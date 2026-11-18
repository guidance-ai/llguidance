//! Offline workload and acceptance checks for completion-row research.
use anyhow::{ensure, Result};
use llguidance::{api::TopLevelGrammar, toktrie::ApproximateTokEnv, ParserFactory, TokenParser};
use serde_json::{json, Value};

pub struct Case {
    pub name: String,
    pub grammar: TopLevelGrammar,
    pub input: Vec<u8>,
    pub row_limit: usize,
    pub step_limit: usize,
}

pub const CASE_NAMES: &[&str] = &[
    "expression",
    "blog_schema",
    "wide_32",
    "wide_256",
    "wide_1024",
    "ambiguous_16",
    "ambiguous_64",
    "ambiguous_128",
];

pub fn case(name: &str) -> Result<Case> {
    ensure!(CASE_NAMES.contains(&name), "unknown case: {name}");
    Ok(match name {
        "expression" => Case {
            name: name.into(),
            grammar: TopLevelGrammar::from_lark(
                "start: expr\nexpr: term | expr \"+\" term\nterm: atom | term \"*\" atom\natom: /[0-9]+/ | \"(\" expr \")\"\n".into(),
            ),
            input: b"(12+3)*4+5*(6+7)".to_vec(),
            row_limit: 2000,
            step_limit: 50_000,
        },
        "blog_schema" => Case {
            name: name.into(),
            grammar: TopLevelGrammar::from_json_schema(
                serde_json::from_str(include_str!("../../sample_parser/data/blog.schema.json"))
                    .unwrap(),
            ),
            input: include_bytes!("../../sample_parser/data/blog.sample.json").to_vec(),
            row_limit: 2000,
            step_limit: 50_000,
        },
        name if name.starts_with("wide_") => wide_case(name[5..].parse()?),
        _ => {
            let length: usize = name[10..].parse()?;
            Case {
                name: name.into(),
                grammar: TopLevelGrammar::from_lark(
                    "start: expr \"!\"\nexpr: expr expr | \"a\"\n".into(),
                ),
                input: [vec![b'a'; length], b"!".to_vec()].concat(),
                row_limit: 2000,
                step_limit: 50_000,
            }
        }
    })
}

pub fn wide_case(width: usize) -> Case {
    // Recursion keeps each alternative a nonterminal rather than one regex.
    // Completing a recursive alternative searches a wide origin row; the
    // compiler may inline or factor the shared atom before parsing.
    let mut grammar = format!(
        "start: {}\natom: \"a\"\n",
        (0..width)
            .map(|i| format!("alt{i}"))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    for i in 0..width {
        grammar.push_str(&format!("alt{i}: \"(\" alt{i} \")\" | atom \"tail{i};\"\n"));
    }
    Case {
        name: format!("wide_{width}"),
        grammar: TopLevelGrammar::from_lark(grammar),
        input: b"((atail0;))".to_vec(),
        // Only the largest stress case overrides the default per-row budget.
        row_limit: if width > 256 { 16_384 } else { 2000 },
        step_limit: if width > 256 { 2_000_000 } else { 50_000 },
    }
}

pub fn template(case: &Case) -> Result<TokenParser> {
    let env = ApproximateTokEnv::single_byte_env();
    // No slicing or fast-forwarding: every input byte takes the same API path.
    let mut factory = ParserFactory::new(&env, Default::default(), &[])?;
    factory.quiet();
    factory.limits_mut().max_items_in_row = case.row_limit;
    factory.limits_mut().step_max_items = case.step_limit;
    factory.create_parser(case.grammar.clone())
}

/// Full byte-token masks are hashed at every prefix, including EOS at the end.
/// This stable FNV-1a checksum is a differential tripwire, not a proof of equivalence.
pub fn replay(parser: &mut TokenParser, input: &[u8]) -> Result<u64> {
    parser.start_without_prompt();
    let mut hash = 0xcbf29ce484222325u64;
    let eos = parser.token_env.tok_trie().eos_token();
    for token in input
        .iter()
        .map(|b| u32::from(*b))
        .chain(std::iter::once(eos))
    {
        let mask = parser.compute_mask()?;
        for word in mask.as_slice() {
            for byte in word.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        ensure!(mask.is_allowed(token), "token {token} rejected");
        ensure!(parser.consume_token(token)? == 0, "unexpected backtracking");
    }
    ensure!(parser.is_accepting(), "input is not accepting");
    Ok(hash)
}

pub fn stats(parser: &TokenParser) -> Value {
    json!(parser.parser_stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_accepts_and_replays_identically() -> Result<()> {
        for name in CASE_NAMES {
            let case = case(name)?;
            let base = template(&case)?;
            let mut first = base.deep_clone();
            let first_hash = replay(&mut first, &case.input)?;
            let mut second = base.deep_clone();
            assert_eq!(
                first_hash,
                replay(&mut second, &case.input)?,
                "{}",
                case.name
            );
            assert_eq!(first.final_bytes(), case.input);
        }
        Ok(())
    }

    #[test]
    fn wide_grammar_acceptance_has_an_independent_oracle() -> Result<()> {
        let case = wide_case(4);
        let base = template(&case)?;
        for text in ["atail0;", "atail3;", "(atail2;)", "((atail1;))"] {
            replay(&mut base.deep_clone(), text.as_bytes())?;
        }
        for text in ["", "atail4;", "atail0", "(atail0;", "atail0;)", "b"] {
            assert!(
                replay(&mut base.deep_clone(), text.as_bytes()).is_err(),
                "{text}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "completion-stats")]
    #[test]
    fn completion_counts_have_known_relationships() -> Result<()> {
        let case = wide_case(32);
        let mut parser = template(&case)?;
        replay(&mut parser, &case.input)?;
        let stats = parser.parser_stats();
        let c = &stats.completion;
        assert!(c.searches > 0);
        assert!(c.candidates > c.matches);
        assert!(c.matches > 0);
        assert!(c.candidates >= c.searches);
        assert!(c.max_origin_row_items > 32);
        assert!(c.max_row_items >= c.max_origin_row_items);
        assert_eq!(stats.delta(&Default::default()).completion, *c);
        assert_eq!(
            llguidance::earley::ParserStats::default()
                .delta(stats)
                .completion
                .candidates,
            0
        );
        let delta = stats.delta(stats);
        assert_eq!(delta.completion.candidates, 0);
        assert_eq!(delta.completion.searches, 0);
        assert_eq!(delta.completion.matches, 0);
        assert_eq!(delta.completion.max_row_items, c.max_row_items);
        assert_eq!(stats.max(&Default::default()).completion, *c);
        // Older saved stats without the optional field still deserialize.
        let mut value = serde_json::to_value(stats)?;
        value.as_object_mut().unwrap().remove("completion");
        let old: llguidance::earley::ParserStats = serde_json::from_value(value)?;
        assert_eq!(old.completion.candidates, 0);
        Ok(())
    }
}
