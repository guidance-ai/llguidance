use std::{collections::BTreeMap, sync::Arc};

use llguidance::{
    api::TopLevelGrammar,
    toktrie::{
        ApproximateTokEnv, InferenceCapabilities, TokEnv, TokEnvWithTrie, TokRxInfo, TokTrie,
    },
    Matcher, ParserFactory,
};
use serde_json::json;

/// Checks raw output, singleton spelling, ordering, padding and literal whitespace.
#[test]
fn test_int_ranges_language() {
    assert_language(
        r#"start: "{\"content\":\"" ranges "\"}"
        ranges: %int_ranges {"min":0,"max":999,"width":3,"max_ranges":20}"#,
        &[
            r#"{"content":""}"#,
            r#"{"content":"031-031,108-208,300-420"}"#,
        ],
        &[
            r#"{"content":"31-031"}"#,
            r#"{"content":"031"}"#,
            r#"{"content":"032-031"}"#,
            r#"{"content":"031-100,100-101"}"#,
            r#"{"content":"031-100,099-101"}"#,
            r#"{"content":"031-100, 101-101"}"#,
            r#"{"content":"031-100,"}"#,
        ],
    );
    assert_language(
        r#"start: "[" ranges "]"
        ranges: %int_ranges {"min":0,"max":20,"separator":", "}
        %ignore /[ \t\n]+/"#,
        &["[]", "[0-0]", "[0-9, 10-20]", "[ 0-9, 10-20 ]"],
        &[
            "[00-0]",
            "[0-00]",
            "[0 -9]",
            "[0- 9]",
            "[0-9,10-20]",
            "[0-9,  10-20]",
            "[0-9 , 10-20]",
        ],
    );
    assert_language(
        r#"start: ranges "!"
        ranges: %int_ranges {"min":7,"max":10,"min_ranges":1,"max_ranges":1}"#,
        &["7-7!", "7-10!", "10-10!"],
        &["!", "6-7!", "7-11!", "7-7,8-8!", "07-07!", "-7-7!"],
    );
    assert_language(
        r#"start: ranges "!"
        ranges: %int_ranges {"min":0,"max":0,"max_ranges":0}"#,
        &["!"],
        &["0-0!"],
    );
    // The separator limit counts UTF-8 bytes, including literal punctuation.
    for separator in [";".repeat(16), format!("{},", "→".repeat(5))] {
        let config = json!({"min":0,"max":1,"separator":separator,"min_ranges":2});
        let accepted = format!("0-0{separator}1-1!");
        let rejected = format!("0-0{}1-1!", &separator[..15]);
        assert_language(
            &format!("start: ranges \"!\"\nranges: %int_ranges {config}"),
            &[&accepted],
            &[&rejected],
        );
    }
}

/// Checks u32 arithmetic at the boundary and widths above the endpoint digit count.
#[test]
fn test_int_ranges_numeric_edges() {
    assert_language(
        r#"start: ranges "!"
        ranges: %int_ranges {"min":0,"max":10,"min_ranges":2}"#,
        &["9-9,10-10!", "1-9,10-10!"],
        &["10-10!", "9-10,10-10!", "1-1,02-3!"],
    );
    assert_language(
        r#"start: ranges "!"
        ranges: %int_ranges {"min":4294967294,"max":4294967295,"min_ranges":2}"#,
        &["4294967294-4294967294,4294967295-4294967295!"],
        &[
            "4294967295-4294967295!",
            "4294967294-4294967295!",
            "4294967294-4294967294,4294967295-4294967295,0-0!",
        ],
    );
    assert_language(
        r#"start: ranges "!"
        ranges: %int_ranges {"min":0,"max":4294967295}"#,
        &["0-4294967295!", "4294967295-4294967295!"],
        &["4294967296-4294967296!", "0-4294967295,0-0!"],
    );
    assert_language(
        r#"start: ranges "!"
        ranges: %int_ranges {"min":9,"max":10,"width":10,"min_ranges":1}"#,
        &["0000000009-0000000010!"],
        &["000000009-0000000010!", "00000000009-0000000010!"],
    );
    assert_language(
        r#"start: ranges "!"
        ranges: %int_ranges {"min":0,"max":4294967295,"width":10}"#,
        &["0000000000-4294967295!", "4294967295-4294967295!"],
        &["0-4294967295!", "4294967296-4294967296!"],
    );
}

/// Hundreds of intervals remain a single lexeme and do not grow an Earley
/// production chain. There is no implicit twenty-range default limit.
#[test]
fn test_int_ranges_long_sequence() {
    let factory = factory(&[], &[]);
    let mut parser = factory
        .create_parser(TopLevelGrammar::from_lark(
            r#"start: ranges "!"
        ranges: %int_ranges {"min":0,"max":999}"#
                .to_string(),
        ))
        .unwrap();
    parser.start_without_prompt();
    let text = format!(
        "{}!",
        (0..256)
            .map(|n| format!("{n}-{n}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    // Apply the bytes directly so the row count measures committed parsing only.
    parser.parser.apply_token(text.as_bytes(), 0).unwrap();
    assert!(parser.is_accepting());
    assert!(
        parser.parser_stats().rows < 10,
        "{:?}",
        parser.parser_stats()
    );
}

/// Long selections compute actual masks over numeric tokens. Equivalent numeric
/// prefixes must share states so speculative scans stay within the default budget.
#[test]
fn test_int_ranges_long_sequence_masks() {
    let tokens = (0..100)
        .map(|n| format!("{n:02}"))
        .chain((0..1000).map(|n| format!("{n:03}")))
        .collect::<Vec<_>>();
    let factory = factory(&tokens.iter().map(String::as_str).collect::<Vec<_>>(), &[]);
    for width in [0, 6] {
        let config = json!({"min":0,"max":100000,"width":width});
        let mut parser = factory
            .create_parser(TopLevelGrammar::from_lark(format!(
                "start: %int_ranges {config}"
            )))
            .unwrap();
        parser.start_without_prompt();
        let text = (0..1000)
            .map(|n| format!("{:0width$}-{:0width$}", n * 97, n * 97 + 1))
            .collect::<Vec<_>>()
            .join(",");
        for token in factory.tok_env().tokenize(&text) {
            assert!(parser.compute_mask().unwrap().is_allowed(token));
            parser.consume_token(token).unwrap();
        }
        assert!(parser.is_accepting());
        assert_eq!(parser.parser.get_bytes(), text.as_bytes());
    }
}

/// Even a uniquely determined sequence leaves all valid tokenizations available
/// to the model. Ordinary literals outside the construct still fast-forward.
#[test]
fn test_int_ranges_does_not_force_tokens() {
    let vocab = factory(&["00", "000", "00-00,01-01", "00-01"], &[]);
    // Canonical byte tokenization would force individual zeros; the vocabulary
    // also permits longer tokens which the mask must leave available.
    let env: TokEnv = Arc::new(TokEnvWithTrie::new(
        ApproximateTokEnv::single_byte_env(),
        vocab.tok_env().tok_trie().clone(),
    ));
    let mut factory = ParserFactory::new_simple(&env).unwrap();
    factory.quiet();
    let mut parser = factory
        .create_parser(TopLevelGrammar::from_lark(
            r#"start: "[" ranges "]" | "abc"
            ranges: %int_ranges {"min":0,"max":1,"width":2,"min_ranges":2}"#
                .to_string(),
        ))
        .unwrap();
    parser.start_without_prompt();

    let mut ordinary = parser.clone();
    ordinary.consume_token(b'a' as u32).unwrap();
    assert_eq!(
        ordinary.consume_ff_tokens().unwrap(),
        vec![b'b' as u32, b'c' as u32]
    );
    assert!(ordinary.is_accepting());

    parser.consume_token(b'[' as u32).unwrap();
    assert!(parser.compute_ff_tokens().is_empty());
    let mask = parser.compute_mask().unwrap();
    assert!(mask.is_allowed(b'0' as u32));
    assert!(mask.is_allowed(256)); // 00
    assert!(!mask.is_allowed(257)); // 000
    assert!(mask.is_allowed(258)); // 00-00,01-01
    assert!(!mask.is_allowed(259)); // 00-01 leaves no ID for the second range

    for byte in b"00-00,01-01" {
        assert!(parser.force_bytes().is_empty());
        assert!(parser.compute_mask().unwrap().is_allowed(u32::from(*byte)));
        parser.consume_token(u32::from(*byte)).unwrap();
    }
    assert_eq!(parser.consume_ff_tokens().unwrap(), vec![b']' as u32]);
    assert!(parser.is_accepting());
}

/// Compact parameters can describe enormous forced strings. Computing one mask
/// must visit only token prefixes, without expanding padding, counts or separators.
#[test]
fn test_int_ranges_mask_work_stays_within_token() {
    let env = ApproximateTokEnv::single_byte_env();
    let mut factory = ParserFactory::new_simple(&env).unwrap();
    factory.quiet();
    factory.limits_mut().max_lexer_states = 64;
    for (config, prefix, next) in [
        (
            json!({"min":0,"max":4294967295u64,"min_ranges":4294967296u64}),
            "",
            b'0',
        ),
        (json!({"min":0,"max":9,"width":10,"min_ranges":1}), "", b'0'),
        (
            json!({"min":0,"max":1,"min_ranges":2,"separator":",".repeat(16)}),
            "0-0",
            b',',
        ),
    ] {
        let mut parser = factory
            .create_parser(TopLevelGrammar::from_lark(format!(
                "start: %int_ranges {config}"
            )))
            .unwrap();
        parser.start_without_prompt();
        for byte in prefix.bytes() {
            parser.consume_token(u32::from(byte)).unwrap();
        }
        assert!(parser.force_bytes().is_empty());
        let mask = parser.compute_mask().unwrap();
        assert_eq!(mask.iter().collect::<Vec<_>>(), vec![u32::from(next)]);
        assert_eq!(parser.parser.get_bytes(), prefix.as_bytes());
        assert!(parser.parser.lexer_stats().num_states < 64);
    }
}

/// Numeric probes cost fuel, and exhausting it stops within one
/// matcher operation rather than after all overlapping alternatives are visited.
#[test]
fn test_int_ranges_mask_fuel() {
    let tokens = (0..1000).map(|n| format!("{n:03}")).collect::<Vec<_>>();
    let mut factory = factory(&tokens.iter().map(String::as_str).collect::<Vec<_>>(), &[]);
    for alternatives in [1, 100] {
        let choices = (1..=alternatives)
            .map(|count| {
                format!(
                    "%int_ranges {}",
                    json!({"min":0,"max":999,"width":3,"max_ranges":count})
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let grammar = format!("start: ranges \"!\"\nranges: {choices}");
        let mut cases = vec![("", 1), ("031-", 64)];
        if alternatives == 100 {
            cases.push(("", 200_000));
        }
        for (prefix, fuel) in cases {
            factory.limits_mut().step_lexer_fuel = fuel;
            let mut parser = factory
                .create_parser(TopLevelGrammar::from_lark(grammar.clone()))
                .unwrap();
            parser.start_without_prompt();
            for byte in prefix.bytes() {
                parser.consume_token(u32::from(byte)).unwrap();
            }
            let before = parser.parser.lexer_stats().total_fuel_spent;
            let error = parser.compute_mask().unwrap_err().to_string();
            let work = parser.parser.lexer_stats().total_fuel_spent - before;
            assert!(error.contains("too many expressions"), "{error}");
            // One matcher operation may overshoot; the budget must stop work
            // before visiting the rest of the overlapping alternatives.
            assert!(
                (fuel as usize..fuel as usize + 1000).contains(&work),
                "work={work} fuel={fuel}"
            );
        }
    }
}

/// Many overlapping lexemes can keep the outer DFA small while retaining large
/// inner caches. Both compilation and later cache growth must honor the limit.
#[test]
fn test_int_ranges_cache_budget() {
    let choices = (1..=100)
        .map(|count| {
            format!(
                "%int_ranges {}",
                json!({"min":0,"max":999,"width":3,"max_ranges":count})
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    let grammar = format!("start: ranges \"!\"\nranges: {choices}");
    let tokens = (0..1000).map(|n| format!("{n:03}")).collect::<Vec<_>>();
    let mut factory = factory(&tokens.iter().map(String::as_str).collect::<Vec<_>>(), &[]);
    factory.limits_mut().max_lexer_states = 64;
    let error = factory
        .create_parser(TopLevelGrammar::from_lark(grammar.clone()))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("cache exceeds max_lexer_states"), "{error}");

    factory.limits_mut().max_lexer_states = 2000;
    factory.limits_mut().step_lexer_fuel = 2_000_000;
    let mut parser = factory
        .create_parser(TopLevelGrammar::from_lark(grammar))
        .unwrap();
    parser.start_without_prompt();
    let error = parser.compute_mask().unwrap_err().to_string();
    assert!(error.contains("cache exceeds max_lexer_states"), "{error}");
    let stats = parser.parser.lexer_stats();
    assert!(stats.num_states < 500, "{stats}");
    assert!(stats.num_bytes < 4 * 1024 * 1024, "{stats}");
}

/// Initialization scans first-byte transitions even with optional precomputation
/// disabled; exhausting its fuel must fail compilation rather than return a parser.
#[test]
fn test_int_ranges_initialization_fuel() {
    let mut factory = factory(&[], &[]);
    factory.limits_mut().precompute_large_lexemes = false;
    factory.limits_mut().initial_lexer_fuel = 64;
    let error = factory
        .create_parser(TopLevelGrammar::from_lark(
            r#"start: %int_ranges {"min":0,"max":999,"width":3}"#.to_string(),
        ))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("initialization failed"), "{error}");
}

/// Invalid declarations fail at compilation, including impossible minimum counts.
#[test]
fn test_int_ranges_configuration_errors() {
    let factory = factory(&[], &[]);
    for config in [
        json!({}),
        json!({"min":0}),
        json!({"min":-1,"max":3}),
        json!({"min":0,"max":4294967296u64}),
        json!({"min":0.5,"max":3}),
        json!({"min":3,"max":2}),
        json!({"min":0,"max":100,"width":2}),
        json!({"min":0,"max":3,"width":-1}),
        json!({"min":0,"max":3,"width":11}),
        json!({"min":0,"max":3,"separator":""}),
        json!({"min":0,"max":3,"separator":"x1"}),
        json!({"min":0,"max":3,"separator":" - "}),
        json!({"min":0,"max":3,"separator":",".repeat(17)}),
        json!({"min":0,"max":3,"separator":format!("{}, ", "→".repeat(5))}),
        json!({"min":0,"max":3,"min_ranges":2,"max_ranges":1}),
        json!({"min":0,"max":3,"min_ranges":5}),
        json!({"min":0,"max":3,"max_ranges":-1}),
        json!({"min":0,"max":3,"unknown":1}),
    ] {
        assert!(
            factory
                .create_parser(TopLevelGrammar::from_lark(format!(
                    "start: %int_ranges {config}"
                )))
                .is_err(),
            "{config}"
        );
    }
    let err = factory
        .create_parser(TopLevelGrammar::from_lark(
            "start: R\nR: %int_ranges {\"min\":0,\"max\":3}".to_string(),
        ))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("cannot be used in terminals"), "{err}");
}

/// Whole-token validation must visit each dynamic bound, with regex slicing enabled.
#[test]
fn test_int_ranges_speculative_tokens_and_slices() {
    let tokens = [
        "0-0,1-1!",
        "0-1,2-3!",
        "0-3,4-4!",
        "0-2,2-3!",
        "0-0,1-1,2-2!",
        "0-0,1-1,2-2,3-3!",
        "3-3!",
        "0-2,3-3!",
        "0-3,",
        "0-0,1-0!",
        "00-0,1-1!",
        "0-0,1-1,!",
    ];
    for slices in [vec![], vec!["[0-9,-]+!?".to_string(), "[0-9]+".to_string()]] {
        let factory = factory(&tokens, &slices);
        let mut matcher = matcher(
            &factory,
            r#"start: ranges "!"
            ranges: %int_ranges {"min":0,"max":3,"min_ranges":2,"max_ranges":3}"#,
        );
        let expected = [
            true, true, false, false, true, false, false, true, false, false, false, false,
        ];
        let mask = matcher.compute_mask().unwrap();
        for (idx, valid) in expected.into_iter().enumerate() {
            let token = 256 + idx as u32;
            assert_eq!(mask.is_allowed(token), valid, "{}", tokens[idx]);
            assert_eq!(
                matcher.validate_tokens(&[token]).unwrap(),
                usize::from(valid),
                "{}",
                tokens[idx]
            );
            if valid {
                let mut copy = matcher.clone();
                copy.consume_token(token).unwrap();
                assert!(copy.is_accepting().unwrap());
            }
        }
        assert_eq!(mask, matcher.compute_mask().unwrap());
    }
}

/// Rollback and both clone modes must retain numeric state even when the outer
/// grammar row is unchanged. Cached masks must distinguish those numeric states.
#[test]
fn test_int_ranges_rollback_cloning_and_forcing() {
    let factory = factory(&[], &[]);
    let mut original = matcher(
        &factory,
        r#"start: ranges "!"
        ranges: %int_ranges {"min":0,"max":9,"min_ranges":2,"max_ranges":2}"#,
    );
    original
        .consume_tokens(&b"7-".iter().map(|b| u32::from(*b)).collect::<Vec<_>>())
        .unwrap();
    let high_mask = original.compute_mask().unwrap();
    assert!(!high_mask.is_allowed(b'6' as u32));
    assert!(!high_mask.is_allowed(b'9' as u32));
    assert!(high_mask.is_allowed(b'7' as u32));
    assert!(high_mask.is_allowed(b'8' as u32));
    for mut copy in [original.clone(), original.deep_clone()] {
        copy.consume_token(b'8' as u32).unwrap();
        assert!(copy.compute_ff_bytes().is_empty());
        assert_eq!(
            copy.compute_mask().unwrap().iter().collect::<Vec<_>>(),
            vec![b',' as u32]
        );
        copy.consume_tokens(&b",9-9!".iter().map(|b| u32::from(*b)).collect::<Vec<_>>())
            .unwrap();
        assert!(copy.is_accepting().unwrap());
    }
    assert_eq!(high_mask, original.compute_mask().unwrap());
    original.rollback(2).unwrap();
    original
        .consume_tokens(&[b'1' as u32, b'-' as u32])
        .unwrap();
    let low_mask = original.compute_mask().unwrap();
    assert!(low_mask.is_allowed(b'1' as u32));
    assert!(low_mask.is_allowed(b'6' as u32));
    assert!(!low_mask.is_allowed(b'0' as u32));
    assert!(!low_mask.is_allowed(b'9' as u32));
    original
        .consume_tokens(&[b'2' as u32, b',' as u32])
        .unwrap();
    let next_mask = original.compute_mask().unwrap();
    assert!(!next_mask.is_allowed(b'2' as u32));
    assert!(next_mask.is_allowed(b'3' as u32));
    original.rollback(2).unwrap();
    assert_eq!(low_mask, original.compute_mask().unwrap());
}

/// A standalone construct allows EOS only after the minimum count, even when
/// the current decimal endpoint could still consume another digit.
#[test]
fn test_int_ranges_eos() {
    let factory = factory(&[], &[]);
    let eos = factory.tok_env().tok_trie().eos_token();
    let mut m = matcher(
        &factory,
        r#"start: %int_ranges {"min":0,"max":10,"min_ranges":2,"max_ranges":2}"#,
    );
    assert!(!m.compute_mask().unwrap().is_allowed(eos));
    m.consume_tokens(&b"0-0".iter().map(|b| u32::from(*b)).collect::<Vec<_>>())
        .unwrap();
    assert!(!m.compute_mask().unwrap().is_allowed(eos));
    m.consume_tokens(&b",1-1".iter().map(|b| u32::from(*b)).collect::<Vec<_>>())
        .unwrap();
    let mask = m.compute_mask().unwrap();
    assert!(mask.is_allowed(eos));
    assert!(mask.is_allowed(b'0' as u32));
    assert!(!mask.is_allowed(b',' as u32));
    m.consume_token(eos).unwrap();
    assert!(m.is_stopped());

    let mut empty = matcher(&factory, r#"start: %int_ranges {"min":0,"max":10}"#);
    let mask = empty.compute_mask().unwrap();
    assert!(mask.is_allowed(eos));
    assert!(mask.is_allowed(b'0' as u32));
}

/// Exhaustively compares every reachable byte prefix against an independently
/// enumerated finite language, including acceptance and all next-byte choices.
#[test]
fn test_int_ranges_every_prefix_has_completion() {
    let factory = factory(&[], &[]);
    for (min, max, width, separator, min_count, max_count) in [
        (0, 3, 0, ",", 0, Some(3)),
        (0, 3, 2, " ; ", 2, Some(3)),
        (0, 3, 0, "→", 3, Some(3)),
        (8, 11, 0, ",", 2, Some(3)),
        (98, 101, 3, ",", 2, Some(3)),
        (0, 3, 0, ",", 0, None),
        (0, 3, 2, " ; ", 2, None),
        (8, 11, 0, ",", 2, None),
    ] {
        let config = json!({
            "min": min,
            "max": max,
            "width": width,
            "separator": separator,
            "min_ranges": min_count,
            "max_ranges": max_count,
        });
        let grammar = format!("start: ranges \"!\"\nranges: %int_ranges {config}");
        let mut language = Vec::new();
        enumerate(
            &mut language,
            &mut Vec::new(),
            (min, max),
            width,
            separator,
            min_count,
            max_count.unwrap_or((max - min + 1) as usize),
        );
        let mut prefixes: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
        for text in &language {
            for idx in 0..text.len() {
                let next = prefixes.entry(text.as_bytes()[..idx].to_vec()).or_default();
                if !next.contains(&text.as_bytes()[idx]) {
                    next.push(text.as_bytes()[idx]);
                }
            }
        }
        let initial = matcher(&factory, &grammar);
        for (prefix, next) in prefixes {
            let mut m = initial.clone();
            m.consume_tokens(&prefix.iter().map(|b| u32::from(*b)).collect::<Vec<_>>())
                .unwrap();
            let mask = m.compute_mask().unwrap();
            for byte in 0..=255u8 {
                assert_eq!(
                    mask.is_allowed(u32::from(byte)),
                    next.contains(&byte),
                    "config={config} prefix={prefix:?} byte={byte}"
                );
            }
            assert!(!m.is_accepting().unwrap()); // Every enumerated prefix still lacks '!'.
        }
    }
}

/// Multiple occurrences and alternatives get independent interval sequences.
#[test]
fn test_int_ranges_grammar_composition() {
    assert_language(
        r#"start: ranges "/" ranges "!" | "other!"
        ranges[capture]: %int_ranges {"min":0,"max":2,"min_ranges":1}"#,
        &["2-2/0-0!", "0-0,1-2/0-2!", "other!"],
        &["2-2/!", "2-2/1-0!"],
    );
    assert_language(
        r#"start: a "!" | b "?"
        a: %int_ranges {"min":0,"max":3,"min_ranges":2}
        b: %int_ranges {"min":0,"max":3,"min_ranges":1,"max_ranges":1}"#,
        &["0-0,1-1!", "0-3?"],
        &["0-3!", "0-0,1-1?"],
    );
}

/// Range completion must compose with a special token and a nullable suffix
/// lookahead without allowing either to bypass an incomplete endpoint.
#[test]
fn test_int_ranges_special_token_and_nullable_suffix() {
    let env = ApproximateTokEnv::single_byte_env();
    let tool = env.tok_trie().get_special_token("<|tool|>").unwrap();
    let mut factory = ParserFactory::new(&env, InferenceCapabilities::default(), &[]).unwrap();
    factory.quiet();
    let base = Matcher::new(Ok(factory
        .create_parser(TopLevelGrammar::from_lark(
            r#"
        start: ranges <|tool|> tail
        ranges: %int_ranges {"min":0,"max":99}
        tail[suffix="!"]: /[a-z]*/
    "#
            .into(),
        ))
        .unwrap()));
    let consume = |m: &mut Matcher, text: &str| {
        for b in text.bytes() {
            assert!(m.compute_mask_or_eos().unwrap().is_allowed(u32::from(b)));
            m.consume_token(u32::from(b)).unwrap();
        }
    };
    for (ranges, tail) in [("", "!"), ("6-6", "!"), ("1-9", "abc!")] {
        let mut m = base.clone();
        consume(&mut m, ranges);
        let mask = m.compute_mask_or_eos().unwrap();
        assert!(mask.is_allowed(tool));
        assert!(!mask.is_allowed(env.eos_token()));
        m.consume_token(tool).unwrap();
        consume(&mut m, tail);
        assert!(m.compute_mask_or_eos().unwrap().is_allowed(env.eos_token()));
    }
    let mut m = base.clone();
    consume(&mut m, "1-");
    assert!(!m.compute_mask_or_eos().unwrap().is_allowed(tool));
}

/// Builds a self-contained byte vocabulary, optionally with tokens spanning ranges.
fn factory(extra: &[&str], slices: &[String]) -> ParserFactory {
    let mut words: Vec<_> = (0..=255).map(|byte| vec![byte]).collect();
    words.extend(extra.iter().map(|text| text.as_bytes().to_vec()));
    words.push(b"\xff<eos>".to_vec());
    let info = TokRxInfo::new(words.len() as u32, words.len() as u32 - 1);
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
    let mut factory = ParserFactory::new(&env, InferenceCapabilities::default(), slices).unwrap();
    factory.quiet();
    factory
}

/// Compiles and initializes a fresh matcher, keeping compilation errors visible.
fn matcher(factory: &ParserFactory, grammar: &str) -> Matcher {
    Matcher::new(Ok(factory
        .create_parser(TopLevelGrammar::from_lark(grammar.to_string()))
        .unwrap()))
}

/// Checks both token masks and committed parsing for complete example strings.
fn assert_language(grammar: &str, accepted: &[&str], rejected: &[&str]) {
    let factory = factory(&[], &[]);
    for (texts, expected) in [(accepted, true), (rejected, false)] {
        for text in texts {
            let mut m = matcher(&factory, grammar);
            let mut allowed = true;
            for byte in text.bytes() {
                if !m.compute_mask().unwrap().is_allowed(u32::from(byte)) {
                    allowed = false;
                    break;
                }
                m.consume_token(u32::from(byte)).unwrap();
            }
            assert_eq!(
                allowed && m.is_accepting().unwrap(),
                expected,
                "{text:?}\n{grammar}"
            );
        }
    }
}

/// Enumerates the mathematical interval language over a small domain, without
/// using any parser or regex code, to supply the prefix-completion oracle.
fn enumerate(
    language: &mut Vec<String>,
    parts: &mut Vec<String>,
    bounds: (u32, u32),
    width: usize,
    separator: &str,
    min_count: usize,
    max_count: usize,
) {
    if parts.len() >= min_count {
        language.push(format!("{}!", parts.join(separator)));
    }
    if parts.len() == max_count {
        return;
    }
    for start in bounds.0..=bounds.1 {
        for end in start..=bounds.1 {
            parts.push(format!("{start:0width$}-{end:0width$}"));
            enumerate(
                language,
                parts,
                (end + 1, bounds.1),
                width,
                separator,
                min_count,
                max_count,
            );
            parts.pop();
        }
    }
}
