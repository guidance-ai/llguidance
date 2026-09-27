use std::{collections::BTreeMap, sync::Arc};

use llguidance::{
    api::TopLevelGrammar,
    toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie},
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
        ranges: %int_ranges {"min":9,"max":10,"width":12,"min_ranges":1}"#,
        &["000000000009-000000000010!"],
        &["00000000009-000000000010!", "0000000000009-000000000010!"],
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
        json!({"min":0,"max":3,"separator":""}),
        json!({"min":0,"max":3,"separator":"x1"}),
        json!({"min":0,"max":3,"separator":" - "}),
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
        assert_eq!(copy.compute_ff_bytes(), b",9-9!");
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
        (0, 3, 0, ",", 0, 3),
        (0, 3, 2, " ; ", 2, 3),
        (0, 3, 0, "→", 3, 3),
        (8, 11, 0, ",", 2, 3),
        (98, 101, 3, ",", 2, 3),
    ] {
        let config = json!({"min":min,"max":max,"width":width,"separator":separator,"min_ranges":min_count,"max_ranges":max_count});
        let grammar = format!("start: ranges \"!\"\nranges: %int_ranges {config}");
        let mut language = Vec::new();
        enumerate(
            &mut language,
            &mut Vec::new(),
            (min, max),
            width,
            separator,
            min_count,
            max_count,
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
