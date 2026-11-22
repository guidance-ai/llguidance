use llguidance::{
    api::{ParserLimits, TopLevelGrammar},
    toktrie::{ApproximateTokEnv, TokEnv},
    GrammarTemplate, HashMap, Matcher, ParserFactory, StringSlotLimits,
};
use serde_json::json;
use std::sync::Arc;

fn grammar() -> TopLevelGrammar {
    serde_json::from_value(json!({"grammars": [
        {"name": "main", "lark_grammar": "start: \"[\" @target \",\" @agent \",\" ESCAPED_STRING \"]\"\n%import common.ESCAPED_STRING"},
        {"name": "target", "string_slot": "targets", "lark_grammar": "start: ESCAPED_STRING\n%import common.ESCAPED_STRING"},
        {"name": "agent", "string_slot": "agents", "lark_grammar": "start: ESCAPED_STRING\n%import common.ESCAPED_STRING"}
    ]})).unwrap()
}

fn template(limits: StringSlotLimits) -> GrammarTemplate {
    let env: TokEnv = Arc::new(ApproximateTokEnv::single_byte());
    let factory = Arc::new(ParserFactory::new_simple(&env).unwrap());
    GrammarTemplate::new(factory, grammar(), ParserLimits::default(), limits).unwrap()
}

fn bindings(value: &str) -> HashMap<String, Vec<String>> {
    [("targets".into(), vec![value.into()])]
        .into_iter()
        .collect()
}

fn accepts(
    template: &GrammarTemplate,
    bindings: &HashMap<String, Vec<String>>,
    value: &str,
) -> bool {
    let mut matcher = Matcher::new(template.new_parser(bindings));
    for byte in value.bytes() {
        if !matcher.compute_mask().unwrap().is_allowed(byte as u32) {
            return false;
        }
        matcher.consume_token(byte as u32).unwrap();
    }
    matcher.is_accepting().unwrap()
}

#[test]
fn fallback_binding_and_plain_strings_are_independent() {
    let t = template(StringSlotLimits::default());
    assert_eq!(t.slot_names(), ["agents", "targets"]);
    for value in ["Kitchen", "", "Étage 東京", "a\"b\\c\n"] {
        let text = serde_json::to_string(&[value, "any agent", "anything"]).unwrap();
        assert!(accepts(&t, &bindings(value), &text));
        assert!(!accepts(
            &t,
            &bindings(value),
            "[\"wrong\",\"any\",\"free\"]"
        ));
        assert!(accepts(
            &t,
            &HashMap::default(),
            "[\"wrong\",\"any\",\"free\"]"
        ));
    }
}

#[test]
fn invalid_bindings_and_aggregate_limits_leave_template_usable() {
    let t = template(StringSlotLimits {
        max_values: 2,
        max_bytes: 8,
    });
    let invalid: Vec<HashMap<String, Vec<String>>> = vec![
        [("missing".into(), vec!["a".into()])].into_iter().collect(),
        [("targets".into(), vec![])].into_iter().collect(),
        bindings("123456789"),
        [
            ("targets".into(), vec!["a".into(), "a".into()]),
            ("agents".into(), vec!["b".into()]),
        ]
        .into_iter()
        .collect(),
        [
            ("targets".into(), vec!["12345".into()]),
            ("agents".into(), vec!["6789".into()]),
        ]
        .into_iter()
        .collect(),
    ];
    for binding in invalid {
        assert!(t.new_parser(&binding).is_err());
    }
    assert!(accepts(&t, &bindings("a"), "[\"a\",\"b\",\"c\"]"));
}

#[test]
fn concurrent_specializations_do_not_leak() {
    let t = Arc::new(template(StringSlotLimits::default()));
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let t = t.clone();
            std::thread::spawn(move || {
                let value = format!("room {i}");
                let text = serde_json::to_string(&[value.as_str(), "agent", "free"]).unwrap();
                assert!(accepts(&t, &bindings(&value), &text));
                assert!(!accepts(&t, &bindings("other"), &text));
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn duplicate_and_unsupported_fallbacks_are_rejected() {
    let env: TokEnv = Arc::new(ApproximateTokEnv::single_byte());
    let factory = Arc::new(ParserFactory::new_simple(&env).unwrap());
    let mut duplicate = grammar();
    duplicate.grammars[2].string_slot = Some("targets".into());
    let mut multi = grammar();
    multi.grammars[1].lark_grammar =
        Some("start: \"[\" ESCAPED_STRING \"]\"\n%import common.ESCAPED_STRING".into());
    for grammar in [duplicate, multi] {
        assert!(GrammarTemplate::new(
            factory.clone(),
            grammar,
            ParserLimits::default(),
            StringSlotLimits::default()
        )
        .is_err());
    }
}

#[test]
fn prepared_matchers_keep_cache_correct_after_rewind() {
    let env = ApproximateTokEnv::single_byte_env();
    let factory = Arc::new(ParserFactory::new_simple(&env).unwrap());
    let input: TopLevelGrammar = serde_json::from_value(json!({"grammars": [
        {"lark_grammar": "start: @name branch\nbranch: \"a\" middle \"b\" | \"x\" middle \"c\"\nmiddle: /[0-9]+/"},
        {"name": "name", "string_slot": "targets", "lark_grammar": "start: ESCAPED_STRING\n%import common.ESCAPED_STRING"}
    ]})).unwrap();
    let template = GrammarTemplate::new(
        factory,
        input,
        ParserLimits::default(),
        StringSlotLimits::default(),
    )
    .unwrap();
    let tokens = |s: &str| s.bytes().map(u32::from).collect::<Vec<_>>();
    for bindings in [HashMap::default(), bindings("Kitchen")] {
        for reset in [false, true] {
            for clone in [false, true] {
                let fresh = || Matcher::new(template.new_parser(&bindings));
                let mut matcher = fresh();
                matcher.consume_tokens(&tokens("\"Kitchen\"a1")).unwrap();
                matcher.compute_mask().unwrap();
                if clone {
                    matcher = matcher.deep_clone();
                }
                if reset {
                    matcher.reset().unwrap();
                    matcher.consume_tokens(&tokens("\"Kitchen\"")).unwrap();
                } else {
                    matcher.rollback(2).unwrap();
                }
                matcher.consume_tokens(&tokens("x1")).unwrap();
                let actual = matcher.compute_mask().unwrap();
                let mut expected = fresh();
                expected.consume_tokens(&tokens("\"Kitchen\"x1")).unwrap();
                let expected = expected.compute_mask().unwrap();
                for token in 0..env.tok_trie().vocab_size() as u32 {
                    assert_eq!(actual.is_allowed(token), expected.is_allowed(token));
                }
                assert!(!actual.is_allowed(b'b' as u32));
                assert!(actual.is_allowed(b'c' as u32));
            }
        }
    }
}

#[test]
fn specialization_matches_full_compilation_with_boundary_spanning_tokens() {
    use llguidance::toktrie::{TokRxInfo, TokTrie};
    let mut words: Vec<_> = (0..=255).map(|byte| vec![byte]).collect();
    for word in [
        b"\"Kitchen\"".as_slice(),
        b"\",\"",
        b"\"Kitchen\",\"Kitchen\"",
        b"\"Other\"",
    ] {
        words.push(word.to_vec());
    }
    words.push(b"\xff<eos>".to_vec());
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(
        &TokRxInfo::new(words.len() as u32, words.len() as u32 - 1),
        &words,
    )));
    let factory = Arc::new(ParserFactory::new_simple(&env).unwrap());
    let input: TopLevelGrammar = serde_json::from_value(json!({"grammars": [
        {"lark_grammar": "start: \"[\" @name \",\" @name \"]\""},
        {"name": "name", "string_slot": "targets", "lark_grammar": "start: ESCAPED_STRING\n%import common.ESCAPED_STRING"}
    ]})).unwrap();
    let template = GrammarTemplate::new(
        factory.clone(),
        input,
        ParserLimits::default(),
        StringSlotLimits::default(),
    )
    .unwrap();
    let full = TopLevelGrammar::from_lark(
        "start: \"[\" VALUE \",\" VALUE \"]\"\nVALUE: \"\\\"Kitchen\\\"\"".into(),
    );
    for tokens in [
        vec![b'[' as u32, 258, b']' as u32],
        vec![b'[' as u32, 256, b',' as u32, 256, b']' as u32],
    ] {
        let mut bound = Matcher::new(template.new_parser(&bindings("Kitchen")));
        let mut full = Matcher::new(factory.create_parser(full.clone()));
        for token in tokens {
            let actual = bound.compute_mask().unwrap();
            let expected = full.compute_mask().unwrap();
            for id in 0..words.len() as u32 {
                assert_eq!(actual.is_allowed(id), expected.is_allowed(id), "token {id}");
            }
            // Speculative validation must not advance the live matcher.
            assert_eq!(bound.validate_tokens(&[token]).unwrap(), 1);
            bound.consume_token(token).unwrap();
            full.consume_token(token).unwrap();
        }
        assert!(bound.is_accepting().unwrap());
        assert!(full.is_accepting().unwrap());
    }
}

#[test]
fn special_lexeme_fallbacks_are_rejected() {
    let env = ApproximateTokEnv::single_byte_env();
    let factory = Arc::new(ParserFactory::new_simple(&env).unwrap());
    for fallback in [
        "start: %int_ranges {\"min\":1,\"max\":99}",
        "start: /a*/",
        "start[capture=\"value\"]: /a+/",
        "start[max_tokens=3]: /a+/",
    ] {
        let input = serde_json::from_value(
            json!({"grammars": [{"string_slot": "targets", "lark_grammar": fallback}]}),
        )
        .unwrap();
        assert!(
            GrammarTemplate::new(
                factory.clone(),
                input,
                ParserLimits::default(),
                StringSlotLimits::default()
            )
            .is_err(),
            "{fallback}"
        );
    }
}
