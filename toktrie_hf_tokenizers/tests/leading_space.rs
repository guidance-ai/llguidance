#[path = "../examples/leading_space/support.rs"]
mod support;

use llguidance::ParserFactory;
use support::{check_sequence, initial_mask, load, GRAMMARS};

#[test]
fn fixture_pairs_change_only_the_declared_preprocessing_setting() {
    let configs: Vec<serde_json::Value> = support::FIXTURES
        .iter()
        .map(|(_, bytes)| serde_json::from_slice(bytes).unwrap())
        .collect();
    let mut bytelevel = configs[0].clone();
    bytelevel["pre_tokenizer"]["add_prefix_space"] = true.into();
    assert_eq!(bytelevel, configs[1]);
    for (index, scheme) in [(3, "first"), (4, "always")] {
        let mut metaspace = configs[2].clone();
        metaspace["pre_tokenizer"]["prepend_scheme"] = scheme.into();
        assert_eq!(metaspace, configs[index]);
    }
    let mut prepend = configs[6].clone();
    prepend["normalizer"]["normalizers"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    assert_eq!(prepend, configs[5]);
}

// Independent byte-language oracle for the two fixture grammars.
fn accepts_bytes(bytes: &[u8], allow_space: bool) -> bool {
    let bytes = if allow_space {
        bytes.strip_prefix(b" ").unwrap_or(bytes)
    } else {
        bytes
    };
    let bytes = bytes.strip_suffix(b"!").unwrap_or(bytes);
    bytes.split(|b| *b == b' ').all(|word| {
        !word.is_empty() && word.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'_')
    })
}

#[test]
fn source_preprocessing_and_adapter_bytes_are_separate_observations() {
    for (name, bytes) in support::FIXTURES {
        let (source, adapted, env) = load(bytes).unwrap();
        let source_ids = source.encode("hello", false).unwrap();
        let adapted_ids = adapted.encode("hello", false).unwrap();
        let source_prepends = matches!(
            name,
            "bytelevel-true" | "metaspace-first" | "metaspace-always" | "prepend-space"
        );
        assert_eq!(
            env.tok_trie().decode(source_ids.get_ids()),
            if source_prepends {
                b" hello".as_slice()
            } else {
                b"hello".as_slice()
            },
            "source: {name}"
        );
        // The existing adapter disables Metaspace/Prepend preprocessing, but
        // leaves ByteLevel add_prefix_space as configured. This test observes it.
        assert_eq!(
            env.tok_trie().decode(adapted_ids.get_ids()),
            if name == "bytelevel-true" {
                b" hello".as_slice()
            } else {
                b"hello".as_slice()
            },
            "adapter: {name}"
        );
        if name.starts_with("metaspace-") || name.starts_with("prepend-") {
            let space_hello = env.tok_trie().token_id(b" hello").unwrap();
            assert_eq!(source.decode(&[space_hello], false).unwrap(), "hello");
            assert_eq!(env.tok_trie().token(space_hello), b" hello");
        }
    }
}

#[test]
fn environment_encoding_and_literal_input_have_explicit_byte_checks() {
    for (name, bytes) in support::FIXTURES {
        let (_, _, env) = load(bytes).unwrap();
        let mut factory = ParserFactory::new_simple(&env).unwrap();
        factory.quiet();
        for text in support::samples() {
            let ids = env.tokenize(&text);
            let actual_bytes = env.tok_trie().decode(&ids);
            let mut expected_bytes = text.as_bytes().to_vec();
            if name == "bytelevel-true" && !text.is_empty() && !text.starts_with(' ') {
                expected_bytes.insert(0, b' ');
            }
            assert_eq!(actual_bytes, expected_bytes, "{name}: {text:?}");
            let literal_ids = env.tok_trie().greedy_tokenize(text.as_bytes());
            assert_eq!(env.tok_trie().decode(&literal_ids), text.as_bytes());
            for (index, (_, grammar)) in GRAMMARS.iter().enumerate() {
                let result = check_sequence(&factory, grammar, &ids).unwrap();
                assert_eq!(
                    result["accepted"],
                    accepts_bytes(&actual_bytes, index == 1),
                    "{name}, grammar {index}, input {text:?}: {result}"
                );
                let literal = check_sequence(&factory, grammar, &literal_ids).unwrap();
                assert_eq!(
                    literal["accepted"],
                    accepts_bytes(text.as_bytes(), index == 1)
                );
            }
        }
    }
}

#[test]
fn masks_and_validity_depend_on_token_bytes_not_encoding_preprocessing() {
    let mut reference = None;
    for (name, bytes) in support::FIXTURES {
        let (_, _, env) = load(bytes).unwrap();
        let mut factory = ParserFactory::new_simple(&env).unwrap();
        factory.quiet();
        let masks: Vec<_> = GRAMMARS
            .iter()
            .map(|(_, grammar)| initial_mask(&factory, grammar).unwrap())
            .collect();
        if let Some(reference) = &reference {
            assert_eq!(
                &masks, reference,
                "same token bytes, different preprocessing: {name}"
            );
        } else {
            reference = Some(masks.clone());
        }
        let trie = env.tok_trie();
        let hello = trie.token_id(b"hello").unwrap();
        let space_hello = trie.token_id(b" hello").unwrap();
        let space = trie.token_id(b" ").unwrap();
        for (index, (_, grammar)) in GRAMMARS.iter().enumerate() {
            assert!(masks[index].is_allowed(hello), "{name}");
            assert_eq!(masks[index].is_allowed(space_hello), index == 1, "{name}");
            assert_eq!(masks[index].is_allowed(space), index == 1, "{name}");
            assert!(
                !masks[index].is_allowed(trie.eos_token()),
                "empty must reject"
            );
            // Exercise every non-special vocabulary entry, including partial words,
            // alone and after one literal leading space. The byte oracle checks
            // final acceptance; check_sequence also compares validation and masks.
            for id in 0..trie.vocab_size() as u32 {
                if trie.is_special_token(id) {
                    continue;
                }
                for ids in [vec![id], vec![space, id], vec![hello, id]] {
                    let bytes = trie.decode(&ids);
                    let result = check_sequence(&factory, grammar, &ids).unwrap();
                    assert_eq!(
                        result["accepted"],
                        accepts_bytes(&bytes, index == 1),
                        "{name}, grammar {index}, ids {ids:?}, bytes {bytes:?}: {result}"
                    );
                }
            }
        }
    }
}

#[test]
fn report_records_observations_without_inference_claims() {
    for (name, bytes) in support::FIXTURES {
        let report = support::analyze(name, bytes).unwrap();
        assert_eq!(report["vocab_size"], 22);
        assert_eq!(report["initial_mask"]["no_leading_space_count"], 17);
        assert_eq!(report["initial_mask"]["optional_leading_space_count"], 20);
        assert_eq!(report["initial_mask"]["added_count"], 3);
        assert_eq!(report["samples"].as_array().unwrap().len(), 10);
        for row in report["samples"].as_array().unwrap() {
            assert_eq!(row["validity"].as_array().unwrap().len(), 6);
            let input = row["input"].as_str().unwrap();
            assert_eq!(
                row["llguidance_env"]["preserves_input_bytes"],
                name != "bytelevel-true" || input.is_empty() || input.starts_with(' ')
            );
        }
    }
}
