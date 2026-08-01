use llg_test_utils::*;
use llguidance::{api::TopLevelGrammar, ParserFactory};
use std::collections::BTreeSet;

const LONG_LOOKAHEAD: usize = 80;

fn long_backtracking_grammar() -> String {
    format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: T S
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        S: /b{{{LONG_LOOKAHEAD}}}d/
    "#
    )
}

fn spanning_token_grammar() -> &'static str {
    r#"
        %llguidance {"no_forcing": true, "lexer_backtracking": true}
        start: T S
        T: /a|abc/
        S: /by/
    "#
}

fn branching_long_backtracking_grammar() -> String {
    format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: T (S | X)
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        S: /b{{{LONG_LOOKAHEAD}}}d/
        X: "x"
    "#
    )
}

#[test]
fn test_lexer_backtracking_recovers_after_unfinished_longer_attempt() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "lexer_backtracking": true}
            start: STEM SUFFIX
            STEM: "list" | "listen"
            SUFFIX: "ed"
        "#,
        &["listed", "listened"],
        &["listd", "listeed", "FINAL_REJECT:list"],
    );
}

#[test]
fn test_lexer_backtracking_preserves_global_maximal_munch() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "lexer_backtracking": true}
            start: A B | AB C
            A: "a"
            B: "b"
            AB: "ab"
            C: "c"
        "#,
        &["abc"],
        &["FINAL_REJECT:ab"],
    );
}

#[test]
fn test_lexer_backtracking_contextual_ipv6() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "lexer_backtracking": true}
            start: "ping " ipv6
                 | "connect " (ipv6 | HOST_PORT)
            ipv6: HEX "::" HEX
            HEX: /[0-9a-f]+/
            HOST_PORT: /[a-z0-9]+:[0-9]+/
        "#,
        &["ping fe80::1", "connect fe80::1", "connect server:80"],
        &["connect fe80:xyz"],
    );
}

#[test]
fn test_lexer_backtracking_fstring_chunk() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "lexer_backtracking": true}
            start: "f\"" FCHUNK? ("{" NAME "}" FCHUNK?)* "\""
            FCHUNK: /(?:[^{}"\\]|\\.|\{\{|\}\})+/
            NAME: /[A-Za-z_][A-Za-z0-9_]*/
        "#,
        &[r#"f"User {name} done""#],
        &[r#"f"User {name done""#],
    );
}

#[test]
fn test_lexer_backtracking_at_eos() {
    lark_str_test_many(
        r#"
            %llguidance {"no_forcing": true, "lexer_backtracking": true}
            start: T B
            T: "a" | "abc"
            B: "b"
        "#,
        &["ab"],
        &["FINAL_REJECT:a"],
    );
}

#[test]
fn test_lexer_backtracking_accepting_boundary_preserves_state() {
    let grammar = r#"
        %llguidance {"no_forcing": true, "lexer_backtracking": true}
        start: NUMBER
        NUMBER: /[0-9]+/
    "#;
    let mut parser = make_parser(grammar, true).unwrap();
    consume_text(&mut parser, "5");

    let env = get_tok_env();
    let eos = env.tok_trie().eos_token();
    let mut suffix = env.tokenize("6");
    suffix.push(eos);

    for _ in 0..2 {
        let mask = parser.compute_mask().unwrap();
        assert!(mask.is_allowed(eos));
        assert_eq!(parser.validate_tokens_raw(&suffix).unwrap(), suffix.len());
        assert_eq!(parser.final_bytes(), b"5");
    }

    consume_text(&mut parser, "6");
    assert!(parser.is_accepting());
}

#[test]
fn test_lexer_backtracking_after_long_lookahead() {
    let grammar = long_backtracking_grammar();
    let input = format!("a{}d", "b".repeat(LONG_LOOKAHEAD));
    lark_str_test(&grammar, true, &input, true);
}

#[test]
fn test_lexer_backtracking_across_nested_boundaries() {
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: T U V
        T: /a|abd{{{LONG_LOOKAHEAD}}}x/
        U: /b|bd{{{LONG_LOOKAHEAD}}}e/
        V: /d{{{LONG_LOOKAHEAD}}}f/
    "#
    );
    let prefix = format!("ab{}", "d".repeat(LONG_LOOKAHEAD));
    let mut parser = make_parser(&grammar, true).unwrap();
    consume_text(&mut parser, &prefix);

    let token = get_tok_env().tokenize("f")[0];
    for _ in 0..2 {
        let mask = parser.compute_mask().unwrap();
        assert!(mask.is_allowed(token));
        assert_eq!(parser.final_bytes(), prefix.as_bytes());
    }
    consume(&mut parser, token);
    assert!(parser.is_accepting());
}

#[test]
fn test_lexer_backtracking_accepts_at_eos_after_long_lookahead() {
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: T S
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        S: /b{{{LONG_LOOKAHEAD}}}/
    "#
    );
    let input = format!("a{}", "b".repeat(LONG_LOOKAHEAD));
    lark_str_test(&grammar, true, &input, true);
}

fn consume_text(parser: &mut llguidance::TokenParser, text: &str) {
    let env = get_tok_env();
    for tok in env.tokenize(text) {
        let mask = parser.compute_mask().unwrap();
        assert!(
            mask.is_allowed(tok),
            "rejected {}",
            env.tok_trie().token_dbg(tok)
        );
        consume(parser, tok);
    }
}

#[test]
fn test_lexer_backtracking_preserves_capture() {
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: stem S
        stem[capture]: T
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        S: /b{{{LONG_LOOKAHEAD}}}d/
    "#
    );
    let input = format!("a{}d", "b".repeat(LONG_LOOKAHEAD));
    let mut parser = make_parser(&grammar, true).unwrap();
    consume_text(&mut parser, &input);
    assert!(parser.is_accepting());
    assert_eq!(parser.get_capture("stem"), Some(&b"a"[..]));
}

#[test]
fn test_lexer_backtracking_rollback_and_rebuild() {
    let grammar = long_backtracking_grammar();
    let full = format!("a{}d", "b".repeat(LONG_LOOKAHEAD));
    let prefix = format!("a{}", "b".repeat(72));
    let mut parser = make_parser(&grammar, true).unwrap();
    consume_text(&mut parser, &prefix);
    let rollback_tokens = 4.min(parser.num_tokens());
    parser.rollback(rollback_tokens).unwrap();
    let consumed = parser.final_bytes().len();
    consume_text(&mut parser, &full[consumed..]);
    assert!(parser.is_accepting());
}

#[test]
fn test_lexer_backtracking_deep_clone_is_independent() {
    let grammar = branching_long_backtracking_grammar();
    let prefix = format!("a{}", "b".repeat(72));
    let mut shorter = make_parser(&grammar, true).unwrap();
    consume_text(&mut shorter, &prefix);
    let mut longest = shorter.deep_clone();
    consume_text(&mut shorter, &format!("{}d", "b".repeat(8)));
    consume_text(&mut longest, &format!("{}cx", "b".repeat(8)));
    assert!(shorter.is_accepting());
    assert!(longest.is_accepting());
}

#[test]
fn test_lexer_backtracking_shared_lexer_clone_is_independent() {
    let grammar = branching_long_backtracking_grammar();
    let prefix = format!("a{}", "b".repeat(72));
    let mut shorter = make_parser(&grammar, true).unwrap();
    consume_text(&mut shorter, &prefix);
    let mut longest = shorter.clone();
    consume_text(&mut shorter, &format!("{}d", "b".repeat(8)));
    consume_text(&mut longest, &format!("{}cx", "b".repeat(8)));
    assert!(shorter.is_accepting());
    assert!(longest.is_accepting());
}

#[test]
fn test_lexer_backtracking_validate_tokens_restores_state() {
    let grammar = long_backtracking_grammar();
    let prefix = format!("a{}", "b".repeat(72));
    let suffix = format!("{}d", "b".repeat(8));
    let mut parser = make_parser(&grammar, true).unwrap();
    consume_text(&mut parser, &prefix);
    let tokens = get_tok_env().tokenize(&suffix);
    assert_eq!(parser.validate_tokens_raw(&tokens).unwrap(), tokens.len());
    assert_eq!(parser.final_bytes(), prefix.as_bytes());
    consume_text(&mut parser, &suffix);
    assert!(parser.is_accepting());
}

#[test]
fn test_lexer_backtracking_with_forcing_enabled() {
    let grammar = format!(
        r#"
        %llguidance {{"lexer_backtracking": true}}
        start: T S
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        S: /b{{{LONG_LOOKAHEAD}}}d/
    "#
    );
    let input = format!("a{}d", "b".repeat(LONG_LOOKAHEAD));
    lark_str_test(&grammar, true, &input, true);
}

#[test]
fn test_lexer_backtracking_with_stop_suffix_and_max_tokens() {
    let prefix = format!("a{}d", "b".repeat(LONG_LOOKAHEAD));

    let stop_grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: T body "!"
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        body[stop="!"]: /.*/
    "#
    );
    lark_str_test(&stop_grammar, true, &format!("{prefix}!"), true);

    let suffix_grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: T body
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        body[suffix="!"]: /.*/
    "#
    );
    lark_str_test(&suffix_grammar, true, &format!("{prefix}!"), true);

    let max_tokens_grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: T body
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        body[max_tokens=100]: /b{{{LONG_LOOKAHEAD}}}d/
    "#
    );
    lark_str_test(&max_tokens_grammar, true, &prefix, true);
}

#[test]
#[should_panic(expected = "with_recognizer is unavailable with lexer_backtracking")]
fn test_lexer_backtracking_rejects_single_state_recognizer_api() {
    let mut parser = make_parser(spanning_token_grammar(), true).unwrap();
    parser.parser.with_recognizer(|_| ());
}

#[test]
fn test_lexer_backtracking_rollback_then_take_long_match() {
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: stem (S | X)
        stem[capture]: T
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        S: /b{{{LONG_LOOKAHEAD}}}d/
        X: "x"
    "#
    );
    let shorter = format!("a{}d", "b".repeat(LONG_LOOKAHEAD));
    let longest = format!("a{}cx", "b".repeat(LONG_LOOKAHEAD));

    let mut parser = make_parser(&grammar, true).unwrap();
    consume_text(&mut parser, &shorter);
    assert!(parser.is_accepting());
    assert_eq!(parser.get_capture("stem"), Some(&b"a"[..]));

    parser.rollback(1).unwrap();
    let consumed = parser.final_bytes().len();
    assert!(
        longest.as_bytes().starts_with(parser.final_bytes()),
        "rollback did not return to a common prefix: {:?}",
        String::from_utf8_lossy(parser.final_bytes())
    );
    consume_text(&mut parser, &longest[consumed..]);

    assert!(parser.is_accepting());
    assert_eq!(
        parser.get_capture("stem"),
        Some(&longest.as_bytes()[..longest.len() - 1])
    );
}

#[test]
fn test_lexer_backtracking_rollback_after_eos() {
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: T (S | X)
        T: /a|ab{{{LONG_LOOKAHEAD}}}c/
        S: /b{{{LONG_LOOKAHEAD}}}/
        X: "x"
    "#
    );
    let shorter = format!("a{}", "b".repeat(LONG_LOOKAHEAD));
    let longest = format!("a{}cx", "b".repeat(LONG_LOOKAHEAD));

    let mut parser = make_parser(&grammar, true).unwrap();
    consume_text(&mut parser, &shorter);
    assert!(parser.is_accepting());

    parser.rollback(1).unwrap();
    let consumed = parser.final_bytes().len();
    assert!(longest.as_bytes().starts_with(parser.final_bytes()));
    consume_text(&mut parser, &longest[consumed..]);
    assert!(parser.is_accepting());
}

#[test]
fn test_lexer_backtracking_rollback_across_multiple_backtracking_points() {
    let grammar = format!(
        r#"
        %llguidance {{"no_forcing": true, "lexer_backtracking": true}}
        start: first (S1 | X) second (S2 | Y)
        first[capture]: T1
        second[capture]: T2
        T1: /a|ab{{{LONG_LOOKAHEAD}}}c/
        S1: /b{{{LONG_LOOKAHEAD}}}d/
        X: "x"
        T2: /e|ef{{{LONG_LOOKAHEAD}}}g/
        S2: /f{{{LONG_LOOKAHEAD}}}h/
        Y: "y"
    "#
    );
    let both_shorter = format!(
        "a{}de{}h",
        "b".repeat(LONG_LOOKAHEAD),
        "f".repeat(LONG_LOOKAHEAD)
    );
    let second_long = format!(
        "a{}de{}gy",
        "b".repeat(LONG_LOOKAHEAD),
        "f".repeat(LONG_LOOKAHEAD)
    );
    let both_long = format!(
        "a{}cxe{}gy",
        "b".repeat(LONG_LOOKAHEAD),
        "f".repeat(LONG_LOOKAHEAD)
    );

    let mut parser = make_parser(&grammar, true).unwrap();
    consume_text(&mut parser, &both_shorter);
    assert!(parser.is_accepting());

    parser.rollback(1).unwrap();
    let consumed = parser.final_bytes().len();
    assert!(second_long.as_bytes().starts_with(parser.final_bytes()));
    consume_text(&mut parser, &second_long[consumed..]);
    assert!(parser.is_accepting());
    assert_eq!(
        parser.get_capture("second"),
        Some(&second_long.as_bytes()[LONG_LOOKAHEAD + 2..second_long.len() - 1])
    );

    while !both_long.as_bytes().starts_with(parser.final_bytes()) {
        assert!(parser.num_tokens() > 0);
        parser.rollback(1).unwrap();
    }
    let consumed = parser.final_bytes().len();
    consume_text(&mut parser, &both_long[consumed..]);
    assert!(parser.is_accepting());
    assert_eq!(
        parser.get_capture("first"),
        Some(&both_long.as_bytes()[..LONG_LOOKAHEAD + 2])
    );
    assert_eq!(
        parser.get_capture("second"),
        Some(&both_long.as_bytes()[LONG_LOOKAHEAD + 3..both_long.len() - 1])
    );
}

#[test]
fn test_lexer_backtracking_counts_branch_work_toward_item_limit() {
    let grammar = long_backtracking_grammar();
    let prefix = format!("a{}", "b".repeat(LONG_LOOKAHEAD));
    let mut factory = ParserFactory::new_simple(get_tok_env()).unwrap();
    factory.limits_mut().step_max_items = 1;
    let mut parser = factory
        .create_parser(TopLevelGrammar::from_lark(grammar))
        .unwrap();
    parser.start_without_prompt();
    for token in get_tok_env().tokenize(&prefix) {
        consume(&mut parser, token);
    }

    let error = parser.compute_mask().unwrap_err().to_string();
    assert!(error.contains("Too many items (limit 1; mask)"), "{error}");
}

#[test]
fn test_lexer_backtracking_stats_stay_cumulative_after_rollback() {
    let input = format!("a{}d", "b".repeat(LONG_LOOKAHEAD));
    let mut parser = make_parser(&long_backtracking_grammar(), true).unwrap();
    consume_text(&mut parser, &input);

    let before = parser.parser_stats().clone();
    parser.rollback(1).unwrap();
    let after = parser.parser_stats();
    assert!(after.all_items >= before.all_items);
    assert!(after.rows >= before.rows);
    assert!(after.definitive_bytes >= before.definitive_bytes);
    assert!(after.trie_nodes_walked >= before.trie_nodes_walked);
}

#[test]
fn test_lexer_backtracking_temperature_includes_backtracked_state() {
    let grammar = r#"
        %llguidance {"no_forcing": true, "lexer_backtracking": true}
        start: t s
        t[temperature=0.1]: /a|abc/
        s[temperature=0.9]: /by/
    "#;
    let mut parser = make_parser(grammar, true).unwrap();
    consume_text(&mut parser, "ab");
    let _ = parser.compute_mask().unwrap();
    assert_eq!(parser.temperature(), Some(0.9));
}

#[test]
fn test_lexer_backtracking_temperature_is_ready_before_mask() {
    let grammar = r#"
        %llguidance {"no_forcing": true, "lexer_backtracking": true}
        start: t s
        t[temperature=0.1]: /a|abc/
        s[temperature=0.9]: /xy/
    "#;
    let mut parser = make_parser(grammar, true).unwrap();
    let token = get_tok_env().tokenize("a")[0];
    let mask = parser.compute_mask().unwrap();
    assert!(mask.is_allowed(token));
    consume(&mut parser, token);
    assert_eq!(parser.temperature(), Some(0.9));
}

#[test]
fn test_lexer_backtracking_token_ranges_include_backtracked_state() {
    let grammar = r#"
        %llguidance {"no_forcing": true, "lexer_backtracking": true}
        start: T S hd
        T: /a|abc/
        S: "b"
        hd: <[32006]>
    "#;
    let mut parser = make_parser(grammar, true).unwrap();
    consume_text(&mut parser, "ab");
    let mask = parser.compute_mask().unwrap();
    assert!(mask.is_allowed(32006));
    assert_eq!(parser.validate_tokens_raw(&[32006]).unwrap(), 1);
    assert_eq!(parser.final_bytes(), b"ab");
}

fn assert_finite_language_masks(grammar: &str, words: &[&str]) {
    let env = get_tok_env();
    let trie = env.tok_trie();
    let mut prefixes = BTreeSet::from([String::new()]);
    for word in words {
        prefixes.extend((0..=word.len()).map(|end| word[..end].to_string()));
    }

    let template = make_parser(grammar, true).unwrap();
    for prefix in prefixes {
        let mut parser = template.clone();
        for byte in prefix.chars() {
            let token = env.tokenize(&byte.to_string())[0];
            let mask = parser.compute_mask().unwrap();
            assert!(mask.is_allowed(token));
            consume(&mut parser, token);
        }
        let mask = parser.compute_mask().unwrap();
        for token in 0..trie.vocab_size() as u32 {
            let expected = if trie.eos_tokens().contains(&token) {
                words.contains(&prefix.as_str())
            } else {
                let token_bytes = trie.token(token);
                !token_bytes.is_empty()
                    && words.iter().any(|word| {
                        word.as_bytes()
                            .strip_prefix(prefix.as_bytes())
                            .is_some_and(|rest| rest.starts_with(token_bytes))
                    })
            };
            assert_eq!(
                mask.is_allowed(token),
                expected,
                "prefix={prefix:?} token={}: {}",
                trie.token_dbg(token),
                trie.token_set_dbg(&mask)
            );
        }
    }
}

#[test]
fn test_lexer_backtracking_prefix_masks_match_finite_languages() {
    assert_finite_language_masks(
        r#"
            %llguidance {"no_forcing": true, "lexer_backtracking": true}
            start: T S
            T: /a|abc/
            S: /x|by/
        "#,
        &["ax", "aby", "abcx", "abcby"],
    );
    assert_finite_language_masks(
        r#"
            %llguidance {"no_forcing": true, "lexer_backtracking": true}
            start: T U V
            T: /a|abdx/
            U: /b|bde/
            V: /df/
        "#,
        &["abdf", "abdedf", "abdxbdf", "abdxbdedf"],
    );
    assert_finite_language_masks(
        r#"
            %llguidance {"no_forcing": true, "lexer_backtracking": true}
            start: T S
            T: /a|aa|aaa|aaabc/
            S: /x|bdy/
        "#,
        &[
            "ax", "abdy", "aax", "aabdy", "aaax", "aaabdy", "aaabcx", "aaabcbdy",
        ],
    );
}

#[test]
fn test_lexer_backtracking_within_single_token() {
    let grammar = spanning_token_grammar();
    let env = get_tok_env();
    let tokens = env.tokenize("aby");
    assert_eq!(
        tokens.len(),
        1,
        "test requires a token spanning the backtracking point"
    );
    let mut parser = make_parser(grammar, true).unwrap();
    let mask = parser.compute_mask().unwrap();
    assert!(
        mask.is_allowed(tokens[0]),
        "{}",
        env.tok_trie().token_set_dbg(&mask)
    );
    consume(&mut parser, tokens[0]);
    assert!(parser.is_accepting());
}

#[test]
fn test_lexer_backtracking_twice_within_single_token() {
    let grammar = r#"
        %llguidance {"no_forcing": true, "lexer_backtracking": true}
        start: T U V
        T: /a|abx/
        U: /b|bix/
        V: /ility/
    "#;
    let env = get_tok_env();
    let tokens = env.tokenize("ability");
    assert_eq!(
        tokens.len(),
        1,
        "test requires two backtracking points inside one token"
    );
    let mut parser = make_parser(grammar, true).unwrap();
    let mask = parser.compute_mask().unwrap();
    assert!(
        mask.is_allowed(tokens[0]),
        "{}",
        env.tok_trie().token_set_dbg(&mask)
    );
    assert_eq!(parser.validate_tokens_raw(&tokens).unwrap(), 1);
    consume(&mut parser, tokens[0]);
    assert!(parser.is_accepting());
}

#[test]
fn test_lexer_backtracking_validates_single_spanning_token() {
    let grammar = spanning_token_grammar();
    let tokens = get_tok_env().tokenize("aby");
    assert_eq!(
        tokens.len(),
        1,
        "test requires a token spanning the backtracking point"
    );
    let mut parser = make_parser(grammar, true).unwrap();
    let before = parser.final_bytes().to_vec();
    assert_eq!(parser.validate_tokens_raw(&tokens).unwrap(), 1);
    let mut with_eos = tokens.clone();
    with_eos.push(get_tok_env().tok_trie().eos_token());
    assert_eq!(parser.validate_tokens_raw(&with_eos).unwrap(), 2);
    assert_eq!(parser.final_bytes(), before);
}

#[test]
fn test_lexer_backtracking_rolls_back_single_spanning_token() {
    let grammar = spanning_token_grammar();
    let env = get_tok_env();
    let short = env.tokenize("aby");
    assert_eq!(
        short.len(),
        1,
        "test requires a token spanning the backtracking point"
    );
    let mut parser = make_parser(grammar, true).unwrap();
    let mask = parser.compute_mask().unwrap();
    assert!(mask.is_allowed(short[0]));
    consume(&mut parser, short[0]);
    assert!(parser.is_accepting());

    parser.rollback(1).unwrap();
    assert!(parser.final_bytes().is_empty());
    consume_text(&mut parser, "abcby");
    assert!(parser.is_accepting());
}
