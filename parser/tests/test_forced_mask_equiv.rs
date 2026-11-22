//! Masks inside a long forced run give the same tokens as compute_ff_tokens().
//!
//! compute_mask() tokenizes only a window of the forced bytes ahead. These tests walk forced
//! runs that are longer than that window, one mask and one token at a time. The walk must
//! give exactly the tokens that compute_ff_tokens() gives for the whole run at the start.
//! The runs put special tokens, multi-byte characters, non-UTF-8 bytes and multi-byte
//! tokens at the points where the window cuts the forced bytes.

use std::sync::Arc;

use llguidance::{
    api::TopLevelGrammar,
    toktrie::{
        ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv,
    },
    Matcher, ParserFactory,
};
use serde_json::{json, Map, Value};

fn factory(env: &TokEnv) -> ParserFactory {
    let mut f = ParserFactory::new(env, InferenceCapabilities::default(), &[]).unwrap();
    f.quiet();
    f
}

fn lark_matcher(f: &ParserFactory, grammar: &str) -> Matcher {
    let m = Matcher::new(f.create_parser(TopLevelGrammar::from_lark(grammar.to_string())));
    assert!(!m.is_error(), "{:?}", m.get_error());
    m
}

/// Takes the token of a mask that must hold exactly one token.
fn forced_token(m: &mut Matcher, trie: &TokTrie, step: usize) -> TokenId {
    let mask = m
        .compute_mask()
        .unwrap_or_else(|e| panic!("step {step}: compute_mask() failed: {e}"));
    assert_eq!(
        mask.num_set(),
        1,
        "step {step}: mask has {} tokens, first {:?}",
        mask.num_set(),
        mask.first_bit_set().map(|t| trie.token_dbg(t as TokenId))
    );
    mask.first_bit_set().unwrap() as TokenId
}

/// Walks the forced run mask by mask and compares the tokens with compute_ff_tokens().
/// Returns the number of forced tokens.
fn walk_matches_ff_tokens(m: &mut Matcher, trie: &TokTrie) -> usize {
    let expected = m.deep_clone().compute_ff_tokens();
    let mut got = Vec::new();
    while got.len() < expected.len() {
        let t = forced_token(m, trie, got.len());
        got.push(t);
        m.consume_token(t)
            .unwrap_or_else(|e| panic!("step {}: consume_token() failed: {e}", got.len()));
    }
    assert_eq!(trie.tokens_dbg(&got), trie.tokens_dbg(&expected));
    assert_eq!(got, expected);
    expected.len()
}

#[test]
fn forced_run_masks_match_ff_tokens() {
    let env = ApproximateTokEnv::single_byte_env();
    let f = factory(&env);
    // Each part is longer than the window, so the end of the window moves over each
    // special token and each multi-byte character, one token per step.
    let head = "#".repeat(1100);
    let body = "h\u{e9}llo w\u{f6}rld 12345 \u{1f600} ".repeat(20);
    let grammar = format!("start: \"{head}\" <|tool|> \"{body}\" <|user|> \"{head}\" /[a-z]+/");
    let mut m = lark_matcher(&f, &grammar);
    let n = walk_matches_ff_tokens(&mut m, env.tok_trie());
    assert!(n > 2500, "{n} forced tokens");
    // after the run, the mask is free again
    assert!(m.compute_mask().unwrap().num_set() > 1);
}

/// A forced run of 1,100 UTF-8 continuation bytes, longer than the window.
#[test]
fn non_utf8_forced_run_masks_match_ff_tokens() {
    let env = ApproximateTokEnv::single_byte_env();
    let f = factory(&env);
    let mut m = lark_matcher(
        &f,
        "%llguidance {\"allow_invalid_utf8\": true}\nstart: \"a\" /\\x80{1100}/ \"b\" /[a-z]+/",
    );
    let n = walk_matches_ff_tokens(&mut m, env.tok_trie());
    assert!(n > 1100, "{n} forced tokens");
}

/// The same, after one consumed token, so that the last token is part of the tokenized bytes.
#[test]
fn non_utf8_forced_run_after_first_token() {
    let env = ApproximateTokEnv::single_byte_env();
    let f = factory(&env);
    let mut m = lark_matcher(
        &f,
        "%llguidance {\"allow_invalid_utf8\": true}\nstart: /[a-z]/ /\\x80{1100}/ \"b\" /[a-z]+/",
    );
    m.consume_token(b'q' as TokenId).unwrap();
    let n = walk_matches_ff_tokens(&mut m, env.tok_trie());
    assert!(n > 1100, "{n} forced tokens");
}

/// A canonical greedy tokenizer with multi-byte tokens, so that tokens cross the window cut.
fn multibyte_env() -> TokEnv {
    let mut words = (0..=255u8).map(|x| vec![x]).collect::<Vec<_>>();
    for w in [
        "ab", "abab", "abababab", "ba", "bab", "aaa", "aaaaaa", "xy", "xyz", "yz", "zx", "zxyz",
        "\u{e9}l", "llo", " w", "\u{f6}r", "12", "345",
    ] {
        words.push(w.as_bytes().to_vec());
    }
    words.push(b"\xFF<|tool|>".to_vec());
    words.push(b"\xFF<|end|>".to_vec());
    let info = TokRxInfo {
        vocab_size: words.len() as u32,
        tok_eos: words.len() as u32 - 1,
        tok_bos: None,
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    Arc::new(Canonical(ApproximateTokEnv::new(TokTrie::from(
        &info, &words,
    ))))
}

struct Canonical(ApproximateTokEnv);

impl TokenizerEnv for Canonical {
    fn tok_trie(&self) -> &TokTrie {
        self.0.tok_trie()
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.0.tokenize_bytes(s)
    }

    fn tokenize_is_canonical(&self) -> bool {
        true
    }
}

#[test]
fn multibyte_greedy_masks_match_ff_tokens() {
    let env = multibyte_env();
    let f = factory(&env);
    let body = "abababababa xyzxyzx h\u{e9}llo w\u{f6}rld 12345 aaaaaaaa ".repeat(60);
    let mut m = lark_matcher(
        &f,
        &format!("start: \"{body}\" <|tool|> \"{body}\" /[a-z]+/"),
    );
    let n = walk_matches_ff_tokens(&mut m, env.tok_trie());
    assert!(n > 1500, "{n} forced tokens");
}

/// A walk with rollbacks: every 7 tokens, roll back 3 tokens and walk them again.
#[test]
fn forced_run_rollback_walk_matches_ff_tokens() {
    let env = multibyte_env();
    let trie = env.tok_trie();
    let f = factory(&env);
    let body = "abababababa xyzxyzx h\u{e9}llo w\u{f6}rld 12345 aaaaaaaa ".repeat(50);
    let mut m = lark_matcher(&f, &format!("start: \"{body}\" /[a-z]+/"));
    let expected = m.deep_clone().compute_ff_tokens();
    assert!(expected.len() > 1000, "{} forced tokens", expected.len());
    let mut pos = 0;
    let mut last_rollback = 0;
    while pos < expected.len() {
        let t = forced_token(&mut m, trie, pos);
        assert_eq!(
            t,
            expected[pos],
            "step {pos}: {} instead of {}",
            trie.token_dbg(t),
            trie.token_dbg(expected[pos])
        );
        m.consume_token(t).unwrap();
        pos += 1;
        if pos % 7 == 0 && pos > last_rollback {
            last_rollback = pos;
            m.rollback(3).unwrap();
            pos -= 3;
        }
    }
}

/// A real BPE tokenizer (Phi-3.5) on about 4.8 KB of forced JSON text.
#[test]
fn bpe_forced_run_masks_match_ff_tokens() {
    let env = llg_test_utils::get_tok_env();
    let f = factory(env);
    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, \
                sed do eiusmod tempor  incididunt ut labore et dolore magna aliqua. \
                Caf\u{e9} na\u{ef}ve \u{fc}ber \u{1f600} 12345.678 ";
    let props: Map<String, Value> = (0..3)
        .map(|i| (format!("field_{i}"), json!({ "const": text.repeat(10) })))
        .collect();
    let required: Vec<String> = props.keys().cloned().collect();
    let schema = json!({
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": false,
        "x-guidance": { "whitespace_flexible": false },
    });
    let mut m = Matcher::new(f.create_parser(TopLevelGrammar::from_json_schema(schema)));
    assert!(!m.is_error(), "{:?}", m.get_error());
    let ff_bytes = m.deep_clone().compute_ff_bytes().len();
    assert!(ff_bytes > 4 * 1024, "{ff_bytes} forced bytes");
    walk_matches_ff_tokens(&mut m, env.tok_trie());
}
