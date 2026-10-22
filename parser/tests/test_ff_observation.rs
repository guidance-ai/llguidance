use std::sync::Arc;

use llguidance::{
    api::{StopReason, TopLevelGrammar},
    toktrie::{
        ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv,
    },
    Cancelled, FastForwardStatus, Matcher, ParserFactory, TokenParser,
};

fn parser_with_env(grammar: &str, env: &TokEnv) -> TokenParser {
    let factory = ParserFactory::new(env, InferenceCapabilities::default(), &[]).unwrap();
    let mut parser = factory
        .create_parser(TopLevelGrammar::from_lark(grammar.to_string()))
        .unwrap();
    parser.start_without_prompt();
    parser
}

fn parser(grammar: &str) -> TokenParser {
    parser_with_env(grammar, &ApproximateTokEnv::single_byte_env())
}

fn consume(parser: &mut TokenParser, tokens: &[TokenId]) {
    for &token in tokens {
        assert_eq!(parser.consume_token(token).unwrap(), 0);
    }
    parser.check_stop().unwrap();
}

#[test]
fn repeated_observation_preserves_history_masks_and_language() {
    let mut parser = parser(r#"start: "abc" /[xy]/"#);
    let mut control = parser.deep_clone();
    let expected = vec![b'a' as u32, b'b' as u32, b'c' as u32];
    assert_eq!(parser.compute_ff_tokens(), expected);
    let observed = parser.observe_ff_tokens().unwrap();
    assert_eq!(observed.status, FastForwardStatus::Ready);
    assert_eq!(observed.tokens, expected);
    assert_eq!(observed.token_bytes, b"abc");
    assert_eq!((observed.token_count(), observed.byte_count()), (3, 3));
    for _ in 0..3 {
        assert_eq!(parser.observe_ff_tokens().unwrap(), observed);
        assert_eq!(parser.num_tokens(), 0);
        assert_eq!(parser.final_bytes(), b"");
    }
    for byte in b"abcx" {
        assert_eq!(
            parser.compute_mask().unwrap(),
            control.compute_mask().unwrap()
        );
        consume(&mut parser, &[*byte as u32]);
        consume(&mut control, &[*byte as u32]);
    }
    assert_eq!(parser.num_tokens(), 4);
    assert_eq!(parser.final_bytes(), control.final_bytes());
    assert_eq!(parser.final_bytes(), b"abcx");
    assert_eq!(parser.stop_reason(), control.stop_reason());
    assert_eq!(
        parser.observe_ff_tokens().unwrap().status,
        FastForwardStatus::Stopped
    );
}

#[test]
fn observation_tracks_consumption_rollback_reset_and_clones() {
    let mut parser = parser(r#"start: "abc" /[xy]/"#);
    let initial = parser.observe_ff_tokens().unwrap();
    consume(&mut parser, &[b'a' as u32]);
    let after_a = parser.observe_ff_tokens().unwrap();
    assert_eq!(after_a.token_bytes, b"bc");
    assert_eq!(parser.num_tokens(), 1);
    let mut clone = parser.clone();
    parser.rollback(1).unwrap();
    assert_eq!(parser.observe_ff_tokens().unwrap(), initial);
    assert_eq!(clone.observe_ff_tokens().unwrap(), after_a);
    consume(
        &mut parser,
        &[b'a' as u32, b'b' as u32, b'c' as u32, b'x' as u32],
    );
    parser.reset().unwrap();
    assert_eq!(parser.observe_ff_tokens().unwrap(), initial);
    assert_eq!(parser.num_tokens(), 0);
    assert_eq!(parser.final_bytes(), b"");
}

#[test]
fn observation_counts_utf8_bytes_and_incomplete_codepoints() {
    let mut parser = parser("start: \"é🙂\" /[xy]/");
    let initial = parser.observe_ff_tokens().unwrap();
    assert_eq!(initial.token_bytes, "é🙂".as_bytes());
    assert_eq!((initial.token_count(), initial.byte_count()), (6, 6));
    consume(&mut parser, &[0xc3]);
    let rest = parser.observe_ff_tokens().unwrap();
    assert_eq!(rest.token_bytes, &"é🙂".as_bytes()[1..]);
    assert_eq!(rest.byte_count(), 5);
    assert!(std::str::from_utf8(&rest.token_bytes).is_err());
    assert_eq!(parser.num_tokens(), 1);
}

struct GreedyEnv(TokTrie);

impl TokenizerEnv for GreedyEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.0
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.0.greedy_tokenize(bytes)
    }
}

#[test]
fn observation_excludes_prior_token_and_chopped_healing_suffix() {
    let mut words = (0..=255).map(|byte| vec![byte]).collect::<Vec<_>>();
    words.extend([
        b"".to_vec(),
        b"ab".to_vec(),
        b"cd".to_vec(),
        b"cdx".to_vec(),
        b"cdy".to_vec(),
    ]);
    let env: TokEnv = Arc::new(GreedyEnv(TokTrie::from(
        &TokRxInfo::new(words.len() as u32, 256),
        &words,
    )));
    let mut parser = parser_with_env(r#"start: "!abcd" /[xy]/"#, &env);
    consume(&mut parser, &[b'!' as u32]);
    let mut control = parser.deep_clone();
    let observed = parser.observe_ff_tokens().unwrap();
    assert_eq!(observed.tokens, vec![257]);
    assert_eq!(observed.token_bytes, b"ab");
    assert_eq!((observed.token_count(), observed.byte_count()), (1, 2));
    assert_eq!(parser.force_bytes(), b"abcd");
    assert_eq!(parser.final_bytes(), b"!");
    assert_eq!(
        parser.compute_mask().unwrap(),
        control.compute_mask().unwrap()
    );
    consume(&mut parser, &observed.tokens);
    assert!(parser.observe_ff_tokens().unwrap().tokens.is_empty());
    assert_eq!(parser.force_bytes(), b"cd");
    consume(&mut parser, &[259]);
    assert_eq!(parser.final_bytes(), b"!abcdx");
    assert!(parser.is_accepting());
}

#[test]
fn noncanonical_observation_is_distinct_from_empty_supported_span() {
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(
        ApproximateTokEnv::single_byte().tok_trie().clone(),
    ));
    let mut matcher = Matcher::new(Ok(parser_with_env(r#"start: "abc" /[xy]/"#, &env)));
    let mut control = matcher.deep_clone();
    let observed = matcher.observe_ff_tokens().unwrap();
    assert_eq!(observed.status, FastForwardStatus::NonCanonical);
    assert_eq!((observed.token_count(), observed.byte_count()), (0, 0));
    assert!(matcher.compute_ff_tokens().is_empty());
    assert_eq!(matcher.compute_ff_bytes(), b"abc");
    assert_eq!(
        matcher.compute_mask().unwrap(),
        control.compute_mask().unwrap()
    );
    matcher
        .consume_tokens(&[b'a' as u32, b'b' as u32, b'c' as u32, b'x' as u32])
        .unwrap();
    assert!(matcher.is_accepting().unwrap());

    let mut choice = Matcher::new(Ok(parser("start: /[xy]/")));
    let empty = choice.observe_ff_tokens().unwrap();
    assert_eq!(empty.status, FastForwardStatus::Ready);
    assert_eq!((empty.token_count(), empty.byte_count()), (0, 0));
}

#[test]
fn observation_uses_decoded_token_bytes_for_special_tokens() {
    let mut parser = parser("start: <[256]> /[xy]/");
    let observed = parser.observe_ff_tokens().unwrap();
    assert_eq!(observed.tokens, vec![256]);
    assert_eq!(observed.token_bytes, b"<|tool|>");
    assert_eq!((observed.token_count(), observed.byte_count()), (1, 8));
    assert_eq!(parser.num_tokens(), 0);
}

#[test]
fn observation_does_not_count_validation_or_partial_consumption_as_sampling() {
    let mut parser = parser(r#"start: "abc" /[xy]/"#);
    let first = parser.observe_ff_tokens().unwrap();
    assert_eq!(
        parser
            .validate_tokens_raw(&[b'a' as u32, b'b' as u32, b'c' as u32, b'y' as u32])
            .unwrap(),
        4
    );
    assert_eq!(parser.observe_ff_tokens().unwrap(), first);
    assert_eq!(parser.num_tokens(), 0);
    let mut matcher = Matcher::new(Ok(parser));
    assert_eq!(
        matcher
            .try_consume_tokens(&[b'a' as u32, b'b' as u32, b'!' as u32])
            .unwrap(),
        2
    );
    assert_eq!(matcher.observe_ff_tokens().unwrap().token_bytes, b"c");
    assert!(!matcher.is_error());
}

#[test]
fn errors_stay_terminal_and_keep_their_original_cause() {
    let mut matcher = Matcher::new(Ok(parser(r#"start: "abc""#))).into_cancellable();
    matcher.consume_token(999999).unwrap_err();
    let error = matcher.get_error().unwrap();
    matcher.cancellation_handle().unwrap().cancel();
    assert_eq!(matcher.observe_ff_tokens().unwrap_err().to_string(), error);
    assert!(!matcher.is_cancelled());
    assert!(matcher.compute_ff_tokens().is_empty());
    assert!(matcher.reset().is_err());
    assert!(matcher.rollback(0).is_err());
    assert_eq!(matcher.observe_ff_tokens().unwrap_err().to_string(), error);
}

#[test]
fn cancellation_is_terminal_but_prior_clones_are_independent() {
    let mut matcher = Matcher::new(Ok(parser(r#"start: "abc" /[xy]/"#))).into_cancellable();
    let initial = matcher.observe_ff_tokens().unwrap();
    let mut shallow = matcher.clone();
    let mut deep = matcher.deep_clone();
    matcher.cancellation_handle().unwrap().cancel();
    assert!(matcher.observe_ff_tokens().unwrap_err().is::<Cancelled>());
    assert_eq!(matcher.stop_reason(), StopReason::Cancelled);
    assert!(matcher.compute_ff_tokens().is_empty());
    assert!(matcher.reset().unwrap_err().is::<Cancelled>());
    assert!(matcher.rollback(0).unwrap_err().is::<Cancelled>());
    assert!(matcher
        .clone()
        .observe_ff_tokens()
        .unwrap_err()
        .is::<Cancelled>());
    assert_eq!(shallow.observe_ff_tokens().unwrap(), initial);
    assert_eq!(deep.observe_ff_tokens().unwrap(), initial);
}
