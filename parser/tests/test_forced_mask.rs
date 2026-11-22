//! Work done by compute_mask() inside a long forced run.
//!
//! Under canonical forcing, a grammar can force a long run of bytes. Here, a JSON schema of
//! required properties with long `const` values and no flexible whitespace forces its whole
//! document. Inside such a run, each mask holds one token, so a mask needs only a few
//! tokens' worth of the forced bytes. These tests check that a mask does not tokenize the
//! whole run ahead, that the logs of a run stay small, and that a force that stops on the
//! item limit makes compute_mask() return the error. The tests count bytes and check
//! errors; they do not measure time. The ignored tests print timings (run them with
//! `--release -- --ignored --nocapture`).

use std::sync::{
    atomic::{AtomicUsize, Ordering::Relaxed},
    Arc,
};
use std::time::Instant;

use llguidance::{
    api::TopLevelGrammar,
    toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokTrie, TokenId, TokenizerEnv},
    Matcher, ParserFactory,
};
use serde_json::{json, Map, Value};

/// The single-byte test tokenizer (canonical). It counts the bytes passed to `tokenize_bytes`.
struct CountingTokEnv {
    inner: ApproximateTokEnv,
    bytes: AtomicUsize,
}

impl TokenizerEnv for CountingTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        self.inner.tok_trie()
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.bytes.fetch_add(s.len(), Relaxed);
        self.inner.tokenize_bytes(s)
    }

    fn tokenize_is_canonical(&self) -> bool {
        true
    }
}

/// `n` required properties, each with the `const` string `value`. The whole document after
/// the opening brace is forced.
fn const_schema(n: usize, value: &str) -> Value {
    let props: Map<String, Value> = (0..n)
        .map(|i| (format!("p{i}"), json!({ "const": value })))
        .collect();
    let required: Vec<String> = props.keys().cloned().collect();
    json!({
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": false,
        "x-guidance": { "whitespace_flexible": false },
    })
}

/// `n` properties of 2,500 bytes each: about 2,510 forced bytes per property.
fn long_const_schema(n: usize) -> Value {
    const_schema(n, &"x".repeat(2500))
}

struct Setup {
    env: Arc<CountingTokEnv>,
    factory: ParserFactory,
}

fn setup() -> Setup {
    let env = Arc::new(CountingTokEnv {
        inner: ApproximateTokEnv::single_byte(),
        bytes: AtomicUsize::new(0),
    });
    let tok_env: TokEnv = env.clone();
    let mut factory = ParserFactory::new(&tok_env, InferenceCapabilities::default(), &[]).unwrap();
    factory.quiet();
    Setup { env, factory }
}

fn matcher_after_brace(s: &Setup, schema: Value) -> Matcher {
    let mut m = Matcher::new(
        s.factory
            .create_parser(TopLevelGrammar::from_json_schema(schema)),
    );
    assert!(!m.is_error(), "{:?}", m.get_error());
    m.consume_token(b'{' as TokenId).unwrap();
    m
}

/// Walks `steps` tokens along the forced run. Returns the bytes tokenized per mask and the
/// seconds per mask.
fn walk(s: &Setup, m: &mut Matcher, steps: usize) -> (usize, f64) {
    let before = s.env.bytes.load(Relaxed);
    let mut mask_time = 0.0;
    for _ in 0..steps {
        let t0 = Instant::now();
        let mask = m.compute_mask().unwrap();
        mask_time += t0.elapsed().as_secs_f64();
        assert_eq!(mask.num_set(), 1, "a forced run gives a singleton mask");
        let t = mask.first_bit_set().unwrap() as TokenId;
        m.consume_token(t).unwrap();
    }
    (
        (s.env.bytes.load(Relaxed) - before) / steps,
        mask_time / steps as f64,
    )
}

#[test]
fn forced_run_mask_tokenizes_bounded_bytes() {
    let s = setup();
    let mut m = matcher_after_brace(&s, long_const_schema(40));
    let forced = m.compute_ff_bytes().len();
    assert!(forced > 100_000, "forced run is {forced} bytes");
    let max_len = s.env.tok_trie().max_token_len();
    // The first mask may tokenize the run once. The masks after it must stay bounded.
    walk(&s, &mut m, 1);
    let (per_mask, _) = walk(&s, &mut m, 20);
    assert!(
        per_mask <= 8 * 1024 + 64 * max_len,
        "each mask tokenized {per_mask} bytes of a {forced}-byte forced run (max_token_len {max_len})"
    );
}

#[test]
fn forced_run_logs_stay_bounded() {
    let mut s = setup();
    s.factory.set_buffer_log_level(2);
    let grm = TopLevelGrammar::from_json_schema(long_const_schema(40));
    let mut p = s.factory.create_parser(grm).unwrap();
    p.start_without_prompt();
    p.consume_token(b'{' as TokenId).unwrap();
    p.logger.get_and_clear_logs();

    for _ in 0..5 {
        let mask = p.compute_mask().unwrap();
        assert_eq!(mask.num_set(), 1);
        p.consume_token(mask.first_bit_set().unwrap() as TokenId)
            .unwrap();
    }
    let ff = p.compute_ff_tokens();
    assert!(ff.len() > 100_000, "{} forced tokens", ff.len());

    let logs = p.logger.get_and_clear_logs();
    assert!(logs.contains("forced:"), "no forced-token log in: {logs}");
    assert!(
        logs.len() < 64 * 1024,
        "5 masks and 1 compute_ff_tokens() logged {} bytes",
        logs.len()
    );
}

#[test]
fn forced_run_item_limit_error_from_mask() {
    let mut s = setup();
    // Each short property adds Earley items. The force through all of them exceeds this
    // item limit, so the force stops with an error.
    s.factory.limits_mut().step_max_items = 500;
    let mut m = matcher_after_brace(&s, const_schema(300, "abcde"));

    // The mask must return the error, not a forced token that cannot be consumed.
    let err = match m.compute_mask() {
        Ok(mask) => panic!(
            "compute_mask() returned a mask of {} tokens; the next consume_token() gives: {:?}",
            mask.num_set(),
            m.consume_token(mask.first_bit_set().unwrap() as TokenId)
                .err()
        ),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("Too many items"), "{err}");
    assert!(m.is_error());
}

/// Prints how the mask cost changes with the length of the forced run.
#[test]
#[ignore]
fn forced_run_mask_cost_scaling() {
    let s = setup();
    for n in [25, 50, 100, 200, 400] {
        let mut m = matcher_after_brace(&s, long_const_schema(n));
        let t0 = Instant::now();
        walk(&s, &mut m, 1);
        let first = t0.elapsed().as_secs_f64();
        let forced = m.compute_ff_bytes().len();
        let (per_mask, secs) = walk(&s, &mut m, 50);
        let t0 = Instant::now();
        let ff = m.compute_ff_tokens();
        let ff_secs = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        m.consume_token(ff[0]).unwrap();
        let consume_secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "n={n:3} forced={forced:8} first_mask={:8.3}ms mask={:8.3}ms tokenized/mask={per_mask:8} \
             compute_ff_tokens={:8.3}ms ({} tokens) consume_token={:6.3}ms",
            first * 1e3,
            secs * 1e3,
            ff_secs * 1e3,
            ff.len(),
            consume_secs * 1e3,
        );
    }
    eprintln!("{}", s.factory.perf_counters());
}

/// Prints the first mask of a run, a later mask, and the masks after a rollback.
#[test]
#[ignore]
fn forced_run_first_mask_and_rollback_cost() {
    let s = setup();
    for n in [100, 400] {
        let mut m = matcher_after_brace(&s, long_const_schema(n));
        let t0 = Instant::now();
        let mask = m.compute_mask().unwrap();
        let first = t0.elapsed().as_secs_f64() * 1e3;
        m.consume_token(mask.first_bit_set().unwrap() as TokenId)
            .unwrap();
        let t0 = Instant::now();
        let mask = m.compute_mask().unwrap();
        let steady = t0.elapsed().as_secs_f64() * 1e3;
        // a speculative step: consume a draft token, then roll it back
        m.consume_token(mask.first_bit_set().unwrap() as TokenId)
            .unwrap();
        m.rollback(1).unwrap();
        let t0 = Instant::now();
        m.compute_mask().unwrap();
        let after_rollback = t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        m.compute_mask().unwrap();
        let next = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "n={n} first={first:.3}ms steady={steady:.3}ms after_rollback={after_rollback:.3}ms next={next:.3}ms"
        );
    }
}
