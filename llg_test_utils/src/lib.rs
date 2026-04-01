/// Test utilities for llguidance grammar testing.
///
/// This crate provides shared test infrastructure used by integration tests
/// in the `parser` crate and by the `json_schema_test_suite` binary. It is
/// **not** intended as an example of how to use llguidance in production.
/// For that, see the `sample_parser` crate.
///
/// The crate is split into two submodules by testing strategy:
///
/// - [`trace_check`]: **Trace-replay testing** – given a grammar and a recorded
///   sequence of forced / generated tokens, step the parser forward one cycle at
///   a time and verify every intermediate mask and forced-token result.
///   Entry points: [`check_lark_grammar`], [`check_lark_grammar_nested`],
///   [`check_lark_json`], [`check_capture`].
///
/// - [`acceptance`]: **Acceptance testing** – tokenize a complete input string,
///   feed the tokens one-by-one, and assert accept / reject at the expected
///   point.  Entry points: [`lark_str_test`], [`json_schema_check`],
///   [`json_test_many`], [`lark_ok`], [`lark_err_test`].
///
/// Shared state lives here in `lib.rs`:
/// - [`PARSER_FACTORY`]: A [`ParserFactory`] with the Phi-3.5-mini-instruct
///   tokenizer, configured for testing (ff_tokens + backtrack enabled, verbose
///   logging).
/// - [`get_tok_env`] / [`get_parser_factory`]: Accessors for the above.
use std::sync::atomic::{AtomicBool, Ordering};

use lazy_static::lazy_static;
use llguidance::{
    earley::SlicedBiasComputer,
    toktrie::{InferenceCapabilities, TokEnv},
    ParserFactory,
};

// ── Global quiet-mode flag ───────────────────────────────────────────────────

static QUIET_MODE: AtomicBool = AtomicBool::new(false);

/// When true, functions like [`json_schema_check`] suppress per-token parser
/// logging.  Defaults to `false` (verbose), preserving existing behaviour for
/// `cargo test` callers where output is captured anyway.
pub fn set_quiet_mode(quiet: bool) {
    QUIET_MODE.store(quiet, Ordering::Relaxed);
}

pub fn is_quiet_mode() -> bool {
    QUIET_MODE.load(Ordering::Relaxed)
}

mod acceptance;
pub mod rng_utils;
mod trace_check;

#[allow(unused_imports)]
pub use acceptance::*;
pub use rng_utils::*;
#[allow(unused_imports)]
pub use trace_check::*;

// ── Shared parser factory ────────────────────────────────────────────────────

lazy_static! {
    static ref PARSER_FACTORY: ParserFactory = {
        let env =
            toktrie_hf_downloader::byte_tokenizer_from_name("microsoft/Phi-3.5-mini-instruct")
            .unwrap()
            .into_tok_env(Some(35000))
            .unwrap();
        let mut fact = ParserFactory::new(&env,
            InferenceCapabilities {
                ff_tokens: true, // can the engine append multiple tokens?
                backtrack: true, // can the engine remove generated tokens?
                conditional_ff_tokens: false, // not used
                fork: false,                  // not used
            }, &SlicedBiasComputer::general_slices()).unwrap();
        fact.set_stderr_log_level(2);
        fact.set_buffer_log_level(0);
        fact
    };
}

pub fn get_tok_env() -> &'static TokEnv {
    PARSER_FACTORY.tok_env()
}

pub fn get_parser_factory() -> &'static ParserFactory {
    &PARSER_FACTORY
}
