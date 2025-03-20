use crate::{earley::SlicedBiasComputer, ParserFactory};
use anyhow::Result;
use std::sync::Arc;
use toktrie::{InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenizerEnv};

#[cxx::bridge(namespace = "llguidance")]
mod ffi {
    unsafe extern "C++" {
        include!("rust/cxx.h");
        include!("llguidance_cxx_support.h");

        type FactoryInit;

        fn vocab_size(self: &FactoryInit) -> usize;
        fn tok_eos(self: &FactoryInit) -> u32;
        fn token_bytes(self: &FactoryInit, token: usize) -> Vec<u8>;
        fn tokenize(self: &FactoryInit, text: &str) -> Vec<u32>;
        fn slices(self: &FactoryInit) -> Vec<String>;
        fn allow_ff_tokens(self: &FactoryInit) -> bool;
        fn allow_backtracking(self: &FactoryInit) -> bool;
        fn stderr_log_level(self: &FactoryInit) -> u32;
    }

    extern "Rust" {
        type ParserFactory;

        fn parser_factory(tok_init: UniquePtr<FactoryInit>) -> Result<Box<ParserFactory>>;

        /// Returns slices applicable for general grammars.
        /// Currently the same as `json_slices`.
        fn general_slices() -> Vec<String>;

        /// Returns slices applicable for JSON schemas.
        fn json_slices() -> Vec<String>;
    }
}

struct CTokenizer {
    trie: TokTrie,
    tokenize_is_canonical: bool,
    init: cxx::UniquePtr<ffi::FactoryInit>,
}
unsafe impl Send for CTokenizer {}
unsafe impl Sync for CTokenizer {}

impl TokenizerEnv for CTokenizer {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<toktrie::TokenId> {
        if self.tokenize_is_canonical {
            self.trie
                .tokenize_with_greedy_fallback(s, |s| self.init.tokenize(s))
        } else {
            self.trie.greedy_tokenize(s)
        }
    }

    fn tokenize_is_canonical(&self) -> bool {
        self.tokenize_is_canonical
    }
}

fn parser_factory(init: cxx::UniquePtr<ffi::FactoryInit>) -> Result<Box<ParserFactory>> {
    let mut tokens = vec![];
    for tok in 0..init.vocab_size() {
        tokens.push(init.token_bytes(tok));
    }
    let trie = TokTrie::from(
        &TokRxInfo {
            tok_eos: init.tok_eos(),
            vocab_size: tokens.len() as u32,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        },
        &tokens,
    );
    let tokenize_is_canonical = init.tokenize("foobar").len() > 0;
    let slices = init.slices();
    let stderr_log_level = init.stderr_log_level();
    let caps = InferenceCapabilities {
        ff_tokens: init.allow_ff_tokens(),
        backtrack: init.allow_backtracking(),
        ..Default::default()
    };
    let tok_env: TokEnv = Arc::new(CTokenizer {
        trie,
        tokenize_is_canonical,
        init,
    });
    let mut factory = ParserFactory::new(&tok_env, caps, &slices)?;
    factory.set_stderr_log_level(stderr_log_level);
    factory.set_buffer_log_level(0);

    Ok(Box::new(factory))
}

fn general_slices() -> Vec<String> {
    SlicedBiasComputer::general_slices()
}

fn json_slices() -> Vec<String> {
    SlicedBiasComputer::json_slices()
}
