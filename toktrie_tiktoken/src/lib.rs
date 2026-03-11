use anyhow::{bail, Result};
use std::sync::Arc;
use tiktoken_rs::{CoreBPE, Rank};
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

pub struct TikTokenBPE {
    pub bpe: CoreBPE,
    tok_trie: TokTrie,
}

impl TikTokenBPE {
    pub fn new(
        encoder: Vec<(Vec<u8>, Rank)>,
        special_tokens_encoder: Vec<(String, Rank)>,
        pattern: &str,
        n_vocab_override: Option<usize>,
        eos_token: u32,
    ) -> Result<TikTokenBPE> {
        let mut n_vocab = encoder.len() + special_tokens_encoder.len();
        let mut tokens = vec![vec![]; n_vocab];

        for (bytes, idx) in encoder.iter() {
            while tokens.len() <= *idx as usize {
                tokens.push(vec![]);
            }
            tokens[*idx as usize] = bytes.clone();
        }

        for (name, idx) in special_tokens_encoder.iter() {
            while tokens.len() <= *idx as usize {
                tokens.push(vec![]);
            }
            let mut spec_bytes = Vec::with_capacity(name.len() + 1);
            spec_bytes.push(TokTrie::SPECIAL_TOKEN_MARKER);
            spec_bytes.extend_from_slice(name.as_bytes());
            tokens[*idx as usize] = spec_bytes;
        }

        n_vocab = tokens.len();

        if let Some(n_vocab_override) = n_vocab_override {
            if n_vocab_override < n_vocab {
                bail!("vocab size too small; {} vs {}", n_vocab_override, n_vocab);
            }
            n_vocab = n_vocab_override;
            tokens.resize(n_vocab, vec![]);
        }

        for (i, token) in tokens.iter_mut().enumerate() {
            if token.is_empty() {
                let mut name = format!(".<[{i}]>").into_bytes();
                name[0] = TokTrie::SPECIAL_TOKEN_MARKER;
                *token = name;
            }
        }

        let tok_trie = TokTrie::from(
            &TokRxInfo {
                vocab_size: n_vocab as u32,
                tok_eos: eos_token,
                tok_end_of_turn: None,
                tok_unk: None,
                tok_pad: None,
                tok_bos: None,
            },
            &tokens,
        );

        let bpe = CoreBPE::new(
            encoder.into_iter().collect(),
            special_tokens_encoder.into_iter().collect(),
            pattern,
        )?;

        Ok(TikTokenBPE { bpe, tok_trie })
    }

    pub fn tokrx_info(&self) -> TokRxInfo {
        *self.tok_trie.info()
    }

    pub fn set_eos_tokens(&mut self, tokens: &[TokenId]) {
        self.tok_trie = self.tok_trie.with_eos_tokens(tokens);
    }

    pub fn to_env(self) -> TokEnv {
        Arc::new(self)
    }
}

impl TokenizerEnv for TikTokenBPE {
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie
            .tokenize_with_greedy_fallback(s, |s| self.bpe.encode_ordinary(s))
    }

    fn tokenize_bytes_special(&self, s: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(s, |s| {
            self.tok_trie
                .tokenize_with_special(s, |s| self.bpe.encode_ordinary(s))
        })
    }
}
