use std::sync::Arc;

use anyhow::{ensure, Result};
use llguidance::toktrie::{self, TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

type LlamaTokenizeFn = unsafe extern "C" fn(
    vocab: *const std::os::raw::c_void,
    text: *const std::os::raw::c_char,
    text_len: i32,
    tokens: *mut i32,
    n_tokens_max: i32,
    add_special: bool,
    parse_special: bool,
) -> i32;

struct LlamaTokenizer {
    trie: TokTrie,
    tokenize_fn: LlamaTokenizeFn,
    vocab: *const std::os::raw::c_void,
    sentinel: Option<u8>,
    sentinel_tokens: Vec<TokenId>,
}
// SAFETY: tokenize_fn is required to be thread-safe
unsafe impl Send for LlamaTokenizer {}
unsafe impl Sync for LlamaTokenizer {}

impl LlamaTokenizer {
    fn tokenize_with_sentinel(
        &self,
        s: &[u8],
        parse_special: bool,
    ) -> Result<Vec<toktrie::TokenId>> {
        if s.is_empty() {
            return Ok(vec![]);
        }

        if let Some(sentinel) = self.sentinel {
            let mut b = Vec::with_capacity(s.len() + 1);
            b.push(sentinel);
            b.extend_from_slice(s);
            let mut res = self.raw_tokenize(&b, parse_special);
            ensure!(
                res.len() > self.sentinel_tokens.len(),
                "tokenize_with_sentinel: res.len() <= sentinel_tokens.len()"
            );
            ensure!(
                res[0..self.sentinel_tokens.len()] == self.sentinel_tokens,
                "tokenize_with_sentinel: res[0..sentinel_tokens.len()] != sentinel_tokens"
            );
            res.splice(0..self.sentinel_tokens.len(), []);
            Ok(res)
        } else {
            Ok(self.raw_tokenize(s, parse_special))
        }
    }

    fn raw_tokenize(&self, s: &[u8], parse_special: bool) -> Vec<toktrie::TokenId> {
        let mut res_toks = vec![0u32; s.len() / 4 + 5];
        let res = unsafe {
            (self.tokenize_fn)(
                self.vocab,
                s.as_ptr() as *const std::os::raw::c_char,
                s.len().try_into().unwrap(),
                res_toks.as_mut_ptr() as *mut i32,
                res_toks.len().try_into().unwrap(),
                false,
                parse_special,
            )
        };

        let res = if res < 0 {
            let n_toks = (-res) as usize;
            res_toks.resize(n_toks, 0);
            let res2 = unsafe {
                (self.tokenize_fn)(
                    self.vocab,
                    s.as_ptr() as *const std::os::raw::c_char,
                    s.len().try_into().unwrap(),
                    res_toks.as_mut_ptr() as *mut i32,
                    res_toks.len().try_into().unwrap(),
                    false,
                    parse_special,
                )
            };
            assert!(res2 == n_toks as i32);
            res2
        } else {
            res
        };

        res_toks.truncate(res as usize);
        res_toks
    }
}

impl TokenizerEnv for LlamaTokenizer {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<toktrie::TokenId> {
        // llama.cpp tokenizer encodes invalid UTF8 as Unicode replacement character U+FFFD,
        // so we need the greedy fallback
        self.trie.tokenize_with_greedy_fallback(s, |s| {
            self.tokenize_with_sentinel(s.as_bytes(), false)
                .expect("tokenize_with_sentinel failed")
        })
    }

    fn tokenize_bytes_special(&self, s: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(s, |s| {
            self.tokenize_with_sentinel(s.as_bytes(), true)
                .expect("tokenize_with_sentinel failed")
        })
    }
}

pub fn tokenv_from_llamacpp(
    tokens: Vec<Vec<u8>>,
    vocab_ptr: usize,
    tokenize_fptr: usize,
    eos_token: u32,
) -> Result<TokEnv> {
    ensure!(vocab_ptr != 0, "vocab_ptr must be non-null");
    ensure!(tokenize_fptr != 0, "tokenize_fptr must be non-null");

    let info = TokRxInfo::new(tokens.len() as u32, eos_token);
    let trie = TokTrie::from(&info, &tokens);

    let mut llama_tok = LlamaTokenizer {
        trie,
        tokenize_fn: unsafe { std::mem::transmute::<usize, LlamaTokenizeFn>(tokenize_fptr) },
        vocab: vocab_ptr as *const std::os::raw::c_void,
        sentinel: None,
        sentinel_tokens: vec![],
    };

    let trie = &llama_tok.trie;
    let t0 = llama_tok.raw_tokenize(b"a", false);
    if trie.decode(&t0) != b"a" {
        // Now, this likely means that the tokenizer is adding a space in front of the token
        // (or possibly <BOS> token)
        // We fill "fix" this by tokenizing [sentinel] + s instead of just s
        // and then removing tokens corresponding to the sentinel

        // find a good sentinel token - one that doesn't start any other token
        let sentinel = (1u8..32)
            .find(|&b| {
                trie.token_id(&[b]).is_some()
                    && !trie.has_extensions(&[b])
                    && !trie.has_extensions(&[b' ', b])
            })
            .ok_or_else(|| {
                anyhow::anyhow!("could not find a good sentinel token in the range 1..32")
            })?;

        llama_tok.sentinel_tokens = llama_tok.raw_tokenize(&[sentinel], false);
        llama_tok.sentinel = Some(sentinel);

        // now, check if it works
        let t1 = llama_tok.tokenize_with_sentinel(b"a", false)?;
        ensure!(
            trie.decode(&t1) == b"a",
            "tokenizer is not working with the sentinel {} {:?}",
            sentinel,
            trie.decode(&t1)
        );

        // make sure we can tokenize double-sentinel
        let t3 = llama_tok.tokenize_with_sentinel(&[sentinel], false)?;
        ensure!(
            trie.decode(&t3) == [sentinel],
            "tokenizer is not working with the sentinel (rec) {} {:?}",
            sentinel,
            trie.decode(&t3)
        );
    }

    Ok(Arc::new(llama_tok))
}
