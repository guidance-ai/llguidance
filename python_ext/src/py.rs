use std::fmt::Display;
use std::{borrow::Cow, sync::Arc};

use llguidance::api::{GrammarInit, ParserLimits};
use llguidance::earley::SlicedBiasComputer;
use llguidance::toktrie::{
    self, ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId,
    TokenizerEnv,
};
use llguidance::{api::TopLevelGrammar, output::ParserOutput};
use llguidance::{token_bytes_from_tokenizer_json, JsonCompileOptions, ParserFactory};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use serde_json::Value;

struct PyTokenizer {
    tok_trie: Arc<toktrie::TokTrie>,
    tokenizer_fun: Py<PyAny>,
    #[allow(dead_code)]
    tok_bos: Option<u32>,
}

#[derive(Clone)]
#[pyclass]
pub(crate) struct LLTokenizer {
    factory: Arc<ParserFactory>,
}

#[derive(Serialize, Deserialize)]
struct PyMidProcessResult {
    progress: Vec<ParserOutput>,
    stop: bool,
    temperature: f32,
}

#[pymethods]
impl LLTokenizer {
    #[new]
    #[pyo3(signature = (tokenizer, n_vocab=None, eos_token=None, slices=None))]
    fn py_new(
        tokenizer: Bound<'_, PyAny>,
        n_vocab: Option<usize>,
        eos_token: Option<u32>,
        slices: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let tok_env: TokEnv = if let Ok(tokenizer_str) = tokenizer.extract::<String>() {
            if tokenizer_str.starts_with("{") {
                let val = serde_json::from_str(&tokenizer_str).map_err(val_error)?;
                let mut tokens = token_bytes_from_tokenizer_json(&val).map_err(val_error)?;
                if let Some(n_vocab) = n_vocab {
                    while tokens.len() < n_vocab {
                        tokens.push(vec![]);
                    }
                }
                let trie = TokTrie::from(&TokRxInfo::new(tokens.len() as u32, 0), &tokens);
                let candidates = &[
                    "<|im_end|>",
                    "<|eot_id|>",
                    "<|end_of_text|>",
                    "<｜end▁of▁sentence｜>", // deepseek-v3 - weird Unicode bars
                    "</s>",
                    "<|endoftext|>",
                ];
                let eos_token = if let Some(eos_token) = eos_token {
                    eos_token
                } else {
                    candidates
                        .iter()
                        .filter_map(|s| trie.get_special_token(s))
                        .next()
                        .ok_or_else(|| {
                            PyValueError::new_err(
                                "Expecting a tokenizer with an EOS token, but none was found"
                                    .to_string(),
                            )
                        })?
                };
                let trie = trie.with_eos_token(eos_token);
                Arc::new(ApproximateTokEnv::new(trie))
            } else {
                #[cfg(feature = "tokenizers")]
                {
                    let tok =
                        toktrie_hf_tokenizers::ByteTokenizerEnv::from_name(&tokenizer_str, n_vocab)
                            .map_err(val_error)?;
                    tok.to_env()
                }

                #[cfg(not(feature = "tokenizers"))]
                {
                    let _ = n_vocab;
                    return Err(PyValueError::new_err(
                        "Expecting a TokenizerWrapper() class or encoded HF-tokenizers JSON file",
                    ));
                }
            }
        } else {
            Arc::new(PyTokenizer::py_new(tokenizer)?)
        };
        let factory = ParserFactory::new(
            &tok_env,
            InferenceCapabilities::default(),
            &slices.unwrap_or_else(SlicedBiasComputer::general_slices),
        )
        .map_err(val_error)?;

        Ok(LLTokenizer {
            factory: Arc::new(factory),
        })
    }

    fn tokenize_bytes(&self, utf8bytes: &[u8]) -> Vec<TokenId> {
        self.tok_env().tokenize_bytes(utf8bytes)
    }

    fn tokenize_str(&self, text: &str) -> Vec<TokenId> {
        self.tokenize_bytes(text.as_bytes())
    }

    fn greedy_tokenize(&self, text: &str) -> Vec<u32> {
        self.tok_trie().greedy_tokenize(text.as_bytes())
    }

    fn is_special_token(&self, token: u32) -> bool {
        self.tok_trie().is_special_token(token)
    }

    fn test_trace_tokens(&self, tokens: Vec<u32>) -> String {
        self.tok_trie()
            .test_trace_tokens(&tokens)
            .replace("\\n", "\n")
    }

    fn dbg_tokens(&self, tokens: Vec<u32>) -> String {
        self.tok_trie().tokens_dbg(&tokens)
    }

    fn decode_str(&self, tokens: Vec<u32>) -> String {
        self.tok_trie().decode_str(&tokens)
    }

    fn decode_bytes(&self, tokens: Vec<u32>) -> Cow<[u8]> {
        let r = self.tok_trie().decode(&tokens);
        Cow::Owned(r)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.tok_trie().vocab_size()
    }

    #[getter]
    fn eos_token(&self) -> u32 {
        self.tok_trie().eos_token()
    }
}

impl LLTokenizer {
    pub fn tok_env(&self) -> &TokEnv {
        self.factory().tok_env()
    }
    pub fn tok_trie(&self) -> &toktrie::TokTrie {
        self.tok_env().tok_trie()
    }
    pub fn factory(&self) -> &ParserFactory {
        &self.factory
    }
}

impl PyTokenizer {
    fn py_new(tokenizer: Bound<'_, PyAny>) -> PyResult<Self> {
        let is_tokenizer = tokenizer
            .getattr("is_tokenizer_wrapper")
            .map(|v| v.extract::<bool>())
            .unwrap_or(Ok(false))
            .unwrap_or(false);
        if !is_tokenizer {
            return Err(PyValueError::new_err(
                "Expecting a TokenizerWrapper() class",
            ));
        }

        let mut tokens = tokenizer.getattr("tokens")?.extract::<Vec<Vec<u8>>>()?;

        // no eos_token only applies to ByteTokenizer from Guidance, which we
        // hopefully will not actually use
        let tok_eos = tokenizer
            .getattr("eos_token_id")?
            .extract::<Option<u32>>()?
            .unwrap_or_else(|| {
                let r = tokens.len() as u32;
                tokens.push(vec![]);
                r
            });
        let tok_bos = tokenizer
            .getattr("bos_token_id")?
            .extract::<Option<u32>>()?;

        let special_token_ids = tokenizer
            .getattr("special_token_ids")?
            .extract::<Vec<u32>>()?;

        for tok_id in special_token_ids {
            let tok_ix = tok_id as usize;
            if let Some(token) = tokens.get_mut(tok_ix) {
                if token
                    .first()
                    .is_none_or(|&first_byte| first_byte != TokTrie::SPECIAL_TOKEN_MARKER)
                {
                    token.insert(0, TokTrie::SPECIAL_TOKEN_MARKER);
                }
            }
        }

        // we want decode_bytes([EOS]) etc to be empty
        tokens[tok_eos as usize] = vec![];
        // if let Some(t) = tok_bos {
        //     tokens[t as usize] = vec![];
        // }

        let info = TokRxInfo::new(tokens.len() as u32, tok_eos);

        let tok_trie = TokTrie::from(&info, &tokens);
        Ok(PyTokenizer {
            tok_trie: Arc::new(tok_trie),
            tokenizer_fun: tokenizer.into(),
            tok_bos,
        })
    }
}

impl TokenizerEnv for PyTokenizer {
    fn tok_trie(&self) -> &toktrie::TokTrie {
        &self.tok_trie
    }

    fn tokenize_bytes(&self, utf8bytes: &[u8]) -> Vec<TokenId> {
        self.tok_trie.tokenize_with_greedy_fallback(utf8bytes, |s| {
            Python::with_gil(|py| {
                let r = self.tokenizer_fun.call1(py, (s,)).unwrap();
                r.extract::<Vec<TokenId>>(py).unwrap()
            })
        })
    }
}

#[derive(Clone)]
#[pyclass]
struct JsonCompiler {
    item_separator: String,
    key_separator: String,
    whitespace_flexible: bool,
    coerce_one_of: bool,
}

#[pymethods]
impl JsonCompiler {
    #[new]
    #[pyo3(signature = (separators = None, whitespace_flexible = false, coerce_one_of = false))]
    fn py_new(
        separators: Option<(String, String)>,
        whitespace_flexible: bool,
        coerce_one_of: bool,
    ) -> Self {
        let (item_separator, key_separator) = separators.unwrap_or_else(|| {
            if whitespace_flexible {
                (",".to_owned(), ":".to_owned())
            } else {
                (", ".to_owned(), ": ".to_owned())
            }
        });
        JsonCompiler {
            item_separator,
            key_separator,
            whitespace_flexible,
            coerce_one_of,
        }
    }
    fn compile(&self, schema: &str) -> PyResult<String> {
        let mut schema: Value = serde_json::from_str(schema).map_err(val_error)?;
        let compile_options = JsonCompileOptions {
            item_separator: self.item_separator.clone(),
            key_separator: self.key_separator.clone(),
            whitespace_flexible: self.whitespace_flexible,
            coerce_one_of: self.coerce_one_of,
            retriever: None,
        };
        compile_options.apply_to(&mut schema);
        let grm = TopLevelGrammar::from_json_schema(schema);
        let res = serde_json::to_string(&grm).map_err(val_error)?;
        let g_init = GrammarInit::Serialized(grm);
        // this compiles the grammar and signals errors
        let _ = g_init
            .to_internal(None, ParserLimits::default())
            .map_err(val_error)?;
        Ok(res)
    }
}

#[derive(Clone)]
#[pyclass]
struct LarkCompiler {}

#[pymethods]
impl LarkCompiler {
    #[new]
    fn py_new() -> Self {
        LarkCompiler {}
    }
    fn compile(&self, lark: &str) -> PyResult<String> {
        let grammar = TopLevelGrammar::from_lark(lark.to_string());
        serde_json::to_string(&grammar).map_err(val_error)
    }
}

#[derive(Clone)]
#[pyclass]
struct RegexCompiler {}

#[pymethods]
impl RegexCompiler {
    #[new]
    fn py_new() -> Self {
        RegexCompiler {}
    }
    fn compile(&self, regex: &str) -> PyResult<String> {
        let grammar = TopLevelGrammar::from_regex(regex);
        serde_json::to_string(&grammar).map_err(val_error)
    }
}

pub(crate) fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LLTokenizer>()?;
    m.add_class::<JsonCompiler>()?;
    m.add_class::<LarkCompiler>()?;
    m.add_class::<RegexCompiler>()?;
    Ok(())
}

fn val_error(e: impl Display) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}
