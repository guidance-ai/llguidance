use std::fmt::Display;
use std::ops::DerefMut;
use std::{borrow::Cow, sync::Arc};

use llguidance::api::{GrammarInit, ParserLimits};
use llguidance::earley::SlicedBiasComputer;
use llguidance::toktrie::{
    self, ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie, TokenId,
    TokenizerEnv,
};
use llguidance::{api::TopLevelGrammar, output::ParserOutput, TokenParser};
use llguidance::{
    token_bytes_from_tokenizer_json, Constraint, JsonCompileOptions, Logger, ParserFactory,
};
use pyo3::types::{PyByteArray, PyList};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// #[derive(Clone)]
#[pyclass]
struct LLInterpreter {
    inner: Constraint,
    #[pyo3(get, set)]
    log_level: isize,
    borrowed: bool,
}

struct PyTokenizer {
    tok_trie: Arc<toktrie::TokTrie>,
    tokenizer_fun: Py<PyAny>,
    #[allow(dead_code)]
    tok_bos: Option<u32>,
}

#[pyclass]
struct LLExecutor {
    pool: rayon::ThreadPool,
}

#[pymethods]
impl LLExecutor {
    #[new]
    #[pyo3(signature = (num_threads=None))]
    fn py_new(num_threads: Option<usize>) -> PyResult<Self> {
        let num_threads = num_threads.unwrap_or_else(|| {
            let n = std::thread::available_parallelism().unwrap().get();
            // by default run on 80% of available threads but not more than 32
            (n * 80 / 100).clamp(1, 32)
        });
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(val_error)?;
        Ok(LLExecutor { pool })
    }

    fn unsafe_compute_mask_ptr(
        &self,
        interpreters: Bound<'_, PyList>,
        trg_ptr: usize,
        one_mask_bytes: usize,
    ) -> PyResult<String> {
        if interpreters.len() == 0 {
            return Err(PyValueError::new_err("No interpreters"));
        }

        if interpreters.len() == 1 {
            let mut interp = interpreters
                .get_item(0)?
                .extract::<PyRefMut<LLInterpreter>>()?;
            return interp.unsafe_compute_mask_ptr(trg_ptr, one_mask_bytes);
        }

        use rayon::prelude::*;

        let mut ptrs = vec![];
        for ent in interpreters.iter() {
            let mut interp = ent.extract::<PyRefMut<LLInterpreter>>()?;
            interp.validate_mask_ptr(trg_ptr, one_mask_bytes)?;
            if interp.borrowed {
                return Err(PyValueError::new_err("Interpreter already borrowed"));
            }
            let interp = interp.deref_mut() as *mut LLInterpreter;
            ptrs.push(interp);
        }

        let mut ok = true;
        let mut refs = vec![];
        for (idx, interp_ptr) in ptrs.iter_mut().enumerate() {
            unsafe {
                let interp = &mut **interp_ptr;
                if interp.borrowed {
                    ok = false;
                    break;
                }
                interp.borrowed = true;
                refs.push((idx, interp));
            }
        }

        let rlen = refs.len();

        if !ok {
            #[allow(clippy::needless_range_loop)]
            for idx in 0..rlen {
                unsafe { (*ptrs[idx]).borrowed = false };
            }
            return Err(PyValueError::new_err("Duplicate interpreter in list"));
        }

        let strs = self.pool.install(|| {
            refs.into_par_iter()
                .map(|(idx, interp)| {
                    interp.unsafe_compute_mask_ptr(trg_ptr + idx * one_mask_bytes, one_mask_bytes)
                })
                .collect::<Result<Vec<_>, _>>()
        });
        #[allow(clippy::needless_range_loop)]
        for idx in 0..rlen {
            unsafe { (*ptrs[idx]).borrowed = false };
        }
        let strs = strs?;
        Ok(format!("[{}]", strs.join(",")))
    }
}

#[derive(Clone)]
#[pyclass]
struct LLTokenizer {
    factory: Arc<ParserFactory>,
}

impl LLInterpreter {
    fn json_py_result(&mut self) -> String {
        let res = PyMidProcessResult {
            progress: self.inner.flush_progress(),
            stop: self.inner.step_result().is_stop(),
            temperature: self.inner.temperature,
        };
        serde_json::to_string(&res).unwrap()
    }

    fn validate_mask_ptr(&self, mask_ptr: usize, mask_bytes: usize) -> PyResult<()> {
        if mask_ptr == 0 {
            return Err(PyValueError::new_err("Null pointer"));
        }
        if mask_ptr % 4 != 0 {
            return Err(PyValueError::new_err("Pointer not aligned"));
        }
        let n_words = (self.inner.tok_trie().vocab_size() + 31) / 32;
        if mask_bytes != n_words * 4 {
            return Err(PyValueError::new_err("Invalid buffer size"));
        }
        Ok(())
    }
}

// This is the interface from llguidance to the LLM's.
#[pymethods]
impl LLInterpreter {
    #[new]
    #[pyo3(signature = (tokenizer, grammar, enable_backtrack=None, enable_ff_tokens=None, log_level=None))]
    fn py_new(
        tokenizer: &LLTokenizer,
        grammar: &str,
        enable_backtrack: Option<bool>,
        enable_ff_tokens: Option<bool>,
        log_level: Option<isize>,
    ) -> PyResult<Self> {
        let fact = &tokenizer.factory;
        let arg = TopLevelGrammar::from_lark_or_json_schema(grammar).map_err(val_error)?;
        let log_level = log_level.unwrap_or(1);
        let inference_caps = InferenceCapabilities {
            backtrack: enable_backtrack.unwrap_or(true),
            ff_tokens: enable_ff_tokens.unwrap_or(true),
            conditional_ff_tokens: enable_ff_tokens.unwrap_or(true),
            fork: false,
        };
        let logger = Logger::new(0, std::cmp::max(0, log_level) as u32);
        let mut inner = TokenParser::from_grammar(
            fact.tok_env().clone(),
            arg,
            logger,
            inference_caps,
            ParserLimits::default(),
            fact.extra_lexemes(),
        )
        .map_err(val_error)?;
        fact.post_process_parser(&mut inner);
        let inner = Constraint::new(inner);
        Ok(LLInterpreter {
            inner,
            log_level,
            borrowed: false,
        })
    }

    fn deep_copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            log_level: self.log_level,
            borrowed: false,
        }
    }

    fn is_accepting(&mut self) -> bool {
        self.inner.parser.is_accepting()
    }

    fn stop_reason(&self) -> String {
        self.inner.parser.stop_reason().to_string()
    }

    fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        self.inner.process_prompt(prompt)
    }

    fn start_without_prompt(&mut self) {
        self.inner.start_without_prompt()
    }

    fn validate_tokens_raw(&mut self, tokens: Vec<TokenId>) -> PyResult<usize> {
        self.inner.validate_tokens_raw(&tokens).map_err(val_error)
    }

    fn unsafe_compute_mask_ptr(&mut self, trg_ptr: usize, trg_bytes: usize) -> PyResult<String> {
        self.validate_mask_ptr(trg_ptr, trg_bytes)?;
        let r = self.inner.compute_mask().map_err(val_error)?;
        let trg_slice =
            unsafe { std::slice::from_raw_parts_mut(trg_ptr as *mut u32, trg_bytes / 4) };
        if let Some(m) = r.sample_mask.as_ref() {
            let src = m.as_slice();
            trg_slice.copy_from_slice(&src[0..trg_slice.len()]);
        } else {
            trg_slice.fill(0);
            let trie = self.inner.tok_trie();
            let eos = trie.eos_token();
            let eos_ok = (eos as usize) < trie.vocab_size();
            if eos_ok {
                trg_slice[eos as usize / 32] |= 1 << (eos % 32);
            }
        }

        Ok(self.json_py_result())
    }

    // TODO: remove this
    fn compute_mask_into(&mut self, trg: &Bound<'_, PyByteArray>) -> PyResult<String> {
        let r = self.inner.compute_mask().map_err(val_error)?;
        let trg_slice = unsafe { trg.as_bytes_mut() };
        if let Some(m) = r.sample_mask.as_ref() {
            let src = bytemuck::cast_slice::<u32, u8>(m.as_slice());
            if trg_slice.len() > src.len() {
                trg_slice[..src.len()].copy_from_slice(src);
            } else {
                trg_slice.copy_from_slice(&src[..trg_slice.len()]);
            }
        } else {
            trg_slice.fill(0);
        };

        Ok(self.json_py_result())
    }

    fn compute_mask(&mut self, py: Python<'_>) -> PyResult<(Option<Cow<[u8]>>, String)> {
        let r = py
            .allow_threads(|| self.inner.compute_mask())
            .map_err(val_error)?
            .clone();
        let mask = if r.is_stop() || r.unconditional_splice().is_some() {
            None
        } else {
            let m = r
                .sample_mask
                .as_ref()
                .expect("expecting unconditional splice or mask");
            let mut res = vec![0u8; m.len()];
            m.iter_set_entries(|i| res[i] = 200);
            Some(Cow::Owned(res))
        };

        Ok((mask, self.json_py_result()))
    }

    #[pyo3(signature = (sampled_token))]
    fn commit_token(&mut self, sampled_token: Option<TokenId>) -> PyResult<(u32, Vec<TokenId>)> {
        let pres = self.inner.commit_token(sampled_token).map_err(val_error)?;

        if pres.stop {
            // inner.commit_token() only returns stop, when compute_mask()
            // had already returned stop
            Ok((0, vec![]))
        } else {
            Ok((pres.backtrack, pres.ff_tokens))
        }
    }

    fn has_pending_stop(&self) -> bool {
        self.inner.has_pending_stop()
    }
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
        self.factory.tok_env().tokenize_bytes(utf8bytes)
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
    fn tok_trie(&self) -> &toktrie::TokTrie {
        self.factory.tok_env().tok_trie()
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
    m.add_class::<LLInterpreter>()?;
    m.add_class::<LLExecutor>()?;
    m.add_class::<JsonCompiler>()?;
    m.add_class::<LarkCompiler>()?;
    m.add_class::<RegexCompiler>()?;
    Ok(())
}

fn val_error(e: impl Display) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}
