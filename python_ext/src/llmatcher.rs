use std::borrow::Cow;
use std::fmt::Display;
use std::ops::DerefMut;

use anyhow::Result;
use llguidance::api::GrammarInit;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{InferenceCapabilities, SimpleVob, TokEnv, TokenId};
use llguidance::{json_merge, Logger, Matcher, ParserFactory};
use pyo3::types::{PyList, PyTuple};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::json;

use crate::parserlimits::LLParserLimits;
use crate::py::LLTokenizer;
use crate::pyjson::{str_or_dict_to_value, stringify_if_needed, to_json_value};

// #[derive(Clone)]
#[pyclass]
struct LLMatcher {
    inner: Matcher,
    tok_env: TokEnv,
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
        trg_batch_size: usize,
        py: Python<'_>,
    ) -> PyResult<()> {
        if interpreters.len() == 0 {
            return Err(PyValueError::new_err("No interpreters"));
        }

        let mut mut_refs = vec![];
        for ent in interpreters.iter() {
            let tupl = ent.downcast::<PyTuple>()?;
            if tupl.len() != 2 {
                return Err(PyValueError::new_err("Expecting (LLMatcher, int) tuple"));
            }
            let interp = tupl.get_item(0)?.extract::<PyRefMut<LLMatcher>>()?;
            let idx = tupl.get_item(1)?.extract::<usize>()?;
            if idx >= trg_batch_size {
                return Err(PyValueError::new_err("Target index out of bounds"));
            }
            interp.validate_mask_ptr(trg_ptr, one_mask_bytes)?;
            mut_refs.push((interp, idx));
        }

        if mut_refs.len() == 1 {
            let (mut interp, idx) = mut_refs.pop().unwrap();
            return interp.unsafe_compute_mask_ptr(
                trg_ptr + idx * one_mask_bytes,
                one_mask_bytes,
                py,
            );
        }

        let mut_refs2: Vec<_> = mut_refs
            .iter_mut()
            .map(|(x, idx)| (x.deref_mut(), *idx))
            .collect();

        use rayon::prelude::*;

        py.allow_threads(|| {
            self.pool.install(|| {
                mut_refs2.into_par_iter().for_each(|(interp, idx)| {
                    interp.unsafe_compute_mask_ptr_inner(
                        trg_ptr + idx * one_mask_bytes,
                        one_mask_bytes,
                    )
                })
            })
        });

        Ok(())
    }
}

impl LLMatcher {
    fn validate_mask_ptr(&self, mask_ptr: usize, mask_bytes: usize) -> PyResult<()> {
        if mask_ptr == 0 {
            return Err(PyValueError::new_err("Null pointer"));
        }
        if mask_ptr % 4 != 0 {
            return Err(PyValueError::new_err("Pointer not aligned"));
        }
        let n_words = self.tok_env.tok_trie().vocab_size().div_ceil(32);
        if mask_bytes != n_words * 4 {
            return Err(PyValueError::new_err("Invalid buffer size"));
        }
        Ok(())
    }

    fn unsafe_compute_mask_ptr_inner(&mut self, trg_ptr: usize, trg_bytes: usize) {
        let r = self.compute_mask_or_eos();
        let trg_slice =
            unsafe { std::slice::from_raw_parts_mut(trg_ptr as *mut u32, trg_bytes / 4) };
        let src = r.as_slice();
        trg_slice.copy_from_slice(&src[0..trg_slice.len()]);
    }

    fn eos_token_set(&self) -> SimpleVob {
        let trie = self.tok_env.tok_trie();
        trie.singleton_token_set(trie.eos_token())
    }

    fn compute_mask_or_eos(&mut self) -> SimpleVob {
        if self.inner.is_stopped() {
            self.eos_token_set()
        } else {
            self.inner
                .compute_mask()
                .unwrap_or_else(|_| self.eos_token_set())
        }
    }
}

fn new_matcher(
    fact: &ParserFactory,
    grammar: TopLevelGrammar,
    log_level: isize,
    limits: Option<&LLParserLimits>,
    py: Python<'_>,
) -> Matcher {
    let logger = Logger::new(0, std::cmp::max(0, log_level) as u32);
    // constructing a grammar can take on the order of 100ms
    // for very large grammars, so we drop the GIL here
    let inner = py.allow_threads(|| {
        fact.create_parser_from_init_ext(
            GrammarInit::Serialized(grammar),
            logger,
            InferenceCapabilities::default(),
            LLParserLimits::from_option(limits),
        )
    });
    Matcher::new(inner)
}

fn extract_grammar(grammar: Bound<'_, PyAny>) -> Result<TopLevelGrammar> {
    if let Ok(s) = grammar.extract::<String>() {
        TopLevelGrammar::from_lark_or_grammar_list(&s)
    } else {
        Ok(serde_json::from_value(to_json_value(grammar)?)?)
    }
}

// This is the interface from llguidance to the LLM's.
#[pymethods]
impl LLMatcher {
    #[new]
    #[pyo3(signature = (tokenizer, grammar, /, log_level=None, limits=None))]
    fn py_new(
        tokenizer: &LLTokenizer,
        grammar: Bound<'_, PyAny>,
        log_level: Option<isize>,
        limits: Option<&LLParserLimits>,
        py: Python<'_>,
    ) -> Self {
        let fact = tokenizer.factory();
        let inner = match extract_grammar(grammar) {
            Ok(grammar) => new_matcher(fact, grammar, log_level.unwrap_or(1), limits, py),
            Err(e) => Matcher::new(Err(e)),
        };
        LLMatcher {
            inner,
            tok_env: fact.tok_env().clone(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (grammar, tokenizer=None, /, limits=None))]
    fn validate_grammar(
        grammar: Bound<'_, PyAny>,
        tokenizer: Option<&LLTokenizer>,
        limits: Option<&LLParserLimits>,
        py: Python<'_>,
    ) -> String {
        match extract_grammar(grammar) {
            Ok(grammar) => py.allow_threads(|| {
                GrammarInit::Serialized(grammar)
                    .to_internal(
                        tokenizer.map(|t| t.factory().tok_env().clone()),
                        LLParserLimits::from_option(limits),
                    )
                    .err()
                    .map(|e| e.to_string())
                    .unwrap_or_default()
            }),
            Err(e) => e.to_string(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (schema, /, defaults=None, overrides=None))]
    fn grammar_from_json_schema(
        schema: Bound<'_, PyAny>,
        defaults: Option<Bound<'_, PyAny>>,
        overrides: Option<Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        if defaults.is_some() || overrides.is_some() {
            let mut schema = str_or_dict_to_value(schema)?;
            if schema.is_object() {
                let mut options = defaults.map_or_else(|| Ok(json!({})), str_or_dict_to_value)?;
                let in_schema = &schema["x-guidance"];
                if in_schema.is_object() {
                    json_merge(&mut options, in_schema);
                }
                if let Some(overrides) = overrides {
                    let overrides = str_or_dict_to_value(overrides)?;
                    json_merge(&mut options, &overrides);
                }
                schema["x-guidance"] = options;
            } else {
                // we could support "true" and "false" as schemas here but probably not worth it
                return Err(PyValueError::new_err(
                    "Expecting object schema to apply options",
                ));
            }
            let grm = TopLevelGrammar::from_json_schema(schema);
            Ok(serde_json::to_string(&grm).map_err(val_error)?)
        } else {
            Ok(format!(
                "{{ \"grammars\": [{{ \"json_schema\": {} }}] }}",
                stringify_if_needed(schema)?
            ))
        }
    }

    #[staticmethod]
    fn grammar_from_lark(lark: String) -> String {
        // lark can be passed directly
        lark
    }

    #[staticmethod]
    fn grammar_from_regex(regex: &str) -> String {
        serde_json::to_string(&TopLevelGrammar::from_regex(regex)).unwrap()
    }

    fn deep_copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            tok_env: self.tok_env.clone(),
        }
    }

    fn is_accepting(&mut self) -> bool {
        self.inner.is_accepting().unwrap_or(false)
    }

    fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    fn stop_reason(&self) -> String {
        self.inner.stop_reason().to_string()
    }

    fn validate_tokens(&mut self, tokens: Vec<TokenId>) -> usize {
        self.inner.validate_tokens(&tokens).unwrap_or_else(|_| {
            let eos = self.tok_env.tok_trie().eos_token();
            if tokens.first() == Some(&eos) {
                1
            } else {
                0
            }
        })
    }

    fn unsafe_compute_mask_ptr(
        &mut self,
        trg_ptr: usize,
        trg_bytes: usize,
        py: Python<'_>,
    ) -> PyResult<()> {
        self.validate_mask_ptr(trg_ptr, trg_bytes)?;
        py.allow_threads(|| self.unsafe_compute_mask_ptr_inner(trg_ptr, trg_bytes));
        Ok(())
    }

    fn compute_logit_bias(&mut self, py: Python<'_>) -> Cow<[u8]> {
        py.allow_threads(|| {
            let m = self.compute_mask_or_eos();
            let mut res = vec![0u8; m.len()];
            m.iter_set_entries(|i| res[i] = 200);
            Cow::Owned(res)
        })
    }

    fn compute_bitmask(&mut self, py: Python<'_>) -> Cow<[u8]> {
        py.allow_threads(|| {
            let m = self.compute_mask_or_eos();
            Cow::Owned(bytemuck::cast_slice(m.as_slice()).to_vec())
        })
    }

    fn consume_token(&mut self, sampled_token: TokenId) -> bool {
        if self.inner.is_stopped() && sampled_token == self.tok_env.tok_trie().eos_token() {
            true
        } else {
            self.inner.consume_token(sampled_token).is_ok()
        }
    }

    fn consume_tokens(&mut self, tokens: Vec<TokenId>) -> bool {
        self.inner.consume_tokens(&tokens).is_ok()
    }

    fn rollback(&mut self, num_tokens: usize) -> bool {
        self.inner.rollback(num_tokens).is_ok()
    }

    fn reset(&mut self) -> bool {
        self.inner.reset().is_ok()
    }

    fn compute_ff_tokens(&mut self) -> Vec<TokenId> {
        self.inner.compute_ff_tokens()
    }

    fn compute_ff_bytes(&mut self) -> Cow<[u8]> {
        let bytes = self.inner.compute_ff_bytes();
        Cow::Owned(bytes)
    }

    fn try_consume_tokens(&mut self, tokens: Vec<TokenId>) -> usize {
        self.inner.try_consume_tokens(&tokens).unwrap_or(0)
    }

    fn is_error(&self) -> bool {
        self.inner.is_error()
    }

    fn get_error(&self) -> String {
        self.inner.get_error().unwrap_or_default()
    }
}

pub(crate) fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LLMatcher>()?;
    m.add_class::<LLExecutor>()?;
    Ok(())
}

fn val_error(e: impl Display) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}
