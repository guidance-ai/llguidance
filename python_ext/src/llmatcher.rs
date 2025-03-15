use std::borrow::Cow;
use std::fmt::Display;
use std::ops::DerefMut;

use llguidance::api::ParserLimits;
use llguidance::toktrie::{InferenceCapabilities, TokEnv, TokenId};
use llguidance::{api::TopLevelGrammar, TokenParser};
use llguidance::{Logger, Matcher};
use pyo3::types::PyList;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::py::LLTokenizer;

// #[derive(Clone)]
#[pyclass]
struct LLMatcher {
    inner: Matcher,
    tok_env: TokEnv,
    borrowed: bool,
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
    ) -> PyResult<()> {
        if interpreters.len() == 0 {
            return Err(PyValueError::new_err("No interpreters"));
        }

        if interpreters.len() == 1 {
            let mut interp = interpreters.get_item(0)?.extract::<PyRefMut<LLMatcher>>()?;
            return interp.unsafe_compute_mask_ptr(trg_ptr, one_mask_bytes);
        }

        use rayon::prelude::*;

        let mut ptrs = vec![];
        for ent in interpreters.iter() {
            let mut interp = ent.extract::<PyRefMut<LLMatcher>>()?;
            interp.validate_mask_ptr(trg_ptr, one_mask_bytes)?;
            if interp.borrowed {
                return Err(PyValueError::new_err("Interpreter already borrowed"));
            }
            let interp = interp.deref_mut() as *mut LLMatcher;
            ptrs.push(interp);
        }

        let mut ok = true;
        let mut refs = vec![];
        for (idx, &interp_ptr) in ptrs.iter().enumerate() {
            unsafe {
                let interp = &mut *interp_ptr;
                if interp.borrowed {
                    ok = false;
                    break;
                }
                interp.borrowed = true;
                refs.push((idx, interp));
            }
        }

        if !ok {
            for &ptr in &ptrs {
                unsafe { (*ptr).borrowed = false };
            }
            return Err(PyValueError::new_err("Duplicate interpreter in list"));
        }

        let results = self.pool.install(|| {
            refs.into_par_iter()
                .map(|(idx, interp)| {
                    interp.unsafe_compute_mask_ptr(trg_ptr + idx * one_mask_bytes, one_mask_bytes)
                })
                .collect::<Result<Vec<_>, _>>()
        });
        for &ptr in &ptrs {
            unsafe { (*ptr).borrowed = false };
        }
        let _ = results?;
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
        let n_words = (self.tok_env.tok_trie().vocab_size() + 31) / 32;
        if mask_bytes != n_words * 4 {
            return Err(PyValueError::new_err("Invalid buffer size"));
        }
        Ok(())
    }
}

// This is the interface from llguidance to the LLM's.
#[pymethods]
impl LLMatcher {
    #[new]
    #[pyo3(signature = (tokenizer, grammar, log_level=None))]
    fn py_new(tokenizer: &LLTokenizer, grammar: &str, log_level: Option<isize>) -> PyResult<Self> {
        let fact = tokenizer.factory();
        let arg = TopLevelGrammar::from_lark_or_json_schema(grammar)?;
        let log_level = log_level.unwrap_or(1);
        let logger = Logger::new(0, std::cmp::max(0, log_level) as u32);
        let mut inner = TokenParser::from_grammar(
            fact.tok_env().clone(),
            arg,
            logger,
            InferenceCapabilities::default(),
            ParserLimits::default(),
            fact.extra_lexemes(),
        )?;
        fact.post_process_parser(&mut inner);
        let inner = Matcher::new(Ok(inner));
        Ok(LLMatcher {
            inner,
            tok_env: fact.tok_env().clone(),
            borrowed: false,
        })
    }

    fn deep_copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            tok_env: self.tok_env.clone(),
            borrowed: false,
        }
    }

    fn is_accepting(&mut self) -> bool {
        self.inner.is_accepting().unwrap_or(false)
    }

    fn stop_reason(&self) -> String {
        self.inner.stop_reason().to_string()
    }

    fn validate_tokens(&mut self, tokens: Vec<TokenId>) -> PyResult<usize> {
        Ok(self.inner.validate_tokens(&tokens)?)
    }

    fn unsafe_compute_mask_ptr(&mut self, trg_ptr: usize, trg_bytes: usize) -> PyResult<()> {
        self.validate_mask_ptr(trg_ptr, trg_bytes)?;
        let r = self.inner.compute_mask()?;
        let trg_slice =
            unsafe { std::slice::from_raw_parts_mut(trg_ptr as *mut u32, trg_bytes / 4) };
        let src = r.as_slice();
        trg_slice.copy_from_slice(&src[0..trg_slice.len()]);

        Ok(())
    }

    fn compute_logit_bias(&mut self, py: Python<'_>) -> PyResult<Cow<[u8]>> {
        let m = py.allow_threads(|| self.inner.compute_mask())?;
        let mut res = vec![0u8; m.len()];
        m.iter_set_entries(|i| res[i] = 200);
        Ok(Cow::Owned(res))
    }

    fn consume_token(&mut self, sampled_token: TokenId) -> PyResult<()> {
        Ok(self.inner.consume_tokens(&[sampled_token])?)
    }

    fn rollback(&mut self, num_tokens: usize) -> PyResult<()> {
        self.inner.rollback(num_tokens).map_err(val_error)
    }

    fn compute_ff_tokens(&mut self) -> PyResult<Vec<TokenId>> {
        Ok(self.inner.compute_ff_tokens())
    }

    fn compute_ff_bytes(&mut self) -> PyResult<Cow<[u8]>> {
        let bytes = self.inner.compute_ff_bytes();
        Ok(Cow::Owned(bytes))
    }

    fn try_consume_tokens(&mut self, tokens: Vec<TokenId>) -> PyResult<usize> {
        self.inner.try_consume_tokens(&tokens).map_err(val_error)
    }

    fn is_error(&self) -> bool {
        self.inner.is_error()
    }

    fn get_error(&self) -> Option<String> {
        self.inner.get_error()
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
