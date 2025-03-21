use std::borrow::Cow;
use std::fmt::Display;

use llguidance::api::ParserLimits;
use llguidance::toktrie::{InferenceCapabilities, TokenId};
use llguidance::{api::TopLevelGrammar, output::ParserOutput, TokenParser};
use llguidance::{Constraint, Logger};
use pyo3::types::PyByteArray;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

use crate::py::LLTokenizer;

// #[derive(Clone)]
#[pyclass]
struct LLInterpreter {
    inner: Constraint,
    #[pyo3(get, set)]
    log_level: isize,
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
        let fact = tokenizer.factory();
        let arg = TopLevelGrammar::from_lark_or_grammar_list(grammar).map_err(val_error)?;
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
        Ok(LLInterpreter { inner, log_level })
    }

    fn deep_copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            log_level: self.log_level,
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

pub(crate) fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LLInterpreter>()?;
    Ok(())
}

fn val_error(e: impl Display) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}
