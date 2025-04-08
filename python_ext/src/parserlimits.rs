use llguidance::api::ParserLimits;
use pyo3::prelude::*;

#[pyclass]
pub struct LLParserLimits {
    inner: ParserLimits,
}

impl LLParserLimits {
    pub fn from_option(limits: Option<&LLParserLimits>) -> ParserLimits {
        if let Some(limits) = limits {
            limits.inner.clone()
        } else {
            ParserLimits::default()
        }
    }
}

#[pymethods]
impl LLParserLimits {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (/, max_items_in_row=None, initial_lexer_fuel=None, step_lexer_fuel=None,
        step_max_items=None, max_lexer_states=None, max_grammar_size=None, precompute_large_lexemes=None))]
    fn new(
        max_items_in_row: Option<usize>,
        initial_lexer_fuel: Option<u64>,
        step_lexer_fuel: Option<u64>,
        step_max_items: Option<usize>,
        max_lexer_states: Option<usize>,
        max_grammar_size: Option<usize>,
        precompute_large_lexemes: Option<bool>,
    ) -> Self {
        let mut inner = ParserLimits::default();
        if let Some(v) = max_items_in_row {
            inner.max_items_in_row = v;
        }
        if let Some(v) = initial_lexer_fuel {
            inner.initial_lexer_fuel = v;
        }
        if let Some(v) = step_lexer_fuel {
            inner.step_lexer_fuel = v;
        }
        if let Some(v) = step_max_items {
            inner.step_max_items = v;
        }
        if let Some(v) = max_lexer_states {
            inner.max_lexer_states = v;
        }
        if let Some(v) = max_grammar_size {
            inner.max_grammar_size = v;
        }
        if let Some(v) = precompute_large_lexemes {
            inner.precompute_large_lexemes = v;
        }
        Self { inner }
    }

    #[getter]
    fn max_items_in_row(&self) -> usize {
        self.inner.max_items_in_row
    }

    #[getter]
    fn initial_lexer_fuel(&self) -> u64 {
        self.inner.initial_lexer_fuel
    }

    #[getter]
    fn step_lexer_fuel(&self) -> u64 {
        self.inner.step_lexer_fuel
    }

    #[getter]
    fn step_max_items(&self) -> usize {
        self.inner.step_max_items
    }

    #[getter]
    fn max_lexer_states(&self) -> usize {
        self.inner.max_lexer_states
    }

    #[getter]
    fn max_grammar_size(&self) -> usize {
        self.inner.max_grammar_size
    }

    #[getter]
    fn precompute_large_lexemes(&self) -> bool {
        self.inner.precompute_large_lexemes
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

pub(crate) fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LLParserLimits>()?;
    Ok(())
}
