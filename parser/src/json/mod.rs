pub mod compiler;
mod formats;
mod numeric;
mod schema;
mod shared_context;

#[cfg(feature = "referencing")]
mod context_ref;
#[cfg(not(feature = "referencing"))]
mod context_simple;

pub mod context {
    #[cfg(feature = "referencing")]
    pub use super::context_ref::*;
    #[cfg(not(feature = "referencing"))]
    pub use super::context_simple::*;
}

use std::{any::type_name_of_val, sync::Arc};

use serde_json::Value;

#[derive(Debug)]
struct UnsatisfiableSchemaError {
    message: String,
}

impl std::fmt::Display for UnsatisfiableSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unsatisfiable schema: {}", self.message)
    }
}

#[cfg(feature = "lark")]
pub(crate) fn numeric_regex(
    keywords: &serde_json::Map<String, Value>,
    integer: bool,
) -> anyhow::Result<derivre::RegexAst> {
    let keywords = keywords
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    let number = schema::compile_numeric(&keywords, integer)?;
    if integer {
        numeric::json_int(&number)
    } else {
        numeric::json_number(&number)
    }
}

pub fn json_merge(a: &mut Value, b: &Value) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b.iter() {
                json_merge(a.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (a, b) => *a = b.clone(),
    }
}

pub trait Retrieve: Send + Sync {
    fn retrieve(&self, uri: &str) -> Result<Value, Box<dyn std::error::Error + Send + Sync>>;
}

#[derive(Clone)]
pub struct RetrieveWrapper(pub Arc<dyn Retrieve>);
impl RetrieveWrapper {
    pub fn new(retrieve: Arc<dyn Retrieve>) -> Self {
        RetrieveWrapper(retrieve)
    }
}

impl std::fmt::Debug for RetrieveWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", type_name_of_val(&self.0))
    }
}
