pub mod compiler;
mod context;
mod formats;
mod numeric;
mod schema;

use serde_json::Value;
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
