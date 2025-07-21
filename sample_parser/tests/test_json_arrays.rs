use serde_json::json;

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_test_many};

#[test]
fn test_json_array() {
    json_test_many(
        &json!({"type":"array", "items": {"type":"integer"}}),
        &[json!([1, 2, 3]), json!([]), json!([1])],
        &[json!([1, "Hello"]), json!([true, false]), json!([1.0, 2.0])],
    );
}

#[test]
fn test_json_array_length_constraints() {
    json_test_many(
        &json!({"type":"array", "items": {"type":"integer"}, "minItems": 2, "maxItems": 4}),
        &[json!([1, 2]), json!([1, 2, 3]), json!([1, 2, 3, 4])],
        &[json!([1]), json!([]), json!([1, 2, 3, 4, 5])],
    );
    json_err_test(
        &json!({"type":"array", "items": {"type":"integer"}, "minItems": 2, "maxItems": 1}),
        "Unsatisfiable schema",
    );
}
