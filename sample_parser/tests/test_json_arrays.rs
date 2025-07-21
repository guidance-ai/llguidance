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
    // Just prove we can do other primitive types in arrays
    json_test_many(
        &json!({"type":"array", "items": {"type":"boolean"}}),
        &[json!([true, false, true]), json!([]), json!([true])],
        &[json!([1, "Hello"]), json!([1, 2]), json!([1.0, 2.0])],
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
        "Unsatisfiable schema: minItems (2) is greater than maxItems (1)",
    );
}

#[test]
fn test_json_array_nested() {
    json_test_many(
        &json!({"type":"array", "items": {"type":"array", "items": {"type":"integer"}}}),
        &[
            json!([]),
            json!([[1]]),
            json!([[1], []]),
            json!([[], [1]]),
            json!([[1, 2], [3, 4]]),
            json!([[0], [1, 2, 3]]),
            json!([[0], [1, 2, 3], [4, 5]]),
        ],
        &[
            json!([[1, "Hello"]]),
            json!([[true, false]]),
            json!([[1.0, 2.0]]),
        ],
    );
}

#[test]
fn test_json_array_objects() {
    json_test_many(
        &json!({"type":"array", "items": {"type":"object", "properties": {"a": {"type":"integer"}}, "required": ["a"]}}),
        &[json!([]), json!([{"a": 1}]), json!([{"a": 1}, {"a": 2}])],
        &[
            json!([{"b": 1}]),
            json!([{"a": "Hello"}]),
            json!([{"a": 1}, {"b": 2}]),
        ],
    );
}
