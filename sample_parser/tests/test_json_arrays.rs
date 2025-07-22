use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_schema_check, json_test_many};

#[rstest]
#[case(&json!([]),)]
#[case(&json!([1]),)]
#[case(&json!([1, 2, 3]),)]
fn test_json_array_integer(#[case] sample_array: &Value) {
    let schema = &json!({"type":"array", "items": {"type":"integer"}});
    json_schema_check(schema, sample_array, true);
}
#[rstest]
#[case(&json!([1, "Hello"]),)]
#[case(&json!([true, false]),)]
#[case(&json!([1.0, 3.0]),)]
fn test_json_array_integer_failures(#[case] sample_array: &Value) {
    let schema = &json!({"type":"array", "items": {"type":"integer"}});
    json_schema_check(schema, sample_array, false);
}

#[rstest]
#[case(&json!([]),)]
#[case(&json!([true]),)]
#[case(&json!([false]),)]
#[case(&json!([false, true]),)]
fn test_json_array_boolean(#[case] sample_array: &Value) {
    let schema = &json!({"type":"array", "items": {"type":"boolean"}});
    json_schema_check(schema, sample_array, true);
}
#[rstest]
#[case(&json!([true, 0]),)]
#[case(&json!([false, 1]),)]
#[case(&json!([1.0, 0.0]),)]
fn test_json_array_boolean_failures(#[case] sample_array: &Value) {
    let schema = &json!({"type":"array", "items": {"type":"boolean"}});
    json_schema_check(schema, sample_array, false);
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
