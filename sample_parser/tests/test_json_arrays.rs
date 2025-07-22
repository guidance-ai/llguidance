use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_schema_check};

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

#[rstest]
#[case(&json!([1,2]))]
#[case(&json!([1,2, 3]))]
#[case(&json!([1,2, 3, 4]))]
fn test_json_array_length(#[case] sample_array: &Value) {
    let schema =
        &json!({"type":"array", "items": {"type":"integer"}, "minItems": 2, "maxItems": 4});
    json_schema_check(schema, sample_array, true);
}

#[rstest]
#[case(&json!([]))]
#[case(&json!([1]))]
#[case(&json!([1,2,3,4,5]))]
fn test_json_array_length_failures(#[case] sample_array: &Value) {
    let schema =
        &json!({"type":"array", "items": {"type":"integer"}, "minItems": 2, "maxItems": 4});
    json_schema_check(schema, sample_array, false);
}

#[test]
fn test_json_array_length_bad_constraints() {
    json_err_test(
        &json!({"type":"array", "items": {"type":"integer"}, "minItems": 2, "maxItems": 1}),
        "Unsatisfiable schema: minItems (2) is greater than maxItems (1)",
    );
}

#[rstest]
#[case(&json!([]))]
#[case(&json!([[1]]))]
#[case(&json!([[1], []]))]
#[case(&json!([[], [1]]))]
#[case(&json!([[1, 2], [3, 4]]))]
#[case(&json!([[0], [1, 2, 3]]))]
#[case(&json!([[0], [1, 2, 3], [4, 5]]))]
fn test_json_nested_array(#[case] sample_array: &Value) {
    let schema = &json!({"type":"array", "items": {"type":"array", "items": {"type":"integer"}}});
    json_schema_check(schema, sample_array, true);
}

#[rstest]
#[case(&json!([[1, "Hello"]]))]
#[case(&json!([[true, false]]))]
#[case(&json!([[1.0, 2.0]]))]
fn test_json_nested_array_failures(#[case] sample_array: &Value) {
    let schema = &json!({"type":"array", "items": {"type":"array", "items": {"type":"integer"}}});
    json_schema_check(schema, sample_array, false);
}

#[rstest]
#[case(&json!([]))]
#[case(&json!([{"a": 1}]))]
#[case(&json!([{"a": 1}, {"a": 2}]))]
fn test_json_array_of_objects(#[case] sample_array: &Value) {
    let schema = &json!(
        {
            "type":"array",
            "items": {
                "type":"object",
                "properties":
                 {
                    "a": {"type":"integer"}
                },
                "required": ["a"]
            }
        }
    );

    json_schema_check(schema, sample_array, true);
}

#[rstest]
#[case(&json!([{"b": 1}]))]
#[case(&json!([{"a": "Hello"}]))]
#[case(&json!([{"a": 1}, {"b": 2}]))]
fn test_json_array_of_objects_failures(#[case] sample_array: &Value) {
    let schema = &json!(
        {
            "type":"array",
            "items": {
                "type":"object",
                "properties":
                 {
                    "a": {"type":"integer"}
                },
                "required": ["a"]
            }
        }
    );

    json_schema_check(schema, sample_array, false);
}
