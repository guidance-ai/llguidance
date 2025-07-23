use lazy_static::lazy_static;
use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_schema_check};

lazy_static! {
    static ref INTEGER_ARRAY: Value = json!({"type":"array", "items": {"type":"integer"}});
}

#[rstest]
#[case::empty_list(&json!([]),)]
#[case::single_item(&json!([1]),)]
#[case(&json!([1, 2, 3]),)]
fn array_integer(#[case] sample_array: &Value) {
    json_schema_check(&INTEGER_ARRAY, sample_array, true);
}
#[rstest]
#[case(&json!([1, "Hello"]),)]
#[case(&json!([true, false]),)]
#[case(&json!([1.0, 3.0]),)]
fn array_integer_failures(#[case] sample_array: &Value) {
    json_schema_check(&INTEGER_ARRAY, sample_array, false);
}

lazy_static! {
    static ref BOOLEAN_ARRAY: Value = json!({"type":"array", "items": {"type":"boolean"}});
}

#[rstest]
#[case::empty_list(&json!([]),)]
#[case::single_item(&json!([true]),)]
#[case(&json!([false]),)]
#[case(&json!([false, true]),)]
fn array_boolean(#[case] sample_array: &Value) {
    json_schema_check(&BOOLEAN_ARRAY, sample_array, true);
}
#[rstest]
#[case(&json!([true, 0]),)]
#[case(&json!([false, 1]),)]
#[case(&json!([1.0, 0.0]),)]
fn array_boolean_failures(#[case] sample_array: &Value) {
    json_schema_check(&BOOLEAN_ARRAY, sample_array, false);
}

lazy_static! {
    static ref LENGTH_CONSTRAINED_ARRAY: Value =
        json!({"type":"array", "items": {"type":"integer"}, "minItems": 2, "maxItems": 4});
}

#[rstest]
#[case::lower_bound(&json!([1,2]))]
#[case::between_bounds(&json!([1,2, 3]))]
#[case::upper_bound(&json!([1,2, 3, 4]))]
fn array_length_constraints(#[case] sample_array: &Value) {
    json_schema_check(&LENGTH_CONSTRAINED_ARRAY, sample_array, true);
}

#[rstest]
#[case::empty_list(&json!([]))]
#[case::single_item(&json!([1]))]
#[case::too_long(&json!([1,2,3,4,5]))]
fn array_length_failures(#[case] sample_array: &Value) {
    json_schema_check(&LENGTH_CONSTRAINED_ARRAY, sample_array, false);
}

#[test]
fn array_length_bad_constraints() {
    json_err_test(
        &json!({"type":"array", "items": {"type":"integer"}, "minItems": 2, "maxItems": 1}),
        "Unsatisfiable schema: minItems (2) is greater than maxItems (1)",
    );
}

lazy_static! {
    static ref NESTED_ARRAY: Value =
        json!({"type":"array", "items": {"type":"array", "items": {"type":"integer"}}});
}

#[rstest]
#[case::empty_list(&json!([]))]
#[case(&json!([[1]]))]
#[case(&json!([[1], []]))]
#[case(&json!([[], [1]]))]
#[case(&json!([[1, 2], [3, 4]]))]
#[case(&json!([[0], [1, 2, 3]]))]
#[case(&json!([[0], [1, 2, 3], [4, 5]]))]
fn nested_array(#[case] sample_array: &Value) {
    json_schema_check(&NESTED_ARRAY, sample_array, true);
}

#[rstest]
#[case(&json!([[1, "Hello"]]))]
#[case(&json!([[true, false]]))]
#[case(&json!([[1.0, 2.0]]))]
#[case(&json!([[1], [2.0]]))]
fn nested_array_failures(#[case] sample_array: &Value) {
    json_schema_check(&NESTED_ARRAY, sample_array, false);
}

lazy_static! {
    static ref ARRAY_OF_OBJECTS: Value = json!({
        "type":"array",
        "items": {
            "type":"object",
            "properties":
             {
                "a": {"type":"integer"}
            },
            "required": ["a"]
        }
    });
}

#[rstest]
#[case::empty_list(&json!([]))]
#[case::single_item(&json!([{"a": 1}]))]
#[case::multiple_items(&json!([{"a": 1}, {"a": 2}]))]
fn array_of_objects(#[case] sample_array: &Value) {
    json_schema_check(&ARRAY_OF_OBJECTS, sample_array, true);
}

#[rstest]
#[case(&json!([{"b": 1}]))]
#[case(&json!([{"a": "Hello"}]))]
#[case(&json!([{"a": 1}, {"b": 2}]))]
fn array_of_objects_failures(#[case] sample_array: &Value) {
    json_schema_check(&ARRAY_OF_OBJECTS, sample_array, false);
}
