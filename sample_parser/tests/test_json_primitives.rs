use serde_json::json;

mod common_lark_utils;
use common_lark_utils::*;

#[test]
fn test_json_boolean() {
    json_test_many(
        &json!({"type":"boolean"}),
        &[json!(true), json!(false)],
        &[json!(1), json!("True"), json!(0), json!("False")],
    );
}

#[test]
fn test_json_integer() {
    json_test_many(
        &json!({"type":"integer"}),
        &[json!(1), json!(-1), json!(0), json!(10001), json!(-20002)],
        &[json!(1.0), json!("1"), json!(-1.0), json!("0")],
    );
}

#[test]
fn test_json_integer_limits() {
    json_test_many(
        &json!({"type":"integer", "minimum": -100, "maximum": 100}),
        &[json!(0), json!(-100), json!(100)],
        &[json!(-101), json!(101), json!(1.0)],
    );
    json_test_many(
        &json!({"type":"integer", "exclusiveMinimum": 0, "maximum": 100}),
        &[json!(1), json!(50), json!(100)],
        &[json!(0), json!(-1), json!(101)],
    );
    json_test_many(
        &json!({"type":"integer", "minimum": 0, "exclusiveMaximum": 100}),
        &[json!(0), json!(50), json!(99)],
        &[json!(-1), json!(100), json!(101)],
    );
    json_test_many(
        &json!({"type":"integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100}),
        &[json!(1), json!(50), json!(99)],
        &[json!(-1), json!(0), json!(100), json!(101)],
    );
    json_err_test(
        &json!({
            "type": "integer",
            "minimum": 1, "maximum": -1
        }),
        "Unsatisfiable schema",
    );
    json_err_test(
        &json!({
            "type": "integer",
            "exclusiveMinimum": 1, "maximum": -1
        }),
        "Unsatisfiable schema",
    );
    json_err_test(
        &json!({
            "type": "integer",
            "minimum": 1, "exclusiveMaximum": -1
        }),
        "Unsatisfiable schema",
    );
    json_err_test(
        &json!({
            "type": "integer",
            "exclusiveMinimum": 1, "exclusiveMaximum": -1
        }),
        "Unsatisfiable schema",
    );
    json_err_test(
        &json!({
            "type": "integer",
            "exclusiveMinimum": 0, "exclusiveMaximum": 1
        }),
        "Failed to generate regex for integer range",
    );
}
