use serde_json::json;

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_test_many};

#[test]
fn test_json_null() {
    json_test_many(
        &json!({"type":"null"}),
        &[json!(null)],
        &[json!(true), json!(false), json!(1), json!("Hello")],
    );
}

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
        "Unsatisfiable schema: minimum (1) is greater than maximum (-1)",
    );
    json_err_test(
        &json!({
            "type": "integer",
            "exclusiveMinimum": 1, "maximum": -1
        }),
        "Unsatisfiable schema: minimum (1) is greater than maximum (-1)",
    );
    json_err_test(
        &json!({
            "type": "integer",
            "minimum": 1, "exclusiveMaximum": -1
        }),
        "Unsatisfiable schema: minimum (1) is greater than maximum (-1)",
    );
    json_err_test(
        &json!({
            "type": "integer",
            "exclusiveMinimum": 1, "exclusiveMaximum": -1
        }),
        "Unsatisfiable schema: minimum (1) is greater than maximum (-1)",
    );
    json_err_test(
        &json!({
            "type": "integer",
            "exclusiveMinimum": 0, "exclusiveMaximum": 1
        }),
        "Failed to generate regex for integer range",
    );
}

#[test]
fn test_json_number() {
    json_test_many(
        &json!({"type":"number"}),
        &[
            json!(0),
            json!(0.0),
            json!(1.0),
            json!(-1.0),
            json!(-1),
            json!(1),
            json!(142.4),
            json!(-213.1),
            json!(1.23e23),
            json!(-9.2e-132),
        ],
        &[json!("1.0"), json!("1"), json!("Hello")],
    );
}

#[test]
fn test_json_number_limits() {
    json_test_many(
        &json!({"type":"number", "minimum": -100, "maximum": 100}),
        &[json!(0.0), json!(-100.0), json!(100.0)],
        &[json!(-100.0001), json!(100.0001)],
    );
    json_test_many(
        &json!({"type":"number", "exclusiveMinimum": -1, "maximum": 100}),
        &[json!(-0.99999), json!(1.0), json!(50), json!(100.0)],
        &[json!(-1.0), json!(-1), json!(100.0001)],
    );
    json_test_many(
        &json!({"type":"number", "minimum": -0.5, "exclusiveMaximum": 5}),
        &[json!(-0.5), json!(0), json!(0.1), json!(4.999999)],
        &[json!(-0.50001), json!(5.000001)],
    );
    json_test_many(
        &json!({"type":"number", "exclusiveMinimum": 0, "exclusiveMaximum": 1.5}),
        &[json!(0.00001), json!(1.0), json!(1.49999)],
        &[json!(-0.0), json!(1.5)],
    );
    json_err_test(
        &json!({
            "type": "number",
            "minimum": 1.5, "maximum": -1
        }),
        "Unsatisfiable schema: minimum (1.5) is greater than maximum (-1)",
    );
    json_err_test(
        &json!({
            "type": "number",
        // Note coercion of 1.0 to 1
            "exclusiveMinimum": 1.0, "maximum": -1
        }),
        "Unsatisfiable schema: minimum (1) is greater than maximum (-1)",
    );
    json_err_test(
        &json!({
            "type": "number",
        // Note coercion of 1.0 to 1
            "minimum": 1.0, "exclusiveMaximum": -1.5
        }),
        "Unsatisfiable schema: minimum (1) is greater than maximum (-1.5)",
    );
    json_err_test(
        &json!({
            "type": "number",
            "exclusiveMinimum": 1.0, "exclusiveMaximum": -2.5
        }),
        // Note coercion of 1.0 to 1
        "Unsatisfiable schema: minimum (1) is greater than maximum (-2.5)",
    );
}

#[test]
fn test_json_string() {
    json_test_many(
        &json!({"type":"string"}),
        &[
            json!(""),
            json!("Hello"),
            json!("123"),
            json!("!@#$%^&*()_+"),
            json!("'"),
            json!("\""),
            json!(
                r"Hello\nWorld
            
            With some extra line breaks etc.
            "
            ),
        ],
        &[json!(1), json!(true), json!(null)],
    );
}

#[test]
fn test_json_string_regex() {
    json_test_many(
        &json!({"type":"string", "pattern": r"a[A-Z]"}),
        &[json!("aB"), json!("aC"), json!("aZ")],
        &[json!("Hello World"), json!("aa"), json!("a1")],
    );
}

#[test]
fn test_json_string_length() {
    json_test_many(
        &json!({"type":"string", "minLength": 3, "maxLength": 5}),
        &[json!("abc"), json!("abcd"), json!("abcde")],
        &[json!("ab"), json!("abcdef"), json!("a")],
    );
    json_test_many(
        &json!({"type":"string", "minLength": 3, "maxLength": 3}),
        &[json!("abc")],
        &[json!("ab"), json!("abcd"), json!("a")],
    );
    json_test_many(
        &json!({"type":"string", "minLength": 0, "maxLength": 0}),
        &[json!("")],
        &[json!("a"), json!("abc")],
    );
    json_err_test(
        &json!({"type":"string", "minLength": 2, "maxLength": 1}),
        "Unsatisfiable schema: minLength (2) is greater than maxLength (1)",
    );
}
