use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_schema_check, json_test_many};

#[test]
fn null_schema() {
    let schema = &json!({"type":"null"});
    json_schema_check(schema, &json!(null), true);
}

#[rstest]
#[case::boolean(&json!(true))]
#[case::integer(&json!(1))]
#[case::string(&json!("Hello"))]
fn null_schema_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"null"});
    json_schema_check(schema, sample_value, false);
}

// ============================================================================

#[rstest]
#[case::bool_false(&json!(false))]
#[case::bool_true(&json!(true))]
fn boolean(#[case] sample_value: &Value) {
    let schema = &json!({"type":"boolean"});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case::int_0(&json!(0))]
#[case::int_1(&json!(1))]
#[case::str_false(&json!("false"))]
#[case::str_true(&json!("true"))]
fn boolean_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"boolean"});
    json_schema_check(schema, sample_value, false);
}

// ============================================================================

#[rstest]
#[case::one(&json!(1))]
#[case::minus_1(&json!(-1))]
#[case::zero(&json!(0))]
#[case::large(&json!(10001))]
#[case::negative_large(&json!(-20002))]
fn integer(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer"});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case::float(&json!(1.0))]
#[case::string_one(&json!("1"))]
#[case::negative_float(&json!(-1.0))]
#[case::string_zero(&json!("0"))]
fn integer_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer"});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!(0))]
#[case(&json!(-100))]
#[case(&json!(100))]
fn integer_limits_inc_inc(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer", "minimum": -100, "maximum": 100});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(-101))]
#[case(&json!(101))]
#[case(&json!(1.0))]
fn integer_limits_inc_inc_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer", "minimum": -100, "maximum": 100});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!(0))]
#[case(&json!(-99))]
#[case(&json!(100))]
fn integer_limits_excl_inc(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer", "exclusiveMinimum": -100, "maximum": 100});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(-101))]
#[case(&json!(-100))]
#[case(&json!(101))]
#[case(&json!(1.0))]
fn integer_limits_excl_inc_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer", "exclusiveMinimum": -100, "maximum": 100});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!(0))]
#[case(&json!(-100))]
#[case(&json!(99))]
fn integer_limits_inc_excl(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer", "minimum": -100, "exclusiveMaximum": 100});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(-101))]
#[case(&json!(100))]
#[case(&json!(101))]
#[case(&json!(1.0))]
fn integer_limits_inc_excl_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer", "minimum": -100, "exclusiveMaximum": 100});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!(1))]
#[case(&json!(50))]
#[case(&json!(99))]
fn integer_limits_excl_excl(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(0))]
#[case(&json!(100))]
#[case(&json!(-1))]
#[case(&json!(101))]
fn integer_limits_excl_excl_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
fn integer_limits_incompatible(
    #[values("minimum", "exclusiveMinimum")] min_type: &str,
    #[values("maximum", "exclusiveMaximum")] max_type: &str,
) {
    let schema = &json!({
        "type": "integer",
        min_type: 1,
        max_type: -1
    });
    json_err_test(
        schema,
        "Unsatisfiable schema: minimum (1) is greater than maximum (-1)",
    );
}

#[rstest]
fn integer_limits_empty() {
    json_err_test(
        &json!({
            "type": "integer",
            "exclusiveMinimum": 0, "exclusiveMaximum": 1
        }),
        "Failed to generate regex for integer range",
    );
}

// ============================================================================

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

// ============================================================================

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
