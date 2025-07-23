use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_schema_check};

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

#[rstest]
#[case::zero(&json!(0))]
#[case::zero_float(&json!(0.0))]
#[case::one(&json!(1))]
#[case::one_float(&json!(1.0))]
#[case::minus_1(&json!(-1))]
#[case::minus_1_float(&json!(-1.0))]
#[case::large(&json!(10001.1))]
#[case::negative_large(&json!(-20002.231))]
#[case::positive_exponent(&json!(8.231e2))]
#[case::negative_exponent(&json!(8.231e-2))]
#[case(&json!(-1.61e28))]
#[case(&json!(-8.4e-8))]
fn number(#[case] sample_value: &Value) {
    let schema = &json!({"type":"number"});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case::string_one_float(&json!("1.0"))]
#[case::string_one(&json!("1"))]
#[case::boolean(&json!(false))]
#[case::string_alpha(&json!("Hello"))]
fn number_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"number"});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!(0))]
#[case(&json!(-100))]
#[case(&json!(100))]
fn number_limits_inc_inc(#[case] sample_value: &Value) {
    let schema = &json!({"type":"number", "minimum": -100, "maximum": 100});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(-100.0000001))]
#[case(&json!(100.000001))]
#[case(&json!(2.0e2))]
fn number_limits_inc_inc_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"number", "minimum": -100, "maximum": 100});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!(0))]
#[case(&json!(-0.999999))]
#[case(&json!(-1e-2))]
#[case(&json!(100))]
fn number_limits_excl_inc(#[case] sample_value: &Value) {
    let schema = &json!({"type":"number", "exclusiveMinimum": -1, "maximum": 100});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(-1))]
#[case(&json!(-1.0))]
#[case(&json!(100.000001))]
#[case(&json!(2.0e2))]
fn number_limits_exclu_inc_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"number", "exclusiveMinimum": -1, "maximum": 100});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!(-2))]
#[case(&json!(-100))]
#[case(&json!(-1.00001))]
#[case(&json!(-1.00001e0))]
fn number_limits_inc_excl(#[case] sample_value: &Value) {
    let schema = &json!({"type":"number", "minimum": -100, "exclusiveMaximum": -1});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(-100.0000001))]
//#[case(&json!(-1))] See issue 210
//#[case(&json!(-1.0))] See issue 210
#[case(&json!(-2.12e6))]
#[case(&json!(-4.6e-6))]
fn number_limits_inc_excl_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"number", "minimum": -100, "exclusiveMaximum": -1});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
fn number_limits_incompatible(
    #[values("minimum", "exclusiveMinimum")] min_type: &str,
    #[values("maximum", "exclusiveMaximum")] max_type: &str,
) {
    let schema = &json!({
        "type": "number",
        min_type: -0.1,
        max_type: -1.0
    });
    json_err_test(
        schema,
        "Unsatisfiable schema: minimum (-0.1) is greater than maximum (-1)",
    );
}

// ============================================================================

#[rstest]
#[case::empty(&json!(""))]
#[case::hello(&json!("Hello"))]
#[case::number_string(&json!("123"))]
#[case::special_chars(&json!("!@#$%^&*{}()_+"))]
#[case::single_quote(&json!("'"))]
#[case::double_quote(&json!("\""))]
#[case::unbalanced_brace(&json!("}"))]
#[case::multiline_string(&json!(
    r"Hello\nWorld
            
            With some extra line breaks etc.
            "
))]
fn string(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string"});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case::integer(&json!(1))]
#[case::boolean(&json!(true))]
#[case::null(&json!(null))]
fn string_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string"});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!("aB"))]
#[case(&json!("aC"))]
#[case(&json!("aZ"))]
fn string_regex(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string", "pattern": r"a[A-Z]"});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!("aa"))]
#[case(&json!("a1"))]
#[case(&json!("Hello World!"))]
fn string_regex_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string", "pattern": r"a[A-Z]"});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!("abc"))]
#[case(&json!("abcd"))]
#[case(&json!("abcde"))]
fn string_length_many(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string", "minLength": 3, "maxLength": 5});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(""))]
#[case(&json!("ab"))]
#[case(&json!("abcdef"))]
fn string_length_many_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string", "minLength": 3, "maxLength": 5});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!("abc"))]
#[case(&json!("def"))]
fn string_length_single(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string", "minLength": 3, "maxLength": 3});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!(""))]
#[case(&json!("ab"))]
#[case(&json!("abcd"))]
fn string_length_single_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string", "minLength": 3, "maxLength": 3});
    json_schema_check(schema, sample_value, false);
}

#[rstest]
#[case(&json!(""))]
fn string_length_empty(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string", "minLength": 0, "maxLength": 0});
    json_schema_check(schema, sample_value, true);
}

#[rstest]
#[case(&json!("a"))]
#[case(&json!("abc"))]
fn string_length_empy_failures(#[case] sample_value: &Value) {
    let schema = &json!({"type":"string", "minLength": 0, "maxLength": 0});
    json_schema_check(schema, sample_value, false);
}

#[test]
fn string_length_unsatisfiable() {
    json_err_test(
        &json!({"type":"string", "minLength": 2, "maxLength": 1}),
        "Unsatisfiable schema: minLength (2) is greater than maxLength (1)",
    );
}
