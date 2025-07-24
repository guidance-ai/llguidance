use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_schema_check};

#[derive(Debug)]
enum NumericBounds {
    Inclusive,
    Exclusive,
}

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
fn integer_lower_bound(
    #[values(NumericBounds::Inclusive, NumericBounds::Exclusive)] bound_type: NumericBounds,
    #[values(-11, -2, -1, 0, 1, 2, 11)] lower_bound: i64,
) {
    assert!(lower_bound.abs() < 20);
    let iterate_range = 200;
    let schema = match bound_type {
        NumericBounds::Inclusive => {
            json!({"type":"integer", "minimum": lower_bound})
        }
        NumericBounds::Exclusive => {
            json!({"type":"integer", "exclusiveMinimum": lower_bound})
        }
    };
    for i in -iterate_range..=iterate_range {
        let sample_value = json!(i);
        let should_pass = match bound_type {
            NumericBounds::Inclusive => i >= lower_bound,
            NumericBounds::Exclusive => i > lower_bound,
        };
        json_schema_check(&schema, &sample_value, should_pass);
    }
}

#[rstest]
fn integer_upper_bound(
    #[values(NumericBounds::Inclusive, NumericBounds::Exclusive)] bound_type: NumericBounds,
    #[values(-11, -2, -1, 0, 1, 2, 11)] upper_bound: i64,
) {
    assert!(upper_bound.abs() < 20);
    let iterate_range = 101;
    let schema = match bound_type {
        NumericBounds::Inclusive => {
            json!({"type":"integer", "maximum": upper_bound})
        }
        NumericBounds::Exclusive => {
            json!({"type":"integer", "exclusiveMaximum": upper_bound})
        }
    };
    for i in -iterate_range..=iterate_range {
        let sample_value = json!(i);
        let should_pass = match bound_type {
            NumericBounds::Inclusive => i <= upper_bound,
            NumericBounds::Exclusive => i < upper_bound,
        };
        json_schema_check(&schema, &sample_value, should_pass);
    }
}

#[rstest]
#[case(-10, -2)]
#[case(-10, -1)]
#[case(-10, 0)]
#[case(-10, 2)]
#[case(-3, -1)]
#[case(-3, 0)]
#[case(-3, 2)]
#[case(0, 2)]
#[case(0, 10)]
#[case(1, 3)]
#[case(1, 10)]
#[case(2, 10)]
fn integer_both_bounds(
    #[values(NumericBounds::Inclusive, NumericBounds::Exclusive)] lower_bound_type: NumericBounds,
    #[values(NumericBounds::Inclusive, NumericBounds::Exclusive)] upper_bound_type: NumericBounds,
    #[case] lower_bound: i32,
    #[case] upper_bound: i32,
) {
    assert!(lower_bound.abs() < 20);
    assert!(upper_bound.abs() < 20);

    let lb_str = match lower_bound_type {
        NumericBounds::Inclusive => "minimum",
        NumericBounds::Exclusive => "exclusiveMinimum",
    };
    let ub_str = match upper_bound_type {
        NumericBounds::Inclusive => "maximum",
        NumericBounds::Exclusive => "exclusiveMaximum",
    };
    let schema = json!({"type":"integer", lb_str:lower_bound, ub_str:upper_bound});

    let iterate_range = 101;
    for i in -iterate_range..=iterate_range {
        let sample_value = json!(i);
        let should_pass = match (&lower_bound_type, &upper_bound_type) {
            (NumericBounds::Inclusive, NumericBounds::Inclusive) => {
                i >= lower_bound && i <= upper_bound
            }
            (NumericBounds::Inclusive, NumericBounds::Exclusive) => {
                i >= lower_bound && i < upper_bound
            }
            (NumericBounds::Exclusive, NumericBounds::Inclusive) => {
                i > lower_bound && i <= upper_bound
            }
            (NumericBounds::Exclusive, NumericBounds::Exclusive) => {
                i > lower_bound && i < upper_bound
            }
        };
        json_schema_check(&schema, &sample_value, should_pass);
    }
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
#[case(&json!(-1))]
#[case(&json!(-1.0))]
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
