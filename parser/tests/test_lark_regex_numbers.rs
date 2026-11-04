use llg_test_utils::{lark_err_test, lark_str_test_many_quiet};
use rstest::rstest;
use serde_json::{json, Value};

fn grammar(keyword: &str, schema: &Value) -> String {
    format!("start: {keyword} {schema}")
}

#[rstest]
#[case::integer(json!({"type": "integer"}), &["0", "-7", "123456789"], &["01", "+1", "1.5", "1e3", "FINAL_REJECT:-"])]
#[case::integer_range(json!({"type": "integer", "minimum": 1, "maximum": 7}), &["1", "7"], &["0", "8", "-1", "10"])]
#[case::integer_exclusive_multiple(json!({"type": "integer", "exclusiveMinimum": 0, "multipleOf": 10}), &["10", "2500"], &["0", "-10", "FINAL_REJECT:15"])]
#[case::integer_draft4_exclusive(json!({"type": "integer", "minimum": 0, "exclusiveMinimum": true}), &["1"], &["0"])]
#[case::number(json!({"type": "number"}), &["0", "-0.5", "3.25", "1e3", "2E-7"], &["FINAL_REJECT:1.", ".5", "+1", "01"])]
#[case::number_range(json!({"type": "number", "minimum": -17.3, "maximum": 33.721}), &["-17.3", "0", "33.7", "33.721"], &["-17.31", "33.7211", "34"])]
#[case::number_exclusive(json!({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}), &["0.5", "0.999"], &["FINAL_REJECT:0", "1", "-0.5"])]
#[case::number_multiple(json!({"type": "number", "multipleOf": 0.5}), &["0", "1.5", "-2"], &["1.25"])]
fn numbers_match_like_json(
    #[case] schema: Value,
    #[case] valid: &[&str],
    #[case] invalid: &[&str],
) {
    for keyword in ["%json", "%regex"] {
        lark_str_test_many_quiet(&grammar(keyword, &schema), valid, invalid);
    }
}

#[rstest]
#[case::empty_integer_range(json!({"type": "integer", "minimum": 5, "maximum": 1}), "Unsatisfiable schema")]
#[case::no_integer_in_range(json!({"type": "integer", "minimum": 1.2, "maximum": 1.8}), "contains no integer values")]
#[case::multiple_of_too_precise(json!({"type": "number", "multipleOf": 0.12345678912}), "too many digits")]
#[case::null_minimum(json!({"type": "integer", "minimum": null}), "Expected f64 for 'minimum'")]
#[case::string_minimum(json!({"type": "integer", "minimum": "1"}), "Expected f64 for 'minimum'")]
#[case::null_multiple_of(json!({"type": "number", "multipleOf": null}), "Expected f64 for 'multipleOf'")]
fn number_errors_match_json(#[case] schema: Value, #[case] error: &str) {
    for keyword in ["%json", "%regex"] {
        lark_err_test(&grammar(keyword, &schema), error);
    }
}

#[test]
fn numbers_are_terminals() {
    lark_str_test_many_quiet(
        r#"
            start: DAY
            DAY: %regex {"type": "integer", "minimum": 1, "maximum": 7} & ~/7/
        "#,
        &["1", "6"],
        &["7", "0"],
    );
    lark_str_test_many_quiet(
        r#"
            start: VALUE
            VALUE: "(" %regex {"type": "number", "maximum": 1} ")"
        "#,
        &["(0.5)", "(-3)"],
        &["(2)", "FINAL_REJECT:(0.5"],
    );
}

#[rstest]
#[case::unsupported_type(json!({"type": "boolean"}), "unsupported %regex type")]
#[case::unknown_keyword(json!({"type": "integer", "foo": 1}), "unknown field `foo`")]
#[case::keyword_without_type(json!({"minimum": 1}), "requires \"type\"")]
#[case::keyword_with_substring(json!({"substring_words": "a b", "minimum": 1}), "requires \"type\"")]
#[case::mixed_with_substring(json!({"type": "integer", "substring_words": "a b"}), "only one field can be set on %regex")]
fn invalid_number_regexes_are_rejected(#[case] schema: Value, #[case] error: &str) {
    lark_err_test(&grammar("%regex", &schema), error);
}
