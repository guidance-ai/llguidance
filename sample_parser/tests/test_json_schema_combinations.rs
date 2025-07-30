// This is for testing anyOf and allOf in JSON schema

use lazy_static::lazy_static;
use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_schema_check};

lazy_static! {
    static ref SIMPLE_ANYOF: Value = json!({"anyOf": [
        {"type": "integer"},
        {"type": "boolean"}
    ]});
}

#[rstest]
fn simple_anyof(#[values(json!(42), json!(true))] sample: Value) {
    json_schema_check(&SIMPLE_ANYOF, &sample, true);
}

#[rstest]
fn simple_anyof_failures(#[values(json!("string"), json!(1.2), json!([1, 2]))] sample: Value) {
    json_schema_check(&SIMPLE_ANYOF, &sample, false);
}

#[rstest]
#[case(&json!(true), true)]
#[case(&json!(42), true)]
#[case(&json!("string"), false)]
#[case(&json!([1, 2]), false)]
fn type_as_list(#[case] sample: &Value, #[case] expected_pass: bool) {
    // Turns out that "type" can be a list, which acts like anyOf
    let schema = json!({"type": ["boolean", "integer"]});
    json_schema_check(&schema, sample, expected_pass);
}

lazy_static! {
    static ref SIMPLE_ALLOF: Value = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "allOf": [
            {"properties": {"foo": {"type": "string"}}, "required": ["foo"]},
            {"properties": {"bar": {"type": "integer"}}, "required": ["bar"]},
        ],
    });
}

#[rstest]
fn simple_allof(#[values(json!({"foo": "hello", "bar": 42}))] sample: Value) {
    json_schema_check(&SIMPLE_ALLOF, &sample, true);
}

#[rstest]
fn simple_allof_failures(
    #[values(json!({"foo": "hello"}), json!({"bar": 42}), json!({"foo": "hello", "bar": "not a number"}) )]
    sample: Value,
) {
    json_schema_check(&SIMPLE_ALLOF, &sample, false);
}

lazy_static! {
    static ref ALLOF_WITH_BASE: Value = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {"bar": {"type": "integer"}},
            "required": ["bar"],
            "allOf": [
                {"properties": {"foo": {"type": "string"}}, "required": ["foo"]},
                {"properties": {"baz": {"type": "null"}}, "required": ["baz"]},
            ],
    });
}

#[rstest]
#[case(&json!({"bar": 2, "foo": "quux", "baz": null}), true)]
#[case(&json!({"foo": "quux", "baz": null}), false)]
#[case(&json!({"bar": 2, "baz": null}), false)]
#[case(&json!({"bar": 2, "foo": "quux"}), false)]
#[case(&json!({"bar": 2}), false)]
fn allof_with_base(#[case] sample: &Value, #[case] expected_pass: bool) {
    json_schema_check(&ALLOF_WITH_BASE, sample, expected_pass);
}

#[rstest]
#[case(-35, false)]
#[case(0, false)]
#[case(29, false)]
#[case(30, true)]
#[case(35, true)]
#[case(381925, true)]
fn allof_simple_minimum(#[case] value: i32, #[case] expected_pass: bool) {
    let schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "allOf": [{"minimum": 30}, {"minimum": 20}],
    });
    json_schema_check(&schema, &json!(value), expected_pass);
}

#[rstest]
#[case("a", true)]
#[case("b", true)]
// Issue 224 #[case("bb", false)]
// Issue 224 #[case("aa", false)]
#[case("", false)]
#[case(" ", false)]
fn allof_string_patterns(#[case] value: &str, #[case] expected_pass: bool) {
    let schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "allOf": [
            {"type": "string", "pattern": r"\w+"},
            {"type": "string", "pattern": r"\w?"}
        ]
    });
    json_schema_check(&schema, &json!(value), expected_pass);
}

#[rstest]
fn allof_unsatisfiable_false_schema(#[values(true, false)] other_schema: bool) {
    let schema = &json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "allOf": [other_schema, false],
    });
    json_err_test(schema, "Unsatisfiable schema: schema is false");
}

#[rstest]
fn allof_unsatisfiable() {
    let schema = &json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "allOf": [
            {"type": "integer", "minimum": 10},
            {"type": "integer", "maximum": 5}
        ]
    });
    json_err_test(
        schema,
        "Unsatisfiable schema: minimum (10) is greater than maximum (5)",
    );
}

#[rstest]
fn allof_anyof_oneof_combined() {
    let schema = &json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [{"enum": [2, 6, 10, 30]}],
            "anyOf": [{"enum": [3, 6, 15, 30]}],
            "oneOf": [{"enum": [5, 10, 15, 30]}],
    });

    for i in -35..=35 {
        let value = json!(i);
        let expected_pass = i == 30;
        json_schema_check(schema, &value, expected_pass);
    }
}
