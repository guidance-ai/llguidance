// This is for testing anyOf and allOf in JSON schema

use lazy_static::lazy_static;
use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::json_schema_check;

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
