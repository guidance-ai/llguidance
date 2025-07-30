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
