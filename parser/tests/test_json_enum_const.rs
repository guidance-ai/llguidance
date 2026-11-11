use rstest::*;
use serde_json::{json, Value};

use llg_test_utils::{json_err_test, json_schema_check};

#[rstest]
fn const_null() {
    // This may fall into the 'Excessive Pedantry' bin
    let schema = &json!({"type":"null", "const":null});
    json_schema_check(schema, &json!(null), true);
}

#[rstest]
#[case(&json!(true), false)]
#[case(&json!(false), true)]
fn const_boolean(#[case] sample: &Value, #[case] expected_pass: bool) {
    // Some lovely ambiguity here. The point is that this is a boolean
    // which must always be 'false'.
    let schema = json!({
      "type": "boolean",
      "const": false
    });
    json_schema_check(&schema, sample, expected_pass);
}

#[rstest]
#[case(&json!({"name" : "John", "age": 42}), true)]
#[case(&json!({"name" : "Jane", "age": 42}), false)]
#[case(&json!({"name" : "John", "age": 52}), false)]
#[case(&json!({"name" : "John", "age": 42, "location" : "USA"}), false)]
fn const_object(#[case] sample: &Value, #[case] expected_pass: bool) {
    let schema = json!({
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "const": { "name": "John", "age": 42 }
    });
    json_schema_check(&schema, sample, expected_pass);
}

#[rstest]
#[case(&json!({"country": "US"}), true)]
#[case(&json!({"country": "CA"}), false)]
fn const_property(#[case] sample: &Value, #[case] expected_pass: bool) {
    let schema = json!({
      "properties": {
        "country": {
          "const": "US"
        }
      }
    });
    json_schema_check(&schema, sample, expected_pass);
}

#[rstest]
#[case(&json!([10]), false)]
#[case(&json!([10, 20]), true)]
#[case(&json!([10, 20, 30]), false)]
#[case(&json!([10, "a"]), false)]
fn const_array(#[case] sample: &Value, #[case] expected_pass: bool) {
    let schema = json!({
      "type": "array",
      "const": [10, 20]
    });
    json_schema_check(&schema, sample, expected_pass);
}

#[rstest]
#[case(&json!([2, 1]), false)]
#[case(&json!([2, 3]), true)]
#[case(&json!([2, 3, true, false]), true)]
#[case(&json!([2, 3, 1]), false)]
fn array_with_const_prefix(#[case] sample_array: &Value, #[case] expected_pass: bool) {
    let schema = json!({ "type": "array",
      "prefixItems": [
        { "const": 2 },
        { "const": 3 }
      ],
      "items": { "type": "boolean" }, // Remaining items must be booleans
    });
    json_schema_check(&schema, sample_array, expected_pass);
}

#[rstest]
#[case(&json!(6), true)]
#[case(&json!(9), true)]
#[case(&json!(13), true)]
#[case(&json!(42), false)]
fn enum_check(#[case] sample: &Value, #[case] expected_pass: bool) {
    let schema = json!({"enum": [6, 9, 13]});

    json_schema_check(&schema, sample, expected_pass);
}

fn item_ids(n: usize) -> Vec<Value> {
    (1..=n).map(|i| json!(format!("item-{i}"))).collect()
}

fn array_property(items: Value) -> Value {
    json!({
        "type": "object",
        "properties": { "r": { "type": "array", "items": items } },
        "required": ["r"],
        "additionalProperties": false
    })
}

#[rstest]
#[case(&json!({ "r": ["item-7", "item-1000"] }), true)]
#[case(&json!({ "r": ["item-1001"] }), false)]
#[case(&json!({ "r": ["ITEM-7"] }), false)]
#[case(&json!({ "r": ["item-12345"] }), false)]
fn large_string_enum_with_pattern_and_length(#[case] sample: &Value, #[case] expected_pass: bool) {
    let mut values = item_ids(1000);
    values.push(json!("ITEM-7"));
    values.push(json!("item-12345"));
    let schema = array_property(json!({
        "type": "string",
        "enum": values,
        "pattern": "^item-[0-9]{1,4}$",
        "minLength": 1,
        "maxLength": 12
    }));
    json_schema_check(&schema, sample, expected_pass);
}

#[rstest]
#[case(&json!({ "r": ["item-7", "item-12"] }), true)]
#[case(&json!({ "r": ["ITEM-7"] }), false)]
fn large_string_any_of_const_with_pattern(#[case] sample: &Value, #[case] expected_pass: bool) {
    let mut values = item_ids(300);
    values.push(json!("ITEM-7"));
    let options: Vec<Value> = values.into_iter().map(|v| json!({ "const": v })).collect();
    let schema = array_property(json!({
        "type": "string",
        "pattern": "^item-[0-9]{1,4}$",
        "anyOf": options
    }));
    json_schema_check(&schema, sample, expected_pass);
}

#[rstest]
#[case(&json!({ "r": ["item-7", "item-12"] }), true)]
#[case(&json!({ "r": ["ITEM-7"] }), false)]
fn large_string_enum_with_pattern_in_all_of(#[case] sample: &Value, #[case] expected_pass: bool) {
    let mut values = item_ids(300);
    values.push(json!("ITEM-7"));
    let schema = array_property(json!({
        "allOf": [
            { "type": "string", "enum": values },
            { "type": "string", "pattern": "^item-[0-9]{1,4}$", "maxLength": 12 }
        ]
    }));
    json_schema_check(&schema, sample, expected_pass);
}

#[rstest]
#[case(&json!("a"), true)]
#[case(&json!("bb"), false)]
#[case(&json!("C"), false)]
fn string_enum_filtered_by_pattern(#[case] sample: &Value, #[case] expected_pass: bool) {
    let schema = json!({
        "type": "string",
        "enum": ["a", "bb", "C"],
        "pattern": "^[a-z]$"
    });
    json_schema_check(&schema, sample, expected_pass);
}

#[rstest]
#[case(&json!({ "type": "string", "enum": ["A", "B"], "pattern": "^[a-z]+$" }))]
#[case(&json!({ "type": "string", "const": "abc", "pattern": "^[0-9]+$" }))]
#[case(&json!({ "type": "string", "const": "abc", "format": "date" }))]
fn string_literal_not_matching_pattern_is_unsatisfiable(#[case] schema: &Value) {
    json_err_test(schema, "Unsatisfiable schema");
}
