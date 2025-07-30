use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::json_schema_check;

#[rstest]
#[case(&json!({"country": "US"}), true)]
#[case(&json!({"country": "CA"}), false)]
fn const_check(#[case] sample: &Value, #[case] expected_pass: bool) {
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
#[case(&json!(6), true)]
#[case(&json!(9), true)]
#[case(&json!(13), true)]
#[case(&json!(42), false)]
fn enum_check(#[case] sample: &Value, #[case] expected_pass: bool) {
    let schema = json!({"enum": [6, 9, 13]});

    json_schema_check(&schema, sample, expected_pass);
}
