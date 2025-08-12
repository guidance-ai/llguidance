// This is for testing JSON string formats
// Only smoke testing for now

use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::json_schema_check;

#[rstest]
#[case(&json!("1963-06-19T08:30:06.283185Z"))]
pub fn valid_date_time(#[case] value: &Value) {
    let schema = json!({"type":"string", "format":"date-time"});
    json_schema_check(&schema, &value, true);
}

#[rstest]
#[case("08:30:06.283185Z")]
pub fn valid_time(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"time"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("1963-06-19")]
pub fn valid_date(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"date"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("P1M")]
pub fn valid_duration(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"duration"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("joe.bloggs@example.com")]
pub fn valid_email(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"email"});
    json_schema_check(&schema, &json!(s), true);
}
