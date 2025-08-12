// This is for testing JSON string formats
// Only smoke testing for now; more comprehensive tests are in Python

use rstest::*;
use serde_json::json;

mod common_lark_utils;
use common_lark_utils::json_schema_check;

#[rstest]
#[case("1963-06-19T08:30:06.283185Z")]
pub fn valid_date_time(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"date-time"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("1963-06-38T08:30:06.283185Z")]
pub fn bad_date_time(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"date-time"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("08:30:06.283185Z")]
pub fn valid_time(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"time"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("28:30:06.283185Z")]
pub fn bad_time(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"time"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("1963-06-19")]
pub fn valid_date(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"date"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("1963-13-19")]
pub fn bad_date(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"date"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("P1M")]
pub fn valid_duration(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"duration"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("P2D1Y")]
pub fn bad_duration(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"duration"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("joe.bloggs@example.com")]
pub fn valid_email(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"email"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("joe.bloggs@@example.com")]
pub fn bad_email(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"email"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("hostnam3")]
pub fn valid_hostname(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"hostname"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("hostnam3-")]
pub fn bad_hostname(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"hostname"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("192.168.0.1")]
pub fn valid_ipv4(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"ipv4"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("192.168.0.0.1")]
pub fn bad_ipv4(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"ipv4"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("::42:ff:1")]
pub fn valid_ipv6(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"ipv6"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("1:1:1:1:1:1:1:1:1:1:1:1:1:1:1:1")]
pub fn bad_ipv6(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"ipv6"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("2eb8aa08-AA98-11ea-B4Aa-73B441D16380")]
pub fn valid_uuid(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"uuid"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("2eb8-aa08-aa98-11ea-b4aa73b44-1d16380")]
pub fn bad_uuid(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"uuid"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("Some string")]
pub fn valid_unknown(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"unknown"});
    json_schema_check(&schema, &json!(s), true);
}
