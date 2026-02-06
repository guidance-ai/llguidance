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

// RFC 3986 Section 1.1.2 examples
#[rstest]
#[case("ftp://ftp.is.co.za/rfc/rfc1808.txt")]
#[case("http://www.ietf.org/rfc/rfc2396.txt")]
#[case("ldap://[2001:db8::7]/c=GB?objectClass?one")]
#[case("mailto:John.Doe@example.com")]
#[case("news:comp.infosystems.www.servers.unix")]
#[case("tel:+1-816-555-1212")]
#[case("telnet://192.0.2.16:80/")]
#[case("urn:oasis:names:specification:docbook:dtd:xml:4.1.2")]
// Additional valid URIs
#[case("http://example.com")]
#[case("https://example.com/path/to/resource")]
#[case("https://example.com?query=value")]
#[case("https://example.com#section")]
#[case("http://example.com:8080/")]
#[case("ftp://user:pass@ftp.example.com/")]
#[case("http://192.168.1.1/")]
#[case("http://[::1]:8080/path")]
#[case("https://example.com/path%20with%20spaces")]
#[case("file:///path/to/file")]
#[case("tel:+1-201-555-0123;ext=456")]
pub fn valid_uri(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"uri"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("//example.com/path")]        // No scheme
#[case("123://example.com")]          // Invalid scheme (starts with digit)
#[case("http://example.com/%GG")]     // Invalid percent-encoding
#[case("/path/to/resource")]          // Bare path (no scheme)
#[case("http://example.com/path with spaces")] // Unencoded spaces
pub fn bad_uri(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"uri"});
    json_schema_check(&schema, &json!(s), false);
}

#[rstest]
#[case("Some string")]
pub fn valid_unknown(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"unknown"});
    json_schema_check(&schema, &json!(s), true);
}
