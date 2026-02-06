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
// IPv6 address variations
#[case("http://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]/")] // Full IPv6
#[case("http://[2001:db8:85a3::8a2e:370:7334]/")]
// Compressed IPv6
// #[case("http://[::ffff:192.0.2.1]/")]                      // IPv4-mapped IPv6 style - TODO: not yet supported
#[case("http://[::1]/")] // Loopback
#[case("http://[::]/")]
// All zeros
// #[case("http://[fe80::1%25eth0]/")]                        // Link-local with zone ID - TODO: not yet supported
#[case("http://[2001:db8::1]:8080/path")] // IPv6 with port and path
#[case("http://user:pass@[2001:db8::1]:8080/path/to/resource")] // IPv6 with userinfo and path
#[case("http://[v1.test]/")] // IPvFuture format
// Additional schemes
#[case("ssh://git@github.com:22/user/repo.git")] // SSH
#[case("git://github.com/user/repo.git")] // Git protocol
#[case("svn://svn.example.com/repo/trunk")] // Subversion
#[case("sftp://user@host.example.com/path/to/file")] // SFTP
#[case("s3://bucket-name/key/path")] // Amazon S3
#[case("data:text/plain;base64,SGVsbG8=")] // Data URI
#[case("javascript:void(0)")] // JavaScript (common in web)
#[case("magnet:?xt=urn:btih:abc123")] // Magnet link
#[case("redis://localhost:6379/0")] // Redis
#[case("postgres://user:pass@localhost:5432/db")] // PostgreSQL
#[case("mysql://user:pass@localhost:3306/db")] // MySQL
#[case("mongodb://localhost:27017/mydb")] // MongoDB
#[case("amqp://user:pass@host:5672/vhost")] // AMQP (RabbitMQ)
#[case("ws://example.com/socket")] // WebSocket
#[case("wss://example.com/socket")] // WebSocket Secure
#[case("irc://irc.example.com:6667/channel")] // IRC
#[case("xmpp:user@example.com")] // XMPP/Jabber
#[case("sip:user@example.com")] // SIP
#[case("sips:user@example.com:5061")] // SIPS (secure SIP)
#[case("rtsp://media.example.com:554/stream")] // RTSP (streaming)
#[case("spotify:track:4uLU6hMCjMI75M1A2tKUQC")] // Spotify
#[case("slack://channel?team=T123&id=C456")] // Slack
#[case("vscode://file/path/to/file.txt")] // VS Code
#[case("x-custom-scheme://anything/goes/here")] // Custom scheme with x- prefix
// Schemes with +, -, . characters
#[case("git+ssh://git@github.com/user/repo.git")] // Git over SSH
#[case("coap+tcp://example.com/sensor")] // CoAP over TCP
#[case("ms-windows-store://pdp?productid=abc123")] // Microsoft Store
// Complex URI with multiple components
#[case("https://user:pass@api.example.com:8443/v2/users/123/profile?name=John%20Doe&active=true&sort=desc#contact-info")]
// JSON Schema Test Suite cases (draft2020-12)
#[case("http://foo.bar/?baz=qux#quux")] // URL with anchor tag
#[case("http://foo.com/blah_(wikipedia)_blah#cite-1")] // URL with parentheses and anchor
#[case("http://foo.bar/?q=Test%20URL-encoded%20stuff")] // URL-encoded query
#[case("http://xn--nw2a.xn--j6w193g/")] // Puny-coded URL
#[case("http://-.~_!$&'()*+,;=:%40:80%2f::::::@example.com")] // Many special characters
#[case("http://223.255.255.254")] // IPv4-based URL
pub fn valid_uri(#[case] s: &str) {
    let schema = json!({"type":"string", "format":"uri"});
    json_schema_check(&schema, &json!(s), true);
}

#[rstest]
#[case("//example.com/path")] // No scheme
#[case("123://example.com")] // Invalid scheme (starts with digit)
#[case("http://example.com/%GG")] // Invalid percent-encoding
#[case("/path/to/resource")] // Bare path (no scheme)
#[case("http://example.com/path with spaces")] // Unencoded spaces
// Invalid IPv6
#[case("http://[:::]/")] // Too many colons
#[case("http://2001:db8::1/")] // IPv6 without brackets
#[case("http://[2001:db8:85a3:0000:0000:8a2e:0370:7334:extra]/")] // Too many groups
// JSON Schema Test Suite invalid cases (draft2020-12)
#[case("\\\\WINDOWS\\fileshare")] // Windows path
#[case("abc")] // Just a string, no scheme
#[case("http:// shouldfail.com")] // Space after scheme
#[case(":// should fail")] // Missing scheme with spaces
#[case("bar,baz:foo")]
// Comma in scheme
#[case("https://[@example.org/test.txt")] // Invalid userinfo with [
#[case("https://example.org/foobar\\.txt")] // Backslash in path
#[case("https://example.org/foobar<>.txt")] // Invalid <> characters
#[case("https://example.org/foobar{}.txt")] // Invalid {} characters
#[case("https://example.org/foobar^.txt")] // Invalid ^ character
#[case("https://example.org/foobar`.txt")] // Invalid ` character
#[case("https://example.org/foo bar.txt")] // Invalid SPACE character
#[case("https://example.org/foobar|.txt")] // Invalid | character
// Scheme issues
#[case("a@b://example.com")] // Invalid @ in scheme
#[case("://example.com")] // Empty scheme
// Authority issues
#[case("http://example.com:abc/")] // Non-numeric port
#[case("http://user@@example.com/")] // Double @
#[case("http://[::1/path")] // Unclosed IPv6 bracket
#[case("http://exa mple.com/")] // Space in host
// Encoding issues
#[case("http://example.com/%")] // Incomplete percent encoding
#[case("http://example.com/%a")] // Incomplete percent encoding (one hex digit)
// Control characters
#[case("http://example.com/path\x00")] // Null byte
#[case("http://example.com/\t")] // Tab character
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
