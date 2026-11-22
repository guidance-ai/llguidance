// Tests for `$id` (`id` in draft 4) and the `$ref` lookups that use it.

use rstest::*;
use serde_json::{json, Value};

use llg_test_utils::{json_err_test, json_test_many};

const ROOT_ID: &str = "https://example.com/root";
const DUPLICATE_ROOT_ID: &str =
    "duplicate schema URI: more than one schema has the URI https://example.com/root";
const DUPLICATE_X_ID: &str =
    "duplicate schema URI: more than one schema has the URI https://example.com/x";

const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";
const DRAFT_2019_09: &str = "https://json-schema.org/draft/2019-09/schema";
const DRAFT_07: &str = "http://json-schema.org/draft-07/schema#";
const DRAFT_06: &str = "http://json-schema.org/draft-06/schema#";
const DRAFT_04: &str = "http://json-schema.org/draft-04/schema#";

/// An object schema with the root's id and one string property "s".
/// Draft 4 uses the keyword `id`. The other drafts use `$id`.
fn root_schema(draft: &str, s: Value) -> Value {
    let id_key = if draft == DRAFT_04 { "id" } else { "$id" };
    let mut schema = json!({"$schema": draft, "type": "object", "properties": {"s": s}});
    schema[id_key] = json!(ROOT_ID);
    schema
}

fn integer_with_id(draft: &str, id: &str) -> Value {
    let id_key = if draft == DRAFT_04 { "id" } else { "$id" };
    let mut schema = json!({"type": "integer"});
    schema[id_key] = json!(id);
    schema
}

// `contentSchema` is an annotation. The `referencing` crate treats it as a subschema in
// these drafts, so it can claim the root's URI.
#[rstest]
fn content_schema_with_root_id_is_refused(
    #[values(DRAFT_2020_12, DRAFT_2019_09, DRAFT_04)] draft: &str,
) {
    let s = json!({"type": "string", "contentSchema": integer_with_id(draft, ROOT_ID)});
    json_err_test(&root_schema(draft, s), DUPLICATE_ROOT_ID);
}

#[test]
fn content_schema_with_root_id_is_refused_in_default_draft() {
    let schema = json!({
        "$id": ROOT_ID,
        "type": "object",
        "properties": {"s": {"type": "string", "contentSchema": {"$id": ROOT_ID, "type": "integer"}}}
    });
    json_err_test(&schema, DUPLICATE_ROOT_ID);
}

// Drafts 6 and 7 do not define `contentSchema`. Its `$id` does not identify a schema.
#[rstest]
fn content_schema_with_root_id_is_ignored(#[values(DRAFT_07, DRAFT_06)] draft: &str) {
    let s = json!({"type": "string", "contentSchema": integer_with_id(draft, ROOT_ID)});
    json_test_many(
        &root_schema(draft, s),
        &[json!({"s": "x"}), json!({})],
        &[json!(5), json!({"s": 5})],
    );
}

// Before draft 2019-09, the registry ignores an `$id` beside `$ref`.
#[test]
fn draft_07_id_beside_ref_is_ignored() {
    let schema = json!({
        "$schema": DRAFT_07,
        "$id": ROOT_ID,
        "type": "object",
        "properties": {"s": {"$id": ROOT_ID, "$ref": "#/definitions/x"}},
        "definitions": {"x": {"type": "string"}}
    });
    json_test_many(&schema, &[json!({"s": "x"})], &[json!(5), json!({"s": 5})]);
}

#[rstest]
fn property_with_root_id_is_refused(
    #[values(DRAFT_2020_12, DRAFT_2019_09, DRAFT_07, DRAFT_06, DRAFT_04)] draft: &str,
) {
    let schema = root_schema(draft, integer_with_id(draft, ROOT_ID));
    json_err_test(&schema, DUPLICATE_ROOT_ID);
}

#[rstest]
#[case::trailing_hash("https://example.com/root#")]
#[case::empty_fragment("#")]
#[case::empty("")]
#[case::upper_case_host("HTTPS://EXAMPLE.COM/root")]
#[case::dot_segment("https://example.com/a/../root")]
#[case::relative("root")]
fn property_with_root_uri_in_other_form_is_refused(#[case] id: &str) {
    let schema = root_schema(DRAFT_2020_12, integer_with_id(DRAFT_2020_12, id));
    json_err_test(&schema, DUPLICATE_ROOT_ID);
}

// The root lookup uses the root's URI, so a `$defs` entry with the root's `$id` is an error,
// although no `$ref` uses the entry.
#[test]
fn definition_with_root_id_is_refused() {
    let schema = json!({
        "$id": ROOT_ID,
        "type": "object",
        "properties": {"s": {"type": "string"}},
        "$defs": {"d": {"$id": ROOT_ID, "type": "integer"}}
    });
    json_err_test(&schema, DUPLICATE_ROOT_ID);
}

// A relative root `$id` "a/b.json" resolves against itself, so the registry also has the root
// under json-schema:///a/a/b.json. A subschema `$id` "b.json" resolves to that URI.
#[test]
fn subschema_with_relative_root_uri_is_refused() {
    let schema = json!({
        "$id": "a/b.json",
        "type": "object",
        "properties": {"s": {"$id": "b.json", "type": "integer"}}
    });
    json_err_test(
        &schema,
        "duplicate schema URI: more than one schema has the URI json-schema:///a/a/b.json",
    );
}

// In a draft 4 root without `id`, a subschema `id` "." resolves to the default URI.
#[test]
fn draft_04_subschema_with_dot_id_is_refused() {
    let schema = json!({
        "$schema": DRAFT_04,
        "type": "object",
        "properties": {"s": {"id": ".", "type": "integer"}}
    });
    json_err_test(
        &schema,
        "duplicate schema URI: more than one schema has the URI json-schema:///",
    );
}

#[test]
fn subschema_with_default_root_uri_is_refused() {
    let schema = json!({
        "type": "object",
        "properties": {"s": {"$id": "json-schema:///", "type": "integer"}}
    });
    json_err_test(
        &schema,
        "duplicate schema URI: more than one schema has the URI json-schema:///",
    );
}

// The `$ref` targets in the root must not resolve inside another schema with the root's `$id`.
#[rstest]
#[case::pointer("#/$defs/x")]
#[case::absolute_pointer("https://example.com/root#/$defs/x")]
#[case::recursive("#")]
fn ref_into_root_is_refused_when_subschema_has_root_id(#[case] reference: &str) {
    let schema = json!({
        "$id": ROOT_ID,
        "type": "object",
        "$defs": {"x": {"type": "string"}},
        "properties": {
            "s": {"$ref": reference},
            "t": {"type": "string", "contentSchema": {"$id": ROOT_ID, "$defs": {"x": {"type": "integer"}}}}
        }
    });
    json_err_test(&schema, DUPLICATE_ROOT_ID);
}

#[rstest]
#[case::relative("x")]
#[case::absolute("https://example.com/x")]
#[case::with_pointer("x#/properties/a")]
#[case::with_anchor("x#foo")]
fn ref_to_duplicate_id_is_refused(#[case] reference: &str) {
    let schema = json!({
        "$id": ROOT_ID,
        "type": "object",
        "properties": {"s": {"$ref": reference}},
        "$defs": {
            "a": {"$id": "https://example.com/x", "type": "object", "properties": {"a": {"$anchor": "foo", "type": "string"}}},
            "b": {"$id": "https://example.com/x", "type": "object", "properties": {"a": {"type": "integer"}}}
        }
    });
    json_err_test(&schema, DUPLICATE_X_ID);
}

#[test]
fn ref_to_duplicate_id_without_root_id_is_refused() {
    let schema = json!({
        "type": "object",
        "properties": {"s": {"$ref": "x.json"}},
        "$defs": {
            "a": {"$id": "x.json", "type": "string"},
            "b": {"$id": "x.json", "type": "integer"}
        }
    });
    json_err_test(
        &schema,
        "duplicate schema URI: more than one schema has the URI json-schema:///x.json",
    );
}

// A duplicate `$id` that no lookup uses does not change the grammar, so it is not an error.
#[test]
fn unused_duplicate_id_compiles() {
    let schema = json!({
        "$id": ROOT_ID,
        "type": "object",
        "properties": {"s": {"type": "string"}},
        "$defs": {
            "a": {"$id": "https://example.com/x", "type": "string"},
            "b": {"$id": "https://example.com/x", "type": "integer"}
        }
    });
    json_test_many(&schema, &[json!({"s": "x"})], &[json!(5), json!({"s": 5})]);
}

// Copies with equal JSON values (key order ignored) are not an error. A lookup returns the
// last copy.
#[rstest]
#[case::relative("x")]
#[case::with_anchor("x#foo")]
fn ref_to_equal_copies_resolves(#[case] reference: &str) {
    let copy = json!({"$id": "https://example.com/x", "$anchor": "foo", "type": "string"});
    let schema = json!({
        "$id": ROOT_ID,
        "type": "object",
        "properties": {"s": {"$ref": reference}},
        "$defs": {"a": copy, "b": copy}
    });
    json_test_many(&schema, &[json!({"s": "x"})], &[json!({"s": 5})]);
}

#[test]
fn duplicate_root_id_is_refused_in_lenient_mode() {
    let mut schema = root_schema(DRAFT_2020_12, integer_with_id(DRAFT_2020_12, ROOT_ID));
    schema["x-guidance"] = json!({"lenient": true});
    json_err_test(&schema, DUPLICATE_ROOT_ID);
}

#[rstest]
#[case::no_id(json!({"type": "string", "contentSchema": {"type": "integer"}}))]
#[case::other_id(json!({"type": "string", "contentSchema": {"$id": "https://example.com/other", "type": "integer"}}))]
#[case::id_in_default(json!({"type": "string", "default": {"$id": ROOT_ID, "type": "integer"}}))]
#[case::id_in_examples(json!({"type": "string", "examples": [{"$id": ROOT_ID}]}))]
#[case::id_in_const(json!({"anyOf": [{"type": "string"}, {"const": {"$id": ROOT_ID, "type": "integer"}}]}))]
#[case::id_in_enum(json!({"enum": ["x", {"$id": ROOT_ID, "type": "integer"}]}))]
fn root_grammar_is_kept_without_duplicate_id(#[case] s: Value) {
    json_test_many(
        &root_schema(DRAFT_2020_12, s),
        &[json!({"s": "x"})],
        &[json!(5), json!({"s": 5})],
    );
}

#[test]
fn distinct_ids_resolve() {
    let schema = json!({
        "$id": ROOT_ID,
        "type": "object",
        "properties": {
            "s": {"$ref": "https://example.com/str"},
            "n": {"$ref": "int"}
        },
        "$defs": {
            "str": {"$id": "https://example.com/str", "type": "string"},
            "int": {"$id": "int", "type": "integer"}
        }
    });
    json_test_many(
        &schema,
        &[json!({"s": "x", "n": 5})],
        &[json!({"s": 5}), json!({"n": "x"})],
    );
}

// A `$ref` inside a resource resolves against the URI of that resource.
#[test]
fn nested_resources_resolve_against_own_id() {
    let schema = json!({
        "$id": "https://example.com/a/root",
        "type": "object",
        "properties": {"s": {"$ref": "other"}, "t": {"$ref": "#/$defs/b"}, "u": {"$ref": "../inner"}},
        "$defs": {
            "b": {"type": "string"},
            "o": {
                "$id": "other",
                "$ref": "#/$defs/b",
                "$defs": {
                    "b": {"type": "integer"},
                    "inner": {"$id": "../inner", "type": "boolean"}
                }
            }
        }
    });
    json_test_many(
        &schema,
        &[json!({"s": 5, "t": "x", "u": true})],
        &[json!({"s": "x"}), json!({"t": 5}), json!({"u": 5})],
    );
}

#[test]
fn recursive_ref_with_root_id_resolves() {
    let schema = json!({
        "$id": ROOT_ID,
        "type": "object",
        "properties": {"s": {"type": "string"}, "r": {"$ref": "#"}}
    });
    json_test_many(
        &schema,
        &[json!({"r": {"s": "x"}})],
        &[json!({"r": 5}), json!({"r": {"s": 5}})],
    );
}
