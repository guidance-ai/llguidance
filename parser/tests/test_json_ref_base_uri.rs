//! A relative `$ref` resolves against the base URI of the schema resource that
//! holds it. When a `$ref` reaches into another resource through a fragment (a
//! JSON pointer or an anchor), the relative references inside the target must
//! resolve against the target's resource, not against the referring resource.

use rstest::*;
use serde_json::{json, Value};

use llg_test_utils::json_test_many;

const ROOT: &str = "https://example.com/root";

/// An object with one required property `s`, and the given definitions.
fn object_with(s: Value, defs: Value) -> Value {
    json!({
        "$id": ROOT,
        "type": "object",
        "properties": {"s": s},
        "required": ["s"],
        "additionalProperties": false,
        "$defs": defs
    })
}

/// The embedded resource `https://example.com/other`. Its `b` is an integer.
fn other() -> Value {
    json!({"$id": "other", "$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"type": "integer"}}})
}

fn integer_only(schema: Value) -> (Value, Vec<Value>, Vec<Value>) {
    (schema, vec![json!({"s": 5})], vec![json!({"s": "x"})])
}

#[rstest]
// `other#/$defs/a` is in `other`, so its `#/$defs/b` is the integer, not the root's string.
#[case::pointer_into_resource(integer_only(object_with(
    json!({"$ref": "other#/$defs/a"}),
    json!({"b": {"type": "string"}, "o": other()}),
)))]
// The same through an anchor.
#[case::anchor_into_resource(integer_only(object_with(
    json!({"$ref": "other#foo"}),
    json!({"b": {"type": "string"},
           "o": {"$id": "other", "$defs": {"a": {"$anchor": "foo", "$ref": "#/$defs/b"},
                                           "b": {"type": "integer"}}}}),
)))]
// The pointer crosses the nested resource `n` (https://example.com/n), so `#/$defs/b` is in `n`.
#[case::pointer_through_nested_resource(integer_only(object_with(
    json!({"$ref": "other#/$defs/n/$defs/a"}),
    json!({"b": {"type": "string"},
           "o": {"$id": "other", "$defs": {"b": {"type": "string"},
                                           "n": {"$id": "n", "$defs": {"a": {"$ref": "#/$defs/b"},
                                                                       "b": {"type": "integer"}}}}}}),
)))]
// The root has no `b`. The schema is valid, and `#/$defs/b` is found in `other`.
#[case::target_only_in_its_resource(integer_only(object_with(
    json!({"$ref": "other#/$defs/a"}),
    json!({"o": other()}),
)))]
// Draft-07 uses `definitions`, with the same rule.
#[case::draft_07_pointer_into_resource(integer_only(json!({
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": ROOT,
    "type": "object",
    "properties": {"s": {"$ref": "other#/definitions/a"}},
    "required": ["s"],
    "additionalProperties": false,
    "definitions": {
        "b": {"type": "string"},
        "o": {"$id": "other", "definitions": {"a": {"$ref": "#/definitions/b"}, "b": {"type": "integer"}}}
    }
})))]
// A `$ref` with sibling keywords goes through the same lookup.
#[case::pointer_with_sibling_keywords(integer_only(object_with(
    json!({"$ref": "other#/$defs/a", "minimum": 0}),
    json!({"b": {"type": "string"}, "o": other()}),
)))]
fn relative_ref_in_target_resolves_in_target_resource(
    #[case] case: (Value, Vec<Value>, Vec<Value>),
) {
    let (schema, passing, failing) = case;
    json_test_many(&schema, &passing, &failing);
}

#[test]
fn pointer_back_into_root_resolves_in_root() {
    // Inside `other`, `root#/$defs/c` reaches into the root. The `#/$defs/d` in `c` is the root's string.
    let schema = object_with(
        json!({"$ref": "other"}),
        json!({
            "c": {"$ref": "#/$defs/d"},
            "d": {"type": "string"},
            "o": {"$id": "other", "type": "object", "properties": {"t": {"$ref": "root#/$defs/c"}},
                  "required": ["t"], "additionalProperties": false,
                  "$defs": {"d": {"type": "integer"}}}
        }),
    );
    json_test_many(
        &schema,
        &[json!({"s": {"t": "x"}})],
        &[json!({"s": {"t": 5}})],
    );
}

#[rstest]
// A `$ref` to a whole embedded resource uses that resource's base URI.
#[case::whole_resource(integer_only(object_with(
    json!({"$ref": "other"}),
    json!({"b": {"type": "string"},
           "o": {"$id": "other", "$ref": "#/$defs/b", "$defs": {"b": {"type": "integer"}}}}),
)))]
// A relative `$id` with a path is applied once, not twice.
#[case::whole_resource_with_path_id(integer_only(json!({
    "$id": "https://example.com/schemas/root.json",
    "type": "object",
    "properties": {"s": {"$ref": "sub/x.json"}},
    "required": ["s"],
    "additionalProperties": false,
    "$defs": {
        "k": {"type": "string"},
        "o": {"$id": "sub/x.json", "$ref": "#/$defs/k", "$defs": {"k": {"type": "integer"}}}
    }
})))]
// A pointer to a target that has its own `$id` with a path.
#[case::pointer_to_target_with_path_id(integer_only(json!({
    "$id": "https://example.com/schemas/root.json",
    "type": "object",
    "properties": {"s": {"$ref": "#/$defs/o"}},
    "required": ["s"],
    "additionalProperties": false,
    "$defs": {
        "k": {"type": "string"},
        "o": {"$id": "sub/x.json", "$ref": "#/$defs/k", "$defs": {"k": {"type": "integer"}}}
    }
})))]
// A pointer inside one document, with no `$id` on the way.
#[case::same_document_pointer(integer_only(object_with(
    json!({"$ref": "#/$defs/a"}),
    json!({"a": {"$ref": "#/$defs/b"}, "b": {"type": "integer"}}),
)))]
fn relative_ref_keeps_resolving_as_before(#[case] case: (Value, Vec<Value>, Vec<Value>)) {
    let (schema, passing, failing) = case;
    json_test_many(&schema, &passing, &failing);
}

#[test]
fn recursive_root_ref_with_id() {
    let schema = json!({
        "$id": ROOT,
        "type": "object",
        "properties": {"n": {"type": "integer"}, "c": {"$ref": "#"}},
        "required": ["n"],
        "additionalProperties": false
    });
    json_test_many(
        &schema,
        &[
            json!({"n": 1}),
            json!({"n": 1, "c": {"n": 2, "c": {"n": 3}}}),
        ],
        &[json!({"n": "x"}), json!({"n": 1, "c": {"n": "x"}})],
    );
}

#[test]
fn recursive_root_ref_without_id() {
    let schema = json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}, "c": {"$ref": "#"}},
        "required": ["n"],
        "additionalProperties": false
    });
    json_test_many(
        &schema,
        &[json!({"n": 1}), json!({"n": 1, "c": {"n": 2}})],
        &[json!({"n": 1, "c": {"n": "x"}})],
    );
}

#[test]
fn inline_resource_uses_its_own_base() {
    // No `$ref` reaches the embedded resource; it sits inline under `s`.
    let schema = object_with(
        json!({"$id": "other", "$ref": "#/$defs/b", "$defs": {"b": {"type": "integer"}}}),
        json!({"b": {"type": "string"}}),
    );
    json_test_many(&schema, &[json!({"s": 5})], &[json!({"s": "x"})]);
}

#[test]
fn draft_04_definition_with_id_follows_the_registry() {
    // The root has no `$schema`, so the registry reads the whole document as
    // draft 2020-12, where `id` is not an identifier. The definition `r` has
    // its own draft-04 `$schema` and `"id": "schema/r"`, and its references
    // are relative to the root. They resolve against the root, as the
    // registry stores them. Hyper-schemas in this style are common.
    let schema = json!({
        "type": "object",
        "properties": {"r": {"$ref": "#/definitions/r"}},
        "required": ["r"],
        "definitions": {
            "r": {
                "$schema": "http://json-schema.org/draft-04/schema",
                "id": "schema/r",
                "type": ["object"],
                "properties": {"d": {"$ref": "#/definitions/r/definitions/d"}},
                "required": ["d"],
                "definitions": {"d": {"type": ["integer"]}}
            }
        }
    });
    json_test_many(
        &schema,
        &[json!({"r": {"d": 5}})],
        &[json!({"r": {"d": "x"}})],
    );
}
