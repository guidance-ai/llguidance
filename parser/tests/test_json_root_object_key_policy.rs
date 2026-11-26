use llg_test_utils::json_schema_check;
use serde_json::{json, Value};

fn named(mut schema: Value, min: usize, max: usize) -> Value {
    schema["x-guidance"] = json!({"root_object_key_policy":{"min":min,"max":max}});
    schema
}

#[test]
fn named_root_keys_preserve_nested_dynamic_names_and_property_values() {
    let schema = named(
        json!({
            "type":"object",
            "properties":{"outer":{"type":"object","additionalProperties":{"type":"integer"}}},
            "required":["outer"],
            "additionalProperties":{"type":"integer"}
        }),
        1,
        8,
    );
    json_schema_check(&schema, &json!({"outer":{"":7,"dynamic":8}}), true);
    json_schema_check(&schema, &json!({"outer":{"bad":"value"}}), false);
    json_schema_check(&schema, &json!({"outer":{},"unknown":1}), false);
}

#[test]
fn named_root_values_use_the_original_pattern_and_additional_intersections() {
    let schema = named(
        json!({
            "type":"object", "properties":{"a":{"type":"number"}},
            "patternProperties":{"^a$":{"type":"integer"}},
            "required":["a"], "additionalProperties":false
        }),
        1,
        8,
    );
    json_schema_check(&schema, &json!({"a":2}), true);
    json_schema_check(&schema, &json!({"a":2.5}), false);
    json_schema_check(&schema, &json!({"a":2,"ax":3}), false);

    let required_only = named(
        json!({
            "type":"object", "required":["declared"],
            "additionalProperties":{"type":"integer"}
        }),
        1,
        8,
    );
    json_schema_check(&required_only, &json!({"declared":3}), true);
    json_schema_check(&required_only, &json!({"declared":"bad"}), false);
    json_schema_check(&required_only, &json!({"declared":3,"extra":4}), false);
}

#[test]
fn named_root_reference_cache_does_not_restrict_the_same_definition_when_nested() {
    let schema = named(
        json!({
            "$id":"https://example.com/named-root",
            "$defs":{"Args":{
                "type":"object",
                "properties":{
                    "valid":{"type":"integer"},
                    "lengthy":{"type":"integer"},
                    "next":{"$ref":"#/$defs/Args"}
                },
                "required":["valid"], "additionalProperties":false
            }},
            "$ref":"#/$defs/Args"
        }),
        1,
        5,
    );
    json_schema_check(&schema, &json!({"valid":1}), true);
    json_schema_check(
        &schema,
        &json!({"next":{"lengthy":3,"valid":2},"valid":1}),
        true,
    );
    json_schema_check(&schema, &json!({"lengthy":3,"valid":1}), false);
}

#[test]
fn named_root_cardinality_preserves_exact_optional_and_required_counts() {
    let schema = named(
        json!({
            "type":"object", "properties":{"a":{"type":"integer"},"b":{"type":"integer"}},
            "additionalProperties":false, "minProperties":1, "maxProperties":1
        }),
        1,
        8,
    );
    json_schema_check(&schema, &json!({"a":1}), true);
    json_schema_check(&schema, &json!({"b":2}), true);
    json_schema_check(&schema, &json!({}), false);
    json_schema_check(&schema, &json!({"a":1,"b":2}), false);

    let properties = (0..12)
        .map(|i| (format!("k{i:02}"), json!({"type":"integer"})))
        .collect::<serde_json::Map<_, _>>();
    let required = properties.keys().cloned().collect::<Vec<_>>();
    let schema = named(
        json!({
            "type":"object", "properties":properties, "required":required,
            "minProperties":12, "maxProperties":12, "additionalProperties":false
        }),
        1,
        3,
    );
    let mut value = (0..12)
        .map(|i| (format!("k{i:02}"), json!(i)))
        .collect::<serde_json::Map<_, _>>();
    json_schema_check(&schema, &Value::Object(value.clone()), true);
    value.remove("k11");
    json_schema_check(&schema, &Value::Object(value), false);
}

#[test]
fn named_root_policy_alternatives_and_utf8_byte_bounds_are_enforced() {
    let schema = named(
        json!({"anyOf":[{"const":{"":1}},{"const":{"valid":2}}]}),
        1,
        5,
    );
    json_schema_check(&schema, &json!({"valid":2}), true);
    json_schema_check(&schema, &json!({"":1}), false);

    let schema = named(
        json!({
            "type":"object", "properties":{"éé":{"type":"integer"},"ok":{"type":"integer"}},
            "additionalProperties":false
        }),
        1,
        3,
    );
    json_schema_check(&schema, &json!({"ok":1}), true);
    json_schema_check(&schema, &json!({"éé":1}), false);
    let empty = named(json!({"const":{"":1}}), 0, 0);
    json_schema_check(&empty, &json!({"":1}), true);
}

#[test]
fn root_array_items_keep_the_unrestricted_schema_language() {
    let schema = named(
        json!({
            "type":"array", "items":{"type":"object","additionalProperties":{"type":"integer"}}
        }),
        1,
        1,
    );
    json_schema_check(&schema, &json!([{"":1,"long dynamic name":2}]), true);
    json_schema_check(&schema, &json!([{"bad":"value"}]), false);
}
