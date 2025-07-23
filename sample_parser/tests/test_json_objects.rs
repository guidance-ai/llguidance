use rstest::*;
use serde_json::{json, Value};

mod common_lark_utils;
use common_lark_utils::{json_err_test, json_schema_check, json_test_many};

#[rstest]
#[case(&json!({"a":123}))]
#[case(&json!({"a":0}))]
fn single_property(#[case] obj: &Value) {
    let schema =
        &json!({"type":"object", "properties": {"a": {"type":"integer"}}, "required": ["a"]});
    json_schema_check(schema, obj, true);
}

#[rstest]
#[case(&json!({"a":"Hello"}))]
#[case(&json!({"b":0}))]
fn single_property_failures(#[case] obj: &Value) {
    let schema =
        &json!({"type":"object", "properties": {"a": {"type":"integer"}}, "required": ["a"]});
    json_schema_check(schema, obj, false);
}

#[rstest]
#[case(&json!({"a":123, "b": "Hello"}))]
#[case(&json!({"a":0, "b": "World"}))]
fn multiple_properties(#[case] obj: &Value) {
    let schema = &json!({"type":"object", "properties": {
        "a": {"type":"integer"},
        "b": {"type":"string"}
    }, "required": ["a", "b"]});
    json_schema_check(schema, obj, true);
}

#[rstest]
#[case(&json!({"a":123}))]
#[case(&json!({"b": "Hello"}))]
#[case(&json!({"c": 1}))]
fn multiple_properties_failures(#[case] obj: &Value) {
    let schema = &json!({"type":"object", "properties": {
        "a": {"type":"integer"},
        "b": {"type":"string"}
    }, "required": ["a", "b"]});
    json_schema_check(schema, obj, false);
}


#[rstest]
#[case(&json!({"name": "Test", "info": {"a": 123, "b": 456}}))]
fn nested(#[case] obj: &Value) {
    let schema = &json!({"type":"object", "properties": {
        "name": {"type": "string"},
        "info": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    }, "required": ["name", "info"]});
    json_schema_check(schema, obj, true);
}

fn test_json_object_directly_nested() {
    json_test_many(
        &json!({"type":"object", "properties": {
                "name" : {
                    "type": "string"
                },
                "info": {
                    "type": "object",
                    "properties" : {
                        "a" : {
                            "type" : "integer"
                        },
                        "b" : {
                            "type" : "integer"
                        }
                    },
                    "required": ["a", "b"]
                }
            },
            "required": ["name", "info"]
        }),
        &[json!({"name": "Test", "info": {"a": 123, "b": 456}})],
        &[
            json!({"name": "Test", "info": {"a": 123}}),
            json!({"name": "Test", "info": {"a": "123", "b":20}}),
            json!({"name": "Test", "info": {"a": 123, "b": "456"}}),
            json!({"name": "Test", "info": {"b": 456}}),
            json!({"name": "Test", "info": {"c": 1}}),
        ],
    );
}

#[test]
fn test_json_object_with_array() {
    json_test_many(
        &json!({"type":"object", "properties": {
                "name" : {"type": "string"},
                "values": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["name", "values"]
        }),
        &[json!({"name": "Test", "values": [1, 2, 3]})],
        &[
            json!({"name": "Test", "values": [1, 2, "Hello"]}),
            json!({"name": "Test", "values": [1.0, 2.0]}),
            json!({"name": "Test"}),
            json!({"values": [1, 2, 3]}),
        ],
    );
}

#[test]
fn test_json_object_unsatisfiable() {
    json_test_many(
        &json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": false},
            "additionalProperties": false,
        }),
        &[json!({"a": 42})],
        &[json!({"a": 42, "b": 43})],
    );
    json_err_test(
        &json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": false},
            "required": ["b"],
            "additionalProperties": false,
        }),
        "Unsatisfiable schema: required property 'b' is unsatisfiable",
    );
    json_err_test(
        &json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": false,
        }),
        "Unsatisfiable schema",
    );
}

#[test]
fn test_json_linked_list() {
    json_test_many(
        &json!(
        {
            "$defs": {
                "A": {
                    "properties": {
                        "my_str": {
                            "default": "me",
                            "title": "My Str",
                            "type": "string"
                        },
                        "next": {
                            "anyOf": [
                                {
                                    "$ref": "#/$defs/A"
                                },
                                {
                                    "type": "null"
                                }
                            ]
                        }
                    },
                    "required": ["my_str", "next"],
                    "type": "object"
                }
            },
            "type": "object",
            "properties": {
                "my_list": {
                    "anyOf": [
                        {
                            "$ref": "#/$defs/A"
                        },
                        {
                            "type": "null"
                        }
                    ]
                }
            },
            "required": ["my_list"]
        }),
        &[
            json!({"my_list": null}),
            json!({"my_list":{"my_str": "first", "next": null}}),
            json!({"my_list":{"my_str": "first", "next": {"my_str": "second", "next":null}}}),
            json!({"my_list":{"my_str": "first", "next": {"my_str": "second", "next":{"my_str": "third", "next":null}}}}),
        ],
        &[
            json!({"my_list": {"my_str": 1}}),
            json!({"my_list": {"my_str": "first", "next": "second"}}),
        ],
    );
}
