use llg_test_utils::{json_test_many, lark_str_test_many_quiet};
use llguidance::{
    api::{GrammarInit, ParserLimits, TopLevelGrammar},
    JsonCompileOptions, JsonSchemaStyle,
};
use rstest::rstest;
use serde_json::{json, Value};

fn xml_schema(mut schema: Value) -> Value {
    schema["x-guidance"] = json!({"style": "qwen_xml"});
    schema
}

fn xml_test(schema: Value, passing: &[&str], failing: &[&str]) {
    let grammar = format!("start: %json {}", xml_schema(schema));
    lark_str_test_many_quiet(&grammar, passing, failing);
}

fn schema_error(schema: Value, expected: &str) {
    let error = GrammarInit::Serialized(TopLevelGrammar::from_json_schema(schema))
        .to_internal(None, ParserLimits::default())
        .expect_err("expected schema compilation to fail");
    assert!(
        error.to_string().contains(expected),
        "unexpected error: {error}"
    );
}

#[test]
fn rust_options_select_parameter_encoding() {
    let options: JsonCompileOptions = serde_json::from_value(json!({"style": "qwen_xml"})).unwrap();
    let mut schema = json!({
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": false
    });
    options.apply_to(&mut schema);
    lark_str_test_many_quiet(
        &format!("start: %json {schema}"),
        &[
            "<parameter=count>\n3\n</parameter>",
            "<parameter=count>3</parameter>",
        ],
        &[r#"{"count":3}"#, "<parameter=count>three</parameter>"],
    );
}

#[test]
fn lark_composes_parameter_encoding_with_wrappers() {
    let schema = xml_schema(json!({
        "type": "object",
        "properties": {"name": {"type": "string", "enum": ["report.docx"]}},
        "required": ["name"],
        "additionalProperties": false
    }));
    let grammar = format!("start: \"<call>\" parameters \"</call>\"\nparameters: %json {schema}");
    lark_str_test_many_quiet(
        &grammar,
        &["<call><parameter=name>\nreport.docx\n</parameter></call>"],
        &[
            "<call><parameter=name>report.xlsx</parameter></call>",
            "<call><function=read_files><parameter=name>report.docx</parameter></function></call>",
        ],
    );
}

#[test]
fn nested_values_remain_json() {
    xml_test(
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["Čaša 🧪"]},
                "data": {
                    "type": "object",
                    "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
                    "required": ["items"],
                    "additionalProperties": false
                }
            },
            "required": ["name", "data"],
            "additionalProperties": false
        }),
        &["<parameter=name>Čaša 🧪</parameter><parameter=data>{\"items\":[1,2]}</parameter>"],
        &[
            "<parameter=name>\"Čaša 🧪\"</parameter><parameter=data>{\"items\":[1,2]}</parameter>",
            "<parameter=name>Čaša 🧪</parameter><parameter=data>{\"items\":[1,\"two\"]}</parameter>",
            "<parameter=name>Čaša 🧪</parameter><parameter=data><parameter=items>[1,2]</parameter></parameter>",
        ],
    );
}

#[test]
fn optional_parameters_keep_schema_order_and_unique_declared_names() {
    xml_test(
        json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "string", "enum": ["blue"]},
                "c": {"type": "boolean"}
            },
            "required": ["b"],
            "additionalProperties": false
        }),
        &[
            "<parameter=b>blue</parameter>",
            "<parameter=a>1</parameter>\n<parameter=b>blue</parameter>",
            "<parameter=b>blue</parameter>\t<parameter=c>true</parameter>",
            "<parameter=a>1</parameter><parameter=b>blue</parameter><parameter=c>true</parameter>",
        ],
        &[
            "<parameter=b>blue</parameter><parameter=a>1</parameter>",
            "<parameter=b>blue</parameter><parameter=b>blue</parameter>",
            "FINAL_REJECT:<parameter=a>1</parameter>",
        ],
    );
}

#[rstest]
#[case::enumeration(json!({"type": "string", "enum": ["blue"]}))]
#[case::pattern_and_length(json!({"type": "string", "pattern": "^[a-z]+$", "minLength": 4, "maxLength": 4}))]
fn string_constraints_apply_after_one_edge_newline(#[case] value_schema: Value) {
    xml_test(
        json!({
            "type": "object",
            "properties": {"value": value_schema},
            "required": ["value"],
            "additionalProperties": false
        }),
        &[
            "<parameter=value>blue</parameter>",
            "<parameter=value>\nblue\n</parameter>",
            "<parameter=value>\nblue</parameter>",
            "<parameter=value>blue\n</parameter>",
        ],
        &[
            "<parameter=value>\n\nblue\n</parameter>",
            "<parameter=value>\nblue\n\n</parameter>",
            "<parameter=value> blue </parameter>",
        ],
    );
}

#[rstest]
#[case::bare("", "", true)]
#[case::leading_newline("\n", "", true)]
#[case::trailing_newline("", "\n", true)]
#[case::both_newlines("\n", "\n", true)]
#[case::leading_space(" ", "", false)]
#[case::trailing_space("", " ", false)]
#[case::leading_tab("\t", "", false)]
#[case::trailing_tab("", "\t", false)]
#[case::leading_cr("\r", "", false)]
#[case::trailing_cr("", "\r", false)]
#[case::crlf("\r\n", "\r\n", false)]
#[case::extra_leading_newline("\n\n", "\n", false)]
#[case::extra_trailing_newline("\n", "\n\n", false)]
#[case::extra_both_newlines("\n\n", "\n\n", false)]
#[case::tab_and_cr("\t", "\r", false)]
fn booleans_and_null_allow_only_one_edge_newline(
    #[values(("boolean", "true"), ("boolean", "false"), ("null", "null"))] scalar: (&str, &str),
    #[case] leading: &str,
    #[case] trailing: &str,
    #[case] accepted: bool,
) {
    let (kind, literal) = scalar;
    let schema = json!({
        "type": "object",
        "properties": {"value": {"type": kind}},
        "required": ["value"],
        "additionalProperties": false
    });
    let text = format!("<parameter=value>{leading}{literal}{trailing}</parameter>");
    if accepted {
        xml_test(schema, &[&text], &[]);
    } else {
        xml_test(schema, &[], &[&text]);
    }
}

#[rstest]
#[case::omitted(None, true)]
#[case::open(Some(json!(true)), true)]
#[case::closed(Some(json!(false)), false)]
fn encoding_preserves_additional_properties_policy(
    #[case] additional: Option<Value>,
    #[case] allows_extra: bool,
) {
    let mut schema = json!({
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"]
    });
    if let Some(additional) = additional {
        schema["additionalProperties"] = additional;
    }
    let extra = "<parameter=count>1</parameter><parameter=extra>raw text</parameter>";
    if allows_extra {
        xml_test(schema, &["<parameter=count>1</parameter>", extra], &[]);
    } else {
        xml_test(schema, &["<parameter=count>1</parameter>"], &[extra]);
    }
}

#[test]
fn extra_names_may_repeat_but_declared_names_may_not() {
    xml_test(
        json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": true
        }),
        &["<parameter=count>1</parameter><parameter=extra>first</parameter><parameter=extra>second</parameter>"],
        &["<parameter=count>1</parameter><parameter=extra>first</parameter><parameter=count>2</parameter>"],
    );
}

#[test]
fn property_count_bounds_apply_to_extras_with_required_declared_properties() {
    xml_test(
        json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": true,
            "minProperties": 2,
            "maxProperties": 2
        }),
        &["<parameter=count>1</parameter><parameter=extra>2</parameter>"],
        &[
            "FINAL_REJECT:<parameter=count>1</parameter>",
            "<parameter=count>1</parameter><parameter=extra>2</parameter><parameter=another>3</parameter>",
        ],
    );
}

#[test]
fn optional_property_count_bounds_are_unsupported() {
    schema_error(
        xml_schema(json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}, "c": {"type": "integer"}},
            "additionalProperties": false,
            "minProperties": 2,
            "maxProperties": 2
        })),
        "min/maxProperties",
    );
}

#[test]
fn schema_valued_top_level_additional_properties_are_unsupported() {
    schema_error(
        xml_schema(json!({"type": "object", "additionalProperties": {"type": "integer"}})),
        "additionalProperties",
    );
}

#[test]
fn parameter_encoding_requires_an_object_schema() {
    schema_error(xml_schema(json!({"type": "string"})), "object");
}

#[test]
fn root_reference_keeps_parameter_encoding() {
    xml_test(
        json!({
            "$defs": {
                "Args": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": false
                }
            },
            "$ref": "#/$defs/Args"
        }),
        &["<parameter=count>3</parameter>"],
        &[r#"{"count":3}"#, "<parameter=count>wrong</parameter>"],
    );
}

#[test]
fn shared_reference_keeps_raw_and_nested_json_encodings_separate() {
    xml_test(
        json!({
            "$defs": {"Name": {"type": "string", "enum": ["blue"]}},
            "type": "object",
            "properties": {
                "name": {"$ref": "#/$defs/Name"},
                "names": {"type": "array", "items": {"$ref": "#/$defs/Name"}}
            },
            "required": ["name", "names"],
            "additionalProperties": false
        }),
        &["<parameter=name>blue</parameter><parameter=names>[\"blue\"]</parameter>"],
        &[
            "<parameter=name>\"blue\"</parameter><parameter=names>[\"blue\"]</parameter>",
            "<parameter=name>blue</parameter><parameter=names>[blue]</parameter>",
        ],
    );
}

#[test]
fn encoding_style_does_not_leak_into_adjacent_json_grammar() {
    let schema = json!({
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": false
    });
    let xml = xml_schema(schema.clone());
    let grammar =
        format!("start: parameters \"|\" plain\nparameters: %json {xml}\nplain: %json {schema}");
    lark_str_test_many_quiet(
        &grammar,
        &["<parameter=count>3</parameter>|{\"count\":4}"],
        &[
            "<parameter=count>3</parameter>|<parameter=count>4</parameter>",
            "<parameter=count>3</parameter>|{\"count\":\"four\"}",
        ],
    );
}

#[rstest]
fn ordinary_json_encoding_is_preserved(#[values(false, true)] explicit_style: bool) {
    let mut schema = json!({
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"]
    });
    if explicit_style {
        schema["x-guidance"] = json!({"style": "json"});
    }
    json_test_many(
        &schema,
        &[json!({"count": 3}), json!({"count": 3, "extra": true})],
        &[json!({"count": "three"})],
    );
}

#[test]
fn unknown_encoding_style_is_rejected() {
    schema_error(
        json!({"type": "object", "x-guidance": {"style": "unknown"}}),
        "unknown variant",
    );
}

#[rstest]
fn default_encoding_preserves_serialized_options(#[values(false, true)] explicit_style: bool) {
    let options: JsonCompileOptions = if explicit_style {
        serde_json::from_value(json!({"style": "json"})).unwrap()
    } else {
        JsonCompileOptions::default()
    };
    let mut schema = json!({"type": "object"});
    options.apply_to(&mut schema);
    let expected = json!({
        "type": "object",
        "x-guidance": {
            "item_separator": ",",
            "key_separator": ":",
            "whitespace_flexible": true,
            "whitespace_pattern": null,
            "coerce_one_of": false,
            "lenient": false,
            "json_allowed_escapes": null,
            "json_allow_general_unicode_escapes": false
        }
    });
    assert_eq!(
        serde_json::to_string(&schema).unwrap(),
        serde_json::to_string(&expected).unwrap()
    );
}

#[test]
fn qwen_encoding_survives_options_serialization() {
    let options = JsonCompileOptions {
        style: JsonSchemaStyle::QwenXml,
        ..Default::default()
    };
    let mut schema = json!({"type": "object"});
    options.apply_to(&mut schema);
    assert_eq!(schema["x-guidance"]["style"], "qwen_xml");
    let restored: JsonCompileOptions =
        serde_json::from_value(schema["x-guidance"].clone()).unwrap();
    assert_eq!(restored.style, JsonSchemaStyle::QwenXml);
}
