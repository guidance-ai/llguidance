use llguidance::{api::ParserLimits, GrammarBuilder, JsonCompileOptions, UnsatisfiableSchemaError};
use serde_json::{json, Value};

fn compile(schema: Value) -> anyhow::Result<()> {
    JsonCompileOptions::default()
        .json_to_llg_with_overrides(GrammarBuilder::new(None, ParserLimits::default()), schema)
        .map(|_| ())
}

#[test]
fn compiler_proven_empty_schemas_have_a_public_error_type() {
    for schema in [
        json!(false),
        json!({"type": "object", "properties": {"x": false}, "required": ["x"]}),
        json!({"type": "integer", "minimum": 2, "maximum": 1}),
    ] {
        let error = compile(schema.clone()).expect_err("schema must be unsatisfiable");
        assert!(
            error.downcast_ref::<UnsatisfiableSchemaError>().is_some(),
            "{schema}: {error:#}"
        );
        let contextual = error.context("compile a tool's argument schema");
        assert!(contextual.is::<UnsatisfiableSchemaError>());
    }
}

#[test]
fn invalid_or_unsupported_schemas_are_not_classified_as_empty_languages() {
    for schema in [
        json!({"type": 123}),
        json!({"$ref": "#/$defs/missing"}),
        json!({"oneOf": [{"type": "number"}, {"type": "integer"}]}),
    ] {
        let error = compile(schema.clone()).expect_err("schema must fail compilation");
        assert!(
            !error.is::<UnsatisfiableSchemaError>(),
            "{schema}: {error:#}"
        );
    }
}

#[test]
fn resource_limit_errors_are_not_classified_as_empty_languages() {
    let limits = ParserLimits {
        max_grammar_size: 0,
        ..ParserLimits::default()
    };
    let error = JsonCompileOptions::default()
        .json_to_llg_with_overrides(
            GrammarBuilder::new(None, limits),
            json!({
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
                "additionalProperties": false
            }),
        )
        .err()
        .expect("the object grammar exceeds the configured limit");
    assert!(!error.is::<UnsatisfiableSchemaError>(), "{error:#}");
    assert!(error.to_string().contains("grammar size"), "{error:#}");
}

#[test]
fn satisfiable_alternatives_and_optional_false_properties_still_compile() {
    for schema in [
        json!({"anyOf": [false, {"type": "string"}]}),
        json!({"type": "object", "properties": {"x": false}, "additionalProperties": false}),
    ] {
        compile(schema).expect("existing empty-branch pruning must remain unchanged");
    }
}
