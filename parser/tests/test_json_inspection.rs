use llguidance::{api::ParserLimits, JsonCompileOptions};
use serde_json::json;

#[test]
fn inspection_retains_normalization_and_lowering() {
    let result = JsonCompileOptions::default()
        .inspect_json_schema(
            json!({"allOf": [{"type": "number", "minimum": 3}, {"maximum": 8}]}),
            ParserLimits::default(),
        )
        .unwrap();
    let root = result["normalized"]["root"].as_str().unwrap();
    assert!(root.contains("Number"));
    assert!(root.contains("3.0"));
    assert!(root.contains("8.0"));
    assert!(!result["grammar"].as_str().unwrap().is_empty());
    assert!(!result["lexer"].as_str().unwrap().is_empty());
}

#[test]
fn embedded_options_replace_rust_defaults() {
    let options = JsonCompileOptions {
        whitespace_flexible: false,
        ..Default::default()
    };
    let result = options
        .inspect_json_schema(
            json!({"type": "string", "x-guidance": {"lenient": true}}),
            ParserLimits::default(),
        )
        .unwrap();
    assert_eq!(result["effective_options"]["whitespace_flexible"], true);
    assert_eq!(result["effective_options"]["lenient"], true);
}

#[test]
fn unsatisfiable_schema_does_not_return_success() {
    assert!(JsonCompileOptions::default()
        .inspect_json_schema(json!(false), ParserLimits::default())
        .is_err());
}
