use crate::JsonCompileOptions;
use anyhow::Result;
use derivre::RegexAst;
use indexmap::{IndexMap, IndexSet};
use serde_json::Value;

use super::numeric::Decimal;
use super::schema_compiler::SchemaCompiler;
use super::shared_context::BuiltSchema;

// Keywords that are implemented in this module
pub(crate) const IMPLEMENTED: [&str; 27] = [
    // Core
    "anyOf",
    "oneOf",
    "allOf",
    "$ref",
    "const",
    "enum",
    "type",
    // Array
    "items",
    "additionalItems",
    "prefixItems",
    "minItems",
    "maxItems",
    // Object
    "properties",
    "additionalProperties",
    "patternProperties",
    "required",
    "minProperties",
    "maxProperties",
    // String
    "minLength",
    "maxLength",
    "pattern",
    "format",
    // Number
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
];

// Keywords that are used for metadata or annotations, not directly driving validation.
// Note that some keywords like $id and $schema affect the behavior of other keywords, but
// they can safely be ignored if other keywords aren't present
pub(crate) const META_AND_ANNOTATIONS: [&str; 15] = [
    "$anchor",
    "$defs",
    "definitions",
    "$schema",
    "$id",
    "id",
    "$comment",
    "title",
    "description",
    "default",
    "readOnly",
    "writeOnly",
    "examples",
    "contentMediaType",
    "contentEncoding",
];

#[derive(Debug, Clone)]
pub enum Schema {
    Any,
    Unsatisfiable(String),
    Null,
    Number(NumberSchema),
    String(StringSchema),
    Array(ArraySchema),
    Object(ObjectSchema),
    Boolean(Option<bool>),
    AnyOf(Vec<Schema>),
    OneOf(Vec<Schema>),
    Ref(String),
}

#[derive(Debug, Clone, Default)]
pub struct NumberSchema {
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub exclusive_minimum: Option<f64>,
    pub exclusive_maximum: Option<f64>,
    pub integer: bool,
    pub multiple_of: Option<Decimal>,
}

impl NumberSchema {
    pub fn get_minimum(&self) -> (Option<f64>, bool) {
        match (self.minimum, self.exclusive_minimum) {
            (Some(min), Some(xmin)) => {
                if xmin >= min {
                    (Some(xmin), true)
                } else {
                    (Some(min), false)
                }
            }
            (Some(min), None) => (Some(min), false),
            (None, Some(xmin)) => (Some(xmin), true),
            (None, None) => (None, false),
        }
    }

    pub fn get_maximum(&self) -> (Option<f64>, bool) {
        match (self.maximum, self.exclusive_maximum) {
            (Some(max), Some(xmax)) => {
                if xmax <= max {
                    (Some(xmax), true)
                } else {
                    (Some(max), false)
                }
            }
            (Some(max), None) => (Some(max), false),
            (None, Some(xmax)) => (Some(xmax), true),
            (None, None) => (None, false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StringSchema {
    pub min_length: usize,
    pub max_length: Option<usize>,
    pub regex: Option<RegexAst>,
}

#[derive(Debug, Clone)]
pub struct ArraySchema {
    pub min_items: usize,
    pub max_items: Option<usize>,
    pub prefix_items: Vec<Schema>,
    pub items: Option<Box<Schema>>,
}

#[derive(Debug, Clone)]
pub struct ObjectSchema {
    pub properties: IndexMap<String, Schema>,
    pub pattern_properties: IndexMap<String, Schema>,
    pub additional_properties: Option<Box<Schema>>,
    pub required: IndexSet<String>,
    pub min_properties: usize,
    pub max_properties: Option<usize>,
}

pub trait OptSchemaExt {
    fn schema(&self) -> Schema;
    fn schema_ref(&self) -> &Schema;
}

impl OptSchemaExt for Option<Box<Schema>> {
    fn schema(&self) -> Schema {
        match self {
            Some(schema) => schema.as_ref().clone(),
            None => Schema::Any,
        }
    }

    fn schema_ref(&self) -> &Schema {
        match self {
            Some(schema) => schema.as_ref(),
            None => &Schema::Any,
        }
    }
}

impl Schema {
    pub fn unsat(reason: &str) -> Schema {
        Schema::Unsatisfiable(reason.to_string())
    }

    pub fn false_schema() -> Schema {
        Self::unsat("schema is false")
    }

    pub fn any_box() -> Option<Box<Schema>> {
        Some(Box::new(Schema::Any))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, Schema::Unsatisfiable(_))
    }
}

#[derive(Clone)]
pub struct SchemaBuilderOptions {
    pub max_size: usize,
    pub max_stack_level: usize,
    pub lenient: bool,
}

impl Default for SchemaBuilderOptions {
    fn default() -> Self {
        SchemaBuilderOptions {
            max_size: 50_000,
            max_stack_level: 128, // consumes ~2.5k of stack per level
            lenient: false,
        }
    }
}

pub fn build_schema(contents: Value, options: &JsonCompileOptions) -> Result<BuiltSchema> {
    SchemaCompiler::compile(contents, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_problem_child() {
        let schema = json!({
            "allOf" : [
                {"$ref": "#/$defs/tree1"},
                {"$ref": "#/$defs/tree2"}
            ],
            "$defs" : {
                "tree1": {
                    "type": "object",
                    "properties": {
                        "child": {
                            "$ref": "#/$defs/tree1"
                        }
                    }
                },
                "tree2": {
                    "type": "object",
                    "properties": {
                        "child": {
                            "$ref": "#/$defs/tree2"
                        }
                    }
                }
            }
        });
        // Test failure amounts to this resulting in a stack overflow
        let options = JsonCompileOptions::default();
        let _ = build_schema(schema, &options);
    }

    #[test]
    fn test_id_ref_resolve() {
        let schema = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "https://example.com/browser.json",
            "definitions": {
                "browser": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            },
            "$ref": "#/definitions/browser"
        });
        let opts = JsonCompileOptions::default();
        let result = build_schema(schema, &opts);
        assert!(result.is_ok(), "Failed: {:?}", result.err());
    }
}
