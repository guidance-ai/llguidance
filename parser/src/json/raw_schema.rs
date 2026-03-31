//! Serde-based JSON Schema parsing types.
//!
//! `RawSchema` is the input type for the JSON Schema compiler. It maps directly
//! to the JSON Schema keywords we support, and can be deserialized from a JSON value.
//!
//! Keywords are classified at parse time into three categories:
//! - **Meta**: `$schema`, `$id`, `$defs`, `title`, etc. — ignored by the compiler
//! - **Constraint**: `type`, `properties`, `minLength`, etc. — batched and compiled together
//! - **Applicator**: `allOf`, `anyOf`, `$ref`, `const`, etc. — trigger constraint flushes
//!
//! Unknown keywords (not recognized by this parser) are tracked by name so the compiler
//! can check them against the draft's keyword list for unimplemented-keyword errors.

use indexmap::IndexMap;
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use std::fmt;

/// A JSON Schema, which can be a boolean or an object with keywords.
#[derive(Debug, Clone)]
pub enum RawSchema {
    /// `true` (accept everything) or `false` (reject everything)
    Bool(bool),
    /// A schema object with keywords
    Object(SchemaObject),
}

impl Default for RawSchema {
    fn default() -> Self {
        RawSchema::Bool(true)
    }
}

impl<'de> Deserialize<'de> for RawSchema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RawSchemaVisitor;

        impl<'de> Visitor<'de> for RawSchemaVisitor {
            type Value = RawSchema;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a boolean or a JSON Schema object")
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(RawSchema::Bool(v))
            }

            fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Self::Value, M::Error> {
                let obj = SchemaObject::deserialize(de::value::MapAccessDeserializer::new(map))?;
                Ok(RawSchema::Object(obj))
            }
        }

        deserializer.deserialize_any(RawSchemaVisitor)
    }
}

// ---- Applicator enum ----

/// Applicator keywords that compose schemas by combining sub-schemas.
/// These trigger a constraint flush in the compiler's keyword-ordering loop.
#[derive(Debug, Clone)]
pub enum Applicator {
    Const(Value),
    Enum(Vec<Value>),
    AllOf(Vec<RawSchema>),
    AnyOf(Vec<RawSchema>),
    OneOf(Vec<RawSchema>),
    Ref(String),
}

// ---- Constraint enum ----

/// Constraint keywords that narrow what values are valid.
/// Between applicators, these are batched and compiled together.
#[derive(Debug, Clone)]
pub enum Constraint {
    // Type
    Type(TypeValue),

    // String
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
    Format(String),

    // Number
    Minimum(f64),
    Maximum(f64),
    ExclusiveMinimum(BoolOrNumber),
    ExclusiveMaximum(BoolOrNumber),
    MultipleOf(f64),

    // Array
    Items(ItemsValue),
    AdditionalItems(Box<RawSchema>),
    PrefixItems(Vec<RawSchema>),
    MinItems(usize),
    MaxItems(usize),

    // Object
    Properties(Box<IndexMap<String, RawSchema>>),
    AdditionalProperties(Box<RawSchema>),
    PatternProperties(Box<IndexMap<String, RawSchema>>),
    Required(Vec<String>),
    MinProperties(usize),
    MaxProperties(usize),
}

// ---- Keyword ordering ----

/// Entry in the keyword ordering sequence, preserving JSON iteration order.
///
/// The compiler walks this sequence to determine when to flush pending
/// constraints (on encountering an applicator). Meta keywords are not
/// included — they are either pre-extracted (`$id`) or ignored entirely.
#[derive(Debug, Clone)]
pub enum KeywordEntry {
    /// An applicator keyword with its parsed data.
    Applicator(Applicator),
    /// A constraint keyword with its parsed data.
    Constraint(Constraint),
    /// An unrecognized keyword (name preserved for draft-aware unimplemented checking).
    Unknown(String),
}

// ---- Helper types ----

/// `exclusiveMinimum` / `exclusiveMaximum` can be a boolean (Draft 4) or a number (Draft 6+).
#[derive(Debug, Clone)]
pub enum BoolOrNumber {
    Bool(bool),
    Number(f64),
}

impl<'de> Deserialize<'de> for BoolOrNumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct BoolOrNumberVisitor;

        impl<'de> Visitor<'de> for BoolOrNumberVisitor {
            type Value = BoolOrNumber;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a boolean or number")
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(BoolOrNumber::Bool(v))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(BoolOrNumber::Number(v as f64))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(BoolOrNumber::Number(v as f64))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
                Ok(BoolOrNumber::Number(v))
            }
        }

        deserializer.deserialize_any(BoolOrNumberVisitor)
    }
}

/// `items` can be a single schema (Draft 2020-12) or an array of schemas (Draft 4 — treated as
/// `prefixItems`).
#[derive(Debug, Clone)]
pub enum ItemsValue {
    /// A single schema applied to all items beyond prefixItems
    Schema(Box<RawSchema>),
    /// An array of schemas (Draft 4 `items` as array = prefixItems)
    Array(Vec<RawSchema>),
}

impl<'de> Deserialize<'de> for ItemsValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ItemsVisitor;

        impl<'de> Visitor<'de> for ItemsVisitor {
            type Value = ItemsValue;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a schema or array of schemas")
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(ItemsValue::Schema(Box::new(RawSchema::Bool(v))))
            }

            fn visit_map<M: MapAccess<'de>>(self, map: M) -> Result<Self::Value, M::Error> {
                let raw = RawSchema::deserialize(de::value::MapAccessDeserializer::new(map))?;
                Ok(ItemsValue::Schema(Box::new(raw)))
            }

            fn visit_seq<S: de::SeqAccess<'de>>(self, mut seq: S) -> Result<Self::Value, S::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(ItemsValue::Array(items))
            }
        }

        deserializer.deserialize_any(ItemsVisitor)
    }
}

/// `type` can be a single string or an array of strings.
#[derive(Debug, Clone)]
pub enum TypeValue {
    Single(String),
    Array(Vec<String>),
}

impl<'de> Deserialize<'de> for TypeValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TypeVisitor;

        impl<'de> Visitor<'de> for TypeVisitor {
            type Value = TypeValue;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a string or array of strings")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(TypeValue::Single(v.to_string()))
            }

            fn visit_seq<S: de::SeqAccess<'de>>(self, mut seq: S) -> Result<Self::Value, S::Error> {
                let mut types = Vec::new();
                while let Some(item) = seq.next_element::<String>()? {
                    types.push(item);
                }
                Ok(TypeValue::Array(types))
            }
        }

        deserializer.deserialize_any(TypeVisitor)
    }
}

// ---- SchemaObject ----

/// The main schema object containing all supported JSON Schema keywords.
///
/// All keyword data lives in `keyword_order` as `KeywordEntry::Constraint(...)` or
/// `KeywordEntry::Applicator(...)`. Meta keywords (title, description, $schema, etc.)
/// are consumed during parsing but not stored — they don't affect compilation.
#[derive(Debug, Clone, Default)]
pub struct SchemaObject {
    /// Pre-extracted `$id` (Draft 6+).
    pub dollar_id: Option<String>,
    /// Pre-extracted `id` (Draft 4 only).
    pub legacy_id: Option<String>,

    /// Keyword iteration order with all constraint and applicator data.
    /// Meta keywords are excluded — they don't participate in compilation.
    pub keyword_order: Vec<KeywordEntry>,
}

impl<'de> Deserialize<'de> for SchemaObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SchemaObjectVisitor;

        impl<'de> Visitor<'de> for SchemaObjectVisitor {
            type Value = SchemaObject;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a JSON Schema object")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut keyword_order = Vec::new();
                // Track both $id and id separately to prefer $id
                let mut id_new: Option<String> = None;
                let mut id_old: Option<String> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        // --- Constraints ---
                        "type" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::Type(
                                map.next_value()?,
                            )));
                        }
                        "properties" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::Properties(
                                Box::new(map.next_value()?),
                            )));
                        }
                        "additionalProperties" => {
                            keyword_order.push(KeywordEntry::Constraint(
                                Constraint::AdditionalProperties(map.next_value()?),
                            ));
                        }
                        "patternProperties" => {
                            keyword_order.push(KeywordEntry::Constraint(
                                Constraint::PatternProperties(Box::new(map.next_value()?)),
                            ));
                        }
                        "required" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::Required(
                                map.next_value()?,
                            )));
                        }
                        "minProperties" => {
                            keyword_order.push(KeywordEntry::Constraint(
                                Constraint::MinProperties(map.next_value()?),
                            ));
                        }
                        "maxProperties" => {
                            keyword_order.push(KeywordEntry::Constraint(
                                Constraint::MaxProperties(map.next_value()?),
                            ));
                        }
                        "items" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::Items(
                                map.next_value()?,
                            )));
                        }
                        "additionalItems" => {
                            keyword_order.push(KeywordEntry::Constraint(
                                Constraint::AdditionalItems(map.next_value()?),
                            ));
                        }
                        "prefixItems" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::PrefixItems(
                                map.next_value()?,
                            )));
                        }
                        "minItems" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::MinItems(
                                map.next_value()?,
                            )));
                        }
                        "maxItems" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::MaxItems(
                                map.next_value()?,
                            )));
                        }
                        "minLength" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::MinLength(
                                map.next_value()?,
                            )));
                        }
                        "maxLength" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::MaxLength(
                                map.next_value()?,
                            )));
                        }
                        "pattern" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::Pattern(
                                map.next_value()?,
                            )));
                        }
                        "format" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::Format(
                                map.next_value()?,
                            )));
                        }
                        "minimum" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::Minimum(
                                map.next_value()?,
                            )));
                        }
                        "maximum" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::Maximum(
                                map.next_value()?,
                            )));
                        }
                        "exclusiveMinimum" => {
                            keyword_order.push(KeywordEntry::Constraint(
                                Constraint::ExclusiveMinimum(map.next_value()?),
                            ));
                        }
                        "exclusiveMaximum" => {
                            keyword_order.push(KeywordEntry::Constraint(
                                Constraint::ExclusiveMaximum(map.next_value()?),
                            ));
                        }
                        "multipleOf" => {
                            keyword_order.push(KeywordEntry::Constraint(Constraint::MultipleOf(
                                map.next_value()?,
                            )));
                        }

                        // --- Applicators (data stored in keyword_order directly) ---
                        "const" => {
                            let v: Value = map.next_value()?;
                            keyword_order.push(KeywordEntry::Applicator(Applicator::Const(v)));
                        }
                        "enum" => {
                            let v: Vec<Value> = map.next_value()?;
                            keyword_order.push(KeywordEntry::Applicator(Applicator::Enum(v)));
                        }
                        "allOf" => {
                            let v: Vec<RawSchema> = map.next_value()?;
                            keyword_order.push(KeywordEntry::Applicator(Applicator::AllOf(v)));
                        }
                        "anyOf" => {
                            let v: Vec<RawSchema> = map.next_value()?;
                            keyword_order.push(KeywordEntry::Applicator(Applicator::AnyOf(v)));
                        }
                        "oneOf" => {
                            let v: Vec<RawSchema> = map.next_value()?;
                            keyword_order.push(KeywordEntry::Applicator(Applicator::OneOf(v)));
                        }
                        "$ref" => {
                            let v: String = map.next_value()?;
                            keyword_order.push(KeywordEntry::Applicator(Applicator::Ref(v)));
                        }

                        // --- Meta keywords (consumed but not stored) ---
                        "$id" => {
                            id_new = Some(map.next_value()?);
                        }
                        "id" => {
                            // In Draft 4, `id` is the identifier keyword (string).
                            // Some schemas misuse `id` as a regular property with a
                            // non-string value — tolerate that by discarding non-strings.
                            let v: Value = map.next_value()?;
                            if let Value::String(s) = v {
                                id_old = Some(s);
                            }
                        }
                        "$schema" | "$anchor" | "$comment" | "$defs" | "definitions" | "title"
                        | "description" | "default" | "readOnly" | "writeOnly" | "examples"
                        | "contentMediaType" | "contentEncoding" => {
                            let _: Value = map.next_value()?;
                        }

                        // --- Unknown keywords ---
                        _ => {
                            let _: Value = map.next_value()?;
                            keyword_order.push(KeywordEntry::Unknown(key));
                        }
                    }
                }

                Ok(SchemaObject {
                    dollar_id: id_new,
                    legacy_id: id_old,
                    keyword_order,
                })
            }
        }

        deserializer.deserialize_map(SchemaObjectVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_bool_schema() {
        let schema: RawSchema = serde_json::from_value(json!(true)).unwrap();
        assert!(matches!(schema, RawSchema::Bool(true)));
        let schema: RawSchema = serde_json::from_value(json!(false)).unwrap();
        assert!(matches!(schema, RawSchema::Bool(false)));
    }

    #[test]
    fn test_simple_type() {
        let schema: RawSchema = serde_json::from_value(json!({"type": "string"})).unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let has_type = obj.keyword_order.iter().any(|e| {
                    matches!(e, KeywordEntry::Constraint(Constraint::Type(TypeValue::Single(s))) if s == "string")
                });
                assert!(has_type, "Expected type constraint");
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_type_array() {
        let schema: RawSchema =
            serde_json::from_value(json!({"type": ["string", "number"]})).unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let has_type = obj.keyword_order.iter().any(|e| {
                    matches!(e, KeywordEntry::Constraint(Constraint::Type(TypeValue::Array(types))) if types == &vec!["string", "number"])
                });
                assert!(has_type, "Expected array type");
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_object_with_properties() {
        let schema: RawSchema = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"],
            "additionalProperties": false
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                // Check properties
                let props = obj.keyword_order.iter().find_map(|e| match e {
                    KeywordEntry::Constraint(Constraint::Properties(p)) => Some(p.as_ref()),
                    _ => None,
                });
                let props = props.expect("Expected properties constraint");
                assert_eq!(props.len(), 2);
                let keys: Vec<&String> = props.keys().collect();
                assert_eq!(keys, vec!["name", "age"]);

                // Check required
                let required = obj.keyword_order.iter().find_map(|e| match e {
                    KeywordEntry::Constraint(Constraint::Required(r)) => Some(r),
                    _ => None,
                });
                assert_eq!(required, Some(&vec!["name".to_string()]));

                // Check additionalProperties
                let has_ap = obj.keyword_order.iter().any(|e| {
                    matches!(e, KeywordEntry::Constraint(Constraint::AdditionalProperties(s)) if matches!(s.as_ref(), RawSchema::Bool(false)))
                });
                assert!(has_ap);
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_exclusive_minimum_bool() {
        let schema: RawSchema = serde_json::from_value(json!({
            "type": "number",
            "minimum": 0,
            "exclusiveMinimum": true
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let has_excl = obj.keyword_order.iter().any(|e| {
                    matches!(
                        e,
                        KeywordEntry::Constraint(Constraint::ExclusiveMinimum(BoolOrNumber::Bool(
                            true
                        )))
                    )
                });
                assert!(has_excl);
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_exclusive_minimum_number() {
        let schema: RawSchema = serde_json::from_value(json!({
            "type": "number",
            "exclusiveMinimum": 5.0
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let has_excl = obj.keyword_order.iter().any(|e| {
                    matches!(e, KeywordEntry::Constraint(Constraint::ExclusiveMinimum(BoolOrNumber::Number(n))) if *n == 5.0)
                });
                assert!(has_excl);
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_items_single_schema() {
        let schema: RawSchema = serde_json::from_value(json!({
            "type": "array",
            "items": {"type": "string"}
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let has_items = obj.keyword_order.iter().any(|e| {
                    matches!(
                        e,
                        KeywordEntry::Constraint(Constraint::Items(ItemsValue::Schema(_)))
                    )
                });
                assert!(has_items);
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_items_array_draft4() {
        let schema: RawSchema = serde_json::from_value(json!({
            "type": "array",
            "items": [{"type": "string"}, {"type": "number"}],
            "additionalItems": {"type": "boolean"}
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let has_items_array = obj.keyword_order.iter().any(|e| {
                    matches!(e, KeywordEntry::Constraint(Constraint::Items(ItemsValue::Array(items))) if items.len() == 2)
                });
                assert!(has_items_array);
                let has_additional = obj
                    .keyword_order
                    .iter()
                    .any(|e| matches!(e, KeywordEntry::Constraint(Constraint::AdditionalItems(_))));
                assert!(has_additional);
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_all_of() {
        let schema: RawSchema = serde_json::from_value(json!({
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "object", "properties": {"b": {"type": "number"}}}
            ]
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                // allOf is stored in keyword_order as an Applicator
                let has_all_of = obj.keyword_order.iter().any(
                    |e| matches!(e, KeywordEntry::Applicator(Applicator::AllOf(v)) if v.len() == 2),
                );
                assert!(has_all_of, "Expected allOf with 2 schemas");
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_ref() {
        let schema: RawSchema = serde_json::from_value(json!({
            "$ref": "#/$defs/Foo"
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let has_ref = obj.keyword_order.iter().any(|e| {
                    matches!(e, KeywordEntry::Applicator(Applicator::Ref(s)) if s == "#/$defs/Foo")
                });
                assert!(has_ref, "Expected $ref applicator");
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_unknown_keys_captured() {
        let schema: RawSchema = serde_json::from_value(json!({
            "type": "string",
            "if": {"minLength": 5},
            "then": {"pattern": "^a"},
            "uniqueItems": true
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let unknowns: Vec<&str> = obj
                    .keyword_order
                    .iter()
                    .filter_map(|e| match e {
                        KeywordEntry::Unknown(k) => Some(k.as_str()),
                        _ => None,
                    })
                    .collect();
                assert!(unknowns.contains(&"if"));
                assert!(unknowns.contains(&"then"));
                assert!(unknowns.contains(&"uniqueItems"));
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_defs() {
        let schema: RawSchema = serde_json::from_value(json!({
            "$defs": {
                "Foo": {"type": "string"},
                "Bar": {"type": "number"}
            },
            "$ref": "#/$defs/Foo"
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                // $defs is meta — consumed but not stored
                // $ref is stored as an applicator
                let has_ref = obj
                    .keyword_order
                    .iter()
                    .any(|e| matches!(e, KeywordEntry::Applicator(Applicator::Ref(_))));
                assert!(has_ref, "Expected $ref applicator");
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_nested_schemas() {
        let schema: RawSchema = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "deep": {"type": "string", "minLength": 1}
                    }
                }
            }
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let props = obj
                    .keyword_order
                    .iter()
                    .find_map(|e| match e {
                        KeywordEntry::Constraint(Constraint::Properties(p)) => Some(p.as_ref()),
                        _ => None,
                    })
                    .unwrap();
                let inner = props.get("inner").unwrap();
                match inner {
                    RawSchema::Object(inner_obj) => {
                        let inner_props = inner_obj
                            .keyword_order
                            .iter()
                            .find_map(|e| match e {
                                KeywordEntry::Constraint(Constraint::Properties(p)) => {
                                    Some(p.as_ref())
                                }
                                _ => None,
                            })
                            .unwrap();
                        let deep = inner_props.get("deep").unwrap();
                        match deep {
                            RawSchema::Object(deep_obj) => {
                                let has_min = deep_obj.keyword_order.iter().any(|e| {
                                    matches!(e, KeywordEntry::Constraint(Constraint::MinLength(1)))
                                });
                                assert!(has_min);
                            }
                            _ => panic!("Expected object schema for deep"),
                        }
                    }
                    _ => panic!("Expected object schema for inner"),
                }
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_pattern_properties() {
        let schema: RawSchema = serde_json::from_value(json!({
            "type": "object",
            "patternProperties": {
                "^S_": {"type": "string"},
                "^I_": {"type": "integer"}
            }
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let pp = obj.keyword_order.iter().find_map(|e| match e {
                    KeywordEntry::Constraint(Constraint::PatternProperties(p)) => Some(p.as_ref()),
                    _ => None,
                });
                let pp = pp.expect("Expected patternProperties");
                assert_eq!(pp.len(), 2);
                assert!(pp.contains_key("^S_"));
                assert!(pp.contains_key("^I_"));
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_fhir_like_schema_parses() {
        let schema: RawSchema = serde_json::from_value(json!({
            "$schema": "http://json-schema.org/draft-04/schema#",
            "id": "http://example.com/fhir.schema.json",
            "type": "object",
            "properties": {
                "resourceType": {"type": "string", "const": "Patient"},
                "name": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/HumanName"}
                }
            },
            "required": ["resourceType"],
            "definitions": {
                "HumanName": {
                    "type": "object",
                    "properties": {
                        "family": {"type": "string"},
                        "given": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "dependencies": {"name": ["resourceType"]}
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                // id is pre-extracted
                assert_eq!(
                    obj.legacy_id.as_deref(),
                    Some("http://example.com/fhir.schema.json")
                );
                // "dependencies" should be in unknown keys
                let unknowns: Vec<&str> = obj
                    .keyword_order
                    .iter()
                    .filter_map(|e| match e {
                        KeywordEntry::Unknown(k) => Some(k.as_str()),
                        _ => None,
                    })
                    .collect();
                assert!(unknowns.contains(&"dependencies"));
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_keyword_order_preserved() {
        let schema: RawSchema = serde_json::from_value(json!({
            "properties": {"a": true},
            "allOf": [{"properties": {"b": true}}],
            "additionalProperties": false,
            "anyOf": [{"type": "object"}]
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                assert_eq!(obj.keyword_order.len(), 4);
                assert!(matches!(
                    obj.keyword_order[0],
                    KeywordEntry::Constraint(Constraint::Properties(_))
                ));
                assert!(matches!(
                    obj.keyword_order[1],
                    KeywordEntry::Applicator(Applicator::AllOf(_))
                ));
                assert!(matches!(
                    obj.keyword_order[2],
                    KeywordEntry::Constraint(Constraint::AdditionalProperties(_))
                ));
                assert!(matches!(
                    obj.keyword_order[3],
                    KeywordEntry::Applicator(Applicator::AnyOf(_))
                ));
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_const_null() {
        // Ensure const: null is correctly parsed (not treated as absent)
        let schema: RawSchema = serde_json::from_value(json!({"const": null})).unwrap();
        match schema {
            RawSchema::Object(obj) => {
                let has_const = obj
                    .keyword_order
                    .iter()
                    .any(|e| matches!(e, KeywordEntry::Applicator(Applicator::Const(Value::Null))));
                assert!(has_const, "Expected const: null applicator");
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_id_stored_separately() {
        let schema: RawSchema = serde_json::from_value(json!({
            "id": "http://old.com",
            "$id": "http://new.com"
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                assert_eq!(obj.dollar_id.as_deref(), Some("http://new.com"));
                assert_eq!(obj.legacy_id.as_deref(), Some("http://old.com"));
            }
            _ => panic!("Expected object schema"),
        }
    }

    #[test]
    fn test_meta_keywords_not_in_keyword_order() {
        let schema: RawSchema = serde_json::from_value(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Test",
            "description": "A test schema",
            "$defs": {"Foo": {"type": "string"}},
            "type": "string"
        }))
        .unwrap();
        match schema {
            RawSchema::Object(obj) => {
                // Only "type" should appear (as a Constraint)
                assert_eq!(obj.keyword_order.len(), 1);
                assert!(matches!(obj.keyword_order[0], KeywordEntry::Constraint(_)));
            }
            _ => panic!("Expected object schema"),
        }
    }
}
