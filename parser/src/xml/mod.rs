//! XML grammar compilation, inspired by XGrammar: <https://github.com/mlc-ai/xgrammar>.
//!
//! This module holds the format-independent part: checking parameter value
//! schemas and turning their string constraints into Lark regexes. Each style
//! lives in its own submodule.

mod qwen3_coder;

use anyhow::{anyhow, bail, ensure, Context, Result};
use indexmap::IndexSet;

use crate::{
    json::{
        formats::lookup_format,
        numeric::{check_number_bounds, rx_float_range, rx_int_range},
        schema::Schema,
    },
    regex_to_lark, HashMap,
};
use derivre::{RegexAst, RegexBuilder};
use regex_syntax::hir::{Hir, HirKind, Look};
use serde_json::Value;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct XmlOptions {
    // Style can become optional once there is a suitable default.
    style: String,
}

pub(crate) fn schema_to_lark(mut schema: Value) -> Result<String> {
    let options = schema
        .as_object_mut()
        .ok_or_else(|| anyhow!("XML schema must be an object"))?
        .remove("x-guidance")
        .unwrap_or_else(|| serde_json::json!({}));
    let options: XmlOptions = serde_json::from_value(options).context("Invalid %xml options")?;
    match options.style.as_str() {
        "qwen3_coder" => qwen3_coder::schema_to_lark(&schema),
        style => bail!("Unsupported XML style: {style}"),
    }
}

const ANY: &str = "(?s:.*)";
const JSON_TYPES: [&str; 5] = ["number", "boolean", "null", "object", "array"];
const ALL_JSON: u8 = 31;
const MAX_DEPTH: usize = 128;

fn rx(pattern: &str) -> RegexAst {
    RegexAst::Regex(pattern.to_string())
}

fn quoted(text: &str) -> String {
    serde_json::to_string(text).unwrap()
}

fn local_ref<'a>(root: &'a Value, reference: &Value) -> Result<&'a Value> {
    let reference = reference
        .as_str()
        .ok_or_else(|| anyhow!("$ref must be a string"))?;
    let pointer = reference
        .strip_prefix('#')
        .ok_or_else(|| anyhow!("Remote $ref is unsupported: {reference}"))?;
    ensure!(
        pointer.is_empty() || pointer.starts_with('/'),
        "Only JSON pointer $ref is supported: {reference}"
    );
    root.pointer(pointer)
        .ok_or_else(|| anyhow!("Cannot resolve $ref: {reference}"))
}

fn bound(schema: &Value, key: &str) -> Result<Option<u32>> {
    schema
        .get(key)
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .or_else(|| {
                    v.as_f64()
                        .filter(|n| *n >= 0.0 && *n <= u32::MAX as f64 && n.fract() == 0.0)
                        .map(|n| n as u32)
                })
                .ok_or_else(|| anyhow!("{key} must be a nonnegative 32-bit integer"))
        })
        .transpose()
}

fn validate_keywords(schema: &Value, allow_json_keywords: bool) -> Result<()> {
    let Some(object) = schema.as_object() else {
        ensure!(schema.is_boolean(), "Schema must be an object or boolean");
        return Ok(());
    };
    for key in object.keys() {
        ensure!(
            matches!(
                key.as_str(),
                "type"
                    | "enum"
                    | "const"
                    | "allOf"
                    | "anyOf"
                    | "oneOf"
                    | "$ref"
                    | "$defs"
                    | "definitions"
                    | "$schema"
                    | "$comment"
                    | "title"
                    | "description"
                    | "default"
                    | "examples"
                    | "readOnly"
                    | "writeOnly"
                    | "deprecated"
                    | "contentMediaType"
                    | "contentEncoding"
                    | "properties"
                    | "required"
                    | "additionalProperties"
                    | "minProperties"
                    | "maxProperties"
                    | "items"
                    | "prefixItems"
                    | "additionalItems"
                    | "minItems"
                    | "maxItems"
                    | "minLength"
                    | "maxLength"
                    | "pattern"
                    | "format"
                    | "minimum"
                    | "maximum"
                    | "exclusiveMinimum"
                    | "exclusiveMaximum"
                    | "multipleOf"
            ) || (allow_json_keywords && matches!(key.as_str(), "patternProperties" | "$id")),
            "Unsupported schema keyword: {key}"
        );
    }
    Ok(())
}

fn validate_schema(
    root: &Value,
    schema: &Value,
    seen: &mut IndexSet<(*const Value, String)>,
    depth: usize,
) -> Result<()> {
    ensure!(depth <= MAX_DEPTH, "Schema nesting limit exceeded");
    validate_keywords(schema, true)?;
    let root = if schema.get("$id").is_some() {
        schema
    } else {
        root
    };
    if let Some(reference) = schema.get("$ref") {
        let target = local_ref(root, reference)?;
        if seen.insert((root as *const Value, reference.to_string())) {
            validate_schema(root, target, seen, depth + 1)?;
        }
    }
    if let Some(format) = schema.get("format") {
        ensure!(
            format.as_str().and_then(lookup_format).is_some(),
            "Unknown format: {format}"
        );
    }
    if let Some(pattern) = schema.get("pattern") {
        let pattern = pattern
            .as_str()
            .ok_or_else(|| anyhow!("pattern must be a string"))?;
        RegexBuilder::new()
            .mk(&RegexAst::SearchRegex(regex_to_lark(pattern, "dw")))
            .context("Invalid regex in pattern")?;
    }
    for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
        if let Some(children) = schema.get(key) {
            for child in children
                .as_array()
                .ok_or_else(|| anyhow!("{key} must be an array"))?
            {
                validate_schema(root, child, seen, depth + 1)?;
            }
        }
    }
    for key in ["properties", "patternProperties"] {
        if let Some(children) = schema.get(key) {
            for child in children
                .as_object()
                .ok_or_else(|| anyhow!("{key} must be an object"))?
                .values()
            {
                validate_schema(root, child, seen, depth + 1)?;
            }
        }
    }
    for key in ["items", "additionalItems", "additionalProperties"] {
        if let Some(child) = schema.get(key) {
            if let Some(children) = child.as_array() {
                for child in children {
                    validate_schema(root, child, seen, depth + 1)?;
                }
            } else {
                validate_schema(root, child, seen, depth + 1)?;
            }
        }
    }
    Ok(())
}

struct ValueShape {
    string: RegexAst,
    json_types: u8,
}

impl ValueShape {
    fn any() -> Self {
        Self {
            string: rx(ANY),
            json_types: ALL_JSON,
        }
    }
    fn intersect(&mut self, other: Self) {
        self.string = string_intersection(self.string.clone(), other.string);
        self.json_types &= other.json_types;
    }
}

fn literal_shape(value: &Value) -> ValueShape {
    let (string, json_types) = match value {
        Value::String(text) => (RegexAst::Literal(text.clone()), 0),
        Value::Number(_) => (RegexAst::NoMatch, 1),
        Value::Bool(_) => (RegexAst::NoMatch, 2),
        Value::Null => (RegexAst::NoMatch, 4),
        Value::Object(_) => (RegexAst::NoMatch, 8),
        Value::Array(_) => (RegexAst::NoMatch, 16),
    };
    ValueShape { string, json_types }
}

fn value_shape(root: &Value, schema: &Value, depth: usize) -> Result<ValueShape> {
    ensure!(
        depth <= MAX_DEPTH,
        "Cyclic or excessively nested parameter $ref"
    );
    let root = if schema.get("$id").is_some() {
        schema
    } else {
        root
    };
    if schema == &Value::Bool(false) {
        return Ok(ValueShape {
            string: RegexAst::NoMatch,
            json_types: 0,
        });
    }
    let mut shape = ValueShape::any();
    if let Some(kinds) = schema.get("type") {
        let kinds = if let Some(kinds) = kinds.as_array() {
            kinds.clone()
        } else {
            vec![kinds.clone()]
        };
        let mut string = false;
        let mut json_types = 0;
        for kind in kinds {
            let kind = kind
                .as_str()
                .ok_or_else(|| anyhow!("type must be a string or array of strings"))?;
            match kind {
                "string" => string = true,
                "integer" => json_types |= 1,
                _ => {
                    json_types |= 1
                        << JSON_TYPES
                            .iter()
                            .position(|k| *k == kind)
                            .ok_or_else(|| anyhow!("Unknown schema type: {kind}"))?
                }
            }
        }
        shape.string = if string { rx(ANY) } else { RegexAst::NoMatch };
        shape.json_types = json_types;
    }
    let min = bound(schema, "minLength")?.unwrap_or(0);
    let max = bound(schema, "maxLength")?;
    if max.is_some_and(|max| min > max) {
        shape.string = RegexAst::NoMatch;
    } else if min != 0 || max.is_some() {
        shape.string = RegexAst::And(vec![
            shape.string,
            rx(&format!(
                "(?s:.{{{min},{}}})",
                max.map_or(String::new(), |n| n.to_string())
            )),
        ]);
    }
    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
        shape.string = RegexAst::And(vec![shape.string, search_pattern(pattern)?]);
    }
    if let Some(format) = schema.get("format").and_then(Value::as_str) {
        shape.string = RegexAst::And(vec![
            shape.string,
            rx(lookup_format(format).ok_or_else(|| anyhow!("Unknown format: {format}"))?),
        ]);
    }
    if let Some(value) = schema.get("const") {
        shape.intersect(literal_shape(value));
    }
    if let Some(values) = schema.get("enum") {
        let options = values
            .as_array()
            .ok_or_else(|| anyhow!("enum must be an array"))?
            .iter()
            .map(literal_shape)
            .collect::<Vec<_>>();
        shape.intersect(union_shape(options, false));
    }
    if let Some(reference) = schema.get("$ref") {
        shape.intersect(value_shape(root, local_ref(root, reference)?, depth + 1)?);
    }
    for key in ["allOf", "anyOf", "oneOf"] {
        if let Some(options) = schema.get(key) {
            let options = options
                .as_array()
                .ok_or_else(|| anyhow!("{key} must be an array"))?
                .iter()
                .map(|option| value_shape(root, option, depth + 1))
                .collect::<Result<Vec<_>>>()?;
            if key == "oneOf" {
                for (i, option) in options.iter().enumerate() {
                    let containers = options
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .fold(0, |types, (_, other)| types | (other.json_types & 24));
                    if containers != 0 {
                        ensure!(
                            !nonempty(&RegexAst::And(vec![
                                option.string.clone(),
                                container_prefix(containers),
                            ]))?,
                            "oneOf with overlapping raw string and JSON container encodings is unsupported"
                        );
                    }
                }
            }
            if key == "allOf" {
                for option in options {
                    shape.intersect(option);
                }
            } else {
                shape.intersect(union_shape(options, key == "oneOf"));
            }
        }
    }
    Ok(shape)
}

fn container_prefix(types: u8) -> RegexAst {
    let opening = match types & 24 {
        8 => r"\{",
        16 => r"\[",
        _ => r"[\{\[]",
    };
    rx(&format!(r"[ \t\r\n]*{opening}{ANY}"))
}

fn string_intersection(left: RegexAst, right: RegexAst) -> RegexAst {
    match (&left, &right) {
        (RegexAst::NoMatch, _) | (_, RegexAst::NoMatch) => RegexAst::NoMatch,
        (RegexAst::Regex(pattern), _) if pattern == ANY => right,
        (_, RegexAst::Regex(pattern)) if pattern == ANY => left,
        _ => RegexAst::And(vec![left, right]),
    }
}

fn union_shape(options: Vec<ValueShape>, exclusive: bool) -> ValueShape {
    let json_types = options
        .iter()
        .fold(0, |types, option| types | option.json_types);
    let string = union_regex(options.into_iter().map(|s| s.string).collect(), exclusive);
    ValueShape { string, json_types }
}

fn union_regex(options: Vec<RegexAst>, exclusive: bool) -> RegexAst {
    let mut strings: Vec<_> = options
        .into_iter()
        .filter(|s| !matches!(s, RegexAst::NoMatch))
        .collect();
    match strings.len() {
        0 => RegexAst::NoMatch,
        1 => strings.pop().unwrap(),
        _ if exclusive => RegexAst::Or(
            strings
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    RegexAst::And(vec![
                        value.clone(),
                        RegexAst::Not(Box::new(RegexAst::Or(
                            strings
                                .iter()
                                .enumerate()
                                .filter(|(j, _)| *j != i)
                                .map(|(_, value)| value.clone())
                                .collect(),
                        ))),
                    ])
                })
                .collect(),
        ),
        _ => RegexAst::Or(strings),
    }
}

fn scalar_regex(
    schema: &Schema,
    definitions: &HashMap<String, Schema>,
    depth: usize,
) -> Result<RegexAst> {
    ensure!(
        depth <= MAX_DEPTH,
        "Cyclic or excessively nested scalar $ref"
    );
    Ok(match schema {
        Schema::Unsatisfiable(_) => RegexAst::NoMatch,
        Schema::Null => RegexAst::Literal("null".to_string()),
        Schema::Boolean(Some(value)) => RegexAst::Literal(value.to_string()),
        Schema::Boolean(None) => rx("true|false"),
        Schema::Number(number) => {
            if check_number_bounds(number).is_err() {
                return Ok(RegexAst::NoMatch);
            }
            ensure!(
                number.multiple_of.is_none(),
                "multipleOf in a raw string/numeric union is unsupported"
            );
            let (min, exclusive_min) = number.get_minimum();
            let (max, exclusive_max) = number.get_maximum();
            let range = rx(&rx_float_range(min, max, !exclusive_min, !exclusive_max)?);
            if number.integer {
                RegexAst::And(vec![range, rx(&rx_int_range(None, None)?)])
            } else {
                range
            }
        }
        Schema::AnyOf(options) | Schema::OneOf(options) => union_regex(
            options
                .iter()
                .map(|option| scalar_regex(option, definitions, depth + 1))
                .collect::<Result<_>>()?,
            matches!(schema, Schema::OneOf(_)),
        ),
        Schema::Ref(reference) => scalar_regex(
            definitions
                .get(reference)
                .ok_or_else(|| anyhow!("Cannot resolve scalar $ref: {reference}"))?,
            definitions,
            depth + 1,
        )?,
        _ => bail!("Expected a scalar schema"),
    })
}

// Keep JSON Schema search semantics, including alternatives with different anchors.
// Derivre uses the same regex-syntax HIR for JSON patterns; unsupported lookarounds
// remain errors rather than being silently treated as ordinary characters.
fn search_pattern(pattern: &str) -> Result<RegexAst> {
    fn branches(hir: &Hir, budget: &mut usize) -> Result<Vec<(Hir, bool, bool)>> {
        ensure!(*budget > 0, "Pattern expansion limit exceeded");
        *budget -= 1;
        if hir.properties().look_set().is_empty() {
            return Ok(vec![(hir.clone(), false, false)]);
        }
        match hir.kind() {
            HirKind::Look(Look::Start) => Ok(vec![(Hir::empty(), true, false)]),
            HirKind::Look(Look::End) => Ok(vec![(Hir::empty(), false, true)]),
            HirKind::Capture(capture) => branches(&capture.sub, budget),
            HirKind::Alternation(options) => {
                let mut result = Vec::new();
                for option in options {
                    result.extend(branches(option, budget)?);
                }
                Ok(result)
            }
            HirKind::Concat(parts) => {
                let mut result = vec![(Hir::empty(), false, false)];
                for part in parts {
                    let suffixes = branches(part, budget)?;
                    let mut next = Vec::new();
                    for (prefix, start, end) in &result {
                        for (suffix, s, e) in &suffixes {
                            ensure!(*budget > 0, "Pattern expansion limit exceeded");
                            *budget -= 1;
                            ensure!(
                                !s || matches!(prefix.kind(), HirKind::Empty),
                                "Unsupported regex start anchor inside a pattern"
                            );
                            ensure!(
                                !end || matches!(suffix.kind(), HirKind::Empty),
                                "Unsupported regex end anchor inside a pattern"
                            );
                            next.push((
                                Hir::concat(vec![prefix.clone(), suffix.clone()]),
                                start | s,
                                end | e,
                            ));
                        }
                    }
                    result = next;
                }
                Ok(result)
            }
            _ => bail!("Unsupported regex lookaround or repeated anchor"),
        }
    }
    let hir = regex_syntax::Parser::new()
        .parse(&regex_to_lark(pattern, "dw"))
        .context("Invalid pattern regex")?;
    Ok(RegexAst::Or(
        branches(&hir, &mut 10_000)?
            .into_iter()
            .map(|(hir, start, end)| {
                rx(&format!(
                    "{}(?:{hir}){}",
                    if start { "" } else { ANY },
                    if end { "" } else { ANY }
                ))
            })
            .collect(),
    ))
}

fn nonempty(regex: &RegexAst) -> Result<bool> {
    let mut builder = RegexBuilder::new();
    let expr = builder.mk(regex).context("Invalid regex")?;
    if builder.exprset().is_positive(expr) {
        return Ok(true);
    }
    Ok(!builder
        .into_regex_limited(expr, 100_000)
        .context("Cannot determine whether string constraints are satisfiable")?
        .always_empty())
}

fn lark_regex(regex: &RegexAst) -> String {
    let combine = |parts: &[RegexAst], separator: &str| {
        format!(
            "({})",
            parts
                .iter()
                .map(lark_regex)
                .collect::<Vec<_>>()
                .join(separator)
        )
    };
    match regex {
        RegexAst::Regex(text) => format!("/{}/", regex_to_lark(text, "")),
        RegexAst::Literal(text) => quoted(text),
        RegexAst::EmptyString => "\"\"".to_string(),
        RegexAst::NoMatch => "(/(?s:.*)/ & ~/(?s:.*)/)".to_string(),
        RegexAst::And(parts) if parts.is_empty() => format!("/{ANY}/"),
        RegexAst::Or(parts) if parts.is_empty() => lark_regex(&RegexAst::NoMatch),
        RegexAst::Concat(parts) if parts.is_empty() => "\"\"".to_string(),
        RegexAst::And(parts) => combine(parts, " & "),
        RegexAst::Or(parts) => combine(parts, " | "),
        RegexAst::Concat(parts) => combine(parts, " "),
        RegexAst::Not(inner) => format!("~({})", lark_regex(inner)),
        _ => unreachable!("only raw string regex nodes are constructed"),
    }
}
