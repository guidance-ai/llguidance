//! Tool-call grammar generation, inspired by XGrammar: <https://github.com/mlc-ai/xgrammar>.

use anyhow::{anyhow, bail, ensure, Context, Result};
use indexmap::IndexSet;
use serde_json::json;
use std::fmt::Write;

use crate::{
    json::{
        formats::lookup_format,
        numeric::{check_number_bounds, rx_float_range, rx_int_range},
        schema::{build_schema, Schema},
    },
    regex_to_lark, HashMap,
};
use derivre::{RegexAst, RegexBuilder};
use regex_syntax::hir::{Hir, HirKind, Look};
use serde_json::Value;

/// Builds a Lark grammar from OpenAI function tool definitions.
///
/// The helper supplies the call envelope, reasoning regions, and marker handling.
/// Parameter tags and raw strings are Lark rules; JSON values use ordinary `%json`.
/// Scalar alternatives to raw strings share a Lark terminal with the closing tag
/// so greedy lexing cannot discard a shorter scalar match.
/// Mixed string/numeric `multipleOf` and strings overlapping a JSON container's
/// opening prefix are unsupported. Standalone JSON retains its usual constraints.
/// Only `function.strict = true` enables schema constraints. For those tools,
/// omitted top-level `additionalProperties` is treated as `false`. Parameters
/// follow schema order, with optional ones skippable. The reference Qwen3-Coder
/// parser strips one leading and one trailing LF from each parameter value;
/// other whitespace remains part of a string. Strings cannot contain their
/// closing `</parameter>` delimiter.
///
/// `special_tokens` lists protocol markers that also accept a special-token form.
/// Ordinary text forms remain accepted. `None` uses the format defaults; an explicit
/// list replaces them, and an empty list uses only ordinary text forms.
/// For `qwen3_coder`, the defaults are `<tool_call>`, `</tool_call>`, `<think>`,
/// and `</think>`. Names outside the format are invalid. The listed markers must
/// exist as special tokens in the tokenizer used to construct a matcher.
///
/// Unsupported strict schemas and invalid format options return an error.
pub fn grammar_from_tools(
    tools: &Value,
    format: &str,
    initial_reasoning: bool,
    special_tokens: Option<&[String]>,
) -> Result<String> {
    ensure!(
        format == "qwen3_coder",
        "Unsupported tool-call format: {format}"
    );
    const MARKERS: [&str; 4] = ["<tool_call>", "</tool_call>", "<think>", "</think>"];
    let special: IndexSet<&str> = match special_tokens {
        None => MARKERS.into_iter().collect(),
        Some(names) => {
            for name in names {
                ensure!(
                    MARKERS.contains(&name.as_str()),
                    "Unknown special token marker: {name}"
                );
            }
            names.iter().map(String::as_str).collect()
        }
    };
    let tools = tools
        .as_array()
        .ok_or_else(|| anyhow!("tools must be an array"))?;
    let mut names = IndexSet::new();
    let mut strict = Vec::new();
    for tool in tools {
        ensure!(
            tool.get("type").and_then(Value::as_str) == Some("function"),
            "Expected a function tool"
        );
        let function = tool
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow!("Missing function definition"))?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("Missing function name"))?;
        ensure!(
            !name.is_empty()
                && !name
                    .chars()
                    .any(|ch| ch.is_whitespace() || ch == '<' || ch == '>'),
            "Invalid function name: {name:?}"
        );
        ensure!(names.insert(name), "Duplicate function name: {name}");
        if function.get("strict") == Some(&Value::Bool(true)) {
            let schema = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "additionalProperties": false}));
            strict.push((name, schema));
        }
    }
    if strict.is_empty() {
        return Ok("start: <[*]>*\n".to_string());
    }

    let mut grammar = String::from("%llguidance {\"ignore_once\":true}\n");
    writeln!(
        grammar,
        "start: {} response",
        if initial_reasoning { "thought" } else { "" }
    )?;
    // Separate safe-text lexemes let the slicer skip ordinary text tokens.
    // Delimiter-aware lexemes handle the remaining tokens without lazy matching.
    grammar.push_str(
        r#"
response: (tool_head function | think_head thought)* (TEXT | SAFE)
TEXT: /(?s:.*)/ & ~/(?s:.*)(<tool_call>|<think>)(?s:.*)/
SAFE: /[^<]*/
THOUGHT_TEXT: /(?s:.*)/ & ~/(?s:.*)<\/think>(?s:.*)/
LOOSE_TEXT: /(?s:.*)/ & ~/(?s:.*)<\/function>(?s:.*)/
WS: /[ \t\r\n]*/
tool_head_text: TOOL_HEAD | SAFE_TOOL_HEAD
TOOL_HEAD: TEXT "<tool_call>"
SAFE_TOOL_HEAD: SAFE "<tool_call>"
think_head_text: THINK_HEAD | SAFE_THINK_HEAD
THINK_HEAD: TEXT "<think>"
SAFE_THINK_HEAD: SAFE "<think>"
thought_end: THOUGHT_END | SAFE_THOUGHT_END
THOUGHT_END: THOUGHT_TEXT "</think>"
SAFE_THOUGHT_END: SAFE "</think>"
function: WS "<function=" dispatch WS tool_end
loose_body: LOOSE_BODY | SAFE_LOOSE_BODY
LOOSE_BODY: LOOSE_TEXT "</function>"
SAFE_LOOSE_BODY: SAFE "</function>"
"#,
    );
    for (rule, text, marker) in [
        ("tool_head", "tool_head_text", "<tool_call>"),
        ("think_head", "think_head_text", "<think>"),
        ("tool_end", "\"</tool_call>\"", "</tool_call>"),
    ] {
        writeln!(
            grammar,
            "{rule}: {text}{}",
            if special.contains(marker) {
                format!(
                    " | {}{marker}",
                    if rule == "tool_end" {
                        ""
                    } else {
                        "(TEXT | SAFE) "
                    }
                )
            } else {
                String::new()
            }
        )?;
    }
    let mut thought = vec!["thought_end".to_string()];
    if special.contains("</think>") {
        thought.push("(THOUGHT_TEXT | SAFE) </think>".to_string());
    }
    for marker in special.iter().filter(|&&name| name != "</think>") {
        thought.push(format!("(THOUGHT_TEXT | SAFE) {marker} thought"));
    }
    writeln!(grammar, "thought: {}", thought.join(" | "))?;
    let excluded = strict
        .iter()
        .map(|(name, _)| serde_json::to_string(name).unwrap())
        .collect::<Vec<_>>()
        .join(" | ");
    writeln!(grammar, "UNKNOWN_NAME: /[^\\s<>]+/ & ~({excluded})")?;
    let mut dispatch = vec!["UNKNOWN_NAME \">\" loose_body".to_string()];
    let mut values = HashMap::default();
    for (idx, (name, schema)) in strict.iter().enumerate() {
        dispatch.push(format!(
            "{} WS arguments_{idx} WS \"</function>\"",
            serde_json::to_string(&format!("{name}>")).unwrap()
        ));
        parameters(&mut grammar, &mut values, idx, schema)
            .with_context(|| format!("tool {name:?}"))?;
    }
    writeln!(grammar, "dispatch: {}", dispatch.join(" | "))?;
    Ok(grammar)
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

fn name_is_valid(name: &str) -> bool {
    !name.is_empty()
        && !name
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '<' | '>'))
}

fn bound(schema: &Value, key: &str) -> Result<Option<u32>> {
    schema
        .get(key)
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| anyhow!("{key} must be a nonnegative 32-bit integer"))
        })
        .transpose()
}

fn parameters(
    grammar: &mut String,
    values: &mut HashMap<String, String>,
    index: usize,
    root: &Value,
) -> Result<()> {
    let mut schema = root;
    let mut depth = 0;
    while let Some(reference) = schema.get("$ref") {
        depth += 1;
        ensure!(depth <= MAX_DEPTH, "Cyclic root $ref");
        ensure!(
            schema.as_object().unwrap().keys().all(|k| matches!(
                k.as_str(),
                "$ref" | "$defs" | "definitions" | "title" | "description" | "$schema"
            )),
            "Unsupported siblings of root $ref"
        );
        schema = local_ref(root, reference)?;
    }
    ensure!(
        schema.get("type").and_then(Value::as_str) == Some("object"),
        "Tool parameters require an object schema with type: object"
    );
    validate_keywords(schema)?;
    for key in ["allOf", "anyOf", "oneOf", "const", "enum"] {
        ensure!(
            schema.get(key).is_none(),
            "Unsupported top-level {key}; apply it to parameter values"
        );
    }
    ensure!(
        schema.get("patternProperties").is_none(),
        "Top-level patternProperties is unsupported"
    );
    let empty = serde_json::Map::new();
    let properties = match schema.get("properties") {
        None => &empty,
        Some(value) => value
            .as_object()
            .ok_or_else(|| anyhow!("properties must be an object"))?,
    };
    let required: IndexSet<&str> = match schema.get("required") {
        None => IndexSet::new(),
        Some(value) => value
            .as_array()
            .ok_or_else(|| anyhow!("required must be an array"))?
            .iter()
            .map(|name| {
                name.as_str()
                    .ok_or_else(|| anyhow!("required names must be strings"))
            })
            .collect::<Result<_>>()?,
    };
    let extra = match schema.get("additionalProperties") {
        None | Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) => true,
        _ => bail!("Schema-valued top-level additionalProperties is unsupported"),
    };
    let min = bound(schema, "minProperties")?.unwrap_or(0);
    let max = bound(schema, "maxProperties")?;
    ensure!(
        (min == 0 && max.is_none()) || properties.keys().all(|key| required.contains(key.as_str())),
        "min/maxProperties with optional parameters is unsupported"
    );
    ensure!(
        max.is_none_or(|max| max as usize >= required.len()),
        "Unsatisfiable maxProperties"
    );
    ensure!(
        extra || min as usize <= required.len(),
        "Unsatisfiable minProperties"
    );
    let mut names: IndexSet<&str> = properties.keys().map(String::as_str).collect();
    names.extend(required.iter().copied());
    let mut sequence = Vec::new();
    for (i, name) in names.iter().enumerate() {
        ensure!(name_is_valid(name), "Invalid parameter name: {name:?}");
        let property = properties.get(*name).unwrap_or(if extra {
            &Value::Bool(true)
        } else {
            &Value::Bool(false)
        });
        let result = value_rule(grammar, values, root, property)
            .with_context(|| format!("parameter {name:?}"));
        let rule = match result {
            Ok(Some(rule)) => rule,
            Ok(None) if !required.contains(name) => continue,
            Ok(None) => bail!("parameter {name:?}: Unsatisfiable schema"),
            Err(error) => return Err(error),
        };
        let parameter = format!("parameter_{index}_{i}");
        writeln!(
            grammar,
            "{parameter}: {} {rule} WS",
            quoted(&format!("<parameter={name}>"))
        )?;
        sequence.push(format!(
            "{parameter}{}",
            if required.contains(name) { "" } else { "?" }
        ));
    }
    if extra {
        let excluded = names
            .iter()
            .map(|name| quoted(name))
            .collect::<Vec<_>>()
            .join(" | ");
        writeln!(
            grammar,
            "EXTRA_NAME_{index}: /[^\\s<>]+/{}",
            if excluded.is_empty() {
                String::new()
            } else {
                format!(" & ~({excluded})")
            }
        )?;
        let plain = raw_value_rule(grammar, values, rx(ANY), false)?;
        let padded = raw_value_rule(grammar, values, rx(ANY), true)?;
        let free = format!("({plain} | \"\\n\" {padded})");
        writeln!(
            grammar,
            "EXTRA_HEAD_{index}: \"<parameter=\" EXTRA_NAME_{index} \">\"\nextra_{index}: EXTRA_HEAD_{index} {free} WS"
        )?;
        let min_extra = min.saturating_sub(required.len() as u32);
        let max_extra = max.map(|n| n.saturating_sub(required.len() as u32));
        sequence.push(format!(
            "extra_{index}{{{min_extra},{}}}",
            max_extra.map_or(String::new(), |n| n.to_string())
        ));
    }
    writeln!(
        grammar,
        "arguments_{index}: {}",
        if sequence.is_empty() {
            "\"\"".to_string()
        } else {
            sequence.join(" ")
        }
    )?;
    Ok(())
}

fn validate_keywords(schema: &Value) -> Result<()> {
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
            ),
            "Unsupported schema keyword: {key}"
        );
    }
    Ok(())
}

fn validate_schema(
    root: &Value,
    schema: &Value,
    seen: &mut IndexSet<String>,
    depth: usize,
) -> Result<()> {
    ensure!(depth <= MAX_DEPTH, "Schema nesting limit exceeded");
    validate_keywords(schema)?;
    if let Some(reference) = schema.get("$ref") {
        let target = local_ref(root, reference)?;
        if seen.insert(reference.to_string()) {
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
    if let Some(children) = schema.get("properties") {
        for child in children
            .as_object()
            .ok_or_else(|| anyhow!("properties must be an object"))?
            .values()
        {
            validate_schema(root, child, seen, depth + 1)?;
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

fn raw_value_rule(
    grammar: &mut String,
    cache: &mut HashMap<String, String>,
    string: RegexAst,
    leading_newline: bool,
) -> Result<String> {
    let raw = lark_regex(&string);
    let key = format!("raw:{leading_newline}:{raw}");
    if let Some(rule) = cache.get(&key) {
        return Ok(rule.clone());
    }
    let id = cache.len();
    let rule = format!("raw_{id}");
    writeln!(
        grammar,
        "VALUE_{id}: ({raw}) & ~/(?s:.*)<\\/parameter>(?s:.*)/"
    )?;
    // A newline belonging to the value needs an additional framing LF. Split
    // the leading frame in the rule so it cannot hide a competing JSON lexeme.
    let start = if leading_newline {
        String::new()
    } else {
        " & ~(/\\n/ /(?s:.*)/)".to_string()
    };
    writeln!(
        grammar,
        "PADDED_{id}: ((VALUE_{id} & ~(/(?s:.*)/ /\\n/)) | (VALUE_{id} \"\\n\")){start}"
    )?;
    // The slicer can bulk-accept ordinary text with no potential closing tag.
    writeln!(
        grammar,
        "SAFE_VALUE_{id}: (PADDED_{id} & /[^<]*/) \"</parameter>\"\nVALUE_END_{id}: PADDED_{id} \"</parameter>\"\n{rule}: SAFE_VALUE_{id} | VALUE_END_{id}"
    )?;
    cache.insert(key, rule.clone());
    Ok(rule)
}

fn value_rule(
    grammar: &mut String,
    cache: &mut HashMap<String, String>,
    root: &Value,
    schema: &Value,
) -> Result<Option<String>> {
    validate_schema(root, schema, &mut IndexSet::new(), 0)?;
    let shape = value_shape(root, schema, 0)?;
    let mut string = RegexAst::And(vec![
        shape.string,
        RegexAst::Not(Box::new(rx("(?s:.*)</parameter>(?s:.*)"))),
    ]);
    let mut options = Vec::new();
    let mut padded = Vec::new();
    let has_string = nonempty(&string)?;
    // Separate scalars and containers so only container interiors permit JSON
    // whitespace. Newline padding belongs to the parameter envelope.
    for mask in [7, 24] {
        let types: Vec<_> = JSON_TYPES
            .iter()
            .enumerate()
            .filter(|(i, _)| shape.json_types & mask & (1 << i) != 0)
            .map(|(_, kind)| *kind)
            .collect();
        if types.is_empty() {
            continue;
        }
        if mask == 24 && has_string {
            let opening = match shape.json_types & 24 {
                8 => r"\{",
                16 => r"\[",
                _ => r"[\{\[]",
            };
            ensure!(
                !nonempty(&RegexAst::And(vec![
                    string.clone(),
                    rx(&format!(r"[ \t\r\n]*{opening}{ANY}")),
                ]))?,
                "Raw strings overlapping JSON object/array prefixes are unsupported"
            );
        }
        let mut json = json!({"allOf":[schema, {"type":types}]});
        for key in ["$defs", "definitions"] {
            if let Some(definitions) = root.get(key) {
                json[key] = definitions.clone();
            }
        }
        if mask == 7 && has_string {
            let compiled = build_schema(json, &Default::default())?;
            let scalar = scalar_regex(&compiled.schema, &compiled.definitions, 0)?;
            string = union_regex(vec![string, scalar], false);
            continue;
        }
        if mask == 7 {
            json["x-guidance"] = json!({"whitespace_flexible":false});
        }
        let key = json.to_string();
        let rule = if let Some(rule) = cache.get(&key) {
            rule.clone()
        } else {
            let id = cache.len();
            let rule = format!("json_{id}");
            writeln!(grammar, "{rule}: %json {json} /\\n?/ \"</parameter>\"")?;
            cache.insert(key, rule.clone());
            rule
        };
        padded.push(rule.clone());
        options.push(rule);
    }
    if has_string {
        options.push(raw_value_rule(grammar, cache, string.clone(), false)?);
        padded.push(raw_value_rule(grammar, cache, string, true)?);
    }
    if options.is_empty() {
        Ok(None)
    } else {
        Ok(Some(format!(
            "(({}) | \"\\n\" ({}))",
            options.join(" | "),
            padded.join(" | ")
        )))
    }
}
