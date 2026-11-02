//! Qwen3-Coder parameter encoding: `<parameter=NAME>VALUE</parameter>`.
//!
//! Raw strings are Lark terminals; JSON values use ordinary `%json`. Scalar
//! alternatives to raw strings share a terminal with the closing tag so greedy
//! lexing cannot discard a shorter scalar match.
//!
//! Parameters follow schema order, with optional ones skippable and additional
//! parameters allowed by default. The reference Qwen3-Coder parser strips one
//! leading and one trailing LF from each value; other whitespace belongs to the
//! string. Strings cannot contain their closing `</parameter>` delimiter.
//!
//! An unrestricted string alternative covers other `anyOf` alternatives without
//! JSON rules. Overlapping raw string/container `oneOf` encodings, constrained
//! strings overlapping container prefixes, and constrained string/numeric
//! `multipleOf` are unsupported. Standalone JSON retains its usual constraints.

use anyhow::{anyhow, bail, ensure, Context, Result};
use indexmap::IndexSet;
use serde_json::{json, Value};
use std::fmt::Write;

use super::{
    bound, container_prefix, lark_regex, local_ref, nonempty, quoted, rx, scalar_regex,
    union_regex, validate_keywords, validate_schema, value_shape, ANY, JSON_TYPES, MAX_DEPTH,
};
use crate::{json::schema::build_schema, HashMap};
use derivre::RegexAst;

pub(super) fn schema_to_lark(schema: &Value) -> Result<String> {
    let mut grammar = String::from("WS: /[ \\t\\r\\n]*/\n");
    parameters(&mut grammar, &mut HashMap::default(), schema)?;
    Ok(grammar)
}

fn name_is_valid(name: &str) -> bool {
    !name.is_empty()
        && !name
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '<' | '>'))
}

fn parameters(
    grammar: &mut String,
    values: &mut HashMap<String, String>,
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
        "XML parameters require an object schema with type: object"
    );
    validate_keywords(schema, false)?;
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
        Some(Value::Bool(false)) => false,
        None | Some(Value::Bool(true)) => true,
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
        let parameter = format!("parameter_{i}");
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
            "EXTRA_NAME: /[^\\s<>]+/{}",
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
            "EXTRA_HEAD: \"<parameter=\" EXTRA_NAME \">\"\nextra: EXTRA_HEAD {free} WS"
        )?;
        let min_extra = min.saturating_sub(required.len() as u32);
        let max_extra = max.map(|n| n.saturating_sub(required.len() as u32));
        sequence.push(format!(
            "extra{{{min_extra},{}}}",
            max_extra.map_or(String::new(), |n| n.to_string())
        ));
    }
    writeln!(
        grammar,
        "start: {}",
        if sequence.is_empty() {
            "\"\"".to_string()
        } else {
            sequence.join(" ")
        }
    )?;
    Ok(())
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
    let json_types = if matches!(&shape.string, RegexAst::Regex(pattern) if pattern == ANY) {
        0
    } else {
        shape.json_types
    };
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
            .filter(|(i, _)| json_types & mask & (1 << i) != 0)
            .map(|(_, kind)| *kind)
            .collect();
        if types.is_empty() {
            continue;
        }
        if mask == 24 && has_string {
            ensure!(
                !nonempty(&RegexAst::And(vec![
                    string.clone(),
                    container_prefix(json_types),
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
