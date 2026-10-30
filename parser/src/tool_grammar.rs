//! Tool-call grammar generation, inspired by XGrammar: <https://github.com/mlc-ai/xgrammar>.

use anyhow::{anyhow, ensure, Context, Result};
use indexmap::IndexSet;
use serde_json::json;
use std::fmt::Write;

use crate::json::compiler::prepare_tool_schema;
use serde_json::Value;

/// Builds a Lark grammar from OpenAI function tool definitions.
///
/// The helper supplies the call envelope, reasoning regions, and marker handling.
/// Strict argument bodies use `%json` with `x-guidance.style = "qwen_xml"`.
/// Only `function.strict = true` enables schema constraints. For those tools,
/// omitted top-level `additionalProperties` is treated as `false`; the reusable
/// encoding option retains JSON Schema's permissive default.
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
            let schema = prepare_tool_schema(schema).with_context(|| format!("tool {name:?}"))?;
            strict.push((name, schema));
        }
    }
    if strict.is_empty() {
        return Ok("start: <[*]>*\n".to_string());
    }

    let mut grammar = String::from("%llguidance {}\n");
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
    for (idx, (name, schema)) in strict.iter().enumerate() {
        dispatch.push(format!(
            "{} WS arguments_{idx} WS \"</function>\"",
            serde_json::to_string(&format!("{name}>")).unwrap()
        ));
        writeln!(grammar, "arguments_{idx}: %json {schema}")?;
    }
    writeln!(grammar, "dispatch: {}", dispatch.join(" | "))?;
    Ok(grammar)
}
