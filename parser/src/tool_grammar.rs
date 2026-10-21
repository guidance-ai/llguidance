//! Tool-call grammar generation, inspired by XGrammar: <https://github.com/mlc-ai/xgrammar>.

use anyhow::{bail, Result};
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
/// This interface is a placeholder; tool grammar compilation is not implemented.
pub fn grammar_from_tools(
    _tools: &Value,
    _format: &str,
    _initial_reasoning: bool,
    _special_tokens: Option<&[String]>,
) -> Result<String> {
    bail!("tool grammar compilation is not implemented")
}
