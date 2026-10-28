use anyhow::{bail, Result};
use serde_json::Value;

/// Builds a Lark grammar for strict Qwen3-Coder function tools.
///
/// Strings use raw text; other values use JSON. Parameters follow schema order.
/// A missing top-level `additionalProperties` means false for strict tools.
/// One leading and one trailing LF are removed from raw parameter values by the
/// reference Qwen3-Coder parser. Other whitespace is part of a string value.
/// `None` selects the four Qwen protocol special tokens; an empty slice uses text
/// markers only. Explicit marker lists replace the defaults.
pub fn grammar_from_tools(
    _tools: &Value,
    _format: &str,
    _initial_reasoning: bool,
    _special_tokens: Option<&[String]>,
) -> Result<String> {
    bail!("Qwen tool grammar generation is not implemented")
}
