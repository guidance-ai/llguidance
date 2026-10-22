//! Offline tokenizer/grammar observations; no model logits or inference.
#[path = "leading_space/support.rs"]
mod support;

use anyhow::{bail, Context, Result};
use serde_json::json;
use std::{env, fs};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    let mut results = Vec::new();
    if args.is_empty() {
        for (name, bytes) in support::FIXTURES {
            results.push(support::analyze(name, bytes)?);
        }
    } else if args.first().map(String::as_str) == Some("--tokenizer") && args.len() >= 2 {
        for path in &args[1..] {
            let bytes = fs::read(path).with_context(|| format!("reading {path}"))?;
            results.push(support::analyze(path, &bytes)?);
        }
    } else {
        bail!("usage: leading_space [--tokenizer LOCAL_TOKENIZER_JSON ...]");
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "scope": "tokenizer preprocessing, token bytes, and grammar validity; no model inference",
            "add_special_tokens": false,
            "grammars": support::GRAMMARS.iter().map(|(name, lark)| json!({
                "name": name, "lark": lark,
            })).collect::<Vec<_>>(),
            "results": results,
        }))?
    );
    Ok(())
}
