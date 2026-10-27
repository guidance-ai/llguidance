use anyhow::{anyhow, ensure, Result};
use llguidance::{api::TopLevelGrammar, ParserFactory, TokenParser};
use serde_json::{json, Value};
use tokenizers::Tokenizer;
use toktrie::{SimpleVob, TokEnv, TokenId};
use toktrie_hf_tokenizers::ByteTokenizer;

pub const GRAMMARS: [(&str, &str); 2] = [
    (
        "no-leading-space",
        include_str!("fixtures/no-leading-space.lark"),
    ),
    (
        "optional-leading-space",
        include_str!("fixtures/optional-leading-space.lark"),
    ),
];

pub const FIXTURES: [(&str, &[u8]); 7] = [
    (
        "bytelevel-false",
        include_bytes!("fixtures/bytelevel-false.json"),
    ),
    (
        "bytelevel-true",
        include_bytes!("fixtures/bytelevel-true.json"),
    ),
    (
        "metaspace-never",
        include_bytes!("fixtures/metaspace-never.json"),
    ),
    (
        "metaspace-first",
        include_bytes!("fixtures/metaspace-first.json"),
    ),
    (
        "metaspace-always",
        include_bytes!("fixtures/metaspace-always.json"),
    ),
    ("prepend-none", include_bytes!("fixtures/prepend-none.json")),
    (
        "prepend-space",
        include_bytes!("fixtures/prepend-space.json"),
    ),
];

pub fn samples() -> Vec<String> {
    serde_json::from_str(include_str!("fixtures/samples.json")).unwrap()
}

pub fn load(bytes: &[u8]) -> Result<(Tokenizer, Tokenizer, TokEnv)> {
    let source = Tokenizer::from_bytes(bytes).map_err(|e| anyhow!(e.to_string()))?;
    let adapter = ByteTokenizer::from_tokenizer(source.clone())?;
    let adapted = adapter.hf_tokenizer.clone();
    Ok((source, adapted, adapter.into_tok_env(None)?))
}

pub fn parser(factory: &ParserFactory, grammar: &str) -> Result<TokenParser> {
    let mut parser = factory.create_parser(TopLevelGrammar::from_lark(grammar.into()))?;
    parser.start_without_prompt();
    Ok(parser)
}

pub fn initial_mask(factory: &ParserFactory, grammar: &str) -> Result<SimpleVob> {
    parser(factory, grammar)?.compute_mask()
}

// Use a new parser for every sequence: rejection must not affect the next sample.
// Checking both APIs makes disagreement between validation and masks an error.
pub fn check_sequence(factory: &ParserFactory, grammar: &str, tokens: &[TokenId]) -> Result<Value> {
    let validated_prefix = parser(factory, grammar)?.validate_tokens_raw(tokens)?;
    let mut parser = parser(factory, grammar)?;
    for (index, token) in tokens.iter().enumerate() {
        if !parser.compute_mask()?.is_allowed(*token) {
            ensure!(validated_prefix == index, "mask/validation disagreement");
            return Ok(json!({
                "accepted": false,
                "validated_prefix_tokens": validated_prefix,
                "rejected_token_index": index,
            }));
        }
        ensure!(
            parser.consume_token(*token)? == 0,
            "unexpected backtracking"
        );
    }
    ensure!(
        validated_prefix == tokens.len(),
        "mask/validation disagreement"
    );
    Ok(json!({
        "accepted": parser.is_accepting(),
        "validated_prefix_tokens": validated_prefix,
        "rejected_token_index": null,
    }))
}

fn config(tokenizer: &Tokenizer) -> Result<Value> {
    let value = serde_json::to_value(tokenizer)?;
    Ok(json!({
        "normalizer": value["normalizer"],
        "pre_tokenizer": value["pre_tokenizer"],
        "decoder": value["decoder"],
        "post_processor": value["post_processor"],
    }))
}

fn encoding(tokenizer: &Tokenizer, env: &TokEnv, text: &str) -> Result<(Vec<TokenId>, Value)> {
    let encoded = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!(e.to_string()))?;
    let ids = encoded.get_ids().to_vec();
    let decoded = tokenizer
        .decode(&ids, false)
        .map_err(|e| anyhow!(e.to_string()))?;
    let llguidance_bytes = env.tok_trie().decode_raw(&ids);
    let observation = json!({
        "ids": ids,
        "token_spellings": encoded.get_tokens(),
        "hf_decoded_text": decoded,
        "llguidance_bytes": llguidance_bytes,
        "llguidance_text": String::from_utf8_lossy(&llguidance_bytes),
    });
    Ok((ids, observation))
}

pub fn analyze(name: &str, bytes: &[u8]) -> Result<Value> {
    let (source, adapted, env) = load(bytes)?;
    let mut factory = ParserFactory::new_simple(&env)?;
    factory.quiet();
    let masks = GRAMMARS
        .iter()
        .map(|(_, grammar)| initial_mask(&factory, grammar))
        .collect::<Result<Vec<_>>>()?;
    let trie = env.tok_trie();
    let mut counts = [0usize; 2];
    let mut added_count = 0;
    let mut added_examples = Vec::new();
    for id in 0..trie.vocab_size() as TokenId {
        for (index, mask) in masks.iter().enumerate() {
            counts[index] += usize::from(mask.is_allowed(id));
        }
        ensure!(
            !masks[0].is_allowed(id) || masks[1].is_allowed(id),
            "optional grammar must contain the strict grammar's initial mask"
        );
        if masks[1].is_allowed(id) && !masks[0].is_allowed(id) {
            added_count += 1;
            if added_examples.len() < 64 {
                added_examples.push(json!({
                    "id": id,
                    "bytes": trie.token(id),
                    "text": trie.token_str(id),
                }));
            }
        }
    }
    let mut rows = Vec::new();
    for text in samples() {
        let (source_ids, source_encoding) = encoding(&source, &env, &text)?;
        let (adapter_ids, adapter_encoding) = encoding(&adapted, &env, &text)?;
        let env_ids = env.tokenize(&text);
        let env_bytes = trie.decode_raw(&env_ids);
        let paths = [
            ("source_hf", source_ids.as_slice()),
            ("adapter_hf", adapter_ids.as_slice()),
            ("llguidance_env", env_ids.as_slice()),
        ];
        let mut validity = Vec::new();
        for (path, ids) in paths {
            for (grammar_name, grammar) in GRAMMARS {
                validity.push(json!({
                    "path": path,
                    "grammar": grammar_name,
                    "result": check_sequence(&factory, grammar, ids)?,
                }));
            }
        }
        rows.push(json!({
            "input": text,
            "source_hf": source_encoding,
            "adapter_hf": adapter_encoding,
            "llguidance_env": {
                "ids": env_ids,
                "bytes": env_bytes,
                "text": String::from_utf8_lossy(&env_bytes),
                "preserves_input_bytes": env_bytes == text.as_bytes(),
            },
            "validity": validity,
        }));
    }
    Ok(json!({
        "tokenizer": name,
        "vocab_size": trie.vocab_size(),
        "source_config": config(&source)?,
        "adapted_config": config(&adapted)?,
        "initial_mask": {
            "no_leading_space_count": counts[0],
            "optional_leading_space_count": counts[1],
            "added_count": added_count,
            "added_examples_limit": 64,
            "added_examples": added_examples,
        },
        "samples": rows,
    }))
}
