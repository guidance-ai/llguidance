//! Utilities for downloading tokenizer files from [HuggingFace Hub](https://huggingface.co/)
//! and constructing [`toktrie`] tokenizer environments from them.
//! Uses [`toktrie_hf_tokenizers`] under the hood.

use anyhow::{ensure, Result};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;
use toktrie::TokEnv;
use toktrie_hf_tokenizers::ByteTokenizer;

fn strip_suffix(sep: &str, s: &mut String) -> Option<String> {
    let mut parts = s.splitn(2, sep);
    let core = parts.next().unwrap().to_string();
    let suff = parts.next().map(|s| s.to_string());
    *s = core;
    suff
}

/// Downloads the `tokenizer.json` file for a model from HuggingFace Hub.
///
/// The `name` can include a revision suffix in the form `"model_name@revision"`;
/// if no revision is specified, `"main"` is used. Model names and revisions may only
/// contain alphanumeric characters or `'-'`, `'_'`, `'.'`, `'/'`.
pub fn download_tokenizer_json(name: &str) -> Result<PathBuf> {
    let mut name2 = name.to_string();
    let revision = strip_suffix("@", &mut name2).unwrap_or("main".to_string());

    let valid_chars = ['-', '_', '.', '/'];
    let is_valid_char = |x: char| x.is_alphanumeric() || valid_chars.contains(&x);

    ensure!(
        name2.chars().all(is_valid_char),
        "Model \"{}\" contains invalid characters, expected only alphanumeric or {:?}",
        name,
        valid_chars
    );

    ensure!(
        revision.chars().all(is_valid_char),
        "Revision \"{}\" contains invalid characters, expected only alphanumeric or {:?}",
        revision,
        valid_chars
    );

    let builder = ApiBuilder::new();
    let api = builder.build()?;
    let repo = Repo::with_revision(name2, RepoType::Model, revision);
    let api = api.repo(repo);
    Ok(api.get("tokenizer.json")?)
}

/// Returns a local path directly if `name` starts with `"."` or `"/"`, or if it exists
/// as a local path. Otherwise downloads from HuggingFace Hub via [`download_tokenizer_json`].
pub fn maybe_download_tokenizer_json(name: &str) -> Result<PathBuf> {
    if name.starts_with(".") || name.starts_with("/") || std::path::Path::new(name).exists() {
        Ok(PathBuf::from(name))
    } else {
        download_tokenizer_json(name)
    }
}

/// Resolves a tokenizer by name (local path or HuggingFace model) and constructs a
/// [`ByteTokenizer`] from it.
pub fn byte_tokenizer_from_name(name: &str) -> Result<ByteTokenizer> {
    let path = maybe_download_tokenizer_json(name)?;
    ByteTokenizer::from_file(path)
}

/// Resolves a tokenizer by name (local path or HuggingFace model) and constructs a
/// [`TokEnv`] from it.
pub fn tok_env_from_name(name: &str) -> Result<TokEnv> {
    let path = maybe_download_tokenizer_json(name)?;
    ByteTokenizer::from_file(path)?.into_tok_env(None)
}
