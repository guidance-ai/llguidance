mod cases;
mod results;
mod run;

use crate::{
    cases::Case,
    results::{CaseResult, CategoryResult},
    run::run_case_tests,
};
use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use llguidance::toktrie::TokEnv;
use referencing::Retrieve;
use serde_json::{Map, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    rc::Rc,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct CliOptions {
    /// Path to the root directory of the JSON schema test suite
    #[arg(
        long,
        value_name = "ROOT_DIR",
        help = "Path to the root directory of the JSON schema test suite"
    )]
    root_dir: String,

    /// Path to the output directory
    #[arg(
        long,
        value_name = "OUTPUT_PATH",
        help = "Path to the output directory"
    )]
    output_path: String,

    /// Which tests to run (default: main, can be set to format)
    #[arg(
        long,
        value_enum,
        default_value = "main",
        help = "Which tests to run (main or format)"
    )]
    tests: TestsType,

    /// Which schema to use (default: latest, can be set to draft202012)
    #[arg(
        long,
        value_enum,
        default_value = "latest",
        help = "Which draft to use (latest or 2020-12)"
    )]
    draft: Draft,

    /// Tokenizer to use (default: microsoft/Phi-3.5-mini-instruct)
    #[arg(long, default_value = "microsoft/Phi-3.5-mini-instruct")]
    tokenizer: String,
}

#[derive(ValueEnum, Clone, Debug, Default)]
enum TestsType {
    #[default]
    #[value(name = "main")]
    Main,
    #[value(name = "format")]
    Format,
}

#[derive(ValueEnum, Clone, Debug, Default)]
enum Draft {
    #[default]
    #[value(name = "latest")]
    Latest,
    #[value(name = "2020-12")]
    Draft202012,
}

fn list_test_files_in_dir(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            if path.is_file() && path.extension().unwrap() == "json" {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
struct MapRetrieverError(String);
impl std::fmt::Display for MapRetrieverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Resource not found: {}", self.0)
    }
}
impl std::error::Error for MapRetrieverError {}

struct MapRetriever {
    map: Map<String, Value>,
}
impl Retrieve for MapRetriever {
    fn retrieve(
        &self,
        uri: &referencing::Uri<&str>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        match self.map.get(uri.as_str()) {
            Some(value) => Ok(value.clone()),
            None => Err(Box::new(MapRetrieverError(uri.as_str().to_string()))),
        }
    }
}

fn make_remotes_retriever(root_dir: &Path) -> Result<MapRetriever> {
    let output =
        std::process::Command::new(root_dir.join("bin/jsonschema_suite").to_str().unwrap())
            .arg("remotes")
            .output()
            .expect("Failed to execute 'bin/jsonschema_suite remotes'");

    let stdout = String::from_utf8(output.stdout)?;
    let value: Value = serde_json::from_str(&stdout)?;
    if let Value::Object(m) = value {
        Ok(MapRetriever { map: m })
    } else {
        bail!("Expected 'bin/jsonschema_suite remotes' to return a JSON object")
    }
}

fn main() {
    let options = CliOptions::parse();
    let root_dir = Path::new(&options.root_dir);
    assert!(root_dir.exists(), "Root directory does not exist");
    assert!(root_dir.is_dir(), "Root directory is not a directory");

    let draft_dir = root_dir.join(match options.draft {
        Draft::Latest => "tests/latest",
        Draft::Draft202012 => "tests/draft2020-12",
    });
    let test_dir = draft_dir.join(match options.tests {
        TestsType::Format => "optional/format",
        TestsType::Main => ".",
    });
    let test_files = list_test_files_in_dir(&test_dir);

    let retriever = Rc::new(make_remotes_retriever(root_dir).unwrap());

    let tok_env: TokEnv =
        toktrie_hf_tokenizers::ByteTokenizerEnv::from_name(&options.tokenizer, None)
            .unwrap()
            .to_env();

    let mut results = Vec::new();
    for test_file in test_files {
        let mut case_results = Vec::new();
        let test_file = test_file.as_path();
        let file_name = test_file.file_stem().unwrap().to_str().unwrap();
        let cases: Vec<Case> =
            serde_json::from_str(&fs::read_to_string(test_file).unwrap()).unwrap();
        for case in cases.into_iter() {
            let test_results = run_case_tests(
                case,
                tok_env.clone(),
                Rc::clone(&retriever) as Rc<dyn Retrieve>,
            );
            case_results.push(CaseResult {
                tests: test_results,
            });
        }
        results.push(CategoryResult {
            category: file_name.to_string(),
            cases: case_results,
        });
    }
    let output_json = serde_json::to_string(&results).unwrap();
    fs::write(&options.output_path, output_json).unwrap();
}
