/// Runs the official JSON Schema Test Suite against our compiler
/// with ratchet-based regression detection.
///
/// Each test case is compiled and run, then categorized by distance from correctness:
///   pass                      - Instance result matches expectation
///   false_negative            - Instance was rejected but should have been accepted
///   compile_error_all_invalid - Schema failed to compile, all instances invalid
///   skip_compile              - Schema failed to compile on unimplemented feature
///   compile_error_valid       - Schema failed to compile, but has valid instances
///   false_positive            - Instance was accepted but should have been rejected
///
/// Usage:
///   cargo run -p json_schema_test_suite --release -- --expected expected.json
///   cargo run -p json_schema_test_suite --release -- --draft draft7 --expected expected.json --update
///
/// The baseline file is keyed by draft: {"draft2020-12": {...}, "draft7": {...}}.
/// Without --draft, all drafts in the baseline are checked.
/// With --draft, only the specified draft(s) are run.
use anyhow::{bail, Result};
use clap::Parser;
use llg_test_utils::{json_schema_check, make_parser};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Deserialize)]
struct TestGroup {
    description: String,
    schema: Value,
    tests: Vec<TestCase>,
}

#[derive(Deserialize)]
struct TestCase {
    description: String,
    data: Value,
    valid: bool,
}

/// Result categories ordered by badness (lower = better).
const CATEGORIES: &[&str] = &[
    "pass",
    "false_negative",
    "compile_error_all_invalid",
    "skip_compile",
    "compile_error_valid",
    "false_positive",
];

fn category_badness(cat: &str) -> usize {
    CATEGORIES
        .iter()
        .position(|&c| c == cat)
        .unwrap_or_else(|| panic!("Unknown result category '{cat}'. Valid: {CATEGORIES:?}"))
}

fn ensure_test_suite(dir: Option<&str>) -> PathBuf {
    if let Some(d) = dir {
        let p = PathBuf::from(d);
        assert!(p.join("tests").exists(), "No tests/ directory found in {d}");
        return p;
    }
    // Check common locations
    let candidates = [
        PathBuf::from("JSON-Schema-Test-Suite"),
        PathBuf::from("/tmp/JSON-Schema-Test-Suite"),
    ];
    for c in &candidates {
        if c.join("tests").exists() {
            return c.clone();
        }
    }
    // Clone it
    let tmp = &candidates[1];
    eprintln!("Cloning JSON-Schema-Test-Suite...");
    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "https://github.com/json-schema-org/JSON-Schema-Test-Suite",
            tmp.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run git clone");
    assert!(status.success(), "git clone failed");
    tmp.clone()
}

/// Nested results: file → group → test → category
type Results = BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>>;

/// Run all tests in a file, recording per-test-case results.
fn run_test_file(path: &Path, prefix: &str, results: &mut Results) {
    let filename = path.file_name().unwrap().to_str().unwrap();
    let stem = filename.strip_suffix(".json").unwrap_or(filename);
    let file_key = format!("{prefix}{stem}");

    let content = std::fs::read_to_string(path).unwrap();
    let groups: Vec<TestGroup> = serde_json::from_str(&content).unwrap();

    let file_results = results.entry(file_key).or_default();

    for group in &groups {
        let group_results = file_results.entry(group.description.clone()).or_default();

        let lark_grammar = format!(
            r#"start: %json {}"#,
            serde_json::to_string(&group.schema).unwrap()
        );
        let parser_result = make_parser(&lark_grammar, true);

        match parser_result {
            Err(e) => {
                let msg = format!("{e}");
                let is_unimplemented =
                    msg.contains("Unimplemented keys") || msg.contains("not supported");
                let has_valid = group.tests.iter().any(|t| t.valid);

                for test in &group.tests {
                    let category = if is_unimplemented {
                        "skip_compile"
                    } else if has_valid {
                        "compile_error_valid"
                    } else {
                        "compile_error_all_invalid"
                    };
                    group_results.insert(test.description.clone(), category.to_string());
                }
            }
            Ok(_) => {
                for test in &group.tests {
                    let result = std::panic::catch_unwind(|| {
                        json_schema_check(&group.schema, &test.data, test.valid);
                    });
                    let category = match result {
                        Ok(()) => "pass",
                        Err(_) => {
                            if test.valid {
                                "false_negative"
                            } else {
                                "false_positive"
                            }
                        }
                    };
                    group_results.insert(test.description.clone(), category.to_string());
                }
            }
        }
    }
}

fn flatten(results: &Results) -> BTreeMap<String, String> {
    let mut flat = BTreeMap::new();
    for (file, groups) in results {
        for (group, tests) in groups {
            for (test, cat) in tests {
                flat.insert(format!("{file} / {group} / {test}"), cat.clone());
            }
        }
    }
    flat
}

fn compare_results(
    current: &Results,
    baseline: &Results,
) -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let cur_flat = flatten(current);
    let base_flat = flatten(baseline);
    let mut regressions = Vec::new();
    let mut improvements = Vec::new();
    let mut new_tests = Vec::new();
    let mut missing_tests = Vec::new();

    for (test_id, cur_cat) in &cur_flat {
        match base_flat.get(test_id) {
            Some(base_cat) => {
                if cur_cat != base_cat {
                    let cur_bad = category_badness(cur_cat);
                    let base_bad = category_badness(base_cat);
                    if cur_bad > base_bad {
                        regressions.push(format!("{test_id}: {base_cat} → {cur_cat}"));
                    } else {
                        improvements.push(format!("{test_id}: {base_cat} → {cur_cat}"));
                    }
                }
            }
            None => {
                new_tests.push(format!("{test_id}: {cur_cat}"));
            }
        }
    }

    for (test_id, base_cat) in &base_flat {
        if !cur_flat.contains_key(test_id) {
            missing_tests.push(format!("{test_id}: was {base_cat}"));
        }
    }

    (regressions, improvements, new_tests, missing_tests)
}

fn print_category_summary(draft: &str, results: &Results) {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for cat in CATEGORIES {
        counts.insert(cat, 0);
    }
    let mut total = 0;
    for groups in results.values() {
        for tests in groups.values() {
            for cat in tests.values() {
                *counts.entry(cat.as_str()).or_insert(0) += 1;
                total += 1;
            }
        }
    }
    eprintln!("\n=== JSON Schema Test Suite ({draft}) ===");
    eprintln!("Total: {total}");
    for (cat, count) in &counts {
        if *count > 0 {
            eprintln!("  {cat:30} {count}");
        }
    }
}

/// Top-level baseline: draft → file → group → test → category
type Baseline = BTreeMap<String, Results>;

fn run_draft(suite_root: &Path, draft: &str) -> Result<Results> {
    let suite_dir = suite_root.join("tests").join(draft);
    if !suite_dir.exists() {
        let available: Vec<String> = std::fs::read_dir(suite_root.join("tests"))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        bail!(
            "Draft '{draft}' not found. Available: {}",
            available.join(", ")
        );
    }
    let mut results: Results = BTreeMap::new();

    // Core tests
    let mut files: Vec<PathBuf> = std::fs::read_dir(&suite_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    for file in &files {
        run_test_file(file, "", &mut results);
    }

    // Optional format tests
    let format_dir = suite_dir.join("optional").join("format");
    if format_dir.exists() {
        let mut format_files: Vec<PathBuf> = std::fs::read_dir(&format_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
            .collect();
        format_files.sort();
        for file in &format_files {
            run_test_file(file, "optional/format/", &mut results);
        }
    }

    print_category_summary(draft, &results);
    Ok(results)
}

/// Check one draft's results against its baseline. Returns true if there are changes.
fn check_draft(draft: &str, current: &Results, baseline: &Results, update: bool) -> Result<bool> {
    let (regressions, improvements, new_tests, missing_tests) = compare_results(current, baseline);

    let has_changes = !regressions.is_empty()
        || !improvements.is_empty()
        || !new_tests.is_empty()
        || !missing_tests.is_empty();

    if has_changes {
        if !regressions.is_empty() {
            eprintln!("\n--- {draft}: REGRESSIONS ({}) ---", regressions.len());
            for r in &regressions {
                eprintln!("  {r}");
            }
        }
        if !improvements.is_empty() {
            eprintln!("\n--- {draft}: IMPROVEMENTS ({}) ---", improvements.len());
            for i in &improvements {
                eprintln!("  {i}");
            }
        }
        if !new_tests.is_empty() {
            eprintln!("\n--- {draft}: NEW TESTS ({}) ---", new_tests.len());
            for n in &new_tests {
                eprintln!("  {n}");
            }
        }
        if !missing_tests.is_empty() {
            eprintln!("\n--- {draft}: MISSING TESTS ({}) ---", missing_tests.len());
            for m in &missing_tests {
                eprintln!("  {m}");
            }
        }
        if !update {
            eprintln!(
                "\n{draft}: {} regressions, {} improvements, {} new, {} missing",
                regressions.len(),
                improvements.len(),
                new_tests.len(),
                missing_tests.len()
            );
        }
    } else {
        eprintln!("\n{draft}: all results match baseline. ✓");
    }

    Ok(has_changes)
}

/// Run the JSON Schema Test Suite against our compiler with ratchet-based regression detection.
#[derive(Parser)]
struct Args {
    /// Baseline file for ratchet comparison
    #[arg(long)]
    expected: Option<String>,

    /// Draft(s) to run (e.g. draft2020-12, draft7). Repeatable.
    /// Without this flag: runs all drafts in the baseline, or draft2020-12 if no baseline.
    #[arg(long)]
    draft: Vec<String>,

    /// Overwrite the baseline file with current results
    #[arg(long)]
    update: bool,

    /// Enable verbose parser logging (off by default)
    #[arg(long)]
    verbose: bool,

    /// Path to JSON-Schema-Test-Suite checkout
    suite_dir: Option<String>,
}

fn main() -> Result<()> {
    // Suppress panic messages from catch_unwind (expected for false_negative/false_positive cases)
    std::panic::set_hook(Box::new(|_| {}));

    let args = Args::parse();
    llg_test_utils::set_quiet_mode(!args.verbose);
    let mut drafts_arg = args.draft;
    let update = args.update;

    let suite_root = ensure_test_suite(args.suite_dir.as_deref());

    // No baseline — run specified drafts (or default), dump to stdout
    let Some(expected) = args.expected else {
        if drafts_arg.is_empty() {
            drafts_arg.push("draft2020-12".to_string());
        }
        let mut by_draft = BTreeMap::new();
        for d in &drafts_arg {
            let results = run_draft(&suite_root, d)?;
            by_draft.insert(d.clone(), results);
        }
        let json = serde_json::to_string_pretty(&by_draft)?;
        println!("{json}");
        return Ok(());
    };

    let baseline_file = PathBuf::from(&expected);

    // Load baseline (if it exists) and determine which drafts to run
    let mut baseline: Baseline = if baseline_file.exists() {
        let content = std::fs::read_to_string(&baseline_file)?;
        serde_json::from_str(&content)?
    } else {
        BTreeMap::new()
    };

    let drafts: Vec<String> = if !drafts_arg.is_empty() {
        drafts_arg
    } else if !baseline.is_empty() {
        baseline.keys().cloned().collect()
    } else {
        vec!["draft2020-12".to_string()]
    };

    // Run each draft and compare
    let mut any_changes = false;
    for d in &drafts {
        let results = run_draft(&suite_root, d)?;
        if let Some(base) = baseline.get(d.as_str()) {
            let changed = check_draft(d, &results, base, update)?;
            if changed {
                any_changes = true;
                if update {
                    baseline.insert(d.clone(), results);
                }
            }
        } else {
            // New draft — no baseline yet
            eprintln!("\n{d}: new draft (no baseline entry)");
            any_changes = true;
            if update {
                baseline.insert(d.clone(), results);
            }
        }
    }

    if update && any_changes {
        let json = serde_json::to_string_pretty(&baseline)?;
        std::fs::write(&baseline_file, &json)?;
        eprintln!("\nBaseline updated: {expected}");
    } else if any_changes {
        bail!("Baseline mismatch. Run with --update to update the baseline.");
    }

    Ok(())
}
