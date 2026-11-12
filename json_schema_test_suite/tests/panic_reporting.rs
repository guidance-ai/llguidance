use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct TestSuite {
    root: PathBuf,
}

impl TestSuite {
    fn with_test_file(contents: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "llguidance-json-schema-panic-reporting-{}-{unique}",
            std::process::id()
        ));
        let draft_dir = root.join("tests").join("draft2020-12");
        std::fs::create_dir_all(&draft_dir).unwrap();
        std::fs::write(draft_dir.join("fixture.json"), contents).unwrap();
        Self { root }
    }

    fn run(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_json_schema_test_suite"))
            .args(["--draft", "draft2020-12"])
            .arg(self.path())
            .output()
            .unwrap()
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for TestSuite {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

#[test]
fn unexpected_panics_are_reported() {
    let suite = TestSuite::with_test_file("not valid JSON");
    let output = suite.run();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success());
    assert!(
        stderr.contains("panicked at"),
        "unexpected panic diagnostic was suppressed; stderr: {stderr:?}"
    );
}

#[test]
fn expected_mismatches_remain_quiet_and_classified() {
    let suite = TestSuite::with_test_file(
        r#"[
            {
                "description": "intentional mismatch",
                "schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "integer"
                },
                "tests": [
                    {
                        "description": "deliberately mislabeled valid",
                        "data": "not an integer",
                        "valid": true
                    }
                ]
            }
        ]"#,
    );
    let output = suite.run();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(output.status.success(), "stderr: {stderr}");
    assert!(stdout.contains(r#""deliberately mislabeled valid": "false_negative""#));
    assert!(
        !stderr.contains("panicked at"),
        "expected mismatch emitted a panic diagnostic: {stderr}"
    );
}
