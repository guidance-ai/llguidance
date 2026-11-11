//! Generator for `c/llguidance.h`.
//!
//! Requires being run with `+nightly`, see:
//! <https://github.com/mozilla/cbindgen/issues/183>
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let c_dir = crate_dir.parent().unwrap().join("c");
    let config_path = PathBuf::from(&c_dir).join("cbindgen.toml");
    let header_path = PathBuf::from(&c_dir).join("llguidance.h");

    let config = cbindgen::Config::from_file(config_path)
        .unwrap_or_else(|e| panic!("failed to read cbindgen.toml: {e}"));

    let bindings = cbindgen::Builder::new()
        .with_crate_and_name(&c_dir, "llguidance")
        .with_config(config)
        .generate()
        .unwrap_or_else(|e| panic!("failed to generate C bindings: {e}"));

    let mut raw = Vec::new();
    bindings.write(&mut raw);
    let header = String::from_utf8(raw).unwrap();

    // Filter out Rust-specific doc comment lines
    let filtered: String = header
        .lines()
        .filter(|line| !line.contains("* # Safety"))
        .filter(|line| !line.contains("* This function should only be called from C code"))
        .map(|line| format!("{line}\n"))
        .collect();

    let existing = std::fs::read_to_string(&header_path).unwrap_or_default();
    if existing != filtered {
        std::fs::write(&header_path, &filtered)
            .unwrap_or_else(|e| panic!("failed to write llguidance.h: {e}"));
        println!("updated llguidance.h");
    } else {
        println!("generated llguidance.h was already up to date, nothing to do");
    }
}
