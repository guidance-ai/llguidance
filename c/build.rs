use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=llguidance.h");

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = PathBuf::from(&crate_dir).join("llguidance.h");

    // Place header alongside compiled library in target/{profile}/
    if let Ok(out_dir) = env::var("OUT_DIR") {
        let dest = PathBuf::from(out_dir)
            .ancestors()
            .nth(3)
            .expect(
                "unexpected OUT_DIR structure, expected target/{profile}/build/{crate}-{hash}/out",
            )
            .join("llguidance.h");
        std::fs::copy(&header_path, &dest)
            .unwrap_or_else(|e| panic!("Failed to copy llguidance.h to {}: {e}", dest.display()));
    }
}
