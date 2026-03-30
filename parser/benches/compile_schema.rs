use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use llguidance::{api::ParserLimits, GrammarBuilder, JsonCompileOptions};

fn fhir_schema_path() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("bench-data");
    dir.join("fhir.schema.json")
}

fn ensure_fhir_schema() -> String {
    let path = fhir_schema_path();
    if path.exists() {
        return fs::read_to_string(&path).expect("Failed to read cached FHIR schema");
    }

    // Download
    let url = "https://hl7.org/fhir/R4/fhir.schema.json";
    eprintln!("Downloading FHIR schema from {url} ...");
    let resp = std::process::Command::new("curl")
        .args(["-sL", url])
        .output()
        .expect("Failed to run curl");
    assert!(
        resp.status.success(),
        "curl failed: {}",
        String::from_utf8_lossy(&resp.stderr)
    );
    let body = String::from_utf8(resp.stdout).expect("FHIR schema is not valid UTF-8");
    // Validate it's JSON
    serde_json::from_str::<serde_json::Value>(&body).expect("Downloaded content is not valid JSON");

    fs::create_dir_all(path.parent().unwrap()).ok();
    fs::write(&path, &body).expect("Failed to cache FHIR schema");
    eprintln!("Cached to {}", path.display());
    body
}

const BLOG_SCHEMA_JSON: &str = include_str!("../../sample_parser/data/blog.schema.json");

fn compile_schema(schema: serde_json::Value, opts: JsonCompileOptions) {
    let builder = GrammarBuilder::new(None, ParserLimits::default());
    opts.json_to_llg_with_overrides(builder, schema).unwrap();
}

/// Benchmark schema → grammar compilation time.
///
/// Usage:
///   # Save baseline before rewrite:
///   cargo bench --bench compile_schema -- --save-baseline pre-rewrite
///
///   # Compare against baseline after rewrite:
///   cargo bench --bench compile_schema -- --baseline pre-rewrite
fn bench_schema_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("schema_compilation");

    // Small schema: blog (1.3KB)
    let blog_value: serde_json::Value = serde_json::from_str(BLOG_SCHEMA_JSON).unwrap();
    group.bench_with_input(
        BenchmarkId::new("compile", "blog_1kb"),
        &blog_value,
        |b, schema| {
            b.iter(|| {
                compile_schema(schema.clone(), JsonCompileOptions::default());
                black_box(())
            })
        },
    );

    // Large schema: FHIR (~4MB)
    let fhir_text = ensure_fhir_schema();
    let fhir_value: serde_json::Value = serde_json::from_str(&fhir_text).unwrap();
    let fhir_size_kb = fhir_text.len() / 1024;

    group.bench_with_input(
        BenchmarkId::new("compile", format!("fhir_{fhir_size_kb}kb")),
        &fhir_value,
        |b, schema| {
            b.iter(|| {
                let opts = JsonCompileOptions {
                    lenient: true, // FHIR uses some unsupported keywords
                    ..Default::default()
                };
                compile_schema(schema.clone(), opts);
                black_box(())
            })
        },
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(10));
    targets = bench_schema_compilation
}
criterion_main!(benches);
