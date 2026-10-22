use std::{hint::black_box, time::Instant};

use anyhow::{bail, ensure, Result};
use completion_bench::{case, replay, stats, template, CASE_NAMES};
use serde_json::json;

fn main() -> Result<()> {
    ensure!(!cfg!(debug_assertions), "run with --release");
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.as_slice() == ["--list"] {
        for name in CASE_NAMES {
            println!("{name}");
        }
        return Ok(());
    }
    if args.len() != 3 {
        bail!("usage: completion_bench --list | CASE ITERATIONS SAMPLES");
    }
    let case = case(&args[0])?;
    let iterations: usize = args[1].parse()?;
    let samples: usize = args[2].parse()?;
    ensure!(iterations > 0 && samples > 0, "counts must be positive");
    let template_start = Instant::now();
    let base = template(&case)?;
    let template_ns = template_start.elapsed().as_nanos();
    // Warm the lexer and verify the entire accepted sequence before timing.
    let expected_hash = replay(&mut base.clone(), &case.input)?;
    let mut measurements = Vec::new();
    for sample in 0..samples {
        let mut elapsed_ns = 0;
        let mut clone_ns = 0;
        let mut last_stats = json!(null);
        for _ in 0..iterations {
            let start = Instant::now();
            let mut parser = base.clone();
            clone_ns += start.elapsed().as_nanos();
            let start = Instant::now();
            let hash = black_box(replay(black_box(&mut parser), black_box(&case.input))?);
            elapsed_ns += start.elapsed().as_nanos();
            ensure!(hash == expected_hash, "mask checksum changed");
            last_stats = stats(&parser);
        }
        measurements.push(json!({
            "sample": sample, "iterations": iterations, "replay_elapsed_ns": elapsed_ns,
            "clone_elapsed_ns": clone_ns, "last_replay_stats": last_stats,
        }));
    }
    println!(
        "{}",
        json!({
            "case": case.name, "input_bytes": case.input.len(),
            "completion_stats_enabled": cfg!(feature = "completion-stats"),
            "max_items_in_row": case.row_limit, "step_max_items": case.step_limit,
            "template_elapsed_ns": template_ns, "mask_checksum": format!("{expected_hash:016x}"),
            "accepting": true, "samples": measurements,
        })
    );
    Ok(())
}
