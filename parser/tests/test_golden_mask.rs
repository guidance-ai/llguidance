//! Phase 0.3: Baseline mask-differential harness.
//!
//! Runs the EXISTING Earley `compute_mask_immut` on a corpus of (grammar,
//! token sequence) pairs and records the masks. This is the P10 oracle
//! (independent register: the legacy engine, not the PDA). In Phase 3, the
//! PDA `pda_mask` is run on the same corpus and compared against these
//! golden records.
//!
//! The corpus uses the single-byte tokenizer (ApproximateTokEnv) so the token
//! IDs are 0-255 (one per byte). The masks are 256-bit bitsets (SimpleVob).

#[cfg(feature = "dpda")]
mod tests {
    use llguidance::api::TopLevelGrammar;
    use llguidance::earley::SlicedBiasComputer;
    use llguidance::toktrie::{ApproximateTokEnv, InferenceCapabilities, SimpleVob};
    use llguidance::{Matcher, ParserFactory};

    fn factory() -> ParserFactory {
        let env = ApproximateTokEnv::single_byte_env();
        ParserFactory::new(
            &env,
            InferenceCapabilities {
                ff_tokens: true,
                backtrack: false,
                conditional_ff_tokens: false,
                fork: false,
            },
            &SlicedBiasComputer::general_slices(),
        )
        .expect("the factory")
    }

    /// Record a SimpleVob as a sorted space-separated hex list of set token IDs.
    fn vob_to_hex(vob: &SimpleVob) -> String {
        let mut allowed = Vec::new();
        vob.iter_set_entries(|idx| {
            allowed.push(idx);
        });
        allowed.sort();
        allowed.dedup();
        allowed
            .iter()
            .map(|&a| format!("{:02x}", a))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// A corpus of (grammar, token sequence) pairs. The token sequences are
    /// chosen to match the grammar's structure (the single-byte tokenizer
    /// maps each byte to a token ID 0-255).
    fn corpus() -> Vec<(TopLevelGrammar, Vec<u32>)> {
        let mut c = Vec::new();

        // Lark sequence: "a" "b" "c" (the bytes 97, 98, 99).
        let g1 = TopLevelGrammar::from_lark(r#"start: "a" "b" "c""#.to_string());
        c.push((g1, vec![97, 98, 99]));

        // Lark choice: "foo" | "bar" (the bytes for "foo": 102, 111, 111).
        let g2 = TopLevelGrammar::from_lark(r#"start: "foo" | "bar""#.to_string());
        c.push((g2, vec![102, 111, 111]));

        // Lark nested: "a" ("b" "c")* (the bytes: 97, 98, 99, 98, 99).
        let g3 = TopLevelGrammar::from_lark(
            r#"start: "a" ("b" "c")*"#.to_string(),
        );
        c.push((g3, vec![97, 98, 99, 98, 99]));

        c
    }

    /// Run the legacy `compute_mask_immut` on the corpus and record the masks.
    /// This is the P10 golden: the legacy engine's masks are the ground truth
    /// that the PDA must match (or exceed, in the frozen-mask bug case).
    #[test]
    fn golden_mask_corpus() {
        let f = factory();
        let corpus = corpus();
        let mut golden = String::new();

        for (i, (grm, tokens)) in corpus.iter().enumerate() {
            let parser = f.create_parser(grm.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();

            golden.push_str(&format!("--- corpus[{}] ({} tokens) ---\n", i, tokens.len()));

            for (step, &tok) in tokens.iter().enumerate() {
                let mask = m.compute_mask_immut().expect("the mask");
                let mask_hex = vob_to_hex(&mask);
                golden.push_str(&format!("  step {}: {}\n", step, mask_hex));

                match m.consume_token(tok) {
                    Ok(_) => {}
                    Err(_) => {
                        golden.push_str(&format!(
                            "  step {}: CONSUME FAILED (token {} not allowed)\n",
                            step, tok
                        ));
                        break;
                    }
                }
            }
            golden.push_str("\n");
        }

        let golden_path = std::path::Path::new(".golden_mask_corpus.txt");
        std::fs::write(golden_path, &golden).expect("write golden file");
        eprintln!(
            "Golden mask corpus written to {:?} ({} bytes, {} entries)",
            golden_path,
            golden.len(),
            corpus.len()
        );

        // Verify determinism: re-run and check identical.
        let mut golden2 = String::new();
        for (i, (grm, tokens)) in corpus.iter().enumerate() {
            let parser = f.create_parser(grm.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            golden2.push_str(&format!("--- corpus[{}] ({} tokens) ---\n", i, tokens.len()));
            for (step, &tok) in tokens.iter().enumerate() {
                let mask = m.compute_mask_immut().expect("the mask");
                golden2.push_str(&format!("  step {}: {}\n", step, vob_to_hex(&mask)));
                if m.consume_token(tok).is_err() {
                    golden2.push_str(&format!("  step {}: CONSUME FAILED\n", step));
                    break;
                }
            }
            golden2.push_str("\n");
        }
        assert_eq!(
            golden, golden2,
            "the golden mask corpus must be deterministic (re-run identical)"
        );
    }

    /// Benchmark: measure the legacy `compute_mask_immut` time per token.
    /// This is the baseline that the PDA must beat (Phase 3).
    #[test]
    fn bench_legacy_mask_time() {
        let f = factory();
        let corpus = corpus();
        let mut total_tokens = 0usize;
        let t0 = std::time::Instant::now();

        for (grm, tokens) in corpus.iter() {
            let parser = f.create_parser(grm.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            for &tok in tokens.iter() {
                let _ = m.compute_mask_immut().expect("the mask");
                if m.consume_token(tok).is_ok() {
                    total_tokens += 1;
                } else {
                    break;
                }
            }
        }

        let elapsed = t0.elapsed();
        let us_per_token = elapsed.as_micros() as f64 / total_tokens as f64;
        eprintln!(
            "Legacy mask bench: {} tokens in {} ms ({} us/token)",
            total_tokens,
            elapsed.as_millis(),
            us_per_token
        );
        assert!(us_per_token > 0.0, "the legacy mask time must be positive");
    }
}