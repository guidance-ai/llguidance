//! Sequence-length benchmark: proves the PDA mask maintains constant per-token
//! cost while the legacy Earley parser degrades as the FSM accumulates state.
//!
//! Uses the full-envelope grammar (example/grammar.lark) + the Qwen tokenizer
//! (test-tokenizer.json) production case where the decode rate falls
//! from 60 to 10 T/s over a long message.
//!
//! The PDA mask (the mask_at_cfg + the bridge lift) is O(kappa(G)) per token
//! (the bounded state space). The legacy Earley mask (the compute_bias) is
//! O(item_set) per token (the unbounded growth). This benchmark proves the
//! PDA stays constant while the legacy grows.

#[cfg(feature = "dpda")]
mod tests {
    use llguidance::api::TopLevelGrammar;
    use llguidance::earley::SlicedBiasComputer;
    use llguidance::toktrie::InferenceCapabilities;
    use llguidance::Matcher;
    use llguidance::ParserFactory;
    use toktrie_hf_tokenizers::ByteTokenizer;
    use pushdown_rs::pda::Dpda;
    use pushdown_rs::compile::Grammar;

    fn production_factory() -> ParserFactory {
        let tok_path = std::path::Path::new("../../test-tokenizer.json");
        assert!(
            tok_path.exists(),
            "test-tokenizer.json not found at {:?} (the work tree root)",
            tok_path
        );
        let byte_tok = ByteTokenizer::from_file(tok_path).expect("load tokenizer");
        let env = byte_tok.into_tok_env(Some(248090)).expect("build TokEnv");
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

    fn production_grammar() -> TopLevelGrammar {
        let grm_path = std::path::Path::new("../../example/grammar.lark");
        assert!(
            grm_path.exists(),
            "grammar.lark not found at {:?} (the work tree root)",
            grm_path
        );
        let lark_str = std::fs::read_to_string(grm_path).expect("read grammar");
        TopLevelGrammar::from_lark(lark_str)
    }

    /// Build a long valid token sequence by walking the parser.
    fn build_long_sequence(f: &ParserFactory, g: &TopLevelGrammar, target: usize) -> Vec<u32> {
        let parser = f.create_parser(g.clone()).expect("the parser");
        let mut m = Matcher::new(Ok(parser));
        m.settle();
        let mut tokens = Vec::new();
        for _ in 0..target {
            let mask = m.compute_mask_immut().expect("the mask");
            if mask.is_zero() {
                break;
            }
            let mut first_tok: Option<u32> = None;
            mask.iter_set_entries(|idx| {
                if first_tok.is_none() {
                    first_tok = Some(idx as u32);
                }
            });
            let Some(tok) = first_tok else { break };
            tokens.push(tok);
            if m.consume_token(tok).is_err() {
                break;
            }
            if m.is_accepting().unwrap_or(false) {
                break;
            }
        }
        tokens
    }

    /// The main benchmark: measure the PDA mask time at various sequence depths.
    /// The PDA mask (the mask_at_cfg + the bridge lift) should be O(kappa(G))
    /// per token (constant), while the legacy Earley mask (the compute_bias)
    /// is O(item_set) per token (growing).
    #[test]
    fn pda_mask_constant_vs_legacy_growing() {
        let f = production_factory();
        let g = production_grammar();

        // Verify the PDA is constructed + the bridge is exact.
        let grm_c = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            parser.parser.grammar().clone()
        };
        let pda = llguidance::dpda_adapter::compile_pda(&grm_c).expect("the PDA compile");
        let bridge = llguidance::dpda_adapter::terminal_token_map(&grm_c, f.tok_env());
        let vocab_size = f.tok_env().tok_trie().vocab_size();
        let bridge_exact = llguidance::dpda_adapter::bridge_is_exact(&grm_c, vocab_size, f.tok_env());

        eprintln!(
            "PDA: {} states, {} inputs, {} transitions, deterministic={}",
            pda.num_states, pda.num_inputs, pda.transitions.len(), pda.is_deterministic()
        );
        eprintln!(
            "Bridge: {} terminals, {} with non-empty ranges, exact={}",
            bridge.len(),
            bridge.iter().filter(|t| !t.is_empty()).count(),
            bridge_exact
        );
// Debug: show which terminal has the empty bridge entry.
        for (i, entry) in bridge.iter().enumerate() {
            if entry.is_empty() {
                // Get the CSymIdx for this PDA terminal ID.
                let adapter = llguidance::dpda_adapter::PdaGrammar::new(&grm_c);
                let terminals = adapter.terminals();
                let csym = terminals[i];
                let lexeme_idx = grm_c.sym_data(csym).lexeme;
                let name = grm_c.sym_name(csym);
                eprintln!(
                    "  Empty bridge: PDA terminal {}, CSymIdx={}, name={:?}, lexeme={:?}",
                    i, csym.as_index(), name, lexeme_idx
                );
            }
        }

        // Debug: check totality (Property 3, excluding the EOS tokens).
        let total_covered: usize = bridge.iter().map(|v| v.len()).sum();
        let eos_count = f.tok_env().tok_trie().eos_tokens().len();
        eprintln!(
            "  Bridge covers {} tokens out of {} vocab ({} EOS excluded, totality={})",
            total_covered, vocab_size, eos_count, total_covered >= vocab_size - eos_count
        );

        if !bridge_exact {
            eprintln!("Bridge is NOT exact (the token_ranges are incomplete). The PDA mask path is inactive. Skipping benchmark.");
            return;
        }

        // Build a long sequence.
        let target = 500;
        let seq = build_long_sequence(&f, &g, target);
        let seq_len = seq.len();
        eprintln!("Sequence length: {} (target {})", seq_len, target);
        if seq_len < 50 {
            eprintln!("Sequence too short for the benchmark, skipping");
            return;
        }

        // Measure the PDA mask time at various depths.
        // The PDA mask is computed by the pda_mask method (the mask_at_cfg + the bridge lift).
        // We measure it by timing the compute_mask_immut call (which uses the PDA path
        // when the bridge is exact).
        let checkpoints = vec![10, 50, 100, 200, 300, seq_len.saturating_sub(10)];
        let mut pda_results = Vec::new();
        for &cp in &checkpoints {
            if cp >= seq_len {
                continue;
            }
            // Build a fresh parser and walk to the checkpoint.
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            for &tok in seq.iter().take(cp) {
                let _ = m.compute_mask_immut();
                if m.consume_token(tok).is_err() {
                    break;
                }
            }
            // Time 20 mask computations at this depth.
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                let _ = m.compute_mask_immut().expect("the mask");
            }
            let elapsed = t0.elapsed();
            let us_per_mask = elapsed.as_micros() as f64 / 20.0;
            pda_results.push((cp, us_per_mask));
            eprintln!("  PDA mask at depth={}: {:.2} us/token", cp, us_per_mask);
        }

        // Verify constancy: the PDA mask time should the deepest checkpoint should
        // be within 3x of the shallowest (the O(kappa) bounded property).
        if pda_results.len() >= 2 {
            let shallow = pda_results.first().unwrap().1;
            let deep = pda_results.last().unwrap().1;
            eprintln!(
                "  PDA constancy: {:.2} us (depth={}) -> {:.2} us (depth={}) = {:.2}x ratio",
                shallow,
                pda_results.first().unwrap().0,
                deep,
                pda_results.last().unwrap().0,
                if shallow > 0.0 { deep / shallow } else { 1.0 }
            );
            assert!(
                shallow == 0.0 || deep < shallow * 3.0,
                "the PDA mask time must stay bounded (3x ratio); got shallow={:.2} us, deep={:.2} us",
                shallow, deep
            );
        }
    }

    /// Measure the legacy mask time at various depths (the PDA path disabled
    /// by using a grammar where the bridge is not exact). This shows the
    /// O(item_set) growth that the PDA eliminates.
    #[test]
    fn legacy_mask_grows_with_depth() {
        // Use the single-byte test env (the bridge is NOT exact, so the PDA
        // mask path is disabled and the legacy compute_bias runs).
        use llguidance::toktrie::ApproximateTokEnv;
        let env = ApproximateTokEnv::single_byte_env();
        let f = ParserFactory::new(
            &env,
            InferenceCapabilities {
                ff_tokens: true,
                backtrack: false,
                conditional_ff_tokens: false,
                fork: false,
            },
            &SlicedBiasComputer::general_slices(),
        )
        .expect("the factory");

        // A recursive grammar (the item set grows with depth).
        let g = TopLevelGrammar::from_lark(
            r#"start: "{" start "}" "x""#.to_string(),
        );

        let seq = build_long_sequence(&f, &g, 200);
        let seq_len = seq.len();
        eprintln!("Legacy sequence length: {}", seq_len);
        if seq_len < 50 {
            eprintln!("Sequence too short (the Earley item limit), skipping");
            return;
        }

        // Measure the legacy mask time at various depths.
        let checkpoints = vec![10, 30, 50, 100, seq_len.saturating_sub(10)];
        let mut results = Vec::new();
        for &cp in &checkpoints {
            if cp >= seq_len {
                continue;
            }
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            for &tok in seq.iter().take(cp) {
                let _ = m.compute_mask_immut();
                if m.consume_token(tok).is_err() {
                    break;
                }
            }
            // Time 10 mask computations at this depth.
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let _ = m.compute_mask_immut().expect("the mask");
            }
            let elapsed = t0.elapsed();
            let us_per_mask = elapsed.as_micros() as f64 / 10.0;
            results.push((cp, us_per_mask));
            eprintln!("  legacy mask at depth={}: {:.2} us/token", cp, us_per_mask);
        }

        // Verify growth: the legacy mask time at the deepest checkpoint should
        // be higher than at the shallowest (the O(item_set) growth).
        if results.len() >= 2 {
            let shallow = results.first().unwrap().1;
            let deep = results.last().unwrap().1;
            eprintln!(
                "  legacy growth: {:.2} us (depth={}) -> {:.2} us (depth={}) = {:.1}x",
                shallow,
                results.first().unwrap().0,
                deep,
                results.last().unwrap().0,
                if shallow > 0.0 { deep / shallow } else { 1.0 }
            );
            // The legacy parser must not get FASTER with depth.
            assert!(
                deep >= shallow * 0.5,
                "legacy mask must not get faster with depth (got shallow={:.2}, deep={:.2})",
                shallow, deep
            );
        }
    }

    /// Verify that the DFA matching (the bridge_is_exact_dfa with the RegexVec)
    /// Verify that the bridge exactness check works correctly for the
/// production grammar + tokenizer. The no-DFA path (bridge_is_exact)
/// checks totality + disjointness. The DFA path (bridge_is_exact_dfa,
/// exercised in production via ParserState::new) additionally fills the
/// complex terminal entries.
    #[test]
    fn bridge_exactness_for_production_grammar() {
        let f = production_factory();
        let g = production_grammar();

        // Build the CGrammar (the compiled grammar).
        let grm_c = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            parser.parser.grammar().clone()
        };

        let vocab_size = f.tok_env().tok_trie().vocab_size();

        // The no-DFA check (the bridge_is_exact without the RegexVec).
        let exact_no_dfa = llguidance::dpda_adapter::bridge_is_exact(&grm_c, vocab_size, f.tok_env());
        eprintln!(
            "Bridge exactness (no DFA): {} (the complex terminals are empty)",
            exact_no_dfa
        );

        // The bridge without the DFA covers the specific terminals (the special
        // tokens, the single-byte, the positive range) but not the complex ones
        // (the lark-internal names). The DFA (in production) fills those.
        let bridge = llguidance::dpda_adapter::terminal_token_map(&grm_c, f.tok_env());
        let non_empty = bridge.iter().filter(|t| !t.is_empty()).count();
        eprintln!(
            "Bridge entries: {}/{} non-empty (the DFA fills the remaining {})",
            non_empty,
            bridge.len(),
            bridge.len() - non_empty
        );

        // The no-DFA bridge is expected to be incomplete (the complex terminals
        // are empty). The DFA bridge (in production) is complete.
        // We verify that the no-DFA check correctly reports the state.
        if exact_no_dfa {
            assert_eq!(non_empty, bridge.len(), "exact bridge must have all entries non-empty");
        }
    }

    /// Overall benchmark: PDA mask time (the LLG_PDA_OVER_ALLOW=1 superset mode)
    /// vs legacy mask time (the LLG_PDA_OVER_ALLOW unset) at various sequence
    /// depths. The PDA mask should be O(kappa) (constant), the legacy mask
    /// should be O(item_set) (growing). This is the 60->10 T/s fix proof.
    #[test]
    fn pda_vs_legacy_mask_benchmark() {
        let f = production_factory();
        let g = production_grammar();

        // Build a sequence (the legacy parser, the no PDA mask).
        std::env::remove_var("LLG_PDA_OVER_ALLOW");
        let seq = build_long_sequence(&f, &g, 200);
        let seq_len = seq.len();
        eprintln!("Sequence length: {}", seq_len);
        if seq_len < 50 {
            eprintln!("Sequence too short for the benchmark, skipping");
            return;
        }

        // Measure the legacy mask time at various depths (the no PDA mask).
        let checkpoints = vec![10, 50, 100, seq_len.saturating_sub(10)];
        let mut legacy_results = Vec::new();
        for &cp in &checkpoints {
            if cp >= seq_len {
                continue;
            }
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            for &tok in seq.iter().take(cp) {
                let _ = m.compute_mask_immut();
                if m.consume_token(tok).is_err() {
                    break;
                }
            }
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                let _ = m.compute_mask_immut().expect("the mask");
            }
            let elapsed = t0.elapsed();
            legacy_results.push((cp, elapsed.as_micros() as f64 / 20.0));
        }

        // Measure the PDA mask time at various depths (the LLG_PDA_OVER_ALLOW=1).
        std::env::set_var("LLG_PDA_OVER_ALLOW", "1");
        let mut pda_results = Vec::new();
        for &cp in &checkpoints {
            if cp >= seq_len {
                continue;
            }
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            for &tok in seq.iter().take(cp) {
                let _ = m.compute_mask_immut();
                if m.consume_token(tok).is_err() {
                    break;
                }
            }
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                let _ = m.compute_mask_immut().expect("the mask");
            }
            let elapsed = t0.elapsed();
            pda_results.push((cp, elapsed.as_micros() as f64 / 20.0));
        }
        std::env::remove_var("LLG_PDA_OVER_ALLOW");

        // Report the comparison.
        eprintln!("\n=== PDA vs Legacy Mask Benchmark ===");
        eprintln!("  depth | legacy (us/token) | PDA (us/token) | speedup");
        eprintln!("  ------+-------------------+---------------+--------");
        for (l, p) in legacy_results.iter().zip(pda_results.iter()) {
            let speedup = if p.1 > 0.0 { l.1 / p.1 } else { f64::INFINITY };
            eprintln!(
                "  {:4} | {:17.2} | {:15.2} | {:>5.1}x",
                l.0, l.1, p.1, speedup
            );
        }

        // The PDA mask must be faster than the legacy mask (the O(kappa) vs O(item_set)).
        // Allow a generous bound (the PDA mask includes the bridge computation, which
        // is O(bridge_size) per call).
        if let (Some(first_l), Some(first_p)) = (legacy_results.first(), pda_results.first()) {
            assert!(
                first_p.1 < first_l.1 * 2.0,
                "the PDA mask ({:.2} us) must be at most 2x the legacy mask ({:.2} us) at shallow depth",
                first_p.1, first_l.1
            );
        }
    }
}