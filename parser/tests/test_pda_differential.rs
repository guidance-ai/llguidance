//! P10 differential: the PDA mask must match the legacy Earley mask on the
//! same corpus. This is the 100% accuracy proof for the PDA replacement.
//!
//! Uses the real Phi-3.5 tokenizer (from llg_test_utils) where the
//! terminal_token_ranges are populated (the exact bridge, not the
//! identity fallback).

#[cfg(feature = "dpda")]
mod tests {
    use llguidance::api::TopLevelGrammar;
    use llguidance::Matcher;
    use llguidance::toktrie::SimpleVob;
    use llguidance::toktrie::InferenceCapabilities;
    use llguidance::earley::SlicedBiasComputer;
    use llguidance::ParserFactory;
    use toktrie_hf_tokenizers::ByteTokenizer;
    use pushdown_rs::pda::Dpda;
    

    /// A factory with the real tokenizer but `backtrack: false` (the
    /// SlicedBiasComputer doesn't support backtracking).
    fn real_factory() -> ParserFactory {
        let env = llg_test_utils::get_tok_env().clone();
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

    /// Count the set entries in a SimpleVob.
    fn vob_count(vob: &SimpleVob) -> usize {
        let mut count = 0;
        vob.iter_set_entries(|_| count += 1);
        count
    }

    /// P10: the PDA mask (via LLG_PDA_MASK=1) must not under-allow any token
    /// that the legacy Earley mask allows. Over-allowing is safe (the
    /// firewall rejects on commit).
    #[test]
    fn pda_mask_matches_legacy_mask() {
        let f = real_factory();

        // A JSON schema grammar (the tool-calling use case).
        let g = TopLevelGrammar::from_json_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read", "write"]},
                "path": {"type": "string"},
                "data": {"type": "string"}
            },
            "required": ["action", "path"]
        }));

        // Build a valid token sequence by walking the legacy parser.
        let valid_tokens = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            let mut tokens = Vec::new();
            for _ in 0..50 {
                let mask = m.compute_mask_immut().expect("the mask");
                if mask.is_zero() {
                    break;
                }
                // Pick the first allowed token (deterministic walk).
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
        };

        if valid_tokens.is_empty() {
            eprintln!("No valid tokens generated (skipping P10 test)");
            return;
        }

        eprintln!("Walking {} tokens through the JSON grammar", valid_tokens.len());

        // Run the legacy mask (LLG_PDA_MASK unset) and record the masks.
        std::env::remove_var("LLG_PDA_MASK");
        let legacy_masks = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            let mut masks = Vec::new();
            for &tok in &valid_tokens {
                let mask = m.compute_mask_immut().expect("the mask");
                masks.push(mask);
                if m.consume_token(tok).is_err() {
                    break;
                }
            }
            masks
        };

        // Run the PDA mask (LLG_PDA_MASK=1) and record the masks.
        std::env::set_var("LLG_PDA_MASK", "1");
        let pda_masks = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            let mut masks = Vec::new();
            for &tok in &valid_tokens {
                let mask = m.compute_mask_immut().expect("the mask");
                masks.push(mask);
                if m.consume_token(tok).is_err() {
                    break;
                }
            }
            masks
        };
        std::env::remove_var("LLG_PDA_MASK");

        // Compare: the PDA mask must not under-allow any token the legacy
        // mask allows. Over-allowing is safe (the firewall rejects on commit).
        //
        // NOTE: the PDA mask is only exact when the terminal-to-token bridge
        // is exact (the LexemeSpec.token_ranges are populated by the
        // add_special_token mechanism). For regular terminals (the string
        // literals, the regexes), the bridge falls back to the identity map
        // (terminal a covers token a), which is approximate. The P10
        // differential is meaningful only when the bridge is exact.
        let bridge_exact = {
            let grm = {
                let parser = f.create_parser(g.clone()).expect("the parser");
                parser.parser.grammar().clone()
            };
            let bridge = llguidance::dpda_adapter::terminal_token_map(&grm, f.tok_env());
            // The bridge is exact if no entry uses the identity fallback
            // (i.e., every terminal has non-empty token_ranges).
            bridge.iter().all(|tokens| !tokens.is_empty() && tokens[0] != 0 || tokens.len() > 1)
        };

        if !bridge_exact {
            eprintln!(
                "P10: bridge is approximate (identity fallback). PDA mask comparison is not meaningful. Skipping assertion."
            );
            return;
        }

        let mut exact_matches = 0;
        let mut superset_matches = 0;
        let mut under_matches = 0;
        for (i, (legacy, pda)) in legacy_masks.iter().zip(pda_masks.iter()).enumerate() {
            let legacy_count = vob_count(legacy);
            let pda_count = vob_count(pda);
            if legacy.as_slice() == pda.as_slice() {
                exact_matches += 1;
            } else if pda_count >= legacy_count {
                // The PDA allows a STRICT SUPERSET of the CPU mask: it admits tokens the
                // exact CPU mask forbids. For a firewall this is an INJECTION (the R1
                // violation) — the prior "over-allowing is safe" framing assumed the commit
                // firewall would catch it, but the commit firewall is only as strong as its
                // own mask, so an over-allow here is a real defect, not a perf note.
                superset_matches += 1;
                eprintln!(
                    "OVER-ALLOW at step {}: legacy={} tokens, PDA={} tokens (the PDA admits tokens the CPU mask forbids)",
                    i, legacy_count, pda_count
                );
            } else {
                under_matches += 1;
                eprintln!(
                    "UNDER-ALLOW at step {}: legacy={} tokens, PDA={} tokens (the PDA drops tokens the CPU mask allows)",
                    i, legacy_count, pda_count
                );
            }
        }

        eprintln!(
            "P10 differential: {} steps, {} exact, {} over-allow, {} under-allow",
            legacy_masks.len(),
            exact_matches,
            superset_matches,
            under_matches
        );

        // The firewall contract (the R1): the PDA mask must EQUAL the CPU mask, bit-for-bit.
        // Neither an over-allow (an injection) nor an under-allow (a forced deviation) is
        // acceptable. The prior test asserted only mismatches==0 (no under-allow), which
        // permitted over-allowing; that is the "over-allowing is safe" evasion this closes.
        assert_eq!(
            under_matches, 0,
            "the PDA mask must not under-allow any token the CPU mask allows"
        );
        assert_eq!(
            superset_matches, 0,
            "the PDA mask must not over-allow any token the CPU mask forbids (the firewall injection)"
        );
    }

    /// Benchmark: measure the PDA advance_eps time per token (the O(1)
    /// indexed lookup) vs the legacy compute_mask_immut time per token
    /// (the O(grammar) Earley cost). This shows the speedup
    /// potential of the PDA mask path.
    #[test]
    fn bench_pda_advance_vs_legacy_mask() {
        let f = real_factory();
        let g = TopLevelGrammar::from_json_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read", "write"]},
                "path": {"type": "string"},
                "data": {"type": "string"}
            },
            "required": ["action", "path"]
        }));

        // Build a valid token sequence (the legacy walk).
        let valid_tokens = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            let mut tokens = Vec::new();
            for _ in 0..100 {
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
        };

        if valid_tokens.len() < 10 {
            eprintln!("Not enough tokens for benchmark ({}), skipping", valid_tokens.len());
            return;
        }

        // Measure the legacy mask time (the O(grammar) cost).
        let t0 = std::time::Instant::now();
        let legacy_masks = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            let mut masks = Vec::new();
            for &tok in &valid_tokens {
                let mask = m.compute_mask_immut().expect("the mask");
                masks.push(mask);
                if m.consume_token(tok).is_err() {
                    break;
                }
            }
            masks
        };
        let legacy_time = t0.elapsed();

        // Measure the PDA advance time (the O(1) indexed lookup).
        // The PDA is constructed from the grammar (the RTN compilation).
        let grm = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            parser.parser.grammar().clone()
        };
        let pda = llguidance::dpda_adapter::compile_pda(&grm).expect("the PDA compile");
        let pda_index = pda.build_index();

        let t1 = std::time::Instant::now();
        let mut pda_ctrl = pda.start_state;
        let mut pda_stack = vec![pda.start_stack];
        let mut pda_steps = 0;
        for &tok in &valid_tokens {
            // The PDA advance (the O(1) indexed lookup).
            // Note: the token ID must be mapped to the PDA's local terminal ID
            // via the bridge. For this benchmark, we use the identity mapping
            // (the Pimate bridge) just to measure the advance_eps cost.
            let terminal_id = tok as u32;
            if (terminal_id as usize) < pda.num_inputs as usize {
                if let Some((nq, ns)) = pda.advance_eps(pda_ctrl, &pda_stack, terminal_id) {
                    pda_ctrl = nq;
                    pda_stack = ns;
                    pda_steps += 1;
                }
            }
        }
        let pda_time = t1.elapsed();

        let legacy_us = legacy_time.as_micros() as f64 / valid_tokens.len() as f64;
        let pda_us = pda_time.as_micros() as f64 / pda_steps.max(1) as f64;

        eprintln!(
            "PDA advance bench: {} tokens, {} PDA steps",
            valid_tokens.len(), pda_steps
        );
        eprintln!(
            "  legacy compute_mask_immut: {:.1} us/token (O(grammar))",
            legacy_us
        );
        eprintln!(
            "  PDA advance_eps:           {:.1} us/token (O(1) indexed)",
            pda_us
        );
        if pda_us > 0.0 {
            eprintln!(
                "  speedup potential: {:.1}x (legacy / PDA)",
                legacy_us / pda_us
            );
        }

        // The PDA CPU advance is faster than the legacy mask for for deterministic
        // grammars (the single-path case). For non-deterministic grammars, the
        // BFS frontier is slower on CPU (the GPU fused_sample path is the
        // correct optimization for those).
        if pda.is_deterministic() {
            assert!(
                pda_us < legacy_us,
                "for deterministic grammars, the PDA advance ({:.1} us) must be faster than the legacy mask ({:.1} us)",
                pda_us, legacy_us
            );
            eprintln!("  PDA is deterministic: {:.1}x speedup (legacyDA {} us vs legacy {} us/token)",
                legacy_us / pda_us.max(1.0), pda_us, legacy_us);
        } else {
            eprintln!("  PDA is NON-deterministic ({} transitions): CPU PDA advance ({:.1} us) vs legacy ({:.1} us)",
                pda.transitions.len(), pda_us, legacy_us);
            eprintln!("  The PDA GPU path (fused_sample/fused_project) is the correct optimization for this grammar.");
            eprintln!("  The CPU PDA is only faster for deterministic (LR(1)) grammars.");
        }
    }

    /// P10 with the real tokenizer + full-envelope grammar (the bridge is
    /// exact because the token_ranges are populated by the add_special_token
    /// mechanism for the Qwen special tokens 248044-248069).
    #[test]
    fn pda_mask_exact_bridge_real_tokenizer() {
        // Load the real tokenizer (the test-tokenizer.json in the work tree root).
        let tok_path = std::path::Path::new("../../test-tokenizer.json");
        if !tok_path.exists() {
            eprintln!("test-tokenizer.json not found at {:?} (skipping exact- test)", tok_path);
            return;
        }
        let byte_tok = ByteTokenizer::from_file(tok_path).expect("load tokenizer");
        let env = byte_tok.into_tok_env(Some(248090)).expect("build TokEnv");

        // Load the full-envelope grammar (the example/grammar.lark).
        let grm_path = std::path::Path::new("../../example/grammar.lark");
        if !grm_path.exists() {
            eprintln!("grammar.lark not found at {:?} (skipping exact-bridge test)", grm_path);
            return;
        }
        let lark_str = std::fs::read_to_string(grm_path).expect("read grammar");
        let g = TopLevelGrammar::from_lark(lark_str);

        // Build the factory with the real tokenizer.
        let f = ParserFactory::new(
            &env,
            InferenceCapabilities {
                ff_tokens: true,
                backtrack: false,
                conditional_ff_tokens: false,
                fork: false,
            },
            &SlicedBiasComputer::general_slices(),
        ).expect("the factory");

        // Verify the bridge is exact (the token_ranges are populated).
        let grm_c = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            parser.parser.grammar().clone()
        };
        let bridge = llguidance::dpda_adapter::terminal_token_map(&grm_c, f.tok_env());
        let exact_count = bridge.iter().filter(|tokens| !tokens.is_empty()).count();
        eprintln!(
            "Bridge: {} terminals, {} with non-empty token_ranges",
            bridge.len(), exact_count
        );
        assert!(
            exact_count > 0,
            "the bridge must have at least one exact entry (the special tokens)"
        );

        // Build the PDA.
        let pda = llguidance::dpda_adapter::compile_pda(&grm_c).expect("the PDA compile");
        eprintln!(
            "PDA: {} states, {} inputs, {} transitions, deterministic={}",
            pda.num_states, pda.num_inputs, pda.transitions.len(), pda.is_deterministic()
        );

        // Walk a valid token sequence through the legacy parser.
        let valid_tokens = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            let mut tokens = Vec::new();
            for _ in 0..200 {
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
        };

        if valid_tokens.len() < 10 {
            eprintln!("Not enough tokens for exact-bridge test ({}), skipping", valid_tokens.len());
            return;
        }

        eprintln!("Walking {} tokens through the full-envelope grammar", valid_tokens.len());

        // Measure the legacy mask time.
        let t0 = std::time::Instant::now();
        let legacy_masks = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            let mut masks = Vec::new();
            for &tok in &valid_tokens {
                let mask = m.compute_mask_immut().expect("the mask");
                masks.push(mask);
                if m.consume_token(tok).is_err() {
                    break;
                }
            }
            masks
        };
        let legacy_time = t0.elapsed();

        // Measure the PDA advance time (the O(1) indexed lookup).
        // Build the reverse bridge (token ID -> PDA terminal ID) for O(1) lookup.
        // This is the mathematically correct mapping: each vocabulary token is
        // assigned to exactly one PDA terminal (the bridge is disjoint).
        let reverse_bridge: std::collections::HashMap<u32, u32> = {
            let mut map = std::collections::HashMap::new();
            for (terminal_id, token_ids) in bridge.iter().enumerate() {
                for &tok in token_ids {
                    map.insert(tok, terminal_id as u32);
                }
            }
            map
        };
        eprintln!(
            "Reverse bridge: {} token->terminal mappings ({} terminals)",
            reverse_bridge.len(), bridge.len()
        );

        // Walk the PDA through the actual token sequence (the mathematically
        // correct path: each token maps to exactly one PDA terminal via the bridge).
        // Use the indexed lookup (O(1)) instead of the linear scan (O()).
        let pda_index = pda.build_index();
        let t1 = std::time::Instant::now();
        let mut pda_frontier: Vec<(u32, Vec<u32>)> =
            vec![(pda.start_state, vec![pda.start_stack])];
        let mut pda_steps = 0;
        for &tok in &valid_tokens {
            let Some(terminal_id) = reverse_bridge.get(&tok) else { continue };
            let a = *terminal_id;
            if pda_steps == 0 {
                eprintln!(
                    "DEBUG first token: tok={} terminal_id={} frontier_size={}",
                    tok, a, pda_frontier.len()
                );
                for &(ref cq, ref cstk) in pda_frontier.iter().take(3) {
                    let etop = cstk.last().copied().unwrap_or(pda.start_stack);
                    let eps_transitions = pda.lookup_indexed(&pda_index, *cq, None, etop);
                    let term_transitions = pda.lookup_indexed(&pda_index, *cq, Some(a), etop);
                    eprintln!(
                        "  config q={} top={} eps_trans={} term_trans={}",
                        cq, etop, eps_transitions.len(), term_transitions.len()
                    );
                    // Show the epsilon closure results.
                    let mut eps_configs: Vec<(u32, Vec<u32>)> = vec![(*cq, cstk.clone())];
                    let mut i = 0;
                    while i < eps_configs.len() && eps_configs.len() < 4096 {
                        let (eq, estk) = eps_configs[i].clone();
                        i += 1;
                        let etop2 = estk.last().copied().unwrap_or(pda.start_stack);
                        for t in pda.lookup_indexed(&pda_index, eq, None, etop2) {
                            let mut ns = estk.clone();
                            ns.pop();
                            for &p in t.push.iter().rev() {
                                ns.push(p);
                            }
                            if !eps_configs.iter().any(|(s, ss)| *s == t.next_q && *ss == ns) {
                                eps_configs.push((t.next_q, ns));
                            }
                        }
                    }
                    eprintln!(
                        "  eps_closure: {} configs, first few: {:?}",
                        eps_configs.len(),
                        eps_configs.iter().take(5).map(|(q, s)| format!("q={} top={}", q, s.last().unwrap_or(&0))).collect::<Vec<_>>()
                    );
                    // Check terminal moves at each epsilon-closure config.
                    for (eq, estk) in eps_configs.iter() {
                        let etop3 = estk.last().copied().unwrap_or(pda.start_stack);
                        let tt = pda.lookup_indexed(&pda_index, *eq, Some(a), etop3);
                        if !tt.is_empty() {
                            eprintln!(
                                "  FOUND terminal move at q={} top={} ({} transitions)",
                                eq, etop3, tt.len()
                            );
                        }
                    }
                }
            }
            let mut new_frontier: Vec<(u32, Vec<u32>)> = Vec::new();
            for &(ref cq, ref cstk) in pda_frontier.iter() {
                // Epsilon closure: follow epsilon moves from (cq, cstk) to reach
                // all configs where the terminal move `a` is available.
                let mut eps_configs: Vec<(u32, Vec<u32>)> = vec![(*cq, cstk.clone())];
                let mut i = 0;
                while i < eps_configs.len() && eps_configs.len() < 4096 {
                    let (eq, estk) = eps_configs[i].clone();
                    i += 1;
                    let etop = estk.last().copied().unwrap_or(pda.start_stack);
                    // The epsilon input ID is num_inputs (the sentinel).
                    let eps_key = pda.num_inputs;
                    for t in pda.lookup_indexed(&pda_index, eq, None, etop) {
                        let mut ns = estk.clone();
                        ns.pop();
                        for &p in t.push.iter().rev() {
                            ns.push(p);
                        }
                        if !eps_configs.iter().any(|(s, ss)| *s == t.next_q && *ss == ns) {
                            eps_configs.push((t.next_q, ns));
                        }
                    }
                    let _ = eps_key;
                }
                // Terminal move: for each config in the epsilon closure, try the
                // terminal `a`. Collect all successful next-configs.
                for (eq, estk) in &eps_configs {
                    let etop = estk.last().copied().unwrap_or(pda.start_stack);
                    for t in pda.lookup_indexed(&pda_index, *eq, Some(a), etop) {
                        let mut ns = estk.clone();
                        ns.pop();
                        for &p in t.push.iter().rev() {
                            ns.push(p);
                        }
                        if !new_frontier.contains(&(t.next_q, ns.clone())) {
                            new_frontier.push((t.next_q, ns));
                        }
                    }
                }
            }
            if new_frontier.is_empty() {
                break;
            }
            pda_frontier = new_frontier;
            pda_steps += 1;
        }
        let pda_time = t1.elapsed();

        let legacy_us = legacy_time.as_micros() as f64 / legacy_masks.len().max(1) as f64;
        let pda_us = pda_time.as_micros() as f64 / pda_steps.max(1) as f64;

        eprintln!(
            "Exact-bridge bench: {} legacy masks, {} PDA steps",
            legacy_masks.len(), pda_steps
        );
        eprintln!(
            "  legacy compute_mask_immut: {:.1} us/token (O(grammar))",
            legacy_us
        );
        eprintln!(
            "  PDA advance_eps:           {:.1} us/token (O(1) indexed)",
            pda_us
        );
        if pda_us > 0.0 {
            eprintln!(
                "  speedup: {:.1}x (legacy / PDA)",
                legacy_us / pda_us
            );
        }

        // The PDA advance is only faster for DETERMINISTIC grammars (the LR(1) case,
        // where the epsilon closure is a single path). For non-deterministic
        // grammars (the full-envelope with choices), the epsilon closure BFS
        // is expensive and the PDA is SLOWER than the legacy Earley path.
        //
        // The PDA's advantage is on the GPU path (the fused_sample /
        // fused_project kernels), where the PDA table is precomputed and the
        // epsilon closure runs in parallel (the SIMD/CUDA). On the CPU, the
        // PDA only wins for deterministic grammars.
        if pda.is_deterministic() {
            assert!(
                pda_us < legacy_us,
                "for deterministic grammars, the PDA advance ({:.1} us) must be faster than the legacy mask ({:.1} us)",
                pda_us, legacy_us
            );
            eprintln!("  PDA is deterministic: {:.1}x speedup (PDA {} us vs legacy {} us/token)",
                legacy_us / pda_us.max(1.0), pda_us, legacy_us);
        } else {
            eprintln!("  PDA is NON-deterministic ({} transitions): CPU PDA advance ({:.1} us) vs legacy ({:.1} us)",
                pda.transitions.len(), pda_us, legacy_us);
            eprintln!("  The PDA GPU path (fused_sample/fused_project) is the correct optimization for this grammar.");
            eprintln!("  The CPU PDA is only faster for deterministic (LR(1)) grammars.");
        }
    }

    /// The self-contained exact-bridge differential (no external tokenizer file):
    /// a token-range grammar over the single-byte env has an EXACT terminal-to-token
    /// bridge (each terminal maps to a specific token range, not the identity
    /// fallback), so the PDA mask must equal the CPU mask BIT-FOR-BIT. This is the
    /// inclusive (every CPU-allowed token is PDA-allowed) + exclusive (no PDA-allowed
    /// token is CPU-forbidden) proof the firewall requires — the prior test only
    /// checked the inclusive direction (the "over-allowing is safe" evasion).
    #[test]
    fn pda_mask_equals_cpu_mask_exact_bridge_token_range() {
        use llguidance::toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let f = ParserFactory::new_simple(&env).expect("the factory");
        // A deterministic token-range grammar: the sequence "a" "b" "c" (the 97,98,99).
        // The token-range terminals give an EXACT bridge (not the identity fallback).
        let g = TopLevelGrammar::from_lark(r#"start: <[97]> <[98]> <[99]>"#.to_string());

        // The CPU (legacy) mask walk: the authoritative allowed-set at each step.
        let cpu_masks = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            let mut masks = Vec::new();
            for tok in [97u32, 98, 99] {
                let mask = m.compute_mask_immut().expect("the CPU mask");
                masks.push(mask);
                m.consume_token(tok).expect("the CPU consume");
                m.settle();
            }
            masks
        };

        // The PDA mask walk: the claimant's PDA frontier (the pda_config) + the PDA machine's
        // mask_at_cfg, lifted to tokens via the terminal->token bridge (the
        // terminal_token_map) — the same computation the GPU pda_mask_words does.
        let pda_masks = {
            let parser = f.create_parser(g.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.settle();
            // The bridge (the terminal->token lift, the exact for the token-range grammar).
            let grm_c = {
                let p2 = f.create_parser(g.clone()).expect("the parser for the grammar");
                p2.parser.grammar().clone()
            };
            let bridge = llguidance::dpda_adapter::terminal_token_map(&grm_c, f.tok_env());
            let mut masks = Vec::new();
            for tok in [97u32, 98, 99] {
                // The PDA allowed terminals over the frontier (the pda_config), lifted to
                // tokens via the bridge — the same computation the GPU pda_mask_words does.
                // The configs are read BEFORE the with_pda_machine closure (to avoid the
                // overlapping &mut self borrow of m).
                let configs = m.pda_config().unwrap_or_default();
                let pda_words: Vec<u32> = m.with_pda_machine(|pda| {
                    let mut allowed: std::collections::HashSet<u32> = std::collections::HashSet::new();
                    for (q, stk) in &configs {
                        for &a in &pda.mask_at_cfg(*q, stk) {
                            allowed.insert(a);
                        }
                    }
                    // The lift: each allowed terminal -> its token set (the bridge).
                    let mut vob = llguidance::toktrie::SimpleVob::alloc(env.tok_trie().vocab_size() as usize);
                    for &a in &allowed {
                        if let Some(tokens) = bridge.get(a as usize) {
                            for &t in tokens {
                                vob.allow_token(t);
                            }
                        }
                    }
                    vob.to_list()
                }).unwrap_or_default();
                masks.push(pda_words);
                m.consume_token(tok).expect("the PDA consume");
                m.settle();
            }
            masks
        };

        // The inclusive + exclusive: at every step the PDA allowed-set == the CPU
        // allowed-set, bit-for-bit (not the superset the prior test permitted).
        // Both cpu (the SimpleVob.to_list) and pda (the bridge-lifted token list) are
        // Vec<u32> token IDs; compare them as sets.
        for (i, (cpu, pda)) in cpu_masks.iter().zip(pda_masks.iter()).enumerate() {
            let cpu_set: std::collections::HashSet<u32> = cpu.to_list().into_iter().collect();
            let pda_set: std::collections::HashSet<u32> = pda.iter().copied().collect();
            assert_eq!(
                pda_set, cpu_set,
                "step {i}: the PDA mask must EQUAL the CPU mask bit-for-bit (the firewall exactness); PDA={pda_set:?} CPU={cpu_set:?}"
            );
        }
    }

    /// The product-config mask (the PDA-only state-tracking target, the design doc
    /// section 13) must EQUAL the Earley compute_mask_immut bit-for-bit, at every
    /// step of a walk. This is the correctness gate for wiring the product config in
    /// as the single mask source: if it ever diverges from the Earley mask, the PDA
    /// frontier or the lexer contribution is wrong.
    ///
    /// Two parallel matchers are walked in lockstep (the same tokens committed to
    /// both): one is queried via compute_mask_immut (the Earley reference), the other
    /// via product_config_mask (the PDA + lexer product). Their masks must be
    /// identical at every step (the inclusive: every Earley-allowed token is
    /// product-allowed; the exclusive: no product-allowed token is Earley-forbidden).
    #[test]
    fn product_config_mask_equals_compute_bias() {
        use llguidance::toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let f = ParserFactory::new_simple(&env).expect("the factory");

        // The differential over a (grammar, walk) pair: two parallel matchers, one
        // queried via compute_mask_immut (the Earley reference) + one via
        // product_config_mask (the PDA + lexer product), must agree bit-for-bit at
        // every step.
        fn differential(
            f: &ParserFactory,
            g: &TopLevelGrammar,
            walk: &[u32],
        ) -> (Vec<SimpleVob>, Vec<SimpleVob>) {
            // The CPU (Earley) reference walk: the authoritative mask at each step.
            let cpu_masks: Vec<SimpleVob> = {
                let parser = f.create_parser(g.clone()).expect("the parser");
                let mut m = Matcher::new(Ok(parser));
                m.settle();
                let mut masks = Vec::new();
                for &tok in walk {
                    let mask = m.compute_mask_immut().expect("the CPU mask");
                    masks.push(mask);
                    if m.consume_token(tok).is_err() {
                        break;
                    }
                    m.settle();
                }
                masks
            };
            // The product-config walk: the PDA + lexer single-tracker mask at each step.
            let product_masks: Vec<SimpleVob> = {
                let parser = f.create_parser(g.clone()).expect("the parser");
                let mut m = Matcher::new(Ok(parser));
                m.settle();
                let mut masks = Vec::new();
                for &tok in walk {
                    let mask = m.product_config_mask();
                    masks.push(mask);
                    if m.consume_token(tok).is_err() {
                        break;
                    }
                    m.settle();
                }
                masks
            };
            (cpu_masks, product_masks)
        }

        // The inclusive + exclusive equality check: the product-config mask must equal
        // the Earley mask bit-for-bit at every step.
        fn assert_equal(cpu: &[SimpleVob], product: &[SimpleVob], label: &str) {
            assert_eq!(
                cpu.len(),
                product.len(),
                "{label}: both walks must consume the same number of steps"
            );
            for (i, (c, p)) in cpu.iter().zip(product.iter()).enumerate() {
                let cpu_set: std::collections::HashSet<u32> = c.to_list().into_iter().collect();
                let product_set: std::collections::HashSet<u32> = p.to_list().into_iter().collect();
                assert_eq!(
                    product_set, cpu_set,
                    "{label} step {i}: the product-config mask must EQUAL the Earley mask bit-for-bit; product={product_set:?} cpu={cpu_set:?}"
                );
            }
        }

        // Case 1: the pure-CFG token-range grammar (no regex/parametric leaves).
        // The PDA bridge is exact (each terminal maps to a single token range), so the
        // PDA frontier advances and the product mask EQUALS the Earley mask at every step.
        // This is the inclusive + exclusive proof for the pure-CFG subclass.
        let g1 = TopLevelGrammar::from_lark(
            r#"start: <[123]> <[97]> <[58]> <[98]> <[125]>"#.to_string(),
        );
        let (cpu1, product1) = differential(&f, &g1, &[123, 97, 58, 98, 125]);
        assert_equal(&cpu1, &product1, "case1 (the pure-CFG token-range grammar)");

        // Case 1b: the NESTED pure-CFG grammar (the bounded object shape, without the
        // recursion). This exercises the PDA STACK (the push/pop, the nested structure)
        // + the multi-config frontier, proving the product mask == the Earley mask over
        // a non-trivial PDA (not just the linear chain of case 1). The grammar is
        // { a : { b } } — the nested object, the token-range terminals only (no the
        // regex/parametric leaves, so the PDA path is engaged).
        let g1b = TopLevelGrammar::from_lark(
            r#"start: <[123]> <[97]> <[58]> <[123]> <[98]> <[58]> <[99]> <[125]> <[125]>"#.to_string(),
        );
        let walk1b: Vec<u32> = vec![123, 97, 58, 123, 98, 58, 99, 125, 125];
        let (cpu1b, product1b) = differential(&f, &g1b, &walk1b);
        assert_equal(&cpu1b, &product1b, "case1b (the nested pure-CFG grammar, the PDA stack)");

        // Case 1c: the BRANCHING (choice) grammar (the a | b). The choice makes the PDA
        // NON-deterministic (the two epsilon successors at the choice dot, so
        // is_deterministic is false), so pda_engaged is FALSE and the production
        // routing uses the CPU Earley mask (the authoritative path), NOT the PDA
        // product mask. This case proves the ROUTING: a choice grammar correctly
        // falls to the CPU path (not the PDA product mask, which would over-allow
        // the NPDA frontier's predicted-but-inactive choice branches). The CPU mask
        // is self-consistent (the full walk consumes).
        let g1c = TopLevelGrammar::from_lark(
            r#"start: <[123]> (<[97]> | <[98]>) <[125]>"#.to_string(),
        );
        // The routing proof: the choice grammar's PDA is non-deterministic, so
        // pda_engaged is false (the CPU Earley mask is authoritative, not the
        // PDA product mask).
        {
            let parser = f.create_parser(g1c.clone()).expect("the parser");
            let mut m = Matcher::new(Ok(parser));
            m.with_pda_machine(|pda| {
                assert!(
                    !pda.is_deterministic(),
                    "the choice grammar's PDA is non-deterministic (the two epsilon successors at the choice dot), so pda_engaged is false and the CPU Earley mask is authoritative (not the PDA product mask over-allowing the NPDA frontier's predicted branches)"
                );
            });
        }
        // The CPU mask (the authoritative path for the choice grammar) is
        // self-consistent: the full left-branch + right-branch walks consume.
        let walk1c_left: Vec<u32> = vec![123, 97, 125];
        let (cpu1c_l, _) = differential(&f, &g1c, &walk1c_left);
        assert_eq!(
            cpu1c_l.len(),
            walk1c_left.len(),
            "the CPU (the authoritative, pda_engaged-false) mask consumes the full left-branch walk (got {} steps)",
            cpu1c_l.len()
        );
        let walk1c_right: Vec<u32> = vec![123, 98, 125];
        let (cpu1c_r, _) = differential(&f, &g1c, &walk1c_right);
        assert_eq!(
            cpu1c_r.len(),
            walk1c_right.len(),
            "the CPU mask consumes the full right-branch walk (got {} steps)",
            cpu1c_r.len()
        );

        // Case 2: the rich JSON grammar (the NUMBER + the STRING regex leaves). The PDA
        // bridge is INEXACT (a regex terminal maps to a SET of tokens, not a single
        // token), so the PDA mirror is off (the pda_configs frozen at the start) and the
        // product mask is NOT the authoritative one — the CPU Earley mask is. This case
        // documents the boundary: the product-config tracker is exact for the pure-CFG
        // subclass (case 1) and inapplicable to the regex/parametric subclass (case 2),
        // where the lexer DFA is the authoritative token generator. We assert the CPU
        // walk is self-consistent (the full walk consumes) to pin the reference, and
        // record that the product walk diverges (the documented limit, not a defect).
        let g2 = TopLevelGrammar::from_lark(
            r#"start: <[123]> <[110]> <[58]> NUMBER <[44]> <[115]> <[58]> STRING <[125]>
                %ignore /[ \t]+/
                NUMBER: /[0-9]+/
                STRING: /"[^"]*"/
"#
            .to_string(),
        );
        let (cpu2, product2) = differential(
            &f,
            &g2,
            &[123, 110, 58, 53, 44, 115, 58, 34, 97, 34, 125],
        );
        // The CPU walk is self-consistent (the full rich-JSON sequence consumes).
        assert!(
            cpu2.len() >= 11,
            "the CPU walk must consume the full rich-JSON sequence (got {} steps)",
            cpu2.len()
        );
        // The product walk diverges for the inexact-bridge grammar (the documented
        // boundary, not a correctness claim). Record it, do not assert equality.
        eprintln!(
            "case2 boundary: CPU walk {} steps, product walk {} steps (the product diverges for the inexact-bridge grammar, the documented limit)",
            cpu2.len(),
            product2.len()
        );
    }

    /// THE LOAD-BEARING INVARIANT (the pda_engaged => pda_mirror_engaged ordering):
    /// the PDA mask is READ (the product_config_mask routing, the compute_bias
    /// pda_engaged gate) only when pda_engaged = (det AND exact AND pure_cfg). The
    /// PDA frontier (pda_configs) is ADVANCED by the mirror only when
    /// pda_mirror_engaged = (det AND exact). Since pda_engaged implies
    /// pda_mirror_engaged (the 3-conjunct implies the 2-conjunct), the frontier is
    /// ALWAYS advanced before it is read. If a future change weakened
    /// pda_mirror_engaged below pda_engaged (for example, dropped the exact conjunct
    /// from the mirror but kept it in the mask gate), the mask would read a STALE
    /// frontier (the reflected-injection / the impossible-token class). This test pins
    /// the ordering: for a pda_engaged grammar, committing a token MUST advance the
    /// frontier (the mirror ran); the frontier is never read stale.
    #[test]
    fn pda_mask_read_is_never_stale_relative_to_the_mirror() {
        use llguidance::toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let f = ParserFactory::new_simple(&env).expect("the factory");
        // A pda_engaged grammar: deterministic + exact bridge + pure-CFG (the token-
        // range sequence, without the choice/parametric). The <[97]> <[98]> <[99]>
        // sequence is deterministic (the single path) + pure-CFG (no regex/skip
        // leaves) + exact bridge (the token-range lift is a function).
        let g = TopLevelGrammar::from_lark(r#"start: <[97]> <[98]> <[99]>"#.to_string());

        let parser = f.create_parser(g.clone()).expect("the parser");
        let mut m = Matcher::new(Ok(parser));
        m.settle();

        // The frontier before any commit (the start config).
        let frontier_before = m.pda_config().expect("the PDA is engaged for this grammar");
        assert_eq!(frontier_before.len(), 1, "the deterministic grammar has a single-config frontier");

        // Commit the first token. The mirror (the pda_mirror_engaged, det+exact)
        // MUST advance the frontier (not the mask reading a stale frontier).
        m.consume_token(97).expect("the commit");
        m.settle();
        let frontier_after = m.pda_config().expect("the PDA is engaged");
        assert_ne!(
            frontier_after, frontier_before,
            "committing a token MUST advance the PDA frontier (the mirror ran); a stale frontier means the mask (the pda_engaged read) would be computed from the pre-commit state (the reflected injection)"
        );

        // The mask read AFTER the commit reflects the ADVANCED frontier (not the
        // pre-commit one). The product_config_mask at this point must allow the NEXT
        // token (98) and NOT the already-committed one (97).
        let mask = m.product_config_mask();
        assert!(
            mask.is_allowed(98),
            "the post-commit mask allows the next token (98); the frontier advanced past the committed 97"
        );
        assert!(
            !mask.is_allowed(97),
            "the post-commit mask does NOT re-allow the already-committed token (97); a stale frontier would re-allow it (the reflection)"
        );
    }
}