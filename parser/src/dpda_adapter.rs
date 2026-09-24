//! The adapter: implement the pushdown_rs::Grammar trait for the llguidance
//! CGrammar (the compiled grammar). This is the bridge that lets the
//! pushdown-rs core compile the llguidance's grammars to PDAs.
//!
//! The critical detail (the lesson from the global/local index bug): the
//! terminal_id/nonterminal_id must return the LOCAL indices (the 0..num_terminals,
//! the 0..num_nonterminals), NOT the global symbol IDs (the 0..num_symbols).
//! The RTN compilation numbers the states by the local index.
//!
//! Performance: the CGrammar uses a flat CSymIdx over ALL symbols (terminals +
//! nonterminals mixed). The wrapper precomputes the global->local maps once,
//! giving O(1) ID lookups during compilation (vs the O(n) scan per call).

use pushdown_rs::compile::{CfgError, Grammar};

use crate::earley::{CGrammar, CSymIdx};
use derivre::RegexAst;

/// A wrapper around &CGrammar that precomputes the global->local symbol index
/// maps for O(1) terminal_id/nonterminal_id lookups during RTN compilation.
pub struct PdaGrammar<'a> {
    grm: &'a CGrammar,
    /// global CSymIdx (0..num_symbols) -> local terminal index (0..num_terminals), or None
    global_to_local_terminal: Vec<Option<u32>>,
    /// global CSymIdx (0..num_symbols) -> local nonterminal index (0..num_nonterminals), or None
    global_to_local_nonterminal: Vec<Option<u32>>,
    #[allow(dead_code)]
    num_terminals: u32,
    #[allow(dead_code)]
    num_nonterminals: u32,
    start_local: u32,
}

impl<'a> PdaGrammar<'a> {
    pub fn new(grm: &'a CGrammar) -> Self {
        let num_syms = grm.num_symbols();
        let mut global_to_local_terminal = vec![None; num_syms];
        let mut global_to_local_nonterminal = vec![None; num_syms];
        let mut term_count = 0u32;
        let mut nt_count = 0u32;
        for i in 0..num_syms {
            let s = CSymIdx::new_checked(i);
            if grm.is_terminal(s) {
                global_to_local_terminal[i] = Some(term_count);
                term_count += 1;
            } else {
                global_to_local_nonterminal[i] = Some(nt_count);
                nt_count += 1;
            }
        }
        let start_local = global_to_local_nonterminal
            .get(grm.start().as_index())
            .copied()
            .flatten()
            .unwrap_or(0);
        PdaGrammar {
            grm,
            global_to_local_terminal,
            global_to_local_nonterminal,
            num_terminals: term_count,
            num_nonterminals: nt_count,
            start_local,
        }
    }

    pub fn grammar(&self) -> &CGrammar {
        self.grm
    }
}

impl<'a> Grammar for PdaGrammar<'a> {
    type Nonterminal = CSymIdx;
    type Terminal = CSymIdx;
    type Symbol = CSymIdx;

    fn nonterminals(&self) -> Vec<CSymIdx> {
        (0..self.grm.num_symbols())
            .filter(|&i| self.global_to_local_nonterminal[i].is_some())
            .map(CSymIdx::new_checked)
            .collect()
    }

    fn terminals(&self) -> Vec<CSymIdx> {
        (0..self.grm.num_symbols())
            .filter(|&i| self.global_to_local_terminal[i].is_some())
            .map(CSymIdx::new_checked)
            .collect()
    }

    fn start(&self) -> CSymIdx {
        self.grm.start()
    }

    fn productions(&self) -> Vec<(CSymIdx, Vec<CSymIdx>)> {
        let mut prods = Vec::new();
        for i in 0..self.grm.num_symbols() {
            let s = CSymIdx::new_checked(i);
            if self.grm.is_terminal(s) {
                continue;
            }
            for &rule in self.grm.rules_of(s) {
                let rhs = self.grm.rhs_symbols(rule).to_vec();
                prods.push((s, rhs));
            }
        }
        prods
    }

    fn is_terminal(&self, sym: &CSymIdx) -> bool {
        self.grm.is_terminal(*sym)
    }

    fn terminal_id(&self, sym: &CSymIdx) -> Option<u32> {
        self.global_to_local_terminal.get(sym.as_index()).copied().flatten()
    }

    fn terminal_name(&self, sym: &CSymIdx) -> String {
        // The CGrammar terminal's name (the "reasoning_block", the "text", the
        // "tool_call", ...). The consumer (the llguidance mask builder) maps this
        // name back to the terminal's token range (the LexemeSpec.token_ranges), so
        // the PDA input bit is identified with the actual tokenizer lexeme.
        self.grm.sym_name(*sym).to_string()
    }

    fn nonterminal_id(&self, sym: &CSymIdx) -> Option<u32> {
        self.global_to_local_nonterminal.get(sym.as_index()).copied().flatten()
    }

    fn nonterminal_index(&self, nt: &CSymIdx) -> u32 {
        self.global_to_local_nonterminal.get(nt.as_index()).copied().flatten().unwrap_or(0)
    }

    fn start_id(&self) -> u32 {
        self.start_local
    }

    fn validate(&self) -> Result<(), CfgError> {
        // The RTN compilation is CFG-only. Parametric grammars (the 64-bit rule
        // conditions evaluated at Earley-item time) cannot be compiled to a PDA.
        // Reject them here so the PDA path is never taken for aetric grammars.
        if self.grm.parametric() {
            return Err(CfgError::Other(
                "parametric grammars cannot be compiled to a PDA (the RTN is CFG-only)".into(),
            ));
        }
        Ok(())
    }
}

/// Compile the CGrammar to a PDA machine (the RTN construction).
pub fn compile_pda(grm: &CGrammar) -> Result<pushdown_rs::machine::PdaMachine, CfgError> {
    let adapter = PdaGrammar::new(grm);
    pushdown_rs::compile(&adapter)
}

/// The terminal-to-token bridge: for each PDA local terminal ID `a`
/// (0..num_inputs), the set of vocabulary token IDs that terminal covers.
///
/// Derived from `CGrammar.terminal_token_ranges` (the precomputed
/// `LexemeSpec.token_ranges`). When a terminal's ranges are empty, the
/// token set is computed from the tokenizer's trie (the `TokenSpanner`
/// approach): iterate over the vocabulary and check which tokens' byte
/// sequences match the terminal's regex.
pub fn terminal_token_map(grm: &CGrammar, tok_env: &crate::toktrie::TokEnv) -> Vec<Vec<u32>> {
    terminal_token_map_dfa(grm, tok_env, None)
}

/// The full bridge computation with an optional DFA (the `RegexVec`) for
/// matching the complex `RegexAst` terminals (the `Or`, the `Regex`, the
/// `ExprRef`). When the DFA is provided, the bridge entries for the complex
/// terminals are computed by driving the DFA over each vocabulary token's
/// byte sequence and checking if the terminal's lexeme is in the accepting set.
pub fn terminal_token_map_dfa(
    grm: &CGrammar,
    tok_env: &crate::toktrie::TokEnv,
    dfa: Option<&mut crate::earley::regexvec::RegexVec>,
) -> Vec<Vec<u32>> {
    let adapter = PdaGrammar::new(grm);
    let terminals = adapter.terminals(); // the CSymIdx list, in local-ID order
    let trie = tok_env.tok_trie();
    let vocab_size = trie.vocab_size();
    let num_terminals = terminals.len();

    // P3 (the semantic exactness): for each terminal, compute the ACTUAL lexeme
    // acceptance set (the tokens whose bytes the terminal's lexeme accepts). This
    // is the ground truth (the no two). The specific terminals (the
    // token_ranges, the byte patterns) have known sets (the cheap). The complex
    // terminals (the regexes) use the DFA matching (the actual acceptance).
    //
    // Then the partition condition (P1+P2) is tested: are the P3 sets pairwise
    // disjoint + total? If yes, the bridge is a function (the DPDA semantics If no
    // (two terminals accept the same token), the bridge is a relation (the NPDA,
    // the nondeterministic choice).
    let mut bridge: Vec<Vec<u32>> = vec![Vec::new(); num_terminals];

for (i, &csym) in terminals.iter().enumerate() {
        // The specific terminals (the token_ranges, the byte patterns): the known set.
        let ranges = grm.terminal_token_ranges(csym);
        if !ranges.is_empty() {
            for range in ranges.iter() {
                for tok in range.clone() {
                    let idx = tok as usize;
                    if idx < vocab_size {
                        bridge[i].push(tok);
                    }
                }
            }
            continue;
        }
        // The byte pattern terminals (the single-byte, the multi-byte literals): the
        // known set (the greedy tokenization).
        let lexeme_idx = grm.sym_data(csym).lexeme;
        if let Some(lex) = lexeme_idx {
            let lexeme_spec = grm.lexer_spec().lexeme_spec(lex);
            let byte_pattern = extract_byte_pattern(&lexeme_spec.rx);
            if let Some(bytes) = byte_pattern {
                if bytes.len() == 1 {
                    if let Some(tok_id) = trie.token_id(&bytes) {
                        let idx = tok_id as usize;
                        if idx < vocab_size {
                            bridge[i].push(tok_id);
                        }
                    }
                } else if bytes.len() > 1 {
                    let token_ids = trie.greedy_tokenize(&bytes);
                    for tok_id in token_ids {
                        let idx = tok_id as usize;
                        if idx < vocab_size {
                            bridge[i].push(tok_id);
                        }
                    }
                }
            }
        }
    }

    // The complex terminals (the regexes): the DFA matching (the actual acceptance).
    // Done in a separate pass (the no move of the &mut dfa inside the per-terminal loop).
    if let Some(dfa) = dfa {
        for (i, &csym) in terminals.iter().enumerate() {
            if !bridge[i].is_empty() {
                continue; // the specific terminal (the P3 set already computed)
            }
            let lexeme_idx = grm.sym_data(csym).lexeme;
            let Some(lex) = lexeme_idx else { continue }; // the NULL terminal (the no lexeme)
            let mut selected = grm.lexer_spec().alloc_lexeme_set();
            selected.add(lex);
            let initial = dfa.initial_state(&selected);
            for tok_id in 0..vocab_size as u32 {
                let token_str = trie.token_str(tok_id);
                let token_bytes = token_str.as_bytes();
                let mut state = initial;
                for &byte in token_bytes {
                    state = dfa.transition(state, byte);
                }
                let desc = dfa.state_desc(state);
                if desc.greedy_accepting.contains(lex) || desc.lazy_accepting.contains(lex) {
                    bridge[i].push(tok_id);
                }
            }
        }
    }

    // Sort + dedup each bridge entry (the P3 set, the no form).
    for tokens in bridge.iter_mut() {
        tokens.sort();
        tokens.dedup();
    }

    bridge
}

/// The algorithmic bridge contract: the terminal-to-token mapping is
/// sound for the PDA mask iff ALL three properties hold:
///
/// 1. NO FALLBACK: every terminal has non-empty `token_ranges` (the
///    identity fallback is not used).
/// 2. DISJOINT: no vocabulary token ID appears in two different
///    terminals' ranges.
/// 3. TOTAL: every vocabulary token ID in [0, vocab_size) is covered
///    by at least one terminal's range.
///
/// When all three hold, the PDA mask (the `mask_at_cfg` lifted through
/// the bridge) is a precise token-level mask equivalent to the legacy
/// Earley `compute_bias`. When any fails, the PDA mask is an
/// over- or under-approximation and must not replace the legacy path.
pub fn bridge_is_exact(grm: &CGrammar, vocab_size: usize, tok_env: &crate::toktrie::TokEnv) -> bool {
    bridge_is_exact_dfa(grm, vocab_size, tok_env, None)
}

/// The full bridge exactness check with an optional DFA (the `RegexVec`) for
/// matching the complex `RegexAst` terminals. When the DFA is provided, the
/// bridge entries for the complex terminals are computed by driving the DFA
/// over each vocabulary token's byte sequence.
pub fn bridge_is_exact_dfa(
    grm: &CGrammar,
    vocab_size: usize,
    tok_env: &crate::toktrie::TokEnv,
    dfa: Option<&mut crate::earley::regexvec::RegexVec>,
) -> bool {
    let bridge = terminal_token_map_dfa(grm, tok_env, dfa);
    bridge_is_exact_from_map(&bridge, vocab_size, tok_env)
}

/// The exactness check over a PRECOMPUTED bridge (the no re-derivation). This
/// is the independent-register form: given the terminal->token map, verify the
/// algorithmic contract (disjointness + totality, the EOS excluded) WITHOUT
/// recomputing the bridge. A caller that has already built the bridge (the
/// ParserState construction) uses this to avoid the double computation.
///
/// Property 2 (disjointness): no vocabulary token ID appears in two different
/// terminals' entries. Property 3 (totality): every NON-EOS token ID in
/// [0, vocab_size) is covered by at least one terminal's entry.
pub fn bridge_is_exact_from_map(
    bridge: &[Vec<u32>],
    vocab_size: usize,
    tok_env: &crate::toktrie::TokEnv,
) -> bool {
    let trie = tok_env.tok_trie();
    // The EOS tokens are not part of the grammar (they're the end-of-sequence
    // markers). Exclude them from the totality check.
    let eos_tokens: std::collections::HashSet<u32> = trie.eos_tokens().iter().copied().collect();

    // Property 2: disjointness over the full vocabulary.
    let mut covered = vec![false; vocab_size];
    for token_list in bridge.iter() {
        for &tok in token_list {
            let idx = tok as usize;
            if idx >= vocab_size {
                continue;
            }
            if covered[idx] {
                return false; // Property 2 violated: token in two terminals
            }
            covered[idx] = true;
        }
    }
    // Property 3: every NON-EOS token is covered.
    covered.iter().enumerate().all(|(idx, &c)| c || eos_tokens.contains(&(idx as u32)))
}

/// Compile the CGrammar's PDA as a CUDA package (the bitvec + the source
/// primitives). This is the H2D payload for the attention-rs kernels.
pub fn export_pda_package(grm: &CGrammar) -> Result<pushdown_rs::cuda::CudaPackage, CfgError> {
    let machine = compile_pda(grm)?;
    pushdown_rs::cuda::CudaPackage::from_machine(&machine).map_err(|e| {
        CfgError::Other(format!("the CUDA package export failed: {e}"))
    })
}

/// The displacement-based bridge (the CFGzip Theorem 2, the pure functional atom):
/// group the vocabulary tokens by their PDA displacement (the set of
/// (in_config, out_config) pairs induced by the token's terminal sequence). Two
/// tokens are in the same bridge class iff they have the same displacement (the
/// interchangeable inputs, the no DFA).
///
/// The `token_terminal` closure maps a token ID to its terminal sequence (the
/// lexer's output, the single-terminal case for now). The displacement is
/// computed via the PDA's `displacement` method (the pure function). The bridge
/// is the displacement partition (the tokens grouped by their displacement).
#[allow(dead_code)] // the GNF/preterminal conversion (the next increment) will
// wire this into the live pda_bridge (the multi-terminal token sequences).
// For now it equals the existing bridge (the single-terminal case).
pub fn displacement_bridge(
    pda: &pushdown_rs::machine::PdaMachine,
    vocab_size: usize,
    token_terminal: &dyn Fn(u32) -> Vec<u32>,
) -> Vec<Vec<u32>> {
    // The displacement signature for each token (the set of (in_config, out_config) pairs).
    let mut signatures: Vec<(u32, Vec<(u32, Vec<u32>, u32, Vec<u32>)>)> = Vec::with_capacity(vocab_size);
    for tok in 0..vocab_size as u32 {
        let terminal_seq = token_terminal(tok);
        let disp = pda.displacement(&terminal_seq);
        signatures.push((tok, disp));
    }
    // Group the tokens by their displacement signature (the displacement partition).
    let mut groups: Vec<(Vec<(u32, Vec<u32>, u32, Vec<u32>)>, Vec<u32>)> = Vec::new();
    for (tok, disp) in signatures {
        match groups.iter_mut().find(|(g, _)| *g == disp) {
            Some((_, toks)) => toks.push(tok),
            None => groups.push((disp, vec![tok])),
        }
    }
    // The bridge (the displacement class -> the token set).
    groups.into_iter().map(|(_, toks)| toks).collect()
}

/// Extract a fixed byte pattern from a `RegexAst` (the simple cases: the
/// `Literal`, the `Byte`, the `Concat` of literals/bytes). Returns `None`
/// for complex regexes (the `Or`, the `Regex`, the `AndRef`) that require
/// the DFA matching approach (the next increment).
fn extract_byte_pattern(ast: &RegexAst) -> Option<Vec<u8>> {
    match ast {
        RegexAst::Literal(s) => Some(s.as_bytes().to_vec()),
        RegexAst::Byte(b) => Some(vec![*b]),
        RegexAst::Concat(parts) => {
            let mut result = Vec::new();
            for part in parts {
                let bytes = extract_byte_pattern(part)?;
                result.extend(bytes);
            }
            Some(result)
        }
        _ => None, // the complex regexes (the Or, the Regex, the ExprRef) need the DFA
    }
}

/// The terminal classification (the GNF conversion, the first step): each
/// terminal is either byte-level (the known byte sequence, the literal / the
/// single byte / the concatenation), token-level (the token ID range, the
/// <[32006,32010]> form), or complex-regex (the Or / the Regex / the ExprRef,
/// the DFA matching required). This is the terminal -> the preterminal mapping
/// (the byte-level terminals expand to their byte sequence, the token-level
/// terminals stay at the token ID, the complex-regex terminals use the DFA).
#[derive(Debug, Clone, PartialEq)]
pub enum TerminalKind {
    /// The byte-level terminal (the known byte sequence, the literal / the
    /// single byte / the concatenation). The preterminal expansion is the byte
    /// sequence.
    BytePattern(Vec<u8>),
    /// The token-level terminal (the token ID range, the <[32006,32010]> form).
    /// The preterminal is the token ID (the no byte expansion).
    TokenRange(Vec<std::ops::RangeInclusive<u32>>),
    /// The complex-regex terminal (the Or / the Regex / the ExprRef). The DFA
    /// matching is required (the no byte pattern, the no token range).
    ComplexRegex,
}

/// Classify a terminal (the CSymIdx) into its kind (the byte-level / the
/// token-level / the complex-regex). This is the GNF conversion's first step
/// (the terminal -> the preterminal mapping).
pub fn classify_terminal(grm: &CGrammar, csym: CSymIdx) -> TerminalKind {
    // The token-level terminal (the token_ranges are populated).
    let ranges = grm.terminal_token_ranges(csym);
    if !ranges.is_empty() {
        return TerminalKind::TokenRange(ranges.iter().map(|r| *r.start()..=*r.end()).collect());
    }
    // The byte-level terminal (the known byte pattern).
    let lexeme_idx = grm.sym_data(csym).lexeme;
    if let Some(lex) = lexeme_idx {
        let lexeme_spec = grm.lexer_spec().lexeme_spec(lex);
        if let Some(bytes) = extract_byte_pattern(&lexeme_spec.rx) {
            return TerminalKind::BytePattern(bytes);
        }
    }
    // The complex-regex terminal (the no byte pattern, the no token range).
    TerminalKind::ComplexRegex
}

/// The terminal -> the preterminal sequence (the GNF conversion, the second step):
/// the byte-level terminals expand to their byte sequence (the preterminals), the
/// token-level terminals stay at the token ID (the no byte expansion), and the
/// complex-regex terminals use the DFA matching (the no preterminal expansion,
/// the next increment). This is the terminal -> the preterminal mapping that the
/// displacement bridge uses (the no DFA, the pure PDA simulation).
pub fn terminal_preterminals(
    grm: &CGrammar,
    csym: CSymIdx,
) -> Option<Vec<u8>> {
    match classify_terminal(grm, csym) {
        TerminalKind::BytePattern(bytes) => Some(bytes),
        TerminalKind::TokenRange(_) => None, // the token-level (the no byte expansion)
        TerminalKind::ComplexRegex => None, // the DFA matching (the no increment)
    }
}

/// The GNF displacement bridge (the no DFA, the pure PDA simulation): for the
/// terminal, compute its preterminal sequence (the byte-level expansion, the
/// token-level identity), and then compute the displacement of the preterminal
/// sequence (the PDA's stack transformation). The bridge is the displacement
/// partition (the tokens grouped by their displacement).
///
/// The `token_preterminals` closure maps a token ID to its preterminal sequence
/// (the byte-level for the byte-level terminals, the token ID for the
/// token-level terminals). This is the lexer's output (the token -> the
/// preterminal sequence), computed without the DFA (the pure PDA simulation).
#[allow(dead_code)] // the GNF conversion is the next increment (the live bridge)
pub fn gnf_displacement_bridge<F: Fn(u32) -> Vec<u32> + Sync>(
    pda: &pushdown_rs::machine::PdaMachine,
    vocab_size: usize,
    eos_tokens: &std::collections::HashSet<u32>,
    token_preterminals: &F,
) -> Vec<Vec<u32>> {
    // The displacement is a function of the byte sequence (the preterminal
    // sequence), NOT the token ID. So group tokens by their byte sequence, and
    // compute the displacement once per group (the no per-token). This reduces
    // the O(vocab x displacement_cost) to the O(num_groups x displacement_cost +
    // vocab) (the num_groups << the vocab).
    //
    // Step 1: group the tokens by their byte sequence (the preterminal sequence).
    let mut seq_groups: Vec<(Vec<u32>, Vec<u32>)> = Vec::new(); // (byte_seq, token_ids)
    for tok in 0..vocab_size as u32 {
        if eos_tokens.contains(&tok) {
            continue; // the EOS tokens are excluded (the no part of the grammar)
        }
        let preterminals = token_preterminals(tok);
        match seq_groups.iter_mut().find(|(seq, _)| *seq == preterminals) {
            Some((_, toks)) => toks.push(tok),
            None => seq_groups.push((preterminals, vec![tok])),
        }
    }
    // Step 2: compute the displacement once per group (the no per-token). The
    // displacement computation is parallelized (the rayon, the no shared state):
    // each group's displacement is independent (the pure function).
    let displacements: Vec<Vec<(u32, Vec<u32>, u32, Vec<u32>)>> = {
        use rayon::prelude::*;
        seq_groups.par_iter().map(|(seq, _)| pda.displacement(seq)).collect()
    };
    // Step 3: group the groups by their displacement signature (the sorted set of
    // (in_config, out_config) pairs). The bridge is the token sets per signature.
    let mut sig_to_bridge_idx: Vec<(Vec<(u32, Vec<u32>, u32, Vec<u32>)>, usize)> = Vec::new();
    let mut bridge: Vec<Vec<u32>> = Vec::new();
    for (i, (_, toks)) in seq_groups.iter().enumerate() {
        let sig: Vec<_> = displacements[i].clone();
        match sig_to_bridge_idx.iter().position(|(s, _)| *s == sig) {
            Some(pos) => bridge[sig_to_bridge_idx[pos].1].extend(toks.iter().copied()),
            None => {
                let idx = bridge.len();
                bridge.push(toks.clone());
                sig_to_bridge_idx.push((sig, idx));
            }
        }
    }
    bridge
}

/// The GNF displacement bridge for a SUBSET of tokens (the no full vocab, the
/// benchmark speed). The `tokens` is the subset of token IDs to compute the
/// bridge for. The displacement is computed via the PDA's displacement method
/// (the pure functional, the no temporary state).
pub fn gnf_displacement_bridge_subset(
    pda: &pushdown_rs::machine::PdaMachine,
    tokens: &[u32],
    token_preterminals: &dyn Fn(u32) -> Vec<u32>,
) -> Vec<Vec<u32>> {
    // The displacement signature for each token in the subset.
    let mut signatures: Vec<(u32, Vec<(u32, Vec<u32>, u32, Vec<u32>)>)> =
        Vec::with_capacity(tokens.len());
    for &tok in tokens {
        let preterminals = token_preterminals(tok);
        let disp = pda.displacement(&preterminals);
        signatures.push((tok, disp));
    }
    // Group the tokens by their displacement signature (the displacement partition).
    let mut groups: Vec<(Vec<(u32, Vec<u32>, u32, Vec<u32>)>, Vec<u32>)> = Vec::new();
    for (tok, disp) in signatures {
        match groups.iter_mut().find(|(g, _)| *g == disp) {
            Some((_, toks)) => toks.push(tok),
            None => groups.push((disp, vec![tok])),
        }
    }
    // The bridge (the displacement class -> the token set).
    groups.into_iter().map(|(_, toks)| toks).collect()
}