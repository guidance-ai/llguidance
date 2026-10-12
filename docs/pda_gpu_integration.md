# PDA GPU Integration Guide

**For:** the GPU inference runtime integrating llguidance's grammar PDA.
**Depends on:** `pushdown-rs` (the PDA engine), `attention-rs` (the CUDA kernels).

## 1. Model Load (CPU, once)

```rust
use llguidance::dpda_adapter::{compile_pda, export_pda_package};
use llguidance::api::TopLevelGrammar;
use llguidance::ParserFactory;

// 1. Build the grammar (Lark, JSON schema, regex, or explicit token ranges).
let g = TopLevelGrammar::from_lark(grammar_source);

// 2. Compile to a CGrammar (the Earley parser's internal form).
let mut parser = factory.create_parser(g)?;
parser.start_without_prompt();
let grm = parser.parser.grammar().clone();

// 3. Compile to a PDA (the RTN construction, kappa(G) states).
let pda = compile_pda(&grm)?;
// pda.num_states, pda.transitions, pda.accepting, pda.start_state, pda.start_stack

// 4. Export the CUDA package (the flat POD bitvec + source primitives).
let pkg = export_pda_package(&grm)?;
// pkg.bitvec: BitVec<u64> (the GPU-uploadable payload)
// pkg.num_states, pkg.num_inputs, pkg.num_stack_syms, pkg.num_transitions
// pkg.upload_bytes(): the H2D size (~210 KB for the tool-call grammar)

// 5. Upload to GPU (attention-rs).
let table = PdaPushdownTable::from_cuda_package(&pkg, device)?;
// table.transitions: the flat (q, a, top, next_q, push_len, push[]) array
// table.accepting: the accepting state IDs
// table.num_states, table.num_inputs, table.num_stack_syms, table.num_transitions
```

## 2. Per-Token Sampling (GPU, fused)

The fused_sample kernel performs the full per-token PDA step in one launch:
scan the transition table for the current (ctrl, top), emit the VOB mask,
apply it to the logits, sample, and advance the PDA state.

**The CSR index.** The transition table is a flat array of variable-length
records. Without an index, each GPU thread would scan all 8764 records to
find the ~80 that match its control state. The CSR (ctrl_u32_offsets,
ctrl_counts) lets each thread jump directly to its state's records in O(1).
The CSR are computed on the CPU at upload time (one pass over the
transition array) and uploaded as a small [num_states] u32 tensor.

**The logit format.** The PDA mask is applied to F32 or BF16 logits. The
model's lm_head produces logits in the model's compute dtype; the sampler
dequantizes to F32 before the PDA step. FP8/FP4 are weight formats (for
the GEMM), not logit formats - by the time logits reach the sampler they
are already F32/BF16. The PDA kernel does not need FP8/FP4 variants.

**The optional dispatch.** If `ctrl` is null (no grammar active), the kernel
runs plain sampling with zero PDA overhead. The PDA is strictly optional.

```rust
// The fused kernel: ONE launch does mask + sample + advance.
// If no grammar is active, pass ctrl=null (the kernel skips the PDA path).

let (out_ctrl, out_sp, out_tokens) = table.fused_sample(
    &logits,      // [batch, vocab] F32/BF16
    &ctrl,        // [batch] current control state (or null)
    &stack,       // [batch * D] bounded stack
    &sp,          // [batch] stack pointer
    &PdaSampling::TopKTopP { temperature: 0.8, top_k: 50, top_p: 0.9 },
)?;
// out_ctrl: [batch] next control state
// out_sp: [batch] next stack pointer
// out_tokens: [batch] sampled token IDs
```

### The Optional Dispatch

```rust
// No grammar: zero PDA overhead (the existing sampling path).
let tokens = sampler.sample_cuda_vob(&logits, k, p, temp, seed, None)?;

// Grammar active: the PDA mask is fused on-GPU (fused into sampling).
let (ctrl, sp, tokens) = pda_table.fused_sample(&logits, &ctrl, &stack, &sp, &sampling)?;
```

## 3. Drafting (MTP/DFlash, GPU, fused)

The projection kernel walks K draft tokens through the PDA and emits K+1
VOB masks in one launch. K is a runtime parameter (the adaptive-k system
changes it per step based on acceptance rate), so the kernel is launched
individually rather than captured in a CUDA graph. The graph is only used
for the fixed model forward pass (attention + MLP + lm_head).

```rust
// The projection kernel: ONE launch emits K+1 VOB masks.
// Used to constrain a speculative draft: each draft position gets the
// mask of the PDA state reached after the preceding draft tokens.

let projected = table.fused_project(
    &ctrl,        // [batch] current control state
    &stack,       // [batch * D] bounded stack
    &sp,          // [batch] stack pointer
    &draft,       // [batch, K] draft tokens
)?;
// projected: [batch * (K+1) * words_per_vob] U32 VOB masks

// Convert to the DFlash allow matrix format: [batch*(K+1), vocab] F32
let allow = table.vob_to_allow(&projected, batch, k, vocab)?;

// Feed into the DFlash candidate walk (the existing masked path).
let selected = dflash_select_candidates_masked(
    &hidden, &unary_logits, &candidate_ids,
    &predecessor_codebook, &successor_codebook, &anchor_token,
    Some(&allow),  // None for unmasked DFlash
)?;
```

### The MTP Verify Path

```rust
// MTP verify: the target model scores K+1 positions in parallel.
// The PDA projection constrains each position's logits.
// If no grammar is active, the PDA projection is skipped (zero overhead).

let pda_masks: Option<&Tensor> = match pda_table {
    Some(table) => Some(&table.fused_project(&ctrl, &stack, &sp, &draft)?),
    None => None,  // No grammar: plain MTP verify
};
```

## 4. CPU<->GPU State Transfer

```
Model load:
  CPU: export_pda_package -> CudaPackage (bitvec + primitives)
  GPU: PdaPushdownTable::from_cuda_package (H2D upload, once)

Per token:
  GPU: fused_sample -> (out_ctrl, out_sp, out_tokens)
  CPU: (no round-trip needed; the state stays on GPU)

Return boundary (end of generation):
  GPU: D2H (out_ctrl, out_sp) -> CPU
  CPU: pda_validate_draft (the CPU-side check, if needed)
  CPU: sync_from_pda (re-align the Earley parser, O(delta))
```

The CPU advantage over the old DFA approach: the PDA state is
`(ctrl, stack[D], sp)` - a fixed-size tuple that fits in GPU registers.
No CPU round-trip is needed during generation. The state only crosses
the boundary at the return point (end of sequence).

## 5. The Layout-Identity Invariant

```
  pushdown-rs bitvec (CPU)
       |
       |  to_pod_bytes()
       v
  CudaPackage (the flat POD)
       |
       |  H2D upload
       v
  PdaPushdownTable (GPU)
       |
       |  fused_sample / fused_project
       v
  GPU results
       |
       |  D2H + compare to CPU PdaMachine::accepts
       v
  Layout-identity proof: GPU == CPU (same table, same algorithm)
```

The CPU `PdaMachine` is the correctness oracle for the GPU kernel.
If they disagree, the kernel is wrong (the table is proven correct by
the pushdown-rs test suite, 93/93).

## 6. CPU Batched Draft Projection (the consumer API)

For CPU-side speculative decoding (the no GPU fused kernel), a consumer
(xinfer, mistral-rs) holds a `Matcher` (or `TokenParser`) and projects a
batch of drafted tokens through the PDA in one SIMD call, then commits the
legal prefix in a single FSM step. This avoids K sequential PDA advances.

The API surface (all `#[cfg(feature = "dpda")]`):

```rust
use llguidance::Matcher;

// matcher: the per-sequence grammar matcher (the CPU authority).

// 1. Project K drafted tokens through the PDA in one SIMD call.
//    Returns (legal_count, final_config):
//      - legal_count: how many of the K drafts are grammar-legal (the prefix).
//      - final_config: Some((ctrl, stack)) when the PDA is active, None when it is not.
//    If legal_count < K, the draft diverged from the grammar at position
_count.
let (legal, _final_cfg) = matcher.project_drafts(&drafts);

// 2. Commit the legal prefix in ONE FSM step (the no legal sequential consumes).
//    try_consume_tokens returns the count actually consumed (== legal when the
//    PDA projection agreed with the Earley authority).
matcher.try_consume_tokens(&drafts[..legal])?;

// 3. (Optional) drive the SIMD pipeline directly over the static machine, e.g.
//    to compute K+1 projected masks for the draft positions in one call.
//    Read the current config frontier FIRST (it needs &mut), then yield the
//    static machine to the closure (the machine borrow is contained to the body).
let configs = matcher.pda_config()?;
matcher.with_pda_machine(|machine| {
    use pushdown_rs::simd_pipeline::SimdPipeline;
    let pipeline = SimdPipeline::new(machine);
    let projections = pipeline.project_batch(&configs, &[drafts.to_vec()]);
    // projections[0] is the K+1 VOB masks (one per draft position).
    Some(projections)
});
```

Key invariants for the consumer:
- `project_drafts` is the CLAIMANT (the fast SIMD projection); the subsequent
  `try_consume_tokens` is the ORACLE (the Earley authority). They must
  agree on the legal prefix; if they diverge, trust the oracle (the Earley) and
  resync the PDA (the `pda_config` re-read on the next step).
- `with_pda_machine` yields the static machine to a closure; the machine borrow
  is contained to the closure body (it cannot escape), so build + run the
  `SimdPipeline` inside the closure.
- When the PDA is not active (the parametric grammar, or the `dpda` feature is
  off), `project_drafts` returns `(0, None)` + `with_pda_machine` returns
  `None`; the consumer falls back to the sequential `validate_tokens` path.

## 7. File Map

| File | Crate | Role |
|------|-------|------|
| `parser/src/dpda_adapter.rs` | llguidance | The PdaGrammar wrapper + compile/export entry points |
| `parser/tests/test_dpda.rs` | llguidance | 9 accuracy proofs (structure, kappa, bitvec, CUDA, differential) |
| `parser/examples/dpda_real_tokenizer.rs` | llguidance | End-to-end: 248K tokenizer + tool-call grammar + 8 lark grammars |
| `src/machine.rs` | pushdown-rs | The PdaMachine (7-tuple, DPDA/NPDA sim, batched ops) |
| `src/compile.rs` | pushdown-rs | The Grammar trait + RTN compilation + kappa(G) |
| `src/bitvec.rs` | pushdown-rs | Lossless POD serialization |
| `src/simd.rs` | pushdown-rs | The rten-simd mask broadcast (proven bit-exact) |
| `src/cuda.rs` | pushdown-rs | The CudaPackage + FFI declarations |
| `src/oracle.rs` | pushdown-rs | The independent CFG membership oracle |
| `tests/simd_accuracy.rs` | pushdown-rs | 6 SIMD accuracy proofs |
| `benches/simd_bench.rs` | pushdown-rs | Criterion: scalar vs SIMD at 4 vocab sizes |
| `parser/matcher.rs` | llguidance | The Matcher consumer API: `project_drafts` (the batched SIMD projection), `with_pda_machine` (the SIMD pipeline yield), `pda_config` (the live frontier) |
| `src/pda.rs` | attention-rs | The PdaPushdownTable + fused_sample + fused_project + vob_to_allow |
| `src/kernels/src/pda.cu` | attention-rs | The CUDA kernel implementations |
| `src/kernels/src/ffi.rs` | attention-rs | The FFI declarations (pda_fused_sample_f32/bf16, pda_fused_project_masks) |
| `src/speculative/verify.rs` | xinfer | The pda_validate_draft (CPU-side draft check) |
| `src/speculative/dflash.rs` | xinfer | The projected_masks integration (DFlash) |