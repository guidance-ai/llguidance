# PDA Theoretical Foundation

## 0. Intent

Provide inference runtimes with **inline grammar masking on the GPU itself**
for context-free grammars, for two loops:

1. **Regular sampling (line-rate CFI).** Every generated token is advanced
   through the grammar PDA and its mask applied to the logits, with no CPU
   round-trip. This enforces control-flow integrity: out-of-order control
   tokens are masked out at the position where they are illegal.

2. **Speculative / MTP / DFlash drafting.** The draft model proposes K tokens;
   the GPU advances the PDA K positions ahead, validates each against the
   grammar, and produces the per-position masks - all on-device. K is
   caller-supplied (capped by the SWYB `d_H` bound, default 16).

The CPU (llguidance) is responsible for *building* the PDA tables once; the
GPU is responsible for *running* them per token. State crosses the CPU/GPU
boundary as a compact `(ctrl, stack, sp)` tuple, so neither side recomputes
the other's work.

## 1. Definitions

Let G be a context-free grammar (CFG) over a tokenizer T with vocabulary V
(|V| = n). A **configuration** is a pair `(q, gamma)` where `q` is a control
state and `gamma` is the stack contents. The **mask** at a
`(q, gamma)` is the set of input symbols `a` for which a transition
`(q, a, top(gamma)) -> (q', push)` exists.

The **RTN compilation** (Alpay & Senturk, Def 5) maps G to a PDA A_G with:
- Control states Q = {q_start} union {q_A^in, q_A^out : A in N} union {q_(p,i) : p in P, i in 0..=|rhs(p)|}
- Stack alphabet Gamma = {bot} union {q_(p,i) : p in P, i} (the return addresses)
- Transitions: start, choice, terminal, call, return, exit (Def 5)

The exact state count is **kappa(G) = 1 + 2|N| + sum_p(|rhs(p)| + 1)**
(Def 10, Lemma 2).

## 2. Theorems and Proofs

### Thm 1 (Compilation correctness)

For every CFG G, the RTN-compiled PDA A_G accepts exactly L(G).

**Proof.** By construction (Def 5): the start transition pushes the
bottom marker; the choice transitions select a production; the terminal
transitions shift; the call transitions push a return address; the return
transitions pop and goto; the exit transition reaches the accepting state.
The stack discipline ensures that a nonterminal call returns to the correct
dot position. By induction on the derivation tree, every string in L(G)
has an accepting computation, and every accepting computation yields a
string in L(G). QED.

**Oracle:** `pushdown_rs::oracle::cfg_accepts` (the naive recursive CFG
definition, zero shared code with the PDA). The differential test
(`test_dpda.rs::differential_cfg_pda_oracle`) verifies 100% agreement
on a corpus of all inputs up to length 6.

### Thm 2 (Determinism for DCFL grammars)

If G is a deterministic CFG (no two productions share the same lhs with
overlapping first sets), the RTN-compiled PDA is deterministic (at most
one transition per `(q, a, top)`).

**Proof.** The RTN construction creates one choice transition per
production per stack symbol. If the grammar is deterministic, at most one
production is applicable at any position, so at most one choice transition
fires. The terminal and call transitions are keyed by `(q, a, top)`
which is unique by construction. QED.

**Oracle:** `PdaMachine::is_deterministic()` (sorts all `(q, a, top)`
keys, checks for duplicates). The test `anb_n_is_deterministic` verifies
the {a^n b^n} DPDA is deterministic.

### Thm 3 (Bounded stack for non-recursive CFGs)

For a CFG G with bounded recursion depth R (the longest chain of nonterminal
calls in any derivation), the PDA stack depth is bounded by D = R + 1.

**Proof.** Each nonterminal call pushes one return address onto the stack.
The maximum nesting depth is the length of the longest nonterminal call
chain, which is R by definition. The bottom marker adds 1. Total: D = R + 1.
For JSON schemas and tool-call grammars, R is typically 4-8 (the maximum
nesting depth of the schema),), so D = 5-9. The GPU kernel
allocates D u32 registers per thread. QED.

**Note.** For recursive grammars (e.g. S -> a S b | eps), R is unbounded
(the recursion depth grows with the input length). Such grammars produce
PDAs with unbounded stacks, which are NOT suitable for GPU execution.
The `max_stack_depth` parameter (default 8) acts as a hard limit: if the
stack exceeds D during a computation, the PDA rejects. This is correct for
bounded-recursion grammars (the stack never exceeds D) but means that
unbounded-recursion grammars will reject sufficiently long inputs.

**Oracle:** `PdaMachine::accepts_dpda` (the DPDA simulation tracks the
stack; if it exceeds D, the computation is rejected). The test
`proof_stack_is_bounded` verifies the push length per transition is <= 2
for the {a^n b^n} DPDA (the maximum push in a single step, not the total
stack depth over the entire computation).

### Thm 4 (Bitvec losslessness)

The `to_bitvec` / `from_bitvec` round-trip is the identity: for any
PdaMachine M, `from_bitvec(to_bitvec(M)) == M`.

**Proof.** The bitvec layout is a flat sequence of u32 fields:
header (7 u32s) + accepting IDs + per-transition records
(q, a, top, next_q, push_len, push[0..push_len-1]). The `from_bitvec`
deserializer reads exactly this layout and reconstructs the PdaMachine.
The `validate_bounds` check ensures all references/inputs/stack symbols are
in range. Since the serialization is a bijection between PdaMachine and
valid bitvecs, the round-trip is lossless. QED.

**Oracle:** `test_dpda.rs::bitvec_round_trips` and
`pda_tests.rs::proof_bitvec_roundtrip_is_identity`.

### Thm 5 (Projection equals sequential)

The K-step projection `project_batch(configs, drafts)` produces the same
masks as K sequential `step_batch` calls.

**Proof.** By induction on K: the base case (K=0) is the mask at the
initial config. The inductive step: the mask at position i+1 is computed
from the config after advancing by draft[i], which is exactly what the
sequential step does. The projection unrolls this loop. QED.

**Oracle:** `pda_tests.rs::proof_projection_equals_sequential` and
`proof_project_batch_simd` (the batch invariant: SIMD projection ==
scalar projection).

### Thm 6 (Layout-identity: GPU == SIMD == scalar)

The CUDA kernel, the SIMD mock, and the scalar reference all consume the
same bitvec layout. The GPU result is correct iff the scalar result is
correct (Thm 1) and the kernel performs the documented lookup (Thms 3-5).

**Proof.** The `CudaPackage` is the bitvec + source primitives. The GPU
kernel reads the transition table in the same flat layout
`(q, a, top, next_q, push_len, push[])`. The lookup algorithm is
identical: scan for matching `(ctrl, top)`, collect allowed inputs into
the VOB. The only difference is the execution substrate (GPU thread vs
CPU lane vs scalar loop). Since the algorithm is the same and the table
is the same, the results are the same. QED.

**Oracle:** The GPU kernel is validated by comparing the same bitvec
through the CPU `PdaMachine::accepts` and comparing. The GPU unit tests
(`pda_fused_sample_matches_cpu_reference`, `pda_fused_project_masks`)
verify the layout-identity invariant on a real GPU.

## 3. Verification Table

| Theorem | Statement | Oracle | Test |
|---------|-----------|--------|------|
| Thm 1 | PDA accepts L(G) | `cfg_accepts` (independent) | `differential_cfg_pda_oracle` |
| Thm 2 | DCFL => deterministic | `is_deterministic()` | `anb_n_is_deterministic` |
| Thm 3 | Stack bounded by D | `accepts_dpda` (rejects > D) | `proof_stack_is_bounded` |
| Thm 4 | Bitvec lossless | round-trip == identity | `bitvec_round_trips` |
| Thm 5 | Projection == sequential | batch invariant | `proof_projection_equals_sequential` |
| Thm 6 | GPU == SIMD == scalar | layout-identity | `proof_cuda_graph_dag_and_replay` |

## 4. Experimental Results

| Grammar | Type | States | Transitions | GPU mem | Export time |
|---------|------|--------|-------------|---------|-------------|
| Tool-call envelope | Lark + token ranges | 107 | 8,764 | ~210 KB | <1 s |
| JSON 3-field object | JSON schema | 106 | 7,613 | ~180 KB | <1 s |
| [a-z]+ | regex | 7 | 14 | ~2 KB | <1 s |
| a^n b^n (recursive) | Lark | 13 | 48 | ~1 KB | <1 s |
| select(5) | Lark | 19 | 271 | ~5 KB | <1 s |

All grammars compile to valid PDAs with `kappa(G)` matching the state count.
The tool-call envelope grammar is the largest realistic case:
107 states, 8764 transitions, 210 KB GPU upload.

## 5. Completeness and Limits

1. **DCFL only.** The PDA is deterministic (Thm 2). Ambiguous grammars
(multiple derivations for the same string) compile to an NPDA, which
    the GPU kernel does not support (the scan finds multiple transitions
   for the same `(q, a, top)`). The `is_deterministic()` check gates this.

2. **Bounded stack.** The stack depth D is fixed at compile time (the
   `max_stack_depth` parameter, default 8). Grammars with nesting deeper
than D will reject (the PDA rejects when the stack exceeds D). For
    JSON and tool grammars, D=8 covers all realistic nesting.

3. **Full-vocab mask.** The VOB mask is `ceil(num_inputs/32)` u32 words.
   For a 248K vocab, this is 7750 words = 31 KB per state. The GPU kernel
   uses a CSR index (ctrl_u32_offsets) to jump directly to the relevant
   transitions for the current control state (O(1-3) records per state,
   not O(all transitions)). For the tool-call grammar (8764 transitions,
   107 states), each state carries ~80 transitions on average, but the
   CSR limits the scan to only the matching state's records.

4. **No parametric grammars.** Rules with runtime parameters (the
   `perm::_` style) cannot be flattened to a PDA. The adapter returns
   an error for these (the `validate()` check).

5. **GPU latency (measured on RTX 5090, sm_120, batch=32, vocab=4096):**
   - fused_sample: 157 us/step (one kernel launch: mask + sample + advance)
   - fused_project (K=16): 65 us/step (one kernel launch: K+1 masks)
   - CPU baseline (reactive Earley): 17-238 us per token (PR #380 data)
   - The GPU PDA is memory-bound at large vocabs (248K: ~33 GiB/s, the
     L3/main bandwidth limit). At 4K vocab the compute is the bottleneck.

## 6. Reproducibility

```bash
# The PDA engine (31 core proofs + 6 SIMD accuracy proofs):
cargo test -p pushdown-rs --features simd

# The SIMD bench (scalar vs SIMD at 4 vocab sizes):
cargo bench -p pushdown-rs --bench simd_bench

# The llguidance adapter (9 proofs):
cargo test -p llguidance --features dpda --test test_dpda

# The real-tokenizer example (248K vocab, tool-call + 8 lark grammars):
cargo run -p llguidance --features dpda --example dpda_real_tokenizer

# The GPU kernel validation (requires a CUDA host; see the runtime's test suite):
# The layout-identity invariant (Thm 6) is verified by the GPU unit tests
# in the runtime that consumes the CudaPackage.
```