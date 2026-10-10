// PDA GPU kernels for grammar-constrained sampling and drafting.
//
// Extracted from the attention-rs pda.cu (the fused_sample / fused_project
// kernels). These are self-contained: they only depend on the CUDA runtime
// + curand. Copy them into your runtime's kernel directory and call them
// with your PDA table (the pushdown-rs CudaPackage bitvec upload).
//
// The PDA table format (the flat u32 array):
//   Each record: (q, a, top, next_q, push_len, push[0..push_len-1])
//   Total array length = sum over all records of (5 + push_len) u32s.
//
// The CSR index (the ctrl_offsets + ctrl_counts):
//   ctrl_offsets[q] = the u32 offset into the flat array where q's records begin
//   ctrl_counts[q]  = the number of records for state q
//   The kernel uses the CSR to scan only q's records (O(1-3) instead of O(total)).
//
// The:
//   1. Upload the PDA table (the transitions + the accepting + the CSR) to GPU memory
//   2. For sampling: call pda_fused_sample (the mask + sample + advance in one launch)
//   3. For drafting: call pda_fused_project (the K+1 masks in one launch)
//   4. The PDA config (the ctrl + stack + sp) is updated in-place on the GPU
//
// The PDA dispatch is optional: if ctrl is null, the kernel runs plain
// sampling with zero PDA overhead (the no grammar constraint).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>

// ---------------------------------------------------------------------------
// Helper: scan the transition table for (ctrl, top) and emit the VOB mask.
// The epsilon-closure BFS over the (state, top) configs, collecting the
// terminal inputs from every reachable config. Matches the CPU mask_at_cfg.
// ---------------------------------------------------------------------------
__device__ void scan_mask(
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ ctrl_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    uint32_t num_inputs,
    uint32_t ctrl,
    uint32_t top,
    uint32_t* __restrict__ out_vob,
    uint32_t words_per_vob)
{
    for (uint32_t w = 0; w < words_per_vob; w++) {
        out_vob[w] = 0;
    }
    // The BFS over the (state, top) configs (the epsilon closure).
    const int MAXF = 64;
    uint32_t fq[MAXF], ft[MAXF];
    int head = 0, tail = 0;
    fq[tail] = ctrl; ft[tail] = top; tail++;
    while (head < tail) {
        uint32_t cq = fq[head], ctop = ft[head]; head++;
        size_t idx = ctrl_offsets[cq];
        uint32_t count = ctrl_counts[cq];
        for (uint32_t i = 0; i < count; i++) {
            uint32_t a = transitions[idx + 1];
            uint32_t t = transitions[idx + 2];
            uint32_t next_q = transitions[idx + 3];
            uint32_t push_len = transitions[idx + 4];
            if (t == ctop) {
                if (a < num_inputs) {
                    // The terminal move: collect the input.
                    uint32_t word_idx = a / 32;
                    if (word_idx < words_per_vob) {
                        out_vob[word_idx] |= (1u << (a % 32));
                    }
                } else {
                    // The epsilon move: follow it (the new_top = the push[0],
                    // the empty push -> the same top).
                    uint32_t new_top = (push_len > 0) ? transitions[idx + 5] : ctop;
                    bool dup = false;
                    for (int v = 0; v < tail; v++) {
                        if (fq[v] == next_q && ft[v] == new_top) { dup = true; break; }
                    }
                    if (!dup && tail < MAXF) {
                        fq[tail] = next_q; ft[tail] = new_top; tail++;
                    }
                }
            }
            idx += 5 + push_len;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: advance the PDA by a token (the epsilon-closure, then no stuck call
// dots). Matches the CPU advance_eps. Returns (next_ctrl, new_top, push_len);
// holds (ctrl, top, 0) if no terminal move is reachable (the reject path).
// ---------------------------------------------------------------------------
__device__ void advance_pda(
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ ctrl_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    uint32_t num_inputs,
    uint32_t ctrl,
    uint32_t top,
    uint32_t token,
    uint32_t* out_ctrl,
    uint32_t* out_top,
    uint32_t* out_push_len,
    const uint32_t** out_push_base,
    uint32_t* out_moved)
{
    *out_ctrl = ctrl;
    *out_top = top;
    *out_push_len = 0;
    if (out_push_base) *out_push_base = nullptr;
    if (out_moved) *out_moved = 0;
    // The BFS over the (state, top) configs (the epsilon closure).
    const int MAXF = 64;
    uint32_t fq[MAXF], ft[MAXF];
    int head = 0, tail = 0;
    fq[tail] = ctrl; ft[tail] = top; tail++;
    while (head < tail) {
        uint32_t cq = fq[head], ctop = ft[head]; head++;
        size_t idx = ctrl_offsets[cq];
        uint32_t count = ctrl_counts[cq];
        for (uint32_t i = 0; i < count; i++) {
            uint32_t a = transitions[idx + 1];
            uint32_t t = transitions[idx + 2];
            uint32_t next_q = transitions[idx + 3];
            uint32_t push_len = transitions[idx + 4];
            if (t == ctop) {
                if (a == token) {
                    // The terminal move found.
                    *out_ctrl = next_q;
                    *out_push_len = push_len;
                    *out_top = (push_len > 0) ? transitions[idx + 5] : ctop;
                    if (out_push_base) *out_push_base = (push_len > 0) ? (transitions + idx + 5) : nullptr;
                    if (out_moved) *out_moved = 1;
                    return;
                } else if (a == num_inputs) {
                    // The epsilon move: follow it.
                    uint32_t new_top = (push_len > 0) ? transitions[idx + 5] : ctop;
                    bool dup = false;
                    for (int v = 0; v < tail; v++) {
                        if (fq[v] == next_q && ft[v] == new_top) { dup = true; break; }
                    }
                    if (!dup && tail < MAXF) {
                        fq[tail] = next_q; ft[tail] = new_top; tail++;
                    }
                }
            }
            idx += 5 + push_len;
        }
    }
    // The no terminal move in the closure: hold (the reject path).
}

// ---------------------------------------------------------------------------
// The fused sample kernel (the template on the logit dtype): the mask +
// the sample (the temperature + the top-k + the random) + the advance in
// one launch. One thread per sequence. If ctrl is null, the plain sampling
// (the no PDA overhead).
// ---------------------------------------------------------------------------
template <typename T>
__device__ float pda_logit_to_float(const T& x) {
    return (float)x;
}
template <>
__device__ float pda_logit_to_float<__nv_bfloat16>(const __nv_bfloat16& x) {
    return __bfloat162float(x);
}

template <typename T>
__global__ void pda_fused_sample_kernel(
    const T* __restrict__ logits,
    const uint32_t* __restrict__ ctrl,
    uint32_t* __restrict__ stack,
    const uint32_t* __restrict__ sp,
    uint32_t* __restrict__ out_ctrl,
    uint32_t* __restrict__ out_sp,
    uint32_t* __restrict__ out_tokens,
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ accepting,
    const uint32_t* __restrict__ ctrl_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    uint32_t* __restrict__ vob_buf,
    const uint32_t* __restrict__ forbid,
    uint32_t num_states,
    uint32_t num_stack_syms,
    uint32_t num_inputs,
    uint32_t words_per_vob,
    int batch,
    int vocab,
    int top_k,
    float temperature,
    float top_p,
    uint64_t seed,
    int d)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    const T* logit_row = logits + (size_t)seq * vocab;

    // The mask (the no PDA, the all-allowed).
    uint32_t* vob = vob_buf + (size_t)seq * words_per_vob;
    uint32_t c = 0, s = 1, top = 0;
    if (ctrl != nullptr) {
        c = ctrl[seq];
        if (c >= num_states) c = 0;
        s = (sp != nullptr) ? sp[seq] : 1;
        top = (s > 0 && stack != nullptr) ? stack[(size_t)seq * d + s - 1] : 0;
        scan_mask(transitions, ctrl_offsets, ctrl_counts, num_inputs, c, top, vob, words_per_vob);
        // The anti-loop kick (the H2): the clear the forbid bit (the safety floor,
        // the popcount > 1, the no dead-end).
        if (forbid != nullptr) {
            uint32_t f = forbid[seq];
            if (f != UINT32_MAX && f < num_inputs) {
                uint32_t popcount = 0;
                for (uint32_t w = 0; w < words_per_vob; w++) popcount += __popc(vob[w]);
                if (popcount > 1) {
                    uint32_t word_idx = f / 32;
                    if (word_idx < words_per_vob) vob[word_idx] &= ~(1u << (f % 32));
                }
            }
        }
    } else {
        for (uint32_t w = 0; w < words_per_vob; w++) vob[w] = 0xFFFFFFFF;
    }

    // The sampling (the M1): the temperature + the top-k + the random (the curand).
    int max_idx;
    if (top_k <= 1 && temperature <= 0.0f) {
        // The greedy (the argmax over the masked logits).
        float max_val = -FLT_MAX;
        max_idx = 0;
        for (int i = 0; i < vocab; i++) {
            float v = pda_logit_to_float(logit_row[i]);
            uint32_t word_idx = i / 32;
            if (word_idx < words_per_vob && (vob[word_idx] & (1u << (i % 32))) == 0) v = -FLT_MAX;
            if (v > max_val) { max_val = v; max_idx = i; }
        }
    } else {
        // The top-k + the curand.
        float kth = -FLT_MAX;
        if (top_k > 1) {
            int removed[64]; int removed_n = 0;
            for (int n = 0; n < top_k && n < 64; n++) {
                float cur_max = -FLT_MAX; int cur_idx = -1;
                for (int i = 0; i < vocab; i++) {
                    bool rem = false;
                    for (int r = 0; r < removed_n; r++) if (removed[r] == i) { rem = true; break; }
                    if (rem) continue;
                    float v = pda_logit_to_float(logit_row[i]);
                    uint32_t word_idx = i / 32;
                    if (word_idx < words_per_vob && (vob[word_idx] & (1u << (i % 32))) == 0) continue;
                    if (v > cur_max) { cur_max = v; cur_idx = i; }
                }
                if (cur_idx < 0) break;
                if (n == top_k - 1) kth = cur_max;
                if (removed_n < 64) removed[removed_n++] = cur_idx;
            }
        }
        float max_logit = -FLT_MAX;
        for (int i = 0; i < vocab; i++) {
            float v = pda_logit_to_float(logit_row[i]);
            uint32_t word_idx = i / 32;
            if (word_idx < words_per_vob && (vob[word_idx] & (1u << (i % 32))) == 0) continue;
            if (top_k > 1 && v < kth) continue;
            if (v > max_logit) max_logit = v;
        }
        float sum = 0.0f;
        for (int i = 0; i < vocab; i++) {
            float v = pda_logit_to_float(logit_row[i]);
            uint32_t word_idx = i / 32;
            if (word_idx < words_per_vob && (vob[word_idx] & (1u << (i % 32))) == 0) continue;
            if (top_k > 1 && v < kth) continue;
            float scaled = (temperature > 0.0f) ? (v - max_logit) / temperature : (v - max_logit);
            sum += expf(scaled);
        }
        curandStatePhilox4_32_10_t rng;
        curand_init(seed, (unsigned int)seq, 0, &rng);
        float u = curand_uniform(&rng);
        float cumsum = 0.0f;
        max_idx = vocab - 1;
        for (int i = 0; i < vocab; i++) {
            float v = pda_logit_to_float(logit_row[i]);
            uint32_t word_idx = i / 32;
            if (word_idx < words_per_vob && (vob[word_idx] & (1u << (i % 32))) == 0) continue;
            if (top_k > 1 && v < kth) continue;
            float scaled = (temperature > 0.0f) ? (v - max_logit) / temperature : (v - max_logit);
            cumsum += expf(scaled);
            if (u * sum <= cumsum) { max_idx = i; break; }
        }
    }
    out_tokens[seq] = max_idx;

    // The advance (the PDA, the no PDA, the no advance).
    if (ctrl != nullptr) {
        uint32_t next_ctrl, next_top, push_len;
        const uint32_t* push_base = nullptr;
        advance_pda(transitions, ctrl_offsets, ctrl_counts, num_inputs, c, top, (uint32_t)max_idx,
                    &next_ctrl, &next_top, &push_len, &push_base, nullptr);
        out_ctrl[seq] = next_ctrl;
        if (out_sp != nullptr) {
            uint32_t new_sp = s;
            if (new_sp > 0) new_sp--;
            new_sp += push_len;
            if (stack != nullptr && push_base != nullptr) {
                uint32_t base_pos = (s > 0) ? s - 1 : 0;
                for (uint32_t j = 0; j < push_len && base_pos + j < (uint32_t)d; j++) {
                    stack[(size_t)seq * d + base_pos + j] = push_base[j];
                }
            }
            out_sp[seq] = new_sp;
        }
    }
}

// ---------------------------------------------------------------------------
// The fused project kernel: walk K draft tokens, emit K+1 VOB masks.
// One thread per sequence. The anti-loop kick (the anchor bit) is applied
// to the anchor mask ( (the pos 0), not the draft positions.
// ---------------------------------------------------------------------------
__global__ void pda_fused_project_kernel(
    const uint32_t* __restrict__ ctrl,
    const uint32_t* __restrict__ stack,
    const uint32_t* __restrict__ sp,
    const uint32_t* __restrict__ drafts,
    uint32_t* __restrict__ out_masks,
    const uint32_t* __restrict__ transitions,
    const uint32_t* __restrict__ accepting,
    const uint32_t* __restrict__ ctrl_offsets,
    const uint32_t* __restrict__ ctrl_counts,
    const uint32_t* __restrict__ forbid,
    uint32_t num_states,
    uint32_t num_stack_syms,
    uint32_t num_inputs,
    uint32_t words_per_vob,
    int batch,
    int k,
    int d)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    uint32_t c = ctrl[seq];
    if (c >= num_states) c = 0;
    uint32_t s = (sp != nullptr) ? sp[seq] : 1;
    uint32_t top = (s > 0 && stack != nullptr) ? stack[(size_t)seq * d + s - 1] : 0;

    const uint32_t* draft_row = drafts + (size_t)seq * k;

    for (int pos = 0; pos <= k; pos++) {
        // Emit the mask at the current (c, top).
        uint32_t* out = out_masks + ((size_t)seq * (k + 1) + pos) * words_per_vob;
        scan_mask(transitions, ctrl_offsets, ctrl_counts, num_inputs, c, top, out, words_per_vob);

        // The anti-loop kick (the anchor only, the pos 0): clear the forbid bit
        // in the anchor mask (the no draft-triggering the loop at the draft root).
        // The safety floor (the popcount > 1) prevents a dead-end.
        if (pos == 0 && forbid != nullptr) {
            uint32_t f = forbid[seq];
            if (f != UINT32_MAX && f < num_inputs) {
                uint32_t popcount = 0;
                for (uint32_t w = 0; w < words_per_vob; w++) {
                    popcount += __popc(out[w]);
                }
                if (popcount > 1) {
                    uint32_t word_idx = f / 32;
                    if (word_idx < words_per_vob) {
                        out[word_idx] &= ~(1u << (f % 32));
                    }
                }
            }
        }

        if (pos == k) break;

        // Advance by draft[pos] (the epsilon-closure). Break on divergence (the no
        // terminal move, matching the CPU project_batch).
        uint32_t tok = draft_row[pos];
        uint32_t next_c, next_top, next_push_len, moved;
        advance_pda(transitions, ctrl_offsets, ctrl_counts, num_inputs, c, top, tok,
                    &next_c, &next_top, &next_push_len, nullptr, &moved);
        if (!moved) break; // the draft diverged (the no transition)
        c = next_c;
        top = next_top;
    }
}

// ---------------------------------------------------------------------------
// The extern "C" entry points (the host-side launchers).
// ---------------------------------------------------------------------------
extern "C" {

void pda_fused_sample_f32(
    const float* logits,
    const uint32_t* ctrl, uint32_t* stack, const uint32_t* sp,
    uint32_t* out_ctrl, uint32_t* out_sp, uint32_t* out_tokens,
    const uint32_t* transitions, const uint32_t* accepting,
    const uint32_t* ctrl_offsets, const uint32_t* ctrl_counts,
    uint32_t* vob_buf,
    const uint32_t* forbid,
    uint32_t num_states, uint32_t num_stack_syms, uint32_t num_inputs,
    uint32_t words_per_vob, int batch, int vocab,
    int top_k, float temperature, float top_p, uint64_t seed, int d, int64_t stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    pda_fused_sample_kernel<float><<<blocks, threads, 0, s>>>(
        logits, ctrl, stack, sp, out_ctrl, out_sp, out_tokens,
        transitions, accepting, ctrl_offsets, ctrl_counts, vob_buf, forbid,
        num_states, num_stack_syms, num_inputs,
        words_per_vob, batch, vocab, top_k, temperature, top_p, seed, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[pda_fused_sample_f32] launch error: %s\n", cudaGetErrorString(err));
    }
}

void pda_fused_sample_bf16(
    const void* logits,
    const uint32_t* ctrl, uint32_t* stack, const uint32_t* sp,
    uint32_t* out_ctrl, uint32_t* out_sp, uint32_t* out_tokens,
    const uint32_t* transitions, const uint32_t* accepting,
    const uint32_t* ctrl_offsets, const uint32_t* ctrl_counts,
    uint32_t* vob_buf,
    const uint32_t* forbid,
    uint32_t num_states, uint32_t num_stack_syms, uint32_t num_inputs,
    uint32_t words_per_vob, int batch, int vocab,
    int top_k, float temperature, float top_p, uint64_t seed, int d, int64_t stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    pda_fused_sample_kernel<__nv_bfloat16><<<blocks, threads, 0, s>>>(
        (const __nv_bfloat16*)logits, ctrl, stack, sp, out_ctrl, out_sp, out_tokens,
        transitions, accepting, ctrl_offsets, ctrl_counts, vob_buf, forbid,
        num_states, num_stack_syms, num_inputs,
        words_per_vob, batch, vocab, top_k, temperature, top_p, seed, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[pda_fused_sample_bf16] launch error: %s\n", cudaGetErrorString(err));
    }
}

void pda_fused_project_masks(
    const uint32_t* ctrl, const uint32_t* stack, const uint32_t* sp,
    const uint32_t* drafts, uint32_t* out_masks,
    const uint32_t* transitions, const uint32_t* accepting,
    const uint32_t* ctrl_offsets, const uint32_t* ctrl_counts,
    const uint32_t* forbid,
    uint32_t num_states, uint32_t num_stack_syms, uint32_t num_inputs,
    uint32_t words_per_vob, int batch, int k, int d, int64_t stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    pda_fused_project_kernel<<<blocks, threads, 0, s>>>(
        ctrl, stack, sp, drafts, out_masks,
        transitions, accepting, ctrl_offsets, ctrl_counts, forbid,
        num_states, num_stack_syms, num_inputs,
        words_per_vob, batch, k, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[pda_fused_project_masks] launch error: %s\n", cudaGetErrorString(err));
    }
}

} // extern "C"

// ---------------------------------------------------------------------------
// Integration notes:
//
// 1. The PDA table (the transitions + the accepting + the CSR) is uploaded
//    once at model load (the pushdown-rs CudaPackage, the bitvec H2D).
//    The ctrl_offsets + ctrl_counts are the CSR index (the O(1-3) lookup
//    per state, computed from the flat transition array).
//
// 2. The PDA config (the ctrl + stack + sp) is maintained on the GPU
//    (the out_ctrl / out_sp / the in-place stack update). It persists
//    across decode steps (no CPU round-trip).
//
// 3. The forbid parameter is the anti-loop kick (the H2): the token ID to
//    exclude from the mask (the repetition detector's trigger token). The
//    safety floor (the popcount > 1) prevents a dead-end (the no mask with
//    zero allowed tokens).
//
// 4. The drafts parameter (the fused_project) is the K draft tokens from
//    the MTP/DFlash drafter. The kernel emits K+1 masks (the anchor + the
//    K draft positions). If the draft diverges (the no terminal move), the
//    kernel breaks early (the fewer masks, the legal prefix).
//
// 5. The kernel is optional: if ctrl is null, the fused_sample runs plain
//    sampling (the no PDA overhead). This fused_project is skipped entirely.
// ---------------------------------------------------------------------------