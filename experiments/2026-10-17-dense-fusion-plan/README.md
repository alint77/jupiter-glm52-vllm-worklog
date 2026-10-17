# Dense decode GEMMs: fuse into their predecessors? (plan, 2026-10-17)

Paper plan only (no nodes). Per-layer main-stream chain at c=1, M=8, from
`../2026-10-08-decode-dive-3/dive4-win2.txt` (EP-era trace; MoE and DCP tail
have changed since, the dense chain should not have):

| GEMM | predecessor |
|---|---|
| o_proj (decode_gemm 16.5 us) | W_UV bmm (nvjet, 3.9) after the DCP tail |
| fused_qkv_a (12.5) | MoE all-reduce + residual + RMSNorm (FlashInfer lamport) |
| q_b (7.6) | triton_poi_fused_1 (1.5) + triton_red_fused_2 (2.2), unattributed; likely q_c copy and q_a/kv_a RMSNorm |
| indexer wq_b (side stream) | the same normalized q_c |

q-chain qkv_a start -> q_b end 24.7 us vs 13.5 us weight floor; W_UV start ->
o_proj end 20.6 us vs 15.1 us (54.5 MB both weights).

First plan (given to the user): (1a) norm into q_b's prologue, (1b) qkv_a + q_b
in one persistent queue-ordered kernel, (2) W_UV into o_proj's prologue, skip
(3) AR+RMSNorm -> qkv_a; claimed ~0.6-0.8 ms/step.

Astra review (`astra/review.md`, gpt-6-astra xhigh, prompt `astra/review-prompt.md`):

- (2) is wrong as stated: decode_gemm CTAs own 16 rows over the whole K = 16
  heads x 256, so each of 384 CTAs would recompute all of W_UV (~1.5 GiB L2
  reads). Head-split o_proj still recomputes W_UV per (tile, head). If at all:
  W_UV producer tasks once, bf16 output, full-K o_proj consumers. Deferred.
- 0.6-0.8 ms is an unvalidated target: the 3.3 us streaming intercept is not a
  removable per-boundary cost, the glue kernels' 3.7 us carries required work
  (kv norm, contiguous q_c for the indexer), 1a and 1b overlap.
- The trace shows no overlap at norm -> q_b or W_UV -> o_proj although both
  consumers are PDL-launched: whether the Triton/nvjet predecessors release
  dependents (and the graph keeps the edge) is unknown and cheap to check.
- 1b deadlock argument holds only if producers publish completion
  (device-scope release/acquire) before claiming again and never wait;
  counters reset per graph replay; side-stream indexer needs a real stream
  dependency on normalized q_c. 16.8 MB "fits in smem" is not a staging plan.
- Bitwise norm identity needs the generated Triton's eps/casts/reduction tree,
  not just the formula; W_UV must stay bf16-rounded before o_proj.
- decode_gemm fallbacks are per shape: q_b above M=16, qkv_a at M=32, o_proj
  keeps decode_gemm to 32; all above MAX_M=32 (NT stops at 4).

Revised order: (1) Booster rebaseline on current prod: attribute the two Triton
kernels, strides, side-stream events, PDL on captured edges, whole-step across
ranks; (2) one strided-input q/kv norm kernel writing contiguous normalized q_c,
PDL-releasing q_b; (3) then norm-inside-q_b vs that, or M=64 GEMM tuning for
c>=8; (4) persistent q-chain only if a measured residual remains; (5) o-chain
deferred. Every saving as an incremental whole-step delta.
