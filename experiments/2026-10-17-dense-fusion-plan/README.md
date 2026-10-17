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

## Step 1a: the chain on the sliced config, and the glue attributed (no node)

From the 2026-10-11 nsys-sol reports (sliced, vllm 80bd3cce5c, c=1 M=8,
`../2026-10-11-nsys-sol/an/kc-c1-w2-d*.txt`, median of layers 20-40, all four
ranks agree to ~0.1 us), "gap" = start minus the previous kernel's end:

| kernel | us | gap |
|---|--:|--:|
| AR + residual + RMSNorm (lamport) | 7.9-9.1 | |
| decode_gemm fused_qkv_a (164 CTAs) | 12.9 | -0.1 (overlaps the AR) |
| triton_poi_fused_0 (1 CTA) | 1.50 | 0.4 |
| triton_red_fused_1 (8 CTAs) | 2.05 | 0.4 |
| decode_gemm q_b (256 CTAs) | 7.84 | 0.1 |
| ... rope, 2x concat_and_cache, W_UK, gather_cat, FlashMLA, combine, lse_rs | | |
| nvjet W_UV | 3.97 | 0.2 |
| decode_gemm o_proj (384 CTAs) | 16.8 | 0.1 (no overlap) |

Inductor source (`caches/marlin/vllm-cache-glm53-nsol-c1-2272145/
torch_compile_cache/148ba15622/rank_0_0/backbone/artifact_compile_range_1_170_subgraph_12`,
strings) for the graph `qkv_a -> split [2048, 576] -> rms_norm(q_c) -> q_b;
split [512, 64] -> rms_norm(kv_c)`:

- `triton_poi_fused_0` is the **k_pe RoPE** (reads columns 2560-2623 of the
  2624-stride qkv_a output, cos/sin cache by position). q_b does not need it;
  inductor just orders it first.
- `triton_red_fused_1` is a combo kernel: the kv_a RMSNorm (512) and the q_a
  RMSNorm (2048), both reading the strided qkv_a output directly (no copy
  kernel exists). Arithmetic: fp32 sum of squares (2048: R0_BLOCK-wide partial
  sums over a loop, then tl.sum), / n, + 1e-5, rsqrt, x * r in fp32, * weight
  in fp32, one bf16 rounding at the store.
- Neither Triton kernel releases dependents: 0.4 us gaps after them vs
  0.1 / -0.1 after the PDL kernels.

So the glue between qkv_a and q_b is 1.5 + 2.05 + ~0.9 us of gaps = ~4.4 us
per layer (~0.34 ms/step at c=1, 1.8% of 18.8 ms), of which the RoPE does not
belong on that path at all. And W_UV -> o_proj shows no overlap, as Astra said.

## Step 1b: rebaseline on current prod (holds 2375889, 2375891)

`rebase.sh <hold> <c>`: the 2026-10-11 nsys arm (`../2026-10-11-nsys-sol/nsys_arm.sh`)
on frozen tree `worktrees/pf5` (vllm 144f2f4aee), c=1 (DFlash2 k=7, M=8) and
c=4 (MTP3, M=16). Reports `/e/fscratch/profound/$USER/sliced-prof/dfp-c{1,4}-*/`,
chains `an/kc-c{1,4}-w{1,2}-d{0-3}.txt` (`../2026-10-09-tp-sliced-experts/kchain_nsys.py`).
Step period 18.88 ms (c=1, ~50K) / 27.80 (c=4). The dense chain is unchanged
from 10-11 to within ~0.1 us on all four ranks:

| median us (layers 20-40, rank 0) | c=1 | c=4 |
|---|--:|--:|
| AR + residual + RMSNorm | 7.49 | 9.44 |
| fused_qkv_a decode_gemm | 12.80 | 13.73 |
| k_pe RoPE (triton_poi_fused_0) | 1.47 (+0.4 gap) | 1.38 (+0.4) |
| q_a/kv_a RMSNorm (triton_red_fused_1) | 2.05 (+0.4) | 2.02 (+0.4) |
| q_b decode_gemm | 7.78 (+0.1) | 8.51 (+0.1) |
| W_UV nvjet | 3.97 | 4.35 |
| o_proj decode_gemm | 16.64 (+0.1) | 17.44 (+0.2) |

## Step 2: glue.cu prototype (one PDL-aware kernel for both norms + k_pe RoPE)

`glue.cu`: one CTA per token; `griddepcontrol.launch_dependents` at entry (q_b
starts its first weight batch), `griddepcontrol.wait` before reading qkv_a's
output; same arithmetic as the inductor kernels (fp32 sum of squares, / n,
+ eps, rsqrtf, x * r * w in fp32, one bf16 rounding; RoPE in fp32).
`bench_qchain.py`: correctness vs eager and inductor-compiled references, then
the chain qkv_a -> glue -> q_b in one CUDA graph of 64 chains over 8 rotating
weight sets (cold L2), decode_gemm with the served configs.

Correctness: at M = 8 and 16 bitwise equal to both references. Over 4096 rows
with varied scales, 23 of 8.4M q_c and 9 of 2.1M kv_c elements differ by one
bf16 ulp (the 2048/512-wide sum order differs from Triton's); k_pe exact.
Not bitwise identical in general: matching Triton's reduction tree (its
autotuned R0_BLOCK + tl.sum order) would be needed for that.

Timing (`logs/qchain-*.log`, GPU 0 of both nodes, they agree to 0.05 us;
us per chain, median of 5 x 20 replays):

| M | inductor glue | glue.cu PDL | glue.cu no PDL | floor (no glue) |
|---|--:|--:|--:|--:|
| 8 | 21.73 | **20.21** | 21.10 | 18.20 |
| 16 | 23.39 | **21.81** | 22.63 | 19.70 |

-1.5 us/chain (~0.6 from one kernel instead of two, ~0.9 from PDL). Over 78
layers ~0.12 ms/step at c=1 (~0.6% of 18.9 ms), ~0.12 at c=4 (0.4%); 2.0 us of
glue cost remains over the floor (the glue's own latency, which q_b's
griddepcontrol.wait must sit out). Not integrated; no served A/B.
