# In-graph cold prefetch (GLM-5.3, prefill graphs to 1024)

Follows ../2026-09-30-glm-prefill-prefetch (plan section there).

## Capture crash (e3a4143956) -- root cause

`server-*.out` (not `.err`) had the worker traceback:
`cudaErrorStreamCaptureIsolation` in `join_captured`. Piecewise capture makes
one graph per layer; the last MoE layer forks no copy but still joined the copy
stream, which was outside its capture. Fixed in daeecd8bdf (join only after a
fork); the capture test now captures one graph per layer and fails without it.

## Same-node A/B (jpbo-001-44, job 2138596), median TTFT ms +- std, 20 prompts

| arm | 512 | 768 | 1024 |
|---|---|---|---|
| nopf (threshold 1025) | 176 +-3 | 236 +-12 | 240 +-12 |
| inmoe (daeecd8: fork at MoE(L), join after it) | 143 +-1 | 174 +-12 | 203 +-10 |
| wide (fork before o_proj(L), join before attn(L+1), first MoE layer under the dense layer) | 135 +-28 | 172 +-7 | 205 +-13 |

- Prefetch in graphs: -33 to -64 ms. Widening the window: within noise; the
  copy is already mostly hidden under MoE(L) at >=512 tokens.
- Greedy text is not a correctness check here: identical no-prefetch configs
  on two nodes (agentic-bench/greedy-gp-nopf-n{1,2}) already differ.
  Correctness: VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY now also runs each
  captured MoE against Grace and keeps max |diff| on the device, reported by
  the next eager chunk (launch_verify.sh).

## Trace: whole-piece design (trace.sh wide, analyze_trace.py), 4 ranks

One traced prefill each at 512/768/1024 new tokens. Graph replays put every
node on the launch stream; copies are the >=5 MB HtoD at ~400 GB/s.

| tokens | copy L+1 (us, median) | MoE(L) (us, median) | layers where copy > MoE(L) | join stalls (layers, total) |
|---|---|---|---|---|
| 512 | 1350-1400 (~510 MB) | 950-1080 | 53-62 / 73 | 10-17, 1.1-2.7 ms |
| 768 | same | 1160-1310 | 34-41 / 73 | 0-2, <20 us |
| 1024 | same | 1430-1600 | 9-27 / 73 | 0 |

- The MoE-only window is too short for most layers at 512, half at 768, and a
  minority at 1024; the wider window hides all of it from 768 up, and at 512
  still stalls 1-3 ms per prefill (forward ~124 ms first-to-last MoE).
- Dense-layer copy (first MoE layer): 320-440 us, no stall.

## 512-token prefill breakdown (breakdown.py, rank 0 unless noted)

- GPU window 136.7 ms = setup 5.8 + embed/dense layers/first attention 6.3 +
  75 MoE layers 123.9 (1.65 ms/layer) + tail 0.7. Host step 85 ms: GPU-bound
  except setup and the dense part.
- Idle 10.7 ms (8%), 7.6 ms of it host-bound, almost all in the first 12 ms
  (input prep, metadata, dense layers, DCP top-k all-gathers); a few 100-230 us
  join stalls inside the MoE region.
- Not idle but wasted: the post-MoE all-reduce waits for the slowest rank. MoE
  end spread across ranks median 337 us/layer, 26.5 ms over the forward; last
  rank varies (r1 27, r2 25, r0 15, r3 8 layers). Post-MoE AR total 12-22 ms/rank.
- Kernel time: Marlin 109 ms (hot and cold tiers overlap on two streams; MoE
  ~0.92 ms/layer for ~1.36 GB of expert weights/rank, ~1.5 TB/s), collectives
  29.6, dense GEMMs 10.5, MoE act 6.2, attention 3.4, indexer 2.1.
- Layer 40: AR 436 us (mostly waiting) -> norm, q/kv GEMMs, cache write, flash
  attention ~40 us -> o_proj -> AR 46 us -> router -> MoE 746..1667 us; cold
  copy for L+1 599..1941 us (starts before o_proj, ends ~270 us after MoE).

## MoE roofline (roofline.py; 630 TFLOPS bf16, 3.6 TB/s HBM; rank 0)

Per rank per layer: 64 experts x 21.2 MB = 1.36 GB of W4A16 weights (group
32). FLOPs/route 75.5 M. Marlin block_size_m 32 (512, 768) / 48 (1024).

| tokens | useful GFLOP | padded GFLOP | mem SOL | compute SOL useful/padded | measured span | HBM BW | useful / padded TFLOPS |
|---|---|---|---|---|---|---|---|
| 512 | 77 | 155 | 377 us | 123 / 245 us | 985 us | 1.38 TB/s (38%) | 78 (12%) / 157 (25%) |
| 768 | 116 | 155 | 377 | 184 / 245 | 1209 | 1.12 (31%) | 96 (15%) / 128 (20%) |
| 1024 | 155 | 232 | 377 | 245 / 368 | 1424 | 0.95 (27%) | 109 (17%) / 163 (26%) |

- Below both roofs. The span grows ~0.86 us/token with constant bytes:
  marginal ~176 TFLOPS useful, intercept ~545 us (~2.5 TB/s with the cold
  copy's 0.53 GB of writes). Memory and compute look additive, not overlapped.
- Marlin launch: 128 threads/block, 1-2 blocks/SM (4-8 warps/SM, 208 regs,
  est. occupancy 6-13%); hot and cold tiers run concurrently on two streams,
  each with a full-GPU persistent grid (132 or 264).
- Hot/cold attribution by launch order is ambiguous (correlations with cold
  expert count +-0.5). Next: ncu on a standalone Marlin MoE at these shapes
  (dram throughput, tensor pipe active, stall reasons) to name the limiter.

## Isolated Marlin MoE (benchmark_moe_wna16_marlin_prefill.py, marlin_bench.sh)

One tier, 64 experts in HBM, uniform routing, CUDA graph + profiler, jpbo-056-24.
Raw: fscratch/ingraph-prefetch/marlin/{bench.txt,marlin.ncu-rep,raw.csv}.

| M | bm | pad | w13 us | w2 us | HBM TB/s (w13) | useful TFLOPS (w13) | ncu DRAM % | tensor pipe active % | SM clock |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 8 | 7.1x | 82 | 47 | 2.77 | 11 | 71 | 14 | 1.75 GHz |
| 512 | 32 | 2.07x | 510 | 276 | 1.78 | 98 | 49 | 40 | 1.55 |
| 1024 | 48 | 1.47x | 646 | 352 | 1.40 | 163 | 38 | 46 | 1.52 |
| 2048 | 64 | 1.46x | 1186 | 644 | 0.76 | 175 | 23 | 47 | 1.52 |
| 4096 | 64 | 1.27x | 2047 | 1106 | 0.44 | 203 | 18 | 47 | 1.51 |

- Neither roof: DRAM <=49%, tensor pipe (HMMA/mma.sync; GMMA 0) <=47% active.
- Latency/issue-bound: 8 warps/SM (12.5% occupancy, limited by 208-255 regs
  and 115 KB smem/block), 0.5-0.67 eligible warps/scheduler, no warp eligible
  51-61% of cycles. Stalls per issue: wait 1.1-1.6, short scoreboard 0.3-0.55,
  dispatch 0.4-0.5, math throttle 0.34-0.41; long scoreboard (DRAM) only
  0.05-0.27. ALU pipe 34% at 512 (int4 dequant).
- Register spills from M=2048 (bm 64): 7-12 M local-memory requests.
- SM clock 1.51-1.62 GHz under prefill load (1.98 max).
- In the server the two-tier MoE span is 985 us at 512 vs 785 us here for all
  64 experts in one call: tier split + concurrent cold copy cost ~25%.

## Upstream alternatives for INT4 W4A16 MoE on Hopper

- Triton moe_wna16 (upstream fallback, fused_moe_kernel_gptq_awq, default
  config): w13 1669/1720/2490/4265 us at 512/1024/2048/4096 -- 2-3x slower
  than Marlin (bench_triton.txt).
- FlashInfer/TRT-LLM moe_gemm_tma_ws_mixed_input (CUTLASS sm90 array
  mixed-input, wgmma + TMA): int4 group size hard-coded 128
  (mixed_input_utils.hpp), GLM is 32; MXFP4 group 32 is supported (MiMo).
- Both want their own weight layouts; our decode one-kernel reads Marlin's.

Plan: own Hopper kernel that reads the Marlin layout. Marlin packs pairs of
n8 B-fragments; read as W (rows n, cols k) they are the m16k16 A fragment,
which is what wgmma's register-sourced A operand takes per warp. So: dequant
int4 into registers as Marlin does, wgmma RS with W as A (m64 per warpgroup)
and expert-sorted activations as B from smem via TMA, tokens on N (8..256 in
steps of 8, so ~no padding at 16 tokens/expert), persistent grid over
(expert, n-tile), expert -> weight-pointer table so both tiers go in one launch.

## What overlaps the MoE GEMMs (moe_overlaps.py, ranks 0 and 2, 512 and 1024)

Within first..last Marlin of each layer, the only other SM work is the MoE's
own other tier (its act_and_mul 6-8 ms, moe_sum 1-2 ms, fills <0.1 ms per
prefill) -- removed by a single launch over both tiers. No shared-expert,
attention, indexer or collective kernels overlap. The cold copy overlaps
65-72 ms of it but runs on the copy engine (no SMs); it does add ~0.5 GB/layer
of HBM writes, so the memory floor at 512 is ~525 us, not 377.

## Prefill kernel, milestone 1 (vllm/.../fused_moe/tiered_prefill/)

wgmma (register A = decoded Marlin words, B = TMA 128B-swizzled activations),
256 W rows per CTA, every expert on the same N tokens (bench_m1.py, 64 experts,
graph + profiler; check_m1.py vs marlin_quantize's w_ref: within 1 bf16 ulp).

| version | w13 N=16 us | w13 N=64 us |
|---|---|---|
| Marlin (512 tok, ~16/expert) | 510 | - |
| 1 CTA/SM, wait<0> each step | 544 | 775 |
| ld.shared + 2 CTAs/SM (CK 128) | 414 | 770 |
| CTAs/SM by tile (4/3/2), PRMT scale splat | 402 | 763 |
| one LOP3 per nibble pair (constants in regs) | 389 (2.33 TB/s) | 742 (278 TF) |

- Double-buffering A registers inside a warpgroup makes ptxas serialize all
  wgmmas (C7513): overlap must come from other warpgroups/CTAs.
- ncu (single call) N=16: 314 us, DRAM 2.91 TB/s (72%), ALU 52%, GMMA inst
  12%, long_scoreboard 1.8 (mbarrier waits); N=64: barrier 2.1 + long
  scoreboard 1.8, GMMA inst 5%. Sustained SM clock 1.37-1.5 GHz.
- Next: ping-pong consumer warpgroups (shared X tile), then grouped (M2).

### Iterations after milestone 1 (see probes)

- Probes (bench_m1.py --loads-only / --compute-only): w13 N=16 loads only 256 us
  (3.54 TB/s), compute only 217 us, full ~390-400 us: little overlap.
- ncu per mode (single call): loads 245 us @1.82 GHz, compute 225 us @1.71,
  full 327 us @1.42 -- the clock drops with both active (user: design for
  ~1.4-1.5 GHz; clocks cannot be locked, nvidia-smi -lgc is denied).
- Tried, no gain: decode k+1 under wgmma k (no C7513, but no change); CK 64
  with up to 8 stages (no change); shifts as IMAD.HI on the FMA pipe (slower).
- 2 k16 steps per wgmma sync round for N<=32 (3/2 CTAs per SM, no spills),
  node jpbo-121-12: w13 N=16 358 us (2.53 TB/s), N=32 420; compute only 217 /
  259. N=64 regressed to 948 (144 B spills). ncu (base clock) N=16 full:
  ALU 61%, math throttle 0.74, not_selected 0.83, issue 63%: ALU-bound.
- bf16 decode floor ~7.25 ALU/word (3 SHF + 4 LOP3 + PRMT/4): bf16's 7-bit
  mantissa needs the nibble shifted down. fp16 (10-bit) avoids 2 of 3 shifts
  (~5 ALU/word) but rounds w*s to 11 bits instead of Marlin's 8: asked user.

### fp16 (user: "match decode with fp16"), vllm 77ce7f59c7

Exact f16 weights ((code-8)*2^-14, 0x6400 trick: 1 SHF + 4 LOP3 per word),
f16 activations under a power-of-two row scale, group scale in fp32 on the
exact group partial sum (a round = one 32-group). check_m1.py vs the exact
fp32 reference ((code-8)*scale): every output within half a bf16 ulp.
No producer warp (warp 0 lane 0 issues TMA; 128 threads -> 255-reg cap at
2 CTAs). Sweep on jpbo-121-12 (sweep.sh), w13 / w2 at N=16:

| config | w13 N=16 | w2 N=16 | w13 N=32 |
|---|---|---|---|
| bf16 (prev), same node | 358 | 184 | 420 |
| 2 CTAs, deferred scaling | 339 | 174 | 412 |
| 3 CTAs, deferred | 328 | 164 | 411 |
| **3 CTAs, no deferral (default)** | **321** | **163** | 411 |
| 4 CTAs (spills) | 754 | 371 | - |

vs Marlin at the same tokens/expert: 16: 510 -> 321, 32: 646 -> 411,
64: 1186 -> 833 (N>32 still runs as N=32 chunks, re-reading weights).
ncu N=16: no pipe saturated (DRAM 60%, ALU 35%, tensor 22%, issue 54%),
latency-bound with few warps.
Next: a wide-N tile (N 64-128) for long prompts, then grouped routing (M2).

### Wide tiles (vllm 7eae302f55 + working tree)

CTA = 256 W rows over 4/T warpgroups of T tiles, one shared X tile; wide
tiles scale weights in f16 (exact: bf16 scale 8 bits x |code-8| <= 3 bits <=
f16's 11; per-call 2^k into f16's normal range; GLM-5.3 scales span 2^5 to
2^14.5 per layer, sampled layers 3/20/40/60/77), narrow (N <= 32) keep fp32
per-group scaling (cheaper there). Errors vs exact: mean 1.1e-3 rms-rel
(the output's bf16 rounding) vs 1.6e-3 for Marlin's numerics; < 5e-5 of
outputs beyond half an ulp, none beyond one.
Same-process A/B vs 77ce7f5 (bench_ab.py), w13: N=16 335 -> 334, 32 427 ->
429, 64 858 -> 546 (T=4, 2 CTAs), 96 1295 -> ~840, 128 1734 -> ~975.
Tried: producer warp for multi-WG CTAs (288 threads -> 168-reg cap, N=128
spills: worse); non-blocking refill via mbarrier.test_wait (refills later:
96 903, 128 1047, worse than blocking one-stage-late refill).
ncu wide (base clock): N=128 tensor active 62% full / 80% compute-only.

## Real per-expert token counts (expert_counts.py; user: "token count per expert isn't uniform")

All kernel benchmarks so far gave every expert the same N; Marlin's baseline
used uniform random routing. From the agentic route capture (glm53-route-cap,
657 requests, [tokens, 78, 8] ids), chunks at random offsets past the first
2048 tokens, experts mapped to GPUs with the served profile's owners:

| chunk | mean/expert | p50 | p90 | p99 | busiest per GPU (median) | 0 tokens | 1-8 |
|---|---|---|---|---|---|---|---|
| 512 | 16 | 7 | 35 | 123 | 110 | 22% | 31% |
| 1024 | 32 | 17 | 67 | 225 | 200 | 18% | 18% |
| 2048 | 64 | 34 | 130 | 446 | 390 | 18% | 10% |
| 4096 | 128 | 64 | 246 | 959 | 773 | 22% | 4% |

Busiest GPU / mean routed tokens per layer: 1.21-1.26. Counts saved as
fscratch/ingraph-prefetch/counts_<chunk>.npy [samples, 75 layers, 4, 64].
Implications: skip empty experts (~20% of weight bytes), per-expert tile
width in one launch, heaviest-first scheduling. Next: Marlin and our kernel
on real captured top-k ids, then the grouped kernel.

### Correction: use the no-loop capture

`glm53-route-cap/merged` includes requests that looped during capture: 5/64
(512) and 11/64 (2048) sampled chunks had <= 8 experts per GPU taking every
token. `merged-noloop` has none; all real-routing tools now use it. Corrected
distribution (replaces the table above):

| chunk | 0 tokens | p50 | p90 | p99 | busiest per GPU | busiest GPU / mean |
|---|---|---|---|---|---|---|
| 512 | 13% | 9 | 38 | 111 | 106 | 1.14 |
| 1024 | 6% | 21 | 71 | 199 | 190 | 1.13 |
| 2048 | 3% | 44 | 139 | 380 | 366 | 1.12 |
| 4096 | 1% | 92 | 268 | 716 | 690 | 1.11 |

### Grouped kernel on real routing (bench_grouped_real.py, 16 chunks each)

Per-width launches (widest first) serialized and left SMs idle (timeline at
512: the 128- and 96-wide launches had 16 CTAs each). With programmatic
dependent launch they overlap (vllm working tree). w13, same node/process:

| chunk | Marlin | ours (before overlap) | ours | sort/gather | weight floor |
|---|---|---|---|---|---|
| 512 | 783 | 479* | 346 (-56%) | 12 | 218 |
| 1024 | 1150 | 524* | 460 (-60%) | 21 | 239 |
| 2048 | 1821 | 669* | 649 (-64%) | 43 | 245 |
| 4096 | 2980 | 1116* | 1077 (-64%) | 85 | 249 |

(* looping capture.) 4096's real floor is compute: 412 GFLOP -> 654 us.

### Whole MoE on the GPU (vllm 455265cc17, ddf0d9e248 + working tree)

tiered_prefill_moe: route (1 CTA) -> gather (sorted f16 rows) -> w13 per
width -> silu*up in place -> w2 per width -> combine; graph-capturable;
tests/kernels/moe/test_tiered_prefill_moe.py (9 tests). Error vs fp32 with
the same intermediate precisions: mean 1.1e-3 of output RMS (Marlin 3.9e-3).
One class per expert (from its total rows) so results are deterministic
(atomic route order within an expert no longer changes a route's numerics).

Timeline (512, real chunk): route 4.7, gather 9.2, act 12.4, combine 19.9
us; w13 phase 317 us, w2 186. The 16-wide launch starts only at ~117 us:
the GPU dispatches blocks in launch order regardless of stream, so it waits
for earlier widths' blocks and then runs alone (214 us) as the tail.

Schedules, whole MoE, 16 real chunks, same node (us):

| chunk | Marlin | chained widest-first (default) | narrow-first | streams | persistent |
|---|---|---|---|---|---|
| 512 | 831 | 574 (-31%) | 616 | 598 | 769 |
| 1024 | 1294 | 815 (-37%) | 865 | 805 | 993 |
| 2048 | 2064 | 1222 (-41%) | 1288 | 1224 | 1373 |
| 4096 | 3365 | 2231 (-34%) | 2339 | 2143 | 2252 |

Persistent (one CTA/SM walking static items, widest first): SMs balanced
(active cycles min/avg/max 700K/825K/869K) but per item slow; a producer
warp (288 thr) capped regs at 168 -> spills; thread-0 issuer starved the
ring (long_scoreboard 1.26); setmaxnreg producer warpgroup (40/232 regs)
helped (930 -> 764 at 512) but 2 WGs x T=2 is worse for narrow items than
the tuned 3 CTAs x T=4.

### Persistent v2 with a work queue (vllm 398b45e30d)

Independent consumer-warpgroup pipelines (own ring, T=4, 128-row units as
two 64-col items), producer warpgroup with setmaxnreg 40/232 (240 hung:
the producer's decrease must free what the consumers' increase takes),
items claimed from a queue the route kernel resets. Whole MoE, 16 real
chunks, same node: 512 643 (chained 599), 1024 860 (808), 2048 1287
(1227), 4096 2181 (2226). Static assignment was 672/946/1415/2394 (busiest
SM 16% above average). Default stays chained widest-first.

## State: ready for integration (2026-10-02)

API: tiered_prefill.tiered_prefill_moe(x, topk_ids, topk_weights, hot_map,
cold_map, hot, cold, scale_exp, schedule=0) -> this rank's routed sum
(bf16 [T, 6144]); hot / cold are the Marlin tier component dicts the
decode kernel takes (cold may be the staged HBM slot views or the Grace
alias); scale_exp = prefill_scale_exponent(hot, cold), once per layer at
load (host sync). No host sync per call; CUDA-graph capturable; workspace
~20 KB per routed row (cap T * 8 rows), allocated per call.
Tests: tests/kernels/moe/test_tiered_prefill_moe.py (11): two tiers vs
fp32, split experts, one tier, padding / no local routes, graph replay,
all schedules bit-identical, cold tier in Grace, int64 ids / bf16 weights.
Numerics: exact products, fp32 sums; mean error 1.1e-3 of output RMS vs
3.9e-3 for Marlin. Deterministic.
Perf, whole routed MoE on one GPU, real GLM-5.3 routing (16 chunks each),
vs Marlin fused_marlin_moe with all 64 experts in one call (production
Marlin runs two tier calls, so slower): 512 -31%, 1024 -37%, 2048 -41%,
4096 -34%.
Limits: GLM-5.3 W4A16 only (INT4 group 32, 6144 x 2048, top-8); MiMo's
MXFP4 needs its own decode. Known headroom: w13 at 512 ~1.45x and w2 ~1.7x
the weight-read floor; the narrow width runs last (dispatch order);
sorted-activation gather (12-85 us) could move into the GEMM producer.

## Integrated in serving (vllm 23a50e73b2, 2026-10-03)

`VLLM_TIERED_MOE_PREFILL_KERNEL=1` (now serve.sh's default) runs tiered
GLM-5.3 steps of > 8 tokens through tiered_prefill_moe; the cold tier is the
prefetch slot's views when staged, else the Grace alias.

Two bugs found on the way, both invisible to the kernel tests:
- **Wrong maps (integration).** `layer.tiered_cold_expert_map` also maps the
  rank's Grace replicas of other ranks' experts (slots after its own cold
  experts). Marlin runs on the execution maps (`tiers[i][3]`), which drop
  them. With the registered map the kernel computed replicas twice (once
  here, once at home) and indexed past the slot, which holds only the own
  cold experts: illegal address in the 8192-token profile run.
- **NaN rows (kernel).** Real prefill has rows of ~1e-20 (min row max
  8e-21 in a dumped layer-7 call); the SiLU * up row is then denormal and
  `exp2f(-e)` overflowed to inf (0 * inf = NaN). Clamped e to [-126, 126]
  (`row_exp`); regression test `test_tiny_rows_stay_finite` fails without
  it. The NaNs hung FlashInfer's Lamport all-reduce at capture size 128
  (cuda-gdb: ranks 0/1/3 spinning in allreduce_fusion_kernel_oneshot_lamport,
  rank 2 done). Found by dumping the first non-finite call per rank
  (`replay_nan.py`); the four dumps replay finite and within 0.36-0.39% mean
  |diff| / RMS of Marlin after the fix. tiered_decode's `row_scale` has the
  same unclamped exponent (not changed; decode rows have not hit it).

Harness: servers on several nodes shared one VLLM_CACHE_ROOT and one
FlashInfer JIT cache. Concurrent writers broke the trtllm_comm build (one
rank without AR fusion, another hung in its barrier) and torch.compile
artifacts (one rank recompiled alone for 10 min). serve.sh now takes
SERVE_CACHE_ROOT; launch_pk.sh gives each hold its own. GSM8K needs its data
pre-fetched (compute nodes have no internet): bench_node.sh points TMPDIR at
fscratch/caches/gsm8k-data.

### Same-node A/B (launch_pk.sh, passes 5 and 6), median TTFT ms, 20 prompts

| tokens | 512 | 768 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| node A Marlin | 134 | 171 | 202 | 327 | 990 |
| node A kernel | 122 | 135 | 154 | 243 | 873 |
| node B Marlin | 136 | 175 | 205 | 333 | 1005 |
| node B kernel | 122 | 137 | 158 | 248 | 889 |

Both arms with in-graph prefetch from 512 (no prefetch was 176/236/240 at
512/768/1024). GSM8K 400, 5-shot greedy: kernel 90.0%, Marlin 91.25%, 0
invalid (SE ~1.5 points each; earlier GLM run 91.21%). Greedy text diverges
after a few dozen chars, as identical configs on two nodes already did.

### Trace, 512/768/1024 (trace-pk, analyze_trace.py / breakdown.py / trace_pk.py)

MoE per layer (route .. combine) vs Marlin's (trace-wide): 512 596 vs 1001
us, 768 744 vs 1230, 1024 894 vs 1449. Prefetch works: 75 copies per prefill
(~506 MB, ~1.38 ms each), every MoE layer read the slot (log: 150 from slot,
0 from Grace), dense-layer copy never stalls. But the copy is now the
critical path at 512: it outlasts MoE(L) on 69/73 layers and the join
stalls total 19-22 ms per rank (768: 3-4 ms; 1024: 0.1-0.4 ms). 512 rank 0:
window 126.9 ms (Marlin 136.7), idle 30.8 ms = 18.6 join stalls + ~10
host-bound (setup / dense part); copy engine busy 95 of 127 ms. Floor at
512 is the C2C copy (~1.4 ms/layer), not compute; reading the cold tier
straight from Grace instead (isolated 1130 us/layer at 512) is estimated
~1.8 ms/layer vs 1.53 now, so no.

### Why 4096 costs ~3.6x 2048 (trace-big, top_kernels.py)

Not the MoE. GPU busy 187 -> 858 ms: attention 6 -> 256 (dense FA3 72
us/layer -> flash_fwd_splitkv_mla_fp8_sparse 3.28 ms/layer: DSA top-k 2048
makes <= 2048 tokens dense), DCP4 sparse-path collectives (all-gather 7.5 ->
79 ms, reduce-scatter 0 -> 53, _correct_attn_cp_out 17, copies/cat ~80), the
MoE all-reduce 23 -> 96 ms (NCCL RING_LL, 149 -> 618 us/call for 2x bytes),
MoE 113 -> 179 (linear). Marlin jumps the same (333 -> 1005). 2048 is 44%
host-bound in the trace (no graphs above 1024; profiler inflates it).

## Long-context OOM and the 9 GB reserve (2026-10-03)

The Claude Code server (job 2150656, then 2150901) died of a CUDA OOM on its
first long-context 8192-token chunk: FlashMLA's sparse decode kernel (DSA
path, used once context > 2048) asked for 2 GiB with ~1.7 GiB free and
~2.3 GiB reserved but fragmented. stress_long.sh (60K-token prompt, then
turns growing the prefix to ~200K) reproduces it on turn 0 at reserve 7 both
with the prefill MoE kernel and without it (VLLM_TIERED_MOE_PREFILL_KERNEL=0),
so the kernel is not the cause; the startup profile runs the dense path and
never sees that buffer, and the prefill CUDA graphs (1.32 GiB) came out of
the same margin. Slicing the kernel's workspace (vllm c71d95ea74, steps >
4096 tokens in slices) is kept but did not fix it. expandable_segments:
engine init fails. RESERVE_GB=9 (now serve.sh's default): 7.2 GiB free
after startup (was 5.4), 31 turns to 196K tokens, 0 OOM; ~3% fewer hot
experts (2963 vs 3061 per rank in the first layers' plan).
Turn latency at 150-196K context (TTFT + 16 output tokens): 2K new tokens
~0.9 s, 4K ~1.4 s, 8K ~2.5 s -- several times the zero-context sweep
(2048 new: 243 ms TTFT); long-context attention dominates real turns.

## Prefill and decode at 100K context (prof100k.sh, prof_window.py; 2026-10-03)

98K-token prefix of real code (vLLM sources) cached, then 2000 new tokens
(the step recomputes 2464: the cache matches in 256-token blocks), serve.sh
defaults (prefill kernel, reserve 9). Unprofiled, 3 reps: TTFT 643-649 ms,
decode 110-125 tok/s (256 out; 84 on the first, cold rep). Uncached 98K
prefill: 21.5 s.

Prefill step, rank 0 (ranks within 1 ms): wall 584 ms, busy 559, idle 25
(mostly host-bound step setup; no prefetch stalls, the 106 ms of cold copies
hide under MoE). Union time per category: sparse MLA 159 ms (FlashMLA sparse
*decode* kernel, 2.04 ms/layer), collectives 135 (all-gather 52 / 120 calls,
all-reduce 41 / 155, reduce-scatter 33 / 78, one-shot gather 10), MoE 101
(+11 glue), copies/cat 53 (3 direct_copy per layer, 40 ms), dense GEMM 43,
indexer 22 (mqa_logits on 42 of 78 layers), DCP combine 10, norm/rope/misc
~25. Attention-side work (sparse MLA + its DCP collectives/copies + indexer)
is ~60% of the step; MoE ~19%.

Decode (DFlash2 verify, 8 tokens), 34 steps: 27.6 ms/step, ~3.8 tokens
accepted per step on this code text. Union per step: MoE 14.5 ms (53%),
dense GEMM 6.4, collectives 4.5, sparse MLA 1.4 + indexer 0.8 (DSA keeps
attention small at 100K), triton/elementwise/norm/router ~2.1.
