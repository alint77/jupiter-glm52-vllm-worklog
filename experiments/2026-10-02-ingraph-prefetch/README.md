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
