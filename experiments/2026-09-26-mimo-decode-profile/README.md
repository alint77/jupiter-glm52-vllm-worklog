# MiMo-V2.6 decode profile (2026-09-26)

Where a decode step goes in the production MiMo config: tiered MoE with the
routing profile (`profile-3827.json`), DFlash k=7, R3, LM-only, TP4/EP4, c=1.
Job **2074303**; traces in
`/e/project1/profound/alint77/traces/mimo26-decode-2074303/{decode-short,decode-96k}`.

## Capture

`capture.py` reuses the 2026-09-04 harness's gating (the window opens only once
generation tokens are rising). Each context gets one long greedy request; its
step time is measured **unprofiled** from the metrics counters first, then a 2 s
profiler window is opened on the same request.

| | unprofiled step | profiled step | distortion | tokens/step |
| --- | ---: | ---: | ---: | ---: |
| short (PyTorch coding prompt) | 28.61 ms | 29.48 ms | +3.0% | 5.2 |
| ~96K (Claude-Code-shaped) | 34.82 ms | 36.46 ms | +4.7% | **8.0** |

Distortion is small, so absolutes are quotable. **The 96K request degenerated:**
8.0 tokens per step means every draft was accepted, i.e. greedy output was
looping. Step cost is unaffected (the verify is always 8 tokens), but its
routing is not representative traffic, so read the 96K row for attention
scaling only.

## Method

`analyze.py`: kernels are attributed to a step by launch correlation. Each step
is one big verify graph (~2.3K kernels), lm_head + vocab all-gather, the DFlash
drafter (6 small graphs + eager attention), and host-side kernels. Time inside
the verify graph is union-partitioned (each instant shared among the categories
active in it), so shares sum to busy time and the hot/cold overlap is not double
counted. Hot vs cold Marlin by grid: 264 CTAs (hot, 2/SM) vs 132 (cold, 1/SM).
Collectives are also compared across ranks per (step, ordinal): the cross-rank
minimum approximates the transfer, the excess is waiting.

## Result (short context, 29.2 ms/step, mean of 4 GPUs)

![step](figs/step-breakdown.png)

| | ms/step | share |
| --- | ---: | ---: |
| MoE Marlin, hot tier (HBM) | 4.16 | 14% |
| MoE Marlin, cold tier (Grace) | 7.96 | 27% |
| **waiting in reduce-scatter for the slowest GPU** | **5.05** | **17%** |
| MoE routing/align/act/sum | 1.50 | 5% |
| SP MoE comm: all-gather 1.56 + reduce-scatter transfer 0.46 | 2.02 | 7% |
| dense GEMMs (fp8 deep_gemm + cuBLAS) | 3.02 | 10% |
| attention (FA3/FA4 with sinks, KV write) | 1.01 | 3% |
| norms, rope, elementwise | 0.93 | 3% |
| TP custom all-reduce | 0.40 | 1% |
| drafter 0.87 + logits 0.28 + host kernels 0.20 | 1.35 | 5% |
| GPU idle within the step | 1.82 | 6% |

Findings:

1. **The "communication" is load imbalance.** MiMo's MoE runs sequence-parallel
   (69 reduce-scatters + 138 all-gathers per step). The reduce-scatter costs 5.51
   ms/step but its cross-rank minimum is 0.46 ms: 5.05 ms is ranks waiting.
   `imbalance.py` pins it: each rank's waiting equals the slowest rank's Marlin
   time minus its own in that layer (5.05 vs 4.99 ms/step, all four ranks within
   0.06 ms). The per-layer spread is mostly cold (5.8 ms/step) not hot (2.2), and
   the last rank to arrive is uniform (23-26% each): per-step routing variance,
   not a slow GPU.

   ![layer](figs/layer-ranks.png)

2. **MoE is ~60% of the step once waiting is counted**: 12.1 ms of overlapped
   Marlin plus 5.0 ms of waiting. The hot/cold overlap is doing its job: Marlin
   sums to 18.7 ms/step, its union is 12.5 ms.
3. **Cold is still the long pole.** Even with the profile (17% of expert reads
   from Grace), cold Marlin sums to 11.3 ms/step against 7.4 hot: ~87 µs per
   cold expert-layer vs ~11.5 µs hot, the C2C/HBM ratio. Per layer the cold tier
   finishes last on average.
4. Attention is cheap and scales gently (1.0 -> 1.5 ms from short to 96K): 60 of
   70 layers are 128-token sliding window.
5. 1.8 ms/step of GPU idle sits between the verify graph and the drafter
   (rejection sampling needs the host).

## Levers, in order of size

| lever | addresses | size |
| --- | --- | --- |
| balance cold work across GPUs per layer: replica assignment (GLM's exact min-max with secondary Grace copies), not yet built for MiMo | waiting 5.05 ms | up to ~17%; GLM got 5-8% from it |
| less cold work: more residency (reserve, KV) or per-layer slot allocation weighted by cold cost | cold 8.0 ms | each cold expert-layer removed ~76 µs |
| drop sequence-parallel MoE for the all-reduce path GLM uses (Phase 8) | SP comm ~2.0 ms | ~1.6 ms (~5%); waiting would move to the all-reduce, not vanish |
| overlap draft/sampling with the next verify | idle 1.8 ms | ~6% |

Replica assignment is the first thing to try: it targets the largest bucket and
the machinery exists (`2026-07-31-replicated-expert-scheduling/oracle.py`,
`--tiered-moe-replica-assignment exact`), but it needs a MiMo profile v2 with
`secondary_ranks` and the per-step route fingerprint check under DFlash.

## Files

`capture.py`, `job.sbatch` (capture); `analyze.py` -> `breakdown-2074303.{txt,json}`;
`imbalance.py` -> `imbalance-2074303.txt`; `plot_breakdown.py` -> `figs/`.
