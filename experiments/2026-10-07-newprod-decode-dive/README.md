# New-prod decode dive: where the 21.8 ms step goes (2026-10-07)

The first kernel-level breakdown of the **full memory stack** prod default
(skip_host_uva + fp8 drafter KV & weights + drafter KV on Grace + reserve 4.7,
`skipkv-newprod` arm, hold 2208236), from its own profiler window
(`/e/fscratch/profound/naeimitabiei1/agentic-bench/skipkv-newprod/trace-base/
window-0`, 92 decode steps x 4 ranks, 16 agentic requests). The same-day
old-prod window (`skipkv-dkvhost/trace-base`, 48 steps, same node) gives the
before/after. Tools: `dive.py`, `dive2.py`, `wait_layers.py` here over
`../2026-09-26-mimo-decode-profile/analyze.py` (union-partition by launch
correlation); `replay_ranked.py` for the placement replay.

**Careful with analyze.py's buckets on this config.** FUSE_AR_RMS moved the
layer all-reduces into `trtllm_allreduce_fusion` kernels, and
"flashinfer" contains "flash", so the entire fused-AR residency (2.21 ms/step)
files under **attention** -- the 4.25 ms "attention" bucket is really 2.21 AR +
1.44 FlashMLA + 0.63 combine + 0.24 KV writes. analyze.py's "TP all-reduce"
row (0.14, x6) is only the few unfused `cross_device_reduce` calls.

## Step budget (92 steps, mean period 21.84 ms, p50 21.72)

| region | ms/step | note |
|---|---:|---|
| target graph busy (union) | 19.44 | span 19.85; **zero gaps >= 3 us** (0.41 ms of sub-us boundary bubbles, 2,581 kernels) |
| - MoE one-kernel chain (share) | 6.91 | w13 2.04 (26 us/layer, sum 4.4), w2 1.43, route/act/finalize 3.44; PDL-packed |
| - dense GEMM + glue | 5.04 + 1.10 | floor ~3.0 (byte-stream); nvjet 64x8 x336 + splitK x174 + reduce x174 |
| - fused AR+RMS (trtllm, x151) | 2.21 | wire ~1.0 (6.9 us floor/call), **spin-wait 1.16 (mean-rank; 2.82 max-min)** |
| - FlashMLA + combine + KV writes | 2.32 | 18.4 us/layer, anchor == skip (staging never late); combine 8.1 us |
| - DCP one-shots (x177) | 1.35 | wire-bound; gather_cat 9 us, lse_rs 6.5 us x78; c=1 pays for c=4 capacity |
| - DSA indexer | 0.49 | 21 indexer layers |
| draft region (eager DFlash2, fp8) | 1.25 span | busy 1.05, gap 0.21; under-profiler inflation caveat |
| logits / host-side busy | 0.17 / 0.39 | |
| GPU idle besides in-graph bubbles | ~0.4 | boundaries + draft gap |

Old prod (same day, 48 steps): period 23.91; AR wait max-min 4.07 vs 2.82;
w13 mean 67-86 us/layer vs 57-69; MoE chain share 8.03 vs 6.91. The stack's
+330 hot experts are all visible in the MoE window and the AR waiting.

## Findings

1. **The AR spin-wait is continuous, diffuse, and rotating -- ~1.16 ms/step.**
   p50 1.06-1.23 ~= p90 1.41-1.47 on every rank: nearly every step pays it,
   there is no burst structure. The last-arriving rank rotates (rank 0 last in
   38/92 steps, then 2, 3, 1); per-ordinal worsts hold no single layer (worst
   layer = 1.7% of skew vs 1.33% uniform for 75 layers). corr(per-step w13
   skew, AR wait) = 0.70. This is the max-of-4 order statistic of cold-expert
   routing, not a defectable hot spot -- placement inputs and balancing are
   the only handles (see 1 below).
2. **The 2026-09-28 step-entry burst is gone.** AR#0 (step entry) waits 28 us
   mean / 63 us ordinal-mean, 1/92 steps > 0.5 ms (old-prod window: 762 us
   mean -- window-content dependent). Burst steps (period > median+1 ms):
   4/92, all explained by ~1.5-2 ms longer *target graphs*, not entry stalls.
   Do not carry the "18% of steps eat 0.9-11 ms" item forward without a new
   sighting.
3. **There is nothing left inside the target graph's schedule.** Zero union
   gaps >= 3 us across 2,581 kernels/step. The 0.41 ms span-minus-union is
   sub-microsecond boundary bubbles between PDL-chained kernels; graph-node
   fusion caps at that.
4. **Skip-KV staging is invisible.** 38 staging kernels/step, 0.4 ms of
   kernel time on a side stream; slack never negative; skip-layer FlashMLA
   runs at the anchor's 18.4 us. The memory-stack change behaves as designed.
5. **The promotion gap is the top actionable item.** The planner log: profile
   lists 3,239, promoted to 3,508-3,510 -- **~271 experts/GPU (7.7% of the
   hot set) seated in expert-id order with zero frequency information**, a gap
   the reserve sweep widened from ~170. Replay over the live CC capture
   (28,698 verify steps, `replay_ranked.py`):

   | hot set | cold/GPU/layer mean | slowest GPU | vs served |
   |---|---:|---:|---|
   | oldprod 3,180 (id) | 1.996 | 3.425 | -- |
   | newprod 3,510 (id, as served) | 1.537 | 2.773 | -- |
   | newprod 3,510 ranked | 1.424 | 2.600 | **+0.43 mean / +0.66 slowest ms/step** |

   The one-kernel overlaps part of the cold reads (ab-gA/gB measured the
   replay's sibling prediction at ~2x the wall gain), so a realistic live gain
   is ~0.3-0.45 ms/step, for a profile rebuild + restart with no runtime cost.

## Recommended queue (decode-only, execution-only)

1. Rebuild the placement profile at the real budget (3,531 slots from the
   reserve sweep; live-CC capture, frequency-ranked, replicas recomputed),
   gate through `gate0b_replay.py` here, A/B on the node. Expected ~0.3-0.45
   ms/step on live CC traffic.
2. The queued unprofiled bf16-vs-fp8 drafter-timing comparison, then judge
   whether the fp8 drafter's per-token activation-quant launch count is worth
   a same-math epilogue fusion. Draft region: 1.25 ms span. A draft CUDA graph
   (worth <= ~1 ms) stays gated on the Phase 52 acceptance defect.
3. Dense GEMM small-grid work (5.04 vs ~3.0 floor) remains the largest
   engineered item (~0.5-0.8 realistic, `skinny_gemm` parked; MiMo's Inductor
   fused-mm templates are the named fit).

Not opportunities, re-measured dead here: in-graph idle (zero >= 3 us),
act/finalize exposure (PDL-hidden), AR wire time (6.9 us floor/call),
step-entry stalls, skip-KV staging.

## Files

- `dive.py`, `dive2.py`, `wait_layers.py`, `replay_ranked.py`,
  `dive-newprod.json`, `dive2-newprod.json`
- traces: `/e/fscratch/profound/naeimitabiei1/agentic-bench/skipkv-{newprod,dkvhost}/trace-{base,dkvhost}/window-0`

```bash
.venv/bin/python agent_space/experiments/2026-09-26-mimo-decode-profile/analyze.py \
  /e/fscratch/profound/naeimitabiei1/agentic-bench/skipkv-{newprod,dkvhost}/trace-base/window-0
.venv/bin/python agent_space/experiments/2026-10-07-newprod-decode-dive/dive2.py \
  /e/fscratch/profound/naeimitabiei1/agentic-bench/skipkv-newprod/trace-base/window-0
.venv/bin/python agent_space/experiments/2026-10-07-newprod-decode-dive/wait_layers.py \
  /e/fscratch/profound/naeimitabiei1/agentic-bench/skipkv-newprod/trace-base/window-0
.venv/bin/python agent_space/experiments/2026-10-07-newprod-decode-dive/replay_ranked.py
```
