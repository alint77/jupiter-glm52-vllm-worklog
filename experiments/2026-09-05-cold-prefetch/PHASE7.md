# Phase 7 — second-quantizer validation on GLM-5.2 AutoRound W4G64

> **Corrected 2026-09-05.** This phase was written as "enabled in production,
> and profiled there." That premise was wrong. I identified production from
> `claude-local.sh`'s default wrapper and its `--served-model-name`
> (`glm52-w4a16-tiered`) and concluded the prod CC server ran GLM-5.2
> AutoRound W4G64. It does not. `sacct` shows the daily driver is
> `glm53-claude-w4a16-c1-df2` →
> `experiments/2026-09-04-glm53-c1-df2/server.sbatch`, serving **GLM-5.3
> W4A16** with a DFlash2 drafter. The served-model-name is a stable API label
> reused across checkpoints; the checkpoint is `model_name=` in the sbatch.
>
> Two consequences, in opposite directions:
>
> * The enablement landed in the **wrong launcher**, so this phase did not in
>   fact turn the feature on in production. Phase 8 does that.
> * The eval-coverage worry this phase raised evaporates. Phases 3-6 ran
>   `GLM-5.3-W4A16` with `glm53-w4a16-2496.json` and `MIN_TOKENS=1024` —
>   which *is* the production configuration. The McNemar gates (p=0.888,
>   p=1.000) cover production exactly.
>
> What survives, and is worth keeping, is what this phase actually measured:
> the code path works on a **second quantiser** (auto_gptq, group 64) as well
> as on compressed-tensors group 32. That is a real result. Read the rest of
> this file as that, not as a production measurement.

The campaign measured GLM-5.3 W4A16 group 32, loaded through
compressed-tensors. This phase measures **GLM-5.2 AutoRound W4 group 64**
through `auto_gptq`. Different quantiser, different group size, different
placement profile, and a 5 GB HBM reserve instead of 10.

## How it was enabled

`VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1024` in
`experiments/2026-07-26-claude-autoround/server.sbatch`, not as a new default
in `vllm/envs.py`. (That file is the GLM-5.2 launcher, not the production one
— see the correction above.)

A global default is the wrong lever twice over. The gate is a token count, so
`setup_tiered_moe_kernels` now refuses any configuration where
`max_num_seqs * verify_tokens >= MIN_TOKENS` -- defaulting it on would convert
working deployments into load-time failures. And nothing in the campaign
validated this checkpoint. One line in the launch script is what "the prod cc
script uses it" actually requires.

## Preflight, before spending an allocation

* `glm_marlin_components` names 64 explicitly and derives scale shapes as
  `6144 // group_size`; `auto_gptq.py` passes `self.quant_config.group_size`.
  Both correct by inspection.
* Group 64 added to the prefetch tests' parametrisation. The group-32 layout
  bug reached a server launch before its backing-size assert caught it, and 64
  had no coverage.
* Priced offline against the production checkpoint, profile and reserve:
  the fixed point converged, so the launch would not fail closed.

## What production actually plans

| | offline estimate | measured |
| --- | --- | --- |
| residency | 2574 -> 2535 | **2714 -> 2677** |
| hot experts given up | 39 | **37 (1.4%)** |
| slot | 746 MiB | **708 MiB** (689 on rank 1) |

The offline probe used approximate HBM capacities, so it was close but not
exact -- worth remembering before quoting a probe figure as a result. The fixed
point converged in three passes (2714 -> 2678 -> 2677) and the observed reserve
passes at 5.65 GiB free against a 3.73 GiB minimum, with the slot resident.

## The prefill chunk, 8192 tokens, mean of 7

Wall 1616.9 ms, GPU busy 1659.1 ms, cumulative kernel 1773.6 ms.
`cum/busy = 1.069`, which is the copy stream overlapping rather than
serialising.

| kernel | ms | % of wall |
| --- | --- | --- |
| sparse MLA | 578.5 | 34.9 |
| MoE Marlin hot (w13+w2) | 244.8 | 14.8 |
| all-reduce | 153.0 | 9.2 |
| staging copy (HtoD) | 139.1 | 8.4 |
| MoE Marlin cold (w13+w2) | 182.7 | 11.0 |
| DSA indexer | 113.8 | 6.9 |
| dense GEMMs | 130.7 | 7.9 |

## The answer to the open question

Group 64 has a different scale-to-weight byte ratio than group 32, so whether
staged cold Marlin still reached the hot tier's rate was genuinely open.

| | effBW | per-expert |
| --- | --- | --- |
| hot | 315 / 335 GB/s | 91.5 us |
| cold, staged | **338 / 351 GB/s** | **86.0 us** |

Cold is **5.9% faster per expert than hot**. Nothing about group 64 changes the
result. Cold sitting slightly ahead is consistent rather than surprising: cold
layers hold fewer experts on average (28.3 against 35.7), so each Marlin call
loops over fewer experts for the same routed rows.

## Before and after, on the GLM-5.2 checkpoint

Job `1672712` is the same capture with `MIN_TOKENS=0`. Its residency is
2714 hot / 2086 cold and it logs no staging lines at all, so the off arm is
genuinely off.

The staged capture caught **7** chunks and the baseline **6**. Chunk 7 is a
full chunk, not a tail (`sparse_decode_fwd` 581.96 ms, in line with c1-c6),
but it sits deeper in context, and context-growing roles climb monotonically
(`sparse_attn_indexer` 55.0 → 186.8 across the seven). Averaging over
mismatched chunk counts therefore mixes a workload difference into the
comparison. The table below restricts the staged arm to its first six chunks
(`prefill_roofline.py --max-chunks 6`); `prod52-matched-c1c6.txt` has the full
output.

| | baseline (6ch) | staged (c1-c6) | |
| --- | --- | --- | --- |
| prefill wall | 1821.1 ms | **1595.1 ms** | **-12.4%** |
| GPU busy | 1881.2 ms | 1645.0 ms | -12.6% |
| cold Marlin | 440.4 ms | **183.8 ms** | **-58.3%** |
| hot Marlin | 240.5 ms | 242.8 ms | +1.0% |
| all-reduce | 189.9 ms | 155.3 ms | -18.2% |
| `sparse_decode_fwd` | 552.1 ms | 578.0 ms | **+4.7%** |
| `aten::copy_` | 23.5 ms | 139.0 ms | +115.5 ms (the copy) |
| cold effBW | 133 / 153 GB/s | **338 / 351 GB/s** | 2.5x / 2.3x |
| `cum/busy` | 1.000 | 1.070 | overlap appears |

The unmatched 7-chunk mean gave -11.2%; matching the chunk count gives
**-12.4%**. The extra chunk was penalising the staged arm, not flattering it.

**The cold tier went from 2.38x the hot tier's per-expert cost to 0.94x.** Total
MoE Marlin time falls 37.2%, from 680.9 to 427.5 ms per chunk.

### Where the copy actually costs time

`stage(L+1)` is issued *after* `apply_tiered` for layer L and joined at L+1's
MoE. So the copy is in flight during the post-MoE all-reduce and during layer
L+1's attention, indexer and dense mm — but **not** during hot or cold Marlin,
which run after the join and before the next `stage`. Hot Marlin is the one
role that cannot be contended by the copy.

That predicts what the matched table shows, and contradicts what this file
said before:

* **Hot Marlin +1.0%** — near noise, as the ordering requires. The earlier
  claim of "+3.2% per expert, copy-stream contention for HBM bandwidth" was
  wrong twice: the number came from the unmatched 7-chunk mean, and the
  mechanism was attributed to the one role the copy cannot reach.
* **`sparse_decode_fwd` +4.7% (+25.9 ms)** is the contention, and it is the
  real cost of the feature. The earlier version did not report it at all.
* `cum/busy` moves from exactly 1.000 to 1.070 — the copy stream overlapping.
  Of the 139 ms of `aten::copy_`, roughly 115 ms is hidden.

### All-reduce: the earlier attribution was wrong

All-reduce falls, but not for the reason this file gave. The 189.9 → 153.0
figure was **rank 0 alone**, presented as the headline. Per rank, ms/chunk:

| | r0 | r1 | r2 | r3 | mean | spread |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 189.9 | 190.3 | 137.7 | 178.3 | 174.1 | 52.6 |
| staged | 153.0 | 179.3 | 132.9 | 116.3 | 145.4 | 63.0 |

Total wait falls (mean -16.5%), but the rank-to-rank **spread grows**
(52.6 → 63.0 ms) and rank 1's excess over the other three grows from +21.7 to
+45.3 ms. The straggler also moves from rank 2 to rank 3, and it moves on
`sparse_decode_fwd` — attention, which the prefetch does not touch.

So "arrival skew shrinking as per-rank MoE time becomes more uniform" is not
what happened. What the data supports: **total collective wait fell, and the
residual imbalance is now attention-driven rather than MoE-driven.** Every
non-all-reduce role is within a few ms across ranks in both arms, so
all-reduce is the elastic term absorbing arrival differences.

### Against GLM-5.3

The GLM-5.3 comparison this file made (-11.2% vs -14.1%) is not sound; see
`control-check-1665068-vs-1669308.txt`. That -14.1% came from a **cross-job,
cross-commit** pair, and control rows the prefetch cannot touch move +3% to
+15% between them. The matched GLM-5.3 numbers are phase 5's -16.1% (both arms
in one allocation) and phase 8's fresh traced pair.

## Corrections carried into this analysis

The 2026-09-04 roofline table cannot be reused here unchanged. It hardcodes 192
and 64 scale groups per expert -- `6144 // 32` and `2048 // 32`, group size 32 --
which would misprice every Marlin row on a group-64 checkpoint.
`prod_roofline_table.py` derives them from `K // 64`, takes residency from the
run's own log rather than a constant, and prices the cold tier against HBM
rather than C2C. That last one is why PHASE3's table showed cold at 89-94%
efficiency: it was still being judged against the C2C roof it no longer reads
from.

`planner_staging_probe.py` also took its maximum over rank 0 while the budget
is scenario-wide, so its "budget covers requirement" line compared two
different things. Fixed; both figures now agree.

## Caveats

**The trace measures the no-cache-hit path.** Profiling runs with
`--no-enable-prefix-caching` so a repeated prompt cannot skip the prefill being
measured. Production runs prefix caching **on**, so a real session sees shorter
chunks, some below the 1024-token threshold and therefore unstaged. The
realised end-to-end benefit is smaller than the prefill figures suggest. That
is a statement about the workload, not the feature.
