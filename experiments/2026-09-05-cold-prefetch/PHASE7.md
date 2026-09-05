# Phase 7 — enabled in production, and profiled there

The campaign measured GLM-5.3 W4A16 group 32, loaded through
compressed-tensors. The production CC server serves **GLM-5.2 AutoRound W4
group 64** through `auto_gptq`. Different quantiser, different group size,
different placement profile, and a 5 GB HBM reserve instead of 10.

## How it was enabled

`VLLM_TIERED_MOE_COLD_PREFETCH_MIN_TOKENS=1024` in
`experiments/2026-07-26-claude-autoround/server.sbatch`, not as a new default
in `vllm/envs.py`.

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
