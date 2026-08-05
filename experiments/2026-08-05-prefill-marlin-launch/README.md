# Prefill Marlin launch: the occupancy hypothesis is wrong, the tile is

Status: **Hypothesis refuted; a smaller unrelated win found and shipped behind
a flag.**

The [production profile](../2026-08-05-prod-profile/README.md) found prefill's
routed MoE running at 10.6% of the bf16 peak with the pre-Phase-26 legacy
launch — 2 CTAs/SM, 128 threads, 115,200 B of shared memory — and proposed
ungating the tight-shared-memory policy and lifting `ops.cu`'s
`allow_count <= 2` cap for large M. **That proposal was wrong.** Neither
shared memory nor CTAs per SM is the constraint. What is real is that Marlin's
config heuristic picks a 128-thread tile where a 256-thread one is 8-12%
faster.

## Method

`benchmark_moe_wna16_marlin_decode.py --mode prefill` sweeps the launch config
at the production chunk shape. Passing an explicit `(thread_k, thread_n)` makes
`ops.cu` skip `determine_exec_config` entirely, taking `blocks_per_sm` verbatim,
so the `allow_count` cap and the tile can both be swept from Python with no
rebuild.

Every config is checked against the auto launch's output. A wrong tile produces
wrong numbers rather than failing, so speed without a numeric check would be
meaningless. Differences are reported in bf16 ulps at the output magnitude,
since a different tile changes the fp32 reduction order.

Two jobs, `1241464` (10 iterations) and `1241487` (20 iterations), on separate
Booster nodes.

## Result: two refutations and one win

At m=8192, the production chunk size (job `1241487`):

| smem | tile | CTAs/SM | us | vs auto | ulp |
| --- | --- | ---: | ---: | ---: | ---: |
| — | auto | auto | 12,540 | 1.00x | — |
| legacy | 64x256 | 1 | 11,266 | 1.11x | 0.2 |
| **tight** | **64x256** | **1** | **11,195** | **1.12x** | 0.2 |
| tight | 64x256 | 2 | 15,243 | 0.82x | 0.5 |
| tight | 64x256 | 3 | 13,680 | 0.92x | 0.5 |
| tight | 64x128 | 2 | 12,551 | 1.00x | **0.0** |
| tight | 64x128 | 1 | 14,562 | 0.86x | 0.5 |
| tight | 128x64 | 1 | 19,686 | 0.64x | 1.0 |

**1. Shared memory is not the constraint here.** Legacy 11,266 us against tight
11,195 us is **0.3%**. The Phase 26 mechanism, worth 31-45% at decode shapes,
is worth nothing in prefill. The reason is grid size: a prefill chunk launches
**1,051 blocks against 132 SMs**, so the SM is fed by many sequential blocks and
how many are co-resident does not matter. Decode launches 4 blocks and needs
co-residency; prefill does not.

**2. More CTAs per SM is worse, consistently.** 2 CTAs/SM is 0.82x and 3 is
0.92x, at every chunk size. The `allow_count <= 2` cap flagged as a limitation
is not one — the optimum is *below* it, at one block per SM. The profile's
reading of "2 CTAs/SM, 8 warps of the 64 an SM holds" as under-occupancy was
wrong: occupancy per SM is the wrong figure of merit when the grid oversubscribes
the machine eightfold.

**3. The tile is the real lever.** Auto selects `(64, 128)` at 128 threads —
confirmed, because that config reproduces auto's output at **0.0 ulp**, which is
also the harness's positive control. The 256-thread `(64, 256)` tile is faster
at every size measured:

| chunk | auto | best | speedup |
| ---: | ---: | ---: | ---: |
| 512 | 1,120 / 1,067 us | 1,007 / 987 | 1.11x / 1.08x |
| 2,048 | 3,329 / 3,371 | 3,035 / 3,092 | 1.10x / 1.09x |
| 8,192 | 12,398 / 12,540 | 11,101 / 11,195 | 1.12x / 1.12x |

(two jobs; `1241464` / `1241487`.)

The winning config differs from auto by **0.2 ulp**, i.e. reduction-order noise,
not a math change.

Tiles wider than 256 (`(64, 512)`, `(128, 256)`, both 512 threads) fail to
launch, as does `(128, 128)` at this shape.

## Honest sizing

1.10x on a kernel that is 883.7 ms of a 2,619 ms chunk saves **~80 ms/chunk**:

| | |
| --- | ---: |
| of the routed MoE | 9.1% |
| of a prefill chunk | 3.1% |
| of c4 wall clock on the 16K coding shape (71% prefill) | **~2.2%** |

The profile's estimate for this lever was "up to ~440 ms/chunk if it reaches
2x", or ~12% end to end. **That was 5x too optimistic**, because it assumed the
kernel was occupancy-limited and would approach peak once freed. It is not, and
the MoE stays at ~11% of bf16 peak with the better tile. Whatever holds Marlin
to 11% of peak at W4A16 is not addressable from the launch configuration.

## What shipped

`MarlinLaunchPolicy` gains an explicit `thread_k`/`thread_n` tile and a
`min_tokens` bound, and `_fused_marlin_moe` accepts a *list* of policies, taking
the first that covers the batch. That is what lets decode and prefill have
opposite launches — decode wants two tiers co-resident on few blocks, prefill
wants the widest tile on one block per SM — without either special-casing the
other.

`tiered_moe_execution.py` now installs two policies per tier: the existing tight
shared-memory one at or below the decode bound, and the `(64, 256)` tile above
it. `VLLM_TIERED_MOE_PREFILL_TILE=0` restores the heuristic. Non-tiered callers
are unaffected: they pass no policy and take Marlin's own config as before.

The constants are Booster-measured and must not be retuned elsewhere — the
login node prefers different grids, as Phase 26 found the hard way.

### A trap the unit test caught

The first version gated the tile on token count alone, and the unit test failed
with `Unsupported shapes ... thread_n_blocks = 16, thread_k_blocks = 4`. Marlin
only instantiates kernels for some `(thread_m, thread_n, thread_k)` block
combinations, and **an explicit tile with no instantiation raises at launch
rather than falling back to the heuristic**. `(64, 256)` exists at the 64-token
block that large batches select and not at the small blocks a short batch does,
so a token bound alone would have crashed any prefill whose batch happened to
pick a smaller block.

`MarlinLaunchPolicy.requires_block_m` now pins a tile to the block size it was
validated against, and `_fused_marlin_moe` checks it against the `block_size_m`
it is about to pass. The test asserts both directions: a supported tile is used,
and a tile pinned to a block size the call will not use is silently skipped in
favour of the heuristic. This is the fail-closed shape the rest of the tiered
path uses, and without it the change would have been a latent crash rather than
a 2% gain.

## Still to do

The kernel-level number is measured; the **server-level effect is not**. A
same-node A/B on the 16K prompt suite is the gate, and at a predicted ~2.2% it
needs the acceptance-free protocol to resolve — the 1.8% run-to-run spread on
aggregate throughput would swallow it otherwise.

## Files

| File | What |
| --- | --- |
| `job.sh` | The launch sweep |
| `job-test.sh` | Unit test for the multi-policy selection |
| `prefill-sweep-1241464.json`, `prefill-sweep-1241487.json` | Raw results |
| `slurm-*` | Run logs |
