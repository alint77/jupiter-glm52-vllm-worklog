# The Marlin shared-memory monopoly: why hot and cold never overlapped

Every hot/cold overlap experiment in this project has come back null or
negative: the `blocks_per_sm` sweeps (0.0-0.4%), the SM-budget split
(0.0-0.4%), the hot-first reorder (-0.94%), green contexts (FAIL-A, +9-40%
worse), and Grace->HBM staging (+9.5% to +44% worse at q4). This experiment
finds the single mechanism that explains all of them, and removes it.

**Result, all measured on Booster. At the kernel, under CUDA-graph replay, the
two-tier routed-MoE union drops by a median of 34% across 15 realistic
activated-expert cells (best 40.1%, worst 14.6%, every cell positive) and 31-45%
at the production c1/q4 shape, landing on `max(hot, cold)` - the overlap becomes
free. End to end on the production server, with both arms in one allocation and
draft acceptance divided out, step time falls 7.66% and output throughput rises
7.05% on the gate suite and 6.01% on a realistic agentic coding suite.**

## The mechanism

`csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu` computes how much shared
memory the kernel needs, and then launches with a different number:

```cpp
int sh_cache_size = get_kernel_cache_size(...);   // what the kernel indexes
...
kernel<<<blocks, num_threads, max_shared_mem, stream>>>(...)
```

`max_shared_mem` is `cudaDevAttrMaxSharedMemoryPerBlockOptin / blocks_per_sm`
(minus 1024 when `blocks_per_sm > 1`). That is deliberate occupancy control -
carve the SM into exactly `blocks_per_sm` slices - but it is written from a
single-kernel worldview. On GH200:

| `blocks_per_sm` | smem request/CTA | wave total | free per SM |
| ---: | ---: | ---: | ---: |
| 1 | 232,448 B | 232,448 | 1,024 B |
| 2 | 115,200 B | 230,400 | 3,072 B |
| **3 (production)** | **76,458 B** | **229,374** | **4,098 B** |
| 4 | 57,088 B | 228,352 | 5,120 B |

`232,448 / 3 - 1024 = 76,458` is exactly the per-CTA shared memory recorded in
the c4 MTP3 trace. The kernel's actual requirement at that tile
(`thread_k=64, thread_n=128, 128 threads`) is **25,856 B** - the launch asks for
three times what it uses.

At *every* `blocks_per_sm`, one Marlin wave claims ~100% of every SM's shared
memory, and the driver additionally reserves 1 KB per CTA. So **no second Marlin
CTA can ever be placed on any SM**, whatever the stream, the priority, the
launch order, or the tile. The hot and cold tiers are serialized by the block
scheduler before any of the knobs this project swept get a vote.

This is not GLM-specific and not tiered-MoE-specific. Any two concurrent Marlin
MoE launches on one device serialize.

## Why every previous experiment was consistent with this

- **`blocks_per_sm` sweeps.** The knob only re-carves the same 228 KB; the wave
  total is invariant. Structurally a null experiment.
- **"SM budget split across tiers".** `grid = sms * blocks_per_sm` reduces the
  cold tier's *CTA count*, but the hot tier's 396 CTAs still hold every SM's
  shared memory, so cold still cannot be placed.
- **Hot-first reorder.** Order is irrelevant when whichever kernel arrives first
  monopolizes. `apply_tiered` launches cold first (`run_tier(1)` on
  `aux_stream()`), so cold was taking the GPU; flipping it only swaps who waits.
- **Green contexts.** Green contexts are the one mechanism that *does* bypass
  the monopoly, because disjoint SM partitions have disjoint shared memory -
  which is why the mechanism probe passed. But partitioning SMs starves the cold
  tier of the memory-level parallelism it needs (measured there: 297.6 us @8 SM
  -> 117.3 us @32 SM) and shrinks hot at the same time. The right thing to
  partition is **shared memory across all 132 SMs**, not the SMs.
- **`2026-07-25-tier-overlap`'s "81.5% co-resident".** That metric is overlap of
  the two kernels' *time ranges*, which is guaranteed once both are enqueued. It
  cannot distinguish "CTAs sharing SMs" from "second kernel queued while the
  first runs". Measured per `%smid` below, CTAs from the two kernels never share
  an SM at the production request. The "no scheduling headroom" conclusion does
  not hold.

## Evidence 1 - the monopoly, in isolation

`analysis/smem_monopoly.cu`: two identical dummy kernels, two streams, swept
dynamic shared memory. `peak` is the maximum number of CTAs resident on any one
SM across *both* kernels, from per-CTA `%smid` + `%globaltimer`.

| dynSmem/CTA | occ/SM (one kernel) | union | overlap frac | **peak** |
| ---: | ---: | ---: | ---: | ---: |
| **76,800 (production)** | 3 | 23.79 ms | 0.38 | **3** |
| 57,344 | 4 | 22.89 | 1.00 | **4** |
| 48,000 | 4 | 22.88 | 1.00 | **4** |
| 32,768 | 6 | 22.89 | 1.00 | **4** |

At the production request the peak never exceeds the hot wave itself: the second
kernel's CTAs only start as the first's retire. Below 57,344 B they interleave.

## Evidence 2 - HBM and C2C do not fight

`analysis/tier_union.cu`: A streams 246 MB from HBM (19 experts' w13), B streams
39 MB from pinned Grace over C2C (3 experts), production geometry.

| smem/CTA | hot grid | hot solo | GB/s | cold solo | GB/s | **union** | vs `max` | peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 76,458 both | 396 (3/SM) | 0.328 ms | 3,144 | 0.494 ms | 331 | **0.811** | **+64%** | 3 |
| 57,088 both | 528 (4/SM) | 0.296 | 3,488 | 0.496 | 330 | 0.778 | +57% | 4 |
| 76,458 both | **264 (2/SM)** | 0.426 | 2,423 | 0.502 | 326 | **0.498** | **-0.8%** | 3 |
| 50,000 / 32,000 | 396 (3/SM) | 0.326 | 3,163 | 0.494 | 331 | **0.496** | +0.4% | 4 |

3.1 TB/s of HBM traffic and 331 GB/s of C2C traffic overlap with **zero**
dilation once the CTAs can co-reside. L2 `evict_first` cache hints on either
stream changed nothing (+-0.5%), so L2 capacity pressure is not the limiter for
the streaming portion either. The memory systems were never the problem; the SM
shared-memory allocator was.

This also retires the green-context report's leading hypothesis. The +45% hot
dilation measured there is not HBM/C2C contention - forced co-residency here
costs nothing. It is an artifact of confining the tiers to disjoint SM
partitions.

## The fix

Request what the kernel indexes, and let the caller size the grid:

```cpp
int launch_smem = max_shared_mem;
if (smem_mode == MARLIN_SMEM_TIGHT) {
  launch_smem = round_up_to(sh_cache_size, 128);
}
STD_TORCH_CHECK(launch_smem >= sh_cache_size, ...);
int launch_blocks = grid_blocks > 0 ? grid_blocks : blocks;
```

Two new optional arguments on `moe_wna16_marlin_gemm`, defaulting to the
existing behaviour. The tiered path then launches **hot at 2 CTAs/SM and cold at
1 CTA/SM**, both spread over all 132 SMs.

## Evidence 3 - the real Marlin kernel

> **These numbers are login-node measurements.** They are a matched A/B - every
> variant is timed inside the same round of the same process, so the *direction*
> is sound - but the login GH200 runs a 900 W cap against Booster's 680 W, and a
> slightly slower C2C, so neither the absolute times nor the hot/cold balance
> transfer. The balance is what sets the size of the win: going from serial to
> perfect overlap saves `1 - max(h,c)/(h+c)`, which is largest when the tiers are
> balanced. Login's faster SM clocks and slower C2C make cold *relatively*
> slower (242.5 vs 189.0 us here) than Booster's near-balanced 104.4 vs 115.3 us
> measured in [`../2026-07-25-grace-bandwidth`](../2026-07-25-grace-bandwidth/README.md)
> at this exact shape, which would make the login figure an *under*-estimate -
> but that is an argument for measuring, not assuming. The login GPU was also
> co-tenanted during these runs (hot solo drifted 121.6 -> 148.8 -> 189.0 us
> across successive runs). Booster numbers are in "Evidence 5"; the mechanism
> itself is hardware-independent arithmetic and holds identically on both.

`analysis/setup_tight.py` + `build_tight.py` stage an isolated copy of the real
kernel (bf16 x u4b8, `thread_m_blocks=1`, `group_blocks=8`; 6 instantiations, so
the build is minutes) so the fix could be measured before touching the installed
`_moe_C`. `analysis/ab_tight.py` interleaves every variant within each round, so
shared-GPU drift cancels. Login-node GH200, GLM geometry, Grace pages
NUMA-verified 100% local. Spread <=1%.

**c4 MTP3 straggler shape (m=16, 19 hot / 3 cold)** - `ab-m16.json`:

| layer | production | tight (264/132) | ideal `max(hot,cold)` | delta |
| --- | ---: | ---: | ---: | ---: |
| w13 | 424.4 us | **240.5** | 242.5 | **-43.3%** |
| w2 | 234.0 us | **137.7** | 138.4 | **-41.0%** |

**c1 q4 dominant shape (m=4, 6 hot / 1 cold)** - `ab-m4.json`:

| layer | production | tight (264/132) | ideal `max(hot,cold)` | delta |
| --- | ---: | ---: | ---: | ---: |
| w13 | 185.4 us | **95.8** | 106.8 | **-48.3%** |
| w2 | 99.2 us | **51.4** | 58.6 | **-48.1%** |

The fixed union sits **on** `max(hot, cold)`. There is no overlap headroom left
to chase after this.

Hot-solo timings are unchanged by the smem request alone (189.0 vs 188.4 us at
grid 396; 101.2 vs 100.7 for w2), so the win comes from co-residency, not from
some incidental L1-carveout effect.

## Evidence 4 - activated-expert sweep

`analysis/sweep_experts.py`. q4..q16 verification gives 32-128 routed
assignments per rank, i.e. roughly 8-32 distinct activated experts per rank per
layer, ~20% of them cold. Routing is constructed so exactly `hot + cold` experts
are activated with `top_k` distinct experts per token, which is what sets
Marlin's padded block count. Full w13+w2 chain; deltas are paired per-round
ratios (absolute us are not comparable across rows - shared login GPU).

| m | activated | hot/cold | cold% | delta |
| ---: | ---: | --- | ---: | ---: |
| 4 | 8 | 7/1 | 12% | **-57.2%** |
| 4 | 12 | 10/2 | 17% | -48.6% |
| 8 | 12 | 10/2 | 17% | -47.9% |
| 8 | 16 | 13/3 | 19% | -42.6% |
| 16 | 32 | 26/6 | 19% | -35.3% |
| 12 | 16 | 13/3 | 19% | -36.4% |
| 12 | 24 | 19/5 | 21% | -32.3% |
| 16 | 24 | 19/5 | 21% | -32.7% |
| 8 | 24 | 19/5 | 21% | -32.7% |
| 4 | 8 | 6/2 | 25% | -36.8% |
| 12 | 24 | 17/7 | 29% | -21.4% |
| 16 | 32 | 22/10 | 31% | -24.0% |
| 8 | 16 | 11/5 | 31% | -18.6% |
| 4 | 12 | 8/4 | 33% | -26.4% |

**Median -34.0%, best -57.2%, worst -18.6%.** The gain is monotone in cold
share: at the real ~20% operating point it is -32% to -49%, and above ~30% cold
the union becomes cold-bound so there is less to recover. It does *not* fade
with activated-expert count (26/6 at m=16 still returns -35%).

## Evidence 5 - Booster

`job-kernel-ab.sh` reruns the same A/B and the same activated-expert sweep on a
Booster node through the **in-tree** `ops.moe_wna16_marlin_gemm`, so it validates
the shipped integration rather than the staged copy, at production clocks and
C2C, with the GPU-paired Grace node detected from sysfs rather than hardcoded.

```bash
sbatch agent_space/experiments/2026-07-29-marlin-smem-monopoly/job-kernel-ab.sh
```

### The mechanism reproduces exactly

Booster reports the **same** shared-memory geometry as the login node -
`sharedMemPerMultiprocessor 233472 B`, `optin/block 232448 B` - so
`232448 / 3 - 1024 = 76,458` and the 229,374-of-233,472 wave are identical on
both. The monopoly probe:

| dynSmem/CTA | occ/SM | union | overlap frac | **peak CTAs on one SM** |
| ---: | ---: | ---: | ---: | ---: |
| **76,800 (production)** | 3 | 2381.9 ms | 0.39 | **3** |
| 57,344 | 4 | 2292.6 | 1.00 | **4** |
| 48,000 | 4 | 2289.6 | 1.00 | **4** |
| 32,768 | 6 | 2293.4 | 1.00 | **4** |

### The two-tier union, at production clocks

246 MB from HBM against 39 MB from pinned Grace, GLM geometry:

| smem A/B | hot grid | hot solo | GB/s | cold solo | GB/s | **union** | vs `max` | peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 76,458 both | 396 (3/SM) | 0.337 ms | 3,066 | 0.400 ms | 408 | **0.720** | **+79.9%** | 3 |
| 76,458 both | **264 (2/SM)** | 0.430 | 2,399 | 0.397 | 412 | **0.434** | **+0.8%** | 3 |
| 50,000 / 32,000 | 396 (3/SM) | 0.338 | 3,050 | 0.399 | 410 | **0.400** | **+0.1%** | 4 |

**-40% union with the shipped configuration**, landing on `max(hot, cold)`.
Booster's C2C is *faster* than the login node's (408-417 vs 326-331 GB/s) while
its HBM is slightly slower (3,066 vs 3,144 GB/s), so the two tiers are more
balanced here, not less.

### Server end-to-end, matched c1/q4

The first attempt (jobs 1088069/1088070) measured +9.71% output tok/s, but ran
its arms on **different nodes** and with **unmatched draft acceptance**, so it
could not attribute the gain. `job-samenode.sh` reruns both arms sequentially in
one allocation. Job 1090344, 24-prompt suite, one excluded warmup and two
measured repetitions per arm:

| | off | on | delta |
| --- | ---: | ---: | ---: |
| output tok/s | 98.35 [98.24, 98.46] | **105.28** [104.60, 105.95] | **+7.05%** |
| TPOT (ms) | 9.183 | 8.436 | -8.13% |
| tokens per target step | 2.9212 | 2.9360 | +0.51% |
| **step time = TPOT x tokens/step** | **26.82 ms** | **24.77 ms** | **-7.66%** |

With both arms on one node the acceptance difference collapses from 2.4% to
0.51%, so the throughput gain is now almost entirely step time. The
acceptance-adjusted **-7.66%** agrees with the -7.2% derived from the confounded
run, which means the original +9.71% was inflated by roughly 2.5 points of node
and acceptance luck. **The honest server-level figure is about +7%.**

Both arms reproduced the semantic gate with byte-identical output.

### Realistic agentic decode

The suite this project gates on is 24 prompts of roughly 40 input tokens. A real
agentic coding turn carries the file under edit, profiler tables and stack
traces, so `make_agentic_prompts.py` builds 16 prompts that embed **real source
from this checkout** plus authentic profiler tables, build errors and stack
traces, and ask for a diagnosis or a patch: 131-2277 input tokens (median ~800),
512 output tokens. The tasks come from this session's own work - occupancy
analysis, split-K cost, reading a critical-path budget, a shared-workspace
illegal access, CUDA-graph capture with a forked stream, reduction-order
numerics against a golden SHA, and a review of the patch this experiment ships.

Job 1090343, production c1/q4, both arms in one allocation:

| | off | on | delta |
| --- | ---: | ---: | ---: |
| **output tok/s** | 94.13 | **99.79** | **+6.01%** |
| TPOT (ms) | 10.098 | 9.509 | -5.83% |

Agentic prompts gain slightly less than the short-prompt suite (+6.0% vs +7.1%),
as expected: longer inputs add attention and KV work per step, diluting the
routed-MoE share this fix targets. The `off` arm measured 94.13 here against
93.50 on a different node earlier - 0.7% apart, a useful check that the suite is
stable across nodes.

Prefix caching is disabled to match the existing gate methodology, so this is
decode for *fresh* turns; a real multi-turn session re-prefixes heavily, which
improves TTFT but not decode.

### The activated-expert sweep, and a harness error that inverted its conclusion

The first Booster sweep (`booster-kernel-ab.json`) reported roughly 0% at low
activated-expert counts and -20% to -28% at high ones, which would have meant
the fix bought nothing at this deployment's operating point. **That conclusion
was wrong, and the tell was in its own numbers**: at m=4 with 7 hot and 1 cold,
hot solo 105.6 us plus cold solo 106.0 us is 211.5 us serial, but the measured
production union was 319.2 us. Running the tiers concurrently came out *slower
than running them back to back*, which no co-residency story can produce.

The cause was the harness. `both()` performs `aux.wait_stream(main)` and
`main.wait_stream(aux)` **per iteration in eager mode**, so each of the fifty
timed iterations hit two stream barriers that stop the host running ahead and
expose launch latency; the solo timings have no barriers and pipeline freely.
That floor is ~110 us on Booster and ~80 us on login - larger than the entire
kernel at low activated-expert counts, which is exactly where the sweep read
zero. Production pays none of it, because it replays the same fork/join from
inside a CUDA graph.

Re-measured under graph replay, with every variant captured and replayed the way
the production runner does:

| shape | layer | production | shipped (264/132) | best grid | ideal |
| --- | --- | ---: | ---: | ---: | ---: |
| m=4, 7 hot / 1 cold | w13 | 102.6 | **61.4** (-40%) | 56.0 @ 264/66 | 63.9 |
| | w2 | 62.1 | **28.5** (-54%) | 28.5 @ 264/132 | 33.8 |
| m=16, 19 hot / 5 cold | w13 | 520.8 | **392.6** (-25%) | 369.6 @ 132/132 | 364.8 |
| | w2 | 283.6 | **182.2** (-36%) | 182.2 @ 264/132 | 182.3 |

Full w13+w2 chain with the shipped policy: **-45.4% at m=4 (7 hot / 1 cold)**
and -28.5% at m=16 (19 hot / 5 cold). The low-count shape gets the *largest*
win of anything measured, and lands within 6 us of `max(hot, cold)`.

Production union under graph replay is 102.6 us against 113.1 us serial, so the
tiers still barely overlap without the fix - the monopoly is real under graph
replay too, not only in eager mode.

### Grid shape: second-order, and the login optimum does not transfer

The cold tier's cost does depend on its grid, for the expected reason: with one
activated cold expert w13 has only `n_tiles = 32` MN-tiles, so a 132-CTA launch
makes Marlin split K about four ways and cooperate through global barrier
spin-locks plus an fp32 `C_tmp` reduction.

On the **login node** this made 66 CTAs clearly better than 132 for cold w13
(56.6 vs 71.6 us solo), and the shipped fixed policy came out ~3 percentage
points off the best per-shape grid. **On Booster that is not true.** Job 1088768
searched twelve hot/cold grid combinations per layer at four low-count shapes,
and the shipped `{"hot": 2, "cold": 1}` (264/132) is the best of them in five of
six cases and 0.3% off in the sixth:

| shape | layer | production | shipped 264/132 | best searched | ideal |
| --- | --- | ---: | ---: | ---: | ---: |
| m=4, 7 hot / 1 cold | w13 | 86.1 | **54.5 (-36.7%)** | 54.5 @ 264/132 | 53.6 |
| | w2 | 50.9 | **28.1 (-44.8%)** | 28.1 @ 264/132 | 32.5 |
| m=4, 6 hot / 2 cold | w13 | 103.4 | 71.1 (-31.2%) | 70.9 @ 132/132 | 76.2 |
| | w2 | 58.4 | 39.0 (-33.2%) | 38.4 @ 132/132 | 42.1 |
| m=8, 10 hot / 2 cold | w13 | 121.6 | **81.9 (-32.7%)** | 81.9 @ 264/132 | 75.6 |
| | w2 | 69.2 | **41.4 (-40.2%)** | 41.4 @ 264/132 | 43.0 |
| m=8, 13 hot / 3 cold | w13 | 163.6 | **107.8 (-34.1%)** | 107.8 @ 264/132 | 104.4 |
| | w2 | 87.1 | **57.2 (-34.4%)** | 57.2 @ 264/132 | 54.7 |

Booster's cold-solo optimum is 132 CTAs, not the 66 the login node preferred.
Had the login result been trusted, the shipped constant would have been retuned
to a value that is **wrong on the machine that matters** - the same trap as the
absolute-timing transfer, one level down. `{"hot": 2, "cold": 1}` stands, now
for a measured reason rather than a default.

### The full sweep on Booster, under graph replay

`booster-kernel-ab-graph.json`, 15 cells, every variant captured and replayed:

| m | act | hot | cold | cold% | prod | tight | ideal | delta | eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8 | 7 | 1 | 12% | 136.4 | 81.7 | 85.0 | **-40.1%** | 288.2 |
| 4 | 12 | 10 | 2 | 17% | 190.5 | 122.4 | 118.3 | -35.8% | 287.6 |
| 8 | 16 | 13 | 3 | 19% | 251.0 | 164.1 | 159.8 | -34.6% | 322.6 |
| 8 | 24 | 19 | 5 | 21% | 395.5 | 259.5 | 256.4 | -34.4% | 410.5 |
| 16 | 24 | 19 | 5 | 21% | 395.5 | 261.1 | 257.0 | -34.0% | 411.0 |
| 16 | 32 | 26 | 6 | 19% | 489.5 | 322.0 | 299.2 | -34.2% | 507.4 |
| 8 | 16 | 11 | 5 | 31% | 337.0 | 287.8 | 253.3 | -14.6% | 369.5 |
| 16 | 32 | 22 | 10 | 31% | 651.2 | 487.5 | 498.5 | -25.1% | 658.0 |

**Median -34.0%, best -40.1%, worst -14.6%, every cell positive.** `tight` sits
on `ideal` throughout. The `eager` column is the same production configuration
timed without graph capture, and it is where the earlier ~0% readings came from:
288.2 us eager against 136.4 us graphed for the identical cell.

The win is monotone in **cold share**, not in expert count: 12-21% cold gives
-34% to -40%, 31% cold gives -15% to -25%. The production c1/q4 point (~5.9
activated per rank, ~1.1 cold, i.e. 12-19% cold) is at the favourable end.

### Isolating the kernel from MTP acceptance

Because acceptance moved between the arms, the MTP3 numbers cannot cleanly
attribute their delta to the kernel. `job-nomtp.sh` (1088527) removes the
speculative path entirely and drives the decode batch with concurrency instead:
without drafting, every engine step emits exactly one token per sequence, so
mean TPOT **is** the step time and `M` equals the concurrency. Both arms run
sequentially in one allocation at concurrency 4, 8 and 16, with graphs captured
at all three sizes and `tiered_overlap_max_tokens = 1 x 16`, so the overlap path
and the launch policy apply throughout.

This is a different routing regime rather than merely a quieter one, and the
difference cuts against the fix if anything: `M = 4` from four independent
sequences activates more distinct experts than `M = 4` from four consecutive
positions of one sequence, so more experts are live per layer. It is a
diagnostic; the MTP3 run remains the production gate.

Summarize with `analysis/summarize_nomtp.py`.

Two contract guards had to be worked around to get M>1 without speculation:
`max_num_seqs` is pinned to 1-4 (`vllm/config/vllm.py:2302`), and >1 additionally
requires DCP because the replicated 400K MLA cache does not fit more than one
sequence. The job therefore mirrors the qualified c4 configuration (V2 runner,
DCP4, `max_num_seqs 4`, 7 GB reserve) with the speculative config removed.
**Pending** (jobs 1088769/1089276 lost to the scratch inode quota; rerun as
1090345).

## Realistic agentic decode throughput

The suite this project gates on is 24 prompts of roughly 40 input tokens. A real
agentic coding turn carries the file under edit, profiler tables and stack
traces. `make_agentic_prompts.py` builds a 16-prompt suite that embeds **real
source from this checkout** (`ops.cu` launch path, `marlin_template.h` split-K
and barrier, `apply_tiered`, `marlin_moe.py`, `tiered_moe_execution.py`) plus
authentic profiler tables, build errors and stack traces, and asks for a
diagnosis or a patch: 131-2277 input tokens (median ~800), 512 output tokens.
The tasks are drawn from this session's actual work - occupancy analysis,
split-K cost, reading a critical-path budget, a shared-workspace illegal access,
CUDA-graph capture with a forked stream, reduction-order numerics against a
golden SHA, and a review of the very patch this experiment ships.

Production c1/q4 (MTP3, TP4/EP4, tiered, full graphs at size 4), 16 requests:

| metric | off (fix disabled) |
| --- | ---: |
| output tok/s | **93.50** [94.01, 92.99] |
| TPOT (ms) | 10.11 |
| ITL (ms) | 28.71 |
| TTFT (ms) | 308.3 |

The `on` arm is **pending** (job 1090343). Decode is slightly below the 96.53 of
the short-prompt suite, as expected from longer inputs adding attention work per
step. Prefix caching is disabled to match the existing gate methodology, so this
is decode for *fresh* turns; a real multi-turn session re-prefixes heavily, which
improves TTFT but not decode.

## Checkpoint loading: exa_fscratch cuts it from 14 minutes to 2

Checkpoint load is the slowest part of every job in this project. Measured
end-to-end on the real c4 serving path (`claude-local-c4.sh` config: V2 runner,
DCP4, `max_num_seqs 4`, MTP3), same model, same config, only
`TIERED_MOE_MODEL_PATH` differing:

| filesystem | full 86-shard load | s/shard |
| --- | ---: | ---: |
| `exa_project1` (current home) | **14 min 08 s** | 9.87 |
| `exa_fscratch` (staged) | **2 min 11 s** | 1.52 |
| | **-85%** | **6.5x** |

The project1 figure is identical to the second across this morning's three
independent runs on three different nodes (jobs 1090343/1090344/1090345, all
14:08), so it is a solid uncongested baseline. A fourth sample under contention
ran 19.1 s/shard and is deliberately excluded - using it would flatter fscratch.

The mechanism is the one the microbenchmark predicted. Four ranks load
concurrently, and that is precisely where the two filesystems diverge: parallel
O_DIRECT scales 11 -> 21 GB/s on fscratch across 4 to 8 readers while project1
manages 0.55 -> 0.95. The loader was never the bottleneck.

Staging costs 86 s at 5.06 GB/s for 405 GB (`stage_model_fscratch.sh`, which
verifies per-file sizes), against 12 minutes saved per server start - net
positive on the first launch and free thereafter if the copy persists.

**Open question before adopting it:** the baseline README records that
ExaFlash/`exa_fscratch` was investigated and abandoned early in the project. On
performance this is unambiguous, so the reason was probably retention or
capacity. If fscratch purges, this becomes a stage-once-per-session step (still
net positive); if it persists, `TIERED_MOE_MODEL_PATH` can simply point there.

Two incidental findings from the same runs:

- Server *startup* before weight loading is dominated by reading the venv's many
  small files from project1, the access pattern project1 is worst at. Staging
  the venv is a separate lever from staging the model.
- One allocated node (`jpbo-059-16`) read at ~100 KB/s during imports and had to
  be abandoned. It coincided with a cluster health event - 54 rising to 66 nodes
  draining with `pshealthcheck ... Not responding` - which also explains why job
  dispatch stretched from the usual 5-25 minutes to over 40.

## Operational note: the scratch inode quota

Three server jobs died mid-startup with `OSError: [Errno 122] Disk quota
exceeded` inside Inductor autotuning. The cause was **file-count exhaustion on
`/e/scratch`, not space**: a 512 MB write succeeded while only five files could
be created, and the same probe on `/e/project1` created 4000/4000. Compile caches
write tens of thousands of tiny files, so they hit an inode quota long before a
space quota.

Two contributing factors, one of them self-inflicted:

- Every job here was given its own `VLLM_CACHE_ROOT`, so caches multiplied per
  job instead of being reused. Fixed by keying them per *arm* rather than per
  job id, which still keeps concurrent arms off each other's caches.
- The dominant consumer was the shared `vllm-cache` root itself. Clearing it
  restored the probe from 5/4000 to 2000/2000.

Measured inode counts, before the purge:

| directory | inodes | disposition |
| --- | ---: | --- |
| `vllm-cache` | **343,049** | deleted (regenerable) |
| `cache` (uv/triton/pip) | 249,113 | kept |
| `vllm-k3-venv` | 85,959 | kept - active K3 work |
| `vscode-server` | 52,457 | kept |
| `vscode-ext` | 14,575 | kept |
| `claude-routing-profile-1047954-108` | 110 | kept - frozen dataset |
| `glm52-fp8-mtp-layer78` | 7 | kept - model artifact |

Clearing `vllm-cache` alone returned 343k inodes and restored the probe from
5/4000 to 2000/2000. `cache` is the next-largest reserve if more headroom is
ever needed, but it holds uv and triton state in daily use.

The per-job leftovers that look like clutter are a red herring: the ~169
`vllm-cache-<JOBID>` / `trtllm-dg-<JOBID>` directories are empty (512 B and 1
inode respectively) and free nothing. `du --inodes` is also unusable as a survey
tool here - it needs minutes per large directory on this filesystem, and the fact
that only two directories failed to finish was itself the diagnosis. All job scripts here now cache under
`/e/project1/profound/alint77/.marlin-caches/` so this experiment does not
consume the scratch inode budget at all.

Two confounds are being closed by follow-up jobs: the two arms ran on
**different nodes** (`jpbo-049-30` and `jpbo-101-13`), so `job-samenode.sh`
(1088490) runs both arms sequentially inside one allocation; and the
activated-expert sweep (1088491) reruns with strict `numactl` binding, without
which the pinned Grace allocations intermittently land off-node and the tiered
allocator's locality audit fails closed.

## Numerics

- The legacy path is repeatable run to run (`legacy_repeatable=True`).
- Tight smem **at the same grid is bit-exact** against legacy, for both tiers.
- Tight smem at a *different* grid differs by at most one bf16 ulp (0.00195 for
  w13, 0.000488-0.00195 for w2), because the grid changes Marlin's DP/split-K
  decomposition and therefore the fp32 accumulation order. Each grid reproduces
  itself exactly, so the path stays deterministic - the same class of change as
  the staged-vs-direct rounding recorded in
  [`../2026-07-27-green-context-marlin`](../2026-07-27-green-context-marlin/README.md).

**Consequence: the exact-400K golden SHA is expected to change.** A mismatch
after this change is a reduction-order artifact, not a regression; the gate must
be re-established rather than treated as a failure.

## Tests

| Gate | Result |
| --- | --- |
| `test_fused_marlin_moe_launch_policy` | pass - each grid is exactly reproducible, agrees with the default launch within 2e-2, and the `max_tokens` gate reproduces the default launch bit-exactly above the threshold |
| `test_tiered_moe_launch_policy_reaches_the_experts` | pass - the policy lands on `kernel.impl.fused_experts`, and a non-Marlin backend raises |
| `tests/kernels/moe/test_moe.py -k marlin` | 257 passed, 6 skipped |
| `analysis/graph_capture_check.py` | pass - the two-tier fork/join captures under a CUDA graph, replay is bit-exact against eager, and replay is repeatable |

Two integration defects were found and fixed while wiring this up, both of which
would have shipped silently:

1. The policy was applied to every call through the tiered path, including
   large-M prefill, where the tiers do not overlap at all and shrinking the grid
   only loses parallelism. It is now gated on `M <= max_tokens`, wired from the
   same `tiered_overlap_max_tokens` that decides whether `apply_tiered` forks a
   stream.
2. The policy was attached to `kernel.impl`, but the Marlin experts object that
   the gemm reads is one level deeper at `kernel.impl.fused_experts`. The
   `isinstance` check therefore always failed and the policy was never set - a
   complete no-op in production, while every kernel-level test still passed
   because those call `fused_marlin_moe` directly with an explicit policy. The
   lookup now resolves the right object and raises instead of skipping, and
   `test_tiered_moe_launch_policy_reaches_the_experts` covers it.

## Source changes

| File | Change |
| --- | --- |
| `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu` | `MARLIN_SMEM_LEGACY/TIGHT`, `round_up_to`, `smem_mode`/`grid_blocks` on `marlin_mm` and the op, fail-closed check that the request is never below `sh_cache_size` |
| `csrc/libtorch_stable/moe/torch_bindings.cpp` | schema extended by two ints |
| `vllm/_custom_ops.py` | passthrough, fake-op update, `MARLIN_SMEM_*` constants |
| `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` | `MarlinLaunchPolicy` threaded to both gemms; optional per-instance policy on `MarlinExpertsBase` |
| `vllm/model_executor/model_loader/tiered_moe_execution.py` | per-tier policy at kernel construction: hot 2 CTAs/SM, cold 1 CTA/SM, `grid_blocks = SMs x CTAs/SM` from device properties |

Defaults are unchanged, so every non-tiered Marlin user keeps the current
behaviour. `batched_fused_marlin_moe` is deliberately left on the legacy policy.

## Reproduce

```bash
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
D=agent_space/experiments/2026-07-29-marlin-smem-monopoly/analysis

# mechanism, no vLLM needed
nvcc -O3 -arch=sm_90a -o /tmp/smem_monopoly $D/smem_monopoly.cu && /tmp/smem_monopoly
nvcc -O3 -arch=sm_90a -o /tmp/tier_union   $D/tier_union.cu   && /tmp/tier_union 4

# real Marlin, isolated build (does not touch the installed _moe_C)
.venv/bin/python $D/setup_tight.py && .venv/bin/python $D/build_tight.py
.venv/bin/python $D/ab_tight.py --m 16 --hot 19 --cold 3
.venv/bin/python $D/sweep_experts.py
```

`setup_tight.py` patches its staged copy of `ops.cu`; run it against a checkout
that does **not** already contain the in-tree fix, or drop its patch step.

## Post-fix production trace (2026-07-29)

`job-profile-ab.sh` captured a matched c1/q4 MTP3 off/on pair on one node.
A 45-minute backfill copy started at the same time, so the result replicated on
a second node. Both used the staged AutoRound checkpoint on `exa_fscratch`, the
qualified `hybrid-p0.5` placement, one realistic prompt, and twelve profiled
M=4 target steps per rank. Each arm produced four Perfetto-compatible
`*.pt.trace.json.gz` files.

| mean over four ranks and eleven bounded steps | 1092954 off | on | delta | 1092955 off | on | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| engine-step wall | 25.194 ms | 24.281 | **-3.63%** | 25.741 | 24.333 | **-5.47%** |
| GPU busy union, correlation-attributed | 24.800 ms | 23.926 | **-3.52%** | 25.504 | 23.756 | **-6.85%** |
| GPU idle inside the attributed span | 1.925 ms | 1.953 | +1.45% | 1.989 | 2.066 | +3.87% |
| target-graph span | 23.944 ms | 23.126 | **-3.42%** | 24.679 | 22.925 | **-7.11%** |
| routed-layer span, 75 layers | 8.286 ms | 7.577 | **-8.56%** | 9.075 | 7.703 | **-15.12%** |
| Marlin cumulative duration | 12.779 ms | 9.819 | -23.2% | 13.987 | 9.974 | -28.7% |

Across the two nodes, mean step wall falls **25.468 -> 24.307 ms (-4.56%,
1.16 ms)**. The routed-layer critical spans save a mean 1.04 ms, accounting for
about 90% of the engine-step improvement. GPU idle is flat to slightly higher:
the gain is less device work on the critical path, not reduced host gaps.

Read this trace as **mechanism attribution, not magnitude**. It is a single
prompt at c1/M=4 under an active profiler over eleven steps, and the delta is
smaller than the throughput measurements of the same fix (-5.3% to -6.7% step
time without MTP, -8.1% with MTP3 on the same node; see the sections above).
Those suites are the authority on how large the win is; this capture explains
*where* it comes from. Note also that the no-MTP sweep has the win growing with
concurrency, so a c1 capture understates the c4 production path.

The two captures also do not replicate equally well on the two arms:

| cross-node spread, on the same metric | off arm | on arm |
| --- | ---: | ---: |
| engine-step wall | +2.2% | **+0.2%** |
| Marlin cumulative | +9.5% | **+1.6%** |
| routed-layer span | +9.5% | **+1.6%** |
| cold-tier cumulative | +5.4% | **+0.2%** |

With n=2 this is a hypothesis, not a result, but the same pattern appears on
four independent metrics: **under the monopoly the routed path is 5-10x more
sensitive to whatever differs between nodes; with the fix both nodes agree to
within 2%.** The mechanism is consistent with it - serialization makes the span
depend on how fast each tier's queue drains, which tracks clocks and power, and
removing it puts cold on a bandwidth-bound path that does not. If it holds at
larger n, variance reduction is a second benefit of the fix and matters for tail
latency independently of the mean. The reported -4.56% is a mean of -3.63% and
-5.47%; it carries no usable interval at n=2.

The cumulative Marlin reduction is larger than the routed-span reduction
because a profiler kernel interval includes time resident or waiting while
another kernel uses the SM. Tight shared memory removes that queueing and lets
the tier epilogues (`act_and_mul`, `moe_sum_vec`) become resident sooner as
well. Conversely, timeline-overlap percentage falls from about 35% to 23%
because the individual intervals become shorter. As established by the `%smid`
probe, timeline overlap is not CTA co-residency and must not be used as the
physical overlap metric.

### What dominates now

Representative post-fix target graph, averaged across the two captures:

| component | time/rank-step | interpretation |
| --- | ---: | --- |
| routed hot/cold layer span | **7.64 ms** | about 33% of target-graph span |
| dense/shared GEMMs | **5.68 ms cumulative** | compile variants differ, but off/on totals agree within 0.5% |
| TP custom all-reduce | **4.65 ms cumulative** | 157 calls; long synchronization tails |
| FlashMLA | **1.77 ms cumulative** | unchanged by the fix |
| target-graph idle | **0.985 ms** | 2,430 dependency gaps, mean 0.40 us |

The cumulative rows overlap and therefore do not add to the graph span.

### Additive whole-step budget

To make the overlapping trace sum, partition every GPU timeline interval among
the unique kernel families active in that interval (equal share when families
overlap), then keep GPU-empty time as its own bucket.

`analyze_step_budget.py` does this with **launch-correlation attribution**, not
GPU-timestamp-versus-CPU-annotation boundaries. That distinction matters: the
boundary method used by `analyze_profile_ab.py` silently drops the kernels of a
step that execute after the CPU has moved on, and the loss is measurable against
the census `analyze_comms.py` proves exact.

| per rank-step | boundary-attributed | correlation-attributed | truth |
| --- | ---: | ---: | ---: |
| custom all-reduces | 156.7 | **166.0** | 166 |
| Marlin GEMMs | 288.5 | **306.0** | 306 |
| vocabulary all-gathers | 3.6 | **4.0** | 4 |

The corrected census is exact in **every one of the 176 rank-steps across all
four arms**, and the script fails closed if it is not. 306 = 2 GEMMs for each of
the 75 routed target layers plus the three MTP draft layers.

Across both post-fix captures (88 rank-steps):

| additive component | ms/step | share | cross-node spread |
| --- | ---: | ---: | --- |
| routed experts (W4 Marlin) | 6.757 | **26.14%** | 6.694-6.820 |
| dense/shared-expert compiled GEMMs | 5.684 | **21.99%** | 5.682-5.685 |
| TP/EP communication and synchronization | 5.051 | **19.54%** | 4.890-5.211 |
| glue, elementwise and uncategorized kernels | 2.963 | **11.46%** | 2.960-2.966 |
| GPU-empty host/graph gaps | 2.010 | **7.77%** | 1.953-2.066 |
| attention (FlashMLA + DSA + KV) | 1.886 | **7.30%** | 1.880-1.892 |
| MoE routing, activation and sum | 1.500 | **5.80%** | 1.499-1.502 |
| **total (attributed GPU span)** | **25.851** | **100%** | 25.822-25.880 |

The correction **confirms the shares and revises the absolutes**. Every
percentage moves by less than 0.5 points from the boundary-attributed version,
because the dropped kernels were spread nearly uniformly across families, so
numerator and denominator shrank together. The absolute milliseconds are about
6% higher. GPU-empty is essentially unchanged (2.010 against 2.045), so the
earlier suspicion that boundary loss inflated the idle bucket does not hold.

The total is the step's **attributed GPU span**, 25.851 ms, not the 24.307 ms
engine-step wall. The earlier version forced the budget to sum to the wall,
which is wrong by construction: consecutive steps' GPU spans overlap by a mean
1.54 ms, because the CPU has already launched step N+1 while step N's tail is
still draining, and the aux cold stream lets the two coexist. Quote the
percentages, or quote ms against the 25.851 ms span - do not read the ms column
as a partition of the 24.307 ms wall.

The directly attributed routed-MoE path is 31.9% after adding Marlin to its
routing/activation/sum epilogue. The dense/shared bucket also contains shared
expert work and compiled MTP/dense GEMMs that use the same nvjet/Triton kernel
families, so the trace cannot split that 22% cleanly without additional graph
annotations. Communication includes synchronization wait, not just bytes on
NVLink. This budget is for c1/q4 at short realistic context; attention's share
will grow with context length.

### The machine is a quarter idle, not 8%

The budget scores 7.77% as GPU-empty, which reads like a well-fed device. It is
not. A rank that arrives early at a custom all-reduce spins **inside** the
kernel until its peers arrive, and the profiler scores every microsecond of that
spin as GPU-busy. Adding the measured synchronization excess to the empty
bucket:

| doing no useful work, per step | ms | share of the 24.307 ms wall |
| --- | ---: | ---: |
| GPU-empty host/graph gaps | 2.010 | 8.27% |
| custom-AR synchronization excess | 4.128 | 16.98% |
| **total** | **6.138** | **25.3%** |

A quarter of every decode step is spent either with an empty device or with SMs
burning cycles in a barrier. That, not any single kernel, is the headline number
for what remains.

### The cold tier is at the C2C roofline; there is no kernel work left in it

Splitting the routed span by tier, averaged over both post-fix captures:

| post-fix routed path | ms/step | cross-node spread |
| --- | ---: | --- |
| hot cumulative | 5.917 | 5.843-5.991 (**2.5%**) |
| cold cumulative | 3.980 | 3.976-3.984 (**0.2%**) |
| Marlin cumulative | 9.897 | 9.819-9.974 |
| routed layer span | 7.640 | 7.577-7.703 |
| span above `max(hot, cold)` | **1.723** | 1.712-1.734 |

Cold carries about 20% of routed traffic under `hybrid-p0.5` but 40% of Marlin
time, which reads as a 2.7x per-token defect in the cold path. It is not a
defect. One W4G64 expert is 12.58 MB of `w13` plus 6.29 MB of `w2` plus about
1.2 MB of group scales, so **20.1 MB**. Cold time per layer is
3.980 ms / 75 = **53 us**. At roughly one distinct cold expert per rank per layer
at M=4, that is **379 GB/s** - against the **373 GB/s** that
[`2026-07-25-grace-bandwidth`](../2026-07-25-grace-bandwidth/README.md) measured
as the achievable pinned-Grace read rate, 88-95% of the 421 GB/s C2C roof.

Two independent checks support the roofline reading rather than a coincidence.
First, cold is **node-invariant to 0.2%** while hot varies 2.5%: bandwidth-bound
work does not track clocks, compute-bound work does. Second, the pre-fix cold
figure is 5.957 ms, 50% higher, because under the monopoly a cold kernel's
interval includes time resident but starved of SMs - the same artifact that
inflates the pre-fix Marlin cumulative. Only the post-fix number is a
bandwidth measurement.

Three consequences, in decreasing order of how much work they save:

1. **Do not tune the cold Marlin kernel.** It is running at the memory roof of
   the path it reads from. Tile shapes, split-K, grid constants and occupancy
   have nothing to win there, which retires a class of experiment this worklog
   has already run several variants of.
2. **The fix never made cold faster and was never supposed to.** It let hot
   compute proceed *during* cold's transfer. That is why the win is largest at
   low activated-expert counts, where cold's fixed C2C cost dominates a small
   hot chain.
3. **Remaining overlap headroom is 1.723 ms/step**, the span above
   `max(hot, cold)`, including the `moe_sum`/`act_and_mul` epilogue (about
   1.06 ms excluding it). That is a bounded and well-defined target, unlike
   "improve overlap".

Cold *volume* is not buyable either: 11,480 of 19,200 expert slots are already
hot (59.8%, about 57 GB/rank), HBM is the binding constraint, and Grace-to-HBM
staging was tested and refuted twice in this worklog
([staging hurts +16-34%](../2026-07-27-green-context-marlin/README.md)). The
lever on cold is placement quality, not kernels and not more HBM.

### Where the GPU-empty time actually is

`analyze_graph_gaps.py` splits the empty bucket by position relative to the CUDA
graph replays. Averaged over both post-fix captures:

| GPU-empty, per rank-step | ms | note |
| --- | ---: | --- |
| inside graph replays | 1.019 | 0.985 of it in the target graph |
| between graph replays | 0.991 | |
| — after the target graph | **0.745** | one boundary out of seven |
| — across the other six boundaries | 0.246 | 0.02-0.07 ms each |
| host not inside any CUDA API during those gaps | 0.703 | |
| GPU time in kernels outside every graph | 1.071 | 0.561 of it one GEMM |

Two things this settles.

**The MTP drafts are already graphed, so "capture the draft path" is not a
lever.** The step replays seven graphs, not one: the target at 3,466 kernels,
then a 16-kernel and a 20-kernel graph for each of the three draft rounds. The
six draft graphs contain 0.003-0.008 ms of internal idle *each*. There is nothing
to win there.

**Two thirds of the between-graph idle is one boundary: the sampling and logits
region right after the target graph.** It costs 0.745 ms of device idle plus
1.071 ms of un-graphed GPU work, dominated by 0.561 ms of
`nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT` - the vocabulary projection, which the
eager trace confirms as `[4, 6144] x [6144, 38720]` for the target and
`[1, 6144] x [6144, 38720]` for each draft (38,720 = 154,880/4). The rest is the
logits all-gather, MLA metadata, a pinned HtoD copy and sampler reductions. For
0.703 ms of that idle the host is inside no CUDA call at all, so it is Python
sampler work, not launch latency - which means graph capture alone would not
recover it.

The whole region is about 1.8 ms of the 24.307 ms step (7.5%), of which 0.745 ms
is device idle. It is a c1 measurement, so the per-step host cost amortises over
more tokens at higher concurrency.

The gap structure is **identical between the off and on arms** (in-graph 0.991
against 1.019, between-graph 0.966 against 0.991, host-dark 0.703 in both), which
is the right invariance: the shared-memory fix touches the device schedule and
nothing on the host path. It is also a check on the measurement itself.

The target is replayed as one CUDA graph, so its 0.985 ms of idle cannot be
CPU-per-kernel launch latency. The trace contains seven graph launches and 140
eager launches per engine step overall, but the shared-memory fix leaves those
counts exactly unchanged while GPU busy time falls. Within the target graph,
the 2,430 gaps are extremely small (p50 0.42 us, p90 0.51 us, maximum under
1.8 us). Eliminating every one would cap the gain at roughly 1 ms; graph-node
fusion is now a secondary lever, not the first one.

The first remaining lever is **per-layer EP-rank balance**. Aligning the 75
routed layers across all four ranks in each step and summing
`max(rank span) - mean(rank span)` gives **3.615 ms/step post-fix**
(3.547 and 3.682 on the two captures, per-step range 3.095-4.751), or **47% of
the 7.640 ms routed span**. Pre-fix it is 3.885 ms, so the shared-memory fix
barely touched the skew - it is a placement property, not a launch property.

This supersedes an earlier figure of 1.485 ms in this report, which came from a
single capture and did not align layers across all four ranks. The corrected
number matters because it **reconciles with the communication measurement**: the
custom all-reduces show 4.128 ms/step of synchronization excess, and 3.615 ms of
per-layer routed skew is the same phenomenon measured with a different estimator
(`max - mean` over ranks per layer, against `mean - min` over ranks per
collective). Agreement to within 12% is the strongest evidence in this report
that **routed-layer rank skew is what the all-reduce tails are waiting on**. The
old 1.485 ms figure invited the opposite reading, that skew and AR wait were
separate and additive levers worth 5.6 ms together. They are one lever worth
roughly 3.6-4.1 ms.

Per-layer excess averages 48 us against a 102 us mean layer span, so the slowest
rank routinely runs about 1.5x the mean. The worst offenders are stable across
captures - layers 34, 69, 18, 20, 61, 67 at 68-82 us mean excess - which is what
a static placement defect looks like rather than transient jitter. The
actual-Claude routing capture should be used to rebalance expert
ownership/placement per layer under the current per-rank HBM constraint, and
these layer indices are where to start.

Reproduce with:

```bash
.venv/bin/python agent_space/experiments/2026-07-29-marlin-smem-monopoly/analyze_step_budget.py \
  /e/project1/profound/alint77/traces/marlin-smem-profile-1092954 \
  /e/project1/profound/alint77/traces/marlin-smem-profile-1092955 \
  --json step-budget.json
```

Custom all-reduce reflects the same skew. It totals 4.7-5.0 ms/rank-step
post-fix, with p50 6-7 us but p90 about 97 us and p99 189-192 us. Treating the
p5 duration as a universal payload floor would overstate wait because payloads
vary, so the next attribution should bucket reductions by position/payload
before assigning an exact recoverable number.

### Communication-kernel dissection

`analyze_comms.py` assigns GPU kernels to an engine step through their CUDA
launch correlation rather than GPU timestamp boundaries. This removes the
boundary errors in the earlier aggregate: every rank and every bounded MTP3
decode step has exactly:

- 166 `cross_device_reduce_1stage<bf16, 4>` custom all-reduces: 157 in the
  target graph, then one preparation and two forward reductions for each of the
  three MTP drafts;
- four `ncclDevKernel_AllGather_RING_LL` calls: one target vocabulary gather
  and one after each draft.

There are no other NCCL, all-to-all, reduce-scatter, peer-copy, or fused
communication kernels in these traces. In particular, CUDA kernels containing
`splitKreduce` are local GEMM reductions, not communication.

The duration of a custom all-reduce is not pure NVLink transfer time. Its
kernel contains both an entry and exit cross-rank barrier, and an early rank
spins inside the kernel until its peers arrive. For each collective ordinal,
the analysis therefore reports:

- **observed residency**: mean kernel duration over the four ranks;
- **fast-rank floor**: the minimum duration over the four ranks;
- **synchronization excess**: observed residency minus that floor.

The fast-rank floor is a conservative proxy for payload plus protocol with the
least arrival wait. It is still not exact wire time: it retains both barrier
instructions and any residual wait on the fastest rank.

One capture-start custom-AR outlier in 1092954 step 0 lasted 16.872 ms and is
excluded from the steady mean. Across the remaining 21 steps:

| communication in one decode step | calls | observed ms | fast-rank floor ms |
| --- | ---: | ---: | ---: |
| target custom all-reduce | 157 | **4.676** | **0.634** |
| MTP1 prep + forward custom AR | 3 | 0.042 | 0.012 |
| MTP2 prep + forward custom AR | 3 | 0.031 | 0.012 |
| MTP3 prep + forward custom AR | 3 | 0.033 | 0.012 |
| target vocabulary NCCL all-gather | 1 | 0.0179 | 0.0156 |
| three draft vocabulary NCCL all-gathers | 3 | 0.0453 | 0.0321 |
| **all communication kernels** | **170** | **4.846** | **0.718** |

Thus **4.128 ms, or 85.2% of communication-kernel residency, is not payload**.
Relative to the 24.307 ms mean engine step, the communication kernels are
resident for 19.94% of the step, but the fast-rank payload/protocol proxy is only
2.95%. Nearly all of the long residency is in the target model, not MTP: the
three MTP rounds contribute 0.153 ms observed and 0.068 ms at the fast-rank
floor.

**4.128 ms is an upper bound on the wait, not a recoverable budget.** The floor
is a per-call minimum over four ranks, and the fastest rank differs from call to
call, so no rank could actually achieve the 0.718 ms total - it would have to win
all 170 races. Under perfect balance every rank would converge to something at or
above the *slowest* rank's true payload time, not the per-call minimum. The
independent estimator to trust is the routed-layer skew, **3.615 ms/step** of
summed `max - mean`, which is the same phenomenon and is what a placement change
can actually move. Treat 3.6 ms as the target and 4.1 ms as the ceiling, and do
not add the two.

The custom-AR duration distribution makes the barrier tail visible. Over
steady rank-kernel instances it is 6.75 us at p50, 96.37 us at p90, and
189.62 us at p99. The aligned fast-rank floor is much tighter: 3.97 us at p50,
4.22 us at p90, and 4.45 us at p99. NCCL all-gathers are 14.51 us at p50 and
23.65 us at p90; their aligned floors are 10.78 and 15.49 us.

All custom-AR launches use grid 6, block 512. The packed BF16 kernel processes
eight values per thread, matching `[4, 6144]`: 49,152 bytes of local input per
rank per call. Its one-stage algorithm reads the same payload from three remote
ranks, so 166 calls deliver **24,477,696 remote bytes (23.34 MiB) per rank per
step**. The trace records the NCCL inputs directly as `[4, 38720]` BF16 for the
target and `[1, 38720]` for each draft. Those gathers receive another
1,626,240 remote bytes (1.55 MiB). Total one-direction remote payload delivered
is therefore **26,103,936 bytes (24.89 MiB) per rank per step**.

At the 0.718 ms fast-rank floor that is about 36.4 GB/s of delivered remote
payload. This low effective rate is expected for 170 small, barrier-heavy
collectives; the limiting factor is operation count and arrival skew, not bulk
NVLink bandwidth. Optimizing only the transfer loop has less than 0.72 ms/step
to work with in this trace. The larger lever is reducing rank skew before the
157 target reductions, then reducing or fusing the collective count.

Per-step totals, including the marked capture-start outlier:

| capture/step | observed ms | fast-rank floor ms | synchronization excess ms |
| --- | ---: | ---: | ---: |
| 1092954/0 (excluded) | 9.349 | 0.851 | 8.498 |
| 1092954/1 | 4.484 | 0.709 | 3.775 |
| 1092954/2 | 4.476 | 0.707 | 3.769 |
| 1092954/3 | 4.741 | 0.709 | 4.032 |
| 1092954/4 | 4.211 | 0.709 | 3.501 |
| 1092954/5 | 4.629 | 0.706 | 3.923 |
| 1092954/6 | 4.405 | 0.708 | 3.697 |
| 1092954/7 | 4.689 | 0.706 | 3.983 |
| 1092954/8 | 5.083 | 0.707 | 4.376 |
| 1092954/9 | 5.537 | 0.710 | 4.826 |
| 1092954/10 | 5.720 | 0.706 | 5.013 |
| 1092955/0 | 4.640 | 0.841 | 3.799 |
| 1092955/1 | 4.422 | 0.713 | 3.709 |
| 1092955/2 | 4.608 | 0.714 | 3.894 |
| 1092955/3 | 4.548 | 0.711 | 3.837 |
| 1092955/4 | 4.696 | 0.715 | 3.980 |
| 1092955/5 | 4.447 | 0.715 | 3.732 |
| 1092955/6 | 4.491 | 0.718 | 3.773 |
| 1092955/7 | 4.806 | 0.719 | 4.087 |
| 1092955/8 | 5.755 | 0.718 | 5.037 |
| 1092955/9 | 5.853 | 0.714 | 5.139 |
| 1092955/10 | 5.526 | 0.716 | 4.810 |

These per-step totals are **not a stationary sample**. Steps 8-10 are
systematically higher than steps 1-7 in both captures (5.5-5.9 ms against
4.2-4.7 ms), because the capture is twelve consecutive steps of a single prompt
and the context is growing under it. Ratios between arms survive this; absolute
per-step figures should not be quoted as steady-state values, and any future
capture that wants stationary absolutes needs either a fixed context length or
many more steps.

### Artifacts

- Primary: `/e/project1/profound/alint77/traces/marlin-smem-profile-1092954/`
- Replicate: `/e/project1/profound/alint77/traces/marlin-smem-profile-1092955/`
- Reproducers: `analyze_step_budget.py` (correlation-attributed budget, tier
  split and EP skew; supersedes the per-family aggregates in
  `analyze_profile_ab.py`), `analyze_comms.py` (collective dissection),
  `analyze_graph_gaps.py` (GPU-empty time by graph position)
- Machine-readable: `step-budget.json`, `comm-kernel-analysis.json`,
  `graph-gaps.json`
- Full per-step collective table: `comm-step-breakdown.csv`
- Superseded but kept for comparison:
  `profile-ab-{1092954,1092955}-analysis.{txt,json}` and
  `analyze_profile_ab.py`. Its step wall, target-graph span, routed-layer span
  and tier split are correlation- or graph-derived and remain correct; its
  `categories` per-family aggregates and `top_kernels` are boundary-attributed
  and run 4-6% low.

The traces are **22 MB per capture** (four gzipped rank files), small enough to
keep indefinitely. They are the only input to both reproducers, so they must
survive for any of the numbers in this section to be re-derivable: do not let a
scratch cleanup take `/e/project1/profound/alint77/traces/`.

Open any rank file under `off/` or `on/` directly in Perfetto. The two jobs ran
on separate nodes but shared the same per-arm compile-cache roots while starting
concurrently, so they are a hardware replication, not a fully independent
compile replication.
