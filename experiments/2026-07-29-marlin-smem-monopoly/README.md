# The Marlin shared-memory monopoly: why hot and cold never overlapped

Every hot/cold overlap experiment in this project has come back null or
negative: the `blocks_per_sm` sweeps (0.0-0.4%), the SM-budget split
(0.0-0.4%), the hot-first reorder (-0.94%), green contexts (FAIL-A, +9-40%
worse), and Grace->HBM staging (+9.5% to +44% worse at q4). This experiment
finds the single mechanism that explains all of them, and removes it.

**Result, measured on Booster under CUDA-graph replay: the two-tier routed-MoE
union drops by a median of 34% across 15 realistic activated-expert cells (best
40.1%, worst 14.6%, every cell positive), and 31-45% at the production c1/q4
shape. The fixed union lands on `max(hot, cold)` - the overlap becomes free.**

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

Jobs 1088069 (`off`) and 1088070 (`on`), 24-prompt suite, one excluded warmup
and two measured repetitions each. Both arms reproduced the semantic gate with
byte-identical output.

| metric | off (control) | on (tight smem) | delta |
| --- | ---: | ---: | ---: |
| output tok/s | 96.53 [96.37, 96.70] | 105.90 [104.85, 106.96] | **+9.71%** |
| TPOT (ms) | 9.304 [9.287, 9.322] | 8.427 [8.357, 8.496] | **-9.43%** |
| ITL (ms) | 26.746 [26.696, 26.797] | 24.658 [24.614, 24.702] | -7.81% |
| TTFT (ms) | 279.3 [270.3, 288.3] | 268.6 [262.3, 274.9] | within spread, not a result |

**Acceptance is not matched between the arms**, so the +9.71% is not all kernel
speed. The `on` arm accepts 84.31/65.11/46.49 per draft position against
82.81/62.61/43.45, i.e. 2.959 versus 2.889 tokens per target step. Dividing it
out:

| | off | on |
| --- | ---: | ---: |
| tokens per target step | 2.889 | 2.959 (+2.4%) |
| **step time = TPOT x tokens/step** | **26.88 ms** | **24.94 ms (-7.2%)** |

So the defensible kernel claim is **about -7% step time**; the remaining ~2.4%
is an acceptance tailwind caused by the reduction-order change shifting which
drafts are accepted, which is not something to bank - it could fall the other
way on a different prompt set.

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
