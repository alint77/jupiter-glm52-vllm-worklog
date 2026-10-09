# TP-sliced experts: every expert 4-way on its intermediate dim (2026-10-09)

Idea (`../2026-10-09-first-principles`): instead of expert parallelism (each
GPU owns whole experts; the slowest GPU's MoE sets the step), GPU r holds rows
[512 r, 512 r + 512) of every expert's gate / up and the matching columns of
down, for hot (HBM) and cold (its own Grace) experts alike. Every GPU computes
its slice of every touched expert for all tokens; the all-reduce that already
follows the MoE sums the partials. Work per GPU is the node mean by
construction, each cold expert is read over all four C2C links at once, and
replicas / balancer / cost table go away.

`tiered_decode.cu`: the production kernel with build-time `TD_INTER` (2048 or
512), per-projection K chunk / ring depth (`TD_CHUNK0/1`, `TD_STAGES0/1`) and a
`TD_NO_FLUSH` timing probe; defaults rebuild today's kernel. Kept here (the
shared tree serves running A/B arms). `bench_slice.py --check`: the four 512
slices of the same checkpoint experts, summed, against the 2048 kernel and
fp32; `--time`: whole experts at per-GPU counts vs slices at node-wide counts
(graph replay of 20 distinct routings, best of 10). `analyze.py`: per-variant
linear fits, replayed over agentic held-out steps (EP = slowest GPU after the
deployed replica balancer, prod profile at 3,670 hot).

## Round 1 (job 2247626)

Check: slices summed vs fp32 2.2e-3 (whole kernel 2.5-2.7e-3), M = 8 and 32.

| MoE ms / step (75 layers) | M=8 | M=32 |
|---|--:|--:|
| EP, whole experts (1024 chunk, 4 stages), slowest GPU | 8.72 | 22.37 |
| EP, mean GPU | 7.49 | 20.28 |
| TP-sliced, 512 chunk x 8 stages | **8.11** | 22.41 |
| TP-sliced, 512 chunk x 4 stages | 11.84 | 34.47 |

Slices pay ~20% more per byte than whole experts (M=8: 44 hot slices 105.9 us
= 9.6 us per expert-equivalent vs 12 whole hot 96.6 us = 8.0), but cold is far
cheaper (M=32: 112 hot + 12 cold slices 297 us; 28 hot + 8 cold whole experts
on one GPU 447 us). Suspects: the down projection (K = 512) flushes its fp32
atomics after every 16 KB unit (4x the atomics per GPU), and the 512 chunk
on gate / up. Round 2: per-projection chunks, a 6-stage down ring, and
`TD_NO_FLUSH` probes.

## Kernel dev, slice regime (2026-10-09)

Dev loop: `kdev.py` (build with ptxas report + SASS; roof; bench; once; check),
`ab.sh` (same-GPU comparison, 4 GPUs, alternating order) + `ab_sum.py`,
`ncu1.sh` + `stall_regions.py`, `tl2.py` (CTA trace), `sass_loops.py`,
`tma_probe.py`. Nodes: the `develbooster` reservation (30 min holds,
`keep_holds.sh`); the main queue was blocked by a large planned job.

**Roofline** (`kdev.py roof`, plain loads; `tma_probe.py`, TMA ring):
HBM 2.42 TB/s at 132 x 64 KB in flight, 3.43 TB/s at 264 blocks; the
kernel's own TMA box patterns stream at 3.5-3.8 TB/s with 3-4 stages
(w13 box 3.76, contiguous 3.91), so layout and boxes are not the limit. C2C
~11 GB/s per CTA with plain loads, 419 GB/s at >= 48 CTAs.

| version | change |
|---|---|
| v1 | one persistent kernel: w13 -> per-entry activation (last contributor) -> w2 |
| v2 | + shared expert (bf16 TP slice) as units: swizzled TMA, ldmatrix, bf16 mma |
| v3 | Marlin-style INT4 decode, phase-specialised fully unrolled consumer (58 -> 26 instr per group step) |
| v4 | dynamic per-tier work queues (S0, R0, S1, R1 groups), stealing across tiers |
| v5 | route_prep fused in (lists by CTA 0, rows by CTAs 1..T, shared w13 from raw rows): slower, dropped |
| v6 | v4 with the ring addressed off `smem` (v2-v5's uintptr alignment had turned every smem load into a generic LD) |
| v8 | v6 + per-stage flush records in smem (rows, folded scales) written by the producer: no global loads in flushes |

Same GPU, us per call (shared expert included; v0 = the shipped kernel with
INTER 512 + the shared expert as cuBLAS on a side stream, today's shape):

| M=8 cell (hot/cold slices) | 30/2 | 38/0 | 38/4 | 46/6 |
|---|--:|--:|--:|--:|
| v0 + side shared | 93.9 | 102.7 | 115.5 | 135.9 |
| v6 | 90.8 | 106.4 | 111.8 | 133.5 |
| **v8** | **81.5** | **96.5** | **103.6** | **126.3** |
| v8 roofline share | 0.61 | 0.64 | 0.59 | 0.60 |

| M=32 cell | 96/8 | 110/0 | 110/12 | 124/16 |
|---|--:|--:|--:|--:|
| v0 + side shared | 274.7 | 266.8 | 323.6 | 368.2 |
| **v8** | **251.4** | **262.9** | **292.5** | **338.6** |

Cold-preferring CTAs: 16 best (8 collapses with cold slices, 24 / 32 slightly
worse).

### v9-v12 (2026-10-09, same-GPU A/Bs on develbooster)

| version | change | M=8 38/4 | M=32 110/12 |
|---|---|--:|--:|
| v8 | (previous best) | 103.6 | 292.5 |
| v9 | 16 consumer warps + parallel activation | slower (+3%); 8-warp build ~ v8 | |
| v10 | per-warp smem offsets hoisted, 32-bit ld.shared, incremental stage / phase | 103.3 | 288.8 |
| v11 | each warp all 4 row blocks of a 64-row tile over a K slice: one conflict-free LDS.128 per k16 row (4.3M bank conflicts before), activation fragment feeds 4 MMAs | 99.6 | 288.1 |
| **v12** | producer: next group claimed while this one issues, entry record read once per group | **98.2** | **279.5** |

Findings: `TD_COMPUTE_ONLY` (producer signals stages without loads): v8 69.8 /
190.5 us at 38/0 and 110/0 vs 95.3 / 260.3 with loads, i.e. consumer
instruction work alone ~ the memory floor (the decode/MMA blocks are 75% of
instructions at ~30 per group step, near Marlin's dequant minimum; the rest
was bookkeeping); 16 warps are slower even compute-only. ncu (compute-only):
issue 47%, ALU 34%, FMA 28%, LSU 22%, tensor 16%: latency, no pipe saturated.
CTA trace: consumer warp 0 waits on data 14% of the call, the producer is
blocked on a full ring ~20%: the producer's own serial latencies (claim
atomics, record loads) bound the pipeline, hence v12.

Open: v12 with w2 groups of > 1 unit (`TD_GR1`) gives garbage in the first
64-row half of one w2 tile for a few tokens in ~10% of M=32 calls with cold
experts, only when hot CTAs steal cold w2 work (`TD_NO_STEAL_COLD` passes);
the smem weights checksum correct in an instrumented build. Default GR1 = 1
(no race, same speed). `kdev.py check --reps N` is now GPU-side (24 s for 5
reps of 16 cases).

### v13-v15

| version | change | M=8 38/4 | M=32 110/12 |
|---|---|--:|--:|
| v13 | v12 device code, torch-free C entry (`td_forward`, ctypes): builds 97 s -> ~8 s each and in parallel (`kdev.build_many`); 97 s was 95% torch headers (gcc 35 s, cicc 27, cudafe++ 20), ptxas 0.8 s | 98.2 | 278.1 |
| v14 | two 8-warp consumer groups on alternating stages | 101.4 | 295.6 |
| **v15** | w13 units in w2's shape (128 rows x 512 of K): one consumer path, 45 KB stages | **93.8** | **271.9** |

v13 compute-only 67.8 us vs 91.6 with loads at 38/0; 28.6 M warp instructions
(67% in the MMA blocks), issue 43% busy: stall-bound with 2 consumer warps per
scheduler. v14 (4 per scheduler) cuts compute-only to 63.5 but loses overall
(2 stages per group, 96 registers with spills). v15 at 5 stages is slower
than at 4 (default 4). Today's kernel + side-stream shared at the same cells:
115.5 / 323.6 us, so v15 is -19% / -16%.

Open: v14 with one consumer group (`TD_CGROUPS=1`, otherwise ~v13) fails the
check at M=32 hot-only in 6 / 320 runs (rel err 0.007-0.008) while v13 / v14
(2 groups) / v15 pass 320 / 320: likely the same latent race as v12's GR1 > 1,
exposed by timing. Not root-caused yet.

### v16-v17: where the consumer time goes

`kdev.py check` now compares each slice's output with its own fp32 reference
(threshold 5e-3): the summed-slices check flagged bf16 rounding under partial
cancellation (~6-7e-3) as failures. v15 passes 10 reps x 16 cases.

v16 (finalize fused into the layer kernel, the last CTAs to finish): no gain,
dropped.

CTA trace, v15 at M=8 38/4 (`tr.sh`): with loads, hot consumer warps wait on
full stages 9.7 of 85 us; compute-only they wait 3.7 of 67 us, so consumer
work alone (~63 us) is above the memory floor (~61 us). The consumer is the
bound.

`probe_consume.cu` (`pc.sh`): v15's routed consume block alone on fake stage
data, 132 CTAs: 0.72 us per 128 x 512 unit at 8 warps (0.77 with a CTA
barrier per unit), i.e. ~6.3 TB/s of weights, 1.8x the HBM rate; 16 warps
are no faster; non-volatile asm and hoisted loads change nothing. Inside v15
a unit costs ~1.3 us. ncu SASS runs by execution count (`sass_runs.py`),
v15 compute-only: MMA blocks 48% of samples (= the probe's rate), unit loop
head 14% (mbarrier wait, `desc` re-addressed through the generic window each
unit), w2 per-unit flush 9% (16 branchy atomics, a BSSY/BRA each), w13
last-chunk flush 4% (per-thread `fence.sc.gpu`), kernel start 4%.

| version | change | M=8 38/0 | 38/4 | M=32 110/12 | compute-only 38/4 | 110/12 |
|---|---|--:|--:|--:|--:|--:|
| v15 | | 87.0 | 94.4 | 265.9 | 71.0 | 190.6 |
| **v17** | unit records via `ld.shared` off precomputed addresses, predicated `red.global.add` flushes (no branches), `fence.acq_rel` before the done count | **84.0** | **90.5** | **250.1** | 65.0 | 171.1 |

### Reviews and v18-v21 (2026-10-09 afternoon)

Two outside reviews of the design: Codex gpt-6-astra (`logs/codex-review.md`,
prompt = `REVIEW_BRIEF.md` + code) and a fresh-context subagent (`AUDIT.md`).
Both independently found: v18 read `sd_row[s]` in the flush after the warp
released the stage (wrong destination rows, rel err 0.61); v19's deferred
w13 count / activation waits behind the next full stage while this CTA's
producer can spin on that entry's ready flag (deadlock; both v19 builds hung
the check). Byte accounting corrected: 5.308 MB per routed entry with scales,
18.87 MB shared; floors 61.3 us (38/4 HBM) and 167.4 us (110/12 HBM).

Sweeps on v17 (no code change):
- `TD_COLD_CTAS` 12/16/20/24/32: 16 stays best or tied at M=32 (110/12:
  267.0 / 261.1 / 266.7 / 267.5 / 276.7) and M=8 (38/4: 92.8 / 92.3 / 94.0 /
  94.8 / 97.6). More dedicated cold CTAs do not help.
- `TD_GR1` 1/2/4/6 on cold-free cells: M=8 flat (38/0 84.2 / 84.2 / 85.7 /
  86.6); M=32 GR1=4 -4% (110/0 226.8 / 222.3 / 217.3 / 220.9). The w2
  claim/record chain matters at M=32 only. (The audit's "R1 phase starves"
  came from a traced build whose per-unit record adds an atomic on that path.)
- `TD_NO_FLUSH_FENCE` (no per-thread fence before the done13 count; thread 0
  still fences after the CTA barrier): passes 60 reps.

| version | change | 38/0 | 38/4 | 110/0 | 110/12 |
|---|---|--:|--:|--:|--:|
| v17 | | 84.9 | 93.7 | 232.8 | 261.1 |
| v21 | v18's smem-staged vector-red flush, destination rows snapshotted before the stage release | 83.6 | 93.2 | 230.6 | 253.7 |
| v20 | v21 + w2 producer chain: one unconditional record read, xs13 in producer lanes, x2 rows carry their scale (consumer applies wt * xs2), w2 weight TMA before the ready spin | 83.9 | 92.0 | 228.1 | 255.0 |
| **v20 no per-thread fence** | | **83.3** | **90.6** | **226.6** | **245.3** |

(same node, ab.sh; v17 on this node is ~3 us slower than in the v17 table.)
All pass `check` 20 reps.

### v22-v23 (Astra reviews 2 and 3: `logs/codex-review2.md`, `logs/codex-review3.md`)

| us, same node, all TD_NO_FLUSH_FENCE | 38/0 | 38/4 | 110/0 | 110/12 |
|---|--:|--:|--:|--:|
| **v20** | 82.3 | **88.3** | 221.8 | **238.6** |
| v22: the 4 K-quarter warps of a half sum partials in smem (named barriers 2/3), one red.v4 per float4 (4x fewer atomics; user approved the fp32 order change) | 81.5 | 88.1 | 218.6 | 240.4 |
| v23: v22 + shared w13 statically split over CTAs, weight TMAs before griddepcontrol.wait | 83.1 | 90.3 | 223.5 | 245.2 |

v22 = noise, v23 slower: dropped (Astra: both correct; keep v20's flush).
`TD_NO_FLUSH_FENCE` on v20: 100 / 100 check reps pass; both reviews accept the
CTA barrier + thread-0 fence as the ordering. Default from here.

Cold transfer ceiling of the real kernel (cold-only cells, all 132 CTAs on
the cold tier, v20): M=8 0/8 117.0 us = 363 GB/s, 0/12 166.7 us = 382 GB/s
(0.87-0.91 of 419). Compute-only on the same cells: 28-29 us. So the TMA path
can fill C2C; in mixed cells it doesn't because cold CTAs come out of the hot
consumer capacity, which is itself at its limit (cold-CTA sweep). (M=32
cold-only cells put > 8 tokens on an expert: entries re-read weights, not
comparable.)

### Where the time goes (v24 ablations) and v25-v26

v24 = v20 with `TD_NO_FLUSH_FENCE` as the default plus timing-only switches
(`TD_ABL_NOMMA` skips the routed math, `TD_ABL_NOFLUSH` the flush; a no-flush
cell alone is invalid: ptxas dead-codes the MMA block once acc is unused).

| us, --shared 1 | 38/4 | 110/12 |
|---|--:|--:|
| full | 89.4 | 243.7 |
| no routed math | 78.2 | 206.8 |
| loads only (no math, no flush) | 78.1 | 203.6 |
| compute-only full | 61.9 | 162.1 |
| compute-only, no math | 33.2 | 73.9 |

Loads alone are ~10 / ~30 us over (HBM floor + route_prep/finalize ~6.3 us):
~85% of floor throughput. Math adds ~11 / ~40 us on top: the consumer is only
~20-25% faster than delivery, so overlap is poor.

Loads-only sweep (no math/flush): 3 stages, 32 cold CTAs, no claim prefetch
all worse; GR1=4 -9 us at 110/12. Loads-only trace 38/0: routed w13 groups
(12 units on one SM, ~20 us) finish up to 67.8 us; producers spin on ready up
to 22 us.

| version | change | 38/0 | 38/4 | 110/0 | 110/12 |
|---|---|--:|--:|--:|--:|
| v25 | `TD_R0S`: a w13 tile's 12 chunks as R0S adjacent groups; R0S 1/2/3/4 (38/4: 91.6 / 94.4 / 96.3 / 98.4; 110/12: 245.3 / 260.3 / 268.6 / 275.2) -> default 1 | | | | |
| v25 GR1 1/2/3/4 | 110/12: 245.2 / 242.4 / 247.6 / 241.7; 38/4: 92.1 / 91.4 / 90.9 / 93.2 (~1%) | | | | |
| v26 | w13 completion handed to warp 0: warps 1..7 `bar.arrive` after their reds and go on; warp 0 syncs, counts, activates all routes, publishes ready | 84.4 | 91.7 | 229.5 | 251.8 |
| (v25 same node) | | 84.1 | 93.9 | 227.8 | 260.6 |

GR1 > 1 passes 120 / 120 check reps on v25: the v12-era stealing race no longer
reproduces (Astra review 4 found no GR1 race in the current code).

Ready-aware w2 dispatch (Astra's top proposal) sized first with
`TD_ABL_NOREADY` (w2 never waits for its entry; results invalid): full kernel
110/0 221.6 -> 218.3, 110/12 234.4 -> 235.1, 38/0 83.9 -> 82.2, 38/4 87.6 ->
87.3. Only loads-only gains (-6 to -8 us at M=32). Not worth the redesign.

## Against prod EP (the goal)

`logs/ep-vs-slice/` (one node, job 2251858): prod whole-expert kernel
(`bench_slice.py --time --variant full-1024x4`) at round 1's per-GPU cells, v25
at the slice cells (with and without the fused shared expert), `analyze.py`
replay of 2,000 held-out agentic steps at the prod profile (3,670 hot,
replicas + deployed balancer; EP = slowest GPU per layer):

| MoE ms / step, 75 layers | M=8 | M=32 |
|---|--:|--:|
| prod EP, slowest GPU | 8.74 | 22.00 |
| prod EP, mean GPU | 7.51 | 20.13 |
| sliced v25 (no shared, like EP's side-stream shared) | **6.80 (-22%)** | **19.99 (-9%)** |
| sliced v25 + shared expert fused | 7.05 (-19%) | 20.53 (-7%) |

Fits (us per layer): EP M=32 53.3 + 4.87 hot + 26.29 cold (per GPU); sliced
M=32 13.6 + 1.71 hot + 5.98 cold (node-wide slice counts, shared fused). Cold is
where slicing wins (4 C2C links per expert, no slowest-GPU); hot slices cost
6.8 us per expert-equivalent vs EP's 4.9.

### v27 and replay 2 (Astra review 5: `logs/codex-review5.md`)

v27 = v26 + `__syncwarp()` after lane 0's count acquire in the warp-0 handoff
(Astra: `__shfl_sync` passes the decision, not the acquire). Passes 100 reps
(v27 and GR1=2). Replay vs a fresh same-node EP (`logs/ep-vs-slice-v27/`; the
v27 GR1=1 M=8 bench was lost to a concurrent-build race):

| MoE ms / step | M=8 | M=32 |
|---|--:|--:|
| prod EP slowest / mean | 8.74 / 7.52 | 21.95 / 20.06 |
| v25 + shared (previous node) | 7.05 | 20.53 |
| v27 + shared | | 20.64 |
| v27 GR1=2 + shared | 7.54 | 20.36 |

v26/v27 and GR1=2 don't move the replay beyond noise; the per-variant linear fits
have 30-65 us max residuals, so ~2-3% differences are not resolvable this way.
Sliced leads EP by ~14-19% at M=8 and ~6-7% at M=32, and at M=32 that lead
is EP's slowest-vs-mean imbalance; sliced is ~0.3-0.5 ms/step slower than EP's
perfectly balanced mean.
