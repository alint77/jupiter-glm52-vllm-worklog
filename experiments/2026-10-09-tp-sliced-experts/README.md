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

## Served integration (`tp-sliced-moe` branch of vLLM)

`--tiered-moe-layout tp_sliced` (vllm 4bf85a198f .. a64fab4d00): TP MoE with EP
off, every GPU a 512 slice of every expert (hot HBM / cold own Grace, one hot
set for all GPUs), the sliced kernel on decode steps with the shared expert
fused (through the runner's MK-internal shared slot; Astra reviews 6 and 7:
`logs/codex-review6.md`, `logs/codex-review7.md`), Marlin tiers otherwise.
Plan: `INTEGRATION_PLAN.md`. Tests: `tests/kernels/moe/test_tiered_decode_sliced.py`
(kernel vs fp32 and Marlin, shared on/off), `tests/model_executor/model_loader/
test_tiered_moe_sliced_conversion.py` (real checkpoint expert: 4 production-
converted slices sum to the whole expert).

Boot fixes found on the way: the frozen worktree needs every untracked build
artifact (flash-attn .so in subpackages); the tiered plan was built inside the
EP weight-filter setup (EP off -> no plan -> full allocation OOM); Marlin's
synchronous finalize never runs the shared expert in the MK slot; no vLLM
config is current at run time.

First served boot (job 2255845, prod serve.sh: DFlash2 k=7, DCP4, 400K,
reserve 1.7): 14748 hot / 4452 cold slices per GPU (the EP profile's hot set),
weights 325 s (every GPU reads every expert), ready in 9 min, GSM8K
**92.0%** on 250 questions (prod ~90-91%).

A/B vs prod EP (frozen worktrees ep-base = c1aec2e22f, tp-sliced-a =
a64fab4d00): `chain_ab.sh` on holds 2255981 / 2255982, the standard arm
(`../2026-10-07-mem-reclaim/arm.sh`); compare with
`KINDS=ep,sl LOGS=logs/serve ../2026-10-07-mem-reclaim/compare_ba.py`.

### Served A/B vs prod EP (holds 2255981 / 2255982, both orders)

`KINDS=ep,sl LOGS=logs/serve ../2026-10-07-mem-reclaim/compare_ba.py`:

| arm | hot | free GiB | GSM8K | acc | tok/step | ms/step | TTFT 20K/14K | 8K/150K | 60K | 388K | peak MiB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|
| ep-2255981-1 | 3671 | 3.42 | 0.915 | 0.366 | 3.56 | 21.13 | 2.841 | 1.565 | 8.36 | OK | 96142 |
| ep-2255982-2 | 3671 | 3.42 | 0.900 | 0.381 | 3.67 | 21.12 | 2.788 | 1.512 | 8.17 | OK | 95882 |
| sl-2255981-2 | 14748 | 3.71 | 0.930 | 0.356 | 3.49 | 19.35 | 3.787 | 1.965 | 11.14 | OK | 96190 |
| sl-2255982-1 | 14748 | 3.71 | 0.900 | 0.347 | 3.43 | 19.29 | 3.721 | 1.958 | 10.97 | OK | 96615 |

(hot: whole experts per GPU for EP, slices per GPU for sliced: same residency.)

- Agentic decode: **sl - ep = -1.841 +- 0.039 ms/step** (69 requests, 2 nodes;
  fit controls tok/step and context). Long context (50K / 130K): **-1.805 +-
  0.067 ms/step**.
- GSM8K 0.907 (EP) vs 0.915 (sliced), 2 x 200 each: unchanged.
- Acceptance lower on the sliced arms (tok/step 3.49 / 3.43 vs 3.56 / 3.67):
  within run-to-run spread so far (2 arms each) but in one direction; the
  sliced numerics differ from EP (fp32 summation order over four partials,
  shared expert fused in fp32 at 1 / rsf), so it needs more arms.
- TTFT +33% (20K: 2.8 -> 3.75 s; 60K: 8.3 -> 11.1 s): prefill runs on Marlin
  over the sliced tiers with no wgmma prefill kernel and no cold prefetch
  (cold slices read over C2C through UVA). Prefill is not the target, but this
  is the largest regression and is fixable (512-wide wgmma prefill; staging).

### Round 2: TTFT fix (vLLM 1830810344: wgmma prefill at 512, cold prefetch staging)

Holds 2258516 (sl first) / 2258574 (ep first), same harness:

| arm | GSM8K | acc | tok/step | ms/step | TTFT 20K/14K | 8K/150K | 60K | 388K |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| ep-2258516-2 | 0.900 | 0.342 | 3.39 | 21.37 | 2.810 | 1.523 | 8.26 | OK |
| ep-2258574-1 | 0.915 | 0.312 | 3.18 | 20.90 | 2.788 | 1.535 | 8.19 | OK |
| sl-2258516-1 | 0.915 | 0.331 | 3.32 | 19.09 | 2.929 | 1.592 | 8.68 | OK |
| sl-2258574-2 | 0.915 | 0.322 | 3.25 | 19.06 | 2.951 | 1.623 | 8.65 | OK |

- Agentic decode **-1.862 +- 0.079 ms/step** (75 requests), long context
  **-1.956 +- 0.052**. TTFT now +4-6% (was +33%); every prefill chunk stages
  all 75 layers' cold slices from the 2 x 440 MiB slots (0 from Grace).
- Acceptance: round 2 sliced 3.32 / 3.25 vs EP 3.39 / 3.18: no consistent
  direction over the four pairs so far.

## Decode step breakdown, sliced vs EP (holds 2258736 / 2258737)

`prof_launch.sh <ep|sl>` -> `prof_arm.sh` (prod serve.sh, c=1, DFlash2 k=7,
torch profiler 2 s windows from `prof_load.py`: one request at 5K and one at
50K context). Traces: `/e/fscratch/profound/naeimitabiei1/sliced-prof/`.
`breakdown2.py` (copy of ../2026-10-09-c8-profile's, layer_kernel counted as
MoE), `kernels.py`, `layer_chain.py` (median per-layer kernel chain),
`ar_skew.py` (per-rank MoE kernel and AR).

| ms / step, rank 0 | EP 5K | SL 5K | EP 50K | SL 50K |
|---|--:|--:|--:|--:|
| period | 20.87 | 19.08 | 22.98 | 20.28 |
| MoE expert kernels (SL: + shared expert) | 7.44 | 6.70 | 7.98 | 7.65 |
| dense GEMMs (decode_gemm) | 3.12 | 3.14 | 3.13 | 3.14 |
| all-reduce + RMSNorm (incl. wait) | 2.31 | 1.26 | 2.94 | 1.24 |
| outside the verify graph (drafter, sampling) | 1.58 | 1.55 | 2.02 | 1.55 |
| DCP collectives | 1.39 | 1.38 | 1.43 | 1.40 |
| attention (FlashMLA + combine) | 1.37 | 1.37 | 1.38 | 1.39 |
| dense GEMMs (cuBLAS; EP: + shared expert) | 1.02 | 1.00 | 1.02 | 1.00 |
| other verify-graph kernels | 0.64 | 0.69 | 0.63 | 0.69 |
| GPU idle | 0.61 | 0.66 | 0.75 | 0.61 |
| MoE router / route / finalize (exclusive) | 0.57 | 0.49 | 0.57 | 0.49 |
| DSA indexer | 0.57 | 0.57 | 0.87 | 0.86 |
| KV write / skip-KV staging | 0.26 | 0.26 | 0.26 | 0.26 |

The sliced win is the MoE kernel (-0.7) and the MoE all-reduce no longer
waiting on the slowest EP rank (-1.05 / -1.7); everything else is unchanged.

**One layer, sliced, 5K (median, us from grouped_topk; 206 us/layer x 75 =
15.5 ms of the 19.1):**
- MoE region 95.3: grouped_topk 4.1 and route_prep overlap the layer kernel's
  start (it starts at 4.7), layer_kernel 85.5, finalize exposed 1.9 after it,
  then a bf16 zero fill (the owned shared output, 1.0), the runner's
  `shared + rsf * routed` (1.2) and 3 x 0.4 us launch gaps.
- MoE AR + residual + RMSNorm 9.3 (the attention-side AR: 6.5).
- Attention block 101: qkv_a 13.0, q_b 7.8, o_proj 16.4 (decode_gemm), two
  nvjet absorbs 3.7 + 3.9, router 3.7; DCP gather_cat 8.5 + lse
  reduce-scatter 6.5; FlashMLA 14.9 + combine (2.8 exposed); AR 6.5; norms,
  rope, 2 x concat_and_cache 9.4; ~0.4 us launch gap before each kernel.

EP's layer is 236 us: MoE region 107 (gemm 59.7 + act + gemm 37.5, shared
expert on cuBLAS on the side stream) and the MoE AR 18 (ranks' MoE kernels end
14-17 us apart at the median).

**MoE kernel durations** (layer_kernel, rank 0): 5K mean 89.2, p50 85.7, p90
106, p99 152 us; 50K mean 101.8, p50 91.7, p90 140, p99 196. Above-p90 calls
are 14% / 17% of MoE time. EP (gemm<0> start to gemm<1> end): 5K p50 98.0, p90
125; 50K p50 105.5, p90 130. The sliced tail at 50K is heavier than EP's.

**Per-rank skew (sliced)**: rank 3's layer_kernel is 2-3.4 us slower at the
median in both windows (87.9 vs 84.5-86.1 at 5K) and its MoE AR is the
shortest (6.6 vs 8.7-9.9): the other ranks wait for it.

### Optimization opportunities (ms / step at c=1, 75 layers)

| # | lever | evidence | upper bound |
|---|---|---|--:|
| 1 | MoE kernel to the memory floor | p50 85.7 us vs floor 61.3 us at 38/4 (bench); consumer math only ~20-25% faster than delivery | ~1.8 |
| 2 | Cold-heavy tail | 50K mean 101.8 vs p50 91.7; 16 cold CTAs at ~11 GB/s each cap C2C at ~180 GB/s, vs 363 GB/s with all CTAs | 0.26 (5K) - 0.75 (50K) |
| 3 | Dense GEMMs over floor | qkv_a 13.0 vs 8.9, q_b 7.8 vs 4.6, o_proj 16.4 vs 13.8 (`../2026-10-07-skinny-gemm-v2`): mostly per-kernel ramp | ~0.74 |
| 4 | Launch gaps in the verify graph | ~0.4 us before each of ~20 kernels per layer (GPU idle 0.61-0.66) | ~0.6 |
| 5 | MoE epilogue | zero fill + runner add + gaps 3.4 us, finalize exposed 1.9: write rsf * routed + shared in-kernel, finalize in the last CTA | ~0.4 |
| 6 | Rank-3 skew into the MoE AR | MoE AR 8.7-9.9 vs 6.5 floor | ~0.19 |
| 7 | MoE head | grouped_topk + route_prep before the kernel starts (4.7 us) | ~0.15 |

### Levers 5 and 7 (user: "5&7"; lever 6 dropped)

**Lever 7, MoE head: no kernel-side headroom.** Moving route_prep into the
layer kernel (`kernels/td_v28.cu`-`td_v30.cu`: fused prep, per-tile counting,
end-of-kernel finalize) was never faster than route_prep + the PDL boundary.
v27's route_prep exits at 3.7 us and its producers issue at 5.0; in-kernel
prep took 4.3 (v29) or 2-3 us (v30, vectorized) after the wait, and M=32
regressed (110/0: 222 vs 217 us; 110/12: 264 vs 250). `td_v31` (v27 + routed
scale + the slot maps read with the ids, `e24.sh`) passes check (worst 0.0034)
and benches equal to v27 (M=8 38/0 82.0 vs 84.0, 38/4 86.2 vs 86.0, 50/4 107
vs 107; M=32 110/0 215 vs 211-217, 110/12 255 vs 254); its route_prep still
exits at 3.8 us (`probe27.py`), so the map prefetch is dropped. The head's
remaining 4.7 us is grouped_topk + route_prep latency the kernel already
overlaps from 4.4 us.

**Lever 5, MoE epilogue (vLLM f51ed6dec9).** An in-kernel finalize cannot
beat the exposed finalize tail (~2.6 us either way in v29/v30), so the
win left is the runner's glue: the zero fill of the owned shared output and
`shared + rsf * routed`. The tiered setup now calls
`MoERunner.release_output_epilogue()`: the runner gives the shared MLP and its
routed scale to the method, keeps no shared expert and a scale of 1 (plain
`moe_forward` op, then the AR). Decode: route_prep multiplies the router weight
by the routed scale and the kernel adds the shared slice at scale 1. Other
steps: tiers' output `*= rsf` then `mlp(x) +` it, the runner's own ops. The
`tiered_fuse_shared` hook on `mk_can_overlap_shared_experts` is gone.
`tests/kernels/moe/test_tiered_decode_sliced.py` (now with routed_scale 2.5):
6 passed. Served check: `chain_arms.sh run slb sla` / `run sla slb` (holds
2258906 / 2258907) and `chain_arms.sh prof slb sla` (2258908); slb runs from
worktree tp-sliced-b with its own compile cache.

**Served result (slb = f51ed6dec9 vs sla = 1830810344; holds 2258906/7/10,
same-node pairs on 2258907 and 2258910).** The first slb arms on 2258906 and
2258908 died at boot: worktree tp-sliced-b lacked the ignored build outputs
(flash-attn extensions, third_party); all ignored files are now synced from
tp-sliced-a. `KINDS=sla,slb LOGS=logs/serve ../2026-10-07-mem-reclaim/compare_ba.py`:

| arm | GSM8K | acc | tok/step | ms/step | TTFT 20K/14K | 8K/150K | 60K | 388K |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| sla-2258906-2 | 0.910 | 0.328 | 3.30 | 19.18 | 2.950 | 1.598 | 8.73 | OK |
| sla-2258907-1 | 0.910 | 0.360 | 3.52 | 19.26 | 2.974 | 1.621 | 8.78 | OK |
| sla-2258910-2 | 0.910 | 0.332 | 3.33 | 19.14 | 2.914 | 1.562 | 8.57 | OK |
| slb-2258907-2 | 0.910 | 0.317 | 3.22 | 18.80 | 2.983 | 1.590 | 8.78 | OK |
| slb-2258910-1 | 0.915 | 0.321 | 3.25 | 18.90 | 2.916 | 1.578 | 8.58 | OK |

- Agentic decode: **slb - sla = -0.235 +- 0.029 ms/step** (85 requests, 3
  nodes; fit controls tok/step and context). Long context: -0.306 +- 0.058.
- GSM8K 0.910 -> 0.913; TTFT unchanged.
- Acceptance 0.317 / 0.321 vs 0.328-0.360: in the low end of the old arms'
  spread; the decode numerics only moved rsf before the kernel's single bf16
  cast. Watch it over more arms.
- Profile (2258911, rank 0, 5K, `layer_chain.py`): the layer chain drops from
  23 to 21 kernels (FillFunctor and `triton_poi_fused_add_mul_0` gone; the MoE
  AR starts 0.1 us after finalize) and the layer from 211.0 to 207.0 us:
  4.0 us x 75 = 0.30 ms/step, as served.
- Astra review 8 (`logs/codex-review8.md`): no definite bug. Fixed: non-decode
  steps now scale and add with one rounding like the runner's fused kernel
  (`torch.add(shared, out, alpha=rsf)`, fp32 opmath on CUDA, checked on GPU;
  vLLM 6dd6db53bd). Checked: the compile cache key hashes
  VLLM_TIERED_SLICED_FUSE_SHARED and the traced moe_runner.py, so no stale
  graph is reused across the change.

### Lever 1 resumed: ring depth and L2 prefetch (v35, v36; 2026-10-10)

Skeleton trace (`e37.sh`, v32 M=8 38/4, CTA trace): compute-only with no
routed math or flush exits at 32.8 us with consumers waiting 22.7 us of it, so
the producer's serial path (claims, record loads, ready acquires, issue) is
~26 us for ~46 units. In the full kernel the producer empty-waits 19.7 us
while consumers wait ~22 us on full stages: both sides idle on each other.

Hypothesis tested: too few bytes in flight per SM (ring 4 x 46 KB). Both
probes say no.

| us | 38/0 | 38/4 | 50/4 | 110/0 | 110/12 |
|---|--:|--:|--:|--:|--:|
| v32 | 84.8 | 92.9 | 115.9 | 216.5 | 254.5 |
| v35 `TD_XROWS=4` (41 KB stages) | 85.7 | 93.6 | 116.1 | 211.8 | 252.6 |
| v35 `TD_XROWS=4 TD_STAGES=5` | 85.5 | 93.7 | 116.6 | 214.7 | 253.4 |
| v35 `TD_XROWS=2 TD_STAGES=5` | 85.2 | 93.8 | 116.4 | 212.8 | 253.6 |
| loads-only v32 / XROWS=4 / +5 stages (38/0) | 78.3 / 78.8 / 78.4 | | | | |

(`e36.sh`; `TD_XROWS` clamps the x rows per stage, timing only.) A 5th stage
changes nothing, loads-only included. v36 `TD_L2PF=P` (L2 tensor prefetch of
unit ci + P within a w13 / shared group, before the empty wait; passes check)
is slower, monotonically in P: 38/0 84.0 -> 89.7 / 93.9 / 103.1 at P = 2 / 4
/ 8 (`e38.sh`). The ~4.5 us load latency is not a ring-depth artefact; more
outstanding requests only add traffic.

### Trace methodology fixed, and where v32's time actually goes (2026-10-10)

The per-unit trace was distorting what it measured, twice:
1. The unit record's "stage full" time was stamped inside `td_record`
   after its `atomicAdd` on the trace counter, so every unit carried one
   global atomic round trip as fake wait (no unit ever waited < 0.25 us).
2. Warp 0 (the warp that paces the ring: it also runs the w13 handoff) did
   that atomic for every unit, ~0.4-0.9 us under HBM load: loads-only showed
   warp 0 "consuming" 33 us per CTA with no math at all.

v35's trace now stamps right after the barrier wait and writes unit records
into slots each CTA reserves once (plain stores). Traced v35 = 84.2 us vs
~85 untraced. The "latency-bound ring" reading above (and the Little's-law
argument behind the 5-stage and L2-prefetch probes) came from the distorted
trace; with the fixed trace both null results are explained.

`trace_bd.py` (every hot CTA's wall time partitioned, waits charged to
latency / ring / producer), `hand_tl.py` (warp 0's work split), `wait_dist.py`
(true landing latency of waited units), `sass_win.py` / `sass_reg.py` (ncu
SASS windows and per-region stall reasons). M=8 38/4, hot CTAs, us per CTA:

| warp 0, us / CTA | full | compute-only | loads-only |
|---|--:|--:|--:|
| kernel (traced, extra stamps) | 90.0 | 69.8 | 78.4 |
| head (route_prep + PDL) + first fill | 7.1 | 4.6 | 8.4 |
| R0 math (31.4 units) | 28.6 (0.93 / unit) | 23.0 (0.74) | 0.5 |
| R1 math (14.7 units) | 12.7 (0.93) | 10.1 (0.70) | 0.1 |
| loop head: stage full -> math start | 13.5 (0.26 R0 / 0.32 R1) | 11.4 | 7.8 |
| R1 flush / R0 flush | 3.6 / 0.7 | 2.9 / 0.7 | ~0 |
| w13 done count (fence + atomic), 2.6 / CTA | 3.3 (1.3-1.6 each) | 1.6 (0.6) | 7.6 (3.1) |
| activation + ready release (0.3 / CTA) | 1.3 (3.6 each) | 0.9 | 2.2 (5.7) |
| shared expert units | ~4.5 | ~3.3 | ~4.9 |
| waits on loads (all kinds) | ~5 | ~2.6 | ~35 |
| end (last unit + imbalance) | ~4 | ~3.3 | ~1.5 |

- **The full kernel is consumer-bound**: warp 0 consumes 90-96% of the time
  from 16 to 64 us; waits on loads total ~5 us. Loads-only, the same
  delivery finishes at 78 us, so memory is not the limit.
- Per R0 unit the warp-0 cycle is 1.38 us (full) / 1.15 (compute-only);
  the floor needs ~1.2 us per unit for everything (61 us / ~50 units).
- The math slows 25% with loads (0.74 -> 0.93 us / unit); the done-count
  fence + atomic doubles (0.6 -> 1.3-1.6 us) and is on warp 0, which the ring
  waits for (warps 1-7 wait ~22 us vs warp 0 ~15 in the CTA trace).
- ncu (v32 full, base clock): the math regions are issue-bound with
  dependency stalls (selected 33%, wait 25%, not_selected 13%, math 11%,
  dispatch 8-10%, short_sb 7%); per HMMA 13.2 instructions: 4.75 LOP3, 4
  HFMA2, 2 FFMA, 1.1 SHF, 0.84 IMAD, 0.5 LDS. The loop head (27 SASS) is 17%
  of samples, 62% long_sb on the full-barrier try_wait (warps 1-7 idling
  behind warp 0). DRAM bytes 221.6 MB = the model's hot + shared bytes.

### v37-v39: Astra consult 9 switches, w2 claim size (2026-10-10)

Astra consult 9 (`logs/astra9-prompt.md`, `logs/codex-review9.md`) on the
consumer-bound v32. `kernels/mk_v37.py` adds four switches on v35:
`TD_MMA_NV` (non-volatile mma asm), `TD_MB2` (two row blocks interleaved:
independent decode/MMA chains), `TD_GLOOP` (an R0 group's chunks consumed in
an inner loop, flush metadata as plain shared loads read only where needed),
`TD_ACQREL` (acq_rel done count instead of `__threadfence` + atomicAdd).
All pass check (worst rel err 0.00344).

Traced (M=8 38/4, warp 0, us / CTA), all four on vs v35: R0 math 28.6 ->
21.6 (0.93 -> 0.70 / unit), R0 loop head 8.8 -> 5.4, R0 consume 47.9 ->
38.1; but waits rose (R0 1.7 -> 3.5, R1 2.5 -> 6.9) and the done count rose
3.6 -> 5.0 with ACQREL (an acq_rel atomic is slower under load: p50 1.38 ->
1.76 us). Kernel 91.9 -> 86.5 traced: the saved consumer time mostly
reappears as w2-phase waits.

- `TD_GR1=2` (two w2 tiles per claim) helps M=16/32 (amortises the
  producer's claim / record / ready round trips) but not M=8 (tail
  imbalance). `td_v38` (`TD_GUIDED=K`, guided self-scheduling of w2 claims,
  may span entries) lost to static GR1=2.
- `td_v39` (`kernels/mk_v39.py`): v37's MMA_NV + MB2 + GLOOP on by default,
  the claim size per call: `g1 = T > 8 ? 2 : 1`. 162 regs, no spills.

| us (e49) | 38/0 | 38/4 | 50/4 | 70/6 | 110/0 | 110/12 |
|---|--:|--:|--:|--:|--:|--:|
| v32 | 84.7 | 91.6 | 112.9 | 151.7 | 219.6 | 255.4 |
| **v39** | 84.5 | 90.6 | 113.9 | **144.9** | **209.9** | **242.6** |
| v39 + ACQREL | 83.8 | 90.9 | 113.9 | 147.0 | 206.8 | 243.7 |

ACQREL dropped (neutral). M=8 (the served DFlash2 case) stays flat.

### v39 producer timeline: at M=8 the w2 phase is producer-bound

`prod_tl.py` reads producer records added to v39 under `TD_UNIT_TRACE`
(170 group start, 171 per unit empty wait + issue, 172 ready spin, 173
claim wait). M=8 38/4 (`logs/e50.out`): from ~56 us the hot CTAs wait
30-40% of the time; trace_bd charges 12.0 us / CTA to "wait latency R1"
(issued before the consumer arrived, landed after). The producer's cycle per
w2 unit (GR1=1 at M=8), mean / p50 us:

| producer, one w2 unit | mean | p50 |
|---|--:|--:|
| group setup + entry record load | 0.90 | 0.90 |
| empty wait | 0.16 | 0.16 |
| descriptors + weight TMAs (incl. a trace atomic, see below) | 0.73 | 0.70 |
| ready spin (taken every unit: each claim is a new entry) | 0.22 | 0.19 |
| x2 row copies | 0.22 | 0.22 |
| claim wait / loop head | 0.15 | 0.13 |
| **sum** | **2.38** | |

The consumer needs ~1.4 us per w2 unit (0.93 math + 0.22 flush + 0.22 loop
head), so the producer cannot keep the ring ahead: every unit is issued
~1.2 us before it is needed while landing takes ~1.5. The record load is a
serial dependent round trip (the TMA coordinates need `local`) on every unit.
The "W issue" 0.73 includes a `td_record(90 + q)` debug record with a global
atomic under `TD_CTA_TRACE` (now gated behind `TD_DEBUG_SUM` in v40), so the
untraced cycle is ~1.65 us, still above the consumer's 1.4.
At M=32 (GR1=2) the producer's issue -> next-unit gap is 1.73 us p50 for
two units, and the consumer is busy 73-93% throughout.

### v40-v43: chasing the producer's per-unit path (M=8 w2 phase)

Every guess about the ~0.9 us "setup + record" step was wrong until it was
stamped finely; the ablations below record why.

| variant (e51-e53, us) | 38/0 | 38/4 | 50/4 | 70/6 | 110/0 | 110/12 |
|---|--:|--:|--:|--:|--:|--:|
| v39 | 84.6 | 91.2 | 113.8 | 144.9 | 207.7 | 242.4 |
| v40 (claim 2 ahead + prefetch next record) | 86.3 | 94.0 | 117.2 | 151.0 | 211.1 | 247.6 |
| v41 no prefetch (len/n_tier in registers) | 84.6 | 91.2 | 113.9 | 144.9 | 205.3 | 241.5 |
| v42 static strided schedule, no stealing (ablation) | 97.3 | 103.9 | 133.5 | 190.1 | 281.7 | 303.5 |
| v42 dynamic, no stealing | 86.2 | 95.5 | 120.4 | 147.5 | 201.3 | 250.3 |

(e51 and e52/e53 ran on different nodes; compare within a run.)

- v40: the claim atomic and the record loads share SASS scoreboard SB5, so
  the prefetched record's first use waits for this group's contended claim
  atomic (record wait 0.32 -> 0.65 us). The claim prefetch of v32+ has never
  overlapped anything.
- v41: `len[q]` / `n_tier[q]` were a 16 B stack frame (two LDLs per group,
  after the ready spin's CCTL.IVALL). Registers now; neutral.
- v42: with no claim atomics at all the setup step still costs 0.79 us; the
  static schedule loses 12-90 us to imbalance (work stealing matters).
- v43 (`setup_tl.py`, stamps that take the loaded registers as asm inputs):
  the setup step is a chain of small latencies, not one round trip:
  index math (`group_at` + a runtime division by TILES1/g1) 0.22, record
  load 0.32, record -> empty wait (STS, loop setup) 0.32 us. The whole
  per-unit producer path is ~2.2 us of such steps (weight TMA issue 0.44,
  ready spin 0.22-0.30, x copies 0.22, loop head 0.16) vs the consumer's
  ~1.4 us per w2 unit. Astra consult 10 (`logs/astra10-prompt.md`) weighs two
  producer warps sharing the ring vs shortening the chain.

### v44: a scheduler warp (Astra consult 10, option C)

Astra (`logs/codex-review10.md`) ranked a scheduler warp over two producer
warps (whose ring-reservation protocol it showed unsound: slot parity
aliasing across generations, and a w13 -> w2 claim/reservation deadlock) and
over shortening the single producer's chain. No PTX control exists over SASS
scoreboard assignment; another warp is the only isolation.

`td_v44` (`kernels/mk_v44.py`): warp 9 (THREADS 320, 161 regs, no spills)
claims, steals, decodes groups (constant divisions only) and loads entry
records; it publishes whole groups through a `TD_SQ`-slot shared-memory FIFO
(mbarriers `sq_full` / `sq_empty`, a slot freed once its group is issued, so
at most TD_SQ claims ahead). The producer only issues. Check passes.

| us (e55) | 38/0 | 38/4 | 50/4 | 70/6 | 110/0 | 110/12 |
|---|--:|--:|--:|--:|--:|--:|
| v41 (no prefetch) | 85.5 | 91.1 | 113.8 | 146.7 | 206.4 | 241.2 |
| **v44** (SQ 2) | **84.4** | **88.4** | **111.5** | **142.9** | **203.5** | **234.9** |
| v44 SQ 3 | 85.8 | 89.1 | 110.4 | 145.7 | 204.5 | 234.8 |

Trace (M=8 38/4, `sched_tl.py`, `logs/e55-tr.out`): the producer's setup per
w2 unit 0.93 -> 0.23 us; R1 load waits 12 -> 7.5 us / CTA (4.9 at SQ 3). The
scheduler claims in 0.29 us p50 and decodes + loads + publishes in 0.90; it
idles ~45 us / CTA waiting for slots. The producer's w2 path is still ~1.9-2.1
us / unit: weight TMA issue 0.48, ready spin 0.50-0.61 (p50 0.26, long tail),
x2 copies 0.33, FIFO wait 0.2-0.28. Next: the scheduler probes readiness
(v45).

### v45 (scheduler ready probe) and v46 (finisher warp)

| us | 38/0 | 38/4 | 50/4 | 70/6 | 110/0 | 110/12 |
|---|--:|--:|--:|--:|--:|--:|
| v44 (e56) | 84.5 | 89.0 | 111.4 | 145.1 | 202.2 | 235.2 |
| v45 (e56) | 84.8 | 88.8 | 110.5 | 144.1 | 200.8 | 237.2 |
| v45 (e57) | 85.0 | 88.8 | 110.7 | 145.3 | 208.3 | 235.0 |
| **v46** (e57) | 85.3 | 88.6 | 110.4 | **142.0** | **204.1** | **229.0** |
| v46 no SREADY (e57) | 85.0 | 88.2 | 110.0 | 143.0 | 203.4 | 228.7 |

- v45 (`TD_SREADY`): the scheduler probes a w2 entry's ready flag once and
  publishes it with the group; the producer skips its spin when set (the
  acquire reaches it through the FIFO mbarrier; copying lanes still fence the
  async proxy). Hit rate 95% (M=8) / 100% (M=32); the producer's spin goes,
  but the scheduler now limits the w2 phase (producer FIFO wait 8.7 us / CTA).
  Neutral.
- v46 (`TD_FIN`): a finisher warp (THREADS 352, 166 regs) runs the w13
  completion handoff (fence + done count; activation + ready release) that
  consumer warp 0 ran inline. Every consumer thread arrives on a 4-slot
  `hq_full` mbarrier (count 256) after its own y13 reductions; warp 0 lane 0
  writes (q, entry, chunks) first. -3 to -6 us at M=16/32; neutral at M=8.
  Traced M=8 (`logs/e57-tr.out`): R0 consume 34.7 -> 29.6 us / CTA, but the
  finisher's activation is 2x slower than warp 0's was (p50 8.7 vs 4.1 us per
  completed entry), so ready flags come later and R1 waits rose 4.6 -> 8.0.
  (Single traced calls differ from the bench's many-routing average; the
  bench is the verdict.)
- v47 (`kernels/mk_v47.py`): `TD_SPIPE` overlaps the scheduler's three
  round trips per group (next claim issued before the current record load;
  ready probe in flight with the record), `TD_FIN_AR` replaces the
  finisher's MEMBAR.SC.GPU fences with fence.acq_rel.gpu.

| us (e58) | 38/0 | 38/4 | 50/4 | 70/6 | 110/0 | 110/12 |
|---|--:|--:|--:|--:|--:|--:|
| v46 | 85.2 | 89.0 | 110.5 | 144.4 | 203.9 | 229.9 |
| v47 | 87.3 | 86.8 | 106.7 | 145.7 | 208.9 | 230.0 |
| v47 FIN_AR | 85.7 | 87.9 | 106.9 | 143.6 | 210.1 | 230.2 |

  Mixed: -2 to -4 us on the M=8 cells with a cold tier, +2 at 38/0 and +5 at
  110/0. Traced (`logs/e58-tr.out`): the scheduler's decode+record+publish is
  0.77 us p50 per w2 group (was 1.09) and its slot wait dominates its time,
  so it no longer limits; the largest M=8 bucket is now R1 ready waits
  (12-17 us / CTA), and the finisher's activation is 8-10 us p50 per finished
  entry at M=8, 11 us at M=32: one route per L2 round trip, serially.
  FIN_AR shortens the done count (p50 2.18 -> 1.76 us) but is bench-neutral.

Astra review 11 (`logs/codex-review11.md`): v44/v45's FIFO handoffs and v46's
cumulative publication (consumer reds -> mbarrier release -> finisher fence +
count RMW -> fence -> activation -> release) are sound, but v46's `hq_full`
reuse breaks the PTX mbarrier phase rule: warps 1-7 could arrive for a slot's
next generation before the finisher had waited on the completed phase (warp
0 alone waited `hq_empty`). Also the producer built a `Group` from slot
fields the K_END entry leaves unwritten.

- v48 (`kernels/mk_v48.py`): the fixes. Every consumer warp's lane 0 waits
  `hq_empty` before reuse, `__syncwarp`, then lane 0 arrives (count
  CONSUMER_WARPS instead of 256); the producer checks `e.kind == K_END`
  first.
- v49 (`kernels/mk_v49.py`): `TD_ACTK=K` (default 4): the finisher activates
  K routes at once (all their y13 loads in flight), and loads the entry
  record before the done count. 168 regs, no spills at K <= 4 (K=8 spills).

| us (e59) | 38/0 | 38/4 | 50/4 | 70/6 | 110/0 | 110/12 |
|---|--:|--:|--:|--:|--:|--:|
| v47 | 86.2 | 87.7 | 106.9 | 144.4 | 209.0 | 230.2 |
| v48 | 85.8 | 87.3 | 107.3 | 146.7 | 209.8 | 231.8 |
| v48 no SPIPE | 85.1 | 89.8 | 111.0 | 143.1 | 205.9 | 231.5 |
| v49 (ACTK 4) | 87.3 | 88.1 | 107.4 | 145.0 | 208.9 | 230.7 |
| v49 ACTK 2 | 86.1 | 88.3 | 107.0 | 145.3 | 210.0 | 230.9 |

v48's fixes are neutral; SPIPE is worth 2.5-4 us on the M=8 cells with a cold
tier (and costs ~1-4 us at 70/6 and 110/0). v49 is neutral and its trace
(`logs/e59-tr.out`) still has 8 us (M=8) / 11 us (M=32) p50 per finished
entry, so serial per-route loads were not the cost.

v50 (`kernels/mk_v50.py`, trace builds only): sub-stamps on the finisher's done
path, `fin_tl.py` (`logs/e60` run, traces `logs/u60-*.pt`). p50 us per finished
entry, M=8 38/4, one route:

| count RT | post-count fence | y13 loads landed | compute + stores | proxy + GPU fences | release | total |
|--:|--:|--:|--:|--:|--:|--:|
| 1.8 | 1.5 | 1.9 | 1.0 | 1.1 | 0.7 | ~10 |

It is a chain of ~6 serialized global round trips/fences, each 1-2 us under
the kernel's memory load, on one warp. Compute + stores grows with the routes:
1.5 us at 1 route, 18.4 at 8 (M=32), i.e. the stores (16 f16 x2 + 32 fp32 y13
zeroing per lane per route) issue slowly from this warp.

- v51 (`kernels/mk_v51.py`): `TD_ZLATE` zeroes y13 after the ready release
  (float4 stores; next call's reds are after its PDL wait), `TD_FIN_AR` on by
  default, variants `TD_ONEREL` (no GPU fence before the release `__syncwarp`;
  lane 0's st.release.gpu is cumulative) and `TD_ATOM_AR` (count as one
  atom.acq_rel.gpu). Astra review 12 (`logs/astra12-prompt.md`) asked about
  these and the done path's structure.
