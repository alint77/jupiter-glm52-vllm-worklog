# Green Context Marlin — Track A feasibility (§4.1-4.2)

Implements the cheap-first gate of
[`../../gh200_low_sm_marlin_green_context_plan.txt`](../../gh200_low_sm_marlin_green_context_plan.txt):
prove the CUDA green-context mechanism partitions SMs and allows genuine
concurrent dispatch on GH200 + CUDA 13, **before** any Marlin or vLLM work.

## Why this slice first

Track A (green contexts with the *existing* kernels) is the plan's preferred
first track. Its central assumption — that hard spatial SM isolation can make
hot and cold Marlin run near their isolated rates — rests on green contexts
actually working on this stack. The runtime-API wrappers the plan named
(`cudaGreenCtxCreate`, `cudaExecutionCtxStreamCreate`) **do not exist in CUDA
13.0.48** (`CUDA_VERSION 13000`); the driver-API surface does
(`cuGreenCtxCreate`, `cuDeviceGetDevResource`, `cuDevSmResourceSplitByCount`,
`cuDevResourceGenerateDesc`, `cuGreenCtxStreamCreate`, `cuGreenCtxGetId`,
`cuStreamGetGreenCtx` — all in `cuda.h`/`cudaTypedefs.h`). So Track A must use
the driver API, and this probe validates that path end-to-end.

The binary question this probe answers:

> Can two disjoint green contexts launch kernels concurrently such that the
> concurrent union is materially shorter than the serial sum (i.e. true
> spatial overlap, not time-sliced contention)?

If no — green contexts don't partition cleanly, or `<<<>>>` on a green-context
stream isn't supported on this driver, or the union ≈ serial sum — Track A is
dead before any Marlin integration, and we go straight to Track B (low-SM cold
kernel) or stop.

## What runs

`green_context_probe.cu` (standalone, no vLLM/PyTorch):

1. `cuInit` + `cudaFree(0)` (primary context active).
2. `cuDeviceGetDevResource(SM)` → print `smCount`, `minSmPartitionSize`,
   `smCoscheduledAlignment`.
3. `cuDevSmResourceSplitByCount(1 group, minCount=16, &remainder)` → disjoint
   cold (≥16 SMs) + hot remainder; print both, verify sum ≤ device.
4. `cuDevResourceGenerateDesc` + `cuGreenCtxCreate(CU_GREEN_CTX_DEFAULT_STREAM)`
   for both; `cuGreenCtxGetId` → distinct IDs.
5. `cuGreenCtxStreamCreate` per context; `cuStreamGetGreenCtx` verifies
   stream→ctx mapping; `cuGreenCtxGetDevResource` confirms provisioned
   `smCount` per context (disjoint proof).
6. `cudaHostAlloc` pinned buffer + `cudaPointerGetAttributes` (UVA-readable
   check — needed later for Grace-pinned UVA).
7. Busy kernel (`cold_busy`/`hot_busy`, distinct names for nsys attribution):
   solo on each stream, then concurrent. Compare union vs serial sum.
8. Outputs non-zero (correctness sanity).

`CUDA_DEVICE_MAX_CONNECTIONS=8` (plan §4.1 — without ≥2 HW connections two
green contexts serialize on one).

## Reproduce

```bash
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
bash agent_space/experiments/2026-07-27-green-context-marlin/run-probe.sh [iters]
# default iters=20,000,000 (~ms-scale busy loop)
```

Runs on the login-node GH200 (no Slurm). For nsys/ncu on a Booster node:

```bash
sbatch agent_space/experiments/2026-07-27-green-context-marlin/job.sh
```

## Pass/fail (plan §4.5 / §10)

- **PASS-A-mechanism**: distinct greenCtxIds; provisioned SM counts disjoint and
  sum ≤ device; `<<<>>>` launches succeed on green-context streams; concurrent
  union < serial sum (overlap speedup clearly > 1.0, ideally approaching the
  ratio of the two partitions' work). → proceed to Marlin integration (§4.3-4.5).
- **PARTIAL**: launches work and SMs partition, but union ≈ serial (no real
  overlap) → diagnose via nsys greenCtxId attribution + ncu SM-issue stalls
  before declaring Track A dead; likely a shared-workqueue or
  single-HW-connection artifact.
- **FAIL**: `<<<>>>` on green-context streams errors, or split doesn't produce
  disjoint partitions, or union == serial under confirmed disjoint contexts.
  → Track A cannot proceed with the driver API as written; either switch to
  per-thread `cuCtxFromGreenCtx`+`cuCtxSetCurrent` (2 host threads) or abandon
  Track A for Track B.

## nsys verification (plan §4.2)

```bash
nsys profile --stats=true -o nsys/probe ./green_context_probe
```
Confirm: hot/cold kernels carry **different greenCtxId**s, their time ranges
**overlap**, and `gpu-kernsum`/SM activity shows both ran. (Node-level
`--cuda-graph-trace=node` is for the later Marlin-under-graph slice, §4.4.)

## Status — PASS-A-mechanism (2026-07-27)

First run on the login-node GH200 (CUDA 13.0.48, sm_90a). **The green-context
mechanism works and gives true spatial concurrency.**

| | cold | hot | total |
|---|---:|---:|---:|
| provisioned SMs | 16 | 116 | 132 (disjoint) |
| green context ID | 2 | 3 | distinct |
| solo (iters=5M) | 29.790 ms | 29.702 ms | — |
| concurrent (each) | 29.693 ms | 29.679 ms | union 29.693 ms |

- Serial sum 59.492 ms → concurrent union 29.693 ms → **2.00× overlap**.
- nsys `cuda_gpu_kern_sum`: each kernel ran twice (solo + concurrent) at
  **identical ~29.67 ms/call** — neither kernel slowed when the other ran.
  That is the disjoint-SM proof: cold (16 SMs) and hot (~112) execute in
  parallel on non-overlapping partitions, not time-sliced.

Findings that gate the implementation:

1. **Driver API only.** The runtime wrappers the plan names
   (`cudaGreenCtxCreate`, `cudaExecutionCtxStreamCreate`,
   `cudaDevWorkqueueConfigScopeGreenCtxBalanced`) are absent in CUDA 13.0.48.
   Track A must use `cuGreenCtxCreate` / `cuDeviceGetDevResource` /
   `cuDevSmResourceSplitByCount` / `cuDevResourceGenerateDesc` /
   `cuGreenCtxStreamCreate` (all present, `cuda.h`/`cudaTypedefs.h`).
2. **`CU_STREAM_NON_BLOCKING` is mandatory** for `cuGreenCtxStreamCreate`
   (flags=0 returns `CUDA_ERROR_INVALID_VALUE`). Stream priority range on
   GH200 is `[-5, 0]`; using `greatest=-5` works.
3. **`<<<>>>` launches on a `cuGreenCtxStreamCreate` stream succeed** on this
   driver (cast `CUstream`→`cudaStream_t`). No per-thread `cuCtxFromGreenCtx`
   needed for the trivial-kernel path.
4. **Pinned host (UVA) buffers are device-accessible** from green-context
   streams — required for the later Grace-pinned-UVA cold tier.

**Gate met: PASS-A-mechanism.** Track A proceeds to Marlin integration.

## What this probe does NOT answer (next slice)

The trivial busy-loop kernels are purely SM-issue-bound and touch no memory
bandwidth. The real Track A question (and critique #1 of the plan) is whether
**hot Marlin reading HBM** and **cold Marlin reading Grace over C2C** — which
contend for the *shared* HBM port, L2, and power/clock domain — retain this
clean concurrency under disjoint green contexts, or whether the Phase 23
zero-sum dilation reappears because the bottleneck was never SM scheduling but
shared bandwidth. That needs the Marlin kernels + pinned-Grace buffers on a
Booster node (§4.5 matrix). The login GH200 obviously runs the mechanism, but
the production-shape HBM/C2C contention test belongs on Booster with the
tiered NUMA pairing.

nsys report: `nsys/probe-16-116.nsys-rep`. Raw run log in `logs/`.

## Verdict — §4.5 Track A: **FAIL-A. Green contexts are worse than the existing production path (2026-07-28)**

`marlin_green_probe.py` extends the proven Phase-21 Marlin-MoE harness
(`benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py`) with the driver-API
green-context streams validated above. Both tiers run on their **own** green
streams and are launched **independently** (no cross-stream event waits — see
"two implementation bugs" below), timed with per-stream CUDA events plus an
anchor-event union. Grace locality is auto-detected per node via the production
`get_device_numa_node` (nvml) path — never hardcode it; the GPU0↔Grace mapping
is node-specific (node 0 on jpbo-022/025/044, node 2 on jpbo-011).

Representative w13, M=16, 19 hot / 3 cold experts, cold_share=0.13, Grace-cold on
the GPU0-paired node (100% pages local), iters=30, Booster:

| cold SM (hot SM) | hot solo | green hot ov | **green union** | green hot intf | **prod union** | prod hot intf | green union vs prod |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 (124) | 134.0 | 196.6 | 388.5 | +46.7% | 278.4 | +3.2% | **+39% worse** |
| 16 (116) | 131.7 | 191.9 | 307.4 | +45.7% | 278.6 | +6.4% | **+10% worse** |
| 24 (108) | 131.4 | 188.0 | 300.5 | +43.1% | 276.9 | +5.3% | **+9% worse** |
| 32 (100) | 132.8 | 192.6 | 304.1 | +45.1% | 278.4 | +5.8% | **+9% worse** |

(`prod` = same-job production control, plan §4.5 F: hot on main stream, cold on
aux stream with the apply_tiered fork/join. Its union is ~278 µs on every split
because it does not use green contexts — the split is irrelevant to it.)

**Green contexts lose to production on every split and every axis**: ~9–40%
higher union and ~8× higher hot interference. Overlap speedup vs serial is
0.82–1.11× (≥ serial time on 3 of 4 splits — concurrency with green contexts is
slower than just running the tiers back-to-back).

### Why — the mechanism (this is the valuable part)

The hot dilation is **memory-subsystem contention during forced co-residency, not
SM scheduling and not clocks/power.** Evidence, all from this same job:

1. **Split-independent.** Hot interference is +43–47% whether cold has 8, 16, 24,
   or 32 SMs. If it were SM-scheduling contention, shrinking cold's SM share
   would relieve hot. It does not.
2. **Not SM confinement.** Hot *solo* on the 116-SM green context is 131.7 µs;
   hot *concurrent on the same 116 SMs* is 191.9 µs. The +45.7% is purely from
   cold co-running on the *other* 16 SMs.
3. **Source-independent.** HBM-cold (both tiers in HBM) dilates hot +42%;
   Grace-cold (cold over C2C) dilates +46%. Moving cold's weights to Grace/C2C
   does **not** relieve hot — refuting the tiering premise that the C2C path
   frees hot's HBM. The shared resource is downstream of the weight source:
   the L2 cache and the memory fabric that both kernels (and cold's HBM-resident
   output/activations) traverse.
4. **Not clocks/power.** Concurrent SM clock is 1920 MHz vs 1875 MHz hot-solo
   (**+2.4%**, i.e. no throttle), board power 361 W (far under the GH200 cap).
   The plan §3 rule-7 power-wall hypothesis is refuted.
5. **Production already avoids it.** The production fork/join co-runs the tiers
   only ~22% of the time (coresident=0.22–0.23), so hot stays near solo
   (+3–6%). Green contexts *force* co-residency (coresident up to 0.93), which
   *creates* the memory contention. Green contexts move production in the wrong
   direction: more co-residency = more dilation.

### Consequences for Tracks B and C

- **Track B (low-SM cold kernel) is undercut by the same mechanism.** Its goal is
  to shrink cold's SM footprint to free SMs for hot. But hot is HBM-bandwidth-bound
  (solo ~132 µs on 100–124 SMs alike — it does not need the freed SMs), and the
  dilation tracks cold's *co-residency*, not its *SM count*. A smaller cold grid
  moves the same weight bytes through the same shared L2/fabric, so it cannot
  recover hot's isolated rate either.
- **Track C (combine)** inherits both failures.
- The probe also measured that the **stock** cold kernel, confined to few SMs,
  serializes its large persistent grid (cold solo over C2C: 297.6 µs @8 SM →
  117.3 µs @32 SM). A purpose-built low-grid cold kernel (Track B §5.2) would fix
  *that*, but per the above it would not reduce hot's co-run dilation.

### Disposition

**Do not integrate green contexts (Track A) into production. Do not pursue
Track B/C for the goal of relieving hot's co-run dilation** — the bottleneck is
shared memory-bandwidth/L2 during co-residency, which no SM-partitioning or
SM-footprint scheme adds to. The effective levers are the ones that reduce total
co-resident memory traffic (routing/placement: fewer/smaller cold experts per
layer, raising hot-cache hit so cold is invoked less), not kernel/SM scheduling.

### Two implementation bugs worth recording (cost real debugging time)

1. **Shared Marlin workspace = device-side deadlock.** The kernel's `int* locks`
   workspace is a spin-barrier for the cross-CTA reduction (`barrier_acquire`
   spins `while (state != count)` in `marlin_template.h`). Two concurrent kernels
   sharing one workspace corrupt each other's barrier counts → infinite spin →
   100% GPU, host `synchronize()` never returns. This reproduced on login and
   Booster and looked exactly like a "green-context deadlock." Fix: **one
   workspace per tier** (production already does this — plan §1 "separate
   workspace views"). The probe originally shared one.
2. **Cross-green-context `wait_stream` is fragile.** The first concurrent timer
   used `cold.wait_stream(hot)` / `hot.wait_stream(cold)` between two green
   contexts (isolated workqueues) and hung. The C++ mechanism probe never
   exercised cross-context event waits (it launched both kernels independently
   and host-synced), which is why it passed. Fix: launch both tiers
   independently with no cross-stream waits (which also matches production's
   overlap phase — the tiers only join downstream at the output add).

Artifacts: `results-marlin-green.json` (sweep + prod control),
`results-diag-16.json` (solo/green/prod/plain + NVML clock/power),
`marlin_green_probe.py`. Mode flags: `--sweep`, `--diag`, `--hbm-cold`
(pessimistic both-tiers-HBM ablation), `--numa-node -1` (auto-detect).

## Grace→HBM cold-weight staging, W13-only — 18–38% faster (2026-07-28) — **⚠ SUPERSEDED: this was a straggler-workload + W13-only artifact. See the realistic-q4 verdict below, which reverses it.**

The pivot after FAIL-A. Since the co-residency contention is what kills overlap,
**stop co-running cold-Marlin-over-C2C with hot**. Instead: overlap a pure C2C
**weight transfer** (Grace→HBM copy) with hot Marlin, then run cold Marlin from
the staged HBM copy **after hot retires** (no co-residency). This was excluded
from the original plan (§0 "do not implement Grace-to-HBM staging") as a scoping
choice, not a finding; revisited here on the strength of the FAIL-A mechanism.

`--stage` mode measures the four unknowns (transfer solo, transfer∥hot dilation,
cold-from-HBM, end-to-end staged latency) against the production co-run union.

Representative w13, 19 hot / 3 cold, Grace-C2C, NUMA-verified, Booster:

| m | hot solo | transfer | cold_HBM | cold_Grace | hot dil under transfer | prod union | **staged** | staged vs prod |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 108.0 | 100.6 (387 GB/s) | 51.9 | 105.8 | +39.6% | 273.3 | 219.1 | **−19.8%** |
| 64 | 193.2 | 101.6 (383 GB/s) | 53.9 | 195.7 | +27.9% | 450.4 | 314.6 | **−30.2%** |
| 256 | 583.8 | 102.0 (382 GB/s) | 98.0 | 600.2 | +9.9% | 1278.9 | 768–816 | **−36 to −38%** |

**Correctness: PASS.** Weight copy is bit-exact (m=16/64 staged==direct, diff=0).
Both paths are individually deterministic (direct_vs_direct=0, staged_vs_staged=0).
At m=256 staged differs from direct by 7.8e-3 (1–2 bf16 ulps) — reduction-order
rounding from the different weight-fetch latency (Grace C2C vs HBM) reordering the
split-k accumulation, not a math error. Production adopts the staged path
consistently, so this is numerically equivalent (same as a batch/tile change).

### Why it wins

1. **`cold_from_HBM` is 2–6× faster than `cold_from_Grace`** (51.9 vs 105.8 µs at
   m=16; 98.0 vs 600.2 µs at m=256). HBM has ~8× the bandwidth of C2C; the staged
   copy is tiny (only the 3 *activated* cold experts, 38.9 MB w13).
2. **The transfer is weight-bytes-bound, not token-bound** — constant ~102 µs at
   380+ GB/s (90% of C2C roof) regardless of m, while hot grows with m. So it
   hides *better* at scale: hot dilation under the transfer drops +39.6% (m=16) →
   +9.9% (m=256). This is why the win grows with batch.
3. **`cold_from_Grace` scales terribly** (105.8 → 600.2 µs at m=256) — reading
   cold weights over slow C2C becomes the dominant per-layer cost at scale. This
   is the real production pain the staging removes.
4. The transfer overlaps hot as a *copy* (no SM competition), unlike cold-Marlin
   which fights hot for SMs. So phase 1 ({transfer ∥ hot}) is far cheaper than
   the production {cold-Marlin ∥ hot} co-run.

### Caveats / before integration

- **HBM headroom**: staging buffer = activated cold weights, ~39 MB (w13) +
  ~19 MB (w2) per layer, ~116 MB double-buffered for cross-layer prefetch. Must
  fit the existing physical-HBM-reserve gate. Small, but cold-heavy layers
  (more activated cold experts) need the worst-case sized.
- **The transfer still mildly contends at small m** (+39.6% at m=16) — the same
  memory-subsystem contention; it only fades at larger batch. At the m=16 target
  the win is still −19.8%, but the transfer is not free.
- **Variant not yet tested**: cross-layer prefetch (stage layer N+1's cold during
  layer N's hot) would hide the transfer entirely, but needs layer N+1's cold set
  known in advance (static tiering or routing lookahead). The measured variant is
  within-layer (transfer after routing, overlap hot).
- **Remaining confirmations**: full-chain (w13+w2) staging, real-route replay
  (cold-heavy stragglers), and production CUDA-graph integration.

Artifacts: `results-stage-m{16,64,256}.json`. Mode flag: `--stage`.

## Realistic q4 full-chain staging — **staging HURTS (+16–34%) everywhere; the W13-only win was an artifact (2026-07-28)**

This reverses the W13-only result above. Two corrections drive it:

1. **Test the real operating point.** The dominant decode shape is **q4** (4 MTP
   verify tokens, concurrency-1). Measured from the c1q4 routing traces
   (`../../2026-07-19-c1q4-placement/trace-977597`): per layer, per EP4 rank the
   MoE activates **mean 5.9 experts (p90 9)**, of which **mean 1.1 cold
   (18.5% — the "20% cold" figure is right; "32/layer" is the max not the mean,
   "8/rank" is ~p75 not the mean)**. The earlier probe's 19 hot + 3 cold was the
   much larger **c4q4 straggler** regime (m=16, mean 19.6/rank), not the dominant
   point.
2. **Test the full W13+W2 chain, not W13-only.** The full chain **doubles** the
   staged transfer (both weight matrices), and the dominant point has a *short*
   hot phase, so the transfer can no longer hide.

`stage_fullchain_probe.py` measures staged vs direct-Grace per cell with the
review's fixes: full W13+W2 chain, non-contiguous per-expert staging, one
common-start/final-join timing shared by both controls, L2 flush between iters
(production-realistic: 74 other layers evict L2 between reads), deterministic
balanced routing (cells comparable), cold=0 control. Booster, NUMA-verified,
iters=50.

**Staged vs direct-Grace (negative = staging helps):**

| m=4 (dominant q4) | cold=0 | cold=1 | cold=2 | cold=4 |
|---|---:|---:|---:|---:|
| hot=4 | −1.0% | **+16.4%** | **+28.8%** | **+34.1%** |
| hot=6 | +1.7% | **+16.8%** | **+29.1%** | **+29.3%** |
| hot=8 | −1.0% | **+18.7%** | **+27.3%** | **+26.3%** |

| m=16 | hot=6/cold=1 | hot=6/cold=4 | hot=15/cold=4 |
|---|---:|---:|---:|
| | **+17.5%** | **+30.3%** | **+22.0%** |

- **cold=0 → flat (±1.7%)**: staging is neutral with no cold (control passes).
- **cold≥1 → HURT +16–34% everywhere**, including m=16. Stable across two
  independent clean runs.

### Why staging hurts here

- **`cold_from_HBM` is ~flat (~130 µs) regardless of cold count** (HBM is fast;
  the tiny cold tier is overhead-bound, not bandwidth-bound), while
  **`cold_from_Grace` grows** (145→265 µs as cold goes 1→4). So the staging
  *benefit* (coldG−coldH) is real and grows with cold.
- **But the transfer dilates hot +44–107%**, growing faster than the benefit.
- **Root cause:** the transfer reads the same cold weights from Grace over C2C
  that direct cold-from-Grace reads — so staging does **not** avoid the co-residency
  C2C/HBM contention, it **front-loads** it and then adds cold-from-HBM on top.
  And a bulk transfer **steals HBM bandwidth more aggressively** than the
  interleaved cold-from-Grace compute kernel, which time-slices gracefully with
  hot. Front-loading the C2C read as a copy is *worse* than letting the cold
  kernel stream it interleaved.
- Staging only wins when the hot phase is **long enough to hide the transfer**
  (the m=256 W13-only straggler: hot 585 µs vs transfer 102 µs). At the short-hot
  q4 point (hot ~140 µs) even one cold expert's W13+W2 transfer (~70 µs) is ~half
  the hot phase and cannot hide.

### Disposition

**Do not pursue Grace→HBM staging for the dominant q4 decode path.** The direct
cold-from-Grace two-stream path (current production) is better at the real
operating point. Staging is only worth revisiting for the long-hot straggler
regime (large m, many hot experts, few cold) — and even there the full-chain
(W13+W2) transfer cost erodes the W13-only win. This is **not** a production
direction.

Artifacts: `results-fullchain-grid.json`, `stage_fullchain_probe.py`
(`--selftest`, `--iters`, `--numa-node -1`).



