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

## Status — §4.5 Marlin hot/cold probe (in flight, 2026-07-28)

`marlin_green_probe.py` extends the proven Phase-21 Marlin-MoE harness
(`benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py`: `make_tier`,
`build_gemm`, `global_routing`, `tier_map`, `align`) with the driver-API
green-context streams validated above. Both tiers run on their **own** green
streams with the production fork/join (`cold.wait(hot_start)` → both run →
`hot.wait(cold_join)`), timed with per-stream CUDA events plus an outer union
pair (`time_green_both`). The reused harness's `time_both_with_events` was wrong
for this test — it parks hot on the default stream, not the hot green stream.

Two modes:

- `--hbm-cold` (login-valid pessimistic): cold tier also in HBM. Isolates the
  **shared-HBM-port** contention from C2C. No Grace/C2C → login-node-safe.
- default (Booster only): cold tier in Grace-pinned UVA (`numa-node 2`,
  GPU0-paired), real C2C reads.

### HBM-cold login run (partial)

SOLO on a 16/116-SM green split, M=16, 19 hot / 3 cold experts, iters=20:

| | hot (116 SM) | cold (16 SM) | serial sum |
|---|---:|---:|---:|
| solo µs | 127.3 | 141.4 | 268.7 |

Concurrent was not captured on the login node (the first run used the wrong
timer; the green-aware `time_green_both` fix landed after). Deferred to the
Booster sweep — login-node concurrency numbers are not citable anyway
(benchmark-on-Booster rule).

### Booster §4.5 Grace-cold sweep — submitted

`job.sh` (Booster, 4× GH200, `numa-node 2`). First attempt (job 1068961)
failed: `agent_space/jupiter-env.sh: No such file or directory` — `repo_dir`
resolved to `/var/spool` under Slurm before `cd`. Fixed with an absolute
`repo_dir` fallback.

Re-submitted as **job 1069196** (2026-07-28): `--sweep` over cold=8/16/24/32 SM
(hot=124/116/104/96), M=16, 19 hot / 3 cold experts, `cold_share=0.13`, iters=50,
Grace-cold on NUMA node 2. Preceded by the cheap mechanism-probe sanity check.

Verdict (plan §4.5/§10) pending job completion:

- **PASS-A**: hot/cold interference ≤ ~10% AND overlap speedup approaching
  serial/min → green contexts rescue the Phase-23 zero-sum dilation → proceed to
  vLLM integration (§4.3-4.4, CUDA-graph capture under green streams).
- **FAIL-A**: interference high / overlap speedup ≈ 1 → the shared HBM port /
  L2 / power domain is the real bottleneck, not SM scheduling → drop to Track B
  (low-SM cold Marlin kernel).

