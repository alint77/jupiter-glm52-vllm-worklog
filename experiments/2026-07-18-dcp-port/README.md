# DCP4 port for FlashMLA sparse: c=1 400K qualification

Goal: enable decode context parallelism (dcp=4) on the qualified MTP3-overlap
tiered configuration, as the prerequisite for concurrency-4 serving at 400K
(c=4 x 400K needs 76 GiB/rank of replicated MLA KV without DCP; with DCP4 it
is the same 19 GiB footprint as today's single request).

## Implementation (uncommitted at submission time)

- `vllm/v1/attention/backends/mla/flashmla_sparse.py`: DCP support in the
  fp8_ds_mla mixed-batch path. Under DCP the common MLA layer all-gathers
  queries in the head dimension (16 local x 4 dcp = 64 heads, exactly the
  kernel's native head count, eliminating the 75% padding waste), the impl
  filters top-k indices to the rank's KV shard
  (`triton_filter_and_convert_dcp_index`), and returns base-e LSE for the
  cross-rank combine (`cp_lse_ag_out_rs`). The kernel returns lse=+inf for
  an all-filtered row; `mask_empty_dcp_lse` rewrites those to -inf, without
  which the combine produces NaN. `supports_dcp_with_varlen=True` keeps the
  MTP spec-as-decode reorder threshold from being clamped to 1.
- `vllm/config/vllm.py`: tiered contract now admits DCP1 or DCP4 with
  interleave 1; routing-trace capture is rejected under DCP (scheduler
  limitation) - capture traces at DCP1, apply profiles at DCP4.
- `tiered_moe_kv.py` / `tiered_moe_physical.py` / `tiered_moe_plan.py`:
  KV plan shards blocks across the DCP group
  (`ceil(len / (64 * dcp)) + null`).
- `tiered_moe_planner.py`: when the HBM budget outgrows a static residency
  profile (DCP shrinks the cache), cold experts are promoted round-robin
  deterministically; an overfilled profile still fails.

## Pre-submission verification (login-node GH200)

- `tests/v1/attention/test_flashmla_sparse_dcp.py` (new): the real FlashMLA
  sparse FP8 kernel under simulated DCP4 sharding matches the full-index
  reference for q_len 1 and 4, including an adversarial row whose top-k
  entirely misses three shards. This empirically pins the base-e LSE
  convention and the -1 skip behavior.
- `tests/model_executor/model_loader/test_tiered_moe_manifest.py`: 34 pass,
  including new DCP KV-plan and residency-promotion/overfill tests.
- DCP4 `--tiered-moe-plan-only` completes: main cache 20.47 GB -> 5.19 GB
  per rank (num_blocks 6,251 -> 1,564), hot experts 2,870 -> 3,713 of 4,800
  (cold -44%; some layers fully hot; empty cold tiers take the existing
  single-tier path).

## Reproduction

```bash
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

# Offline verification (login-node GH200):
.venv/bin/python -m pytest tests/v1/attention/test_flashmla_sparse_dcp.py \
  tests/model_executor/model_loader/test_tiered_moe_manifest.py -q

# DCP4 physical plan without touching GPUs (ray backend only passes the
# 1-GPU world-size check; plan-only exits before any executor starts):
#   vllm serve <model> ... --decode-context-parallel-size 4 \
#     --distributed-executor-backend ray --tiered-moe-plan-only

# Booster qualification:
#   job.sh <depth> <dcp_size> <label> <profile> [cudagraph_mode] [debug_env]
sbatch --job-name glm52-dcp4-mtp3 \
  agent_space/experiments/2026-07-18-dcp-port/job.sh 3 4 <label> false
```

## Jobs

| Job | Config | Label |
| --- | --- | --- |
| 970100 | MTP3, DCP4 | dcp4-mtp3 |
| 970101 | MTP3, DCP4 repeat | dcp4-mtp3-repeat |
| 970102 | MTP3, DCP1 fresh control | nodcp-control |
| 970103 | MTP3, DCP1 fresh control repeat | nodcp-control-repeat |

All: exact 399,744-input/256-output, seed 13, greedy, ignore EOS, plus the
semantic smoke prompt, via `job.sh <depth> <dcp> <label> <profile>`.

## Correctness gate

Primary: exact-400K continuation SHA-256
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528` and the
eight-token semantic continuation. Note: DCP changes attention reduction
order, so ULP-level logit shifts could legitimately flip a greedy near-tie.
If the SHA differs, do not assume failure: diff the continuations
token-by-token, check the divergence point's logprob margin, and only then
classify as bug versus reduction-order tie-break.

## Round 1 results

Both DCP1 controls passed the full gate: exact-400K SHA matches
`d594e4...cfc528` and the semantic prompt is byte-identical, confirming the
planner/residency changes are inert at DCP1. Control decode: 136.38 and
129.08 tok/s (mean TPOT 7.33/7.75 ms) - within the known run-to-run band
around the historical 123.65-127.67 controls.

Both DCP4 jobs (970100/970101) died identically: worker 3 killed by signal
(no Python traceback) during FULL CUDA graph capture of the size-4 decode
shape, immediately after custom-AR graph address registration. Diagnosis:
`cudagraph_num_of_warmups` defaults to 0, so the first-ever execution of
the DCP decode path (shape-specialized `_correct_attn_cp_out_kernel`
Triton JIT, decode-shape collective paths) happens inside capture; lazy
initialization during capture is fatal. The eager 8192-token profile pass
warms only the prefill shape.

Round 2 (`run-server.sh` now sets `cudagraph_num_of_warmups: 1`):

| Job | Config | Purpose |
| --- | --- | --- |
| 970395 | DCP4, PIECEWISE, warmup 1 | eager-path correctness + SHA |
| 970396 | DCP4, FULL_AND_PIECEWISE, warmup 1, NCCL_DEBUG=WARN, CUDA_LAUNCH_BLOCKING | fix candidate; diagnostics if it still fails |

## Round 2 results: DCP4 correctness qualified

Job 970395 (DCP4, PIECEWISE graphs, warmup 1) served and completed the full
matrix. The exact-400K continuation SHA-256 matches the golden
`d594e4...cfc528` byte-for-byte and the semantic prompt is identical:
**the DCP4 attention path is lossless in the real model.** TTFT at exact
400K is 112.2 s, inside the 111.4-113.0 s control band, so the prefill
query all-gather cost is negligible. Decode is piecewise-slow as expected
(35.5 ms TPOT at 400K vs ~7.5 ms full-graph control; ~3,600 launches/step
without full graphs) - this run is a correctness vehicle only.

Job 970396 (FULL_AND_PIECEWISE, warmup 1, NCCL_DEBUG=WARN) still died
silently at the same point, proving the eager warmup does not fix the
capture crash. NCCL printed no warnings; CUDA_LAUNCH_BLOCKING surfaced no
Python error - a hard signal during FULL-graph capture of the size-4
decode shape. Piecewise capture completes; only the FULL capture
(attention + DCP collectives in-graph) dies. NCCL symmetric-memory AG/RS
is off by default (ruled out); the DCP group uses plain pynccl (custom AR
is TP-only, ruled out). Job 972505 reruns full-graph with
PYTHONFAULTHANDLER=1, NCCL_DEBUG=INFO, and core dumps to capture the
crashing frame.

## Round 3: capture-crash bisect

Job 972505 (faulthandler + core dumps) died identically with no
faulthandler output - faulthandler catches SIGSEGV/SIGABRT but not
SIGKILL-style deaths, and no core file appeared. NCCL 2.28.9 logged no
error; all communicators initialized in 0.1-2.8 s at startup, none
mid-capture. Slurm MaxRSS was ~181 GiB (host), but the piecewise DCP4 run
carries the same memory layout and survives, so steady-state host OOM is
ruled out.

Job 974928 (`job-bisect.sh`) boots the server three times with temporary
env gates in `mla_attention.py` (TEMP-DEBUG, working tree only, never to
be committed) that produce shape-correct but numerically wrong output:

| Variant | In-graph DCP ops | If it captures |
| --- | --- | --- |
| no-comm | index filter + kernel + lse mask only | crash is in the collectives |
| ag-only | + query all-gather (pynccl) | crash is in the LSE merge (ag + triton + reduce-scatter) |
| full-dcp | + LSE merge | control: expected crash |

Health endpoint responding = capture completed for that variant.

## Round 3 results and revised diagnosis

All three variants crashed, including no-comm - which invalidated the
bisect's premise: the env gates only removed the MLA-layer DCP collectives,
but the DSA indexer's own DCP top-k merge
(`sparse_attn_indexer._merge_dcp_topk_global`: Triton candidate pack ->
`get_dcp_group().all_gather` -> CuteDSL stable top-k) runs inside the
captured graph in every variant. Meanwhile all four suspect ops (CuteDSL
stable top-k, Triton pack, DCP index filter, FlashMLA sparse FP8 kernel)
capture and replay cleanly in single-GPU isolation on the login GH200
(`capture_repro.py`).

The surviving hypothesis fits every observation: in the qualified DCP1
config, no NCCL collective is ever captured (the vocabulary all-gather
lives in `compute_logits`, outside the FULL graph), so **the DCP port
introduces the first in-graph NCCL collective on this stack** (PyTorch
2.12, NCCL 2.28.9+cuda13.0, aarch64). Piecewise works (collectives eager),
DCP1 full-graph works (no in-graph collectives), every DCP4 full-graph
variant crashes (indexer all-gather always in-graph).

Round 4: minimal 4-rank pynccl-all-gather-inside-capture repro
(`pynccl_capture_repro.py`, jobs 976071 base / 976072 NCCL_CUMEM_ENABLE=0 /
976073 NCCL_NVLS_ENABLE=0, submitted in parallel). If the base repro
crashes and an env variant survives, that env is the fix; if the base
passes, the composition (side-stream capture, multiple comms, aux streams)
is at fault and the repro grows toward the production pattern.

## Round 4 results: every reduced repro passes

- `capture_repro.py` (login GH200): CuteDSL stable top-k, Triton pack, DCP
  index filter, and the FlashMLA sparse FP8 kernel each capture and replay
  cleanly in isolation.
- `pynccl_capture_repro.py` (4 ranks, jobs 976075-77): pynccl all-gather
  inside plain CUDA graph capture passes 4/4 with default env,
  NCCL_CUMEM_ENABLE=0, and NCCL_NVLS_ENABLE=0 alike.
- `pynccl_capture_repro2.py` (4 ranks, job 976086): full vLLM parallel
  state (world/TP/DCP comms), DCP all-gather captured under vLLM's
  graph_capture() context, custom-AR sharing the same graph, and an
  aux-stream fork in the same graph - all four stages pass 4/4.

The crash therefore requires the real model graph. Remaining deltas: the
graph-memory profiling capture runs against a minimal stand-in KV cache
with a temporary graph pool, the captured graph is the Inductor-compiled
forward with ~3,600 nodes on multiple streams, and two graphs (piecewise
then FULL) are captured back to back.

## Round 5: instrumented server runs (in flight)

TEMP-DEBUG hooks in `gpu_model_runner.py` (env-gated, uncommitted):
phase markers around `_warmup_and_capture` plus
`faulthandler.dump_traceback_later(30s)` - periodic all-thread stack dumps
that survive even a hang-then-SIGKILL death; and
`TIERED_SKIP_FULL_CG_PROFILE=1` to drop FULL graphs from the profiling
capture only, so the later real `capture_model` (normal pool, real KV
cache) is tested directly.

| Job | Config | Question |
| --- | --- | --- |
| 976087 | instrumented, full profiling | which phase dies; stacks at death |
| 976088 | instrumented + skip FULL profiling | does the real capture path survive? |

## Round 5 results: crash localized and fixed

Job 976087's markers: piecewise capture completes, the FULL eager warmup
(DCP path executing for real) completes on all four ranks, and death is
inside the FULL profiling capture itself, within seconds. Job 976088
(skip FULL from the memory-profiling pass only): **CAPTURE_OK** - the real
`capture_model` pass captured the identical FULL graph with all ~300
in-graph DCP collectives cleanly (1.91 GiB, 5 s) and the server served.

Conclusion: the crash is specific to the graph-memory profiling capture's
composition (temporary graph pool + minimal stand-in KV cache); the same
graph captures fine against the persistent pool and real KV cache. Fix
committed (`67e6d48ff`): `profile_cudagraph_memory` skips FULL graphs when
`decode_context_parallel_size > 1` and estimates from piecewise only (the
tiered KV plan overrides available memory, so the estimate is not
load-bearing). All TEMP-DEBUG gates removed.

Round 6: full-performance qualification, jobs 976128/976129 (DCP4,
FULL_AND_PIECEWISE, exact-400K matrix, two runs), gated on the golden SHA
and benchmarked against today's DCP1 controls (129.1/136.4 tok/s).

## Round 6 results: DCP4 full-graph qualified; c=1 decode regresses

Both runs pass the gate: exact-400K SHA matches the golden hash and the
semantic prompt is byte-identical under full CUDA graphs.

| Config | Exact-400K decode | 4K decode | TTFT |
| --- | ---: | ---: | ---: |
| DCP1 controls (today) | 129.1 / 136.4 tok/s | 85.1 / 97.1 | 111.4-113.0 s |
| DCP4 full graphs | 78.4 / 80.6 tok/s | 70.0 / 86.4 | 110.0-110.3 s |

The ~40% c=1 regression is collective-latency arithmetic: DCP adds ~312
small NCCL ops per step (78 layers x indexer-AG + q-AG + lse-AG + RS) at
tens of microseconds each, ~+12 ms/step, outweighing the ~5 ms/step saved
by +843 hot experts, 4x-sharded DSA scans, and unpadded MLA heads. TTFT
improves slightly.

This does not defeat the c=4 goal: DCP4 is the only KV-feasible route to
c=4 x 400K, and the collective tax is per-step, so it amortizes across
concurrent requests. Four agents served sequentially at DCP1 see an
effective ~33 tok/s each; projected c=4 DCP4 is ~80-90 tok/s per request.
The c=4 measurement is the decision point.

In flight: 976153 (profiled DCP4, trace attribution of the added step
time) and 976156 (`--dcp-comm-backend a2a`, halves the merge collectives).

## Round 7: concurrency 4 (in flight)

Side results: the profiled DCP4 c=1 run passed (SHA ok, 81.7 tok/s, trace
at `glm52-dcp4-profiled-profile-976153` on scratch, unanalyzed). The
`--dcp-comm-backend a2a` variant failed the fail-closed HBM audit
(8.86 GiB free vs 9 GiB minimum - the a2a path allocates slightly more);
deprioritized.

c=4 enablement (`4dc352d24`): KV blocks provisioned per concurrent
sequence (c=4 x 400K DCP4 = 6,253 blocks = 20.74 GB/rank, same footprint
class as DCP1 c=1), the hot/cold overlap gate derives from
(depth+1) x max_num_seqs, the tiered contract admits max_num_seqs 1-4
(requiring DCP), and residency profiles now demote deterministically when
the budget shrinks (plan-only confirms hot 2,870 -> 2,869). Full graphs
capture at sizes 4/8/12/16 (`run-server-c4.sh`).

Benchmark (`run-benchmark-c4.sh`): c=4 cases use seed 17 so prompts stay
disjoint from the seed-13 golden prompt; the gate is the c=1 exact-400K
SHA served from the same c=4 server, plus the semantic prompt. Headline:
4 concurrent 399,744-token requests, aggregate and per-request decode.
Jobs 976249/976250.

## Expected effects (to verify against trace)

- Per-rank DSA scan and top-k over context/4 (~1.5 ms/step -> ~0.4 ms).
- Sparse MLA at 64 real heads instead of 16-padded-to-64.
- Routed cold span shrinks with 1,087 vs 1,930 cold slots.
- New per-layer costs: DCP query all-gather + LSE-corrected reduce-scatter.
- Prefill/TTFT may regress: prefill chunks also pay the q all-gather.
