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

The ~40% c=1 regression is collective-latency arithmetic. The later trace
measured ~271 DCP collectives per step (about 190 all-gathers and 81
reduce-scatters) at tens of microseconds each, ~+12 ms/step, outweighing
the ~5 ms/step saved by +843 hot experts, 4x-sharded DSA scans, and
unpadded MLA heads. TTFT improves slightly.

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

## Round 7 results: c=4 works; the first 400K measurement design was wrong

Job 976249 passed every gate on the c=4 server: semantic byte-identical,
and the c=1 exact-400K golden SHA reproduced (76.6 tok/s, consistent with
c=1 DCP4 minus one demoted expert and 4-size graphs). c=4 at 4K ran all
four requests concurrently: 31.2 ms TPOT = 32 tok/s per request, ~128
tok/s aggregate decode.

The c=4 exact-400K case (input 399,744, output 256) reported 825 ms TPOT -
not a decode collapse but a measurement artifact: 400K prefills serialize
(~110 s each; one partial prefill at a time), decode steps of running
requests queue behind 8,192-token prefill chunks, and with only 256 output
tokens the requests finish as the last prefill lands. There is no 4-way
steady decode window in that trace at all (`Running: 2..3` throughout; no
preemptions; KV peaked at 68%).

The repeat on a different node failed the fail-closed HBM audit by 0.2 GiB
(node variance ~0.5 GiB in free HBM); resubmitted with an 11 GB planned
reserve (976254).

Round 8 (v2 benchmark, jobs 976254/976261): input 395,904 + output 4,096
(= exactly 400,000, same per-seq block budget) leaves a multi-minute clean
4-way decode phase after the last prefill; steady-state TPOT to be
extracted from the detailed ITLs of the last-admitted request.

Operational note for agentic serving: at 400K, prefills are ~2 min each
and serialize, so four agents submitting cold simultaneously see TTFTs of
roughly 2/4/6/8 minutes. Cross-turn prefix caching (agents re-prefix only
their delta) is the mitigation, and is already enabled.

## Round 8 results: c=4 x 400K qualified

Both v2 jobs (976254 label `c4-dcp4-r11` with 11 GB reserve, 976261 label
`c4-dcp4-v2` with 10 GB) passed the c=1 golden-SHA gate on the c=4 server
(78.5 / 75.5 tok/s) and completed the steady-state case (input 395,904 +
output 4,096 x 4 concurrent). Steady 4-way decode, measured over the
window after the last prefill from per-request ITLs (tokens stream in
MTP bursts of ~2.8-3.5/step; each ITL is one engine step):

| Run | Steady window | Step p50 | Aggregate | Per request |
| --- | ---: | ---: | ---: | ---: |
| r11 | 74.2 s | 68.9 ms | 178.9 tok/s | 44.7 tok/s |
| v2 | 79.8 s | 71.4 ms | 173.2 tok/s | 43.3 tok/s |

Final serving menu on one Booster node (all lossless, SHA-gated):

| Config | Per-request decode | Aggregate | Use case |
| --- | ---: | ---: | --- |
| c=1, DCP1 (qualified default) | 129-136 tok/s | 129-136 | interactive single session |
| c=1, DCP4 | 76-82 tok/s | 76-82 | not useful alone |
| c=4, DCP4 | 43-45 tok/s | ~176 | agent swarm at 400K |

c=4 DCP4 delivers ~1.35x the aggregate of the best single-request config
and +36% effective per-agent throughput versus queueing four agents on
the DCP1 server (~33 tok/s effective each). The step time doubles
(~35 -> ~69 ms) while serving 4x the tokens: weight streaming amortizes
as predicted, and the per-step DCP collective tax is shared. Prefills
serialize (~110 s each; cold 4-agent TTFT ladder ~2/4/6/8 min) - relying
on cross-turn prefix caching in agentic use.

Unharvested levers, in expected order: MoE support-chain fusion and
indexer/merge collective coalescing. The MTP-aware placement and DCP trace
are analyzed below.

## Expected effects (to verify against trace)

- Per-rank DSA scan and top-k over context/4 (~1.5 ms/step -> ~0.4 ms).
- Sparse MLA at 64 real heads instead of 16-padded-to-64.
- Routed cold span shrinks with 1,087 vs 1,930 cold slots.
- New per-layer costs: DCP query all-gather + LSE-corrected reduce-scatter.
- Prefill/TTFT may regress: prefill chunks also pay the q all-gather.

## Round 9: trace-guided A2A optimization

The eight-step DCP4 c=1 trace at
`/e/scratch/profound/naeimitabiei1/glm52-dcp4-profiled-profile-976153`
attributes the communication portion of each decode step as follows:

| Operation | Calls/step | Time/step |
| --- | ---: | ---: |
| all-gather, grid 1 (mostly LSE) | 81 | 1.81 ms |
| all-gather, grid 18 (query) | 79 | 1.17 ms |
| all-gather, grid 16 (mostly indexer) | 25 | 1.35 ms |
| other all-gathers | 5 | 1.41 ms |
| reduce-scatter | 81 | 1.16 ms |
| LSE correction kernel | 81 | 0.14 ms |

The LSE all-gather, reduce-scatter, and correction chain alone is about
3.10 ms/step. `--dcp-comm-backend a2a` replaces that chain with one packed
all-to-all plus pack/unpack kernels; the query and indexer all-gathers
remain. NCCL symmetric memory (`VLLM_USE_NCCL_SYMM_MEM=1`) is tested as a
separate matched variant.

A four-rank mean over the same eight full-context decode steps gives a more
direct backend comparison:

| Backend | All-gather | Exchange | Auxiliary | Total communication |
| --- | ---: | ---: | ---: | ---: |
| `ag_rs` | 7.089 ms | 1.345 ms RS | 0.135 ms correction | 8.569 ms |
| plain A2A | 4.690 ms | 2.614 ms A2A | 0.330 ms pack/unpack | 7.633 ms |
| A2A + NVLS | 4.862 ms | 2.722 ms A2A | 0.332 ms pack/unpack | 7.916 ms |

Plain A2A saves 0.936 ms/step of communication and NVLS saves 0.653 ms.
At c=1, however, the extra A2A residency forced more experts into Grace:
plain A2A's total GPU span rose from 51.98 to 54.55 ms and NVLS to 55.66 ms.
The routed W4 MoE role alone rose from 9.37 to 11.41/12.10 ms. The c=4 test
is therefore decisive because its communication saving is amortized across
four requests.

### Residency and qualification

A2A needs more persistent HBM than `ag_rs`. The first full-residency runs
failed closed at about 9.50 GB free for plain A2A and 9.05 GB for A2A+NVLS
against the 10 GB observed-reserve gate. A profile with 2,813 hot
experts/rank initially freed only five physical experts because the planner
automatically promoted cold experts into remaining capacity. Commit
`3cd56a570` adds opt-in `VLLM_TIERED_MOE_PROFILE_CAP=1`: only an explicit
placement profile becomes a hard residency ceiling; default promotion is
unchanged. Its 38 focused tests and all pre-commit hooks pass.

Calibration jobs 976379/976380 confirmed that a 2,813 cap was still short at
9.59/9.15 GB. The qualified profiles use 2,771 hot experts/rank for plain
A2A and 2,749 for NVLS. Their held-out routing cold-hit rates are 3.84% and
3.93%, respectively, versus 3.66% at 2,813. Both validate against the exact
serving checkpoint fingerprint. Plain A2A loads 59.82 GiB of model memory
and leaves 9.69-9.70 GiB free after capture; NVLS loads 59.42 GiB and leaves
9.67-9.69 GiB. Neither relaxes the 10,000,000,000-byte runtime gate.

The fixed semantic continuation and the established c=1 400K SHA
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`
pass for plain A2A and NVLS. Concurrent and sequential random-prompt batches
are not a valid byte-equality pair, so the harness instead gates the semantic
continuation, output lengths, server health, and exact c=1 SHA. Jobs
976417-420 reached that gate, but editing the live shell script caused those
workers to resume at an invalid file offset before the long case. Jobs
976450/976452 are the clean resubmissions; this is why launch scripts must not
be modified while jobs execute them.

### c=4 result

The steady window begins after the final 395,904-token prefill and ends when
the first of four 4,096-token decodes completes. Tokens are reconstructed
from detailed ITL event timestamps, using each request's measured speculative
acceptance length.

| Backend | Window | Median step | Aggregate | Per request | Acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ag_rs` r11 | 74.18 s | 68.87 ms | 178.94 tok/s | 44.74 tok/s | 3.023 |
| `ag_rs` v2 | 79.81 s | 71.37 ms | 173.12 tok/s | 43.28 tok/s | 3.045 |
| `ag_rs` v3 | **61.53 s** | **62.32 ms** | 179.56 tok/s | 44.89 tok/s | 2.561 |
| plain A2A, p2771 | 72.08 s | 63.92 ms | 170.12 tok/s | 42.53 tok/s | 2.561 |
| A2A + NVLS, p2749 | 65.09 s | 65.90 ms | **190.15 tok/s** | **47.54 tok/s** | 2.977 |

Plain A2A shortens the median step versus the first two baselines, but not the
fresh v3 control; its generated path also accepts fewer MTP tokens and loses
end-to-end. NVLS retains normal acceptance and improves aggregate throughput
by 5.9% over the best `ag_rs` run and 9.8% over the lower baseline replica.
Acceptance is explicitly reported because small floating-point changes select
different greedy paths in every replica and make throughput content-sensitive.

An attempted follow-up cached A2A send/receive tensors by shape, based on an
estimated 0.784 GiB of duplicated buffers across FULL and piecewise graphs.
Jobs 976454/976455 falsified the hypothesis: graph memory stayed at
2.87 GiB versus 2.86 GiB before the patch, and the p2793 profile failed at
9,957,867,520/9,965,469,696 bytes free. The code and test were removed;
p2813 and p2793-NVLS follow-ups were cancelled before wasting more node time.

### NVLS reliability disposition

The 190.15 tok/s NVLS result is not the production default. Replica job
976497 stalled during MTP profiling: one rank was in
`ncclCommWindowRegister` while the other ranks had entered the matching
all-reduce. [NCCL documents window registration as a collective operation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2277/user-guide/docs/api/comms.html);
DCP can give ranks different allocator histories even when the current
collective has a uniform shape. Commit `990b1d378` therefore disables only
NCCL symmetric-memory all-reduce under DCP, while leaving the NVLS
all-gather/reduce-scatter path available. All commit hooks pass. Runtime
qualification of that safety split was deliberately left for a later session
rather than extending this investigation; job 976565 was cancelled during
model loading and no jobs remain active.

Until that qualification is run, `ag_rs` at 179.56 tok/s is the stable DCP4
default. A2A+NVLS at 190.15 tok/s is an experimental upper result, not a
qualified replacement.

Operationally, only vLLM/Inductor state is job-specific; stable FlashInfer
and TRT-LLM caches are shared. Caller cache overrides now survive nested
`jupiter-env.sh` sourcing. Never give two parallel jobs the same writable
vLLM cache root.
