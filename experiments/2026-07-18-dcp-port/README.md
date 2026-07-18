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

## Expected effects (to verify against trace)

- Per-rank DSA scan and top-k over context/4 (~1.5 ms/step -> ~0.4 ms).
- Sparse MLA at 64 real heads instead of 16-padded-to-64.
- Routed cold span shrinks with 1,087 vs 1,930 cold slots.
- New per-layer costs: DCP query all-gather + LSE-corrected reduce-scatter.
- Prefill/TTFT may regress: prefill chunks also pay the q all-gather.
