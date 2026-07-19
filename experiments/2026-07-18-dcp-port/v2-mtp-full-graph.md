# V2 MTP full-graph follow-up

The c4 trace showed seven CUDA-graph replays separated by 300 eager GPU
operations per decode step. The V2 GPU runner already captures the complete
MTP prefill and recurrent draft routines, but two runtime blockers prevented it
from serving this configuration:

- `GPUWorker` unconditionally called a DSA trace hook that the V2 runner does
  not implement. The hook is now optional unless tracing was explicitly
  requested.
- Draft decode reused capture-time dummy DCP-local sequence lengths. Mixed c4
  prefill/decode batches consequently deadlocked. The draft metadata path now
  refreshes the persistent DCP-local length buffer before graph replay.

The mixed c4 warmup now completes 4/4 requests and the server remains healthy.
The deterministic 399,744-input c1 gate reproduces SHA-256
`d594e4d4268600a0d9e2d51355913907c638ef0c61615547598615384dcfc528`.

## Matched result

Both runs use MTP3, TP4/EP4/DCP4, `ag_rs`, concurrency four, seed 17, and
4,096 input plus 256 output tokens.

| Runner | Median ITL | Acceptance length | Effective aggregate |
| --- | ---: | ---: | ---: |
| V1 control | 64.46 ms | 3.006 | 186.52 tok/s |
| V2 full MTP graphs | 57.37 ms | 3.230 | 225.23 tok/s |

This is an 11.0% step-latency reduction and a 20.8% effective-throughput gain.
The exact-400K c1 median ITL also falls from 41.22 to 32.92 ms while retaining
the golden output SHA. Batched c4 text is not a bytewise gate because prior V1
c4 replicas are themselves not bitwise stable; completion, acceptance, and the
deterministic c1 SHA are used instead.

The requested early wrap-up skipped the full 400K c4 tail and a second
profiler run. The result therefore proves the launch-path improvement in live
latency, but does not yet provide a post-change eager-launch inventory.

## Validation

```text
.venv/bin/python -m pytest \
  tests/v1/worker/test_gpu_autoregressive_speculator.py \
  tests/v1/worker/test_gpu_worker.py -v
# 11 passed

PRE_COMMIT_HOME=/e/scratch/profound/naeimitabiei1/pre-commit-cache \
  .venv/bin/pre-commit run --files <four changed Python files>
# all applicable hooks passed, including ruff and mypy
```

Artifacts are the `dcp4-v1runner-c4-control-*.json` and
`dcp4-v2runner-dcpfix-async-c4-v1-*.json` files in this directory.
