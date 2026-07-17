# Sampled real DSA index trace

Phase 6 completion on JUPITER allocation `967403`, Booster node
`jpbo-002-16`. The run used four GH200 GPUs in TP4/EP4, the 361.06 GiB
GLM-5.2 W4A16 checkpoint, the tail-aware expert placement profile, exact
400,064-token HBM cache capacity, `torch.compile`, and full/piecewise CUDA
graphs.

## Outcome

The opt-in tracer captured 32 real context rows from four independent 32,768
token prompts. Every row contains the 21 full-indexer layer sets and all 2,048
unique positions per set. The trace is request-bound, carries the checkpoint
config/index fingerprints, and passed range and uniqueness validation.

The implementation stays out of the ordinary serving path. When
`VLLM_DSA_INDEX_TRACE_DIR` is set, it:

- binds to both supported DSA indexer implementations immediately after model
  load, before profiling and compilation;
- selects at most one interval-aligned context row per request and model step;
- writes into a bounded
  `max_num_seqs x 21 x 2048 x int32` device buffer (172,032 bytes/rank in this
  `max_num_seqs=1` run);
- drains only rank zero into separate compressed NPZ samples.

The trace-only sidecar uses a synchronous drain to keep the implementation
small. It is not enabled during latency measurement or normal serving.

## Trace

The sampled context positions for every request were:

```text
8176, 16368, 24560, 32752, 32768, 32784, 32800, 32816
```

The 21 layer IDs are `0, 1, 2, 6, 10, ..., 74`, matching GLM's shared-indexer
groups. Across all 672 position sets, the native order was ascending for
86.83% of adjacent pairs. After sorting, the median gap was 1 position, p95
was 49 positions, and the mean set span was 26,616 positions. This is neither
the synthetic contiguous case nor fully random.

Each 32K+64 request took 7.77-8.35 seconds with tracing enabled. The first
request also paid first-use Triton JIT costs for the 32K prefill shape. These
times describe capture overhead and are not serving-performance results.

## Full-footprint cache replay

The highest-context real set was scaled monotonically over the exact 400,064
physical-token range, preserving its order and gap structure. The production
FlashMLA FP8 sparse-decode interface then read 78 layers and 104,792,064 bytes
per simulated token. Results are 100 CUDA-graph replays after five warmups.

| Pattern | HBM median | HBM p95 | Host median | Host p95 |
| --- | ---: | ---: | ---: | ---: |
| Clustered | 1.471 ms | 1.474 ms | 1.713 ms | 1.715 ms |
| Random | 1.496 ms | 1.498 ms | 3.002 ms | 3.009 ms |
| Sorted | 1.488 ms | 1.490 ms | 3.209 ms | 3.305 ms |
| Sampled real | 1.482 ms | 1.484 ms | 2.455 ms | 2.508 ms |

Host and HBM outputs matched exactly for all four patterns. The real pattern
is friendlier to Grace than random or sorted access, but its 2.508 ms host p95
still misses the v2 0.5 ms gate by 5.0x. This reinforces the existing
fail-closed AUTO selection of HBM for the main MLA cache.

## Startup and validation

- weights: 65.66 GiB/rank, 14:54 on the accepted run;
- compile: 172.98 seconds for the instrumented graph;
- cache: 400,064 tokens;
- observed free HBM after graphs: 6.18-6.19 GiB/rank, above the 5.59 GiB
  runtime minimum;
- focused pytest: 34 passed;
- Ruff check/format and diff checks: passed.

Two integration failures were caught rather than recorded as results. The
first binder targeted the newer NVIDIA-specific class while GLM used the
shared indexer. Binding by capability fixed both implementations. A later
capture was structurally present but unwritten because binding happened after
profiling had cached the graph; moving initialization before compilation fixed
it, and strict value validation now rejects that failure mode. Finally,
JUPITER's module-provided CUDA 13 `ptxas` is exported explicitly because a new
Inductor cache key otherwise searched for a nonexistent bundled binary.

## Reproduction

Start the compiled server with:

```bash
export VLLM_DSA_INDEX_TRACE_DIR=agent_space/experiments/dsa-trace
export VLLM_DSA_INDEX_TRACE_INTERVAL=16
```

Then run:

```bash
.venv/bin/python agent_space/benchmarks/capture_dsa_index_trace.py \
  --base-url http://127.0.0.1:8026 \
  --model glm52-w4a16-tiered --model-path "$GLM52_W4A16_MODEL" \
  --trace-dir "$VLLM_DSA_INDEX_TRACE_DIR" \
  --num-requests 4 --prompt-len 32768 --output-len 64

.venv/bin/python agent_space/benchmarks/mla_cache_full_footprint.py \
  --numa-node 0 --warmups 5 --iterations 100 \
  --real-index-trace "$VLLM_DSA_INDEX_TRACE_DIR/sample-00000031.npz"
```

Accepted raw data is under [`raw-v3/`](raw-v3/); the cache report is
[`full-footprint-real-100.json`](full-footprint-real-100.json).
