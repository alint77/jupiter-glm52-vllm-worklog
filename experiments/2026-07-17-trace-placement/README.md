# Trace-derived owner and residency placement

This is the first minimal Phase 6 pass: capture exact logical top-8 routes,
derive a static EP4 placement from whole requests, load arbitrary per-layer
owners, and measure the result against the linear/even HBM-cache tracer bullet.
It is not the final tail-aware optimizer.

## Method

- Captured eight salted synthetic requests, each with 1,024 prompt and 64
  output tokens. Each file contains 1,087 routed token positions by 78 layers
  by top-8 expert IDs.
- Used six complete requests for training and two complete requests for
  held-out replay. Request hashes are embedded in the profile.
- Assigned exactly 64 experts per rank per layer with greedy load balancing,
  then spent the exact 3,176-HBM-slot budget per rank on the most frequent
  locally owned layer-experts.
- Fingerprinted the profile against the checkpoint config and index. The
  loader rejects mismatched, incomplete, unbalanced, or overlapping profiles.
- Loaded the complete 400K HBM-cache server with the profile, Inductor mode 3,
  and full/piecewise CUDA graphs. All 4,800 local expert destinations per rank
  streamed successfully.

The profile changes 14,318 of 19,200 owner assignments (74.6%). Every rank
still owns 4,800 layer-expert slots, with 3,176 in HBM and 1,624 in paired
Grace memory.

## Trace replay

The critical-count proxy sums, over layers, the maximum number of cold routes
assigned to any one rank for each token.

| Split | Placement | Cold route rate | Mean cold critical count | Mean owner max count |
| --- | --- | ---: | ---: | ---: |
| Train | Linear/even | 31.46% | 109.58 | 260.97 |
| Train | Trace-derived | 2.01% | 10.11 | 247.57 |
| Held out | Linear/even | 31.24% | 109.00 | 261.04 |
| Held out | Trace-derived | 2.32% | 11.56 | 248.02 |

Held-out replay preserves the large predicted reduction, but the v2 exit gate
requires a latency prediction within 20% of replay. This first frequency proxy
is not calibrated in latency units, so Phase 6 is not complete yet.

## End-to-end result

Matched batch-one 4,096-input/256-output runs used the same 400K HBM-cache
configuration. Decode rate is `1000 / TPOT` so it excludes TTFT.

| Seed | Placement | TTFT | TPOT | Decode |
| ---: | --- | ---: | ---: | ---: |
| 0 | Linear/even | 1.006 s | 27.564 ms | 36.28 tok/s |
| 0 | Trace-derived | 0.844 s | 26.262 ms | 38.08 tok/s |
| 1 | Linear/even | 0.957 s | 27.541 ms | 36.31 tok/s |
| 1 | Trace-derived | 1.071 s | 25.601 ms | 39.06 tok/s |

Mean TPOT falls from 27.553 to 25.931 ms, a 5.9% latency reduction and 6.3%
decode-throughput improvement. TTFT is noisy and is not claimed as a gain; one
run triggered a first-shape Triton JIT warning.

Random-token greedy continuations were not stable across repeated requests in
the same optimized process, so cross-placement text equality is not a valid
accuracy check for this workload. A higher-margin semantic prompt produced the
same eight tokens and token logprobs across three repeats, beginning with the
correct answer `Paris`; the first-token logprob margin was 3.125. A model eval
is still required before treating this as PR-ready model-affecting work.

## Reproduction

Capture from a server started with `--enable-return-routed-experts`:

```bash
.venv/bin/python agent_space/benchmarks/capture_routing_trace.py \
  --output-dir agent_space/experiments/2026-07-17-trace-placement/trace
```

Build the static profile:

```bash
.venv/bin/python agent_space/benchmarks/optimize_routing_profile.py \
  --trace-dir agent_space/experiments/2026-07-17-trace-placement/trace \
  --model "$GLM52_W4A16_MODEL" \
  --hot-slots-per-rank 3176 --train-requests 6 \
  --output-profile agent_space/experiments/2026-07-17-trace-placement/placement-profile.json \
  --output-report agent_space/experiments/2026-07-17-trace-placement/optimizer-report.json
```

Launch the qualified HBM-cache server as in the long-context experiment, adding
`--tiered-moe-placement-profile` with the generated profile path. The raw
benchmark JSON files contain the exact serving client settings.

## Artifacts

- `trace/`: request-bound exact route captures and hashes
- `placement-profile.json`: fingerprinted arbitrary owner/residency profile
- `optimizer-report.json`: training and held-out replay metrics
- `profile-plan.log`: plan-only validation output for all four ranks
- `phase6-optimized-*.json`: end-to-end serving results
- `checks.txt`: focused verification record
