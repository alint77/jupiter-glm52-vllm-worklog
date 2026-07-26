# Claude Code routing capture

This experiment records routed-expert IDs from natural Claude Code turns on
GLM-5.2 AutoRound W4G64 MTP3. It does not store prompts, responses, tool
arguments, or repository contents. Each trace starts at generation, avoiding
repeated counting of the conversation prefix.

The capture host uses c1/TP4/EP4 and the V1 model runner because routed-expert
return is not supported with DCP4 or Model Runner V2. Claude chooses the
natural stopping point; no 256-token output cap is applied. MTP verification
routes include rejected drafts.

Start a four-hour capture host and use it normally:

```bash
./claude-local-capture.sh
```

After collecting several representative sessions:

```bash
./claude-local-capture.sh --grid
```

The grid, ranking, heatmap, and summary are written under the capture state
directory for the active job.

Implementation source: vLLM fork commits `af5587af3` and `ffae5a399`. Two
focused route-writer tests and all commit hooks passed. The grid builder
completed a synthetic end-to-end check with the expected
`4 * 75 * 8 = 2,400` routes.

Job `1047751` exposed the Model Runner V2 incompatibility before loading
weights. V1 retry `1047770` became ready on `jpbo-008-32` and passed a streamed
Messages-API smoke test. Its trace contained 336 routed positions with shape
`336 * 78 * 8`; aggregation produced the expected
`336 * 75 * 8 = 201,600` selections. The smoke trace was moved to
`routes-1047770-smoke`, leaving `routes-1047770` empty for real Claude Code
activity.

The smoke test also caught a streamed output-token metadata bug. Commit
`ffae5a399` fixes later launches with a cumulative counter. The grid builder
does not depend on the `output_tokens` field; routed-expert arrays and hotness
counts were unaffected.

## OOM recovery

Job `1047770` later wedged during a real request after rank 2 failed a 2 GiB
FlashMLA temporary allocation with 2.10 GiB free. The HTTP health endpoint
continued returning 200 while the remaining ranks waited in a collective.
There were no completed real traces to preserve.

The replacement server reduces `gpu_memory_utilization` from 0.90 to 0.85,
shrinking the per-rank KV cache from about 16.9 GiB to 10.5 GiB while retaining
the 400k-token context limit. Four-hour job `1047954` on `jpbo-036-32` passed
streamed stress requests with 10,020 and 45,521 input tokens. Both generated
64 tokens, completed in 4.2 and 11.5 seconds, and left zero running or waiting
requests with no server errors. Their two valid `84 * 78 * 8` traces aggregated
to `100,800` routed selections and were moved to `routes-1047954-smoke`.
`routes-1047954` is empty for real Claude Code activity.

## Corrected 108-request hotness snapshot

An audit of the initial 109-record aggregation found one invalid 23,451-output
token trace. All 93,800 of its routed positions contained the dummy top-k
`[0, 1, ..., 7]` in every routed layer. It accounted for 35.6% of the original
positions and made the first histogram and EP-rank split invalid.

The grid builder now detects and reports all-default routed traces. Excluding
that record leaves 108 requests, 169,748 routed positions, and 101,848,800
target-router selections. `expert-hotness-distribution.png` compares the
corrected 75-by-256 grid with the earlier 24-prompt, fixed-256-token capture.
Counts are divided by the mean within each layer so capture sizes remain
comparable.

The aggregate distributions are nearly identical: Gini is 0.387 versus 0.385,
mean entropy-effective experts per layer are 197.6 versus 197.5, and p99
relative hotness is 4.146x versus 4.142x. Expert identities still move:
normalized cell-frequency correlation is 0.406, and only 74.3% of the current
profile's HBM slots overlap a frequency profile retrained on this capture.

## Layer concentration ranking

Every routed layer has exactly 1,357,984 selections because each routed
position invokes top-8 routing in all 75 layers. Layers therefore cannot be
ranked by total traffic. `layer-hotness-ranking.csv` instead ranks how
concentrated each layer's traffic is, using entropy-effective expert count:
lower means that fewer experts carry the traffic and the layer is more
placement-sensitive.

Layer 64 is most concentrated at 162.0 effective experts; layers 59, 55, 56,
and 58 follow. Layer 3 is most diffuse at 244.6 effective experts, followed by
layers 7, 5, 4, and 6. The CSV also records Gini, top-8/16/32 shares,
50%/80% coverage sizes, and each layer's hottest expert.

## EP4 rank split

`rank-ep4-layer-hotness.py` applies the actual non-linear ownership map in
`../2026-07-26-autoround-w4g64/hybrid-p0.5-profile.json`. Aggregate routing is
balanced: ranks 0–3 receive 25.36%, 24.78%, 25.16%, and 24.70% of all routes.

The critical rank receives 27.54% of a layer's routes on average and 30.42% at
p95, versus the balanced target of 25%. Six of 75 layers exceed 30%; none
exceed 35%. The worst case is layer 48 on rank 2 at 33.85%. This is a
routing-load proxy rather than a kernel-time measurement.

The four per-rank CSVs rank layers by local route load and include local
expert concentration. `ep4-layer-criticality.csv` ranks layers by their
busiest rank, while `ep4-layer-load.png` shows the complete layer sequence.

## Offline placement opportunity

At the same 2,870 HBM expert slots per rank, the current old-data placement
covers 71.02% of valid Claude routes. Reselecting residency from this capture
with the existing owners covers 83.75% and reduces the mean critical-rank cold
route share from 8.57% to 4.85%. Rebalancing owners as well reduces that proxy
to 4.19% and makes the busiest-rank layer share nearly 25%.

These are in-sample routing estimates, not throughput results. A new profile
should be trained on a chronological subset, checked on held-out requests,
then compared against the current profile at identical HBM budget and reserve.
