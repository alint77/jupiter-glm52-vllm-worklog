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

## 109-request hotness snapshot

The first natural-use snapshot contains 109 requests, 263,548 routed positions,
and 158,128,800 target-router selections. `expert-hotness-distribution.png`
compares its 75-by-256 grid with the earlier 24-prompt, fixed-256-token capture.
Counts are divided by the mean within each layer so the unequal capture sizes
remain comparable; 1x is uniform traffic within a layer.

Natural Claude Code use is substantially more concentrated. Its Gini
coefficient is 0.580 versus 0.385, the top 1% of layer-expert cells receive
12.64% versus 5.99% of routes, and mean entropy-effective experts per layer
fall from 197.5 to 112.0. This makes the real-use grid a better placement
candidate, but throughput still needs a matched placement A/B.

The two count grids, histogram, JSON summary, and plotting script are stored
in this directory.

## Layer concentration ranking

Every routed layer has exactly 2,108,384 selections because each routed
position invokes top-8 routing in all 75 layers. Layers therefore cannot be
ranked by total traffic. `layer-hotness-ranking.csv` instead ranks how
concentrated each layer's traffic is, using entropy-effective expert count:
lower means that fewer experts carry the traffic and the layer is more
placement-sensitive.

Layer 56 is most concentrated at 97.1 effective experts; layers 68, 58, 64,
and 55 follow. Layer 7 is most diffuse at 128.8 effective experts, followed by
layers 3, 4, 5, and 13. The CSV also records Gini, top-8/16/32 shares,
50%/80% coverage sizes, and each layer's hottest expert.
