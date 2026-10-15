# Decode and prefill steps at agentic context lengths (existing traces)

2026-10-15. Post-fix agentic c=4 runs sit at median 205K / max 332K context;
every roofline so far was taken at 5K / 50K. The cluster is busy, so this
uses the traces we have. A fresh run was planned (`prof_longctx.py`,
`launch.sh`: 4 live replay requests at 80K-300K, and prefill chunks of 0.5-4K
at 0-300K) and cancelled before it started.

Traces: `nfr-c4-2277156` (c=4 MTP3, 4 x 5K and 4 x 50K decode windows),
`c4tl-2333343` (agentic c=4 timeline: window 1 one decoder at 237K;
window 2 decoders at 294K + 69K with a ~240K prefill), `nfr-sizes-2277153`
(prefill 0.5-4K, fresh and @60K). Kernels unchanged since, except the
candidate-pack recompile fix (prefill idle only).

`graph_bd.py`: like nsys_bd.py but measures each verify graph over its own
span, so it works where prefill chunks sit between decode steps (nsys_bd.py
drops those as window gaps). Tables in `an/`.

## Decode: context barely matters

Verify-graph span by category (ms, mean of 4 ranks):

| category | 4 x 5K | 4 x 50K | 1 x 237K (+ prefill) | 294K + 69K (+ prefill) |
|---|--:|--:|--:|--:|
| span | 23.08 | 23.17 | 17.29 | 21.30 |
| MoE layer_kernel | 11.77 | 11.96 | 6.82 | 9.67 |
| decode_gemm | 3.41 | 3.38 | 3.23 | 3.39 |
| all-reduce + RMSNorm | 1.76 | 1.39 | 1.21 | 1.51 |
| DCP collectives | 1.55 | 1.59 | 1.28 | 1.56 |
| DSA indexer | 0.60 | 0.95 | 1.01 | 1.24 |
| cuBLAS GEMMs | 1.08 | 1.07 | 1.03 | 1.08 |
| FlashMLA | 1.06 | 1.08 | 0.87 | 1.04 |

The long-context windows ran fewer decode tokens (smaller graphs), so their
MoE is lower. Per kernel, only the indexer (paged mqa_logits 0.08 -> 0.17 ->
0.36 ms, top-k 0.04 -> 0.27 -> 0.30) and the skip-KV gathers (`_gather_rows`,
0.69 -> 1.06 -> 1.20 ms summed, side stream) grow with context. FlashMLA
sparse is flat (top-k 2048 keys per query, whatever the context). So the
c=4 / 50K SOL table (`../2026-10-11-nsys-fresh/an/sol-w2.md`) is the decode
reference at agentic lengths, plus ~0.5 ms of side-stream work:

| c=4, period 25.64 ms | ms/step | floor | gap |
|---|--:|--:|--:|
| MoE layer_kernel | 11.96 | 8.9 (75%) | 3.0 |
| dense GEMMs (decode_gemm + cuBLAS, 534 calls) | 4.58 | 2.48 | 2.1 |
| layer tail: AR+RMS 1.40, DCP gather_cat + lse_rs 1.35, FlashMLA 1.08 + combine 0.25 | 4.1 | ~1.0 | ~3.1 |
| outside the verify graph (MTP drafter, lm_head, sampling, prep) | 2.22 | | |
| indexer + skip-KV staging (side stream; C2C contention with MoE) | 1.2 + 1.3 | | ~0.2-0.5 |
| idle | 0.50 | | |

## Prefill: a 4K chunk at ~240K (c4tl window 2, ~8 chunks, rank 0)

| group | ms/chunk | note |
|---|--:|---|
| prefill MoE | 176.9 | flat in context; was ~240 before prefill v9 + cold prefetch |
| NCCL all-gather (DCP q) | 92.4 | ~226 MB received per layer, ~300 GB/s |
| dense GEMMs | 84.9 | ~48 ms floor (prefill-nsys) |
| indexer mqa_logits | 69.4 | grows with context (0.7 fresh, 15.7 @60K) |
| NCCL all-reduce (TP) | 61.2 | near NVLink bandwidth |
| sparse_attn_fwd | 51.5 | flat (48.5 fresh, 66.8 @60K) |
| DCP lse_combine | 41.4 | |
| indexer top-k | 27.5 | grows with context |

Indexer logits at 240K: 21 anchor layers x 4096 queries x 32 heads x 128 dims
x ~60K local keys (DCP4) = 42 TFLOP per chunk per rank, so 69 ms is ~610
TFLOPS, ~30% of FP8 dense peak (assumes the chunks sat near 240K; lower
context means lower FLOP and lower efficiency). At 300K it would be ~90 ms.

## Ranked levers for the steps

Weight: in the post-fix c=4 run, ~13.7K decode steps and ~380 prefill chunks
per 600 s. 1 ms per decode step is ~2.3% of the run; 10 ms per prefill chunk
is ~0.6%.

Decode:
1. MoE layer_kernel, 3.0 ms over its floor (#69).
2. Layer tail, ~3.1 ms: ~8-14 us latency-bound kernels x 78 layers at 15-23%
   SOL; fewer and fused kernels per layer.
3. Dense GEMMs, 2.1 ms over the HBM floor.
4. Outside the graph, 2.2 ms (drafter passes, lm_head, sampling).

Prefill (4K chunk at long context, ~650-700 ms):
1. DCP comms, 92 + 41 ms (q all-gather + LSE combine): sequence-split DCP
   prefill (prefill-nsys lever 4). At 240K the KV all-gather it needs grows
   to ~157 MB per layer (fp8 656 B rows), so the saving is smaller than the
   -70 to -90 ms estimated at 60K. Needs re-estimating.
2. Indexer, 69 + 28 ms and growing linearly: logits at ~30% of FP8 peak.
3. Dense GEMMs, ~37 ms over the floor.
4. Bigger chunks when no decoder waits (prefill-nsys lever 3): amortizes the
   flat MoE cost.
