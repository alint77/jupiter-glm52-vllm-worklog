# Decode dive 3: c=1 k=7, captured DFlash2 drafter

Trace: `/e/fscratch/profound/naeimitabiei1/c2-k3/c1k7-graph/trace/alone-5K`
(../2026-10-08-dflash2-cudagraph `launch_c2.sh c1k7-graph`; torch profiler,
lone 5K request, 8-token verify, 89 steps x 4 ranks). Caveats: the
c2-k3 `serve.sh` copy uses the pre-frequency hot-set profile, and a 5K
code-explanation prompt rather than agentic traffic. Outputs in
`analysis.txt`.

Step 22.3 ms: target verify 20.1 busy, draft 1.04, logits 0.17, host-side
0.37, idle 0.63. Typical layer ~245 us, indexer layers ~285-310 us.

Per layer (rank 0, layer 40, 291 us): attention + combine + LSE RS 0-29 us;
o_proj 17 us; AR 8 us; router/route_prep 8 us; MoE 65-186 us (w13 79 us,
act overlapped, w2 40 us, finalize 36 us overlapped); MoE AR 58 us on this
layer (waiting for the slowest rank); qkv/indexer/cache/gather_cat 46 us.
Shared expert runs on a side stream under the MoE.

Ranked opportunities (estimates, not measured):
1. MoE rank imbalance: ar_moe wait 1.37 ms/step; arrival spread 18.9 us x 74
   layers; the late rank has the longest MoE chain in 93% of cases.
2. MoE kernel itself: ~121 us/layer span, ~9 ms/step; w13 79 us and w2 40 us.
3. FlashMLA sparse decode: 18 us main + 8.4 us split-KV combine per layer
   (combine 0.63 ms/step) for at most 2048 selected keys split across ranks.
4. Indexer layers (21): +40-60 us each; StableTopKFromGatheredCandidates
   12 us on 8 CTAs; ~0.95 ms/step for the indexer work in total.
5. cuBLAS left on the main stream (nvjet 4 + 3.6 us per layer, ~0.6 ms/step).
6. Drafter eager prepare + context-KV precompute: 43 launches, 0.64 ms span
   (shape is fixed at decode; the profiler inflates eager regions).

## Dive 4: op sequence, streams, SOL on the dd3-w768 windows (2026-10-08)

`dive4_sol.py` / `dive4b_refine.py` over
`/e/fscratch/profound/naeimitabiei1/decode-mem-dive/dd3-w768/trace/{window-2,window-11}`
(rank0 4.6K-ctx agentic, 66 steps, p50 20.66 ms; 130K source-code decode, 140
steps, p50 21.26 ms; raw outputs `dive4-win*.txt`, `dive4b-win*.txt`,
`dive4-launchcfg-win2.txt`). Unit note: this torch writes traces with ts/dur
in us; `analyze.py` divides by 1000 on purpose and reports milliseconds, so
its own numbers (e.g. the 22.3 ms above) are and were correct.
`dive4_sol.py` / `dive4b_refine.py` do per-kernel arithmetic in us and replace
the loader accordingly — earlier `analyze.py`-based results stand.

Step = 2 graph launches (target 2,503 kernels, draft 163) + ~90 eager kernels
(logits lm_head, sampler chain, dflash prepare). Target busy 18.66/19.34 ms,
draft 1.03, logits 0.165, host-side 0.39, idle 0.74/0.56. A ~0.33-0.42 ms gap
sits between logits and the draft; blaming it on CUPTI per-launch cost on the
eager sampler/prepare chain is a hypothesis (the same effect as
../2026-10-08-dflash2-cudagraph's "profiled host-bound eager drafter"), not
measured here. What is measured: unprofiled rows.jsonl gives 20.5-21.3 ms for
the same requests, bounding profile inflation at <=2-4%, against the Phase-54
+12.8% (that reference is real, top-level README).

Per-layer sequence (skip layer ~248 us): FlashMLA main 14.9 -> combine 6.5 ->
lse_rs 6.3 -> W_UV nvjet 3.9 -> o_proj decode_gemm 16.7 -> attn-AR 5.8 ->
cute_dsl 4.1 (shared-expert input) [aux: shared gate_up splitK 8.6 +
splitKreduce + silu -> down 12.7, all under the MoE] -> grouped_topk 4.5 ->
route_prep 5.5 -> w13 62.1 [act PDL-resident inside] -> w2 36.5 [finalize
inside] -> add_mul 1.6 -> MoE-AR (5.4 wire .. 23.4 mean) -> qkv_a 12.7 ->
q_b 7.6 -> glue -> 2x concat_and_cache 1.8 -> router NNT 3.7 ->
one_shot::gather_cat 9.1 (next layer's DCP query). Anchor layers add the
indexer block (~12 us at 5K, ~20 at 130K) and the skip-KV staging chain
(_mark/_compact/_remap/_gather_rows, 26-30 us, slack to first skip reader >=
94 us, n=1254 group-steps). Layers 0-2 replace the tiered MoE with two
decode_gemm's (24.5 + 13.7 us).

Streams: the target graph replay internally rotates across 4 recorded streams
(19/2632/2633/2635, capture-pool artifacts), so stream id != role; the real
concurrency is (a) shared expert on 2634, 2.36-2.42 ms/step busy, fully hidden
(join slack never negative), (b) skip staging interleaved under the anchor
tail, (c) PDL: act 57.3-61.0 us of its duration sits inside w13, finalize
29.2-30.7 inside w2; per 5-kernel window the durations sum 197.4 us but span
103.0 (208.0 / 108.3 at 130K), i.e. 94.4 us/layer (99.7 at 130K) are
overlapped — act/finalize durations must never be summed into budgets.

SOL per call (rank0, 5K / 130K; HBM roof 3.64 TB/s, C2C 421 GB/s). The
decode_gemm classes are position-classified, which only fires on the 56
non-anchor, non-dense layer windows per step, so their means are over
n=56 measured of the true 78 calls (the 21 anchor windows carry the indexer's
wq_b as an extra unclassed call, and layers 0-2 carry the dense-MLP
decode_gemms; the layer sequences show the same durations in those windows):

| kernel | n/step | us | byte model | achieved | % roof |
|---|--:|--:|---|--:|--:|
| decode_gemm o_proj | 78 | 16.7 | 50.3 MB bf16 | 3.02 TB/s | 83 |
| decode_gemm fused_qkv_a | 78 | 12.7 | 32.5 MB | 2.56 | 70 |
| decode_gemm q_b | 78 | 7.6 | 16.8 MB | 2.20 | 60 |
| tiered w13 | 75 | 62.1/65.8 | cold 14.2 MB/exp | t=8+33.6c, c=1.65/1.74 | ~87-93 of cold floor |
| tiered w2 | 75 | 36.5/38.0 | cold 7.1 MB/exp | t=4+16.8c, c=1.94/2.03 | same |
| lm_head target (then AG) | 1 | 140.6 | 475.6 MB | 3.38 TB/s | 93 |
| lm_head drafter | 1 | 147.7 | 475.6 MB | 3.22 | 88 |
| FlashMLA main (width 768) | 78 | 15.0 | 0.34 MB sel | fixed-cost/tile-bound | n/a |
| combine | 78 | 6.5 | 16 splits | fixed | n/a |
| attn-AR lamport | 78 | 6.3 | 96 KB payload | wire 5.6 + RMS | at floor |
| MoE-AR lamport | 72 | 23.4 | 96 KB | wire 5.4, wait 17.9 | 77% imbalance |
| one_shot gather_cat / lse_rs | 78/78 | 9.1 / 6.7 | <=0.6 MB | 2 barrier RTs | latency |
| shared expert gate_up/down (aux) | 75 | 8.6 / 12.7 | 12.6 / 6.3 MB | 1.39 / 0.50 TB/s | 39 / 14, hidden |
| router NNT | 75 | 3.7 | 3.1 MB | 0.85 TB/s | 23 |
| _topk_topp sampler | 1 | 243.6 | 0.6 MB | 8 CTAs, latency | ~0 |
| mqa_logits | 21 | 3.7 -> 8.1 | index K, ctx/4 x 128 B | ~0.5 TB/s at 130K | 14 |

Cross-rank (4 ranks, matched by call ordinal): AR excess is symmetric across
ranks (8.8-9.8 us/call each) -> rotating imbalance, not a fixed laggard rank.
attn-AR wait 0.06 ms/step; MoE-AR wait 1.29 (5K) / 1.25 (130K) ms/step,
p90 47 us — matches item 1 above (1.37 was the estimate). one_shot waits 0.18
/ 0.06 ms.

130K vs 5K: +0.48 ms/step total (~2.3%) = w13 +0.28 (implied cold 1.65 ->
1.74), mqa_logits +0.10, w2 +0.11. Everything else — FlashMLA, combine,
decode_gemm, AR, one-shots, lm_heads — is context-flat; the step grows only
through the cold-expert mix and the indexer.

Files: `dive4_sol.py` (--seq --sol), `dive4b_refine.py` (decode_gemm classes
by window position, attn/MoE AR split, dense layers, staging slack, PDL),
`dive4-launchcfg-win2.txt` (grid/block/regs/smem/occupancy per kernel).

## Review of dive 4 (2026-10-08)

Holds up (consistent with `breakdown-dd3-w768-rank0.txt`, the 10-07 dives and
the raw `dive4*` outputs): the per-layer sequence and PDL model (act / finalize
mostly inside w13 / w2, chain span ~103 us/layer, durations must not be
summed); decode_gemm at 83 / 70 / 60% of the HBM floor (o_proj / qkv_a / q_b,
matching the byte counts); MoE-AR wait 1.29 ms/step (dive 3 estimated 1.37)
with symmetric per-rank excess, i.e. rotating imbalance; and the new leads:
`_topk_topp` 0.24 ms/step on 8 CTAs, the drafter re-reading the 475 MB
lm_head shard (0.15 ms), router NNT at 23% of floor, mqa_logits growing with
context, 130K costing +0.48 ms over 5K, profiler inflation now <= 2-4% (the
Phase-54 +12.8% reference is real: top-level README).

To fix:
1. **The analyze.py "1000x small" note is wrong.** `analyze.py` divides the
   trace's microsecond ts/dur by 1000 on purpose and reports milliseconds
   (`event["t"] = event["ts"] / 1000`); its 22.3 ms/step matches unprofiled
   tok/s. Earlier numbers from it stand. `dive4_sol.py` loading in us for its
   own use is fine; drop the claim.
2. **decode_gemm n/step.** The SOL table says 78 per class, but `dive4b`
   classifies by position and saw 56 per step for each (the 21 indexer layers
   and 3 dense layers are not classed). The means are from those 56 layers;
   say so.
3. **dive4b output bugs.** "5 kernel durations sum 0 renorm'd" (the README's
   ~176 us is not in the printed output: recompute or cite its source) and the
   staging-slack line printing "ms? unit us".
4. **Inferred, not measured:** that the 0.33-0.42 ms logits -> draft gap is
   CUPTI per-launch cost. Plausible (same effect as the eager-drafter case in
   ../2026-10-08-dflash2-cudagraph), but the profiled-vs-unprofiled step
   comparison cited cannot resolve 0.4 ms. State it as a hypothesis.
5. **Wording.** "Step time is context-flat to 130K" right after +0.48 ms: say
   everything except w13 (cold experts) and the indexer is context-flat.

### Fixes applied (same day)

1. Unit note corrected: `analyze.py`'s us -> ms division is intentional; its
   own outputs stand. `dive4_sol.py`'s docstring corrected to match.
2. The SOL-table intro now states the decode_gemm means are position-classified
   over the 56 non-anchor, non-dense windows (of the true 78 calls).
3. `dive4b_refine.py` PDL print fixed to output the real numbers: durations
   sum 197.4 us vs span 103.0 -> 94.4 us/layer overlapped at 5K (208.0 / 108.3
   -> 99.7 at 130K), and the slack line prints "(us)". The README's earlier
   "~176 us sum / ~73 overlapped" is superseded by these.
4. The logits -> draft gap attribution is now marked as a hypothesis; only the
   <=2-4% inflation bound is claimed as measured.
5. Wording fixed as requested.
