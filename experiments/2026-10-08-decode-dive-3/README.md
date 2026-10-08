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
