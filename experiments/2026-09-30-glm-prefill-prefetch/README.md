# 2026-09-30 GLM prefill: double-buffered cold prefetch

Question: does rotating the tiered-MoE cold prefetch through two HBM slots
(`VLLM_TIERED_MOE_COLD_PREFETCH_SLOTS=2`, commit d979012d19) hide the copy,
and does it make a lower threshold (512) pay off?

Why: in the one-slot design layer L+1's copy is issued after layer L's MoE and
waits for it (the slot's last reader), so it only overlaps L+1's attention.
Measured on the 2026-09-30 prefill traces (hold 2123062, windows 1 and 3,
~1.7K new tokens): the copy is ~97 ms per prefill per rank and only 41-52% of
it overlaps compute.

Arms (`launch.sh`, same-node pairs, MiMo agentic task set, 150 requests):

| arm | slots | threshold |
|---|---|---|
| s1 (production) | 1 | 1024 |
| s2 | 2 | 1024 |
| s2x | 2 | 512 |
| off | - | no prefetch (no slot budget, more hot experts) |

Nodes 1-6 run s1/s2/s2x in all six orders; nodes 7-8 run off/s2 both ways;
a ninth node runs s2x with `VLLM_TIERED_MOE_COLD_PREFETCH_VERIFY=1` (25
requests) and then a 4-window prefill trace of s2x. Rows and logs:
`/e/fscratch/profound/naeimitabiei1/agentic-bench/{rows,run,server}-pf-*`.
`agentic_bench.py` now also records prefix-cache hit/query tokens per request,
so new tokens = queried - cached.

Cost to watch: the second slot (~450 MiB/rank) comes out of hot-expert HBM,
demoting ~22 experts per rank, so decode step time is compared too.

## Results

The 9-node full-set run (`launch.sh`) was cancelled: a synthetic prefill
sweep is enough for a kernel/overlap change. Replaced by `launch_short.sh`
(vllm bench serve, random prompts, 1 output token, 10 per length, median TTFT;
then 20 agentic requests for decode) and `analyze_short.py`.

- Bug found first: slot 1 started 8 bytes off 16-byte alignment (an expert is
  21,233,672 bytes, 8 mod 16; the largest tier has 33) -> CUDA misaligned
  address in Marlin. Slots now start on 4 KiB boundaries; test added.
- Correctness (s2x + VERIFY=1): 232 chunks, 0 mismatched, 74 of 75 layers
  from the slot (the chunk's first layer reads Grace by design).
- Same node (n1), median TTFT ms, s2x minus s1:
  512 +15.3, 1024 +24.3, 2048 -3.0, 4096 +0.6; decode +0.4 ms/step (the
  second 668 MiB slot demotes ~32 hot experts per rank).
  Double buffering does not help: prefill is host-bound (see the traces).
  Kept in the tree, default 1 slot.
- s2 (2 slots, threshold 1024) deadlocked at CUDA graph capture 3/3 starts
  (ranks 0-2 in an inductor runtime autotune + synchronize, rank 3 already in
  register_graph_buffers); s2x started 3/3. Not root-caused.

Afterwards, per the user: two slots made the vLLM default (257499f581) and
serve.sh's threshold 512. The s2 deadlock was never seen with threshold 512.

## Prefill CUDA graphs

Piecewise capture for prefill sizes (`launch_cg.sh`, `cg_next.sh`,
`launch_cg1k.sh`), prefetch pinned to 1 slot / 1024 in both arms.

- Capture [8..2048] (13 sizes): graph pool 1.50 GiB vs 0.63 GiB for [8];
  startup failed the observed free-HBM check (5.57 GB free, 6.0 required).
  KV is sized exactly to 400,255 tokens, so there is no slack elsewhere.
- Capture [8..1024] (128-token steps above 512): 5.87 GB free, still short.
  User chose to take it from the margin: `VLLM_TIERED_MOE_OBSERVED_HBM_TOLERANCE_GB`
  (393c286d72), 1.5 in serve.sh -> 5.5 GB required.
- Median TTFT (vllm bench serve, random prompts, 10 each), ms:

| arm | 512 | 1024 | 2048 | 4096 | decode ms/step | tok/step |
|---|---|---|---|---|---|---|
| base n1 | 218 | 249 | 341 | 1009 | 23.6 | 3.4 |
| base n2 | 222 | 266 | 339 | 1003 | 23.8 | 3.4 |
| base n3 | 189 | 229 | - | - | - | - |
| cg1k n2 | 173 | 240 | 337 | 1002 | 23.4 | 3.4 |
| cg1k n3 | 175 | 242 | 334 | 1005 | 23.7 | 3.6 |

  512: graphs 173-175 (tight) vs baselines 189-222 -> -14 to -48 ms.
  1024: no clear gain. >1024 is eager in both, unchanged. Decode unchanged.
- Defaults 2 slots + threshold 512 + capture [8..1024] + tolerance 1.5 started
  cleanly (final-n4); its sweep did not finish before the hold ended (the
  `vllm bench serve` client took minutes to start on a slow filesystem).
- Note: prefetch refuses graph capture, so captured prefills (<=1024) replay
  without it; with threshold 512 it now acts only on eager chunks >1024.
- serve.sh defaults updated accordingly.

Next: trace a graphed 512-1024 prefill to see what the eager attention part
(indexer, sparse MLA, DCP collectives) still costs.
