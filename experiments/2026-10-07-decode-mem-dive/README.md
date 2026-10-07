# Decode timing and HBM deep dive, full-memory-stack prod (2026-10-07)

Prod serve.sh (DFlash2 k=7, DCP4, 400K, skip-layer MLA KV + drafter KV on
Grace, fp8 drafter KV + weights, 4K chunks, reserve 4.7: 3,531 hot/rank).
Holds 2216509 / 2216515 / 2216516. `run.sh <mode> <tag>`:

- `timing`: agentic task set (16 requests, torch-profiler windows) then
  `longctx.py` decode windows at 50K and 130K tokens of real source code
  (temperature 1.0, top_p 0.95).
- `nsys` / `nsyslong`: one Nsight Systems window (graphs traced whole), agentic
  / 130K.
- `mem`: local `MEM_SNAPSHOT_DIR` probe in gpu_worker.py (uncommitted):
  free / torch-reserved per startup phase, allocator snapshots with Python
  stacks at startup, after the profile run and after a 130K request.

Analysis: `layer_timeline.py` (one step's per-layer kernels), `coll_wait.py`
(cross-rank wait per collective), `moe_skew.py` (where the MoE all-reduce wait
comes from), `step_tail.py` / `between_graphs.py` (work between verify
graphs), `mem_snapshot.py` / `mem_peak.py` (itemised live / peak HBM),
`l2_prefetch_probe.py` (M=8 GEMMs with weights pre-loaded into L2).

## Decode step (~22.4 ms unprofiled at ~5K context; nsys: graph 20.55 +
eager 1.53 + idle 0.33, host never late)

One layer (~245 us) on the critical path, rank 0, skip layer 40:

| us | |
|--:|---|
| 18 + 8 + 7 | FlashMLA sparse (128 CTAs) + split combine + DCP LSE reduce-scatter |
| 4 + 21 | W_UV + o_proj (bf16, 96 of 132 SMs) |
| 6 | attention AR + RMSNorm (FlashInfer Lamport) |
| 5 + 5 + 6 | router GEMM + grouped_topk + route_prep |
| ~73 | tiered MoE w13/act/w2/finalize (shared expert overlapped on a side stream) |
| 5 .. 40 | MoE AR + RMSNorm: transfer ~5, the rest waits for the slowest rank |
| 14 + 4 + 8 + 2 | fused_qkv_a (split-K + reduce) + norms + q_b + rope |
| 2 x 1.8 + 4 | KV writes + W_UK |
| 9-12 | DCP query gather_cat |

Anchor layers add the indexer: +48 us at 5K, +64 us at 50-130K (x21).

Ranked (ms/step, ~5K context unless noted):

| # | item | now | lever | est. gain |
|---|---|--:|---|--:|
| 1 | dense bf16 GEMMs at M=8 (o_proj, qkv_a, q_b) | ~3.9 solo | weights to L2 ahead of use; probe: fully L2-resident o_proj 22.7 -> 15.2 us, qkv_a 16.8 -> 10.9, q_b 8.9 -> 5.5 (L2 is 60 MiB) | up to 1.3 |
| 2 | MoE AR wait = MoE-chain spread across ranks (94% of the time the last rank is the one with the longest chain) | 1.36 | replica balancer's COST_US table is MiMo's; a GLM-measured table | unknown, < 1.36 |
| 3 | DCP one-shot collectives (2 barrier round trips each, 177/step) + split combine | 1.24 + 0.63 | Lamport-style protocol (no barriers), as FlashInfer's AR (5.7 us incl. RMSNorm) | 0.35-0.5 |
| 4 | router GEMM + grouped_topk + route_prep, serial | 1.15 | grouped_topk folded into route_prep | ~0.3 |
| 5 | sampler `_topk_topp_kernel` (only with top_p/top_k) | 0.19 | 8 CTAs for 8 rows over 154,880 vocab; multi-CTA | ~0.17 |
| 6 | DFlash2 drafter eager (graph defective since 836f8871d7) | 0.33 idle | fix the captured graph | <= 0.3 |
| 7 | drafter `fc` replicated (189 MB fp8/rank, 74 us) | 0.074 | TP-shard | ~0.05 |
| 8 | indexer top-k kernels, 8 / 64 CTAs (long context) | 0.5 at 130K | multi-CTA | ~0.3 at 130K |

Not opportunities: lm_head x2 (target + drafter, 476 MB each, at the HBM
floor); host launch (nsys host-late ~0); the extra GPU idle seen at 50K under
the torch profiler (cudaGraphLaunch takes 1.7-2.0 ms under CUPTI vs 226 us
unprofiled).

## HBM per GPU

Non-torch (CUDA context + communicators): 0.55 GiB context, +2.16 GiB at
distributed init (TP NCCL comm 0.91 incl. NCCL's module load, DCP NCCL comm
0.47, EP NCCL comm ~0.47, DCP prefill combine buffer 0.25, custom AR + symm
0.07), +0.59 in the profile run, +0.27 at graph capture: 3.57 GiB.

Torch, live at startup (rank 0, 87.8 GiB): routed experts 69.9; bf16 dense
~9.8 (o_proj 3.66, qkv_a 2.44, q_b 1.22, shared expert 1.35, kv_b 0.53,
indexer 0.33, router 0.25); shared workspace 2.22; KV on HBM 1.54;
cold-prefetch slots 1.07; embed + lm_head 0.89; fp8 drafter 0.60; RoPE
caches 0.375 (target 0.125, drafter 0.25); graph pool 0.46 (0.43 free).

| item | GiB | fix |
|---|--:|---|
| FlashMLA bf16 prefill workspace, 5 x max_model_len x 576 x 2 B, never read under DCP | 2.15 | skip under DCP |
| indexer prefill K-gather workspace, 40 x max_model_len rows (sized for 40 concurrent prefills) | 1.97 | max_model_len x min(40, max_num_seqs) |
| duplicate NCCL communicators (TP, DCP, EP are the same 4 ranks) | ~0.9 | share TP's |
| one-time `prefill_scale_exponent` transient in the profile run (cat + float + abs of all scales) | 3.0 peak | chunked min/max, same integer |
| RoPE caches sized past max_model_len, drafter's own copy | ~0.2 | trim / share |
| embedding table (8-row gather per step) | 0.44 | Grace |

The two workspaces shared one buffer (`get_simultaneous` grows a single
allocation), so removing only FlashMLA's let the indexer regrow it to 1.97
GiB: ws0 (FlashMLA only) gained +54 hot and +0.2 GiB observed free.

### Memory fixes measured (startup runs, hold 2216516, uncommitted vLLM edits)

| run | change | hot / rank | startup free (min 2.53) | profile-run max_allocated |
|---|---|--:|--:|--:|
| mem | prod | 3,531 | 2.67-2.79 GiB | 89.5 GiB |
| ws0 | FlashMLA workspace skipped under DCP | 3,585 | 2.88-2.97 | n/a |
| ws2 | + indexer workspace x min(40, max_num_seqs) + chunked scale exponent | **3,585** | **4.87-4.94** | 85.1 |

(ws1 = ws2's code but the AOT compile cache reloaded the old indexer size;
ws2 ran with VLLM_DISABLE_COMPILE_CACHE=1.) ws2 keeps ~2.3 GiB more free than
prod on top of +54 hot: lowering RESERVE_GB by ~2.2 should put it into ~110
more hot experts (~3,690/rank). The 130K-token request still prefills and
decodes (3.2 GiB free after it). Not yet done: reserve sweep with the 388K
stress, GSM8K / acceptance check, NCCL communicator dedupe.

At 130K context, unprofiled (nsys): 22.95 ms/step (graph 21.04 + eager 1.60
+ idle 0.30) vs 22.41 at ~5K; host never late.
