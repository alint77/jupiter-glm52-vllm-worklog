# Verify step: the 4-layer cycle on prod with the glue (2026-10-24)

The trace is prod as the c=4 launcher serves it: the main tree, vllm 23d2796ed5 plus uncommitted changes, with VLLM_MLA_QKV_GLUE=1.
- Hold 2489367, `../2026-10-18-mla-glue/launch_prodnsys.sh`, `node_run2.sh dec4`.
- Four requests decode at once, so verify steps are M=16. There are two nsys windows: w1 at ~5K context and w2 at ~50K.
- Reports are in `/e/fscratch/profound/naeimitabiei1/sliced-prof/prodg-dec4-2489367/`. They were exported to sqlite with Nsight Systems 2025.5.1, because the CUDA/13 nsys is too old to read them.

How the analysis works:
- Only verify graphs are counted: 1879 kernels each, 332 per window over 4 GPUs.
- `cycle_nsys.py` splits a graph into layers. A layer starts at the all-reduce + RMSNorm that follows the previous layer's finalize_kernel.
- An anchor is a layer with `sm90_fp8_paged_mqa_logits`. The anchors are layers 0, 1, 2, 6, 10, …, 74, as in `indexer_types`.
- A cycle is A S1 S2 S3, i.e. layers 6+4i to 9+4i: 18 cycles.
- Per kernel, the script gives median start, duration, idle before it, "adds" (its end past every earlier kernel's end) and exclusive time.
- The skip-KV stager (mark/compact/remap/gather_rows) runs on a side branch, so it gets no adds.
- `edges_nsys.py` prints the predecessor → successor gaps. `cycle_groups.py` sums the cycle by op group.
- Outputs are in `an/`.

## Result (~50K, w2; ~5K, w1)

| | ~50K | ~5K |
|---|--:|--:|
| verify graph span | 22.36 ms | 21.98 ms |
| 4-layer cycle (median) | 1131 us (p10 1038, p90 1273) | 1125 us |
| anchor layer A | 344 us | 328 us |
| skip layers S1 / S2 / S3 | 258 / 256 / 259 us | 262 / 261 / 260 us |
| prologue (dense layers 0-2 + layer 3) | 966 us | 927 us |

Skip-layer chain at ~50K (S2, us; adds sum to the wall):

| kernel | grid | dur | adds |
|---|---|--:|--:|
| all-reduce + residual + RMSNorm (FlashInfer lamport) | 64 | 9.5 | 9.5 |
| qkv_a decode_gemm (N 2624) | 164 | 14.2 | 14.0 |
| qkv_glue (q_a/kv_a norms, k RoPE, KV write) | 16 | 4.1 | 3.4 |
| q_b decode_gemm (N 4096) | 256 | 10.7 | 7.5 |
| q RoPE (Inductor triton_poi) | 128 | 1.6 | 1.6 |
| W_UK bmm (nvjet) | 2x64 | 3.8 | 3.8 |
| DCP q gather (one_shot gather_cat) | 36 | 9.0 | 9.0 |
| FlashMLA sparse decode | 1x16x8 | 14.0 | 14.0 |
| FlashMLA combine (PDL) | 1x16x8 | 6.5 | 3.1 |
| DCP LSE reduce-scatter | 32 | 7.9 | 7.9 |
| W_UV bmm (nvjet) | 2x64 | 4.5 | 4.5 |
| o_proj decode_gemm (N 6144) | 384 | 17.9 | 17.8 |
| all-reduce + residual + RMSNorm | 64 | 7.4 | 6.8 |
| router GEMM (cute DSL split-K) | 1x16x6 | 4.0 | 3.8 |
| grouped_topk | 16 | 4.1 | 3.9 |
| route_prep | 17 | 5.6 | 3.9 |
| MoE layer_kernel | 132 | 139.7 | 136.2 |
| finalize (fp32 → bf16, zero) | 6x16 | 8.7 | 2.5 |

The anchor runs the same chain. Between the q RoPE and W_UK it inserts the indexer, all of it serialized on the critical path (~50K):
- wk split-K 5.1, splitKreduce 1.5, k LayerNorm 1.7;
- wq_b decode_gemm 8.2, q rope/quant 1.9, triton_poi 1.3, k quant/cache 2.1;
- logits 6.5, cooperative top-k 13.2 (2.0 at 5K), pack 1.7, candidate gather 10.2, StableTopK 15.0;
- 2× concat_and_cache 4.1, index convert 4.5.

After FlashMLA it adds:
- the hot-cache `map_kernel`: 11.2 long, 2.9 exposed before FlashMLA;
- the skip-KV stager on a side branch: `_gather_rows` 62 us at 50K (37 at 5K), overlapping o_proj → MoE.

Full tables are in `an/cycle-2489367-w{1,2}.txt`.

Cycle by group (`an/groups-2489367.txt`, ~50K):

| group | us / cycle | share |
|---|--:|--:|
| MoE layer_kernel + finalize | 561 | 51% |
| qkv_a + glue + q_b | 99 | 9% |
| W_UV + o_proj | 92 | 8% |
| FlashMLA + combine | 69 | 6% |
| 2x all-reduce + RMSNorm | 65 | 6% |
| MoE head (router, top-k, route_prep) | 47 | 4% |
| indexer logits + top-k + DCP merge | 47 | 4% (33 at 5K) |
| DCP q gather | 36 | 3% |
| DCP LSE reduce-scatter | 33 | 3% |
| q RoPE + W_UK | 20 | 2% |
| indexer k and q branches | 23 | 2% |
| anchor index convert + hot-cache map, KV writes | 14 | 1% |

Edges (`an/edges-2489367-w*.txt`) fall into two kinds:
- Overlapped. The successor starts before its predecessor ends: qkv_a→glue→q_b, FlashMLA→combine, o_proj→AR, top-k→route_prep→MoE, MoE→finalize, plus AR→qkv_a and W_UV→o_proj, which overlap by under 0.4 us.
- Not overlapped: the remaining edges, at +0.1 to +0.2 us median with p90 about +0.45.

The same edges measured +0.42 to +0.48 median in the 10-23 trace (mg5g-dec4-2479208, glue-ab2). The pure launch-gap total is therefore ~0.1-0.35 ms/step, depending on the run.
