# Skinny GEMM v2: the M=8 decode GEMMs at the HBM floor (2026-10-07)

The verify step's bf16 GEMMs (8 tokens) only stream weights. Per step, at the
calls per verify step:

| GEMM (calls) | weight | floor 3.64 TB/s | cuBLAS isolated | cuBLAS in model |
|---|--:|--:|--:|--:|
| o_proj (78) | 48 MiB | 13.8 us | 17.9 | 21.0 |
| fused_qkv_a (78) | 31 MiB | 8.9 | 13.7 | ~16 (split-K + reduce) |
| q_b (78) | 16 MiB | 4.6 | 7.8 | 7.9 |
| indexer wq_b (21) | 16 MiB | 4.6 | 7.8 | 7.4 |

All at the floor: ~1.7 ms/step (2026-09-27 bench). The first custom kernel
(`vllm/model_executor/layers/skinny_gemm`, untracked: TMA ring, one producer
warp, stream-K) lost to cuBLAS (o_proj 19.9 vs 17.9 us; ncu 2.23 vs 2.66
TB/s, both ~14% occupancy). The L2-prefetch route also failed
(`../2026-10-07-dense-l2-prefetch`): no idle bandwidth window long enough.

## Plan, with gates

1. Streaming ceiling (`stream_probe.py`): TMA bulk ring into shared memory vs
   plain 16 B loads, per CTA count / chunk / depth, at the three sizes; base-
   clock ncu of cuBLAS and the old kernel to see what limited it. Gate: a pure
   stream must reach o_proj <= ~15 us, else stop.
2. GEMM on that streaming core: weights as the MMA A operand (m16n8k16, the 8
   tokens as N), in-kernel split-K reduce (cluster shared memory or last-CTA
   finish). Gate: >= 3 us under cuBLAS on o_proj and qkv_a, exact-to-cuBLAS
   tolerance.
3. Programmatic dependent launch: stream weights while the previous kernel
   finishes, wait only before reading activations; graph bench with the real
   predecessors.
4. Integrate for o_proj, fused_qkv_a, q_b, indexer wq_b; served A/B.

## Results

**Streaming ceiling** (`stream_probe.py`, 50 MB, L2 flushed): plain 16 B
loads 17.3 us; a TMA 1D-bulk ring 21.3 us at best, whatever the depth, chunk
or CTAs per SM. (`../2026-10-07-moe-shared-tma/bw_sweep.py` later showed this
is a ~4.6 us slower ramp, not a bandwidth cap: plain loads 3.3 us + MB / 3.49
TB/s, TMA 8.0 us + MB / 3.65 TB/s.) The first kernel streamed through TMA.

**Kernel iterations** (`bench_v2.py`, isolated, flushed L2):

| | o_proj | fused_qkv_a | q_b |
|---|--:|--:|--:|
| cuBLAS | 22.1-22.7 us | 16.7-17.3 | 8.9 |
| v2: 16 B loads into mma.m16n8k16, CTA per 16 rows, 16 warps split K | 22.1 | 14.9 | 8.7 |
| v3: persistent, deterministic slot reduction (one config wrong; dropped) | 23.5 | | |
| **v4: 4 warps per tile (long K per warp), register double-buffering** | **18.0** | **13.8** | **8.5** |
| v5: + split-K across CTAs | no gain | | |
| v6: + programmatic dependent launch | -0.2 to -0.5 | | |

ncu (base clock) on v2: 81% of warp time waiting on loads, 0.22 eligible
warps per scheduler, 0% excess sectors: latency, not access pattern. v4's
long per-warp K and double-buffered loads fixed it.

All verify-step linears (`bench_shapes.py`): cuBLAS 5.31 ms/step, best-of
4.53 ms (-0.77). In the model the shared expert's two GEMMs (side stream)
must stay on cuBLAS: on them the kernel fills every SM and delayed the router
(+7.7 us) and MoE (+3.9 us); with them excluded the profiled layer goes
254.8 -> 248.4 us (o_proj 20.9 -> 17.0, the pre-attention block 48.5 -> 45.6).

**Served A/B** (VLLM_DECODE_GEMM 0 vs 1, `chain_dg.sh`: 4 nodes alternating,
`../2026-10-07-mem-reclaim/arm.sh`, `KINDS=dg0,dg1 compare_ba.py`):
-0.47 +- 0.03 ms/step agentic (297 requests, 16 arms), -0.50 +- 0.04 at
50-130K (64 requests); GSM8K 0.914 -> 0.921 (8 x 200 each); TTFT and the 388K
stress unchanged (16/16).

vllm 86f871f450 (`model_executor/layers/decode_gemm`, `vllm::decode_unquantized_gemm`,
`tests/kernels/test_decode_gemm.py`); serve.sh defaults VLLM_DECODE_GEMM=1.
