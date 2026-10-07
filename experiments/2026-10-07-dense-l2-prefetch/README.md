# Dense decode GEMMs: weights into L2 ahead of use (2026-10-07)

At M=8 the verify step's bf16 GEMMs only stream weights, and cuBLAS (64-96 of
132 SMs) does not keep enough bytes in flight: o_proj 48 MiB in ~21 us, against
15 us for the floor and 15.2 us with the weight already in L2 (60 MiB on
GH200; `../2026-10-07-decode-mem-dive/l2_prefetch_probe.py`). The idea: keep
cuBLAS and have the weight in L2 when it runs, by prefetching during a
latency-bound stretch before it. Same math, outputs identical.

`vllm/model_executor/layers/l2_prefetch` (JIT, sm_90a): `bulk_prefetch` hands
64 KiB chunks to the TMA unit (`cp.async.bulk.prefetch.L2`, SASS `UBLKPF.L2`)
with an evict_normal / evict_last policy; `line_prefetch` issues one
`prefetch.global.L2` per 128 B line.

## Microbenchmark (`window_bench.py`, node 2219434, CUDA graphs, L2 flushed
per replay, kernel times from profiler device timestamps)

Window A replays attention before o_proj: FlashMLA-like 18 us idle + a 17 MB
split-accumulator write and read (the split combine), 11 us (DCP
reduce-scatter + W_UV), o_proj. Clean run (`window-bench5`), evict_normal,
join after o_proj:

| prefetch | o_proj | combine read | window to o_proj end |
|---|--:|--:|--:|
| none | 21.4 us | 11.7 | 90.1 |
| at FlashMLA start | 21.0-21.4 | 11.6-12.2 | 91.6-92.6 |
| **after the combine, 32 CTAs** | **15.4** | 11.65 | **86.3** |
| after the combine, 132 CTAs | 15.1 | 11.65 | 86.6 |

- Prefetched at FlashMLA start, the lines are gone (the combine's traffic)
  before o_proj reads them.
- After the combine the bandwidth is idle until o_proj: -6 us on o_proj,
  -3.9 us on the window, ~-0.3 ms/step over 78 layers.
- A CTA issuing bulk prefetches stays resident until they drain: with 8 CTAs
  the kernel ran 37-46 us (~1.2 TB/s) and joining before o_proj stretched the
  window to 112 us. 32 CTAs: ~11 us. The join must not come before the GEMM.
- Earlier evict_last numbers (`window-bench2-4`) were inflated: those lines
  survive the flush between replays.

Window B (qkv_a and q_b of the next layer, prefetched during router / top-k /
route prep and the MoE): with an HBM-bound MoE (160 MB streamed) any prefetch
of qkv_a slows the MoE by 4-6 us, more than it saves; with a C2C-bound MoE
(HBM idle) qkv_a 18.3 -> 12.7, q_b 7.6 -> 6.2. The step follows the slowest
rank, which has no all-reduce wait to hide a prefetch in, so window B pays
only on layers where that rank is cold-bound; not pursued for now.

## In the model (VLLM_DENSE_L2_PREFETCH=1)

`MLAAttention.forward_impl` forks `prefetch_on_side_stream(o_proj.weight)`
right after `forward_mqa` (FlashMLA + split combine); the tiered MoE op
rejoins the side stream at its entry, after o_proj. A fork from eager code is
not joined from inside a capture (piecewise graphs run attention eagerly).

Measured in the model (`segments.py`, rank 0, layers 3-76, profiled agentic +
50K/130K decode; all three runs with the new memory config, reserve 1.7):

| median us | off | mode 1 (o_proj) | mode 2 (W_UV, then o_proj) |
|---|--:|--:|--:|
| FlashMLA start -> o_proj start | 34.3 | 39.8 | 35.4 |
| DCP LSE reduce-scatter | 6.3 | 7.3 | 6.7 |
| W_UV | 3.9 | 6.8 | 3.6 |
| o_proj | 20.9 | **15.8** | 22.4 |
| layer | 254.8 | 255.8 | 257.4 |

The ~12 us between the split combine and o_proj is too short to hide a 48 MiB
fetch: mode 1 makes o_proj 5 us faster but W_UV and the reduce-scatter queue
behind the prefetch for the same 5 us; mode 2 keeps W_UV fast but the o_proj
lines no longer arrive in time. No layer-time gain either way: stopped here
(code kept behind VLLM_DENSE_L2_PREFETCH, default off). What remains for these
GEMMs is a kernel that keeps more bytes in flight than cuBLAS's 64-96 CTAs.
