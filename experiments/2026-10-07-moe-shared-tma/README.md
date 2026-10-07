# Shared expert inside the MoE kernel? And does TMA cap the MoE's hot loads? (2026-10-07)

## Shared expert: serialized vs overlapped (`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`)

Profiled agentic + 50K/130K decode, prod config (reserve 1.7, embedding on
Grace), `../2026-10-07-dense-l2-prefetch/segments.py`, rank 0, median us:

| segment | overlapped (prod) | serialized |
|---|--:|--:|
| attn AR end -> route_prep (the shared expert runs here when serial) | 7.9 | 22.9 |
| MoE chain | 105.6 | 103.8 |
| layer | 254.8 | 269.6 |

The shared expert costs ~15 us alone; the side stream hides ~13 us of it and
slows the MoE by ~1.8 us. Folding it into the tiered MoE kernel could save at
most ~2 us per layer (~0.15 ms/step), and its 19 MB would join the hot load on
hot-bound ranks. Not built.

## TMA vs plain loads (`tma_map_probe.py`, `bw_sweep.py`; node 2220517)

The MoE's weight load copied exactly (3D tensor map, 32 KiB boxes, no swizzle,
4 stages, one producer, 1 CTA/SM) vs 16 B loads (132 x 512 threads):

| | fixed | sustained |
|---|--:|--:|
| plain 16 B loads | 3.3 us | 3.49 TB/s |
| TMA tensor map | 8.0 us | 3.65 TB/s |

TMA is not bandwidth-capped (1 GB: 3.57 vs 3.46 TB/s); it ramps ~4.6 us slower,
which dominates 25-50 MB reads (25 MB: 12.3 vs 9.95 us; 101 MB: 38.3 vs 32.1).
Producer warps (1/2/4) and stages (4/6) change nothing. For the MoE: ~7 us on
w13 and ~3.6 us on w2 for an 8-hot-expert rank, paid only where the slowest
rank is hot-bound (w2 may already hide part of it, prefetching during the
activation kernel). Sizing it needs the real kernel on a (hot, cold) grid.

The earlier "TMA tops out near 2.4 TB/s" (`../2026-10-07-skinny-gemm-v2`) was
this ramp read as a ceiling on 16-50 MB transfers.
