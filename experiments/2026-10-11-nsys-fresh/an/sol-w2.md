window 2: per concurrency, us/call (min over ranks) and ms/step; SOL us/call; gap ms/step

### c=1: period 19.81 ms, outside the verify graph 1.59, idle 0.52
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 104.7 (103.1) | 7.86 | 61.3 | 59 | 3.26 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.0 | 4.27 | (2.48 ms) | 58 | 1.80 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 151 | 7.3 (5.6) | 1.10 | 0.7 | 9 | 1.00 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 10.9 (10.7) | 0.85 | 1.4 | 12 | 0.74 | 8 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 7.5 (6.3) | 0.59 | 1.0 | 13 | 0.51 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 6.8 (6.3) | 0.53 | 0.9 | 13 | 0.46 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.0 (1.8) | 0.15 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.6) | 0.71 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.6 (3.5) | 0.28 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 5.7 (5.3) | 1.07 |  |  |  |  |
| KV write | 135 | 1.8 (1.7) | 0.25 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 7.9 (7.5) | 0.60 |  |  |  |  |
| other verify-graph kernels | 293 | 1.8 (1.6) | 0.54 |  |  |  |  |

### c=2: period 20.26 ms, outside the verify graph 1.98, idle 0.60
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 104.0 (102.3) | 7.80 | 61.0 | 59 | 3.23 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.0 | 4.28 | (2.48 ms) | 58 | 1.80 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 7.5 (5.7) | 1.18 | 0.7 | 9 | 1.07 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 11.0 (10.7) | 0.86 | 1.4 | 12 | 0.75 | 8 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 7.5 (6.4) | 0.58 | 1.0 | 14 | 0.50 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 6.7 (6.3) | 0.53 | 0.9 | 13 | 0.46 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.0 (1.8) | 0.15 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.5) | 0.70 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.6 (3.5) | 0.28 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 5.7 (5.4) | 1.08 |  |  |  |  |
| KV write | 135 | 1.8 (1.7) | 0.24 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 9.0 (8.5) | 0.68 |  |  |  |  |
| other verify-graph kernels | 281 | 1.6 (1.5) | 0.45 |  |  |  |  |

### c=4: period 25.64 ms, outside the verify graph 2.22, idle 0.50
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 159.4 (156.8) | 11.96 | 119.0 | 75 | 3.03 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.6 | 4.58 | (2.48 ms) | 54 | 2.10 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 8.9 (6.5) | 1.40 | 1.4 | 15 | 1.18 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 13.8 (13.6) | 1.08 | 2.7 | 20 | 0.86 | 16 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 9.4 (7.9) | 0.73 | 2.0 | 22 | 0.57 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 8.0 (7.5) | 0.62 | 1.8 | 23 | 0.48 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.6 (2.2) | 0.19 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.8 (4.6) | 0.72 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.2 (3.0) | 0.25 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 6.4 (6.0) | 1.21 |  |  |  |  |
| KV write | 135 | 1.9 (1.8) | 0.25 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 17.5 (15.7) | 1.33 |  |  |  |  |
| other verify-graph kernels | 281 | 1.6 (1.5) | 0.46 |  |  |  |  |

### c=8: period 38.58 ms, outside the verify graph 5.14, idle 0.87
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| dense GEMMs | 687 | 7.8 | 5.37 | (2.48 ms) | 46 | 2.90 | weights 8.92 GB/step / HBM |
| MoE layer_kernel | 75 | 254.0 (249.7) | 19.05 | 221.0 | 87 | 2.47 | max(hot/HBM, cold/C2C) |
| AR + RMSNorm | 157 | 12.3 (8.5) | 1.93 | 2.7 | 22 | 1.50 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 18.5 (18.1) | 1.45 | 5.4 | 29 | 1.02 | 32 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 12.4 (11.0) | 0.97 | 4.1 | 33 | 0.65 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 10.9 (10.2) | 0.85 | 3.6 | 33 | 0.57 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 3.3 (2.9) | 0.24 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.9 (4.7) | 0.74 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 4.1 (4.0) | 0.32 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 7.8 (7.2) | 1.47 |  |  |  |  |
| KV write | 135 | 2.0 (1.9) | 0.27 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 31.9 (26.3) | 2.42 |  |  |  |  |
| other verify-graph kernels | 281 | 1.8 (1.7) | 0.50 |  |  |  |  |

### c=16: period 133.47 ms, outside the verify graph 88.51, idle 6.68
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| dense GEMMs | 651 | 8.4 | 5.50 | (2.48 ms) | 45 | 3.02 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 15.7 (10.8) | 2.47 | 5.5 | 35 | 1.60 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 22.4 (21.9) | 1.75 | 10.9 | 48 | 0.90 | 64 tok x 64 h x 768 keys |
| DCP lse_reduce_scatter | 78 | 15.7 (13.6) | 1.22 | 7.2 | 46 | 0.66 | 3.1 MB x T/64 over NVLink |
| DCP gather_cat | 78 | 15.9 (14.7) | 1.24 | 8.1 | 51 | 0.60 | 3.5 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 3.9 (3.5) | 0.29 |  |  |  |  |
| MoE route_prep + topk | 150 | 5.1 (4.9) | 0.76 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 4.0 (3.9) | 0.31 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 9.3 (8.7) | 1.76 |  |  |  |  |
| KV write | 135 | 2.0 (1.9) | 0.27 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 61.4 (53.8) | 4.67 |  |  |  |  |
| other verify-graph kernels | 281 | 1.9 (1.8) | 0.54 |  |  |  |  |
| MoE layer_kernel | 75 | 304.5 (300.1) | 22.84 | 378.0 | 124 | -5.51 | max(hot/HBM, cold/C2C) |
