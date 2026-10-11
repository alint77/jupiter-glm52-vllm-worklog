window 1: per concurrency, us/call (min over ranks) and ms/step; SOL us/call; gap ms/step

### c=1: period 18.09 ms, outside the verify graph 1.59, idle 0.53
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 85.3 (83.5) | 6.40 | 61.3 | 72 | 1.80 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.0 | 4.26 | (2.48 ms) | 58 | 1.79 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 151 | 7.3 (5.6) | 1.11 | 0.7 | 9 | 1.01 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 10.7 (10.5) | 0.83 | 1.4 | 13 | 0.73 | 8 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 7.5 (6.4) | 0.58 | 1.0 | 14 | 0.50 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 6.8 (6.3) | 0.53 | 0.9 | 13 | 0.46 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.0 (1.8) | 0.15 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.6) | 0.70 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.6 (3.4) | 0.28 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 4.1 (3.8) | 0.78 |  |  |  |  |
| KV write | 135 | 1.8 (1.7) | 0.25 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 6.7 (6.4) | 0.51 |  |  |  |  |
| other verify-graph kernels | 293 | 1.9 (1.6) | 0.54 |  |  |  |  |

### c=2: period 19.29 ms, outside the verify graph 1.95, idle 0.62
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 96.1 (94.4) | 7.21 | 61.0 | 63 | 2.63 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.0 | 4.27 | (2.48 ms) | 58 | 1.79 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 7.3 (5.7) | 1.15 | 0.7 | 9 | 1.04 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 10.8 (10.6) | 0.85 | 1.4 | 13 | 0.74 | 8 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 7.3 (6.4) | 0.57 | 1.0 | 14 | 0.49 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 6.7 (6.2) | 0.53 | 0.9 | 13 | 0.46 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.1 (1.8) | 0.15 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.5) | 0.70 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.6 (3.5) | 0.28 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 4.1 (3.8) | 0.78 |  |  |  |  |
| KV write | 135 | 1.8 (1.7) | 0.24 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 7.7 (7.4) | 0.59 |  |  |  |  |
| other verify-graph kernels | 281 | 1.6 (1.5) | 0.45 |  |  |  |  |

### c=4: period 25.64 ms, outside the verify graph 2.20, idle 0.69
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 156.9 (154.4) | 11.77 | 119.0 | 76 | 2.85 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.6 | 4.59 | (2.48 ms) | 54 | 2.11 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 11.2 (6.5) | 1.77 | 1.4 | 12 | 1.55 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 13.5 (13.3) | 1.06 | 2.7 | 20 | 0.84 | 16 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 9.0 (8.0) | 0.70 | 2.0 | 23 | 0.54 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 8.0 (7.5) | 0.62 | 1.8 | 23 | 0.48 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.6 (2.3) | 0.20 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.6) | 0.71 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.2 (3.0) | 0.25 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 4.5 (4.3) | 0.86 |  |  |  |  |
| KV write | 135 | 1.9 (1.8) | 0.25 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 12.5 (11.3) | 0.95 |  |  |  |  |
| other verify-graph kernels | 281 | 1.6 (1.5) | 0.46 |  |  |  |  |

### c=8: period 35.98 ms, outside the verify graph 2.58, idle 0.72
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 261.9 (257.5) | 19.64 | 221.0 | 84 | 3.07 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 687 | 7.7 | 5.28 | (2.48 ms) | 47 | 2.80 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 12.0 (8.5) | 1.88 | 2.7 | 23 | 1.45 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 18.2 (17.8) | 1.42 | 5.4 | 30 | 1.00 | 32 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 12.3 (11.0) | 0.96 | 4.1 | 33 | 0.64 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 10.9 (10.1) | 0.85 | 3.6 | 33 | 0.57 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 3.3 (2.9) | 0.25 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.8 (4.7) | 0.72 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 4.2 (4.0) | 0.32 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 5.7 (5.1) | 1.08 |  |  |  |  |
| KV write | 135 | 2.0 (1.9) | 0.27 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 19.5 (15.3) | 1.48 |  |  |  |  |
| other verify-graph kernels | 281 | 1.8 (1.7) | 0.50 |  |  |  |  |

### c=16: period 99.30 ms, outside the verify graph 55.21, idle 4.29
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| dense GEMMs | 636 | 8.7 | 5.53 | (2.48 ms) | 45 | 3.05 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 14.8 (11.3) | 2.32 | 5.5 | 37 | 1.46 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 23.2 (22.7) | 1.81 | 10.9 | 47 | 0.96 | 64 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 16.7 (15.7) | 1.30 | 8.1 | 49 | 0.67 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 15.1 (14.2) | 1.18 | 7.2 | 48 | 0.62 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 4.1 (3.7) | 0.31 |  |  |  |  |
| MoE route_prep + topk | 150 | 5.1 (5.0) | 0.77 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 4.1 (3.9) | 0.32 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 6.9 (6.5) | 1.31 |  |  |  |  |
| KV write | 135 | 2.1 (2.0) | 0.28 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 49.0 (41.9) | 3.73 |  |  |  |  |
| other verify-graph kernels | 281 | 2.0 (1.9) | 0.55 |  |  |  |  |
| MoE layer_kernel | 75 | 330.0 (325.1) | 24.75 | 378.0 | 115 | -3.60 | max(hot/HBM, cold/C2C) |
