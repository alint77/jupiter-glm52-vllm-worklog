window 2: per concurrency, us/call (min over ranks) and ms/step; SOL us/call; gap ms/step

### c=1: period 18.83 ms, outside the verify graph 1.78, idle 0.58
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 87.0 (85.1) | 6.53 | 61.3 | 70 | 1.93 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.0 | 4.28 | (2.48 ms) | 58 | 1.80 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 151 | 7.7 (5.6) | 1.17 | 0.7 | 9 | 1.06 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 10.9 (10.7) | 0.85 | 1.4 | 12 | 0.75 | 8 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 7.6 (6.4) | 0.60 | 1.0 | 13 | 0.52 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 6.8 (6.3) | 0.53 | 0.9 | 13 | 0.46 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.0 (1.8) | 0.15 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.6) | 0.71 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.6 (3.4) | 0.28 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 5.7 (5.4) | 1.08 |  |  |  |  |
| KV write | 135 | 1.8 (1.7) | 0.24 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 7.5 (7.2) | 0.57 |  |  |  |  |
| other verify-graph kernels | 293 | 1.8 (1.6) | 0.53 |  |  |  |  |

### c=2: period 20.11 ms, outside the verify graph 1.99, idle 0.58
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 102.3 (100.7) | 7.67 | 61.0 | 60 | 3.10 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.0 | 4.28 | (2.48 ms) | 58 | 1.80 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 7.4 (5.6) | 1.17 | 0.7 | 9 | 1.06 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 10.9 (10.8) | 0.85 | 1.4 | 12 | 0.75 | 8 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 7.5 (6.5) | 0.59 | 1.0 | 14 | 0.51 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 6.8 (6.3) | 0.53 | 0.9 | 13 | 0.46 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.0 (1.8) | 0.15 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.5) | 0.70 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.6 (3.5) | 0.28 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 5.7 (5.4) | 1.08 |  |  |  |  |
| KV write | 135 | 1.8 (1.7) | 0.24 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 8.9 (8.5) | 0.68 |  |  |  |  |
| other verify-graph kernels | 281 | 1.6 (1.5) | 0.44 |  |  |  |  |

### c=4: period 25.92 ms, outside the verify graph 2.27, idle 0.59
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 160.4 (158.2) | 12.03 | 119.0 | 74 | 3.11 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.6 | 4.59 | (2.48 ms) | 54 | 2.11 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 9.0 (6.4) | 1.41 | 1.4 | 15 | 1.19 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 13.8 (13.6) | 1.08 | 2.7 | 20 | 0.86 | 16 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 9.4 (7.9) | 0.74 | 2.0 | 22 | 0.58 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 8.1 (7.5) | 0.63 | 1.8 | 22 | 0.49 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.6 (2.3) | 0.19 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.8 (4.6) | 0.72 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.1 (2.9) | 0.24 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 6.6 (6.0) | 1.25 |  |  |  |  |
| KV write | 135 | 1.9 (1.8) | 0.25 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 17.7 (15.4) | 1.35 |  |  |  |  |
| other verify-graph kernels | 281 | 1.6 (1.5) | 0.46 |  |  |  |  |

### c=8: period 36.69 ms, outside the verify graph 2.62, idle 0.62
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 268.7 (264.8) | 20.15 | 221.0 | 82 | 3.57 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 687 | 7.7 | 5.31 | (2.48 ms) | 47 | 2.83 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 11.6 (8.6) | 1.83 | 2.7 | 24 | 1.40 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 18.4 (18.0) | 1.43 | 5.4 | 30 | 1.01 | 32 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 12.3 (10.9) | 0.96 | 4.1 | 33 | 0.65 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 10.8 (10.1) | 0.84 | 3.6 | 33 | 0.56 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 3.3 (2.9) | 0.25 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.9 (4.7) | 0.73 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 4.1 (4.0) | 0.32 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 7.6 (7.1) | 1.43 |  |  |  |  |
| KV write | 135 | 2.0 (1.9) | 0.27 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 33.7 (29.1) | 2.56 |  |  |  |  |
| other verify-graph kernels | 281 | 1.7 (1.6) | 0.48 |  |  |  |  |

### c=16: period 49.83 ms, outside the verify graph 3.31, idle 0.88
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| dense GEMMs | 609 | 9.1 | 5.54 | (2.48 ms) | 45 | 3.07 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 16.9 (12.2) | 2.66 | 5.5 | 32 | 1.80 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 24.6 (23.9) | 1.92 | 10.9 | 44 | 1.07 | 64 tok x 64 h x 768 keys |
| DCP lse_reduce_scatter | 78 | 18.3 (15.0) | 1.42 | 7.2 | 39 | 0.86 | 3.1 MB x T/64 over NVLink |
| DCP gather_cat | 78 | 18.0 (16.9) | 1.40 | 8.1 | 45 | 0.77 | 3.5 MB x T/64 over NVLink |
| MoE layer_kernel | 75 | 386.5 (380.9) | 28.99 | 378.0 | 98 | 0.64 | max(hot/HBM, cold/C2C) |
| MoE finalize (tail) | 75 | 4.4 (3.9) | 0.33 |  |  |  |  |
| MoE route_prep + topk | 150 | 5.3 (5.0) | 0.79 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 4.1 (4.0) | 0.32 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 10.1 (9.5) | 1.90 |  |  |  |  |
| KV write | 135 | 2.1 (2.0) | 0.28 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 88.6 (75.3) | 6.73 |  |  |  |  |
| other verify-graph kernels | 281 | 2.0 (1.9) | 0.57 |  |  |  |  |
