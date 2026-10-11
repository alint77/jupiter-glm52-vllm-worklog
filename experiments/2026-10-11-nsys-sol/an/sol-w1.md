window 1: per concurrency, us/call (min over ranks) and ms/step; SOL us/call; gap ms/step

### c=1: period 18.14 ms, outside the verify graph 1.59, idle 0.54
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 85.9 (84.0) | 6.44 | 61.3 | 71 | 1.85 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.0 | 4.26 | (2.48 ms) | 58 | 1.78 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 151 | 7.4 (5.6) | 1.12 | 0.7 | 9 | 1.01 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 10.8 (10.5) | 0.84 | 1.4 | 13 | 0.73 | 8 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 7.4 (6.4) | 0.58 | 1.0 | 14 | 0.50 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 6.8 (6.3) | 0.53 | 0.9 | 13 | 0.46 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.0 (1.8) | 0.15 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.5) | 0.70 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.5 (3.4) | 0.28 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 4.2 (3.8) | 0.79 |  |  |  |  |
| KV write | 135 | 1.8 (1.7) | 0.24 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 6.7 (6.4) | 0.51 |  |  |  |  |
| other verify-graph kernels | 293 | 1.8 (1.6) | 0.53 |  |  |  |  |

### c=2: period 19.42 ms, outside the verify graph 1.95, idle 0.60
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 98.4 (96.8) | 7.38 | 61.0 | 62 | 2.80 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.0 | 4.27 | (2.48 ms) | 58 | 1.79 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 7.2 (5.6) | 1.13 | 0.7 | 9 | 1.03 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 10.8 (10.6) | 0.84 | 1.4 | 13 | 0.74 | 8 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 7.4 (6.5) | 0.58 | 1.0 | 14 | 0.50 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 6.8 (6.3) | 0.53 | 0.9 | 13 | 0.46 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.0 (1.8) | 0.15 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.7 (4.5) | 0.70 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.6 (3.4) | 0.28 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 4.1 (3.8) | 0.78 |  |  |  |  |
| KV write | 135 | 1.8 (1.7) | 0.24 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 7.8 (7.4) | 0.59 |  |  |  |  |
| other verify-graph kernels | 281 | 1.6 (1.5) | 0.44 |  |  |  |  |

### c=4: period 25.21 ms, outside the verify graph 2.24, idle 0.65
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 154.9 (152.5) | 11.61 | 119.0 | 77 | 2.69 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 534 | 8.6 | 4.59 | (2.48 ms) | 54 | 2.11 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 8.9 (6.4) | 1.39 | 1.4 | 15 | 1.18 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 13.6 (13.3) | 1.06 | 2.7 | 20 | 0.85 | 16 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 9.1 (8.0) | 0.71 | 2.0 | 22 | 0.55 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 8.2 (7.6) | 0.64 | 1.8 | 22 | 0.50 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 2.7 (2.3) | 0.20 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.8 (4.6) | 0.72 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 3.2 (3.0) | 0.25 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 4.8 (4.2) | 0.90 |  |  |  |  |
| KV write | 135 | 1.9 (1.8) | 0.25 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 11.9 (10.7) | 0.91 |  |  |  |  |
| other verify-graph kernels | 281 | 1.6 (1.6) | 0.46 |  |  |  |  |

### c=8: period 36.28 ms, outside the verify graph 2.60, idle 0.63
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| MoE layer_kernel | 75 | 270.4 (266.7) | 20.28 | 221.0 | 82 | 3.71 | max(hot/HBM, cold/C2C) |
| dense GEMMs | 687 | 7.6 | 5.22 | (2.48 ms) | 48 | 2.74 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 11.3 (8.5) | 1.77 | 2.7 | 24 | 1.34 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 18.0 (17.6) | 1.41 | 5.4 | 30 | 0.98 | 32 tok x 64 h x 768 keys |
| DCP gather_cat | 78 | 12.3 (10.9) | 0.96 | 4.1 | 33 | 0.64 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 10.8 (10.1) | 0.84 | 3.6 | 33 | 0.56 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 3.3 (2.9) | 0.25 |  |  |  |  |
| MoE route_prep + topk | 150 | 4.8 (4.6) | 0.72 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 4.1 (4.0) | 0.32 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 5.4 (5.1) | 1.02 |  |  |  |  |
| KV write | 135 | 2.0 (1.9) | 0.26 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 20.5 (16.1) | 1.56 |  |  |  |  |
| other verify-graph kernels | 281 | 1.7 (1.6) | 0.48 |  |  |  |  |

### c=16: period 49.14 ms, outside the verify graph 3.25, idle 0.98
| kernel group | calls/step | us/call (min) | ms/step | SOL us/call | % SOL | gap ms/step | model |
|---|--:|--:|--:|--:|--:|--:|---|
| dense GEMMs | 609 | 9.1 | 5.53 | (2.48 ms) | 45 | 3.05 | weights 8.92 GB/step / HBM |
| AR + RMSNorm | 157 | 16.2 (12.1) | 2.55 | 5.5 | 34 | 1.69 | 2.36 MB x T/64 over NVLink |
| FlashMLA sparse | 78 | 24.1 (23.5) | 1.88 | 10.9 | 45 | 1.03 | 64 tok x 64 h x 768 keys |
| MoE layer_kernel | 75 | 388.3 (382.4) | 29.12 | 378.0 | 97 | 0.77 | max(hot/HBM, cold/C2C) |
| DCP gather_cat | 78 | 17.9 (16.9) | 1.40 | 8.1 | 45 | 0.76 | 3.5 MB x T/64 over NVLink |
| DCP lse_reduce_scatter | 78 | 16.2 (15.0) | 1.26 | 7.2 | 44 | 0.70 | 3.1 MB x T/64 over NVLink |
| MoE finalize (tail) | 75 | 4.4 (3.9) | 0.33 |  |  |  |  |
| MoE route_prep + topk | 150 | 5.2 (5.0) | 0.78 |  |  |  |  |
| FlashMLA combine (tail) | 78 | 4.1 (3.9) | 0.32 |  |  |  |  |
| indexer (gather, logits, top-k, quant) | 189 | 7.2 (6.8) | 1.35 |  |  |  |  |
| KV write | 135 | 2.1 (2.0) | 0.28 |  |  |  |  |
| skip-KV staging (side stream) | 76 | 62.4 (51.2) | 4.74 |  |  |  |  |
| other verify-graph kernels | 281 | 2.0 (1.9) | 0.57 |  |  |  |  |
