| category | c=1 | c=2 | c=4 | c=8 | c=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **step period** | **18.83** | **20.11** | **25.92** | **36.69** | **49.83** |
| MoE expert GEMMs (sliced: + shared expert) | 6.53 | 7.68 | 12.01 | 20.11 | 28.78 |
| dense GEMMs (cuBLAS etc.) | 1.01 | 1.02 | 1.08 | 3.41 | 5.54 |
| DCP collectives | 1.31 | 1.30 | 1.64 | 2.16 | 3.47 |
| outside the verify graph (draft, lm_head, sampling, prep) | 1.78 | 1.99 | 2.27 | 2.62 | 3.31 |
| all-reduce + RMSNorm (incl. wait) | 1.20 | 1.16 | 1.40 | 1.82 | 2.66 |
| attention (FlashMLA) | 0.85 | 0.85 | 1.08 | 1.43 | 1.92 |
| DSA indexer | 0.88 | 0.87 | 0.95 | 1.05 | 0.96 |
| GPU idle | 0.58 | 0.58 | 0.59 | 0.62 | 0.88 |
| other verify-graph kernels | 0.49 | 0.45 | 0.46 | 0.49 | 0.84 |
| MoE route/act/finalize + router | 0.49 | 0.50 | 0.54 | 0.60 | 0.71 |
| KV write / skip-KV staging | 0.26 | 0.26 | 0.26 | 0.29 | 0.45 |
| attention (split-KV combine) | 0.28 | 0.28 | 0.24 | 0.32 | 0.32 |
| dense GEMMs (decode_gemm) | 3.18 | 3.17 | 3.40 | 1.76 | 0.00 |
