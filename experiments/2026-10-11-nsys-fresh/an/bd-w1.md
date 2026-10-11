| category | c=1 | c=2 | c=4 | c=8 | c=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **step period** | **18.09** | **19.29** | **25.64** | **35.98** | **99.30** |
| outside the verify graph (draft, lm_head, sampling, prep) | 1.59 | 1.95 | 2.20 | 2.58 | 55.21 |
| MoE expert GEMMs (sliced: + shared expert) | 6.40 | 7.20 | 11.76 | 19.59 | 24.54 |
| dense GEMMs (cuBLAS etc.) | 1.01 | 1.02 | 1.08 | 3.47 | 5.42 |
| GPU idle | 0.53 | 0.62 | 0.69 | 0.72 | 4.29 |
| DCP collectives | 1.30 | 1.28 | 1.55 | 2.21 | 3.06 |
| all-reduce + RMSNorm (incl. wait) | 1.14 | 1.15 | 1.77 | 1.88 | 2.30 |
| attention (FlashMLA) | 0.83 | 0.85 | 1.06 | 1.42 | 1.80 |
| MoE route/act/finalize + router | 0.49 | 0.50 | 0.54 | 0.60 | 0.68 |
| DSA indexer | 0.57 | 0.58 | 0.60 | 0.66 | 0.66 |
| other verify-graph kernels | 0.50 | 0.45 | 0.46 | 0.51 | 0.59 |
| KV write / skip-KV staging | 0.26 | 0.26 | 0.27 | 0.30 | 0.33 |
| attention (split-KV combine) | 0.28 | 0.28 | 0.25 | 0.32 | 0.32 |
| dense GEMMs (decode_gemm) | 3.17 | 3.17 | 3.42 | 1.71 | 0.09 |
