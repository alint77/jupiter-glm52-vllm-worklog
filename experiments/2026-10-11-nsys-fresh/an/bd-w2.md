| category | c=1 | c=2 | c=4 | c=8 | c=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **step period** | **19.81** | **20.26** | **25.64** | **38.58** | **133.47** |
| outside the verify graph (draft, lm_head, sampling, prep) | 1.59 | 1.98 | 2.22 | 5.14 | 88.51 |
| MoE expert GEMMs (sliced: + shared expert) | 7.84 | 7.81 | 11.95 | 18.95 | 22.57 |
| GPU idle | 0.52 | 0.60 | 0.50 | 0.87 | 6.68 |
| dense GEMMs (cuBLAS etc.) | 1.01 | 1.02 | 1.07 | 3.47 | 5.20 |
| DCP collectives | 1.30 | 1.29 | 1.59 | 2.20 | 2.99 |
| all-reduce + RMSNorm (incl. wait) | 1.14 | 1.17 | 1.39 | 1.92 | 2.47 |
| attention (FlashMLA) | 0.85 | 0.86 | 1.08 | 1.45 | 1.74 |
| DSA indexer | 0.87 | 0.88 | 0.95 | 1.06 | 0.94 |
| other verify-graph kernels | 0.50 | 0.45 | 0.46 | 0.51 | 0.77 |
| MoE route/act/finalize + router | 0.49 | 0.50 | 0.54 | 0.60 | 0.66 |
| KV write / skip-KV staging | 0.26 | 0.26 | 0.27 | 0.30 | 0.36 |
| attention (split-KV combine) | 0.28 | 0.28 | 0.25 | 0.32 | 0.31 |
| dense GEMMs (decode_gemm) | 3.17 | 3.17 | 3.38 | 1.78 | 0.26 |
