| category | c=1 | c=2 | c=4 | c=8 | c=16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **step period** | **18.14** | **19.42** | **25.21** | **36.28** | **49.14** |
| MoE expert GEMMs (sliced: + shared expert) | 6.44 | 7.37 | 11.62 | 20.27 | 29.02 |
| dense GEMMs (cuBLAS etc.) | 1.01 | 1.02 | 1.09 | 3.41 | 5.53 |
| DCP collectives | 1.30 | 1.29 | 1.62 | 2.15 | 3.29 |
| outside the verify graph (draft, lm_head, sampling, prep) | 1.59 | 1.95 | 2.24 | 2.60 | 3.25 |
| all-reduce + RMSNorm (incl. wait) | 1.15 | 1.13 | 1.39 | 1.76 | 2.55 |
| attention (FlashMLA) | 0.84 | 0.84 | 1.06 | 1.41 | 1.88 |
| GPU idle | 0.54 | 0.60 | 0.65 | 0.63 | 0.98 |
| MoE route/act/finalize + router | 0.50 | 0.50 | 0.55 | 0.60 | 0.70 |
| DSA indexer | 0.57 | 0.57 | 0.61 | 0.65 | 0.66 |
| other verify-graph kernels | 0.49 | 0.45 | 0.46 | 0.49 | 0.60 |
| KV write / skip-KV staging | 0.26 | 0.26 | 0.27 | 0.29 | 0.36 |
| attention (split-KV combine) | 0.28 | 0.28 | 0.25 | 0.32 | 0.32 |
| dense GEMMs (decode_gemm) | 3.17 | 3.17 | 3.41 | 1.69 | 0.00 |
