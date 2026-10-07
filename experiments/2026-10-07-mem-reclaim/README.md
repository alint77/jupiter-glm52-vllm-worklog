# Reclaimed HBM -> hot experts: reserve sweep and before/after (2026-10-07)

vLLM `dflash2-backport` (local commits): f3c882485e (DCP prefill workspaces
sized to what they hold, chunked scale exponent: -2.15 GiB), d0ec991bf4 (one
NCCL communicator per rank set; TP, DCP and EP share it: -0.94 GiB),
b306c22850 (input embedding on Grace, VLLM_TIERED_MOE_EMBED_HOST: -0.44 GiB),
794eda5446 (planned-reserve floor 2 -> 0.5 GB; the fail-closed startup check
stays the guard). Diagnosis: `../2026-10-07-decode-mem-dive`.

## Reserve sweep (QUICK=1 arm.sh: TTFT + 388K stress; embedding on Grace)

At reserve 4.7 the new code leaves 6.24-6.33 GiB free after startup (prod
2.67-2.79; minimum 2.53).

| RESERVE_GB | hot / GPU | tightest-rank free | result |
|--:|--:|--:|---|
| 2.0 | 3,662 | 2.96 GiB | 388K OK, 0 OOM |
| 1.6 | 3,681 | 2.62 | 388K OK, 0 OOM |
| 1.3 / 1.0 / 0.85 / 0.7 | | 2.49 / 2.26 / 2.08 / 1.92 GB < 2.71 | refuse to start |

Free memory falls ~1.2 GB per GB of reserve (whole experts per layer), and
ranks 2-3 run ~0.8 GiB tighter than 0-1. Chosen: **1.7** (~0.17 GiB margin on
the tightest rank, prod's 4.7 had ~0.14).

## Before vs after (arm.sh, 6 nodes alternating, `compare_ba.py`)

before = HEAD ba5dc7c7aa from a worktree (PYTHONPATH, own cache root), prod
serve.sh (reserve 4.7); after = this tree, VLLM_TIERED_MOE_EMBED_HOST=1,
RESERVE_GB=1.7. Each arm: GSM8K 200, 32 agentic requests, 50K/130K decode (2
seeds each), TTFT at three shapes, stress to 388K.

| | before (12 arms) | after (9 arms) |
|---|--:|--:|
| hot experts / GPU | 3,531 | **3,676 (+145)** |
| tightest startup free (min 2.53) | 2.67-2.70 GiB | 2.70-2.72 GiB |
| agentic decode, fit over 383 requests (tok/step, ctx, node) | | **-0.223 +- 0.030 ms/step** |
| 50K / 130K decode, 84 requests | | **-0.265 +- 0.033 ms/step** |
| GSM8K 200 | 0.913 | 0.913 |
| TTFT 20K/14K, 8K/150K, 60K | 2.76-2.91, 1.51-1.61, 8.07-8.56 s | 2.76-2.90, 1.52-1.59, 8.09-8.54 s |
| 388K stress | 11/12 (one cuBLAS failure) | 9/9 |
| GPU peak | 96.8-97.3 GB | 97.0-97.3 GB |

The one failure is in the old code (before-2218527-4): CUBLAS_STATUS_EXECUTION_FAILED
in a strided-batched GEMM on rank 3 at 314K tokens, 97,271 MiB peak (the run's
highest), no OOM line. Not seen in the new code, but a 1-in-21 event: watch for
it at long context. Three after arms (2218523-3, 2218524-4, 2218527-3) died at
startup because the shared tree was edited while they imported; excluded.
