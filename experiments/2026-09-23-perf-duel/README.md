# Perf duel: MiMo-V2.6-Pro vs GLM-5.3 through opencode (2026-09-23)

Same task, prompt, 90-minute budget and GPU type (one GH200 each), each model
served locally (MiMo: tiered + DFlash k=7, 250K; GLM-5.3 W4A16: tiered +
DFlash2, 400K), driven by `opencode run --standalone --auto` with an empty
config home (no external subagents). Graded by `sandbox.sbatch` with pristine
copies of `task/bench.py` and `task/reference.py`.

Task (`task/README.md`): fused linear + cross-entropy, forward and backward,
PyTorch + Triton only; shapes up to N=16384, H=6144, V=152576; score =
geomean(speedup x memory reduction) over `main` and `large`. The unmodified
stub (== baseline) scores 0.99 (`validate-1964216.out`).

## Results (grader)

| | GLM-5.3 (job 1964236) | MiMo-V2.6 (job 1964348) |
|---|---|---|
| correct on all shapes | yes | yes |
| main: time / speedup | 118.5 ms / **0.61x** | 211.5 ms / 0.33x |
| main: extra memory | 0.004 GiB (2948x less) | 0.004 GiB (3029x less) |
| large: time / speedup | 470.6 ms / 0.42x | 475.7 ms / 0.40x |
| large: extra memory | 0.002 GiB (**12898x** less) | 0.008 GiB (3104x less) |
| small: speedup | 1.11x | 0.10x |
| score | **3116** | 1119 |
| wall time used | 42 min (stopped itself) | 60 min (stopped itself) |
| tool calls | 55 | 74 |
| output tokens | 117K | 263K |

## What they built

Independently, nearly the same design: never materialise `[N, V]`; stream
logit tiles from bf16 cuBLAS GEMMs; Triton online-softmax for the logsumexp;
recompute tiles in backward, turn them into `p` in place, and accumulate
`gx += p @ W_tile` and `gw += p^T @ x`; apply the dominant one-hot term
(`-W[t]`, `-x`) once at the end with gathers and bf16 atomics so bf16
accumulation only ever rounds the small `p` part. Both found that precision
trick and both reasoned explicitly that the score rewards memory.

The speed gap on `main` is tiling: GLM tiles rows and vocab (`[512, 4096]`),
keeping GEMMs large; MiMo tiles vocab only at `BLOCK_V=128` over all N rows,
so every GEMM is 128 wide (and small shapes pay per-tile launch cost: 0.10x).

## Caveats

* The score is flawed: memory reduction reaches 10^3-10^4x while speedup
  stays near 1, so both agents traded speed for memory and neither beat the
  baseline's time. A rerun should gate on memory (e.g. <= 1 GiB extra) and
  score speed.
* One run each. MiMo's first attempt (job 1964236) died after 33 s: the
  server applied generation_config.json's `max_new_tokens: 2048` as a
  server-wide cap and cut a reasoning turn; fixed in `claude-server.sbatch`
  and rerun from scratch (1964348). GLM's run is the original.
* GLM self-measured 96 ms on main; the grader measured 118 ms (run-to-run).
