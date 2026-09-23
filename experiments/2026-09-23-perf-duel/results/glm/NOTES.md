# Fused linear + cross-entropy — approach notes

## Problem recap
`loss = CE(x @ W.T, target, ignore_index=-100)`, mean over valid rows, with
backward for both `x` (bf16 [N,H]) and `W` (bf16 [V,H]).  At LLM vocab sizes
the naive version materialises N×V logits + softmax + grad (≈13 GiB extra for
`main`, 26 GiB for `large`).  Baseline (PyTorch, fp32 CE): **main 72.3 ms /
12.69 GiB**, **large 197 ms / 26.0 GiB**.

## Core design
Never materialise any [N, V] tensor.  Everything is computed from small
"logits tiles" `[NB, W]` (rows × vocab-chunk), produced by bf16 cuBLAS GEMMs,
inside a custom `torch.autograd.Function`:

* **forward** — for each tile `T = x[rb] @ W[vc].T`:
  * a Triton kernel does an *online-softmax* running update of per-row stats
    `M[i]` (max) and `S[i]` (sumexp) and extracts the target logit
    (`picked[i]`) with a single extra load when the target lies in this chunk.
  * `lse = M + log(S)`; `loss = Σ valid·(lse − picked) / count`.
* **backward** — for each tile: recompute it, convert **in place** to
  `p = softmax(logits) × scale` (one Triton kernel), then
  * `gw[vc] += p.T @ x[rb]` — bf16 `addmm_` accumulate across row-blocks,
  * `gx[rb] += p @ W[vc]` — bf16 `addmm_` accumulate across vocab chunks,
  * and once at the end the one-hot part is added by two tiny kernels:
    `gx[r] −= scale·W[t_r]` and `gw[t_r] −= scale·x[r]` (bf16 atomics; duplicate
    targets are possible, hence atomics).

This is 4 GEMM passes total (fwd stats + bwd recompute + gw + gx), the same
matmul work as the baseline's 3 passes plus one recompute pass, but with
**megabytes** instead of gigabytes of intermediate memory.

### Precision tricks (the part that cost the most debugging)
1. **bf16 tile softmax** is fine for this data: logit rounding noise averages
   out over N rows → loss rel err ≈ 2e-6.
2. **Do NOT accumulate the full dlogits in bf16.**  Splitting off the one-hot
   term is essential: `gx = Σ_c p_c@W_c − scale·W[t]`.  Accumulating
   `(p − onehot)` bf16 chunk-by-chunk re-rounds the dominant `-W[t]` value at
   every `addmm_` (measured grad_x error 9.5e-3 — right at the 1e-2 tolerance).
   Accumulating only the *tiny* p-part in bf16 and subtracting `scale·W[t]`
   exactly once gives grad_x error 1.7e-3 (6× margin).  Same idea for gw.

## Score-driven optimisation
`score = geomean(spd × mem_reduction)`.  Since extra memory can drop from GiB
to MB, `mem_reduction` dominates; the optimal operating point trades a
moderate slowdown for a tiny footprint.  Tile shape `[NB, W]` controls both:
buffer bytes `= NB·W·2 + tail`, and time degrades when `NB` or `W` get small
(worse GEMM shapes, more launches, more bf16-accumulate traffic:
`gw traffic ≈ 2·V·H·(N/NB)`, `gx traffic ≈ 2·N·H·(V/W)`).

Measured (real `bench.py`-style runs, KeyError-free, errors ≪ tol):

| shape | tile (NB,W) | ms | speedup | extra | mem_red | shape score |
|---|---|---|---|---|---|---|
| main (8192,4096,151936) | (512, 4096) | ~96 | 0.75 | 4.4 MB | 2948× | ~2200 |
| large (16384,6144,152576) | (2048, 512) | ~345-365 | 0.54-0.58 | 2.06 MB | 12898× | ~7400 |

Rejected / not worth it:
* saving bf16 logits to skip the recompute pass: 3 GB extra → mem ratio ~5×,
  ~3 orders of magnitude worse score than the memory-lean scheme.
* CUDA graphs / pointer-keyed capture: the grader uses fresh tensors; recapture
  per call is slow and fragile.  Skipped.
* stream-overlap of the p-kernel with the next GEMM: needs double buffering →
  doubles the tiny buffer → net score loss.
* W=512 for `main`, W=384 for `large`: K too small, big GEMM slowdown.
* Triton block-size sweeps for the stat/p kernels: all within run-to-run noise.

## Final numbers (bench.py, this machine, 2026-09-23)
```
[small]  4.20 ms (baseline 4.64, 1.11x); extra 0.03 GiB (52.2x less)
[main]  96.26 ms (baseline 72.41, 0.75x); extra 0.00 GiB (2947.5x less)
[large] 336.68 ms (baseline 196.73, 0.58x); extra 0.00 GiB (12898.3x less)
SCORE (geomean speedup x mem_reduction over main,large): 4087.70
```
Run-to-run wall-time variance on this shared machine is ~±10% (large ranged
335-400 ms across identical runs); the score ranged ~3650-4300 accordingly.
`small` uses a 1D tile (single row-block, direct gw write) and is both faster
and leaner than baseline; it does not enter the score.  Tile configs live in
`TILE_TABLE` (overridable via `FLCE_NB`/`FLCE_W`).

Correctness margins: loss ~2e-6 (tol 1e-3), grads 1.7e-3 / 2.0-2.3e-3
(tol 1e-2) — 5-6× headroom on every error metric, all shapes.
