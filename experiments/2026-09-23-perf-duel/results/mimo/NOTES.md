# Fused linear + cross-entropy — solution notes

## Result (final, `bench.py`, GH200 / H100-class)

| shape | N,H,V | time (ms) | baseline (ms) | extra mem (GiB) | baseline (GiB) | speedup | mem reduction |
|---|---|---|---|---|---|---|---|
| small | 4096, 2048, 32000   |  46.47 |   4.44 | 0.00 |  1.33 | 0.10x |   633x |
| main  | 8192, 4096, 151936  | 218.63 |  70.47 | 0.00 | 12.69 | 0.32x |  3029x |
| large | 16384, 6144, 152576 | 477.29 | 190.81 | 0.01 | 26.00 | 0.40x |  3104x |

**SCORE (geomean speedup × mem_reduction over main,large): 1100.60**

Errors (vs fp32 reference, tolerances 1e-3 loss / 1e-2 grads): loss ≈ 2e-6,
grad_x ≈ 2–4e-3, grad_w ≈ 2–4e-3 on every shape — all pass with margin.
Tile size is tunable (`FLCE_BV`, default 128 = score-optimal); BV=256 trades
~2× memory for ~1.4× speed (score ≈ 870), BV=512 gives score ≈ 670.

## Approach

**Exact gradient decomposition.** With `d = (p − onehot(t)) / count`:

    gx[n]  = (Σ_v p[n,v]·W[v] − W[t[n]]) / count
    gw[v]  = (Σ_n p[n,v]·x[n] − Σ_{n:t[n]=v} x[n]) / count

The softmax ("p") part is only ~2% of gx and ~1% of gw in norm (measured), but
cannot be skipped (1e-2 tolerance). The onehot part dominates the magnitude.

**Streaming formulation ("X''"), 4 GEMMs total.** Vocab is processed in tiles of
`BLOCK_V` classes; only the `[N, BLOCK_V]` bf16 logit tile is ever materialized
(a few MiB — this is the whole transient memory budget):

1. forward pass over tiles: Triton `_lse_kernel` carries a running (max,
   sum-exp) per row across tiles and stores `lse`; the row's own-target logit
   is folded into the same kernel (zero extra passes). Loss =
   `Σ_n (lse[n] − tlog[n])·valid[n] / count` in fp64.
2. backward pass over tiles: recompute the logit tile (`x @ W_tile^T`), Triton
   `_softmax_kernel` converts it in place to `p = exp(logit − lse)/count`, then
   `gw[tile] = pᵀ·x` (exact tile write, `out=`) and `gx += p @ W_tile`
   (`addmm_` accumulating directly into the grad buffer).

**Onehot terms, added last ("dominant-late").** A single temp-free Triton pass
does `gx[n] -= (1/count)·W[t[n]]` (row gather) and
`gw[t[n]] += (−1/count)·x[n]` (bf16 `atomic_add` scatter) — no `[R,H]`
scratch buffers. Adding the dominant magnitude last means all bf16 rounding of
the accumulated p-part happens at the scale of that tiny term, so the
`[N,H]`/`[V,H]` bf16 accumulations are safe (this is what keeps grad error at
~3e-3 instead of ~1e-2).

**No autograd clones.** Gradients are computed eagerly in `forward()` and the
`backward()` hook returns tensors that alias the saved grad storages
(`detach()`); PyTorch's autograd engine otherwise clones returned grads
(~1.3 GiB extra on `large`). x/weight/target are saved for backward as aliases
of the inputs (no memory cost), and grads land directly in `x.grad`/`w.grad`.

**Ignored targets (−100)** are handled with `1/count` scaling per row (zero on
ignored rows) and `t = target.clamp(min=0)` for safe gathers — no control flow.

## Trade-off: time vs memory

Score is `speedup × mem_reduction`, and memory counts linearly down to a 1 MiB
floor, so tile size `BLOCK_V` is the dial: transient memory is `N·BLOCK_V·2`
bytes plus a `BLOCK_V·H·4`-byte GEMM scratch, while time grows as tiles shrink
(more GEMM-launches and more `gx += p@W` accumulate traffic over the `[N,H]`
output). Measured on `main` (time / extra): BV=2048: 76 ms / 65 MiB,
BV=1024: 80 / 33, BV=512: 88 / 17, BV=256: 117 / 9.3, BV=128: 195 / 5.3 —
score peaks at BV≈128 for both scored shapes. Everything is Triton + vanilla
PyTorch (matmul/mm/addmm_/index-free scatter), no other libraries.

## Reproduce

    /e/project1/profound/alint77/vllm/.venv/bin/python bench.py
