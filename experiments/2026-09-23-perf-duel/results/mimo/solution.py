"""Fused linear + cross-entropy (forward and backward), Triton streaming design.

Streaming "X''" formulation: vocab is processed in tiles of BLOCK_V classes.
For each tile we materialize only the [N, BLOCK_V] logit tile:

  pass 1 (forward):  lse accumulation (running max/sum in Triton) + own-target
                     logit, folded into the same kernel -> scalar loss
  pass 2 (backward): p_tile = exp(logit - lse) * (1/count), then
                     gw[tile]  = p_tile^T @ x          (exact tile write)
                     gx       += p_tile @ W[tile]      (accumulates [N, H])

plus the dominant-magnitude onehot terms, added LAST so bf16 rounding happens
at the scale of the tiny softmax term (exact gradient decomposition):

  gx[n]  -= x-count^-1 * W[target[n]]
  gw[tgt[n]] -= count^-1 * x[n]        (atomic scatter-add, no temp buffers)

Peak transient memory is one [N, BLOCK_V] bf16 logit tile plus a handful of
[N] vectors; gradients accumulate directly into the (excluded) grad tensors.
"""

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100

BLOCK_V = int(__import__("os").environ.get("FLCE_BV", "128"))  # vocab columns per tile


@triton.jit
def _lse_kernel(
    lg,  # [n, b] bf16 logits (read only)
    lse,  # [n] fp32 out
    mrun,  # [n] fp32 running max
    srun,  # [n] fp32 running sum exp(l - max)
    tlog,  # [n] fp32 out: logit of the row's own target (0 if not in tile)
    idx,  # [n] int64 clamped target index
    v0,  # first vocab column of this tile
    n,
    b,
    stride_lg,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < n
    mm = tl.load(mrun + r, mask=rm, other=float("-inf"))
    ss = tl.load(srun + r, mask=rm, other=0.0)
    for c0 in range(0, b, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        x = tl.load(
            lg + r[:, None] * stride_lg + c[None, :],
            mask=rm[:, None] & (c[None, :] < b),
            other=float("-inf"),
        ).to(tl.float32)
        tmax = tl.max(x, axis=1)
        newm = tl.maximum(mm, tmax)
        ss = ss * tl.exp(mm - newm) + tl.sum(tl.exp(x - newm[:, None]), axis=1)
        mm = newm
    out = mm + tl.log(ss)
    tl.store(lse + r, out, mask=rm)
    tl.store(mrun + r, mm, mask=rm)
    tl.store(srun + r, ss, mask=rm)
    ti = tl.load(idx + r, mask=rm, other=0)
    tc = ti - v0
    inb = rm & (tc >= 0) & (tc < b)
    tv = tl.load(lg + r * stride_lg + tc, mask=inb, other=0.0).to(tl.float32)
    tl.store(tlog + r, tv, mask=inb)


@triton.jit
def _softmax_kernel(
    lg,  # [n, b] bf16, in place: logits -> exp(logits - lse) / count
    lse,  # [n] fp32
    scale,  # [n] fp32 (valid / count)
    n,
    b,
    stride_lg,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n * b
    row = off // b
    x = tl.load(lg + off, mask=mask, other=0.0).to(tl.float32)
    l = tl.load(lse + row, mask=mask, other=0.0)
    sc = tl.load(scale + row, mask=mask, other=0.0)
    tl.store(lg + off, tl.exp(x - l) * sc, mask=mask)


@triton.jit
def _onehot_kernel(
    gx,  # [n, h] bf16, accumulate target pull-out
    gw,  # [v, h] bf16, atomic scatter-add of -x/count
    X,  # [n, h] bf16
    W,  # [v, h] bf16
    idx,  # [n] int64 clamped target
    pc,  # [n] bf16 1/count on valid rows (0 elsewhere)
    mc,  # [n] bf16 -1/count on valid rows
    n,
    h,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pr = tl.program_id(0)
    ph = tl.program_id(1)
    r = pr * BLOCK_R + tl.arange(0, BLOCK_R)
    hc = ph * BLOCK_H + tl.arange(0, BLOCK_H)
    m = (r < n)[:, None] & (hc < h)[None, :]
    vi = tl.load(idx + r, mask=r < n, other=0)
    pcv = tl.load(pc + r, mask=r < n, other=0.0)
    mcv = tl.load(mc + r, mask=r < n, other=0.0)
    wr = tl.load(W + vi[:, None] * h + hc[None, :], mask=m, other=0.0)
    gr = tl.load(gx + r[:, None] * h + hc[None, :], mask=m, other=0.0)
    tl.store(gx + r[:, None] * h + hc[None, :], gr - pcv[:, None] * wr, mask=m)
    xr = tl.load(X + r[:, None] * h + hc[None, :], mask=m, other=0.0)
    tl.atomic_add(gw + vi[:, None] * h + hc[None, :], mcv[:, None] * xr, mask=m)


def fused_linear_cross_entropy(x, weight, target):
    return _Flce.apply(x, weight, target)


class _Flce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target):
        n, h = x.shape
        v = weight.shape[0]
        dev = x.device

        ok = target != IGNORE_INDEX
        cnt = ok.sum().clamp(min=1).float()
        valid = ok.to(torch.float32)
        idx = target.clamp(min=0)
        inv_c = valid / cnt                      # [n] 1/count on valid rows
        pc = inv_c.to(torch.bfloat16)
        mc = -pc

        mrun = torch.full((n,), float("-inf"), dtype=torch.float32, device=dev)
        srun = torch.zeros(n, dtype=torch.float32, device=dev)
        lse = torch.empty(n, dtype=torch.float32, device=dev)
        tlog = torch.zeros(n, dtype=torch.float32, device=dev)

        # ---------------- forward: lse over vocab tiles -------------------
        for v0 in range(0, v, BLOCK_V):
            v1 = min(v0 + BLOCK_V, v)
            b = v1 - v0
            logits = x.matmul(weight[v0:v1].t())            # [n, b] bf16
            _lse_kernel[(triton.cdiv(n, 16),)](
                logits, lse, mrun, srun, tlog, idx, v0, n, b, logits.stride(0),
                BLOCK_R=16, BLOCK_C=256,
            )

        loss = ((lse - tlog) * valid).sum(dtype=torch.float64) / cnt.double()

        # ---------------- backward work: both gradients -------------------
        gx = torch.zeros(n, h, dtype=torch.bfloat16, device=dev)   # -> x.grad
        gw = torch.zeros(v, h, dtype=torch.bfloat16, device=dev)   # -> w.grad
        for v0 in range(0, v, BLOCK_V):
            v1 = min(v0 + BLOCK_V, v)
            b = v1 - v0
            logits = x.matmul(weight[v0:v1].t())            # recompute [n, b]
            _softmax_kernel[(triton.cdiv(n * b, 1024),)](
                logits, lse, inv_c, n, b, logits.stride(0), BLOCK=1024,
            )
            torch.mm(logits.t(), x, out=gw[v0:v1])          # p.T @ x, once
            gx.addmm_(logits, weight[v0:v1])                # accumulate p @ W
        del logits

        # onehot parts (dominant magnitude, so rounding is relative to grad);
        # done in one temp-free Triton pass: gather into gx, scatter-add gw
        _onehot_kernel[(triton.cdiv(n, 32), triton.cdiv(h, 256))](
            gx, gw, x, weight, idx, pc, mc, n, h, BLOCK_R=32, BLOCK_H=256,
        )

        ctx.x_grad = gx
        ctx.w_grad = gw
        return loss.to(torch.float32)

    @staticmethod
    def backward(ctx, grad_out):
        gx = ctx.x_grad
        gw = ctx.w_grad
        gx.mul_(grad_out)
        gw.mul_(grad_out)
        # detach(): returning the raw tensors makes autograd clone both grads
        return gx.detach(), gw.detach(), None
