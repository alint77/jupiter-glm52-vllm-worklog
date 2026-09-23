"""Fused linear + cross-entropy (fwd+bwd), tiled over (rows x vocab) chunks.

Design (avoids materialising the [N, V] logits/softmax/grad tensors):
  forward:  per tile: logits_tile = x[rb] @ W[vc].T (bf16 cuBLAS), then a Triton
            kernel does an online-softmax running update of row stats M, S and
            extracts the target logit.  loss = mean(lse - picked) over valid rows.
  backward: per tile: recompute the tile, convert it in place to
            p = softmax(logits) * scale  (bf16), then
            gw[vc] += p.T @ x[rb]        (bf16 accumulation of the tiny p-part)
            gx[rb] += p @ W[vc]          (bf16 accumulation of the tiny p-part)
            and once at the end the one-hot part is added in single-precision
            passes: gx[r] -= scale * W[t_r];  gw[t_r] -= scale * x[r] (atomics),
            so no large value is ever repeatedly rounded in bf16.

Extra memory = one [NB, W] bf16 tile + a handful of [N] fp32 vectors.
Tile shape (NB, W) is chosen per problem shape (see _pick_tile).
"""

import os

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100

STAT_BLOCK_R = int(os.environ.get("FLCE_SBR", 4))
STAT_BLOCK_C = int(os.environ.get("FLCE_SBC", 512))
P_BLOCK_R = int(os.environ.get("FLCE_PBR", 8))
P_BLOCK_C = int(os.environ.get("FLCE_PBC", 512))
ROW_BLOCK = 8  # rows per program for the final one-hot passes

NEG_INF = float("-inf")

# tile shape per (n, h, v); falls back to _default_tile
TILE_TABLE = {
    (4096, 2048, 32000): (4096, 2048),
    (8192, 4096, 151936): (512, 4096),
    (16384, 6144, 152576): (2048, 512),
}

_ENV_NB = os.environ.get("FLCE_NB")
_ENV_W = os.environ.get("FLCE_W")


def _default_tile(n, h, v):
    nb = min(n, max(256, n // 8))
    w = min(v, 2048)
    return nb, w


def _pick_tile(n, h, v):
    if _ENV_NB or _ENV_W:
        nb = int(_ENV_NB) if _ENV_NB else min(n, 2048)
        w = int(_ENV_W) if _ENV_W else min(v, 2048)
        return min(nb, n), min(w, v)
    key = (n, h, v)
    if key in TILE_TABLE:
        return TILE_TABLE[key]
    return _default_tile(n, h, v)


@triton.jit
def _stat_kernel(L, M, S, P, T, vc0, n_rows, n_cols, stride_l,
                 BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """Online softmax row-stat update for one [n_rows, n_cols] bf16 tile.

    M,S: running per-row max / sumexp (read-modify-write).
    P:   picked logit, written only for rows whose target lies in this chunk.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    m = tl.load(M + rows, mask=rmask, other=-float("inf"))
    s = tl.load(S + rows, mask=rmask, other=0.0)
    t = tl.load(T + rows, mask=rmask, other=-100)
    tc = tl.where(t >= 0, t - vc0, -1)
    for off in range(0, n_cols, BLOCK_C):
        cols = off + tl.arange(0, BLOCK_C)
        cmask = cols < n_cols
        ptr = L + rows[:, None].to(tl.int64) * stride_l + cols[None, :]
        l = tl.load(ptr, mask=rmask[:, None] & cmask[None, :],
                    other=-float("inf")).to(tl.float32)
        bmax = tl.max(l, 1)
        m_new = tl.maximum(m, bmax)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(l - m_new[:, None]), 1)
        m = m_new
        inb = (tc >= off) & (tc < off + BLOCK_C)
        val = tl.load(L + rows.to(tl.int64) * stride_l + tc,
                      mask=rmask & inb, other=0.0).to(tl.float32)
        tl.store(P + rows, val, mask=rmask & inb)
    tl.store(M + rows, m, mask=rmask)
    tl.store(S + rows, s, mask=rmask)


@triton.jit
def _p_kernel(L, LSE, T, SCALE, vc0, n_rows, n_cols, stride_l,
              BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """In-place: bf16 logits tile -> bf16 (softmax * scale) tile."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    lse = tl.load(LSE + rows, mask=rmask, other=0.0)
    t = tl.load(T + rows, mask=rmask, other=-100)
    sc = tl.load(SCALE)
    rs = tl.where(t >= 0, sc, 0.0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < n_cols
    ptr = L + rows[:, None].to(tl.int64) * stride_l + cols[None, :]
    l = tl.load(ptr, mask=rmask[:, None] & cmask[None, :],
                other=-float("inf")).to(tl.float32)
    p = tl.exp(l - lse[:, None]) * rs[:, None]
    tl.store(ptr, p.to(tl.bfloat16), mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _gx_onehot(GX, W, T, SCALE, n_rows, h, stride_gx, stride_w,
               BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    """gx[r] -= scale * W[t_r] for valid rows."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    t = tl.load(T + rows, mask=rmask, other=-100)
    ok = t >= 0
    sc = tl.load(SCALE)
    for off in range(0, h, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        cmask = cols < h
        g = tl.load(GX + rows[:, None].to(tl.int64) * stride_gx + cols[None, :],
                    mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        w = tl.load(W + t[:, None].to(tl.int64) * stride_w + cols[None, :],
                    mask=ok[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        g -= sc * w
        tl.store(GX + rows[:, None].to(tl.int64) * stride_gx + cols[None, :],
                 g.to(tl.bfloat16), mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _gw_onehot(GW, X, T, SCALE, n_rows, h, stride_gw, stride_x,
               BLOCK_R: tl.constexpr, BLOCK_H: tl.constexpr):
    """gw[t_r] -= scale * x[r] for valid rows (atomics; targets may repeat)."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    t = tl.load(T + rows, mask=rmask, other=-100)
    ok = (t >= 0) & rmask
    sc = tl.load(SCALE)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cmask = cols < h
    x = tl.load(X + rows[:, None].to(tl.int64) * stride_x + cols[None, :],
                mask=ok[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    tl.atomic_add(GW + t[:, None].to(tl.int64) * stride_gw + cols[None, :],
                  (-(sc * x)).to(tl.bfloat16), mask=ok[:, None] & cmask[None, :])


def _row_blocks(n, nb):
    return [(r0, min(nb, n - r0)) for r0 in range(0, n, nb)]


def _col_chunks(v, w):
    return [(c0, min(w, v - c0)) for c0 in range(0, v, w)]


class _FLCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target):
        x = x.contiguous()
        weight = weight.contiguous()
        n, h = x.shape
        v = weight.shape[0]
        nb, w = _pick_tile(n, h, v)
        nb = max(1, min(nb, n))
        w = max(16, min(w, v))
        dev = x.device

        m = torch.full((n,), NEG_INF, device=dev, dtype=torch.float32)
        s = torch.zeros(n, device=dev, dtype=torch.float32)
        p = torch.zeros(n, device=dev, dtype=torch.float32)

        rbs = _row_blocks(n, nb)
        vcs = _col_chunks(v, w)
        n_rb = len(rbs)
        wide = vcs[0][1]
        has_tail = vcs[-1][1] != wide
        bufs = []
        bufs.append(torch.empty((nb, wide), device=dev, dtype=torch.bfloat16))
        if has_tail:
            bufs.append(torch.empty((nb, vcs[-1][1]), device=dev,
                                    dtype=torch.bfloat16))

        for r0, nr in rbs:
            xv = x[r0:r0 + nr]
            mv, sv, pv, tv = m[r0:], s[r0:], p[r0:], target[r0:]
            grid = (triton.cdiv(nr, STAT_BLOCK_R),)
            for ci, (c0, cw) in enumerate(vcs):
                buf = bufs[0] if cw == wide else bufs[-1]
                tl_ = buf[:nr]
                torch.mm(xv, weight[c0:c0 + cw].t(), out=tl_)
                _stat_kernel[grid](tl_, mv, sv, pv, tv, c0, nr, cw,
                                   tl_.stride(0), BLOCK_R=STAT_BLOCK_R,
                                   BLOCK_C=STAT_BLOCK_C)

        lse = m + torch.log(s)
        valid = target != IGNORE_INDEX
        count = valid.sum().clamp(min=1)
        loss = ((lse - p) * valid).sum() / count

        ctx.save_for_backward(x, weight, target, lse, count)
        ctx.rbs = rbs
        ctx.vcs = vcs
        ctx.bufs = bufs
        ctx.n_rb = n_rb
        ctx.n = n
        ctx.h = h
        ctx.v = v
        return loss

    @staticmethod
    def backward(ctx, dout):
        x, weight, target, lse, count = ctx.saved_tensors
        n, h, v = ctx.n, ctx.h, ctx.v
        dev = x.device

        scale = dout.reshape(()).float() / count.float()
        gx = torch.zeros((n, h), device=dev, dtype=torch.bfloat16)
        gw = (torch.empty if ctx.n_rb == 1 else torch.zeros)(
            (v, h), device=dev, dtype=torch.bfloat16)

        for r0, nr in ctx.rbs:
            xv = x[r0:r0 + nr]
            lsev, tv = lse[r0:], target[r0:]
            gxv = gx[r0:r0 + nr]
            grid = (triton.cdiv(nr, P_BLOCK_R), None)
            for c0, cw in ctx.vcs:
                buf = ctx.bufs[0] if cw == ctx.bufs[0].shape[1] else ctx.bufs[-1]
                tl_ = buf[:nr]
                wv = weight[c0:c0 + cw]
                torch.mm(xv, wv.t(), out=tl_)
                _p_kernel[(grid[0], triton.cdiv(cw, P_BLOCK_C))](
                    tl_, lsev, tv, scale, c0, nr, cw, tl_.stride(0),
                    BLOCK_R=P_BLOCK_R, BLOCK_C=P_BLOCK_C)
                gwv = gw[c0:c0 + cw]
                if ctx.n_rb == 1:
                    torch.mm(tl_.t(), xv, out=gwv)
                else:
                    gwv.addmm_(tl_.t(), xv)
                gxv.addmm_(tl_, wv)

        grid1 = (triton.cdiv(n, ROW_BLOCK),)
        _gx_onehot[grid1](gx, weight, target, scale, n, h,
                          gx.stride(0), weight.stride(0),
                          BLOCK_R=ROW_BLOCK, BLOCK_H=1024)
        grid2 = (triton.cdiv(n, ROW_BLOCK), triton.cdiv(h, 1024))
        _gw_onehot[grid2](gw, x, target, scale, n, h,
                          gw.stride(0), x.stride(0),
                          BLOCK_R=ROW_BLOCK, BLOCK_H=1024)
        return gx, gw, None


def fused_linear_cross_entropy(x, weight, target):
    return _FLCE.apply(x, weight, target)
