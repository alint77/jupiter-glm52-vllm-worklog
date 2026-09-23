"""PyTorch baseline and fp32 ground truth. Do not edit."""

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def baseline_linear_cross_entropy(x, weight, target):
    """What a framework does by default: fp32 logits, then cross-entropy."""
    logits = (x @ weight.t()).float()
    return F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)


@torch.no_grad()
def fp32_reference(x, weight, target, chunk=1024):
    """Exact fp32 loss and gradients, chunked so it fits in memory."""
    xf = x.float()
    wf = weight.float()
    valid = target != IGNORE_INDEX
    count = valid.sum().clamp(min=1).float()
    loss = torch.zeros((), device=x.device, dtype=torch.float64)
    grad_x = torch.empty_like(xf)
    grad_w = torch.zeros_like(wf)
    for start in range(0, x.shape[0], chunk):
        rows = slice(start, start + chunk)
        logits = xf[rows] @ wf.t()
        lse = torch.logsumexp(logits, dim=-1)
        tgt = target[rows]
        ok = tgt != IGNORE_INDEX
        picked = logits.gather(1, tgt.clamp(min=0).unsqueeze(1)).squeeze(1)
        loss += ((lse - picked) * ok).sum().double()
        dlogits = torch.softmax(logits, dim=-1)
        dlogits[torch.arange(len(tgt), device=x.device), tgt.clamp(min=0)] -= 1.0
        dlogits *= (ok.float() / count).unsqueeze(1)
        grad_x[rows] = dlogits @ wf
        grad_w += dlogits.t() @ xf[rows]
    return (loss / count.double()).float(), grad_x, grad_w
