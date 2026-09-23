"""Your fused linear + cross-entropy. Replace this naive version."""

import torch
import torch.nn.functional as F


def fused_linear_cross_entropy(x, weight, target):
    logits = (x @ weight.t()).float()
    return F.cross_entropy(logits, target, ignore_index=-100)
