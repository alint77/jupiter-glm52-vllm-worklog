"""Rank-promotion gap at the newprod budget, on live Claude Code routing.

The newprod server (skipkv-newprod, 2026-10-07) serves 3,508-3,510 hot
experts per GPU from the agentic profile's 3,239 names, the remaining ~271
promoted in expert-id order (planner log). This replays the live CC capture
(steps from job 2173771, heldout) against:
  oldprod   runtime_hot(profile, 3180): the Sept-30 budget (reference)
  newprod   runtime_hot(profile, 3510): exactly what the trace served
  ranked    the profile's 3,239 + each GPU's 271 most frequent cold pairs
            (train split): what a profile rebuilt at the larger budget holds
Cold experts per GPU per layer (mean, and the slowest GPU's mean), ms/step
at ~51 us per cold expert. Replicas ignored (owner count only).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
sys.path.insert(0, str(HERE.parent / "2026-09-27-glm53-mtp7-profile"))
from mimo_replicas import EP, active_mask, load_steps  # noqa: E402
from plot_glm import runtime_hot  # noqa: E402

US = 51.0
LIVE = Path("/e/fscratch/profound/naeimitabiei1/routes-datasets/glm53-cc-20261004-job2173771")
TRAIN = Path("/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged")
prof = json.loads((HERE.parents[1] / "profiles/glm53-w4a16-agentic-3239-r2000.json").read_text())if False else None
import json
prof = json.loads((HERE.parents[1] / "profiles/glm53-w4a16-agentic-3239-r2000.json").read_text())
layers, owners = prof["routed_layers"], np.asarray(prof["owners"])
n = owners.shape[1]


def ranked(budget: int) -> np.ndarray:
    freq = active_mask(load_steps(TRAIN, "train", layers), n).sum(0)  # [layers, n]
    hot = np.zeros(owners.shape, dtype=bool)
    for li, lst in enumerate(prof["hot_experts"]):
        for e in lst:
            if owners[li, e]:
                hot[li, e] = True
    for r in range(EP):
        owned = int(hot[(owners == r)].sum())
        cand = np.argwhere((owners == r) & ~hot)
        order = np.argsort(-freq[cand[:, 0], cand[:, 1]], kind="stable")[:budget - owned]
        hot[cand[order, 0], cand[order, 1]] = True
    return hot


def cold_per_gpu(act: np.ndarray, hot: np.ndarray) -> np.ndarray:
    cold = act & ~hot[None]
    return np.stack([(cold & (owners == r)[None]).sum(-1) for r in range(EP)], -1)


def id_promoted(slots: int) -> np.ndarray:
    """runtime_hot with the planner's skip-a-full-layer guard (plot_glm raises
    StopIteration when a rank-layer already holds all 64 owned experts)."""
    hot = np.zeros(owners.shape, dtype=bool)
    for r in range(EP):
        lists = [[e for e in prof["hot_experts"][li] if owners[li, e] == r]
                 for li in range(len(layers))]
        extra = slots - sum(map(len, lists))
        guard = 0
        while extra and guard < 100000:
            guard += 1
            hit = False
            for li in range(len(layers)):
                if not extra:
                    break
                have = set(lists[li])
                cand = next((e for e in range(n) if owners[li, e] == r and e not in have), None)
                if extra < 0:
                    if lists[li]:
                        lists[li].pop()
                        extra += 1
                    hit = True
                elif cand is not None:
                    lists[li].append(cand)
                    extra -= 1
                    hit = True
            if not hit:
                break
        for li, lst in enumerate(lists):
            hot[li, lst] = True
    return hot


act = active_mask(load_steps(LIVE, "heldout", layers), n)
print(f"{act.shape[0]} live verify steps, {len(layers)} MoE layers")
rows = [("oldprod 3180 (id)", id_promoted(3180)),
        ("newprod 3510 (id) == served", id_promoted(3510)),
        ("newprod 3510 ranked", ranked(3510))]
ref = None
for name, h in rows:
    c = cold_per_gpu(act, h)
    mean, worst = c.mean(), c.max(-1).mean()
    if ref is None:
        ref = (mean, worst)
    print(f"{name:28s} hot/GPU {h.sum() / EP:6.0f}  cold mean {mean:.3f} slowest {worst:.3f}"
          f" -> vs oldprod: mean {(ref[0] - mean) * len(layers) * US / 1e3:+.3f} ms/step,"
          f" slowest {(ref[1] - worst) * len(layers) * US / 1e3:+.3f} ms/step")
