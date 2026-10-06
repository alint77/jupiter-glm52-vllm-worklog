"""Gate 0b: what ~174 more hot experts per GPU buy on live traffic.

Replays the live Claude Code routing capture (8-token verify steps) against
hot sets at the served budget (3,180 per GPU) and at +EXTRA, two ways:
  planner  the served profile promoted as the planner does it today
           (expert-id order, round-robin over layers: plot_glm.runtime_hot)
  ranked   the served set plus each GPU's EXTRA most frequent cold
           (layer, expert) pairs on the agentic capture's train split, which
           is what a profile rebuilt at the larger budget would hold
Reports cold experts per GPU per layer (mean) and at the slowest GPU per layer
(the post-MoE all-reduce waits for it), and ms/step at ~51 us per cold expert.
Replicas ignored: an expert counts on its owner."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
sys.path.insert(0, str(HERE.parent / "2026-09-27-glm53-mtp7-profile"))
from mimo_replicas import EP, active_mask, load_steps  # noqa: E402
from plot_glm import runtime_hot  # noqa: E402

BASE, EXTRA, US = 3180, int(sys.argv[1]) if len(sys.argv) > 1 else 174, 51.0
LIVE = Path("/e/fscratch/profound/naeimitabiei1/routes-datasets/glm53-cc-20261004-job2173771")
TRAIN = Path("/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged")
prof = json.loads((HERE.parents[1] / "profiles/glm53-w4a16-agentic-3239-r2000.json").read_text())
layers, owners = prof["routed_layers"], np.asarray(prof["owners"])
n = owners.shape[1]


def ranked(hot: np.ndarray, extra: int) -> np.ndarray:
    freq = active_mask(load_steps(TRAIN, "train", layers), n).sum(0)  # [layers, n]
    out = hot.copy()
    for r in range(EP):
        cand = np.argwhere((owners == r) & ~hot)
        order = np.argsort(-freq[cand[:, 0], cand[:, 1]], kind="stable")[:extra]
        out[cand[order, 0], cand[order, 1]] = True
    return out


def cold_per_gpu(act: np.ndarray, hot: np.ndarray) -> np.ndarray:
    cold = act & ~hot[None]  # [steps, layers, n]
    return np.stack([(cold & (owners == r)[None]).sum(-1) for r in range(EP)], -1)


act = active_mask(load_steps(LIVE, "heldout", layers), n)
base = runtime_hot(prof, BASE)
sets = {"served (3180)": base,
        f"planner +{EXTRA}": runtime_hot(prof, BASE + EXTRA),
        f"ranked +{EXTRA}": ranked(base, EXTRA)}
print(f"{act.shape[0]} live verify steps, {len(layers)} MoE layers")
ref = None
for name, h in sets.items():
    c = cold_per_gpu(act, h)
    mean, worst = c.mean(), c.max(-1).mean()
    if ref is None:
        ref = (mean, worst)
    print(f"{name:16s} hot/GPU {h.sum() / EP:6.0f}  cold/GPU/layer mean {mean:.3f}"
          f"  slowest GPU {worst:.3f}  -> vs served: mean "
          f"{(ref[0] - mean) * len(layers) * US / 1e3:.3f} ms/step, slowest "
          f"{(ref[1] - worst) * len(layers) * US / 1e3:.3f} ms/step")
