"""Is the profiled hot set better than a random one, by what decode pays for:
distinct cold experts per GPU per layer per 8-token step? The served profile
at the served budget (3,180 hot per GPU) vs random hot sets with the same
per-(layer, GPU) hot counts, on held-out steps of the agentic capture (the
profile's own workload) and of a Claude Code capture. Replicas ignored: an
expert counts on its owner."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26-mimo-routing-profile"))
sys.path.insert(0, str(HERE.parent / "2026-09-27-glm53-mtp7-profile"))
from mimo_replicas import EP, active_mask, load_steps  # noqa: E402
from plot_glm import runtime_hot  # noqa: E402

HOT = 3180
prof = json.loads((HERE.parents[1] / "profiles/glm53-w4a16-agentic-3239-r2000.json").read_text())
layers, owners = prof["routed_layers"], np.asarray(prof["owners"])
hot = runtime_hot(prof, HOT)
rng = np.random.default_rng(0)


def random_hot():
    h = np.zeros_like(hot)
    for li in range(owners.shape[0]):
        for r in range(EP):
            mine = np.flatnonzero(owners[li] == r)
            k = int(hot[li, mine].sum())
            h[li, rng.choice(mine, k, replace=False)] = True
    return h


def stats(act, h):
    cold = act & ~h[None]
    a = act.sum()
    return {"cold share of active": cold.sum() / a,
            "cold / GPU / layer": cold.sum() / (act.shape[0] * act.shape[1] * EP),
            "active / GPU / layer": a / (act.shape[0] * act.shape[1] * EP)}


print(f"hot fraction resident: {hot.mean():.3f} ({hot.sum() / EP:.0f} per GPU)")
for name, d in (("agentic capture (profile's workload)", "/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged"),
                ("Claude Code capture", "/e/fscratch/profound/naeimitabiei1/caches/routes/snap-1535650-a")):
    act = active_mask(load_steps(Path(d), "heldout", layers), owners.shape[1])
    s = stats(act, hot)
    rs = [stats(act, random_hot()) for _ in range(5)]
    r = {k: np.mean([x[k] for x in rs]) for k in s}
    print(f"{name}: {act.shape[0]} held-out steps")
    for k in s:
        print(f"   {k:22s} profiled {s[k]:.3f}   random {r[k]:.3f}")
