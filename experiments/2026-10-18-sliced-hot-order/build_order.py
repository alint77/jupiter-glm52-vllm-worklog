"""Hot set for the TP-sliced layout: one priority order over every routed
(layer, expert), the planner taking the first N for whatever HBM allows.

Prod's profiles rank per EP owner GPU (3,676 owned experts each) and the
sliced planner loads the union, then trims round-robin over layers. Here:
- metric: MTP3 verify windows (4 consecutive routed positions of one request)
  that touch the expert, ties by route count; on the agentic capture's train
  split plus the even live Claude Code captures (as touch3676);
- scored on the agentic held-out families plus the odd live captures: steps of
  n requests (one window each, n = 1, 2, 4, 8 -> M = 4..32), distinct cold
  experts per step summed over layers, at the c=4 runtime count (14,290) and
  others. Sliced: every GPU streams a slice of every cold expert, so the step
  waits for the node-wide count.

    build_order.py [--write OUT.json] [--slots 14290]
"""
import argparse
import json
from pathlib import Path

import numpy as np

AGENTIC = Path("/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged")
LIVE = Path("/e/fscratch/profound/naeimitabiei1/claude-glm53-df2-dcp4-cap/routes")
PROFILES = Path(__file__).resolve().parents[2] / "profiles"
PROD = PROFILES / "glm53-w4a16-agentic-3239-r2000-ccfreq3676.json"
TOUCH = PROFILES / "glm53-w4a16-touch3676-r2000.json"
W, E = 4, 256


def requests(split):
    m = json.loads((AGENTIC / "manifest.json").read_text())
    out = [np.load(AGENTIC / r["file"]) for r in m if r["split"] == split]
    files = [json.loads(l)["file"] for l in (LIVE / "manifest.jsonl").read_text().splitlines()]
    out += [np.load(LIVE / f) for f in files[{"train": 0, "heldout": 1}[split]::2]]
    return out


def windows(reqs, layers):
    """[n windows, W, layers, 8] and the request index of each window."""
    ws, owner = [], []
    for i, r in enumerate(reqs):
        r = r[: len(r) // W * W, layers]
        ws.append(r.reshape(-1, W, len(layers), r.shape[2]))
        owner += [i] * (len(r) // W)
    return np.concatenate(ws), np.asarray(owner)


def presence(ids):
    """ids [S, tokens, L, k] -> [S, L, E] bool."""
    s, _, l, _ = ids.shape
    p = np.zeros((s, l, E), dtype=bool)
    for j in range(l):
        np.put_along_axis(p[:, j], ids[:, :, j].reshape(s, -1).astype(np.int64), True, 1)
    return p


def runtime_hot(lists, slots):
    """The sliced planner: the profile's per-layer lists, trimmed (or filled in
    expert order) one per layer round-robin to the slot count."""
    hot = [list(x) for x in lists]
    n = sum(map(len, hot)) - slots
    while n > 0:
        for h in hot:
            if n and h:
                h.pop()
                n -= 1
    while n < 0:
        for h in hot:
            if n < 0 and len(h) < E:
                h.append(next(e for e in range(E) if e not in h))
                n += 1
    m = np.zeros((len(lists), E), dtype=bool)
    for l, h in enumerate(hot):
        m[l, h] = True
    return m


def order_hot(order, slots, layers):
    m = np.zeros((layers, E), dtype=bool)
    for l, e in order[:slots]:
        m[l, e] = True
    return m


def score(wins, owner, hots, n, steps=6000, seed=0):
    """Mean distinct cold experts per step (summed over layers) for steps of n
    windows from distinct requests, per hot mask."""
    rng = np.random.default_rng(seed)
    picks = []
    while len(picks) < steps:
        w = rng.integers(0, len(wins), n)
        if len(set(owner[w])) == n:
            picks.append(w)
    picks = np.asarray(picks)
    tot = {k: 0.0 for k in hots}
    for b in range(0, steps, 1000):
        ids = wins[picks[b:b + 1000]]  # [S, n, W, L, k]
        s = ids.shape
        p = presence(ids.reshape(s[0], s[1] * s[2], s[3], s[4]))
        for k, h in hots.items():
            tot[k] += (p & ~h[None]).sum((1, 2)).sum()
    return {k: v / steps for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--slots", type=int, nargs="+", default=[13500, 14290, 15000])
    a = ap.parse_args()
    prod = json.loads(PROD.read_text())
    layers = prod["routed_layers"]
    L = len(layers)
    tr, _ = windows(requests("train"), layers)
    held, owner = windows(requests("heldout"), layers)
    print(f"windows: train {len(tr)}, held-out {len(held)} ({owner.max() + 1} requests)")
    touch = presence(tr).sum(0).astype(float)
    route = np.stack([np.bincount(tr[:, :, l].ravel(), minlength=E) for l in range(L)])
    order = sorted(((l, e) for l in range(L) for e in range(E)),
                   key=lambda le: (-touch[le], -route[le], le))
    by_route = sorted(((l, e) for l in range(L) for e in range(E)),
                      key=lambda le: (-route[le], -touch[le], le))
    touch_prof = json.loads(TOUCH.read_text())
    for slots in a.slots:
        hots = {
            "prod as served (lists id-sorted)": runtime_hot(
                [sorted(x) for x in prod["hot_experts"]], slots),
            "prod ccfreq3676, round-robin trim": runtime_hot(prod["hot_experts"], slots),
            "touch3676 (per-GPU touch), round-robin": runtime_hot(touch_prof["hot_experts"], slots),
            "global route count": order_hot(by_route, slots, L),
            "global touch (this)": order_hot(order, slots, L),
        }
        cold = {k: L * E - h.sum() for k, h in hots.items()}
        print(f"\n{slots} hot experts (check: cold {set(cold.values())})")
        print(f"  {'hot set':40s}" + "".join(f"  M={4 * n:<4d}" for n in (1, 2, 4, 8)))
        res = {n: score(held, owner, hots, n) for n in (1, 2, 4, 8)}
        for k in hots:
            print(f"  {k:40s}" + "".join(f"  {res[n][k]:6.1f}" for n in (1, 2, 4, 8)))
        base = "prod as served (lists id-sorted)"
        print(f"  {'this vs prod':40s}" + "".join(
            f"  {res[n]['global touch (this)'] - res[n][base]:+6.1f}" for n in (1, 2, 4, 8)))
        per_layer = hots["global touch (this)"].sum(1)
        print(f"  this: hot per layer min {per_layer.min()} max {per_layer.max()}")
    if a.write:
        n0 = 14704
        hot0 = order_hot(order, n0, L)
        lists = [[e for (l2, e) in order if l2 == l and hot0[l, e]] for l in range(L)]
        out = dict(prod, profile_version=3, hot_experts=lists,
                   secondary_ranks=[[-1] * E] * L,
                   residency_order=[[layers[l], e] for l, e in order],
                   optimizer=f"sliced-global-steps-touched-w{W}-order")
        Path(a.write).write_text(json.dumps(out) + "\n")
        print(f"wrote {a.write}")


if __name__ == "__main__":
    main()
