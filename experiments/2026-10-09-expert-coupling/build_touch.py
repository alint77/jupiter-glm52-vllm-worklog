"""Hot set ranked by steps touched instead of routes, offline, then a profile.

Prod (`profiles/glm53-w4a16-agentic-3239-r2000-ccfreq3676.json`): 3,239 hot
per GPU ranked by route count on the agentic capture's training split
(../2026-09-28-glm53-route-cap), 437 more promoted by route count on the live
Claude Code capture, 2,000-per-GPU replica budget (mimo_replicas.py). Here
each GPU's 3,676 hot experts are ranked over all layers by how many 8-token
steps touch them, each layer's list in that order (the planner demotes from
the tail), and replicas re-placed by the same builder at the same budget.

Hot sets: prod as served; route count (agentic train); steps touched (agentic
train); steps touched (agentic train + even live-capture files). Scored at the
c=1 runtime count (3,670) on the agentic held-out task families and the odd
live-capture files: active cold experts per step summed over layers, busiest
GPU after the replica assignment (what the step waits for) and mean GPU.

    build_touch.py [--write CANDIDATE OUT.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "2026-09-26-mimo-routing-profile"))
from mimo_replicas import active_mask, evaluate, load_steps, place, score  # noqa: E402

sys.path.insert(0, str(HERE))
from coupling import PROFILE, pg, runtime_hot  # noqa: E402

AGENTIC = Path("/e/fscratch/profound/naeimitabiei1/glm53-route-cap/merged")
SLOTS, RUNTIME, BUDGET = 3676, 3670, 2000


def live(layers, part):
    src = pg.SOURCES[0]
    files = [json.loads(l)["file"] for l in (src / "manifest.jsonl").read_text().splitlines()]
    out = [np.load(src / f).reshape(-1, 8, 78, 8)[:, :, layers] for f in files[part::2]]
    return np.concatenate(out)


def ranked(metric, owners, tie):
    """Per GPU, the SLOTS owned (layer, expert) with the largest metric; each
    layer's list in descending metric."""
    lists = [[] for _ in range(owners.shape[0])]
    for g in range(4):
        cand = [(metric[l, e], tie[l, e], l, e) for l in range(owners.shape[0])
                for e in range(owners.shape[1]) if owners[l, e] == g]
        cand.sort(reverse=True)
        for m, t, l, e in cand[:SLOTS]:
            lists[l].append((m, t, e))
    return [[e for _, _, e in sorted(x, reverse=True)] for x in lists]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", nargs=2, metavar=("CANDIDATE", "OUT"))
    a = ap.parse_args()
    prof = json.loads(PROFILE.read_text())
    layers = prof["routed_layers"]
    owners = np.asarray(prof["owners"])
    E = owners.shape[1]
    tr_steps = load_steps(AGENTIC, "train", layers)
    train = active_mask(tr_steps, E)
    held = active_mask(load_steps(AGENTIC, "heldout", layers), E)
    live_even = active_mask(live(layers, 0), E)
    live_odd = active_mask(live(layers, 1), E)
    print(f"steps: agentic train {len(train)}, held-out {len(held)}; live even "
          f"{len(live_even)}, odd {len(live_odd)}", flush=True)
    route = np.stack([np.bincount(tr_steps[:, :, l].ravel(), minlength=E)
                      for l in range(len(layers))]).astype(float)
    touch = train.sum(0).astype(float)
    touch_live = touch + live_even.sum(0)
    lv = live(layers, 0)
    route_live = route + np.stack([np.bincount(lv[:, :, l].ravel(), minlength=E)
                                   for l in range(len(layers))])
    cands = {
        "prod (ccfreq3676)": (prof["hot_experts"], np.asarray(prof["secondary_ranks"])),
        "route count, agentic train + live even": (ranked(route_live, owners, touch_live), None),
        "steps touched, agentic train + live even": (ranked(touch_live, owners, route_live), None),
    }
    out = {}
    for name, (lists, sec) in cands.items():
        p = dict(prof, hot_experts=lists)
        hot = runtime_hot(p, RUNTIME)
        if sec is None:
            sec = place(score(train, owners, hot), owners, BUDGET)
        r = {"replicas/GPU": np.bincount(sec[sec >= 0], minlength=4).tolist()}
        for split, act in (("agentic held-out", held), ("live odd", live_odd)):
            ev = evaluate(act, owners, hot, sec)
            r[split] = (round(ev["critical_cold_per_step"], 1),
                        round(ev["mean_cold_per_rank_per_step"], 1))
        out[name] = (lists, sec)
        print(f"{name:42s} busiest / mean GPU, cold per step: agentic held-out "
              f"{r['agentic held-out'][0]:6.1f} / {r['agentic held-out'][1]:6.1f}   live odd "
              f"{r['live odd'][0]:6.1f} / {r['live odd'][1]:6.1f}   replicas {r['replicas/GPU']}",
              flush=True)
    if a.write:
        lists, sec = out[a.write[0]]
        written = dict(prof, hot_experts=lists, secondary_ranks=sec.tolist(),
                       optimizer=f"profile-owner+steps-touched-{SLOTS}+replicas-minmax-cold-{BUDGET}")
        Path(a.write[1]).write_text(json.dumps(written) + "\n")
        print(f"wrote {a.write[1]}")


if __name__ == "__main__":
    main()
