"""How much does the order in which extra hot slots are filled matter?

Prod's profile lists 3,239 hot experts per rank (frequency-ordered); at 3,676
slots the planner promotes the rest in expert-id order. Held-out check on the
live Claude Code capture: rank experts by route count on the even files,
evaluate on the odd ones. Hot sets per rank at 3,676 slots:
  id-order   : profile list + id-order promotion (what the server does)
  freq-ext   : profile list + promotion in held-in frequency order
  freq-full  : each rank's 3,676 most-routed owned experts (held-in counts)
Each scored with the kernel's assignment (shipped table) and the measured INT4
cell costs: the slowest rank's MoE time, the mean rank's, and cold experts per
rank per layer.

    replay_hotset.py <grid jsonl> [--steps 3000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-04-tiered-decode-kernel-review"))
import profile_grid as pg  # noqa: E402
from replay import expand, runtime_hot, table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("grid")
ap.add_argument("--slots", type=int, default=3676)
ap.add_argument("--steps", type=int, default=3000)
a = ap.parse_args()
profile, primary, secondary, _, shipped, pair, delta, target = pg.configuration()
owners = np.asarray(profile["owners"])
true = expand(table(a.grid, "int4"))
info = json.loads((HERE.parent / "2026-10-04-tiered-decode-kernel-review/profile-grid.json").read_text())
files = sorted(f for f in info["files"] if "claude-glm53-df2-dcp4-cap" in f)
train, test = files[0::2], files[1::2]
counts = np.zeros((75, 256))
for f in train:
    arr = np.load(f, mmap_mode="r")
    for li in range(75):
        ids = np.asarray(arr[:, li + 3]).ravel()
        counts[li] += np.bincount(ids[ids >= 0], minlength=256)

base = runtime_hot(profile, a.slots)


def freq_ext():
    hot = np.zeros_like(base)
    for r in range(4):
        lists = [[e for e in profile["hot_experts"][li] if owners[li, e] == r] for li in range(75)]
        extra = a.slots - sum(map(len, lists))
        cand = sorted(((counts[li, e], li, e) for li in range(75) for e in range(256)
                       if owners[li, e] == r and e not in set(lists[li])), reverse=True)
        for _, li, e in cand[:extra]:
            lists[li].append(e)
        for li in range(75):
            hot[li, lists[li]] = True
    return hot


def freq_full():
    hot = np.zeros_like(base)
    for r in range(4):
        cand = sorted(((counts[li, e], li, e) for li in range(75) for e in range(256)
                       if owners[li, e] == r), reverse=True)
        for _, li, e in cand[:a.slots]:
            hot[li, e] = True
    return hot


sets = {"id-order (prod)": base, "freq-ext": freq_ext(), "freq-full": freq_full()}
for k, h in sets.items():
    assert all(int(h[owners == r].sum()) == a.slots for r in range(4)), k
rng = np.random.default_rng(1)
res = {k: [0.0, 0.0, 0.0] for k in sets}
n = 0
per = max(1, a.steps // len(test))
for f in test:
    arr = np.load(f, mmap_mode="r")
    steps = arr.shape[0] // 8
    for step in rng.choice(steps, size=min(per, steps), replace=False):
        n += 1
        for li in range(75):
            ids = np.asarray(arr[step * 8:(step + 1) * 8, li + 3], dtype=np.int64)
            cnt = np.bincount(ids[ids >= 0].ravel(), minlength=256)
            for k, h in sets.items():
                hs, cs, _ = pg.assign(cnt, primary[li], secondary[li], h[li], shipped,
                                      pair, delta, target)
                t = true[hs, cs]
                res[k][0] += t.max()
                res[k][1] += t.mean()
                res[k][2] += cs.mean()
print(f"held-out: {n} steps x 75 layers from {len(test)} files (ranked on {len(train)}); "
      f"{a.slots} hot slots per rank")
for k, (mx, mean, cold) in res.items():
    print(f"  {k:16s} slowest rank {mx / n / 1e3:6.3f} ms/step   mean rank {mean / n / 1e3:6.3f}"
          f"   cold per rank-layer {cold / n / 75:.2f}")
