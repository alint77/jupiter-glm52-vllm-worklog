"""Is GLM-5.3's routing coupled enough to matter for us? (Zyphra, "Expert
coupling in MoE pretraining": co-selection within a layer, prediction of a
token's layer-l+1 experts from its layer-l experts.)

Live Claude Code decode captures ([steps, 8 tokens, 78 layers, top-8]), files
split in two: transitions learned on even files, evaluated on odd ones.

1. Within-layer co-selection: share of tokens whose top-8 contains one of the
   most frequent 0.8% of pairs, vs the same pairs under independent routing.
2. Cold experts at layer l+1 predicted at layer l, budget B experts per layer
   (summed over the 4 GPUs): recall of the cold experts the step actually
   touches, for
     coupling   score(e') = sum over the step's layer-l routes e of P(e' | e)
     frequency  the B most frequently routed cold experts of layer l+1
     previous   the previous step's cold experts at l+1 (same request)
   at 8 tokens (one request) and 32 (four requests from different files).

    coupling.py [--eval-steps 3000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP / "2026-10-04-tiered-decode-kernel-review"))
import profile_grid as pg  # noqa: E402

runtime_hot = pg.load("plot_glm_c", EXP / "2026-09-27-glm53-mtp7-profile/plot_glm.py").runtime_hot
PROFILE = EXP.parent / "profiles/glm53-w4a16-agentic-3239-r2000-ccfreq3676.json"
E, L0, NL = 256, 3, 75


def load():
    src = pg.SOURCES[0]
    files = [json.loads(l)["file"] for l in (src / "manifest.jsonl").read_text().splitlines()]
    return [np.load(src / f).reshape(-1, 8, 78, 8)[:, :, L0:].astype(np.int64) for f in files]  # [s, 8, 75, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-steps", type=int, default=3000)
    ap.add_argument("--slots", type=int, default=3381)
    a = ap.parse_args()
    reqs = load()
    train, test = reqs[0::2], reqs[1::2]
    hot = runtime_hot(json.loads(PROFILE.read_text()), a.slots)  # [75, 256]
    tr = np.concatenate([r.reshape(-1, NL, 8) for r in train])  # [tokens, 75, 8]
    print(f"{len(reqs)} requests, train tokens {len(tr)}, test steps "
          f"{sum(len(r) for r in test)}; hot/GPU {hot.sum() / 4:.0f}")

    # 1. within-layer co-selection
    iu = np.triu_indices(8, 1)
    for li in (0, 20, 40, 60, 74):
        x = np.sort(tr[:, li], axis=1)
        pid = x[:, iu[0]] * E + x[:, iu[1]]  # [tokens, 28]
        pc = np.bincount(pid.ravel(), minlength=E * E)
        top = np.argsort(pc)[::-1][: int(0.008 * E * (E - 1) / 2)]
        mask = np.zeros(E * E, bool)
        mask[top] = True
        share = mask[pid].any(1).mean()
        # independent routing with the same loads: Gumbel top-8 on the marginals
        p = np.bincount(tr[:, li].ravel(), minlength=E) + 1e-9
        g = np.log(p) + np.random.default_rng(li).gumbel(size=(len(x), E))
        y = np.sort(np.argpartition(-g, 8, axis=1)[:, :8], axis=1)
        ind = mask[y[:, iu[0]] * E + y[:, iu[1]]].any(1).mean()
        print(f"layer {li + L0:2d}: top 0.8% of pairs ({len(top)}) appear in {share:.0%} of "
              f"tokens (independent routing, same loads: {ind:.0%})")

    # 2. transitions P(e' at l+1 | e at l)
    T = np.zeros((NL - 1, E, E))
    for li in range(NL - 1):
        a0 = np.repeat(tr[:, li], 8, axis=1).ravel()
        a1 = np.tile(tr[:, li + 1], (1, 8)).ravel()
        T[li] = np.bincount(a0 * E + a1, minlength=E * E).reshape(E, E)
    T /= np.maximum(T.sum(2, keepdims=True), 1)
    freq = np.stack([np.bincount(tr[:, li].ravel(), minlength=E) for li in range(NL)])
    print(f"P(likeliest successor | e) (per route, sums to 1 over e'), median over (l, e): "
          f"{np.median(T.max(2)):.3f}; uniform: {1 / E:.4f}")

    rng = np.random.default_rng(0)
    budgets = (1, 2, 4, 8, 16)
    for m in (8, 32):
        hit = {k: np.zeros(len(budgets)) for k in ("coupling", "frequency", "previous")}
        prev_bytes = 0
        cold_total = 0
        for _ in range(a.eval_steps):
            picks, prevs = [], []
            used = set()
            while len(picks) < m // 8:
                fi = int(rng.integers(len(test)))
                if fi in used or len(test[fi]) < 2:
                    continue
                used.add(fi)
                s = int(rng.integers(1, len(test[fi])))
                picks.append(test[fi][s])
                prevs.append(test[fi][s - 1])
            r = np.concatenate(picks)  # [m, 75, 8]
            rp = np.concatenate(prevs)
            for li in range(NL - 1):
                nxt = np.unique(r[:, li + 1])
                cold = set(nxt[~hot[li + 1][nxt]].tolist())
                if not cold:
                    continue
                cold_total += len(cold)
                cmask = ~hot[li + 1]
                score = T[li][r[:, li].ravel()].sum(0)
                score[~cmask] = -1
                order_c = np.argsort(score)[::-1]
                f = freq[li + 1].astype(float)
                f[~cmask] = -1
                order_f = np.argsort(f)[::-1]
                pu = np.unique(rp[:, li + 1])
                pcold = set(pu[~hot[li + 1][pu]].tolist())
                prev_bytes += len(pcold)
                for bi, b in enumerate(budgets):
                    hit["coupling"][bi] += len(cold & set(order_c[:b].tolist()))
                    hit["frequency"][bi] += len(cold & set(order_f[:b].tolist()))
                hit["previous"][:] += len(cold & pcold)
        n = a.eval_steps * (NL - 1)
        print(f"\nM={m}: cold experts touched per layer (4 GPUs) {cold_total / n:.2f}; "
              f"previous step's cold set {prev_bytes / n:.2f} per layer")
        print("  budget B (experts/layer): " + "  ".join(f"{b:5d}" for b in budgets))
        for k in ("coupling", "frequency"):
            print(f"  recall {k:9s}:         " + "  ".join(f"{v / cold_total:5.1%}" for v in hit[k]))
        print(f"  recall previous step: {hit['previous'][0] / cold_total:.1%} "
              f"(at {prev_bytes / n:.2f} experts/layer)")


if __name__ == "__main__":
    main()
