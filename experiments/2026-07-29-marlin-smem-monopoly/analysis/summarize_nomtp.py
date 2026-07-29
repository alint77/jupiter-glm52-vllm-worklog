"""Summarize the no-MTP concurrency sweep.

Without speculation each engine step emits exactly one token per sequence, so
mean TPOT is the step time and no acceptance term has to be divided out. The
decode batch M equals the concurrency.
"""

import json
import pathlib
import statistics

D = pathlib.Path(__file__).resolve().parent.parent
CONCURRENCIES = (1, 2, 4)


def reps(arm, conc):
    out = []
    for r in (1, 2):
        p = D / f"nomtp-{arm}-c{conc}-r{r}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def main():
    print(f"{'M':>3} {'off TPOT ms':>22} {'on TPOT ms':>22} {'step time':>10} "
          f"{'off tok/s':>10} {'on tok/s':>10}")
    any_row = False
    for conc in CONCURRENCIES:
        o, n = reps("off", conc), reps("on", conc)
        if not o or not n:
            continue
        any_row = True
        ot = [r["mean_tpot_ms"] for r in o]
        nt = [r["mean_tpot_ms"] for r in n]
        oth = [r["output_throughput"] for r in o]
        nth = [r["output_throughput"] for r in n]
        om, nm = statistics.mean(ot), statistics.mean(nt)
        spread = max(max(ot) - min(ot), max(nt) - min(nt))
        flag = "" if abs(nm - om) > spread else "  (within spread)"
        print(f"{conc:>3} {om:>10.3f} [{min(ot):.3f},{max(ot):.3f}]".ljust(30)
              + f"{nm:>10.3f} [{min(nt):.3f},{max(nt):.3f}]".ljust(24)
              + f"{nm / om - 1:>+9.2%} {statistics.mean(oth):>10.2f} "
              + f"{statistics.mean(nth):>10.2f}{flag}")
    if not any_row:
        print("no results yet")


if __name__ == "__main__":
    main()
