"""probe.json summary per server, as probe_summary.txt: MTP acceptance and the
mean token logprob per request, averaged, greedy and sampled.

    probe_sum.py <sliced-prof tag glob>      e.g. 'dgf-*'
"""
import json
import statistics as st
import sys
from pathlib import Path

R = Path("/e/fscratch/profound/naeimitabiei1/sliced-prof")
print(f"{'server':22s} acc_greedy acc_sampled greedy_lp sampled_lp")
for d in sorted(R.glob(sys.argv[1])):
    f = d / "probe.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    lp = {k: st.mean(st.mean(o["lp"]) for o in r[k]["outs"] if o["lp"])
          for k in ("greedy", "sampled")}
    print(f"{d.name:22s} {r['greedy']['acc_len']:10.2f} {r['sampled']['acc_len']:11.2f} "
          f"{lp['greedy']:9.3f} {lp['sampled']:10.3f}")
