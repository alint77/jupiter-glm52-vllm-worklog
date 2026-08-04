import sys, collections
from pathlib import Path
sys.path.insert(0,"/e/project1/profound/alint77/vllm/agent_space/experiments/2026-07-29-marlin-smem-monopoly")
from analyze_step_budget import load_trace, step_windows, family, GPU_CATEGORIES
p = sorted(Path(sys.argv[1]).glob("*rank0*.trace.json.gz"))[0]
events = load_trace(p)
n = len(step_windows(events))
agg = collections.defaultdict(float); cnt = collections.Counter()
for e in events:
    if e.get("cat") in GPU_CATEGORIES:
        key = (family(e["name"]), e["name"][:56])
        agg[key] += e["dur"]/1000.0; cnt[key] += 1
print(f"{n} steps on rank0")
for (fam, name), ms in sorted(agg.items(), key=lambda kv: -kv[1])[:15]:
    print(f"{ms/n:8.3f} ms  x{cnt[(fam,name)]/n:6.1f}  [{fam[:24]:24}] {name}")
