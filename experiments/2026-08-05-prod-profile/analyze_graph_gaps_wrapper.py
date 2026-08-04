import sys, json
from pathlib import Path
sys.path.insert(0, "/e/project1/profound/alint77/vllm/agent_space/experiments/2026-07-29-marlin-smem-monopoly")
from analyze_graph_gaps import summarize, print_arm
root = Path(sys.argv[1]); out = {}
for label in sys.argv[2:]:
    s = summarize(root / label)
    out[label] = s
    print_arm(label, s)
Path("graph-gaps.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
