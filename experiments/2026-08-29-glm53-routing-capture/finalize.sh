#!/usr/bin/env bash
# Turn a finished routing capture into the GLM-5.3 placement profile and the
# evidence for it.
#
#   ./finalize.sh [job-id] [hot-slots-per-rank] [replica-budget]
#
# Produces, under results-<job>/:
#   manifest.json                 trace split, written back into the trace dir
#   <mode>-profile.json           version 1 profiles, frequency and tail residency
#   <mode>-report.json            train/held-out metrics per mode
#   replicas-<budget>.json        version 2 profile, the shippable artefact
#   comparison.json               every profile scored on the same held-out split
#   grid/                         hotness counts, ranking, heatmap

set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir=/e/project1/profound/alint77/vllm
python="${repo_dir}/.venv/bin/python"
state_dir="/e/scratch/profound/${USER:-$(id -un)}/claude-glm53-capture"

job="${1:-$(<"${state_dir}/job-id")}"
slots="${2:-2400}"
budget="${3:-985}"
trace_dir="${state_dir}/routes-${job}"
out_dir="${here}/results-${job}"
model="${TIERED_MOE_MODEL_PATH:-/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-NVFP4}"
shipped="${repo_dir}/agent_space/experiments/2026-08-28-nvfp4-tiered/nvfp4-profile-2400.json"

mkdir -p "${out_dir}"

"${python}" "${here}/traces_to_manifest.py" \
  --trace-dir "${trace_dir}" \
  --attribution "${state_dir}/driver-${job}/requests.jsonl" \
  | tee "${out_dir}/capture-summary.json"

for mode in frequency tail; do
  "${python}" "${repo_dir}/agent_space/benchmarks/optimize_routing_profile.py" \
    --trace-dir "${trace_dir}" \
    --model "${model}" \
    --hot-slots-per-rank "${slots}" \
    --residency-mode "${mode}" \
    --output-profile "${out_dir}/${mode}-profile.json" \
    --output-report "${out_dir}/${mode}-report.json" \
    >/dev/null
  printf 'wrote %s-profile.json\n' "${mode}"
done

# Replicas are charged to Grace, so the budget matches the shipped profile's
# 3940 non-(-1) secondary ranks: 985 copies per rank across four ranks.
"${python}" "${repo_dir}/agent_space/experiments/2026-07-31-replicated-expert-scheduling/oracle.py" \
  --trace-dir "${trace_dir}" \
  --profile "${out_dir}/tail-profile.json" \
  --hot-slots-per-rank "${slots}" \
  --budgets "${budget}" \
  --placement-dir "${out_dir}" \
  --output "${out_dir}/oracle-report.json"

PYTHONPATH="${repo_dir}" "${python}" "${here}/compare_rankings.py" \
  --trace-dir "${trace_dir}" \
  --profile "glm52-placeholder=${shipped}" \
  --profile "rederived-frequency=${out_dir}/frequency-profile.json" \
  --profile "rederived-tail=${out_dir}/tail-profile.json" \
  --profile "rederived-replicas=${out_dir}/replicas-${budget}.json" \
  --output "${out_dir}/comparison.json" \
  >/dev/null
printf 'wrote comparison.json\n'

"${python}" "${repo_dir}/agent_space/benchmarks/build_claude_routing_grid.py" \
  --trace-dir "${trace_dir}" \
  --output-dir "${out_dir}/grid" \
  >"${out_dir}/grid-summary.json"

"${python}" - "${out_dir}" "${budget}" "${model}" <<'PY'
import json, sys
from pathlib import Path

from vllm.model_executor.model_loader.tiered_moe_manifest import (
    build_glm_w4a16_manifest,
)
from vllm.model_executor.model_loader.tiered_moe_placement import (
    load_tiered_moe_placement_profile,
)

out_dir, budget, model = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = build_glm_w4a16_manifest(model)
final = out_dir / f"replicas-{budget}.json"
profile = load_tiered_moe_placement_profile(final, manifest, 4)
comparison = json.loads((out_dir / "comparison.json").read_text())
print(f"\n{final} validates against the NVFP4 checkpoint")
print(f"  hot experts {sum(len(h) for h in profile.hot_experts)}")
print(f"  replicas    {sum(1 for row in profile.secondary_ranks for r in row if r >= 0)}")
print(f"\nheld-out split: {comparison['requests']} requests, "
      f"{comparison['routed_positions']} routed positions")
for label, metrics in comparison["profiles"].items():
    print(f"  {label:24s} cold-hit {metrics['routing_cold_hit_rate']:.4f}  "
          f"tail {metrics['tail_objective']:.3f}")
for pair, stats in comparison["hot_set_overlap"].items():
    print(f"  overlap {pair}: {stats['hot_expert_overlap']:.3f}")
PY
