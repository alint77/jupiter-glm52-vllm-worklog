#!/usr/bin/env bash
# Turn the real-usage routing capture into a shippable GLM-5.3 W4A16 profile.
#
#   ./finalize.sh [trace-dir] [hot-slots-per-rank] [replica-budget]
#
# Mirrors 2026-08-29-glm53-routing-capture/finalize.sh, retargeted at the
# W4A16 checkpoint the c4 host now serves and at 2496 hot slots per rank, so
# the output is a drop-in replacement for agent_space/profiles/
# glm53-w4a16-2496.json with the ranking as the only difference.

set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir=/e/project1/profound/alint77/vllm
python="${repo_dir}/.venv/bin/python"
capture="${repo_dir}/agent_space/experiments/2026-08-29-glm53-routing-capture"

trace_dir="${1:-/e/fscratch/profound/${USER:-$(id -un)}/caches/routes/snap-1535650-a}"
slots="${2:-2496}"
budget="${3:-985}"
out_dir="${here}/results-$(basename -- "${trace_dir}")"
model="${TIERED_MOE_MODEL_PATH:-/e/fscratch/profound/${USER:-$(id -un)}/models/GLM-5.3-W4A16}"
shipped="${repo_dir}/agent_space/profiles/glm53-w4a16-2496.json"

mkdir -p "${out_dir}"

# manifest.json already exists if describe_routing.py has been run; regenerate
# only when absent so the analysed split and the shipped split are the same.
[[ -s "${trace_dir}/manifest.json" ]] || \
  "${python}" "${capture}/traces_to_manifest.py" --trace-dir "${trace_dir}" \
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

# 985 copies per rank across four ranks == the shipped profile's 3940 replicas,
# so the Grace-side budget is unchanged and the ranking is the only variable.
"${python}" "${repo_dir}/agent_space/experiments/2026-07-31-replicated-expert-scheduling/oracle.py" \
  --trace-dir "${trace_dir}" \
  --profile "${out_dir}/tail-profile.json" \
  --hot-slots-per-rank "${slots}" \
  --budgets "${budget}" \
  --placement-dir "${out_dir}" \
  --output "${out_dir}/oracle-report.json"

PYTHONPATH="${repo_dir}" "${python}" "${capture}/compare_rankings.py" \
  --trace-dir "${trace_dir}" \
  --profile "shipped-synthetic=${shipped}" \
  --profile "realusage-frequency=${out_dir}/frequency-profile.json" \
  --profile "realusage-tail=${out_dir}/tail-profile.json" \
  --profile "realusage-replicas=${out_dir}/replicas-${budget}.json" \
  --output "${out_dir}/comparison.json" \
  >/dev/null
printf 'wrote comparison.json\n'

"${python}" - "${out_dir}" "${budget}" "${model}" "${shipped}" <<'PY'
import json, sys
from pathlib import Path

from vllm.model_executor.model_loader.tiered_moe_manifest import (
    build_glm_w4a16_manifest,
)
from vllm.model_executor.model_loader.tiered_moe_placement import (
    load_tiered_moe_placement_profile,
)

out_dir, budget, model, shipped = (
    Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
)
manifest = build_glm_w4a16_manifest(model)
final = out_dir / f"replicas-{budget}.json"
profile = load_tiered_moe_placement_profile(final, manifest, 4)
reference = load_tiered_moe_placement_profile(shipped, manifest, 4)
print(f"\n{final} validates against the W4A16 checkpoint")
hot = sum(len(h) for h in profile.hot_experts)
rep = sum(1 for row in profile.secondary_ranks for r in row if r >= 0)
hot_ref = sum(len(h) for h in reference.hot_experts)
rep_ref = sum(1 for row in reference.secondary_ranks for r in row if r >= 0)
print(f"  hot experts {hot} (shipped {hot_ref})")
print(f"  replicas    {rep} (shipped {rep_ref})")
assert hot == hot_ref, "hot budget must match the shipped profile"

comparison = json.loads((out_dir / "comparison.json").read_text())
print(f"\nheld-out split: {comparison['requests']} requests, "
      f"{comparison['routed_positions']} routed positions")
for label, metrics in comparison["profiles"].items():
    print(f"  {label:22s} cold-hit {metrics['routing_cold_hit_rate']:.4f}  "
          f"tail {metrics['tail_objective']:.3f}")
for pair, stats in comparison.get("hot_set_overlap", {}).items():
    print(f"  overlap {pair}: {stats['hot_expert_overlap']:.3f}")
PY
