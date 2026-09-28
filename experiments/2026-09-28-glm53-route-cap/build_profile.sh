#!/usr/bin/env bash
# GLM-5.3 profile from the MiMo-workload capture, built the way MiMo's was
# (../2026-09-26-mimo-routing-profile): merge the shards, manifest with a
# task-family split, frequency residency, then Grace replicas.
#
#   build_profile.sh <job-id>...
#
# Hot slots: 3239 per GPU, the largest DCP4 runtime count, so the planner only
# demotes; each layer's list is then ordered by training frequency so it
# demotes the least-used first. Owners are kept from the served profile.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
H=agent_space/experiments/2026-09-28-glm53-route-cap
G=agent_space/experiments/2026-08-29-glm53-routing-capture
out=/e/fscratch/profound/${USER}/glm53-route-cap/merged
rm -rf "${out}"; mkdir -p "${out}"
for job in "$@"; do
  r=/e/fscratch/profound/${USER}/glm53-route-cap/${job}
  for f in "${r}"/routes/*.npy; do ln -s "${f}" "${out}/$(basename "${f}")"; done
  cat "${r}/routes/manifest.jsonl" >>"${out}/manifest.jsonl"
  cat "${r}/driver/requests.jsonl" >>"${out}/requests.jsonl"
done
echo "$(wc -l <"${out}/manifest.jsonl") traces"
.venv/bin/python "${G}/traces_to_manifest.py" --trace-dir "${out}" --split-by domain \
  --attribution "${out}/requests.jsonl"
.venv/bin/python agent_space/benchmarks/optimize_routing_profile.py \
  --trace-dir "${out}" --model "/e/fscratch/profound/${USER}/models/GLM-5.3-W4A16" \
  --hot-slots-per-rank 3239 --residency-mode frequency \
  --owners-profile agent_space/profiles/glm53-w4a16-2496.json \
  --output-profile "${H}/profile-3239.json" --output-report "${H}/report-3239.json"
.venv/bin/python "${H}/finish_profile.py" --trace-dir "${out}" \
  --profile "${H}/profile-3239.json" --replicas 2000 \
  --out agent_space/profiles/glm53-w4a16-agentic-3239-r2000.json
