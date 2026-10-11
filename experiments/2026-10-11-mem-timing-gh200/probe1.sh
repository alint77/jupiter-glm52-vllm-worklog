#!/usr/bin/env bash
# probe1: read-only GH200 memory inventory + HBM access-pattern bandwidth on
# one GPU. Run via node-submit.sh, which pins CUDA_VISIBLE_DEVICES. The
# contrast between streaming patterns and random gathers at several access
# granularities is the readout for how much DRAM timing state each pattern
# is exposed to (row activation/precharge vs plain sector streaming).
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
cd "$repo_dir"
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-10-11-mem-timing-gh200
gpu="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${E}/logs"
ts="$(date +%H%M%S)"
inv="${E}/logs/probe1-inventory-${ts}.txt"
{
  echo "# probe1 inventory $(date -Is) node=$(hostname -s) cvd=${gpu} job=${SLURM_JOB_ID} step=${SLURM_STEP_ID}"
  nvidia-smi
  echo; nvidia-smi -L
  echo; nvidia-smi topo -m
  echo; nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.mem,clocks.mem,clocks.sm,power.draw,temperature.gpu --format=csv
  echo; echo '## SUPPORTED_CLOCKS'; nvidia-smi -q -d SUPPORTED_CLOCKS
  echo; echo '## ECC'; nvidia-smi -q -d ECC
} >"${inv}" 2>&1
echo "inventory: ${inv}"

numa="$(nvidia-smi topo -m | awk -v d="GPU${gpu}" '$1==d {print $NF}' | head -1)"
out="${E}/logs/probe1-bw-${ts}.json"
if command -v numactl >/dev/null 2>&1 && [ -n "${numa}" ]; then
  echo "binding to NUMA node ${numa} for GPU ${gpu}"
  numactl --cpunodebind="${numa}" --membind="${numa}" \
    .venv/bin/python "${E}/probe1.py" --out "${out}"
else
  echo "numactl or NUMA affinity unavailable, running unbound"
  .venv/bin/python "${E}/probe1.py" --out "${out}"
fi
nvidia-smi --query-gpu=index,clocks.mem,clocks.sm,power.draw,temperature.gpu \
  --format=csv >"${E}/logs/probe1-postclocks-${ts}.txt"
