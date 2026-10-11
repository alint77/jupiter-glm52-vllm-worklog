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

# Topo rows end with "... CPU Affinity NUMA Affinity GPU NUMA ID": $(NF-1) is
# the CPU/NUMA node (0-3, one per superchip); $NF is the memory-only GPU-HBM
# node (4/12/20/28) with no schedulable CPUs.
numa="$(nvidia-smi topo -m | awk -v d="GPU${gpu}" '$1==d {print $(NF-1)}' | head -1)"
out="${E}/logs/probe1-bw-${ts}.json"
bind=(numactl --cpunodebind="${numa}" --membind="${numa}")
if ! command -v numactl >/dev/null 2>&1 || [ -z "${numa}" ]; then
  bind=()
  echo "numactl or NUMA affinity unavailable, running unbound"
elif numactl --cpunodebind="${numa}" --membind="${numa}" true 2>/dev/null; then
  echo "binding to NUMA node ${numa} for GPU ${gpu}"
  numactl -H 2>/dev/null | grep -E "node [0-9]+ (cpus|size):" | head -8
else
  bind=()
  echo "WARN: cpunodebind=${numa} failed, running unbound"
fi
"${bind[@]}" .venv/bin/python "${E}/probe1.py" --out "${out}"
nvidia-smi --query-gpu=index,clocks.mem,clocks.sm,power.draw,temperature.gpu \
  --format=csv >"${E}/logs/probe1-postclocks-${ts}.txt"
