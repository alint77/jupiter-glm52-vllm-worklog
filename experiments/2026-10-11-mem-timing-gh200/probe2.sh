#!/usr/bin/env bash
# probe2: HBM random-gather granularity sweep (2B..64KiB), cold rows vs hot
# rows. Cold = uniform random indices over 8 GiB (row-miss dominated);
# hot = indices resampled from a fixed 4K-row set (open-page hits dominate).
# The cold/hot gap at fine granularity is the row-activation tax that DRAM
# timing parameters (tRP/tRCD/tRC/tFAW) act on.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." 2>/dev/null && pwd)"
[[ -f "${repo_dir}/agent_space/jupiter-env.sh" ]] || repo_dir=/e/project1/profound/alint77/vllm
cd "$repo_dir"
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-10-11-mem-timing-gh200
gpu="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${E}/logs"
ts="$(date +%H%M%S)"
numa="$(nvidia-smi topo -m | awk -v d="GPU${gpu}" '$1==d {print $(NF-1)}' | head -1)"
out="${E}/logs/probe2-sweep-${ts}.json"
bind=(numactl --cpunodebind="${numa}" --membind="${numa}")
if ! command -v numactl >/dev/null 2>&1 || [ -z "${numa}" ]; then
  bind=(); echo "running unbound"
elif numactl --cpunodebind="${numa}" --membind="${numa}" true 2>/dev/null; then
  echo "binding to NUMA node ${numa} for GPU ${gpu}"
else
  bind=(); echo "WARN: bind to node ${numa} failed, running unbound"
fi
nvidia-smi --query-gpu=index,clocks.current.memory,clocks.current.sm,power.draw,temperature.gpu --format=csv
"${bind[@]}" .venv/bin/python "${E}/probe2.py" --out "${out}"
nvidia-smi --query-gpu=index,clocks.current.memory,clocks.current.sm,power.draw,temperature.gpu --format=csv
