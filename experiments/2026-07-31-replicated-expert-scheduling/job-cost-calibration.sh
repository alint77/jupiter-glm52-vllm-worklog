#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=replica-cost-cal
#SBATCH --output=agent_space/experiments/2026-07-31-replicated-expert-scheduling/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-31-replicated-expert-scheduling/slurm-%x-%j.err

set -euo pipefail

repo=/e/project1/profound/alint77/vllm
result_dir="${repo}/agent_space/experiments/2026-07-31-replicated-expert-scheduling"

cd "${repo}"
source agent_space/jupiter-env.sh

.venv/bin/python -m pytest \
  tests/engine/test_arg_utils.py::test_tiered_moe_config_enforces_reserves_and_enablement \
  tests/engine/test_arg_utils.py::test_replica_assignment_rejects_nonidentical_routing_layout \
  tests/model_executor/model_loader/test_tiered_moe_manifest.py::test_greedy_replica_assignment_balances_predicted_rank_span \
  tests/model_executor/model_loader/test_tiered_moe_manifest.py::test_replica_route_check_rejects_divergent_rank_routes \
  tests/model_executor/model_loader/test_tiered_moe_manifest.py::test_replica_route_check_accepts_identical_rank_routes \
  tests/model_executor/model_loader/test_tiered_moe_manifest.py::test_gpu_greedy_replica_assignment_executes_every_route_once \
  -q

numa_node="$(
  .venv/bin/python \
    agent_space/experiments/2026-07-29-marlin-smem-monopoly/detect_numa.py 0
)"
echo "GPU0 paired NUMA node: ${numa_node}"
numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
  .venv/bin/python benchmarks/kernels/benchmark_moe_wna16_marlin_decode.py \
  --mode cost-calibration \
  --tokens 4 16 \
  --numa-node "${numa_node}" \
  --output "${result_dir}/cost-calibration-${SLURM_JOB_ID}.json"
