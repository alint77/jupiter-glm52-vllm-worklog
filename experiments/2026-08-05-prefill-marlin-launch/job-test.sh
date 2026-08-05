#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=prefill-tile-test
#SBATCH --output=agent_space/experiments/2026-08-05-prefill-marlin-launch/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-08-05-prefill-marlin-launch/slurm-%x-%j.err
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
echo "node: $(hostname)"
.venv/bin/python -m pytest tests/kernels/moe/test_moe.py::test_fused_marlin_moe_launch_policy -v 2>&1 | tail -20
echo "=== done ==="
