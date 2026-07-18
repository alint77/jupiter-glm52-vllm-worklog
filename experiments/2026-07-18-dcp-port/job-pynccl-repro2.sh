#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=agent_space/experiments/2026-07-18-dcp-port/slurm-%x-%j.out
#SBATCH --error=agent_space/experiments/2026-07-18-dcp-port/slurm-%x-%j.err

set -uo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29511
srun --ntasks=4 --cpu-bind=none \
  .venv/bin/python agent_space/experiments/2026-07-18-dcp-port/pynccl_capture_repro2.py
echo "repro exit: $?"
