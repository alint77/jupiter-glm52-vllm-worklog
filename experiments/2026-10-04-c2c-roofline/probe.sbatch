#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --job-name=c2c-roofline
#SBATCH --output=/e/fscratch/profound/naeimitabiei1/holds/c2c-roofline-%j.out
# The C2C roofline probe as its own short job (backfills more easily than a hold).
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/prefill-kernel-iter
export TORCH_EXTENSIONS_DIR=${VLLM_CACHE_ROOT}/torch_extensions PATH=${PWD}/.venv/bin:${PATH}
numactl --cpunodebind=0 --membind=0 .venv/bin/python agent_space/experiments/2026-10-04-c2c-roofline/c2c_roofline.py 2>&1 \
  | grep -v -i warn > agent_space/experiments/2026-10-04-c2c-roofline/run-${SLURM_JOB_ID}.log
