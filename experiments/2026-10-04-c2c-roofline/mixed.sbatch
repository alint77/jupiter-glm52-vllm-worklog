#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --job-name=c2c-mixed
#SBATCH --output=/e/fscratch/profound/naeimitabiei1/holds/c2c-mixed-%j.out
# tiered_decode_moe whole-call time over hot,cold mixes (fresh experts per call,
# 1000-expert hot pool in HBM, 100-expert cold pool in Grace), next to the
# cold-only roofline in this directory.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/prefill-kernel-iter
export TORCH_EXTENSIONS_DIR=${VLLM_CACHE_ROOT}/torch_extensions PATH=${PWD}/.venv/bin:${PATH}
D=agent_space/experiments/2026-10-04-c2c-roofline
numactl --cpunodebind=0 --membind=0 .venv/bin/python \
  agent_space/experiments/2026-09-27-glm53-mtp7-profile/bench_fmt.py --fmt int4 \
  --pool-hot 1000 --pool-cold 100 --grid \
  0,1 0,2 0,3 0,4  4,0 9,0 14,0 20,0 \
  4,1 4,2 4,3  9,1 9,2 9,3  14,1 14,2 14,3  20,1 20,2 \
  2>&1 | grep '^{' > ${D}/mixed-${SLURM_JOB_ID}.jsonl
