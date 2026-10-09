#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --job-name=sliced-boot
#SBATCH --output=/e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts/logs/serve/boot-%j.out
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh >/dev/null 2>&1
export WT=/e/fscratch/profound/naeimitabiei1/worktrees/tp-sliced-a TAG=a1-${SLURM_JOB_ID}
srun --ntasks=1 --gres=gpu:4 --cpus-per-task=288 bash agent_space/experiments/2026-10-09-tp-sliced-experts/boot1.sh
