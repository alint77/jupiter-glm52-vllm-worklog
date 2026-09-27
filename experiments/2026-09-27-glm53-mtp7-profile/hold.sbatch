#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --job-name=glm-hold
#SBATCH --output=agent_space/experiments/2026-09-27-glm53-mtp7-profile/hold-%j.out

# Standby allocation: work is run on it with `srun --jobid=<id> --overlap`.
echo "holding $(hostname) from $(date +%T)"
sleep infinity  # the job time limit ends the hold
