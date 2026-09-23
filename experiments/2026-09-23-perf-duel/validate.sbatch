#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=perf-duel-validate
#SBATCH --output=agent_space/experiments/2026-09-23-perf-duel/validate-%j.out
#SBATCH --error=agent_space/experiments/2026-09-23-perf-duel/validate-%j.err
# The unmodified stub (== the baseline) must pass the harness on every shape.
set -euo pipefail
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-09-23-perf-duel/task
CUDA_VISIBLE_DEVICES=0 /e/project1/profound/alint77/vllm/.venv/bin/python bench.py
