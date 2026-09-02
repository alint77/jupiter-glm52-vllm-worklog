#!/usr/bin/env bash
# DCP4 with the sliding-window fix (4bfece3f59), plus a DCP1 regression control.
# Unique labels per arm: arms share $label-server.out, and reusing a label made
# an earlier watcher read a previous job's output.
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-fix-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"; echo "frozen: ${snap}"
submit() {
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 --job-name="df2-$1" \
    --output="${here}/results/slurm-$1-%j.out" \
    --error="${here}/results/slurm-$1-%j.err" \
    --wrap "RESULT_DIR=${here}/results DCP=$2 SEQS=$3 $4 \
              bash ${snap} $1 dflash2 ${profile} gsm8k"
}
submit fix-eager-dcp4 4 4 "VLLM_DFLASH2_PROBE=999999"
submit fix-graph-dcp4 4 4 ""
submit fix-eager-dcp1 1 1 "VLLM_DFLASH2_PROBE=999999"
