#!/usr/bin/env bash
# Is the drafter's KV actually dense at DCP4?
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-audit-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"; echo "frozen: ${snap}"
submit() {
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:00:00 --job-name="df2-$1" \
    --output="${here}/results/slurm-$1-%j.out" \
    --error="${here}/results/slurm-$1-%j.err" \
    --wrap "RESULT_DIR=${here}/results DCP=$2 SEQS=$3 \
              VLLM_DFLASH_SLOT_AUDIT=1 VLLM_DFLASH2_PROBE=999999 SAMPLES=4 \
              bash ${snap} $1 dflash2 ${profile} gsm8k"
}
submit audit-dcp4 4 4
submit audit-dcp1 1 1
