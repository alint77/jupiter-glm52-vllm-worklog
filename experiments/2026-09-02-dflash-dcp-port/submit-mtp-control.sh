#!/usr/bin/env bash
# Does the TARGET lose acceptance under DCP4, independent of the drafter?
#
# DFlash2 lands 3.4434 at DCP4 against 5.6265 at DCP1, depressed ~20% at every
# position including position 0. A uniform hit that includes position 0 points
# at the drafter's inputs -- the target's hidden states, which under DCP4 come
# through the MLA all-gather/LSE-combine -- rather than at the drafter's own
# slot math, which would degrade with depth.
#
# MTP is the control: it shares the target and its hidden states but none of
# the DFlash KV plumbing. If MTP shows a similar DCP4 discount, the deficit is
# the target's DCP path and not the replicated-KV work.
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-mtp-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"; echo "frozen: ${snap}"
submit() {
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 --job-name="mtp-$1" \
    --output="${here}/results/slurm-$1-%j.out" \
    --error="${here}/results/slurm-$1-%j.err" \
    --wrap "RESULT_DIR=${here}/results DCP=$2 SEQS=$3 \
              bash ${snap} $1 mtp ${profile} gsm8k"
}
submit mtp-dcp4 4 4
submit mtp-dcp1 1 1
