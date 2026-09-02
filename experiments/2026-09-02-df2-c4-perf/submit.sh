#!/usr/bin/env bash
# Prefill + decode at c=4, under real concurrent load.
#
#   df2-dcp1-c4   the config claude-glm53-c4-df2.sh launches
#   mtp3-dcp4-c4  what runs in production today, as the reference
#
# Two arms because one number alone is not interpretable: DFlash2 should win on
# decode via acceptance (5.7046 AL against MTP3's ~4.9) and the question is
# whether DCP1's replicated MLA cache costs enough prefill or residency to
# give that back.
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-df2-c4-perf"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-perf-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-perf.sh" "${snap}"; echo "frozen: ${snap}"
submit() {  # label mode concurrency dcp
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 --job-name="perf-$1" \
    --output="${here}/results/slurm-$1-%j.out" \
    --error="${here}/results/slurm-$1-%j.err" \
    --wrap "bash ${snap} $1 $2 $3 $4"
}
submit df2-dcp1-c4  dflash2 4 1
submit mtp3-dcp4-c4 mtp3    4 4
