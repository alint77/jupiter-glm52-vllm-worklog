#!/usr/bin/env bash
# Separate DCP from concurrency.
#
# Every DFlash2 comparison so far varied BOTH: the DCP4 arms ran
# max_num_seqs=4 and the DCP1 arms max_num_seqs=1, so 3.4434 vs 5.6265
# cannot be attributed. The slot audit showed the drafter's KV is dense at
# both DCP sizes (0% PAD), which removes the slot-math explanation and makes
# concurrency the leading suspect.
#
# These two arms complete the 2x2 against the existing
# repl2-eager-dcp1 (dcp1/c1, 5.6265) and repl2-eager-dcp4 (dcp4/c4, 3.4434).
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-2x2-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"; echo "frozen: ${snap}"
submit() {
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 --job-name="df2-$1" \
    --output="${here}/results/slurm-$1-%j.out" \
    --error="${here}/results/slurm-$1-%j.err" \
    --wrap "RESULT_DIR=${here}/results DCP=$2 SEQS=$3 VLLM_DFLASH2_PROBE=999999 \
              bash ${snap} $1 dflash2 ${profile} gsm8k"
}
submit x-dcp1-c4 1 4   # concurrency 4 without DCP
submit x-dcp4-c1 4 1   # DCP without concurrency 4
