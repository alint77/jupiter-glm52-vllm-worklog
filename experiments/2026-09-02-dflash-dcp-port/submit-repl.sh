#!/usr/bin/env bash
# Replicated drafter KV (e796779f6c): DCP4 with the drafter's group opting out
# of DCP sharding, plus a DCP1 control.
#
# Expect repl-eager-dcp4 ~= 5.65 (the DCP1 figure): the drafter's attention path
# is untouched, only its cache stops being sharded. Landing near 1.76 again
# means the opt-out did not reach the runtime; landing in between means some
# other DCP coupling remains.
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-repl-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"; echo "frozen: ${snap}"
submit() {
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 --job-name="df2-$1" \
    --output="${here}/results/slurm-$1-%j.out" \
    --error="${here}/results/slurm-$1-%j.err" \
    --wrap "RESULT_DIR=${here}/results DCP=$2 SEQS=$3 $4 \
              bash ${snap} $1 dflash2 ${profile} gsm8k"
}
submit repl2-eager-dcp4 4 4 "VLLM_DFLASH2_PROBE=999999"
submit repl2-eager-dcp1 1 1 "VLLM_DFLASH2_PROBE=999999"
