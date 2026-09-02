#!/usr/bin/env bash
# Phase 53 arms: does the #52188 DCP port work, and did B/C regress DCP1?
#
# 1-2. DCP1 regression pair. The gates prove the prepare kernel is
#      bit-identical at cp_size=1, but they test that kernel in isolation --
#      only an end-to-end arm shows commit B (rejected suffix rows masked to
#      PAD) did not move acceptance. Compare against Phase 52's paired
#      eager 5.7190 / graph 3.9626 on the same protocol.
# 3-4. The prod config the NotImplementedError used to refuse: DCP4, c=4,
#      400K, DFlash2. Run both draft-graph states, since the graph defect is
#      still open and worth ~44% acceptance.
#
# VLLM_DFLASH2_PROBE is set past the end of the run in the eager arms: it
# forces need_eager without the probe ever firing (Phase 52's method).
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash-dcp-port"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
# Freeze the harness per submission: editing a live arm script has destroyed
# running arms before.
snap="${here}/snapshots/arm-$(date +%Y%m%d-%H%M%S).sh"
cp "${here}/arm-dcp.sh" "${snap}"
echo "frozen: ${snap}"

submit() {  # name dcp seqs extra_env
  local name="$1" dcp="$2" seqs="$3" extra="$4"
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 \
    --job-name="df2-${name}" \
    --output="${here}/results/slurm-${name}-%j.out" \
    --error="${here}/results/slurm-${name}-%j.err" \
    --wrap "RESULT_DIR=${here}/results DCP=${dcp} SEQS=${seqs} ${extra} \
              bash ${snap} ${name} dflash2 ${profile} gsm8k"
}

submit bc-eager-dcp1 1 1 "VLLM_DFLASH2_PROBE=999999"
submit bc-graph-dcp1 1 1 ""
submit prod-eager-dcp4 4 4 "VLLM_DFLASH2_PROBE=999999"
submit prod-graph-dcp4 4 4 ""
