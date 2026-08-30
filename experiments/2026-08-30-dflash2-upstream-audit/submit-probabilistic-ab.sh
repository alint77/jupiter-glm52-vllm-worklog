#!/usr/bin/env bash
# Does DFlash2's proposal distribution reaching the verifier close the gap?
#
# draft_sample_method defaults to "greedy", which makes the candidate selector
# take an argmax walk: the 16-candidate lattice's distribution q never reaches
# the verifier, and the ratio test degenerates from sum(min(p,q)) to p(argmax q).
# SGLang -- the stack that produces the card's 5.94 -- samples the path per
# request and hands q to verification. MTP is unaffected by this because its
# next-token head is near-peaked, so argmax is close to sampling; DFlash2's
# lattice is not, and it carries an fp32 logits cache whose only consumer is
# the ratio test.
#
# Prerequisite already landed: upstream #54282 salts the draft's Gumbel stream
# (7c1b93ebf5). Without it the proposal and the acceptance draw share a noise
# vector and the resulting number would be uninterpretable.
#
# Three benchmarks, because the gap is already known not to be uniform:
# HumanEval sits 10% under the card while GSM8K sits 32% under it. A fix that
# only moves one of them is not a fix.

set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-prob-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

for task in gsm8k humaneval longcode; do
  sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
    --gres=gpu:4 --cpus-per-task=288 --time=04:00:00 \
    --job-name="dflash2-prob-${task}" \
    --output="${here}/results/slurm-prob-${task}-%j.out" \
    --error="${here}/results/slurm-prob-${task}-%j.err" \
    --wrap "RESULT_DIR=${here}/results DRAFT_SAMPLE_METHOD=probabilistic \
              bash ${snap} prob-${task} dflash2 ${profile} ${task}"
done
