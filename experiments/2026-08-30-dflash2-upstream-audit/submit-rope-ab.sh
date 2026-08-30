#!/usr/bin/env bash
# P1: does the draft's RoPE layout explain DFlash2's 32% acceptance gap?
#
# Upstream #51655 copies the target's rotary layout into the draft head before
# constructing it, because a DFlash head must rotate Q/K the way the target it
# was distilled against does; a mismatch is silent, since the target verifies
# every drafted token, so output stays correct while acceptance degrades. This
# fork had no such copy and get_rope defaults to NeoX, while every rotary in a
# DeepseekV2-derived GLM-5.3 target is interleaved.
#
# Two arms, one allocation, everything else identical to the Phase 43 run:
#
#   rope-neox         VLLM_DFLASH_DRAFT_ROPE_NEOX=1, the pre-fix behaviour
#   rope-interleaved  unset, so the layout is derived from the target
#
# The neox arm is also a reproduction check: it must land near Phase 43's
# 4.0293, and if it does not, something else moved and the comparison is void.
#
# Gates: 4.0293 (Phase 43 post-fix), 5.94 (card, SGLang/GB300), 4.9121 (our
# MTP7 control at the same width).

set -euo pipefail

repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"

[[ -s "${arm}" ]] || { printf 'missing arm: %s\n' "${arm}" >&2; exit 1; }
[[ -s "${profile}" ]] || { printf 'missing profile: %s\n' "${profile}" >&2; exit 1; }

snap="${here}/snapshots/arm-replicate-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=04:00:00 \
  --job-name=dflash2-rope-ab \
  --output="${here}/results/slurm-%j.out" \
  --error="${here}/results/slurm-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH_DRAFT_ROPE_NEOX=1 \
            bash ${snap} rope-neox dflash2 ${profile} gsm8k
          RESULT_DIR=${here}/results \
            bash ${snap} rope-interleaved dflash2 ${profile} gsm8k"
