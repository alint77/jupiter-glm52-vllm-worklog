#!/usr/bin/env bash
# Does the fork's DFlash2 draft actually attend non-causally within its block?
#
# The checkpoint sets is_causal=false and the draft is all-SWA, so every query
# slot should see every other slot. This fork expresses that through attention
# METADATA (causal=self._group_causal) while keeping attn_type=DECODER
# (qwen3_dflash.py:301); upstream sglang -- which measures 5.71 where this fork
# measures 3.995 -- instead sets attn_type=ENCODER_ONLY (dflash.py:104-135).
#
# The probe perturbs late query slots and re-runs the forward eagerly: if the
# first mask slot's output moves, attention is non-causal; if it is
# bit-identical, the early slots are blind to the later ones and position 0 --
# where Phase 49f localised the entire deficit -- is starved of context.
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-02-dflash2-causality-probe"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=01:00:00 \
  --job-name=df2-causality \
  --output="${here}/results/slurm-causality-%j.out" \
  --error="${here}/results/slurm-causality-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_CAUSALITY_PROBE=3 SAMPLES=2 \
            bash ${snap} causality dflash2 ${profile} gsm8k"
