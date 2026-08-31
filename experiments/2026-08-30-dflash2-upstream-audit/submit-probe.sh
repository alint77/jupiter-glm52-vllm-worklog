#!/usr/bin/env bash
# In-serving numerical probe: the one surface the offline audit cannot reach.
#
# The offline harness proved the draft's MATH is faithful to z-lab/dflash
# (forward rel 2e-5, selector paths identical) using the real weights. What it
# cannot check is the SERVING PATH around that math: whether the context KV
# actually landed in the cache where the draft's attention reads it, at the
# right slots, with the right positions, under TP4 + CUDA graphs + the tiered
# MoE target.
#
# VLLM_DFLASH2_PROBE=N recomputes, on the Nth propose call, the whole
# context-KV chain (hidden_norm -> fused kv_proj -> k_norm -> RoPE) in plain
# torch from the live weights and diffs it against what is actually sitting in
# each layer's KV cache at the slots the draft wrote, plus the selector chain
# (unary logits, lattice scores, walk replay) against reference math.
#
# Reading: bf16 rounding is ~1e-2 relative. Anything at 1e-1 or above, or a
# per-layer pattern (one layer differing, or K differing while V matches ->
# RoPE/k_norm; both differing -> slot mapping or projection), is the bug.

set -euo pipefail
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-08-30-dflash2-upstream-audit"
arm="${repo}/agent_space/experiments/2026-08-28-nvfp4-tiered/arm-replicate.sh"
profile="${PROFILE:-${repo}/agent_space/experiments/2026-08-29-glm53-routing-capture/results-1532971/replicas-985.json}"
mkdir -p "${here}/results" "${here}/snapshots"
snap="${here}/snapshots/arm-probe-$(date +%Y%m%d-%H%M%S).sh"
cp "${arm}" "${snap}"

# Probe on a mid-stream decode call (past warmup, real context, steady state).
sbatch --account=profound --partition=booster --nodes=1 --ntasks=1 \
  --gres=gpu:4 --cpus-per-task=288 --time=01:30:00 \
  --job-name=df2-probe \
  --output="${here}/results/slurm-probe-%j.out" \
  --error="${here}/results/slurm-probe-%j.err" \
  --wrap "RESULT_DIR=${here}/results VLLM_DFLASH2_PROBE=50 \
            bash ${snap} probe dflash2 ${profile} gsm8k"
