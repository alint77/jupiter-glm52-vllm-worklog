#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:50:00
#SBATCH --job-name=mimo26-serve
#SBATCH --output=agent_space/experiments/2026-09-22-mimo-v26-pro/serve-%j.out
#SBATCH --error=agent_space/experiments/2026-09-22-mimo-v26-pro/serve-%j.err

# MiMo-V2.6-Pro-RL on tiered MoE: 250K context, concurrency 1, no speculative
# decoding, EP4/TP4, linear expert placement.
#
# No placement profile exists for this model, so the hot set is a linear split
# rather than a routing-derived one. What tiered buys here is the hot/cold
# overlap and NUMA-pinned expert residency in Grace, not placement quality.
#
# Differences from the GLM launcher, all forced by the architecture:
#   * kv-cache-dtype auto, not fp8_ds_mla -- that dtype is MLA-only and MiMo is
#     GQA. plan_mimo_v2_kv_cache budgets bf16 to match.
#   * max-model-len 250000. The 400K pin is GLM memory arithmetic and is now
#     scoped by model_type rather than relaxed.
#   * trust-remote-code: the config carries an auto_map and transformers does
#     not know model_type mimo_v2 natively.
#   * no placement profile.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh

model=/e/fscratch/profound/${USER}/models/MiMo-V2.6-Pro-RL
port="${MIMO_PORT:-8031}"
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/marlin/vllm-cache-mimo26
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/caches/triton
export TORCHINDUCTOR_CACHE_DIR=/e/fscratch/profound/${USER}/caches/inductor
export FLASHINFER_CACHE_DIR=/e/fscratch/profound/${USER}/caches/flashinfer
mkdir -p "${VLLM_CACHE_ROOT}"

reserve_gb="${MIMO_HBM_RESERVE_GB:-8}"
model_len="${MIMO_MAX_MODEL_LEN:-250000}"
echo "mimo: model_len=${model_len} reserve=${reserve_gb}GB port=${port}"

exec .venv/bin/vllm serve "${model}" \
  --served-model-name mimo26-pro \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "${port}" \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --distributed-executor-backend mp \
  --numa-bind \
  --block-size 64 \
  --max-model-len "${model_len}" \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --optimization-level 2 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1],"compile_sizes":[],"cudagraph_num_of_warmups":1,"pass_config":{"fuse_allreduce_rms":false}}' \
  --enable-tiered-moe \
  --tiered-moe-backend uva \
  --tiered-moe-hbm-reserve-gb "${reserve_gb}" \
  --tiered-moe-host-reserve-gb 8 \
  --tiered-moe-numa-strict \
  --mla-cache-tier hbm \
  --grace-machine-profile agent_space/profiles/jupiter-gh200-baseline.json
