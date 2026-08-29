#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --job-name=glm53-autoround
#SBATCH --output=agent_space/experiments/2026-08-29-glm53-autoround/slurm-%j.out
#SBATCH --error=agent_space/experiments/2026-08-29-glm53-autoround/slurm-%j.err

# AutoRound W4A16 G64 for GLM-5.3, reproducing the recipe of the HF GLM-5.2
# quant (GLM-5.2-AutoRound-W4G64-MTP-e1ba887) tensor for tensor.
#
# Every flag below is a value that checkpoint's quantization_config records.
# The parameters it does NOT record -- iters, lr, minmax_lr, seqlen, dataset --
# are deliberately left unset: auto-round is pinned to 0.14.0, the same version
# that produced the 5.2 quant, so its own defaults fill them in identically.
# Overriding them with guesses would diverge from the recipe, not match it.
#
# Structure confirmed against the 5.2 weight index: only routed expert MLPs in
# layers 3-78 carry qweight/qzeros/scales. Router gates, shared experts, all
# attention including the DSA indexer, eh_proj and dense layers 0-2 stay BF16.
# The 5.2 checkpoint quantizes the MTP block (78); this one does not, by
# instruction. Layer 78 is also absent from the transformers graph anyway.

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
here="${repo_dir}/agent_space/experiments/2026-08-29-glm53-autoround"
venv=/e/fscratch/profound/naeimitabiei1/venvs/autoround
src=/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-BF16
out=/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-AutoRound-W4G64-MTP

cd "${repo_dir}"
source agent_space/jupiter-env.sh
# jupiter-env.sh activates the vLLM venv and sets HF_HUB_OFFLINE=1. Drop the
# venv vars so the isolated auto-round env wins; keep the offline flags -- the
# calibration set (NeelNanda/pile-10k) is pre-cached under HF_HOME.
unset VIRTUAL_ENV PYTHONPATH

export HF_HOME=/e/fscratch/profound/naeimitabiei1/caches/hf
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRITON_CACHE_DIR=/e/fscratch/profound/naeimitabiei1/caches/triton-autoround
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p "${out}" "${HF_HOME}" "${TRITON_CACHE_DIR}"

echo "node:   $(hostname)"
echo "source: ${src}"
echo "output: ${out}"
"${venv}/bin/python" -c "import auto_round;print('auto_round', auto_round.__version__)"

# Layers 3..77: every routed-expert block. The MTP head (78) is deliberately
# left BF16 -- it is also absent from the HF graph, so this needs no exclusion.
blocks="$(seq -s, -f 'model.layers.%g' 3 77)"

exec "${venv}/bin/auto-round" \
  --model_name "${src}" \
  --output_dir "${out}" \
  --bits 4 \
  --group_size 64 \
  --data_type int \
  --batch_size 2 \
  --gradient_accumulate_steps 4 \
  --nsamples 512 \
  --low_gpu_mem_usage \
  --device_map auto \
  --format "auto_round:auto_gptq" \
  --layer_config "${here}/layer-config.json" \
  --to_quant_block_names "${blocks}"
