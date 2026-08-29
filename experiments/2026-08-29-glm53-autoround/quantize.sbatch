#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=24:00:00
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
# Layer 78 is the MTP block and IS quantized there -- worth 1.21 GiB per rank
# against the 4.50 GiB the NVFP4 checkpoint spends leaving it BF16.

set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
here="${repo_dir}/agent_space/experiments/2026-08-29-glm53-autoround"
venv=/e/fscratch/profound/naeimitabiei1/venvs/autoround
src=/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-BF16
out=/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-AutoRound-W4G64-MTP

cd "${repo_dir}"
source agent_space/jupiter-env.sh

export HF_HOME=/e/fscratch/profound/naeimitabiei1/caches/hf
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRITON_CACHE_DIR=/e/fscratch/profound/naeimitabiei1/caches/triton-autoround
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p "${out}" "${HF_HOME}" "${TRITON_CACHE_DIR}"

echo "node:   $(hostname)"
echo "source: ${src}  ($(du -sh "${src}" 2>/dev/null | cut -f1))"
echo "output: ${out}"
"${venv}/bin/python" -c "import auto_round;print('auto_round', auto_round.__version__)"

# Layers 3..78: the routed-expert blocks plus the MTP block.
blocks="$(seq -s, -f 'model.layers.%g' 3 78)"

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
