#!/usr/bin/env bash
# XiaomiMiMo/MiMo-V2.6-Pro-RL -- ~1T params, 70 layers, 384 routed experts.
# Ships pre-quantized: quant_method fp8, store_dtype mxfp4, block [128,128].
# 132 safetensors shards, 0.52 TiB. Includes a dflash/ drafter subdirectory.
set -euo pipefail
VENV=/e/fscratch/profound/naeimitabiei1/venvs/autoround
DEST=/e/fscratch/profound/naeimitabiei1/models/MiMo-V2.6-Pro-RL
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/e/fscratch/profound/naeimitabiei1/caches/hf
mkdir -p "${DEST}" "${HF_HOME}"
exec "${VENV}/bin/hf" download XiaomiMiMo/MiMo-V2.6-Pro-RL \
  --local-dir "${DEST}" --max-workers 16
