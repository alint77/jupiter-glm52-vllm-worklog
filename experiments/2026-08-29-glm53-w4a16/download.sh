#!/usr/bin/env bash
# JANGQ-AI/GLM-5.3-W4A16 -- compressed-tensors pack-quantized, group 32.
set -euo pipefail
VENV=/e/fscratch/profound/naeimitabiei1/venvs/autoround
DEST=/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-W4A16
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/e/fscratch/profound/naeimitabiei1/caches/hf
mkdir -p "${DEST}" "${HF_HOME}"
exec "${VENV}/bin/hf" download JANGQ-AI/GLM-5.3-W4A16 \
  --local-dir "${DEST}" --max-workers 16
