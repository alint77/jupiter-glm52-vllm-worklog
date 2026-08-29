#!/usr/bin/env bash
# Pull zai-org/GLM-5.3-BF16 (753.3B params, ~1.5 TB) to fscratch.
# Resumable: hf download skips files already complete, so re-running after an
# interruption continues rather than restarting.
set -euo pipefail
VENV=/e/fscratch/profound/naeimitabiei1/venvs/autoround
DEST=/e/fscratch/profound/naeimitabiei1/models/GLM-5.3-BF16
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/e/fscratch/profound/naeimitabiei1/caches/hf
mkdir -p "${DEST}" "${HF_HOME}"
exec "${VENV}/bin/hf" download zai-org/GLM-5.3-BF16 \
  --local-dir "${DEST}" --max-workers 16
