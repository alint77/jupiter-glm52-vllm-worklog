#!/usr/bin/env bash
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
NV=$(dirname $(/e/project1/profound/alint77/vllm/.venv/bin/python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"))/$(basename $(/e/project1/profound/alint77/vllm/.venv/bin/python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"))/bin/nvcc
for v in td_v32 td_v33; do echo "## $v"; $NV -O3 -gencode=arch=compute_90a,code=sm_90a -std=c++17 -Xptxas -v -c kernels/$v.cu -o /dev/null 2>&1 | grep -A2 "layer_kernel" | grep -E "registers|spill|layer"; done
