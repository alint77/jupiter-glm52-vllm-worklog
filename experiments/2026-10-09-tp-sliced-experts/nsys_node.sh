#!/usr/bin/env bash
# NSYS_BIN for serve.sh's NSYS_OUT mode: kernels inside CUDA graphs traced
# per node, and every profile window (cudaProfilerApi range) its own report.
exec /e/software/default/stages/2026/software/Nsight-Systems/2025.5.1-GCCcore-14.3.0/bin/nsys \
  "${1}" $(printf '%s\n' "${@:2}" | sed 's/^--cuda-graph-trace=graph$/--cuda-graph-trace=node/; s/^--capture-range-end=stop$/--capture-range-end=repeat/')
