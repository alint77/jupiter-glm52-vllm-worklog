#!/usr/bin/env bash
# Build the prefill kernel per define set; print wgmma serialization warnings
# and register/spill lines of the wide mode-0 GEMMs (96, 128).
cd /e/project1/profound/alint77/vllm; export PATH=$PWD/.venv/bin:$PATH
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/prefill-kernel-iter
for d in "$@"; do
  echo "== ${d:-(default)}"
  VLLM_TIERED_PREFILL_BUILD_VERBOSE=1 VLLM_TIERED_PREFILL_DEFINES="$d" .venv/bin/python -c \
    "from vllm.model_executor.layers.fused_moe import tiered_prefill as t; t._extension()" 2>&1 |
    awk '/Compiling entry function/ {k=$0; sub(/.*function ./,"",k); sub(/. for.*/,"",k)}
         /C75[0-9][0-9]/ {print "  " $0}
         /Used [0-9]+ registers/ && (k ~ /gemm_kernelILi[0-9]+ELi0E/) {print "  " k ": " $0}
         /spill/ && (k ~ /gemm_kernelILi[0-9]+ELi0E/) && !/0 bytes spill stores, 0 bytes spill loads/ {print "  SPILL " k ": " $0}' | sort -u
done
