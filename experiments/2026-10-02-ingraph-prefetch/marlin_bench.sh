#!/usr/bin/env bash
# Isolated Marlin MoE prefill benchmark + Nsight Compute, on a held node.
#   ./onnode.sh <D>/marlin_bench.sh
cd /e/project1/profound/alint77/vllm
OUT=/e/fscratch/profound/${USER}/ingraph-prefetch/marlin; mkdir -p "${OUT}"
export TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/cache/triton
NCU=/e/software/default/stages/2026/software/Nsight-Compute/2025.3.1-GCCcore-14.3.0/ncu
B=benchmarks/kernels/benchmark_moe_wna16_marlin_prefill.py
T="8 512 1024 2048 4096"
numactl --cpunodebind=0 --membind=0 .venv/bin/python $B --tokens $T 2>&1 | grep -v -i warn | tee "${OUT}/bench.txt"
numactl --cpunodebind=0 --membind=0 "${NCU}" --target-processes all -k regex:Marlin --clock-control none \
  --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section SchedulerStats \
  -f -o "${OUT}/marlin" .venv/bin/python $B --ncu --tokens $T > "${OUT}/ncu.log" 2>&1
"${NCU}" --import "${OUT}/marlin.ncu-rep" --page details --csv > "${OUT}/marlin_details.csv"
echo "=== marlin done $(date +%T)"
