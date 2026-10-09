#!/usr/bin/env bash
# One iteration on a kdev hold (NUMA node 0 / GPU 0 unless GPU=n):
#   step.sh "<variant>" <m> "<cells>" [trace-cell] [ncu-cell]
# builds, benches the cells, a traced single call (TD_CTA_TRACE build), and
# an ncu report of the GEMM kernels at ncu-cell.
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
G=${GPU:-0}
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
mkdir -p $TMPDIR logs
V=$1 M=$2 CELLS=$3 TC=${4:-} NC=${5:-}
tag=$(echo "$V" | md5sum | cut -c1-8)-m$M-g$G
P="numactl --cpunodebind=$G --membind=$G /e/project1/profound/alint77/vllm/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=$G
$P kdev.py bench --v "$V" --m $M --numa-node $G --cells $CELLS 2>logs/$tag.err | grep "^{"
if [[ -n $TC ]]; then
  $P kdev.py once --v "$V TD_CTA_TRACE" --m $M --numa-node $G --cell $TC --n 1 --trace logs/$tag.trace.pt \
    2>>logs/$tag.err >/dev/null && python3 -c "print()" && \
    /e/project1/profound/alint77/vllm/.venv/bin/python tl.py logs/$tag.trace.pt
fi
if [[ -n $NC ]]; then
  ncu --target-processes all --profile-from-start off -k regex:"gemm_kernel|act_kernel|route_prep|finalize" \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats \
    --section LaunchStats --section Occupancy \
    --metrics dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_srcunit_tex_aperture_sysmem_op_read.sum,lts__t_sectors_aperture_sysmem_op_read.sum,gpu__time_duration.sum \
    -o logs/$tag-ncu -f $P kdev.py once --v "$V" --m $M --numa-node $G --cell $NC --n 1 >/dev/null 2>>logs/$tag.err
  ncu --import logs/$tag-ncu.ncu-rep --page raw --csv --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors_aperture_sysmem_op_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed 2>/dev/null | cut -c1-400
fi
