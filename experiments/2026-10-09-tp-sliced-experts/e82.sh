#!/usr/bin/env bash
# Roofline position per concurrency: served kernel, all hot (MTP3 mean expert counts),
# instructions, issue, tensor-pipe use and DRAM bytes per call. One M per GPU.
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
V="td_v57:TD_MAX_TOKENS=64"
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "$V"
MET=gpu__time_duration.sum,dram__bytes_read.sum,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor_op_hmma.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_lsu.sum,sm__cycles_elapsed.avg.per_second,smsp__warps_active.avg.per_cycle_active
run() { g=$1 m=$2 c=$3
  CUDA_VISIBLE_DEVICES=$g ncu --target-processes all -k regex:"layer_kernel" --metrics $MET --csv \
    numactl --cpunodebind=$g --membind=$g $PY kdev.py once --v "$V" --m $m --cell $c --n 1 --shared 1 --numa-node $g \
    2>logs/e82-$m.err | grep layer_kernel > logs/e82-$m.csv
}
run 0 4 25,0 & run 1 16 78,0 & run 2 32 125,0 & run 3 64 177,0 & wait
run 0 8 45,0 & wait
for m in 4 8 16 32 64; do echo "== M=$m"; awk -F'","' '{print $(NF-2), $NF}' logs/e82-$m.csv | sed 's/"//g' | sort -u; done
