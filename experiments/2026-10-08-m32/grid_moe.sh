#!/usr/bin/env bash
# All 95%-cells per M on one node, one GPU per M (NUMA-bound to its Grace
# node), 2 rounds alternating kernel order:
#   GPU 0: M=8  decode (this tree) vs decode@HEAD
#   GPU 1: M=16 decode vs prefill;  GPU 2: M=32 decode vs prefill
#   ./onnode.sh <D>/grid_moe.sh <tag>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-m32
B=$D/bench_moe_m32.py
export TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-m32; mkdir -p $TMPDIR
run() {  # gpu m "argsA" "argsB"
  for rep in 1 2; do
    if (( rep % 2 )); then o=("$3" "$4"); else o=("$4" "$3"); fi
    for args in "${o[@]}"; do
      CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 .venv/bin/python $B --m $2 \
        --numa-node $1 $args 2>>$D/grid-$TAG-m$2.err | grep "^{" | sed "s/}$/, \"rep\": $rep}/" >> $D/grid-$TAG-m$2.jsonl
    done
  done
}
TAG=$1
run 0 8 "--kernel decode" "--kernel decode --rev HEAD" &
run 1 16 "--kernel decode" "--kernel prefill" &
run 2 32 "--kernel decode" "--kernel prefill" &
wait
echo "=== done $(date +%T)"
