#!/usr/bin/env bash
# The tiered decode MoE kernel (this tree) over the COST_US grid (hot 0..24 x
# cold 0..6), INT4 (GLM) and MXFP4 (MiMo, the format the shipped table was
# measured in), production-sized pools (GLM prod: ~49 hot / ~34 cold slots per
# layer per rank), 3 repeats alternating the format order. NUMA-bound to GPU 0's
# Grace node. JSON lines -> grid-<job>.jsonl.
#   ./onnode.sh <G>/grid.sh <job>
cd /e/project1/profound/alint77/vllm
G=agent_space/experiments/2026-10-08-moe-cost-table
B=agent_space/experiments/2026-10-08-moe-cost-table/bench_fmt_distinct.py
cells=""
for h in $(seq 0 24); do for c in $(seq 0 6); do
  [[ $h == 0 && $c == 0 ]] || cells="$cells $h,$c"
done; done
for rep in 1 2 3; do
  if (( rep % 2 )); then order="int4 mxfp4"; else order="mxfp4 int4"; fi
  for fmt in $order; do
    CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 .venv/bin/python $B --fmt $fmt \
      --numa-node 0 --pool-hot 49 --pool-cold 34 --grid $cells 2>>$G/grid-$1.err \
      | grep "^{" | sed "s/}$/, \"rep\": $rep}/" >> $G/grid-$1.jsonl
  done
done
echo "=== done $(date +%T)"
