#!/usr/bin/env bash
# M=8: this tree's decode kernel vs HEAD's, 3 rounds alternating, GPU 0.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-m32
export TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-m32; mkdir -p $TMPDIR
for rep in 1 2 3; do
  if (( rep % 2 )); then o=("--kernel decode" "--kernel decode --rev HEAD"); else o=("--kernel decode --rev HEAD" "--kernel decode"); fi
  for args in "${o[@]}"; do
    CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 .venv/bin/python $D/bench_moe_m32.py --m 8 $args \
      2>>$D/m8-$1.err | grep "^{" | sed "s/}$/, \"rep\": $rep}/" >> $D/m8-$1.jsonl
  done
done
