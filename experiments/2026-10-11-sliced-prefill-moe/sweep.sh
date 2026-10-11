#!/usr/bin/env bash
# Up to 4 define sets in parallel, one per GPU, bench_sliced.py each:
#   sweep.sh <tag> "<defines 0>" "<defines 1>" ... -> logs/sw-<tag>-<i>.log
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-11-sliced-prefill-moe
tag=$1; shift
i=0
for d in "$@"; do
  ( echo "DEFINES: $d"; GPU=$i DEFINES="$d" $X/run.sh ${TREE:+TREE=$TREE} $X/bench_sliced.py ${SIZES:+--sizes $SIZES} ) > $X/logs/sw-$tag-$i.log 2>&1 &
  i=$((i + 1))
done
wait
for j in $(seq 0 $((i - 1))); do echo "== $j"; grep -E "DEFINES|T=|error|Error" $X/logs/sw-$tag-$j.log | head -12; done
