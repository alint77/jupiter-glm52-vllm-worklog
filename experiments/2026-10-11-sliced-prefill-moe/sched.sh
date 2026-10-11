#!/usr/bin/env bash
# bench_sliced.py with schedules 0 / 1 / 3 on GPUs 0-2 (+ 0 again on GPU 3)
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-11-sliced-prefill-moe
i=0
for s in 0 1 3 0; do
  ( echo "schedule $s"; GPU=$i $X/run.sh $X/bench_sliced.py --schedule $s ${SIZES:+--sizes $SIZES} ) > $X/logs/sched-$1-$i.log 2>&1 &
  i=$((i + 1))
done
wait
for j in 0 1 2 3; do echo "== GPU $j"; grep -E "schedule|T=" $X/logs/sched-$1-$j.log; done
