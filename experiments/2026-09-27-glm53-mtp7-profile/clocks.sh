#!/usr/bin/env bash
# SM clock, power and throttle reasons at 100 ms while (1) the GLM server decodes
# (bench.py short, greedy) and (2) bench_fmt.py times the MoE kernel alone.
#   ./onnode.sh <E>/clocks.sh <tag>      -> clocks-<tag>-{serve,kernel}.csv
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
tag="$1"
q=timestamp,index,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active
sample() { nvidia-smi --query-gpu="${q}" --format=csv,noheader -lms 100 >"$1" & echo $!; }

bash "${E}/serve.sh" >"${E}/server-clocks-${tag}.out" 2>"${E}/server-clocks-${tag}.err" &
pid=$!
trap 'kill ${pid} 2>/dev/null || true; wait ${pid} 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; exit 1; }
  sleep 5
done
echo "server ready $(date +%T)"
s=$(sample "${E}/clocks-${tag}-serve.csv")
.venv/bin/python "${E}/bench.py" --out "${E}/ab-clocks-${tag}.json" --contexts short --reps 3
kill "${s}"
kill "${pid}"; wait "${pid}" 2>/dev/null || true
trap - EXIT
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 2; done

s=$(sample "${E}/clocks-${tag}-kernel.csv")
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 .venv/bin/python \
  "${E}/bench_fmt.py" --fmt int4 --numa-node 0 --grid 9,2 9,2 9,2 9,2 9,2 9,2
kill "${s}"
echo "=== done $(date +%T)"
