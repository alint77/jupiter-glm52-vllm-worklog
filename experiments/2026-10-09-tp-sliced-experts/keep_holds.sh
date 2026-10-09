#!/usr/bin/env bash
# Keep N one-hour kdev-hold allocations alive: count holds pending or running
# with > 15 min left; submit until there are N. Stop by touching keep_holds.stop.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
N=${1:-2}
while [[ ! -e $D/keep_holds.stop ]]; do
  live=$(squeue -h -u $USER -n kdev-hold -o "%T %L" | awk '
    $1 == "PENDING" {n++; next}
    { split($2, a, ":"); m = (NF && length(a) == 3) ? a[1] * 60 + a[2] : a[1]; if (m > 15) n++ }
    END {print n + 0}')
  for ((i = live; i < N; i++)); do
    sbatch --parsable --time=01:00:00 --job-name=kdev-hold $E/hold.sbatch >> $D/keep_holds.log
  done
  sleep 120
done
