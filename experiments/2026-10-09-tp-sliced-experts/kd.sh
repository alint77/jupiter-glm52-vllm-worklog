#!/usr/bin/env bash
# Run a command on the running kdev-hold with the most time left:  kd.sh <cmd...>
cd /e/project1/profound/alint77/vllm
j=$(squeue -h -u $USER -n kdev-hold -t RUNNING -o "%i %L" | awk '{
  n = split($2, a, /[-:]/); s = 0
  if (n == 1) s = a[1]; else if (n == 2) s = a[1] * 60 + a[2]; else if (n == 3) s = a[1] * 3600 + a[2] * 60 + a[3]; else s = a[1] * 86400 + a[2] * 3600 + a[3] * 60 + a[4]
  print s, $1}' | sort -rn | head -1 | cut -d' ' -f2)
[[ -n $j ]] || { echo "no running kdev-hold" >&2; exit 2; }
HOLD_JOB=$j exec agent_space/experiments/2026-09-27-glm53-mtp7-profile/onnode.sh "$@"
