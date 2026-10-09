#!/usr/bin/env bash
# Run a command on the running kdev-hold with the most time left:  kd.sh <cmd...>
cd /e/project1/profound/alint77/vllm
j=$(squeue -h -u $USER -n kdev-hold -t RUNNING -o "%i %L" | sort -k2 -r | head -1 | cut -d' ' -f1)
[[ -n $j ]] || { echo "no running kdev-hold" >&2; exit 2; }
HOLD_JOB=$j exec agent_space/experiments/2026-09-27-glm53-mtp7-profile/onnode.sh "$@"
