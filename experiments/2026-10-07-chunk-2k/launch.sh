#!/usr/bin/env bash
# One fresh hold per arm "<chunk>:<reserve>", detached.  bash launch.sh 4096:4.7 2048:4.7 ...
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
D=agent_space/experiments/2026-10-07-chunk-2k
# "<chunk>:<reserve>:g" also captures a piecewise CUDA graph at <chunk> tokens.
for arm in "$@"; do
  IFS=: read -r c r g <<<"${arm}"; tag="c${c}${g:+g}-r${r}"
  caps="8,16,32,64,128,256,384,512,640,768,896,1024${g:+,${c}}"
  j=$(sbatch --parsable --time=01:15:00 "${E}/hold.sbatch")
  echo "${j} ${tag}"
  setsid nohup bash -c "
    until [[ \$(squeue -h -j ${j} -o %T) == RUNNING ]]; do sleep 20; done; sleep 10
    CAPTURE_SIZES=${caps} MAX_NUM_BATCHED_TOKENS=${c} RESERVE_GB=${r} HOLD_JOB=${j} ${E}/onnode.sh '${D}/run.sh ${tag}' > ${D}/run-${tag}.log 2>&1
    scancel ${j}" >/dev/null 2>&1 < /dev/null &
done
