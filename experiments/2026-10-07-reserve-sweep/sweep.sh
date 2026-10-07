#!/usr/bin/env bash
# HBM reserve sweep for the prod default (serve.sh: skip-layer KV + drafter
# fp8 / Grace KV): one fresh hold per RESERVE_GB value, each runs
# ../2026-10-02-ingraph-prefetch/stress_long.sh to 390K tokens (60K prompt,
# +2-8K per turn, +40K uncached every 10th turn). Detaches.
#   bash sweep.sh 5.14 4.9 4.7 4.55 4.4
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
S=agent_space/experiments/2026-10-02-ingraph-prefetch/stress_long.sh
D=agent_space/experiments/2026-10-07-reserve-sweep
for r in "$@"; do
  j=$(sbatch --parsable --time=01:15:00 "${E}/hold.sbatch")
  echo "${j} reserve ${r}"
  setsid nohup bash -c "
    until [[ \$(squeue -h -j ${j} -o %T) == RUNNING ]]; do sleep 20; done; sleep 10
    RESERVE_GB=${r} STRESS_MAX_CTX=390000 HOLD_JOB=${j} ${E}/onnode.sh '${S} reserve-${r}' > ${D}/run-${r}.log 2>&1
    scancel ${j}" >/dev/null 2>&1 < /dev/null &
done
