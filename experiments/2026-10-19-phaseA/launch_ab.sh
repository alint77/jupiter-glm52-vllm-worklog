#!/usr/bin/env bash
# Phase A served A/B at the c=4 launcher config, ABBA on 4 fresh holds:
#   a = prod (worktrees/pf6 = 65d9b56fb1; 44153148ba only changes a launcher)
#   n = worktrees/pf11 (44153148ba + phaseA-pf11.diff: native next_n=4 DSA
#       indexer decode on SM90 (#52164) + no retained FlashMLA-sparse workspace
#       views (#56902))
# Both arms log workspace resizes (VLLM_DEBUG_WORKSPACE=1); node_run2.sh runs
# mem_sampler for the startup memory phases. Kinds: dec1 / dec4 (n decoders at
# ~5K and ~50K, nsys windows). The first hold also runs the kernel microbench.
#   launch_ab.sh
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-18-c4-swarm-timeline
X=agent_space/experiments/2026-10-19-phaseA
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/phaseA
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
P="PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_DEBUG_WORKSPACE=1"
a_env="PYTHONPATH=${W}/pf6 ${P}"
n_env="PYTHONPATH=${W}/pf11 ${P}"
mkdir -p ${X}/logs
for spec in "dec1 an bench" "dec1 na" "dec4 an" "dec4 na"; do (
  set -- ${spec}; kind=$1 order=$2 bench=${3:-}
  j=$(sbatch --parsable --time=01:45:00 "${E}/hold.sbatch")
  echo "${kind} ${order} hold ${j}"
  until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
  sleep 5
  if [[ -n ${bench} ]]; then
    HOLD_JOB=${j} ${E}/onnode.sh "PYTHONPATH=${W}/pf11 .venv/bin/python ${X}/bench_indexer_next_n.py --ctx 5000,12500,50000,100000,200000 --batch 1,2,4" > ${X}/logs/bench-${j}.log 2>&1
  fi
  for arm in ${order:0:1} ${order:1:1}; do
    tag=pa${arm}-${kind}-${j}; envv=${arm}_env
    cb=${C}/vllm-${tag} tc=${C}/triton-${tag}; mkdir -p ${cb} ${tc}
    cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
    cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
    HOLD_JOB=${j} ${E}/onnode.sh "${D}/node_run2.sh ${tag} ${kind} ${cb} ${tc} ${!envv}" > ${X}/logs/${tag}.log 2>&1
    echo "$(date +%T) ${tag}: $(grep -a '=== ' ${X}/logs/${tag}.log | tail -1)"
  done
) & done
wait
