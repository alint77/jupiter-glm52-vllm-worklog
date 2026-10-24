#!/usr/bin/env bash
# Drafter arms on one hold, in order, at the c=4 prod config (frozen worktree
# df2h = vllm 23d2796ed5, TP-sliced MoE, glue on, SRPT, reserve 3.6), each the
# 600 s c=4 Claude Code replay (timeline_save.py, no profiler), one cache copy
# per hold:
#   mtp3   MTP3
#   inco   DFlash2 k=3, incoai/GLM-5.3-DFlash2 (the head used so far)
#   sub    DFlash2 k=3, SubconsciousDev/glm-5.3-dflash2
#   alaya  DFlash2 k=3, AlayaNeW/GLM-5.3-DFlash2
# DFlash2 drafters at serve.sh's fp8_per_channel weights and fp8 KV, as prod ran.
#     run.sh <hold> <run tag> <arm>...       DUR=<s> shortens the replay
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-24-dflash2-heads
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
S=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees/df2h
M=/e/fscratch/profound/${USER}/models
j=$1 run=$2; shift 2
CS=1,2,3,4,5,6,7,8,12,16,64,128,256,384,512,640,768,896,1024
P="PYTHONPATH=${W} PROFILE=glm53-w4a16-sliced-touch-order.json VLLM_MLA_QKV_GLUE=1 CAPTURE_SIZES=${CS} SCRIPT=${PWD}/agent_space/experiments/2026-10-16-mtp-depth-c4/timeline_save.py TIMELINE_NOPROF=1 TIMELINE_DURATION=${DUR:-600}"
mtp3_env="SPEC=mtp"
inco_env="SPEC=dflash2 DRAFTER=${M}/GLM-5.3-DFlash2"
sub_env="SPEC=dflash2 DRAFTER=${M}/GLM-5.3-DFlash2-SubconsciousDev"
alaya_env="SPEC=dflash2 DRAFTER=${M}/GLM-5.3-DFlash2-AlayaNeW"
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do
  [[ -z $(squeue -h -j "${j}" -o %T) ]] && { echo "hold ${j} left the queue"; exit 1; }
  sleep 20
done
sleep 5
cb=${S}/vllm-df2h-${j} tc=${S}/triton-df2h-${j}; mkdir -p ${cb} ${tc}
cp -an ${S}/vllm-mgq-prod-2396113/. ${cb}/ 2>/dev/null
cp -an ${S}/triton-mgq-prod-2396113/. ${tc}/ 2>/dev/null
echo "$(date +%T) hold ${j} on $(squeue -h -j ${j} -o %N), df2h $(git -C ${W} rev-parse --short HEAD)"
for arm in "$@"; do
  envv=${arm}_env tag=${run}-${arm}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${X}/arm.sh ${tag} script ${P} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=${tc} ${!envv}" > ${X}/logs/${tag}.log 2>&1
  echo "$(date +%T) ${tag}: $(grep -a 'ready\|=== \|server failed\|Error' ${X}/logs/${tag}.log | tail -3 | tr '\n' ' ')"
done
rm -rf ${cb} ${tc}
echo "=== run ${run} done"
