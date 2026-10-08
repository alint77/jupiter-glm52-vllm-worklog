#!/usr/bin/env bash
# One held node: the c=2 / k=3 server with two profiled windows (a lone
# 4-token step and a 5K+5K pair), then the trace analyses; releases the node.
#   launch_prof.sh <hold job>
cd /e/project1/profound/alint77/vllm
C=agent_space/experiments/2026-10-08-c2-k3
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
X=agent_space/experiments
j=$1
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "MAX_NUM_SEQS=2 SPEC_K=3 CAPTURE_SIZES=4,8,16,32,64,128,256,384,512,640,768,896,1024 ${C}/run.sh c2k3prof --profile-only" > ${C}/run-c2k3prof.log 2>&1
scancel "${j}"
T=/e/fscratch/profound/${USER}/c2-k3/c2k3prof/trace
for w in pair-5K_5K alone-5K; do
  echo "######## ${w}"
  .venv/bin/python ${X}/2026-09-28-agentic-decode-bench/kernel_tally.py ${T}/${w} --top 30
  .venv/bin/python ${X}/2026-10-07-decode-mem-dive/coll_wait.py ${T}/${w}
  .venv/bin/python ${X}/2026-10-07-decode-mem-dive/layer_timeline.py ${T}/${w}/*rank0.*.gz --layers 40
done > ${C}/analysis3.txt 2>&1
echo done > ${C}/analysis3.done
