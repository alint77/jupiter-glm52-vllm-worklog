#!/usr/bin/env bash
# Round 3 served A/B: prod EP vs tp_sliced, both from the same frozen worktree
# (tp-sliced-b = vLLM 18bb8c945f, the shipped kernel) so only the layout flag
# differs. Standard harness (../2026-10-07-mem-reclaim/arm.sh: GSM8K 200,
# agentic task set with per-request prefill/decode, long-context decode, TTFT,
# 388K stress), arms back to back on one fresh hold:
#   chain_r3m2.sh <first arm: ep|sl> <n arms>   (MTP3, no GSM8K; vLLM 7df917dd4d: MTP drafter MoE untiered under tp_sliced)
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
A=agent_space/experiments/2026-10-07-mem-reclaim/arm_nogsm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
WT=/e/fscratch/profound/${USER}/worktrees/tp-sliced-b
first=$1 n=$2
j=$(sbatch --parsable --time=01:30:00 "${E}/hold.sbatch")
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
arm=${first}
for i in $(seq 1 "${n}"); do
  # per-layout compile caches for this round: the runner's epilogue changed
  # since the older caches were traced; one per hold, as holds start at once
  if [[ ${arm} == ep ]]; then
    cb=${C}/vllm-cache-glm53-r3m-ep-${j}; src=${C}/vllm-cache-glm53-ep-base
    extra=""
  else
    cb=${C}/vllm-cache-glm53-r3m-sl-${j}; src=${C}/vllm-cache-glm53-sliced
    extra="TIERED_MOE_LAYOUT=tp_sliced REPLICAS="
  fi
  mkdir -p ${cb}; cp -an ${src}/{modelinfos,torch_extensions} ${cb}/ 2>/dev/null
  HOLD_JOB=${j} ${E}/onnode.sh "SPEC=mtp SPEC_K=3 PYTHONPATH=${WT} SERVE_CACHE_ROOT=${cb} ${extra} ${A} ${arm}m-${j}-${i}" \
    > ${D}/logs/serve/run-${arm}m-${j}-${i}.log 2>&1
  [[ ${arm} == ep ]] && arm=sl || arm=ep
done
scancel "${j}"
