#!/usr/bin/env bash
# Correctness bisection of the glue arms' acceptance drop (run 2: 3.1-3.4 ->
# 2.0; run 3: 1.6): one server per hold at the c=4 config, probe.py (prompt
# logprobs, greedy and sampled 4 x 5K chats). pf9 = pf8 + debug knobs
# (VLLM_MLA_GLUE_NOPDL, VLLM_MLA_GLUE_MAX_LAYER=78 keeps the MTP layer off
# glue, VLLM_MLA_GLUE_KROPE=0 = original k RoPE).
#   launch_probe.sh
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-18-mla-glue
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/mla-glue-probe
W=/e/fscratch/profound/${USER}/worktrees
P=PROFILE=glm53-w4a16-sliced-touch-order.json
G="PYTHONPATH=${W}/pf9 ${P} VLLM_MLA_QKV_GLUE=1"
if [[ ${RUN:-mgp} == mgp ]]; then
arms=("2396112 prod PYTHONPATH=${W}/pf6 ${P}"
      "2396113 glue ${G}"
      "2396114 nopdl ${G} VLLM_MLA_GLUE_NOPDL=1"
      "2396115 nodraft ${G} VLLM_MLA_GLUE_MAX_LAYER=78"
      "2396116 krope ${G} VLLM_MLA_GLUE_KROPE=0"
      "2396117 prod2 PYTHONPATH=${W}/pf6 ${P}"
      "2396118 nopdlnodraft ${G} VLLM_MLA_GLUE_NOPDL=1 VLLM_MLA_GLUE_MAX_LAYER=78"
      "2396119 glue2 ${G}")
else  # mgq: split the norms (VLLM_MLA_GLUE_QORIG / KVORIG = original module), 8 greedy prompts
arms=("2396112 glue ${G}"
      "2396113 prod PYTHONPATH=${W}/pf6 ${P}"
      "2396114 krope ${G} VLLM_MLA_GLUE_KROPE=0"
      "2396115 qorig ${G} VLLM_MLA_GLUE_QORIG=1"
      "2396116 prod2 PYTHONPATH=${W}/pf6 ${P}"
      "2396117 krope2 ${G} VLLM_MLA_GLUE_KROPE=0"
      "2396118 kvorig ${G} VLLM_MLA_GLUE_KVORIG=1 VLLM_MLA_GLUE_KROPE=0"
      "2396119 glue2 ${G}")
fi
for h in "${arms[@]}"; do (
  set -- ${h}; j=$1 name=$2; shift 2
  tag=${RUN:-mgp}-${name}-${j}
  HOLD_JOB=${j} ${E}/onnode.sh "${X}/probe_arm.sh ${tag} 4 4 SERVE_CACHE_ROOT=${C}/vllm-${tag} TRITON_CACHE_DIR=${C}/triton-${tag} VLLM_ENFORCE_STRICT_TOOL_CALLING=false $*" > ${X}/logs/${tag}.log 2>&1
  echo "$(date +%T) ${tag}: $(grep -a 'acc greedy\|=== \|server failed' ${X}/logs/${tag}.log | tail -2 | tr '\n' ' ')"
) & done
wait
