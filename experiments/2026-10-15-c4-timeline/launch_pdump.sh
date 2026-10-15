#!/usr/bin/env bash
# Prefix-miss root cause: prefix_repro.py (#112-114, #16-17 at c=1) on pf4
# (877df9b262 + VLLM_DEBUG_PREFIX_DUMP probe in KVCacheManager.get_computed_blocks:
# per request, hit, per-group cached prefix, block hashes, prompt token ids).
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-15-c4-timeline
A=agent_space/experiments/2026-10-11-nsys-fresh/arm.sh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
W=/e/fscratch/profound/${USER}/worktrees/pf4
j=$(sbatch --parsable --time=${TIME:-01:00:00} "${E}/hold.sbatch")
cb=${C}/vllm-cache-glm53-pdump-${j}; mkdir -p ${cb}
cp -an ${C}/vllm-cache-glm53-c4tl-2333343/{modelinfos,tiered_sliced,torch_extensions,torch_compile_cache} ${cb}/ 2>/dev/null
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
tag=pdump-${j}
OUT=/e/fscratch/profound/${USER}/sliced-prof/${tag}
HOLD_JOB=${j} ${E}/onnode.sh "${A} ${tag} script PYTHONPATH=${W} SERVE_CACHE_ROOT=${cb} TRITON_CACHE_DIR=/e/fscratch/profound/${USER}/caches/triton-srpt1-2334590 SCRIPT=${D}/prefix_repro.py PREFIX_SEQS=112,113,114__16,17 VLLM_DEBUG_PREFIX_DUMP=${OUT}/dump" > ${D}/logs/${tag}.log 2>&1
scancel "${j}"
