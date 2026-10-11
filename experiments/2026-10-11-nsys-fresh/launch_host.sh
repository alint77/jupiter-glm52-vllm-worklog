#!/usr/bin/env bash
# Host-time probe (probe_host.py) on one fresh hold, prod tree nsys2, no profiler.
# Variants: TAGS=-ft EXTRA_PATH=:<fastokens dir> EXTRA_ENV=VLLM_USE_FASTOKENS=1
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-11-nsys-fresh
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
j=$(sbatch --parsable --time=01:00:00 "${E}/hold.sbatch")
cb=${C}/vllm-cache-glm53-nfr-host-${j}; mkdir -p ${cb}
cp -an ${C}/vllm-cache-glm53-c16-2264992/{modelinfos,tiered_sliced,torch_extensions} ${cb}/ 2>/dev/null
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
tag=nfr-host${TAGS:-}-${j}
HOLD_JOB=${j} ${E}/onnode.sh "${D}/arm.sh ${tag} host PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/nsys2${EXTRA_PATH:-} SERVE_CACHE_ROOT=${cb} ${EXTRA_ENV:-}" > ${D}/logs/${tag}.log 2>&1
scancel "${j}"
