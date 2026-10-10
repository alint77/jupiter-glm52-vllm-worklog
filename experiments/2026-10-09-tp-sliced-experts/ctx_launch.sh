cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
C=/e/fscratch/profound/${USER}/caches/marlin
j=$(sbatch --parsable --time=00:45:00 "${E}/hold.sbatch")
cb=${C}/vllm-cache-glm53-ctx-sl-${j}
cp -a ${C}/vllm-cache-glm53-sw-sl-2263521 ${cb}; rm -f ${cb}/torch_extensions/*/lock
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do sleep 20; done
sleep 5
HOLD_JOB=${j} ${E}/onnode.sh "${D}/ctx_arm.sh ctx-sl-${j} PYTHONPATH=/e/fscratch/profound/${USER}/worktrees/tp-sliced-b SERVE_CACHE_ROOT=${cb} TIERED_MOE_LAYOUT=tp_sliced REPLICAS=" > ${D}/logs/serve/ctx-sl-${j}.log 2>&1
scancel "${j}"
