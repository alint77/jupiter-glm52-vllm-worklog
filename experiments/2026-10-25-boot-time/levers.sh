#!/usr/bin/env bash
# The boot levers end to end on one hold, from the pinned prod tree
# (worktrees/prod = f722d708e2: read-ahead loader; serve.sh's parallel
# prebuild): boot A on a copy of the prod c=4 cache (first boot after a pin:
# new tree path, so compile and extension caches miss), boot B on the same
# copy (warm), then the real prewarm of the prod cache root.
#     levers.sh <hold>
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-25-boot-time
j=$1
C=/e/fscratch/profound/${USER}/caches/marlin/vllm-cache-glm53-sliced-mtp3-c4
cb=/e/fscratch/profound/${USER}/caches/boot-probe/vllm-${j}
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do
  [[ -z $(squeue -h -j "${j}" -o %T) ]] && { echo "hold ${j} left the queue"; exit 1; }
  sleep 20
done
echo "$(date +%T) hold ${j} on $(squeue -h -j ${j} -o %N)"
mkdir -p ${cb}; cp -a ${C}/. ${cb}/; rm -f ${cb}/prewarm-*
summ() {
  local l; l=$(ls -t $1/prewarm-*.out | head -1)
  echo "  $(grep -a TP0 ${l} | grep -a -E 'Loading weights took|Model loading took|torch.compile took|profiling/warmup run took|Graph capturing finished' | sed 's/.*\] //' | tr '\n' ';')"
  grep -a "prebuild" ${l%.out}.err | tr '\n' ';'; echo
}
for b in A B; do
  SERVE_CACHE_ROOT=${cb} agent_space/scripts/prod_prewarm.sh ${j} 4 && summ ${cb}
  cp $(ls -t ${cb}/prewarm-*.out | head -1) ${X}/logs/levers-${b}-${j}.out
  cp $(ls -t ${cb}/prewarm-*.err | head -1) ${X}/logs/levers-${b}-${j}.err
done
rm -rf ${cb}
squeue -u ${USER} -h -o %j | grep -q "glm53-claude-sliced-mtp3-c4" \
  && { echo "prod c4 server running: skip the real prewarm"; exit 0; }
agent_space/scripts/prod_prewarm.sh ${j} 4 && summ ${C}
echo "$(date +%T) done"
