#!/usr/bin/env bash
# Boot-time probe on a fresh hold: copy the prod c=4 cache root (one copy per
# hold, deleted after), run node_probe.sh there.     probe.sh <hold>
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-25-boot-time
j=$1 O=${X}/out-$1${TAG:+-${TAG}}
C=/e/fscratch/profound/${USER}/caches/marlin/vllm-cache-glm53-sliced-mtp3-c4
cb=/e/fscratch/profound/${USER}/caches/boot-probe/vllm-${j}
mkdir -p ${O} ${cb}
cp -a ${C}/. ${cb}/
until [[ $(squeue -h -j "${j}" -o %T) == RUNNING ]]; do
  [[ -z $(squeue -h -j "${j}" -o %T) ]] && { echo "hold ${j} left the queue"; exit 1; }
  sleep 20
done
echo "$(date +%T) hold ${j} on $(squeue -h -j ${j} -o %N)"
srun --jobid=${j} --overlap --ntasks=1 --gres=gpu:4 --cpus-per-task=288 \
  env LOAD_EXTRA="${LOAD_EXTRA:-}" SKIP_RD="${SKIP_RD:-}" bash ${X}/node_probe.sh ${O} ${cb}
rm -rf ${cb}
echo "$(date +%T) probe done"; cat ${O}/marks.txt ${O}/gpfs.txt
