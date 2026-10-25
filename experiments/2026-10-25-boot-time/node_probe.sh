#!/usr/bin/env bash
# On the hold node (srun, no environment loaded yet): one prod c=4 boot from the
# main tree on a copy of the prod cache, with import times (PYTHONPROFILEIMPORTTIME),
# vmstat, and py-spy on the TP0 worker during weight loading; then GPFS read
# probes on checkpoint shards the boot has evicted from the page cache.
#     node_probe.sh <out dir> <cache root copy>
set -uo pipefail
cd /e/project1/profound/alint77/vllm
O=$1 cb=$2
M=/e/fscratch/profound/${USER}/models/GLM-5.3-W4A16
ts() { date +%T.%N | cut -c1-12; }
echo "$(ts) start on $(hostname -s)" > ${O}/marks.txt
{ time bash -c 'source agent_space/jupiter-env.sh >/dev/null 2>&1'; } 2> ${O}/env-cold.txt
echo "$(ts) env cold done" >> ${O}/marks.txt
source agent_space/jupiter-env.sh >/dev/null 2>&1
vmstat -t 2 > ${O}/vmstat.txt &
vm=$!
export TIERED_MOE_LAYOUT=tp_sliced REPLICAS= PROFILE=glm53-w4a16-sliced-touch-order.json
export SPEC=mtp SPEC_K=3 COMPILE_SIZES= MAX_NUM_SEQS=4 VLLM_TIERED_MOE_KV_POOL_SEQS=2
export RESERVE_GB=3.6 VLLM_MLA_QKV_GLUE=1 SERVE_CACHE_ROOT=${cb}
export CAPTURE_SIZES=1,2,3,4,5,6,7,8,12,16,64,128,256,384,512,640,768,896,1024
export SERVE_EXTRA="--host 127.0.0.1 --port 8027 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 ${LOAD_EXTRA:-}"
export PYTHONPROFILEIMPORTTIME=1
echo "$(ts) serve.sh" >> ${O}/marks.txt
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh > ${O}/server.out 2> ${O}/server.err &
srv=$!
unset PYTHONPROFILEIMPORTTIME
until grep -aq "Worker_TP0.*Loading model from scratch" ${O}/server.out 2>/dev/null; do
  kill -0 ${srv} 2>/dev/null || { echo "$(ts) server died" >> ${O}/marks.txt; exit 1; }
  sleep 1
done
pid=$(grep -a -m1 "Worker_TP0.*Loading model from scratch" ${O}/server.out | sed 's/.*pid=\([0-9]*\).*/\1/')
echo "$(ts) load start, TP0 pid ${pid}" >> ${O}/marks.txt
py-spy record -p ${pid} --nonblocking --idle -r 10 -f raw -o ${O}/load-tp0.raw -d 600 > ${O}/pyspy.log 2>&1 &
spy=$!
until grep -aq "Worker_TP0.*Model loading took" ${O}/server.out; do kill -0 ${srv} 2>/dev/null || { echo "$(ts) server died in load" >> ${O}/marks.txt; kill -INT ${spy}; exit 1; }; sleep 1; done
kill -INT ${spy}; wait ${spy}
echo "$(ts) load done" >> ${O}/marks.txt
for _ in $(seq 1 240); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 ${srv} 2>/dev/null || break
  sleep 5
done
echo "$(ts) health ok" >> ${O}/marks.txt
kill ${srv}; wait ${srv}
sleep 10
pkill -f "vllm serve" ; sleep 5
kill ${vm}
echo "$(ts) server down" >> ${O}/marks.txt
{ time .venv/bin/python -X importtime -c 'import vllm.entrypoints.openai.api_server' 2> ${O}/importtime-warm.txt; } 2> ${O}/import-warm-time.txt
# GPFS: whole shards with dd, distinct shards per test (from the start of the
# checkpoint, the first the boot read and so the first evicted)
rd() {  # rd <tag> <readers> <first shard> <shards per reader> [same]
  local tag=$1 n=$2 s=$3 k=$4 same=${5:-} files=() i r bytes t0 t1
  for r in $(seq 0 $((n - 1))); do
    files=()
    for i in $(seq 0 $((k - 1))); do
      [[ -n ${same} ]] && idx=$((s + i)) || idx=$((s + r * k + i))
      files+=($(printf "%s/model-%05d-of-00282.safetensors" ${M} ${idx}))
    done
    echo "${files[@]}" > ${O}/rd-${tag}-${r}.files
  done
  bytes=0
  for r in $(seq 0 $((n - 1))); do
    for f in $(cat ${O}/rd-${tag}-${r}.files); do bytes=$((bytes + $(stat -c %s ${f}))); done
  done
  t0=$(date +%s.%N)
  for r in $(seq 0 $((n - 1))); do
    (for f in $(cat ${O}/rd-${tag}-${r}.files); do dd if=${f} of=/dev/null bs=16M status=none; done) &
  done
  wait
  t1=$(date +%s.%N)
  echo "${tag} readers=${n} bytes=${bytes} s=$(echo "${t1} - ${t0}" | bc) GiB/s=$(echo "scale=2; ${bytes} / (${t1} - ${t0}) / 1073741824" | bc)${same:+ (same files: per-reader bytes counted once each)}" >> ${O}/gpfs.txt
}
[[ -n ${SKIP_RD:-} ]] && exit 0
rd r1 1 2 6
rd r4 4 8 4
rd r16 16 24 2
rd r32 32 56 1
rd s4 4 88 6 same
echo "$(ts) probes done" >> ${O}/marks.txt
