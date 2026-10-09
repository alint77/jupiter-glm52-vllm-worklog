#!/usr/bin/env bash
# On one node: the slice correctness check (GPU 0), then every variant over
# its cells, M=8 on GPUs 0/1 and M=32 on GPUs 2/3, the second GPU of each
# pair in reverse variant order. NUMA-bound to each GPU's Grace node.
#   ./onnode.sh <D>/run_bench.sh <tag>
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/slicebench TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-slice
mkdir -p $VLLM_CACHE_ROOT $TMPDIR
TAG=$1
EP8="4,0 6,0 8,0 10,0 12,0 14,0 16,0 20,0 4,1 6,1 8,1 10,1 12,1 14,1 16,1 20,1 4,2 8,2 12,2 16,2 20,2 4,3 8,3 12,3 16,3"
SL8="24,0 30,0 36,0 40,0 44,0 50,0 56,0 24,2 30,2 36,2 40,2 44,2 50,2 56,2 24,4 30,4 36,4 40,4 44,4 50,4 24,6 36,6 44,6 50,6 24,8 36,8 44,8 56,8"
EP32="16,0 20,0 24,0 28,0 32,0 40,0 16,2 20,2 24,2 28,2 32,2 40,2 16,4 24,4 32,4 40,4 16,6 24,6 32,6 20,8 28,8 40,8"
SL32="72,4 84,4 96,4 104,4 112,4 128,4 144,4 72,8 84,8 96,8 104,8 112,8 128,8 144,8 72,12 96,12 112,12 128,12 84,16 104,16 128,16 72,20 96,20 112,20 84,24 128,24"
bench() {  # gpu m variant cells
  CUDA_VISIBLE_DEVICES=$1 numactl --cpunodebind=$1 --membind=$1 .venv/bin/python $D/bench_slice.py \
    --time --variant $3 --m $2 --numa-node $1 --cells $4 2>>$D/bench-$TAG-gpu$1.err | grep "^{" \
    | sed "s/}$/, \"gpu\": $1}/" >> $D/bench-$TAG.jsonl
}
[[ -z "${SKIP_CHECK:-}" ]] && CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  .venv/bin/python $D/bench_slice.py --check 2>$D/check-$TAG.err | tee $D/check-$TAG.jsonl
pair() {  # gpu m order...
  local g=$1 m=$2; shift 2
  for v in "$@"; do
    case $v in full*) c=EP$m;; *) c=SL$m;; esac
    bench $g $m $v "${!c}"
  done
}
# VARIANTS: space-separated; the second GPU of each pair runs them reversed
V=(${VARIANTS:-full-1024x4 full-512x8 slice-512x8 slice-512x4})
R=(); for ((i=${#V[@]}-1; i>=0; i--)); do R+=("${V[i]}"); done
pair 0 8 "${V[@]}" &
pair 1 8 "${R[@]}" &
pair 2 32 "${V[@]}" &
pair 3 32 "${R[@]}" &
wait
echo "=== done $(date +%T)"
