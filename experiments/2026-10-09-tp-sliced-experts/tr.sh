#!/usr/bin/env bash
# Traced single calls + tl2 summary:  tr.sh <m> <cell> <variant>...
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
m=$1 cell=$2; shift 2
vs=(); for v in "$@"; do [[ $v == *:* ]] && vs+=("$v TD_CTA_TRACE") || vs+=("$v:TD_CTA_TRACE"); done
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "${vs[@]}" || exit 1
for v in "${vs[@]}"; do
  echo "## $v  m=$m cell=$cell"
  t=logs/tr-$(echo "$v $m $cell" | md5sum | cut -c1-8).pt
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m $m --numa-node 0 \
    --cell $cell --n 1 --trace $t --shared 1 2>>logs/tr.err >/dev/null && $PY tl2.py $t
done
