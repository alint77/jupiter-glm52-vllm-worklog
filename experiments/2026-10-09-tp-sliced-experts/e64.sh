#!/usr/bin/env bash
# adversarial check (adv_check.py), one variant per GPU. e64.sh "<v1>" "<v2>" ...
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
set -- "${@//,/ }"
$PY -c "import kdev,sys; kdev.build_many(sys.argv[1:])" "$@"
i=0
for v in "$@"; do
  CUDA_VISIBLE_DEVICES=$i numactl --cpunodebind=$i --membind=$i $PY adv_check.py --v "$v" --reps ${REPS:-3} > logs/adv-$(echo "$v" | tr ': ' '_-').out 2>&1 &
  i=$((i + 1))
done
wait
for v in "$@"; do echo "== $v"; tail -3 logs/adv-$(echo "$v" | tr ': ' '_-').out; done
