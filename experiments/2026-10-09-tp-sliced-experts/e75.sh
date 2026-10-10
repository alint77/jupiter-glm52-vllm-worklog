#!/usr/bin/env bash
# stress one adversarial case per GPU (fresh inputs each call): e75.sh "<variant>|<case>" x4
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
tag=$1; shift; g=0
$PY -c "import kdev, sys; kdev.build_many(sorted(set(sys.argv[1:])))" $(for s in "$@"; do printf "%q " "${s%%|*}"; done) >/dev/null 2>&1
for spec in "$@"; do
  CUDA_VISIBLE_DEVICES=$g numactl --cpunodebind=$g --membind=$g $PY adv_check.py --v "${spec%%|*}" --seed $((g + 50)) --stress "${spec#*|}" > logs/e75-$tag-g$g.out 2>&1 &
  g=$((g + 1))
done
wait
for g in 0 1 2 3; do echo "== g$g"; grep -v "^ptxas\|^built" logs/e75-$tag-g$g.out | cut -c1-400 | tail -6; done
echo E75_DONE
