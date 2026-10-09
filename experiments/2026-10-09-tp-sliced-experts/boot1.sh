#!/usr/bin/env bash
# First tp_sliced boot on a glm-hold: serve, smoke, GSM8K subset.
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
G=agent_space/experiments/2026-09-05-cold-prefetch/gsm8k_paired.py
export TMPDIR=/e/fscratch/profound/${USER}/caches/gsm8k
tag=${TAG:-a1}
echo "=== boot ${tag} $(date +%T) on $(hostname -s)"
WT=${WT} bash ${D}/serve-sliced.sh >${D}/logs/serve/server-${tag}.out 2>${D}/logs/serve/server-${tag}.err &
pid=$!
ok=false
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && { ok=true; break; }
  kill -0 ${pid} 2>/dev/null || break
  sleep 5
done
echo "ready=${ok} $(date +%T)"
if [[ ${ok} == true ]]; then
  .venv/bin/python ${G} --num-questions 2 --out ${D}/logs/serve/gsm-${tag}-smoke.json &&
    .venv/bin/python ${G} --num-questions ${NQ:-250} --out ${D}/logs/serve/gsm-${tag}.json
  grep -h "Tiered MoE residency\|TP-sliced kernel\|Loading weights took\|Streamed" ${D}/logs/serve/server-${tag}.err | head -8
fi
[[ -n ${KEEP:-} ]] && wait ${pid}
kill ${pid} 2>/dev/null; sleep 20; pkill -f "vllm serve" 2>/dev/null; true
