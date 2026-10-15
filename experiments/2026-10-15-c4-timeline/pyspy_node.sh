#!/usr/bin/env bash
# py-spy (non-blocking, timestamped speedscope) on the vLLM processes during
# timeline_run.py: API server + EngineCore for the whole replay, worker 0 for
# 70 s around nsys window 2 (t0+180).
OUT=$1
T=${OUT}/timings.jsonl
until [[ -s ${T} ]]; do sleep 1; done
t0=$(head -1 "${T}" | sed 's/.*"t0_unix": \([0-9.]*\).*/\1/')
api=$(pgrep -f "python3 .*bin/vllm serve" | head -1)
ec=$(pgrep -f "VLLM::EngineCore" | head -1)
w0=$(pgrep -f "VLLM::Worker_TP0" | head -1)
echo "t0=${t0} api=${api} ec=${ec} w0=${w0}"
PS=/e/project1/profound/alint77/vllm/.venv/bin/py-spy
rec() { ${PS} record --nonblocking --threads -r "${4:-50}" -f speedscope -d "$2" -p "$1" -o "${OUT}/pyspy-$3.json" > "${OUT}/pyspy-$3.log" 2>&1; }
rec ${api} 320 api & rec ${ec} 320 enginecore &
sleep "$(.venv/bin/python -c "import time;print(max(0,${t0}+140-time.time()))")"
date +%s.%N > ${OUT}/pyspy-w0.start
rec ${w0} 70 worker0 100 &
wait
ls -la ${OUT}/pyspy-*
