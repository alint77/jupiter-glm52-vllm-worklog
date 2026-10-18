#!/usr/bin/env bash
# py-spy (non-blocking, active samples, per thread, speedscope) on the API
# server, EngineCore and worker 0 for the whole timeline_run.py replay.
#   pyspy_run.sh <out dir> <duration s>
OUT=$1 DUR=$2
until [[ -s ${OUT}/timings.jsonl ]]; do sleep 1; done
api=$(pgrep -f "python3 .*bin/vllm serve" | head -1)
ec=$(pgrep -f "VLLM::EngineCore" | head -1)
w0=$(pgrep -f "VLLM::Worker_TP0" | head -1)
date +%s.%N > ${OUT}/pyspy.start
echo "api=${api} ec=${ec} w0=${w0}" >> ${OUT}/pyspy.start
PS=/e/project1/profound/alint77/vllm/.venv/bin/py-spy
rec() { ${PS} record --nonblocking --threads -r "$3" -f speedscope -d "$((DUR + 20))" -p "$1" -o "${OUT}/pyspy-$2.json" > "${OUT}/pyspy-$2.log" 2>&1; }
rec ${api} api 50 & rec ${ec} enginecore 50 & rec ${w0} worker0 100 &
wait
