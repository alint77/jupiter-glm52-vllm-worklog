#!/usr/bin/env bash
# One c=4, DFlash2 k=7 server (prod serve.sh; extra env from the caller), then
# conc_probe.py --quad: profiled windows first (lone 8-token, 16-token pair,
# 32-token quad), then 4 reps of every case.
#   ./onnode.sh "<env> <D>/arm_conc.sh <tag>"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
C=agent_space/experiments/2026-10-08-c2-k3
tag=$1; OUT=/e/fscratch/profound/${USER}/m32/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export MAX_NUM_SEQS=4 SPEC_K=7 CAPTURE_SIZES=8,16,24,32,64,128,256,384,512,640,768,896,1024
export TRACE_ROOT="${OUT}/trace"; mkdir -p "${TRACE_ROOT}"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
grep -ah "observed HBM\|GPU KV cache size\|Maximum concurrency" "${OUT}/server.out" | sed 's/^.*\] //' | sort | uniq -c
grep -a "residency:" "${OUT}/server.out" | grep Worker_TP0 | tail -1 | sed 's/^.*\] //' | cut -c1-80
.venv/bin/python ${C}/conc_probe.py --out "${OUT}/probe.jsonl" --trace-root "${TRACE_ROOT}" --quad --reps 4 --with-profile
echo "=== done $(date +%T)"
