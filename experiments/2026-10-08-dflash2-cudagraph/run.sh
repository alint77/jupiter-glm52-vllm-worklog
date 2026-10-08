#!/usr/bin/env bash
# One prod server (2026-09-27 serve.sh) with extra env, then acc_probe.py.
#   onnode.sh "<env...> <D>/run.sh <tag>"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-dflash2-cudagraph
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
tag=$1; shift
OUT=/e/fscratch/profound/${USER}/dflash2-cudagraph/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
bash ${E}/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
grep -ah "observed HBM\|Capturing dflash\|dflash.*graph\|DFlash2 selector" "${OUT}/server.out" "${OUT}/server.err" | sed 's/^.*\] //' | sort | uniq -c | head
.venv/bin/python ${D}/acc_probe.py --out "${OUT}/acc.json" "$@"
grep -ah "GRAPH PROBE" "${OUT}/server.out" "${OUT}/server.err" | grep "Worker_TP0\|TP0" | sed 's/^.*\] //' | head -40
echo "=== done $(date +%T)"
