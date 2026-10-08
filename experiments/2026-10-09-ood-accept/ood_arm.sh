#!/usr/bin/env bash
# One c=1 server (prod serve.sh, SPEC / SPEC_K from the caller), then
# ood_accept.py.   ./onnode.sh "<D>/ood_arm.sh <tag> <spec> <k>"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-ood-accept
tag=$1 spec=$2 k=$3
OUT=/e/fscratch/profound/${USER}/ood-accept/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export SPEC=${spec} SPEC_K=${k} COMPILE_SIZES=
if [[ ${spec} == mtp ]]; then
  export RESERVE_GB=3.6 CAPTURE_SIZES=1,$((k + 1)),16,32,64,128,256,384,512,640,768,896,1024
else
  export CAPTURE_SIZES=$((k + 1)),16,32,64,128,256,384,512,640,768,896,1024
fi
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
.venv/bin/python ${D}/ood_accept.py --out "${OUT}/acc.jsonl" --seeds 3
echo "=== done $(date +%T)"
