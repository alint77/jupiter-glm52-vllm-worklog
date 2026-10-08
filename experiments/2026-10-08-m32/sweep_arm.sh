#!/usr/bin/env bash
# One server: prod serve.sh with SPEC=<mtp|dflash2>, SPEC_K=3, MAX_NUM_SEQS=<c>,
# then conc_sweep.py --max-conc <c>.
#   ./onnode.sh "<D>/sweep_arm.sh <tag> <spec> <c> [extra env...]"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-08-m32
tag=$1 spec=$2 c=$3; shift 3
OUT=/e/fscratch/profound/${USER}/m32-sweep/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export SPEC=${spec} SPEC_K=3 MAX_NUM_SEQS=${c} COMPILE_SIZES=
export CAPTURE_SIZES=1,2,3,4,5,6,7,8,12,16,20,24,28,32,64,128,256,384,512,640,768,896,1024
[[ ${spec} == mtp ]] && export RESERVE_GB=${RESERVE_GB:-3.0}
for kv in "$@"; do export "${kv}"; done
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
.venv/bin/python ${D}/conc_sweep.py --max-conc ${c} --reps 3 --out "${OUT}/sweep.jsonl"
echo "=== done $(date +%T)"
