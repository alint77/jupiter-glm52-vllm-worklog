#!/usr/bin/env bash
# One server (this dir's serve.sh, prod defaults + extra env), then conc_probe.py.
#   ./onnode.sh <C>/run.sh <tag> [--profile]
#   c=2, k=3: MAX_NUM_SEQS=2 SPEC_K=3 CAPTURE_SIZES=4,8,16,32,...
set -uo pipefail
cd /e/project1/profound/alint77/vllm
C=agent_space/experiments/2026-10-08-c2-k3
tag=$1; shift
OUT=/e/fscratch/profound/${USER}/c2-k3/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export TRACE_ROOT="${OUT}/trace"; mkdir -p "${TRACE_ROOT}"
bash ${C}/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
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
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 >"${OUT}/mem.csv" &
smi=$!
.venv/bin/python ${C}/conc_probe.py --out "${OUT}/probe.jsonl" --trace-root "${TRACE_ROOT}" "$@"
kill ${smi}
awk -F, '{if ($2 > m[$1]) m[$1] = $2} END {for (g in m) printf "gpu %s peak %d MiB\n", g, m[g]}' "${OUT}/mem.csv"
echo "=== done $(date +%T)"
