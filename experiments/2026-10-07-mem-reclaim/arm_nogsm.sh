#!/usr/bin/env bash
# arm.sh without GSM8K (user, 2026-10-10: accuracy is settled across these
# layouts; GSM8K=1 restores it). One arm on one node: prod serve.sh, then
# the agentic task set (32 requests, no profiler), long-context
# decode (50K / 130K, 2 seeds each), TTFT at three shapes and the stress_long
# session to 388K with per-second GPU memory. Extra env passes to serve.sh
# (before: PYTHONPATH=<HEAD worktree> SERVE_CACHE_ROOT=<own cache>; after:
# VLLM_TIERED_MOE_EMBED_HOST=1 RESERVE_GB=<swept>).
#   ./onnode.sh <D>/arm_nogsm.sh <tag>
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-07-mem-reclaim
B=agent_space/experiments/2026-09-28-agentic-decode-bench
tag=$1; OUT=/e/fscratch/profound/${USER}/mem-reclaim/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T)"
grep -ah "Tiered MoE observed HBM\|selected\|Input embedding" "${OUT}/server.out" | sed 's/^.*\] //' | sort | uniq -c
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 >"${OUT}/mem.csv" &
smi=$!
name=glm53-w4a16-tiered
# QUICK=1 (reserve sweep): only TTFT and the 388K stress
if [[ -z "${QUICK:-}" ]]; then
if [[ -n "${GSM8K:-}" ]]; then
TMPDIR=/e/fscratch/profound/${USER}/caches/gsm8k-data .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --port 8027 \
  --num-questions 200 --save-results "${OUT}/gsm8k.json" 2>&1 | tail -4
fi
.venv/bin/python ${B}/agentic_bench.py --model ${name} --out "${OUT}/rows.jsonl" \
  --tasks agent_space/experiments/2026-09-26-mimo-routing-profile/tasks-{0,1,2,3}.json --limit-requests 32 | tail -3
.venv/bin/python ${D}/decode_long.py --out "${OUT}/long.jsonl"
fi
.venv/bin/python ${D}/ttft_stress.py
kill ${smi}
alive=$(kill -0 "${pid}" 2>/dev/null && echo alive || echo dead)
echo "server ${alive}; OOM lines: $(grep -ac 'out of memory' "${OUT}/server.err")"
awk -F, '{if ($2 > m[$1]) m[$1] = $2} END {for (g in m) printf "gpu %s peak %d MiB\n", g, m[g]}' "${OUT}/mem.csv"
echo "=== done $(date +%T)"
