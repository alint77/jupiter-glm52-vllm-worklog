#!/usr/bin/env bash
# c4 launcher config (claude-glm53-sliced-mtp3-c4.sh: MTP3, max_num_seqs 4,
# pool 2, reserve 3.6) with VLLM_SCHED_INTERLEAVE_PREFILL=<k>, then
# bench_interleave.py and 200 GSM8K questions (the policy changes which steps
# requests skip under async spec decode).
#   ./onnode.sh "<X>/arm.sh <tag> <k> [env...]"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
X=agent_space/experiments/2026-10-11-interleave
tag=$1 k=$2; shift 2
OUT=/e/fscratch/profound/${USER}/sliced-prof/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export TIERED_MOE_LAYOUT=tp_sliced REPLICAS=
export SPEC=mtp SPEC_K=3 COMPILE_SIZES= MAX_NUM_SEQS=4 VLLM_TIERED_MOE_KV_POOL_SEQS=2 RESERVE_GB=3.6
export CAPTURE_SIZES=1,2,3,4,5,6,7,8,12,16,64,128,256,384,512,640,768,896,1024
export VLLM_SCHED_INTERLEAVE_PREFILL=${k}
for kv in "$@"; do export "${kv}"; done
bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh >"${OUT}/server.out" 2>"${OUT}/server.err" &
pid=$!
trap 'pkill -f "bin/vllm [s]erve" 2>/dev/null; kill ${pid} 2>/dev/null; wait ${pid} 2>/dev/null' EXIT
for _ in $(seq 1 480); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "server failed"; grep -ohE "[A-Za-z]*(Error|Exception): .{0,200}" "${OUT}"/server.* | sort -u | head; exit 1; }
  sleep 5
done
echo "ready $(date +%T) k=${k}"
grep -ah "observed HBM" "${OUT}/server.out" | sed 's/^.*\] //' | sort | uniq -c
.venv/bin/python ${X}/bench_interleave.py --out "${OUT}/bench.jsonl"
TMPDIR=/e/fscratch/profound/${USER}/caches/gsm8k-data .venv/bin/python tests/evals/gsm8k/gsm8k_eval.py --port 8027 \
  --num-questions 200 --save-results "${OUT}/gsm8k.json" 2>&1 | tail -4
grep -a "SpecDecoding metrics" "${OUT}/server.out" | tail -1 | sed 's/^.*\] //' | cut -c1-200
echo "=== done $(date +%T)"
