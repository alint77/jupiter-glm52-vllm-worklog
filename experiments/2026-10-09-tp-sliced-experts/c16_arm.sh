#!/usr/bin/env bash
# MTP3 tp_sliced at up to 16 in flight (64-token sliced decode, skip-KV 64):
# the sweep config (1.6M pool), MAX_NUM_SEQS=16, captures every 4 tokens to 64,
# then conc_scale.py at n = 8, 12, 16.   ./onnode.sh "<D>/c16_arm.sh <tag> [env...]"
set -uo pipefail
cd /e/project1/profound/alint77/vllm
D=agent_space/experiments/2026-10-09-tp-sliced-experts
tag=$1; shift
OUT=/e/fscratch/profound/${USER}/sliced-prof/${tag}; mkdir -p "${OUT}"
export PREFIX_CACHING=1 SERVE_EXTRA="--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
export SPEC=mtp SPEC_K=3 MAX_NUM_SEQS=16 VLLM_TIERED_MOE_KV_POOL_SEQS=4 COMPILE_SIZES= RESERVE_GB=3.6
export TIERED_MOE_LAYOUT=tp_sliced REPLICAS=
export CAPTURE_SIZES=1,2,3,4,5,6,7,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,128,256,384,512,640,768,896,1024
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
grep -a "residency:" "${OUT}/server.out" | grep Worker_TP0 | tail -1 | sed 's/^.*\] //' | cut -c1-90
.venv/bin/python ${D}/conc_scale.py --out "${OUT}/scale.jsonl" --ns "${SCALE_NS:-8,12,16}" --ns50 "${SCALE_NS50:-8,12,16}" --reps "${REPS:-2}"
echo "=== done $(date +%T)"
