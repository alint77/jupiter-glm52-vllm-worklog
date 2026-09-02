#!/usr/bin/env bash
# One arm: bring up a 2-node TP8 sglang server for GLM-5.3-NVFP4, run the
# card's acceptance protocol, tear down. Invoked under `srun --ntasks=2
# --ntasks-per-node=1`; rank 0 also runs the benchmark.
set -euo pipefail
label="${1:?label}"; arm="${2:?arm}"; nsamples="${SAMPLES:-64}"
source /e/fscratch/profound/naeimitabiei1/sglang-fresh-20260902/env.sh
sglang_tree_check

RESULTS="${RESULT_DIR:?RESULT_DIR}"; mkdir -p "$RESULTS"
rank="${SLURM_PROCID:-0}"
head="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
port=30000; dist_port=29500
echo "rank=${rank} host=$(hostname) head=${head} arm=${arm} label=${label}"

case "$arm" in
  dflash2)
    # Card: block size 8 = 7 draft tokens + anchor. fa4 draft attention.
    spec=(--speculative-algorithm DFLASH
          --speculative-draft-model-path "$DFLASH2_DRAFT"
          --speculative-num-draft-tokens 8
          --speculative-draft-attention-backend "${DRAFT_ATTN:-fa4}") ;;
  mtp3)
    spec=(--speculative-algorithm NEXTN --speculative-num-steps 3
          --speculative-eagle-topk 1 --speculative-num-draft-tokens 4) ;;
  mtp7)
    spec=(--speculative-algorithm NEXTN --speculative-num-steps 7
          --speculative-eagle-topk 1 --speculative-num-draft-tokens 8) ;;
  none) spec=() ;;
  *) echo "unknown arm ${arm}"; exit 1 ;;
esac

python -m sglang.launch_server \
  --model-path "$GLM53_NVFP4" \
  --tp-size 8 --nnodes 2 --node-rank "${rank}" \
  --dist-init-addr "${head}:${dist_port}" \
  --host 127.0.0.1 --port "${port}" \
  --context-length 16384 \
  --mem-fraction-static "${MEM_FRAC:-0.85}" \
  --max-running-requests 1 \
  --trust-remote-code \
  --ep-size 8 \
  --dsa-prefill-backend "${DSA_PREFILL:-flashmla_sparse}" \
  --dsa-decode-backend "${DSA_DECODE:-flashmla_kv}" \
  ${EXTRA_ARGS:-} \
  "${spec[@]}" \
  >"${RESULTS}/${label}-server-r${rank}.out" 2>&1 &
pid=$!
done_flag="${RESULTS}/${label}.done"
rm -f "${done_flag}"

# Rank 1 owns no benchmark, so it must not simply `wait`: if rank 0 dies during
# dist-init the worker would sit idle for the whole allocation. Poll for the
# head's completion flag, for its own server dying, and for a wall-clock cap.
if [ "${rank}" != "0" ]; then
  for _ in $(seq 1 ${WORKER_POLLS:-2700}); do
    [ -f "${done_flag}" ] && { echo "head finished; worker shutting down"; break; }
    kill -0 "${pid}" 2>/dev/null || { echo "worker server exited"; break; }
    sleep 1
  done
  kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
  exit 0
fi

# Rank 0 must publish the flag on every exit path, or rank 1 polls to its cap.
trap 'touch "${done_flag}"' EXIT

for _ in $(seq 1 480); do
  curl -fsS "http://127.0.0.1:${port}/health_generate" >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || { echo "SERVER EXITED EARLY"
    grep -ohE "[A-Za-z]*(Error|Exception): .{0,160}" "${RESULTS}/${label}-server-r0.out" \
      | sort -u | head -5; exit 1; }
  sleep 5
done
echo "server ready"
# Record what the draft attention actually resolved to: fa4 may not be
# available on SM90, and a silent fallback would change the throughput story.
grep -ihoE "attention backend[^,]{0,60}|fa_impl_ver=[0-9]|flashinfer is unavailable[^\"]{0,60}" \
  "${RESULTS}/${label}-server-r0.out" | sort -u | head -5 || true

python "$SGLANG_ROOT/bench_al.py" --label "${label}" \
  --out "${RESULTS}/${label}-al.json" --base "http://127.0.0.1:${port}" \
  --model "$GLM53_NVFP4" --num-samples "${nsamples}" \
  --dataset /e/fscratch/profound/naeimitabiei1/models/datasets/gsm8k/test.jsonl \
  --prompt-field question
kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
touch "${done_flag}"
