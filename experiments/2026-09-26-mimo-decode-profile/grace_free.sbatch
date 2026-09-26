#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=mimo26-gracefree
#SBATCH --output=agent_space/experiments/2026-09-26-mimo-decode-profile/grace-free-%j.out
#SBATCH --error=agent_space/experiments/2026-09-26-mimo-decode-profile/grace-free-%j.err

# How much Grace memory is actually free per NUMA node while the production
# MiMo server (routing profile, DFlash k=7) is up and has served traffic --
# the headroom a replica tier could use. Samples per-node meminfo before
# launch, after warmup, and after a decode + 128K prefill, plus per-process
# RSS by node.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
out=agent_space/experiments/2026-09-26-mimo-decode-profile/grace-free-${SLURM_JOB_ID}
mkdir -p "${out}"

snap() {
  local tag=$1
  for n in /sys/devices/system/node/node[0-9]*; do
    echo "== $(basename "$n")"
    grep -E "MemTotal|MemFree|MemUsed|FilePages|Active\(file\)|Inactive\(file\)|Mlocked|Unevictable|AnonPages|Shmem:|SReclaimable|HugePages_Total" "$n/meminfo"
  done > "${out}/meminfo-${tag}.txt"
  ps -eo pid,rss,comm --sort=-rss | head -20 > "${out}/ps-${tag}.txt"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "${out}/smi-${tag}.txt"
}

snap before
export MIMO_PORT=8027
export MIMO_EXTRA_ARGS="--language-model-only --tiered-moe-placement-profile ${PWD}/agent_space/experiments/2026-09-26-mimo-routing-profile/profile-3827.json"
sed -e 's/^exec \.venv/.venv/' -e '/^#SBATCH/d' \
  agent_space/experiments/2026-09-22-mimo-v26-pro/serve.sbatch > "${out}/serve-inline.sh"
bash "${out}/serve-inline.sh" >"${out}/server.out" 2>"${out}/server.err" &
pid=$!
trap 'kill ${pid} 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1 && break
  kill -0 "${pid}" 2>/dev/null || exit 1
  sleep 5
done
echo "server ready $(date +%T)"
snap ready
model=/e/fscratch/profound/${USER}/models/MiMo-V2.6-Pro-RL
.venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:8027 \
  --endpoint /v1/completions --model mimo26-pro --served-model-name mimo26-pro \
  --tokenizer "${model}" --trust-remote-code --max-concurrency 1 --request-rate inf \
  --dataset-name random --random-input-len 131072 --random-output-len 256 --num-prompts 1 \
  --ignore-eos --disable-tqdm 2>&1 | grep -E "TTFT|TPOT" | head -3
snap served
for tag in before ready served; do
  echo "##### ${tag}"
  awk '/^==/{n=$2} /MemTotal/{t=$4} /MemFree/{f=$4} /FilePages/{fp=$4} /Mlocked/{ml=$4} /AnonPages/{an=$4}
       /HugePages_Total/{printf "%s total %.1f GiB free %.1f file %.1f anon %.1f mlocked %.1f\n", n, t/1048576, f/1048576, fp/1048576, an/1048576, ml/1048576}' \
    "${out}/meminfo-${tag}.txt"
done
echo "=== done"
