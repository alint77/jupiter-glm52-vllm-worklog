#!/usr/bin/env bash
#SBATCH --account=profound
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=perf-duel
#SBATCH --output=agent_space/experiments/2026-09-23-perf-duel/sandbox-%j.out
#SBATCH --error=agent_space/experiments/2026-09-23-perf-duel/sandbox-%j.err

# Run the same GPU perf task through opencode against two local models at
# once, one GPU each, then grade both with pristine copies of the harness.
#
# Each agent gets an empty XDG config home: the user's global opencode config
# defines subagents on external models, which would contaminate the duel.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
E=agent_space/experiments/2026-09-23-perf-duel
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
BUDGET_S="${DUEL_BUDGET_S:-5400}"
run="/e/fscratch/profound/${USER}/perf-duel/${SLURM_JOB_ID}"
mkdir -p "${run}"
echo "run dir ${run}, budget ${BUDGET_S}s"

# name  model-id  state-dir  job  gpu
agents=(
  "mimo mimo26-pro /e/fscratch/profound/${USER}/claude-mimo26 ${MIMO_JOB} 0"
  "glm glm53-w4a16-tiered /e/fscratch/profound/${USER}/claude-glm53-c1-df2 ${GLM_JOB} 1"
)

if [[ -n "${DUEL_AGENTS:-}" ]]; then
  selected=()
  for spec in "${agents[@]}"; do
    [[ " ${DUEL_AGENTS} " == *" ${spec%% *} "* ]] && selected+=("${spec}")
  done
  agents=("${selected[@]}")
fi

for spec in "${agents[@]}"; do
  read -r name model state job gpu <<<"${spec}"
  until [[ -e "${state}/ready-${job}" ]]; do
    squeue -h -j "${job}" | grep -q . || { echo "${name} server ${job} gone"; exit 1; }
    sleep 20
  done
  source "${state}/endpoint-${job}"
  ws="${run}/${name}"
  mkdir -p "${ws}/xdg/config" "${ws}/xdg/data" "${ws}/xdg/state"
  cp "${E}/task/"{README.md,bench.py,reference.py,solution.py} "${ws}/"
  cat >"${ws}/opencode.json" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local vLLM",
      "options": {"baseURL": "http://${NODE}:${PORT}/v1", "apiKey": "$(<"${state}/api-token")"},
      "models": {"${model}": {"name": "${model}", "tool_call": true, "reasoning": true,
                  "limit": {"context": 240000, "output": 32000}}}
    }
  },
  "model": "local/${model}",
  "small_model": "local/${model}"
}
EOF
  echo "${name}: ${model} at ${NODE}:${PORT}, GPU ${gpu}, workspace ${ws}"
done

prompt="Read README.md in this directory and complete the task it describes. \
The Python interpreter to use is ${PY} (PyTorch 2.11 and Triton 3.6 are \
installed); your GPU is already selected via CUDA_VISIBLE_DEVICES. You have \
$((BUDGET_S / 60)) minutes of wall-clock time in total, after which you are \
stopped and the current solution.py is graded, so keep it passing at all \
times. Work autonomously: do not ask questions, iterate on performance with \
measurements, and record your approach and final numbers in NOTES.md."

pids=()
for spec in "${agents[@]}"; do
  read -r name model state job gpu <<<"${spec}"
  ws="${run}/${name}"
  (
    cd "${ws}"
    export CUDA_VISIBLE_DEVICES="${gpu}" PY="${PY}"
    export XDG_CONFIG_HOME="${ws}/xdg/config" XDG_DATA_HOME="${ws}/xdg/data"
    export XDG_STATE_HOME="${ws}/xdg/state"
    export TRITON_CACHE_DIR="${ws}/.triton"
    date +%s >"${ws}/start"
    timeout --signal=INT --kill-after=60 "${BUDGET_S}" \
      opencode run --standalone --auto -m "local/${model}" --format json \
      "${prompt}" >"${ws}/transcript.jsonl" 2>"${ws}/opencode.err" || true
    date +%s >"${ws}/end"
  ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "agents done at $(date +%T)"

# Grade with pristine harness copies; the agent's solution.py only.
for spec in "${agents[@]}"; do
  read -r name model state job gpu <<<"${spec}"
  ws="${run}/${name}"
  grade="${ws}/grade"
  mkdir -p "${grade}"
  cp "${E}/task/"{bench.py,reference.py} "${grade}/"
  cp "${ws}/solution.py" "${grade}/solution.py"
  (
    cd "${grade}"
    CUDA_VISIBLE_DEVICES="${gpu}" TRITON_CACHE_DIR="${grade}/.triton" \
      "${PY}" bench.py --json result.json >bench.out 2>&1 || true
  ) &
done
wait
for spec in "${agents[@]}"; do
  read -r name _ <<<"${spec}"
  echo "===== ${name}"
  tail -n 20 "${run}/${name}/grade/bench.out"
done
echo "DUEL DONE ${run}"
