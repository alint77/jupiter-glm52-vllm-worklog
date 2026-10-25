#!/usr/bin/env bash
# Boot the pinned prod tree (prod_pin.sh) once on a hold, with a c=<c> TP-sliced
# MTP3 Claude launcher's environment, and stop it once healthy: the launcher's
# compile and extension caches are then warm for its next real boot. Run it
# after each prod_pin.sh, while no server of that launcher is running (two
# writers corrupt a compile cache). SERVE_CACHE_ROOT overrides the cache root.
#     prod_prewarm.sh <hold> <c>        c: 2 4 8 16
set -euo pipefail
cd /e/project1/profound/alint77/vllm
hold=$1 c=$2
tree=${CLAUDE_PROD_TREE:-/e/fscratch/profound/${USER}/worktrees/prod}
launcher=claude-glm53-sliced-mtp3-c${c}.sh
[[ -f ${launcher} && -d ${tree}/vllm ]] || { echo "need ${launcher} and ${tree}" >&2; exit 2; }
override_root=${SERVE_CACHE_ROOT:-}
eval "$(grep -E '^export (GLM53_|VLLM_MLA_QKV_GLUE)' ${launcher})"
conc=${GLM53_CONC} user=${USER}
eval "$(sed -n '/^conc=/,/^export SERVE_CACHE_ROOT/p' claude-glm53-sliced-mtp3.sh | grep -v '^conc=')"
[[ -n ${override_root} ]] && SERVE_CACHE_ROOT=${override_root}
export SERVE_CACHE_ROOT PYTHONPATH=${tree}
export SERVE_EXTRA="--host 127.0.0.1 --port 8027 --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45"
log=${SERVE_CACHE_ROOT}/prewarm-$(date +%Y%m%d-%H%M%S)
echo "$(date +%T) prewarm ${launcher} on hold ${hold}, $(git -C ${tree} log --oneline -1), cache ${SERVE_CACHE_ROOT}, log ${log}.{out,err}"
srun --jobid=${hold} --overlap --ntasks=1 --gres=gpu:4 --cpus-per-task=288 bash -c '
  source agent_space/jupiter-env.sh >/dev/null 2>&1
  t0=$(date +%s)
  bash agent_space/experiments/2026-09-27-glm53-mtp7-profile/serve.sh > '"${log}"'.out 2> '"${log}"'.err &
  srv=$!
  until curl -fsS http://127.0.0.1:8027/health >/dev/null 2>&1; do
    kill -0 ${srv} 2>/dev/null || { echo "server failed, see '"${log}"'.err"; exit 1; }
    sleep 5
  done
  echo "$(date +%T) healthy after $(( $(date +%s) - t0 )) s"
  kill ${srv}; wait ${srv} || true
  pkill -u ${USER} -f "[b]in/vllm serve" || true'
