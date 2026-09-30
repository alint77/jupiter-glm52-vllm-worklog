#!/usr/bin/env bash
# Same-node A/B of the cold prefetch slot count on the MiMo agentic task set.
#   s1   one slot, threshold 1024 (production)
#   s2   two slots (double buffered), threshold 1024
#   s2x  two slots, threshold 512
#   off  no prefetch (no slot budget: more hot experts)
# Six nodes run s1/s2/s2x in all six orders; two run off/s2 both ways; one runs
# a byte-verified s2x pass, then a prefill trace of s2x.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
E=agent_space/experiments/2026-09-27-glm53-mtp7-profile
B=agent_space/experiments/2026-09-28-agentic-decode-bench
D=agent_space/experiments/2026-09-30-glm-prefill-prefetch
OUT=/e/fscratch/profound/${USER}/agentic-bench
P=VLLM_TIERED_MOE_COLD_PREFETCH
s1="${P}_SLOTS=1"; s2="${P}_SLOTS=2"; s2x="${P}_SLOTS=2+${P}_MIN_TOKENS=512"
off="${P}_MIN_TOKENS=0"
hold() { sbatch --parsable --time="$1" "${E}/hold.sbatch"; }
run_node() {  # <job> <arm>...
  local j=$1; shift
  ( bash "${B}/node_ix.sh" "${j}" "$@"; scancel "${j}" ) \
    >"${OUT}/node-${j}.out" 2>&1 </dev/null &
}
jobs_file="${D}/jobs-$(date +%H%M%S).txt"
n=0
for order in "s1 s2 s2x" "s2 s2x s1" "s2x s1 s2" "s1 s2x s2" "s2 s1 s2x" "s2x s2 s1"; do
  n=$((n + 1)); j=$(hold 02:40:00); arms=()
  for a in ${order}; do arms+=("pf-${a}-n${n}:${!a}"); done
  echo "${j} ${arms[*]}" >>"${jobs_file}"; run_node "${j}" "${arms[@]}"
done
for order in "off s2" "s2 off"; do
  n=$((n + 1)); j=$(hold 01:55:00); arms=()
  for a in ${order}; do arms+=("pf-${a}-n${n}:${!a}"); done
  echo "${j} ${arms[*]}" >>"${jobs_file}"; run_node "${j}" "${arms[@]}"
done
j=$(hold 01:40:00); echo "${j} pf-vfy pf-prof" >>"${jobs_file}"
(
  until [[ "$(squeue -h -j "${j}" -o %T)" == RUNNING ]]; do
    [[ -z "$(squeue -h -j "${j}" -o %T)" ]] && exit 1; sleep 20
  done; sleep 10
  env PREFIX_CACHING=1 ${s2x//+/ } ${P}_VERIFY=1 HOLD_JOB="${j}" "${E}/onnode.sh" \
    "${B}/bench_node.sh glm pf-vfy --limit-requests 25" >"${OUT}/run-pf-vfy.log" 2>&1
  env PREFIX_CACHING=1 ${s2x//+/ } HOLD_JOB="${j}" \
    TRACE_ROOT="/e/fscratch/profound/${USER}/traces/glm53-prefill-s2x-${j}" PROFILE_WINDOWS=4 \
    "${E}/onnode.sh" "${B}/bench_node.sh glm pf-prof --limit-requests 30 --profile-prefill --profile-window 1.5" \
    >"${OUT}/run-pf-prof.log" 2>&1
  scancel "${j}"
) >"${OUT}/node-${j}.out" 2>&1 </dev/null &
cat "${jobs_file}"
