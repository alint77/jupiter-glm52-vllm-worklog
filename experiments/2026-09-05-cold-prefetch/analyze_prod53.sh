#!/usr/bin/env bash
# Analyse the production GLM-5.3 traced pair from job-prod53.sh.
#
# Both arms are restricted to the same number of chunks: two captures rarely
# catch the same count, and context-growing roles (sparse_attn_indexer,
# top_k_per_row_prefill) climb monotonically, so an unmatched mean mixes a
# workload difference into the comparison.
set -euo pipefail

job="${1:?usage: analyze_prod53.sh <jobid>}"
repo=/e/project1/profound/alint77/vllm
here="${repo}/agent_space/experiments/2026-09-05-cold-prefetch"
rl="${repo}/agent_space/experiments/2026-09-04-mtp3-profile/prefill_roofline.py"
py="${repo}/.venv/bin/python"

declare -A root
for arm in baseline staged; do
  root[$arm]="$(cat "${here}/prod53-${arm}-tracepath-${job}.txt")/prefill"
  [[ -d "${root[$arm]}" ]] || { echo "missing trace: ${root[$arm]}" >&2; exit 1; }
done

# Chunk count each arm actually caught, so both can be cut to the smaller.
count_chunks() {
  "${py}" "${rl}" --root "$1" 2>/dev/null | grep -m1 "mean of" \
    | sed -E 's/.*mean of ([0-9]+) chunks.*/\1/'
}
nb="$(count_chunks "${root[baseline]}")"
ns="$(count_chunks "${root[staged]}")"
n=$(( nb < ns ? nb : ns ))
echo "chunks caught: baseline ${nb}, staged ${ns} -> comparing on ${n}"

for arm in baseline staged; do
  "${py}" "${rl}" --root "${root[$arm]}" --max-chunks "${n}" \
    --out "${here}/prod53-${arm}-roles-${job}.json" \
    > "${here}/prod53-${arm}-table-${job}.txt"
  echo "wrote prod53-${arm}-table-${job}.txt"
done

# Residency from each arm's own log, not a constant: the staged arm demotes
# hot experts to pay for the slot, so the two arms have different counts.
residency() {
  grep -ohE "Tiered MoE residency: [0-9]+ hot / [0-9]+ cold" \
    "${here}/prod53-${1}-server."{out,err} 2>/dev/null \
    | tail -1 | grep -oE "[0-9]+" | tr '\n' ' '
}
for arm in baseline staged; do
  read -r hot cold <<<"$(residency "${arm}")"
  tier=c2c; [[ "${arm}" == staged ]] && tier=hbm
  echo "=== ${arm}: ${hot} hot / ${cold} cold, cold tier reads ${tier} ==="
  "${py}" "${here}/prod_roofline_table.py" \
    --roles "${here}/prod53-${arm}-roles-${job}.json" \
    --hot-experts "${hot}" --cold-experts "${cold}" \
    --group 32 --cold-tier "${tier}" \
    | tee "${here}/prod53-${arm}-eff-${job}.txt"
done
