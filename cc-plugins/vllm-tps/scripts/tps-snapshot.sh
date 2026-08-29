#!/usr/bin/env bash
#
# tps-snapshot.sh - one-shot throughput report for the backing vLLM server.
#
# Backs the /tps slash command. Takes two scrapes a short window apart so the
# live decode rate is a real measurement rather than a lifetime average.

set -uo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=vllm-metrics.sh
source "${here}/vllm-metrics.sh"

WINDOW="${VLLM_TPS_WINDOW:-2}"
export VLLM_TPS_TIMEOUT="${VLLM_TPS_TIMEOUT:-5}"

url="$(vllm_tps_base_url)"
if [[ -z "${url}" ]]; then
  echo "No vLLM backend configured (ANTHROPIC_BASE_URL / VLLM_TPS_BASE_URL unset)."
  exit 0
fi

first="$(vllm_tps_scrape)"
if [[ -z "${first}" ]]; then
  echo "vLLM backend at ${url} is unreachable."
  exit 0
fi
t0="$(date +%s.%N)"
sleep "${WINDOW}"
second="$(vllm_tps_scrape)"
t1="$(date +%s.%N)"
[[ -n "${second}" ]] || second="${first}"

g() { vllm_tps_sum "$1" "$2" 2>/dev/null || printf '0\n'; }

awk -v url="${url}" -v dt="$(awk -v a="${t0}" -v b="${t1}" 'BEGIN{printf "%.6f", b-a}')" \
    -v gen0="$(g "${first}" 'vllm:generation_tokens_total')" \
    -v gen1="$(g "${second}" 'vllm:generation_tokens_total')" \
    -v acc0="$(g "${first}" 'vllm:spec_decode_num_accepted_tokens_total')" \
    -v acc1="$(g "${second}" 'vllm:spec_decode_num_accepted_tokens_total')" \
    -v drf0="$(g "${first}" 'vllm:spec_decode_num_draft_tokens_total')" \
    -v drf1="$(g "${second}" 'vllm:spec_decode_num_draft_tokens_total')" \
    -v running="$(g "${second}" 'vllm:num_requests_running')" \
    -v waiting="$(g "${second}" 'vllm:num_requests_waiting')" \
    -v kv="$(g "${second}" 'vllm:kv_cache_usage_perc')" \
    -v reqs="$(g "${second}" 'vllm:request_generation_tokens_count')" \
    -v pf_tok="$(g "${second}" 'vllm:request_prefill_kv_computed_tokens_sum')" \
    -v pf_sec="$(g "${second}" 'vllm:request_prefill_time_seconds_sum')" \
    -v dc_tok="$(g "${second}" 'vllm:request_generation_tokens_sum')" \
    -v dc_sec="$(g "${second}" 'vllm:request_decode_time_seconds_sum')" \
    -v ttft_sum="$(g "${second}" 'vllm:time_to_first_token_seconds_sum')" \
    -v ttft_n="$(g "${second}" 'vllm:time_to_first_token_seconds_count')" \
    -v pc_hit="$(g "${second}" 'vllm:prefix_cache_hits_total')" \
    -v pc_q="$(g "${second}" 'vllm:prefix_cache_queries_total')" \
    -v prompt_tok="$(g "${second}" 'vllm:prompt_tokens_total')" \
    -v gen_tok="$(g "${second}" 'vllm:generation_tokens_total')" \
    -v preempt="$(g "${second}" 'vllm:num_preemptions_total')" '
function rate(num, den, unit) {
  return (den > 0) ? sprintf("%.1f %s", num / den, unit) : "n/a"
}
BEGIN {
  printf "vLLM %s\n\n", url
  live = (dt > 0 && gen1 > gen0) ? sprintf("%.1f tok/s decode", (gen1 - gen0) / dt) : "idle"
  printf "now       %-22s %d running, %d waiting\n", live, running, waiting
  if (drf1 > drf0)
    printf "          %.1f%% draft accept\n", 100 * (acc1 - acc0) / (drf1 - drf0)
  printf "kv cache  %.1f%% used\n", 100 * kv
  printf "\nlifetime (%d requests)\n", reqs
  printf "  prefill   %s\n", rate(pf_tok, pf_sec, "tok/s")
  printf "  decode    %s\n", rate(dc_tok, dc_sec, "tok/s")
  printf "  ttft      %s\n", rate(ttft_sum, ttft_n, "s avg")
  if (drf1 > 0)
    printf "  accept    %.1f%% of %.0f draft tokens\n", 100 * acc1 / drf1, drf1
  if (pc_q > 0)
    printf "  prefix hit %.1f%%\n", 100 * pc_hit / pc_q
  printf "  tokens    %.0f prompt / %.0f generated\n", prompt_tok, gen_tok
  if (preempt > 0)
    printf "  preempted %.0f\n", preempt
}'
