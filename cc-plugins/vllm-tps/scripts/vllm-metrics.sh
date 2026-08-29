#!/usr/bin/env bash
#
# Shared scrape helper for the vllm-tps plugin.
#
# Reads the backend endpoint from ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN,
# which claude-local.sh (and its per-model wrappers) export before exec'ing
# Claude Code. VLLM_TPS_BASE_URL / VLLM_TPS_AUTH_TOKEN override them.

vllm_tps_base_url() {
  printf '%s' "${VLLM_TPS_BASE_URL:-${ANTHROPIC_BASE_URL:-}}"
}

vllm_tps_auth_token() {
  printf '%s' "${VLLM_TPS_AUTH_TOKEN:-${ANTHROPIC_AUTH_TOKEN:-}}"
}

# Prints the raw Prometheus exposition, or nothing if the server is unreachable.
vllm_tps_scrape() {
  local url token timeout
  url="$(vllm_tps_base_url)"
  token="$(vllm_tps_auth_token)"
  timeout="${VLLM_TPS_TIMEOUT:-1}"
  [[ -n "${url}" ]] || return 1
  curl -fsS --connect-timeout "${timeout}" --max-time "${timeout}" \
    ${token:+-H "Authorization: Bearer ${token}"} \
    "${url%/}/metrics" 2>/dev/null
}

# vllm_tps_sum <metrics-blob> <metric-name>
#
# Sums one metric across every label set: vLLM emits one series per
# engine/model_name, and a data-parallel deployment emits one per replica.
vllm_tps_sum() {
  awk -v name="$2" '
    index($1, name "{") == 1 { total += $NF; found = 1 }
    END { if (!found) exit 1; printf "%.17g\n", total }
  ' <<<"$1"
}
