#!/usr/bin/env bash
#
# statusline.sh - Claude Code status line showing the backing vLLM server's
# throughput.
#
# Claude Code passes session JSON on stdin and renders stdout as the status
# line. The status line is a settings-level command, not a plugin component, so
# wire it up in ~/.claude/settings.json:
#
#   "statusLine": {
#     "type": "command",
#     "command": "<plugin>/statusline.sh",
#     "refreshInterval": 2,
#     "padding": 0
#   }
#
# Two regimes, because they answer different questions:
#
#   Generating - vllm:generation_tokens_total is a counter bumped every engine
#   step, so its delta over wall-clock between two scrapes is the aggregate
#   decode rate right now, across every request in flight. This is the number
#   you want while watching a response stream.
#
#   Idle - the counter rate collapses to zero the moment you stop prompting, so
#   fall back to the per-request histograms
#   (request_{prefill,decode}_time_seconds_sum against
#   request_{prefill_kv_computed,generation}_tokens_sum). Those accumulate per
#   completed request, so token delta over time delta is work divided by the
#   time that work actually took - independent of scrape cadence and of idle
#   gaps.
#
# When ANTHROPIC_BASE_URL is unset - an ordinary session against Anthropic's
# API - only the model and directory are rendered.

set -uo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/vllm-metrics.sh
source "${here}/scripts/vllm-metrics.sh"

MIN_INTERVAL="${VLLM_TPS_MIN_INTERVAL:-1}"
STATE_ROOT="${VLLM_TPS_STATE_DIR:-${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}/claude-vllm-tps}"

payload="$(cat)"

json_field() {
  printf '%s' "${payload}" | jq -r "${1} // empty" 2>/dev/null
}

session="$(json_field '.session_id')"
[[ -n "${session}" ]] || session="default"
model="$(json_field '.model.display_name')"
[[ -n "${model}" ]] || model="$(json_field '.model.id')"
cwd="$(json_field '.workspace.current_dir')"
[[ -n "${cwd}" ]] || cwd="${PWD}"

prefix="${model:-claude}"
[[ -n "${CLAUDE_LOCAL_LABEL:-}" ]] && prefix="${prefix} ${CLAUDE_LOCAL_LABEL}"
prefix="${prefix} · $(basename -- "${cwd}")"

emit() {
  printf '%s\n' "$*"
  exit 0
}

[[ -n "$(vllm_tps_base_url)" ]] || emit "${prefix}"
mkdir -p "${STATE_ROOT}" 2>/dev/null || emit "${prefix}"

state_file="${STATE_ROOT}/${session}.state"
render_file="${STATE_ROOT}/${session}.render"
last_file="${STATE_ROOT}/${session}.last"

# Throttle: Claude Code repaints the status line far more often than the server
# produces new numbers.
if [[ -s "${render_file}" ]]; then
  age="$(( $(date +%s) - $(stat -c %Y "${render_file}" 2>/dev/null || echo 0) ))"
  (( age < MIN_INTERVAL )) && emit "${prefix} · $(<"${render_file}")"
fi

metrics="$(vllm_tps_scrape)"
if [[ -z "${metrics}" ]]; then
  # Server asleep, restarting, or the job ended. Keep the last known figures
  # rather than flapping the status line to empty.
  [[ -s "${last_file}" ]] && emit "${prefix} · $(<"${last_file}") (stale)"
  emit "${prefix} · server unreachable"
fi

get() { vllm_tps_sum "${metrics}" "$1" 2>/dev/null || printf '0\n'; }

now="$(date +%s.%N)"
pf_tok="$(get 'vllm:request_prefill_kv_computed_tokens_sum')"
pf_sec="$(get 'vllm:request_prefill_time_seconds_sum')"
dc_tok="$(get 'vllm:request_generation_tokens_sum')"
dc_sec="$(get 'vllm:request_decode_time_seconds_sum')"
reqs="$(get 'vllm:request_generation_tokens_count')"
acc="$(get 'vllm:spec_decode_num_accepted_tokens_total')"
drf="$(get 'vllm:spec_decode_num_draft_tokens_total')"
gen="$(get 'vllm:generation_tokens_total')"
running="$(get 'vllm:num_requests_running')"
kv="$(get 'vllm:kv_cache_usage_perc')"

current="${now} ${pf_tok} ${pf_sec} ${dc_tok} ${dc_sec} ${reqs} ${acc} ${drf} ${gen} ${running} ${kv}"
previous=""
[[ -s "${state_file}" ]] && previous="$(<"${state_file}")"
printf '%s\n' "${current}" >"${state_file}"

read -r live completed < <(awk -v cur="${current}" -v prev="${previous}" '
BEGIN {
  n = split(cur, c, " ")
  m = split(prev, p, " ")
  # First sample of a session, or the server restarted and reset its counters.
  if (m < 11 || c[6] < p[6] || c[9] < p[9]) { print "- -"; exit }

  dt     = c[1] - p[1]
  d_gen  = c[9] - p[9]
  d_reqs = c[6] - p[6]
  d_acc  = c[7] - p[7]
  d_drf  = c[8] - p[8]

  acc_txt = (d_drf > 0) ? sprintf(" · acc %.0f%%", 100 * d_acc / d_drf) : ""

  live = "-"
  # Both samples busy, so the elapsed window is genuinely decode time rather
  # than a request that finished somewhere inside it.
  if (dt > 0 && d_gen > 0 && c[10] > 0 && p[10] > 0) {
    live = sprintf("▸ %.0f tok/s%s", d_gen / dt, acc_txt)
    if (c[10] > 1) live = live sprintf(" · %d req", c[10])
    if (c[11] > 0.005) live = live sprintf(" · kv %.0f%%", 100 * c[11])
    gsub(/ /, "|", live)
  }

  done = "-"
  if (d_reqs >= 1) {
    d_pf_tok = c[2] - p[2]; d_pf_sec = c[3] - p[3]
    d_dc_tok = c[4] - p[4]; d_dc_sec = c[5] - p[5]
    done = (d_pf_sec > 0 && d_pf_tok > 0) \
      ? sprintf("pf %.0f", d_pf_tok / d_pf_sec) : "pf -"
    done = done ((d_dc_sec > 0 && d_dc_tok > 0) \
      ? sprintf(" · dec %.1f tok/s", d_dc_tok / d_dc_sec) : " · dec -")
    done = done acc_txt
    if (d_reqs > 1) done = done sprintf(" · %d reqs", d_reqs)
    gsub(/ /, "|", done)
  }
  print live " " done
}')

live="${live//|/ }"
completed="${completed//|/ }"

[[ "${completed}" != "-" ]] && printf '%s' "${completed}" >"${last_file}"

if [[ "${live}" != "-" ]]; then
  render="${live}"
elif [[ "${completed}" != "-" ]]; then
  render="${completed}"
elif [[ -s "${last_file}" ]]; then
  # Idle: hold the last completed request's figures rather than blanking.
  render="$(<"${last_file}")"
else
  render="idle"
fi

printf '%s' "${render}" >"${render_file}"
emit "${prefix} · ${render}"
