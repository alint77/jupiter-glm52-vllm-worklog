#!/usr/bin/env bash
# Sustained runs of each kernel mode with nvidia-smi sampling (GPU power and SM
# clock), to see where the 680 W cap goes.  power_probe.sh <gpu> <outdir> <bin:args>...
set -uo pipefail
gpu=$1; out=$2; shift 2
E=$(cd "$(dirname "$0")" && pwd)
for spec in "$@"; do
  bin=${spec%%:*}; args=${spec#*:}
  nvidia-smi -i "${gpu}" --query-gpu=power.draw.instant,clocks.sm,clocks_event_reasons.active --format=csv,noheader,nounits -lms 100 > "${out}/smi.txt" &
  pid=$!
  res=$(BENCH_SECS=4 "${E}/run_bound.sh" "${gpu}" "${E}/${bin}" bench ${args} | head -1)
  kill ${pid}; wait ${pid} 2>/dev/null
  # middle of the sustained window: drop the first 1.5 s and last 0.5 s of samples
  n=$(wc -l < "${out}/smi.txt")
  stats=$(sed -n "16,$((n - 5))p" "${out}/smi.txt" | awk -F', ' '{p+=$1; c+=$2; k++} END {printf "%.0f W  %.0f MHz  (%d samples)", p/k, c/k, k}')
  printf '%-8s %-22s %s | %s\n' "${bin}" "${args}" "${stats}" "$(echo "${res}" | sed 's/.*stages=4: *//')"
done
