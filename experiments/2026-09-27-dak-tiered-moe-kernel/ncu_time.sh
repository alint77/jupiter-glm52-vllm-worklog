#!/usr/bin/env bash
# Median kernel time at the pinned base clock (KREGEX, default sk_kernel):
#   ncu_time.sh <gpu> <binary> bench <args...>
set -uo pipefail
gpu=$1; shift
E=$(cd "$(dirname "$0")" && pwd)
"${E}/run_bound.sh" "${gpu}" ncu --clock-control base --csv --metrics gpu__time_duration.sum \
  -k "regex:${KREGEX:-sk_kernel}" -c 7 "$@" 2>/dev/null | grep gpu__time |
  awk -F'","' '{gsub(/[",]/, "", $NF); print $NF / 1000}' | sort -n |
  awk '{a[NR] = $1} END {printf "%.1f us\n", a[int((NR + 1) / 2)]}'
