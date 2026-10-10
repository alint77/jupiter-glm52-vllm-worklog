#!/usr/bin/env bash
# Is the kernel SM-bound per CTA? Served kernel at the mix with 8..32 cold CTAs.
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
V="td_v57:TD_MAX_TOKENS=64"
for m in "16 69,9" "32 108,17"; do set -- $m
  ./ab.sh e80-$1 $1 "$2" "$V TD_COLD_CTAS=8|--shared 1" "$V TD_COLD_CTAS=12|--shared 1" \
    "$V|--shared 1" "$V TD_COLD_CTAS=24|--shared 1" "$V TD_COLD_CTAS=32|--shared 1"
done
