#!/usr/bin/env bash
# ncu (full set + source counters) of one variant at one cell, GPU 0:
#   ncu1.sh <tag> <variant> <m> <cell> [flags]
cd /e/project1/profound/alint77/vllm/agent_space/experiments/2026-10-09-tp-sliced-experts
export VLLM_CACHE_ROOT=/e/fscratch/profound/${USER}/caches/kdev TMPDIR=/e/fscratch/profound/${USER}/caches/tmp-kdev
PY=/e/project1/profound/alint77/vllm/.venv/bin/python
tag=$1 v=$2 m=$3 cell=$4; shift 4
CUDA_VISIBLE_DEVICES=0 ncu --target-processes all --profile-from-start off -k regex:"layer_kernel" --set full \
  --import-source yes -o logs/ncu-$tag -f numactl --cpunodebind=0 --membind=0 $PY kdev.py once --v "$v" --m $m \
  --cell $cell --n 1 "$@" >/dev/null 2>>logs/ncu-$tag.err
ncu --import logs/ncu-$tag.ncu-rep --page raw 2>/dev/null | grep -E "issue_stalled.*per_issue_active.ratio|gpu__time_duration.sum |dram__throughput.avg.pct|dram__bytes_read.sum |lts__t_sector_hit_rate.pct |lts__t_sectors_srcunit_tex_op_read.sum |smsp__issue_active.avg.pct|sm__warps_active|l1tex__throughput.avg.pct|lts__throughput.avg.pct" | awk '{print $NF, $1}' | sort -rn | head -24
ncu --import logs/ncu-$tag.ncu-rep --page source --csv --print-source sass 2>/dev/null > logs/ncu-$tag-sass.csv
$PY sass_runs.py logs/ncu-$tag-sass.csv; $PY stall_regions.py logs/ncu-$tag-sass.csv --top 25
