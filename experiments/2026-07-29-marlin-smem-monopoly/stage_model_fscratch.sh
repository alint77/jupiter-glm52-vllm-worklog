#!/usr/bin/env bash
# Stage the served checkpoint onto exa_fscratch.
#
# exa_project1, where the models live, is latency-bound at shallow queue depth
# and scales badly with concurrency (0.55 -> 0.95 GB/s from 4 to 8 parallel
# O_DIRECT readers). exa_fscratch reaches 11 -> 21 GB/s over the same range,
# which is the pattern four ranks loading simultaneously actually produce.
#
# The copy itself runs at over 2 GB/s, so staging 405 GB costs a few minutes.
set -euo pipefail

SRC="${1:-/e/project1/profound/alint77/models/GLM-5.2-AutoRound-W4G64-MTP-e1ba887}"
DST_ROOT="${2:-/e/fscratch/profound/naeimitabiei1/models}"
name="$(basename "$SRC")"
DST="${DST_ROOT}/${name}"

mkdir -p "$DST"
echo "staging $(du -sh "$SRC" 2>/dev/null | cut -f1) from $SRC"
echo "                          to $DST"
t0=$(date +%s)
cp -r --preserve=timestamps "$SRC"/. "$DST"/
t1=$(date +%s)
elapsed=$((t1 - t0))

dst_bytes=$(du -sb "$DST" 2>/dev/null | cut -f1)
rate=$(awk -v b="$dst_bytes" -v s="$elapsed" 'BEGIN{if(s>0) printf "%.2f", b/s/1e9; else print "n/a"}')

# Compare per-file sizes, not `du -sb` totals: du counts directory inodes, whose
# apparent size differs between filesystems, so a correct copy reports a
# spurious mismatch (16 KB on this tree).
diff_out=$(diff \
  <(cd "$SRC" && find . -type f -printf "%s %p\n" | sort -k2) \
  <(cd "$DST" && find . -type f ! -name '.stage_done' -printf "%s %p\n" | sort -k2) || true)

{
  echo "STAGE_DONE elapsed=${elapsed}s rate=${rate} GB/s"
  echo "files=$(find "$SRC" -type f | wc -l)"
  echo "size_mismatches=$(printf '%s' "$diff_out" | grep -c '^[<>]' || true)"
} | tee "${DST}/.stage_done"

if [[ -n "$diff_out" ]]; then
  echo "WARNING: per-file sizes differ - staging incomplete" >&2
  printf '%s\n' "$diff_out" | head -10 >&2
  exit 1
fi
echo "all files match in size"
