#!/usr/bin/env bash
# Merge the capture shards' route traces into one directory for the optimizer.
#
#   ./merge_traces.sh <out-dir> <job-id>...
#
# Traces are symlinked, and each shard's server manifest and driver attribution
# are concatenated. Request ids are unique across servers (random suffixes).
set -euo pipefail
out="${1:?usage: merge_traces.sh <out-dir> <job-id>...}"
shift
root="/e/fscratch/profound/${USER}/mimo26-route-cap"
rm -rf "${out}"
mkdir -p "${out}"
for job in "$@"; do
  for f in "${root}/${job}/routes/"*.npy; do
    ln -s "${f}" "${out}/$(basename "${f}")"
  done
  cat "${root}/${job}/routes/manifest.jsonl" >>"${out}/manifest.jsonl"
  cat "${root}/${job}/driver/requests.jsonl" >>"${out}/requests.jsonl"
done
echo "$(wc -l <"${out}/manifest.jsonl") traces from $# shards in ${out}"
