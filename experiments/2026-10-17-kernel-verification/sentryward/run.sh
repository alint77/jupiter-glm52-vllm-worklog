#!/bin/bash
# SentryWard 60a1ebc on skinny_kernel<64,128> sm_90a PTX: baseline + mutants.
# Usage: sentryward/run.sh [workdir]   (CPU only; ~3 min)
set -u
D=$(cd "$(dirname "$0")" && pwd)
E=$(dirname "$D")
R=$(cd "$E/../../.." && pwd)
W=${1:-/e/fscratch/profound/naeimitabiei1/sentryward-runs}
mkdir -p "$W"
source "$D/swenv.sh"
P=$R/.venv/bin/python
INC=$($P -c "import torch.utils.cpp_extension as c; print(' '.join('-I'+p for p in c.include_paths('cuda')))")
PYI=$($P -c "import sysconfig; print(sysconfig.get_paths()['include'])")
ENTRY=skinny_kernelILi64ELi128

for m in 0 1 2 3; do
  nvcc -ptx -O3 -std=c++17 -arch=compute_90a $INC -I"$PYI" \
    -DTORCH_EXTENSION_NAME=skinny -DMUT=$m -I"$R/vllm/model_executor/layers/skinny_gemm" \
    "$E/mutants/skinny_mut.cu" -o "$W/mut$m.ptx" 2>/dev/null
  $P -I "$D/ptx_prep.py" "$W/mut$m.ptx" $ENTRY "$W/mut${m}_al.ptx" --assume-aligned
done
# K=512 grid 2: one tile per CTA, 8 K-steps = no ring wrap.
# K=4096 grid 1: two tiles x 64 K-steps, the 8-stage ring wraps 8x per tile.
$P -I "$D/mk_skinny.py" 2 512 "$W/g2_k512.json"
$P -I "$D/mk_skinny.py" 1 4096 "$W/g1_k4096.json"

run() {  # name ptx config
  timeout 2400 "$SWB" --ptx "$2" --config "$3" --report "$W/$1.json" >/dev/null 2>&1
  $P -I -c "
import collections, json, sys
r = json.load(open(sys.argv[1]))
codes = collections.Counter(d.get('code') for d in r.get('diagnostics', []))
print(f'{sys.argv[2]:<16} {r[\"result\"]:<16}', r['checks']['synchronization'],
      r['checks']['memory'], dict(codes) or '')" "$W/$1.json" "$1"
}
for m in 0 1 2; do
  run mut${m}_g2_k512 "$W/mut${m}_al.ptx" "$W/g2_k512.json"
  run mut${m}_g1_k4096 "$W/mut${m}_al.ptx" "$W/g1_k4096.json"
done
# Split-K (grid 3): pin the cross-CTA last-arriver branch to "flush".
L=$(awk '/ld.shared.u32.*4last\]/{f=1} f && /bra/{print NR; exit}' "$W/mut0_al.ptx")
$P -I -c "
import json, sys
c = json.load(open(sys.argv[1])); c['launch']['grid_dim'] = [3, 1, 1]
c['branch_decisions'] = [{'line': int(sys.argv[2]), 'taken': False,
                          'note': 'cross-CTA last-arriver flag: this CTA flushes'}]
json.dump(c, open(sys.argv[3], 'w'), indent=1)" "$W/g2_k512.json" "$L" "$W/g3_k512_last.json"
run mut0_g3_splitk "$W/mut0_al.ptx" "$W/g3_k512_last.json"
