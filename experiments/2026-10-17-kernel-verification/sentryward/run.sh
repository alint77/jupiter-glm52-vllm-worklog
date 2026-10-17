#!/bin/bash
# SentryWard 60a1ebc on skinny_kernel<64,128> sm_90a PTX: baseline + mutants.
# Usage: sentryward/run.sh [rundir]   (CPU only)
# Exits nonzero if any step fails or any verdict differs from its expectation.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
E=$(dirname "$D")
R=$(cd "$E/../../.." && pwd)
W=${1:-/e/fscratch/profound/naeimitabiei1/sentryward-runs/$(date +%Y%m%d-%H%M%S)}
[[ -e $W ]] && { echo "run dir $W exists; use a fresh one" >&2; exit 2; }
mkdir -p "$W"
source "$D/swenv.sh"
P=$R/.venv/bin/python
INC=$($P -c "import torch.utils.cpp_extension as c; print(' '.join('-I'+p for p in c.include_paths('cuda')))")
PYI=$($P -c "import sysconfig; print(sysconfig.get_paths()['include'])")
ENTRY=skinny_kernelILi64ELi128
{ nvcc --version | tail -1; git -C "$R" rev-parse HEAD; sha256sum "$E/mutants/skinny_mut.cu"; } > "$W/provenance.txt"

MUTS="0 1 2 4 5 6 7 8 9 10 11 12"
for m in $MUTS; do
  nvcc -ptx -O3 -std=c++17 -arch=compute_90a $INC -I"$PYI" \
    -DTORCH_EXTENSION_NAME=skinny -DMUT=$m -I"$R/vllm/model_executor/layers/skinny_gemm" \
    "$E/mutants/skinny_mut.cu" -o "$W/mut$m.ptx" 2> "$W/mut$m.nvcc.log"
  $P -I "$D/ptx_prep.py" "$W/mut$m.ptx" $ENTRY "$W/mut${m}_al.ptx" --assume-aligned \
    --expect-prefetch 2 --expect-align 3 --expect-bfi 1 > "$W/mut$m.prep.log"
done
sha256sum "$W"/mut*_al.ptx >> "$W/provenance.txt"

# One stage iteration is a 128-wide K chunk. Grid 2 at N=128 gives each CTA
# one tile, so a CTA runs K/128 iterations over the 8-stage ring:
# 4 (no reuse), 7 = S-1, 8 = S, 9 = S+1, 17 = 2S+1, 25 = 3S+1.
# Grid 1 at K=4096: 2 tiles x 32 chunks = 64 iterations, 8 uses per stage.
for k in 512 896 1024 1152 2176 3200; do
  $P -I "$D/mk_skinny.py" 2 $k "$W/g2_k$k.json"
done
$P -I "$D/mk_skinny.py" 1 4096 "$W/g1_k4096.json"
L=$(awk '/ld.shared.u32.*4last\]/{f=1} f && /bra/{print NR; exit}' "$W/mut0_al.ptx")
$P -I -c "
import json, sys
c = json.load(open(sys.argv[1])); c['launch']['grid_dim'] = [3, 1, 1]
c['branch_decisions'] = [{'line': int(sys.argv[2]), 'taken': False,
                          'note': 'cross-CTA last-arriver flag: flush path'}]
json.dump(c, open(sys.argv[3], 'w'), indent=1)" "$W/g2_k512.json" "$L" "$W/g3_k512_last.json"

bad=0
run() {  # name ptx config expect(ok|bug|unsupported|any)
  local rc=0
  timeout 2400 "$SWB" --ptx "$2" --config "$3" --report "$W/$1.json" \
    > "$W/$1.log" 2>&1 || rc=$?
  [[ -s $W/$1.json ]] || { echo "$1: no report (rc=$rc)"; bad=1; return; }
  $P -I -c "
import collections, json, sys
name, expect = sys.argv[2], sys.argv[3]
r = json.load(open(sys.argv[1]))
res = r['result']
codes = dict(collections.Counter(d.get('code') for d in r.get('diagnostics', [])))
ok = {'ok': res == 'race_free',
      'bug': res in ('not_wellsynced', 'potential_race'),
      'unsupported': res == 'unsupported', 'any': True}[expect]
print(f'{name:<18} {res:<15} sync={r[\"checks\"][\"synchronization\"]:<10} '
      f'mem={r[\"checks\"][\"memory\"]:<8} expect={expect:<11} '
      f'{\"OK \" if ok else \"MISMATCH\"} {codes or \"\"}')
sys.exit(0 if ok else 1)" "$W/$1.json" "$1" "$4" || bad=1
}

for k in 512 896 1024 1152 2176 3200; do
  run mut0_g2_k$k "$W/mut0_al.ptx" "$W/g2_k$k.json" ok
done
run mut0_g1_k4096 "$W/mut0_al.ptx" "$W/g1_k4096.json" ok
run mut1_g2_k512 "$W/mut1_al.ptx" "$W/g2_k512.json" bug
run mut2_g2_k512 "$W/mut2_al.ptx" "$W/g2_k512.json" ok  # no reuse: harmless
for m in 1 2 4 5 6 7 8 9 12; do
  run mut${m}_g2_k2176 "$W/mut${m}_al.ptx" "$W/g2_k2176.json" bug
done
# outside the model (no fence semantics; ldmatrix is warp-collective)
for m in 10 11; do
  run mut${m}_g2_k2176 "$W/mut${m}_al.ptx" "$W/g2_k2176.json" any
done
run mut0_g3_splitk "$W/mut0_al.ptx" "$W/g3_k512_last.json" unsupported
echo "run dir: $W"
exit $bad
