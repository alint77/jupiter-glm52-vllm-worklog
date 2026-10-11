#!/usr/bin/env bash
# Build a probe variant of _flashmla_C: recompile api.cpp and the V3.2 h64
# sparse decode instantiation with extra defines, relink with the other
# objects of the fmla-partial build.
#   build_variant.sh <name> [-DFMLA_TRACE] [-DFMLA_CONST_FILL] [-DFMLA_PDL]
# Output: /e/fscratch/profound/$USER/build/fmla-variants/<name>/_flashmla_C.abi3.so
set -euo pipefail
name=$1; shift
defs="$*"
B=/e/fscratch/profound/${USER}/build/fmla-partial
V=/e/fscratch/profound/${USER}/build/fmla-variants/${name}
mkdir -p "${V}"
cd "${B}"
O=CMakeFiles/_flashmla_C.dir/e/fscratch/profound/${USER}/src/FlashMLA/csrc
API=${O}/api/api.cpp.o
H64=${O}/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu.o
/e/project1/profound/alint77/vllm/.venv/bin/ninja -t commands _flashmla_C.abi3.so \
  | grep -E "api/api.cpp|v32_persistent_h64|-o _flashmla_C" > "${V}/cmds.txt"
sed -i -e "s#${API}#${V}/api.cpp.o#g" -e "s#${H64}#${V}/v32_h64.cu.o#g" \
  -e "s#-o _flashmla_C.abi3.so#-o ${V}/_flashmla_C.abi3.so#" \
  -e "s#--dependency-file=CMakeFiles/_flashmla_C.dir/link.d#--dependency-file=${V}/link.d#" \
  -e "s#-DPy_LIMITED_API=3#-DPy_LIMITED_API=3 ${defs}#" "${V}/cmds.txt"
sed -n 1p "${V}/cmds.txt" | bash & p1=$!
sed -n 2p "${V}/cmds.txt" | bash & p2=$!
wait $p1; wait $p2
sed -n 3p "${V}/cmds.txt" | bash
ls -la "${V}/_flashmla_C.abi3.so"
