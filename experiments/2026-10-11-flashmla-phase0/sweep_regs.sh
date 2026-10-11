#!/usr/bin/env bash
# Compile the V3.2 h64 sparse decode object for several warpgroup register
# splits (setmaxnreg WG0/WG1/producer) and count serialized wgmmas in SASS.
#   sweep_regs.sh "192,160,152 200,160,152 ..."
source /e/project1/profound/alint77/vllm/agent_space/jupiter-env.sh >/dev/null 2>&1
B=/e/fscratch/profound/${USER}/build/fmla-partial
H64=CMakeFiles/_flashmla_C.dir/e/fscratch/profound/${USER}/src/FlashMLA/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu.o
cd "${B}"
cmd=$(/e/project1/profound/alint77/vllm/.venv/bin/ninja -t commands _flashmla_C.abi3.so | grep v32_persistent_h64 | grep -v "abi3.so")
for r in $1; do
  IFS=, read r0 r1 r2 <<< "${r}"
  V=/e/fscratch/profound/${USER}/build/fmla-variants/regs-${r0}-${r1}-${r2}
  mkdir -p "${V}"
  ( c=${cmd//${H64}/${V}/v32_h64.cu.o}
    c=${c/-DPy_LIMITED_API=3/-DPy_LIMITED_API=3 -DFMLA_REG0=${r0} -DFMLA_REG1=${r1} -DFMLA_REG2=${r2} ${EXTRA:-}}
    bash -c "${c}" > "${V}/compile.log" 2>&1
    cuobjdump -sass "${V}/v32_h64.cu.o" > "${V}/sass.txt" 2>/dev/null
    echo "${r0}/${r1}/${r2}: HGMMA $(grep -c HGMMA ${V}/sass.txt) DEPBAR $(grep -c WARPGROUP.DEPBAR ${V}/sass.txt) C7512 $(grep -c C7512 ${V}/compile.log) spill $(grep -oE '[0-9]+ bytes spill stores' ${V}/compile.log | head -1)" ) &
done
wait
