"""Stage an isolated copy of the Marlin MoE kernel with the shared-memory fix.

Nothing in the vLLM checkout is modified; the running server's _moe_C.so is
untouched.  Only the bf16 x u4b8, thread_m_blocks=1, group_blocks=8 (group 128)
instantiations are kept so the build is minutes rather than hours.
"""

import pathlib
import re
import shutil

SRC = pathlib.Path(
    "/e/project1/profound/alint77/vllm/csrc/libtorch_stable/moe/marlin_moe_wna16"
)
DST = pathlib.Path(__file__).parent / "marlin_tight"
DST.mkdir(exist_ok=True)

for f in ("kernel.h", "marlin_template.h"):
    shutil.copy(SRC / f, DST / f)

# ---- keep only the instantiations we need -------------------------------
KEEP = re.compile(
    r"vllm::kBFloat16\.id\(\), vllm::kU4B8\.id\(\), vllm::kBFloat16\.id\(\), "
    r"vllm::kBFloat16\.id\(\), (\d+), 1, (\d+), (\d+), (true|false), 4, 8, false"
)
inst = (SRC / "sm80_kernel_bfloat16_u4b8_bfloat16.cu").read_text().splitlines()
head = [ln for ln in inst[:8]]
kept = [ln for ln in inst if ln.startswith("template") and KEEP.search(ln)]
(DST / "sm80_kernel.cu").write_text("\n".join(head + kept) + "\n}\n")
print(f"instantiations kept: {len(kept)}")

# ---- matching selector chain -------------------------------------------
COND = re.compile(
    r"a_type == vllm::kBFloat16 && b_type == vllm::kU4B8 .*"
    r"thread_m_blocks == 1 .*stages == 4 && group_blocks == 8 "
    r"&& is_zp_float == false"
)
sel = (SRC / "kernel_selector.h").read_text().splitlines()
pairs = []
for i, ln in enumerate(sel):
    if COND.search(ln) and i + 1 < len(sel) and "kernel = Marlin<" in sel[i + 1]:
        cond = ln.split("if (", 1)[1]
        pairs.append((cond, sel[i + 1]))
out = ["// filtered by setup_tight.py", "// clang-format off"]
for j, (cond, asg) in enumerate(pairs):
    out.append(("if (" if j == 0 else "else if (") + cond)
    out.append(asg)
(DST / "kernel_selector.h").write_text("\n".join(out) + "\n")
print(f"selector branches kept: {len(pairs)}")
assert len(pairs) == len(kept), (len(pairs), len(kept))

# ---- ops.cu: drop the torch-stable wrapper, add the fix ------------------
ops = (SRC / "ops.cu").read_text()
ops = ops[: ops.index("torch::stable::Tensor moe_wna16_marlin_gemm(")]
ops = ops.replace(
    '#include <torch/csrc/stable/accelerator.h>\n', ""
).replace('#include <torch/csrc/stable/ops.h>\n', "")

# 1. two extra knobs on marlin_mm
ops = ops.replace(
    "int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,\n"
    "               bool use_fp32_reduce, bool is_zp_float) {",
    "int thread_n, int sms, int blocks_per_sm, bool use_atomic_add,\n"
    "               bool use_fp32_reduce, bool is_zp_float, int smem_override,\n"
    "               int grid_override) {",
)

# 2. request what the kernel needs, and let the caller size the grid.
old_launch = """  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  // avoid ">>>" being formatted to "> > >"
  // clang-format off
  kernel<<<blocks, num_threads, max_shared_mem, stream>>>("""
new_launch = """  // TIGHT-SMEM FIX.  Upstream launches with `max_shared_mem`, i.e.
  // deviceSharedMemOptin / blocks_per_sm, so every Marlin wave claims ~100% of
  // every SM's shared memory and no second Marlin kernel can ever become
  // co-resident.  Request `sh_cache_size` (what the kernel actually indexes)
  // instead, and let the caller choose the grid.
  int launch_smem = max_shared_mem;
  if (smem_override == -2) {
    launch_smem = ((sh_cache_size + 127) / 128) * 128;
  } else if (smem_override > 0) {
    launch_smem = smem_override;
  }
  STD_TORCH_CHECK(launch_smem >= sh_cache_size, "launch_smem = ", launch_smem,
                  " < sh_cache_size = ", sh_cache_size);
  int launch_blocks = grid_override > 0 ? grid_override : blocks;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       launch_smem);
  // avoid ">>>" being formatted to "> > >"
  // clang-format off
  kernel<<<launch_blocks, num_threads, launch_smem, stream>>>("""
assert old_launch in ops
ops = ops.replace(old_launch, new_launch)

# 3. report the chosen geometry so the benchmark can print it
ops = ops.replace(
    "  int thread_k_blocks = thread_k / 16;",
    """  if (getenv("MARLIN_TIGHT_VERBOSE")) {
    printf("[marlin] tile k=%d n=%d thr=%d bps=%d cache=%d launch_smem=%d "
           "grid=%d\\n",
           thread_k, thread_n, num_threads, exec_cfg.blocks_per_sm,
           get_kernel_cache_size(thread_tfg, m_block_size_8, thread_m_blocks,
                                 prob_m, prob_n, prob_k, num_bits, group_size,
                                 has_act_order, is_k_full, has_zp, is_zp_float,
                                 is_a_8bit, stages),
           smem_override == -2 ? -2 : max_shared_mem,
           grid_override > 0 ? grid_override : blocks);
  }
  int thread_k_blocks = thread_k / 16;""",
)
ops = ops.replace('#include "kernel.h"', '#include "kernel.h"\n#include <cstdio>\n#include <cstdlib>')
(DST / "ops.cu").write_text(ops)
print("ops.cu patched")
