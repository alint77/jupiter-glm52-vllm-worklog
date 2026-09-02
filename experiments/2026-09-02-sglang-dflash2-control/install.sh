#!/usr/bin/env bash
# Fresh sglang install for the DFlash2/MTP acceptance replication.
#
# Rust extensions are deliberately NOT built. This is a decision, not a
# workaround for a missing toolchain (JUPITER does ship Rust as a module):
#
#   - the rust/ workspace builds sglang-grpc, sglang-mm, sglang-radix-tree and
#     sglang-server. Nothing under srt/ imports sglang_radix_tree -- the Python
#     radix cache is what serves -- and the only consumers of the artifacts are
#     srt/rust_server/*, an optional embedded-server path that
#     sglang.launch_server does not take in this configuration (verified by
#     grep over the installed package);
#   - the workspace pins rust-toolchain channel 1.92, while the module built
#     against our GCCcore-14.3.0 toolchain is Rust/1.88.0. Rust/1.94.1 exists
#     but only under Stages/2027 + GCCcore/15.2.0, which would change CUDA and
#     NCCL underneath the whole install.
#
# So: skipped because unused by the serving path, not because cargo was absent.
set -euo pipefail
ROOT=/e/fscratch/profound/naeimitabiei1/sglang-fresh-20260902
export UV_CACHE_DIR=/e/fscratch/profound/naeimitabiei1/uv-cache
export XDG_CACHE_HOME=/e/fscratch/profound/naeimitabiei1/cache
export VIRTUAL_ENV="$ROOT/venv"
module load Stages/2026 >/dev/null 2>&1 || true
module load GCC/14.3.0 CUDA/13 CMake/3.31.8 NCCL/default-CUDA-13 Ninja/1.13.0 >/dev/null 2>&1 || true
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export SGLANG_BUILD_RUST_EXTS=none
echo "=== arch=$(uname -m) CUDA_HOME=$CUDA_HOME rust_exts=none"
uv pip install --python "$ROOT/venv/bin/python" \
  "sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python"
