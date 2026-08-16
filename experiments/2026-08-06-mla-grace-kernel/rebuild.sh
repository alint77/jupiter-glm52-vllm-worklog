#!/usr/bin/env bash
# Rebuild the patched FlashMLA sparse kernel and install it into the tree.
#
# The kernel source is a CMake FetchContent checkout at .deps/flashmla-src, so
# edits there are not tracked by this repo and a clean rebuild would discard
# them. This script snapshots them to kernel.patch after every build, which is
# what makes the work recoverable. `git -C .deps/flashmla-src checkout .`
# reverts; `git apply` from the patch restores.
set -euo pipefail
cd /e/project1/profound/alint77/vllm
source agent_space/jupiter-env.sh
B=build/temp.linux-aarch64-cpython-312
D=agent_space/experiments/2026-08-06-mla-grace-kernel

ninja -C "$B" _flashmla_C
cp "$B/_flashmla_C.abi3.so" vllm/_flashmla_C.abi3.so
git -C .deps/flashmla-src diff > "$D/kernel.patch"
echo "installed $(stat -c%s vllm/_flashmla_C.abi3.so) bytes; patch $(wc -l < "$D/kernel.patch") lines"
