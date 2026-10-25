#!/usr/bin/env bash
# Pin the Claude Code servers to one vLLM commit in a worktree at a fixed path,
# so their torch.compile and JIT extension caches stay valid across work in the
# main tree (2026-10-25-boot-time: a cache miss costs ~4 min a boot, and the
# caches are tied to the tree's path as well as its code). claude-server.sbatch
# runs from this tree (PYTHONPATH) when it exists. The main tree's built
# artifacts (.so extensions, vllm_flash_attn/, third_party/, _version.py) are
# copied in; the venv stays the main tree's.
#     prod_pin.sh [<commit>]      default HEAD; prints the pinned commit
set -euo pipefail
repo=/e/project1/profound/alint77/vllm
tree=${CLAUDE_PROD_TREE:-/e/fscratch/profound/${USER}/worktrees/prod}
commit=$(git -C ${repo} rev-parse --verify "${1:-HEAD}^{commit}")
if [[ -d ${tree} ]]; then
  [[ -z $(git -C ${tree} status --porcelain --untracked-files=no) ]] \
    || { echo "${tree} has local changes" >&2; exit 1; }
  git -C ${tree} checkout -q --detach ${commit}
else
  git -C ${repo} worktree add -q --detach ${tree} ${commit}
fi
cd ${repo}
git status --ignored --porcelain vllm | sed -n 's/^!! //p' | grep -v __pycache__ \
  | while read -r p; do
      rm -rf "${tree:?}/${p}"
      cp -a --parents "${p%/}" ${tree}/
    done
echo "prod tree ${tree} at $(git -C ${tree} log --oneline -1)"
