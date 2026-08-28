#!/usr/bin/env bash
# Submit the Phase 42 acceptance job with verified provenance.
#
# git does not exist on Booster compute nodes, so the working tree can only be
# checked here, on the login node. The resolved commit is passed to the job,
# which refuses to run if HEAD moved between submit and start.
set -euo pipefail

repo_dir=/e/project1/profound/alint77/vllm
cd "${repo_dir}"

git diff --quiet || { echo "REFUSING: tracked files are modified"; git status --short; exit 1; }
git diff --cached --quiet || { echo "REFUSING: staged changes present"; exit 1; }

commit="$(git rev-parse HEAD)"
branch="$(git rev-parse --abbrev-ref HEAD)"
echo "submitting ${commit:0:10} (${branch})"

exec sbatch --export=ALL,SOURCE_COMMIT="${commit}" \
  agent_space/experiments/2026-08-28-dflash2-backport/job-c1-acceptance.sh "$@"
