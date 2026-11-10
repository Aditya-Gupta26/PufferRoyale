#!/bin/bash
# All advisor tests (advisor/tests): encoder exactness + mutation check, policy wiring, tracker smoke test.
#     sbatch advisor/hpc/tests.sbatch          (from the advisor worktree)
#SBATCH --job-name=pr-advtest
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
export PR_CHECKOUT="${PR_CHECKOUT:-/scratch/$USER/pufferroyale/PufferRoyale}"   # the checkout with advisor/ (bound under $PR_ROOT)
source /scratch/$USER/pufferroyale/env.sh
cd "$PR_CHECKOUT"
echo "advisor @ $(git rev-parse --short HEAD) (+ uncommitted advisor/), host $(hostname)"
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" --env PYTHONPATH="$PR_ROOT/envs/advisor_py" "$SIF" \
    python3 -m pytest -q -p no:cacheprovider advisor/tests
