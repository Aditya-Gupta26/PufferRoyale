#!/bin/bash
# Advisor encoder exactness check on many simulated matches (advisor/tests/test_encoder.py; docs/ADVISOR.md).
#     sbatch advisor/hpc/encoder_check.sbatch          (from the advisor worktree)
#SBATCH --job-name=pr-advenc
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
export PR_CHECKOUT="${PR_CHECKOUT:-/scratch/$USER/pufferroyale/PufferRoyale}"   # the checkout with advisor/ (bound under $PR_ROOT)
source /scratch/$USER/pufferroyale/env.sh
cd "$PR_CHECKOUT"
echo "advisor @ $(git rev-parse --short HEAD) (+ uncommitted advisor/), host $(hostname)"
for seed in 101 202 303; do
    apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" "$SIF" python3 -m advisor.tests.test_encoder --matches 200 --seed $seed
done
