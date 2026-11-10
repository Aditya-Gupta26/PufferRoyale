#!/bin/bash
# Check 6 (docs/ADVISOR.md): unit-detector accuracy on KataCR's labelled frames (all downloaded episodes).
#     sbatch advisor/hpc/katacr_eval.sbatch          (from the advisor worktree)
#SBATCH --job-name=pr-advkata
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
export PR_CHECKOUT="${PR_CHECKOUT:-/scratch/$USER/pufferroyale/PufferRoyale}"   # the checkout with advisor/ (bound under $PR_ROOT)
source /scratch/$USER/pufferroyale/env.sh
cd "$PR_CHECKOUT"
echo "advisor @ $(git rev-parse --short HEAD) (+ uncommitted advisor/), host $(hostname)"
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" --env PYTHONPATH="$PR_ROOT/envs/advisor_py" "$SIF" \
    python3 -m advisor.eval.katacr_eval --data "$PR_ROOT/research/ext/crds/images/part2" --threads 8 \
    --out "$PR_ROOT/runs/advisor/${OUT_NAME:-katacr_eval}.json"
