#!/bin/bash
# Check 5, second round (docs/ADVISOR.md 5b): tracker v2 (robust to a real detector) and the real detector's
# measured error rates. One variant per array task; results in $PR_ROOT/runs/advisor/in_loop_v2/.
#     sbatch --array=0-4 advisor/hpc/in_loop_v2.sbatch          (from the advisor worktree)
#SBATCH --job-name=pr-advloop2
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail
export PR_CHECKOUT="${PR_CHECKOUT:-/scratch/$USER/pufferroyale/PufferRoyale}"   # the checkout with advisor/ (bound under $PR_ROOT)
source /scratch/$USER/pufferroyale/env.sh
cd "$PR_CHECKOUT"
VARIANTS=(oracle tracker real real_delay1 real_delay2)
MATCHES=(1000 400 400 400 400)
SEEDS=(2 1 1 1 1)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
N=${MATCHES[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPT:-$PR_ROOT/runs/league/S4_A_s1/snap_476523.pt}
echo "advisor @ $(git rev-parse --short HEAD) (+ uncommitted advisor/), host $(hostname), variant $V, $N matches, seed $S, $CKPT"
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" "$SIF" python3 -m advisor.eval.in_loop --variant "$V" \
    --matches "$N" --seed "$S" --threads 4 --ckpt "$CKPT" --out "$PR_ROOT/runs/advisor/in_loop_v2/in_loop_${V}.json"
