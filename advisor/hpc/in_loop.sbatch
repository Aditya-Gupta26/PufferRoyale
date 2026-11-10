#!/bin/bash
# Check 5 (docs/ADVISOR.md): advisor pipeline variants playing in the simulator, one variant per array task.
#     sbatch --array=0-9 advisor/hpc/in_loop.sbatch          (from the advisor worktree; 9 = realistic)
#SBATCH --job-name=pr-advloop
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail
export PR_CHECKOUT="${PR_CHECKOUT:-/scratch/$USER/pufferroyale/PufferRoyale}"   # the checkout with advisor/ (bound under $PR_ROOT)
source /scratch/$USER/pufferroyale/env.sh
cd "$PR_CHECKOUT"
VARIANTS=(oracle no_spells no_hp blind tracker tracker_noisy delay1 delay2 noisy_delay2 realistic)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
MATCHES=${MATCHES:-400}
CKPT=${CKPT:-$PR_ROOT/runs/league/S4_A_s1/snap_476523.pt}
echo "advisor @ $(git rev-parse --short HEAD) (+ uncommitted advisor/), host $(hostname), variant $V, $MATCHES matches, $CKPT"
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" "$SIF" python3 -m advisor.eval.in_loop --variant "$V" \
    --matches "$MATCHES" --seed 1 --threads 4 --ckpt "$CKPT" --out "$PR_ROOT/runs/advisor/in_loop_${V}.json"
