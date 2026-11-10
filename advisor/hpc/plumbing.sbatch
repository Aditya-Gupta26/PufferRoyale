#!/bin/bash
# The full advisor app on a KataCR episode (arena crops, no HUD): end-to-end run + per-stage timing.
#     sbatch advisor/hpc/plumbing.sbatch          (from the advisor worktree)
#SBATCH --job-name=pr-advplumb
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
export PR_CHECKOUT="${PR_CHECKOUT:-/scratch/$USER/pufferroyale/PufferRoyale}"   # the checkout with advisor/ (bound under $PR_ROOT)
source /scratch/$USER/pufferroyale/env.sh
cd "$PR_CHECKOUT"
O=$PR_ROOT/runs/advisor/plumbing
EP=${EP:-WTY_20240301/1}
NAME=${EP//\//_}
echo "advisor @ $(git rev-parse --short HEAD) (+ uncommitted advisor/), host $(hostname), episode $EP"
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" --env PYTHONPATH="$PR_ROOT/envs/advisor_py" "$SIF" \
    python3 -m advisor.app --ckpt "$PR_ROOT/runs/league/S4_A_s1/snap_476523.pt" \
    --deck "Hog Rider,Musketeer,Cannon,Ice Golem,Ice Spirit,Skeletons,Fireball,The Log" \
    --source images --images "$PR_ROOT/research/ext/crds/images/part2/$EP" --fps 2 \
    --layout-file "${LAYOUT:-$O/katacr_layout.json}" --assume-battle --threads 4 \
    --out "$O/$NAME.mp4" --log "$O/$NAME.jsonl"
