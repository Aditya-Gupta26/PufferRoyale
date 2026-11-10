#!/bin/bash
# Stage-4 collapse diagnostics, GPU part (hpc/collapse_gpu_check.py): one learner snapshot vs earlier snapshots
# through the training fast path (with and without the card floor) and the plain CPU path; bank logit checks.
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     RUN=S4_C_s0 LEARNER=1.5 OPPONENTS="1.0 1.1 1.2 1.3 1.4" PR_CHECKOUT=<pinned> \
#         sbatch --export=ALL $PR_CHECKOUT/hpc/collapse_gpu_check.sbatch
#SBATCH --job-name=pr-gpucheck
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=a100|h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
RUN="${RUN:?set RUN}"
LEARNER="${LEARNER:?set LEARNER (billions of steps)}"
read -r -a OPPS <<< "${OPPONENTS:?set OPPONENTS}"
OUT="$DATA_DIR/eval/collapse_diag/gpu_check_${RUN}_${LEARNER}B.json"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): $RUN learner ${LEARNER}B vs ${OPPS[*]}; repo $PROJECT @ $(git -C "$PROJECT" rev-parse --short HEAD)"
nvidia-smi || true
pr_heartbeat_start
pr_isolation_check
export OMP_NUM_THREADS=16
run python hpc/collapse_gpu_check.py --run "$RUN" --learner "$LEARNER" --opponents "${OPPS[@]}" --out "$OUT"
echo "done: $OUT"
