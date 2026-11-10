#!/bin/bash
# Stage-4 evaluation at an analysis point (hpc/stage4_eval.py), on CPU nodes: array element k of K runs the tasks
# with id % K == k of DIR/tasks.jsonl, one single-threaded process per CPU (xargs -P); finished tasks are skipped,
# so re-submitting resumes. Merge afterwards: python hpc/stage4_eval.py merge DIR.
#
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     PR_CHECKOUT=<pinned> run python hpc/stage4_eval.py plan --point 2B --out $DATA_DIR/eval/stage4_2B
#     sbatch --array=0-7 --export=ALL,EVAL_DIR=$DATA_DIR/eval/stage4_2B,PR_CHECKOUT=<pinned> hpc/stage4_eval.sbatch
#SBATCH --job-name=pr-s4eval
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -uo pipefail
: "${EVAL_DIR:?set EVAL_DIR (the plan directory)}"
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
K="${SLURM_ARRAY_TASK_COUNT:-1}"
k="${SLURM_ARRAY_TASK_ID:-0}"
P="${SLURM_CPUS_PER_TASK:-4}"
echo "job ${SLURM_JOB_ID:-local} element $k of $K on $(hostname): $EVAL_DIR, $P processes, repo $PROJECT @ $(git -C "$PROJECT" rev-parse --short HEAD)"
pr_isolation_check
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N=$(wc -l < "$EVAL_DIR/tasks.jsonl")
seq 0 $((N - 1)) | awk -v K="$K" -v k="$k" '$1 % K == k' | \
    xargs -P "$P" -I{} "${PR_RUN[@]}" python hpc/stage4_eval.py run "$EVAL_DIR" --task {}
echo "done: $(ls "$EVAL_DIR/out" | grep -c '^[0-9]*\.json$') of $N task outputs present"
