#!/bin/bash
# PufferRoyale stage 3: one task of the job array that hpc/stage3_submit.sh submits. MANIFEST (the only
# argument) lists one env file per run; task i sources line i's file (RUN_ID, SEED, pool settings, sizes,
# EXTRA_ARGS, EVAL_*, PR_CHECKOUT) and runs hpc/league.sbatch's body with it, so every run is exactly a
# league.sbatch run (resume, heartbeat, isolation check, eval sidecar). `--array=0-N%K` keeps at most K
# runs (= K GPUs, one per task) going at once; re-submitting the same manifest resumes every run.
#
#SBATCH --job-name=pr-S3
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail
MANIFEST="${1:?usage: sbatch --array=0-N%K stage3_array.sbatch MANIFEST}"
i="${SLURM_ARRAY_TASK_ID:?not an array task}"
ENVFILE=$(sed -n "$((i + 1))p" "$MANIFEST")
[ -n "$ENVFILE" ] && [ -r "$ENVFILE" ] || { echo "task $i: no env file on line $((i + 1)) of $MANIFEST" >&2; exit 2; }
echo "task $i of array ${SLURM_ARRAY_JOB_ID:-?}: $ENVFILE"
set -a
# shellcheck disable=SC1090
source "$ENVFILE"
set +a
exec bash "${PR_CHECKOUT:?env file must set PR_CHECKOUT}/hpc/league.sbatch"
