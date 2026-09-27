#!/bin/bash
# PufferRoyale: plain PPO training (scripts/train.py) on one GPU of the NYU Torch cluster.
# Template -- replace the account / partition placeholders before submitting (see hpc/README.md):
#     sbatch hpc/train.sbatch
#     TOTAL=200000000 NUM_AGENTS=1 OPPONENT=heuristic sbatch hpc/train.sbatch
#
#SBATCH --job-name=pr-train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---- paths (override from the environment) ------------------------------------------------
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/$USER}"
SIF="${SIF:-$SCRATCH_DIR/containers/pufferroyale.sif}"   # built from hpc/pufferroyale.def
PROJECT="${PROJECT:-/opt/pufferroyale}"                   # inside the image (or a bind-mounted checkout)
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/pufferroyale}"         # checkpoints, summaries

# ---- run parameters ---------------------------------------------------------------------------
TOTAL="${TOTAL:-100000000}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_AGENTS="${NUM_AGENTS:-2}"             # 2 = self-play, 1 = vs the scripted OPPONENT
OPPONENT="${OPPONENT:-heuristic}"
DECK0="${DECK0:-hog26}"
DECK1="${DECK1:-hog26}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-65536}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-16384}"

mkdir -p "$DATA_DIR/train"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): total=$TOTAL envs=$NUM_ENVS agents=$NUM_AGENTS data=$DATA_DIR"
nvidia-smi || true

apptainer exec --nv --bind "$SCRATCH_DIR:$SCRATCH_DIR" --pwd "$PROJECT" "$SIF" \
    python scripts/train.py \
        --total-timesteps "$TOTAL" \
        --device cuda \
        --num-envs "$NUM_ENVS" \
        --num-agents "$NUM_AGENTS" \
        --opponent "$OPPONENT" \
        --deck0 "$DECK0" \
        --deck1 "$DECK1" \
        --batch-size "$BATCH_SIZE" \
        --minibatch-size "$MINIBATCH_SIZE" \
        --seed "$SEED" \
        --data-dir "$DATA_DIR/train" \
        --quiet \
        --summary-json "$DATA_DIR/train/summary_${SLURM_JOB_ID:-local}.json"

echo "done: $DATA_DIR/train"
