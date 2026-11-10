#!/bin/bash
# PufferRoyale: plain PPO training (scripts/train.py) on one GPU of NYU Torch. Every training stage
# runs through hpc/league.sbatch / hpc/stages.sbatch; this one is the train.py reference (Phase 3).
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     TOTAL=200000000 NUM_AGENTS=1 OPPONENT=heuristic sbatch $REPO/hpc/train.sbatch
#     BACKEND=Multiprocessing WORKERS=8 EXTRA_ARGS="--rnn --train.learning-rate 1e-4" sbatch $REPO/hpc/train.sbatch
# Extra flags (--env.*, --train.*, --rnn, --resume DIR, --wandb ...) go through EXTRA_ARGS, never
# literally here. HEARTBEAT=0 disables the GPU heartbeat (>= 75% utilization, DECISIONS D79).
#
#SBATCH --job-name=pr-train
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=a100|h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"

# ---- paths (env.sh sets them; override from the environment) -----------------------------
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/$USER/pufferroyale}"
SIF="${SIF:-$SCRATCH_DIR/containers/pufferroyale.sif}"   # built from hpc/pufferroyale.def
PROJECT="${PROJECT:-/opt/pufferroyale}"                   # our checkout (env.sh) or the image's copy
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/runs}"                 # checkpoints, summaries

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
BACKEND="${BACKEND:-PufferEnv}"           # or Multiprocessing (WORKERS processes of NUM_ENVS matches each)
WORKERS="${WORKERS:-}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
VEC=()
[ -n "$WORKERS" ] && VEC+=(--workers "$WORKERS")

mkdir -p "$DATA_DIR/train"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): total=$TOTAL envs=$NUM_ENVS agents=$NUM_AGENTS data=$DATA_DIR" \
     "repo=$PROJECT @ $(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi || true
pr_heartbeat_start
pr_isolation_check

apptainer exec --nv --no-home --bind "$SCRATCH_DIR:$SCRATCH_DIR" --pwd "$PROJECT" "$SIF" \
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
        --backend "$BACKEND" \
        --seed "$SEED" \
        --data-dir "$DATA_DIR/train" \
        --quiet \
        ${VEC[@]+"${VEC[@]}"} \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        --summary-json "$DATA_DIR/train/summary_${SLURM_JOB_ID:-local}.json"

echo "done: $DATA_DIR/train"
