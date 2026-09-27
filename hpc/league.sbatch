#!/bin/bash
# PufferRoyale: league training (scripts/league_train.py) on one GPU of the NYU Torch cluster,
# followed by a round-robin tournament of the latest snapshots against the scripted bots.
# Template -- replace the account / partition placeholders before submitting (see hpc/README.md):
#     RUN_ID=league_a sbatch hpc/league.sbatch
#     RUN_ID=league_mmd MMD_COEF=0.05 MMD_REF_INTERVAL=50 sbatch hpc/league.sbatch
# Re-submitting (or a --requeue after preemption) with the same RUN_ID resumes the run from its
# last saved state (the last snapshot epoch; matches in progress are re-dealt); TOTAL is the
# absolute target. DECK0 / DECK1 may be presets or 8 comma-separated card names.
#
#SBATCH --job-name=pr-league
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---- paths (override from the environment) ------------------------------------------------
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/$USER}"
SIF="${SIF:-$SCRATCH_DIR/containers/pufferroyale.sif}"   # built from hpc/pufferroyale.def
PROJECT="${PROJECT:-/opt/pufferroyale}"                   # inside the image (or a bind-mounted checkout)
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/pufferroyale}"         # league runs: $DATA_DIR/league/<run_id>/

# ---- league parameters ------------------------------------------------------------------------
RUN_ID="${RUN_ID:-league_${SLURM_JOB_ID:-local}}"
TOTAL="${TOTAL:-200000000}"
NUM_ENVS="${NUM_ENVS:-256}"
ANCHORS="${ANCHORS:-bot:heuristic,bot:random}"
PFSP="${PFSP:-hard}"
SELF_PLAY_FRAC="${SELF_PLAY_FRAC:-0.3}"
ANCHOR_FRAC="${ANCHOR_FRAC:-0.2}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-50}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-32}"
MMD_COEF="${MMD_COEF:-0}"
MMD_REF_INTERVAL="${MMD_REF_INTERVAL:-0}"
DECK0="${DECK0:-hog26}"
DECK1="${DECK1:-hog26}"
SEED="${SEED:-0}"
TOURNAMENT_MATCHES="${TOURNAMENT_MATCHES:-20}"

RUN_DIR="$DATA_DIR/league/$RUN_ID"
mkdir -p "$DATA_DIR/league"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): run=$RUN_DIR total=$TOTAL envs=$NUM_ENVS"
nvidia-smi || true

APPTAINER=(apptainer exec --nv --bind "$SCRATCH_DIR:$SCRATCH_DIR" --pwd "$PROJECT" "$SIF")

if [ -f "$RUN_DIR/league_state.json" ]; then
    echo "resuming $RUN_DIR"
    "${APPTAINER[@]}" python scripts/league_train.py \
        --resume "$RUN_DIR" \
        --total-timesteps "$TOTAL" \
        --device cuda \
        --num-envs "$NUM_ENVS" \
        --snapshot-interval "$SNAPSHOT_INTERVAL" \
        --summary-json "$RUN_DIR/summary_${SLURM_JOB_ID:-local}.json"
else
    "${APPTAINER[@]}" python scripts/league_train.py \
        --total-timesteps "$TOTAL" \
        --device cuda \
        --num-envs "$NUM_ENVS" \
        --anchors "$ANCHORS" \
        --pfsp "$PFSP" \
        --self-play-frac "$SELF_PLAY_FRAC" \
        --anchor-frac "$ANCHOR_FRAC" \
        --snapshot-interval "$SNAPSHOT_INTERVAL" \
        --max-snapshots "$MAX_SNAPSHOTS" \
        --mmd-coef "$MMD_COEF" \
        --mmd-ref-interval "$MMD_REF_INTERVAL" \
        --seed "$SEED" \
        --data-dir "$DATA_DIR" \
        --deck0 "$DECK0" \
        --deck1 "$DECK1" \
        --run-id "$RUN_ID" \
        --summary-json "$RUN_DIR/summary_${SLURM_JOB_ID:-local}.json"
fi

# ---- evaluation: the 4 newest snapshots vs the scripted bots, payoff / Elo / meta-game Nash
mapfile -t SNAPS < <(ls -1v "$RUN_DIR"/snap_*.pt 2>/dev/null | tail -n 4)   # GNU ls: numeric order
DECKS=("$DECK0")                                       # one deck when both teams use the same one
if [ "$DECK1" != "$DECK0" ]; then DECKS+=("$DECK1"); fi
if [ "${#SNAPS[@]}" -gt 0 ]; then
    "${APPTAINER[@]}" python scripts/tournament.py \
        --agents bot:noop bot:random bot:heuristic "${SNAPS[@]}" \
        --decks "${DECKS[@]}" \
        --matches "$TOURNAMENT_MATCHES" \
        --seed "$SEED" \
        --device cuda \
        --out "$RUN_DIR/tournament_${SLURM_JOB_ID:-local}.json"
fi

echo "done: $RUN_DIR"
