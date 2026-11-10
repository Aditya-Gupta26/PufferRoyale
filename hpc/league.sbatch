#!/bin/bash
# PufferRoyale: league training (scripts/league_train.py) on one GPU of NYU Torch, then a round robin
# (scripts/tournament.py, sampled actions) of the newest snapshots against the scripted bots.
# Submit from our logs directory (docs/HPC_HANDOFF.md §6a, §9):
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     RUN_ID=A_s0 SEED=0 ANCHORS=none SELF_PLAY_FRAC=1.0 ANCHOR_FRAC=0 TOTAL=... NUM_ENVS=... BATCH_SIZE=... \
#         MINIBATCH_SIZE=... SNAPSHOT_INTERVAL=... EXTRA_ARGS="--env.reward-tower 0.3 ... --train.reward-clip 0" \
#         sbatch --job-name=pr-A_s0 $REPO/hpc/league.sbatch
# A NEW run needs ANCHORS, SELF_PLAY_FRAC and ANCHOR_FRAC set explicitly (no defaults: TRAINING_PLAN T10).
# Re-submitting (or a --requeue after preemption) with the same RUN_ID resumes the run from its last
# save (the last snapshot epoch). Re-submit with EXACTLY the same variables: TOTAL is the absolute
# target, EXTRA_ARGS is passed again, and NUM_ENVS / SNAPSHOT_INTERVAL override the saved run's values
# only when set; BATCH_SIZE / MINIBATCH_SIZE and the pool settings always come from the saved run.
# Extra flags (--env.*, --train.*, --rnn, --wandb ...) go through EXTRA_ARGS, never literally here
# (tests/spec/test_hpc_templates.py checks every literal flag against --help). HEARTBEAT=0 disables the
# GPU heartbeat (on by default: >= 75% utilization, DECISIONS D79). DECK0 / DECK1 may be presets or 8
# comma-separated card names. PR_CHECKOUT=<copy under $PR_ROOT> runs a pinned checkout.
# EVAL_EVERY=N (> 0) adds --eval-export-interval N and runs hpc/eval_watch.py next to the training: the
# learner vs the heuristic / random / noop bots every N epochs, EVAL_EPISODES matches per seat and bot
# (default 50), logged to <run>/eval_history.jsonl and, with EVAL_WANDB_GROUP=G, to the wandb run
# "<RUN_ID>-eval" in group G (DECISIONS D91). EVAL_ARGS passes further hpc/eval_watch.py options (D99: deck
# sets for the bots, frozen checkpoint opponents). OMP/MKL/OpenBLAS threads default to 1 (D90).
#
#SBATCH --job-name=pr-league
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=a100|h100|h200
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
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"

# ---- paths (env.sh sets them; override from the environment) -----------------------------
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/$USER/pufferroyale}"
SIF="${SIF:-$SCRATCH_DIR/containers/pufferroyale.sif}"   # built from hpc/pufferroyale.def
PROJECT="${PROJECT:-/opt/pufferroyale}"                   # our checkout (env.sh) or the image's copy
DATA_DIR="${DATA_DIR:-$SCRATCH_DIR/runs}"                 # league runs: $DATA_DIR/league/<run_id>/

# ---- league parameters ------------------------------------------------------------------------
RUN_ID="${RUN_ID:-league_${SLURM_JOB_ID:-local}}"
TOTAL="${TOTAL:-200000000}"
NUM_ENVS_SET="${NUM_ENVS+set}"                           # given by the submitter? (resume: override)
SNAPSHOT_INTERVAL_SET="${SNAPSHOT_INTERVAL+set}"
NUM_ENVS="${NUM_ENVS:-256}"
BATCH_SIZE="${BATCH_SIZE:-}"                             # new runs only; default num_envs x bptt_horizon
MINIBATCH_SIZE="${MINIBATCH_SIZE:-}"
PFSP="${PFSP:-hard}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-50}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-32}"
MMD_COEF="${MMD_COEF:-0}"
MMD_REF_INTERVAL="${MMD_REF_INTERVAL:-0}"
DECK0="${DECK0:-hog26}"
DECK1="${DECK1:-hog26}"
SEED="${SEED:-0}"
TOURNAMENT_MATCHES="${TOURNAMENT_MATCHES:-20}"          # 0 = no tournament
EVAL_EVERY="${EVAL_EVERY:-0}"                            # > 0: eval exports + hpc/eval_watch.py (D91)
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_WANDB_GROUP="${EVAL_WANDB_GROUP:-}"
EVAL_ARGS="${EVAL_ARGS:-}"                               # more hpc/eval_watch.py options (D99)
EVAL_WAIT="${EVAL_WAIT:-1200}"                           # seconds the sidecar may take to finish after training
# idle OpenMP workers spin and burn cores (SPS research, D90): one thread unless the submitter says otherwise
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
       OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
EVALX=()
if [ "$EVAL_EVERY" -gt 0 ]; then EVALX=(--eval-export-interval "$EVAL_EVERY"); fi

RUN_DIR="$DATA_DIR/league/$RUN_ID"
mkdir -p "$DATA_DIR/league"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): run=$RUN_DIR total=$TOTAL envs=$NUM_ENVS repo=$PROJECT" \
     "@ $(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi || true
pr_heartbeat_start
pr_isolation_check

APPTAINER=(apptainer exec --nv --no-home --bind "$SCRATCH_DIR:$SCRATCH_DIR" --pwd "$PROJECT" "$SIF")

# ---- evaluation sidecar (D91): the bots stay evaluation-only; nothing it does reaches the training
EVAL_PID=""
stop_eval() {   # SIGTERM (the sidecar also terminates its eval.py processes), at most 30 s, then SIGKILL
    if [ -n "$EVAL_PID" ]; then
        kill "$EVAL_PID" 2>/dev/null || true
        for _ in $(seq 30); do kill -0 "$EVAL_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$EVAL_PID" 2>/dev/null || true
        wait "$EVAL_PID" 2>/dev/null || true
        EVAL_PID=""
    fi
}
trap 'stop_eval; pr_heartbeat_stop' EXIT                 # replaces lib.sh's heartbeat-only trap

# ---- one job per run directory: refuse to start a second trainer on a run another live job owns
mkdir -p "$RUN_DIR"
OWNER_FILE="$RUN_DIR/.owner_job"
if [ -r "$OWNER_FILE" ]; then
    OTHER=$(cat "$OWNER_FILE" 2>/dev/null || true)
    if [ -n "$OTHER" ] && [ "$OTHER" != "${SLURM_JOB_ID:-local}" ] \
       && squeue -h -j "$OTHER" -t PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED 2>/dev/null | grep -q .; then
        echo "job $OTHER (still queued or running) owns $RUN_DIR: not starting a second trainer" >&2
        exit 5
    fi
fi
echo "${SLURM_JOB_ID:-local}" > "$OWNER_FILE"

if [ "$EVAL_EVERY" -gt 0 ]; then
    rm -f "$RUN_DIR/eval_ckpt/DONE"
    EW=(--episodes "$EVAL_EPISODES")
    if [ -n "$EVAL_WANDB_GROUP" ]; then EW+=(--wandb --wandb-group "$EVAL_WANDB_GROUP"); fi
    read -r -a EVX <<< "$EVAL_ARGS"
    EW+=(${EVX[@]+"${EVX[@]}"})
    nice -n 5 "${APPTAINER[@]}" python hpc/eval_watch.py "$RUN_DIR" "${EW[@]}" >> "$RUN_DIR/eval_watch.log" 2>&1 &
    EVAL_PID=$!
    echo "eval sidecar started (pid $EVAL_PID): every $EVAL_EVERY epochs, $EVAL_EPISODES matches per seat and bot"
fi

if [ -f "$RUN_DIR/league_state.json" ]; then
    echo "resuming $RUN_DIR"
    OVERRIDES=()
    [ -n "$NUM_ENVS_SET" ] && OVERRIDES+=(--num-envs "$NUM_ENVS")
    [ -n "$SNAPSHOT_INTERVAL_SET" ] && OVERRIDES+=(--snapshot-interval "$SNAPSHOT_INTERVAL")
    "${APPTAINER[@]}" python scripts/league_train.py \
        --resume "$RUN_DIR" \
        --total-timesteps "$TOTAL" \
        --device cuda \
        ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
        ${EVALX[@]+"${EVALX[@]}"} \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        --summary-json "$RUN_DIR/summary_${SLURM_JOB_ID:-local}.json"
else
    : "${ANCHORS:?a new run needs ANCHORS (e.g. none, or bot:random)}"
    : "${SELF_PLAY_FRAC:?a new run needs SELF_PLAY_FRAC}"
    : "${ANCHOR_FRAC:?a new run needs ANCHOR_FRAC}"
    SIZES=()
    [ -n "$BATCH_SIZE" ] && SIZES+=(--batch-size "$BATCH_SIZE")
    [ -n "$MINIBATCH_SIZE" ] && SIZES+=(--minibatch-size "$MINIBATCH_SIZE")
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
        ${SIZES[@]+"${SIZES[@]}"} \
        ${EVALX[@]+"${EVALX[@]}"} \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        --summary-json "$RUN_DIR/summary_${SLURM_JOB_ID:-local}.json"
fi

# ---- let the sidecar finish the remaining exports, then stop it
if [ -n "$EVAL_PID" ]; then
    mkdir -p "$RUN_DIR/eval_ckpt"
    touch "$RUN_DIR/eval_ckpt/DONE"
    for _ in $(seq "$EVAL_WAIT"); do kill -0 "$EVAL_PID" 2>/dev/null || break; sleep 1; done
    stop_eval
fi

# ---- evaluation: the 4 newest snapshots vs the scripted bots, payoff / Elo / meta-game Nash
mapfile -t SNAPS < <(ls -1v "$RUN_DIR"/snap_*.pt 2>/dev/null | tail -n 4)   # GNU ls: numeric order
DECKS=("$DECK0")                                       # one deck when both teams use the same one
if [ "$DECK1" != "$DECK0" ]; then DECKS+=("$DECK1"); fi
if [ "$TOURNAMENT_MATCHES" -gt 0 ] && [ "${#SNAPS[@]}" -gt 0 ]; then
    "${APPTAINER[@]}" python scripts/tournament.py \
        --agents bot:noop bot:random bot:heuristic "${SNAPS[@]}" \
        --decks "${DECKS[@]}" \
        --matches "$TOURNAMENT_MATCHES" \
        --sample \
        --seed "$SEED" \
        --device cuda \
        --out "$RUN_DIR/tournament_${SLURM_JOB_ID:-local}.json"
fi

echo "done: $RUN_DIR"
