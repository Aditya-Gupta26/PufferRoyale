#!/bin/bash
# PufferRoyale stage 3 (TRAINING_PLAN §8.3, HPC_HANDOFF §9): prepare -- and, with SUBMIT=1, submit -- the
# course comparison A / B / C (+ the MMD-sensitivity runs Clo / Chi) with the frozen stage-1 settings
# (DECISIONS D89: main) for either placement grid:
#     GRID=1  the default configuration ("main"; run ids S3_<run>_s<seed>)
#     GRID=2  the grid2 ablation the user keeps ready (D89; run ids S3g2_<run>_s<seed>)
# Every run is a hpc/league.sbatch run on one GPU. The runs go into ONE job array
# (hpc/stage3_array.sbatch) with at most MAX_CONCURRENT tasks at a time (the user's choice: 4).
# Nothing is submitted unless SUBMIT=1 (stage 3 needs the user's approval of its pre-registration).
#
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     TOTAL=1000000000 NUM_ENVS=256 BATCH_SIZE=16384 MINIBATCH_SIZE=1024 SNAPSHOT_INTERVAL=477 \
#         PR_CHECKOUT=$PR_ROOT/pinned/<freeze> bash $REPO/hpc/stage3_submit.sh      # writes + prints the plan
#     ... SUBMIT=1 bash $REPO/hpc/stage3_submit.sh                                  # submits the array
#
# Required: TOTAL, NUM_ENVS, BATCH_SIZE, MINIBATCH_SIZE, SNAPSHOT_INTERVAL (the pre-registered sizes;
# HANDOFF §9: SNAPSHOT_INTERVAL ~ (TOTAL / BATCH_SIZE) / 128 with MAX_SNAPSHOTS 64) and PR_CHECKOUT (the
# pinned checkout at the engine-freeze commit; every task runs it).
# Optional:
#   GRID (1), RUNS ("A B C Clo Chi"; Clo / Chi run seed 0 only, as the plan says), SEEDS ("0 1 2"),
#   MAX_SNAPSHOTS (64), PREFIX (S3), MAX_CONCURRENT (4), CPUS (24), CONSTRAINT (h100), MEM (96G),
#   TIME (72:00:00: a wall time over 48 h puts the tasks in QOS gpu168, 4 GPUs per user, separate from
#   gpu48 -- whose cap fell to 2 per user on 2026-10-04 -- and leaves margin for a ~26 h 1B-step run);
#   SHAPING (1: the frozen shaping + --shaping-anneal-frac 0.5; 0 drops it);
#   FAST (1: --env-threads $ENV_THREADS --fast-opponents --cuda-graphs, D90; 0 = the plain paths), ENV_THREADS (16);
#   EVAL_EVERY (100 epochs; 0 = no eval sidecar, D91), EVAL_EPISODES (50 per seat and bot);
#   WANDB (1), TOURNAMENT_MATCHES (league.sbatch's end-of-run check vs the bots, evaluation only), EXTRA.
# Output: SUBMIT=1 writes a NEW directory $DATA_DIR/stage3/<tag>/sub_<time>/ (one self-contained env file
# per run + manifest.txt) that the array's tasks read, so a later run of this script can never change what
# queued or requeued tasks do; a dry run writes $DATA_DIR/stage3/<tag>/dryrun/ (never used by a job).
# SUBMIT=1 refuses while a job named pr-<tag> is queued or running (and league.sbatch refuses a second
# trainer on a run directory). Submitting again later resumes every run from its last save (TOTAL is the
# absolute target; the shaping anneal is recomputed from the same TOTAL).
set -euo pipefail
: "${TOTAL:?TOTAL (learner steps per run)}" "${NUM_ENVS:?NUM_ENVS}" "${BATCH_SIZE:?BATCH_SIZE}"
: "${MINIBATCH_SIZE:?MINIBATCH_SIZE}" "${SNAPSHOT_INTERVAL:?SNAPSHOT_INTERVAL}"
: "${PR_CHECKOUT:?PR_CHECKOUT=<pinned checkout at the engine-freeze commit>}"
: "${DATA_DIR:?source env.sh first}"
GRID="${GRID:-1}"
RUNS="${RUNS:-A B C Clo Chi}"
SEEDS="${SEEDS:-0 1 2}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-64}"
PREFIX="${PREFIX:-S3}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
CPUS="${CPUS:-24}"
CONSTRAINT="${CONSTRAINT:-h100}"
MEM="${MEM:-96G}"
TIME="${TIME:-72:00:00}"
SHAPING="${SHAPING:-1}"
FAST="${FAST:-1}"
ENV_THREADS="${ENV_THREADS:-16}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
WANDB="${WANDB:-1}"
SUBMIT="${SUBMIT:-0}"
case "$GRID" in 1) TAG="$PREFIX" ;; 2|4) TAG="${PREFIX}g$GRID" ;; *) echo "GRID must be 1, 2 or 4" >&2; exit 2 ;; esac
[ -r "$PR_CHECKOUT/hpc/league.sbatch" ] || { echo "no hpc/league.sbatch in $PR_CHECKOUT" >&2; exit 2; }

# frozen stage-1 settings (D89): conditional head, feed-forward, royale.ini hyperparameters, fast cuDNN
COMMON="--train.torch-deterministic="
if [ "$SHAPING" = 1 ]; then
    COMMON="$COMMON --env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0 --shaping-anneal-frac 0.5"
fi
if [ "$GRID" != 1 ]; then COMMON="$COMMON --env.placement-grid $GRID"; fi
if [ "$FAST" = 1 ]; then COMMON="$COMMON --env-threads $ENV_THREADS --fast-opponents --cuda-graphs"; fi
if [ -n "${EXTRA:-}" ]; then COMMON="$COMMON $EXTRA"; fi

if [ "$SUBMIT" = 1 ]; then
    if squeue -h -u "$USER" --name "pr-$TAG" -t PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED | grep -q .; then
        echo "a job named pr-$TAG is still queued or running: not submitting another array" >&2
        squeue -u "$USER" --name "pr-$TAG" >&2
        exit 3
    fi
    OUT="$DATA_DIR/stage3/$TAG/sub_$(date +%Y%m%d_%H%M%S)"
    [ ! -e "$OUT" ] || { echo "$OUT exists" >&2; exit 3; }
else
    OUT="$DATA_DIR/stage3/$TAG/dryrun"
fi
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.txt"
: > "$MANIFEST.new"
q() { printf "'%s'" "${1//\'/\'\\\'\'}"; }          # single-quote a value for the env file
# array order: seed by seed (s0: A B C, s1: A B C, ...), then the seed-0-only runs (Clo, Chi), so that with
# MAX_CONCURRENT 4 every wave compares the methods on the same seeds
PAIRS=()
for s in $SEEDS; do
    for run in $RUNS; do
        case "$run" in Clo|Chi) ;; *) PAIRS+=("$run:$s") ;; esac
    done
done
for run in $RUNS; do
    case "$run" in Clo|Chi) PAIRS+=("$run:0") ;; esac
done
for pair in "${PAIRS[@]}"; do
    run="${pair%%:*}"
    s="${pair#*:}"
    # pool flags per run (TRAINING_PLAN §8.3): no scripted bots anywhere in training
    case "$run" in
        A)   POOL=(ANCHORS=none SELF_PLAY_FRAC=1.0 ANCHOR_FRAC=0 PFSP=hard MMD_COEF=0 MMD_REF_INTERVAL=0) ;;
        B)   POOL=(ANCHORS=none SELF_PLAY_FRAC=0.35 ANCHOR_FRAC=0 PFSP=hard MMD_COEF=0 MMD_REF_INTERVAL=0) ;;
        C)   POOL=(ANCHORS=none SELF_PLAY_FRAC=0.35 ANCHOR_FRAC=0 PFSP=hard MMD_COEF=0.05 MMD_REF_INTERVAL=50) ;;
        Clo) POOL=(ANCHORS=none SELF_PLAY_FRAC=0.35 ANCHOR_FRAC=0 PFSP=hard MMD_COEF=0.01 MMD_REF_INTERVAL=50) ;;
        Chi) POOL=(ANCHORS=none SELF_PLAY_FRAC=0.35 ANCHOR_FRAC=0 PFSP=hard MMD_COEF=0.1 MMD_REF_INTERVAL=50) ;;
        *) echo "unknown run '$run' (A B C Clo Chi)" >&2; exit 2 ;;
    esac
    {
        id="${TAG}_${run}_s$s"
        extra="$COMMON"
        if [ "$WANDB" = 1 ]; then extra="$extra --wandb --wandb-project pufferroyale --wandb-group ${TAG}_$run --tag $id"; fi
        f="$OUT/$id.env"
        {
            echo "# $id: written by hpc/stage3_submit.sh $(date -Iseconds)"
            echo "PR_CHECKOUT=$(q "$PR_CHECKOUT")"
            echo "RUN_ID=$(q "$id")"
            echo "SEED=$(q "$s")"
            for kv in "${POOL[@]}"; do echo "${kv%%=*}=$(q "${kv#*=}")"; done
            echo "TOTAL=$(q "$TOTAL")"
            echo "NUM_ENVS=$(q "$NUM_ENVS")"
            echo "BATCH_SIZE=$(q "$BATCH_SIZE")"
            echo "MINIBATCH_SIZE=$(q "$MINIBATCH_SIZE")"
            echo "SNAPSHOT_INTERVAL=$(q "$SNAPSHOT_INTERVAL")"
            echo "MAX_SNAPSHOTS=$(q "$MAX_SNAPSHOTS")"
            echo "EVAL_EVERY=$(q "$EVAL_EVERY")"
            echo "EVAL_EPISODES=$(q "$EVAL_EPISODES")"
            if [ "$WANDB" = 1 ]; then echo "EVAL_WANDB_GROUP=$(q "${TAG}_$run")"; else echo "EVAL_WANDB_GROUP=''"; fi
            echo "TOURNAMENT_MATCHES=$(q "${TOURNAMENT_MATCHES:-20}")"
            echo "DECK0='hog26'"
            echo "DECK1='hog26'"
            echo "HEARTBEAT='1'"
            echo "OMP_NUM_THREADS='1'"
            echo "MKL_NUM_THREADS='1'"
            echo "OPENBLAS_NUM_THREADS='1'"
            echo "PR_OPT_FLAGS=$(q "${PR_OPT_FLAGS--O3 -march=x86-64-v3}")"
            echo "EXTRA_ARGS=$(q "$extra")"
        } > "$f"
        echo "$f" >> "$MANIFEST.new"
    }
done
mv "$MANIFEST.new" "$MANIFEST"
N=$(wc -l < "$MANIFEST")
echo "manifest: $MANIFEST ($N runs)"
cat -n "$MANIFEST"
echo "--- first env file:"; cat "$(head -n 1 "$MANIFEST")"
CMD=(sbatch --array="0-$((N - 1))%$MAX_CONCURRENT" --job-name="pr-$TAG" --cpus-per-task="$CPUS"
     --constraint="$CONSTRAINT" --mem="$MEM" --time="$TIME" "$PR_CHECKOUT/hpc/stage3_array.sbatch" "$MANIFEST")
echo "submit: ${CMD[*]}"
if [ "$SUBMIT" = 1 ]; then
    "${CMD[@]}"
fi
