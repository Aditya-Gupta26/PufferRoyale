#!/bin/bash
# PufferRoyale stage 4 (TRAINING_PLAN §8.4, DECISIONS D99): prepare -- and, with SUBMIT=1, submit -- the
# general-policy runs on all 64 cards: method C (primary) and method A (the pure self-play ablation), each run on
# one GPU, trained to TOTAL learner steps in a CHAIN of jobs of at most TIME each (the user's choice: 20 h). Link
# k+1 starts when link k has ended (afterany) and resumes the run from its last save (hpc/league.sbatch); a link
# that finds the run finished does nothing. Every link is one task of hpc/stage3_array.sbatch reading the run's
# self-contained env file, exactly as in stage 3.
#
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     TOTAL=10000000000 LINKS=12 PR_CHECKOUT=$PR_ROOT/pinned/<commit> bash $REPO/hpc/stage4_submit.sh   # plan only
#     ... SUBMIT=1 bash $REPO/hpc/stage4_submit.sh                                                     # submits
#
# Required: TOTAL (learner steps per run), LINKS (jobs per run), PR_CHECKOUT (the pinned checkout every link runs).
# Optional:
#   RUNS ("C:0:h100|h200 A:0:h100|h200 C:1:a100 A:1:a100" = method:seed:GPU constraint, D99), PREFIX (S4),
#   TIME (20:00:00), CPUS (24), MEM (96G), NUM_ENVS (256), BATCH_SIZE (16384), MINIBATCH_SIZE (1024),
#   SNAPSHOT_INTERVAL (477 epochs, stage 3's), MAX_SNAPSHOTS (64), SHAPING_STEPS (500000000: the shaping fades over
#   the first 500M learner steps as in stage 3, whatever TOTAL is; 0 = constant, D102), DECK_POOL, RANDOM_DECK_FRAC (0.5),
#   HELDOUT ("xbow;royal_hogs;random:50:20261003"), EVAL_EVERY (6104 epochs = 100,007,936 steps), EVAL_EPISODES (10
#   per seat and deck), EVAL_BOTS ("heuristic random"), SPECIALISTS (stage-3 finals the sidecar plays in the Hog 2.6
#   mirror; default A_s1 and C_s0, the best final per method by E2 Elo), CKPT_MATCHES (200), LOG_INTERVAL (10),
#   CARD_FLOOR (0.005) and CARD_FLOOR_STEPS (5000000000: the card-probability floor of D100, faded to 0 over the
#   first 5B learner steps; 0 = constant, D102)
#   first 5B learner steps), ENV_THREADS (16), EXTRA.
# Claude games (D101): LLM_RUNS ("C:0" = method:seed of the runs that get them; "" = none), LLM_EVERY (250000000
#   learner steps per game), LLM_LINKS (7) jobs of LLM_TIME (2-00:00:00) each on a CPU node (LLM_CPUS 4, LLM_MEM
#   16G), LLM_MODEL (claude-opus-5-5), LLM_EFFORT (low): one hpc/llm.sbatch chain per run, started at once (it waits
#   for the run's saved policies), job name pr-<run>-llm.
# Output: SUBMIT=1 writes a NEW directory $DATA_DIR/stage4/<PREFIX>/sub_<time>/ (one env file per run, manifest.txt
# and chains.txt with every link's job id); a dry run writes $DATA_DIR/stage4/<PREFIX>/dryrun/. SUBMIT=1 refuses
# while a job named pr-<PREFIX>_* is queued or running.
set -euo pipefail
: "${TOTAL:?TOTAL (learner steps per run)}" "${LINKS:?LINKS (jobs per run)}"
: "${PR_CHECKOUT:?PR_CHECKOUT=<pinned checkout>}" "${DATA_DIR:?source env.sh first}"
RUNS="${RUNS:-C:0:h100|h200 A:0:h100|h200 C:1:a100 A:1:a100}"
PREFIX="${PREFIX:-S4}"
TIME="${TIME:-20:00:00}"
CPUS="${CPUS:-24}"
MEM="${MEM:-96G}"
NUM_ENVS="${NUM_ENVS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-1024}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-477}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-64}"
SHAPING_STEPS="${SHAPING_STEPS:-500000000}"
DECK_POOL="${DECK_POOL:-hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge}"
RANDOM_DECK_FRAC="${RANDOM_DECK_FRAC:-0.5}"
HELDOUT="${HELDOUT:-xbow;royal_hogs;random:50:20261003}"
EVAL_EVERY="${EVAL_EVERY:-6104}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_BOTS="${EVAL_BOTS:-heuristic random}"
SPECIALISTS="${SPECIALISTS:-S3_A_s1=$DATA_DIR/league/S3_A_s1/model_061036.pt S3_C_s0=$DATA_DIR/league/S3_C_s0/model_061036.pt}"
CKPT_MATCHES="${CKPT_MATCHES:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CARD_FLOOR="${CARD_FLOOR:-0.005}"
CARD_FLOOR_STEPS="${CARD_FLOOR_STEPS:-5000000000}"
ENV_THREADS="${ENV_THREADS:-16}"
LLM_RUNS="${LLM_RUNS-C:0}"
LLM_EVERY="${LLM_EVERY:-250000000}"
LLM_LINKS="${LLM_LINKS:-7}"
LLM_TIME="${LLM_TIME:-2-00:00:00}"
LLM_CPUS="${LLM_CPUS:-4}"
LLM_MEM="${LLM_MEM:-16G}"
LLM_MODEL="${LLM_MODEL:-claude-opus-5-5}"
LLM_EFFORT="${LLM_EFFORT:-low}"
SUBMIT="${SUBMIT:-0}"
[ -r "$PR_CHECKOUT/hpc/league.sbatch" ] && [ -r "$PR_CHECKOUT/hpc/stage3_array.sbatch" ] \
    || { echo "no hpc/league.sbatch or hpc/stage3_array.sbatch in $PR_CHECKOUT" >&2; exit 2; }
[ -z "$LLM_RUNS" ] || [ -r "$PR_CHECKOUT/hpc/llm.sbatch" ] || { echo "no hpc/llm.sbatch in $PR_CHECKOUT" >&2; exit 2; }
case "$LINKS" in ''|*[!0-9]*) echo "LINKS must be a positive integer" >&2; exit 2 ;; esac
[ "$LINKS" -ge 1 ] || { echo "LINKS must be >= 1" >&2; exit 2; }
for kv in $SPECIALISTS; do
    [ -r "${kv#*=}" ] || { echo "specialist checkpoint ${kv#*=} not found" >&2; exit 2; }
done
# the shaping fades over SHAPING_STEPS learner steps: league_train.py takes it as a fraction of TOTAL;
# SHAPING_STEPS=0 keeps the shaping weights constant for the whole run (D102)
SHAPING_FRAC=$(python3 -c "import sys; t, s = int(sys.argv[1]), int(sys.argv[2]); assert 0 <= s <= t; print(repr(s / t) if s else '0')" \
               "$TOTAL" "$SHAPING_STEPS")

# stage-3 settings (D89 / D90): conditional head, feed-forward, royale.ini hyperparameters, constant learning rate
# (no --anneal-lr), the frozen shaping, the throughput package; plus the stage-4 deck sampler (§8.4), D99 logging
# and the D100 card-probability floor
COMMON="--train.torch-deterministic= --env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01"
COMMON="$COMMON --train.reward-clip 0 --shaping-anneal-frac $SHAPING_FRAC"
COMMON="$COMMON --env-threads $ENV_THREADS --fast-opponents --cuda-graphs"
COMMON="$COMMON --env.deck-pool $DECK_POOL --env.random-deck-frac $RANDOM_DECK_FRAC"
COMMON="$COMMON --env.heldout-decks $HELDOUT --env.deck-draw independent"
COMMON="$COMMON --log-interval $LOG_INTERVAL"
COMMON="$COMMON --card-floor $CARD_FLOOR --card-floor-steps $CARD_FLOOR_STEPS"
if [ -n "${EXTRA:-}" ]; then COMMON="$COMMON $EXTRA"; fi
EVAL_ARGS="--bots $EVAL_BOTS --decks pool=$DECK_POOL heldout=$HELDOUT --ckpt $SPECIALISTS"
EVAL_ARGS="$EVAL_ARGS --ckpt-matches $CKPT_MATCHES --ckpt-deck hog26"

if [ "$SUBMIT" = 1 ]; then
    if squeue -h -u "$USER" -o '%j' -t PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED | grep -q "^pr-${PREFIX}_"; then
        echo "a job named pr-${PREFIX}_* is still queued or running: not submitting again" >&2
        exit 3
    fi
    OUT="$DATA_DIR/stage4/$PREFIX/sub_$(date +%Y%m%d_%H%M%S)"
    [ ! -e "$OUT" ] || { echo "$OUT exists" >&2; exit 3; }
else
    OUT="$DATA_DIR/stage4/$PREFIX/dryrun"
fi
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.txt"
: > "$MANIFEST.new"
q() { printf "'%s'" "${1//\'/\'\\\'\'}"; }          # single-quote a value for the env file
IDS=() CONSTRAINTS=()
for item in $RUNS; do
    IFS=: read -r run s gpu <<< "$item"
    [ -n "$gpu" ] || { echo "RUNS item '$item' is not method:seed:constraint" >&2; exit 2; }
    case "$run" in   # pool flags per method (TRAINING_PLAN §8.3); no scripted bots anywhere in training
        A) POOL=(ANCHORS=none SELF_PLAY_FRAC=1.0 ANCHOR_FRAC=0 PFSP=hard MMD_COEF=0 MMD_REF_INTERVAL=0) ;;
        C) POOL=(ANCHORS=none SELF_PLAY_FRAC=0.35 ANCHOR_FRAC=0 PFSP=hard MMD_COEF=0.05 MMD_REF_INTERVAL=50) ;;
        *) echo "unknown method '$run' (stage 4: A or C, D99)" >&2; exit 2 ;;
    esac
    id="${PREFIX}_${run}_s$s"
    extra="$COMMON --wandb --wandb-project pufferroyale --wandb-group ${PREFIX}_$run --tag $id"
    f="$OUT/$id.env"
    {
        echo "# $id: written by hpc/stage4_submit.sh $(date -Iseconds); GPU constraint $gpu, $LINKS links of $TIME"
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
        echo "EVAL_WANDB_GROUP=$(q "${PREFIX}_$run")"
        echo "EVAL_ARGS=$(q "$EVAL_ARGS")"
        echo "TOURNAMENT_MATCHES='0'"
        echo "HEARTBEAT='1'"
        echo "OMP_NUM_THREADS='1'"
        echo "MKL_NUM_THREADS='1'"
        echo "OPENBLAS_NUM_THREADS='1'"
        echo "PR_OPT_FLAGS=$(q "${PR_OPT_FLAGS--O3 -march=x86-64-v3}")"
        echo "EXTRA_ARGS=$(q "$extra")"
    } > "$f"
    echo "$f" >> "$MANIFEST.new"
    IDS+=("$id")
    CONSTRAINTS+=("$gpu")
done
mv "$MANIFEST.new" "$MANIFEST"
echo "manifest: $MANIFEST (${#IDS[@]} runs, $LINKS links of $TIME each)"
cat -n "$MANIFEST"
echo "--- first env file:"; cat "$(head -n 1 "$MANIFEST")"
CHAINS="$OUT/chains.txt"
for i in "${!IDS[@]}"; do
    CMD=(sbatch --parsable --array="$i" --job-name="pr-${IDS[$i]}" --cpus-per-task="$CPUS" --constraint="${CONSTRAINTS[$i]}"
         --mem="$MEM" --time="$TIME" "$PR_CHECKOUT/hpc/stage3_array.sbatch" "$MANIFEST")
    echo "${IDS[$i]}: ${CMD[*]}  (links 2..$LINKS add --dependency=afterany:<previous link>)"
    if [ "$SUBMIT" = 1 ]; then
        prev=""
        jobs=()
        for k in $(seq "$LINKS"); do
            if [ -z "$prev" ]; then
                jid=$("${CMD[@]}")
            else
                jid=$("${CMD[@]:0:1}" --dependency="afterany:$prev" "${CMD[@]:1}")
            fi
            jid="${jid%%;*}"
            jobs+=("$jid")
            prev="$jid"
        done
        echo "${IDS[$i]} ${CONSTRAINTS[$i]} ${jobs[*]}" | tee -a "$CHAINS"
    fi
done

# ---- Claude games (D101): one CPU chain per LLM_RUNS run
for item in $LLM_RUNS; do
    IFS=: read -r run s <<< "$item"
    id="${PREFIX}_${run}_s$s"
    case " ${IDS[*]} " in *" $id "*) ;; *) echo "LLM_RUNS item '$item': $id is not one of this submission's runs" >&2; exit 2 ;; esac
    f="$OUT/$id.llm.env"
    {
        echo "# $id Claude games: written by hpc/stage4_submit.sh $(date -Iseconds); $LLM_LINKS links of $LLM_TIME"
        echo "PR_CHECKOUT=$(q "$PR_CHECKOUT")"
        echo "RUN_ID=$(q "$id")"
        echo "LLM_EVERY=$(q "$LLM_EVERY")"
        echo "LLM_MODEL=$(q "$LLM_MODEL")"
        echo "LLM_EFFORT=$(q "$LLM_EFFORT")"
        echo "LLM_GROUP=$(q "${PREFIX}_$run")"
        echo "PR_OPT_FLAGS=$(q "${PR_OPT_FLAGS--O3 -march=x86-64-v3}")"
    } > "$f"
    CMD=(sbatch --parsable --job-name="pr-$id-llm" --cpus-per-task="$LLM_CPUS" --mem="$LLM_MEM" --time="$LLM_TIME"
         "$PR_CHECKOUT/hpc/llm.sbatch" "$f")
    echo "$id-llm: ${CMD[*]}  (links 2..$LLM_LINKS add --dependency=afterany:<previous link>)"
    if [ "$SUBMIT" = 1 ]; then
        prev=""
        jobs=()
        for k in $(seq "$LLM_LINKS"); do
            if [ -z "$prev" ]; then
                jid=$("${CMD[@]}")
            else
                jid=$("${CMD[@]:0:1}" --dependency="afterany:$prev" "${CMD[@]:1}")
            fi
            jid="${jid%%;*}"
            jobs+=("$jid")
            prev="$jid"
        done
        echo "$id-llm cpu ${jobs[*]}" | tee -a "$CHAINS"
    fi
done
