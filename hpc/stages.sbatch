#!/bin/bash
# PufferRoyale stage 1: one bot ladder (scripts/stages.py; docs/HPC_HANDOFF.md §8) on one GPU.
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     PREFIX=S1main_s0 SEED=0 NUM_ENVS=... BATCH_SIZE=... MINIBATCH_SIZE=... \
#         EXTRA_ARGS="--env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0" \
#         sbatch --job-name=pr-S1main_s0 $REPO/hpc/stages.sbatch
# Rungs: RUNGS (default: stages.py's noop -> random -> heuristic ladder). A rung whose run directory
# exists is resumed, so re-submitting with EXACTLY the same variables continues a preempted or timed-out
# ladder. Outputs: $DATA_DIR/league/<PREFIX>_<i>_<bot>/ and $DATA_DIR/stages_<PREFIX>.json; exit 0 = all
# gates passed, 1 = a gate failed, 2 = a rung failed. To stack several ladders on one GPU use
# hpc/pack.sbatch. HEARTBEAT=0 disables the GPU heartbeat (>= 75% utilization, DECISIONS D79).
#
#SBATCH --job-name=pr-stages
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

PREFIX="${PREFIX:?set PREFIX (e.g. S1main_s0)}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-256}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-50}"             # must be > 0: the league saves only then
DECK0="${DECK0:-hog26}"
DECK1="${DECK1:-hog26}"
OPT=()
[ -n "${BATCH_SIZE:-}" ] && OPT+=(--batch-size "$BATCH_SIZE")
[ -n "${MINIBATCH_SIZE:-}" ] && OPT+=(--minibatch-size "$MINIBATCH_SIZE")
[ -n "${RUNGS:-}" ] && OPT+=(--rungs "$RUNGS")
[ -n "${GATE_WINDOW:-}" ] && OPT+=(--gate-window "$GATE_WINDOW")
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname): ladder $PREFIX seed $SEED envs $NUM_ENVS repo=$PROJECT" \
     "@ $(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null)"
nvidia-smi || true
pr_heartbeat_start
pr_isolation_check
run python scripts/stages.py \
    --run-prefix "$PREFIX" \
    --data-dir "$DATA_DIR" \
    --device cuda \
    --num-envs "$NUM_ENVS" \
    --snapshot-interval "$SNAPSHOT_INTERVAL" \
    --seed "$SEED" \
    --deck0 "$DECK0" \
    --deck1 "$DECK1" \
    ${OPT[@]+"${OPT[@]}"} \
    ${EXTRA[@]+"${EXTRA[@]}"}
echo "done: $DATA_DIR/stages_$PREFIX.json"
