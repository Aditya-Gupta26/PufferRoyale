#!/bin/bash
# PufferRoyale: evaluation on one GPU (HANDOFF §10 E2-E4): MODE=bots (scripts/eval.py),
# MODE=tournament (scripts/tournament.py) or MODE=transitivity (scripts/transitivity.py). Sampled actions.
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     MODE=bots CHECKPOINT=$DATA_DIR/league/A_s0 EPISODES=500 OUT=$DATA_DIR/eval/bots_A_s0.json \
#         sbatch --job-name=pr-eval_A_s0 $REPO/hpc/eval.sbatch
#     MODE=tournament AGENTS="<ckpt> <ckpt> ..." MATCHES=200 OUT=$DATA_DIR/eval/stage3_roundrobin.json \
#         EXTRA_ARGS="--deck-mode mirror" sbatch --job-name=pr-rr $REPO/hpc/eval.sbatch
#     MODE=transitivity RUN=$DATA_DIR/league/B_s0 SNAPSHOTS=10 MATCHES=100 OUT=$DATA_DIR/eval/transitivity_B_s0.json \
#         sbatch --job-name=pr-tr_B_s0 $REPO/hpc/eval.sbatch
# DEVICE=cpu runs the policy on the CPU (submit a copy without the #SBATCH --gres/--constraint lines to a CPU
# node: HANDOFF-style gate confirmations need no GPU). HEARTBEAT=0 disables the GPU heartbeat (D79).
#
#SBATCH --job-name=pr-eval
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=a100|h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
MODE="${MODE:?set MODE (bots | tournament | transitivity)}"
OUT="${OUT:?set OUT (the JSON output path)}"
SEED="${SEED:-0}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
mkdir -p "$(dirname "$OUT")"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): $MODE -> $OUT"
nvidia-smi || true
pr_heartbeat_start
pr_isolation_check
case "$MODE" in
    bots)
        run python scripts/eval.py \
            --checkpoint "${CHECKPOINT:?set CHECKPOINT}" \
            --bots ${BOTS:-noop random heuristic} \
            --seats 0 1 \
            --episodes "${EPISODES:-500}" \
            --device "${DEVICE:-cuda}" \
            --seed "$SEED" \
            --json "$OUT" \
            ${EXTRA[@]+"${EXTRA[@]}"}
        ;;
    tournament)
        read -r -a AG <<< "${AGENTS:?set AGENTS (space-separated specs or checkpoint paths)}"
        run python scripts/tournament.py \
            --agents "${AG[@]}" \
            --decks ${DECKS:-hog26} \
            --matches "${MATCHES:-200}" \
            --sample \
            --device "${DEVICE:-cuda}" \
            --seed "$SEED" \
            --out "$OUT" \
            ${EXTRA[@]+"${EXTRA[@]}"}
        ;;
    transitivity)
        run python scripts/transitivity.py \
            --run "${RUN:?set RUN (a league run directory)}" \
            --snapshots "${SNAPSHOTS:-10}" \
            --matches "${MATCHES:-100}" \
            --device "${DEVICE:-cuda}" \
            --out "$OUT" \
            ${EXTRA[@]+"${EXTRA[@]}"}
        ;;
    *) echo "unknown MODE $MODE" >&2; exit 2 ;;
esac
echo "done: $OUT"
