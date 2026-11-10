#!/bin/bash
# PufferRoyale: one best-response (exploitability) probe (scripts/best_response.py; HANDOFF §10 E1).
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     TARGET=$DATA_DIR/league/A_s0 BR_SEED=0 BR_STEPS=150000000 NUM_ENVS=... \
#         EXTRA_ARGS="--init-from-target --env.reward-tower 0.3 ..." sbatch --job-name=pr-br_A_s0_s0 $REPO/hpc/br.sbatch
# Each probe gets its own data dir ($DATA_DIR/br/<name>_s<seed>) and writes $DATA_DIR/eval/br_<name>_s<seed>.json
# (concurrent probes must never share a data dir, HANDOFF §13). HEARTBEAT=0 disables the GPU heartbeat.
#
#SBATCH --job-name=pr-br
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=a100|h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
TARGET="${TARGET:?set TARGET (a run directory, model_*.pt or snap_*.pt)}"
BR_SEED="${BR_SEED:-0}"
BR_STEPS="${BR_STEPS:?set BR_STEPS (learner steps of the best response)}"
MATCHES="${MATCHES:-1000}"
NUM_ENVS="${NUM_ENVS:-256}"
NAME="${BR_NAME:-$(basename "${TARGET%/}" .pt)}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
mkdir -p "$DATA_DIR/br/${NAME}_s$BR_SEED" "$DATA_DIR/eval"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): BR vs $TARGET seed $BR_SEED steps $BR_STEPS"
nvidia-smi || true
pr_heartbeat_start
pr_isolation_check
run python scripts/best_response.py \
    --target "$TARGET" \
    --device cuda \
    --num-envs "$NUM_ENVS" \
    --total-timesteps "$BR_STEPS" \
    --matches "$MATCHES" \
    --seed "$BR_SEED" \
    --data-dir "$DATA_DIR/br/${NAME}_s$BR_SEED" \
    --out "$DATA_DIR/eval/br_${NAME}_s$BR_SEED.json" \
    ${EXTRA[@]+"${EXTRA[@]}"}
echo "done: $DATA_DIR/eval/br_${NAME}_s$BR_SEED.json"
