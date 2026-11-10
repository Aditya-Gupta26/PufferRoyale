#!/bin/bash
# Claude-vs-policy games for one league run (DECISIONS D101), on a CPU node: hpc/llm_game.py (in the image) plays
# one recorded game per LLM_EVERY learner steps against the run's saved policy and logs it to W&B
# ("<RUN_ID>-llm"); hpc/llm_relay.py (on the HOST, where the Claude Code CLI and its claude.ai login live) answers
# each of Claude's decisions with one headless `claude -p` call -- no SDK, no API key. The turn/reply files go
# through a node-local directory (bound into the image), so NFS attribute caching never delays a reply.
#
#     sbatch [--dependency=afterany:<previous link>] --job-name=pr-<RUN_ID>-llm hpc/llm.sbatch ENV_FILE
#
# ENV_FILE (hpc/stage4_submit.sh writes it): PR_CHECKOUT, RUN_ID, LLM_EVERY, LLM_MODEL, LLM_EFFORT, LLM_GROUP
# (W&B group; empty = no W&B), PR_OPT_FLAGS (must equal the training's: the C extension of the shared checkout
# is never rebuilt under a running trainer), optional LLM_ARGS (more hpc/llm_game.py options). Chained like the training: a
# link resumes from <run>/llm/state.json; a link that finds <run>/llm/DONE exits at once.
#SBATCH --job-name=pr-llm
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
ENV_FILE="${1:?usage: sbatch hpc/llm.sbatch ENV_FILE}"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
: "${PR_CHECKOUT:?env file must set PR_CHECKOUT}" "${RUN_ID:?env file must set RUN_ID}"
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
LLM_EVERY="${LLM_EVERY:-250000000}"
LLM_MODEL="${LLM_MODEL:-claude-opus-5-5}"
LLM_EFFORT="${LLM_EFFORT:-low}"
LLM_GROUP="${LLM_GROUP:-}"
read -r -a LX <<< "${LLM_ARGS:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN_DIR="$DATA_DIR/league/$RUN_ID"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): run=$RUN_DIR every=$LLM_EVERY model=$LLM_MODEL effort=$LLM_EFFORT" \
     "repo=$PROJECT @ $(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null)"
if [ -f "$RUN_DIR/llm/DONE" ]; then echo "$RUN_DIR/llm/DONE exists: every game is played"; exit 0; fi
CLAUDE_BIN="$(command -v claude || echo "$HOME/.local/bin/claude")"
[ -x "$CLAUDE_BIN" ] || { echo "no Claude Code CLI ($CLAUDE_BIN)" >&2; exit 3; }
pr_isolation_check

RELAY="${SLURM_TMPDIR:-/tmp}/pr-llm-${SLURM_JOB_ID:-local}"
mkdir -p "$RELAY" "$RUN_DIR/llm"
RELAY_PID=""
cleanup() {
    touch "$RELAY/STOP" 2>/dev/null || true
    if [ -n "$RELAY_PID" ]; then
        pkill -TERM -P "$RELAY_PID" 2>/dev/null || true        # the relay itself, then its restart loop
        kill "$RELAY_PID" 2>/dev/null || true
        wait "$RELAY_PID" 2>/dev/null || true
        RELAY_PID=""
    fi
    mkdir -p "$RUN_DIR/llm" && cp "$RELAY/calls.jsonl" "$RUN_DIR/llm/calls_${SLURM_JOB_ID:-local}.jsonl" 2>/dev/null
    rm -rf "$RELAY"
}
trap cleanup EXIT
# the relay restarts if it ever dies; it stops when $RELAY/STOP appears
( while [ ! -f "$RELAY/STOP" ]; do
      /usr/bin/python3 "$PROJECT/hpc/llm_relay.py" --relay-dir "$RELAY" --model "$LLM_MODEL" --effort "$LLM_EFFORT" \
          --claude "$CLAUDE_BIN"
      [ -f "$RELAY/STOP" ] || sleep 10
  done ) >> "$RUN_DIR/llm/relay.log" 2>&1 &
RELAY_PID=$!
echo "relay started (pid $RELAY_PID, $CLAUDE_BIN), log $RUN_DIR/llm/relay.log"

WB=()
if [ -n "$LLM_GROUP" ]; then WB=(--wandb --wandb-group "$LLM_GROUP"); fi
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" --bind "$RELAY:$RELAY" --pwd "$PROJECT" "$SIF" \
    python hpc/llm_game.py "$RUN_DIR" --relay-dir "$RELAY" --every-steps "$LLM_EVERY" --model "$LLM_MODEL" \
    --effort "$LLM_EFFORT" ${WB[@]+"${WB[@]}"} ${LX[@]+"${LX[@]}"} &
GAME_PID=$!
trap 'kill -TERM "$GAME_PID" 2>/dev/null; wait "$GAME_PID"; cleanup' TERM
wait "$GAME_PID"
echo "llm_game exit: $?"
