#!/bin/bash
# PufferRoyale: Phase 3 throughput calibration on ONE GPU type (docs/HPC_HANDOFF.md §6), with
# hpc/calibrate.py. Pick the type with --constraint; run one job per type:
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     PR_CHECKOUT=$PR_ROOT/pinned/<commit> sbatch --constraint=h100 --job-name=pr-cal-h100 $REPO/hpc/calibrate.sbatch
#     STAGES="s1grid s1stack" SECONDS_PER=90 sbatch --constraint=a100 $REPO/hpc/calibrate.sbatch
# Results: $DATA_DIR/calibration/<GPU>_<job>/<stage>/results.jsonl + a markdown table per stage in the
# job log. The GPU heartbeat (>= 75% utilization, DECISIONS D79) runs throughout, as in real jobs.
# Every league configuration uses the stage-1 main shaping flags (TRAINING_PLAN §8.1).
#
#SBATCH --job-name=pr-calibrate
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=a100|h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | tr ' ' '_')"
OUT="${OUT:-$DATA_DIR/calibration/${GPU:-gpu}_${SLURM_JOB_ID:-local}}"
STAGES="${STAGES:-s1grid s1mb s1stack A ckpt rnn train}"
SECONDS_PER="${SECONDS_PER:-120}"
SHAPING=(--env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0)
mkdir -p "$OUT"
cd "$REPO" || exit 2
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is); repo $REPO @ $(git rev-parse --short HEAD); out $OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$OUT/gpu.csv"
{ lscpu | sed -n 's/^Model name: *//p'; echo "nproc $(nproc)"; } | tee "$OUT/cpu.txt"
pr_heartbeat_start
pr_isolation_check > "$OUT/isolation.txt" 2>&1 || exit 3      # (exits the job itself on failure)
cat "$OUT/isolation.txt"

cal() {  # cal <stage> calibrate.py args...
    local stage=$1; shift
    [[ " $STAGES " == *" $stage "* ]] || return 0
    echo "== stage $stage ($(date -Is))"
    run python hpc/calibrate.py --out "$OUT/$stage" --name "$stage" --device cuda --seconds "$SECONDS_PER" "$@" \
        2> "$OUT/$stage.stderr" | grep -v '^\[calibrate\]'
}

cal s1grid --script league --shape stage1 --num-envs 128 256 512 --batch-k 1 2 --minibatch-size 8192 -- "${SHAPING[@]}"
cal s1mb --script league --shape stage1 --num-envs 512 --batch-k 2 --minibatch-size 4096 16384 -- "${SHAPING[@]}"
cal s1stack --script league --shape stage1 --num-envs 256 512 --batch-k 2 --minibatch-size 8192 --copies 2 4 \
    -- "${SHAPING[@]}"
cal A --script league --shape A --num-envs 512 --batch-k 2 --minibatch-size 8192 -- "${SHAPING[@]}"
cal ckpt --script league --shape ckpt16 ckpt32 ckpt64 --num-envs 512 --batch-k 2 --minibatch-size 8192 \
    -- "${SHAPING[@]}"
cal rnn --script league --shape stage1 --num-envs 512 --batch-k 2 --minibatch-size 8192 --rnn on -- "${SHAPING[@]}"
cal train --script train --num-envs 64 256 --batch-size 65536 --minibatch-size 16384 --workers 0 4 8 \
    -- "${SHAPING[@]}"
echo "done: $OUT ($(date -Is))"
