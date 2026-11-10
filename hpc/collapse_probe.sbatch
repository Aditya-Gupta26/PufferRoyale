#!/bin/bash
# Stage-4 collapse diagnostics: gradient probes (hpc/collapse_probe.py), one per array element, on CPU nodes.
# SPECS is a file of lines "RUN AT MODE [extra args...]"; element k runs line k+1 and writes $OUT_DIR/<RUN>_<AT>_<MODE>.json
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     SPECS=... OUT_DIR=... PR_CHECKOUT=<pinned> sbatch --array=0-8 --export=ALL $PR_CHECKOUT/hpc/collapse_probe.sbatch
#SBATCH --job-name=pr-probe
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -uo pipefail
: "${SPECS:?set SPECS}" "${OUT_DIR:?set OUT_DIR}"
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
k="${SLURM_ARRAY_TASK_ID:-0}"
read -r RUN AT MODE EXTRA <<< "$(sed -n "$((k + 1))p" "$SPECS")"
echo "job ${SLURM_JOB_ID:-local} element $k on $(hostname): $RUN $AT $MODE $EXTRA; repo $PROJECT @ $(git -C "$PROJECT" rev-parse --short HEAD)"
pr_isolation_check
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$OUT_DIR"
# shellcheck disable=SC2086
run python hpc/collapse_probe.py --run "$RUN" --at "$AT" --mode "$MODE" --epochs 8 --env-threads 8 \
    --out "$OUT_DIR/${RUN}_${AT}_${MODE}.json" $EXTRA
