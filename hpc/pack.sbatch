#!/bin/bash
# PufferRoyale: several runs stacked on ONE GPU (user: one GPU per job; stacking allowed whenever it
# raises total throughput -- DECISIONS D79, Phase 3 measures it). PACK_FILE lists one command per line,
# run concurrently inside our image from our checkout (blank lines and # comments are skipped):
#     scripts/stages.py --run-prefix S1main_s0 --seed 0 --data-dir /scratch/.../runs --device cuda ...
#     scripts/stages.py --run-prefix S1main_s1 --seed 1 --data-dir /scratch/.../runs --device cuda ...
# Each line's output goes to $PR_ROOT/logs/<job name>_<job id>_<line index>.log. The job waits for all
# of them and exits 1 if any failed. Every script used this way resumes from its run directories, so a
# re-submit with the same PACK_FILE continues everything. Give the job enough CPUs for all lines.
# MAX_PARALLEL=N runs at most N lines at a time (default: all at once); PACK_THREADS=T sets OMP/MKL/
# OpenBLAS threads per process (default: the job's CPUs / the lines running at once, at least 1 -- unset,
# torch and numpy each start one thread per CPU in every process, and a 6-line pack on 16 CPUs ran 96
# threads: its startup took 16 min instead of ~2 and starved the heartbeat, job 19153517). On a CPU node (e.g. eval.py --device cpu lines), submit a copy
# without the #SBATCH --gres/--constraint lines: no GPU means no --nv and no heartbeat.
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     PACK_FILE=$PR_ROOT/runs/packs/s1_main.txt sbatch --job-name=pr-S1main --cpus-per-task=16 $REPO/hpc/pack.sbatch
#
#SBATCH --job-name=pr-pack
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=a100|h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
PACK_FILE="${PACK_FILE:?set PACK_FILE (one command per line)}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"                        # 0 = every line at once
[ -r "$PACK_FILE" ] || { echo "cannot read $PACK_FILE" >&2; exit 2; }
if [ -z "${PACK_THREADS:-}" ]; then
    NLINES=$(grep -cvE '^[[:space:]]*(#|$)' "$PACK_FILE" || true)
    CONC=$NLINES; [ "$MAX_PARALLEL" -gt 0 ] && [ "$MAX_PARALLEL" -lt "$NLINES" ] && CONC=$MAX_PARALLEL
    PACK_THREADS=$(( ${SLURM_CPUS_PER_TASK:-$(nproc)} / (CONC > 0 ? CONC : 1) ))
    [ "$PACK_THREADS" -ge 1 ] || PACK_THREADS=1
fi
export OMP_NUM_THREADS=$PACK_THREADS MKL_NUM_THREADS=$PACK_THREADS OPENBLAS_NUM_THREADS=$PACK_THREADS
NAME="${SLURM_JOB_NAME:-pr-pack}_${SLURM_JOB_ID:-local}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): pack $PACK_FILE repo=$PROJECT @ $(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null), $PACK_THREADS threads per line"
nvidia-smi || true
pr_heartbeat_start
pr_isolation_check

PIDS=()
i=0
while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    read -r -a ARGS <<< "$line"
    if [ "$MAX_PARALLEL" -gt 0 ]; then
        # poll, do not `wait -n`: reaping a line here would hide its exit status from the final waits
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 5; done
    fi
    echo "[$i] python ${ARGS[*]}  -> $PR_ROOT/logs/${NAME}_$i.log"
    run python "${ARGS[@]}" > "$PR_ROOT/logs/${NAME}_$i.log" 2>&1 &
    PIDS+=($!)
    i=$((i + 1))
done < "$PACK_FILE"
FAILED=0
for k in "${!PIDS[@]}"; do
    if wait "${PIDS[$k]}"; then echo "[$k] done"; else rc=$?; echo "[$k] exit $rc"; FAILED=1; fi
done
exit $FAILED
