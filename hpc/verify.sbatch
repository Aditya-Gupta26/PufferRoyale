#!/bin/bash
# PufferRoyale: gate G0 on one GPU (docs/HPC_HANDOFF.md §5): isolation check, C tests, ASan, pytest,
# golden hashes, the e2e ladder, and the two GPU smoke runs (with their peak GPU memory), each logged to
# $OUT/<step>.log, then a PASS/FAIL table. A failing step does not stop the ones after it; the job exits
# non-zero if any gating step failed.
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs && sbatch $REPO/hpc/verify.sbatch
#     STEPS="build-ext isolation smoke-train smoke-league" sbatch $REPO/hpc/verify.sbatch   # a subset
# The checkout is read when the job STARTS: do not edit $REPO while a verify job is pending or running
# (or point REPO/PROJECT at a pinned copy).
#
#SBATCH --job-name=pr-verify
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --constraint=a100|h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
source /scratch/$USER/pufferroyale/env.sh
source "$REPO/hpc/lib.sh"
OUT="${OUT:-$DATA_DIR/g0/${SLURM_JOB_ID:-local}}"
STEPS="${STEPS:-}"
mkdir -p "$OUT"
cd "$REPO" || exit 2
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is); repo $REPO @ $(git rev-parse --short HEAD); out $OUT"
nvidia-smi 2>&1 | tee "$OUT/nvidia-smi.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.csv" 2>&1
lscpu > "$OUT/lscpu.txt" 2>&1
echo "nproc $(nproc)" | tee -a "$OUT/lscpu.txt"

declare -a NAMES=() RESULTS=()
record() { NAMES+=("$1"); RESULTS+=("$2"); echo "== $1: $2 ($(date -Is))"; }
step() {  # step <name> <gating 1/0> <timeout> cmd... (cmd runs inside the image)
    local name=$1 gating=$2 limit=$3; shift 3
    if [ -n "$STEPS" ] && [[ " $STEPS " != *" $name "* ]]; then return 0; fi
    local t0=$SECONDS
    timeout --kill-after=60 "$limit" "${PR_RUN[@]}" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    local res; [ $rc -eq 0 ] && res=PASS || res="FAIL(rc=$rc)"
    [ "$gating" = "0" ] && res="$res[non-gating]"
    record "$name" "$res ($((SECONDS - t0))s)"
    tail -4 "$OUT/$name.log"
}

pr_heartbeat_start                         # keep the GPU >= 75% busy (user instruction; HEARTBEAT=0 disables)
rm -f pufferroyale/binding.cpython-*-darwin.so
step build-ext 1 20m python setup.py build_ext --inplace
step isolation 1 10m python hpc/isolation_check.py
cp "$OUT/isolation.log" "$OUT/versions.txt" 2>/dev/null
step test-c 1 30m make test-c
step asan 1 60m make asan
step pytest 1 120m python -m pytest tests -q -p no:cacheprovider
step golden-hashes 1 20m python scripts/golden_hashes.py
step fast-tests 1 30m python -m pytest hpc/tests -q -p no:cacheprovider     # D90 throughput paths
step e2e 1 150m python scripts/e2e_check.py
SMI=
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$OUT/gpu_util_smoke.csv" 2>&1 &
SMI=$!
PR_PEAK_MEM_JSON="$OUT/peak_smoke_train.json" step smoke-train 1 30m python hpc/peak_mem.py scripts/train.py \
    --device cuda --total-timesteps 300000 --num-envs 64 --batch-size 16384 --minibatch-size 4096 --quiet \
    --data-dir "$OUT/train" --summary-json "$OUT/smoke_train.json"
PR_PEAK_MEM_JSON="$OUT/peak_smoke_league.json" step smoke-league 1 30m python hpc/peak_mem.py scripts/league_train.py \
    --device cuda --total-timesteps 200000 --num-envs 64 --anchors bot:heuristic --snapshot-interval 2 \
    --data-dir "$OUT" --run-id smoke --summary-json "$OUT/smoke_league.json"
[ -n "$SMI" ] && kill "$SMI" 2>/dev/null
step asan-py 0 90m make asan-py

echo
echo "G0 summary (job ${SLURM_JOB_ID:-local}, $(cat "$OUT/gpu.csv" 2>/dev/null)):" | tee "$OUT/summary.txt"
FAILED=0
for i in "${!NAMES[@]}"; do
    printf '  %-16s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}" | tee -a "$OUT/summary.txt"
    case "${RESULTS[$i]}" in FAIL*non-gating*) ;; FAIL*) FAILED=1 ;; esac
done
for f in test-c asan; do
    [ -f "$OUT/$f.log" ] || continue
    awk -v f="$f" '/ checks, [0-9]+ failures$/ {n++; c += $1; x += $3} END {printf "%s: %d binaries, %d checks, %d failures\n", f, n, c, x}' "$OUT/$f.log"
done | tee -a "$OUT/summary.txt"
grep -shE 'isolation OK' "$OUT/isolation.log" | tee -a "$OUT/summary.txt"
grep -sE '[0-9]+ (passed|failed)' "$OUT/pytest.log" | tail -1 | tee -a "$OUT/summary.txt"
grep -sE 'golden hashes' "$OUT/golden-hashes.log" | tail -1 | tee -a "$OUT/summary.txt"
grep -sE 'OVERALL' "$OUT/e2e.log" | tail -1 | tee -a "$OUT/summary.txt"
cat "$OUT"/peak_smoke_*.json 2>/dev/null | tee -a "$OUT/summary.txt"; echo
echo "logs: $OUT"
exit $FAILED
