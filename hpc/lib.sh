# shellcheck shell=bash
# Helpers shared by the PufferRoyale job scripts (docs/HPC_HANDOFF.md §4d, §6a item 6). Source it after
# our env script:   source /scratch/$USER/pufferroyale/env.sh && source "$REPO/hpc/lib.sh"
#   run CMD...              CMD inside our image: only $PR_ROOT bound, no $HOME, cwd = $PROJECT
#   "${PR_RUN[@]}" CMD...   the same, as an array (for timeout / nice)
#   pr_ensure_ext           build the C extension in $PROJECT if missing or stale (locked)
#   pr_isolation_check      pr_ensure_ext, then hpc/isolation_check.py in the image (exits the job if it fails)
#   pr_heartbeat_start      hpc/gpu_heartbeat.py in the background (HEARTBEAT=0 disables it)
#   pr_heartbeat_stop       stop it (also done by the EXIT trap pr_heartbeat_start installs)
: "${PR_ROOT:?source /scratch/\$USER/pufferroyale/env.sh first}"
PR_NV=(--nv)
command -v nvidia-smi >/dev/null 2>&1 || PR_NV=()          # CPU nodes: no GPU to expose
PR_HEARTBEAT_PID=""

# the same command as an array, for wrappers that need an executable (e.g. `timeout 30m "${PR_RUN[@]}" make`)
PR_RUN=(apptainer exec "${PR_NV[@]}" --no-home --bind "$PR_ROOT:$PR_ROOT" --pwd "${PROJECT:-$REPO}" "$SIF")

run() {
    "${PR_RUN[@]}" "$@"
}

_pr_ext_current() {
    # the .so exists, is newer than its sources and was built with the requested PR_OPT_FLAGS
    # (.build_flags records them; no file = the default flags)
    local so newest
    # `|| true`: under the templates' `set -euo pipefail` a failing ls (no .so yet) must not end the job
    so=$(ls -t "$PROJECT"/pufferroyale/binding.cpython-*-linux-gnu.so 2>/dev/null | head -n 1 || true)
    newest=$(ls -t "$PROJECT"/pufferroyale/csrc/*.h "$PROJECT"/pufferroyale/binding.c "$PROJECT"/pufferroyale/env_binding.h \
             "$PROJECT"/setup.py 2>/dev/null | head -n 1 || true)
    [ -n "$so" ] && [ "$so" -nt "$newest" ] && [ "$(cat "$PROJECT/.build_flags" 2>/dev/null || true)" = "${PR_OPT_FLAGS:-}" ]
}

pr_ensure_ext() {
    # build the C extension in the job's checkout only if it is missing, older than its sources or built
    # with other PR_OPT_FLAGS (DECISIONS D90), under a lock; the check is repeated inside the lock, so jobs
    # that start together on the same checkout build once and never overwrite a .so another job has loaded
    if _pr_ext_current; then return 0; fi
    exec 9>"$PROJECT/.build_ext.lock"
    flock 9
    if ! _pr_ext_current; then
        echo "building the C extension in $PROJECT (PR_OPT_FLAGS='${PR_OPT_FLAGS:-}')"
        "${PR_RUN[@]}" python setup.py build_ext --inplace --force > "$PROJECT/.build_ext.log" 2>&1 \
            || { tail -20 "$PROJECT/.build_ext.log" >&2; echo "build_ext FAILED" >&2; flock -u 9; exit 4; }
        printf '%s' "${PR_OPT_FLAGS:-}" > "$PROJECT/.build_flags"
    fi
    flock -u 9
    exec 9>&-
}

pr_isolation_check() {
    pr_ensure_ext
    if ! run python "$REPO/hpc/isolation_check.py"; then
        echo "isolation check FAILED: stopping this job" >&2
        exit 3
    fi
}

pr_heartbeat_stop() {
    if [ -n "$PR_HEARTBEAT_PID" ]; then
        kill "$PR_HEARTBEAT_PID" 2>/dev/null || true
        wait "$PR_HEARTBEAT_PID" 2>/dev/null || true
        PR_HEARTBEAT_PID=""
    fi
}

pr_heartbeat_start() {
    [ "${HEARTBEAT:-1}" = "1" ] || { echo "heartbeat disabled (HEARTBEAT=${HEARTBEAT:-})"; return 0; }
    [ "${#PR_NV[@]}" -gt 0 ] || return 0
    # nice 0 (HEARTBEAT_NICE): at nice 19 it never even started while the job's cores were saturated (D87)
    nice -n "${HEARTBEAT_NICE:-0}" apptainer exec --nv --no-home --bind "$PR_ROOT:$PR_ROOT" --pwd "$REPO" "$SIF" \
        python "$REPO/hpc/gpu_heartbeat.py" --target "${HEARTBEAT_TARGET:-75}" &
    PR_HEARTBEAT_PID=$!
    trap pr_heartbeat_stop EXIT
    echo "gpu heartbeat started (pid $PR_HEARTBEAT_PID, floor ${HEARTBEAT_TARGET:-75}% utilization)"
}
