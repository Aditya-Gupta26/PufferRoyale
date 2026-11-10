# PufferRoyale on NYU Torch: the environment of every PufferRoyale shell and job (docs/HPC_HANDOFF.md
# §4b). Installed as $PR_ROOT/env.sh (cp hpc/env.sh /scratch/$USER/pufferroyale/env.sh) and sourced by
# literal path: `source /scratch/$USER/pufferroyale/env.sh`. Never source it from ~/.bashrc.
# No secrets here: the wandb key is read from $PR_ROOT/secrets/wandb_key (chmod 600) if it exists.
export PR_ROOT=/scratch/$USER/pufferroyale
# PR_CHECKOUT (set at submission) runs a job from a pinned copy, e.g. a worktree under $PR_ROOT/pinned/,
# so edits to the main checkout never reach a queued job; it must lie inside $PR_ROOT (bound).
export REPO=${PR_CHECKOUT:-$PR_ROOT/PufferRoyale}
export PROJECT=$REPO
export SCRATCH_DIR=$PR_ROOT                          # the sbatch templates bind ONLY this directory
export SIF=$PR_ROOT/containers/pufferroyale.sif
export DATA_DIR=$PR_ROOT/runs
export PIP_CACHE_DIR=$PR_ROOT/cache/pip UV_CACHE_DIR=$PR_ROOT/cache/uv XDG_CACHE_HOME=$PR_ROOT/cache/xdg
export APPTAINER_CACHEDIR=$PR_ROOT/cache/apptainer SINGULARITY_CACHEDIR=$PR_ROOT/cache/apptainer
export APPTAINER_TMPDIR=$PR_ROOT/tmp TMPDIR=$PR_ROOT/tmp
export MPLCONFIGDIR=$PR_ROOT/cache/matplotlib
export WANDB_DIR=$PR_ROOT/wandb WANDB_CACHE_DIR=$PR_ROOT/cache/wandb WANDB_CONFIG_DIR=$PR_ROOT/wandb/config
export WANDB_DATA_DIR=$PR_ROOT/cache/wandb-data WANDB_ARTIFACT_DIR=$PR_ROOT/cache/wandb-artifacts
export WANDB_PROJECT=${WANDB_PROJECT:-pufferroyale}
export WANDB_ENTITY=${WANDB_ENTITY:-}       # your W&B workspace (empty = your default; D83)
if [ -r "$PR_ROOT/secrets/wandb_key" ]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$PR_ROOT/secrets/wandb_key")"
    export WANDB_API_KEY
fi
# C extension build flags for every job (DECISIONS D90): one value for all jobs, so jobs sharing a checkout
# never rebuild it with different flags; x86-64-v3 = AVX2 / FMA (EPYC 9454, Xeon 8592+ / 8380 all have it)
export PR_OPT_FLAGS="${PR_OPT_FLAGS--O3 -march=x86-64-v3}"
export PYTHONNOUSERSITE=1                            # ignore ~/.local site-packages (could hold another pufferlib)
unset PYTHONPATH
# Torch's site profile (/share/apps/profile.d/nyu.sh) exports BASH_ENV=/etc/profile.d/cluster.sh, so every
# non-interactive bash -- including the bash launcher behind the `apptainer` command -- re-sources it and
# re-exports APPTAINER_BINDPATH and APPTAINER_CACHEDIR/SINGULARITY_CACHEDIR (/scratch/$USER/...-cache),
# undoing the settings below. Our processes do not inherit it (DECISIONS D80).
unset BASH_ENV ENV
# The Torch login profile exports APPTAINER_BINDPATH=/scratch,/projects,/share/apps,/mnt,/state/partition1,
# which would show all of /scratch (incl. $PD_REF) inside our containers and breaks image builds; our
# jobs bind only $PR_ROOT (DECISIONS D79).
unset APPTAINER_BINDPATH SINGULARITY_BINDPATH APPTAINER_BIND SINGULARITY_BIND
export PD_REF=/scratch/$USER/EMERGE/PufferDrive      # read-only reference; used only by the isolation check
mkdir -p "$TMPDIR" "$APPTAINER_CACHEDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_DIR" \
    "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$DATA_DIR" "$PR_ROOT/logs" 2>/dev/null || true
