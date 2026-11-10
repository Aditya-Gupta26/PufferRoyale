#!/bin/bash
# PufferRoyale: build our Apptainer image (Path A, docs/HPC_HANDOFF.md §4c) on a CPU node.
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs && sbatch $REPO/hpc/build_image.sbatch
# Builds $SIF from $REPO/hpc/pufferroyale.def with --fakeroot (allowed on Torch; no /etc/subuid entry,
# so a root-mapped namespace), in the job's node-local temp directory, then `apptainer test`s it and
# moves it into place. The previous image, if any, is kept as $SIF.prev. Records versions in the log.
#
#SBATCH --job-name=pr-build
#SBATCH --account=torch_pr_355_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
NODE_TMP="${TMPDIR:-/tmp}"                     # SLURM's job-private node-local dir (env.sh moves TMPDIR)
source /scratch/$USER/pufferroyale/env.sh
# a fakeroot sandbox on node-local disk: fast, and no ownership tricks on the network filesystem
if [ "$(df -Pk "$NODE_TMP" | awk 'NR==2 {print $4}')" -gt 41943040 ]; then
    export APPTAINER_TMPDIR="$NODE_TMP/pr_build_${SLURM_JOB_ID:-local}"
fi
mkdir -p "$APPTAINER_TMPDIR" "$(dirname "$SIF")"
cd "$REPO"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is): building $SIF from $(git rev-parse --short HEAD)" \
     "(tmp $APPTAINER_TMPDIR, cache $APPTAINER_CACHEDIR); apptainer $(apptainer --version)"
NEW="$SIF.build_${SLURM_JOB_ID:-local}"
apptainer build --fakeroot "$NEW" hpc/pufferroyale.def
apptainer test "$NEW"
apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" "$NEW" python -c \
    "import sys, numpy, torch, pufferlib, wandb, matplotlib, pytest; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'wandb', wandb.__version__, 'matplotlib', matplotlib.__version__, 'pytest', pytest.__version__)"
[ -f "$SIF" ] && mv -f "$SIF" "$SIF.prev"
mv -f "$NEW" "$SIF"
rm -rf "$APPTAINER_TMPDIR"
ls -la "$SIF"
echo "done: $SIF ($(date -Is))"
