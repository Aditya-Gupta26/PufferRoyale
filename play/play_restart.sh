#!/bin/bash
# Restart the play site's game server (play/play_server.py) with the current code and page, keeping the link: the
# site job (play/play_site.sbatch) runs the server in a loop and its Cloudflare tunnel stays up. Matches in progress
# end and are saved as unfinished; players reload the page.
#
#     bash /scratch/$USER/pufferroyale/PufferRoyale/play/play_restart.sh [JOBID]
#
# Works for site jobs started from the 2026-11-09 version of play/play_site.sbatch on (they write
# $PR_ROOT/play_site/server_<jobid>.pid); an older job has no loop: resubmit it instead (the link then changes).
set -euo pipefail
source /scratch/$USER/pufferroyale/env.sh
JOB="${1:-$(squeue -u "$USER" -h -n pr-play -t RUNNING -o %i | head -1)}"
[ -n "$JOB" ] || { echo "no running pr-play job" >&2; exit 1; }
PIDF="$PR_ROOT/play_site/server_${JOB}.pid"
[ -s "$PIDF" ] || { echo "no $PIDF: job $JOB predates the restart loop; resubmit play/play_site.sbatch instead" >&2; exit 2; }
PID="$(cat "$PIDF")"
[[ "$PID" =~ ^[0-9]+$ ]] || { echo "bad pid in $PIDF" >&2; exit 2; }
srun --jobid="$JOB" --overlap --ntasks=1 --cpus-per-task=1 kill -TERM "$PID"
echo "job $JOB: game server $PID told to stop; the loop starts the current code in ~2 s (ready ~30 s later)"
