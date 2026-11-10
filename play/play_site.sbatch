#!/bin/bash
# PufferRoyale human-play site on a CPU node of NYU Torch (play/play_server.py): anyone with the link plays the
# policy in a browser, at most MAX_PLAYERS matches at once (later players wait in line). Three processes:
#   1. the game server (in the image), on 127.0.0.1 when a public tunnel is used, else on the node's network;
#   2. hpc/llm_relay.py on the HOST: the Claude Code CLI with the owner's own login answers the Claude commentary /
#      win-probability requests (owner sessions only; no API key; at most CLAUDE_DAILY_CALLS calls per day);
#   3. optional public tunnel: $PR_ROOT/tools/cloudflared (installed by the owner, not by this script) opens a
#      Cloudflare "quick tunnel" (https://<random>.trycloudflare.com, no account): reachable from anywhere without the
#      NYU VPN. Without it, the site is reachable only through an SSH tunnel to the node.
#
#     source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
#     sbatch $REPO/play/play_site.sbatch
#     cat $PR_ROOT/play_site/links.txt          # the public link to share and your owner link (Claude toggles)
#
# The site token and the owner key are kept in $PR_ROOT/secrets/ (created once, mode 600), so the links keep their
# ?token= part across restarts; a quick tunnel's host name changes at every start. Players, decks and match records
# live in $DATA_DIR/human_play/ and survive restarts. scancel the job to take the site down.
# The game server runs in a loop: `bash $REPO/play/play_restart.sh` restarts it with the current code and page while
# the tunnel stays up, so the link does not change (matches in progress end; they are saved as unfinished).
# Optional: CHECKPOINT (S4_A_s1's latest), MAX_PLAYERS (5), PUBLIC (1 = use the tunnel if installed, 0 = never),
# CLAUDE (1 = start the relay, 0 = no Claude), CLAUDE_DAILY_CALLS (200), GREEDY=1, PORT (first free from 8765),
# TTS (1 = start play/tts_worker.py when its engine is installed; 0 = never), TTS_ENGINE (kokoro, installed by
# play/kokoro_install.sh; chatterbox needs a GPU), TTS_VOICE (am_fenrir), TTS_THREADS (4).
# 4. optional neural voice: play/tts_worker.py (offline, on this CPU node) turns each commentary line into audio.
#SBATCH --job-name=pr-play
#SBATCH --account=torch_pr_355_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
source /scratch/$USER/pufferroyale/env.sh
CHECKPOINT="${CHECKPOINT:-$REPO/models/S4_A_s1_10B.pt}"            # = S4_A_s1 model_610352.pt (same sha256)
MAX_PLAYERS="${MAX_PLAYERS:-5}"
PUBLIC="${PUBLIC:-1}"
CLAUDE="${CLAUDE:-1}"
CLAUDE_DAILY_CALLS="${CLAUDE_DAILY_CALLS:-200}"
CLOUDFLARED="$PR_ROOT/tools/cloudflared"
TTS="${TTS:-1}"
TTS_THREADS="${TTS_THREADS:-4}"
TTS_ENGINE="${TTS_ENGINE:-kokoro}"                       # kokoro (CPU, ~1 s per line) or chatterbox (needs a GPU)
TTS_VOICE="${TTS_VOICE:-am_fenrir}"
TTS_PY="$PR_ROOT/tools/$([ "$TTS_ENGINE" = chatterbox ] && echo tts_py || echo kokoro_py)"
# per-job working directories: a new site job may start while the old one still runs (the old one's exit touches
# STOP files in ITS directories and must not stop the new one's relay or voice worker)
VOICE="$DATA_DIR/human_play/voice/job_${SLURM_JOB_ID:-local}"
[ -r "$CHECKPOINT" ] || { echo "checkpoint $CHECKPOINT not readable" >&2; exit 2; }
NODE=$(hostname -s)
TUNNEL=0
if [ "$PUBLIC" = 1 ] && [ -x "$CLOUDFLARED" ]; then TUNNEL=1; fi
HOST=$([ "$TUNNEL" = 1 ] && echo 127.0.0.1 || echo 0.0.0.0)
PORT="${PORT:-$(python3 -c '
import socket
for p in range(8765, 8865):
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", p)); print(p); break
    except OSError:
        pass
    finally:
        s.close()
')}"
[ -n "$PORT" ] || { echo "no free port in 8765-8864 on $NODE" >&2; exit 3; }

umask 077
mkdir -p "$PR_ROOT/play_site" "$PR_ROOT/secrets" "$DATA_DIR/human_play"
chmod 700 "$PR_ROOT/play_site" "$PR_ROOT/secrets"
for f in play_site_token play_owner_key; do
    [ -s "$PR_ROOT/secrets/$f" ] || python3 -c 'import secrets; print(secrets.token_urlsafe(18))' > "$PR_ROOT/secrets/$f"
done
TOKEN=$(cat "$PR_ROOT/secrets/play_site_token")
OWNER=$(cat "$PR_ROOT/secrets/play_owner_key")
LINKS="$PR_ROOT/play_site/links.txt"
RELAY="$PR_ROOT/play_site/claude_relay_${SLURM_JOB_ID:-local}"
echo "pr-play job $SLURM_JOB_ID on $NODE: port $PORT, host $HOST, tunnel $TUNNEL, claude $CLAUDE, checkout $REPO @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"

PIDS=()
SERVER_PID="$PR_ROOT/play_site/server_${SLURM_JOB_ID:-local}.pid"          # play/play_restart.sh reads it
STOP_SERVER="$PR_ROOT/play_site/STOP_SERVER_${SLURM_JOB_ID:-local}"
MYLINKS="$PR_ROOT/play_site/links_${SLURM_JOB_ID:-local}.txt"
PAGES_KEY="$PR_ROOT/secrets/github_pages_token"           # the permanent link's key (play/play_link.py); optional
cleanup() {
    [ -d "$RELAY" ] && touch "$RELAY/STOP"
    [ -d "$VOICE" ] && touch "$VOICE/STOP"
    touch "$STOP_SERVER"
    [ -s "$SERVER_PID" ] && kill "$(cat "$SERVER_PID")" 2>/dev/null || true
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
    # the permanent link: marked down only if it still names this job (play/play_link.py checks)
    if [ "${TUNNEL:-0}" = 1 ] && [ -s "$PAGES_KEY" ]; then
        timeout 20 /usr/bin/python3 "$REPO/play/play_link.py" down --job "$SLURM_JOB_ID" >> "$PR_ROOT/play_site/link.log" 2>&1 || true
    fi
    # links.txt says DOWN only if it still shows this job (a newer site job may have taken it over)
    if [ ! -s "$LINKS" ] || grep -q "job $SLURM_JOB_ID " "$LINKS"; then
        printf 'PufferRoyale play site is DOWN (job %s ended %s)\n' "$SLURM_JOB_ID" "$(date '+%F %H:%M')" > "$LINKS"
    fi
    rm -f "$MYLINKS"
}
trap cleanup EXIT

CLAUDE_ARGS=()
if [ "$CLAUDE" = 1 ]; then
    CLAUDE_BIN="$(command -v claude || echo "$HOME/.local/bin/claude")"
    if [ -x "$CLAUDE_BIN" ]; then
        mkdir -p "$RELAY"; rm -f "$RELAY/STOP"; rm -rf "$RELAY"/g*
        /usr/bin/python3 "$REPO/hpc/llm_relay.py" --relay-dir "$RELAY" --model claude-opus-5-5 --effort low \
            --claude "$CLAUDE_BIN" --timeout 90 --max-wait 100 >> "$PR_ROOT/play_site/claude_relay.log" 2>&1 &
        PIDS+=($!)
        CLAUDE_ARGS=(--claude-relay-dir "$RELAY" --claude-daily-calls "$CLAUDE_DAILY_CALLS")
    else
        echo "no Claude Code CLI at $CLAUDE_BIN: the Claude toggles stay off" >&2
    fi
fi

VOICE_ARGS=()
if [ "$TTS" = 1 ] && [ -d "$TTS_PY/$TTS_ENGINE" ]; then
    mkdir -p "$VOICE"; rm -f "$VOICE/STOP" "$VOICE/HEARTBEAT"
    apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" --pwd "$REPO" \
        --env "PYTHONPATH=$TTS_PY,HF_HOME=$PR_ROOT/tools/hf_cache,HF_HUB_OFFLINE=1,OMP_NUM_THREADS=$TTS_THREADS,TMPDIR=/tmp" "$SIF" \
        python play/tts_worker.py --dir "$VOICE" --engine "$TTS_ENGINE" --voice "$TTS_VOICE" --threads "$TTS_THREADS" \
        >> "$PR_ROOT/play_site/tts.log" 2>&1 &
    PIDS+=($!)
    VOICE_ARGS=(--voice-dir "$VOICE")
fi

EXTRA=()
[ "${GREEDY:-0}" = 1 ] && EXTRA+=(--greedy)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
serve() {                       # the game server, restarted whenever it exits (play/play_restart.sh) until STOP_SERVER
    local n=0 rc
    while [ ! -f "$STOP_SERVER" ]; do
        apptainer exec --no-home --bind "$PR_ROOT:$PR_ROOT" --pwd "$REPO" "$SIF" \
            python play/play_server.py --checkpoint "$CHECKPOINT" --host "$HOST" --port "$PORT" \
            --token-file "$PR_ROOT/secrets/play_site_token" --owner-key-file "$PR_ROOT/secrets/play_owner_key" \
            --data-dir "$DATA_DIR/human_play" --max-players "$MAX_PLAYERS" --threads 2 \
            ${CLAUDE_ARGS[@]+"${CLAUDE_ARGS[@]}"} ${VOICE_ARGS[@]+"${VOICE_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} &
        echo $! > "$SERVER_PID"
        wait $! && rc=0 || rc=$?
        n=$((n + 1))
        [ -f "$STOP_SERVER" ] && break
        echo "[play] game server exited (status $rc, start $n) at $(date '+%F %T'); restarting" >&2
        sleep $([ "$n" -lt 5 ] && echo 2 || echo 30)          # back off if it keeps failing
    done
}
rm -f "$STOP_SERVER" "$SERVER_PID"
serve &
SERVER=$!
PIDS+=($SERVER)
for _ in $(seq 120); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/api/meta?token=$TOKEN" && break
    kill -0 "$SERVER" 2>/dev/null || { echo "the game server exited" >&2; exit 4; }
    sleep 1
done

if [ "$TUNNEL" = 1 ]; then
    TLOG="$PR_ROOT/play_site/cloudflared_${SLURM_JOB_ID}.log"
    "$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$TLOG" 2>&1 &
    PIDS+=($!)
    BASE=""
    for _ in $(seq 90); do
        BASE=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TLOG" | head -1 || true)
        [ -n "$BASE" ] && break
        sleep 1
    done
    [ -n "$BASE" ] || { echo "the Cloudflare tunnel gave no URL (see $TLOG)" >&2; exit 5; }
    PERM=""                                                # the permanent link now forwards here
    if [ -s "$PAGES_KEY" ]; then
        if timeout 60 /usr/bin/python3 "$REPO/play/play_link.py" set "$BASE" --job "$SLURM_JOB_ID" >> "$PR_ROOT/play_site/link.log" 2>&1; then
            PERM=$(/usr/bin/python3 "$REPO/play/play_link.py" url)
        else
            echo "the permanent link was not updated (see $PR_ROOT/play_site/link.log)" >&2
        fi
    fi
    {
        echo "PufferRoyale play site -- job $SLURM_JOB_ID on $NODE, up since $(date '+%F %H:%M'), at most 2 days"
        echo
        if [ -n "$PERM" ]; then
            echo "Permanent link (share this one; it always forwards to the current server; open link, players sign"
            echo "in with name + PIN; the page adds the site token itself):"
            echo "  ${PERM}"
            echo "  owner (Claude toggles; do NOT share):  ${PERM}?owner=$OWNER"
            echo
            echo "This job's own address (changes at every restart):"
        else
            echo "Share this link (anyone, no VPN; max $MAX_PLAYERS matches at once, others wait in line):"
        fi
        echo "  $BASE/?token=$TOKEN"
        echo
        echo "Your owner link (adds the Claude commentary / win-probability toggles; do NOT share):"
        echo "  $BASE/?token=$TOKEN&owner=$OWNER"
        echo
        echo "Stop: scancel $SLURM_JOB_ID   (and touch $PR_ROOT/play_site/NO_RENEW to stop the renewal chain)"
    } > "$MYLINKS"
else
    {
        echo "PufferRoyale play site -- job $SLURM_JOB_ID on $NODE, port $PORT, up since $(date '+%F %H:%M') (no public tunnel)"
        echo
        echo "1) On your computer (keep it open):  ssh -N -L $PORT:$NODE:$PORT $USER@<your Torch login host>"
        echo "2) Open:  http://localhost:$PORT/?token=$TOKEN"
        echo "   Owner (Claude toggles):  http://localhost:$PORT/?token=$TOKEN&owner=$OWNER"
        echo
        echo "Stop: scancel $SLURM_JOB_ID"
    } > "$MYLINKS"
fi
cp "$MYLINKS" "$LINKS"
echo "links: $LINKS (also $MYLINKS)"

# Renewal (user's choice, 2026-11-09): queue the next site job to become eligible RENEW_AFTER_MIN minutes after this
# one started (default 47 h; the limit is 48 h), so it can take over before this one ends. One successor at a time;
# none while $PR_ROOT/play_site/NO_RENEW exists. The successor queues its own successor when it starts.
RENEW="${RENEW:-1}"
RENEW_AFTER_MIN="${RENEW_AFTER_MIN:-2820}"
if [ "$RENEW" = 1 ] && [ ! -f "$PR_ROOT/play_site/NO_RENEW" ]; then
    if [ -n "$(squeue -u "$USER" -h -n pr-play -t PENDING -o %i 2>/dev/null)" ]; then
        echo "renewal: a pr-play job is already queued; none added"
    elif NEXT=$(cd "$PR_ROOT/logs" && env $(compgen -e | grep -E '^(SLURM|SBATCH)_' | sed 's/^/-u /') \
                sbatch --parsable --dependency="after:$SLURM_JOB_ID+$RENEW_AFTER_MIN" "$REPO/play/play_site.sbatch"); then
        echo "renewal: next site job $NEXT queued (eligible $RENEW_AFTER_MIN min after this job started)"
    else
        echo "renewal: could not queue the next site job" >&2
    fi
fi
wait "$SERVER"
