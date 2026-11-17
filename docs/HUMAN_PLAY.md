# Human play: play the policy in a browser

`play/play_server.py` serves a web page where people play PufferRoyale against a trained policy (default:
S4_A_s1's latest model, `model_610352.pt`, 10.0B steps). `play/play_site.sbatch` runs it on a Torch CPU node.
Tests: `tests/builder/test_play_site.py`.

## Running it

| Where | Command | Then |
|---|---|---|
| Torch | `sbatch $REPO/play/play_site.sbatch` (from `$PR_ROOT/logs`, after `source env.sh`) | `cat $PR_ROOT/play_site/links.txt` |
| Any machine with the package built | `python play/play_server.py --checkpoint <model.pt>` | open the printed URL |

The Torch job runs up to 2 days (the CPU queue's limit for this job size); `scancel` it to take the site down.
**Updating the running site without changing the link:** the job runs the game server in a loop, so
`bash $REPO/play/play_restart.sh` restarts it with the current code and page while the Cloudflare tunnel (the link)
stays up. Matches in progress end and are saved as unfinished. (Jobs started before 2026-11-09 have no loop.)
Each site job keeps its own Claude relay directory (`play_site/claude_relay_<job>`), voice directory
(`human_play/voice/job_<job>`, so replays of older matches read their lines with the browser's voice), pid file
and `links_<job>.txt`, so a new site job can start while the old one still runs; `links.txt` is the newest job's,
and an ending job marks it DOWN only if it still shows that job.
`links.txt` holds the link to share and the owner link (see Claude below). Without a public tunnel it holds an SSH
tunnel command instead (the node is reachable only from inside NYU).

**Public link (no VPN).** With an executable `$PR_ROOT/tools/cloudflared` (Cloudflare's tunnel client, installed
by the owner), the job opens a Cloudflare quick tunnel: `https://<random>.trycloudflare.com/?token=...`, reachable
from anywhere, no Cloudflare account. The host name changes at every job start (the `?token=` part does not).
Quick tunnels have no uptime guarantee. A fixed host name needs a Cloudflare account and domain (a named tunnel).

## Permanent link and renewal (2026-11-09, user's choice: GitHub Pages, $0)

A quick tunnel's address changes at every site job, and a job lasts at most 2 days. So the link people keep is
**`https://aditya-gupta26.github.io/pr-play/`**, a forwarding page on GitHub Pages. Since 2026-11-09 19:21 it is an
**open link** (the owner's choice, together with name + PIN sign-in): no `?token=` is needed.

- **The repo** `Aditya-Gupta26/pr-play` (public, Pages on `main`) holds `index.html` (from `play/play_redirect.html`)
  and `target.json` (the current tunnel address, its job id, up/down, and the site token). The owner key is never there.
- **The forwarding page** reads `target.json` (from the GitHub API, fresh; the Pages copy as a fallback), checks the
  server's `/api/alive` (no token needed; it answers only `{"ok": true}`), then sends the visitor to `<tunnel>/` with
  the token from `target.json` (replacing any older `?token=` in the visitor's link) and any other parameter of the
  visitor's link kept (the owner's `?owner=...`). While no server is up it says so and retries by itself.
- **The token is public now.** Anyone can read it in `target.json`. It still keeps out traffic that does not come
  through the link. To change it: write a new token to `$PR_ROOT/secrets/play_site_token`, restart the server
  (`play/play_restart.sh`), then `python3 play/play_link.py set <tunnel> --job <id>`: the link people have keeps
  working. To close the site to the public again: `set ... --private` (visitors then need a link with `?token=`).
- **Owner link:** `https://aditya-gupta26.github.io/pr-play/?owner=<owner key>` (in the site job's links file).
- **Updating it:** `play/play_link.py` (standard library). Each site job runs `set <tunnel> --job <id>` when its
  tunnel is up and `down --job <id>` when it ends; `down` changes nothing if a newer job has taken over. The key is a
  fine-grained GitHub token with Contents read/write on that one repo, in `$PR_ROOT/secrets/github_pages_token`
  (made and saved by the owner from their own terminal, never printed). When it expires, make a new one and save it there.
  The key reader uses the one well-formed key in the file: on 2026-11-09 the first paste brought 21 terminal control
  codes along (GitHub answered 401). Pages was switched on in the repo's Settings -> Pages (the key had Contents
  write only; `play_link.py enable-pages` got 403). Live since 2026-11-09 15:55. Open link checked 2026-11-09 19:25 in
  headless Chromium (desktop and a 390 px phone): the plain link, and an old-style link with a wrong `?token=`, both
  land in the lobby of the current tunnel.
- **Renewal:** each site job queues the next one with `--dependency=after:<this job>+2820` (eligible 47 h after it
  started; the limit is 48 h), unless a `pr-play` job is already queued or `$PR_ROOT/play_site/NO_RENEW` exists. When
  the next one starts, it takes over the permanent link; the old one runs to its end. How early it starts depends on
  the CPU queue. To stop the chain: `touch $PR_ROOT/play_site/NO_RENEW`, then `scancel` the queued `pr-play` job.
  Note: `after:<job>+<min>` counts from that job's start, or from its cancellation if it is cancelled.
- **Overlap safety:** while two site jobs run they share the data. Each keeps its own relay, voice, pid and links
  files; `players.json` is merged on every save (players only the file has are kept, and the newer PIN of a player
  wins), so a player or PIN made on one is not lost by the other. The leaderboard of a server shows matches saved by
  the other one after its next start.

## What a player sees

- **Anti-stall rule (since 2026-11-17, the owner's request for "a simple fix RIGHT NOW").** The released model can
  sit and wait when the opponent goes quiet (see the passive-opponent diagnosis, runs/diag/passive_probe_20261116/).
  On the site, once the model has been on full elixir (10) for 5 s (`ANTI_STALL_S`, `--anti-stall`, 0 = off; the
  owner chose 5 s over 2 s), its "wait" action is masked out for that decision. It then plays the card and spot its
  own policy prefers among plays, sampled as always. The lobby label says so, each forced play is marked
  `"forced": true` in the match events, and the record's `opponent` block holds `anti_stall_s` and
  `forced_plays`. Replays are unaffected (they replay the recorded plays). Hint and Autopilot do not use the rule.
  Test: runs/diag/anti_stall_20261117/.
- **Welcome window (since 2026-11-17, the owner's request).** A first-time visitor who is not signed in gets a
  two-step window. Step 1 is how to play: how to win, how to play a card, elixir, keys, pinching out on a phone, and
  Hint. Step 2 is the name and PIN, with two tabs:
  - **New player:** name, PIN, and the PIN typed again. It sends `new: true` to `/api/hello`, so a name that is
    already taken gets "that name is taken" before any PIN check and never uses up that player's PIN tries.
    Tested: six "New player" tries on a taken name with wrong PINs, then one wrong PIN on "I have played before",
    which said "4 tries left".
  - **I have played before:** name and PIN, the same login as the lobby form.

  "Look around first" or Esc closes it. The browser remembers that (localStorage `pr_welcome_seen`), so it does not
  come back. A signed-in browser never gets it. "How to play" in the lobby header opens step 1 again. On phones,
  inputs and menus use 16 px text, because iOS Safari zooms into any text box under 16 px when it gets focus.
- **Lobby.** Name, deck, opponent's deck, seat and speed, plus the leaderboard and recent matches.
  - **Deck:** one of the 7 training-pool presets, or your own 8 cards from the 64.
  - **The name alone is the login (since 2026-11-17, the owner's request: "get rid of the PIN system whatsoever -
    just let the user enter their name and let them play").** The same name (any case or spacing, with or without
    `_ - . '`) is the same player on any device; a new name makes a new player. No PIN is asked for, checked or
    made: the page hides every PIN box (`META.pins` is false), `/api/pin` answers "this site has no PINs", and the
    "New player" tab and the "taken" check are gone with them. Nothing was deleted: the PIN records players
    already have stay in players.json, unused. The old login below comes back with `--pins` (tests run it that
    way). Anyone who types a name plays as that player.
  - **Remembered sign-in (2026-11-17, the owner's request):** a browser that signed in once is signed in the next
    time, also after the game's address changes. The game's address changes every two days and a browser keeps a
    sign-in per address, so the permanent link's page (hpc/play_redirect.html, an address that never changes) keeps
    the player id on the device and hands it to the game in the address's #fragment (`#home=..&pid=..`), which
    browsers never send to a server; the game removes it from the address bar at once. The game trusts `home` only
    if its SHA-256 is in `HOME_SHA` (a hash, so the page names no account). After a sign-in, or the new **Sign
    out**, the game makes a quick round trip to that page (`#save=<id>&back=<game>` / `#forget=1&back=<game>`;
    `back` must be a trycloudflare address). A sign-out also leaves a "signed out" mark at the permanent link, handed over as `#out=1`, so an older game address that still remembers the player forgets it too; with neither mark nor id, a sign-in the game address already had (from before this change) is kept and sent up once. Private windows or cleared site data forget it, as with any site.
  - **Name + PIN login (2026-11-09 19:20 to 2026-11-17, now only with `--pins`):** a name with its 4-digit PIN (any case
    or spacing, on any device or tunnel address) continues that player, with their record, saved deck and settings.
    A new name must choose a PIN and makes a new player; no record is made until a name is given. The browser also
    remembers the player id per address (localStorage; that id is itself a long secret, so that browser needs no
    PIN), but a quick tunnel's address changes at every restart, so name + PIN is what carries a player across.
    - Only a salted PBKDF2-SHA256 hash of each PIN is stored (100,000 iterations); the page never receives it.
    - After 5 wrong PINs in a row a name is locked for 10 minutes (10,000 PINs at 5 per 10 min: about 14 days to
      try them all). The lock is per name and in memory (a server restart clears it).
    - **Players from before PINs** (the 5 players of 2026-11-06..09): at their next sign-in, by name or by their
      browser's stored id, the site gives them a random PIN, shown once ("I saved it"). The owner chose this; the
      risk is that someone else types their name first.
      Since 2026-11-17 PINs are off (see above), so this no longer applies.
    - **Names are unique up to case, spaces and `_ - . '` (since 2026-11-17).** `name_key` drops those characters,
      so "Sam", "sam." and "S am" are one name with one PIN. A name with no letter or digit is refused.
      No existing names collided under the new key.
    - **Change my PIN:** a signed-in player can choose a new PIN (link under the name box).
    - **Owner maintenance:** `$DATA_DIR/human_play/admin.json` = `{"drop": [names], "reset_pin": [names]}`, applied
      once at the next server start (`play/play_restart.sh`), then renamed `admin.done.<time>.json` and logged in
      `merge_log.jsonl`. `drop` moves the player's match records to `removed/` (not deleted); `reset_pin` gives the
      player a new PIN at their next sign-in. Use it while one server runs (an overlapping old server does not know
      the change). 2026-11-09 19:20: one duplicate player dropped at the owner's request (1 match moved to `removed/`).
  - **Old duplicates:** player records made before 2026-11-06 16:29 with the same name were merged into the one
    with the most matches. Their matches were moved to it, unnamed records without matches were dropped, and every
    change is in `merge_log.jsonl`. The backup is `runs/human_play_backup_20261106_162904/`.
- **Seats.** At most `--max-players` (5) matches run at once. Later players wait in a line that moves on its own;
  the page shows their place.
- **Match.** The engine runs on the server in real time, 20 ticks/s times the chosen speed, with the training
  settings: frame skip 10, deploy lockout 90 ticks, absolute tiebreak, Princess towers.
  - **The policy's beat, the player's tick:** every 10 ticks (0.5 s) the policy acts on its own observation, its
    training cadence. The player's plays go in at the next tick, at most one per tick, so within 50 ms at 1x, like a
    real player's. Before 2026-11-09 they also waited for the beat (0-500 ms, median 250 ms over the 411 plays then
    recorded); matches from before that date had that handicap.
  - **Playing a card:** tap a card, then a tile; or drag the card onto the board (mouse or finger). A tap or drop up
    to 2 tiles outside the allowed area (one tile over the river, a tower's edge) lands on the nearest allowed tile,
    and the preview shows where. The card appears at once where it will land, its cost leaves the elixir bar, and it
    turns real when the server applies it; a refused play disappears with the reason. A card can be on its way only
    once, and a play queued behind another is checked against the elixir the first one leaves.
  - **Every tick reaches the page:** units walk, fly and fight; spells fly to their target; each new play shows the
    card's name where it lands.
  - **Same information as a real player:** the player never sees the policy's hand or elixir.
  - **Like a real match:** until the match ends, the page also gets nothing about the policy's deck (no list of its
    cards seen so far, no deck name). Its plays still appear on the board as they happen. The result screen shows
    its deck. Claude's input drops the opponent's elixir estimate and its seen-card list, and its prompt forbids
    talking about them. (The lobby still lets a player choose the policy's deck on purpose; the default is a
    random preset, revealed only at the end.)
  - **Look:** everything is drawn procedurally: grass tiles tinted by half, stone pads under the towers, plank
    bridges, animated water and an arena wall.
    - **Towers** are stone towers with a team-coloured roof and a crown or pennant emblem. A large HP bar with the
      number sits on the side away from the river. A hit flashes the tower, a lighter trail shows the damage just
      taken, cracks appear below 67% and 34% HP, and a destroyed tower leaves rubble.
    - **Units** are shaded team tokens that flash when hit, with a spawn ring, a shadow (raised for flyers), name
      labels and HP bars.
    - **Names (since 2026-11-16, user request "every cards name should be visible as it moves"):** every card on
      the board is named, and the name moves with it. Names are drawn after all units and projectiles, so nothing
      covers them. A unit of a one- or two-unit card shows its own name (Hog Rider, Golemite). A swarm card (3 or
      more units: Skeletons, Skeleton Army, Goblin Gang, Bats, Minion Horde, ...) shows its card name once per group
      of its units within 2.5 tiles of each other, so 15 skeletons walking together show one "Skeleton Army" and a
      group that splits off gets its own. Spells are named while they fly (Fireball, Rocket, Giant Snowball, Goblin
      Barrel) or act on an area (Arrows, Poison, Earthquake, The Log, Barbarian Barrel). Zap, Freeze and Lightning
      leave nothing on the board, so they show only the name that pops up where any card lands. The **Aa** button
      (key L) hides or shows all names.
    - **Icons:** where the browser has no colour-emoji font, cards and units show short name tags (for example
      "HR" for Hog Rider) instead of emoji.
    - **Towers panel:** the side panel lists all six towers' HP.
- **Toggles.**
  - **Assist: Off / Hint / Autopilot.** Any level makes the match unranked.
    - **Hint:** the same policy, from the player's side, shows its top 3 cards, each with its most likely tile and
      probability; a gold ring marks its top pick.
    - **Autopilot:** that policy also *plays* the player's side, with its own random stream, on every beat where the
      player has queued nothing. A card the player plays always comes first, so the player can take over at any
      moment. A yellow badge on the river says so, and autopilot plays are marked in the records (`"auto": true`)
      and replay exactly.
  - **Claude commentary and Claude win probability** (owner only, see below).
    - **The commentary** is a dramatic arena announcer in a playful roast mood, teasing the plays (the player's and
      the AI's), never the person.
    - **User's rule (2026-11-06):** one complete line of at most 10 words and no numbers: no hit points, levels,
      elixir, times or counts. The prompt asks for it, and `clean_commentary()` enforces it.
      - Words with digits are dropped.
      - A longer line is cut at its last comma or sentence end within the first 10 words, else at 10 words without
        a dangling "its" or "the".
    - **The win probability** is drawn as a step line over the match.
  - **Voice:** a per-browser setting, also available to spectators and in replays that have commentary.
    - **Neural voice:** when `play/tts_worker.py` runs, each line becomes an audio clip
      (`$DATA_DIR/human_play/voice/<match>_<n>.wav`). The worker runs on the site's CPU node, offline, and the site
      job starts it.
      - **Engine:** Kokoro-82M (Apache 2.0), installed by `play/kokoro_install.sh`.
      - **Speed (measured 2026-11-06, 4 threads):** 0.6-1.4 s per 10-word line; the worker loads in about 40-60 s.
      - **Voices:** Fenrir, Michael and Puck (US male), George (UK male) and Heart (US female). The owner picks one in
        the game; the choice is per browser.
      - **Chatterbox** (`play/tts_install.sh`, more dramatic) took 8-12 s per line on a CPU node; it is kept as
        `--engine chatterbox` for a GPU.
    - **How the page gets it:** the server marks a clip ready and serves it at `/api/audio?id=`; the page plays it.
    - **Fallback:** without the worker, or if a clip is not ready within 8 s, the browser's own speech reads the
      line. It picks the most natural voice available (Edge's "Natural", Apple's "Enhanced"), with livelier pitch
      and pace on "!" and "?".
- **Records.** Every match is saved to `$DATA_DIR/human_play/matches/<id>.json`:
  - result, crowns, decks, settings, toggles used, Claude's lines and probabilities, every applied play (with the
    page's play id and how long it waited on the server);
  - the policy's decision times, and (from 2026-11-09) how the player's page was connected (`ws`, `poll`) and the
    round trips it measured;
  - a **replay record**: seed, decks, every applied play of both sides with its tick, and the engine state hash
    every 600 ticks and at the end.
- **Watching live.** The lobby's *Live now* panel lists the matches being played (player, clock, crowns, number of
  watchers). **Watch live** shows a match in real time from the player's side: their hand and elixir, never the
  policy's deck or elixir (the same view as the player, so a watcher cannot pass on more than the player sees).
  - **Read only:** spectators cannot play, pause, change speed or toggle anything (the server answers 400).
  - **Seats and limits:** watching takes no seat; at most 20 spectators per match and 60 in all.
  - **Players:** a player sees "N watching".
  - **No zombie matches:** a spectator never keeps a match alive; only the player's own requests count.
  - **When the player leaves:** spectators see "<name> left the match".
- **Replays.** "Watch" re-simulates a match from its record (the engine is deterministic). The page shows "✓
  verified" when the final state hash matches, offers seek and speed, and can save the replay as a WebM video,
  recorded in the browser.
- **Leaderboard.** It ranks players by score against the policy, (wins + ½ draws) per match, over **ranked**
  matches only: finished, 1× speed throughout, advisor never on. Player ids are never shown (they work as the login).
- **Games against language models** (added 2026-11-14). A lobby panel shows the 100 games four Claude models
  played against the released policy (25 each, the same games for every model): per model W-D-L, Claude's score
  with its 95% range and one square per game, the score per setup, and a game list per model.
  - **Data:** `docs/figures/claude_games.json`, written by `docs/figures/make_figures.py --only claude_games`
    (no costs and no Claude text), served at `GET /api/llm_games`. The server re-reads it when the file changes.
  - **Replays:** "Watch" (or a square) opens the game's GIF from `GET /api/llm_replay?model=<id>&k=<game>`. Only
    GIFs that the JSON names and that lie inside `--llm-replays` are served (default:
    `<data-dir>/../claude_games/S4_A_s1_10B` when it exists; elsewhere the panel shows results without replays).
  - Same site token as every other request; `play/play_server.py` prints how many games and replays it found at start.

## Network and phones

- **Live channel: a WebSocket (from 2026-11-09).** The page opens `/api/ws`. The server pushes every engine tick
  (20 a second; when a page is slower, it gets the newest frame, never a backlog), and the page sends its plays on
  the same connection (acknowledged at once). Frames carry only the plays of the last 5 s: about 2 KB each.
  - **Fallback: polling.** The page asks for each frame (`/api/state?after=<version>`); the server answers as soon
    as a newer frame exists, or after at most 0.5 s, and plays go by POST. The page switches to it for the rest of
    the session when the WebSocket does not open within 4 s, sends no first frame within 3 s, or goes quiet for
    1.5 s while the match runs (a proxy holding it back). The ping label then ends in "(poll)".
  - **Why not a stream:** Cloudflare's quick tunnel held a streamed response (server-sent events) back. Measured
    2026-11-06 through the tunnel, gaps between frames were up to 3.2 s, so the board froze and then jumped; polling
    through the tunnel then gave an update every ~140 ms (one round trip), worst gap 244 ms.
  - **Measured 2026-11-09 on the server's own machine (headless Chromium, no tunnel), old page vs new:** 20 frames
    per second both; tap card then tile: on the board 200-480 ms after the tap (old: the beat) vs 37-65 ms (new);
    dragging a card onto the board: never played in the old page (mouse or touch: the release went to the card,
    not the board) vs 27-90 ms; WebSocket, polling, a silent WebSocket (switches to polling), pause (the WebSocket
    stays), watching live and replays all checked. Through the tunnel each step adds the player's network round trip.
  - **Measured 2026-11-09 through the deployed site's Cloudflare tunnel** (job 22442925, from a Torch login node, a
    replay session, 20 s): WebSocket 20.2 frames/s, gaps median 50 ms, p99 54 ms, max 66 ms, ~2.7 KB per frame, ping
    round trip 8 ms; polling the same session: 6.0 frames/s, gaps median 163 ms, max 261 ms. The tunnel does not hold
    a WebSocket back (it did hold back server-sent events). Probe: `$PR_ROOT/tmp/play_lag/ws_tunnel_probe.py`.
  - **Smooth motion:** units glide over the measured time between updates.
  - **Ping:** the page shows the round trip next to the connection dot.
  - **Server side:** the event-stream endpoint still exists (tests use it); the page does not use it.
- **Touch input.** The hand is redrawn only when it changes, so a finger never lands on an element that is being
  replaced; a card reacts on touch-down. Canvases draw at most at 2x resolution (3x phone screens look the same and
  draw 2.25x fewer pixels).
- **Zoom (fixed 2026-11-16, user report: "if it zooms in - I cant zoom out or it does not react").**
  - **Cause:** the page asks phones not to zoom (`user-scalable=no`), but Safari on iPhone zooms anyway. The board
    and the hand took every touch for the game (`touch-action: none`), so once zoomed in, a two-finger pinch on the
    board or a card never reached the browser, and those fill most of a zoomed-in screen.
  - **Fix:** one finger stays the game's (tap, drag a card); a two-finger pinch on the board or a card now goes to
    the browser (`touch-action: pinch-zoom`). While the page is zoomed in, the board and cards give all gestures to
    the browser, and a note at the top of the visible area says "Zoomed in: pinch with two fingers to zoom out".
    Double-tap zoom is off on the game screen. A pinch that starts on a card leaves the selection as it was.
  - **Checked** in headless Chromium (390x844 touch screen, zoom allowed as on an iPhone), old page vs new page:
    zoomed in to 3.5-4x from the header row, a pinch out on the board stayed at 3.5-4x on the old page (also on a
    second try) and went back to 1x on the new one. A two-finger spread on a card reached the browser only on the
    new page. A one-finger drag of a card onto the board still plays it.
- **Server stalls seen 2026-11-09.** In one match (14:08) the policy took 100 ms per decision instead of the usual
  ~3.5 ms (2.4 ms re-measured on the node afterwards). The job's 8 cores are its own (cgroup cpuset, no throttling),
  but cluster-wide agents (`ir_agent`, `splunkd`) may run on any core during the node security patching. Not
  reproduced; the match records keep the decision times, so a repeat shows up.
- **Phone layout.** It applies at width 760 px or less, or a touch screen up to 1024 px.
  - **On screen:** the board takes the screen, with a slim header row (the policy's crowns, the clock, Claude's
    win chance when on, and a **Menu** button) and your elixir and hand below.
  - **In the Menu sheet:** Pause, speed, names, Leave, the toggles and the Towers / advisor / Claude panels.
  - **Claude on a phone:** its latest comment shows as a caption on the board for 7 s.
  - **Checked** in headless Chromium at 390×844, 360×740 and 375×667: no overlap, no scrolling, and plays made by
    tapping a card, then a tile.

## Claude (owner only, no API key)

The Claude toggles run through `hpc/llm_relay.py`, the relay of the Claude games (D101). It runs on the host, and
each request is one headless `claude -p` call with the owner's own Claude Code login, model `claude-opus-5-5`,
effort `low`.

- **What Claude gets:** the player's public view (`pufferroyale.llm.render_state`, which never contains the
  policy's hand or elixir), the plays since the last update, and its own previous lines.
- **What it returns:** a final JSON line `{"commentary": ..., "p_win": ...}`. One real call took 5.5 s.
- **Who can use it:** only sessions opened with the owner link (`&owner=<key>` from
  `$PR_ROOT/secrets/play_owner_key`), because it uses a personal subscription. There are at most
  `--claude-daily-calls` (200) calls per day across all matches.
- **Extra charges:** nothing is billed per call. Keep "extra usage" off in the Claude account, so that reaching the
  plan's limit pauses the commentary instead of charging.

## Security

- **Token:** every request needs the site token (`?token=` in the link, `$PR_ROOT/secrets/play_site_token`).
  Without it the server answers 403. Since the open link (2026-11-09) the token is public (in `target.json`); what
  protects players is name + PIN.
- **Input checks:** names are filtered to letters, digits, spaces and `_-.'`; decks are validated (8 different
  cards); POST bodies are capped at 16 KB.
- **Limits:** at most 60 open live connections (WebSockets and event streams) and 30 players in line; a WebSocket
  message is at most 16 KB, a client frame must be masked (RFC 6455), and a WebSocket idle for 20 s is closed.
  The page's Content-Security-Policy allows a WebSocket to its own host only.
- **Browser hardening:** the page sends a strict Content-Security-Policy, and player names and Claude's text are
  inserted as text, never as HTML.
- **Behind a tunnel:** with a tunnel the server listens on 127.0.0.1 only.

The page states that this is an unofficial research simulator, not affiliated with or endorsed by Supercell. It
uses no game art: the board is drawn with shapes and generic emoji (PLAN B6).
