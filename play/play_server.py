#!/usr/bin/env python3
"""Play PufferRoyale against a trained policy in a web browser (human-play mode; several players at once).

    # any machine with the package built (pip install -e .): plays the repo's model; open the printed URL
    python play/play_server.py
    python play/play_server.py --checkpoint path/to/model.pt --port 8765

    # NYU Torch: play/play_site.sbatch starts it on a CPU node (see that file for the public link)

Players open the shared link, sign in with a name and a 4-digit PIN (a new name chooses its PIN; the same name +
PIN on any device continues the same player), pick a deck (one of the 7 training-pool presets or their own 8
cards; saved per player, so the next visit starts with it) and the opponent's deck, and play. At most
--max-players matches run at once; later players wait in a line that moves on its own. Players made before PINs
(2026-11-09) get a random PIN shown once at their next sign-in. After PIN_TRIES wrong PINs a name is locked for
PIN_LOCK_S seconds. Only a salted PBKDF2 hash of each PIN is stored.

The match runs in real time on the server: 20 engine ticks per second (x the chosen speed), with the
training settings (frame skip 10, deploy lockout 90 ticks, absolute tiebreak, Princess towers). The policy
acts on its training cadence: every 10 ticks (0.5 s at 1x) it picks one action from its observation of the
current state (sampled from its masked distribution as in the evaluations; --greedy for the card-first greedy
rule). The player's plays go in at the next tick (at most one per tick, so within 50 ms at 1x), like a real
player's. A player sees what a player sees: the board, their hand / next card / elixir, the cards the opponent
has played and the crowns, not the opponent's hand or elixir. Every engine tick is pushed to the page over a
WebSocket (polling as fallback), so units walk, fly and fight as the engine moves them; the page sends its
plays on the same WebSocket.

In-game toggles:
  * Assist, two levels. Hint: the same policy, run on the PLAYER's observation, shows its top cards with their
    best tile and probability. Autopilot: that policy also PLAYS the player's side (its own random stream) on
    every beat (every 10 ticks) where the player has queued nothing; a play of the player's own always comes first, so the player
    can take over at any moment. A match with any assist on at any moment is "assisted" (not ranked).
  * Voice (in the page): reads Claude's commentary aloud with the browser's own speech synthesis (no cost).
  * Claude commentary and Claude win probability (owner only, see below): every --comment-every ticks the
    player's public view (pufferroyale.llm.render_state, which never contains the opponent's hand or elixir)
    and the plays since the last call go to Claude through the Claude Code CLI relay (hpc/llm_relay.py,
    running on the host with the owner's own login: no API key, no per-call billing); the reply is one or two
    sentences of dramatic, playful roast commentary and the probability that the player wins, drawn as a line.
    Because the relay uses the owner's personal subscription, only a session opened with the owner key
    (?owner=...) can switch these on, at most --claude-daily-calls calls per day.

Records (--data-dir): players.json (name, PIN hash, saved deck and settings per player id) and matches/<id>.json per
match: the result, the settings and toggles, and a replay record (seed, decks, every applied play of both
sides with its tick, and the engine state hash every 600 ticks and at the end). The engine is deterministic,
so the "Watch" button re-simulates a recorded match exactly (the page checks the final hash) and can save it
as a video. The leaderboard ranks players by their score against the policy over RANKED matches: finished,
at 1x speed throughout, assist never on.

Every request must carry the site token (?token=... in the shared link, or the X-Token header); requests
without it get 403.

Owner maintenance (applied once when the server starts, then renamed to admin.done.<time>.json and logged in
merge_log.jsonl): <data-dir>/admin.json = {"drop": [names], "reset_pin": [names]}. "drop" removes a player and
moves its match records to <data-dir>/removed/; "reset_pin" clears a PIN (the player gets a new one, shown once).
"""
import argparse
import base64
import copy
import glob
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import signal
import socket
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

TICKS_PER_SECOND = 20          # SPEC §4: 180 s = 3600 ticks
FRAME_SKIP = 10                # the env cadence the policies were trained with (both seats act every 10 ticks)
REGULATION_TICKS = 3600
MAX_TICKS = 6000
BOARD_W, BOARD_H = 18, 32      # tiles; 1 tile = 1000 engine units
ELIXIR_UNIT = 2800
MAX_QUEUED = 3                 # a player's plays waiting to go in (one per tick)
SPEEDS = (0.25, 0.5, 0.75, 1.0)
REPLAY_SPEEDS = (0.5, 1.0, 2.0, 4.0)
POOL_DECKS = ("hog26", "giant", "bait", "golem", "lavaloon", "miner_poison", "pekka_bridge")   # the training pool
HASH_EVERY = 600               # replay check points (ticks)
IDLE_ABANDON_S = 90            # a match with no page connected for this long is abandoned (frees the seat)
PAUSE_MAX_S = 120              # a pause longer than this resumes by itself
QUEUE_TICKET_S = 60            # a waiting player whose page is gone for this long leaves the line
MAX_LINE = 30                  # players waiting at most
MAX_STREAMS = 60               # open live connections (WebSockets, event streams) at most
WS_TIMEOUT_S = 20.0            # a WebSocket that neither sends nor takes data for this long is closed
WS_PING_S = 10.0               # the server pings an idle WebSocket this often (Cloudflare drops idle ones)
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"   # RFC 6455
VOICES = ("am_fenrir", "am_michael", "am_puck", "bm_george", "af_heart")   # play/tts_worker.py's Kokoro voices
MAX_SPECTATORS = 20            # people watching one live match at most
MAX_SPECTATE = 60              # spectator sessions at most (all matches)
MAX_BODY = 16384               # bytes per POST body
HTML_FILE = os.path.join(ROOT, "play", "play_page.html")
NAME_BAD = re.compile(r"[^A-Za-z0-9 _\-.']")
HOST_RE = re.compile(r"^[A-Za-z0-9.\-]{1,253}(:[0-9]{1,5})?$")
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,40}$")
PIN_RE = re.compile(r"^[0-9]{4}$")
PIN_TRIES = 5                  # wrong PINs in a row before a name is locked
PIN_LOCK_S = 600               # how long the lock lasts (10,000 PINs at 5 per 10 min: about 14 days to try them all)
PIN_ITER = 100_000             # PBKDF2-SHA256 iterations of the stored PIN hash

INFER = threading.Lock()       # one policy forward at a time (torch threads are pinned; matches share the weights)


# ====================================================================================== helpers
def ws_accept(key):
    """The Sec-WebSocket-Accept value for a client's Sec-WebSocket-Key (RFC 6455 section 4.2.2)."""
    return base64.b64encode(hashlib.sha1((key.strip() + WS_GUID).encode()).digest()).decode()


def ws_frame(payload, opcode=1):
    """One unmasked server frame (FIN set): opcode 1 text, 8 close, 9 ping, 10 pong."""
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", 0x80 | opcode, n)
    elif n < 1 << 16:
        head = struct.pack("!BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
    return head + payload


def _read_exact(f, n):
    b = f.read(n) if n else b""
    if b is None or len(b) < n:
        raise EOFError("the WebSocket closed")
    return b


def ws_read(f, limit=MAX_BODY):
    """One client message from file `f` as (opcode, payload): continuation frames are joined, control frames
    (close 8, ping 9, pong 10) come back on their own. Client frames must be masked (RFC 6455 section 5.1)."""
    data, first = b"", None
    while True:
        b0, b1 = _read_exact(f, 2)
        op, n = b0 & 0x0F, b1 & 0x7F
        if not b1 & 0x80:
            raise ValueError("unmasked client frame")
        if n == 126:
            n = struct.unpack("!H", _read_exact(f, 2))[0]
        elif n == 127:
            n = struct.unpack("!Q", _read_exact(f, 8))[0]
        if len(data) + n > limit:
            raise ValueError("message too large")
        mask = _read_exact(f, 4)
        chunk = _read_exact(f, n)
        if n:
            chunk = (int.from_bytes(chunk, "big") ^ int.from_bytes((mask * (n // 4 + 1))[:n], "big")).to_bytes(n, "big")
        if op >= 8:
            return op, chunk
        first = op if first is None else first
        data += chunk
        if b0 & 0x80:
            return first, data

def to_frame(team, x, y):
    """Engine frame (team 0 at the bottom) <-> own frame of `team` (a 180-degree rotation for team 1;
    its own inverse). Checked against the engine: team 1's own tile (3, 20) spawns at (14500, 11500)."""
    return (x, y) if team == 0 else (BOARD_W * 1000 - x, BOARD_H * 1000 - y)


def terrain_rows(game, team):
    """The 32 x 18 terrain of the board in `team`'s own frame, from the engine's text board at a fresh
    state: '.' ground, ':' off-limits edge, '~' river, '=' bridge, 'T' a crown tower's footprint."""
    rows = []
    for line in game.ansi().splitlines():
        m = re.match(r"^\s*(\d+) (.{18})$", line)
        if m:
            rows.append("".join(ch if ch in ".:~=" else "T" if ch in "KPkp" else "." for ch in m.group(2)))
    if len(rows) != BOARD_H:
        raise RuntimeError(f"could not read the board from Game.ansi() ({len(rows)} rows)")
    if team == 1:
        rows = [r[::-1] for r in rows[::-1]]
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


DEFAULT_MODEL = os.path.join(ROOT, "models", "S4_A_s1_10B.pt")   # = S4_A_s1's model_610352.pt (10,000,007,168 steps)


def checkpoint_label(path):
    """'S4_A_s1 / model_610352.pt (10,000,007,168 steps)' when the run's league_state.json gives the batch size."""
    if os.path.abspath(path) == os.path.abspath(DEFAULT_MODEL):
        return "S4_A_s1 (10,000,007,168 steps)"
    run_dir, name = os.path.split(os.path.abspath(path))
    label = f"{os.path.basename(run_dir)} / {name}"
    m = re.match(r"^(?:model|snap)_(\d+)\.pt$", name)
    state = os.path.join(run_dir, "league_state.json")
    if m and os.path.isfile(state):
        try:
            batch = int(json.load(open(state))["args"]["batch_size"])
            label += f" ({int(m.group(1)) * batch:,} steps)"
        except (KeyError, ValueError, TypeError):
            pass
    return label


def clean_name(name):
    name = NAME_BAD.sub("", str(name or "")).strip()[:24]
    return re.sub(r"\s+", " ", name)


def deck_spec(spec, pr, allow_random=False):
    """A deck from the page: a preset name (the 7 training-pool decks; 'random' = one of them drawn per
    match when allowed) or a list of 8 distinct card ids. Returns (label, list of 8 ids or 'random')."""
    if isinstance(spec, str):
        if spec in POOL_DECKS:
            return spec, list(pr.DECKS[spec])
        if spec == "random" and allow_random:
            return "random", "random"
        raise ValueError(f"unknown deck {spec!r}")
    if isinstance(spec, (list, tuple)) and len(spec) == 8:
        ids = [int(c) for c in spec]
        if len(set(ids)) != 8 or not all(0 <= c < pr.N_CARDS for c in ids):
            raise ValueError("a deck is 8 different cards")
        return "custom", ids
    raise ValueError("a deck is a preset name or 8 card ids")


def name_key(name):
    """The login key of a name: the cleaned name, lower case ('  aditya ' and 'Aditya' are the same player)."""
    return clean_name(name).lower()


def public_id(pid):
    return hashlib.sha256(("pr-player:" + pid).encode()).hexdigest()[:10]


def pin_hash(pin, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", str(pin).encode(), bytes.fromhex(salt_hex), PIN_ITER).hex()


def make_pin_record(pin):
    salt = secrets.token_hex(16)
    return {"salt": salt, "hash": pin_hash(pin, salt), "set": time.strftime("%Y-%m-%dT%H:%M:%S")}


def pin_ok(rec, pin):
    if not (isinstance(rec, dict) and isinstance(pin, str) and PIN_RE.match(pin)):
        return False
    try:
        return hmac.compare_digest(pin_hash(pin, rec["salt"]), str(rec["hash"]))
    except (KeyError, ValueError, TypeError):
        return False


def public_profile(pl):
    """A player's profile as the page gets it: never the PIN record."""
    return {k: v for k, v in pl.items() if k != "pin"}


def public_summary(s, name):
    """A match summary without the player id (which works as the player's login)."""
    out = {k: v for k, v in s.items() if k != "player_id"}
    out["name"] = name
    out["pub"] = public_id(s["player_id"])
    return out


def atomic_json(path, obj):
    tmp = f"{path}.tmp{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def advise(policy, game, team, grid):
    """The policy's own choice for `team` right now: P(wait) and, per playable card, P(card) with its most
    likely tile (own frame) and that tile's share of the card's probability."""
    import torch
    from pufferroyale import decode_action
    with INFER, torch.no_grad():
        logits, _ = policy.forward_eval(torch.as_tensor(game.obs(team)[None, :]), {})
    p = torch.softmax(logits.float(), dim=-1)[0].numpy().astype(np.float64)
    n = (len(p) - 1) // 4
    per = p[1:].reshape(4, n)
    hand = game.state()["hand"][team]
    cards = []
    for s in range(4):
        pc = float(per[s].sum())
        if pc < 0.005:
            continue
        best = int(per[s].argmax())
        a = 1 + s * n + best
        if grid > 1:
            a = game.coarse_to_fine(team, a, grid)
        d = decode_action(a) if a else None
        if d is None:
            continue
        cards.append({"card": int(hand[s]), "p": round(pc, 4), "tx": d[1], "ty": d[2],
                      "p_tile": round(float(per[s, best]) / pc, 4)})
    cards.sort(key=lambda c: -c["p"])
    return {"wait": round(float(p[0]), 4), "cards": cards[:3]}


# ====================================================================================== records
class Store:
    """players.json and matches/<id>.json under one directory; the leaderboard is computed from an
    in-memory index of the match summaries (rebuilt from the files at start)."""

    def __init__(self, root):
        self.root = root
        self.mdir = os.path.join(root, "matches")
        os.makedirs(self.mdir, exist_ok=True)
        self.lock = threading.Lock()
        p = os.path.join(root, "players.json")
        self.players = json.load(open(p)) if os.path.isfile(p) else {}
        self.index = []
        for f in sorted(glob.glob(os.path.join(self.mdir, "*.json"))):
            try:
                self.index.append(json.load(open(f))["summary"])
            except (ValueError, KeyError, OSError):
                continue
        u = os.path.join(root, "claude_usage.json")
        self.usage = json.load(open(u)) if os.path.isfile(u) else {}
        self.dropped = set()            # player ids merged away or dropped on purpose (never taken back from disk)
        self.fails = {}                 # name key -> [wrong PINs in a row, locked until (time.time())]
        self._apply_admin()
        self._merge_duplicates()
        self.by_name = {name_key(v["name"]): k for k, v in self.players.items() if name_key(v.get("name") or "")}

    def _apply_admin(self):
        """The owner's one-off maintenance file <root>/admin.json: {"drop": [names], "reset_pin": [names]} (see the
        module docstring). Applied once, then renamed to admin.done.<time>.json; every change goes to merge_log.jsonl."""
        p = os.path.join(self.root, "admin.json")
        if not os.path.isfile(p):
            return
        try:
            req = json.load(open(p))
        except (ValueError, OSError) as e:
            print(f"[play] admin.json unreadable ({e}): ignored", flush=True)
            return
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        log = []
        for name in req.get("drop") or []:
            key = name_key(name)
            for pid in [q for q, pl in self.players.items() if key and name_key(pl.get("name") or "") == key]:
                moved = []
                rdir = os.path.join(self.root, "removed")
                os.makedirs(rdir, exist_ok=True)
                for f in sorted(glob.glob(os.path.join(self.mdir, "*.json"))):
                    try:
                        if json.load(open(f)).get("summary", {}).get("player_id") == pid:
                            os.replace(f, os.path.join(rdir, os.path.basename(f)))
                            moved.append(os.path.basename(f))
                    except (ValueError, OSError):
                        continue
                self.index = [s for s in self.index if s.get("player_id") != pid]
                del self.players[pid]
                self.dropped.add(pid)
                log.append({"time": now, "admin_drop": pid, "name": name, "matches_moved_to_removed": moved})
        for name in req.get("reset_pin") or []:
            key = name_key(name)
            for pid, pl in self.players.items():
                if key and name_key(pl.get("name") or "") == key and pl.pop("pin", None) is not None:
                    pl["pin_reset"] = now           # newer than the old PIN: a merge from disk will not bring it back
                    log.append({"time": now, "admin_reset_pin": pid, "name": pl.get("name")})
        with open(os.path.join(self.root, "merge_log.jsonl"), "a") as f:
            for e in log or [{"time": now, "admin": "nothing matched", "request": req}]:
                f.write(json.dumps(e) + "\n")
        os.replace(p, os.path.join(self.root, f"admin.done.{time.strftime('%Y%m%d_%H%M%S')}.json"))
        self._save_players()
        print(f"[play] admin.json applied: {len(log)} change(s)", flush=True)

    def _merge_duplicates(self):
        """Records made before names were logins (2026-11-06): several players with one name become the one with
        the most matches (then the latest visit); their matches are moved to it; players without a name and without
        matches are dropped. Every change is appended to merge_log.jsonl."""
        counts = {}
        for sm in self.index:
            counts[sm["player_id"]] = counts.get(sm["player_id"], 0) + 1
        groups = {}
        for pid, pl in self.players.items():
            groups.setdefault(name_key(pl.get("name") or ""), []).append(pid)
        remap, drop = {}, []
        for key, pids in groups.items():
            if not key:
                drop += [p for p in pids if not counts.get(p)]
                continue
            if len(pids) > 1:
                keep = max(pids, key=lambda q: (counts.get(q, 0), self.players[q].get("last_seen") or ""))
                remap.update({q: keep for q in pids if q != keep})
        if not remap and not drop:
            return
        for f in glob.glob(os.path.join(self.mdir, "*.json")):
            try:
                rec = json.load(open(f))
            except (ValueError, OSError):
                continue
            if rec.get("summary", {}).get("player_id") in remap:
                rec["summary"]["player_id"] = remap[rec["summary"]["player_id"]]
                atomic_json(f, rec)
        for sm in self.index:
            sm["player_id"] = remap.get(sm["player_id"], sm["player_id"])
        with open(os.path.join(self.root, "merge_log.jsonl"), "a") as log:
            for old, new in remap.items():
                log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "merged": old, "into": new,
                                      "name": self.players[new].get("name"), "matches_moved": counts.get(old, 0)}) + "\n")
            for q in drop:
                log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "dropped_unnamed": q}) + "\n")
        for q in list(remap) + drop:
            del self.players[q]
        self.dropped.update(remap, drop)
        self._save_players()

    def _save_players(self):
        """Write players.json. Two servers can share the data for a while (a new site job starts before the old one
        ends), so players that only the file has (made by the other server) are taken in first, not overwritten."""
        p = os.path.join(self.root, "players.json")
        try:
            disk = json.load(open(p)) if os.path.isfile(p) else {}
        except (ValueError, OSError):
            disk = {}
        by_name = getattr(self, "by_name", None)
        for pid, pl in (disk.items() if isinstance(disk, dict) else ()):
            if pid in self.dropped or not isinstance(pl, dict):
                continue
            if pid not in self.players:
                self.players[pid] = pl
                k = name_key(pl.get("name") or "")
                if by_name is not None and k and k not in by_name:
                    by_name[k] = pid
            else:
                self._take_newer_pin(self.players[pid], pl)
        atomic_json(p, self.players)

    @staticmethod
    def _take_newer_pin(mine, disk):
        """PINs set by the other server of an overlap (or before its reset) are merged by time: the newest PIN or
        PIN reset wins, so neither server wipes a PIN the other one set."""
        def when(pl):
            return max((pl.get("pin") or {}).get("set") or "", pl.get("pin_reset") or "")
        if when(disk) > when(mine):
            for k in ("pin", "pin_reset"):
                if k in disk:
                    mine[k] = disk[k]
                else:
                    mine.pop(k, None)

    def _refresh_from_disk(self, pid):
        p = os.path.join(self.root, "players.json")
        try:
            disk = json.load(open(p))
        except (ValueError, OSError):
            return
        if isinstance(disk, dict) and isinstance(disk.get(pid), dict):
            self._take_newer_pin(self.players[pid], disk[pid])

    def _check_pin(self, key, pl, pin):
        """Raise ValueError unless pin is this player's PIN; counts wrong PINs per name and locks the name."""
        f = self.fails.setdefault(key, [0, 0.0])
        wait = f[1] - time.time()
        if wait > 0:
            raise ValueError(f"too many wrong PINs for this name: try again in {max(1, round(wait / 60))} min")
        if pin in (None, ""):
            raise ValueError("this name has a PIN: enter it")
        if pin_ok(pl.get("pin"), str(pin)):
            self.fails.pop(key, None)
            return
        f[0] += 1
        if f[0] >= PIN_TRIES:
            f[0], f[1] = 0, time.time() + PIN_LOCK_S
            raise ValueError(f"wrong PIN: this name is locked for {PIN_LOCK_S // 60} min")
        raise ValueError(f"wrong PIN ({PIN_TRIES - f[0]} {'try' if PIN_TRIES - f[0] == 1 else 'tries'} left)")

    def hello(self, pid, name=None, pin=None):
        """(player id or None, profile, created, new PIN or None). With a name: name + PIN is the login. A known
        name needs its PIN; a new name needs a PIN of 4 digits and makes a new player. Without a name: the browser's
        stored id (itself a long secret), when it is known; nobody is created until a name is given. A player
        without a PIN (made before PINs, or reset by the owner) gets a random one, returned ONCE as the new PIN."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock:
            key = name_key(name) if name is not None else ""
            created = False
            if key:
                if key in self.by_name:
                    pid = self.by_name[key]
                    self._refresh_from_disk(pid)
                    if self.players[pid].get("pin"):
                        self._check_pin(key, self.players[pid], pin)
                else:
                    if not (isinstance(pin, str) and PIN_RE.match(pin)):
                        raise ValueError("new name: choose a PIN of 4 digits (you need it to sign in again)")
                    pid, created = secrets.token_urlsafe(12), True
                    self.players[pid] = {"name": clean_name(name), "created": now, "pin": make_pin_record(pin)}
                    self.by_name[key] = pid
            elif not (isinstance(pid, str) and ID_RE.match(pid) and pid in self.players):
                return None, {}, False, None
            else:
                self._refresh_from_disk(pid)
            pl = self.players[pid]
            new_pin = None
            if not pl.get("pin"):
                new_pin = f"{secrets.randbelow(10000):04d}"
                pl["pin"] = make_pin_record(new_pin)
            pl["last_seen"] = now
            self._save_players()
            return pid, public_profile(pl), created, new_pin

    def set_pin(self, pid, pin):
        """A signed-in player (the page holds its secret player id) chooses a new PIN."""
        if not (isinstance(pin, str) and PIN_RE.match(pin)):
            raise ValueError("a PIN is 4 digits")
        with self.lock:
            if pid not in self.players:
                raise ValueError("unknown player: reload the page")
            self.players[pid]["pin"] = make_pin_record(pin)
            self.players[pid].pop("pin_reset", None)
            self._save_players()
            return True

    def update(self, pid, **fields):
        with self.lock:
            if pid not in self.players:
                raise ValueError("unknown player")
            self.players[pid].update(fields)
            self._save_players()
            return public_profile(self.players[pid])

    def save_match(self, rec):
        path = os.path.join(self.mdir, rec["summary"]["id"] + ".json")
        atomic_json(path, rec)
        with self.lock:
            self.index.append(rec["summary"])
        return path

    def load_match(self, mid):
        if not (isinstance(mid, str) and re.match(r"^[0-9]{8}_[0-9]{6}_[A-Za-z0-9_\-]{6,16}$", mid)):
            raise ValueError("bad match id")
        path = os.path.join(self.mdir, mid + ".json")
        if not os.path.isfile(path):
            raise ValueError("no such match")
        return json.load(open(path))

    def leaderboard(self, recent=15):
        with self.lock:
            idx = list(self.index)
            names = {k: v.get("name") or "?" for k, v in self.players.items()}
        rows = {}
        tot = {"n": 0, "w": 0, "d": 0, "l": 0}
        for s in idx:
            if not s.get("ranked"):
                continue
            r = rows.setdefault(s["player_id"], {"player_id": s["player_id"], "n": 0, "w": 0, "d": 0, "l": 0,
                                                 "best_crowns": 0})
            r["n"] += 1
            k = "w" if s["result"] == 1 else "d" if s["result"] == 0 else "l"
            r[k] += 1
            tot["n"] += 1
            tot[k] += 1
            r["best_crowns"] = max(r["best_crowns"], s["crowns"][0])
        out = []
        for r in rows.values():
            r["name"] = names.get(r["player_id"], "?")
            r["score"] = round((r["w"] + 0.5 * r["d"]) / r["n"], 4)
            r["pub"] = public_id(r.pop("player_id"))       # the player id is a secret: it is the login
            out.append(r)
        out.sort(key=lambda r: (-r["score"], -r["n"], r["name"]))
        rec = [public_summary(s, names.get(s["player_id"], s.get("name", "?"))) for s in idx[-recent:]][::-1]
        return {"rows": out, "recent": rec, "totals": tot}

    def history(self, pid, n=20):
        with self.lock:
            return [public_summary(s, s.get("name", "?")) for s in self.index if s["player_id"] == pid][-n:][::-1]

    def claude_take(self, cap):
        """Count one Claude call against today's cap; False when the cap is reached."""
        day = time.strftime("%Y-%m-%d")
        with self.lock:
            used = int(self.usage.get(day, 0))
            if used >= cap:
                return False
            self.usage = {day: used + 1}
            atomic_json(os.path.join(self.root, "claude_usage.json"), self.usage)
            return True


# ====================================================================================== the board
class Board:
    """An engine game drawn from one team's side (`ht`, always at the bottom of the page). All engine access
    happens under `lock`; the latest view is published as JSON bytes in `frame` (with `cond` notified)."""

    kind = "board"

    def __init__(self, pr, game, ht):
        self.pr, self.game = pr, game
        self.ht, self.mt = int(ht), 1 - int(ht)
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.version = 0
        self.frame = b"{}"
        self.events = []
        self.model_seen = []
        self.done = False
        self.spectators = 0             # changed under the server lock only
        self.tick_now, self.crowns_now = 0, [0, 0]
        # where each crown tower stands, in the player's frame (fixed; destroyed towers vanish from entities())
        self.tower_xy = {(e["team"], e["tower_idx"]): to_frame(self.ht, e["x"], e["y"])
                         for e in game.entities() if e["kind"] == "tower"}

    def _event(self, who, card, slot, tx, ty, err, tick):
        team = self.ht if who == "human" else self.mt
        ex, ey = to_frame(team, tx * 1000 + 500, ty * 1000 + 500)       # tile centre, engine frame
        hx, hy = to_frame(self.ht, ex, ey)                              # ... in the player's frame
        ev = {"tick": int(tick), "who": who, "team": team, "card": int(card), "name": self.pr.CARD_NAMES[card],
              "slot": int(slot), "tile_own": [int(tx), int(ty)], "xy_human_frame": [hx, hy],
              "result": self.pr.PlayError(int(err)).name}
        self.events.append(ev)
        if who == "model" and ev["result"] == "OK" and card not in self.model_seen:
            self.model_seen.append(int(card))
        return ev

    def view(self, st):
        g, ht, mt = self.game, self.ht, self.mt
        ents = []
        for e in g.entities():
            x, y = to_frame(ht, e["x"], e["y"])
            ents.append([e["id"], int(e["team"] == ht), e["unit"], e["card_id"], e["kind"], x, y, e["hp"], e["max_hp"],
                         e["shield"], e["radius"], int(e["flying"]), int(e["deploying"]), int(e["stunned"]),
                         int(e["slowed"]), int(e["hidden"]), e["target_id"], int(e["attacking"]),
                         int(e.get("burrowing", False)), e["deploy_ms"], e.get("tower_idx", -1)])
        projs = []
        for p in g.projectiles():
            x, y = to_frame(ht, p["x"], p["y"])
            tx, ty = to_frame(ht, p["target_x"], p["target_y"])
            projs.append([p["id"], int(p["team"] == ht), p["name"], p["card_id"], x, y, tx, ty, p["radius"]])
        effs = []
        for f in g.effects():
            x, y = to_frame(ht, f["x"], f["y"])
            effs.append([f["id"], int(f["team"] == ht), f["name"], f["card_id"], f["kind"], x, y, f["radius"],
                         f.get("radius_y", 0), f.get("remaining", 0)])
        towers = []
        for side, team in (("me", ht), ("foe", mt)):
            for i, tw in enumerate(st["towers"][team]):
                x, y = self.tower_xy.get((team, i), (0, 0))
                towers.append({"side": side, "idx": i, "hp": tw["hp"], "max_hp": tw["max_hp"],
                               "alive": tw["alive"], "active": tw["active"], "id": tw["id"], "x": x, "y": y})
        tick = int(st["tick"])
        return {
            "kind": self.kind, "seat": self.ht, "tick": tick, "over": bool(st["over"]),
            "result": st["result"][ht] if st["over"] else None, "end_reason": st["end_reason"],
            "overtime": bool(st["overtime"]), "elixir_rate": st["elixir_rate"],
            "elixir": st["elixir"][ht] / ELIXIR_UNIT, "hand": [int(c) for c in st["hand"][ht]],
            "next": int(st["queue"][ht][0]), "crowns": [st["crowns"][ht], st["crowns"][mt]],
            "towers": towers, "ents": ents, "projs": projs, "effs": effs,
            "lockout": max(0, int(st["lockout_ticks"]) - tick),
            "events": [e for e in self.events[-8:] if e["tick"] >= tick - 100],   # recent plays only (frames go out
            "model_seen": self.model_seen,                                       # every tick: keep them small)
            "model_last": [int(c) for c in st["last_played"][mt] if c >= 0],
        }

    def publish(self, st=None):
        with self.cond:
            st = st or self.game.state()
            self.tick_now, self.crowns_now = int(st["tick"]), [st["crowns"][self.ht], st["crowns"][self.mt]]
            v = self.view(st)
            v.update(self.extra(st))
            self.version += 1
            v["v"] = self.version
            self.frame = json.dumps(v, separators=(",", ":")).encode()
            self.cond.notify_all()

    def extra(self, st):
        return {}

    def wake(self):
        with self.cond:
            self.version += 1
            self.cond.notify_all()


class LiveMatch(Board):
    """A player against the policy."""

    kind = "match"

    def __init__(self, server, session, params):
        import torch
        pr = server.pr
        self.server = server
        self.session = session
        self.params = params
        ht = int(params["seat"])
        self.seed = secrets.randbelow(2 ** 31)
        self.deck_labels = {"human": params["deck_label"], "model": params["opp_label"]}
        human_ids, model_ids = params["deck_ids"], params["opp_ids"]
        if model_ids == "random":
            name = POOL_DECKS[secrets.randbelow(len(POOL_DECKS))]
            self.deck_labels["model"] = f"random: {name}"
            model_ids = list(pr.DECKS[name])
        decks = {ht: human_ids, 1 - ht: model_ids}
        game = pr.Game(deck0=list(decks[0]), deck1=list(decks[1]), seed=self.seed)   # training defaults otherwise
        super().__init__(pr, game, ht)
        st = game.state()
        self.deck_ids = {"human": list(st["deck"][self.ht]), "model": list(st["deck"][self.mt])}
        self.player = copy.copy(server.player)          # shares the weights; own generator and state
        self.player.gen = torch.Generator().manual_seed((self.seed * 2 + self.mt) & ((1 << 62) - 1))
        self.player.state = {"lstm_h": None, "lstm_c": None}
        self.speed = float(params["speed"])
        self.min_speed = self.speed
        self.paused = False
        self.pause_t0 = None
        self.paused_s = 0.0
        self.queue = []
        self.plays = []                 # applied plays: [tick, team, slot, tx, ty, card] (the replay record)
        self.hashes = {}
        self.voice = VOICES[0]          # the neural voice for this match's commentary (the owner picks it)
        self.assist = 0                 # 0 off, 1 hint (the policy's pick is shown), 2 autopilot (it plays for you)
        self.assist_max = 0
        self.copilot = copy.copy(server.player)          # the policy on the player's side (hint and autopilot)
        self.copilot.gen = torch.Generator().manual_seed((self.seed * 2 + self.ht + 7919) & ((1 << 62) - 1))
        self.copilot.state = {"lstm_h": None, "lstm_c": None}
        self.advice = None
        self.claude = {"commentary": False, "winprob": False}
        self.comments = []              # [{tick, text}]
        self.winprob = []               # [[tick, p]]
        self.claude_pending = None      # (n, tick, path)
        self.claude_n = 0
        self.claude_last_tick = -10 ** 9
        self.claude_note = None
        self.last_client = time.time()
        self.started = time.time()
        self.finished = None
        self.abandoned = False
        self.id = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.started)) + "_" + secrets.token_urlsafe(6)
        self.decision_ms = []
        self.client_rtt = []            # round trips the page measured (ms), for the record
        self.transports = set()         # how the player's page was connected: "ws", "poll", "stream"
        self.publish()

    # ---------------------------------------------------------------- one tick
    def step(self):
        """One engine tick. The policy acts on its training cadence (every FRAME_SKIP ticks, from its own
        observation). The player's oldest queued play goes in at the very next tick, at most one per tick, so a
        tap waits at most one tick (50 ms at 1x) instead of up to a whole beat (500 ms). Every applied play is
        recorded with its tick, so replays stay exact."""
        g, pr = self.game, self.pr
        t = g.tick_count
        beat = t % FRAME_SKIP == 0
        st = None
        if beat:
            st = g.state()
            t0 = time.perf_counter()
            with INFER:
                a = self.player.act(g.obs(self.mt))
            a = self.player.fine(g, self.mt, a)
            self.decision_ms.append((time.perf_counter() - t0) * 1000.0)
            if a:
                slot, tx, ty = pr.decode_action(a)
                card = int(st["hand"][self.mt][slot])
                err = g.play_tile(self.mt, slot, tx, ty)
                self._event("model", card, slot, tx, ty, err, t)
                if err == pr.PlayError.OK:
                    self.plays.append([t, self.mt, slot, tx, ty, card])
        if self.queue:
            st = st or g.state()                         # the policy's play does not touch the player's hand
            q = self.queue.pop(0)
            hand = list(st["hand"][self.ht])
            if q["card"] in hand:
                slot = hand.index(q["card"])
                err = g.play_tile(self.ht, slot, q["tx"], q["ty"])
            else:
                slot, err = -1, pr.PlayError.NOT_IN_HAND
            ev = self._event("human", q["card"], slot, q["tx"], q["ty"], err, t)
            ev["waited_ms"] = round((time.time() - q["t"]) * 1000, 1)
            if q.get("pid") is not None:
                ev["pid"] = q["pid"]                     # the page's id for this play (it clears its placeholder)
            if err == pr.PlayError.OK:
                self.plays.append([t, self.ht, slot, q["tx"], q["ty"], q["card"]])
        elif beat:
            if self.assist == 2:                         # autopilot: a play of the player's own always comes first
                with INFER:
                    a = self.copilot.act(g.obs(self.ht))
                a = self.copilot.fine(g, self.ht, a)
                if a:
                    slot, tx, ty = pr.decode_action(a)
                    card = int(st["hand"][self.ht][slot])
                    err = g.play_tile(self.ht, slot, tx, ty)
                    ev = self._event("human", card, slot, tx, ty, err, t)
                    ev["auto"] = True
                    if err == pr.PlayError.OK:
                        self.plays.append([t, self.ht, slot, tx, ty, card])
        g.tick(1)
        t1 = g.tick_count
        if t1 % HASH_EVERY == 0:
            self.hashes[str(t1)] = int(g.hash())
        st = g.state()
        if self.assist >= 1 and t1 % FRAME_SKIP == 0 and not st["over"]:
            self.advice = advise(self.player.policy, g, self.ht, self.player.grid)
        if (self.claude["commentary"] or self.claude["winprob"]) and not st["over"]:
            self._claude(st)
        if st["over"] and not self.done:
            self.done = True
            self.finished = time.time()
        self.publish(st)

    # ---------------------------------------------------------------- Claude (through the host relay)
    def _claude(self, st):
        srv = self.server
        if self.claude_pending is not None:
            n, tick, path = self.claude_pending
            if os.path.isfile(path):
                try:
                    reply = json.load(open(path))
                except (ValueError, OSError):
                    return
                self.claude_pending = None
                parsed = parse_claude_reply(reply.get("text", ""))
                if parsed is None:
                    self.claude_note = "Claude's reply could not be read"
                    return
                text, p = parsed
                self.claude_note = None
                if text:
                    c = {"tick": tick, "text": text}
                    if srv.voice_on():                       # the neural voice (play/tts_worker.py) is running
                        c["audio"] = f"{self.id}_{n:04d}"
                        srv.request_voice(c["audio"], text, self.voice)
                    self.comments.append(c)
                if p is not None:
                    self.winprob.append([tick, p])
            elif time.time() - self.claude_req_t > srv.args.claude_timeout:
                self.claude_pending = None
                self.claude_note = "Claude did not answer in time"
            return
        if st["tick"] - self.claude_last_tick < srv.args.comment_every or st["tick"] < 60:
            return
        if not srv.store.claude_take(srv.args.claude_daily_calls):
            self.claude_note = "today's Claude calls are used up"
            self.claude = {"commentary": False, "winprob": False}
            return
        self.claude_n += 1
        d = os.path.join(srv.args.claude_relay_dir, "g" + self.id)
        os.makedirs(d, exist_ok=True)
        turn = os.path.join(d, f"turn_{self.claude_n:04d}.txt")
        with open(turn + ".tmp", "w") as f:
            f.write(claude_turn_text(self, st))
        os.replace(turn + ".tmp", turn)
        self.claude_pending = (self.claude_n, int(st["tick"]), os.path.join(d, f"reply_{self.claude_n:04d}.json"))
        self.claude_req_t = time.time()
        self.claude_last_tick = int(st["tick"])

    # ---------------------------------------------------------------- the page's view
    def extra(self, st):
        return {
            "paused": self.paused, "speed": self.speed,
            "legal": base64.b64encode(np.packbits(np.asarray(self.game.legal_mask(self.ht), dtype=np.uint8)[1:]).tobytes()).decode(),
            "queued": [dict(card=q["card"], tx=q["tx"], ty=q["ty"], pid=q.get("pid")) for q in self.queue],
            "voice": self.voice, "advisor": self.assist >= 1, "assist": self.assist, "advice": self.advice if self.assist >= 1 else None,
            "claude": dict(self.claude), "comments": self.server.voice_marks(self.comments[-4:]), "winprob": self.winprob,
            "claude_wait": self.claude_pending is not None, "claude_note": self.claude_note,
            "ranked": self.ranked(), "match_id": self.id, "saved": self.done and self.finished is not None,
            "max_ticks": MAX_TICKS, "player_name": self.session.name, "spectators": self.spectators,
            "abandoned": bool(self.abandoned),
        }

    def view(self, st):
        """Like a real match: until it ends the page gets nothing about the policy's deck -- not the list of
        its cards seen so far, not its last plays, not its deck's name (its hand and elixir are never sent).
        Its plays still show on the board as they happen. After the end: the deck and its name."""
        v = super().view(st)
        if st["over"] or self.done:
            v["model_deck"] = self.deck_ids["model"]
            v["deck_labels"] = dict(self.deck_labels)
        else:
            v["model_seen"], v["model_last"] = [], []
        return v

    def note_client(self, transport, rtt=None):
        """The player's page reported in: its connection kind and, when it has one, its measured round trip."""
        self.transports.add(transport)
        if isinstance(rtt, (int, float)) and 0 < rtt < 60000 and len(self.client_rtt) < 5000:
            self.client_rtt.append(float(rtt))

    def ranked(self):
        return self.assist_max == 0 and self.min_speed == 1.0 and not self.abandoned

    # ---------------------------------------------------------------- the player's input
    def queue_play(self, card, tx, ty, pid=None):
        """Check a play against the current state and queue it for the next tick (step applies one per tick).
        A play behind other queued plays is checked against what they will leave: elixir minus their costs,
        and a card can be queued once. `pid` is the page's id for the play (echoed in its event)."""
        pr = self.pr
        with self.lock:
            if self.done:
                return False, "the match is over"
            if not (0 <= tx < BOARD_W and 0 <= ty < BOARD_H):
                return False, "outside the board"
            st = self.game.state()
            hand = list(st["hand"][self.ht])
            if card not in hand:
                return False, "that card is not in your hand"
            if any(q["card"] == card for q in self.queue):
                return False, "that card is already on its way"
            if st["tick"] < st["lockout_ticks"]:
                return False, "deploy lockout (the first 4.5 s)"
            spent = sum(pr.CARD_COSTS[q["card"]] for q in self.queue)
            if st["elixir"][self.ht] < (pr.CARD_COSTS[card] + spent) * ELIXIR_UNIT:
                return False, "not enough elixir"
            mask = self.game.legal_mask(self.ht)
            if not mask[1 + hand.index(card) * pr.N_TILES + ty * BOARD_W + tx]:
                return False, "you cannot place that card there"
            if len(self.queue) >= MAX_QUEUED:
                return False, "wait: plays are already queued"
            q = {"card": int(card), "tx": int(tx), "ty": int(ty), "t": time.time()}
            if pid is not None:
                q["pid"] = str(pid)[:32]
            self.queue.append(q)
            self.publish()
            return True, "queued for the next tick"

    def control(self, body, owner):
        with self.lock:
            if "paused" in body:
                want = bool(body["paused"])
                if want and not self.paused:
                    self.pause_t0 = time.time()
                if not want and self.paused and self.pause_t0:
                    self.paused_s += time.time() - self.pause_t0
                self.paused = want
            if "speed" in body:
                sp = float(body["speed"])
                if sp not in SPEEDS:
                    raise ValueError(f"speed must be one of {SPEEDS}")
                self.speed = sp
                self.min_speed = min(self.min_speed, sp)
            if "advisor" in body or "assist" in body:    # "advisor" (true / false) = assist level 1 / 0
                level = int(body["assist"]) if "assist" in body else (1 if body["advisor"] else 0)
                if level not in (0, 1, 2):
                    raise ValueError("assist must be 0 (off), 1 (hint) or 2 (autopilot)")
                self.assist = level
                self.assist_max = max(self.assist_max, level)
                if level >= 1 and not self.done:
                    self.advice = advise(self.player.policy, self.game, self.ht, self.player.grid)
            if "voice" in body:
                if body["voice"] not in VOICES:
                    raise ValueError(f"voice must be one of {VOICES}")
                self.voice = body["voice"]
            for k in ("commentary", "winprob"):
                if k in body:
                    if body[k] and not (owner and self.server.claude_on):
                        raise ValueError("the Claude toggles are available to the owner only")
                    self.claude[k] = bool(body[k])
            self.publish()

    def record(self):
        st = self.game.state()
        d = np.asarray(self.decision_ms) if self.decision_ms else np.zeros(1)
        srv = self.server
        summary = {
            "id": self.id, "player_id": self.session.player_id, "name": self.session.name,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.finished)) if self.finished else None,
            "result": st["result"][self.ht] if st["over"] else None, "end_reason": st["end_reason"],
            "crowns": [st["crowns"][self.ht], st["crowns"][self.mt]], "ticks": int(st["tick"]),
            "abandoned": self.abandoned, "ranked": self.ranked() and bool(st["over"]),
            "advisor_used": self.assist_max >= 1, "autopilot_used": self.assist_max >= 2, "assist_max": self.assist_max,
            "min_speed": self.min_speed, "seat": self.ht,
            "decks": dict(self.deck_labels), "claude_used": bool(self.comments or self.winprob),
        }
        return {
            "summary": summary,
            "opponent": {"checkpoint": os.path.abspath(srv.args.checkpoint), "sha256": srv.checkpoint_sha,
                         "label": srv.label, "greedy": bool(srv.args.greedy)},
            "settings": {"frame_skip": FRAME_SKIP, "deploy_lockout_ticks": self.game.deploy_lockout_ticks,
                         "tiebreak": self.game.tiebreak, "tower_troops": list(self.game.tower_troops),
                         "paused_s": round(self.paused_s, 1)},
            "replay": {"seed": self.seed, "human_team": self.ht, "deck0": self.deck_ids["human"] if self.ht == 0
                       else self.deck_ids["model"], "deck1": self.deck_ids["model"] if self.ht == 0
                       else self.deck_ids["human"], "plays": self.plays, "hashes": self.hashes,
                       "final_tick": int(st["tick"]), "final_hash": int(self.game.hash())},
            "events": self.events, "comments": self.comments, "winprob": self.winprob,
            "tower_hp": {"human": [t["hp"] for t in st["towers"][self.ht]],
                         "model": [t["hp"] for t in st["towers"][self.mt]]},
            "policy_decision_ms": {"n": len(self.decision_ms), "mean": float(d.mean()),
                                   "p99": float(np.percentile(d, 99))},
            "client": {"transports": sorted(self.transports), "rtt_ms": {
                "n": len(self.client_rtt),
                "median": float(np.median(self.client_rtt)) if self.client_rtt else None,
                "p90": float(np.percentile(self.client_rtt, 90)) if self.client_rtt else None}},
        }


CLAUDE_SYSTEM = """You are the live commentator of a match in PufferRoyale, a research simulator of a two-player real-time card \
strategy game. A human player ("you", always at the bottom of the board) plays against an AI policy trained by self-play \
("the AI"). Every few seconds you get the human's view of the match: exactly what the human can see (their own hand and \
elixir, the board, the AI's played cards; never the AI's hand or elixir), plus the plays since your last update and your \
own previous lines.

Your style: a DRAMATIC, over-the-top arena announcer in a fun ROAST mood. Write ONE punchy, COMPLETE sentence of AT \
MOST 10 WORDS -- count them; anything longer is cut off mid-sentence (it is read aloud, and the next update comes \
about 10 seconds later), addressed to the human: what just happened, told \
with theatrical drama, with a playful roast of whoever earned it -- the human's misplays, the AI's cold robotic \
ruthlessness, or both. Mock outrage, gasps, ridiculous comparisons. Keep it good-natured: tease the plays, never the \
person; no profanity, slurs or anything about real-world traits. NO NUMBERS of any kind: never say hit points, levels, \
elixir amounts, times or counts, in digits or in words; name cards, lanes and towers instead. Never repeat your \
previous lines; do not give orders. The line is read aloud, so no emoji, symbols or stage directions. As in a real match, \
never state or estimate the AI's elixir, and never list the AI's deck or the cards it may still hold; talk only about \
what is on the board and what was just played. Then estimate \
the probability that the HUMAN wins the match (a draw is not a win), as a number between 0 and 1, from the crowns, tower \
hit points, elixir, the clock and the units on the board.

Your reply must end with one line of JSON and nothing after it:
{"commentary": "<your line, at most 10 words, no numbers>", "p_win": <probability between 0 and 1>}

The reference below describes the game's rules and cards. It was written for a player; ignore its instructions about \
how to reply with plays.

"""


def claude_turn_text(m, st):
    from pufferroyale import llm
    hidden = ("Reply with", "OPPONENT CARDS SEEN", "OPPONENT LAST PLAYED", "OPPONENT ELIXIR")   # not shown to a player
    lines = [ln for ln in llm.render_state(m.game, m.ht, legal=False).splitlines() if not ln.startswith(hidden)]
    since = [e for e in m.events if e["tick"] > m.claude_last_tick and e["result"] == "OK"]
    plays = "; ".join(f"{'you' if e['who'] == 'human' else 'the AI'} played {e['name']} at ({e['tile_own'][0]},"
                      f"{e['tile_own'][1]}) at {fmt_clock(e['tick'])}" for e in since) or "none"
    prev = " | ".join(c["text"] for c in m.comments[-2:]) or "none yet"
    prob = f"{m.winprob[-1][1]:.2f}" if m.winprob else "none yet"
    return ("\n".join(lines) + f"\nPLAYS SINCE YOUR LAST UPDATE: {plays}\nYOUR PREVIOUS LINES: {prev}\n"
            f"YOUR PREVIOUS P_WIN: {prob}\nCoordinates of plays are in each side's own frame (tile x 0-17, y 0-31, "
            "y = 31 at that side's own back row).\n")


def fmt_clock(tick):
    reg = tick < REGULATION_TICKS
    left = max(0, ((REGULATION_TICKS if reg else MAX_TICKS) - tick) // TICKS_PER_SECOND)
    return ("" if reg else "overtime ") + f"{left // 60}:{left % 60:02d} left"


DANGLING = {"a", "an", "the", "its", "his", "her", "their", "your", "my", "our", "of", "to", "and", "or", "but", "with",
            "at", "in", "on", "for", "from", "by", "as", "is", "are", "was", "be", "that", "this", "than", "like"}


def clean_commentary(text):
    """The user's rule (2026-11-06): at most 10 words and no numbers (no hit points, levels, elixir, times).
    The prompt asks for it; this enforces it: words with digits are dropped, then the line is cut to 10 words.
    Number words are left alone ("one" is common in ordinary speech), the prompt covers them."""
    words = [w for w in re.sub(r"\s+", " ", text).strip().split(" ") if w and not re.search(r"\d", w)]
    if len(words) <= 10:
        return " ".join(words)[:200]
    head = words[:10]
    # cut at the last natural break inside the first 10 words (a sentence end or a comma), else at 10 words without
    # a dangling little word ("...sharpens its" -> "...sharpens"); always end on "!"
    breaks = [i for i, w in enumerate(head[:-1]) if re.search(r"[.!?,;:\u2014-]$", w) and i >= 2]
    if breaks:
        head = head[:breaks[-1] + 1]
    else:
        while len(head) > 3 and head[-1].lower().strip(",;:") in DANGLING:
            head = head[:-1]
    return (" ".join(head).rstrip(",;:\u2014- ").rstrip(".!?") + "!")[:200]


def parse_claude_reply(text):
    """(commentary, p_win or None) from the last JSON object in the reply; None when there is none. A relay
    failure ('WAIT') is None too."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text or ""))):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        c = d.get("commentary")
        c = clean_commentary(c) if isinstance(c, str) else ""
        p = d.get("p_win")
        try:
            p = float(p)
            p = min(1.0, max(0.0, p)) if math.isfinite(p) else None
        except (TypeError, ValueError):
            p = None
        if c or p is not None:
            return c, p
    return None


class Replay(Board):
    """A recorded match re-simulated from its seed and its applied plays (the engine is deterministic); the
    hashes every 600 ticks and at the end are compared as it goes."""

    kind = "replay"

    def __init__(self, server, rec):
        pr = server.pr
        rp = rec["replay"]
        game = pr.Game(deck0=list(rp["deck0"]), deck1=list(rp["deck1"]), seed=int(rp["seed"]),
                       deploy_lockout_ticks=int(rec["settings"]["deploy_lockout_ticks"]),
                       tiebreak=rec["settings"]["tiebreak"], tower_troop0=rec["settings"]["tower_troops"][0],
                       tower_troop1=rec["settings"]["tower_troops"][1])
        super().__init__(pr, game, rp["human_team"])
        self.server = server
        self.rec = rec
        self.by_tick = {}
        for t, team, slot, tx, ty, card in rp["plays"]:
            self.by_tick.setdefault(int(t), []).append((int(team), int(slot), int(tx), int(ty), int(card)))
        self.speed = 1.0
        self.paused = False
        self.mismatch = None
        self.verified = False
        self.final_tick = int(rp["final_tick"])
        self.last_client = time.time()
        self.publish()

    def step(self):
        g, pr = self.game, self.pr
        t = g.tick_count
        for team, slot, tx, ty, card in self.by_tick.get(t, ()):
            hand = g.state()["hand"][team]
            err = g.play_tile(team, slot, tx, ty) if hand[slot] == card else pr.PlayError.NOT_IN_HAND
            self._event("human" if team == self.ht else "model", card, slot, tx, ty, err, t)
            if err != pr.PlayError.OK and self.mismatch is None:
                self.mismatch = f"play at tick {t} did not apply ({pr.PlayError(int(err)).name})"
        g.tick(1)
        t1 = g.tick_count
        want = self.rec["replay"]["hashes"].get(str(t1))
        if want is not None and int(g.hash()) != int(want) and self.mismatch is None:
            self.mismatch = f"state differs from the recording at tick {t1}"
        st = g.state()
        if t1 >= self.final_tick or st["over"]:
            self.done = True
            self.verified = self.mismatch is None and int(g.hash()) == int(self.rec["replay"]["final_hash"])
        self.publish(st)

    def seek(self, tick):
        """Re-simulate from the start to `tick` (fast: no policy, no publishing in between)."""
        with self.lock:
            rp = self.rec["replay"]
            self.game.reset(seed=int(rp["seed"]))
            self.events, self.model_seen, self.done, self.mismatch, self.verified = [], [], False, None, False
            tick = max(0, min(int(tick), self.final_tick))
            while self.game.tick_count < tick and not self.done:
                g, t = self.game, self.game.tick_count
                for team, slot, tx, ty, card in self.by_tick.get(t, ()):
                    err = g.play_tile(team, slot, tx, ty)
                    self._event("human" if team == self.ht else "model", card, slot, tx, ty, err, t)
                g.tick(1)
                if g.tick_count >= self.final_tick or g.state()["over"]:
                    self.done = True
                    self.verified = int(g.hash()) == int(rp["final_hash"])
            self.publish()

    def extra(self, st):
        rec = self.rec
        t = int(st["tick"])
        return {"paused": self.paused, "speed": self.speed, "legal": "", "queued": [], "advisor": False, "assist": 0, "advice": None,
                "claude": {"commentary": bool(rec["comments"]), "winprob": bool(rec["winprob"])},
                "comments": self.server.voice_marks([c for c in rec["comments"] if c["tick"] <= t][-4:]),
                "winprob": [w for w in rec["winprob"] if w[0] <= t],
                "replay": {"final_tick": self.final_tick, "verified": self.verified, "mismatch": self.mismatch,
                           "summary": rec["summary"]},
                "match_id": rec["summary"]["id"], "deck_labels": rec["summary"]["decks"], "max_ticks": MAX_TICKS,
                "ranked": rec["summary"]["ranked"], "claude_wait": False, "claude_note": None}


# ====================================================================================== sessions
class Session:
    def __init__(self, player_id, name, owner, kind):
        self.sid = secrets.token_urlsafe(16)
        self.player_id, self.name, self.owner, self.kind = player_id, name, owner, kind
        self.status = "queued"
        self.board = None
        self.params = None
        self.last_seen = time.time()
        self.created = time.time()


class Server:
    def __init__(self, args, player=None):
        """`player` (tests): a ready PolicyPlayer-like object instead of loading --checkpoint."""
        import torch
        import pufferroyale as pr
        from watch import PolicyPlayer

        torch.set_num_threads(max(1, int(args.threads)))
        self.pr = pr
        self.args = args
        if player is None:
            self.player = PolicyPlayer(args.checkpoint, greedy=args.greedy, seed=0)
            self.checkpoint_sha = sha256_file(self.player.path)
            self.label = args.label or checkpoint_label(self.player.path)
        else:
            self.player, self.checkpoint_sha, self.label = player, "0" * 64, args.label or "test policy"
        self.store = Store(args.data_dir)
        self.claude_on = bool(args.claude_relay_dir)
        self.voice_dir = getattr(args, "voice_dir", None)
        if self.voice_dir:
            os.makedirs(os.path.join(self.voice_dir, "req"), exist_ok=True)
        if self.claude_on:
            os.makedirs(args.claude_relay_dir, exist_ok=True)
            with open(os.path.join(args.claude_relay_dir, "RULES_PROMPT.txt.tmp"), "w") as f:
                from pufferroyale import llm
                f.write(CLAUDE_SYSTEM + llm.RULES_PROMPT)
            os.replace(os.path.join(args.claude_relay_dir, "RULES_PROMPT.txt.tmp"),
                       os.path.join(args.claude_relay_dir, "RULES_PROMPT.txt"))
        g = pr.Game(seed=0)
        self.terrain = [terrain_rows(g, 0), terrain_rows(g, 1)]
        self.lock = threading.Lock()
        self.sessions = {}
        self.line = []                  # sids waiting for a seat
        self.streams = 0
        self.stop = threading.Event()
        threading.Thread(target=self._janitor, daemon=True).start()

    # ---------------------------------------------------------------- seats and the line
    def live_count(self):
        return sum(1 for s in self.sessions.values() if s.kind == "match" and s.status == "playing")

    def join(self, player_id, name, owner, params):
        with self.lock:
            if len(self.line) >= MAX_LINE:
                raise ValueError("the waiting line is full, please try again later")
            for s in self.sessions.values():          # one live match or ticket per player
                if s.player_id == player_id and s.kind == "match" and s.status in ("queued", "playing"):
                    self._end_session(s, abandoned=True)
            s = Session(player_id, name, owner, "match")
            s.params = params
            self.sessions[s.sid] = s
            self.line.append(s.sid)
            self._promote()
            return s

    def _promote(self):
        """Start matches for the first players in the line while seats are free (caller holds the lock)."""
        while self.line and self.live_count() < self.args.max_players:
            sid = self.line.pop(0)
            s = self.sessions.get(sid)
            if s is None or s.status != "queued":
                continue
            s.status = "playing"
            s.board = LiveMatch(self, s, s.params)
            threading.Thread(target=self._run_match, args=(s, s.board), daemon=True).start()
            print(f"[play] match {s.board.id} started for {s.name!r} ({self.live_count()}/{self.args.max_players} "
                  f"seats, {len(self.line)} waiting)", flush=True)

    def position(self, s):
        with self.lock:
            return self.line.index(s.sid) + 1 if s.sid in self.line else 0

    def _end_session(self, s, abandoned=False):
        """Leave the line, or end a live match (saved as abandoned when it was not over), or stop watching one.
        Caller holds the lock."""
        if s.kind == "spectate":
            if s.status != "ended" and s.board is not None:
                s.board.spectators = max(0, s.board.spectators - 1)
            s.status = "ended"
            return
        if s.sid in self.line:
            self.line.remove(s.sid)
        if s.kind == "match" and s.board is not None and not s.board.done:
            with s.board.lock:
                s.board.abandoned = abandoned
                s.board.done = True
                s.board.finished = time.time()
        s.status = "ended"

    def leave(self, s):
        with self.lock:
            self._end_session(s, abandoned=True)
            self._promote()

    def _run_match(self, s, m):
        nxt = time.monotonic()
        try:
            while not self.stop.is_set() and not m.done:
                if m.paused:
                    if m.pause_t0 and time.time() - m.pause_t0 > PAUSE_MAX_S:
                        m.control({"paused": False}, owner=False)
                    time.sleep(0.05)
                    nxt = time.monotonic()
                    continue
                with m.lock:
                    if not m.done:
                        m.step()
                nxt += 1.0 / (TICKS_PER_SECOND * m.speed)
                now = time.monotonic()
                if nxt > now:
                    time.sleep(nxt - now)
                elif now - nxt > 0.5:
                    nxt = now
        finally:
            with m.lock:
                rec = m.record()
                path = self.store.save_match(rec)
                m.publish()
            if self.claude_on:
                d = os.path.join(self.args.claude_relay_dir, "g" + m.id)
                for f in glob.glob(os.path.join(d, "*")):
                    os.remove(f)
                if os.path.isdir(d):
                    os.rmdir(d)
            sm = rec["summary"]
            print(f"[play] match {m.id} of {sm['name']!r}: result {sm['result']} {sm['crowns']} {sm['end_reason']}"
                  f"{' (abandoned)' if sm['abandoned'] else ''}{' ranked' if sm['ranked'] else ''} -> {path}", flush=True)
            with self.lock:
                if s.status == "playing":
                    s.status = "ended"
                self._promote()

    def watch(self, player_id, owner, mid):
        rec = self.store.load_match(mid)
        with self.lock:
            n = sum(1 for s in self.sessions.values() if s.kind == "replay" and s.status == "playing")
            if n >= self.args.max_replays:
                raise ValueError("too many replays are running, please try again in a minute")
            s = Session(player_id, "", owner, "replay")
            s.board = Replay(self, rec)
            s.status = "playing"
            self.sessions[s.sid] = s
        threading.Thread(target=self._run_replay, args=(s, s.board), daemon=True).start()
        return s

    def _run_replay(self, s, r):
        nxt = time.monotonic()
        while not self.stop.is_set() and s.status == "playing":
            if r.paused or r.done:
                time.sleep(0.05)
                nxt = time.monotonic()
                continue
            with r.lock:
                if not r.done:
                    r.step()
            nxt += 1.0 / (TICKS_PER_SECOND * r.speed)
            now = time.monotonic()
            if nxt > now:
                time.sleep(nxt - now)
            elif now - nxt > 0.5:
                nxt = now

    def _janitor(self):
        while not self.stop.is_set():
            time.sleep(2.0)
            self.sweep(time.time())

    def sweep(self, now):
        """Abandon matches whose player's page is gone, drop waiting players and spectators whose page is gone,
        forget old sessions. A spectator never keeps a match alive: only the player's own requests count."""
        with self.lock:
            for s in list(self.sessions.values()):
                if s.status == "queued" and now - s.last_seen > QUEUE_TICKET_S:
                    self._end_session(s)
                elif s.kind == "spectate" and s.status == "playing" and now - s.last_seen > QUEUE_TICKET_S:
                    self._end_session(s)
                elif s.kind != "spectate" and s.status == "playing" and s.board is not None \
                        and now - s.board.last_client > IDLE_ABANDON_S:
                    self._end_session(s, abandoned=True)
                if s.status == "ended" and now - s.last_seen > 600:
                    del self.sessions[s.sid]
            self._promote()

    def session(self, sid):
        s = self.sessions.get(sid) if isinstance(sid, str) else None
        if s is None:
            raise ValueError("unknown or expired session")
        s.last_seen = time.time()
        if s.board is not None and s.kind != "spectate":     # a spectator does not keep the player's match alive
            s.board.last_client = time.time()
        return s

    def spectate(self, owner, mid):
        """Watch a live match: the same frames as its player (their hand and elixir; the policy's cards, deck and
        elixir stay hidden until the end), read only, not a seat."""
        with self.lock:
            if sum(1 for x in self.sessions.values() if x.kind == "spectate" and x.status == "playing") >= MAX_SPECTATE:
                raise ValueError("too many people are watching right now, please try again later")
            for x in self.sessions.values():
                if x.kind == "match" and x.status == "playing" and x.board is not None and x.board.id == mid \
                        and not x.board.done:
                    if x.board.spectators >= MAX_SPECTATORS:
                        raise ValueError("this match has the most watchers already")
                    s = Session(None, "", owner, "spectate")
                    s.board, s.status = x.board, "playing"
                    x.board.spectators += 1
                    self.sessions[s.sid] = s
                    return s
        raise ValueError("that match is not live any more")

    def live_list(self):
        """The matches being played now, for the lobby (no player ids)."""
        with self.lock:
            out = [{"match_id": x.board.id, "name": x.name, "tick": x.board.tick_now, "crowns": list(x.board.crowns_now),
                    "speed": x.board.speed, "spectators": x.board.spectators, "paused": x.board.paused}
                   for x in self.sessions.values()
                   if x.kind == "match" and x.status == "playing" and x.board is not None and not x.board.done]
        return sorted(out, key=lambda r: r["match_id"])

    def voice_on(self):
        """The neural voice worker is alive (its heartbeat file is fresh)."""
        if not self.voice_dir:
            return False
        try:
            return time.time() - os.path.getmtime(os.path.join(self.voice_dir, "HEARTBEAT")) < 10
        except OSError:
            return False

    def request_voice(self, cid, text, voice=None):
        path = os.path.join(self.voice_dir, "req", cid + ".txt")
        with open(path + ".tmp", "w") as f:
            json.dump({"text": text, "voice": voice if voice in VOICES else VOICES[0]}, f)
        os.replace(path + ".tmp", path)

    def voice_marks(self, comments):
        """Comments for a frame, each with an audio clip marked ready once the worker has written it."""
        if not self.voice_dir:
            return comments
        return [dict(c, ready=os.path.exists(os.path.join(self.voice_dir, c["audio"] + ".wav"))) if c.get("audio") else c
                for c in comments]

    def voice_clip(self, cid):
        if not (self.voice_dir and isinstance(cid, str) and re.match(r"^[A-Za-z0-9_\-]{6,80}$", cid)):
            return None
        path = os.path.join(self.voice_dir, cid + ".wav")
        return open(path, "rb").read() if os.path.isfile(path) else None

    def meta(self, owner):
        pr = self.pr
        cards = []
        for i, name in enumerate(pr.CARD_NAMES):
            info = pr.card_info(i)
            unit = info.get("unit")
            cards.append({"id": i, "name": name, "cost": pr.CARD_COSTS[i], "kind": info.get("kind"),
                          "unit": unit.get("name") if isinstance(unit, dict) else unit,
                          "radius": info.get("radius") or 0, "count": info.get("count") or 1})
        return {"label": self.label, "greedy": self.args.greedy, "cards": cards,
                "decks": {k: list(pr.DECKS[k]) for k in POOL_DECKS}, "terrain": self.terrain,
                "speeds": list(SPEEDS), "replay_speeds": list(REPLAY_SPEEDS), "tps": TICKS_PER_SECOND,
                "frame_skip": FRAME_SKIP, "regulation": REGULATION_TICKS, "max_ticks": MAX_TICKS,
                "max_players": self.args.max_players, "owner": bool(owner), "claude": bool(owner and self.claude_on),
                "claude_daily_calls": self.args.claude_daily_calls if owner else None,
                "neural_voice": self.voice_on(), "voices": list(VOICES)}

    def status(self):
        with self.lock:
            return {"playing": self.live_count(), "max_players": self.args.max_players, "waiting": len(self.line)}

    def shutdown(self):
        self.stop.set()
        with self.lock:
            for s in list(self.sessions.values()):
                if s.kind == "match" and s.status in ("queued", "playing"):
                    self._end_session(s, abandoned=True)
        time.sleep(0.5)                 # the match threads save their records


# ====================================================================================== HTTP
def make_handler(server, token, owner_key, html):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *a):
            pass

        def _q(self):
            u = urlparse(self.path)
            return u, parse_qs(u.query)

        def _authorised(self, q):
            got = self.headers.get("X-Token") or (q.get("token") or [""])[0]
            return hmac.compare_digest(got.encode(), token.encode())

        def _owner(self, q):
            if not owner_key:
                return False
            got = self.headers.get("X-Owner") or (q.get("owner") or [""])[0]
            return bool(got) and hmac.compare_digest(got.encode(), owner_key.encode())

        def _send(self, code, body, ctype="application/json", headers=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, separators=(",", ":")).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            if ctype.startswith("text/html"):
                host = self.headers.get("Host") or ""
                ws = f" wss://{host} ws://{host}" if HOST_RE.match(host) else ""   # 'self' misses ws: in some browsers
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                 "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' data: blob:; "
                                 f"connect-src 'self'{ws}; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u, q = self._q()
            if u.path == "/api/alive":                       # no token: the permanent link's forwarding page (another
                return self._send(200, {"ok": True},         # site) checks that this address is up before sending people
                                  headers={"Access-Control-Allow-Origin": "*"})
            if not self._authorised(q):
                return self._send(403, "forbidden: open the link you were given (it ends in ?token=...)", "text/plain")
            owner = self._owner(q)
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, html, "text/html; charset=utf-8")
                if u.path == "/api/audio":
                    clip = server.voice_clip((q.get("id") or [""])[0])
                    if clip is None:
                        return self._send(404, {"error": "no such clip (yet)"})
                    return self._send(200, clip, "audio/wav")
                if u.path == "/api/meta":
                    return self._send(200, server.meta(owner))
                if u.path == "/api/leaderboard":
                    return self._send(200, dict(server.store.leaderboard(), status=server.status(), live=server.live_list()))
                if u.path == "/api/history":
                    return self._send(200, {"matches": server.store.history((q.get("player_id") or [""])[0])})
                if u.path == "/api/ws":
                    return self._ws(server.session((q.get("sid") or [""])[0]))
                if u.path == "/api/state":
                    # ?after=<version>: wait (up to 0.5 s) for a newer frame. The page polls one request at a time,
                    # because Cloudflare's quick tunnels hold streamed responses back for seconds (measured
                    # 2026-11-06: event-stream gaps up to 3.2 s through the tunnel, polling gaps <= 135 ms).
                    s = server.session((q.get("sid") or [""])[0])
                    b = s.board
                    if b is not None and s.status != "queued":
                        try:
                            after = int((q.get("after") or ["-1"])[0])
                        except ValueError:
                            after = -1
                        if s.kind == "match":
                            try:
                                b.note_client("poll", float((q.get("rtt") or ["0"])[0]))
                            except ValueError:
                                b.note_client("poll")
                        with b.cond:
                            if b.version <= after and not server.stop.is_set():
                                b.cond.wait(timeout=0.5)
                    return self._send(200, self._session_frame(s))
                if u.path == "/api/stream":
                    return self._stream(server.session((q.get("sid") or [""])[0]))
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            return self._send(404, {"error": "not found"})

        def _session_frame(self, s):
            if s.status == "queued":
                return json.dumps({"kind": "queued", "position": server.position(s), **server.status()}).encode()
            if s.board is None:
                return b'{"kind":"ended"}'
            return s.board.frame

        def _stream(self, s):
            with server.lock:
                if server.streams >= MAX_STREAMS:
                    return self._send(503, {"error": "too many connections"})
                server.streams += 1
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                last = -1
                while not server.stop.is_set() and s.sid in server.sessions:
                    s.last_seen = time.time()
                    if s.status == "queued":
                        self.wfile.write(b"data: " + self._session_frame(s) + b"\n\n")
                        self.wfile.flush()
                        time.sleep(1.0)
                        continue
                    b = s.board
                    if b is None:
                        break
                    if s.kind != "spectate":
                        b.last_client = time.time()
                    if s.kind == "match":
                        b.note_client("stream")
                    with b.cond:
                        if b.version == last:
                            b.cond.wait(timeout=10)
                        ver, frame = b.version, b.frame
                    if ver == last:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        self.wfile.write(b"data: " + frame + b"\n\n")
                        last = ver
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                with server.lock:
                    server.streams -= 1

        def _ws(self, s):
            """The page's live channel (WebSocket, RFC 6455). The server pushes each new frame of the session (the
            newest one when the page is slower than the match: no backlog builds up), and the page sends its plays,
            cancels and pings on the same connection. Measured 2026-11-09: a streamed HTTP response is held back
            for seconds by Cloudflare's quick tunnel, a WebSocket is not (see docs/HUMAN_PLAY.md)."""
            key = self.headers.get("Sec-WebSocket-Key")
            if not key or "websocket" not in (self.headers.get("Upgrade") or "").lower():
                return self._send(400, {"error": "a WebSocket upgrade was expected"})
            with server.lock:
                if server.streams >= MAX_STREAMS:
                    return self._send(503, {"error": "too many connections"})
                server.streams += 1
            sock = self.connection
            out = threading.Lock()
            gone = threading.Event()

            def send(payload, op=1):
                with out:
                    sock.sendall(ws_frame(payload, op))

            def reader():
                try:
                    while not gone.is_set():
                        op, data = ws_read(self.rfile)
                        if op == 8:                              # close
                            break
                        if op == 9:                              # ping -> pong
                            send(data, 10)
                            continue
                        if op != 1:
                            continue
                        s.last_seen = time.time()
                        b = s.board
                        if b is not None and s.kind != "spectate":
                            b.last_client = time.time()
                        msg = None
                        try:
                            msg = json.loads(data)
                            if not isinstance(msg, dict):
                                raise ValueError("bad message")
                            reply = self._ws_message(s, msg)
                        except (KeyError, ValueError, TypeError) as e:
                            reply = {"t": "error", "id": msg.get("id") if isinstance(msg, dict) else None,
                                     "error": str(e) or "bad message"}
                        if reply is not None:
                            send(json.dumps(reply, separators=(",", ":")).encode())
                except (OSError, EOFError, ValueError):
                    pass
                finally:
                    gone.set()

            try:
                self.wfile.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                                  f"Sec-WebSocket-Accept: {ws_accept(key)}\r\n\r\n").encode())
                self.wfile.flush()
                sock.settimeout(WS_TIMEOUT_S)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(target=reader, name="ws-reader", daemon=True).start()
                last, pinged = -1, time.time()
                while not gone.is_set() and not server.stop.is_set() and s.sid in server.sessions:
                    s.last_seen = time.time()
                    if s.status == "queued" or s.board is None:
                        send(self._session_frame(s))
                        if s.board is None and s.status != "queued":
                            break                               # ended before it started
                        gone.wait(1.0)
                        continue
                    b = s.board
                    if s.kind != "spectate":
                        b.last_client = time.time()
                    if s.kind == "match":
                        b.note_client("ws")
                    with b.cond:
                        if b.version == last:
                            b.cond.wait(timeout=1.0)
                        ver, frame = b.version, b.frame
                    if ver != last:
                        send(frame)
                        last = ver
                    elif time.time() - pinged > WS_PING_S:
                        send(b"", 9)
                        pinged = time.time()
            except (OSError, ValueError):
                pass
            finally:
                gone.set()
                try:
                    send(struct.pack("!H", 1000), 8)
                except OSError:
                    pass
                with server.lock:
                    server.streams -= 1
                self.close_connection = True

        def _ws_message(self, s, msg):
            """One message from the page on its WebSocket; returns the reply (or None)."""
            t = msg.get("t")
            b = s.board
            if t == "ping":                                      # the page's clock goes back as is: it measures the round trip
                if b is not None and s.kind == "match" and isinstance(msg.get("rtt"), (int, float)):
                    b.note_client("ws", float(msg["rtt"]))
                return {"t": "pong", "c": msg.get("c")}
            if s.kind == "spectate":
                raise ValueError("you are watching: spectators cannot play or change the match")
            if b is None:
                raise ValueError("your match has not started yet")
            if t == "play" and s.kind == "match":
                ok, why = b.queue_play(int(msg["card"]), int(msg["tx"]), int(msg["ty"]), pid=msg.get("id"))
                return {"t": "ack", "id": msg.get("id"), "ok": ok, "reason": why}
            if t == "cancel" and s.kind == "match":
                with b.lock:
                    b.queue.clear()
                    b.publish()
                return {"t": "ack", "id": msg.get("id"), "ok": True, "reason": "cancelled"}
            raise ValueError("unknown message")

        def do_POST(self):
            u, q = self._q()
            if not self._authorised(q):
                return self._send(403, {"error": "forbidden"})
            owner = self._owner(q)
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError("bad request")
                return self._post(u.path, body, owner)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e) or "bad request"})

        def _post(self, path, body, owner):
            pr, st = server.pr, server.store
            if path == "/api/hello":
                if body.get("name") is not None and not name_key(body["name"]):
                    raise ValueError("please enter a name (letters, digits, spaces)")
                pin = body.get("pin")
                pid, pl, created, new_pin = st.hello(body.get("player_id"), body.get("name"),
                                                     None if pin is None else str(pin).strip())
                out = {"player_id": pid, "profile": pl, "created": created, "history": st.history(pid) if pid else []}
                if new_pin:
                    out["new_pin"] = new_pin                   # shown once by the page; only its hash is kept
                return self._send(200, out)
            if path == "/api/pin":
                st.set_pin(body["player_id"], str(body.get("pin") or "").strip())
                return self._send(200, {"ok": True})
            if path == "/api/profile":
                fields = {}                                    # the name is the login: it changes only via hello
                if "deck" in body:
                    label, ids = deck_spec(body["deck"], pr)
                    fields["deck"] = label if label != "custom" else ids
                if "opp_deck" in body:
                    fields["opp_deck"] = deck_spec(body["opp_deck"], pr, allow_random=True)[0]
                if "seat" in body:
                    fields["seat"] = 1 if int(body["seat"]) == 1 else 0
                if "speed" in body and float(body["speed"]) in SPEEDS:
                    fields["speed"] = float(body["speed"])
                return self._send(200, {"profile": st.update(body["player_id"], **fields)})
            if path == "/api/join":
                pid = body["player_id"]
                if pid not in st.players:
                    raise ValueError("unknown player: reload the page")
                name = st.players[pid].get("name")
                if not name:
                    raise ValueError("please enter a name first")
                d_label, d_ids = deck_spec(body["deck"], pr)
                o_label, o_ids = deck_spec(body.get("opp_deck", "random"), pr, allow_random=True)
                speed = float(body.get("speed", 1.0))
                if speed not in SPEEDS:
                    raise ValueError(f"speed must be one of {SPEEDS}")
                params = {"seat": 1 if int(body.get("seat", 0)) == 1 else 0, "deck_label": d_label, "deck_ids": d_ids,
                          "opp_label": o_label, "opp_ids": o_ids, "speed": speed}
                st.update(pid, deck=d_label if d_label != "custom" else d_ids, opp_deck=o_label,
                          seat=params["seat"], speed=speed)
                s = server.join(pid, name, owner, params)
                return self._send(200, {"sid": s.sid, "status": s.status, "position": server.position(s)})
            if path == "/api/watch":
                s = server.watch(body.get("player_id"), owner, body["match_id"])
                return self._send(200, {"sid": s.sid})
            if path == "/api/spectate":
                s = server.spectate(owner, str(body["match_id"]))
                return self._send(200, {"sid": s.sid})
            s = server.session(body.get("sid"))
            b = s.board
            if path == "/api/leave":
                if s.kind in ("match", "spectate"):
                    server.leave(s)
                else:
                    s.status = "ended"
                return self._send(200, {"ok": True})
            if s.kind == "spectate":
                raise ValueError("you are watching: spectators cannot play or change the match")
            if b is None:
                raise ValueError("your match has not started yet")
            if path == "/api/play" and s.kind == "match":
                ok, why = b.queue_play(int(body["card"]), int(body["tx"]), int(body["ty"]), pid=body.get("id"))
                return self._send(200, {"ok": ok, "reason": why})
            if path == "/api/cancel" and s.kind == "match":
                with b.lock:
                    b.queue.clear()
                    b.publish()
                return self._send(200, {"ok": True})
            if path == "/api/control" and s.kind == "match":
                b.control(body, s.owner)
                return self._send(200, {"ok": True})
            if path == "/api/control" and s.kind == "replay":
                with b.lock:
                    if "paused" in body:
                        b.paused = bool(body["paused"])
                    if "speed" in body:
                        sp = float(body["speed"])
                        if sp not in REPLAY_SPEEDS:
                            raise ValueError(f"speed must be one of {REPLAY_SPEEDS}")
                        b.speed = sp
                    b.publish()
                if "seek" in body:
                    b.seek(int(body["seek"]))
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})

    return Handler


def read_secret(path, what):
    v = open(path).read().strip()
    if len(v) < 16:
        raise SystemExit(f"the {what} in {path} must be at least 16 characters")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_MODEL,
                    help="model .pt / snapshot / run directory of the opponent policy (default: the repo's models/S4_A_s1_10B.pt)")
    ap.add_argument("--label", default=None, help="opponent name shown on the page (default: run / file / steps)")
    ap.add_argument("--greedy", action="store_true", help="card-first greedy actions instead of sampling")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token-file", default=None, help="the site token (default: a new random one, printed)")
    ap.add_argument("--owner-key-file", default=None, help="the owner key (?owner=...): unlocks the Claude toggles")
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("DATA_DIR", os.path.join(ROOT, "experiments")),
                                                       "human_play"), help="players.json and matches/")
    ap.add_argument("--max-players", type=int, default=5, help="matches at once; later players wait in line")
    ap.add_argument("--max-replays", type=int, default=5, help="replays at once")
    ap.add_argument("--claude-relay-dir", default=None,
                    help="directory of a running hpc/llm_relay.py (Claude Code CLI on the host); none = no Claude")
    ap.add_argument("--claude-daily-calls", type=int, default=200, help="Claude calls per day at most (all matches)")
    ap.add_argument("--comment-every", type=int, default=200, help="ticks between Claude updates (200 = 10 s)")
    ap.add_argument("--claude-timeout", type=float, default=120.0, help="seconds to wait for one Claude reply")
    ap.add_argument("--voice-dir", default=None, help="directory of a running play/tts_worker.py (neural voice); none = "
                    "the page uses the browser's own speech")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads (torch's default is far slower here)")
    args = ap.parse_args()

    token = read_secret(args.token_file, "site token") if args.token_file else secrets.token_urlsafe(16)
    owner_key = read_secret(args.owner_key_file, "owner key") if args.owner_key_file else None
    html = open(HTML_FILE, encoding="utf-8").read()
    server = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server, token, owner_key, html))
    httpd.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"[play] opponent: {server.label} (sha256 {server.checkpoint_sha[:12]}), "
          f"{'greedy' if args.greedy else 'sampled'} actions; {args.max_players} seats; records in {args.data_dir}; "
          f"Claude {'via ' + args.claude_relay_dir + ' (owner only)' if server.claude_on else 'off'}", flush=True)
    if not args.token_file:
        print(f"[play] open http://{shown}:{args.port}/?token={token}", flush=True)
    else:
        print(f"[play] listening on {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        print("[play] stopped", flush=True)


if __name__ == "__main__":
    main()
