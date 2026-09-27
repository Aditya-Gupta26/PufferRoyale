"""pufferroyale.llm -- play PufferRoyale through text, e.g. against a language model (SPEC §17).

A provider-agnostic harness: the game state is rendered as text in the acting team's own frame
(`render_state`), a model function answers with one command line (`WAIT` or
`PLAY <slot|card> AT <tx>,<ty>`), and `parse_action` turns that into a Discrete(2305) action
(anything unparseable or illegal is a no-op).

    RULES_PROMPT                 static system prompt (rules, coordinates, the 64 cards, the
                                 response format); identical across turns and matches, so it
                                 can be prompt-cached
    render_state(game, team)     one decision's state text (public information only)
    parse_action(text, g, team)  -> (action, info)
    LLMAgent(model_fn)           model_fn(system, user) -> str; act(game, team) -> action,
                                 with a transcript entry per decision
    play_llm_match(agent, opp, deck_agent, deck_opp, seed, ...)   one full match via Game
    mock_wait / mock_first_legal / mock_random(seed)   deterministic text-only model functions
    anthropic_model_fn(model, effort)                  optional adapter for the Anthropic API
                                                        (lazy import; never used by tests)

Hidden information (SPEC §9 / §17.1): `render_state` reads only the viewer's own hand, queue
and elixir; everything about the opponent comes from the viewer's own observation vector
(`Game.obs(team)`: cards seen, last four played, the deduced hand once all eight cards were
seen, the elixir upper bound) or from the public board (units, towers, spells, crowns, tower
troops). The opponent's current elixir, hand, queue and unrevealed deck never enter the text.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import game as _game
from .game import CARD_COSTS, CARD_NAMES, N_CARDS, N_TILES, TOWER_TROOPS, Game, card_id, card_info

# ------------------------------------------------------------------------------------ constants
TILES_X, TILES_Y = 18, 32
ARENA_W, ARENA_H = 18000, 32000
TICKS_PER_SECOND = 20
REGULATION_TICKS = 3600            # SPEC §4: 180 s
MAX_TICKS = 6000                   # + 120 s sudden-death overtime
ELIXIR_UNIT = _game.ELIXIR_UNIT    # 2800 units = 1 elixir
ELIXIR_MAX = 10 * ELIXIR_UNIT
ELIXIR_RATE_1X = 50                # units per tick
MELEE_RANGE = 1900                 # obs encoder: range > 1900 counts as ranged
TOWER_TROOP_LABEL = {"princess": "Princess", "cannoneer": "Cannoneer", "dagger_duchess": "Dagger Duchess",
                     "royal_chef": "Royal Chef"}
KNOWN_ANTHROPIC_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5",
                          "claude-haiku-4-5")

_INFO = [card_info(c) for c in range(N_CARDS)]
_KIND = [ci["kind"] for ci in _INFO]
_PRIMARY_UNIT = [ci.get("unit") for ci in _INFO]

# One-line roles (game knowledge; the numbers next to them come from the compiled card data).
CARD_ROLES = {
    "Knight": "cheap melee mini-tank and defender",
    "Archers": "ranged pair; cheap anti-air support",
    "Musketeer": "long-range single-target damage; anti-air",
    "Giant": "slow tank win condition",
    "Hog Rider": "fast win condition; jumps the river",
    "Minions": "fast flying damage trio",
    "Baby Dragon": "flying splash mini-tank",
    "Valkyrie": "melee 360-degree splash; anti-swarm",
    "Skeleton Army": "15-skeleton swarm; shreds single targets, dies to splash",
    "Skeletons": "cheap distraction and cycle card",
    "Ice Golem": "cheap tank; slows nearby enemies on death",
    "Ice Spirit": "kamikaze that freezes a small area; cycle card",
    "Prince": "charges after a run-up for double damage",
    "Wizard": "ranged splash vs air and ground swarms",
    "Cannon": "cheap defensive building vs ground units",
    "Tesla": "defensive building hidden underground until it attacks; hits air",
    "Fireball": "medium damage spell with knockback",
    "Arrows": "wide low-damage spell (3 waves) vs swarms",
    "Zap": "instant small spell; stuns 0.5 s",
    "The Log": "rolling ground spell with knockback; own side only",
    "Goblin Barrel": "barrel that drops 3 Goblins on the target (tower bait)",
    "Barbarians": "5 melee brawlers",
    "Mini P.E.K.K.A": "heavy single-target melee damage",
    "P.E.K.K.A": "armoured melee tank killer",
    "Mega Minion": "sturdy flying single-target damage",
    "Bats": "fragile fast flying swarm",
    "Spear Goblins": "cheap ranged trio; hits air",
    "Goblins": "cheap fast melee swarm",
    "Goblin Gang": "3 melee Goblins + 3 Spear Goblins",
    "Royal Giant": "tank that hits buildings from range",
    "Bomber": "fragile ground splash vs swarms",
    "Princess": "very long-range splash",
    "Dart Goblin": "long range, very fast attacks",
    "Minion Horde": "6 Minions; huge air damage, dies to splash",
    "Fire Spirit": "kamikaze splash; cycle card",
    "Ice Wizard": "splash attacks that slow",
    "Golem": "huge slow tank; splits into 2 Golemites on death",
    "Lava Hound": "flying tank; bursts into 6 Lava Pups on death",
    "Balloon": "flying win condition; bomb on death",
    "Giant Skeleton": "tank with a huge bomb on death",
    "Witch": "ranged splash; keeps summoning Skeletons",
    "Tombstone": "building that spawns Skeletons (distraction)",
    "Inferno Tower": "defensive building whose damage ramps up; melts tanks",
    "Inferno Dragon": "flying ramping beam; melts tanks",
    "Bandit": "dashes onto targets, invulnerable while dashing",
    "Battle Ram": "charging win condition; releases 2 Barbarians",
    "Royal Hogs": "4 hogs that jump the river and target buildings",
    "Miner": "deploys almost anywhere (burrows there); chips towers",
    "Mortar": "siege building; long range, cannot hit within 3.5 tiles",
    "X-Bow": "siege building that hits towers from your side",
    "Elixir Collector": "pump: +1 elixir every 13 s while alive",
    "Bomb Tower": "defensive splash building vs ground",
    "Rocket": "heavy damage spell, small radius",
    "Poison": "damage-over-time area for 8 s; slows",
    "Freeze": "freezes everything in the area for 4 s",
    "Earthquake": "ground damage over 3 s; strong vs buildings",
    "Lightning": "strikes the 3 highest-HP targets; stuns",
    "Giant Snowball": "small damage + knockback + slow",
    "Barbarian Barrel": "rolling spell that leaves a Barbarian; own side only",
    "Rascals": "tanky Rascal Boy + 2 ranged Rascal Girls",
    "Elite Barbarians": "fast strong melee pair",
    "Skeleton Dragons": "2 flying splash attackers",
    "Wall Breakers": "fast kamikaze pair vs buildings",
    "Night Witch": "summons Bats; releases Bats on death",
}


def _num(x, nd=1):
    """x with at most `nd` decimals, trailing zeros of the fraction dropped (30.0 -> '30', 2.5 -> '2.5')."""
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _speed_label(v):
    if not v:
        return None
    return "slow" if v <= 45 else "medium" if v <= 60 else "fast" if v <= 90 else "very fast"


def _card_facts(c: int) -> str:
    """Compact facts of card c from the compiled level-11 data."""
    ci = _INFO[c]
    kind = ci["kind"]
    parts = []
    if kind in ("troop", "building"):
        cnt = int(ci.get("count") or 1)
        parts.append(kind + (f" x{cnt}" if cnt > 1 else ""))
        if kind == "troop" and (ci.get("flying_height") or 0) > 0:
            parts.append("air")
        hp = ci.get("hitpoints") or 0
        if hp:
            parts.append(f"{hp} HP" + (" each" if cnt > 1 else ""))
        dmg, hs = ci.get("damage") or 0, ci.get("hit_speed_ms") or 0
        if dmg and ci.get("kamikaze"):
            parts.append(f"{dmg} dmg on impact (dies)")
        elif dmg and hs:
            parts.append(f"{dmg} dmg / {_num(hs / 1000)} s")
        if dmg or ci.get("kamikaze"):
            rng = ci.get("range_milli") or ci.get("range") or 0
            if rng:
                parts.append("melee" if rng <= MELEE_RANGE else f"range {_num(rng / 1000)}")
            splash = (ci.get("area_damage_radius") or 0) > 0 or (
                isinstance(ci.get("projectile"), dict) and (ci["projectile"].get("radius") or 0) > 0)
            if splash:
                parts.append("splash")
            if ci.get("target_only_buildings"):
                parts.append("targets buildings only")
            elif ci.get("attacks_air") and ci.get("attacks_ground"):
                parts.append("hits air+ground")
            elif ci.get("attacks_ground"):
                parts.append("hits ground only")
            elif ci.get("attacks_air"):
                parts.append("hits air only")
        sp = _speed_label(ci.get("speed")) if kind == "troop" else None
        if sp:
            parts.append(sp)
        if kind == "building" and ci.get("lifetime_ms"):
            parts.append(f"lasts {_num(ci['lifetime_ms'] / 1000, 0)} s")
    else:
        parts.append("spell")
        dmg = ci.get("damage") or 0
        n = int(ci.get("events") or ci.get("waves") or 1)
        if dmg:
            s = f"{dmg} dmg" + (f" x{n}" if n > 1 else "")
            ct = ci.get("crown_tower_damage")
            if ct is not None and ct != dmg:
                s += f" ({ct} to crown towers)"
            parts.append(s)
        r = ci.get("radius") or 0
        if r:
            parts.append(f"radius {_num(r / 1000)}")
        sp = ci.get("spawn")
        if isinstance(sp, dict) and sp.get("name"):
            cnt = int(ci.get("count") or 1)
            parts.append(f"drops {cnt} {sp['name']}{'s' if cnt > 1 else ''} ({sp.get('hitpoints', 0)} HP, "
                         f"{sp.get('damage', 0)} dmg)")
    return ", ".join(parts)


def _card_line(c: int) -> str:
    name = CARD_NAMES[c]
    return f"- {name} ({CARD_COSTS[c]}): {_card_facts(c)} - {CARD_ROLES.get(name, '')}".rstrip(" -")


def _build_rules_prompt() -> str:
    cards = "\n".join(_card_line(c) for c in range(N_CARDS))
    return f"""You are playing PufferRoyale, a faithful 1v1 simulation of a Clash Royale ladder battle (every card at level 11). Each turn you receive the current battle state as text and choose one action for your side.

GOAL
- Destroy enemy crown towers. Each enemy Princess tower is 1 crown; the enemy King tower is 3 crowns and ends the match at once.
- Regulation lasts 3:00; if the crowns differ at the end, the leader wins. If tied, 2:00 of sudden-death overtime follow: the first crown wins. If still tied, the side whose weakest standing tower has more HP wins; equal HP is a draw.
- A King tower does not attack until it is damaged or one of its Princess towers is destroyed; it wakes about 3.5 s later.

ELIXIR AND CARDS
- You start with 6 elixir and gain 1 every 2.8 s (double speed in the last 1:00 of regulation and the first 1:00 of overtime, triple in the last 1:00 of overtime), up to 10; anything above 10 is wasted, so keep spending.
- You hold 4 cards (hand slots 0-3) out of an 8-card deck. Playing a card costs its elixir; that slot is refilled with your "next" card and the played card goes to the back of the cycle.
- No card can be played during the first 4.5 s (deploy lockout). Units spend about 1 s deploying before they act.

ARENA (always in YOUR OWN frame: you are at the bottom)
- 18 x 32 tiles, tile (tx, ty): tx = 0..17 from left to right, ty = 0..31 from the enemy back row (ty 0) to your back row (ty 31).
- Enemy half: ty 0-14. Enemy King tower: tx 7-10, ty 1-4. Enemy Princess towers: left tx 2-4, right tx 13-15, both ty 5-7.
- River: ty 15-16, crossed on foot only over the two bridges at tx 3 and tx 14 (jumpers such as Hog Rider cross anywhere; air units fly over).
- Your half: ty 17-31. Your Princess towers: left tx 2-4, right tx 13-15, both ty 24-26. Your King tower: tx 7-10, ty 27-30.
- Troops walk down their lane (the side of the arena they are on) toward the nearest enemy target they can attack; "targets buildings only" units ignore troops and head for buildings and towers.

PLACEMENT
- Troops and buildings go on free tiles of your half (ty 17-31, not on towers or buildings). When an enemy Princess tower falls, that lane's side of the enemy half (ty 11-14) opens for your troops too.
- Spells can target any tile (The Log and Barbarian Barrel only your side); the Miner can go almost anywhere except onto towers.
- The state lists, for every hand slot, the exact tiles where that card can be played right now; any other tile is rejected. A card you cannot afford has no legal tiles.

TIMING
- You are asked for a decision at regular intervals (by default once per game second). The game does not advance while you think; a play is applied on the next tick.

CARDS (name (elixir): level-11 facts - role; HP and damage per unit, range in tiles)
{cards}

RESPONSE FORMAT
Think as briefly as you like, then end your reply with exactly one final line, either
WAIT
or
PLAY <slot> AT <tx>,<ty>
where <slot> is your hand slot number 0-3 (or the card's exact name) and (tx, ty) is one of that slot's legal tiles, in your own frame. Example: PLAY 2 AT 9,20
Only the last line of that form counts. An unreadable reply, a card that is not in your hand, or a tile that is not legal is treated as WAIT."""


#: The static system prompt (SPEC §17.1): no timestamps or ids, identical for every call.
RULES_PROMPT: str = _build_rules_prompt()


# ------------------------------------------------------------------------------------ rendering
def _own_tile(team: int, x: int, y: int) -> Tuple[int, int]:
    """Own-frame tile of an engine-frame point (the obs encoder's convention)."""
    ox, oy = (x, y) if team == 0 else (ARENA_W - x, ARENA_H - y)
    return min(TILES_X - 1, max(0, ox // 1000)), min(TILES_Y - 1, max(0, oy // 1000))


def _clock(tick: int) -> str:
    if tick < REGULATION_TICKS:
        left, phase = REGULATION_TICKS - tick, "regulation"
    else:
        left, phase = MAX_TICKS - tick, "overtime, sudden death: the next crown wins"
    secs = -(-left // TICKS_PER_SECOND)
    return f"{secs // 60}:{secs % 60:02d} left ({phase})"


def _elixir_1d(units: int) -> str:
    """Elixir to one decimal, rounded DOWN (never shows a card as affordable when it is not)."""
    return f"{math.floor(units * 10 / ELIXIR_UNIT + 1e-9) / 10:.1f}"


def _runs(cols: Sequence[int]) -> str:
    out, start, prev = [], None, None
    for c in cols:
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = c
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def legal_regions(mask: np.ndarray, slot: int) -> List[Tuple[int, int, str]]:
    """Row-run encoding of the legal tiles of hand `slot`: [(ty_from, ty_to, "tx runs")], rows with
    identical runs merged, in increasing ty."""
    blk = np.asarray(mask[1 + slot * N_TILES:1 + (slot + 1) * N_TILES]).reshape(TILES_Y, TILES_X) > 0
    rows = []
    for ty in range(TILES_Y):
        cols = np.flatnonzero(blk[ty])
        if len(cols):
            r = _runs(cols.tolist())
            if rows and rows[-1][2] == r and rows[-1][1] == ty - 1:
                rows[-1] = (rows[-1][0], ty, r)
            else:
                rows.append((ty, ty, r))
    return rows


def _fmt_rows(ty0, ty1) -> str:
    return f"ty={ty0}" if ty0 == ty1 else f"ty={ty0}-{ty1}"


def _unit_label(e: dict) -> str:
    c = e["card_id"]
    if 0 <= c < N_CARDS:
        name = CARD_NAMES[c]
        prim = _PRIMARY_UNIT[c]
        if prim is not None and e["unit"] != prim and _game._norm(e["unit"]) not in _game._norm(name):
            name += f" ({e['unit']})"
        return name
    return e["unit"]


def _unit_flags(e: dict) -> str:
    f = []
    if e["kind"] == "building":
        f.append("building")
    if e.get("flying"):
        f.append("air")
    if e.get("burrowing"):
        f.append("underground")
    elif e.get("deploying"):
        f.append("deploying")
    if e.get("hidden") and not e.get("burrowing"):
        f.append("hidden")
    if e.get("stunned"):
        f.append("stunned")
    if e.get("slowed"):
        f.append("slowed")
    return f" [{', '.join(f)}]" if f else ""


def _units_block(ents: List[dict], team: int) -> List[str]:
    """One line per unit, sorted by id; consecutive identical lines are merged as 'Nx ...'."""
    lines: List[str] = []
    for e in ents:
        tx, ty = _own_tile(team, e["x"], e["y"])
        s = f"{_unit_label(e)} @ ({tx},{ty}) {e['hp']}/{e['max_hp']}{_unit_flags(e)}"
        if lines and lines[-1][1] == s:
            lines[-1] = (lines[-1][0] + 1, s)
        else:
            lines.append((1, s))
    return [f"  {n}x {s}" if n > 1 else f"  {s}" for n, s in lines] or ["  none"]


def _tower_line(st: dict, owner: int, viewer: int) -> str:
    towers = st["towers"][owner]
    left, right = (1, 2) if viewer == 0 else (2, 1)       # own-frame left/right (obs convention)
    label = TOWER_TROOP_LABEL.get(st["tower_troops"][owner], st["tower_troops"][owner])
    king = towers[0]
    parts = [f"King {king['hp']}/{king['max_hp']} ({'awake' if king['active'] else 'asleep'})" if king["alive"]
             else "King destroyed"]
    for side, i in (("left", left), ("right", right)):
        t = towers[i]
        parts.append(f"{side} {label} {t['hp']}/{t['max_hp']}" if t["alive"] else f"{side} {label} destroyed")
    return " | ".join(parts)


def _spells_block(game: Game, team: int) -> List[str]:
    out = []
    for p in game.projectiles():
        c = p.get("card_id", -1)
        if not (0 <= c < N_CARDS) or _KIND[c] != "spell" or p.get("target_id", -1) != -1:
            continue
        tx, ty = _own_tile(team, p["target_x"], p["target_y"])
        who = "your" if p["team"] == team else "enemy"
        out.append(f"  {who} {CARD_NAMES[c]} in flight, lands at ({tx},{ty})")
    for f in game.effects():
        c = f.get("card_id", -1)
        name = CARD_NAMES[c] if 0 <= c < N_CARDS else f["name"]
        tx, ty = _own_tile(team, f["x"], f["y"])
        who = "your" if f["team"] == team else "enemy"
        what = "rolling at" if f.get("kind") == "rolling" else "active at"
        out.append(f"  {who} {name} {what} ({tx},{ty}), radius {_num(f['radius'] / 1000)}")
    return out


_LAYOUT = _game._b.royale_layout()          # the obs layout straight from the binding (no pufferlib needed)


def _scalar(obs: np.ndarray, name: str) -> np.ndarray:
    off, n = _LAYOUT["SCALAR_FIELDS"][name]
    base = _LAYOUT["SCALAR_OFFSET"] + off
    return obs[base:base + n]


def _opp_public(obs: np.ndarray) -> dict:
    """The opponent information of the viewer's own observation (SPEC §9 deductions only)."""
    seen = np.flatnonzero(_scalar(obs, "opp_seen")[:N_CARDS] > 0.5).tolist()
    last = [int(round(v)) - 1 for v in _scalar(obs, "opp_last4")]
    deduced = np.flatnonzero(_scalar(obs, "opp_deduced_hand")[:N_CARDS] > 0.5).tolist()
    ub_units = int(round(float(_scalar(obs, "opp_elixir_ub")[0]) * ELIXIR_MAX))
    return {"seen": seen, "last": [c for c in last if 0 <= c < N_CARDS], "deduced": deduced, "ub_units": ub_units}


def render_state(game: Game, team: int, *, legal: bool = True) -> str:
    """The state text for one decision of `team`, in its own frame (SPEC §17.1). Deterministic;
    public information only (see the module docstring)."""
    team = int(team)
    if team not in (0, 1):
        raise ValueError("team must be 0 or 1")
    opp = 1 - team
    st = game.state()
    tick = int(st["tick"])
    elixir = int(st["elixir"][team])                     # own elixir: allowed
    hand = [int(c) for c in st["hand"][team]]            # own hand / queue: allowed
    nxt = int(st["queue"][team][0])
    lockout = tick < int(st["lockout_ticks"])
    mask = game.legal_mask(team)                         # own legality (also drives 'playable')
    opub = _opp_public(game.obs(team))
    rate = max(1, int(round(int(st["elixir_rate"]) / ELIXIR_RATE_1X)))

    L = []
    L.append(f"TIME {_clock(tick)} | tick {tick} of {MAX_TICKS} | elixir speed {rate}x")
    L.append(f"CROWNS you {st['crowns'][team]} - {st['crowns'][opp]} opponent")
    if lockout:
        L.append(f"DEPLOY LOCKOUT: no card can be played for another "
                 f"{_num((int(st['lockout_ticks']) - tick) / TICKS_PER_SECOND)} s")
    L.append(f"YOUR ELIXIR {_elixir_1d(elixir)} / 10")
    L.append("YOUR HAND (slot: card (elixir)):")
    for s, c in enumerate(hand):
        if not 0 <= c < N_CARDS:
            L.append(f"  {s}: empty")
            continue
        cost = CARD_COSTS[c]
        has_tile = bool(np.any(mask[1 + s * N_TILES:1 + (s + 1) * N_TILES]))
        if lockout:
            status = "deploy lockout"
        elif elixir < cost * ELIXIR_UNIT:
            short = cost * ELIXIR_UNIT - elixir
            status = f"need {math.ceil(short * 10 / ELIXIR_UNIT) / 10:.1f} more elixir"
        elif has_tile:
            status = "playable"
        else:
            status = "no legal tile"
        L.append(f"  {s}: {CARD_NAMES[c]} ({cost}) - {status}")
    if 0 <= nxt < N_CARDS:
        L.append(f"  next: {CARD_NAMES[nxt]} ({CARD_COSTS[nxt]})")
    L.append(f"YOUR TOWERS: {_tower_line(st, team, team)}")
    L.append(f"ENEMY TOWERS: {_tower_line(st, opp, team)}")

    ents = [e for e in game.entities() if e["kind"] != "tower"]
    L.append("YOUR UNITS (card @ (tx,ty) hp/max [flags]):")
    L.extend(_units_block([e for e in ents if e["team"] == team], team))
    L.append("ENEMY UNITS:")
    L.extend(_units_block([e for e in ents if e["team"] == opp], team))
    spells = _spells_block(game, team)
    if spells:
        L.append("SPELLS:")
        L.extend(spells)

    names = lambda cs: ", ".join(CARD_NAMES[c] for c in cs) if cs else "none"  # noqa: E731
    L.append(f"OPPONENT CARDS SEEN ({len(opub['seen'])} of 8): {names(opub['seen'])}")
    L.append(f"OPPONENT LAST PLAYED (most recent first): {names(opub['last'])}")
    if opub["deduced"]:
        L.append(f"OPPONENT HAND (deduced: the 4 seen cards not among the last 4 played): {names(opub['deduced'])}")
    L.append(f"OPPONENT ELIXIR: at most {_elixir_1d(opub['ub_units'])} (6 at the start + regeneration - elixir "
             f"spent; the exact value is hidden)")

    if legal:
        L.append('LEGAL TILES per hand slot (own frame; "ty=<rows>: tx <columns>"):')
        for s, c in enumerate(hand):
            if not 0 <= c < N_CARDS:
                continue
            regions = legal_regions(mask, s)
            head = f"slot {s} {CARD_NAMES[c]}:"
            if not regions:
                why = ("deploy lockout" if lockout else
                       f"costs {CARD_COSTS[c]}, you have {_elixir_1d(elixir)}" if elixir < CARD_COSTS[c] * ELIXIR_UNIT
                       else "no free tile")
                L.append(f"{head} none ({why})")
                continue
            L.append(head)
            for ty0, ty1, runs in regions:
                L.append(f"  {_fmt_rows(ty0, ty1)}: tx {runs}")
    L.append("Reply with your reasoning if you like, then a final line: WAIT or PLAY <slot> AT <tx>,<ty>")
    return "\n".join(L)


# ------------------------------------------------------------------------------------ text parsing (for text-only agents)
_HAND_RE = re.compile(r"^\s{2}([0-3]): (.+) \((\d+)\) - (.+)$")
_SLOT_RE = re.compile(r"^slot ([0-3]) (.+?):(.*)$")
_ROW_RE = re.compile(r"^\s+ty=(\d+)(?:-(\d+))?: tx ([0-9,\-]+)$")


def parse_rendered_hand(text: str) -> List[dict]:
    """[{slot, card, cost, playable}] from the YOUR HAND section of render_state text."""
    out, in_hand = [], False
    for line in text.splitlines():
        if line.startswith("YOUR HAND"):
            in_hand = True
            continue
        if in_hand:
            m = _HAND_RE.match(line)
            if m:
                out.append({"slot": int(m.group(1)), "card": m.group(2), "cost": int(m.group(3)),
                            "playable": m.group(4).strip() == "playable"})
            elif not line.startswith("  "):
                break
    return out


def parse_rendered_legal(text: str) -> Dict[int, List[Tuple[int, int]]]:
    """{slot: [(tx, ty), ...]} legal tiles from the LEGAL TILES section, in listed order (rows
    ascending, columns ascending)."""
    out: Dict[int, List[Tuple[int, int]]] = {}
    cur, in_legal = None, False
    for line in text.splitlines():
        if line.startswith("LEGAL TILES"):
            in_legal = True
            continue
        if not in_legal:
            continue
        m = _SLOT_RE.match(line)
        if m:
            cur = int(m.group(1))
            out[cur] = []
            continue
        m = _ROW_RE.match(line)
        if m and cur is not None:
            ty0 = int(m.group(1))
            ty1 = int(m.group(2)) if m.group(2) else ty0
            cols = []
            for part in m.group(3).split(","):
                a, _, b = part.partition("-")
                cols.extend(range(int(a), int(b or a) + 1))
            for ty in range(ty0, ty1 + 1):
                out[cur].extend((tx, ty) for tx in cols)
            continue
        if line and not line.startswith(" "):
            cur = None
    return out


# ------------------------------------------------------------------------------------ action parsing
_PLAY_RE = re.compile(r"^PLAY\s+(?P<card>.+?)\s+AT\s*\(?\s*(?P<tx>-?\d+)\s*(?:,|\s)\s*(?P<ty>-?\d+)\s*\)?$", re.I)
_WAIT_RE = re.compile(r"^WAIT$", re.I)
_PREFIX_RE = re.compile(r"^(?:final\s+)?(?:action|answer|move|command|decision)\s*[:=\-]\s*", re.I)


def _clean_line(line: str) -> str:
    s = line.strip().strip("`*_>#\"' \t").strip()
    s = _PREFIX_RE.sub("", s).strip().strip("`*_\"' ").rstrip(".!").strip()
    return s


def parse_action(text: str, game: Game, team: int) -> Tuple[int, dict]:
    """(action, info) from a model reply (SPEC §17.1). The LAST line that reads `WAIT` or
    `PLAY <slot|card name> AT <tx>,<ty>` (case-insensitive; markdown emphasis, quotes, a leading
    'Action:' and a trailing period are tolerated) decides. Coordinates are own-frame tiles.
    Unparseable, unknown card / slot, or illegal (the engine mask) -> (0, {"error": ...}).
    info also carries "kind" ('play' | 'wait' | None) and, for errors, "error_kind"
    ('parse' for format / card / slot problems, 'illegal' for a rejected tile)."""
    team = int(team)
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    matched = None
    for raw in reversed(text.splitlines()):
        s = _clean_line(raw)
        if _WAIT_RE.match(s):
            matched = ("wait", None, s)
            break
        m = _PLAY_RE.match(s)
        if m:
            matched = ("play", m, s)
            break
    if matched is None:
        return 0, {"kind": None, "error": "unparseable: no line reads WAIT or PLAY <slot> AT <tx>,<ty>",
                   "error_kind": "parse"}
    kind, m, line = matched
    if kind == "wait":
        return 0, {"kind": "wait", "line": line}
    info = {"kind": "play", "line": line}
    st = game.state()
    hand = [int(c) for c in st["hand"][team]]
    ref = re.sub(r"\s*\(\d+\)\s*$", "", m.group("card").strip()).strip("[]() ")
    ref = re.sub(r"^slot\s*", "", ref, flags=re.I)
    if re.fullmatch(r"-?\d+", ref):
        slot = int(ref)
        if not 0 <= slot < 4:
            return 0, dict(info, error=f"bad slot {slot} (use 0-3)", error_kind="parse")
    else:
        try:
            c = card_id(ref)
        except ValueError:
            return 0, dict(info, error=f"unknown card {ref!r}", error_kind="parse")
        if c not in hand:
            return 0, dict(info, error=f"{CARD_NAMES[c]} is not in your hand", error_kind="parse")
        slot = hand.index(c)
    c = hand[slot]
    info.update(slot=slot, card=CARD_NAMES[c] if 0 <= c < N_CARDS else None)
    tx, ty = int(m.group("tx")), int(m.group("ty"))
    info.update(tx=tx, ty=ty)
    if not (0 <= tx < TILES_X and 0 <= ty < TILES_Y):
        return 0, dict(info, error=f"tile ({tx},{ty}) is outside the 18x32 arena", error_kind="illegal")
    action = 1 + slot * N_TILES + ty * TILES_X + tx
    if not game.legal_mask(team)[action]:
        if st["over"]:
            why = "the match is over"
        elif st["tick"] < st["lockout_ticks"]:
            why = "deploy lockout"
        elif 0 <= c < N_CARDS and st["elixir"][team] < CARD_COSTS[c] * ELIXIR_UNIT:
            why = "not enough elixir"
        else:
            why = f"({tx},{ty}) is not a legal tile for {info['card']}"
        return 0, dict(info, error=f"illegal: {why}", error_kind="illegal")
    return action, info


def describe_action(action: int, game: Optional[Game] = None, team: int = 0) -> str:
    """'WAIT' or 'PLAY <slot> AT <tx>,<ty>' (with the card name when a game is given)."""
    if action == 0:
        return "WAIT"
    slot, cell = divmod(int(action) - 1, N_TILES)
    tx, ty = cell % TILES_X, cell // TILES_X
    card = ""
    if game is not None:
        c = int(game.state()["hand"][team][slot])
        card = f" ({CARD_NAMES[c]})" if 0 <= c < N_CARDS else ""
    return f"PLAY {slot}{card} AT {tx},{ty}"


# ------------------------------------------------------------------------------------ agent
class LLMAgent:
    """A text agent: `model_fn(system, user) -> str` is called with RULES_PROMPT and the rendered
    state; the reply is parsed into an action. Every decision appends a transcript entry
    {tick, team, prompt_sha256, response, action, decoded, error, error_kind, latency_s, meta}
    (`meta` = the model function's `last_call` attribute, when it has one: e.g. the Anthropic
    adapter's stop reason, usage, refusal or API error). A model function that raises is a WAIT
    recorded with error_kind 'model'.

    decision_interval: the default cadence of this agent in play_llm_match (SPEC §17.4.10).
    max_history = k > 0 (SPEC §17.4.7) includes the last k (state, response) pairs as prior turns,
    embedded before the current state in the user text (model_fn is single-turn); 0 = stateless."""

    def __init__(self, model_fn: Callable[[str, str], str], decision_interval: int = 20, max_history: int = 0,
                 name: Optional[str] = None, legal: bool = True, keep_prompts: bool = False):
        if not callable(model_fn):
            raise TypeError("model_fn must be callable: model_fn(system, user) -> str")
        if int(decision_interval) < 1:
            raise ValueError("decision_interval must be >= 1")
        self.model_fn = model_fn
        self.decision_interval = int(decision_interval)
        self.max_history = max(0, int(max_history))
        self.name = name or getattr(model_fn, "__name__", type(model_fn).__name__)
        self.legal = bool(legal)
        self.keep_prompts = bool(keep_prompts)
        self.transcript: List[dict] = []
        self._turns: List[Tuple[str, str, Optional[str]]] = []    # (state text, response, rejection)

    def _history(self) -> str:
        past = self._turns[-self.max_history:]
        if not past:
            return ""
        out = [f"PREVIOUS TURNS (your last {len(past)}, oldest first):"]
        for k, (state, reply, err) in enumerate(past, 1):
            out.append(f"=== turn -{len(past) - k + 1}: state ===")
            out.append(state)
            out.append(f"=== turn -{len(past) - k + 1}: your reply ===")
            out.append(reply.strip() or "(empty)")
            if err:
                out.append(f"(treated as WAIT: {err})")
        out.append("=== CURRENT TURN ===")
        return "\n".join(out) + "\n"

    def prompt(self, game: Game, team: int) -> str:
        user = render_state(game, team, legal=self.legal)
        if self.max_history:
            user = self._history() + user
        return user

    def act(self, game: Game, team: int) -> int:
        user = self.prompt(game, team)
        t0 = time.perf_counter()
        model_error = None
        try:
            reply = self.model_fn(RULES_PROMPT, user)
        except Exception as e:  # noqa: BLE001 -- a failing model is a WAIT, recorded
            reply, model_error = "", f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - t0
        if model_error is None:
            action, info = parse_action(reply, game, team)
        else:
            action, info = 0, {"kind": None, "error": f"model_fn raised {model_error}", "error_kind": "model"}
        meta = getattr(self.model_fn, "last_call", None)
        entry = {
            "tick": int(game.state()["tick"]),
            "team": int(team),
            "prompt_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest()[:16],
            "response": reply if isinstance(reply, str) else str(reply),
            "action": int(action),
            "decoded": describe_action(action, game, team),
            "error": info.get("error"),
            "error_kind": info.get("error_kind"),
            "latency_s": round(latency, 6),
            "meta": dict(meta) if isinstance(meta, dict) else None,
        }
        if self.keep_prompts:
            entry["prompt"] = user
        self.transcript.append(entry)
        if self.max_history:
            self._turns.append((render_state(game, team, legal=self.legal), entry["response"], entry["error"]))
            del self._turns[:-self.max_history]
        return int(action)


# ------------------------------------------------------------------------------------ matches
class _BotOpponent:
    def __init__(self, kind, seed, interval):
        self.bot = _game.Bot(kind, seed=seed)
        self.interval = interval

    def act(self, game, team):
        return self.bot.act(game, team)


class _PolicyOpponent:
    def __init__(self, player, interval):
        self.player = player
        self.interval = interval

    def act(self, game, team):
        return self.player.act(game, team)


def _make_opponent(opponent, seed, team, interval, greedy, device):
    if isinstance(opponent, LLMAgent):
        return opponent, opponent.decision_interval
    if isinstance(opponent, str) and opponent.startswith("bot:"):
        kind = opponent[4:]
        if kind not in _game.BOT_KINDS:
            raise ValueError(f"unknown bot {opponent!r}")
        k = int(interval or 10)
        return _BotOpponent(kind, (int(seed) * 2 + team) & ((1 << 62) - 1), k), k
    from . import metagame
    if interval is None:                     # a checkpoint decides at the frame_skip it was trained with
        cfg = metagame.agent_train_config(opponent)
        interval = int(cfg["frame_skip"]) if cfg and cfg.get("frame_skip") else 10
    return _PolicyOpponent(metagame._player(opponent, seed, team, greedy, device), interval), int(interval)


def _next_multiple(t, k):
    return (t // k + 1) * k


def play_llm_match(agent: LLMAgent, opponent, deck_agent, deck_opp, seed: int, agent_team: int = 0,
                   decision_interval: Optional[int] = None, max_ticks: int = MAX_TICKS, *,
                   opponent_interval: Optional[int] = None,
                   greedy: bool = True, device: str = "cpu", tower_troop_agent: str = "princess",
                   tower_troop_opp: str = "princess", skip_idle: bool = False) -> dict:
    """One match through Game (SPEC §17.1). `agent` plays `agent_team` with `deck_agent` and decides
    at every tick that is a multiple of `decision_interval` (default: agent.decision_interval, 20
    ticks = 1 s; SPEC §17.4.10). `opponent`: a bot spec ("bot:heuristic" ...; acts every
    `opponent_interval` ticks, default 10 = the env cadence), a checkpoint path / "ckpt:<path>" /
    loaded policy (default cadence: the frame_skip it was trained with, else 10; greedy unless
    greedy=False), or another LLMAgent (at its own decision_interval). Plays are queued for the
    next tick.
    skip_idle=True skips the model call when the agent has no legal play at all (deploy lockout,
    nothing affordable) -- the only possible answer is then WAIT; counted as `skipped`.

    Returns {result (+1/0/-1 for the agent), crowns [agent, opponent], end_reason (None if cut
    by max_ticks), ticks, decisions (model calls), illegal, parse_errors, model_errors, plays,
    waits, skipped, agent_team, finished, mean_latency_s}."""
    agent_team = int(agent_team)
    if agent_team not in (0, 1):
        raise ValueError("agent_team must be 0 or 1")
    k = int(agent.decision_interval if decision_interval is None else decision_interval)
    if k < 1:
        raise ValueError("decision_interval must be >= 1")
    opp_team = 1 - agent_team
    decks = (deck_agent, deck_opp) if agent_team == 0 else (deck_opp, deck_agent)
    tts = (tower_troop_agent, tower_troop_opp) if agent_team == 0 else (tower_troop_opp, tower_troop_agent)
    g = Game(deck0=decks[0], deck1=decks[1], seed=int(seed), tower_troop0=tts[0], tower_troop1=tts[1])
    opp, ko = _make_opponent(opponent, seed, opp_team, opponent_interval, greedy, device)
    n0 = len(agent.transcript)
    skipped = 0
    max_ticks = min(int(max_ticks), MAX_TICKS)
    while True:
        st = g.state()
        t = int(st["tick"])
        if st["over"] or t >= max_ticks:
            break
        if t % k == 0:
            if skip_idle and not g.legal_mask(agent_team)[1:].any():
                skipped += 1
            else:
                a = agent.act(g, agent_team)
                if a:
                    g.play_action(agent_team, a)
        if t % ko == 0:
            b = opp.act(g, opp_team)
            if b:
                g.play_action(opp_team, b)
        g.tick(min(_next_multiple(t, k), _next_multiple(t, ko), max_ticks) - t)
    entries = agent.transcript[n0:]
    lat = [e["latency_s"] for e in entries]
    return {
        "result": int(st["result"][agent_team]),
        "crowns": [int(st["crowns"][agent_team]), int(st["crowns"][opp_team])],
        "end_reason": st["end_reason"],
        "ticks": int(st["tick"]),
        "decisions": len(entries),
        "illegal": sum(e["error_kind"] == "illegal" for e in entries),
        "parse_errors": sum(e["error_kind"] == "parse" for e in entries),
        "model_errors": sum(e["error_kind"] == "model" or bool(e["meta"] and (e["meta"].get("error") or
                                                                                e["meta"].get("refusal")))
                            for e in entries),
        "plays": sum(e["action"] != 0 for e in entries),
        "waits": sum(e["action"] == 0 and not e["error"] for e in entries),
        "skipped": skipped,
        "agent_team": agent_team,
        "finished": bool(st["over"]),
        "mean_latency_s": float(np.mean(lat)) if lat else 0.0,
    }


# ------------------------------------------------------------------------------------ mock model functions
def mock_wait(system: str, user: str) -> str:
    """Always WAIT."""
    return "WAIT"


def mock_first_legal(system: str, user: str) -> str:
    """Plays the first affordable card at its first legal tile, reading ONLY the rendered text
    (the YOUR HAND and LEGAL TILES sections); WAIT when nothing is playable."""
    legal = parse_rendered_legal(user)
    for h in parse_rendered_hand(user):
        if h["playable"] and legal.get(h["slot"]):
            tx, ty = legal[h["slot"]][0]
            return f"PLAY {h['slot']} AT {tx},{ty}"
    return "WAIT"


def mock_random(seed: int = 0) -> Callable[[str, str], str]:
    """A seeded text-only random player (SPEC §17.4.8): uniformly one of WAIT and every legal
    command listed in the text (each playable slot x each of its legal tiles)."""
    rng = random.Random(int(seed))

    def mock_random_fn(system: str, user: str) -> str:
        legal = parse_rendered_legal(user)
        playable = [h["slot"] for h in parse_rendered_hand(user) if h["playable"]]
        commands = ["WAIT"] + [f"PLAY {s} AT {tx},{ty}" for s in playable for tx, ty in legal.get(s, [])]
        return commands[rng.randrange(len(commands))]

    mock_random_fn.__name__ = f"mock_random(seed={seed})"
    return mock_random_fn


MOCKS = {"mock_wait": lambda seed=0: mock_wait, "mock_first_legal": lambda seed=0: mock_first_legal,
         "mock_random": lambda seed=0: mock_random(seed)}


# ------------------------------------------------------------------------------------ Anthropic adapter
FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5")   # get server-side refusal fallbacks
NO_EFFORT_MODELS = ("claude-haiku-4-5",)                    # output_config.effort unsupported


def anthropic_model_fn(model: str = "claude-opus-5-5", effort: Optional[str] = "low", max_tokens: int = 4096,
                       client=None) -> Callable[[str, str], str]:
    """A model_fn backed by the official `anthropic` SDK (SPEC §17.2; imported lazily, never used
    by the tests). Each call sends the static system prompt as a cached block and the state text
    as the single user message -- no assistant prefill; thinking stays at the model default (it
    cannot be disabled on claude-opus-5-5 / claude-fable-5-1; `effort` trades depth for latency
    and is omitted for claude-haiku-4-5). claude-fable-5-1 / claude-opus-5 requests go through
    client.beta.messages.create with server-side refusal fallbacks. A refusal
    (stop_reason == "refusal") or a failed call (RateLimitError / APIStatusError /
    APIConnectionError) returns "WAIT"; the details go to `fn.last_call`, which LLMAgent copies
    into the transcript. `client` may be an existing anthropic.Anthropic() instance."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("anthropic_model_fn needs the official Anthropic Python SDK, which is not installed "
                          "(pip install anthropic, then set ANTHROPIC_API_KEY or run `ant auth login`). "
                          "The mock model functions (mock_wait, mock_first_legal, mock_random) need nothing.") from e
    if client is None:
        client = anthropic.Anthropic()
    use_fallbacks = model in FALLBACK_MODELS
    send_effort = effort is not None and model not in NO_EFFORT_MODELS

    def fn(system: str, user: str) -> str:
        kwargs = dict(
            model=model,
            max_tokens=int(max_tokens),
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if send_effort:
            kwargs["output_config"] = {"effort": effort}
        call = {"model": model, "effort": effort if send_effort else None, "fallbacks": use_fallbacks}
        fn.last_call = call
        try:
            if use_fallbacks:
                resp = client.beta.messages.create(**kwargs, betas=["server-side-fallback-2026-07-01"],
                                                   fallbacks="default")
            else:
                resp = client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:          # a subclass of APIStatusError: catch it first
            call.update(error=f"rate_limit: {getattr(e, 'message', e)}", status=getattr(e, "status_code", 429))
            return "WAIT"
        except anthropic.APIStatusError as e:
            call.update(error=f"api_status {getattr(e, 'status_code', '?')}: {getattr(e, 'message', e)}",
                        status=getattr(e, "status_code", None))
            return "WAIT"
        except anthropic.APIConnectionError as e:
            call.update(error=f"connection: {e}")
            return "WAIT"
        call["stop_reason"] = getattr(resp, "stop_reason", None)
        call["served_by"] = getattr(resp, "model", None)
        call["request_id"] = getattr(resp, "_request_id", None)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            call["usage"] = {k: getattr(usage, k, None) for k in
                             ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        if call["stop_reason"] == "refusal":
            details = getattr(resp, "stop_details", None)
            call["refusal"] = {"category": getattr(details, "category", None),
                               "explanation": getattr(details, "explanation", None)}
            return "WAIT"
        text = "".join(getattr(b, "text", "") for b in (getattr(resp, "content", None) or [])
                       if getattr(b, "type", None) == "text")
        return text

    fn.last_call = None
    fn.__name__ = f"anthropic:{model}"
    return fn


def make_model_fn(spec: str, seed: int = 0, effort: Optional[str] = "low", max_tokens: int = 4096,
                  allow_network: bool = False) -> Callable[[str, str], str]:
    """'mock_wait' | 'mock_first_legal' | 'mock_random' | 'anthropic:<model-id>' -> a model_fn.
    Network-backed models are refused unless allow_network=True."""
    if spec in MOCKS:
        return MOCKS[spec](seed)
    if spec.startswith("anthropic:"):
        if not allow_network:
            raise PermissionError(f"{spec} would call the Anthropic API over the network; pass allow_network=True "
                                  f"(llm_match.py: --allow-network) to allow it")
        return anthropic_model_fn(spec.split(":", 1)[1], effort=effort, max_tokens=max_tokens)
    raise ValueError(f"unknown model {spec!r}: use {sorted(MOCKS)} or anthropic:<model-id>")


__all__ = ["RULES_PROMPT", "render_state", "parse_action", "LLMAgent", "play_llm_match", "mock_wait",
           "mock_first_legal", "mock_random", "anthropic_model_fn", "make_model_fn", "parse_rendered_hand",
           "parse_rendered_legal", "legal_regions", "describe_action", "CARD_ROLES", "KNOWN_ANTHROPIC_MODELS"]
