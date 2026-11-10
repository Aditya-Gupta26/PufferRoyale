"""Facts the observation encoder needs about units and projectiles, read from the engine's generated card
DB (pufferroyale/csrc/pr_card_db.h) so they always match the compiled engine; plus a few engine constants.

The header holds designated-initializer tables (`{.name = "Knight", .kind = PR_KIND_TROOP, ...}`); only the
fields listed in _UNIT_FIELDS / _PROJ_FIELDS are kept. tests check every value against the live engine
(card_info / entities), so a parse error cannot pass silently.
"""
from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass

import pufferroyale
from pufferroyale import game as _game

# ---- engine constants (pr_types.h / pr_arena_db.h; asserted against the binding in tests) ----
ARENA_W, ARENA_H = 18000, 32000
TILES_X, TILES_Y = 18, 32
N_TILES = TILES_X * TILES_Y
N_ACTIONS = 1 + 4 * N_TILES
ELIXIR_UNIT = 2800          # engine units per elixir
ELIXIR_START = 16800        # 6 elixir (ledger match.START_MANA)
ELIXIR_MAX = 28000
ELIXIR_RATE_1X = 50         # units per tick
TICK_2X, TICK_3X = 2400, 4800
TICKS_REGULATION, TICKS_MAX = 3600, 6000
MELEE_RANGE = 1900          # range <= this is melee (ranged plane counts range > MELEE_RANGE)
TICK_MS = 50
CARD_SLOTS = _game.CARD_SLOTS
N_CARDS = _game.N_CARDS
CARD_NAMES = _game.CARD_NAMES
CARD_COSTS = _game.CARD_COSTS
DECKS = _game.DECKS
TOWER_TROOPS = _game.TOWER_TROOPS

# Crown towers in the viewer's OWN frame: [King, own-frame left, own-frame right] (SPEC §9.3 order).
OWN_TOWER_XY = ((9000, 29000), (3500, 25500), (14500, 25500))
ENEMY_TOWER_XY = ((9000, 3000), (3500, 6500), (14500, 6500))

_UNIT_FIELDS = ("kind", "only_buildings", "projectile", "hp", "shield", "range", "flying_height",
                "area_radius", "footprint_tiles", "radius")
_PROJ_FIELDS = ("radius",)


@dataclass(frozen=True)
class UnitDef:
    name: str
    kind: str              # 'troop' | 'building' | 'tower'
    only_buildings: bool
    projectile: int        # PR_PROJS index, -1 = melee / instant hit
    hp: int                # level-11 hitpoints
    shield: int
    range: int             # millitiles
    flying: bool           # flying_height > 0
    area_radius: int       # melee splash radius
    footprint_tiles: int   # buildings / towers; 0 for troops
    radius: int            # collision radius
    speed: int = 0         # millitiles per tick
    deploy_ms: int = 0
    spawn_unit: str = ""   # periodic spawner product (Witch -> Skeleton), "" = none
    dspawn_unit: str = ""  # death-spawn product (Golem -> Golemite), "" = none

    @property
    def ranged(self) -> bool:
        return self.range > MELEE_RANGE


def card_db_path() -> str:
    return os.path.join(os.path.dirname(pufferroyale.__file__), "csrc", "pr_card_db.h")


def _table_rows(text: str, decl: str) -> list[str]:
    """Top-level `{...}` rows of the C array initializer that starts with `decl` (brace-depth scan, so
    nested braces such as `.seq_mult = {100, 100, 70, 90}` stay inside their row)."""
    start = text.index(decl)
    i = text.index("{", start + len(decl))
    depth, rows, row_start = 0, [], None
    while True:
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth == 2:
                row_start = i
        elif ch == "}":
            depth -= 1
            if depth == 1:
                rows.append(text[row_start + 1:i])
            elif depth == 0:
                return rows
        i += 1


_FIELD_RE = re.compile(r"\.(\w+)\s*=\s*(\"[^\"]*\"|\{[^}]*\}|'[^']*'|[^,]+)")


def _fields(row: str) -> dict:
    return {k: v.strip() for k, v in _FIELD_RE.findall(row)}


_ENUM_RE = re.compile(r"^\s*(PR_[A-Z0-9_]+)\s*=\s*(-?\d+)\s*,?\s*$", re.M)


@functools.lru_cache(maxsize=None)
def tables() -> tuple[dict, list]:
    """(units: name -> UnitDef, proj_radius: list indexed by PR_PROJS index)."""
    with open(card_db_path()) as f:
        text = f.read()
    enums = {k: int(v) for k, v in _ENUM_RE.findall(text)}  # PR_PROJ_* / PR_UNIT_* / PR_AREA_* indices

    def _int(v: str) -> int:
        return enums[v] if v in enums else int(v, 0)

    kinds = {"PR_KIND_TROOP": "troop", "PR_KIND_BUILDING": "building", "PR_KIND_TOWER": "tower"}
    unit_rows = [_fields(row) for row in _table_rows(text, "static const PrUnitDef PR_UNITS[PR_N_UNITS] =")]
    names = [f["name"].strip('"') for f in unit_rows]  # row index == PR_UNIT_* value (checked below)
    for i, n in enumerate(names):
        sym = "PR_UNIT_" + re.sub(r"[^A-Z0-9]", "", n.upper())
        if sym in enums and enums[sym] != i:
            raise ValueError(f"{sym} = {enums[sym]} but it is row {i} of PR_UNITS")

    def _unit_ref(v: str) -> str:
        i = _int(v)
        return names[i] if i >= 0 else ""

    units = {}
    for name, f in zip(names, unit_rows):
        units[name] = UnitDef(
            name=name, kind=kinds[f["kind"]], only_buildings=bool(_int(f["only_buildings"])),
            projectile=_int(f["projectile"]), hp=_int(f["hp"]), shield=_int(f["shield"]), range=_int(f["range"]),
            flying=_int(f["flying_height"]) > 0, area_radius=_int(f["area_radius"]),
            footprint_tiles=_int(f["footprint_tiles"]), radius=_int(f["radius"]), speed=_int(f["speed"]),
            deploy_ms=_int(f["deploy_ms"]), spawn_unit=_unit_ref(f["spawn_unit"]),
            dspawn_unit=_unit_ref(f["dspawn_unit"]))
    proj_rows = _table_rows(text, "static const PrProjDef PR_PROJS[PR_N_PROJS] =")
    proj_radius = [_int(_fields(row)["radius"]) for row in proj_rows]
    # the enum order must be the table order (PR_PROJ_<NAME> = row index)
    for i, row in enumerate(proj_rows):
        sym = "PR_PROJ_" + re.sub(r"[^A-Z0-9]", "", _fields(row)["name"].strip('"').upper())
        if sym in enums and enums[sym] != i:
            raise ValueError(f"{sym} = {enums[sym]} but it is row {i} of PR_PROJS")
    return units, proj_radius


def unit(name: str) -> UnitDef:
    return tables()[0][name]


def unit_splash(u: UnitDef) -> bool:
    """The observation's splash flag (pr_obs.h): melee splash, or a projectile with a radius."""
    proj_radius = tables()[1]
    return u.area_radius > 0 or (u.projectile >= 0 and proj_radius[u.projectile] > 0)


@functools.lru_cache(maxsize=None)
def card_info(card: int) -> dict:
    return _game.card_info(card)


def card_kind(card: int) -> str:
    """'troop' | 'building' | 'spell' (the card's kind, not its unit's)."""
    return card_info(card)["card_kind"]


def card_unit(card: int) -> str | None:
    """Engine unit name a troop / building card deploys (its main unit), None for spells."""
    return card_info(card)["unit"]


def tower_units() -> dict:
    """Tower-troop index -> the engine unit name of its Princess-tower replacement, plus 'king'."""
    names = [d["name"] for d in _game._DB["tower_troop_units"]]
    return {"king": "KingTower", **{i: n for i, n in enumerate(names)}}


@functools.lru_cache(maxsize=None)
def play_compositions() -> dict:
    """card id -> {unit name: count} of the units a play of that card puts on the board, measured on the
    engine: the card is spawned alone on an empty board and every new entity within 1.5 s is counted
    (Goblin Barrel / Barbarian Barrel land after a flight / roll). Periodic-spawner products (the Witch's
    first Skeleton wave, the Night Witch's first Bats) are removed: they are spawner output, not the play.
    Spells without units map to {}."""
    from collections import Counter
    out = {}
    for c in range(N_CARDS):
        g = _game.Game(deck0="hog26", deck1="giant", seed=0, deploy_lockout_ticks=0)
        seen = {e["id"] for e in g.entities()}
        g.spawn(1, c, 9000, 12000, deployed=False)
        found = Counter()
        for _ in range(30):
            for e in g.entities():
                if e["id"] not in seen:
                    seen.add(e["id"])
                    found[e["unit"]] += 1
            g.tick(1)
        main = card_unit(c)
        if main and unit(main).spawn_unit:
            found.pop(unit(main).spawn_unit, None)
        out[c] = dict(found)
    return out


@functools.lru_cache(maxsize=None)
def unit_cards() -> dict:
    """unit name -> card ids whose PLAY puts that unit on the board (ascending)."""
    out = {}
    for c, comp in play_compositions().items():
        for u in comp:
            out.setdefault(u, []).append(c)
    return out


def elixir_rate(tick: int) -> int:
    """Regen units per tick (pr_rules.h pr_elixir_rate)."""
    if tick < TICK_2X:
        return ELIXIR_RATE_1X
    if tick < TICK_3X:
        return 2 * ELIXIR_RATE_1X
    return 3 * ELIXIR_RATE_1X


def income(tick: int) -> int:
    """Total elixir units regenerated by the start of `tick` (pr_obs.h pr_obs_income)."""
    a = min(tick, TICK_2X)
    b = max(0, min(tick, TICK_3X) - TICK_2X)
    c = max(0, tick - TICK_3X)
    return a * ELIXIR_RATE_1X + b * 2 * ELIXIR_RATE_1X + c * 3 * ELIXIR_RATE_1X
