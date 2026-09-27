"""Independent reference computations for the spec suite.

Nothing here reads the builder's code. Stats come from data/source/cards-15.535.json with the
SPEC §2 level-11 rule, geometry from data/source/royalesim-arena.json with the SPEC §3 rules.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "source")

# ---------------------------------------------------------------------------------------------
# Constants from SPEC
# ---------------------------------------------------------------------------------------------
TICK_MS = 50
ELIXIR_UNIT = 2800              # 1 elixir = 2800 units (§4)
START_ELIXIR = 16800            # 6 elixir (§4)
MAX_ELIXIR = 28000              # 10 elixir (§4)
LOCKOUT_TICKS = 90              # default deploy_lockout_ticks (§4)
REGULATION_TICKS = 3600
MATCH_TICKS = 6000
KING_ACTIVATION_TICKS = 71      # 3550 ms (§4)
N_ACTIONS = 2305                # 1 + 4*576 (§9)
N_CELLS = 576
ARENA_W, ARENA_H = 18000, 32000

# PlayError codes (§10)
OK, NOT_IN_HAND, NOT_ENOUGH_ELIXIR, ILLEGAL_POSITION, LOCKOUT, GAME_OVER = 0, 1, 2, 3, 4, 5

# (id, display name, source key, kind, count) -- SPEC §0 table
CARDS = [
    (0, "Knight", "Knight", "troop"),
    (1, "Archers", "Archer", "troop"),
    (2, "Musketeer", "Musketeer", "troop"),
    (3, "Giant", "Giant", "troop"),
    (4, "Hog Rider", "HogRider", "troop"),
    (5, "Minions", "Minions", "troop"),
    (6, "Baby Dragon", "BabyDragon", "troop"),
    (7, "Valkyrie", "Valkyrie", "troop"),
    (8, "Skeleton Army", "SkeletonArmy", "troop"),
    (9, "Skeletons", "Skeletons", "troop"),
    (10, "Ice Golem", "IceGolemite", "troop"),
    (11, "Ice Spirit", "IceSpirits", "troop"),
    (12, "Prince", "Prince", "troop"),
    (13, "Wizard", "Wizard", "troop"),
    (14, "Cannon", "Cannon", "building"),
    (15, "Tesla", "Tesla", "building"),
    (16, "Fireball", "Fireball", "spell"),
    (17, "Arrows", "Arrows", "spell"),
    (18, "Zap", "Zap", "spell"),
    (19, "The Log", "Log", "spell"),
    (20, "Goblin Barrel", "GoblinBarrel", "spell"),
]
KNIGHT, ARCHERS, MUSKETEER, GIANT, HOG, MINIONS, BABY_DRAGON, VALKYRIE, SKARMY, SKELETONS, \
    ICE_GOLEM, ICE_SPIRIT, PRINCE, WIZARD, CANNON, TESLA, FIREBALL, ARROWS, ZAP, LOG, \
    GOBLIN_BARREL = range(21)
CARD_NAMES = tuple(c[1] for c in CARDS)
CARD_KIND = {c[0]: c[3] for c in CARDS}
TROOPS = [c[0] for c in CARDS if c[3] == "troop"]
BUILDINGS = [CANNON, TESLA]
SPELLS = [FIREBALL, ARROWS, ZAP, LOG, GOBLIN_BARREL]

# SPEC §4.1 cost table (elixir)
COST = {KNIGHT: 3, ARCHERS: 3, MUSKETEER: 4, GIANT: 5, HOG: 4, MINIONS: 3, BABY_DRAGON: 4,
        VALKYRIE: 4, SKARMY: 3, SKELETONS: 1, ICE_GOLEM: 2, ICE_SPIRIT: 1, PRINCE: 5, WIZARD: 5,
        CANNON: 3, TESLA: 4, FIREBALL: 4, ARROWS: 3, ZAP: 2, LOG: 2, GOBLIN_BARREL: 3}

# SPEC §0 preset decks
DECKS = {
    "hog26": [HOG, MUSKETEER, CANNON, ICE_GOLEM, ICE_SPIRIT, SKELETONS, FIREBALL, LOG],
    "giant": [GIANT, PRINCE, BABY_DRAGON, WIZARD, MINIONS, KNIGHT, ARROWS, ZAP],
    "bait": [GOBLIN_BARREL, SKARMY, TESLA, VALKYRIE, ARCHERS, KNIGHT, LOG, FIREBALL],
}

# Crown towers (§3). index 0 King, 1 Left (x 3500), 2 Right (x 14500), absolute engine x.
TOWER_POS = {
    0: [(9000, 29000), (3500, 25500), (14500, 25500)],
    1: [(9000, 3000), (3500, 6500), (14500, 6500)],
}
TOWER_RADIUS = [1400, 1000, 1000]
TOWER_F = [4, 3, 3]
PRINCESS_HP, KING_HP = 3052, 4824
TOWER_MAX_HP = [KING_HP, PRINCESS_HP, PRINCESS_HP]
TOWER_DMG = 109
POCKET_RECT_TILES = [(18, 16), (11, 21)]    # King, Princess (W x H tiles), closed rect (§3.1)


# ---------------------------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------------------------
_CACHE = {}


def card_data():
    if "cards" not in _CACHE:
        with open(os.path.join(DATA, "cards-15.535.json")) as f:
            _CACHE["cards"] = json.load(f)
    return _CACHE["cards"]


def src_card(card_id):
    key = CARDS[card_id][2]
    for c in card_data()["cards"]:
        if c["name"] == key:
            return c
    raise KeyError(key)


def src_unit(name):
    return card_data()["units"][name]


def arena_grid():
    if "arena" not in _CACHE:
        with open(os.path.join(DATA, "royalesim-arena.json")) as f:
            a = json.load(f)
        g = np.array(a["grid"], dtype=np.int32)
        assert g.shape == (64, 36), g.shape
        _CACHE["arena"] = g
    return _CACHE["arena"]


WATER_BIT, NODEPLOY_BIT = 32, 16


def L11(base):
    """SPEC §2: level-11 value of a scaling stat = floor(base * 256 / 100)."""
    return (int(base) * 256) // 100


def ct_damage(d, pct):
    """SPEC §2: crown-tower reduced damage = ceil(D * P / 100) for P < 100."""
    if pct >= 100:
        return d
    return -((-d * pct) // 100)


def ms_to_ticks(ms):
    """SPEC §1: ticks = ms / 50, rounded up if not a multiple."""
    return -((-int(ms)) // TICK_MS)


def expected_unit_stats(card_id):
    """Independent level-11 stats for the unit a troop/building card deploys."""
    c = src_card(card_id)
    st = dict(
        hp=L11(c["hitpoints"]),
        damage=L11(c["damage"]) if c.get("damage") is not None else None,
        hit_speed_ms=c["hit_speed_ms"],
        load_time_ms=c["load_time_ms"],
        range=c["range_milli"],
        sight=c["sight_range_milli"],
        speed=c.get("speed") or 0,
        radius=c["collision_radius_milli"],
        count=c["count"],
        deploy_time_ms=c["deploy_time_ms"],
        flying=(c.get("flying_height") or 0) > 0,
        attacks_air=bool(c["attacks_air"]),
        attacks_ground=bool(c["attacks_ground"]),
        target_only_buildings=bool(c["target_only_buildings"]),
        lifetime_ms=c.get("lifetime_ms"),
        mass=c.get("mass"),
    )
    if c.get("projectile") and c["projectile"].get("damage") is not None:
        st["damage"] = L11(c["projectile"]["damage"])
        st["projectile_speed"] = c["projectile"]["speed"]
    return st


GOBLIN = dict(hp=L11(79), damage=L11(49), speed=120, range=500, deploy_ms=1100, radius=500)


# ---------------------------------------------------------------------------------------------
# Frames (§1): team 1's own frame is the 180-degree rotation of the engine frame
# ---------------------------------------------------------------------------------------------
def own_to_engine_point(team, x, y):
    return (x, y) if team == 0 else (ARENA_W - x, ARENA_H - y)


engine_to_own_point = own_to_engine_point  # the rotation is an involution


def own_to_engine_tile(team, tx, ty):
    return (tx, ty) if team == 0 else (17 - tx, 31 - ty)


def tile_centre_engine(team, tx, ty):
    etx, ety = own_to_engine_tile(team, tx, ty)
    return etx * 1000 + 500, ety * 1000 + 500


def action_id(slot, tx, ty):
    return 1 + slot * N_CELLS + ty * 18 + tx


def decode_action(a):
    assert 1 <= a < N_ACTIONS
    slot, cell = divmod(a - 1, N_CELLS)
    return slot, cell % 18, cell // 18


def mirror_point(x, y):
    return ARENA_W - x, ARENA_H - y


# ---------------------------------------------------------------------------------------------
# Arena geometry oracle (§3)
# ---------------------------------------------------------------------------------------------
def point_in_water(x, y):
    """Engine point (millitiles) lies on a water half-cell."""
    hc, hr = int(x) // 500, int(y) // 500
    if not (0 <= hc < 36 and 0 <= hr < 64):
        return False
    return bool(arena_grid()[hr, hc] & WATER_BIT)


def tile_bits_engine(etx, ety):
    g = arena_grid()
    return int(g[2 * ety, 2 * etx] | g[2 * ety, 2 * etx + 1] | g[2 * ety + 1, 2 * etx] |
               g[2 * ety + 1, 2 * etx + 1])


def tile_is_water_engine(etx, ety):
    return bool(tile_bits_engine(etx, ety) & WATER_BIT)


def footprint(cx, cy, F):
    h = F * 500
    return (cx - h, cy - h, cx + h, cy + h)


def overlap_pos(a, b):
    """Positive-area overlap of two half-open rects (x0, y0, x1, y1)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def F_of_radius(radius):
    """SPEC §3: F = ceil((2*radius + 1000) / 1000) tiles."""
    return -((-(2 * radius + 1000)) // 1000)


def all_alive():
    return {0: [True, True, True], 1: [True, True, True]}


def tile_rect_engine(team, tx, ty):
    etx, ety = own_to_engine_tile(team, tx, ty)
    return (etx * 1000, ety * 1000, etx * 1000 + 1000, ety * 1000 + 1000)


def rotate_rect(team, r):
    """Own-frame half-open rect -> engine-frame half-open rect."""
    if team == 0:
        return r
    return (ARENA_W - r[2], ARENA_H - r[3], ARENA_W - r[0], ARENA_H - r[1])


def territory(team, tx, ty, alive):
    """SPEC §3.1 troop territory: own half (ty >= 17) or an opened pocket."""
    if ty >= 17:
        return True
    etx, ety = own_to_engine_tile(team, tx, ty)
    cx, cy = etx * 1000 + 500, ety * 1000 + 500
    if 15000 <= cy < 17000:
        return False
    enemy = 1 - team
    for i in range(3):
        if alive[enemy][i]:
            tx0, ty0 = TOWER_POS[enemy][i]
            w, h = POCKET_RECT_TILES[0 if i == 0 else 1]
            if (tx0 - w * 500 <= cx <= tx0 + w * 500) and (ty0 - h * 500 <= cy <= ty0 + h * 500):
                return False
    return True


def troop_tile_legal(team, tx, ty, alive, buildings=()):
    """SPEC v0.2 §3.1 rule 1 (own frame). `buildings`: (engine cx, cy, F) of every LIVING building
    of either team."""
    etx, ety = own_to_engine_tile(team, tx, ty)
    bits = tile_bits_engine(etx, ety)
    if bits & (WATER_BIT | NODEPLOY_BIT):
        return False
    trect = tile_rect_engine(team, tx, ty)
    for t in (0, 1):
        for i in range(3):
            if alive[t][i]:
                cx, cy = TOWER_POS[t][i]
                if overlap_pos(trect, footprint(cx, cy, TOWER_F[i])):
                    return False
    for (bx, by, bF) in buildings:
        if overlap_pos(trect, footprint(bx, by, bF)):
            return False
    return territory(team, tx, ty, alive)


def building_centre_own(tx, ty, F):
    """SPEC v0.2 §3.1 rule 2 anchor: odd F -> tile centre; even F -> tile's own-frame top-left."""
    if F % 2 == 1:
        return tx * 1000 + 500, ty * 1000 + 500
    return tx * 1000, ty * 1000


def building_centre_engine(team, tx, ty, card_id):
    F = F_of_radius(expected_unit_stats(card_id)["radius"])
    return own_to_engine_point(team, *building_centre_own(tx, ty, F))


def building_tile_legal(team, tx, ty, card_id, alive, buildings=()):
    """SPEC v0.2 §3.1 rule 2 (+ rule 6: destroyed towers block nothing)."""
    if ty < 17:
        return False
    F = F_of_radius(expected_unit_stats(card_id)["radius"])
    orect = footprint(*building_centre_own(tx, ty, F), F)
    if orect[0] < 0 or orect[2] > ARENA_W or orect[1] < 17000 or orect[3] > ARENA_H:
        return False
    erect = rotate_rect(team, orect)
    g = arena_grid()
    for hr in range(erect[1] // 500, erect[3] // 500):
        for hc in range(erect[0] // 500, erect[2] // 500):
            if g[hr, hc] & (WATER_BIT | NODEPLOY_BIT):
                return False
    for t in (0, 1):
        for i in range(3):
            if alive[t][i]:
                tcx, tcy = TOWER_POS[t][i]
                if overlap_pos(erect, footprint(tcx, tcy, TOWER_F[i])):
                    return False
    for (bx, by, bF) in buildings:
        if overlap_pos(erect, footprint(bx, by, bF)):
            return False
    return True


def log_tile_legal(team, tx, ty, alive):
    """SPEC v0.2 §3.1 rule 5: troop territory, not water; buildings/towers/nodeploy allowed."""
    etx, ety = own_to_engine_tile(team, tx, ty)
    if tile_is_water_engine(etx, ety):
        return False
    return territory(team, tx, ty, alive)


def spell_tile_legal(card_id, team, tx, ty, alive=None):
    if card_id == GOBLIN_BARREL:
        etx, ety = own_to_engine_tile(team, tx, ty)
        return not tile_is_water_engine(etx, ety)
    if card_id == LOG:
        return log_tile_legal(team, tx, ty, alive if alive is not None else all_alive())
    return True


def tile_legal(card_id, team, tx, ty, alive, buildings=()):
    kind = CARD_KIND[card_id]
    if kind == "troop":
        return troop_tile_legal(team, tx, ty, alive, buildings)
    if kind == "building":
        return building_tile_legal(team, tx, ty, card_id, alive, buildings)
    return spell_tile_legal(card_id, team, tx, ty, alive)


def expected_mask(hand, elixir, tick, team, alive, lockout=LOCKOUT_TICKS, over=False, buildings=()):
    """Reference legality mask (SPEC v0.2 §3.1 + play conditions)."""
    m = np.zeros(N_ACTIONS, dtype=np.int8)
    m[0] = 1
    if over or tick < lockout:
        return m
    for slot, c in enumerate(hand):
        if elixir < COST[c] * ELIXIR_UNIT:
            continue
        for ty in range(32):
            for tx in range(18):
                m[action_id(slot, tx, ty)] = int(bool(tile_legal(c, team, tx, ty, alive, buildings)))
    return m


def isqrt_dist(a, b):
    dx, dy = int(a[0]) - int(b[0]), int(a[1]) - int(b[1])
    return math.isqrt(dx * dx + dy * dy)
