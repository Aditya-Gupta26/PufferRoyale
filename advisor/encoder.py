"""BoardState -> the SPEC §9 observation vector the policy was trained on (float32[17715]).

This mirrors pufferroyale/csrc/pr_obs.h (pr_obs_write) field by field, in float32 with the same operation
order, so a perfect BoardState gives a bit-identical vector (tests/test_encoder.py checks it against
Game.obs on simulated matches). Placement legality (the action mask and the troop / building legality
planes) is NOT re-implemented: it comes from a "shadow" engine game holding the same structures, towers,
hand and elixir, so it is the engine's own legality function.
"""
from __future__ import annotations

import numpy as np

from pufferroyale import binding as _binding
from pufferroyale.game import Game

from . import engine_db as D
from .board import BoardState

_L = _binding.royale_layout()
OBS_SIZE = _L["OBS_SIZE"]
SPATIAL_OFFSET, ENTITY_OFFSET = _L["SPATIAL_OFFSET"], _L["ENTITY_OFFSET"]
SCALAR_OFFSET, MASK_OFFSET = _L["SCALAR_OFFSET"], _L["MASK_OFFSET"]
N_CH = _L["SPATIAL_SHAPE"][0]
ENT_N, ENT_F = _L["ENTITY_SHAPE"]
ENT_HALF = ENT_N // 2
SC = {k: tuple(v) for k, v in _L["SCALAR_FIELDS"].items()}
CH = {name: i for i, name in enumerate(_L["SPATIAL_CHANNELS"])}
PER_SIDE = CH["enemy_ground"]                       # 10
PLANE = D.N_TILES
# per-side channel offsets (pr_obs.h PR_CH_*)
GROUND, AIR, BUILDING, TOWER, HP, BT, RANGED, SPLASH, DEPLOYING, DEBUFFED = range(10)
CLIP_CH = (GROUND, AIR, HP, BT, RANGED, SPLASH, DEPLOYING, DEBUFFED)

_F = np.float32
_Q = _F(0.25)


def _div(a, b) -> np.float32:
    """(float)a / (float)b in float32, as the C encoder computes it."""
    return _F(_F(a) / _F(b))


def _c_div(a: int, b: int) -> int:
    """C integer division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _tile(x: int, y: int) -> int:
    tx = min(max(_c_div(x, 1000), 0), D.TILES_X - 1)
    ty = min(max(_c_div(y, 1000), 0), D.TILES_Y - 1)
    return ty * D.TILES_X + tx


def _fill_rect(plane: np.ndarray, x0: int, y0: int, x1: int, y1: int, v) -> None:
    """Set every tile overlapped (positive area) by the rect to max(current, v) (pr_obs_fill_rect)."""
    tx0, tx1 = max(0, (x0 // 1000)), min(D.TILES_X - 1, (x1 - 1) // 1000)
    ty0, ty1 = max(0, (y0 // 1000)), min(D.TILES_Y - 1, (y1 - 1) // 1000)
    v = _F(v)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = ty * D.TILES_X + tx
            if plane[t] < v:
                plane[t] = v


def _fill_disc(plane: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Mark tiles whose centre lies within r of (cx, cy) (pr_obs_fill_disc)."""
    tx0, tx1 = max(0, (cx - r) // 1000), min(D.TILES_X - 1, (cx + r) // 1000)
    ty0, ty1 = max(0, (cy - r) // 1000), min(D.TILES_Y - 1, (cy + r) // 1000)
    if r < 0:
        return
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            dx, dy = cx - (tx * 1000 + 500), cy - (ty * 1000 + 500)
            if dx * dx + dy * dy <= r * r:
                plane[ty * D.TILES_X + tx] = _F(1.0)


def _card_id(c: int) -> np.float32:
    return _F(c + 1) if 0 <= c < D.N_CARDS else _F(0.0)


def _building_card(unit_name: str) -> int:
    for c in range(D.N_CARDS):
        if D.card_kind(c) == "building" and D.card_unit(c) == unit_name:
            return c
    raise ValueError(f"no building card deploys unit {unit_name!r}")


def shadow_legality(b: BoardState):
    """(mask uint8[2305], water, troop_legal, building_legal float32[576]) from an engine game holding the
    board's living towers and buildings and the viewer's deck, hand and elixir (viewer = engine team 0)."""
    deck = [int(c) for c in b.deck]
    if len(deck) != 8 or len(set(deck)) != 8 or not all(0 <= c < D.N_CARDS for c in deck):
        raise ValueError(f"the viewer's deck must be 8 distinct known cards, got {b.deck}")
    # deck order matters only for the building-legality plane (the deck's FIRST building, else Cannon)
    g = Game(deck0=deck, deck1="hog26", seed=0, deploy_lockout_ticks=0)
    for team, towers in ((0, b.own_towers), (1, b.enemy_towers)):
        for idx, t in enumerate(towers):          # own-frame [King, left, right] == engine idx for team 0
            if not t.alive and idx != 0:
                g.set_tower_hp(team, idx, 0)
    for u in b.units:
        if D.unit(u.unit).kind == "building" and u.hp > 0:
            card = u.card if 0 <= u.card < D.N_CARDS and D.card_kind(u.card) == "building" else _building_card(u.unit)
            g.spawn(0 if u.own else 1, card, u.x, u.y, deployed=True)
    o = g.obs(0)
    sp = o[SPATIAL_OFFSET:ENTITY_OFFSET].reshape(N_CH, PLANE)
    water, troop, building = sp[CH["water"]].copy(), sp[CH["own_troop_legal"]].copy(), sp[CH["own_building_legal"]].copy()

    # set_hand needs the 8 deck cards: real hand cards in their slots, any unused deck card as a filler for
    # an unknown slot (or a duplicate read), whose mask segment is then cleared
    hand = [int(c) for c in b.hand]
    fillers = [c for c in deck if c not in hand]
    order, unknown_slots = [], []
    for s, c in enumerate(hand):
        if c in deck and c not in order:
            order.append(c)
        else:
            order.append(fillers.pop(0))
            unknown_slots.append(s)
    order += [c for c in deck if c not in order]
    g.set_hand(0, order)
    g.set_elixir(0, int(b.elixir))
    mask = g.legal_mask(0).copy()
    for s in unknown_slots:  # the engine would refuse an unknown / empty slot (BAD_SLOT)
        mask[1 + s * D.N_TILES:1 + (s + 1) * D.N_TILES] = 0
    if b.tick < b.lockout_ticks or b.over:
        mask[1:] = 0
    return mask, water, troop, building


def encode(b: BoardState) -> np.ndarray:
    out = np.zeros(OBS_SIZE, np.float32)
    sp = out[SPATIAL_OFFSET:ENTITY_OFFSET].reshape(N_CH, PLANE)  # a view
    ent = out[ENTITY_OFFSET:SCALAR_OFFSET].reshape(ENT_N, ENT_F)
    sc = out[SCALAR_OFFSET:MASK_OFFSET]
    tower_units = D.tower_units()

    # ---- crown towers (they come first in the engine's entity array; fill_rect takes the max anyway) ----
    for side, towers, xy, tt in ((0, b.own_towers, D.OWN_TOWER_XY, b.tower_troops[0]),
                                 (PER_SIDE, b.enemy_towers, D.ENEMY_TOWER_XY, b.tower_troops[1])):
        for idx, t in enumerate(towers):
            if not t.alive:
                continue
            u = D.unit(tower_units["king"] if idx == 0 else tower_units[tt])
            h = u.footprint_tiles * 500
            ox, oy = xy[idx]
            _fill_rect(sp[side + TOWER], ox - h, oy - h, ox + h, oy + h, _div(t.hp, t.max_hp))

    # ---- troops and buildings: spatial counts and the entity list ----
    n_own = n_enemy = 0
    for e in b.units:
        if e.hp <= 0:
            continue
        d = D.unit(e.unit)
        side = 0 if e.own else PER_SIDE
        t = _tile(e.x, e.y)
        if d.kind == "building":
            h = d.footprint_tiles * 500
            _fill_rect(sp[side + BUILDING], e.x - h, e.y - h, e.x + h, e.y + h, 1.0)
        else:
            sp[side + (AIR if d.flying else GROUND)][t] += _Q
        sp[side + HP][t] += _div(e.hp, 4000)
        if d.only_buildings:
            sp[side + BT][t] += _Q
        if d.ranged:
            sp[side + RANGED][t] += _Q
        if D.unit_splash(d):
            sp[side + SPLASH][t] += _Q
        if e.inactive:
            sp[side + DEPLOYING][t] += _Q
        if e.stunned or e.slowed:
            sp[side + DEBUFFED][t] += _Q
        if e.own:
            if n_own >= ENT_HALF:
                continue
            slot, n_own = n_own, n_own + 1
        else:
            if n_enemy >= ENT_HALF:
                continue
            slot, n_enemy = ENT_HALF + n_enemy, n_enemy + 1
        f = ent[slot]
        f[0] = _card_id(e.card)
        f[1] = _div(e.x, D.ARENA_W)
        f[2] = _div(e.y, D.ARENA_H)
        f[3] = _div(e.hp, e.max_hp)
        f[4] = min(_F(1.0), _div(e.hp, 2000))
        f[5] = _F(d.flying)
        f[6] = _F(e.inactive)
        f[7] = _F(e.stunned)
        f[8] = _F(e.slowed)
        f[9] = _F(d.kind == "building")
        f[10] = _F(d.only_buildings)
    for s in (0, PER_SIDE):
        for ch in CLIP_CH:
            np.minimum(sp[s + ch], _F(1.0), out=sp[s + ch])

    # ---- static / legality planes and the mask (the engine's own legality via a shadow game) ----
    mask, water, troop, building = shadow_legality(b)
    sp[CH["water"]] = water
    sp[CH["own_troop_legal"]] = troop
    sp[CH["own_building_legal"]] = building

    # ---- active spell areas ----
    for a in b.spells:
        pl = sp[CH["own_spells"] if a.own else CH["enemy_spells"]]
        if a.shape == "disc":
            _fill_disc(pl, a.a, a.b, a.c)
        else:
            _fill_rect(pl, a.a, a.b, a.c, a.d, 1.0)

    # ---- scalars (pr_obs.h order) ----
    def put(name, values):
        off, n = SC[name]
        vals = np.asarray(values, np.float32).ravel()
        assert vals.size == n, (name, vals.size, n)
        sc[off:off + n] = vals

    put("elixir", [_div(b.elixir, D.ELIXIR_MAX)])
    hand_ids, costs, afford = [], [], []
    for c in b.hand:
        hand_ids.append(_card_id(c))
        if 0 <= c < D.N_CARDS:
            costs.append(_div(D.CARD_COSTS[c], 10))
            afford.append(_F(b.elixir >= D.CARD_COSTS[c] * D.ELIXIR_UNIT))
        else:
            costs.append(_F(0))
            afford.append(_F(0))
    put("hand", hand_ids)
    put("hand_cost", costs)
    put("next_card", [_card_id(b.next_card)])
    put("affordable", afford)
    put("tick", [_div(b.tick, D.TICKS_MAX)])
    put("overtime", [_F(b.tick >= D.TICKS_REGULATION)])
    put("elixir_rate", [_div(D.elixir_rate(b.tick), 3 * D.ELIXIR_RATE_1X)])
    put("lockout", [_F(b.tick < b.lockout_ticks)])
    put("own_towers", [_div(t.hp, t.max_hp) if t.alive else _F(0) for t in b.own_towers])
    put("enemy_towers", [_div(t.hp, t.max_hp) if t.alive else _F(0) for t in b.enemy_towers])
    put("king_active", [_F(b.king_active[0]), _F(b.king_active[1])])
    put("crowns", [_div(b.crowns[0], 3), _div(b.crowns[1], 3)])
    seen = np.zeros(D.CARD_SLOTS, np.float32)
    for c in b.opp_seen:
        if 0 <= c < D.N_CARDS:
            seen[c] = 1.0
    put("opp_seen", seen)
    put("opp_spent", [_F(_div(b.opp_spent, D.ELIXIR_UNIT) / _F(200.0))])
    put("opp_last4", [_card_id(c) for c in b.opp_last4])
    deduced = np.zeros(D.CARD_SLOTS, np.float32)
    known_seen = [c for c in b.opp_seen if 0 <= c < D.N_CARDS]
    if len(known_seen) == 8:  # all 8 revealed: the hand is the 4 not among the last 4 played
        for c in known_seen:
            if c not in b.opp_last4:
                deduced[c] = 1.0
    put("opp_deduced_hand", deduced)
    ub = min(max(D.ELIXIR_START + D.income(b.tick) - b.opp_spent, 0), D.ELIXIR_MAX)
    put("opp_elixir_ub", [_div(ub, D.ELIXIR_MAX)])
    tt = np.zeros(len(D.TOWER_TROOPS), np.float32)
    tt[min(max(b.tower_troops[0], 0), len(tt) - 1)] = 1.0
    put("own_tower_troop", tt)
    tt = np.zeros(len(D.TOWER_TROOPS), np.float32)
    tt[min(max(b.tower_troops[1], 0), len(tt) - 1)] = 1.0
    put("enemy_tower_troop", tt)
    put("own_deck", [_card_id(c) for c in sorted(b.deck)])

    out[MASK_OFFSET:] = mask.astype(np.float32)
    return out
