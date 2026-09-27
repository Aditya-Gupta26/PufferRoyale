"""Thin, tolerant accessors over the SPEC §10 `pufferroyale.Game` API plus scenario builders.

Where SPEC §10 leaves a container shape open (e.g. `result` per team as a list vs a scalar),
the accessor accepts the plausible shapes and fails loudly otherwise.
"""
import numpy as np

import helpers as H


def mod():
    import pufferroyale
    return pufferroyale


def new_game(deck0="hog26", deck1="hog26", seed=0, **kw):
    return mod().Game(deck0=deck0, deck1=deck1, seed=seed, **kw)


def st(g):
    s = g.state()
    assert isinstance(s, dict), f"Game.state() must return a dict, got {type(s)}"
    return s


def tick_of(g):
    return int(st(g)["tick"])


def _per_team(v, team, name):
    try:
        return v[team]
    except Exception as e:  # pragma: no cover - diagnostic
        raise AssertionError(f"state()['{name}'] must be indexable per team, got {v!r}") from e


def elixir(g, team):
    return int(_per_team(st(g)["elixir"], team, "elixir"))


def hand(g, team):
    return [int(c) for c in _per_team(st(g)["hand"], team, "hand")]


def queue(g, team):
    return [int(c) for c in _per_team(st(g)["queue"], team, "queue")]


def crowns(g):
    c = st(g)["crowns"]
    return int(c[0]), int(c[1])


def tower(g, team, idx):
    t = _per_team(st(g)["towers"], team, "towers")[idx]
    return t


def tower_hp(g, team, idx):
    return int(tower(g, team, idx)["hp"])


def tower_alive(g, team, idx):
    return bool(tower(g, team, idx)["alive"])


def tower_active(g, team, idx):
    return bool(tower(g, team, idx)["active"])


def living_buildings(g):
    """(engine cx, cy, F) of every living building of either team (SPEC §3.1 footprints)."""
    return [(int(e["x"]), int(e["y"]), H.F_of_radius(int(e["radius"])))
            for e in ents(g) if e["kind"] == "building"]


def alive_map(g):
    return {t: [tower_alive(g, t, i) for i in range(3)] for t in (0, 1)}


def over(g):
    return bool(st(g)["over"])


def result(g, team):
    """SPEC §13.8: state()['result'] == [r0, r1] with r in {+1, -1, 0}."""
    r = st(g)["result"]
    assert isinstance(r, (list, tuple)) and len(r) == 2, f"result must be [r0, r1], got {r!r}"
    assert int(r[0]) in (-1, 0, 1) and int(r[0]) == -int(r[1]), f"result {r!r} not zero-sum"
    return int(r[team])


END_REASONS = ("KING", "REGULATION_CROWNS", "OVERTIME_CROWNS", "TIEBREAK", "DRAW")


def end_reason(g):
    """SPEC §13.8: None while running, else one of the five strings."""
    r = st(g)["end_reason"]
    if not over(g):
        assert r is None, f"end_reason must be None while running, got {r!r}"
        return None
    assert isinstance(r, str) and r in END_REASONS, f"end_reason {r!r} not one of {END_REASONS}"
    return r


def ents(g):
    e = g.entities()
    assert isinstance(e, list), "Game.entities() must return a list of dicts"
    return e


def ent(g, eid):
    for e in ents(g):
        if int(e["id"]) == int(eid):
            return e
    return None


def hp_of(g, eid):
    e = ent(g, eid)
    return None if e is None else int(e["hp"])


def units(g, team=None, card=None, kind=None):
    out = []
    for e in ents(g):
        if team is not None and int(e["team"]) != team:
            continue
        if card is not None and int(e["card_id"]) != card:
            continue
        if kind is not None and e["kind"] != kind:
            continue
        out.append(e)
    return out


def troops(g, team=None, card=None):
    return units(g, team=team, card=card, kind="troop")


def tower_entity(g, team, idx):
    x, y = H.TOWER_POS[team][idx]
    for e in units(g, team=team, kind="tower"):
        if int(e["x"]) == x and int(e["y"]) == y:
            return e
    return None


def deck_of(g, team):
    return hand(g, team) + queue(g, team)


def put_first(g, team, card):
    """Reorder the team's current deck so `card` is in hand slot 0 (debug set_hand)."""
    d = deck_of(g, team)
    assert card in d, f"card {card} not in team {team}'s deck {d}"
    order = [card] + [c for c in d if c != card]
    g.set_hand(team, order)
    assert hand(g, team)[0] == card, "set_hand(team, order) must put order[0] in slot 0"
    return order


def cast(g, team, card, x, y, elixir_units=H.MAX_ELIXIR):
    """Put `card` in slot 0, top up elixir, queue a play at engine point (x, y)."""
    put_first(g, team, card)
    g.set_elixir(team, elixir_units)
    return g.play(team, 0, int(x), int(y))


def cast_tile(g, team, card, tx, ty, elixir_units=H.MAX_ELIXIR):
    put_first(g, team, card)
    g.set_elixir(team, elixir_units)
    return g.play_tile(team, 0, int(tx), int(ty))


def spawn(g, team, card, x, y, deployed=True):
    ids = g.spawn(team, card, int(x), int(y), deployed=deployed)
    ids = [int(i) for i in ids]
    assert len(ids) >= 1, f"spawn({team},{card},{x},{y}) returned no entities"
    return ids


def spawn1(g, team, card, x, y, deployed=True):
    ids = spawn(g, team, card, x, y, deployed)
    assert len(ids) == 1, f"expected exactly one entity for card {card}, got {ids}"
    return ids[0]


def run_until(g, pred, max_ticks, what="condition"):
    """Tick one at a time until pred(g) is true; return the number of ticks advanced."""
    for n in range(1, max_ticks + 1):
        g.tick(1)
        if pred(g):
            return n
    raise AssertionError(f"{what} not reached within {max_ticks} ticks (tick={tick_of(g)})")


def hp_trace(g, eids, n):
    """Tick n times; return list of dicts {eid: hp or None} observed after each tick."""
    out = []
    for _ in range(n):
        g.tick(1)
        cur = {int(e["id"]): int(e["hp"]) for e in ents(g)}
        out.append({i: cur.get(i) for i in eids})
    return out


def drop_ticks(trace, eid, start_hp):
    """Indices (1-based tick counts) at which hp of eid decreased, and the amounts."""
    res = []
    prev = start_hp
    for k, row in enumerate(trace, start=1):
        v = row[eid]
        if v is None:
            res.append((k, None))
            break
        if v < prev:
            res.append((k, prev - v))
        prev = v
    return res


def tower_hp_trace(g, team, idx, n):
    out = []
    for _ in range(n):
        g.tick(1)
        out.append(tower_hp(g, team, idx))
    return out


def first_not_deploying(g, eid, max_ticks=60):
    """Tick until the entity reports deploying == False; return ticks advanced."""
    return run_until(g, lambda gg: not bool(ent(gg, eid)["deploying"]), max_ticks,
                     f"entity {eid} finishing deploy")


def pos(e):
    return int(e["x"]), int(e["y"])


def destroy_tower(g, team, idx, max_ticks=120):
    """Destroy crown tower (team, idx) through a real damage path: set its hp to 1 with the
    debug setter, then have the OTHER team cast a damaging spell from its deck at the tower
    centre. Returns the number of ticks until `alive` became False."""
    attacker = 1 - team
    d = deck_of(g, attacker)
    spell = next((c for c in (H.ZAP, H.ARROWS, H.FIREBALL) if c in d), None)
    assert spell is not None, f"team {attacker}'s deck {d} has no damaging spell"
    g.set_tower_hp(team, idx, 1)
    assert tower_hp(g, team, idx) == 1, "set_tower_hp must set the hp exactly"
    x, y = H.TOWER_POS[team][idx]
    code = cast(g, attacker, spell, x, y)
    assert code == H.OK, f"casting {H.CARD_NAMES[spell]} on the tower returned {code}"
    return run_until(g, lambda gg: not tower_alive(gg, team, idx), max_ticks,
                     f"tower {team}/{idx} destruction")
