"""Builder tests: plays, legality, the mask, setters and reset semantics (SPEC §3.1, §4, §10)."""
import random

import numpy as np
import pytest

from pufferroyale import Game, PlayError, action_index, decode_action

KNIGHT, ARCHERS, GIANT, HOG, CANNON, TESLA, FIREBALL, ZAP, LOG, BARREL, SKARMY = 0, 1, 3, 4, 14, 15, 16, 18, 19, 20, 8
HAND = [KNIGHT, ARCHERS, CANNON, TESLA, FIREBALL, BARREL, SKARMY, ZAP]


def fresh(seed=0, **kw):
    g = Game(deck0="bait", deck1="giant", seed=seed, deploy_lockout_ticks=kw.pop("lockout", 0), **kw)
    g.set_hand(0, HAND)
    g.set_hand(1, HAND)
    return g


def units(g, team=None, unit=None):
    return [e for e in g.entities() if e["kind"] != "tower" and (team is None or e["team"] == team)
            and (unit is None or e["unit"] == unit)]


def test_play_xy_tap_point_semantics_both_teams():
    """play(x, y): legality on the own-frame tile containing the tap; troops and spells land
    at the tap point itself, buildings at that tile's centre."""
    for team in (0, 1):
        tx, ty = 6, 21
        etx, ety = (tx, ty) if team == 0 else (17 - tx, 31 - ty)
        x, y = etx * 1000 + 137, ety * 1000 + 999
        cx, cy = (13, 21) if team == 0 else (17 - 13, 31 - 21)  # Cannon on another tile
        g = fresh()
        assert g.play(team, 0, x, y) == PlayError.OK       # Knight
        assert g.play(team, 2, cx * 1000 + 700, cy * 1000 + 100) == PlayError.OK  # Cannon
        g.tick()
        k = units(g, team, "Knight")[0]
        assert (k["x"], k["y"]) == (x, y)
        c = units(g, team, "Cannon")[0]
        assert (c["x"], c["y"]) == (cx * 1000 + 500, cy * 1000 + 500)
        # play_tile == play at the tile centre
        a, b = fresh(), fresh()
        assert a.play_tile(team, 1, tx, ty) == PlayError.OK
        assert b.play(team, 1, etx * 1000 + 500, ety * 1000 + 500) == PlayError.OK
        a.tick()
        b.tick()
        assert a.hash() == b.hash()


def test_play_xy_boundary_is_seat_symmetric():
    """A tap exactly on a tile boundary belongs to the tile containing it in the ACTING team's
    own frame: team 0 at y=17000 is its tile 17 (legal), and so is team 1 at y=15000."""
    g = fresh()
    assert g.play(0, 0, 9000, 17000) == PlayError.OK
    assert g.play(1, 0, 9000, 15000) == PlayError.OK
    assert g.play(0, 1, 9000, 16999) == PlayError.ILLEGAL_POSITION
    assert g.play(1, 1, 9000, 15001) == PlayError.ILLEGAL_POSITION
    g.tick()
    k0 = units(g, 0, "Knight")[0]
    k1 = units(g, 1, "Knight")[0]
    assert (k0["x"], k0["y"]) == (18000 - k1["x"], 32000 - k1["y"])
    assert k0["y"] > 17000 and k1["y"] < 15000  # never left on the water line


def test_error_codes():
    g = fresh(lockout=90)
    assert g.play_tile(0, 0, 9, 25) == PlayError.LOCKOUT
    g.tick(90)
    assert g.play_tile(0, 4, 9, 25) == PlayError.BAD_SLOT
    assert g.play_tile(0, 0, 9, 16) == PlayError.ILLEGAL_POSITION
    assert g.play(0, 0, -5, 20000) == PlayError.ILLEGAL_POSITION
    assert g.play(0, 0, 18000, 20000) == PlayError.ILLEGAL_POSITION
    g.set_elixir(0, 3 * 2800 - 1)
    assert g.play_tile(0, 0, 9, 25) == PlayError.NOT_ENOUGH_ELIXIR
    g.set_elixir(0, 3 * 2800)
    assert g.play_tile(0, 0, 9, 25) == PlayError.OK
    g.set_tower_hp(1, 0, 0)
    assert g.state()["over"] is True
    assert g.play_tile(0, 1, 9, 25) == PlayError.GAME_OVER


def test_play_applies_next_tick_and_cycles():
    g = fresh()
    assert g.play_tile(0, 1, 5, 20) == PlayError.OK
    assert units(g) == []  # queued, not applied
    g.tick()
    s = g.state()
    assert s["hand"][0] == [KNIGHT, FIREBALL, CANNON, TESLA]
    assert s["queue"][0] == [BARREL, SKARMY, ZAP, ARCHERS]
    assert s["elixir"][0] == 16800 + 50 - 3 * 2800
    arch = units(g, 0, "Archer")
    assert sorted((e["x"], e["y"]) for e in arch) == [(5000, 20500), (6000, 20500)]
    assert all(e["deploying"] for e in arch) and all(e["card_id"] == ARCHERS for e in arch)


def test_mask_equals_play_acceptance():
    rng = random.Random(5)
    g = Game(deck0="random", deck1="random", seed=11, deploy_lockout_ticks=0)
    for step in range(12):
        for team in (0, 1):
            snap = g.snapshot()
            mask = g.legal_mask(team)
            sample = rng.sample(range(1, 2305), 300)
            for a in sample:
                slot, tx, ty = decode_action(a)
                ok = g.play_tile(team, slot, tx, ty) == PlayError.OK
                assert ok == bool(mask[a]), (step, team, a)
                g.restore(snap)
            legal = np.flatnonzero(mask[1:]) + 1
            if len(legal) and rng.random() < 0.5:
                slot, tx, ty = decode_action(int(rng.choice(legal)))
                assert g.play_tile(team, slot, tx, ty) == PlayError.OK
        g.tick(40)
        if step == 6:
            g.set_tower_hp(1, 2, 0)


def test_mask_accepted_play_really_spends_elixir():
    """Validate the mask against engine BEHAVIOUR, not against the legality predicate."""
    rng = random.Random(9)
    g = Game(deck0="bait", deck1="hog26", seed=2, deploy_lockout_ticks=0)
    for _ in range(25):
        mask = g.legal_mask(0)
        legal = np.flatnonzero(mask[1:]) + 1
        if len(legal):
            a = int(rng.choice(legal))
            slot, tx, ty = decode_action(a)
            s0 = g.state()
            card = s0["hand"][0][slot]
            assert g.play_tile(0, slot, tx, ty) == PlayError.OK
            g.tick()
            s1 = g.state()
            cost = [3, 3, 4, 5, 4, 3, 4, 4, 3, 1, 2, 1, 5, 5, 3, 4, 4, 3, 2, 2, 3][card]
            assert s1["elixir"][0] == min(28000, s0["elixir"][0] + 50) - cost * 2800 or s0["elixir"][0] + 50 > 28000
            assert s1["queue"][0][3] == card
        g.tick(60)


def test_two_plays_one_tick_revalidated_in_order():
    """SPEC §13.1: play() validates against the current state; queued plays are validated
    again, in order, in the next Upkeep, where one that no longer fits is dropped."""
    g = fresh()
    g.set_elixir(0, 6 * 2800)
    assert g.play_tile(0, 0, 9, 25) == PlayError.OK            # Knight 3
    assert g.play_tile(0, 1, 5, 25) == PlayError.OK            # Archers 3
    assert np.array_equal(g.legal_mask(0), fresh_mask := g.legal_mask(0))  # mask ignores pending plays
    assert fresh_mask[1 + 0 * 576 + 25 * 18 + 9] == 1
    g.tick()
    assert len(units(g, 0, "Knight")) == 1 and len(units(g, 0, "Archer")) == 2
    s = g.state()
    assert s["elixir"][0] == 50 and s["dropped_plays"] == 0
    g.set_hand(0, HAND)
    g.set_elixir(0, 28000)
    assert g.play_tile(0, 0, 9, 25) == PlayError.OK
    assert g.play_tile(0, 0, 9, 23) == PlayError.OK            # same slot: dropped in Upkeep
    g.tick()
    s = g.state()
    assert s["plays"][0] == 3 and s["dropped_plays"] == 1


def test_spawn_test_hook():
    g = fresh()
    ids = g.spawn(0, SKARMY, 9000, 22000)
    assert len(ids) == 15
    es = {e["id"]: e for e in g.entities()}
    assert all(not es[i]["deploying"] for i in ids)
    ids2 = g.spawn(1, "Archers", 9000, 10000, deployed=False)
    assert len(ids2) == 2 and all(g.entity(i)["deploying"] for i in ids2)
    assert g.spawn(0, "Fireball", 9000, 6000) == []  # spells are cast, no entities
    assert len(g.projectiles()) == 1


def test_setters():
    g = fresh()
    g.set_elixir(1, 1234)
    assert g.state()["elixir"][1] == 1234
    with pytest.raises(ValueError):
        g.set_elixir(0, 28001)
    g.set_tower_hp(0, 1, 100)
    assert g.state()["towers"][0][1]["hp"] == 100
    g.set_tower_hp(0, 1, 0)
    s = g.state()
    assert s["towers"][0][1]["alive"] is False and s["towers"][0][1]["hp"] == 0
    assert s["crowns"] == [0, 1]
    assert s["over"] is False
    with pytest.raises(ValueError):
        g.set_tower_hp(0, 1, 50)
    with pytest.raises(ValueError):
        g.set_hand(0, [0, 1, 2, 3, 4, 5, 6, 6])
    g.set_hand(0, list(range(10, 18)))
    s = g.state()
    assert s["hand"][0] == [10, 11, 12, 13] and s["queue"][0] == [14, 15, 16, 17]


def test_reset_semantics():
    a = Game(seed=5)
    h0 = a.hash()
    a.tick(200)
    a.reset(seed=5)
    assert a.hash() == h0
    a.reset()
    assert a.hash() != h0          # continues the RNG stream: a new deal
    b = Game(seed=5)
    b.reset()
    assert a.hash() == b.hash()    # ... deterministically


def test_action_index_roundtrip():
    for a in (1, 576, 577, 2304):
        slot, tx, ty = decode_action(a)
        assert action_index(slot, tx, ty) == a
    assert decode_action(0) is None
