"""SPEC §4 economy and clock.

Tick convention used throughout (SPEC §4/§10): state()['tick'] == number of engine ticks processed
since reset; the next tick to be processed has that index. Regen for tick t is applied at the
START of tick t; its rate is decided by t (1x for t < 2400, 2x for 2400 <= t < 4800, 3x after).
A play queued with Game.play takes effect during the NEXT tick's Upkeep, after that tick's regen.
"""
import numpy as np
import pytest

import gamekit as K
import helpers as H


def rate(t):
    return 50 if t < 2400 else (100 if t < 4800 else 150)


def test_start_elixir_and_tick(pr):
    g = K.new_game()
    assert K.tick_of(g) == 0
    assert K.elixir(g, 0) == H.START_ELIXIR and K.elixir(g, 1) == H.START_ELIXIR
    assert K.over(g) is False
    assert bool(K.st(g)["overtime"]) is False
    assert K.crowns(g) == (0, 0)


def test_tick_counter_advances_by_n(pr):
    g = K.new_game()
    g.tick(1)
    assert K.tick_of(g) == 1
    g.tick(7)
    assert K.tick_of(g) == 8
    g.tick()
    assert K.tick_of(g) == 9


def test_regen_1x_and_cap_tick_by_tick(pr):
    """16800 + 50/tick, capped at 28000 (reached after exactly 224 ticks)."""
    g = K.new_game()
    for n in range(1, 301):
        g.tick(1)
        want = min(H.START_ELIXIR + 50 * n, H.MAX_ELIXIR)
        assert K.elixir(g, 0) == want, f"after {n} ticks: elixir {K.elixir(g, 0)} want {want}"
        assert K.elixir(g, 1) == want
    assert K.elixir(g, 0) == H.MAX_ELIXIR


def test_lockout_does_not_stop_regen(pr):
    g = K.new_game()
    g.tick(H.LOCKOUT_TICKS)
    assert K.elixir(g, 0) == H.START_ELIXIR + 50 * 90 == 21300


@pytest.mark.parametrize("boundary", [2400, 4800])
def test_regen_rate_boundaries(pr, boundary):
    g = K.new_game()
    g.tick(boundary - 5)
    assert not K.over(g), "no crowns: the match must still be running"
    g.set_elixir(0, 0)
    g.set_elixir(1, 1000)
    e0, e1 = 0, 1000
    for t in range(boundary - 5, boundary + 5):   # t = index of the tick being processed
        g.tick(1)
        e0 += rate(t)
        e1 += rate(t)
        assert K.elixir(g, 0) == e0, f"tick {t}: team0 elixir {K.elixir(g, 0)} want {e0}"
        assert K.elixir(g, 1) == e1, f"tick {t}: team1 elixir {K.elixir(g, 1)} want {e1}"


def test_regen_3x_until_end(pr):
    g = K.new_game()
    g.tick(5900)
    g.set_elixir(0, 0)
    g.tick(50)
    assert K.elixir(g, 0) == 150 * 50


def test_cap_and_leak_in_2x(pr):
    g = K.new_game()
    g.tick(2500)
    g.set_elixir(0, H.MAX_ELIXIR - 150)
    g.tick(1)
    assert K.elixir(g, 0) == H.MAX_ELIXIR - 50
    g.tick(1)
    assert K.elixir(g, 0) == H.MAX_ELIXIR
    g.tick(10)
    assert K.elixir(g, 0) == H.MAX_ELIXIR


@pytest.mark.parametrize("team", [0, 1])
def test_cost_deduction_units(pr, team):
    g = K.new_game("giant", "giant")
    g.tick(100)
    K.put_first(g, team, H.KNIGHT)
    g.set_elixir(team, 20000)
    assert g.play_tile(team, 0, 9, 21) == H.OK
    g.tick(1)                       # tick index 100: regen +50 then the play (-3*2800)
    assert K.elixir(g, team) == 20000 + 50 - 3 * 2800 == 11650
    assert K.elixir(g, 1 - team) == min(H.MAX_ELIXIR, H.START_ELIXIR + 50 * 101)


def test_regen_applies_before_play_at_cap(pr):
    """Upkeep order (§5): regen first (lost at the cap), then the play's deduction."""
    g = K.new_game("giant", "giant")
    g.tick(100)
    K.put_first(g, 0, H.KNIGHT)
    g.set_elixir(0, H.MAX_ELIXIR)
    assert g.play_tile(0, 0, 9, 21) == H.OK
    g.tick(1)
    assert K.elixir(g, 0) == H.MAX_ELIXIR - 3 * 2800 == 19600


@pytest.mark.parametrize("card", [H.HOG, H.ICE_SPIRIT, H.FIREBALL, H.LOG, H.CANNON, H.SKELETONS])
def test_not_enough_elixir_boundary(pr, card):
    g = K.new_game("hog26", "hog26")
    g.tick(H.LOCKOUT_TICKS)
    K.put_first(g, 0, card)
    cost = H.COST[card] * H.ELIXIR_UNIT
    tile = (9, 21)
    g.set_elixir(0, cost - 1)
    assert g.play_tile(0, 0, *tile) == H.NOT_ENOUGH_ELIXIR
    mask = np.asarray(g.legal_mask(0))
    assert mask[1:1 + H.N_CELLS].sum() == 0, "slot 0 unaffordable => no slot-0 action legal"
    g.set_elixir(0, cost)
    mask = np.asarray(g.legal_mask(0))
    assert mask[H.action_id(0, *tile)] == 1
    assert g.play_tile(0, 0, *tile) == H.OK


def test_rejected_play_changes_nothing(pr):
    g = K.new_game("hog26", "hog26")
    g.tick(H.LOCKOUT_TICKS)
    K.put_first(g, 0, H.HOG)
    g.set_elixir(0, 5000)
    h0, q0 = K.hand(g, 0), K.queue(g, 0)
    assert g.play_tile(0, 0, 9, 21) == H.NOT_ENOUGH_ELIXIR
    g.tick(1)
    assert K.hand(g, 0) == h0 and K.queue(g, 0) == q0
    assert K.elixir(g, 0) == 5050
    assert K.troops(g, 0) == []


@pytest.mark.parametrize("lockout", [90, 30, 1])
def test_lockout_boundary(pr, lockout):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=lockout)
    K.put_first(g, 0, H.KNIGHT)
    if lockout > 1:
        g.tick(lockout - 1)
    g.set_elixir(0, H.MAX_ELIXIR)
    assert g.play_tile(0, 0, 9, 21) == H.LOCKOUT
    mask = np.asarray(g.legal_mask(0))
    assert mask[0] == 1 and mask[1:].sum() == 0, "during lockout only the no-op is legal"
    g.tick(1)
    mask = np.asarray(g.legal_mask(0))
    assert mask[H.action_id(0, 9, 21)] == 1
    assert g.play_tile(0, 0, 9, 21) == H.OK


def test_lockout_zero_allows_play_at_reset(pr):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    K.put_first(g, 0, H.KNIGHT)
    assert g.play_tile(0, 0, 9, 21) == H.OK
    g.tick(1)
    assert len(K.troops(g, 0, H.KNIGHT)) == 1


def test_default_lockout_is_90(pr):
    g = K.new_game("giant", "giant")
    K.put_first(g, 0, H.KNIGHT)
    g.tick(89)
    assert g.play_tile(0, 0, 9, 21) == H.LOCKOUT
    g.tick(1)
    assert g.play_tile(0, 0, 9, 21) == H.OK


def test_overtime_flag(pr):
    g = K.new_game()
    g.tick(3599)
    assert bool(K.st(g)["overtime"]) is False
    g.tick(2)
    assert not K.over(g)
    assert bool(K.st(g)["overtime"]) is True


def test_both_teams_play_same_tick(pr):
    g = K.new_game("giant", "giant")
    g.tick(H.LOCKOUT_TICKS)
    for t in (0, 1):
        K.put_first(g, t, H.KNIGHT)
        g.set_elixir(t, 10000)
    assert g.play_tile(0, 0, 9, 21) == H.OK
    assert g.play_tile(1, 0, 9, 21) == H.OK
    g.tick(1)
    assert len(K.troops(g, 0, H.KNIGHT)) == 1 and len(K.troops(g, 1, H.KNIGHT)) == 1
    assert K.elixir(g, 0) == K.elixir(g, 1) == 10000 + 50 - 8400
