"""SPEC §4 King activation (dormant until King damage or own Princess destroyed; active exactly
71 ticks after the FIRST trigger) and §6.1 (dormant King does not target).

Timing convention: let n0 = the first observation (state tick) at which the trigger is visible
(King hp below max, or Princess alive == False). The trigger happened while processing tick
n0 - 1; the countdown is advanced in Upkeep, so the King is first observed active at n0 + 71.
"""
import pytest

import gamekit as K
import helpers as H


def king_target(g, team):
    e = K.tower_entity(g, team, 0)
    assert e is not None
    return int(e["target_id"])


def test_dormant_king_ignores_units_in_range(pr):
    g = K.new_game("giant", "giant")
    # Giant 6403 from the team-1 King (inside 7000 + 1400 + 750) but its nearest building is the
    # right princess (edge 1121 vs King edge 5003), which it attacks in place: the King is never hit.
    kid = K.spawn1(g, 0, H.GIANT, 13000, 8000)
    k_hp = K.tower_hp(g, 1, 0)
    for _ in range(150):
        g.tick(1)
        assert K.tower_active(g, 1, 0) is False
        assert king_target(g, 1) == -1, "a dormant King must not acquire a target"
        assert K.tower_hp(g, 1, 0) == k_hp
    assert K.ent(g, kid) is not None


def _activation_after_trigger(g, team, triggered):
    n = K.run_until(g, triggered, 400, "activation trigger")
    n0 = K.tick_of(g)
    assert K.tower_active(g, team, 0) is False
    for k in range(1, 71):
        g.tick(1)
        assert K.tower_active(g, team, 0) is False, f"King active only {k} ticks after trigger (tick {n0})"
    g.tick(1)
    assert K.tower_active(g, team, 0) is True, "King must be active exactly 71 ticks after the trigger"
    return n0


@pytest.mark.parametrize("team", [0, 1])
def test_activation_71_ticks_after_king_damage_arrows(pr, team):
    g = K.new_game("giant", "giant")
    g.tick(100)
    x, y = H.TOWER_POS[team][0]
    assert K.cast(g, 1 - team, H.ARROWS, x, y) == H.OK
    _activation_after_trigger(g, team, lambda gg: K.tower_hp(gg, team, 0) < H.KING_HP)


@pytest.mark.parametrize("team", [0, 1])
def test_activation_not_delayed_by_zap_stun(pr, team):
    g = K.new_game("giant", "giant")
    g.tick(100)
    x, y = H.TOWER_POS[team][0]
    assert K.cast(g, 1 - team, H.ZAP, x, y) == H.OK
    g.tick(1)
    assert K.tower_hp(g, team, 0) == H.KING_HP - 48, "Zap deals 48 to a crown tower (ceil(192*25/100))"
    n0 = K.tick_of(g)
    for _ in range(70):
        g.tick(1)
        assert K.tower_active(g, team, 0) is False
    g.tick(1)
    assert K.tower_active(g, team, 0) is True, f"activation at {n0}+71 expected"


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("idx", [1, 2])
def test_activation_71_ticks_after_own_princess_destroyed(pr, team, idx):
    g = K.new_game("giant", "giant")
    g.tick(100)
    g.set_tower_hp(team, idx, 1)
    x, y = H.TOWER_POS[team][idx]
    assert K.cast(g, 1 - team, H.ZAP, x, y) == H.OK
    _activation_after_trigger(g, team, lambda gg: not K.tower_alive(gg, team, idx))
    assert K.tower_hp(g, team, 0) == H.KING_HP, "the Zap never touched the King"
    assert K.tower_active(g, 1 - team, 0) is False, "destroying an ENEMY princess does not wake your King"


def test_first_trigger_counts(pr):
    """A second trigger (princess falls later) must not restart the countdown."""
    g = K.new_game("giant", "giant")
    g.tick(100)
    assert K.cast(g, 0, H.ZAP, *H.TOWER_POS[1][0]) == H.OK
    g.tick(1)
    n0 = K.tick_of(g)
    g.tick(30)
    g.set_tower_hp(1, 1, 1)
    assert K.cast(g, 0, H.ZAP, *H.TOWER_POS[1][1]) == H.OK
    g.tick(1)
    assert not K.tower_alive(g, 1, 1)
    g.tick(n0 + 70 - K.tick_of(g))
    assert K.tower_active(g, 1, 0) is False
    g.tick(1)
    assert K.tower_active(g, 1, 0) is True


def test_active_king_targets_and_shoots(pr):
    g = K.new_game("giant", "giant")
    g.tick(100)
    assert K.cast(g, 0, H.ZAP, *H.TOWER_POS[1][0]) == H.OK
    g.tick(80)
    assert K.tower_active(g, 1, 0) is True
    gid = K.spawn1(g, 0, H.GIANT, 9000, 9000)
    seen = K.run_until(g, lambda gg: king_target(gg, 1) == gid, 40, "active King acquiring the Giant")
    assert seen <= 5
    hp0 = H.expected_unit_stats(H.GIANT)["hp"]
    K.run_until(g, lambda gg: K.hp_of(gg, gid) is not None and K.hp_of(gg, gid) < hp0, 60,
                "King/princess damage on the Giant")
