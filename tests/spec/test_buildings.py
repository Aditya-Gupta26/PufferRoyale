"""SPEC §6.5 buildings: linear lifetime drain (Cannon 30 s, Tesla 25 s) and Tesla hide/rise."""
import pytest

import gamekit as K
import helpers as H


def drain_series(max_hp, L, n, off):
    """hp after elapsed drain tick i (i = 0..n-1): max_hp - floor(max_hp*(i+1+off)*50/L)."""
    return [max_hp - (max_hp * (i + 1 + off) * 50) // L for i in range(n)]


@pytest.mark.parametrize("card,deck,L", [(H.CANNON, "hog26", 30000), (H.TESLA, "bait", 25000)])
@pytest.mark.parametrize("team", [0, 1])
def test_lifetime_linear_drain_exact(pr, card, deck, L, team):
    g = K.new_game(deck, deck, deploy_lockout_ticks=0)
    assert K.cast_tile(g, team, card, 9, 21) == H.OK
    g.tick(1)
    b = K.units(g, team=team, card=card)
    assert len(b) == 1
    eid = int(b[0]["id"])
    max_hp = H.expected_unit_stats(card)["hp"]
    D = 0
    while bool(K.ent(g, eid)["deploying"]):
        assert int(K.ent(g, eid)["hp"]) == max_hp, "no drain while deploying"
        D += 1
        assert D < 40
        g.tick(1)
    assert D == 20, f"exactly deploy_time/50 = 20 deploying observations (SPEC §13.2), got {D}"
    series = []
    while True:
        e = K.ent(g, eid)
        if e is None:
            break
        series.append(int(e["hp"]))
        assert len(series) < L // 50 + 5, "building outlived its lifetime"
        g.tick(1)
    n_ticks = L // 50
    # SPEC §13.4: drain tick k = 0 is the first non-deploying tick; hp reaches 0 and the building is
    # reaped at the end of drain tick L/50 - 1, so it is observed alive after drain ticks 0..L/50-2.
    assert len(series) == n_ticks - 1, (f"{H.CARD_NAMES[card]} observed {len(series)} ticks after "
                                        f"deploy; want {n_ticks - 1}")
    assert series == drain_series(max_hp, L, n_ticks - 1, 0), \
        f"hp series must be max_hp - floor(max_hp*(k+1)*50/L); first values {series[:6]}"


def test_tesla_hidden_after_deploy_and_immune(pr):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    t = K.spawn1(g, 1, H.TESLA, 9000, 9000, deployed=False)
    K.run_until(g, lambda gg: not bool(K.ent(gg, t)["deploying"]), 40, "Tesla deploy")
    g.tick(1)
    assert bool(K.ent(g, t)["hidden"]) is True, "Tesla hides after deploying with no enemy in sight"
    hp0 = K.hp_of(g, t)
    assert K.cast(g, 0, H.ZAP, 9000, 9000) == H.OK
    g.tick(1)
    e = K.ent(g, t)
    assert hp0 - int(e["hp"]) <= 3, "hidden Tesla is immune to Zap (only lifetime drain, <= 3/tick)"
    assert bool(e["stunned"]) is False
    hp1 = int(e["hp"])
    assert K.cast(g, 0, H.ARROWS, 9000, 9000) == H.OK
    for _ in range(40):
        g.tick(1)
    # drain over 40 ticks is ~40 * 1182/500 = 95; one Arrows wave alone would be 122
    assert hp1 - K.hp_of(g, t) <= 100, "hidden Tesla is immune to Arrows"


def test_hidden_tesla_is_untargetable(pr):
    g = K.new_game()
    t = K.spawn1(g, 1, H.TESLA, 9000, 7700, deployed=True)
    g.tick(2)
    assert bool(K.ent(g, t)["hidden"]) is True
    # Musketeer 6300 away: it could see the Tesla (6000 + 500) but the Tesla cannot see it
    # (5500 + 500), so the Tesla stays hidden and must not be acquired.
    m = K.spawn1(g, 0, H.MUSKETEER, 9000, 14000, deployed=True)
    g.tick(1)
    assert bool(K.ent(g, t)["hidden"]) is True
    assert int(K.ent(g, m)["target_id"]) != t, "a hidden Tesla is not a valid target"


def test_tesla_rises_then_attacks_after_up_time(pr):
    g = K.new_game()
    t = K.spawn1(g, 1, H.TESLA, 9000, 9000, deployed=True)
    g.tick(2)
    assert bool(K.ent(g, t)["hidden"]) is True
    kn = K.spawn1(g, 0, H.KNIGHT, 9000, 14000, deployed=True)   # 5000 away: in Tesla sight
    R = K.run_until(g, lambda gg: not bool(K.ent(gg, t)["hidden"]), 3, "Tesla rising")
    prev = K.hp_of(g, kn)
    first = None
    for n in range(1, 40):
        g.tick(1)
        cur = K.hp_of(g, kn)
        if prev - cur >= 200 and (prev - cur) % 109 != 0:
            first = n
            assert (prev - cur) in (220, 220 + 109, 220 + 218), f"Tesla hit {prev - cur}"
            break
        prev = cur
    assert first is not None, "risen Tesla never zapped the Knight"
    assert first >= 16, f"Tesla attacked {first} ticks after starting to rise; up_time is 16 ticks"
    assert first <= 16 + 7 + 3, f"Tesla took {first} ticks to attack after rising"
    assert R >= 1


def test_tesla_rehides_without_target(pr):
    g = K.new_game()
    t = K.spawn1(g, 1, H.TESLA, 9000, 9000, deployed=True)
    g.tick(2)
    K.spawn(g, 0, H.SKELETONS, 9000, 14000, deployed=True)
    K.run_until(g, lambda gg: not bool(K.ent(gg, t)["hidden"]), 5, "Tesla rising")
    K.run_until(g, lambda gg: len(K.troops(gg, 0)) == 0, 200, "Tesla killing the Skeletons")
    for k in range(1, 14):
        g.tick(1)
        assert bool(K.ent(g, t)["hidden"]) is False, f"re-hid only {k} ticks after losing its target"
    K.run_until(g, lambda gg: bool(K.ent(gg, t)["hidden"]), 20, "Tesla re-hiding (hide_time 800 ms)")
