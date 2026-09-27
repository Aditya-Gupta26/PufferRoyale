"""SPEC §16.3 tower troops: `tower_troop0/1` on Game and Royale replace BOTH Princess towers of that
team; stats = data rows scaled by the §2 tower ladder at level 11 (Princess rates = 218%):
Cannoneer 1200 -> 2616 HP / 125 -> 272 dmg, Dagger Duchess 1270 -> 2768 / 42 -> 91,
Royal Chef 1240 -> 2703 / 50 -> 109. The opponent's tower troop is public (visible in obs)."""
import math

import numpy as np
import pytest

import cardsv3 as C
import envkit as E
import envkit3 as E3
import gamekit as K
import helpers as H

TT = C.TOWER_TROOPS                     # §16.3 listing order: princess, cannoneer, dagger_duchess, royal_chef


def tt_game(t0="princess", t1="princess", d0="hog26", d1="giant", **kw):
    return K.new_game(d0, d1, deploy_lockout_ticks=0, tower_troop0=t0, tower_troop1=t1, **kw)


def watch_drops(g, eid, n):
    prev, out = K.hp_of(g, eid), []
    for k in range(1, n + 1):
        g.tick(1)
        cur = K.hp_of(g, eid)
        assert cur is not None, "setup: the victim must survive the observation window"
        if cur < prev:
            out.append((k, prev - cur))
        prev = cur
    return out


def test_tester_tower_ladder_values():
    assert C.tower_troop_stats("cannoneer")["hp"] == 2616 and C.tower_troop_stats("cannoneer")["damage"] == 272
    assert C.tower_troop_stats("dagger_duchess")["hp"] == 2768 and C.tower_troop_stats("dagger_duchess")["damage"] == 91
    assert C.tower_troop_stats("royal_chef")["hp"] == 2703 and C.tower_troop_stats("royal_chef")["damage"] == 109
    assert (1400 * C.TOWER_PCT) // 100 == H.PRINCESS_HP and (50 * C.TOWER_PCT) // 100 == H.TOWER_DMG


@pytest.mark.parametrize("troop", TT)
@pytest.mark.parametrize("team", [0, 1])
def test_tower_troop_replaces_both_princess_towers(pr, troop, team):
    kw = {f"tower_troop{team}": troop}
    g = K.new_game("hog26", "giant", deploy_lockout_ticks=0, **kw)
    want = C.tower_troop_stats(troop)["hp"]
    for idx in (1, 2):
        e = K.tower_entity(g, team, idx)
        assert e is not None and int(e["max_hp"]) == int(e["hp"]) == want, \
            f"team {team} tower {idx} with {troop}: {e and (e['hp'], e['max_hp'])}, want {want}"
        assert int(K.tower(g, team, idx)["max_hp"]) == want
        other = K.tower_entity(g, 1 - team, idx)
        assert int(other["max_hp"]) == H.PRINCESS_HP, "the other team keeps its default Princess towers"
    for t in (0, 1):
        assert int(K.tower_entity(g, t, 0)["max_hp"]) == H.KING_HP, "the King tower is unchanged"


def test_default_tower_troop_is_princess(pr):
    a = K.new_game("hog26", "giant", seed=3, deploy_lockout_ticks=0)
    b = tt_game("princess", "princess", seed=3)
    for _ in range(50):
        a.tick(1)
        b.tick(1)
    assert a.hash() == b.hash(), "tower_troop defaults to 'princess'"
    for t in (0, 1):
        for idx in (1, 2):
            assert int(K.tower_entity(a, t, idx)["max_hp"]) == H.PRINCESS_HP


def test_invalid_tower_troop_is_rejected(pr):
    for t in TT:
        K.new_game("hog26", "giant", tower_troop0=t, tower_troop1=t)        # every listed value is accepted
    with pytest.raises(ValueError):                                          # §16.6.19
        K.new_game("hog26", "giant", tower_troop0="musketeer")
    with pytest.raises(ValueError):
        E.make(tower_troop1="knight")


GOLEM_AT = (3500, 9000)          # 2500 from team 1's left tower = Golem range 750 + 750 + 1000


@pytest.mark.parametrize("troop,dmg,gap", [("princess", 109, 16), ("cannoneer", 272, 44), ("royal_chef", 109, 20)])
def test_tower_troop_damage_and_cadence(pr, troop, dmg, gap):
    """A team-0 Golem standing 2500 below team 1's left tower, hitting it (so it never moves): hits
    of exactly `dmg`, every hit_speed/50 ticks (Princess 800 ms, Cannoneer 2200 ms, Chef 1000 ms)."""
    g = tt_game("princess", troop)
    g.tick(100)
    golem = K.spawn1(g, 0, C.GOLEM, *GOLEM_AT, deployed=True)
    hits = watch_drops(g, golem, 150)
    assert K.pos(K.ent(g, golem)) == GOLEM_AT, "setup: the Golem stands in range of the tower"
    assert len(hits) >= 3 and {d for _, d in hits} == {dmg}, f"{troop} hits {hits}"
    assert {b[0] - a[0] for a, b in zip(hits, hits[1:])} == {gap}, f"{troop} hit spacing {hits}"
    if troop == "cannoneer":
        # in range at tick 1; (2200 - 1400)/50 - 1 = 15 ticks to the shot; 2200 of flight at 1000/tick
        assert 18 <= hits[0][0] <= 21, f"first Cannoneer hit at observation {hits[0][0]}"


def test_dagger_duchess_burst_then_reload(pr):
    """Daggers of 91; within the attack sequence shots follow at hit_speed x (100,100,70,90)% =
    10/10/7/9 ticks; after the sequence it reloads (a longer gap)."""
    g = tt_game("princess", "dagger_duchess")
    g.tick(100)
    golem = K.spawn1(g, 0, C.GOLEM, *GOLEM_AT, deployed=True)
    hits = watch_drops(g, golem, 400)
    assert len(hits) >= 8 and {d for _, d in hits} == {91}, f"Dagger Duchess hits {hits}"
    gaps = [b[0] - a[0] for a, b in zip(hits, hits[1:])]
    assert min(gaps) >= 7, f"no shot faster than 350 ms: {gaps}"
    assert any(x <= 10 for x in gaps) and any(x > 10 for x in gaps), f"burst then reload: {gaps}"


LADDER = [256, 281, 309, 339, 372, 409]                     # Common ladder from level 11 upward


def test_tester_chef_level_values():
    assert (690 * 281) // 100 == 1938 == (1766 * 281) // 256


def test_royal_chef_levels_up_an_allied_troop(pr):
    """Knights keep arriving next to team 0's Chef towers; within 120 s some Knight is levelled to
    L12 (max_hp 1766 -> 1938 = x281/256), and every Knight's max_hp is a ladder value."""
    g = tt_game("royal_chef", "princess")
    allowed = {(690 * p) // 100 for p in LADDER}
    v = 1766
    for p0, p1 in zip(LADDER, LADDER[1:]):
        v = v * p1 // p0
        allowed.add(v)
    seen = set()
    for k in range(2400):
        if k % 100 == 0:
            K.spawn(g, 0, H.KNIGHT, 3500 if (k // 100) % 2 == 0 else 14500, 22000, deployed=True)
        g.tick(1)
        for e in K.units(g, team=0, card=H.KNIGHT):
            seen.add(int(e["max_hp"]))
        if 1938 in seen:
            break
    assert 1938 in seen, f"no allied Knight was levelled up in 120 s (max_hp values {sorted(seen)})"
    assert seen <= allowed, f"levelled max_hp not on the ladder: {sorted(seen - allowed)}"


def test_royal_chef_levels_the_highest_cost_troop(pr):
    """§16.6.18: the allied troop with the highest elixir cost within 7500 of an own Chef tower is
    levelled: Giants (5) and Knights (3) keep arriving next to the left Chef tower, and the first
    troop to be levelled is a Giant (max_hp above its L11 value), never a Knight."""
    g = tt_game("royal_chef", "princess")
    base = {H.GIANT: H.expected_unit_stats(H.GIANT)["hp"], H.KNIGHT: 1766}
    first = None
    for k in range(2400):
        if k % 50 == 0:
            K.spawn(g, 0, H.GIANT, 4200, 22500, deployed=True)
            K.spawn(g, 0, H.KNIGHT, 2800, 22500, deployed=True)
        g.tick(1)
        up = [e for e in K.units(g, team=0, kind="troop")
              if int(e["card_id"]) in base and int(e["max_hp"]) > base[int(e["card_id"])]]
        if up:
            first = sorted({int(e["card_id"]) for e in up})
            break
    assert first is not None, "no allied troop was levelled up within 120 s"
    assert first == [H.GIANT], f"first levelled troop cards {first}: want only Giants (cost 5 > Knight 3)"


def _levelled(e):
    base = {H.GIANT: H.expected_unit_stats(H.GIANT)["hp"], H.ICE_GOLEM: H.expected_unit_stats(H.ICE_GOLEM)["hp"],
            H.KNIGHT: 1766}
    if e["unit"] == "Skeleton":
        return int(e["max_hp"]) > 81
    c = int(e["card_id"])
    return c in base and int(e["max_hp"]) > base[c]


def test_royal_chef_timers_are_per_tower(pr):
    """§16.6.28 / §16.6.26: each living Chef tower runs its own 28 s period, so with Giants kept
    next to BOTH towers two troops are levelled at the first full period (observation 560), one
    near each tower."""
    g = tt_game("royal_chef", "princess")
    ups = {}
    for k in range(1, 640):
        if (k - 1) % 50 == 0:
            K.spawn(g, 0, H.GIANT, 3500, 22500, deployed=True)
            K.spawn(g, 0, H.GIANT, 14500, 22500, deployed=True)
        g.tick(1)
        for e in K.units(g, team=0, card=H.GIANT):
            if _levelled(e) and int(e["id"]) not in ups:
                ups[int(e["id"])] = (k, int(e["x"]))
    first = sorted(v for v in ups.values() if v[0] == min(t for t, _ in ups.values()))
    assert [t for t, _ in first] == [560, 560], f"level-ups (observation, x): {sorted(ups.values())}"
    assert first[0][1] < 9000 < first[1][1], "one level-up next to each Chef tower"


def test_royal_chef_ranks_spawned_units_as_cost_zero(pr):
    """§16.6.28: units released by spawners/deaths/spells rank as cost 0. A Tombstone's death-spawned
    Skeletons (carrying the cost-3 Tombstone card, §16.6.6) stand next to the left Chef tower at the
    first level-up, together with Ice Golems (cost 2): the Ice Golem must be chosen."""
    g = tt_game("royal_chef", "princess", d1="hog26")
    tomb_at = (6500, 23500)
    tomb = K.spawn1(g, 0, C.TOMBSTONE, *tomb_at, deployed=True)
    chef = H.TOWER_POS[0][1]
    first = None
    for k in range(1, 600):
        if (k - 1) % 50 == 0:
            K.spawn(g, 0, H.ICE_GOLEM, 2000, 22500, deployed=True)
        if k == 515:                                 # Fireball (35 ticks of flight) kills it at ~549
            assert K.cast(g, 1, H.FIREBALL, *tomb_at) == H.OK
        if k == 559:
            assert K.ent(g, tomb) is None, "setup: the Tombstone died before the first level-up"
            near = [e for e in K.units(g, team=0, kind="troop")
                    if e["unit"] == "Skeleton" and math.dist(K.pos(e), chef) <= 7000]
            assert near, "setup: Skeletons stand within 7500 of the left Chef tower"
        g.tick(1)
        up = [e for e in K.units(g, team=0, kind="troop") if _levelled(e)]
        if up:
            first = (k, [(e["unit"], int(e["card_id"])) for e in up])
            break
    assert first is not None and first[0] == 560, f"first level-up {first}"
    assert all(cid == H.ICE_GOLEM for _, cid in first[1]), \
        f"levelled {first[1]}: a spawned Skeleton (cost 0) must lose to the Ice Golem (cost 2)"


# ------------------------------------------------------------------------------------------
# Env: config accepted, one-hots in both views
# ------------------------------------------------------------------------------------------
def onehot(name):
    v = np.zeros(4, np.float32)
    v[TT.index(name)] = 1.0
    return v


@pytest.mark.parametrize("t0,t1", [("cannoneer", "royal_chef"), ("dagger_duchess", "princess"),
                                   ("princess", "princess")])
def test_tower_troops_visible_in_both_views(pr, t0, t1):
    lay = E3.v3_layout()
    env = E.make(tower_troop0=t0, tower_troop1=t1)
    obs, _ = env.reset(seed=0)
    for k in range(3):
        for r, own, enemy in ((0, t0, t1), (1, t1, t0)):
            assert np.array_equal(E3.scalar(obs[r], lay, "own_tower_troop"), onehot(own)), (r, own)
            assert np.array_equal(E3.scalar(obs[r], lay, "enemy_tower_troop"), onehot(enemy)), (r, enemy)
        obs, *_ = E.step(env, [0, 0])
    env.close()


@pytest.mark.parametrize("side", [0, 1])
def test_single_agent_learner_sees_its_own_tower_troop(pr, side):
    lay = E3.v3_layout()
    env = E.make(num_agents=1, opponent="noop", learner_side=side, tower_troop0="cannoneer",
                 tower_troop1="dagger_duchess")
    obs, _ = env.reset(seed=1)
    own, enemy = ("cannoneer", "dagger_duchess") if side == 0 else ("dagger_duchess", "cannoneer")
    assert np.array_equal(E3.scalar(obs[0], lay, "own_tower_troop"), onehot(own))
    assert np.array_equal(E3.scalar(obs[0], lay, "enemy_tower_troop"), onehot(enemy))
    env.close()


def test_env_tower_hp_fractions_use_tower_troop_max_hp(pr):
    lay = E3.v3_layout()
    env = E.make(tower_troop0="cannoneer", tower_troop1="royal_chef")
    obs, _ = env.reset(seed=0)
    assert np.array_equal(E3.scalar(obs[0], lay, "own_tower_hp"), np.ones(3, np.float32))
    assert np.array_equal(E3.scalar(obs[1], lay, "enemy_tower_hp"), np.ones(3, np.float32))
    env.close()
