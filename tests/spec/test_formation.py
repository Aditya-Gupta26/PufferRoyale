"""SPEC §5 deploy + §5.1 formations: member counts, n=2 / n=3 layouts (own frame, apex toward the
enemy), larger n within 1600 of the tap point, ground members never in water."""
import itertools
import math

import pytest

import gamekit as K
import helpers as H

MULTI = [(H.ARCHERS, "bait", 2), (H.MINIONS, "giant", 3), (H.SKELETONS, "hog26", 3),
         (H.SKARMY, "bait", 15)]


def play_and_collect(team, card, deck, tile, count, max_ticks=15):
    g = K.new_game(deck, deck, deploy_lockout_ticks=0)
    assert K.cast_tile(g, team, card, *tile) == H.OK
    for _ in range(max_ticks):
        g.tick(1)
        us = K.troops(g, team, card)
        if len(us) >= count:
            return g, us
    raise AssertionError(f"{H.CARD_NAMES[card]}: only {len(K.troops(g, team, card))} members "
                         f"after {max_ticks} ticks, want {count}")


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("card,deck,count", MULTI, ids=[H.CARD_NAMES[m[0]] for m in MULTI])
def test_member_count(pr, team, card, deck, count):
    assert H.src_card(card)["count"] == count
    g, us = play_and_collect(team, card, deck, (9, 21), count)
    assert len(us) == count
    g.tick(25)
    assert len(K.troops(g, team, card)) == count, "no extra members appear later"
    for u in us:
        assert int(u["team"]) == team
        assert int(u["max_hp"]) == H.expected_unit_stats(card)["hp"]


def own_offsets(team, us, tile):
    cx, cy = tile[0] * 1000 + 500, tile[1] * 1000 + 500
    out = []
    for u in us:
        ox, oy = H.engine_to_own_point(team, *K.pos(u))
        out.append((ox - cx, oy - cy))
    return sorted(out)


def match_offsets(got, want, tol):
    for perm in itertools.permutations(got):
        if all(abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol for a, b in zip(perm, want)):
            return True
    return False


@pytest.mark.parametrize("team", [0, 1])
def test_two_member_layout(pr, team):
    g, us = play_and_collect(team, H.ARCHERS, "bait", (9, 21), 2)
    got = own_offsets(team, us, (9, 21))
    assert match_offsets(got, [(-500, 0), (500, 0)], 2), f"Archers own-frame offsets {got}"


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("card,deck", [(H.MINIONS, "giant"), (H.SKELETONS, "hog26")])
def test_three_member_triangle_apex_toward_enemy(pr, team, card, deck):
    g, us = play_and_collect(team, card, deck, (9, 21), 3)
    got = own_offsets(team, us, (9, 21))
    want = [(0, -577), (500, 289), (-500, 289)]
    assert match_offsets(got, want, 3), f"{H.CARD_NAMES[card]} own-frame offsets {got}, want {want}"


@pytest.mark.parametrize("team", [0, 1])
def test_large_formation_within_radius(pr, team):
    g, us = play_and_collect(team, H.SKARMY, "bait", (9, 21), 15)
    cx, cy = H.tile_centre_engine(team, 9, 21)
    ps = [K.pos(u) for u in us]
    assert len(set(ps)) == 15, "no two members may share a centre"
    for p in ps:
        # 1600 is the layout bound; one tick of collision separation is allowed on top
        assert math.dist(p, (cx, cy)) <= 1600 + 500, f"member at {p} too far from the tap point"
    assert all(int(u["card_id"]) == H.SKARMY for u in us), "Skeleton Army members carry card id 8"


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("card,deck,count", [(H.SKELETONS, "hog26", 3), (H.SKARMY, "bait", 15),
                                             (H.ARCHERS, "bait", 2)])
def test_ground_members_never_in_water(pr, team, card, deck, count):
    g, us = play_and_collect(team, card, deck, (9, 17), count)
    for _ in range(30):
        for u in K.troops(g, team, card):
            assert not H.point_in_water(*K.pos(u)), f"ground member in water at {K.pos(u)}"
            x, y = K.pos(u)
            assert 0 <= x < H.ARENA_W and 0 <= y < H.ARENA_H
        g.tick(1)


@pytest.mark.parametrize("team", [0, 1])
def test_hand_played_troop_deploy_duration(pr, team):
    """SPEC §13.2: deploying in exactly D/50 = 20 observations, acting from tick t + 20."""
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    assert K.cast_tile(g, team, H.KNIGHT, 9, 21) == H.OK
    g.tick(1)
    kn = K.troops(g, team, H.KNIGHT)[0]
    kid = int(kn["id"])
    D = 0
    p0 = K.pos(kn)
    while bool(K.ent(g, kid)["deploying"]):
        assert K.pos(K.ent(g, kid)) == p0, "deploying units do not move"
        D += 1
        assert D <= 25
        g.tick(1)
    assert D == 20, f"Knight showed deploying for {D} observations (SPEC §13.2: exactly 1000/50)"
    g.tick(2)
    assert K.pos(K.ent(g, kid)) != p0, "the deployed Knight walks its default route"


STAGGER = [(H.ARCHERS, "bait", [21, 23]), (H.MINIONS, "giant", [21, 23, 25]),
           (H.SKELETONS, "hog26", [21, 21, 21]), (H.SKARMY, "bait", [21] * 15)]


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("card,deck,want", STAGGER, ids=[H.CARD_NAMES[s[0]] for s in STAGGER])
def test_staggered_member_deploy(pr, team, card, deck, want):
    """SPEC §13.2: member i deploys as if created at t + i*summon_deploy_delay/50 (Archers and
    Minions: 100 ms -> 2 ticks; no delay for Skeletons / Skeleton Army). With the play applied in
    tick t (observation 1), member i is first seen not deploying at observation 21 + 2i."""
    delay = H.src_card(card).get("summon_deploy_delay_ms") or 0
    assert [21 + (delay // 50) * i for i in range(len(want))] == want     # tester arithmetic
    g = K.new_game(deck, deck, deploy_lockout_ticks=0)
    assert K.cast_tile(g, team, card, 9, 21) == H.OK
    done = {}
    for n in range(1, 40):
        g.tick(1)
        for e in K.troops(g, team, card):
            i = int(e["id"])
            if i not in done and not bool(e["deploying"]):
                done[i] = n
    assert sorted(done.values()) == want, f"first non-deploying observations {sorted(done.values())}"
