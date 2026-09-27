"""SPEC §6.3 movement & pathing, §6.4 collision, §6.1 default route (lane commitment).

Includes the absorbing-state sweep: a lone ground unit on ANY legal tile of its half must reach
and attack an enemy tower (SPEC §6.3 'must never leave a unit stuck').
"""
import math

import pytest

import gamekit as K
import helpers as H


def step_lengths(g, eid, n):
    out = []
    p = K.pos(K.ent(g, eid))
    for _ in range(n):
        g.tick(1)
        e = K.ent(g, eid)
        assert e is not None
        q = K.pos(e)
        out.append(math.dist(p, q))
        p = q
    return out


@pytest.mark.parametrize("card", [H.KNIGHT, H.HOG, H.GIANT, H.VALKYRIE, H.PRINCE])
def test_ground_speed_matches_data(pr, card):
    """Straight walk up the left lane in the unit's own half (no enemy in sight)."""
    g = K.new_game()
    speed = H.expected_unit_stats(card)["speed"]
    eid = K.spawn1(g, 0, card, 3500, 23400, deployed=True)
    g.tick(3)                               # let pathing settle
    # Prince: stay below the 2500 run-up so the x2 charge speed cannot start (3+30 ticks * 60)
    n = 30 if card == H.PRINCE else min(40, 5000 // speed)
    steps = step_lengths(g, eid, n)
    assert max(steps) <= speed + 1e-9, f"step {max(steps):.1f} exceeds speed {speed} (overshoot?)"
    mean = sum(steps) / len(steps)
    assert mean >= 0.95 * speed, f"{H.CARD_NAMES[card]}: mean step {mean:.1f}, data speed {speed}"


def test_flying_speed_and_straight_line(pr):
    """A lone flyer moves in a straight line: every step is speed minus component truncation."""
    g = K.new_game()
    speed = H.expected_unit_stats(H.BABY_DRAGON)["speed"]
    eid = K.spawn1(g, 0, H.BABY_DRAGON, 9500, 22000, deployed=True)
    g.tick(1)
    steps = step_lengths(g, eid, 30)
    for s in steps:
        assert speed - 2 <= s <= speed + 1e-9, f"flying step {s:.2f}, speed {speed}"


def test_minion_group_speed(pr):
    g = K.new_game()
    speed = H.expected_unit_stats(H.MINIONS)["speed"]
    ids = K.spawn(g, 0, H.MINIONS, 9500, 22000, deployed=True)
    g.tick(1)
    p0 = {i: K.pos(K.ent(g, i)) for i in ids}
    g.tick(30)
    for i in ids:
        d = math.dist(p0[i], K.pos(K.ent(g, i)))
        # air-air separation inside the group may add a little displacement
        assert 0.9 * speed * 30 <= d <= 1.05 * speed * 30, f"minion {i} moved {d:.0f} in 30 ticks"


def test_flying_crosses_river_off_bridge(pr):
    g = K.new_game()
    ids = K.spawn(g, 0, H.MINIONS, 9500, 19000, deployed=True)
    seen = False
    for _ in range(80):
        g.tick(1)
        for e in K.troops(g, 0, H.MINIONS):
            x, y = K.pos(e)
            if 15000 <= y < 17000 and not (2500 <= x < 4500 or 13500 <= x < 15500):
                seen = True
    assert seen, "Minions heading to the right princess must fly straight over the water"


@pytest.mark.parametrize("card", [H.KNIGHT, H.GIANT, H.VALKYRIE, H.MUSKETEER])
def test_ground_units_use_bridges(pr, card):
    g = K.new_game()
    eid = K.spawn1(g, 0, card, 8500, 18500, deployed=True)
    crossed = False
    for _ in range(400):
        g.tick(1)
        e = K.ent(g, eid)
        if e is None:
            break
        x, y = K.pos(e)
        assert not H.point_in_water(x, y), f"{H.CARD_NAMES[card]} in water at {(x, y)}"
        if 15000 <= y < 17000:
            assert 2500 <= x < 4500 or 13500 <= x < 15500
        if y < 15000:
            crossed = True
    assert crossed, f"{H.CARD_NAMES[card]} never crossed the river"


@pytest.mark.parametrize("card", [H.HOG, H.PRINCE])
def test_jumpers_cross_water(pr, card):
    """Hog Rider and Prince may cross water anywhere; from (8500, 18500) toward the left princess
    the straight line crosses the river near x~7500, far from a bridge."""
    g = K.new_game()
    eid = K.spawn1(g, 0, card, 8500, 18500, deployed=True)
    jumped = False
    for _ in range(200):
        g.tick(1)
        e = K.ent(g, eid)
        if e is None:
            break
        x, y = K.pos(e)
        if 15000 <= y < 17000 and not (2500 <= x < 4500 or 13500 <= x < 15500):
            jumped = True
        if y < 14000:
            break
    assert jumped, f"{H.CARD_NAMES[card]} did not use its river jump"


# ------------------------------------------------------------------------------------------
# Absorbing-state sweep
# ------------------------------------------------------------------------------------------
def any_enemy_tower_damaged(g, team):
    enemy = 1 - team
    return any(K.tower_hp(g, enemy, i) < H.TOWER_MAX_HP[i] for i in range(3))


def sweep(team, card, tiles, limit):
    g = K.new_game()
    snap = g.snapshot()
    stuck = []
    for (tx, ty) in tiles:
        g.restore(snap)
        x, y = H.tile_centre_engine(team, tx, ty)
        ids = K.spawn(g, team, card, x, y, deployed=True)
        ok = False
        for _ in range(limit):
            g.tick(1)
            if any_enemy_tower_damaged(g, team):
                ok = True
                break
            if all(K.ent(g, i) is None for i in ids):
                break
        if not ok:
            left = [K.pos(K.ent(g, i)) for i in ids if K.ent(g, i) is not None]
            stuck.append(((tx, ty), left))
    assert not stuck, f"{H.CARD_NAMES[card]} (team {team}) never damaged a tower from {len(stuck)} tiles: {stuck[:12]}"


def own_half_tiles(team):
    a = H.all_alive()
    return [(tx, ty) for ty in range(17, 32) for tx in range(18) if H.troop_tile_legal(team, tx, ty, a)]


REPRESENTATIVE = [(1, 17), (16, 17), (0, 18), (17, 18), (6, 31), (11, 31), (8, 31), (9, 31),
                  (8, 23), (9, 23), (0, 30), (17, 30), (5, 26), (12, 26), (1, 24), (16, 24),
                  (6, 27), (11, 27), (8, 19), (9, 19)]


def test_representative_tiles_are_legal():
    for team in (0, 1):
        for t in REPRESENTATIVE:
            assert H.troop_tile_legal(team, *t, H.all_alive()), t


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("card", [H.KNIGHT, H.GIANT, H.HOG, H.MINIONS])
def test_no_absorbing_states_representative(pr, team, card):
    sweep(team, card, REPRESENTATIVE, 1000)


@pytest.mark.slow
@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("card", [H.KNIGHT, H.GIANT, H.HOG])
def test_no_absorbing_states_all_tiles(pr, team, card):
    sweep(team, card, own_half_tiles(team), 1000)


# ------------------------------------------------------------------------------------------
# Default route / lane commitment (SPEC §6.1)
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("card", [H.KNIGHT, H.GIANT])
def test_lane_default_tower(pr, team, card):
    """Own-frame lane (SPEC §13.6); for these off-centre positions it coincides with absolute x:
    x < 9000 walks to the enemy princess at x 3500, otherwise to x 14500."""
    for x, idx in ((5500, 1), (12500, 2)):
        g = K.new_game()
        y = 20000 if team == 0 else 12000
        eid = K.spawn1(g, team, card, x, y, deployed=True)
        K.run_until(g, lambda gg: any_enemy_tower_damaged(gg, team), 900, "tower damage")
        enemy = 1 - team
        assert K.tower_hp(g, enemy, idx) < H.TOWER_MAX_HP[idx], f"x={x}: wrong lane tower"
        assert K.tower_hp(g, enemy, 3 - idx) == H.TOWER_MAX_HP[3 - idx]
        assert K.tower_hp(g, enemy, 0) == H.KING_HP
        assert K.ent(g, eid) is not None


@pytest.mark.parametrize("team", [0, 1])
def test_lane_tie_goes_own_left(pr, team):
    """SPEC §13.6: lane choice in the unit's OWN frame, exactly x = 9000 -> own-left. For team 0
    that is the enemy's absolute-left princess (idx 1); for team 1 the enemy's absolute-right (idx 2)."""
    g = K.new_game()
    y = 20000 if team == 0 else 12000
    K.spawn1(g, team, H.KNIGHT, 9000, y, deployed=True)
    K.run_until(g, lambda gg: any_enemy_tower_damaged(gg, team), 900, "tower damage")
    enemy = 1 - team
    idx = 1 if team == 0 else 2
    assert K.tower_hp(g, enemy, idx) < H.PRINCESS_HP, "the own-frame-left princess must be attacked"
    assert K.tower_hp(g, enemy, 3 - idx) == H.PRINCESS_HP


@pytest.mark.parametrize("team", [0, 1])
def test_lane_commitment_after_princess_falls(pr, team):
    """With the lane's princess down, the unit goes for the KING, not the far princess."""
    g = K.new_game("giant", "giant")
    g.tick(H.LOCKOUT_TICKS)
    enemy = 1 - team
    K.destroy_tower(g, enemy, 1)
    x, y = (5500, 20000) if team == 0 else (5500, 12000)
    eid = K.spawn1(g, team, H.GIANT, x, y, deployed=True)
    far_id = int(K.tower_entity(g, enemy, 2)["id"])
    for _ in range(900):
        g.tick(1)
        e = K.ent(g, eid)
        assert e is not None, "Giant died before reaching the King"
        assert int(e["target_id"]) != far_id, "unit crossed to the far lane's princess"
        if K.tower_hp(g, enemy, 0) < H.KING_HP:
            break
    assert K.tower_hp(g, enemy, 0) < H.KING_HP, "the Giant never attacked the King"
    assert K.tower_hp(g, enemy, 2) == H.PRINCESS_HP


# ------------------------------------------------------------------------------------------
# Collision (SPEC §6.4)
# ------------------------------------------------------------------------------------------
def test_coincident_units_separate(pr):
    g = K.new_game()
    a = K.spawn1(g, 0, H.KNIGHT, 9000, 22000, deployed=True)
    b = K.spawn1(g, 0, H.KNIGHT, 9000, 22000, deployed=True)
    g.tick(3)
    for _ in range(20):
        g.tick(1)
        d = math.dist(K.pos(K.ent(g, a)), K.pos(K.ent(g, b)))
        assert d >= 900, f"two Knights stacked: centre distance {d:.0f} (radii sum 1000)"


def test_ground_unit_pushed_out_of_tower_circle(pr):
    g = K.new_game()
    kn = K.spawn1(g, 0, H.KNIGHT, 3500, 25500 - 800, deployed=True)   # inside own princess circle
    g.tick(1)
    d = math.dist(K.pos(K.ent(g, kn)), H.TOWER_POS[0][1])
    assert d >= 1500 - 1, f"Knight left {d:.0f} from the tower centre (needs >= 1000 + 500)"


def test_buildings_and_towers_never_move(pr):
    g = K.new_game()
    c = K.spawn1(g, 1, H.CANNON, 9000, 10000, deployed=True)
    K.spawn1(g, 0, H.GIANT, 9000, 14000, deployed=True)
    K.spawn(g, 0, H.SKARMY, 9000, 12500, deployed=True)
    towers0 = {(int(e["team"]), K.pos(e)) for e in K.units(g, kind="tower")}
    for _ in range(120):
        g.tick(1)
        e = K.ent(g, c)
        if e is None:
            break
        assert K.pos(e) == (9000, 10000), "a building moved"
        assert {(int(t["team"]), K.pos(t)) for t in K.units(g, kind="tower")} <= towers0


def test_units_never_overlap_buildings_after_collide(pr):
    g = K.new_game()
    c = K.spawn1(g, 1, H.CANNON, 9000, 10000, deployed=True)
    ids = K.spawn(g, 0, H.SKARMY, 9000, 12000, deployed=True)
    for _ in range(80):
        g.tick(1)
        ce = K.ent(g, c)
        if ce is None:
            break
        for e in K.troops(g, 0):
            if not bool(e["flying"]):
                assert math.dist(K.pos(e), K.pos(ce)) >= 600 + 500 - 1, "skeleton inside the Cannon"
