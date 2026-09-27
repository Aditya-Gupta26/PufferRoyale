"""SPEC §7.2 spells (Fireball, Arrows, Zap, The Log, Goblin Barrel), §2 crown-tower reduction,
§6.7 stun, §7 knockback.

Flight-time tests are RELATIVE where the spec leaves the launch tick's alignment open: two casts
at different distances d1 < d2 from the caster's King centre must land exactly
ceil(d2/v) - ceil(d1/v) ticks apart; the absolute landing tick is checked within +/-1.
"""
import math

import pytest

import gamekit as K
import helpers as H

KING0 = H.TOWER_POS[0][0]      # (9000, 29000): team 0 casts from here


def ceil_div(a, b):
    return -((-a) // b)


def landing_obs(g, pred, limit=120):
    """Number of g.tick(1) calls until pred(g) (1-based)."""
    return K.run_until(g, pred, limit, "spell effect")


# ------------------------------------------------------------------------------------------
# Crown-tower reduction (SPEC §2)
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("card,deck,total", [(H.FIREBALL, "hog26", 172), (H.ZAP, "giant", 48),
                                             (H.ARROWS, "giant", 75), (H.LOG, "hog26", 35)])
def test_spell_damage_to_princess(pr, card, deck, total):
    g = K.new_game(deck, deck, deploy_lockout_ticks=0)
    x, y = H.TOWER_POS[1][1]
    # v0.2: the Log is own-half only; from own tile (3,17) it rolls 10100 up into the princess
    tap = (x, 17500) if card == H.LOG else (x, y)
    assert K.cast(g, 0, card, *tap) == H.OK
    drops = []
    hp = H.PRINCESS_HP
    for _ in range(90):
        g.tick(1)
        cur = K.tower_hp(g, 1, 1)
        if cur < hp:
            drops.append(hp - cur)
            hp = cur
    assert sum(drops) == total, f"{H.CARD_NAMES[card]} dealt {drops} to the princess, want {total}"
    per = {H.ARROWS: [25, 25, 25]}.get(card, [total])
    assert drops == per
    assert K.tower_hp(g, 1, 0) == H.KING_HP, "no splash onto the King"


@pytest.mark.parametrize("card,deck,total", [(H.FIREBALL, "hog26", 688), (H.ZAP, "giant", 192),
                                             (H.ARROWS, "giant", 366), (H.LOG, "hog26", 268)])
def test_spell_full_damage_to_troops_and_buildings(pr, card, deck, total):
    """Cannon is a building but NOT a crown tower: full damage (SPEC §2)."""
    g = K.new_game(deck, deck, deploy_lockout_ticks=0)
    c = K.spawn1(g, 1, H.CANNON, 9000, 11000, deployed=True)
    tap = (9500, 17500) if card == H.LOG else (9000, 11000)
    assert K.cast(g, 0, card, *tap) == H.OK
    prev = K.hp_of(g, c)
    dealt = 0
    for _ in range(80):
        g.tick(1)
        cur = K.hp_of(g, c)
        if cur is None:
            break
        if prev - cur >= 50:
            dealt += prev - cur
        prev = cur
    n_hits = 3 if card == H.ARROWS else 1
    assert total <= dealt <= total + 2 * n_hits, f"{H.CARD_NAMES[card]} dealt {dealt} (+drain) to a Cannon"


# ------------------------------------------------------------------------------------------
# Zap (instant, stun)
# ------------------------------------------------------------------------------------------
def test_zap_instant_damage_stun_and_radius(pr):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    a = K.spawn1(g, 1, H.KNIGHT, 9000, 11000, deployed=True)        # centre
    b = K.spawn1(g, 1, H.KNIGHT, 9000 + 2900, 11000, deployed=True)  # 2900 <= 2500 + 500
    c = K.spawn1(g, 1, H.KNIGHT, 9000 - 3100, 11000, deployed=True)  # 3100 > 3000
    m = K.spawn(g, 1, H.MINIONS, 9000, 12000, deployed=True)          # air, inside
    own = K.spawn1(g, 0, H.KNIGHT, 9000, 11600, deployed=True)        # own unit, inside
    assert K.cast(g, 0, H.ZAP, 9000, 11000) == H.OK
    g.tick(1)
    kn = H.expected_unit_stats(H.KNIGHT)["hp"]
    assert K.hp_of(g, a) == kn - 192 and bool(K.ent(g, a)["stunned"])
    assert K.hp_of(g, b) == kn - 192 and bool(K.ent(g, b)["stunned"])
    assert K.hp_of(g, c) == kn and not bool(K.ent(g, c)["stunned"])
    for i in m:
        e = K.ent(g, i)
        assert e is not None and int(e["hp"]) == 230 - 192 and bool(e["stunned"])
    assert K.hp_of(g, own) == kn and not bool(K.ent(g, own)["stunned"]), "no friendly fire"


def test_zap_stun_duration_and_freeze_in_place(pr):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    a = K.spawn1(g, 1, H.KNIGHT, 9000, 11000, deployed=True)
    g.tick(5)                                   # walking
    assert K.cast(g, 0, H.ZAP, *K.pos(K.ent(g, a))) == H.OK
    g.tick(1)
    p = K.pos(K.ent(g, a))
    n = 0
    while bool(K.ent(g, a)["stunned"]):
        n += 1
        assert K.pos(K.ent(g, a)) == p, "stunned unit moved"
        assert int(K.ent(g, a)["target_id"]) == -1, "stun clears the target"
        assert n <= 15
        g.tick(1)
    assert n == 10, f"Zap stun observed for {n} ticks; SPEC §13.3: blocks ticks P..P+9"
    g.tick(3)
    assert K.pos(K.ent(g, a)) != p, "unit resumes moving after the stun"


def test_zap_stuns_crown_tower_and_resets_its_attack(pr):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    gi = K.spawn1(g, 0, H.GIANT, 3500, 6500 + 2600, deployed=True)   # the princess shoots it
    g.tick(30)
    assert K.cast(g, 0, H.ZAP, *H.TOWER_POS[1][1]) == H.OK
    g.tick(1)
    t = K.tower_entity(g, 1, 1)
    assert bool(t["stunned"]) is True, "crown towers can be stunned"
    assert int(t["target_id"]) == -1
    assert K.tower_hp(g, 1, 1) <= H.PRINCESS_HP - 48


def test_zap_resets_attack_progress(pr):
    """Knight hitting a static princess; Zap it mid-cycle. After the stun it starts a FRESH cycle:
    its next hit lands exactly (1200-700)/50 - 1 = 9 ticks after its first unstunned tick."""
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    kn = K.spawn1(g, 0, H.KNIGHT, 3500, 6500 + 2400, deployed=False)
    hp = H.PRINCESS_HP
    K.run_until(g, lambda gg: K.tower_hp(gg, 1, 1) < hp, 80, "first Knight hit")
    g.tick(12)                                  # 12 ticks into the 24-tick cycle
    hp = K.tower_hp(g, 1, 1)
    # team 1 zaps its own attacker's position (team 1 casts on the team-0 Knight)
    assert K.cast(g, 1, H.ZAP, *K.pos(K.ent(g, kn))) == H.OK
    g.tick(1)
    assert bool(K.ent(g, kn)["stunned"])
    K.run_until(g, lambda gg: not bool(K.ent(gg, kn)["stunned"]), 15, "stun end")
    assert K.tower_hp(g, 1, 1) == hp, "no hit while stunned"
    n = K.run_until(g, lambda gg: K.tower_hp(gg, 1, 1) < hp, 40, "first hit after stun")
    assert n == 9, f"first hit {n} ticks after the first unstunned observation, want 9"


# ------------------------------------------------------------------------------------------
# Fireball (projectile from the King, radius, knockback)
# ------------------------------------------------------------------------------------------
def _fireball_land(tap):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    c = K.spawn1(g, 1, H.CANNON, *tap, deployed=True)
    hp0 = K.hp_of(g, c)
    assert K.cast(g, 0, H.FIREBALL, *tap) == H.OK
    return landing_obs(g, lambda gg: hp0 - K.hp_of(gg, c) >= 600)


def test_fireball_flight_time_from_king(pr):
    t1, t2 = (9000, 14100), (9000, 8100)            # 14900 and 20900 from the caster's King
    d1, d2 = math.dist(t1, KING0), math.dist(t2, KING0)
    a, b = _fireball_land(t1), _fireball_land(t2)
    assert b - a == ceil_div(int(d2), 600) - ceil_div(int(d1), 600) == 10, (a, b)
    # SPEC §13.12: lands in tick P + ceil(d/v) - 1, i.e. observation ceil(d/v) after queueing
    assert a == ceil_div(int(d1), 600) == 25 and b == 35, (a, b)


def test_fireball_radius_and_air(pr):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    tap = (9000, 11000)
    inside = K.spawn1(g, 1, H.CANNON, 9000 + 3000, 11000, deployed=True)     # 3000 <= 2500 + 600
    outside = K.spawn1(g, 1, H.CANNON, 9000 - 3200, 11000, deployed=True)    # 3200 > 3100
    # a flyer that drifts from (9000, 9000) toward team 0 and is near the tap on arrival (~25 ticks)
    bd = K.spawn1(g, 1, H.BABY_DRAGON, 9000, 9000, deployed=True)
    hp = {i: K.hp_of(g, i) for i in (inside, outside, bd)}
    assert K.cast(g, 0, H.FIREBALL, *tap) == H.OK
    landing_obs(g, lambda gg: hp[inside] - K.hp_of(gg, inside) >= 600, 60)
    assert hp[outside] - K.hp_of(g, outside) < 150, "outside 2500 + r_target (only lifetime drain)"
    e = K.ent(g, bd)
    assert e is not None and hp[bd] - int(e["hp"]) == 688, "Fireball hits air for 688"


def test_fireball_no_friendly_fire(pr):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    own = K.spawn1(g, 0, H.GIANT, 9000, 20000, deployed=False)      # deploying: stands still
    enemy = K.spawn1(g, 1, H.CANNON, 9000, 20000 + 1500, deployed=False)
    hp_o, hp_e = K.hp_of(g, own), K.hp_of(g, enemy)
    assert K.cast(g, 0, H.FIREBALL, 9000, 20500) == H.OK
    landing_obs(g, lambda gg: hp_e - K.hp_of(gg, enemy) >= 600, 40)
    assert K.hp_of(g, own) == hp_o, "the caster's own units are never hit"


@pytest.mark.parametrize("card,pushed", [(H.KNIGHT, True), (H.GIANT, False), (H.PRINCE, False)])
def test_fireball_knockback_radial_1000(pr, card, pushed):
    """Team-1 unit attacking team 0's left princess from 2400 below-right... pushed +x by 1000
    from an impact 1000 to its left; it stays in range, so its final position is exact."""
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    st = H.expected_unit_stats(card)
    ux, uy = 3500, 25500 - 2400 if card != H.GIANT else 25500 - 2600
    u = K.spawn1(g, 1, card, ux, uy, deployed=True)
    g.tick(3)
    assert K.pos(K.ent(g, u)) == (ux, uy), "unit in range of the tower must stand still"
    hp0 = K.hp_of(g, u)
    assert K.cast(g, 0, H.FIREBALL, ux - 1000, uy) == H.OK
    landing_obs(g, lambda gg: hp0 - K.hp_of(gg, u) >= 600, 40)
    g.tick(20)
    want = (ux + 1000, uy) if pushed else (ux, uy)
    got = K.pos(K.ent(g, u))
    assert abs(got[0] - want[0]) <= 5 and abs(got[1] - want[1]) <= 5, \
        f"{H.CARD_NAMES[card]} ended at {got}, want {want} (ignore_pushback={not pushed})"
    assert st["hp"] > 688


# ------------------------------------------------------------------------------------------
# Arrows (3 waves)
# ------------------------------------------------------------------------------------------
def _arrows_waves(tap):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    c = K.spawn1(g, 1, H.CANNON, *tap, deployed=True)
    assert K.cast(g, 0, H.ARROWS, *tap) == H.OK
    prev = K.hp_of(g, c)
    hits = []
    for n in range(1, 60):
        g.tick(1)
        cur = K.hp_of(g, c)
        if prev - cur >= 100:
            hits.append((n, prev - cur))
        prev = cur
    return hits


def test_arrows_three_waves_200ms_apart(pr):
    t1, t2 = (9000, 14100), (9000, 8300)            # 14900 and 20700 from the caster's King
    h1, h2 = _arrows_waves(t1), _arrows_waves(t2)
    for h in (h1, h2):
        assert len(h) == 3, f"Arrows waves {h}"
        assert [t - h[0][0] for t, _ in h] == [0, 4, 8], f"waves not 4 ticks apart: {h}"
        for _, amt in h:
            assert 122 <= amt <= 124
    d1, d2 = math.dist(t1, KING0), math.dist(t2, KING0)
    assert h2[0][0] - h1[0][0] == ceil_div(int(d2), 1100) - ceil_div(int(d1), 1100) == 5
    assert h1[0][0] == ceil_div(int(d1), 1100) == 14, "SPEC §13.12: first wave in tick P + ceil(d/1100) - 1"


def test_arrows_radius_air_and_one_hit_per_wave(pr):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    tap = (9000, 11000)
    mins = K.spawn(g, 1, H.MINIONS, 9000, 11000, deployed=True)
    edge = K.spawn1(g, 1, H.CANNON, 9000 + 4000, 11000, deployed=True)    # 4000 <= 3500 + 600
    far = K.spawn1(g, 1, H.CANNON, 9000 - 4200, 11000, deployed=True)     # 4200 > 4100
    hp_e, hp_f = K.hp_of(g, edge), K.hp_of(g, far)
    assert K.cast(g, 0, H.ARROWS, *tap) == H.OK
    g.tick(45)
    assert hp_e - K.hp_of(g, edge) >= 3 * 122
    assert hp_e - K.hp_of(g, edge) <= 3 * 122 + 2 * 45
    assert hp_f - K.hp_of(g, far) <= 2 * 45, "outside 3500 + r_target"
    assert all(K.ent(g, i) is None for i in mins), "two waves of 122 kill 230-hp Minions"


# ------------------------------------------------------------------------------------------
# The Log
# ------------------------------------------------------------------------------------------
def test_log_ground_only_and_once_each(pr):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    gi = K.spawn1(g, 1, H.GIANT, 9000, 13000, deployed=True)
    kn = K.spawn1(g, 1, H.KNIGHT, 10500, 12000, deployed=True)
    mins = K.spawn(g, 1, H.MINIONS, 9000, 12500, deployed=True)
    hp = {i: K.hp_of(g, i) for i in [gi, kn] + mins}
    assert K.cast(g, 0, H.LOG, 9500, 17500) == H.OK
    for _ in range(70):
        g.tick(1)
    assert hp[gi] - K.hp_of(g, gi) == 268, "the Log hits each ground target exactly once"
    assert hp[kn] - K.hp_of(g, kn) == 268
    for i in mins:
        e = K.ent(g, i)
        assert e is not None and int(e["hp"]) == hp[i], "the Log never hits air units"


def test_log_no_friendly_fire(pr):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    own = K.spawn1(g, 0, H.KNIGHT, 9000, 20000, deployed=False)
    enemy = K.spawn1(g, 1, H.KNIGHT, 9000, 19000, deployed=False)
    hp_o, hp_e = K.hp_of(g, own), K.hp_of(g, enemy)
    assert K.cast(g, 0, H.LOG, 9500, 21500) == H.OK
    landing_obs(g, lambda gg: hp_e - K.hp_of(gg, enemy) >= 268, 30)
    assert K.hp_of(g, own) == hp_o


def test_log_roll_length_and_width(pr):
    """Log from the tile centre (9500, 22500) rolls to y = 12400; hitbox half-width 1950 and
    half-depth 600 against target circles (Cannon radius 600)."""
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    x0, y0 = 9500, 22500
    y_end = y0 - 10100
    near = K.spawn1(g, 1, H.CANNON, x0, y_end - 600 - 600 + 400, deployed=True)   # overlaps by 400
    beyond = K.spawn1(g, 1, H.CANNON, x0, y_end - 600 - 600 - 400, deployed=True)
    side_in = K.spawn1(g, 1, H.CANNON, x0 + 1950 + 600 - 300, 13000, deployed=True)
    side_out = K.spawn1(g, 1, H.CANNON, x0 - 1950 - 600 - 300, 13000, deployed=True)
    ids = [near, beyond, side_in, side_out]
    hp = {i: K.hp_of(g, i) for i in ids}
    assert K.cast(g, 0, H.LOG, x0, y0) == H.OK
    for _ in range(80):
        g.tick(1)
    lost = {i: hp[i] - K.hp_of(g, i) for i in ids}
    drain = 80 * 2
    assert 268 <= lost[near] <= 268 + drain, lost
    assert 268 <= lost[side_in] <= 268 + drain, lost
    assert lost[beyond] <= drain, lost
    assert lost[side_out] <= drain, lost


@pytest.mark.parametrize("card,dist", [(H.KNIGHT, 1900), (H.GIANT, 2100)])
def test_log_knockback_700_in_roll_direction(pr, card, dist):
    """Team-1 unit attacking team 0's left princess from `dist` below... the Log (team 0) rolls
    toward decreasing y and pushes it 700 along -y (pushback_all: the Giant too); it stays in
    range afterwards, so the final position is exact."""
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    ux, uy = 3500, 25500 - dist
    u = K.spawn1(g, 1, card, ux, uy, deployed=True)
    g.tick(3)
    assert K.pos(K.ent(g, u)) == (ux, uy)
    hp0 = K.hp_of(g, u)
    assert K.cast(g, 0, H.LOG, 3500, 25500) == H.OK       # own tile (3,25), over own princess
    landing_obs(g, lambda gg: hp0 - K.hp_of(gg, u) >= 268, 30)
    g.tick(20)
    got = K.pos(K.ent(g, u))
    assert abs(got[0] - ux) <= 5 and abs(got[1] - (uy - 700)) <= 5, \
        f"{H.CARD_NAMES[card]} at {got}, want {(ux, uy - 700)}"
    tower_id = int(K.tower_entity(g, 0, 1)["id"])
    assert int(K.ent(g, u)["target_id"]) == tower_id, "knockback keeps the target"


# ------------------------------------------------------------------------------------------
# Goblin Barrel
# ------------------------------------------------------------------------------------------
def goblins(g, team):
    return [e for e in K.troops(g, team) if int(e["max_hp"]) == H.GOBLIN["hp"]]


def _barrel_first_seen(tap, team=0):
    g = K.new_game("bait", "bait", deploy_lockout_ticks=0)
    assert K.cast(g, team, H.GOBLIN_BARREL, *tap) == H.OK
    n = landing_obs(g, lambda gg: len(goblins(gg, team)) > 0, 120)
    return g, n


def test_goblin_barrel_flight_and_exactly_three(pr):
    t1, t2 = (9000, 14100), (9000, 8300)
    g1, a = _barrel_first_seen(t1)
    g2, b = _barrel_first_seen(t2)
    d1, d2 = math.dist(t1, KING0), math.dist(t2, KING0)
    assert b - a == ceil_div(int(d2), 400) - ceil_div(int(d1), 400) == 14, (a, b)
    # SPEC §13.12: lands in tick P + ceil(d/400) - 1; Goblins committed in the next tick's Spawn
    assert a == ceil_div(int(d1), 400) + 1 == 39, a
    for g in (g1, g2):
        gs = goblins(g, 0)
        assert len(gs) == 3, f"{len(gs)} goblins"
        g.tick(10)
        assert len(goblins(g, 0)) == 3, "exactly 3 goblins, no late extras"


@pytest.mark.parametrize("team", [0, 1])
def test_goblin_barrel_triangle_and_stats(pr, team):
    tap_own = (9500, 12500)
    tap = H.own_to_engine_point(team, *tap_own)
    g, _ = _barrel_first_seen(tap, team)
    gs = goblins(g, team)
    offs = sorted((H.engine_to_own_point(team, *K.pos(e))[0] - tap_own[0],
                   H.engine_to_own_point(team, *K.pos(e))[1] - tap_own[1]) for e in gs)
    want = sorted([(0, -577), (500, 289), (-500, 289)])
    assert all(abs(a[0] - b[0]) <= 3 and abs(a[1] - b[1]) <= 3 for a, b in zip(offs, want)), offs
    for e in gs:
        assert int(e["hp"]) == 202 and bool(e["deploying"]) is True and not bool(e["flying"])
        assert int(e["radius"]) == 500


def test_goblin_card_id_maps_to_goblin_barrel(pr):
    g, _ = _barrel_first_seen((9000, 12000))
    assert all(int(e["card_id"]) == H.GOBLIN_BARREL for e in goblins(g, 0))


def test_goblin_deploy_1100ms_and_damage(pr):
    g = K.new_game("bait", "bait", deploy_lockout_ticks=0)
    tap = (3500, 6500 + 1600)
    assert K.cast(g, 0, H.GOBLIN_BARREL, *tap) == H.OK
    landing_obs(g, lambda gg: len(goblins(gg, 0)) == 3, 120)
    ids = [int(e["id"]) for e in goblins(g, 0)]
    D = 0                                # observations with every goblin still deploying
    while all(K.ent(g, i) is not None and bool(K.ent(g, i)["deploying"]) for i in ids):
        D += 1
        assert D < 40
        g.tick(1)
    assert D == 22, f"goblins deploying for {D} observations (SPEC §13.2: 1100/50 = 22)"
    hp = K.tower_hp(g, 1, 1)
    for _ in range(80):
        g.tick(1)
        cur = K.tower_hp(g, 1, 1)
        if cur < hp:
            assert (hp - cur) % 125 == 0, f"goblin damage {hp - cur} not a multiple of 125"
            hp = cur
    assert hp < H.PRINCESS_HP, "goblins never hit the princess"


def test_goblin_barrel_no_impact_damage_on_king(pr):
    g = K.new_game("bait", "bait", deploy_lockout_ticks=0)
    assert K.cast(g, 0, H.GOBLIN_BARREL, *H.TOWER_POS[1][0]) == H.OK
    landing_obs(g, lambda gg: len(goblins(gg, 0)) == 3, 150)
    assert K.tower_hp(g, 1, 0) == H.KING_HP, "the barrel itself deals no damage"
    for e in goblins(g, 0):
        d = math.dist(K.pos(e), H.TOWER_POS[1][0])
        assert d >= 1400 + 500 - 1, "goblins are pushed out of the King's collision circle"


@pytest.mark.parametrize("team", [0, 1])
def test_goblin_barrel_goblins_not_in_water(pr, team):
    """Tap on the bank tile (9,17) own frame: the apex member (0,-577) would be in the river."""
    tap = H.tile_centre_engine(team, 9, 17)
    g, _ = _barrel_first_seen(tap, team)
    for _ in range(25):
        for e in goblins(g, team):
            assert not H.point_in_water(*K.pos(e)), f"goblin in water at {K.pos(e)}"
        g.tick(1)
