"""SPEC §6.1 targeting, §6.2 attack cycle (progress credit), §7.1 projectiles, splash, damage.

Timing model (derived from SPEC §6.2 / §7.1, written out so it can be audited):
  * A unit spawned with deployed=False shows deploying=True for D observations, then acts on the
    next tick E (the first observation with deploying=False). Its load_timer starts at load_time
    and loses 50 every processed tick, including tick E (decrement happens first in Attack).
  * On E (target in range, fresh cycle): progress = load_time - load_timer_E, then += 50.
    The hit lands on E + k, k = min k >= 0 with progress_E + 50k >= hit_speed. With a fully
    loaded unit this is the SPEC's (hit_speed - load_time)/50 - 1.
  * Later hits every hit_speed/50 ticks.
  * A projectile is created on the hit tick H at start_radius toward the target and first moves
    on H+1; it impacts on the first tick whose remaining centre distance is <= speed:
    impact = H + max(1, ceil((d - start_radius) / speed)) for a static target at distance d.
  * Melee / Tesla damage resolves on the hit tick itself.
All damage is observed in the state after the tick in which Resolve applied it.
"""
import pytest

import gamekit as K
import helpers as H

T1_LEFT = H.TOWER_POS[1][1]      # (3500, 6500): team-1 left princess, used as a static target


def first_hit_k(hs, lt, D):
    rem = max(0, lt - 50 * (D + 1))
    p = lt - rem + 50
    k = 0
    while p + 50 * k < hs:
        k += 1
    return k


def flight_ticks(d, start_radius, speed):
    return max(1, -((-(d - start_radius)) // speed))


def observe_deploy(g, eid, max_ticks=60):
    """Return (D, obs_index_of_E) where obs indices count g.tick(1) calls from now (1-based)."""
    D = 0
    for n in range(1, max_ticks + 1):
        g.tick(1)
        e = K.ent(g, eid)
        assert e is not None, f"entity {eid} vanished while deploying"
        if bool(e["deploying"]):
            D += 1
            assert int(e["target_id"]) == -1, "deploying units do not target (SPEC §5)"
        else:
            return D, n
    raise AssertionError("deploy never finished")


# (card, distance below the tower, projectile (start_radius, speed) or None for melee)
VS_TOWER = [
    (H.KNIGHT, 2400, None),
    (H.VALKYRIE, 2400, None),
    (H.GIANT, 2600, None),
    (H.HOG, 2000, None),
    (H.ICE_GOLEM, 2000, None),
    (H.PRINCE, 2800, None),
    (H.MUSKETEER, 7000, (450, 1000)),
    (H.WIZARD, 6500, (550, 600)),
    (H.BABY_DRAGON, 4400, (500, 500)),
    (H.CANNON, 6800, (1000, 1000)),
]


def test_tester_projectile_data_matches_source():
    for card, _, proj in VS_TOWER:
        if proj is None:
            assert H.src_card(card).get("projectile") is None
            continue
        c = H.src_card(card)
        assert (c["projectile_start_radius_milli"], c["projectile"]["speed"]) == proj


@pytest.mark.parametrize("card,d,proj", VS_TOWER, ids=[H.CARD_NAMES[c[0]] for c in VS_TOWER])
def test_first_hit_cadence_and_damage_vs_static_tower(pr, card, d, proj):
    g = K.new_game()
    st = H.expected_unit_stats(card)
    x, y = T1_LEFT[0], T1_LEFT[1] + d
    assert d <= st["range"] + st["radius"] + 1000, "tester setup: must start in attack range"
    eid = K.spawn1(g, 0, card, x, y, deployed=False)
    D, iE = observe_deploy(g, eid)
    assert D == st["deploy_time_ms"] // 50, f"exactly D/50 deploying observations (SPEC §13.2), got {D}"
    tower_id = int(K.tower_entity(g, 1, 1)["id"])
    assert int(K.ent(g, eid)["target_id"]) == tower_id, "unit must target the princess on its first acting tick"
    hs = st["hit_speed_ms"]
    k = first_hit_k(hs, st["load_time_ms"], D)
    j = 0 if proj is None else flight_ticks(d, proj[0], proj[1])
    cad = hs // 50
    want = [iE + k + j + m * cad for m in range(3)]
    hp = H.PRINCESS_HP
    assert K.tower_hp(g, 1, 1) == hp, "no damage may land before the first hit"
    drops = []
    for n in range(iE + 1, want[-1] + 3):
        g.tick(1)
        cur = K.tower_hp(g, 1, 1)
        if cur != hp:
            drops.append((n, hp - cur))
            hp = cur
    got_ticks = [t for t, _ in drops]
    assert got_ticks[:3] == want, (f"{H.CARD_NAMES[card]}: hits observed at {got_ticks} "
                                   f"(D={D}, E at obs {iE}); expected {want} "
                                   f"[k={k}, flight={j}, cadence={cad}]")
    for t, amt in drops:
        assert amt == st["damage"], f"{H.CARD_NAMES[card]} hit for {amt}, want {st['damage']}"


@pytest.mark.parametrize("card,d,proj", VS_TOWER, ids=[H.CARD_NAMES[c[0]] for c in VS_TOWER])
def test_spawn_deployed_true_equals_just_deployed_unit(pr, card, d, proj):
    """SPEC §13.10: spawn(deployed=True) = a hand-played unit in the first tick after its deploy,
    i.e. load_timer = max(0, load_time - deploy_time). It acts on the first tick (obs 1), so its
    first hit is at obs 1 + k(D=deploy/50) (+ flight), exactly like a deployed=False unit."""
    g = K.new_game()
    st = H.expected_unit_stats(card)
    eid = K.spawn1(g, 0, card, T1_LEFT[0], T1_LEFT[1] + d, deployed=True)
    assert not bool(K.ent(g, eid)["deploying"])
    k = first_hit_k(st["hit_speed_ms"], st["load_time_ms"], st["deploy_time_ms"] // 50)
    j = 0 if proj is None else flight_ticks(d, proj[0], proj[1])
    cad = st["hit_speed_ms"] // 50
    want = [1 + k + j + m * cad for m in range(2)]
    hp, drops = H.PRINCESS_HP, []
    for n in range(1, want[-1] + 2):
        g.tick(1)
        cur = K.tower_hp(g, 1, 1)
        if cur != hp:
            drops.append(n)
            hp = cur
    assert drops[:2] == want, f"{H.CARD_NAMES[card]}: hits at {drops}, want {want}"


@pytest.mark.parametrize("card,per_hit", [(H.ARCHERS, 112), (H.MINIONS, 107), (H.TESLA, 220),
                                          (H.SKELETONS, 81), (H.SKARMY, 81)])
def test_multi_or_special_units_damage_quantum(pr, card, per_hit):
    """Every tower hp drop must be a whole number of hits of the level-11 damage."""
    g = K.new_game()
    st = H.expected_unit_stats(card)
    assert st["damage"] == per_hit
    d = min(st["range"] + st["radius"] + 1000 - 300, 6000)
    K.spawn(g, 0, card, T1_LEFT[0], T1_LEFT[1] + max(d, 1800), deployed=True)
    hp = H.PRINCESS_HP
    total = 0
    for _ in range(160):
        g.tick(1)
        if not K.tower_alive(g, 1, 1):
            break                       # the killing blow may be clamped at 0 hp
        cur = K.tower_hp(g, 1, 1)
        if cur < hp:
            assert (hp - cur) % per_hit == 0, f"drop {hp - cur} is not a multiple of {per_hit}"
            total += hp - cur
            hp = cur
    assert total > 0, f"{H.CARD_NAMES[card]} never damaged the princess"


def test_princess_first_shot_and_cadence_on_static_building(pr):
    """Tower load_time 0, hit_speed 800: first arrow 15 ticks after acquisition, then every 16;
    projectile speed 600 from start radius 300 (SPEC §2)."""
    g = K.new_game()
    d = 5500
    cid = K.spawn1(g, 0, H.CANNON, T1_LEFT[0], T1_LEFT[1] + d, deployed=True)
    # acquisition on the first tick after spawn (obs 1); hit on obs 1 + 15; flight
    j = flight_ticks(d, 300, 600)            # ceil(5200/600) = 9
    want = [1 + 15 + j + 16 * m for m in range(3)]
    prev = H.expected_unit_stats(H.CANNON)["hp"]
    big = []
    for n in range(1, want[-1] + 3):
        g.tick(1)
        cur = K.hp_of(g, cid)
        assert cur is not None
        if prev - cur >= 50:                 # lifetime drain is <= 2 per tick; a hit is 109
            big.append((n, prev - cur))
        prev = cur
    assert [t for t, _ in big][:3] == want, f"princess arrows landed at {big}, want ticks {want}"
    for _, amt in big:
        assert H.TOWER_DMG <= amt <= H.TOWER_DMG + 2, f"arrow damage {amt} (109 + drain <= 2)"


def test_valkyrie_self_centred_splash(pr):
    g = K.new_game()
    c1 = K.spawn1(g, 1, H.CANNON, 9000, 12000, deployed=False)    # target: 2000 away
    c2 = K.spawn1(g, 1, H.CANNON, 11000, 13000, deployed=False)   # 2236 <= 2000 + 600
    c3 = K.spawn1(g, 1, H.CANNON, 6300, 14000, deployed=False)    # 2700 > 2600: outside
    v = K.spawn1(g, 0, H.VALKYRIE, 9000, 14000, deployed=False)
    cs = [c1, c2, c3]
    prev = {c: K.hp_of(g, c) for c in cs}
    hits = {c: [] for c in cs}
    first_target = None
    for n in range(1, 80):              # window holds exactly the Valkyrie's first two swings
        g.tick(1)
        t = int(K.ent(g, v)["target_id"])
        if first_target is None and t != -1:
            first_target = t
        for c in cs:
            cur = K.hp_of(g, c)
            assert cur is not None
            if prev[c] - cur >= 100:
                hits[c].append((n, prev[c] - cur))
            prev[c] = cur
    assert first_target == c1, "nearest by (dist - r_target) is the 2000-away Cannon"
    assert len(hits[c1]) >= 2, f"Valkyrie target hits: {hits[c1]}"
    assert [t for t, _ in hits[c2]][:2] == [t for t, _ in hits[c1]][:2], "splash lands on the same tick"
    for c in (c1, c2):
        for _, amt in hits[c]:
            assert 266 <= amt <= 268, f"Valkyrie splash {amt} (266 + drain)"
    assert hits[c3] == [], f"unit outside 2000 + r_target was hit: {hits[c3]}"


def test_wizard_projectile_splash(pr):
    g = K.new_game()
    c1 = K.spawn1(g, 1, H.CANNON, 9000, 9000, deployed=False)     # target, 5000 from Wizard
    c2 = K.spawn1(g, 1, H.CANNON, 10800, 9000, deployed=False)    # 1800 <= 1500 + 600 of impact
    c3 = K.spawn1(g, 1, H.CANNON, 9000, 6700, deployed=False)     # 2300 > 2100: outside
    w = K.spawn1(g, 0, H.WIZARD, 9000, 14000, deployed=False)
    cs = [c1, c2, c3]
    prev = {c: K.hp_of(g, c) for c in cs}
    hits = {c: [] for c in cs}
    for n in range(1, 52):              # covers the first impact only (Wizard may die later)
        g.tick(1)
        for c in cs:
            cur = K.hp_of(g, c)
            if cur is None:
                continue
            if prev[c] - cur >= 100:
                hits[c].append((n, prev[c] - cur))
            prev[c] = cur
    assert len(hits[c1]) >= 1, "Wizard never hit its target"
    assert [t for t, _ in hits[c2]] == [t for t, _ in hits[c1]]
    for c in (c1, c2):
        for _, amt in hits[c]:
            assert 281 <= amt <= 283
    assert hits[c3] == []


@pytest.mark.parametrize("attacker,expect_targets_air", [(H.KNIGHT, False), (H.VALKYRIE, False),
                                                         (H.MUSKETEER, True), (H.WIZARD, True),
                                                         (H.ARCHERS, True)])
def test_air_ground_filter(pr, attacker, expect_targets_air):
    g = K.new_game()
    mids = K.spawn(g, 1, H.MINIONS, 9000, 12500, deployed=True)
    aids = K.spawn(g, 0, attacker, 9000, 14000, deployed=True)
    targeted = set()
    for _ in range(40):
        g.tick(1)
        for a in aids:
            e = K.ent(g, a)
            if e is not None:
                targeted.add(int(e["target_id"]))
    hit_air = bool(targeted & set(mids))
    assert hit_air == expect_targets_air, f"{H.CARD_NAMES[attacker]} targets={targeted} minions={mids}"


def test_cannon_ignores_air(pr):
    g = K.new_game()
    c = K.spawn1(g, 0, H.CANNON, 9000, 21000, deployed=True)
    mids = K.spawn(g, 1, H.MINIONS, 9000, 18500, deployed=True)
    for _ in range(40):
        g.tick(1)
        assert int(K.ent(g, c)["target_id"]) not in mids


@pytest.mark.parametrize("bt", [H.GIANT, H.HOG, H.ICE_GOLEM])
def test_building_only_targeters_ignore_troops(pr, bt):
    g = K.new_game()
    b = K.spawn1(g, 0, bt, 9000, 14200, deployed=True)
    kn = K.spawn1(g, 1, H.KNIGHT, 9000, 12800, deployed=True)
    for _ in range(40):
        g.tick(1)
        e = K.ent(g, b)
        assert e is not None
        assert int(e["target_id"]) != kn, f"{H.CARD_NAMES[bt]} targeted a troop"


def test_building_only_targeter_targets_enemy_building(pr):
    g = K.new_game()
    c = K.spawn1(g, 1, H.CANNON, 9000, 10000, deployed=True)
    gi = K.spawn1(g, 0, H.GIANT, 9000, 14000, deployed=True)
    K.run_until(g, lambda gg: int(K.ent(gg, gi)["target_id"]) == c, 5, "Giant acquiring the Cannon")


@pytest.mark.parametrize("d,acquire", [(6000, True), (6100, True), (6200, False)])
def test_sight_range_uses_target_radius(pr, d, acquire):
    """Sight test: dist <= sight + r_target (Knight 5500 + Cannon 600 = 6100)."""
    g = K.new_game()
    c = K.spawn1(g, 1, H.CANNON, 9000, 14000 - d, deployed=True)
    kn = K.spawn1(g, 0, H.KNIGHT, 9000, 14000, deployed=True)
    g.tick(1)
    tid = int(K.ent(g, kn)["target_id"])
    assert (tid == c) == acquire, f"d={d}: target_id={tid}, cannon={c}"


@pytest.mark.parametrize("d,acquire", [(8400, True), (8500, True), (8600, False)])
def test_crown_tower_extra_sight(pr, d, acquire):
    """Crown towers get +2000 sight: Knight sees the princess at 5500 + 1000 + 2000 = 8500."""
    g = K.new_game()
    kn = K.spawn1(g, 0, H.KNIGHT, T1_LEFT[0], T1_LEFT[1] + d, deployed=True)
    g.tick(1)
    tid = int(K.ent(g, kn)["target_id"])
    tower_id = int(K.tower_entity(g, 1, 1)["id"])
    assert (tid == tower_id) == acquire, f"d={d}: target {tid}"


def test_acquisition_picks_min_edge_distance(pr):
    """Minimum (dist - r_target): a Giant (r 750) at 3000 beats a Knight (r 500) at 2800."""
    g = K.new_game()
    kn = K.spawn1(g, 1, H.KNIGHT, 9000 + 2800, 13000, deployed=True)
    gi = K.spawn1(g, 1, H.GIANT, 9000 - 3000, 13000, deployed=True)
    m = K.spawn1(g, 0, H.MUSKETEER, 9000, 13000, deployed=True)
    g.tick(1)
    assert int(K.ent(g, m)["target_id"]) == gi


def test_deploying_unit_is_targetable_and_damageable(pr):
    g = K.new_game()
    kn = K.spawn1(g, 0, H.KNIGHT, T1_LEFT[0], T1_LEFT[1] + 5000, deployed=False)
    g.tick(1)
    assert bool(K.ent(g, kn)["deploying"]) is True
    assert int(K.tower_entity(g, 1, 1)["target_id"]) == kn, "a deploying unit is a valid target"


def test_dead_units_are_removed_same_tick(pr):
    g = K.new_game()
    sk = K.spawn(g, 1, H.SKELETONS, 9000, 12000, deployed=True)
    kn = K.spawn1(g, 0, H.KNIGHT, 9000, 13400, deployed=True)
    for _ in range(80):
        g.tick(1)
        for e in K.ents(g):
            assert int(e["hp"]) > 0, "entities with hp <= 0 must be reaped in the same tick"
            assert int(e["hp"]) <= int(e["max_hp"])


def test_tower_keeps_target_while_in_range(pr):
    """Buildings/towers keep their target while it stays in attack range, even if a nearer
    enemy appears (SPEC §6.1)."""
    g = K.new_game()
    gi = K.spawn1(g, 0, H.GIANT, T1_LEFT[0], T1_LEFT[1] + 8000, deployed=True)   # in range (9250)
    K.run_until(g, lambda gg: int(K.tower_entity(gg, 1, 1)["target_id"]) == gi, 3, "tower acquiring the Giant")
    kn = K.spawn1(g, 0, H.KNIGHT, T1_LEFT[0] + 2500, T1_LEFT[1] + 1500, deployed=True)
    for _ in range(30):
        g.tick(1)
        assert int(K.tower_entity(g, 1, 1)["target_id"]) == gi, "tower switched to a nearer unit"
    assert K.ent(g, kn) is not None


def test_acquisition_tie_breaks_on_lowest_entity_id(pr):
    g = K.new_game()
    a = K.spawn1(g, 1, H.KNIGHT, 9000 - 2000, 13000, deployed=True)
    b = K.spawn1(g, 1, H.KNIGHT, 9000 + 2000, 13000, deployed=True)
    m = K.spawn1(g, 0, H.MUSKETEER, 9000, 13000, deployed=True)
    g.tick(1)
    assert int(K.ent(g, m)["target_id"]) == min(a, b)


def test_knockback_resets_attack_cycle(pr):
    """A Knight hitting a static princess is Fireballed 4 ticks before its next scheduled hit. The
    knockback resets its cycle (SPEC §7), so the next hit cannot come on the old schedule; with a
    fully loaded fresh cycle it is at least 9 ticks after the impact."""
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    kn = K.spawn1(g, 1, H.KNIGHT, 3500, 25500 - 2400, deployed=True)   # attacks team 0's princess
    hp = H.PRINCESS_HP
    K.run_until(g, lambda gg: K.tower_hp(gg, 0, 1) < hp, 60, "first Knight hit")
    # Fireball flight from team 0's King to (2500, 23100) is ceil(8778/600) = 15 ticks (+/-1):
    # cast now + 5 so it lands ~20 ticks after this hit (next scheduled hit: +24).
    g.tick(5)
    hp = K.tower_hp(g, 0, 1)
    hk = K.hp_of(g, kn)
    assert K.cast(g, 0, H.FIREBALL, 2500, 23100) == H.OK
    L = K.run_until(g, lambda gg: hk - K.hp_of(gg, kn) >= 600, 30, "Fireball impact")
    assert K.tower_hp(g, 0, 1) == hp, "setup: the impact must precede the next scheduled hit (+24)"
    assert L <= 18
    hp = K.tower_hp(g, 0, 1)
    n = K.run_until(g, lambda gg: K.tower_hp(gg, 0, 1) < hp, 60, "first hit after the knockback")
    assert n >= 9, f"Knight hit {n} ticks after being knocked back: attack cycle was not reset"
    assert int(K.ent(g, kn)["target_id"]) == int(K.tower_entity(g, 0, 1)["id"])


@pytest.mark.parametrize("d,moves", [(2700, False), (2850, True)])
def test_attack_range_boundary(pr, d, moves):
    """Attack-range test dist <= range + r_attacker + r_target (Knight vs princess: 2700)."""
    g = K.new_game()
    kn = K.spawn1(g, 0, H.KNIGHT, T1_LEFT[0], T1_LEFT[1] + d, deployed=True)
    p0 = K.pos(K.ent(g, kn))
    K.run_until(g, lambda gg: K.tower_hp(gg, 1, 1) < H.PRINCESS_HP, 60, "Knight hit")
    moved = K.pos(K.ent(g, kn)) != p0
    assert moved == moves, f"d={d}: Knight moved={moved}"


def test_king_first_shot_timing(pr):
    """Active King (hit speed 1000, load 500, projectile 1000 from start radius 750, damage 109):
    first arrow 9 ticks after acquisition, flight ceil((8000-750)/1000) = 8, then every 20."""
    g = K.new_game("giant", "giant")
    g.tick(H.LOCKOUT_TICKS)
    K.destroy_tower(g, 1, 1)
    K.destroy_tower(g, 1, 2)
    g.tick(80)
    assert K.tower_active(g, 1, 0)
    cid = K.spawn1(g, 0, H.CANNON, 9000, 3000 + 8000, deployed=True)   # King reach 9000, Cannon 7500
    want = [1 + 9 + 8 + 20 * m for m in range(3)]
    prev = K.hp_of(g, cid)
    big = []
    for n in range(1, want[-1] + 3):
        g.tick(1)
        cur = K.hp_of(g, cid)
        if prev - cur >= 50:
            big.append((n, prev - cur))
        prev = cur
    assert [t for t, _ in big][:3] == want, f"King arrows at {big}, want {want}"
    for _, amt in big:
        assert 109 <= amt <= 111
    assert K.tower_hp(g, 1, 0) == H.KING_HP, "the Cannon cannot reach the King (5500+600+1400 < 8000)"


def test_knockback_into_river_is_clamped_to_land(pr):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    K.spawn1(g, 0, H.CANNON, 9000, 13000, deployed=True)          # static target for the Knight
    kn = K.spawn1(g, 1, H.KNIGHT, 9000, 14700, deployed=True)     # 300 from the water
    g.tick(3)
    assert K.pos(K.ent(g, kn)) == (9000, 14700)
    hk = K.hp_of(g, kn)
    assert K.cast(g, 0, H.FIREBALL, 9000, 13700) == H.OK         # pushes the Knight +y by 1000
    K.run_until(g, lambda gg: hk - K.hp_of(gg, kn) >= 600, 40, "Fireball impact")
    for _ in range(25):
        e = K.ent(g, kn)
        assert e is not None
        assert not H.point_in_water(*K.pos(e)), f"knocked-back Knight left in water at {K.pos(e)}"
        g.tick(1)


def test_princess_targets_air(pr):
    g = K.new_game()
    bd = K.spawn1(g, 0, H.BABY_DRAGON, T1_LEFT[0], T1_LEFT[1] + 6000, deployed=True)
    K.run_until(g, lambda gg: int(K.tower_entity(gg, 1, 1)["target_id"]) == bd, 3, "tower acquiring a flyer")
    K.run_until(g, lambda gg: K.hp_of(gg, bd) < 1152, 40, "arrow hitting the Baby Dragon")
    assert 1152 - K.hp_of(g, bd) == 109


@pytest.mark.parametrize("d,targeted", [(8950, True), (9000, True), (9050, False)])
def test_princess_attack_range_boundary(pr, d, targeted):
    """Towers target within range + r_tower + r_target = 7500 + 1000 + 500 = 9000. A deploying
    Knight is a static, valid target."""
    g = K.new_game()
    kn = K.spawn1(g, 0, H.KNIGHT, T1_LEFT[0], T1_LEFT[1] + d, deployed=False)
    g.tick(1)
    tid = int(K.tower_entity(g, 1, 1)["target_id"])
    assert (tid == kn) == targeted, f"d={d}: princess target {tid}"
