"""SPEC §16.1 / §16.2 mechanics of the new cards: behavioural scenarios with values derived from
data/source/cards-15.535.json (level 11 = floor(base*256/100)).

Scene design rule: every measured victim is damaged ONLY by the mechanic under test (passive
Elixir Collectors or deploying Golems as targets, scenes placed outside the reach of crown
towers that could interfere). Where §16 leaves a timing open the tests pin what the SPEC states
(counts, spacing, values) and use a small documented window for the alignment."""
import math

import numpy as np
import pytest

import cardsv3 as C
import gamekit as K
import helpers as H


def game(d0="hog26", d1="giant", **kw):
    kw.setdefault("deploy_lockout_ticks", 0)
    return K.new_game(d0, d1, **kw)


def unit_ents(g, team, unit):
    return [e for e in K.ents(g) if int(e["team"]) == team and e["unit"] == unit]


def card_ents(g, team, cid):
    return [e for e in K.ents(g) if int(e["team"]) == team and int(e["card_id"]) == cid]


def track_births(g, team, unit, n_ticks, on_tick=None):
    """Tick n times; return [(tick_index, [new entities])] for new ids of `unit` (1-based ticks)."""
    seen = {int(e["id"]) for e in unit_ents(g, team, unit)}
    out = []
    for n in range(1, n_ticks + 1):
        if on_tick:
            on_tick(g, n)
        g.tick(1)
        new = [e for e in unit_ents(g, team, unit) if int(e["id"]) not in seen]
        if new:
            out.append((n, new))
            seen |= {int(e["id"]) for e in new}
    return out


def watch(g, ids, n, towers=(), stop=None):
    """Tick n times. Returns {key: [(tick, drop)]} of hp decreases for entity ids and for crown
    towers given as (team, idx) keys; a key whose entity vanished gets (tick, None) once."""
    prev = {i: K.hp_of(g, i) for i in ids}
    prev.update({t: K.tower_hp(g, *t) for t in towers})
    out = {k: [] for k in prev}
    for k in range(1, n + 1):
        g.tick(1)
        cur = {i: K.hp_of(g, i) for i in ids}
        cur.update({t: K.tower_hp(g, *t) for t in towers})
        for key, v in cur.items():
            p = prev[key]
            if p is None:
                continue
            if v is None:
                out[key].append((k, None))
            elif v < p:
                out[key].append((k, p - v))
        prev = cur
        if stop and stop(g, k, out):
            break
    return out


def ring(c, r, n, a0=0.0, a1=2 * math.pi, closed=False):
    """n points on a circle arc around c from angle a0 to a1 (engine frame, y down = 'north' up)."""
    pts = []
    for i in range(n):
        a = a0 + (a1 - a0) * i / (n - (1 if closed else 0))
        pts.append((int(round(c[0] + r * math.cos(a))), int(round(c[1] - r * math.sin(a)))))
    return pts


def first_gone(g, eid, max_ticks, what):
    return K.run_until(g, lambda gg: K.ent(gg, eid) is None, max_ticks, what)


# ==========================================================================================
# Periodic spawners (§16.2)
# ==========================================================================================
WITCH_AT = (2000, 30000)          # far corner of team 0's half: ~20 s before any tower can reach it


def test_tester_spawner_data():
    w = C.unit("Witch")["spawner"]
    assert (w["character"], w["number"], w["start_time_ms"], w["pause_time_ms"]) == ("Skeleton", 4, 1000, 7000)
    t = C.unit("Tombstone")["spawner"]
    assert (t["character"], t["number"], t["interval_ms"], t["pause_time_ms"]) == ("Skeleton", 2, 500, 3500)
    nw = C.unit("DarkWitch")
    assert nw["spawner"]["number"] == 2 and nw["spawner"]["pause_time_ms"] == 5000
    assert nw["death_spawn"]["character"] == "Bat" and nw["death_spawn"]["count"] == 1
    assert C.unit("Tombstone")["death_spawn"]["count"] == 4


def test_witch_waves(pr):
    """§16.6.5: first wave start_time/50 = 20 ticks after deploy completes (elapsed k = 19, k = 0 =
    first non-deploying tick = observation 1 for a deployed spawn), 4 Skeletons, then every 7000 ms
    (140 ticks) start-to-start; born deployed with hp 81, card_id = Witch (§16.6.6)."""
    g = game()
    K.spawn1(g, 0, C.WITCH, *WITCH_AT, deployed=True)
    births = track_births(g, 0, "Skeleton", 320)
    ticks = [t for t, _ in births]
    assert len(births) >= 3, f"expected 3 waves in 320 ticks, got births at {ticks}"
    assert [len(es) for _, es in births[:3]] == [4, 4, 4], f"wave sizes {[len(es) for _, es in births]}"
    assert ticks[:3] == [20, 160, 300], f"wave observations {ticks[:3]}, want [20, 160, 300]"
    for _, es in births[:3]:
        for e in es:
            assert int(e["max_hp"]) == 81 and not bool(e["deploying"])
            assert int(e["card_id"]) == C.WITCH, "spawned units map to the card that created them"


def test_witch_spawner_pauses_while_stunned(pr):
    base = game()
    K.spawn1(base, 0, C.WITCH, *WITCH_AT, deployed=True)
    t0 = [t for t, _ in track_births(base, 0, "Skeleton", 170)]
    g = game()
    w = K.spawn1(g, 0, C.WITCH, *WITCH_AT, deployed=True)

    def zap(gg, n):
        if n == t0[0] + 50:
            assert K.cast(gg, 1, H.ZAP, *K.pos(K.ent(gg, w))) == H.OK
    t1 = [t for t, _ in track_births(g, 0, "Skeleton", 180, zap)]
    assert t1[0] == t0[0] and t1[1] == t0[1] + 10, \
        f"a 10-tick Zap stun must delay the next wave by exactly 10 ticks: {t0[:2]} vs {t1[:2]}"


def test_tombstone_waves_and_death_spawn(pr):
    """start_time null -> first wave pause_time/50 = 70 ticks after deploy (observation 70), 2 units
    500 ms apart, waves every 3500 ms; destroyed by one Fireball -> exactly 4 Skeletons committed
    in the next Spawn phase (death observation + 1)."""
    g = game(d0="hog26", d1="hog26")
    at = (9000, 21000)
    tomb = K.spawn1(g, 0, C.TOMBSTONE, *at, deployed=True)
    seen = set()
    births, death = [], None
    for n in range(1, 400):
        if n == 225:                                        # lands ~30 obs later, mid-gap
            assert K.cast(g, 1, H.FIREBALL, *at) == H.OK
        g.tick(1)
        cur = unit_ents(g, 0, "Skeleton")
        new = [e for e in cur if int(e["id"]) not in seen]
        seen |= {int(e["id"]) for e in cur}
        if new:
            births.append((n, new))
        if death is None and K.ent(g, tomb) is None:
            death = n
        if death is not None and n >= death + 3:
            break
    assert death is not None, "setup: the Fireball (688) must destroy the Tombstone (hp 529)"
    periodic = [(t, len(es)) for t, es in births if t <= death]
    assert periodic[:6] == [(t, 1) for t in (70, 80, 140, 150, 210, 220)], f"periodic births {periodic}"
    assert [(t, len(es)) for t, es in births if t > death] == [(death + 1, 4)], \
        f"death spawn: births after the death {[(t, len(es)) for t, es in births if t > death]}"
    for t, es in births:
        if t == death + 1:
            for e in es:
                assert int(e["max_hp"]) == 81 and int(e["card_id"]) == C.TOMBSTONE
                assert math.dist(K.pos(e), at) <= 2000


def test_tombstone_death_spawn_on_expiry(pr):
    """§16.6.7: death spawns fire on any death, lifetime expiry included (30 s: gone at observation
    600 for a deployed spawn, §13.4); the 4 Skeletons appear at observation 601."""
    g = game()
    tomb = K.spawn1(g, 0, C.TOMBSTONE, 9000, 21000, deployed=True)
    seen, births, death = set(), [], None
    for n in range(1, 640):
        g.tick(1)
        cur = unit_ents(g, 0, "Skeleton")
        new = [e for e in cur if int(e["id"]) not in seen]
        seen |= {int(e["id"]) for e in cur}
        if new:
            births.append((n, len(new)))
        if death is None and K.ent(g, tomb) is None:
            death = n
        if death is not None and n >= death + 2:
            break
    assert death == 600, f"Tombstone lifetime: gone at observation {death}, want 600"
    assert (death, 4) not in births and (death + 1, 4) in births, f"births around expiry {births[-4:]}"


def test_night_witch_death_spawns_one_bat(pr):
    """Killed while still deploying (so no periodic wave has started): exactly one Bat appears, in
    the Spawn phase after the death (death observation + 1)."""
    g = game()
    P = (9000, 11000)
    nw = K.spawn1(g, 0, C.NIGHT_WITCH, *P, deployed=False)
    for x, y in ring(P, 3000, 8):
        K.spawn1(g, 1, H.MUSKETEER, x, y, deployed=True)
    births, death, seen = [], None, set()
    for n in range(1, 60):
        g.tick(1)
        if death is None and K.ent(g, nw) is None:
            death = n
        for e in unit_ents(g, 0, "Bat"):
            if int(e["id"]) not in seen:
                seen.add(int(e["id"]))
                births.append((n, K.pos(e), int(e["card_id"])))
        if death is not None and n > death + 5:
            break
    assert death is not None and death <= 20, f"setup: 8 Musketeers kill the deploying Night Witch ({death})"
    assert len(births) == 1, f"exactly one death-spawn Bat expected, got {births}"
    t, p, cid = births[0]
    assert t == death + 1 and math.dist(p, P) <= 500 + 300 and cid == C.NIGHT_WITCH, births


# ==========================================================================================
# Death damage / death spawns
# ==========================================================================================
G = (9000, 14000)            # centre of team 1's half, outside every crown tower's reach


def test_golem_death_damage_and_golemites(pr):
    """12 Musketeers kill a deploying Golem where it stands; at death observation + 1 its death
    damage L11(88) = 225 hits the enemy Elixir Collector 1838 away (<= 2000 + 1000), not the
    Musketeers 5000 away, and exactly 2 Golemites (hp L11(406) = 1039) are committed."""
    g = game()
    golem = K.spawn1(g, 0, C.GOLEM, *G, deployed=False)
    ec = K.spawn1(g, 1, C.ELIXIR_COLLECTOR, 10300, 12700, deployed=True)
    musk = [K.spawn1(g, 1, H.MUSKETEER, x, y, deployed=True)
            for x, y in ring(G, 5000, 12, 0, math.pi, closed=True)]
    hp_m = {i: K.hp_of(g, i) for i in musk}
    death = None
    for n in range(1, 60):
        g.tick(1)
        if K.ent(g, golem) is None:
            death = n
            break
    assert death is not None, "setup: the Musketeers must kill the Golem while it deploys (60 ticks)"
    assert unit_ents(g, 0, "Golemite") == [], "death spawns are committed in the NEXT Spawn phase"
    pre = K.hp_of(g, ec)
    g.tick(1)
    lost = pre - K.hp_of(g, ec)
    assert lost in (225, 226), f"Elixir Collector lost {lost} at death + 1; death damage is 225 (+ <=1 drain)"
    kids = unit_ents(g, 0, "Golemite")
    assert len(kids) == 2, f"Golem death spawns exactly 2 Golemites, got {len(kids)}"
    for k in kids:
        assert int(k["max_hp"]) == 1039 and int(k["card_id"]) == C.GOLEM
        assert math.dist(K.pos(k), G) <= 1500 + 1000
    g.tick(2)
    for i in musk:
        assert K.hp_of(g, i) == hp_m[i], "Musketeers 5000 away are outside 2000 + r_target"


def test_lava_hound_death_spawns_six_pups(pr):
    g = game()
    lh = K.spawn1(g, 0, C.LAVA_HOUND, *G, deployed=False)
    for x, y in ring(G, 5000, 12, 0, math.pi, closed=True):
        K.spawn1(g, 1, H.MUSKETEER, x, y, deployed=True)
    last, death = K.pos(K.ent(g, lh)), None
    seen = set()
    for n in range(1, 200):
        g.tick(1)
        e = K.ent(g, lh)
        if e is not None:
            last = K.pos(e)
            continue
        if death is None:
            death = n
        pups = [p for p in unit_ents(g, 0, "LavaPups") if int(p["id"]) not in seen]
        if pups:
            assert n == death + 1, "death spawns are committed in the next Spawn phase"
            assert len(pups) == 6, f"exactly 6 LavaPups, got {len(pups)}"
            for p in pups:
                assert bool(p["flying"]) and int(p["max_hp"]) == 215 and int(p["card_id"]) == C.LAVA_HOUND
                assert math.dist(K.pos(p), last) <= 2500 + 500
            return
        if n > death + 2:
            break
    raise AssertionError(f"no LavaPups after the Hound died (death at {death})")


def test_battle_ram_charged_kamikaze_and_barbarians(pr):
    """Walks 4750 (> charge 3000) -> charged hit L11(224) = 573 on the Princess (crown % 100); it is
    gone in the same observation (kamikaze); 2 Barbarians (hp 716) at radius 600 appear at the next
    observation and deploy for exactly 20 observations (death_spawn deploy_time 1000 ms)."""
    g = game()
    ram = K.spawn1(g, 0, C.BATTLE_RAM, 3500, 13500, deployed=True)
    hp = H.PRINCESS_HP
    last = K.pos(K.ent(g, ram))
    for n in range(1, 200):
        g.tick(1)
        cur = K.tower_hp(g, 1, 1)
        e = K.ent(g, ram)
        if cur < hp:
            assert hp - cur == 573, f"charged Battle Ram hit {hp - cur}, want 573"
            break
        last = K.pos(e)
    else:
        raise AssertionError("the Battle Ram never hit the princess")
    assert K.ent(g, ram) is None, "kamikaze: the Ram dies in the tick its hit lands"
    assert unit_ents(g, 0, "Barbarian") == [], "death spawns are committed in the NEXT Spawn phase"
    g.tick(1)
    barbs = unit_ents(g, 0, "Barbarian")
    assert len(barbs) == 2 and all(int(b["max_hp"]) == 716 for b in barbs), barbs
    assert all(math.dist(K.pos(b), last) <= 600 + 500 for b in barbs)
    assert all(int(b["card_id"]) == C.BATTLE_RAM for b in barbs)
    ids = [int(b["id"]) for b in barbs]
    n_dep = 0
    while all(K.ent(g, i) is not None and bool(K.ent(g, i)["deploying"]) for i in ids):
        n_dep += 1
        g.tick(1)
        assert n_dep < 40
    assert n_dep == 20, f"death-spawned Barbarians deploy 1000 ms: {n_dep} deploying observations"


# ==========================================================================================
# Death bombs (timed area hit after the bomb's deploy_time 3000 ms, radius 3000 + r_target)
# ==========================================================================================
def _bomb_window(g, ids, towers, death, n=80):
    out = watch(g, ids, n, towers)
    return {k: [(death + t, d) for t, d in v] for k, v in out.items()}


def test_giant_skeleton_bomb(pr):
    """GS (team 0) fights a PEKKA and dies at G; GiantSkeletonBomb L11(269) = 688 is observed at
    exactly death + 3000/50 + 1 (§16.6.8) on the enemy Elixir Collector 2500 away (<= 3000 + 1000)
    and never on the one 4700 away."""
    g = game()
    gs = K.spawn1(g, 0, C.GIANT_SKELETON, *G, deployed=True)
    K.spawn1(g, 1, C.PEKKA, G[0], G[1] - 1600, deployed=True)
    ec_in = K.spawn1(g, 1, C.ELIXIR_COLLECTOR, G[0] + 2500, G[1], deployed=True)
    ec_out = K.spawn1(g, 1, C.ELIXIR_COLLECTOR, G[0] - 4700, G[1], deployed=True)
    for x, y in ring(G, 4500, 5, math.radians(50), math.radians(130), closed=True):
        K.spawn1(g, 1, H.CANNON, x, y, deployed=True)
    last, death = G, None
    for n in range(1, 300):
        g.tick(1)
        e = K.ent(g, gs)
        if e is None:
            death = n
            break
        last = K.pos(e)
    assert death is not None, "setup: the Giant Skeleton must die"
    assert math.dist(last, G) <= 300, "setup: the GS fought in place"
    w = _bomb_window(g, [ec_in, ec_out], (), death)
    big_in = [(t, d) for t, d in w[ec_in] if d is not None and d >= 100]
    assert len(big_in) == 1 and big_in[0][1] in (688, 689), f"bomb hits on the inside EC: {big_in}"
    assert big_in[0][0] == death + 61, f"bomb observed {big_in[0][0] - death} ticks after death, want 61"
    assert all(d is not None and d < 100 for _, d in w[ec_out]), "outside 3000 + r_target"


def test_bomb_tower_bomb_hits_enemies_never_friends(pr):
    """Team-0 Bomb Tower 3500 below the enemy left Princess is destroyed by two Fireballs; its bomb
    (L11(87) = 222, crown % 100) hits the Princess once at death + 61, never the team-0 Elixir
    Collector 3750 away (inside 3000 + 1000) and not the enemy one 4500 away."""
    g = game(d0="hog26", d1="hog26")
    B = (3500, 10000)
    bt = K.spawn1(g, 0, C.BOMB_TOWER, *B, deployed=True)
    friend = K.spawn1(g, 0, C.ELIXIR_COLLECTOR, 3500, 13750, deployed=True)
    out = K.spawn1(g, 1, C.ELIXIR_COLLECTOR, 8000, 10000, deployed=True)
    assert K.cast(g, 1, H.FIREBALL, *B) == H.OK
    g.tick(1)
    assert K.cast(g, 1, H.FIREBALL, *B) == H.OK
    death = 1 + first_gone(g, bt, 80, "Bomb Tower destroyed by two Fireballs")
    w = _bomb_window(g, [friend, out], [(1, 1)], death)
    late = [(t, d) for t, d in w[(1, 1)] if t >= death + 30]
    assert late == [(death + 61, 222)], f"late princess damage {late}: exactly the 222 bomb at death + 61"
    assert all(d is not None and d < 200 for t, d in w[friend] if t >= death + 30), \
        f"friendly Elixir Collector hit by its own bomb: {w[friend]}"
    assert all(d is not None and d < 100 for t, d in w[out] if t >= death + 30), "outside 3000 + r_target"


def test_bomb_tower_bomb_on_expiry(pr):
    """§16.6.7: the Bomb Tower's bomb also fires when its 30 s lifetime expires (gone at
    observation 600); a deploying enemy Golem 2700 away takes exactly 222 at 661."""
    g = game()
    bt = K.spawn1(g, 0, C.BOMB_TOWER, 9000, 17500, deployed=True)
    assert not H.point_in_water(9000, 14800)
    victim, death = None, None
    for n in range(1, 700):
        if n == 590:
            victim = K.spawn1(g, 1, C.GOLEM, 9000, 14800, deployed=False)
        g.tick(1)
        if K.ent(g, bt) is None:
            death = n
            break
    assert death == 600, f"Bomb Tower lifetime: gone at observation {death}, want 600"
    w = _bomb_window(g, [victim], (), death, 70)
    late = [(t, d) for t, d in w[victim] if t >= death + 30]
    assert late == [(death + 61, 222)], f"expiry bomb on the Golem: {late}"


def test_balloon_death_bomb(pr):
    """Balloon killed while deploying; BalloonBomb L11(94) = 240 hits the enemy Elixir Collector
    1500 away once, at death + 61."""
    g = game()
    P = (9000, 11000)
    ball = K.spawn1(g, 0, C.BALLOON, *P, deployed=False)
    ec = K.spawn1(g, 1, C.ELIXIR_COLLECTOR, P[0] + 1500, P[1], deployed=True)
    for x, y in ring(P, 3500, 12):
        K.spawn1(g, 1, H.MUSKETEER, x, y, deployed=True)
    death = first_gone(g, ball, 80, "Balloon death")
    w = _bomb_window(g, [ec], (), death)
    big = [(t, d) for t, d in w[ec] if d is not None and d >= 100 and t >= death + 30]
    assert len(big) == 1 and big[0][1] in (240, 241) and big[0][0] == death + 61, f"late bomb hits {big}"


# ==========================================================================================
# Variable damage (Inferno Tower / Dragon)
# ==========================================================================================
INFERNO = {C.INFERNO_TOWER: (43, 158, 847), C.INFERNO_DRAGON: (35, 120, 422)}
ATT_AT, VICTIM_AT = (9000, 18000), (9000, 21600)     # both outside team 0's Princess reach


def test_tester_inferno_stage_values():
    for cid, key in ((C.INFERNO_TOWER, "InfernoTower"), (C.INFERNO_DRAGON, "InfernoDragon")):
        raw = C.unit(key)["raw"]
        assert (H.L11(raw["Damage"]), H.L11(raw["VariableDamage2"]), H.L11(raw["VariableDamage3"])) == INFERNO[cid]
        assert raw["VariableDamageTime1"] == raw["VariableDamageTime2"] == 2000


def _inferno_scene(cid):
    g = game(d0="giant", d1="hog26")
    att = K.spawn1(g, 1, cid, *ATT_AT, deployed=True)
    golem = K.spawn1(g, 0, C.GOLEM, *VICTIM_AT, deployed=False)
    return g, att, golem


@pytest.mark.parametrize("cid", [C.INFERNO_TOWER, C.INFERNO_DRAGON])
def test_inferno_damage_ramps_in_three_stages(pr, cid):
    """§16.6.9: the stage clock starts at the first hit on the target; 5 hits per stage."""
    s1, s2, s3 = INFERNO[cid]
    g, att, golem = _inferno_scene(cid)
    hits = watch(g, [golem], 100)[golem]
    vals = [d for _, d in hits]
    assert len(vals) >= 12 and vals == [s1] * 5 + [s2] * 5 + [s3] * (len(vals) - 10), \
        f"{C.ALL_NAMES[cid]} hits {vals}: want 5 x {s1}, 5 x {s2}, then {s3}"
    gaps = {b[0] - a[0] for a, b in zip(hits, hits[1:])}
    assert gaps == {8}, f"hit every 400 ms (8 ticks): gaps {gaps}"


@pytest.mark.parametrize("cid", [C.INFERNO_TOWER, C.INFERNO_DRAGON])
def test_inferno_resets_to_stage_one_after_zap(pr, cid):
    s1, s2, s3 = INFERNO[cid]
    g, att, golem = _inferno_scene(cid)
    seq = []
    zapped = False
    for n in range(1, 200):
        pre = K.hp_of(g, golem)
        g.tick(1)
        d = pre - K.hp_of(g, golem)
        if d > 0:
            seq.append(d)
            if zapped:
                break
            if d == s3:
                assert K.cast(g, 0, H.ZAP, *K.pos(K.ent(g, att))) == H.OK
                zapped = True
    assert zapped, f"setup: stage 3 never reached ({seq})"
    assert seq[-1] == s1, f"after a Zap stun the next hit must be stage 1 ({s1}): {seq}"


def test_inferno_tower_resets_on_retarget(pr):
    """IT kills a Knight at stage 3, then its first hit on the next target (a Golem) is stage 1."""
    g = game()
    it = K.spawn1(g, 1, C.INFERNO_TOWER, *ATT_AT, deployed=True)
    kn = K.spawn1(g, 0, H.KNIGHT, ATT_AT[0], ATT_AT[1] + 2000, deployed=False)
    golem = K.spawn1(g, 0, C.GOLEM, ATT_AT[0], ATT_AT[1] + 4800, deployed=False)
    kn_dead = None
    golem_hits = []
    for n in range(1, 200):
        pre = K.hp_of(g, golem)
        g.tick(1)
        if kn_dead is None and K.ent(g, kn) is None:
            kn_dead = n
        d = pre - K.hp_of(g, golem)
        if d > 0:
            golem_hits.append((n, d))
            break
    assert kn_dead is not None and K.ent(g, it) is not None, "setup: the IT must kill the Knight first"
    assert golem_hits and golem_hits[0][0] > kn_dead and golem_hits[0][1] == 43, \
        f"first hit on the new target must be stage 1 (43): {golem_hits} (knight died {kn_dead})"


# ==========================================================================================
# Mortar minimum range
# ==========================================================================================
def test_mortar_minimum_range(pr):
    """Mortar (team 1) with a team-0 Elixir Collector 3000 away (< 3500 + 600 + 1000: invalid)
    and team 0's left Princess 7906 away (valid): it fires at the Princess (266, crown % 100) and
    never at the nearer Collector."""
    g = game()
    K.spawn1(g, 1, C.MORTAR, 6000, 18000, deployed=True)
    near = K.spawn1(g, 0, C.ELIXIR_COLLECTOR, 6000, 21000, deployed=True)
    w = watch(g, [near], 120, [(0, 1)])
    assert all(d is not None and d <= 1 for _, d in w[near]), f"minimum range violated: {w[near]}"
    assert [d for _, d in w[(0, 1)]] == [266], f"Princess damage {w[(0, 1)]}: one Mortar shell of 266"


def test_mortar_shell_lands_where_the_target_was(pr):
    """§16.6.2: non-homing shells fly to the target's position AT LAUNCH. A Hog Rider running at
    120/tick is ~3400 away from that point when the shell lands (> 2000 + 600 splash reach), so the
    shot misses (a homing shell would hit for 266)."""
    g = game()
    K.spawn1(g, 1, C.MORTAR, 9000, 12000, deployed=True)
    hog = K.spawn1(g, 0, H.HOG, 9000, 24000, deployed=True)
    hp0 = K.hp_of(g, hog)
    fired = False
    for n in range(1, 56):
        g.tick(1)
        fired = fired or len(g.projectiles()) > 0              # the Mortar is the only shooter here
        assert K.hp_of(g, hog) == hp0, f"the Mortar shell hit the moving Hog Rider (observation {n})"
    assert fired, "setup: the Mortar must have fired at the Hog Rider"


# ==========================================================================================
# Bandit dash
# ==========================================================================================
def _bandit_scene(dist):
    g = game()
    golem = K.spawn1(g, 1, C.GOLEM, 9000, 18000, deployed=False)
    K.spawn1(g, 1, C.XBOW, 11000, 18000, deployed=True)          # shoots the Bandit meanwhile
    b = K.spawn1(g, 0, C.BANDIT, 9000, 18000 + dist, deployed=True)
    steps, first = [], None
    prev_p, prev_hp, prev_g = K.pos(K.ent(g, b)), K.hp_of(g, b), K.hp_of(g, golem)
    for n in range(1, 160):
        g.tick(1)
        e = K.ent(g, b)
        assert e is not None, "setup: the Bandit must survive until its first hit"
        p, hp, gh = K.pos(e), int(e["hp"]), K.hp_of(g, golem)
        steps.append((math.dist(p, prev_p), prev_hp - hp))
        if gh < prev_g:
            first = prev_g - gh
            break
        prev_p, prev_hp, prev_g = p, hp, gh
    return steps, first


@pytest.mark.parametrize("dist", [5000, 6600])
def test_bandit_dashes_with_dash_damage_and_immunity(pr, dist):
    """§16.6.14: dash window 3500 + 600 + 750 = 4850 <= dist <= 6000 + 600 + 750 = 7350 (6600 is
    outside a plain 3500..6000 centre-distance reading): a dash step at speed 500 (>> walking 90),
    first hit = dash damage L11(152) = 389, no damage taken on dash ticks."""
    steps, first = _bandit_scene(dist)
    assert max(s for s, _ in steps) >= 300, f"no dash step: {[round(s) for s, _ in steps]}"
    assert first == 389, f"first Bandit hit {first}, want the dash damage 389"
    assert all(dmg <= 0 for s, dmg in steps if s >= 300), "invulnerable while dashing"


@pytest.mark.parametrize("dist", [2400, 4500])
def test_bandit_does_not_dash_inside_minimum_range(pr, dist):
    """4500 < 4850 (and 2400): no dash, plain first hit L11(76) = 194."""
    steps, first = _bandit_scene(dist)
    assert max(s for s, _ in steps) <= 95, f"dash inside the minimum range: {[round(s) for s, _ in steps]}"
    assert first == 194, f"plain hit L11(76) = 194, got {first}"


# ==========================================================================================
# Miner
# ==========================================================================================
MINER_TILES = [((9, 8), True), ((13, 12), True), ((9, 21), True), ((3, 8), True),
               ((0, 14), True), ((2, 0), True), ((17, 17), True), ((3, 31), True),       # no-deploy cells
               ((3, 6), False), ((14, 5), False), ((9, 2), False),                      # enemy footprints
               ((3, 25), False), ((14, 26), False), ((9, 28), False),                   # own footprints
               ((9, 15), False), ((6, 16), False)]                                      # water


@pytest.mark.parametrize("tile,ok", MINER_TILES)
def test_miner_placement(pr, tile, ok):
    """§16.6.3: any non-water tile outside every living crown-tower footprint (either team);
    no-deploy cells allowed."""
    g = game(d0="miner_poison")
    K.put_first(g, 0, C.MINER)
    g.set_elixir(0, H.MAX_ELIXIR)
    m = np.asarray(g.legal_mask(0))
    assert bool(m[H.action_id(0, *tile)]) == ok, f"mask for Miner at own tile {tile}"
    code = g.play_tile(0, 0, *tile)
    assert code == (H.OK if ok else H.ILLEGAL_POSITION), f"Miner at {tile}: code {code}"


def test_miner_may_land_on_building_footprints_and_destroyed_towers(pr):
    g = game(d0="miner_poison", d1="miner_poison")
    K.spawn1(g, 1, C.INFERNO_TOWER, 9500, 10500, deployed=True)              # enemy building footprint
    K.spawn1(g, 0, C.XBOW, 9500, 22500, deployed=True)                       # own building footprint
    g.set_tower_hp(1, 1, 1)
    assert K.cast(g, 0, C.POISON, *H.TOWER_POS[1][1]) == H.OK
    K.run_until(g, lambda gg: not K.tower_alive(gg, 1, 1), 60, "enemy left Princess destroyed")
    for tile in ((9, 10), (9, 22), (3, 6)):
        K.put_first(g, 0, C.MINER)
        g.set_elixir(0, H.MAX_ELIXIR)
        assert np.asarray(g.legal_mask(0))[H.action_id(0, *tile)] == 1, f"Miner at {tile}"


def test_tester_miner_oracle_agrees():
    for tile, ok in MINER_TILES:
        assert C.miner_tile(0, *tile, H.all_alive()) is ok, tile


def _miner_trace(g, tap, n_ticks, zap_at=None):
    """Cast the Miner at `tap`; per observation record the single team-0 Miner entity."""
    assert K.cast(g, 0, C.MINER, *tap) == H.OK
    out = []
    for n in range(1, n_ticks + 1):
        if zap_at is not None and n == zap_at:
            assert K.cast(g, 1, H.ZAP, *tap) == H.OK
        g.tick(1)
        ms = card_ents(g, 0, C.MINER)
        assert len(ms) == 1, f"observation {n}: the Miner must exist (hidden while burrowing), got {len(ms)}"
        out.append(ms[0])
    return out


MINER_TAP = (9500, 10500)
MINER_N = math.ceil(int(math.dist(MINER_TAP, H.TOWER_POS[0][0])) / 650)        # 18506 / 650 -> 29


def test_miner_travels_underground_and_emerges_at_tap(pr):
    """§16.6.4: hidden from the play tick for exactly ceil(d/650) observations, moving along the
    King -> tap segment; surfaces exactly at the tap, then deploys for exactly 20 observations."""
    tr = _miner_trace(game(d0="miner_poison"), MINER_TAP, MINER_N + 22)
    king = H.TOWER_POS[0][0]
    hidden = [bool(e["hidden"]) for e in tr]
    assert hidden[:MINER_N] == [True] * MINER_N and not hidden[MINER_N], \
        f"hidden pattern {hidden[:MINER_N + 2]} (want {MINER_N} hidden observations)"
    prev = math.dist(king, MINER_TAP) + 1
    for e in tr[:MINER_N]:
        p = K.pos(e)
        cross = abs((MINER_TAP[0] - king[0]) * (p[1] - king[1]) - (MINER_TAP[1] - king[1]) * (p[0] - king[0]))
        assert cross / math.dist(king, MINER_TAP) <= 20, f"burrowing Miner off the King->tap line at {p}"
        assert math.dist(p, MINER_TAP) < prev
        prev = math.dist(p, MINER_TAP)
    s = tr[MINER_N]
    assert abs(int(s["x"]) - MINER_TAP[0]) <= 1 and abs(int(s["y"]) - MINER_TAP[1]) <= 1, K.pos(s)
    assert int(s["hp"]) == int(s["max_hp"]) == 1210
    dep = [bool(e["deploying"]) for e in tr[MINER_N:]]
    assert dep[:20] == [True] * 20 and not dep[20], f"Miner deploy after surfacing: {dep}"


def test_miner_is_immune_while_underground(pr):
    """A Zap on the tap point while the Miner burrows 650 away does nothing; one applied while it
    deploys at the tap hits it (targetable while deploying)."""
    tr = _miner_trace(game(d0="miner_poison", d1="giant"), MINER_TAP, MINER_N + 3, zap_at=MINER_N - 1)
    assert all(int(e["hp"]) == 1210 for e in tr), "a Zap on the burrowing Miner must not damage it"
    tr = _miner_trace(game(d0="miner_poison", d1="giant"), MINER_TAP, MINER_N + 3, zap_at=MINER_N + 2)
    assert int(tr[-2]["hp"]) == 1210 - 192 and bool(tr[-2]["deploying"]), "Zap hits the deploying Miner"


def test_miner_crown_tower_damage_is_20_percent(pr):
    g = game(d0="miner_poison")
    assert K.cast_tile(g, 0, C.MINER, 3, 8) == H.OK          # (3500, 8500): next to the left Princess
    w = watch(g, [], 220, [(1, 1)])
    hits = [d for _, d in w[(1, 1)]]
    assert len(hits) >= 3 and set(hits) == {39}, f"Miner vs Princess: ceil(194 * 20 / 100) = 39, got {hits}"


# ==========================================================================================
# Elixir Collector
# ==========================================================================================
def test_elixir_collector_production_cap_and_death_bonus(pr):
    """§16.6.15: +2800 at elapsed tick 259 after deploy completes (deploying observations 1..20, so
    observation 280), then every 260 ticks; capped at 10 elixir; +2800 in the tick it is destroyed."""
    g = game(d0="hog26", d1="hog26")
    g.set_elixir(0, 0)
    ec = K.spawn1(g, 0, C.ELIXIR_COLLECTOR, 9000, 22000, deployed=False)
    extra = []
    for k in range(1, 561):
        e0 = K.elixir(g, 0)
        if e0 > 20000:
            g.set_elixir(0, 0)
            e0 = 0
        g.tick(1)
        d = K.elixir(g, 0) - e0
        if d != 50:
            extra.append((k, d))
    assert extra == [(280, 2850), (540, 2850)], f"per-tick elixir deltas != 50: {extra}"
    # cap: the third production (observation 800) overflows
    while K.tick_of(g) < 798:
        g.tick(1)
    g.set_elixir(0, 27000)
    g.tick(2)
    assert K.elixir(g, 0) == H.MAX_ELIXIR, "production is capped at 10 elixir"
    # death bonus
    g.set_elixir(0, 0)
    assert K.cast(g, 1, H.FIREBALL, 9000, 22000) == H.OK
    g.tick(1)
    assert K.cast(g, 1, H.FIREBALL, 9000, 22000) == H.OK
    extra, death = [], None
    for k in range(1, 60):
        e0 = K.elixir(g, 0)
        g.tick(1)
        if K.elixir(g, 0) - e0 != 50:
            extra.append((k, K.elixir(g, 0) - e0))
        if death is None and K.ent(g, ec) is None:
            death = k
    assert death is not None, "setup: two Fireballs destroy the Collector"
    assert extra == [(death, 2850)], f"death bonus +2800 once, in the tick it is destroyed: {extra} (death {death})"


def test_elixir_collector_no_bonus_on_expiry(pr):
    g = game()
    ec = K.spawn1(g, 0, C.ELIXIR_COLLECTOR, 9000, 22000, deployed=False)
    extra, gone = [], None
    for k in range(1, 1950):
        e0 = K.elixir(g, 0)
        if e0 > 20000:
            g.set_elixir(0, 0)
            e0 = 0
        g.tick(1)
        d = K.elixir(g, 0) - e0
        if d != 50:
            extra.append((k, d))
        if K.ent(g, ec) is None:
            gone = k
            break
    assert gone == 20 + 1860, f"lifetime 93 s after a 1 s deploy: gone at observation {gone}, want 1880"
    assert extra == [(280 + 260 * i, 2850) for i in range(7)], f"7 productions, no expiry bonus: {extra}"


# ==========================================================================================
# Spells
# ==========================================================================================
KING0 = H.TOWER_POS[0][0]
PRINCESS1 = H.TOWER_POS[1][1]


def _crown_drops(g, card, tap, n):
    assert K.cast(g, 0, card, *tap) == H.OK
    return watch(g, [], n, [(1, 1)])[(1, 1)]


def rocket_game():
    return C.game_with([C.ROCKET], "hog26", deploy_lockout_ticks=0)


def test_rocket_crown_damage_and_flight(pr):
    """Lands like Fireball (§16.6.12 / §13.12): observation ceil(d/350) after queueing."""
    g = rocket_game()
    drops = _crown_drops(g, C.ROCKET, PRINCESS1, 100)
    want = math.ceil(int(math.dist(PRINCESS1, KING0)) / 350)
    assert drops == [(want, 342)], f"Rocket vs Princess: ceil(1484 * 23 / 100) = 342 at {want}, got {drops}"
    assert K.tower_hp(g, 1, 0) == H.KING_HP


def test_rocket_radius(pr):
    g = rocket_game()
    tap = (9000, 11000)
    inside = K.spawn1(g, 1, H.CANNON, 9000 + 2500, 11000, deployed=True)     # 2500 <= 2000 + 600
    outside = K.spawn1(g, 1, H.CANNON, 9000 - 2700, 11000, deployed=True)    # 2700 > 2600
    hp_o = K.hp_of(g, outside)
    assert K.cast(g, 0, C.ROCKET, *tap) == H.OK
    K.run_until(g, lambda gg: K.ent(gg, inside) is None, 80, "Rocket (1484) destroys the inside Cannon (824)")
    assert hp_o - K.hp_of(g, outside) < 150, "outside 2000 + r_target (only lifetime drain)"


def _knockback_scene(card):
    """Team-1 Knight in melee range of a deploying team-0 Golem at (9000, 18000); impact 1000 to
    its left pushes it +x by 1800 -> (10800, 18000), still in range, so it stands there."""
    g = C.game_with([card], "hog26", deploy_lockout_ticks=0)
    K.spawn1(g, 0, C.GOLEM, 9000, 19300, deployed=False)
    kn = K.spawn1(g, 1, H.KNIGHT, 9000, 18000, deployed=True)
    g.tick(2)
    assert K.pos(K.ent(g, kn)) == (9000, 18000)
    hp0 = K.hp_of(g, kn)
    assert K.cast(g, 0, card, 8000, 18000) == H.OK
    n = K.run_until(g, lambda gg: K.hp_of(gg, kn) < hp0, 60, "spell impact")
    return g, kn, hp0, n


@pytest.mark.parametrize("card,dmg,speed", [(C.ROCKET, 1484, 350), (C.SNOWBALL, 179, 800)])
def test_projectile_spell_damage_and_knockback_1800(pr, card, dmg, speed):
    g, kn, hp0, n = _knockback_scene(card)
    assert n == math.ceil(int(math.dist((8000, 18000), KING0)) / speed), f"landed at observation {n}"
    assert hp0 - K.hp_of(g, kn) == dmg
    g.tick(6)
    x, y = K.pos(K.ent(g, kn))
    assert abs(x - 10800) <= 5 and abs(y - 18000) <= 5, f"{C.ALL_NAMES[card]}: Knight at {(x, y)}, want (10800, 18000)"


def test_snowball_slows_for_3_seconds_and_crown_damage(pr):
    g, kn, hp0, n = _knockback_scene(C.SNOWBALL)
    count = 0
    while bool(K.ent(g, kn)["slowed"]):
        count += 1
        g.tick(1)
        assert count <= 80
    assert count == 60, f"Snowball slow 3000 ms = 60 observations, got {count}"
    g2 = C.game_with([C.SNOWBALL], "hog26", deploy_lockout_ticks=0)
    want = math.ceil(int(math.dist(PRINCESS1, KING0)) / 800)
    assert _crown_drops(g2, C.SNOWBALL, PRINCESS1, 60) == [(want, 45)]


def test_freeze(pr):
    """Applied in tick P (§16.6.12): 148 damage at observation 1, frozen for exactly 80 observations
    (4000 ms), crown tower frozen and hit for 37, hidden Tesla frozen but not damaged (§16.6.21)."""
    g = C.game_with([C.FREEZE], "hog26", deploy_lockout_ticks=0)
    T = (3500, 9500)
    kn = K.spawn1(g, 1, H.KNIGHT, 3500, 10000, deployed=True)
    tesla = K.spawn1(g, 1, H.TESLA, 6500, 9500, deployed=True)
    g.tick(1)
    assert bool(K.ent(g, tesla)["hidden"])
    hp_k, hp_t = K.hp_of(g, kn), K.hp_of(g, tesla)
    assert K.cast(g, 0, C.FREEZE, *T) == H.OK
    g.tick(1)
    assert hp_k - K.hp_of(g, kn) == 148, "Freeze applies in tick P"
    assert H.PRINCESS_HP - K.tower_hp(g, 1, 1) == 37
    assert bool(K.tower_entity(g, 1, 1)["stunned"]), "Freeze affects crown towers"
    assert bool(K.ent(g, tesla)["stunned"]), "Freeze affects hidden buildings"
    assert hp_t - K.hp_of(g, tesla) <= 3, "a hidden Tesla takes no damage (lifetime drain only)"
    p = K.pos(K.ent(g, kn))
    count = 0
    while bool(K.ent(g, kn)["stunned"]):
        assert K.pos(K.ent(g, kn)) == p, "frozen unit moved"
        count += 1
        g.tick(1)
        assert count <= 100
    assert count == 80, f"frozen for {count} observations, want 80"


def _pulse_scene(deck, card, n=200):
    g = game(d0=deck)
    T = (9000, 11000)
    xb = K.spawn1(g, 1, C.XBOW, T[0], T[1] - 1500, deployed=True)
    golem = K.spawn1(g, 1, C.GOLEM, *T, deployed=False)      # walks < 3800 before the Poison ends
    mm = K.spawn1(g, 1, C.MEGA_MINION, T[0] - 1500, T[1], deployed=False)
    assert K.cast(g, 0, card, *T) == H.OK
    slowed = []

    def stop(gg, k, out):
        e = K.ent(gg, golem)
        slowed.append(bool(e["slowed"]) if e is not None else None)
        return False
    w = watch(g, [xb, golem, mm], n, stop=stop)
    return w, slowed, xb, golem, mm


def xbow_drain(n):
    """§13.4 lifetime drain of a deployed-spawned X-Bow (1600 HP, 30 s) at observation n (k = n-1)."""
    return (1600 * n * 50) // 30000 - (1600 * (n - 1) * 50) // 30000


def test_poison(pr):
    """§16.6.11: exactly 8 events of 92 at observations 1, 21, ..., 141 (tick P + 20 i); buildings
    alike; air hit; crown towers ceil(92 * 23 / 100) = 22; slowed while inside + 1 s."""
    w, slowed, xb, golem, mm = _pulse_scene("miner_poison", C.POISON)
    ticks = [1 + 20 * i for i in range(8)]
    assert w[golem] == [(t, 92) for t in ticks], f"Poison events on the Golem: {w[golem]}"
    xe = [(t, d) for t, d in w[xb] if d >= 50]
    assert xe == [(t, 92 + xbow_drain(t)) for t in ticks], f"Poison events on the X-Bow: {xe}"
    # the Mega Minion leaves the area and later flies into team 0's Princess reach (109 from ~161),
    # so only its first 150 observations are attributable to the Poison
    mm_ev = [(t, d) for t, d in w[mm] if t <= 150]
    assert mm_ev and mm_ev[0] == (1, 92) and all(d == 92 and t in ticks for t, d in mm_ev), \
        f"Poison hits air: {mm_ev}"
    assert slowed[0] and slowed[149], "slowed while inside the Poison"
    assert not any(slowed[184:]), "slow ends within 1 s of the 8 s area expiring"
    g = game(d0="miner_poison")
    assert _crown_drops(g, C.POISON, PRINCESS1, 200) == [(t, 22) for t in ticks]


def test_two_poisons_stack(pr):
    """enable_stacking: a second Poison one tick later adds its own 8 events of 92."""
    g = game(d0="miner_poison")
    T = (9000, 11000)
    golem = K.spawn1(g, 1, C.GOLEM, *T, deployed=False)
    assert K.cast(g, 0, C.POISON, *T) == H.OK
    w1 = watch(g, [golem], 1)[golem]
    assert K.cast(g, 0, C.POISON, *T) == H.OK
    w2 = watch(g, [golem], 160)[golem]
    got = [(1, d) for _, d in w1] + [(1 + t, d) for t, d in w2]
    want = sorted([(1 + 20 * i, 92) for i in range(8)] + [(2 + 20 * i, 92) for i in range(8)])
    assert got == want, f"two stacked Poisons: {got}"


def test_earthquake(pr):
    """§16.6.10: 3 events at observations 1, 21, 41: 81 to troops, floor(81*350/100) = 283 to
    non-crown buildings, ceil(81*60/100) = 49 to crown towers; ground only; slowed."""
    w, slowed, xb, golem, mm = _pulse_scene("royal_hogs", C.EARTHQUAKE, 70)
    ticks = [1, 21, 41]
    assert w[golem] == [(t, 81) for t in ticks], f"Earthquake events on the Golem: {w[golem]}"
    xe = [(t, d) for t, d in w[xb] if d >= 50]
    assert xe == [(t, 283 + xbow_drain(t)) for t in ticks], f"Earthquake vs building: {xe}"
    assert w[mm] == [], "Earthquake never hits air"
    assert slowed[0] and slowed[40], "Earthquake slows (-50%) while affected"
    g = game(d0="royal_hogs")
    assert _crown_drops(g, C.EARTHQUAKE, PRINCESS1, 70) == [(t, 49) for t in ticks]


def test_lightning_hits_three_highest_hp_targets(pr):
    """§16.6.13: the 3 highest current-HP enemies (Golem 5120 > PEKKA 3760 > Knight 1766 >
    Skeletons 81), bolt k in tick P + ceil(460k/50): observations 1, 11, 20; 1057 each + stun."""
    g = game(d0="golem")
    T = (9000, 11000)
    golem = K.spawn1(g, 1, C.GOLEM, *T, deployed=False)
    pekka = K.spawn1(g, 1, C.PEKKA, T[0] - 2000, T[1], deployed=False)
    kn = K.spawn1(g, 1, H.KNIGHT, T[0] + 2000, T[1], deployed=False)
    sk = K.spawn(g, 1, H.SKELETONS, T[0], T[1] + 1500, deployed=False)
    assert K.cast(g, 0, C.LIGHTNING, *T) == H.OK
    stunned = {}

    def stop(gg, k, out):
        for i in (golem, pekka, kn):
            if out[i] and out[i][-1][0] == k:
                stunned[i] = bool(K.ent(gg, i)["stunned"])
        return False
    w = watch(g, [golem, pekka, kn] + sk, 40, stop=stop)
    for i, t in ((golem, 1), (pekka, 11), (kn, 20)):
        assert w[i] == [(t, 1057)], f"Lightning bolt on target {i}: {w[i]}, want [({t}, 1057)]"
        assert stunned.get(i), "each bolt stuns for 500 ms"
    assert all(w[s] == [] for s in sk), "lower-HP enemies beyond the top 3 are not struck"
    g2 = game(d0="golem")
    assert _crown_drops(g2, C.LIGHTNING, (3500, 9000), 40) == [(1, 265)]


def test_barbarian_barrel_rolls_then_spawns_one_barbarian(pr):
    """§16.6.20: rolls 4500 from the tap (each ground enemy hit once, 232), then one Barbarian
    (card_id 58) at the end point, deploying for exactly 20 observations."""
    g = game(d0="royal_hogs")
    tap = H.tile_centre_engine(0, 9, 18)                       # (9500, 18500)
    end = (tap[0], tap[1] - 4500)
    near = K.spawn1(g, 1, H.KNIGHT, 9500, 17800, deployed=False)
    far = K.spawn1(g, 1, H.KNIGHT, 9500, 12000, deployed=False)
    air = K.spawn1(g, 1, C.MEGA_MINION, 9500, 16500, deployed=False)
    before = {int(e["id"]) for e in K.ents(g)}
    assert K.cast_tile(g, 0, C.BARB_BARREL, 9, 18) == H.OK
    born, dep = None, 0

    def stop(gg, k, out):
        nonlocal born, dep
        barbs = [e for e in K.ents(gg) if int(e["id"]) not in before and int(e["team"]) == 0]
        if barbs:
            assert len(barbs) == 1 and barbs[0]["unit"] == "Barbarian", barbs
            b = barbs[0]
            if born is None:
                born = k
                assert int(b["max_hp"]) == 716 and int(b["card_id"]) == C.BARB_BARREL
                assert math.dist(K.pos(b), end) <= 100, f"Barbarian at {K.pos(b)}, end of roll {end}"
            dep += int(bool(b["deploying"]))
        return False
    w = watch(g, [near, far, air], 70, stop=stop)
    # the roll is over by ~23 observations; after ~50 the Knights / Mega Minion walk or fly into
    # team 0's Princess reach, so only the first 40 observations are attributable to the barrel
    early = {k: [(t, d) for t, d in v if t <= 40] for k, v in w.items()}
    assert [d for _, d in early[near]] == [232], f"rolling barrel L11(91) = 232 once: {early[near]}"
    assert early[far] == [] and early[air] == [], "range 4500, ground only"
    assert born is not None and dep == 20, f"Barbarian born at {born}, deploying observations {dep}"


def test_princess_splash_projectile_168(pr):
    """§16.6.1: Princess fires PrincessProjectile, 66 -> 168, splash 2000, and the Deco volleys do
    no damage: two deploying Golems 1500 apart both lose exactly 168, one 3000 away nothing."""
    g = game()
    a = K.spawn1(g, 1, C.GOLEM, 9000, 11000, deployed=False)
    b = K.spawn1(g, 1, C.GOLEM, 10500, 11000, deployed=False)             # 1500 <= 2000 + 750
    c = K.spawn1(g, 1, C.GOLEM, 6000, 11000, deployed=False)              # 3000 > 2750
    K.spawn1(g, 0, C.PRINCESS, 9000, 14000, deployed=True)
    w = watch(g, [a, b, c], 58, stop=lambda gg, k, out: bool(out[a]))
    assert w[a] and w[a][0][1] == 168, f"Princess shot on its target: {w[a]}"
    assert w[b] == [(w[a][0][0], 168)], f"splash on the Golem 1500 away: {w[b]}"
    assert w[c] == []


# ==========================================================================================
# Kamikaze, jumpers, fliers, Royal Giant, compositions
# ==========================================================================================
def test_fire_spirit_kamikaze_splash(pr):
    g = game()
    golem = K.spawn1(g, 1, C.GOLEM, 9000, 11000, deployed=False)
    near = K.spawn1(g, 1, H.KNIGHT, 11000, 11000, deployed=False)       # 2000 <= 2300 + 500
    far = K.spawn1(g, 1, H.KNIGHT, 5500, 11000, deployed=False)         # 3500 > 2800
    fs = K.spawn1(g, 0, C.FIRE_SPIRIT, 9000, 14000, deployed=True)
    gone = None

    def stop(gg, k, out):
        nonlocal gone
        if gone is None and K.ent(gg, fs) is None:
            gone = k
        return False
    w = watch(g, [golem, near, far], 40, stop=stop)
    assert [d for _, d in w[golem]] == [207] and [d for _, d in w[near]] == [207], (w[golem], w[near])
    assert w[far] == []
    assert gone is not None and gone < w[golem][0][0], "kamikaze: the Fire Spirit dies when it fires"


def test_wall_breakers_ignore_troops_and_die_on_hit(pr):
    g = game()
    ec = K.spawn1(g, 1, C.ELIXIR_COLLECTOR, 9000, 11000, deployed=True)
    kn = K.spawn1(g, 1, H.KNIGHT, 6500, 13000, deployed=False)
    wbs = K.spawn(g, 0, C.WALL_BREAKERS, 9000, 13500, deployed=True)
    assert len(wbs) == 2
    w = watch(g, [ec, kn], 60)
    assert sum(d for _, d in w[ec] if d >= 50) == 2 * 281, f"two kamikaze hits of 281: {w[ec]}"
    assert all(K.ent(g, i) is None for i in wbs), "Wall Breakers die when their hit lands"
    assert w[kn] == [], "buildings-only: the Knight is ignored"


def test_royal_hogs_jump_the_river(pr):
    g = game()
    hogs = K.spawn(g, 0, C.ROYAL_HOGS, 9000, 19500, deployed=True)
    assert len(hogs) == 4
    wet = False
    for _ in range(150):
        g.tick(1)
        for i in hogs:
            e = K.ent(g, i)
            if e is not None and H.point_in_water(*K.pos(e)):
                wet = True
    assert wet, "Royal Hogs (JumpEnabled) cross the river away from the bridges"


@pytest.mark.parametrize("cid", [C.MEGA_MINION, C.BATS, C.MINION_HORDE, C.BALLOON, C.LAVA_HOUND,
                                 C.INFERNO_DRAGON, C.SKELETON_DRAGONS])
def test_air_units_fly_over_water(pr, cid):
    g = game()
    ids = K.spawn(g, 0, cid, 9000, 19500, deployed=True)
    assert all(bool(K.ent(g, i)["flying"]) for i in ids)
    for _ in range(200):
        g.tick(1)
        if any(K.ent(g, i) is not None and H.point_in_water(*K.pos(K.ent(g, i))) for i in ids):
            return
    raise AssertionError(f"{C.ALL_NAMES[cid]} never flew over the river")


def test_royal_giant_ranged_buildings_only(pr):
    g = game()
    rg = K.spawn1(g, 0, C.ROYAL_GIANT, 3500, 13000, deployed=True)      # 6500 <= 5000 + 750 + 1000
    kn = K.spawn1(g, 1, H.KNIGHT, 3500, 11500, deployed=False)         # nearer, but not a building
    p0 = K.pos(K.ent(g, rg))
    hp_k = K.hp_of(g, kn)
    hits = watch(g, [], 60, [(1, 1)])[(1, 1)]
    assert hits and hits[0][1] == 307, f"Royal Giant vs Princess {hits}"
    assert K.pos(K.ent(g, rg)) == p0, "in range from 6500: it never moves"
    assert K.hp_of(g, kn) == hp_k, "buildings-only: the nearer Knight is never hit"


@pytest.mark.parametrize("cid,deck", [(C.GOBLIN_GANG, "royal_hogs"), (C.RASCALS, None)])
def test_goblin_gang_and_rascals_played_composition(pr, cid, deck):
    g = game(d0=deck) if deck else C.game_with([cid], "hog26", deploy_lockout_ticks=0)
    before = {int(e["id"]) for e in K.ents(g)}
    assert K.cast_tile(g, 0, cid, 9, 21) == H.OK
    g.tick(1)
    new = [e for e in K.ents(g) if int(e["id"]) not in before]
    want = sorted(u for u, n in C.composition(cid) for _ in range(n))
    assert sorted(e["unit"] for e in new) == want, f"{C.ALL_NAMES[cid]} members {[e['unit'] for e in new]}"
    centre = H.tile_centre_engine(0, 9, 21)
    for e in new:
        assert int(e["card_id"]) == cid and int(e["max_hp"]) == C.unit_stats(e["unit"])["hp"]
        assert math.dist(K.pos(e), centre) <= 1600 + 1, "formation bound 1600 at spawn time"


def test_ice_wizard_hit_slows(pr):
    g = game()
    golem = K.spawn1(g, 1, C.GOLEM, 9000, 11000, deployed=False)
    K.spawn1(g, 0, C.ICE_WIZARD, 9000, 14000, deployed=True)
    K.run_until(g, lambda gg: K.hp_of(gg, golem) < 5120, 80, "Ice Wizard hit")
    assert bool(K.ent(g, golem)["slowed"]), "Ice Wizard projectile applies its slow target_buff"


# ------------------------------------------------------------------------------------------
# Generic behavioural smoke: every new troop/building card damages an enemy it can target
# ------------------------------------------------------------------------------------------
ATTACKERS = [cid for cid, _, _ in C.NEW_CARDS
             if C.kind(cid) in ("troop", "building") and cid not in (C.ELIXIR_COLLECTOR, C.TOMBSTONE)]


@pytest.mark.parametrize("cid", ATTACKERS)
def test_every_new_unit_card_deals_damage(pr, cid):
    g = game()
    tob = bool(C.src(cid).get("target_only_buildings"))
    ty = 8000 if cid == C.MORTAR else 11000
    if tob:
        tgt = K.spawn1(g, 1, C.ELIXIR_COLLECTOR, 9000, ty, deployed=True)
    else:
        tgt = K.spawn1(g, 1, C.GOLEM, 9000, ty, deployed=False)
    K.spawn(g, 0, cid, 9000, 14000, deployed=True)
    w = watch(g, [tgt], 200, stop=lambda gg, k, out: any(d is None or d >= 10 for _, d in out[tgt]))
    assert any(d is None or d >= 10 for _, d in w[tgt]), f"{C.ALL_NAMES[cid]} never damaged its target"
