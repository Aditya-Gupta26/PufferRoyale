"""SPEC §6.2 charge (Prince), §6.6 deaths (Ice Golem), kamikaze + freeze (Ice Spirit), §6.7
status effects, and charge resets (stun, knockback)."""
import math

import pytest

import gamekit as K
import helpers as H

T1_LEFT = H.TOWER_POS[1][1]


# ------------------------------------------------------------------------------------------
# Prince charge
# ------------------------------------------------------------------------------------------
def _walk_prince_to_tower(g, pid, limit=150):
    """Tick until the princess takes damage; return (steps, charged_seen, first_drop, tick_of_last_move)."""
    steps, charged = [], []
    p = K.pos(K.ent(g, pid))
    hp = H.PRINCESS_HP
    for n in range(1, limit + 1):
        g.tick(1)
        e = K.ent(g, pid)
        assert e is not None, "Prince died on the way"
        q = K.pos(e)
        steps.append(math.dist(p, q))
        charged.append(bool(e["charged"]))
        p = q
        cur = K.tower_hp(g, 1, 1)
        if cur < hp:
            return steps, charged, hp - cur, n
    raise AssertionError("Prince never hit the princess")


def test_prince_charge_speed_and_special_damage(pr):
    g = K.new_game()
    # 3000 of walking to reach attack range (1600 + 600 + 1000 = 3200 from the tower centre)
    pid = K.spawn1(g, 0, H.PRINCE, T1_LEFT[0], T1_LEFT[1] + 3200 + 3000, deployed=True)
    steps, charged, first, n = _walk_prince_to_tower(g, pid)
    assert any(charged), "Prince must become charged after an uninterrupted 2500 run-up"
    k = charged.index(True)
    walked = sum(steps[:k + 1])
    assert 2400 <= walked <= 2500 + 120, f"charged after walking {walked:.0f} (charge_range 2500)"
    fast = [s for s in steps[k + 1:] if s > 0]
    assert fast and max(fast) > 100, f"charged speed must be x2 (120/tick); steps after charge {fast}"
    assert max(steps) <= 120 + 1e-9
    assert first == 783, f"charged hit dealt {first}, want damage_special 783"
    last_move = max(i for i, s in enumerate(steps, start=1) if s > 0)
    assert n - last_move <= 2, f"charged hit landed {n - last_move} ticks after stopping (no windup)"
    g.tick(1)
    assert bool(K.ent(g, pid)["charged"]) is False, "the charge is consumed by the hit"
    hp = K.tower_hp(g, 1, 1)
    K.run_until(g, lambda gg: K.tower_hp(gg, 1, 1) < hp, 60, "second Prince hit")
    assert hp - K.tower_hp(g, 1, 1) == 391, "uncharged Prince hit is 391"


def test_prince_uncharged_when_run_up_too_short(pr):
    g = K.new_game()
    pid = K.spawn1(g, 0, H.PRINCE, T1_LEFT[0], T1_LEFT[1] + 3200 + 1500, deployed=True)
    steps, charged, first, n = _walk_prince_to_tower(g, pid)
    assert not any(charged)
    assert first == 391


def test_zap_resets_prince_charge(pr):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    pid = K.spawn1(g, 0, H.PRINCE, T1_LEFT[0], T1_LEFT[1] + 3200 + 3000, deployed=True)
    K.run_until(g, lambda gg: bool(K.ent(gg, pid)["charged"]), 60, "Prince charging")
    assert K.cast(g, 1, H.ZAP, *K.pos(K.ent(g, pid))) == H.OK
    g.tick(1)
    assert bool(K.ent(g, pid)["charged"]) is False, "stun resets the charge"
    hp = K.tower_hp(g, 1, 1)
    K.run_until(g, lambda gg: K.tower_hp(gg, 1, 1) < hp, 80, "Prince hit after the stun")
    assert hp - K.tower_hp(g, 1, 1) == 391, "the remaining ~500 of run-up cannot re-charge"


def test_log_knockback_resets_prince_charge(pr):
    """A charged team-0 Prince walking up the left lane into team 1's half is hit by a team-1 Log
    cast from team 1's own half just ahead of it (v0.2: the Log is own-half only); the Log rolls
    toward team 0 (+y), hits it, and the knockback resets the charge."""
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    pid = K.spawn1(g, 0, H.PRINCE, 3500, 20000, deployed=True)
    K.run_until(g, lambda gg: bool(K.ent(gg, pid)["charged"]) and K.pos(K.ent(gg, pid))[1] < 13500,
                120, "charged Prince entering team 1's half")
    x, y = K.pos(K.ent(g, pid))
    tap = (3500, ((y - 1200) // 1000) * 1000 + 500)          # a tile centre in team 1's half
    hp0, tower0 = K.hp_of(g, pid), K.tower_hp(g, 1, 1)
    assert K.cast(g, 1, H.LOG, *tap) == H.OK
    K.run_until(g, lambda gg: K.hp_of(gg, pid) < hp0 - 200, 15, "Log hitting the Prince")
    assert K.tower_hp(g, 1, 1) == tower0, "setup: the Prince has not hit the tower yet"
    assert bool(K.ent(g, pid)["charged"]) is False, "knockback resets the charge"


# ------------------------------------------------------------------------------------------
# Ice Golem death damage + death slow area
# ------------------------------------------------------------------------------------------
def test_ice_golem_death_damage_and_slow(pr):
    g = K.new_game()
    gx, gy = 9000, 11000
    ig = K.spawn1(g, 0, H.ICE_GOLEM, gx, gy, deployed=False)
    near = [K.spawn1(g, 1, H.CANNON, gx, gy - 1200, deployed=True),
            K.spawn1(g, 1, H.CANNON, gx + 1200, gy, deployed=True),
            K.spawn1(g, 1, H.CANNON, gx - 1200, gy, deployed=True),
            K.spawn1(g, 1, H.CANNON, gx, gy + 1200, deployed=True)]   # 1200 <= 2000 + 600
    far = K.spawn1(g, 1, H.CANNON, gx, gy + 3200, deployed=True)      # 3200 > 2600
    cs = near + [far]
    prev = {c: K.hp_of(g, c) for c in cs}
    for n in range(1, 120):
        g.tick(1)
        if K.ent(g, ig) is None:
            break
        prev = {c: K.hp_of(g, c) for c in cs}
    else:
        raise AssertionError("five Cannons failed to kill the Ice Golem")
    # death damage is buffered for the next Resolve; the slow is applied once, for 2000 ms
    # (SPEC §13.3 / §13.7): exactly 40 slowed observations
    slowed, far_slowed, all_near = [], [], []
    for _ in range(50):
        slowed.append(bool(K.ent(g, near[0])["slowed"]))
        all_near.append(all(bool(K.ent(g, c)["slowed"]) for c in near))
        far_slowed.append(bool(K.ent(g, far)["slowed"]))
        g.tick(1)
    for c in near:
        lost = prev[c] - K.hp_of(g, c)
        assert 84 <= lost <= 84 + 2 * 51, f"near Cannon lost {lost} (death damage 84 + drain)"
    assert not any(far_slowed), "a Cannon outside 2000 + r was slowed"
    assert all_near == slowed, "every Cannon inside the area is slowed for the same window"
    assert sum(slowed) == 40, f"slow observed {sum(slowed)} ticks; SPEC: applied once for 2000 ms"
    first = slowed.index(True)
    assert first <= 1 and all(slowed[first:first + 40]), "the slow is one contiguous 40-tick window"


def test_ice_golem_death_damage_exact(pr):
    g = K.new_game()
    gx, gy = 9000, 11000
    ig = K.spawn1(g, 0, H.ICE_GOLEM, gx, gy, deployed=False)
    near = [K.spawn1(g, 1, H.CANNON, gx + dx, gy + dy, deployed=True)
            for dx, dy in ((0, -1200), (1200, 0), (-1200, 0), (0, 1200))]
    far = K.spawn1(g, 1, H.CANNON, gx, gy + 3200, deployed=True)
    cs = near + [far]
    hist = []
    for _ in range(120):
        hist.append({c: K.hp_of(g, c) for c in cs})
        g.tick(1)
        if K.ent(g, ig) is None:
            break
    hist.append({c: K.hp_of(g, c) for c in cs})
    for _ in range(3):
        g.tick(1)
        hist.append({c: K.hp_of(g, c) for c in cs})
    # the 84 lands in a single tick: some one-tick drop of 84 + drain (1-2) for every near Cannon
    for c in near:
        drops = [hist[i][c] - hist[i + 1][c] for i in range(len(hist) - 4, len(hist) - 1)]
        assert any(84 <= d <= 86 for d in drops), f"no single-tick 84 death-damage drop: {drops}"
    for i in range(len(hist) - 4, len(hist) - 1):
        assert hist[i][far] - hist[i + 1][far] <= 2


def test_slowed_troop_moves_at_70_percent(pr):
    g = K.new_game()
    ig = K.spawn1(g, 0, H.ICE_GOLEM, 9000, 11000, deployed=False)
    for dx in (-1200, 1200):
        K.spawn1(g, 1, H.CANNON, 9000 + dx, 11000 - 1000, deployed=True)
    K.spawn1(g, 1, H.CANNON, 9000, 9800, deployed=True)
    kn = K.spawn1(g, 1, H.KNIGHT, 9000, 12300, deployed=False)       # deploying next to it
    K.run_until(g, lambda gg: K.ent(gg, ig) is None, 150, "Ice Golem death")
    K.run_until(g, lambda gg: bool(K.ent(gg, kn)["slowed"]) and not bool(K.ent(gg, kn)["deploying"]),
                40, "slowed, deployed Knight")
    p = K.pos(K.ent(g, kn))
    steps = []
    for _ in range(10):
        g.tick(1)
        e = K.ent(g, kn)
        if not bool(e["slowed"]):
            break
        q = K.pos(e)
        steps.append(math.dist(p, q))
        p = q
    moving = [s for s in steps if s > 0]
    assert moving, "the slowed Knight never walked"
    assert max(moving) <= 42 + 1e-9, f"slowed Knight step {max(moving):.1f} > floor(60*70/100) = 42"


# ------------------------------------------------------------------------------------------
# Ice Spirit: kamikaze, splash, freeze
# ------------------------------------------------------------------------------------------
def test_ice_spirit_kamikaze_splash_freeze(pr):
    g = K.new_game()
    c1 = K.spawn1(g, 1, H.CANNON, 9000, 11700, deployed=False)            # target, 2300 away
    c2 = K.spawn1(g, 1, H.CANNON, 10200, 11700, deployed=False)           # 1200 from impact
    c3 = K.spawn1(g, 1, H.CANNON, 9000 - 2300, 11700, deployed=False)     # 2300 > 1500 + 600
    sp = K.spawn1(g, 0, H.ICE_SPIRIT, 9000, 14000, deployed=False)
    cs = [c1, c2, c3]
    prev = {c: K.hp_of(g, c) for c in cs}
    K.run_until(g, lambda gg: K.ent(gg, sp) is None, 60, "Ice Spirit firing (kamikaze)")
    assert len(g.projectiles()) >= 1, "the kamikaze projectile continues after the Spirit dies"
    hit = K.run_until(g, lambda gg: prev[c1] - K.hp_of(gg, c1) >= 100, 20, "Ice Spirit impact")
    for c in (c1, c2):
        lost = prev[c] - K.hp_of(g, c)
        assert 110 <= lost <= 110 + 2 * 30, f"splash lost {lost}"
        e = K.ent(g, c)
        assert bool(e["stunned"]) is True, "Freeze (-100 speed/hit speed) is a stun"
        assert int(e["target_id"]) == -1, "freeze clears the target"
    assert not bool(K.ent(g, c3)["stunned"])
    assert prev[c3] - K.hp_of(g, c3) <= 2 * 30
    n = 0                                  # stunned observations, starting with the impact one
    while bool(K.ent(g, c1)["stunned"]):
        n += 1
        assert n < 40
        g.tick(1)
    assert n == 22, f"freeze observed {n} ticks; SPEC §13.3: impact tick N .. N+21, acts in N+22"
    assert hit >= 1


def test_same_buff_reapplication_refreshes_to_max(pr):
    """Two Zaps (same buff, 500 ms) 5 ticks apart: the stun ends 10 ticks after the SECOND one
    (remaining = max(old remaining, new)), i.e. ~15 stunned observations in total."""
    g = K.new_game("hog26", "giant", deploy_lockout_ticks=0)
    c1 = K.spawn1(g, 0, H.CANNON, 9000, 20300, deployed=True)
    assert K.cast(g, 1, H.ZAP, 9000, 20300) == H.OK
    g.tick(1)
    assert bool(K.ent(g, c1)["stunned"])
    n = 1
    for _ in range(4):
        g.tick(1)
        n += 1
        assert bool(K.ent(g, c1)["stunned"])
    assert K.cast(g, 1, H.ZAP, 9000, 20300) == H.OK
    while True:
        g.tick(1)
        if not bool(K.ent(g, c1)["stunned"]):
            break
        n += 1
        assert n < 40
    assert n in (14, 15, 16), f"stunned for {n} observations; refresh-to-max gives 5 + 10 = 15"
