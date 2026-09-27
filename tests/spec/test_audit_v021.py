"""SPEC v0.2.1 §14 audit amendments: same-tick validation, melee mid-swing retarget, hit-once
capacity, formation ejection side, snapshot restore validation, learner log keys, robustness,
and a large-battle performance/determinism stress test."""
import configparser
import json
import math
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import envkit as E
import gamekit as K
import helpers as H

HERE = os.path.dirname(os.path.abspath(__file__))
KING0 = H.TOWER_POS[0][0]


def ceil_div(a, b):
    return -((-a) // b)


# ------------------------------------------------------------------------------------------
# §14.1 same-tick play validation
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("builder", [0, 1])
def test_same_tick_building_does_not_invalidate_enemy_troop(pr, builder):
    """The builder team places a Cannon covering a pocket tile in the same tick the other team plays
    a troop on that tile. Both plays were valid at the start of Upkeep, so both are applied."""
    troop_team = 1 - builder
    decks = ("hog26", "giant") if builder == 0 else ("giant", "hog26")
    g = K.new_game(*decks, deploy_lockout_ticks=0)
    # open the troop team's pocket on the builder's own-frame-left side
    K.destroy_tower(g, builder, 1 if builder == 0 else 2)
    g.tick(5)
    # builder: own tile (5,19) -> Cannon footprint own tiles 4-6 x 18-20; the troop team's own tile
    # (12,12) is the rotation of the builder's own tile (5,19), i.e. inside that footprint
    assert H.troop_tile_legal(troop_team, 12, 12, K.alive_map(g), K.living_buildings(g))
    K.put_first(g, builder, H.CANNON)
    K.put_first(g, troop_team, H.KNIGHT)
    g.set_elixir(builder, 20000)
    g.set_elixir(troop_team, 20000)
    hand_t, queue_t = K.hand(g, troop_team), K.queue(g, troop_team)
    assert g.play_tile(builder, 0, 5, 19) == H.OK
    assert g.play_tile(troop_team, 0, 12, 12) == H.OK
    g.tick(1)
    assert len(K.units(g, team=builder, card=H.CANNON)) == 1, "the building was placed"
    assert len(K.troops(g, troop_team, H.KNIGHT)) == 1, "the same-tick troop play must be applied"
    assert K.elixir(g, troop_team) == 20000 + 50 - 3 * 2800
    assert K.hand(g, troop_team) == [queue_t[0]] + hand_t[1:]
    assert K.queue(g, troop_team) == queue_t[1:] + [H.KNIGHT]


@pytest.mark.slow
def test_env_every_mask_legal_action_is_applied(pr):
    """Long self-play soak (§14.1): plays counted == mask-legal non-noop actions issued, per team,
    and dropped_plays == 0 in every episode log."""
    for d0, d1 in (("hog26", "hog26"), ("bait", "giant"), ("giant", "bait")):
        env = E.make(num_envs=1, num_agents=2, deck0=d0, deck1=d1, log_interval=1, seed=21)
        obs, _ = env.reset(seed=21)
        rng = np.random.default_rng(5)
        issued = [0, 0]
        episodes = 0
        for _ in range(1800):
            acts = [E.random_bot_action(rng, r, 0.5) for r in obs]
            for r in range(2):
                issued[r] += int(acts[r] != 0)
            obs, rew, term, trunc, infos = E.step(env, acts)
            for lg in E.logs_in(infos):
                assert "dropped_plays" in lg, "log key dropped_plays (SPEC §14.1/§14.6)"
                assert lg["dropped_plays"] == 0, "a mask-legal play was dropped"
                assert lg["plays_0"] == issued[0] and lg["plays_1"] == issued[1], \
                    f"plays {lg['plays_0']},{lg['plays_1']} vs mask-legal issued {issued}"
                assert lg["illegal_actions"] == 0
                issued = [0, 0]
                episodes += 1
        assert episodes >= 2
        env.close()


# ------------------------------------------------------------------------------------------
# §14.2 melee mid-swing retarget
# ------------------------------------------------------------------------------------------
# Scene (all spawned deployed=True, so the Knight is fully loaded: SPEC §13.10):
#   our Knight K (team 0) at (9000, 14000); enemy Knight A at (11100, 14000) (2100 away, edge 1600:
#   chosen first); enemy Cannon B at (6750, 14000) (2250 away, edge 1650, inside reach 2300).
#   K hits A at obs 1+9 = 10, 34, 58 (cadence 24). Our Fireball lands on tile centre (10500, 14500)
#   at obs cast+25 (d = 14577 -> ceil(d/600) = 25): cast at 28 -> lands at 53, 5 ticks before the
#   hit due at 58, knocking A ~1000 away (out of reach 2200); B is 3783 from the impact (unhit).
K_POS, A_POS, B_POS, IMPACT = (9000, 14000), (11100, 14000), (6750, 14000), (10500, 14500)


def _melee_scene(with_b):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    k = K.spawn1(g, 0, H.KNIGHT, *K_POS, deployed=True)
    a = K.spawn1(g, 1, H.KNIGHT, *A_POS, deployed=True)
    b = K.spawn1(g, 1, H.CANNON, *B_POS, deployed=True) if with_b else None
    return g, k, a, b


def test_tester_melee_scene_arithmetic():
    d = math.dist(KING0, IMPACT)
    assert ceil_div(int(d), 600) == 25 and 28 + 25 == 53
    assert math.dist(K_POS, A_POS) <= 1200 + 500 + 500 < math.dist(K_POS, A_POS) + 1000
    assert math.dist(K_POS, B_POS) <= 1200 + 500 + 600
    assert math.dist(K_POS, A_POS) - 500 < math.dist(K_POS, B_POS) - 600
    assert math.dist(IMPACT, B_POS) > 2500 + 600 and math.dist(IMPACT, A_POS) <= 2500 + 500


def test_melee_retargets_mid_swing_keeping_progress(pr):
    g, k, a, b = _melee_scene(True)
    hp_a, hp_b = K.hp_of(g, a), K.hp_of(g, b)
    a_hits, b_hits = [], []
    for n in range(1, 66):
        if n == 29:                                   # queued at state tick 28
            assert K.cast(g, 0, H.FIREBALL, *IMPACT) == H.OK
        g.tick(1)
        ca, cb = K.hp_of(g, a), K.hp_of(g, b)
        if hp_a - ca > 0:
            a_hits.append((n, hp_a - ca))
        if hp_b - cb >= 150:
            b_hits.append((n, hp_b - cb))
        hp_a, hp_b = ca, cb
    assert [t for t, d in a_hits if d == 202][:2] == [10, 34], f"setup: K's hits on A {a_hits}"
    assert (53, 688) in a_hits, f"setup: Fireball on A at obs 53 {a_hits}"
    assert not [t for t, d in a_hits if t > 53], f"K hit A after it was knocked out of reach: {a_hits}"
    assert b_hits and b_hits[0][0] == 58 and 202 <= b_hits[0][1] <= 204, \
        f"the started swing must land on schedule (obs 58) on the in-reach enemy B: {b_hits}"


def test_melee_swing_cancelled_when_nothing_in_reach(pr):
    g, k, a, _ = _melee_scene(False)
    hp_a = K.hp_of(g, a)
    events, back_in_reach = [], None
    for n in range(1, 90):
        if n == 29:
            assert K.cast(g, 0, H.FIREBALL, *IMPACT) == H.OK
        g.tick(1)
        ca = K.hp_of(g, a)
        if hp_a - ca > 0:
            events.append((n, hp_a - ca))
        hp_a = ca
        d = math.dist(K.pos(K.ent(g, k)), K.pos(K.ent(g, a)))
        if n > 53 and back_in_reach is None and d <= 2200:
            back_in_reach = n
    assert (53, 688) in events
    later = [t for t, d in events if t > 53]
    assert 58 not in later, "no damage at distance: the started swing must be cancelled"
    assert back_in_reach is not None, "the Knights should close in again"
    assert later and later[0] >= back_in_reach + 9, \
        f"after cancelling, a fresh cycle: first hit {later[:1]} vs back in reach at {back_in_reach}"


def test_projectile_attacker_keeps_target_within_reach_plus_500(pr):
    """Enemy Knight A stands at (3500, 23000) hitting our left princess (2500 away). Our Musketeer M
    is spawned (loaded, §13.10) at (3500, 16600), 6400 from A, at state tick 30: it acquires A at
    obs 31 and fires at obs 44. Our Fireball on tile centre (3500, 22500) (500 north of A;
    d = 8515 -> ceil(d/600) = 15) is cast at state tick 24 and lands at obs 39, pushing A 1000
    south to (3500, 24000) (touching the princess circle): 7400 from M, beyond reach 7000 but
    within 7500. M keeps A mid-swing and fires at obs 44; the shell (7400 - 450, 1000/tick ->
    7 ticks) hits A at obs 51."""
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    a = K.spawn1(g, 1, H.KNIGHT, 3500, 23000, deployed=True)
    assert ceil_div(int(math.dist(KING0, (3500, 22500))), 600) == 15
    hp = K.hp_of(g, a)
    drops, targets, m = [], {}, None
    for n in range(1, 60):
        if n == 25:
            assert K.cast(g, 0, H.FIREBALL, 3500, 22500) == H.OK
        if n == 31:
            m = K.spawn1(g, 0, H.MUSKETEER, 3500, 16600, deployed=True)
        g.tick(1)
        cur = K.hp_of(g, a)
        if hp - cur > 0:
            drops.append((n, hp - cur))
        hp = cur
        if m is not None:
            targets[n] = int(K.ent(g, m)["target_id"])
    assert any(t == 39 and d >= 688 for t, d in drops), f"setup: Fireball on A at obs 39: {drops}"
    d_after = math.dist(K.pos(K.ent(g, m)), K.pos(K.ent(g, a)))
    assert targets[31] == a, "setup: M acquires A on its first tick"
    assert all(targets[n] == a for n in range(39, 45)), \
        f"M must keep A (within reach + 500) until the shot is fired: {[targets[n] for n in range(39, 45)]}"
    shells = [t for t, d in drops if d in (217, 217 + 109)]
    assert 51 in shells, f"the shot fired at obs 44 must hit A at obs 51: {drops} (A now {d_after:.0f} away)"


# ------------------------------------------------------------------------------------------
# §14.3 hit-once capacity
# ------------------------------------------------------------------------------------------
def _big_drop_counts(g, ids, ticks, thresh):
    prev = {i: K.hp_of(g, i) for i in ids}
    hits = {i: [] for i in ids}
    for _ in range(ticks):
        g.tick(1)
        for i in ids:
            if prev[i] is None:
                continue
            cur = K.hp_of(g, i)
            if cur is None:
                hits[i].append("died")        # only repeated hits could kill an 824-hp Cannon here
            elif prev[i] - cur >= thresh:
                hits[i].append(prev[i] - cur)
            prev[i] = cur
    return hits


def test_log_hits_eighty_entities_exactly_once(pr):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    ids = [K.spawn1(g, 1, H.CANNON, x, 7700 + 450 * i, deployed=True)
           for x in (7700, 8600, 9500, 10400, 11300) for i in range(16)]
    assert len(ids) == 80 and len(K.ents(g)) == 86
    assert K.cast(g, 0, H.LOG, 9500, 17500) == H.OK      # rolls 10100 up to y = 7400
    hits = _big_drop_counts(g, ids, 70, 200)
    bad = {i: h for i, h in hits.items() if len(h) != 1 or h[0] == "died" or not 268 <= h[0] <= 270}
    assert not bad, f"{len(bad)} of 80 entities not hit exactly once for 268: {list(bad.items())[:5]}"


def test_arrows_hit_eighty_entities_once_per_wave(pr):
    g = K.new_game("giant", "giant", deploy_lockout_ticks=0)
    cx, cy = 9500, 10500
    pts = sorted(((math.dist((cx, cy), (cx + 650 * i, cy + 650 * j)), i, j)
                  for i in range(-6, 7) for j in range(-6, 7)
                  if math.dist((cx, cy), (cx + 650 * i, cy + 650 * j)) <= 3800))[:80]
    ids = [K.spawn1(g, 1, H.CANNON, cx + 650 * i, cy + 650 * j, deployed=True) for _, i, j in pts]
    assert len(ids) == 80
    assert K.cast(g, 0, H.ARROWS, cx, cy) == H.OK
    hits = _big_drop_counts(g, ids, 40, 100)
    bad = {i: h for i, h in hits.items()
           if len(h) != 3 or not all(d != "died" and 122 <= d <= 124 for d in h)}
    assert not bad, f"{len(bad)} of 80 entities not hit exactly once per wave: {list(bad.items())[:5]}"


# ------------------------------------------------------------------------------------------
# §14.4 formation ejection stays on the placing team's side
# ------------------------------------------------------------------------------------------
EDGE_TAPS = [(9000, 17000), (6000, 17000), (12000, 17000), (1500, 17000), (16500, 17000),
             (4000, 17000), (14000, 17000), (9000, 17400), (3000, 17000), (15000, 17200)]


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("card,deck", [(H.SKARMY, "bait"), (H.SKELETONS, "hog26"), (H.ARCHERS, "bait")])
def test_formation_ejection_never_crosses_river(pr, team, card, deck):
    for (x, y) in EDGE_TAPS:
        g = K.new_game(deck, deck, deploy_lockout_ticks=0)
        px, py = (x, y) if team == 0 else (H.ARENA_W - x, H.ARENA_H - y - 1)
        assert K.cast(g, team, card, px, py) == H.OK, f"tap {px, py} refused"
        for _ in range(20):
            g.tick(1)
            for e in K.troops(g, team, card):
                if not bool(e["deploying"]):
                    continue
                ex, ey = K.pos(e)
                # never across the river (the enemy half); a bridge cell is land inside the river
                # band and is not 'across' (SPEC §14.4 only relocates members that land in water)
                not_across = ey >= 15000 if team == 0 else ey < 17000
                assert not_across, f"team {team} {H.CARD_NAMES[card]} tapped at {(px, py)}: member across the river at {(ex, ey)}"
                assert not H.point_in_water(ex, ey), f"member in water at {(ex, ey)}"


# ------------------------------------------------------------------------------------------
# §14.5 snapshot restore validation (fuzzed in a subprocess: a crash fails the test)
# ------------------------------------------------------------------------------------------
FUZZ = textwrap.dedent(r'''
    import json, random, sys
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    import numpy as np
    import pufferroyale as pr
    import gamekit as K, helpers as H

    g = pr.Game(deck0="bait", deck1="giant", seed=3, deploy_lockout_ticks=0)
    states = []
    def rec():
        b = bytes(g.snapshot())
        a = np.frombuffer(b[:len(b) // 4 * 4], dtype="<i4")
        states.append((a, len(g.entities()), len(g.projectiles()), len(g.effects()), int(g.state()["tick"])))
    rec()
    K.spawn(g, 1, H.SKARMY, 9000, 11000); rec()
    K.cast(g, 0, H.FIREBALL, 3500, 6500); g.tick(1); rec()
    K.cast(g, 0, H.GOBLIN_BARREL, 14500, 6500); g.tick(1); rec()
    K.cast(g, 1, H.ARROWS, 9000, 21000); g.tick(1); rec()
    K.cast(g, 0, H.LOG, 9500, 17500); g.tick(1); rec()
    K.cast(g, 1, H.ZAP, 9000, 21000); g.tick(1); rec()
    K.spawn(g, 0, H.ARCHERS, 9000, 14000); g.tick(30); rec()
    g.tick(40); rec(); g.tick(60); rec()
    offsets = {}
    for idx, name in ((1, "entities"), (2, "projectiles"), (3, "effects"), (4, "tick")):
        cand = None
        for s in states:
            c = set(np.nonzero(s[0] == s[idx])[0].tolist())
            cand = c if cand is None else cand & c
        offsets[name] = sorted(cand)
    # rebuild a busy base state
    g.reset(seed=3)
    K.spawn(g, 1, H.SKARMY, 9000, 11000); K.spawn(g, 0, H.KNIGHT, 9000, 21000)
    K.cast(g, 0, H.FIREBALL, 3500, 6500); g.tick(2)
    K.cast(g, 1, H.ARROWS, 9000, 21000); g.tick(1)
    base = bytes(g.snapshot())
    order = K.deck_of(g, 0)
    g.set_hand(0, order[::-1]); swapped = bytes(g.snapshot()); g.restore(base)
    hand_bytes = [i for i in range(len(base)) if base[i] != swapped[i]]

    res = {"offsets": offsets, "hand_bytes": hand_bytes[:16], "rejected": [], "accepted": 0, "bad": []}
    def attempt(blob, must_reject, label):
        g.restore(base)
        h = g.hash()
        try:
            g.restore(blob)
        except ValueError:
            if g.hash() != h:
                res["bad"].append(["state changed after a failed restore", label])
            res["rejected"].append(label)
            return
        except Exception as e:
            res["bad"].append(["non-ValueError exception", label, repr(e)])
            return
        if must_reject:
            res["bad"].append(["invalid snapshot accepted", label])
        g.hash(); g.state(); g.entities(); g.projectiles(); g.effects()
        res["accepted"] += 1

    def put32(b, word, value):
        a = bytearray(b)
        a[4 * word:4 * word + 4] = int(value & 0xFFFFFFFF).to_bytes(4, "little")
        return bytes(a)

    for label, blob in (("truncated-1", base[:-1]), ("half", base[:len(base) // 2]), ("empty", b""),
                        ("extended", base + b"\0"), ("doubled", base + base)):
        attempt(blob, True, label)
    for name in ("entities", "projectiles", "effects"):
        for w in offsets[name]:
            for v in (257, 1000, 2**31 - 1, -1, 0xFFFFFFFF):
                attempt(put32(base, w, v), True, f"{name}@{w}={v}")
    for w in offsets["tick"]:
        for v in (6001, 7000, -1, 2**31 - 1):
            attempt(put32(base, w, v), True, f"tick@{w}={v}")
    for i in hand_bytes[:8]:
        a = bytearray(base); a[i] = 0xFF
        attempt(bytes(a), True, f"hand byte {i}=255")
    rng = random.Random(1234)
    for trial in range(400):
        a = bytearray(base)
        for _ in range(rng.randint(1, 8)):
            a[rng.randrange(len(a))] ^= 1 << rng.randrange(8)
        attempt(bytes(a), False, f"flip {trial}")
    for trial in range(200):
        attempt(put32(base, rng.randrange(len(base) // 4), 0xFFFFFFFF), False, f"word {trial}")
    g.restore(base)
    print(json.dumps(res))
''')


@pytest.mark.slow
def test_restore_rejects_corrupt_snapshots_without_crashing(pr):
    code = f"ROOT = {H.ROOT!r}\nHERE = {HERE!r}\n" + FUZZ
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                         cwd=H.ROOT)
    assert out.returncode == 0, f"restore fuzz crashed (rc={out.returncode}):\n{out.stderr[-3000:]}"
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["offsets"]["entities"] and res["offsets"]["projectiles"] and res["offsets"]["tick"], \
        f"tester could not locate pool counts/tick in the snapshot: {res['offsets']}"
    assert res["hand_bytes"], "tester could not locate the hand in the snapshot"
    assert not res["bad"], f"restore validation problems (SPEC §14.5): {res['bad'][:12]}"
    for must in ("truncated-1", "half", "empty", "extended", "doubled"):
        assert must in res["rejected"]


# ------------------------------------------------------------------------------------------
# §14.6 learner-centric logs, tiebreak flag
# ------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_learner_keys_passive_learner_vs_heuristic(pr):
    env = E.make(num_agents=1, opponent="heuristic", learner_side="random", log_interval=1, seed=31)
    obs, _ = env.reset(seed=31)
    logs = []
    for _ in range(8 * 601):
        obs, rew, term, trunc, infos = E.step(env, [0])
        logs += E.logs_in(infos)
        if len(logs) >= 8:
            break
    assert len(logs) >= 8
    for lg in logs:
        for k in ("learner_score", "learner_return", "learner_win"):
            assert k in lg, f"log key {k!r} missing (SPEC §14.6)"
        assert abs(lg["learner_score"] - (lg["learner_return"] + 1) / 2) < 1e-6
        assert lg["learner_win"] == float(lg["learner_return"] == 1.0)
    mean = np.mean([lg["learner_score"] for lg in logs])
    assert mean <= 0.1, f"a passive learner vs the heuristic must score ~0, got {mean:.2f}"
    team0 = [lg["score"] for lg in logs]
    assert max(team0) == 1.0, "with learner_side='random' the learner must sometimes be team 1 " \
                              "(team 0 = heuristic wins) - learner_* must follow the learner, not team 0"
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize("fs", [600, 10])
def test_selfplay_learner_keys_and_tiebreak_flag(pr, fs):
    """Self-play: learner_* equal team 0's values. For every episode: tiebreak == 1 iff the
    end-of-overtime comparison ran, i.e. iff the final crowns are equal (a crown difference in
    overtime or a King kill ends the match earlier with unequal crowns)."""
    env = E.make(num_agents=2, frame_skip=fs, deck0="hog26", deck1="hog26", log_interval=1, seed=7)
    obs, _ = env.reset(seed=7)
    rng = np.random.default_rng(8)
    logs = []
    want_eps = 30 if fs == 600 else 6
    for _ in range(want_eps * (6000 // fs) + 100):
        acts = [E.random_bot_action(rng, r, 0.6 if fs == 600 else 0.2) for r in obs]
        obs, rew, term, trunc, infos = E.step(env, acts)
        logs += E.logs_in(infos)
        if len(logs) >= want_eps:
            break
    assert logs
    for lg in logs:
        assert lg["learner_score"] == lg["score"] and lg["learner_return"] == lg["episode_return"]
        assert lg["learner_win"] == lg["win_0"]
        if lg["crowns_0"] == 3 and lg["crowns_1"] == 3:
            continue
        want = float(lg["crowns_0"] == lg["crowns_1"])
        assert lg["tiebreak"] == want, f"tiebreak={lg['tiebreak']} with crowns {lg['crowns_0']}-{lg['crowns_1']}"
    env.close()


@pytest.mark.parametrize("how", ["KING", "OVERTIME_CROWNS"])
def test_game_ending_on_tick_5999_is_not_a_tiebreak(pr, how):
    g = K.new_game("giant", "giant")
    g.tick(5999)
    idx = 0 if how == "KING" else 2
    g.set_tower_hp(1, idx, 1)
    assert K.cast(g, 0, H.ZAP, *H.TOWER_POS[1][idx]) == H.OK      # applied in tick 5999
    g.tick(1)
    assert K.over(g) and K.tick_of(g) == 6000
    assert K.end_reason(g) == how and K.result(g, 0) == 1


def test_sweep_metric_is_learner_score(pr):
    cfg = configparser.ConfigParser()
    path = os.path.join(H.ROOT, "pufferroyale", "config", "royale.ini")
    assert os.path.isfile(path), "pufferroyale/config/royale.ini missing"
    cfg.read(path)
    assert cfg.has_section("sweep") and cfg.get("sweep", "metric").strip() == "learner_score"


# ------------------------------------------------------------------------------------------
# Robustness
# ------------------------------------------------------------------------------------------
SPAWN_FUZZ = textwrap.dedent(r'''
    import json, sys
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    import pufferroyale as pr
    import helpers as H
    g = pr.Game(deck0="hog26", deck1="giant", seed=0)
    out = {"inside": True, "bad": [], "raised": []}
    pts = [(-100000, -100000), (10**6, 10**6), (2**31 - 1, -2**31), (-2**31, 2**31 - 1),
           (18000, 32000), (17999, 31999), (0, 0), (-1, 16000), (9000, -5)]
    for team in (0, 1):
        for card in (H.KNIGHT, H.MINIONS, H.CANNON, H.GIANT, H.SKARMY):
            for (x, y) in pts:
                ids = g.spawn(team, card, x, y, True)
                for e in g.entities():
                    if e["id"] in ids and not (0 <= e["x"] < 18000 and 0 <= e["y"] < 32000):
                        out["bad"].append([team, card, x, y, e["x"], e["y"]])
    for (x, y) in ((2**40, 5), (5, -2**40)):
        try:
            g.spawn(0, H.KNIGHT, x, y, True)
        except (OverflowError, ValueError, TypeError) as e:
            out["raised"].append(repr(e))
    g.tick(100)
    for e in g.entities():
        if not (0 <= e["x"] < 18000 and 0 <= e["y"] < 32000):
            out["bad"].append(["after ticks", e["unit"], e["x"], e["y"]])
    print(json.dumps(out))
''')


def test_spawn_extreme_coordinates_clamped(pr):
    code = f"ROOT = {H.ROOT!r}\nHERE = {HERE!r}\n" + SPAWN_FUZZ
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         cwd=H.ROOT)
    assert out.returncode == 0, f"spawn with extreme coordinates crashed:\n{out.stderr[-3000:]}"
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert not res["bad"], f"entities outside the arena: {res['bad'][:10]}"


def test_policy_masks_on_lstm_train_path(pr):
    """PuffeRL's LSTM train path calls policy(obs, state) with obs shaped (segments, horizon, OBS):
    every masked logit must equal the dtype minimum, on every row."""
    import torch
    import pufferroyale.torch as prt
    env = E.make(num_envs=2, num_agents=2, seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    seq = []
    for _ in range(12):
        seq.append(obs.copy())
        obs, *_ = E.step(env, [E.random_bot_action(rng, r, 0.5) for r in obs])
    batch = torch.as_tensor(np.stack(seq, axis=1))          # (segments=4, horizon=12, OBS)
    for policy in (prt.Recurrent(env, prt.Policy(env)), prt.Policy(env)):
        with torch.no_grad():
            logits, value = policy(batch if isinstance(policy, prt.Recurrent) else batch.reshape(-1, batch.shape[-1]),
                                   dict(action=None, lstm_h=None, lstm_c=None))
        logits = logits.reshape(-1, H.N_ACTIONS)
        masks = batch.reshape(-1, batch.shape[-1])[:, E.R().MASK_OFFSET:E.R().MASK_OFFSET + H.N_ACTIONS]
        assert logits.shape[0] == masks.shape[0] == 48
        minv = torch.finfo(logits.dtype).min
        assert torch.all(logits[masks < 0.5] == minv), f"{type(policy).__name__}: masked logits not at dtype min"
        assert torch.all(logits[masks > 0.5] > minv)
        assert torch.isfinite(value).all()
    env.close()


# ------------------------------------------------------------------------------------------
# Large-battle stress: ~210 skeletons
# ------------------------------------------------------------------------------------------
def _stress_game():
    g = K.new_game("bait", "bait", seed=0)
    for i, x in enumerate((1500, 3500, 5500, 9000, 12500, 14500, 16500)):
        K.spawn(g, 0, H.SKARMY, x, 19500, deployed=True)
        K.spawn(g, 1, H.SKARMY, H.ARENA_W - x, H.ARENA_H - 19500, deployed=True)
    return g


def test_stress_skeleton_armies_deterministic(pr):
    a, b = _stress_game(), _stress_game()
    assert len(K.ents(a)) == 6 + 7 * 2 * 15
    for _ in range(400):
        a.tick(1)
        b.tick(1)
        assert a.hash() == b.hash()


@pytest.mark.perf
def test_stress_skeleton_armies_throughput(pr):
    best = 0.0
    for rep in range(3):
        g = _stress_game()
        g.tick(20)                                      # warm-up
        t0 = time.perf_counter()
        g.tick(600)
        best = max(best, 600 / (time.perf_counter() - t0))
    assert best >= 10000, f"~210-skeleton battle runs at {best:.0f} ticks/s < 10,000"
