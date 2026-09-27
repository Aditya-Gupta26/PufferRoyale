"""SPEC §18 (v0.4.1) second-audit regression tests.

18.1 crown-tower acquisition ties -> own-frame left; 18.2 Royal Chef serving order (own-frame left
first, a levelled troop is not eligible again); 18.3 dash invulnerability evaluated when damage is
dealt; 18.4 seat-symmetric bots; 18.5 snapshot validation of multiplied values and pending spawns;
18.6 bf16 training never trains on stale weights; 18.7 league run robustness; plus the audit's
low-severity items (batch-size / --matches validation, strict JSON outputs, best_response inputs,
Recurrent init scale)."""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import cardsv3 as C
import gamekit as K
import helpers as H
import leaguekit as LK

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = H.ROOT


def rot(p):
    return (H.ARENA_W - p[0], H.ARENA_H - p[1])


def strict_json(text):
    """json.loads that refuses NaN / Infinity (the audit: outputs must be valid JSON)."""
    def bad(c):
        raise ValueError(f"non-standard JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def last_line(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    assert lines, "no stdout"
    return lines[-1]


def run(args, timeout=900):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


# ==========================================================================================
# 18.1 crown-tower ties
# ==========================================================================================
@pytest.mark.parametrize("card,y0", [(H.GIANT, 14000), (H.HOG, 17500)])
def test_crown_tower_tie_goes_to_own_frame_left(pr, card, y0):
    """At exactly x = 9000 both enemy Princess towers are tied for acquisition; team 0 must take
    the engine-left one (its own-left), team 1 the engine-right one (its own-left), and the two
    walks are 180-degree rotations of each other."""
    g0 = K.new_game("hog26", "hog26", seed=1, deploy_lockout_ticks=0)
    g1 = K.new_game("hog26", "hog26", seed=1, deploy_lockout_ticks=0)
    u0 = K.spawn1(g0, 0, card, 9000, y0)
    u1 = K.spawn1(g1, 1, card, *rot((9000, y0)))
    g0.tick(1)
    g1.tick(1)
    t0, t1 = int(K.ent(g0, u0)["target_id"]), int(K.ent(g1, u1)["target_id"])
    assert t0 == int(K.tower_entity(g0, 1, 1)["id"]), f"team 0 {H.CARD_NAMES[card]} targets {t0}: want own-left"
    assert t1 == int(K.tower_entity(g1, 0, 2)["id"]), f"team 1 {H.CARD_NAMES[card]} targets {t1}: want own-left"
    for n in range(60):
        g0.tick(1)
        g1.tick(1)
        p0, p1 = K.pos(K.ent(g0, u0)), K.pos(K.ent(g1, u1))
        assert rot(p0) == p1, f"tick {n + 2}: team-0 walk {p0} vs team-1 walk {p1} not rotations"


# ==========================================================================================
# 18.2 Royal Chef serving order
# ==========================================================================================
def _chef_scene(team, musk_side):
    g = K.new_game("hog26", "giant", seed=1, deploy_lockout_ticks=0, **{f"tower_troop{team}": "royal_chef"})
    own = lambda p: p if team == 0 else rot(p)
    g.tick(555)
    golem = K.spawn1(g, team, C.GOLEM, *own((9000, 22000)))            # 6519 from both Chef towers
    musk_at = (2000, 22000) if musk_side == "own_left" else (16000, 22000)
    musk = K.spawn1(g, team, H.MUSKETEER, *own(musk_at))               # near ONE tower only
    lev = {}
    while K.tick_of(g) < 566:
        g.tick(1)
        for i, base in ((golem, 5120), (musk, 721)):
            e = K.ent(g, i)
            if e is not None and int(e["max_hp"]) > base and i not in lev:
                lev[i] = K.tick_of(g)
    return lev, golem, musk


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("musk_side", ["own_left", "own_right"])
def test_chef_towers_are_served_own_frame_left_first(pr, team, musk_side):
    """The Golem (8) is within 7500 of both Chef towers, the Musketeer (4) of one. At the first
    period (560) the own-left tower serves first and takes the Golem; the own-right tower may not
    take the Golem again, so it levels the Musketeer only if the Musketeer is near IT."""
    lev, golem, musk = _chef_scene(team, musk_side)
    assert lev.get(golem) == 560, f"team {team}, musketeer {musk_side}: level-ups {lev}"
    if musk_side == "own_left":
        assert musk not in lev, f"team {team}: the own-left tower took the Golem; nobody is left for the " \
                                f"own-right tower (the Musketeer is out of its range): {lev}"
    else:
        assert lev.get(musk) == 560, f"team {team}: the own-right tower levels the Musketeer: {lev}"


# ==========================================================================================
# 18.3 dash invulnerability
# ==========================================================================================
def _freeze_seed(team):
    for seed in range(4000):
        d = ("random", "hog26") if team == 0 else ("hog26", "random")
        if C.FREEZE in K.deck_of(K.new_game(*d, seed=seed), team):
            return d, seed
    raise AssertionError("no random deck with Freeze")


def _dash_game(bandit_team, spell):
    caster = 1 - bandit_team
    if spell == C.FREEZE:
        decks, seed = _freeze_seed(caster)
    else:
        deck = {H.ZAP: "giant", H.LOG: "hog26"}[spell]
        decks, seed = ((deck, "hog26") if caster == 0 else ("hog26", deck)), 1
    g = K.new_game(*decks, seed=seed, deploy_lockout_ticks=0)
    own = lambda p: p if bandit_team == 0 else rot(p)
    K.spawn1(g, caster, C.GOLEM, *own((9000, 9000)), deployed=False)       # the dash target
    b = K.spawn1(g, bandit_team, C.BANDIT, *own((9000, 14000)))            # 5000 away: dash window
    return g, b


@pytest.mark.parametrize("bandit_team", [0, 1])
@pytest.mark.parametrize("spell", [H.ZAP, C.FREEZE, H.LOG])
def test_dash_invulnerability_holds_against_cancelling_spells(pr, bandit_team, spell):
    """The caster hits the Bandit with Zap / Freeze / The Log while its dash is in the moving phase
    (dash_state 2 from `entities()`, cross-checked by its 500/tick steps). The spell's stun or
    knockback lands (and cancels the dash) but deals 0 damage (§18.3)."""
    g, b = _dash_game(bandit_team, spell)
    m = None
    for n in range(1, 60):
        prev = K.pos(K.ent(g, b))
        g.tick(1)
        e = K.ent(g, b)
        if int(e["dash_state"]) == 2 and math.dist(prev, K.pos(e)) >= 400:
            m = n
            break
    assert m is not None, "setup: the Bandit never entered the moving dash phase"
    e = K.ent(g, b)
    hp0, pos0 = int(e["hp"]), K.pos(e)
    assert K.cast(g, 1 - bandit_team, spell, *pos0) == H.OK
    effect = None
    for k in range(1, 4):
        g.tick(1)
        e = K.ent(g, b)
        assert e is not None and int(e["hp"]) == hp0, \
            f"{C.ALL_NAMES[spell]} damaged the dashing Bandit ({hp0} -> {e and e['hp']}) at +{k}"
        if bool(e["stunned"]) or int(e["dash_state"]) != 2:
            effect = k
            break
    assert effect is not None, f"setup: {C.ALL_NAMES[spell]} never reached the Bandit"


# ==========================================================================================
# 18.4 seat-symmetric bots
# ==========================================================================================
SINGLES = [H.KNIGHT, H.GIANT, H.MUSKETEER, C.MINI_PEKKA, H.VALKYRIE, H.PRINCE, H.HOG, H.WIZARD,
           H.BABY_DRAGON, C.MEGA_MINION, C.PEKKA]
PRESET_NAMES = ["hog26", "giant", "bait", "golem", "lavaloon", "xbow", "miner_poison", "pekka_bridge", "royal_hogs"]


def mirrored_games(d0, d1, seed, hands=None, elixir=(28000, 28000), towers=None, units=(), ticks=0):
    """Game A and its seat-swapped 180-degree mirror B (A.team t <-> B.team 1-t)."""
    ga = K.new_game(d0, d1, seed=seed, deploy_lockout_ticks=0)
    gb = K.new_game(d1, d0, seed=seed, deploy_lockout_ticks=0)
    hands = hands or (K.deck_of(ga, 0), K.deck_of(ga, 1))
    for t in (0, 1):
        ga.set_hand(t, list(hands[t]))
        gb.set_hand(1 - t, list(hands[t]))
        ga.set_elixir(t, elixir[t])
        gb.set_elixir(1 - t, elixir[t])
    for (t, i), hp in (towers or {}).items():
        ga.set_tower_hp(t, i, hp)
        gb.set_tower_hp(1 - t, i if i == 0 else 3 - i, hp)
    for t, card, x, y in units:
        K.spawn(ga, t, card, x, y)
        K.spawn(gb, 1 - t, card, *rot((x, y)))
    for _ in range(ticks):
        ga.tick(1)
        gb.tick(1)
    return ga, gb


def bot_pair(ga, gb, kind, seed, team=0):
    a = pr_mod().Bot(kind=kind, seed=seed, play_prob=1.0).act(ga, team)
    b = pr_mod().Bot(kind=kind, seed=seed, play_prob=1.0).act(gb, 1 - team)
    return int(a), int(b)


def pr_mod():
    import pufferroyale
    return pufferroyale


def test_heuristic_finish_tower_tie_is_seat_symmetric(pr):
    """Both enemy Princesses at 100 HP and a Fireball in hand: team 0 (scene A) and team 1 (the
    mirrored scene B) must choose the same own-frame action."""
    oh = [H.FIREBALL, H.HOG, H.CANNON, H.ICE_SPIRIT, H.MUSKETEER, H.ICE_GOLEM, H.SKELETONS, H.LOG]
    og = [H.KNIGHT, H.ARROWS, H.GIANT, H.MINIONS, H.PRINCE, H.BABY_DRAGON, H.WIZARD, H.ZAP]
    ga, gb = mirrored_games("hog26", "giant", 3, hands=(oh, og), towers={(1, 1): 100, (1, 2): 100})
    a, b = bot_pair(ga, gb, "heuristic", 7)
    assert a != 0, "setup: the heuristic must finish a 100-HP tower with its Fireball"
    assert a == b, f"team 0 chose {H.decode_action(a)}, team 1 chose {H.decode_action(b)} (own frame)"


def test_heuristic_weaker_lane_tie_is_seat_symmetric(pr):
    og = [H.GIANT, H.KNIGHT, H.MINIONS, H.BABY_DRAGON, H.PRINCE, H.ARROWS, H.WIZARD, H.ZAP]
    oh = list(H.DECKS["hog26"])
    for s in range(6):
        ga, gb = mirrored_games("giant", "hog26", s, hands=(og, oh), towers={(1, 1): 2000, (1, 2): 2000})
        a, b = bot_pair(ga, gb, "heuristic", 11 + s)
        assert a == b, f"seed {s}: team 0 {H.decode_action(a)} vs team 1 {H.decode_action(b)}"


def _random_scene(seed):
    rng = np.random.default_rng(seed)
    d0, d1 = rng.choice(PRESET_NAMES, 2)
    probe = K.new_game(str(d0), str(d1), seed=seed)
    hands = tuple([int(c) for c in rng.permutation(K.deck_of(probe, t))] for t in (0, 1))
    elixir = tuple(int(rng.integers(0, 28001)) for _ in range(2))
    towers = {(t, i): int(rng.integers(1, [4824, 3052, 3052][i] + 1)) for t in (0, 1) for i in range(3)
              if rng.random() < 0.3}
    units = []
    for _ in range(int(rng.integers(0, 7))):
        t = int(rng.integers(0, 2))
        x, y = int(rng.integers(10, 170)) * 100, int(rng.integers(10, 310)) * 100
        if H.point_in_water(x, y):
            continue
        units.append((t, int(rng.choice(SINGLES)), x, y))
    return mirrored_games(str(d0), str(d1), seed, hands, elixir, towers, units, int(rng.integers(0, 31)))


def test_bots_are_seat_symmetric_on_random_mirrored_scenes(pr):
    plays = 0
    for s in range(40):
        ga, gb = _random_scene(s)
        for team in (0, 1):
            assert np.array_equal(np.asarray(ga.legal_mask(team)), np.asarray(gb.legal_mask(1 - team))), \
                f"setup: scene {s} is not mirrored (legal masks differ)"
            for kind in ("heuristic", "random"):
                a, b = bot_pair(ga, gb, kind, 1000 + s, team)
                assert a == b, f"scene {s}, {kind} bot for team {team}: {H.decode_action(a) if a else 0} " \
                               f"vs mirrored {H.decode_action(b) if b else 0}"
                plays += (a != 0 and kind == "heuristic")
    assert plays >= 5, f"only {plays} heuristic plays across the scenes: too few to test symmetry"


# ==========================================================================================
# 18.5 snapshot validation (subprocess)
# ==========================================================================================
SNAP_FUZZ = textwrap.dedent(r'''
    import json, math, random, sys
    sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
    import numpy as np
    import pufferroyale as pr
    import gamekit as K, helpers as H, cardsv3 as C

    def words(g):
        b = bytes(g.snapshot())
        return b, np.frombuffer(b[:len(b) // 4 * 4], dtype="<i4").copy()

    def put32(b, w, v):
        a = bytearray(b)
        a[4 * w:4 * w + 4] = int(v & 0xFFFFFFFF).to_bytes(4, "little")
        return bytes(a)

    found = {}
    # --- max_hp word: a zapped Knight vs a zapped Golem on the same entity slot ---------------
    def zapped(card):
        g = pr.Game(deck0="giant", deck1="giant", seed=3, deploy_lockout_ticks=0)
        K.spawn(g, 0, card, 9000, 21000)
        K.cast(g, 1, H.ZAP, 9000, 21000); g.tick(1)
        return g
    _, a = words(zapped(H.KNIGHT)); _, b = words(zapped(C.GOLEM))
    found["max_hp"] = [int(i) for i in np.nonzero((a == 1766) & (b == 5120))[0]]
    # --- lifetime word: a Cannon after 5 vs 15 ticks (elapsed or remaining counter) -------------
    def cannon(t):
        g = pr.Game(deck0="giant", deck1="giant", seed=3, deploy_lockout_ticks=0)
        K.spawn(g, 0, H.CANNON, 9000, 21000); g.tick(t)
        return g
    gc = cannon(5)
    _, a = words(gc); _, b = words(cannon(15))
    life = [int(i) for i in np.nonzero(np.abs(b.astype(np.int64) - a) == 10)[0]]
    xi = [i for i in range(len(a) - 1) if a[i] == 9000 and a[i + 1] == 21000]      # the Cannon's record
    found["lifetime"] = [i for i in life if a[i] in (5, 6, 594, 595) and any(x - 8 <= i < x + 48 for x in xi)]
    # --- pending spawns: right after a Golem / Lava Hound death, before the next Spawn phase -----
    def death(card):
        g = pr.Game(deck0="giant", deck1="giant", seed=3, deploy_lockout_ticks=0)
        gid = K.spawn1(g, 0, card, 9000, 14000, deployed=False)
        for i in range(12):
            ang = math.pi * i / 11
            K.spawn(g, 1, H.MUSKETEER, int(9000 + 5000 * math.cos(ang)), int(14000 - 5000 * math.sin(ang)))
        for t in range(200):
            g.tick(1)
            if K.ent(g, gid) is None:
                return g
        raise SystemExit("setup: carrier never died")
    gg, gl = death(C.GOLEM), death(C.LAVA_HOUND)
    bg, sg = words(gg); _, sl = words(gl)
    found["pending_card"] = [int(i) for i in range(len(sg))
                             if ((int(sg[i]) >> 16) & 0xFF) == C.GOLEM and ((int(sl[i]) >> 16) & 0xFF) == C.LAVA_HOUND]
    # tower unit index: the packed word two before the (x, y) pair of a crown tower's record
    tw = [i for i in range(len(sg) - 1) if sg[i] == 3500 and sg[i + 1] == 6500]
    found["tower_packed"] = [int(sg[i - 2]) for i in tw]

    res = {"found": found, "rejected": [], "bad": [], "accepted": 0}

    def attempt(g, base, blob, must_reject, label):
        g.restore(base)
        h = g.hash()
        try:
            g.restore(blob)
        except ValueError:
            if g.hash() != h:
                res["bad"].append(["state changed after a rejected restore", label])
            res["rejected"].append(label)
            return
        except Exception as e:
            res["bad"].append(["non-ValueError exception", label, repr(e)])
            return
        if must_reject:
            res["bad"].append(["invalid snapshot accepted", label])
            return
        try:
            g.tick(50)
            s2 = bytes(g.snapshot()); h2 = g.hash()
            g.restore(s2)
            if g.hash() != h2:
                res["bad"].append(["re-restore changed the state", label])
            g.entities(); g.state()
        except Exception as e:
            res["bad"].append(["accepted state failed after 50 ticks", label, repr(e)])
        res["accepted"] += 1

    base_c, _ = words(gc)
    for w in found["lifetime"]:
        for v in (10**9, 2**31 - 1, 602 if a[w] < 100 else -1):
            attempt(gc, base_c, put32(base_c, w, v), True, f"lifetime@{w}={v}")
    gk = zapped(H.KNIGHT); base_k, _ = words(gk)
    for w in found["max_hp"]:
        for v in (10**9, 2**31 - 1):
            attempt(gk, base_k, put32(base_k, w, v), True, f"max_hp@{w}={v}")
    for w in found["pending_card"]:
        cur = int(sg[w]) & 0xFFFFFFFF
        attempt(gg, bg, put32(bg, w, (cur & ~(0xFF << 16)) | (0xFF << 16)), True, f"pending card -1 @{w}")
        for tp in sorted(set(found["tower_packed"]))[:2]:
            unit = tp & 0xFFFF
            attempt(gg, bg, put32(bg, w, (cur & ~0xFFFF) | unit), True, f"pending tower unit {unit} @{w}")
    rng = random.Random(4321)
    targets = found["max_hp"] + found["lifetime"]
    for g, base, name in ((gk, base_k, "knight"), (gc, base_c, "cannon"), (gg, bg, "pending")):
        for trial in range(150):
            blob = bytearray(base)
            for _ in range(rng.randint(1, 6)):
                blob[rng.randrange(len(blob))] ^= 1 << rng.randrange(8)
            attempt(g, base, bytes(blob), False, f"{name} flip {trial}")
        for trial in range(100):
            attempt(g, base, put32(base, rng.randrange(len(base) // 4), rng.randrange(2**32)), False,
                    f"{name} word {trial}")
        for trial in range(40):
            if targets:
                attempt(g, base, put32(base, rng.choice(targets), rng.randrange(2**31)), False,
                        f"{name} targeted {trial}")
    print(json.dumps(res))
''')


@pytest.mark.slow
def test_snapshot_validation_bounds_multiplied_values_and_pending_spawns(pr):
    code = f"ROOT = {ROOT!r}\nHERE = {HERE!r}\n" + SNAP_FUZZ
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, cwd=ROOT)
    assert out.returncode == 0, f"snapshot fuzz crashed (rc={out.returncode}):\n{out.stderr[-3000:]}"
    res = json.loads(out.stdout.strip().splitlines()[-1])
    f = res["found"]
    assert f["max_hp"] and f["lifetime"] and f["pending_card"] and f["tower_packed"], \
        f"tester could not locate the fields black-box: {f}"
    assert not res["bad"], f"§18.5 validation problems: {res['bad'][:12]}"
    labels = res["rejected"]
    for need in ("lifetime@", "max_hp@", "pending card -1", "pending tower unit"):
        assert any(l.startswith(need) for l in labels), f"no rejection recorded for {need!r}"
    assert res["accepted"] > 50, "the fuzz must also exercise accepted states"


# ==========================================================================================
# 18.6 training precision (bf16 on CPU)
# ==========================================================================================
@pytest.mark.slow
def test_bf16_training_learns_or_is_rejected(pr, tmp_path):
    """MMDPuffeRL with precision='bfloat16' on CPU: either a clear error naming the precision, or
    real learning: parameters change, the trainer's own KL to the initial reference (interval 0)
    becomes > 0, and a fresh forward differs from the initial one. Never silent stale training."""
    import torch
    import pufferroyale
    import pufferroyale.torch as prt
    import pufferroyale.trainer as T
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0)
    torch.manual_seed(0)
    policy = prt.Policy(env)
    init = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    obs, _ = env.reset(seed=0)
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    with torch.no_grad():
        out0 = policy.forward_eval(x, None)[0] if hasattr(policy, "forward_eval") else policy(x)[0]
    cfg = LK.ppo_config(tmp_path, batch_size=64, horizon=16, minibatch=32, total=64 * 20, lr=3e-3,
                        precision="bfloat16", mmd_coef=0.5, mmd_ref_interval=0)
    try:
        tr = T.MMDPuffeRL(cfg, env, policy)
        kls = []
        for _ in range(4):
            tr.last_log_time = -1e18
            tr.evaluate()
            tr.train()
            kls.append(float(tr.losses.get("mmd_kl", float("nan"))))
    except Exception as e:                                             # rejected: must be clear
        msg = f"{type(e).__name__}: {e}"
        assert any(w in msg.lower() for w in ("precision", "bfloat16", "bf16", "float32")), \
            f"bf16 was rejected with an unclear error: {msg}"
        env.close()
        return
    changed = any(not torch.equal(init[k], v) for k, v in policy.state_dict().items())
    assert changed, "bf16 training did not change the parameters"
    assert kls[-1] > 0 and math.isfinite(kls[-1]), f"KL(pi || pi_init) stayed {kls}: stale weights in training"
    with torch.no_grad():
        out1 = policy.forward_eval(x, None)[0] if hasattr(policy, "forward_eval") else policy(x)[0]
    assert not torch.allclose(out0.float(), out1.float()), "the live forward never changed"
    tr.close() if hasattr(tr, "close") else None


# ==========================================================================================
# 18.7 league run robustness (subprocess runs of league_train.py)
# ==========================================================================================
BASE = ["--device", "cpu", "--num-envs", "4", "--anchors", "bot:noop", "--snapshot-interval", "1",
        "--seed", "1", "--deck0", "hog26", "--deck1", "giant", "--bptt-horizon", "16", "--batch-size", "256",
        "--minibatch-size", "256"]


def league(data_dir, run_id, total, *extra, timeout=900):
    return run(["scripts/league_train.py", "--total-timesteps", str(total), "--data-dir", str(data_dir),
                "--run-id", run_id, *BASE, *extra], timeout=timeout)


def resume(run_dir, total, *extra, timeout=900):
    return run(["scripts/league_train.py", "--resume", str(run_dir), "--total-timesteps", str(total),
                "--device", "cpu", "--num-envs", "4", "--bptt-horizon", "16", "--batch-size", "256",
                "--minibatch-size", "256", *extra], timeout=timeout)


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    data = tmp_path_factory.mktemp("league41")
    out = league(data, "base", 1024, "--learning-rate", "3e-3")
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    return os.path.join(str(data), "league", "base")


def copy_run(base_run, dst):
    shutil.copytree(base_run, dst)
    return str(dst)


def finite_tensors(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    bad = []

    def walk(o, pre):
        if isinstance(o, torch.Tensor):
            if o.is_floating_point() and not torch.isfinite(o).all():
                bad.append(pre)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{pre}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{pre}[{i}]")
    walk(obj, os.path.basename(path))
    return bad


@pytest.mark.slow
def test_league_never_saves_a_nonfinite_learner(pr, tmp_path):
    out = league(tmp_path, "nan", 2048, "--learning-rate", "1e12")
    assert out.returncode != 0, "a run whose losses/weights become non-finite must exit non-zero"
    run_dir = tmp_path / "league" / "nan"
    pts = sorted(glob.glob(str(run_dir / "*.pt")))
    bad = [b for p in pts for b in finite_tensors(p)]
    assert not bad, f"non-finite tensors saved: {bad[:6]}"


@pytest.mark.slow
def test_league_resume_of_corrupted_run_fails_promptly(pr, base_run, tmp_path):
    d = copy_run(base_run, tmp_path / "corrupt")
    lp = os.path.join(d, "learner.pt")
    with open(lp, "r+b") as f:
        f.truncate(os.path.getsize(lp) // 2)
    try:
        out = resume(d, 2048, timeout=60)
    except subprocess.TimeoutExpired:
        raise AssertionError("--resume of a corrupted run hung for > 60 s")
    assert out.returncode != 0, "--resume of a truncated learner.pt must exit non-zero"


def _saved_lrs(run_dir):
    import torch
    lrs = []
    ts = torch.load(os.path.join(run_dir, "trainer_state.pt"), map_location="cpu", weights_only=False)

    def walk(o):
        if isinstance(o, dict):
            if "param_groups" in o:
                lrs.extend(float(gr["lr"]) for gr in o["param_groups"])
            for v in o.values():
                walk(v)
    walk(ts)
    return lrs


@pytest.mark.slow
def test_league_resume_reapplies_cli_learning_rate(pr, base_run, tmp_path):
    d = copy_run(base_run, tmp_path / "lr")
    out = resume(d, 1536, "--learning-rate", "1e-4")
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    lrs = _saved_lrs(d)
    assert lrs and all(abs(v - 1e-4) < 1e-12 for v in lrs), f"optimizer lr after resume: {lrs} (want 1e-4)"
    st = json.load(open(os.path.join(d, "league_state.json")))
    if isinstance(st.get("args"), dict) and "learning_rate" in st["args"]:
        assert abs(float(st["args"]["learning_rate"]) - 1e-4) < 1e-12, st["args"]["learning_rate"]


@pytest.mark.slow
def test_league_enabling_mmd_at_resume_starts_from_the_learner(pr, base_run, tmp_path):
    """One minibatch per epoch: the first resumed update's KL(pi || pi_ref) is ~0 because the
    reference IS the loaded learner (not a fresh initialisation)."""
    d = copy_run(base_run, tmp_path / "mmd")
    epochs_before = [json.loads(l)["epoch"] for l in open(os.path.join(d, "history.jsonl")) if l.strip()]
    out = resume(d, 1280, "--mmd-coef", "0.5")
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    recs = [json.loads(l) for l in open(os.path.join(d, "history.jsonl")) if l.strip()]
    new = [r for r in recs if r["epoch"] > max(epochs_before)]
    assert new, "the resumed run must train at least one epoch"
    kl = new[0].get("losses", {}).get("mmd_kl")
    assert kl is not None, f"history.jsonl lacks losses.mmd_kl: {sorted(new[0].get('losses', {}))}"
    assert abs(float(kl)) < 1e-6, f"first resumed mmd_kl = {kl}: the reference must be the loaded learner"


@pytest.mark.slow
def test_league_run_directory_is_relocatable(pr, tmp_path):
    out = league(tmp_path / "a", "mv", 768)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    src = tmp_path / "a" / "league" / "mv"
    dst = tmp_path / "b" / "moved"
    shutil.copytree(src, dst)
    shutil.rmtree(tmp_path / "a")
    out = resume(dst, 1280)
    assert out.returncode == 0, f"a moved run must resume:\n{(out.stdout + out.stderr)[-3000:]}"
    st = json.load(open(dst / "league_state.json"))
    assert st.get("epoch", 0) >= 5 or len(glob.glob(str(dst / "snap_*.pt"))) >= 5
    assert strict_json(last_line(out.stdout)) is not None


# ==========================================================================================
# Low-severity items
# ==========================================================================================
@pytest.mark.slow
def test_total_timesteps_below_one_batch_is_rejected_up_front(pr, tmp_path):
    out = league(tmp_path, "tiny", 10, timeout=120)
    assert out.returncode != 0, "league_train.py: --total-timesteps below one batch (256) must error"
    assert not glob.glob(str(tmp_path / "league" / "tiny" / "snap_*.pt"))
    out = run(["scripts/best_response.py", "--target", LK.make_ckpt(tmp_path / "t.pt"), "--total-timesteps", "10",
               "--matches", "2", "--device", "cpu", "--num-envs", "2", "--bptt-horizon", "16"], timeout=120)
    assert out.returncode != 0, "best_response.py: --total-timesteps below one batch must error"


@pytest.mark.slow
def test_matches_zero_is_rejected(pr, tmp_path):
    ck = LK.make_ckpt(tmp_path / "t.pt")
    cmds = [["scripts/best_response.py", "--target", ck, "--total-timesteps", "64", "--matches", "0",
             "--device", "cpu", "--num-envs", "2", "--bptt-horizon", "16"],
            ["scripts/tournament.py", "--agents", "bot:noop", "bot:random", "--decks", "hog26", "--matches", "0",
             "--out", str(tmp_path / "t.json")],
            ["scripts/llm_match.py", "--model", "mock_wait", "--opponent", "bot:noop", "--deck-agent", "hog26",
             "--deck-opp", "giant", "--matches", "0", "--out", str(tmp_path / "l.json")]]
    for c in cmds:
        out = run(c, timeout=300)
        assert out.returncode != 0, f"{c[0]} --matches 0 must error"


@pytest.mark.slow
@pytest.mark.parametrize("target_kind", ["ckpt", "run_dir"])
def test_best_response_accepts_card_lists_and_run_dir_targets(pr, base_run, tmp_path, target_kind):
    target = f"ckpt:{LK.make_ckpt(tmp_path / 't.pt')}" if target_kind == "ckpt" else base_run
    deck = "Knight,Archers,Musketeer,Giant,Hog Rider,Minions,Baby Dragon,Valkyrie"
    out = run(["scripts/best_response.py", "--target", target, "--total-timesteps", "64", "--matches", "2",
               "--device", "cpu", "--num-envs", "2", "--bptt-horizon", "16", "--deck0", deck, "--deck1", "hog26",
               "--seed", "1", "--out", str(tmp_path / "br.json")], timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    res = strict_json(last_line(out.stdout))
    assert 0.0 <= res["br_score"] <= 1.0
    strict_json(open(tmp_path / "br.json").read())


@pytest.mark.slow
def test_script_json_outputs_are_strict_json(pr, base_run, tmp_path):
    out = run(["scripts/tournament.py", "--agents", "bot:noop", "bot:noop", "--decks", "hog26", "--matches", "2",
               "--out", str(tmp_path / "t.json")], timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    strict_json(open(tmp_path / "t.json").read())                  # tournament: the --out file (§15.4)
    out = run(["scripts/llm_match.py", "--model", "mock_wait", "--opponent", "bot:noop", "--deck-agent", "hog26",
               "--deck-opp", "giant", "--matches", "1", "--decision-interval", "100",
               "--out", str(tmp_path / "l.json")], timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    strict_json(last_line(out.stdout))
    out = resume(base_run, 1024)                                   # already at target: summary only
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    strict_json(last_line(out.stdout))


def test_recurrent_wrapper_keeps_the_policy_init_scale(pr):
    import torch
    import pufferroyale
    import pufferroyale.torch as prt
    env = pufferroyale.Royale(num_envs=1, num_agents=2)

    def actor_std(module):
        ws = [v for k, v in module.state_dict().items()
              if k.endswith("weight") and v.ndim == 2 and v.shape[0] == H.N_ACTIONS]
        assert len(ws) == 1, "tester cannot locate the actor head"
        return float(ws[0].float().std())
    ratios = []
    for seed in range(3):
        torch.manual_seed(seed)
        s_plain = actor_std(prt.Policy(env))
        torch.manual_seed(seed)
        s_rec = actor_std(prt.Recurrent(env, prt.Policy(env)))
        ratios.append(s_rec / s_plain)
    env.close()
    assert all(0.8 <= r <= 1.25 for r in ratios), f"Recurrent(Policy) actor init std / Policy's: {ratios}"
