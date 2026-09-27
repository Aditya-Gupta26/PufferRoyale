"""Builder tests: determinism, snapshot/restore, cross-process hashes and an invariant soak
through the Python API (SPEC §1, §11)."""
import os
import random
import subprocess
import sys

import numpy as np

from pufferroyale import Game, PlayError, decode_action

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def random_match(seed, bot_seed, p=0.12, decks=("random", "random"), check=None, every=1):
    rng = random.Random(bot_seed)
    g = Game(deck0=decks[0], deck1=decks[1], seed=seed)
    hashes = []
    while not g.state()["over"]:
        for team in (0, 1):
            if rng.random() < p:
                mask = g.legal_mask(team)
                legal = np.flatnonzero(mask[1:]) + 1
                if len(legal):
                    slot, tx, ty = decode_action(int(legal[rng.randrange(len(legal))]))
                    assert g.play_tile(team, slot, tx, ty) == PlayError.OK
        g.tick(every)
        hashes.append(g.hash())
        if check:
            check(g)
    return g, hashes


def invariants(g):
    s = g.state()
    assert 0 <= s["elixir"][0] <= 28000 and 0 <= s["elixir"][1] <= 28000
    for team in (0, 1):
        assert sorted(s["hand"][team] + s["queue"][team]) == sorted(s["deck"][team])
    for e in g.entities():
        assert 0 < e["hp"] <= e["max_hp"], e
        assert 0 <= e["x"] < 18000 and 0 <= e["y"] < 32000, e
        # jumpers over the river and a burrowing Miner (hidden, underground) may cross water
        if e["kind"] == "troop" and not e["flying"] and not e["jumping"] and not e["burrowing"]:
            wet = 15000 <= e["y"] < 17000 and not (2500 <= e["x"] < 4500 or 13500 <= e["x"] < 15500)
            assert not wet, e


def test_same_seed_same_actions_same_hashes():
    _, h1 = random_match(7, 70, every=5)
    _, h2 = random_match(7, 70, every=5)
    assert h1 == h2
    _, h3 = random_match(8, 70, every=5)
    assert h3 != h1


def test_snapshot_restore_reproduces_future():
    rng = random.Random(3)
    g = Game(deck0="random", deck1="random", seed=21)
    plays = []
    for t in range(1500):
        for team in (0, 1):
            if rng.random() < 0.08:
                mask = g.legal_mask(team)
                legal = np.flatnonzero(mask[1:]) + 1
                if len(legal):
                    a = int(legal[rng.randrange(len(legal))])
                    plays.append((t, team, a))
        g.tick()
    # replay: snapshot at 600, record, restore (into a different Game object), replay
    def run(game, start, stop, trail):
        for t in range(start, stop):
            for (pt, team, a) in plays:
                if pt == t:
                    slot, tx, ty = decode_action(a)
                    game.play_tile(team, slot, tx, ty)
            game.tick()
            trail.append(game.hash())
    a = Game(deck0="random", deck1="random", seed=21)
    ta = []
    run(a, 0, 600, ta)
    snap = a.snapshot()
    tb = []
    run(a, 600, 1500, tb)
    other = Game(deck0="hog26", deck1="bait", seed=999)
    other.tick(123)
    other.restore(snap)
    tc = []
    run(other, 600, 1500, tc)
    assert tb == tc


def test_hash_is_stable_across_processes():
    code = (
        "import random, numpy as np\n"
        "from pufferroyale import Game, decode_action\n"
        "g = Game(deck0='bait', deck1='giant', seed=12)\n"
        "r = random.Random(4)\n"
        "for t in range(2000):\n"
        "    if t % 17 == 0:\n"
        "        for team in (0, 1):\n"
        "            m = g.legal_mask(team); l = np.flatnonzero(m[1:]) + 1\n"
        "            if len(l):\n"
        "                s, x, y = decode_action(int(l[r.randrange(len(l))])); g.play_tile(team, s, x, y)\n"
        "    g.tick()\n"
        "print(g.hash())\n"
    )
    outs = set()
    for _ in range(2):
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONHASHSEED=str(random.randrange(10000)))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env,
                           check=True)
        outs.add(r.stdout.strip().splitlines()[-1])
    assert len(outs) == 1


def test_soak_invariants_and_valid_results():
    reasons = {}
    for m in range(8):
        g, _ = random_match(100 + m, 200 + m, p=0.05 + 0.03 * (m % 4), check=invariants, every=2)
        s = g.state()
        assert s["over"] and s["tick"] <= 6000
        assert s["end_reason"] in {"KING", "REGULATION_CROWNS", "OVERTIME_CROWNS", "TIEBREAK", "DRAW"}
        assert s["result"][0] + s["result"][1] == 0
        if s["end_reason"] == "DRAW":
            assert s["result"] == [0, 0]
        else:
            assert sorted(s["result"]) == [-1, 1]
        reasons[s["end_reason"]] = reasons.get(s["end_reason"], 0) + 1
    assert sum(reasons.values()) == 8


def test_noop_match_is_a_tiebreak_draw_at_6000():
    g = Game(seed=1)
    g.tick(10000)
    s = g.state()
    assert s["over"] and s["tick"] == 6000 and s["end_reason"] == "DRAW" and s["result"] == [0, 0]
    assert s["overtime"] is True
