"""SPEC §11 non-functional requirements: throughput, soak invariants, capacities, training smoke,
build targets (test-c, ASan/UBSan, warning-free), docs, e2e check."""
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

import gamekit as K
import helpers as H

JUMPERS = (H.HOG, H.PRINCE)


def random_play(g, rng, team, p=0.5):
    if rng.random() >= p:
        return
    m = np.asarray(g.legal_mask(team))
    leg = np.nonzero(m[1:])[0] + 1
    if len(leg):
        s, tx, ty = H.decode_action(int(rng.choice(leg)))
        code = g.play_tile(team, s, tx, ty)
        assert code == H.OK, f"legal_mask allowed slot {s} tile {(tx, ty)} but play returned {code}"


# ------------------------------------------------------------------------------------------
# Throughput (Apple M4, one core). Warm-up, best of 3.
# ------------------------------------------------------------------------------------------
@pytest.mark.perf
def test_engine_ticks_per_second(pr):
    rng = np.random.default_rng(0)
    best = 0.0
    for rep in range(4):
        g = K.new_game("bait", "hog26", seed=rep)
        engine_time, ticks = 0.0, 0
        while not K.over(g) and ticks < 6000:
            for t in (0, 1):
                random_play(g, rng, t, 0.6)
            t0 = time.perf_counter()
            g.tick(20)
            engine_time += time.perf_counter() - t0
            ticks += 20
        if rep > 0:                                  # rep 0 is the warm-up
            best = max(best, ticks / engine_time)
    assert best >= 20000, f"engine throughput {best:.0f} ticks/s < 20,000 (SPEC §11)"


@pytest.mark.perf
def test_env_steps_per_second(pr):
    import pufferroyale
    env = pufferroyale.Royale(num_envs=1, num_agents=2, frame_skip=10, seed=0)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    import pufferroyale.royale as R
    acts = np.zeros(2, dtype=np.int32)
    best = 0.0
    for rep in range(4):
        n, t_env = 0, 0.0
        for _ in range(1500):
            for r in range(2):
                acts[r] = 0
                if rng.random() < 0.1:
                    leg = np.nonzero(obs[r, R.MASK_OFFSET + 1:R.MASK_OFFSET + R.MASK_SIZE] > 0.5)[0]
                    if len(leg):
                        acts[r] = int(rng.choice(leg)) + 1
            t0 = time.perf_counter()
            obs, *_ = env.step(acts)
            t_env += time.perf_counter() - t0
            n += 1
        if rep > 0:
            best = max(best, n / t_env)
    env.close()
    assert best >= 2000, f"env throughput {best:.0f} steps/s < 2,000 (SPEC §11)"


# ------------------------------------------------------------------------------------------
# Soak: full random matches, invariants every tick
# ------------------------------------------------------------------------------------------
def check_invariants(g, decks):
    s = K.st(g)
    for t in (0, 1):
        e = K.elixir(g, t)
        assert 0 <= e <= H.MAX_ELIXIR, f"elixir {e}"
        assert sorted(K.hand(g, t) + K.queue(g, t)) == sorted(H.DECKS[decks[t]])
        assert 0 <= K.crowns(g)[t] <= 3
        for i in range(3):
            tw = K.tower(g, t, i)
            assert 0 <= int(tw["hp"]) <= int(tw["max_hp"])
    es = K.ents(g)
    assert len(es) <= 256, "MAX_ENTITIES"
    ids = [int(e["id"]) for e in es]
    assert len(ids) == len(set(ids)), "duplicate entity ids"
    for e in es:
        assert 0 < int(e["hp"]) <= int(e["max_hp"]), f"hp {e['hp']}/{e['max_hp']}"
        x, y = int(e["x"]), int(e["y"])
        assert 0 <= x < H.ARENA_W and 0 <= y < H.ARENA_H, f"entity outside the arena at {(x, y)}"
        if e["kind"] == "troop" and not bool(e["flying"]) and int(e["card_id"]) not in JUMPERS:
            assert not H.point_in_water(x, y), f"ground troop {e['unit']} in water at {(x, y)}"
    return s


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(16))
def test_soak_random_matches(pr, seed):
    decks = [("hog26", "giant"), ("giant", "bait"), ("bait", "hog26"), ("bait", "bait")][seed % 4]
    p = [0.35, 0.12, 0.6, 0.05][(seed // 4) % 4]
    g = K.new_game(*decks, seed=seed)
    rng = np.random.default_rng(100 + seed)
    for _ in range(6001):
        if K.over(g):
            break
        if K.tick_of(g) % 5 == 0:
            for t in (0, 1):
                random_play(g, rng, t, p)
        g.tick(1)
        check_invariants(g, decks)
    assert K.over(g), "every match ends by tick 6000"
    assert K.tick_of(g) <= 6000
    r0, r1 = K.result(g, 0), K.result(g, 1)
    assert r0 in (-1, 0, 1) and r0 == -r1
    assert K.end_reason(g) in ("KING", "REGULATION_CROWNS", "OVERTIME_CROWNS", "TIEBREAK", "DRAW")
    c0, c1 = K.crowns(g)
    reason = K.end_reason(g)
    assert (reason == "DRAW") == (r0 == 0), f"SPEC §13.8: DRAW iff drawn (reason {reason}, result {r0})"
    if reason == "KING" and r0 != 0:
        assert (c0 == 3) if r0 == 1 else (c1 == 3)
    if K.end_reason(g) in ("REGULATION_CROWNS", "OVERTIME_CROWNS"):
        assert (c0 > c1) == (r0 == 1)


def test_entity_capacity_overflow_is_safe(pr):
    g = K.new_game()
    for i in range(20):                               # 20 x 15 skeletons > 256 slots
        try:
            g.spawn(0, H.SKARMY, 2000 + 700 * (i % 20), 20000 + 400 * (i % 5), True)
        except Exception as e:                         # must not raise on overflow
            raise AssertionError(f"spawn raised on overflow: {e!r}")
    assert len(K.ents(g)) <= 256
    g.tick(50)
    assert len(K.ents(g)) <= 256


# ------------------------------------------------------------------------------------------
# Training integration (SPEC §11)
# ------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_train_script_cli_smoke(pr, tmp_path):
    """SPEC §13.15: scripts/train.py accepts the pinned flags and runs a short CPU smoke run."""
    import re
    cmd = [sys.executable, "scripts/train.py", "--total-timesteps", "4096", "--device", "cpu",
           "--num-envs", "4", "--num-agents", "2", "--opponent", "random", "--deck0", "hog26",
           "--deck1", "giant", "--seed", "1", "--data-dir", str(tmp_path)]
    out = subprocess.run(cmd, cwd=H.ROOT, capture_output=True, text=True, timeout=900)
    log = out.stdout + out.stderr
    assert out.returncode == 0, log[-3000:]
    assert not re.search(r"\bnan\b", log, re.I), "NaN in the training output"


def test_training_scripts_exist(pr):
    for f in ("scripts/train.py", "scripts/eval.py", "scripts/e2e_check.py"):
        p = os.path.join(H.ROOT, f)
        assert os.path.isfile(p), f"{f} missing (SPEC §11/§12)"
        subprocess.run([sys.executable, "-m", "py_compile", p], check=True)


def test_docs_exist(pr):
    for f in ("README.md", "docs/FIDELITY.md", "docs/DECISIONS.md"):
        p = os.path.join(H.ROOT, f)
        assert os.path.isfile(p) and os.path.getsize(p) > 200, f"{f} missing or empty (SPEC §11)"


def test_binding_importable(pr):
    import importlib
    importlib.import_module("pufferroyale.binding")


# ------------------------------------------------------------------------------------------
# Build targets
# ------------------------------------------------------------------------------------------
def make_target(*args, timeout=900):
    return subprocess.run(["make", *args], cwd=H.ROOT, capture_output=True, text=True, timeout=timeout)


@pytest.mark.slow
def test_make_test_c_warning_free(pr):
    out = make_target("-B", "test-c")
    log = out.stdout + out.stderr
    assert out.returncode == 0, log[-3000:]
    warns = [l for l in log.splitlines() if "warning:" in l]
    assert not warns, "C build must be warning-free with -Wall -Wextra:\n" + "\n".join(warns[:30])


@pytest.mark.slow
def test_make_asan_clean(pr):
    out = make_target("asan", timeout=1800)
    log = out.stdout + out.stderr
    assert out.returncode == 0, log[-3000:]
    for marker in ("ERROR: AddressSanitizer", "runtime error:", "ERROR: LeakSanitizer"):
        assert marker not in log, f"sanitizer report: {marker}"


@pytest.mark.slow
def test_e2e_check_prints_pass(pr):
    out = subprocess.run([sys.executable, "scripts/e2e_check.py"], cwd=H.ROOT, capture_output=True,
                         text=True, timeout=1800)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    assert "PASS" in out.stdout
