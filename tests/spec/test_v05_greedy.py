"""SPEC §19.11 (v0.5-G.5) sampling by default; card-first greedy.

1. Greedy = card first, then tile: with p = softmax(logits), P(wait) = p_0, P(s) = sum_j p_{1+sB+j};
   c* = argmax over (wait, slot 0..3) of these marginals, ties -> earliest; wait -> 0, slot ->
   1 + c*B + argmax_j logits[1 + c*B + j], ties -> smallest j. The single greedy rule everywhere:
   pufferroyale.league.greedy_actions(logits) -> int32 array, used by select_actions(greedy=True) and
   every --greedy path; an illegal action is never chosen.
2. Sampling is the default (watch.py, tournament.py [--sample a no-op], metagame.play_match /
   deck_metagame greedy=False, LLM-match policy opponents); sampling stays seeded (deterministic).
3. JSON outputs that record the mode keep their `greedy` field (false by default).
"""
import inspect
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import envkit as E
import helpers as H
import leaguekit as L
import v05kit as V

MINF = np.finfo(np.float32).min


def lg():
    import pufferroyale.league as mod
    return mod


def ref_greedy(logits):
    """Tester reference of §19.11.1 (float64 softmax; np.argmax returns the first maximum)."""
    z = np.asarray(logits, np.float64)
    lead, A = z.shape[:-1], z.shape[-1]
    B = (A - 1) // 4
    z2 = z.reshape(-1, A)
    out = np.zeros(len(z2), np.int64)
    for i, row in enumerate(z2):
        legal = row > MINF
        if not legal.any():
            out[i] = 0
            continue
        r = np.where(legal, row, -np.inf)
        p = np.exp(r - r.max())
        p /= p.sum()
        marg = np.concatenate([[p[0]], p[1:].reshape(4, B).sum(axis=1)])
        c = int(np.argmax(marg))
        if c == 0:
            out[i] = 0
        else:
            seg = row[1 + (c - 1) * B:1 + c * B]
            out[i] = 1 + (c - 1) * B + int(np.argmax(seg))
    return out.reshape(lead)


def blank(A, n=1):
    return np.full((n, A), MINF, np.float32)


def G(x):
    return np.asarray(lg().greedy_actions(x))


# ==========================================================================================
# greedy_actions on hand-built logits
# ==========================================================================================
@pytest.mark.parametrize("A", [2305, 577, 161])
def test_spread_slot_beats_wait(pr, A):
    """Plain argmax waits, card-first plays: P(slot 2) = B e^-2 / (1 + B e^-2) > P(wait)."""
    B = (A - 1) // 4
    x = blank(A)
    x[0, 0] = 0.0
    x[0, 1 + 2 * B:1 + 3 * B] = -2.0
    x[0, 1 + 2 * B + 7] = -1.9                                     # best tile of slot 2
    assert int(np.argmax(x[0])) == 0, "setup: the plain argmax waits"
    got = G(x)
    assert got.dtype == np.int32 and got.shape == (1,)
    assert int(got[0]) == 1 + 2 * B + 7


@pytest.mark.parametrize("A", [2305, 577, 161])
def test_wait_marginal_largest_waits(pr, A):
    B = (A - 1) // 4
    x = blank(A)
    x[0, 0] = 8.0
    x[0, 1:1 + B] = 0.0                                             # slot 0: B tiles at 0 -> B < e^8
    assert int(G(x)[0]) == 0


@pytest.mark.parametrize("A", [2305, 577, 161])
def test_ties(pr, A):
    """Marginal ties -> the earliest of (wait, slot 0..3); tile ties -> the smallest j."""
    B = (A - 1) // 4
    x = blank(A, 4)
    x[0, 0] = 0.0                                                   # wait vs slot 0 with one tile: tie -> wait
    x[0, 1 + 5] = 0.0
    x[1, 0] = -50.0                                                 # slot 1 vs slot 2, equal two-tile sums
    x[1, 1 + B + 3] = 0.25
    x[1, 1 + B + 9] = -0.5
    x[1, 1 + 2 * B + 3] = -0.5
    x[1, 1 + 2 * B + 9] = 0.25
    x[2, 0] = -50.0                                                 # tile tie inside slot 3 -> smallest j
    x[2, 1 + 3 * B + 11] = 1.0
    x[2, 1 + 3 * B + 4] = 1.0
    x[2, 1 + 3 * B + 20 % B] = 0.5
    x[3, 0] = -50.0                                                 # slot 3 wins on mass, not on its max tile
    x[3, 1 + 0 * B + 0] = 1.0
    x[3, 1 + 3 * B:1 + 3 * B + min(B, 30)] = 0.5
    got = G(x)
    assert int(got[0]) == 0, "P(wait) == P(slot 0): ties go to wait"
    assert int(got[1]) == 1 + B + 3, "P(slot 1) == P(slot 2): ties go to slot 1, then its best tile"
    assert int(got[2]) == 1 + 3 * B + 4, "tile ties go to the smallest j"
    assert int(got[3]) == 1 + 3 * B, "the slot with the largest total probability, then its best tile"
    assert np.array_equal(got, ref_greedy(x))


@pytest.mark.parametrize("A", [2305, 577, 161])
def test_wait_only_and_fully_masked_rows(pr, A):
    x = blank(A, 2)
    x[0, 0] = -3.0                                                  # wait-only row
    got = G(x)
    assert int(got[0]) == 0
    assert int(got[1]) == 0, "a row without any legal entry gives the no-op (an illegal action is never chosen)"


@pytest.mark.parametrize("A", [2305, 577, 161])
def test_random_masked_logits_match_reference(pr, A):
    import torch
    rng = np.random.default_rng(A)
    n = 300
    B = (A - 1) // 4
    x = rng.normal(0, rng.uniform(0.1, 4.0, size=(n, 1)), size=(n, A)).astype(np.float32)
    mask = rng.random((n, A)) < rng.uniform(0.02, 0.9, size=(n, 1))
    slot_off = rng.random((n, 4)) < 0.3
    mask[:, 1:] &= ~np.repeat(slot_off, B, axis=1)
    mask[:, 0] = True
    mask[: n // 10, 1:] = False                                     # wait-only rows
    x = np.where(mask, x, MINF).astype(np.float32)
    want = ref_greedy(x)
    got = G(x)
    assert np.array_equal(got, want), f"{int((got != want).sum())} rows differ from the §19.11.1 rule"
    assert np.all(x[np.arange(n), got] > MINF), "an illegal action is never chosen"
    assert np.array_equal(np.asarray(lg().greedy_actions(torch.as_tensor(x))).reshape(-1), want), "torch input"
    xb = x[:240].reshape(2, 3, 40, A)
    gb = G(xb)
    assert gb.shape == (2, 3, 40) and np.array_equal(gb.reshape(-1), want[:240]), "batch shapes"


def test_select_actions_greedy_uses_card_first(pr):
    import torch
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=(50, 577)).astype(np.float32)
    x[:, 0] += 2.0
    gen = torch.Generator().manual_seed(0)
    got = np.asarray(lg().select_actions(torch.as_tensor(x), True, gen)).reshape(-1)
    assert np.array_equal(got, ref_greedy(x))


# ==========================================================================================
# Defaults: sampling everywhere
# ==========================================================================================
def test_python_defaults_are_sampling(pr):
    import pufferroyale.llm as llm
    import pufferroyale.metagame as mg
    assert inspect.signature(mg.play_match).parameters["greedy"].default is False
    assert inspect.signature(mg.deck_metagame).parameters["greedy"].default is False
    assert inspect.signature(llm.play_llm_match).parameters["greedy"].default is False


def _ckpt(tmp_path, bias):
    return L.make_ckpt(tmp_path / f"bias{bias}.pt", noop_bias=bias)


def test_play_match_deterministic_and_greedy_rule(pr, tmp_path):
    """Seeded sampling is deterministic; greedy with a +20 wait bias (wait's marginal largest) never
    plays, so the match vs bot:noop is an untouched-tower draw."""
    import pufferroyale.metagame as mg
    ck = _ckpt(tmp_path, 6.0)
    a = [mg.play_match(ck, "bot:random", "hog26", "giant", s) for s in (1, 2)]
    b = [mg.play_match(ck, "bot:random", "hog26", "giant", s) for s in (1, 2)]
    assert a == b, "sampling is seeded: same seeds, same results"
    waiter = _ckpt(tmp_path, 20.0)
    assert mg.play_match(waiter, "bot:noop", "hog26", "hog26", 3, greedy=True) == 0


def _policy_and_obs(ck, steps=40):
    import torch
    import pufferroyale.league as lgm
    pol, _ = lgm.load_policy(ck)
    env = E.make(num_envs=2, num_agents=2, seed=0, deck0="hog26", deck1="hog26")
    obs, _ = env.reset(seed=0)
    rows = []
    for _ in range(steps):
        obs, *_ = E.step(env, [0] * 4)
        rows.append(obs.copy())
    env.close()
    with torch.no_grad():
        logits, _ = pol.forward_eval(torch.as_tensor(np.concatenate(rows)), dict(lstm_h=None, lstm_c=None))
    return logits.float().numpy()


def test_league_opponent_greedy_is_card_first(pr, tmp_path):
    """§19.11.1 replaces the plain argmax for league --opponent-greedy: with a +5 wait bias the plain
    argmax would always wait, but slots whose legal tiles carry more total probability are played
    (checked on the policy's own logits with the tester reference); with +20 wait dominates every
    marginal and the greedy opponent never plays."""
    def plays(ck, greedy):
        pool = L.make_pool(anchors=(f"ckpt:{ck}",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
        lv = L.make_league(pool, num_envs=2, seed=0, opponent_greedy=greedy, frame_skip=100, log_interval=1)
        lv.async_reset(0)
        total = 0
        for _ in range(130):
            *_, infos, _, _ = lv.recv()
            total += sum(i["plays_0"] + i["plays_1"] for i in infos if isinstance(i, dict) and "plays_0" in i)
            lv.send(np.zeros(2, np.int32))
        lv.close()
        return total
    ck5, ck20 = _ckpt(tmp_path, 5.0), _ckpt(tmp_path, 20.0)
    lg5 = _policy_and_obs(ck5)
    assert np.all(np.argmax(lg5, axis=1) == 0), "setup: the plain argmax of the +5 policy always waits"
    assert (ref_greedy(lg5) != 0).any(), "setup: the card-first rule plays on these observations"
    assert np.array_equal(G(lg5), ref_greedy(lg5))
    assert plays(ck5, True) > 0, "opponent_greedy uses the card-first rule (plays the high-mass slot)"
    assert np.all(ref_greedy(_policy_and_obs(ck20)) == 0), "setup: +20 makes wait's marginal the largest"
    assert plays(ck20, True) == 0


# ==========================================================================================
# Tools: watch.py, tournament.py, llm_match.py
# ==========================================================================================
def run(args, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=H.ROOT, capture_output=True, text=True, timeout=timeout)


def last_line(out):
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    assert out.returncode == 0 and lines, (out.stdout + out.stderr)[-2000:]
    return lines[-1]


def test_watch_samples_by_default_and_is_seeded(pr, tmp_path):
    ck = _ckpt(tmp_path, 0.0)
    base = ["scripts/watch.py", "--checkpoint", ck, "--p1", "noop", "--no-clear", "--fps", "0", "--every", "1000",
            "--max-steps", "150"]
    a = last_line(run([*base, "--seed", "3"]))
    b = last_line(run([*base, "--seed", "3"]))
    assert a == b, "seeded sampling: same seed, same match"
    assert re.search(r"sampl", a, re.I) and not re.search(r"greedy", a, re.I), f"default is sampling: {a!r}"
    g = last_line(run([*base, "--seed", "3", "--greedy"]))
    assert re.search(r"greedy", g, re.I), f"--greedy reported: {g!r}"
    w = last_line(run(["scripts/watch.py", "--checkpoint", _ckpt(tmp_path, 20.0), "--p1", "noop", "--no-clear",
                       "--fps", "0", "--every", "1000", "--max-steps", "150", "--seed", "3", "--greedy"]))
    m = re.search(r"(\d+) plays", w)
    assert m and int(m.group(1)) == 0, f"card-first greedy with wait dominating never plays: {w!r}"


def test_tournament_default_sampling_and_flags(pr, tmp_path):
    ck = _ckpt(tmp_path, 4.0)
    outs = {}
    for name, extra in (("default", []), ("default2", []), ("sample", ["--sample"]), ("greedy", ["--greedy"])):
        f = tmp_path / f"{name}.json"
        out = run(["scripts/tournament.py", "--agents", ck, "bot:noop", "--decks", "hog26", "--matches", "2",
                   "--seed", "1", "--quiet", "--out", str(f), *extra])
        assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
        outs[name] = json.loads(f.read_text())
    assert outs["default"]["greedy"] is False, "§19.11.3: greedy field false by default"
    assert outs["sample"]["greedy"] is False and outs["greedy"]["greedy"] is True
    assert outs["default"]["payoff"] == outs["default2"]["payoff"], "seeded sampling is deterministic"
    assert outs["sample"]["payoff"] == outs["default"]["payoff"], "--sample is a no-op"


def test_llm_match_policy_opponent_samples_by_default(pr, tmp_path):
    ck = _ckpt(tmp_path, 20.0)
    base = ["scripts/llm_match.py", "--model", "mock_wait", "--opponent", ck, "--matches", "1", "--seed", "2",
            "--max-ticks", "600"]
    d = json.loads(last_line(run([*base, "--out", str(tmp_path / "a.json")])))
    g = json.loads(last_line(run([*base, "--opponent-greedy", "--out", str(tmp_path / "b.json")])))
    if "greedy" in d or "greedy" in g:                              # §19.11.3: outputs that record the mode
        assert d.get("greedy") is False and g.get("greedy") is True
    import pufferroyale.llm as llm
    agent = llm.LLMAgent(llm.mock_wait)
    r1 = llm.play_llm_match(agent, ck, "hog26", "hog26", seed=4, max_ticks=600)
    r2 = llm.play_llm_match(llm.LLMAgent(llm.mock_wait), ck, "hog26", "hog26", seed=4, max_ticks=600)
    assert r1["result"] == r2["result"] and r1["crowns"] == r2["crowns"], "seeded"
