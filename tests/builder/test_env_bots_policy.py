"""Builder tests: the PufferLib env (SPEC §9), scripted bots (§8), the policy and the scripts (§11)."""
import os
import subprocess
import sys

import numpy as np
import pytest

import pufferroyale
from pufferroyale import Bot, Game, PlayError, bot_action
from pufferroyale import royale as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SPEC §16.4 / §16.6.16 (v0.3): card identities as integer ids (card_id + 1), 128-wide multi-hots,
# the tower-troop one-hots appended after opp_elixir_ub
SPEC_SCALARS = [("elixir", 1), ("hand", 4), ("hand_cost", 4), ("next_card", 1), ("affordable", 4),
                ("tick", 1), ("overtime", 1), ("elixir_rate", 1), ("lockout", 1), ("own_towers", 3),
                ("enemy_towers", 3), ("king_active", 2), ("crowns", 2), ("opp_seen", 128), ("opp_spent", 1),
                ("opp_last4", 4), ("opp_deduced_hand", 128), ("opp_elixir_ub", 1), ("own_tower_troop", 4),
                ("enemy_tower_troop", 4)]


def ids(v):
    return [int(x) - 1 for x in v]


def test_layout_matches_spec_v03():
    assert R.SPATIAL_SHAPE == (25, 32, 18)
    assert R.ENTITY_SHAPE == (64, 11) and R.CARD_SLOTS == 128
    assert R.ENTITY_FEATURES[0] == "card_id" and len(R.ENTITY_FEATURES) == 11
    assert R.SCALAR_SIZE == 298 and R.MASK_SIZE == 2305
    assert R.OBS_SIZE == 25 * 576 + 64 * 11 + 298 + 2305
    off = 0
    assert list(R.SCALAR_INDEX) == [n for n, _ in SPEC_SCALARS]
    for (name, n), (k, (o, ln)) in zip(SPEC_SCALARS, R.SCALAR_INDEX.items()):
        assert (o, ln) == (off, n), name
        off += n
    assert R.SPATIAL_OFFSET == 0 and R.ENTITY_OFFSET == 14400
    assert R.MASK_OFFSET + R.MASK_SIZE == R.OBS_SIZE


def test_game_obs_mask_equals_legal_mask_and_scalars():
    g = Game(deck0="bait", deck1="giant", seed=3)
    g.tick(150)
    for team in (0, 1):
        o = g.obs(team)
        assert o.dtype == np.float32 and o.shape == (R.OBS_SIZE,)
        assert np.array_equal(R.mask(o).astype(np.uint8), g.legal_mask(team))
        s = g.state()
        assert R.scalar(o, "elixir")[0] == np.float32(s["elixir"][team] / 28000)
        assert R.scalar(o, "tick")[0] == np.float32(150 / 6000)
        assert ids(R.scalar(o, "hand")) == s["hand"][team]              # card_id + 1
        assert ids(R.scalar(o, "next_card")) == [s["queue"][team][0]]
        assert np.array_equal(R.scalar(o, "own_tower_troop"), np.array([1, 0, 0, 0], np.float32))


def test_opponent_deduction_scalars():
    g = Game(deck0="hog26", deck1="giant", seed=5, deploy_lockout_ticks=0)
    rng = np.random.default_rng(0)
    played = []
    while len(set(played)) < 8:           # team 1 plays until all 8 of its cards are revealed
        m = g.legal_mask(1)
        leg = np.flatnonzero(m[1:]) + 1
        s = g.state()
        if len(leg):
            a = int(leg[rng.integers(len(leg))])
            slot = (a - 1) // 576
            card = s["hand"][1][slot]
            assert g.play_action(1, a) == PlayError.OK
            played.append(card)
        g.set_elixir(1, 28000)
        g.tick(1)
        o = g.obs(0)
        seen = set(np.flatnonzero(R.scalar(o, "opp_seen")))
        assert seen == set(played)
        assert abs(R.scalar(o, "opp_spent")[0] - sum(pufferroyale.CARD_COSTS[c] for c in played) / 200) < 1e-6
        deduced = R.scalar(o, "opp_deduced_hand")
        if len(seen) < 8:
            assert deduced.sum() == 0
    s = g.state()
    last4 = ids(R.scalar(g.obs(0), "opp_last4"))
    assert last4 == played[::-1][:4]
    deduced = set(np.flatnonzero(R.scalar(g.obs(0), "opp_deduced_hand")))
    assert deduced == set(s["hand"][1])    # exactly the true (hidden) hand


def test_bots_legal_and_deterministic():
    for kind in ("noop", "random", "heuristic"):
        seqs = []
        for rep in range(2):
            g = Game(deck0="bait", deck1="hog26", seed=11)
            b0, b1 = Bot(kind, seed=1, play_prob=0.5), Bot(kind, seed=2, play_prob=0.5)
            seq = []
            for step in range(400):
                for team, b in ((0, b0), (1, b1)):
                    a = b.act(g, team)
                    if a:
                        assert g.legal_mask(team)[a] == 1
                        assert g.play_action(team, a) == PlayError.OK
                    seq.append(a)
                g.tick(10)
                if g.state()["over"]:
                    break
            seqs.append((seq, g.hash()))
        assert seqs[0] == seqs[1], kind
        plays = sum(1 for a in seqs[0][0] if a)
        assert (plays == 0) if kind == "noop" else (plays > 0), kind


def test_bot_action_helper_persists_bot():
    g = Game(seed=2)
    g.tick(90)
    a = bot_action(g, 0, "heuristic")
    assert a == 0 or g.legal_mask(0)[a] == 1
    assert ("0", "heuristic") not in g._bots and (0, "heuristic") in g._bots


def test_env_alternate_first_flag_and_rotation():
    env = pufferroyale.Royale(num_envs=1, num_agents=2, alternate_first=True, seed=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (2, R.OBS_SIZE)
    env.close()
    # both seats of one env see rotated views of the same unit
    g = Game(seed=0, deploy_lockout_ticks=0)
    g.spawn(0, "Knight", 5500, 20500)
    e0, e1 = R.entities(g.obs(0)), R.entities(g.obs(1))
    assert e0[0, 0] == 1.0 and e1[32, 0] == 1.0                     # Knight = id 0 -> 1
    assert abs(e0[0, 1] + e1[32, 1] - 1) < 1e-6 and abs(e0[0, 2] + e1[32, 2] - 1) < 1e-6


def test_policy_mask_and_shapes():
    import torch
    import pufferlib.pytorch
    import pufferroyale.torch as prt

    env = pufferroyale.Royale(num_envs=2, num_agents=2, seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(12):
        obs, *_ = env.step(np.zeros(4, np.int32))
    pol = prt.Policy(env)
    x = torch.as_tensor(obs)
    with torch.no_grad():
        lg, v = pol.forward_eval(x)
        lg2, v2 = pol.forward(x)
        assert torch.equal(lg, lg2) and v.shape == (4, 1)
        m = x[:, R.MASK_OFFSET:] > 0.5
        for _ in range(100):
            a, lp, ent = pufferlib.pytorch.sample_logits(lg)
            assert bool(m[torch.arange(4), a].all())
            assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
    rec = prt.Recurrent(env, prt.Policy(env))
    st = {"lstm_h": None, "lstm_c": None}
    with torch.no_grad():
        lg, v = rec.forward_eval(x, st)
    assert lg.shape == (4, R.N_ACTIONS) and st["lstm_h"].shape == (4, 256)
    env.close()


def test_config_loads_and_train_cli():
    r = subprocess.run([sys.executable, "scripts/train.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    for flag in ("--total-timesteps", "--device", "--num-envs", "--num-agents", "--opponent", "--deck0", "--deck1",
                 "--seed", "--data-dir"):
        assert flag in r.stdout, flag
    for script in ("eval.py", "bench.py", "watch.py", "e2e_check.py"):
        r = subprocess.run([sys.executable, os.path.join("scripts", script), "--help"], cwd=ROOT, capture_output=True,
                           text=True)
        assert r.returncode == 0, script


def test_ansi_render_both_ways():
    env = pufferroyale.Royale(render_mode="ansi", seed=0)
    env.reset(seed=0)
    txt = env.render()
    assert isinstance(txt, str) and "tick 0/6000" in txt and txt.count("\n") > 32
    env.close()
    g = Game(seed=0)
    assert "tick 0/6000" in g.ansi()
