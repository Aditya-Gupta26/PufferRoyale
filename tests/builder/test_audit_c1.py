"""Builder tests for the Phase C.1 audit fixes (SPEC v0.2.1 §14)."""
import configparser
import os
import random
import sys

import numpy as np
import pytest

import pufferroyale
from pufferroyale import Game, PlayError, action_index
from pufferroyale import royale as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _snapshots():
    g = Game(deck0="bait", deck1="giant", seed=21, deploy_lockout_ticks=0)
    out = [g.snapshot()]
    rng = np.random.default_rng(0)
    for step in range(120):
        for team in (0, 1):
            m = g.legal_mask(team)
            legal = np.flatnonzero(m[1:]) + 1
            if len(legal) and rng.random() < 0.4:
                g.play_action(team, int(rng.choice(legal)))
        if step % 30 == 7:
            g.spawn(0, "The Log", 9000, 20000)
            g.spawn(1, "Arrows", 9000, 20000)
            out.append(g.snapshot())  # plays queued, effects and projectiles in flight
        g.tick(10)
    out.append(g.snapshot())
    return out


def _exercise(g):
    for _ in range(10):
        for team in (0, 1):
            g.obs(team)
            g.legal_mask(team)
            pufferroyale.bot_action(g, team, "heuristic")
        g.entities(), g.projectiles(), g.effects(), g.state(), g.ansi(), g.hash()
        g.tick(3)


def test_restore_rejects_corrupt_snapshots_and_keeps_state():
    """SPEC §14.5: random byte flips raise ValueError (game unchanged) or restore a state that
    simulates without crashing (the C fuzz test repeats this under ASan)."""
    snaps = _snapshots()
    g = Game(seed=3)
    rnd = random.Random(7)
    accepted = rejected = 0
    for it in range(1500):
        base = bytearray(snaps[it % len(snaps)])
        for _ in range(rnd.randint(1, 4)):
            pos = rnd.randrange(len(base)) if rnd.random() < 0.3 else rnd.randrange(2500)  # header + live records
            base[pos] ^= 1 << rnd.randrange(8)
        before = g.hash()
        try:
            g.restore(bytes(base))
        except ValueError as e:
            assert "invalid snapshot" in str(e)
            assert g.hash() == before  # unchanged
            rejected += 1
            continue
        accepted += 1
        _exercise(g)
    assert accepted > 20 and rejected > 200
    # structural errors named in SPEC §14.5
    good = bytearray(snaps[1])
    import struct
    g.restore(bytes(good))
    for field_off, value, msg in [
        (4 * 4, 7000, "tick"),                                  # tick outside [0, 6000]
    ]:
        bad = bytearray(good)
        struct.pack_into("<i", bad, field_off, value)
        with pytest.raises(ValueError, match=msg):
            g.restore(bytes(bad))
    with pytest.raises(ValueError):
        g.restore(b"\0" * len(good))


def test_pocket_cross_team_same_tick_both_applied():
    """SPEC §14.1 (the auditor's pocket_drop.py): mask-legal plays of both teams on one tile in
    the same tick are both applied; dropped_plays stays 0."""
    g = Game(deck0="hog26", deck1="giant", seed=0, deploy_lockout_ticks=0)
    g.set_hand(0, ["Cannon", "Hog Rider", "Musketeer", "Ice Golem", "Ice Spirit", "Skeletons", "Fireball", "The Log"])
    g.set_hand(1, ["Knight", "Giant", "Prince", "Baby Dragon", "Wizard", "Minions", "Arrows", "Zap"])
    g.set_tower_hp(0, 1, 0)
    g.set_elixir(0, 28000)
    g.set_elixir(1, 28000)
    a0, a1 = action_index(0, 6, 18), action_index(0, 17 - 6, 31 - 18)
    assert g.legal_mask(0)[a0] == 1 and g.legal_mask(1)[a1] == 1
    assert g.play_action(0, a0) == PlayError.OK and g.play_action(1, a1) == PlayError.OK
    g.tick(1)
    s = g.state()
    assert s["dropped_plays"] == 0 and s["plays"] == [1, 1]
    units = sorted((e["unit"], e["team"]) for e in g.entities() if e["kind"] != "tower")
    assert units == [("Cannon", 0), ("Knight", 1)]


def test_env_logs_learner_keys_and_no_dropped_plays():
    """SPEC §14.6: learner_score / learner_return / learner_win / dropped_plays are logged; in
    env play every mask-legal action is applied, so dropped_plays is 0."""
    env = pufferroyale.Royale(num_envs=4, num_agents=1, opponent="heuristic", learner_side="random",
                              frame_skip=10, seed=5, log_interval=1)
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    logs = []
    for _ in range(1400):
        acts = np.zeros(env.num_agents, np.int32)
        for r in range(env.num_agents):
            legal = np.flatnonzero(R.mask(obs[r]))
            if rng.random() < 0.3:
                acts[r] = rng.choice(legal)
        obs, rew, term, trunc, info = env.step(acts)
        logs.extend(info)
    env.close()
    assert logs, "no episode finished"
    for d in logs:
        for k in ("learner_score", "learner_return", "learner_win", "dropped_plays", "tiebreak"):
            assert k in d
        assert d["dropped_plays"] == 0 and d["illegal_actions"] == 0
        assert 0.0 <= d["learner_win"] <= d["learner_score"] <= 1.0


def test_sweep_metric_is_learner_score():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, "pufferroyale", "config", "royale.ini"))
    assert cfg["sweep"]["metric"] == "learner_score"


def _rollout_obs(num_envs=2, steps=24):
    env = pufferroyale.Royale(num_envs=num_envs, num_agents=2, seed=1)
    obs, _ = env.reset(seed=1)
    frames = []
    rng = np.random.default_rng(1)
    for _ in range(steps):
        frames.append(obs.copy())
        acts = np.array([rng.choice(np.flatnonzero(R.mask(o))) if rng.random() < 0.3 else 0 for o in obs], np.int32)
        obs, *_ = env.step(acts)
    env.close()
    return np.stack(frames, 1)  # (agents, horizon, OBS) = PuffeRL's (segments, bptt, ...) layout


def test_policy_mask_comes_from_each_call_on_the_train_path():
    """The mask is sliced from the observations of every call, including the LSTM training
    path where observations are (segments, horizon, OBS); no state carries over between
    calls, and a mismatch raises instead of skipping the mask."""
    import torch
    import pufferroyale.torch as prt

    seq = torch.as_tensor(_rollout_obs())            # (4, 24, OBS)
    env = pufferroyale.Royale(num_envs=2, num_agents=2, seed=1)
    rec = prt.Recurrent(env, prt.Policy(env))
    ff = prt.Policy(env)
    assert not hasattr(ff, "_mask")
    finfo_min = torch.finfo(torch.float32).min
    with torch.no_grad():
        state = {"lstm_h": None, "lstm_c": None}
        logits, values = rec.forward(seq, state)     # PuffeRL train path
        mask = seq.reshape(-1, R.OBS_SIZE)[:, R.MASK_OFFSET:R.MASK_OFFSET + R.MASK_SIZE] > 0.5
        assert logits.shape == mask.shape and values.shape == seq.shape[:2]
        assert bool((logits[~mask] == finfo_min).all())
        assert bool((torch.softmax(logits, -1)[~mask] == 0).all())
        # an eval call on a different batch in between must not leak its mask
        rec.forward_eval(seq[:2, 0], {"lstm_h": None, "lstm_c": None})
        logits2, _ = rec.forward(seq, {"lstm_h": None, "lstm_c": None})
        assert torch.equal(logits2 == finfo_min, ~mask)
        # feed-forward policy on the same (segments, horizon, OBS) layout
        lf, _ = ff.forward(seq.reshape(-1, R.OBS_SIZE))
        assert torch.equal(lf == finfo_min, ~mask)
        with pytest.raises(ValueError):
            prt.mask_logits(lf[:-1], seq)
    env.close()


def test_masking_holds_inside_puffer_rl_training():
    """Run real PuffeRL epochs with the LSTM policy and check every forward call's output."""
    import torch
    import pufferlib.pufferl as pufferl
    import pufferlib.vector
    import pufferroyale.torch as prt

    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        args = pufferl.load_config_file(os.path.join(ROOT, "pufferroyale", "config", "royale.ini"))
    finally:
        sys.argv = argv
    tr = args["train"]
    tr.update(device="cpu", total_timesteps=2048, batch_size=1024, minibatch_size=512, max_minibatch_size=512,
              bptt_horizon=16, use_rnn=True, env="pufferroyale", data_dir=os.path.join(ROOT, "build", "pytest_runs"),
              checkpoint_interval=10 ** 6, compile=False)
    vecenv = pufferlib.vector.make(pufferroyale.Royale, env_kwargs=dict(num_envs=16, num_agents=2, seed=3),
                                   backend=pufferlib.vector.PufferEnv)
    policy = prt.Recurrent(vecenv.driver_env, prt.Policy(vecenv.driver_env))
    seen = {"train": 0, "eval": 0}
    finfo_min = torch.finfo(torch.float32).min

    def checked(fn, kind):
        def wrapper(obs, state):
            logits, values = fn(obs, state)
            mask = obs.reshape(-1, R.OBS_SIZE)[:, R.MASK_OFFSET:R.MASK_OFFSET + R.MASK_SIZE] > 0.5
            assert bool((logits.detach()[~mask] == finfo_min).all())
            if kind == "train":
                assert obs.dim() == 3  # (segments, horizon, OBS)
            seen[kind] += 1
            return logits, values
        return wrapper

    policy.forward = checked(policy.forward, "train")
    policy.forward_eval = checked(policy.forward_eval, "eval")
    trainer = pufferl.PuffeRL(tr, vecenv, policy)
    trainer.print_dashboard = lambda *a, **k: None
    try:
        while trainer.global_step < tr["total_timesteps"]:
            trainer.evaluate()
            trainer.train()
    finally:
        trainer.utilization.stop()
        vecenv.close()
    assert seen["train"] > 0 and seen["eval"] > 0


def test_eval_counts_full_matches_evenly_per_env():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        import eval as ev
    finally:
        sys.path.pop(0)
    w, d, l = ev.play(None, False, "noop", 0, 5, "hog26", "hog26", "cpu", False, 0, bot_policy="heuristic",
                      num_envs=2)
    assert w + d + l == 5 and w == 5
