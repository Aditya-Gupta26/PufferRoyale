"""Builder tests for Phase D (SPEC §15): league wrapper internals, opponent isolation from the PPO
buffer, determinism, recurrent opponents, result attribution under reward shaping, the MMD
trainer on recurrent batches, checkpoint loading, and the metagame helpers."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import pufferroyale
from pufferroyale import binding
from pufferroyale import royale as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------------------------ helpers
def league():
    import pufferroyale.league as L
    return L


def make_ckpt(path, recurrent=False, seed=0, noop_bias=None, **policy_kw):
    import torch
    from pufferroyale.torch import Policy, Recurrent
    env = pufferroyale.Royale(num_envs=1, num_agents=2)
    torch.manual_seed(seed)
    pol = Policy(env, **policy_kw)
    if noop_bias is not None:
        with torch.no_grad():
            pol.actor.bias[0] += noop_bias
    if recurrent:
        pol = Recurrent(env, pol, input_size=pol.hidden_size, hidden_size=pol.hidden_size)
    env.close()
    torch.save(pol.state_dict(), str(path))
    return str(path)


def scal(obs, name):
    return R.scalar(obs, name)


def drive(lv, steps, act=None, seed=0):
    """Run a LeagueVecEnv; returns per-step (obs, rew, term, infos, seats-before, opponents-before)."""
    rng = np.random.default_rng(seed)
    lv.async_reset(seed)
    out = []
    for _ in range(steps):
        obs, rew, term, trunc, infos, ids, masks = lv.recv()
        seats, opps = lv.seats.copy(), list(lv.opponents)
        a = np.zeros(lv.num_agents, np.int32)
        if act == "random":
            for r in range(lv.num_agents):
                leg = np.flatnonzero(obs[r, R.MASK_OFFSET:] > 0.5)
                if len(leg) > 1 and rng.random() < 0.5:
                    a[r] = int(rng.choice(leg[1:]))
        lv.send(a)
        o2, r2, t2, _, i2, _, _ = lv.recv()
        out.append((o2.copy(), r2.copy(), t2.copy(), list(i2), seats, opps))
    return out


# ------------------------------------------------------------------------------------ binding helper
def test_env_log_peek_reads_without_clearing():
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=100, log_interval=10 ** 9)
    env.reset(seed=0)
    for _ in range(61):
        env.step(np.zeros(4, np.int32))
    peeks = [binding.env_log_peek(h) for h in env._c_env_list]
    assert all(p["n"] == 1.0 and p["draw"] == 1.0 for p in peeks), peeks
    assert binding.env_log_peek(env._c_env_list[0]) == peeks[0], "peeking must not clear or change the log"
    log = binding.vec_log(env.c_envs)
    assert log["n"] == 2.0 and log["draw"] == 1.0, "vec_log still sees the peeked episodes"
    assert binding.env_log_peek(env._c_env_list[0])["n"] == 0.0, "vec_log cleared the accumulators"
    env.close()


# ------------------------------------------------------------------------------------ isolation
def test_opponent_rows_never_enter_the_ppo_buffer():
    """Learner forced to no-op vs the heuristic bot: every buffered observation must be the
    learner's own view -- its OWN towers get damaged, the ENEMY towers never do (the enemy is
    the only side that plays). Buffered rewards are therefore never positive."""
    import torch
    import pufferlib.pufferl as pufferl
    from pufferroyale.torch import Policy
    L = league()
    pool = L.OpponentPool(anchors=("bot:heuristic",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.LeagueVecEnv(pool, num_envs=4, seed=0, frame_skip=50, log_interval=10 ** 9)
    torch.manual_seed(0)
    policy = Policy(lv.driver_env)
    with torch.no_grad():
        policy.actor.bias[0] += 200.0                       # always the no-op
    lv.set_policy(policy)
    horizon = 64
    cfg = dict(env="pufferroyale", seed=0, torch_deterministic=True, cpu_offload=False, device="cpu",
               optimizer="adam", precision="float32", total_timesteps=4 * horizon * 2, learning_rate=0.0,
               anneal_lr=False, min_lr_ratio=0.0, gamma=0.99, gae_lambda=0.9, update_epochs=1, clip_coef=0.2,
               vf_coef=1.0, vf_clip_coef=0.2, max_grad_norm=1.0, ent_coef=0.0, adam_beta1=0.9, adam_beta2=0.999,
               adam_eps=1e-8, data_dir=ROOT, checkpoint_interval=10 ** 9, batch_size=4 * horizon,
               minibatch_size=4 * horizon, max_minibatch_size=4 * horizon, bptt_horizon=horizon, compile=False,
               vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.8, prio_beta0=0.2, use_rnn=False)
    Quiet = type("Quiet", (pufferl.PuffeRL,), {"print_dashboard": lambda self, *a, **k: None})
    tr = Quiet(cfg, lv, policy)
    tr.save_checkpoint = lambda: None
    try:
        own_hit = False
        for _ in range(2):
            tr.evaluate()
            obs = tr.observations.reshape(-1, R.OBS_SIZE).numpy()
            assert (tr.actions == 0).all(), "setup: the learner must only no-op"
            assert np.all(scal(obs, "enemy_towers") == 1.0), "an opponent-view row entered the buffer"
            own_hit |= bool(np.any(scal(obs, "own_towers") < 1.0))
            assert (tr.rewards <= 0).all()
            tr.train()
        assert own_hit, "setup: the heuristic bot should have damaged the learner's towers"
    finally:
        tr.utilization.stop()
        lv.close()


def test_learner_row_is_the_seat_row_of_the_underlying_env():
    L = league()
    pool = L.OpponentPool(anchors=("bot:random",), self_play_frac=0.0, anchor_frac=1.0, seed=2)
    lv = L.LeagueVecEnv(pool, num_envs=3, seed=4, frame_skip=100, log_interval=1)
    lv.async_reset(4)
    for _ in range(150):
        obs = lv.recv()[0]
        rows = 2 * np.arange(3) + lv.seats
        assert np.array_equal(obs, lv.env.observations[rows])
        lv.send(np.zeros(3, np.int32))
    lv.close()


# ------------------------------------------------------------------------------------ determinism
def _mixed_run(tmp_path, seed):
    L = league()
    ck = make_ckpt(tmp_path / "opp.pt", seed=1)
    rck = make_ckpt(tmp_path / "ropp.pt", recurrent=True, seed=2, hidden_size=64)
    pool = L.OpponentPool(anchors=("bot:heuristic", "bot:random", f"ckpt:{ck}", f"ckpt:{rck}"),
                          self_play_frac=0.2, anchor_frac=0.8, seed=seed)
    lv = L.LeagueVecEnv(pool, num_envs=4, seed=seed, frame_skip=50, log_interval=1)
    import torch
    from pufferroyale.torch import Policy
    torch.manual_seed(0)
    lv.set_policy(Policy(lv.driver_env))
    recs = drive(lv, 260, act="random", seed=seed)
    res = list(lv.results)
    lv.close()
    return recs, res, pool.state_dict()


def test_league_is_deterministic_given_seeds(tmp_path):
    a, ra, pa = _mixed_run(tmp_path, 5)
    b, rb, pb = _mixed_run(tmp_path, 5)
    assert len(ra) >= 4, "setup: several finished episodes"
    assert {s for s, _, _ in ra} >= {"bot:heuristic"}
    assert ra == rb and pa == pb
    for x, y in zip(a, b):
        assert np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) and np.array_equal(x[4], y[4])
        assert x[5] == y[5]


# ------------------------------------------------------------------------------------ attribution
def test_results_attributed_per_match_under_shaping():
    """With shaped rewards the terminal reward is not +-1, so results must come from the engine
    (env_log_peek). vs bot:noop the learner never loses; with log_interval=1 and one match every
    league log is exactly one episode, checkable against win_0 / win_1 and the seat."""
    L = league()
    pool = L.OpponentPool(anchors=("bot:noop",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.LeagueVecEnv(pool, num_envs=1, seed=9, frame_skip=100, log_interval=1, reward_tower=0.5,
                        reward_crown=0.25)
    recs = drive(lv, 600, act="random", seed=3)
    lv.close()
    shaped, n = False, 0
    ep_ret = 0.0
    for obs, rew, term, infos, seats, opps in recs:
        ep_ret += float(rew[0])
        if not term[0]:
            continue
        lg = [i for i in infos if "learner_score" in i][-1]
        seat = int(seats[0])
        r0 = 1 if lg["win_0"] == 1.0 else (-1 if lg["win_1"] == 1.0 else 0)
        learner = r0 if seat == 0 else -r0
        assert learner >= 0, "the noop bot cannot beat the learner"
        assert lg["learner_score"] == 0.5 * (learner + 1)
        assert abs(lg["learner_return"] - ep_ret) < 1e-4, (lg["learner_return"], ep_ret)
        shaped |= float(rew[0]) not in (-1.0, 0.0, 1.0)
        ep_ret, n = 0.0, n + 1
    st = pool.stats()["bot:noop"]
    assert n >= 3 and st["games"] == n and st["losses"] == 0
    assert shaped, "setup: shaping must make some terminal reward differ from +-1/0"


# ------------------------------------------------------------------------------------ recurrent / pool edge cases
def test_recurrent_ckpt_opponent_state_is_reset_per_episode(tmp_path):
    L = league()
    rck = make_ckpt(tmp_path / "rec.pt", recurrent=True, seed=3, hidden_size=64)
    spec = f"ckpt:{rck}"
    pool = L.OpponentPool(anchors=(spec,), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.LeagueVecEnv(pool, num_envs=2, seed=0, frame_skip=100, log_interval=1)
    lv.async_reset(0)
    checked = 0
    for _ in range(130):
        lv.send(np.zeros(2, np.int32))
        term = lv.recv()[2]
        h, c = lv._lstm[spec]
        assert h.shape == (2, 64)
        for i in np.flatnonzero(term):
            assert float(h[i].abs().sum()) == 0.0 and float(c[i].abs().sum()) == 0.0, "state not reset"
            checked += 1
        live = [i for i in range(2) if not term[i]]
        if live:
            assert float(h[live[0]].abs().sum()) > 0.0
    assert checked >= 2
    lv.close()


def test_evicted_snapshot_mid_match_is_ignored_and_uncached(tmp_path):
    L = league()
    a = make_ckpt(tmp_path / "a.pt", seed=1)
    b = make_ckpt(tmp_path / "b.pt", seed=2)
    pool = L.OpponentPool(anchors=(), max_snapshots=1, self_play_frac=0.0, anchor_frac=0.0, seed=0)
    pool.add_snapshot(a)
    lv = L.LeagueVecEnv(pool, num_envs=2, seed=0, frame_skip=100, log_interval=1)
    lv.async_reset(0)
    lv.send(np.zeros(2, np.int32))                      # loads and caches a
    assert f"ckpt:{a}" in lv._ckpt_cache
    pool.add_snapshot(b)                                # evicts a while both matches still play it
    for _ in range(70):
        lv.send(np.zeros(2, np.int32))
    assert f"ckpt:{a}" not in pool.stats(), "results vs an evicted snapshot are dropped"
    assert f"ckpt:{a}" not in lv._ckpt_cache, "evicted and no longer playing: uncached"
    assert pool.stats()[f"ckpt:{b}"]["games"] >= 0
    lv.close()


def test_pool_zero_eps_falls_back_to_uniform(tmp_path):
    L = league()
    pool = L.OpponentPool(anchors=(), pfsp="variance", pfsp_eps=0.0, self_play_frac=0.0, anchor_frac=0.0)
    pool.add_snapshot("x")
    pool.add_snapshot("y")
    pool.record("ckpt:x", 1)
    pool.record("ckpt:y", -1)                            # p = 1 and p = 0 -> variance weight 0 for both
    w = pool.weights()
    assert w == {"ckpt:x": 0.5, "ckpt:y": 0.5}
    assert pool.sample() in w
    with pytest.raises(ValueError):
        pool.record("self", 2)
    with pytest.raises(ValueError):
        L.OpponentPool(anchors=("self",))


def test_load_policy_infers_non_default_sizes(tmp_path):
    L = league()
    p = make_ckpt(tmp_path / "small.pt", hidden_size=64, cnn_channels=16, entity_hidden=32, scalar_hidden=16)
    pol, rec = L.load_policy(p)
    assert not rec and pol.hidden_size == 64 and pol.actor.in_features == 64
    assert not any(t.requires_grad for t in pol.parameters())
    rp = make_ckpt(tmp_path / "rsmall.pt", recurrent=True, hidden_size=64, cnn_channels=16)
    rpol, rrec = L.load_policy(rp)
    assert rrec and rpol.hidden_size == 64


def test_sampled_opponent_actions_are_legal():
    import torch
    L = league()
    env = pufferroyale.Royale(num_envs=2, num_agents=2, deploy_lockout_ticks=0, seed=1)
    obs, _ = env.reset(seed=1)
    env.close()
    logits = torch.randn(4, R.N_ACTIONS)
    from pufferroyale.torch import mask_logits
    ml = mask_logits(logits, torch.as_tensor(obs))
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        a = L.select_actions(ml, False, g)
        assert all(obs[r, R.MASK_OFFSET + a[r]] == 1.0 for r in range(4))
    assert np.array_equal(L.select_actions(ml, True, g), ml.argmax(-1).numpy())


# ------------------------------------------------------------------------------------ MMD on recurrent batches
def test_mmd_trainer_on_recurrent_league():
    import torch
    from pufferroyale.torch import Policy, Recurrent
    from pufferroyale.trainer import MMDPuffeRL
    L = league()
    pool = L.OpponentPool(anchors=("bot:random",), self_play_frac=0.5, anchor_frac=0.5, seed=0)
    lv = L.LeagueVecEnv(pool, num_envs=4, seed=0, frame_skip=50)
    torch.manual_seed(0)
    policy = Recurrent(lv.driver_env, Policy(lv.driver_env, hidden_size=64), input_size=64, hidden_size=64)
    lv.set_policy(policy)
    horizon = 16
    cfg = dict(env="pufferroyale", seed=0, torch_deterministic=True, cpu_offload=False, device="cpu",
               optimizer="adam", precision="float32", total_timesteps=10 ** 6, learning_rate=1e-3,
               anneal_lr=False, min_lr_ratio=0.0, gamma=0.99, gae_lambda=0.9, update_epochs=1, clip_coef=0.2,
               vf_coef=1.0, vf_clip_coef=0.2, max_grad_norm=1.0, ent_coef=0.0, adam_beta1=0.9, adam_beta2=0.999,
               adam_eps=1e-8, data_dir=ROOT, checkpoint_interval=10 ** 9, batch_size=8 * horizon,
               minibatch_size=8 * horizon, max_minibatch_size=8 * horizon, bptt_horizon=horizon, compile=False,
               vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.8, prio_beta0=0.2, use_rnn=True,
               mmd_coef=0.5, mmd_ref_interval=0)
    Quiet = type("Quiet", (MMDPuffeRL,), {"print_dashboard": lambda self, *a, **k: None})
    tr = Quiet(cfg, lv, policy)
    tr.save_checkpoint = lambda: None
    try:
        kls = []
        for _ in range(3):
            tr.last_log_time = -1e18
            tr.evaluate()
            tr.train()
            kls.append(tr.losses["mmd_kl"])
            assert all(math.isfinite(float(v)) for k, v in tr.losses.items() if k != "explained_variance")
        assert kls[0] < 1e-6 < kls[-1], f"KL starts at 0 (pi == pi_ref) and grows as pi moves: {kls}"
    finally:
        tr.utilization.stop()
        lv.close()


# ------------------------------------------------------------------------------------ metagame helpers
def test_elo_regularised_only_when_ml_diverges():
    from pufferroyale import metagame as mg
    P = np.array([[0.5, 1.0, 1.0], [0.0, 0.5, 0.7], [0.0, 0.3, 0.5]])
    r, info = mg.elo(P, np.full((3, 3), 10), return_info=True)
    assert info["regularised"] and np.all(np.isfinite(r)) and r[0] > r[1] > r[2]
    assert abs(r.mean() - 1500) < 1e-9
    P2 = np.array([[0.5, 0.9, 0.95], [0.1, 0.5, 0.7], [0.05, 0.3, 0.5]])
    _, info2 = mg.elo(P2, np.full((3, 3), 10), return_info=True)
    assert not info2["regularised"]


def test_payoff_counts_and_nash_of_dominated_meta_game():
    from pufferroyale import metagame as mg
    res = [(0, 1, 1), (1, 0, -1), (0, 2, 1), (2, 1, 0), (1, 2, 1)]
    P, G = mg.payoff_matrix(res, 3, return_counts=True)
    assert G[0, 1] == G[1, 0] == 2 and G[1, 2] == 2 and G[0, 0] == 0
    x, v = mg.metagame_nash(P)
    assert np.allclose(x, [1, 0, 0], atol=1e-9) and abs(v) < 1e-9


# ------------------------------------------------------------------------------------ scripts
def test_league_train_run_history_and_eval_compatibility(tmp_path):
    """A tiny league run writes history.jsonl / config.json / model_*.pt, and scripts/eval.py loads
    the run directory."""
    out = subprocess.run([sys.executable, "scripts/league_train.py", "--total-timesteps", "512", "--num-envs", "4",
                          "--bptt-horizon", "32", "--anchors", "bot:heuristic", "--snapshot-interval", "2",
                          "--seed", "2", "--data-dir", str(tmp_path), "--run-id", "t", "--frame-skip", "50"],
                         cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    summary = json.loads(out.stdout.strip().splitlines()[-1])
    run = tmp_path / "league" / "t"
    assert summary["run_dir"] == str(run) and summary["global_step"] == 512 and summary["epoch"] == 4
    hist = [json.loads(l) for l in open(run / "history.jsonl")]
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4]
    assert all(math.isfinite(h["losses"]["policy_loss"]) for h in hist)
    assert set(os.listdir(run)) >= {"config.json", "history.jsonl", "league_state.json", "learner.pt",
                                    "model_000004.pt", "snap_2.pt", "snap_4.pt", "trainer_state.pt"}
    state = json.load(open(run / "league_state.json"))
    assert state["pool"]["snapshots"] == ["ckpt:snap_2.pt", "ckpt:snap_4.pt"], "stored relative to the run dir"
    ev = subprocess.run([sys.executable, "scripts/eval.py", "--checkpoint", str(run), "--episodes", "1",
                         "--bots", "noop", "--seats", "0"], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert ev.returncode == 0 and "model_000004.pt" in ev.stdout, (ev.stdout + ev.stderr)[-2000:]


def test_load_policy_rejects_layout_mismatch_clearly(tmp_path):
    """A checkpoint whose tensors do not fit the current observation layout (e.g. pre-v0.3) raises a
    ValueError naming the cause instead of a bare torch size-mismatch error."""
    import torch
    L = league()
    p = make_ckpt(tmp_path / "ok.pt")
    sd = torch.load(p, weights_only=True)
    sd["scalars.0.weight"] = torch.zeros(sd["scalars.0.weight"].shape[0], 377)   # the v0.2 scalar width
    torch.save(sd, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="v0.3"):
        L.load_policy(str(tmp_path / "old.pt"))
