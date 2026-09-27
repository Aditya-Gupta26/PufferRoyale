"""SPEC §9 env robustness: vectorisation shapes, deterministic mid-episode reset, close(),
PufferLib Multiprocessing backend, and a long self-play soak with log-aggregation checks."""
import numpy as np
import pytest

import envkit as E
import helpers as H

DECKS = ["hog26", "giant", "bait"]


@pytest.mark.parametrize("n_envs,n_agents", [(1, 1), (1, 2), (2, 1), (3, 2), (5, 1), (8, 2)])
def test_vectorised_shapes_and_invariants(pr, n_envs, n_agents):
    env = E.make(num_envs=n_envs, num_agents=n_agents, opponent="random", seed=5)
    obs, infos = env.reset(seed=5)
    rows = n_envs * n_agents
    assert obs.shape == (rows, E.R().OBS_SIZE) and obs.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(60):
        acts = [E.random_bot_action(rng, r, 0.3) for r in obs]
        obs, rew, term, trunc, infos = E.step(env, acts)
        assert obs.shape == (rows, E.R().OBS_SIZE) and rew.shape == (rows,)
        assert term.shape == (rows,) and trunc.shape == (rows,) and not trunc.any()
        assert np.isfinite(obs).all() and np.isfinite(rew).all()
        assert np.all(E.R().MASK_SIZE == 2305)
        for r in range(rows):
            assert E.mask_of(obs[r])[0] == 1.0
        if n_agents == 2:
            assert np.allclose(rew.reshape(n_envs, 2).sum(axis=1), 0.0)
    env.close()


def test_reset_mid_episode_is_deterministic(pr):
    def rollout(env, seed):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(99)
        out = [obs.copy()]
        for _ in range(40):
            acts = [E.random_bot_action(rng, r, 0.3) for r in obs]
            obs, rew, *_ = E.step(env, acts)
            out += [obs.copy(), rew.copy()]
        return out
    fresh = E.make(num_envs=2, num_agents=2, seed=0)
    a = rollout(fresh, 17)
    fresh.close()
    used = E.make(num_envs=2, num_agents=2, seed=0)
    rollout(used, 3)                      # play part of another episode first
    b = rollout(used, 17)                 # then reset mid-episode with the same seed
    used.close()
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b)), \
        "env.reset(seed) mid-episode must restart exactly like a fresh env reset with that seed"


def test_close_is_safe(pr):
    env = E.make(num_envs=2, num_agents=2)
    env.reset(seed=0)
    E.step(env, [0, 0, 0, 0])
    env.close()


@pytest.mark.slow
def test_multiprocessing_backend_smoke(pr):
    import pufferlib.vector
    import pufferroyale
    vec = pufferlib.vector.make(pufferroyale.Royale, backend=pufferlib.vector.Multiprocessing,
                                num_envs=2, num_workers=2,
                                env_kwargs=dict(num_envs=2, num_agents=2, frame_skip=10))
    try:
        obs, infos = vec.reset(seed=0)
        assert obs.shape == (8, E.R().OBS_SIZE)
        rng = np.random.default_rng(3)
        for _ in range(50):
            acts = np.array([E.random_bot_action(rng, r, 0.3) for r in obs], dtype=np.int32)
            obs, rew, term, trunc, infos = vec.step(acts)
            assert np.isfinite(obs).all()
            assert np.allclose(rew.reshape(4, 2).sum(axis=1), 0.0)
    finally:
        vec.close()


@pytest.mark.slow
@pytest.mark.parametrize("d0", DECKS)
@pytest.mark.parametrize("d1", DECKS)
def test_selfplay_soak_and_log_aggregation(pr, d0, d1):
    """Random (SPEC §8, p=0.2) self-play on 4 native envs for T steps; the single log emitted at
    step T (log_interval = T) must aggregate exactly the episodes that finished."""
    T, n_envs = 1500, 4
    env = E.make(num_envs=n_envs, num_agents=2, deck0=d0, deck1=d1, log_interval=T, seed=11)
    obs, _ = env.reset(seed=11)
    rng = np.random.default_rng(3 * DECKS.index(d0) + DECKS.index(d1))
    ep_len = np.zeros(n_envs, int)
    finished = []                          # (length, team-0 terminal reward)
    logs = []
    for t in range(1, T + 1):
        acts = [E.random_bot_action(rng, r, 0.2) for r in obs]
        obs, rew, term, trunc, infos = E.step(env, acts)
        logs += E.logs_in(infos)
        assert not trunc.any() and np.isfinite(obs).all()
        pair = rew.reshape(n_envs, 2)
        assert np.allclose(pair.sum(axis=1), 0.0), f"step {t}: not zero-sum {pair}"
        ep_len += 1
        tm = term.reshape(n_envs, 2)
        for i in range(n_envs):
            assert tm[i, 0] == tm[i, 1], "both agents of an env terminate together"
            if tm[i, 0]:
                assert ep_len[i] <= 600, "a match lasts at most 600 steps at frame_skip 10"
                assert pair[i, 0] in (-1.0, 0.0, 1.0)
                assert E.mask_of(obs[2 * i])[1:].sum() == 0, "auto-reset: new match starts in lockout"
                finished.append((ep_len[i], float(pair[i, 0])))
                ep_len[i] = 0
            else:
                assert pair[i, 0] == 0.0, "no shaping by default: reward only on terminal steps"
    assert len(logs) == 1, f"exactly one log dict expected at step T, got {len(logs)}"
    lg = logs[0]
    n = len(finished)
    assert n >= n_envs, "every env should finish at least one match in 1500 steps"
    assert lg["n"] == n, f"log n={lg['n']} but {n} episodes finished"
    lens = np.array([f[0] for f in finished], float)
    r0 = np.array([f[1] for f in finished], float)
    assert abs(lg["episode_length"] - lens.mean()) < 1e-3
    assert abs(lg["episode_return"] - r0.mean()) < 1e-5
    assert abs(lg["win_0"] - np.mean(r0 > 0)) < 1e-5 and abs(lg["win_1"] - np.mean(r0 < 0)) < 1e-5
    assert abs(lg["draw"] - np.mean(r0 == 0)) < 1e-5
    assert abs(lg["score"] - np.mean((r0 + 1) / 2)) < 1e-5 and abs(lg["perf"] - lg["score"]) < 1e-6
    assert lg["illegal_actions"] == 0
    assert lg["dropped_plays"] == 0, "SPEC §14.1: every mask-legal action is applied"
    env.close()
