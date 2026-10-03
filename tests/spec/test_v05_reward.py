"""SPEC §19.1 Reward v2 (C env; `Royale` kwargs), v0.5-G.

Pinned by §19.1:
- kwargs reward_tower / reward_crown / reward_elixir / reward_play (defaults 0), reward_elixir_cap /
  reward_play_cap (20), reward_gamma (1.0), shaping_anneal_steps N (0), shaping_step_offset n0 (0);
  negative weights, caps <= 0, gamma outside (0, 1], N < 0 or n0 < 0 -> ValueError.
- Phi_hat(s) = w_t (T_0 - T_1) + w_c (C_0 - C_1) - w_e clip(L_0 - L_1, +-L_cap) + w_p clip(P_0 - P_1, +-P_cap)
  with T_k = sum of the observation encoder's tower fractions, C_k crowns, L_k = leaked/2800, P_k plays.
- m_n = max(0, 1 - (n0 + n)/N) (N > 0), n = index of the C env's c_step since creation (never reset).
- per c_step: Phi_new = 0 if the match ended, else m_n Phi_hat(s'); F = gamma Phi_new - Phi_prev
  (Phi_prev = previous Phi_new of the same match, 0 on a match's first step);
  r_0 = F + result_0 [ended], r_1 = -r_0 exactly.
- Properties: all-zero weights -> v0.4 rewards bit for bit; self-play r_0 + r_1 = 0 every step;
  per finished match and team sum_t gamma^t F_t = 0 (|.| <= 1e-4, float64 from the float32 rewards),
  also while annealing; episode_return / learner_return = per-episode reward sums.

The tower/crown part of Phi_hat is reconstructed from the acting row's own observation (§9 scalars:
tower HP fractions, crowns/3). The leak/play part is reconstructed in a scenario where team 1 never
plays (its leak follows from §4's regen schedule) and team 0 never leaks (checked step by step).
"""
import math

import numpy as np
import pytest

import envkit as E
import helpers as H
import leaguekit as L
import v05kit as V

W_ALL = dict(reward_tower=0.3, reward_crown=0.2, reward_elixir=0.05, reward_play=0.02)
NO_SHAPING = dict(reward_tower=0.0, reward_crown=0.0, reward_elixir=0.0, reward_play=0.0)


def rand_policy(rng, p=0.35):
    """Random legal play for every row (SPEC §8 'random' style): uses only the fine mask.
    `p` is a play probability, or a list of per-row probabilities."""
    def pol(obs):
        acts = []
        for i, row in enumerate(obs):
            pi = p[i % len(p)] if isinstance(p, (list, tuple)) else p
            leg = E.legal_actions(row)
            acts.append(int(rng.choice(leg[1:])) if len(leg) > 1 and rng.random() < pi else 0)
        return acts
    return pol


def run_matches(env, policy, n_matches, max_steps, seed):
    """Step a (num_envs=1, num_agents=2, log_interval=1) env; split the trace into matches.
    Returns a list of dicts: rews (T, 2) float32, result (team 0), steps, log."""
    obs, _ = env.reset(seed=seed)
    out, cur = [], []
    for _ in range(max_steps):
        obs, rew, term, trunc, infos = E.step(env, policy(obs))
        cur.append(rew.copy())
        if term.any():
            assert term.all(), "both agents of a self-play env terminate together"
            logs = E.logs_in(infos)
            assert logs, "log_interval=1: a finished match must emit its log"
            out.append(dict(rews=np.array(cur, dtype=np.float32), result=V.match_result_from_log(logs[-1]),
                            steps=len(cur), log=logs[-1]))
            cur = []
            if len(out) >= n_matches:
                break
    assert len(out) >= n_matches, f"only {len(out)} matches finished in {max_steps} steps"
    return out


def discounted(x, gamma):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(x * gamma ** np.arange(len(x), dtype=np.float64)))


# ==========================================================================================
# kwargs: defaults and validation
# ==========================================================================================
BAD_KWARGS = [
    dict(reward_tower=-0.1), dict(reward_crown=-1e-6), dict(reward_elixir=-0.01), dict(reward_play=-1.0),
    dict(reward_elixir_cap=0.0), dict(reward_elixir_cap=-3.0), dict(reward_play_cap=0.0),
    dict(reward_play_cap=-1.0), dict(reward_gamma=0.0), dict(reward_gamma=-0.5),
    dict(reward_gamma=1.0001), dict(reward_gamma=2.0), dict(reward_gamma=float("nan")),
    dict(shaping_anneal_steps=-1), dict(shaping_step_offset=-1),
]


@pytest.mark.parametrize("kw", BAD_KWARGS, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
def test_invalid_reward_kwargs_raise_value_error(pr, kw):
    """§19.1: negative weights, caps <= 0, gamma not in (0, 1], N < 0 or n0 < 0 -> ValueError."""
    with pytest.raises(ValueError):
        E.make(**kw)


@pytest.mark.parametrize("kw", [dict(reward_gamma=1.0), dict(reward_gamma=1e-3), dict(reward_elixir_cap=1e-3),
                                dict(reward_play_cap=1e-3), dict(shaping_anneal_steps=0),
                                dict(shaping_anneal_steps=1, shaping_step_offset=5), dict(**W_ALL)])
def test_valid_reward_kwargs_accepted(pr, kw):
    """§19.1 boundary values inside the allowed ranges are accepted."""
    env = E.make(**kw)
    env.reset(seed=0)
    E.step(env, [0, 0])
    env.close()


def _trace(env, policy, steps, seed):
    obs, _ = env.reset(seed=seed)
    o, r, t = [obs.copy()], [], []
    for _ in range(steps):
        obs, rew, term, trunc, _ = E.step(env, policy(obs))
        o.append(obs.copy())
        r.append(rew.copy())
        t.append(term.copy())
    return np.array(o), np.array(r), np.array(t)


def test_all_zero_weights_reproduce_v04_rewards_bit_for_bit(pr):
    """§19.1 / §19.8: with all four weights 0 the rewards are v0.4's (terminal +-1/0 only) bit for bit,
    whatever gamma / caps / anneal settings are given; the default constructor is unshaped."""
    configs = [dict(), dict(NO_SHAPING),
               dict(NO_SHAPING, reward_gamma=0.9, reward_elixir_cap=1.0, reward_play_cap=2.0,
                    shaping_anneal_steps=50, shaping_step_offset=7)]
    traces = []
    for kw in configs:
        env = E.make(num_envs=2, num_agents=2, seed=3, **kw)
        traces.append(_trace(env, rand_policy(np.random.default_rng(11), 0.4), 900, seed=5))
        env.close()
    o0, r0, t0 = traces[0]
    for o, r, t in traces[1:]:
        assert np.array_equal(o, o0) and np.array_equal(t, t0)
        assert np.array_equal(r.view(np.uint32), r0.view(np.uint32)), "rewards differ bitwise from the default env"
    assert t0.any(), "setup: at least one match must finish in 900 steps"
    assert np.all(r0[~t0] == 0.0), "unshaped: non-terminal rewards are exactly 0"
    assert set(np.unique(r0[t0]).tolist()) <= {-1.0, 0.0, 1.0}, "unshaped: terminal rewards are +-1/0"


def _passive_scenario(env, steps, seed, rng):
    """Team 0 plays a random mask-legal card whenever it can; team 1 never plays."""
    obs, _ = env.reset(seed=seed)
    rews, obs_hist = [], []
    for _ in range(steps):
        leg = E.legal_actions(obs[0])
        a0 = int(rng.choice(leg[1:])) if len(leg) > 1 else 0
        obs, rew, term, trunc, _ = E.step(env, [a0, 0])
        rews.append(rew.copy())
        obs_hist.append(obs.copy())
    return np.array(rews), np.array(obs_hist)


def test_default_caps_gamma_and_anneal_values(pr):
    """§19.1 defaults: reward_elixir_cap 20, reward_play_cap 20, reward_gamma 1.0, N = 0, n0 = 0.
    In the passive scenario team 1's leak passes 20 elixir (so the elixir cap binds); the defaults
    must equal the explicit values, and a different cap / gamma / N must change the rewards."""
    def run(**kw):
        env = E.make(num_envs=1, num_agents=2, seed=2, reward_elixir=0.05, reward_play=0.01,
                     reward_tower=0.1, **kw)
        r, _ = _passive_scenario(env, 330, 4, np.random.default_rng(8))
        env.close()
        return r
    base = run()
    explicit = run(reward_elixir_cap=20.0, reward_play_cap=20.0, reward_gamma=1.0, shaping_anneal_steps=0,
                   shaping_step_offset=0)
    assert np.array_equal(base.view(np.uint32), explicit.view(np.uint32)), \
        "defaults must equal reward_elixir_cap=20, reward_play_cap=20, reward_gamma=1, N=0, n0=0"
    for kw in (dict(reward_elixir_cap=19.0), dict(reward_gamma=0.999), dict(shaping_anneal_steps=100000)):
        assert not np.array_equal(run(**kw), base), f"setup: {kw} must be observable in this scenario"


# ==========================================================================================
# Zero-sum and the discounted-shaping invariant on real matches
# ==========================================================================================
def test_selfplay_rewards_are_exact_negations(pr):
    """§19.1 step 3: r_1 = -r_0 (exact float negation); zero-sum every step, several envs."""
    env = E.make(num_envs=3, num_agents=2, seed=7, reward_gamma=0.97, shaping_anneal_steps=500,
                 shaping_step_offset=13, reward_elixir_cap=4.0, reward_play_cap=3.0, **W_ALL)
    obs, _ = env.reset(seed=7)
    pol = rand_policy(np.random.default_rng(7), 0.4)
    nonzero = 0
    for t in range(700):
        obs, rew, term, trunc, _ = E.step(env, pol(obs))
        pair = rew.reshape(3, 2)
        assert np.array_equal(pair[:, 1], -pair[:, 0]), f"step {t}: r_1 != -r_0: {pair.tolist()}"
        assert np.all(pair.sum(axis=1) == 0.0)
        nonzero += int(np.count_nonzero(pair[:, 0] * (~term.reshape(3, 2)[:, 0])))
    assert nonzero > 50, "setup: shaped rewards must be nonzero on many non-terminal steps"
    env.close()


INVARIANT_CONFIGS = {
    "constant_g095": dict(reward_gamma=0.95),
    "constant_g1": dict(reward_gamma=1.0),
    "anneal_crosses_zero": dict(reward_gamma=0.97, shaping_anneal_steps=700),
    "anneal_with_offset": dict(reward_gamma=0.99, shaping_anneal_steps=1000, shaping_step_offset=600),
    "small_caps": dict(reward_gamma=0.98, reward_elixir_cap=1.5, reward_play_cap=2.0, shaping_anneal_steps=400),
}


@pytest.mark.parametrize("name", sorted(INVARIANT_CONFIGS))
def test_discounted_shaping_sums_to_zero_per_match(pr, name):
    """§19.1 Properties: for every finished match and team, sum_t gamma^t F_t = 0 within 1e-4
    (float64 from the stored float32 rewards), F_t = r_t - result [terminal]; hence the discounted
    shaped return equals gamma^(T-1) * result -- with constant weights and while annealing."""
    kw = dict(W_ALL, **INVARIANT_CONFIGS[name])
    gamma = kw["reward_gamma"]
    env = E.make(num_envs=1, num_agents=2, log_interval=1, seed=21, **kw)
    matches = run_matches(env, rand_policy(np.random.default_rng(5), 0.35), 2, 1300, seed=9)
    env.close()
    shaped_steps = 0
    for k, m in enumerate(matches):
        rews = m["rews"].astype(np.float64)
        T = m["steps"]
        assert np.array_equal(m["rews"][:, 1], -m["rews"][:, 0])
        for team in (0, 1):
            res = m["result"] if team == 0 else -m["result"]
            F = rews[:, team].copy()
            F[-1] -= res
            s = discounted(F, gamma)
            assert abs(s) <= 1e-4, f"{name} match {k} team {team}: sum gamma^t F_t = {s:.3e} (T={T})"
            ret = discounted(rews[:, team], gamma)
            assert abs(ret - gamma ** (T - 1) * res) <= 1e-4, \
                f"{name} match {k} team {team}: discounted return {ret} != gamma^(T-1) * result {gamma ** (T - 1) * res}"
        shaped_steps += int(np.count_nonzero(rews[:-1, 0]))
    assert shaped_steps > 20, f"setup: the shaping must be active in {name} ({shaped_steps} nonzero steps)"


# ==========================================================================================
# Exact per-step values
# ==========================================================================================
def _check_exact_towers_crowns(gamma, N, n0, steps, seed, w):
    """Both teams play randomly; with w_e = w_p = 0 the potential is fully observable."""
    env = E.make(num_envs=1, num_agents=2, log_interval=1, seed=seed, reward_tower=w["t"], reward_crown=w["c"],
                 reward_gamma=gamma, shaping_anneal_steps=N, shaping_step_offset=n0)
    lay = V.layout()
    obs, _ = env.reset(seed=seed)
    pol = rand_policy(np.random.default_rng(seed), [0.9, 0.05])     # one-sided pressure: crowns happen
    phi_prev = 0.0
    ends, crowned, worst = 0, 0, 0.0
    for n in range(steps):
        obs, rew, term, trunc, infos = E.step(env, pol(obs))
        if term[0]:
            res = V.match_result_from_log(E.logs_in(infos)[-1])
            exp = res - phi_prev                                  # Phi_new = 0 at the terminal state
            phi_prev = 0.0                                        # the next match starts at 0
            ends += 1
        else:
            phi_new = V.anneal(n, N, n0) * V.phi_hat(obs[0], lay, w)
            exp = gamma * phi_new - phi_prev
            phi_prev = phi_new
            crowned += int(sum(V.crowns_of(obs[0], lay)) > 0)
        err = abs(float(rew[0]) - exp)
        worst = max(worst, err)
        assert err <= 1e-5 * max(1.0, abs(exp)), \
            f"step n={n}: r_0 = {float(rew[0])!r}, SPEC §19.1 gives {exp!r} (terminal={bool(term[0])})"
        assert float(rew[1]) == -float(rew[0])
    env.close()
    return ends, crowned


def test_exact_reward_tower_and_crown_terms_with_annealing(pr):
    """§19.1 steps 1-3 value by value: r_0 = gamma m_n Phi_hat(s') - Phi_prev (terminal: result - Phi_prev),
    Phi_hat from the observation's tower fractions and crowns, m_n crossing zero mid-run (N = 900,
    n0 = 100), across a match boundary (Phi_prev = 0 for the new match)."""
    ends, crowned = _check_exact_towers_crowns(0.96, 900, 100, 1100, 31, dict(t=0.3, c=0.2))
    assert ends >= 1, "setup: a match must finish during the run"
    assert crowned > 0, "setup: crowns must change during the run (exercise the crown term)"


def test_exact_reward_constant_weights_gamma_one(pr):
    """§19.1 with gamma = 1 and N = 0 (constant weights): F = Phi_hat(s') - Phi_hat(s)."""
    ends, _ = _check_exact_towers_crowns(1.0, 0, 0, 700, 12, dict(t=0.5, c=0.25))
    assert ends >= 1


@pytest.mark.parametrize("caps", [(2.5, 3.0), (20.0, 20.0)], ids=["small_caps", "default_caps"])
def test_exact_reward_all_four_terms_and_caps(pr, caps):
    """§19.1: all four terms, incl. the clipped elixir-leak and play differences. Team 1 never plays
    (L_1 follows §4's regen from 16800 units up to the 28000 cap, P_1 = 0); team 0 plays a random legal
    card whenever possible and never reaches the cap (L_0 = 0; checked every step), P_0 = its count of
    mask-legal plays (all applied, §14.1)."""
    L_cap, P_cap = caps
    w = dict(t=0.3, c=0.2, e=0.05, p=0.1)
    gamma, N, n0 = 0.98, 0, 0
    kw = dict(reward_tower=w["t"], reward_crown=w["c"], reward_elixir=w["e"], reward_play=w["p"],
              reward_gamma=gamma)
    if caps != (20.0, 20.0):
        kw.update(reward_elixir_cap=L_cap, reward_play_cap=P_cap)
    env = E.make(num_envs=1, num_agents=2, log_interval=1, seed=6, **kw)
    lay = V.layout()
    rng = np.random.default_rng(6)
    obs, _ = env.reset(seed=6)
    phi_prev, ticks, plays0 = 0.0, 0, 0
    max_P, max_L, ends = 0, 0.0, 0
    for n in range(520):
        e0 = int(round(float(V.sc(obs[0], lay, "own_elixir")[0]) * 28000))
        inc = V.elixir_income(ticks + 10) - V.elixir_income(ticks)
        assert e0 + inc <= 28000, f"setup: team 0 could leak at step {n} (elixir {e0} + {inc})"
        leg = E.legal_actions(obs[0])
        a0 = int(rng.choice(leg[1:])) if len(leg) > 1 else 0
        obs, rew, term, trunc, infos = E.step(env, [a0, 0])
        plays0 += int(a0 != 0)
        ticks += 10
        if term[0]:
            res = V.match_result_from_log(E.logs_in(infos)[-1])
            assert E.logs_in(infos)[-1]["plays_0"] == plays0 and E.logs_in(infos)[-1]["plays_1"] == 0
            exp = res - phi_prev
            phi_prev, ticks, plays0 = 0.0, 0, 0
            ends += 1
        else:
            L1 = V.passive_leak(ticks)
            max_P, max_L = max(max_P, plays0), max(max_L, L1)
            ph = V.phi_hat(obs[0], lay, w, L_diff=0.0 - L1, P_diff=plays0 - 0, L_cap=L_cap, P_cap=P_cap)
            phi_new = V.anneal(n, N, n0) * ph
            exp = gamma * phi_new - phi_prev
            phi_prev = phi_new
        assert abs(float(rew[0]) - exp) <= 1e-5 * max(1.0, abs(exp)), \
            f"step {n}: r_0 = {float(rew[0])!r}, SPEC §19.1 gives {exp!r} (plays0={plays0}, ticks={ticks})"
        assert float(rew[1]) == -float(rew[0])
    env.close()
    assert max_L > L_cap, f"setup: team 1's leak ({max_L:.2f}) must exceed the elixir cap {L_cap}"
    if caps == (2.5, 3.0):
        assert max_P > P_cap, f"setup: team 0's plays ({max_P}) must exceed the play cap {P_cap}"


def test_reset_starts_a_new_match_but_not_a_new_anneal_count(pr):
    """§19.1: a new match (here: env.reset(seed) mid-match) starts with Phi_prev = 0, while n is counted
    since the C env was created and never reset by reset(). Play-only shaping (w_p = 0.25)."""
    gamma, N, n0, wp = 0.9, 200, 10, 0.25
    env = E.make(num_envs=1, num_agents=2, seed=1, reward_play=wp, reward_gamma=gamma, shaping_anneal_steps=N,
                 shaping_step_offset=n0)
    rng = np.random.default_rng(3)
    obs, _ = env.reset(seed=1)
    plays = 0
    for n in range(37):                                           # c_steps n = 0..36, team 0 plays
        leg = E.legal_actions(obs[0])
        a0 = int(rng.choice(leg[1:])) if len(leg) > 1 else 0
        plays += int(a0 != 0)
        obs, rew, term, *_ = E.step(env, [a0, 0])
        assert not term.any()
    assert plays >= 3, "setup: team 0 must have played before the reset (Phi_prev != 0)"
    obs, _ = env.reset(seed=3)
    for k in range(9):                                            # n = 37..45: lockout, Phi_hat(s') = 0
        obs, rew, term, *_ = E.step(env, [0, 0])
        assert rew[0] == 0.0, f"after reset(): Phi_prev must be 0 for the new match (step {k}: r_0 = {rew[0]})"
    leg = E.legal_actions(obs[0])
    assert len(leg) > 1, "setup: lockout over after 9 steps"
    obs, rew, term, *_ = E.step(env, [int(leg[1]), 0])            # n = 46: P_0 - P_1 = 1
    exp = gamma * (1 - (n0 + 46) / N) * wp
    assert abs(float(rew[0]) - exp) <= 1e-6, \
        f"r_0 = {float(rew[0])} but gamma * m_46 * w_p = {exp} (n counts c_steps since creation)"
    env.close()


def test_anneal_to_zero_makes_shaping_vanish(pr):
    """§19.1: once m_n = 0 (n0 + n >= N) Phi_new = 0; from the next step on F = 0 exactly, so rewards
    are terminal +-1/0 only. With n0 >= N from the start the env equals the unshaped one bit for bit."""
    N = 150
    env = E.make(num_envs=2, num_agents=2, log_interval=1, seed=4, reward_gamma=0.97, shaping_anneal_steps=N,
                 **W_ALL)
    obs, _ = env.reset(seed=4)
    pol = rand_policy(np.random.default_rng(4), 0.4)
    before = 0
    for n in range(900):
        obs, rew, term, *_ = E.step(env, pol(obs))
        if n < N:
            before += int(np.count_nonzero(rew[~term]))
        elif n >= N + 1:
            assert np.all(rew[~term] == 0.0), f"step {n} > N: shaping must have vanished: {rew}"
            assert set(np.unique(rew[term]).tolist()) <= {-1.0, 0.0, 1.0}
    assert before > 0, "setup: shaping active before the anneal reaches 0"
    env.close()

    traces = []
    for kw in (dict(), dict(W_ALL, reward_gamma=0.9, shaping_anneal_steps=10, shaping_step_offset=10)):
        e = E.make(num_envs=2, num_agents=2, seed=8, **kw)
        traces.append(_trace(e, rand_policy(np.random.default_rng(1), 0.4), 700, seed=2))
        e.close()
    (o0, r0, t0), (o1, r1, t1) = traces
    assert np.array_equal(r0.view(np.uint32), r1.view(np.uint32)) and np.array_equal(o0, o1)


# ==========================================================================================
# Logs, single-agent mode, LeagueVecEnv
# ==========================================================================================
def test_episode_return_logs_are_reward_sums(pr):
    """§19.1: episode_return / learner_return (self-play: team 0, §14.6) stay the per-episode sums of
    the rewards; score/win keys stay outcome based."""
    env = E.make(num_envs=1, num_agents=2, log_interval=1, seed=17, reward_gamma=0.95, **W_ALL)
    matches = run_matches(env, rand_policy(np.random.default_rng(17), 0.35), 2, 1300, seed=17)
    env.close()
    for m in matches:
        lg = m["log"]
        s = float(np.sum(m["rews"][:, 0].astype(np.float64)))
        assert abs(lg["episode_return"] - s) <= 1e-4, f"episode_return {lg['episode_return']} != sum {s}"
        assert abs(lg["learner_return"] - s) <= 1e-4, f"learner_return {lg['learner_return']} != sum {s}"
        assert abs(lg["learner_score"] - (m["result"] + 1) / 2) < 1e-6
        assert abs(lg["score"] - (m["result"] + 1) / 2) < 1e-6
        assert lg["episode_length"] == m["steps"]


@pytest.mark.parametrize("side", [0, 1])
def test_single_agent_learner_reward_follows_its_own_potential(pr, side):
    """§19.1 in 1-agent mode (§9: rows get their team's value): the learner row's reward is its team's
    r_k (team 1: -r_0), reconstructed from the learner's own-frame observation; learner_return is its
    per-episode sum and the discounted shaping still telescopes."""
    gamma, N = 0.97, 400
    w = dict(t=0.4, c=0.3)
    env = E.make(num_envs=1, num_agents=1, opponent="heuristic", learner_side=side, log_interval=1, seed=40 + side,
                 reward_tower=w["t"], reward_crown=w["c"], reward_gamma=gamma, shaping_anneal_steps=N)
    lay = V.layout()
    rng = np.random.default_rng(side)
    obs, _ = env.reset(seed=40 + side)
    phi_prev, rews, done = 0.0, [], 0
    for n in range(700):
        leg = E.legal_actions(obs[0])
        a = int(rng.choice(leg[1:])) if len(leg) > 1 and rng.random() < 0.4 else 0
        obs, rew, term, trunc, infos = E.step(env, [a])
        rews.append(float(rew[0]))
        if term[0]:
            lg = E.logs_in(infos)[-1]
            res = int(round(2 * lg["learner_score"] - 1))
            exp = res - phi_prev
            assert abs(float(rew[0]) - exp) <= 1e-5 * max(1, abs(exp))
            F = np.array(rews, np.float64)
            F[-1] -= res
            assert abs(discounted(F, gamma)) <= 1e-4
            assert abs(lg["learner_return"] - float(np.sum(np.array(rews, np.float64)))) <= 1e-4
            done += 1
            break
        phi_new = V.anneal(n, N, 0) * V.phi_hat(obs[0], lay, w)
        exp = gamma * phi_new - phi_prev
        phi_prev = phi_new
        assert abs(float(rew[0]) - exp) <= 1e-5 * max(1.0, abs(exp)), f"side {side} step {n}: {rew[0]} vs {exp}"
    assert done == 1, "setup: the match must finish within 700 steps"
    env.close()


def test_league_learner_rewards_keep_the_invariant(pr):
    """§15.7.4 forwards the §19.1 kwargs to Royale; §15.2: the learner's reward equals its row's env
    reward, so sum_t gamma^t F_t = 0 holds for the learner's matches (result from learner_score)."""
    gamma = 0.96
    pool = L.make_pool(anchors=("bot:random",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.make_league(pool, num_envs=1, seed=3, log_interval=1, reward_gamma=gamma, shaping_anneal_steps=500,
                       **W_ALL)
    lv.async_reset(3)
    rng = np.random.default_rng(3)
    rews, done, shaped = [], 0, 0
    obs, rew, term, trunc, infos, _, _ = lv.recv()
    for _ in range(1300):
        leg = E.legal_actions(obs[0])
        a = int(rng.choice(leg[1:])) if len(leg) > 1 and rng.random() < 0.4 else 0
        lv.send(np.array([a], dtype=np.int32))
        obs, rew, term, trunc, infos, _, _ = lv.recv()
        rews.append(float(rew[0]))
        shaped += int(rew[0] != 0 and not term[0])
        if term[0]:
            lgs = [i for i in infos if isinstance(i, dict) and "learner_score" in i]
            assert lgs, "the league forwards the Royale log with learner keys"
            res = int(round(2 * lgs[-1]["learner_score"] - 1))
            F = np.array(rews, np.float64)
            F[-1] -= res
            assert abs(discounted(F, gamma)) <= 1e-4, f"league match: sum gamma^t F_t = {discounted(F, gamma)}"
            done += 1
            rews = []
            if done == 2:
                break
    assert done >= 1, "setup: a league match must finish"
    assert shaped > 0, "the §19.1 kwargs must reach the league's Royale (shaped non-terminal rewards)"
    lv.close()
