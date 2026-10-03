"""SPEC §15.1 OpponentPool, §15.2 LeagueVecEnv, §15.4 league_train.py / best_response.py."""
import glob
import json
import math
import os

import numpy as np
import pytest

import envkit as E
import helpers as H
import leaguekit as L


# ==========================================================================================
# §15.1 OpponentPool
# ==========================================================================================
def snaps(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"s{i}.pt"
        p.write_bytes(b"")
        out.append(str(p))
    return out


def pfsp_w(mode, p, eps):
    w = {"hard": (1 - p) ** 2, "variance": p * (1 - p), "uniform": 1.0}[mode]
    return max(w, eps)


def check_split(w, spf, af, anchors, snap_specs):
    """SPEC §15.7.1: absolute split - self = self_play_frac, anchors = anchor_frac (uniform),
    snapshots = 1 - self_play_frac - anchor_frac (PFSP)."""
    assert abs(sum(w.values()) - 1.0) < 1e-9, f"weights must sum to 1: {w}"
    assert abs(w.get("self", 0.0) - spf) < 1e-9, f"self mass {w.get('self')} != self_play_frac {spf}"
    a_mass = sum(w[a] for a in anchors)
    s_mass = sum(w[s] for s in snap_specs)
    assert abs(a_mass - af) < 1e-9, f"anchor mass {a_mass} != anchor_frac {af}"
    assert abs(s_mass - (1 - spf - af)) < 1e-9, f"snapshot mass {s_mass} != 1 - {spf} - {af}"
    for a in anchors:
        assert abs(w[a] - af / len(anchors)) < 1e-9, "anchors are chosen uniformly"
    return a_mass, s_mass


# snapshot results: s0 3 wins + 1 draw (p = 0.875), s1 1 loss (p = 0), s2 1 draw (p = 0.5),
# s3 unseen (p = 0.5)
RESULTS = {0: [1, 1, 1, 0], 1: [-1], 2: [0], 3: []}
P_OF = {0: 0.875, 1: 0.0, 2: 0.5, 3: 0.5}


@pytest.mark.parametrize("mode", ["hard", "variance", "uniform"])
def test_pfsp_weights_and_mass_split(pr, tmp_path, mode):
    anchors = ("bot:noop", "bot:heuristic")
    pool = L.make_pool(anchors=anchors, max_snapshots=16, pfsp=mode, pfsp_eps=0.05,
                       self_play_frac=0.2, anchor_frac=0.3, seed=0)
    paths = snaps(tmp_path, 4)
    for p in paths:
        pool.add_snapshot(p)
    specs = [f"ckpt:{p}" for p in paths]
    for i, rs in RESULTS.items():
        for r in rs:
            pool.record(specs[i], r)
    w = pool.weights()
    assert set(w) == {"self", *anchors, *specs}, f"weights() keys {sorted(w)}"
    _, s_mass = check_split(w, 0.2, 0.3, anchors, specs)
    raw = [pfsp_w(mode, P_OF[i], 0.05) for i in range(4)]
    for i, s in enumerate(specs):
        want = s_mass * raw[i] / sum(raw)
        assert abs(w[s] - want) < 1e-9, f"{mode}: snapshot {i} (p={P_OF[i]}) weight {w[s]}, want {want}"


def test_draw_counts_half(pr, tmp_path):
    """A single draw gives p = 0.5, i.e. hard weight 0.25 - not 1 (loss) and not eps (win)."""
    pool = L.make_pool(anchors=(), pfsp="hard", pfsp_eps=0.05, self_play_frac=0.0, anchor_frac=0.0)
    a, b = snaps(tmp_path, 2)
    pool.add_snapshot(a)
    pool.add_snapshot(b)
    pool.record(f"ckpt:{a}", 0)
    pool.record(f"ckpt:{b}", -1)
    w = pool.weights()
    assert abs(w[f"ckpt:{a}"] / w[f"ckpt:{b}"] - 0.25) < 1e-9


def test_record_ignores_anchor_and_self_weights(pr, tmp_path):
    pool = L.make_pool(anchors=("bot:random",), self_play_frac=0.25, anchor_frac=0.25, seed=1)
    for p in snaps(tmp_path, 2):
        pool.add_snapshot(p)
    before = pool.weights()
    for r in (1, 1, -1, 0):
        pool.record("self", r)
        pool.record("bot:random", r)
    assert pool.weights() == before, "records against self/anchors do not change the distribution"


@pytest.mark.parametrize("spf,anchors,want", [
    (0.2, ("bot:noop", "bot:heuristic"), {"self": 0.2, "bot:noop": 0.4, "bot:heuristic": 0.4}),
    (0.0, ("bot:heuristic",), {"bot:heuristic": 1.0}),
    (0.3, (), {"self": 1.0}),
])
def test_no_snapshot_fallback(pr, spf, anchors, want):
    """§15.7.1: no snapshots -> the snapshot mass goes to the anchors (to "self" if none)."""
    pool = L.make_pool(anchors=anchors, self_play_frac=spf, anchor_frac=0.2, seed=0)
    w = {k: v for k, v in pool.weights().items() if v > 0}
    assert set(w) == set(want), f"weights {w}"
    for k, v in want.items():
        assert abs(w[k] - v) < 1e-9, f"{k}: {w[k]} want {v}"


def test_no_anchor_fallback_goes_to_snapshots(pr, tmp_path):
    """§15.7.1: no anchors -> the anchor mass goes to the snapshots (PFSP-weighted)."""
    pool = L.make_pool(anchors=(), pfsp="hard", pfsp_eps=0.05, self_play_frac=0.2, anchor_frac=0.3, seed=0)
    a, b = snaps(tmp_path, 2)
    pool.add_snapshot(a)
    pool.add_snapshot(b)
    pool.record(f"ckpt:{b}", -1)                            # p = 0 -> w = 1 ; a unseen -> w = 0.25
    w = {k: v for k, v in pool.weights().items() if v > 0}
    assert set(w) == {"self", f"ckpt:{a}", f"ckpt:{b}"}
    assert abs(w["self"] - 0.2) < 1e-9
    assert abs(w[f"ckpt:{a}"] - 0.8 * 0.25 / 1.25) < 1e-9 and abs(w[f"ckpt:{b}"] - 0.8 * 1 / 1.25) < 1e-9


def test_constructor_rejects_fractions_above_one(pr):
    with pytest.raises((ValueError, AssertionError)):
        L.make_pool(self_play_frac=0.7, anchor_frac=0.5)


def test_stats_bookkeeping(pr, tmp_path):
    """§15.7.2: stats() -> {spec: {games, wins, draws, losses, p}}, draws count 0.5."""
    pool = L.make_pool(anchors=("bot:noop",), self_play_frac=0.2, anchor_frac=0.3, seed=0)
    a, b = snaps(tmp_path, 2)
    pool.add_snapshot(a)
    pool.add_snapshot(b)
    for r in (1, 1, 0, -1, 0):
        pool.record(f"ckpt:{a}", r)
    for r in (-1, -1, 1):
        pool.record("bot:noop", r)
    pool.record("self", 0)
    st = pool.stats()
    want = {f"ckpt:{a}": (5, 2, 2, 1, 3 / 5), "bot:noop": (3, 1, 0, 2, 1 / 3), "self": (1, 0, 1, 0, 0.5)}
    for spec, (g, w_, d, l, p) in want.items():
        s_ = st[spec]
        assert (s_["games"], s_["wins"], s_["draws"], s_["losses"]) == (g, w_, d, l), f"{spec}: {s_}"
        assert abs(s_["p"] - p) < 1e-12, f"{spec}: p {s_['p']} want {p}"
    if f"ckpt:{b}" in st:
        assert st[f"ckpt:{b}"]["games"] == 0 and abs(st[f"ckpt:{b}"]["p"] - 0.5) < 1e-12


def test_evicted_snapshot_stats_dropped_and_no_filesystem(pr, tmp_path):
    pool = L.make_pool(anchors=(), max_snapshots=2, self_play_frac=0.0, anchor_frac=0.0, seed=0)
    ghosts = [str(tmp_path / f"missing_{i}.pt") for i in range(3)]     # never created
    pool.add_snapshot(ghosts[0])
    pool.record(f"ckpt:{ghosts[0]}", 1)
    pool.add_snapshot(ghosts[1])
    pool.add_snapshot(ghosts[2])
    assert f"ckpt:{ghosts[0]}" not in pool.stats(), "an evicted snapshot's stats are dropped"
    assert set(pool.weights()) >= {f"ckpt:{ghosts[1]}", f"ckpt:{ghosts[2]}"}
    assert not any(os.path.exists(g) for g in ghosts), "add_snapshot never touches the filesystem"


def test_eviction_oldest_non_anchor(pr, tmp_path):
    pool = L.make_pool(anchors=("bot:heuristic",), max_snapshots=3, seed=0)
    paths = snaps(tmp_path, 5)
    for p in paths:
        pool.add_snapshot(p)
    keys = set(pool.weights())
    assert "bot:heuristic" in keys and "self" in keys
    assert {k for k in keys if k.startswith("ckpt:")} == {f"ckpt:{p}" for p in paths[2:]}, \
        f"after 5 snapshots with max_snapshots=3 only the 3 newest remain: {sorted(keys)}"


def test_sample_seeded_and_matches_weights(pr, tmp_path):
    def build(seed):
        pool = L.make_pool(anchors=("bot:noop", "bot:heuristic"), pfsp="hard", pfsp_eps=0.05,
                           self_play_frac=0.2, anchor_frac=0.3, seed=seed)
        for p in snaps(tmp_path, 4):
            pool.add_snapshot(p)
        for i, rs in RESULTS.items():
            for r in rs:
                pool.record(f"ckpt:{tmp_path / f's{i}.pt'}", r)
        return pool
    a, b, c = build(7), build(7), build(8)
    sa = [a.sample() for _ in range(300)]
    assert sa == [b.sample() for _ in range(300)], "same seed -> same samples"
    assert sa != [c.sample() for _ in range(300)], "different seed -> different samples"
    pool = build(11)
    w = pool.weights()
    n = 40000
    draws = [pool.sample() for _ in range(n)]
    assert set(draws) <= set(w)
    for k, p in w.items():
        f = draws.count(k) / n
        sigma = math.sqrt(p * (1 - p) / n)
        assert abs(f - p) <= 4.5 * sigma + 1e-4, f"{k}: frequency {f:.4f} vs weight {p:.4f}"
    assert pool.weights() == w, "sampling does not change the weights"


def test_state_dict_round_trip(pr, tmp_path):
    kw = dict(anchors=("bot:noop",), max_snapshots=4, pfsp="variance", pfsp_eps=0.05,
              self_play_frac=0.2, anchor_frac=0.2)
    a = L.make_pool(seed=3, **kw)
    for p in snaps(tmp_path, 3):
        a.add_snapshot(p)
    a.record(f"ckpt:{tmp_path / 's0.pt'}", 1)
    a.record(f"ckpt:{tmp_path / 's1.pt'}", 0)
    for _ in range(57):
        a.sample()
    sd = json.loads(json.dumps(a.state_dict()))            # must be JSON-serialisable
    b = L.make_pool(seed=999, **kw)
    b.load_state_dict(sd)
    assert b.weights() == a.weights()
    assert b.stats() == a.stats()
    assert [b.sample() for _ in range(200)] == [a.sample() for _ in range(200)], \
        "the restored pool must continue with exactly the same samples (RNG state included)"


# ==========================================================================================
# §15.2 LeagueVecEnv
# ==========================================================================================
def noop_pool(**kw):
    args = dict(anchors=("bot:noop",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    args.update(kw)
    return L.make_pool(**args)


def run(lv, n_steps, policy=None, rng=None, seed=0):
    """Drive a LeagueVecEnv; returns per-step records."""
    rng = rng or np.random.default_rng(seed)
    lv.async_reset(seed)
    recs = []
    for _ in range(n_steps):
        obs, rew, term, trunc, infos, env_ids, masks = lv.recv()
        acts = np.array([policy(rng, r) if policy else 0 for r in obs], dtype=np.int32)
        recs.append(dict(obs=obs.copy(), rew=np.array(rew).copy(), term=np.array(term).copy(),
                         trunc=np.array(trunc).copy(), infos=list(infos) if infos else []))
        lv.send(acts)
    return recs


def league_infos(recs):
    return [i for r in recs for i in r["infos"] if isinstance(i, dict) and "league/episodes" in i]


def royale_logs(recs):
    return [i for r in recs for i in r["infos"] if isinstance(i, dict) and "plays_0" in i]


def rand_policy(rng, row):
    return E.random_bot_action(rng, row, 0.5)


def test_league_interface(pr):
    import pufferroyale
    lv = L.make_league(noop_pool(), num_envs=3, seed=0)
    ref = pufferroyale.Royale(num_envs=1, num_agents=2)
    assert lv.num_agents == 3
    assert lv.agents_per_batch == 3 and lv.agent_per_batch == 3
    assert lv.emulated is False and lv.driver_env is not None
    assert lv.single_observation_space == ref.single_observation_space
    assert lv.single_action_space == ref.single_action_space
    lv.async_reset(0)
    out = lv.recv()
    assert len(out) == 7, "recv() -> (obs, rewards, terminals, truncations, infos, env_ids, masks)"
    obs, rew, term, trunc, infos, env_ids, masks = out
    assert obs.shape == (3, E.R().OBS_SIZE) and obs.dtype == np.float32
    assert np.shape(rew) == (3,) and np.shape(term) == (3,) and np.shape(trunc) == (3,)
    assert np.shape(masks)[0] == 3 and len(np.asarray(env_ids)) == 3
    assert isinstance(infos, list)
    for r in range(3):
        assert E.mask_of(obs[r])[0] == 1.0 and E.mask_of(obs[r])[1:].sum() == 0, "lockout at reset"
    lv.send(np.zeros(3, dtype=np.int32))
    lv.recv()
    lv.close()
    ref.close()


def _seats(seed):
    lv = L.make_league(noop_pool(), num_envs=1, seed=seed, frame_skip=100, log_interval=1)
    recs = run(lv, 60 * 12 + 10, rand_policy, seed=5)
    lv.close()
    seats = []
    for lg in royale_logs(recs):
        if lg["plays_1"] == 0 and lg["plays_0"] > 0:
            seats.append(0)
        elif lg["plays_0"] == 0 and lg["plays_1"] > 0:
            seats.append(1)
        else:
            raise AssertionError(f"with a bot:noop opponent only the learner may play: {lg}")
    return seats


def test_learner_seat_randomised_and_seeded(pr):
    s1, s2 = _seats(3), _seats(3)
    assert len(s1) >= 8, f"expected >= 8 finished episodes, got {len(s1)}"
    assert 0 in s1 and 1 in s1, f"both seats must occur: {s1}"
    assert s1 == s2, "seat sequence must be deterministic given the wrapper seed"


def test_noop_opponent_rewards_terminals_and_records(pr):
    """vs bot:noop the learner can never lose a tower: own tower fractions stay 1, rewards are 0
    except on terminals where they are 0/+1, and the pool records every outcome."""
    lay = E.scalar_layout()
    pool = noop_pool()
    lv = L.make_league(pool, num_envs=2, seed=1, frame_skip=100, log_interval=1)
    recs = run(lv, 400, rand_policy, seed=2)
    lv.close()
    results, checked = [], 0
    for r in recs:
        for row in range(2):
            assert np.array_equal(E.scalar(r["obs"][row], lay, "own_tower_hp"), np.ones(3, np.float32)), \
                "the learner lost tower hp to a noop opponent"
        assert not r["trunc"].any()
        for row in range(2):
            if r["term"][row]:
                assert r["rew"][row] in (0.0, 1.0), f"terminal reward {r['rew'][row]} vs noop"
                results.append(float(r["rew"][row]))
            else:
                assert r["rew"][row] == 0.0, "no shaping: non-terminal rewards are 0"
        # league stats emitted with this step must account for every episode finished so far
        for info in r["infos"]:
            if isinstance(info, dict) and "league/episodes" in info:
                assert info["league/episodes"] == len(results), \
                    f"league/episodes {info['league/episodes']} vs completed episodes {len(results)}"
                if results:
                    p = (results.count(1.0) + 0.5 * results.count(0.0)) / len(results)
                    assert abs(info["league/p/bot:noop"] - p) < 1e-6, \
                        "league/p/<opp> = (wins + 0.5 draws) / games from the learner's view"
                assert "league/pool_size" in info
                checked += 1
    assert len(results) >= 4
    assert checked, "infos must carry league/* stats (SPEC §15.2)"
    for lg in royale_logs(recs):
        assert lg["illegal_actions"] == 0


def test_ckpt_opponent_loads_and_plays(pr, tmp_path):
    ck = L.make_ckpt(tmp_path / "opp.pt")
    pool = L.make_pool(anchors=(f"ckpt:{ck}",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.make_league(pool, num_envs=2, seed=0, frame_skip=100, log_interval=1)
    recs = run(lv, 200)                                    # the learner never plays
    lv.close()
    logs = royale_logs(recs)
    assert logs, "no finished episode"
    assert sum(lg["plays_0"] + lg["plays_1"] for lg in logs) > 0, \
        "the frozen checkpoint opponent never played (its actions are sampled, SPEC §15.7.6)"
    assert all(lg["illegal_actions"] == 0 for lg in logs)


def _opponent_plays(tmp_path, greedy, bias=6.0):
    ck = L.make_ckpt(tmp_path / f"noopish_{greedy}_{bias}.pt", noop_bias=bias)
    pool = L.make_pool(anchors=(f"ckpt:{ck}",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.make_league(pool, num_envs=2, seed=0, opponent_greedy=greedy, frame_skip=100, log_interval=1)
    recs = run(lv, 130)
    lv.close()
    logs = royale_logs(recs)
    assert logs
    return sum(lg["plays_0"] + lg["plays_1"] for lg in logs)


def test_opponent_greedy_vs_sampled(pr, tmp_path):
    """§15.7.6 as amended by §19.11.1 (greedy = card first, then tile). A checkpoint whose no-op logit is
    raised by +20 makes P(wait) the largest marginal on every row (e^20 >> 4 x 576 legal tiles at
    ~0), so the greedy opponent never plays; with +6, sampling (the default) still plays. The learner
    never plays, so every play is the opponent's. (The card-first rule itself, incl. the +5 case where
    a slot's total probability beats wait, is tested in test_v05_greedy.py.)"""
    assert _opponent_plays(tmp_path, greedy=True, bias=20.0) == 0, \
        "opponent_greedy=True with wait's marginal the largest must always wait (§19.11.1)"
    assert _opponent_plays(tmp_path, greedy=False) > 0, "default opponents sample (SPEC §15.7.6, §19.11.2)"


def test_learner_keys_rewritten_to_learner_seat(pr):
    """§15.7.5: inside the league learner_score/learner_return/learner_win refer to the learner's
    seat (the underlying self-play env would report team 0)."""
    lv = L.make_league(noop_pool(), num_envs=1, seed=3, frame_skip=100, log_interval=1)
    recs = run(lv, 60 * 14, rand_policy, seed=5)
    lv.close()
    seen_seat1_decisive = False
    for r in recs:
        if not r["term"][0]:
            continue
        ret = float(r["rew"][0])
        lgs = [i for i in r["infos"] if isinstance(i, dict) and "learner_return" in i]
        assert lgs, "the terminal step's infos must carry the episode log (log_interval=1)"
        lg = lgs[-1]
        assert lg["learner_return"] == ret, f"learner_return {lg['learner_return']} vs learner reward {ret}"
        assert abs(lg["learner_score"] - (ret + 1) / 2) < 1e-6
        assert lg["learner_win"] == float(ret == 1.0)
        seat = 0 if lg["plays_0"] > 0 else 1
        if seat == 1 and ret != 0:
            seen_seat1_decisive = True
            assert lg["score"] == 1 - lg["learner_score"], "team-0 score is the opponent's when the learner is team 1"
    assert seen_seat1_decisive, "setup: need a decisive episode with the learner in seat 1"


def test_self_opponent_uses_set_policy_without_grad(pr):
    import torch
    import pufferroyale.torch as prt
    import pufferroyale

    calls = {"n": 0, "grad": []}

    class Counting(prt.Policy):
        def forward_eval(self, obs, state=None):
            calls["n"] += 1
            calls["grad"].append(torch.is_grad_enabled())
            return super().forward_eval(obs, state)

        def forward(self, obs, state=None):
            calls["n"] += 1
            calls["grad"].append(torch.is_grad_enabled())
            return super().forward(obs, state)

    env = pufferroyale.Royale(num_envs=1, num_agents=2)
    pol = Counting(env)
    env.close()
    pool = L.make_pool(anchors=(), self_play_frac=1.0, anchor_frac=0.0, seed=0)
    lv = L.make_league(pool, num_envs=2, seed=0, frame_skip=50)
    lv.set_policy(pol)
    run(lv, 30)
    lv.close()
    assert calls["n"] > 0, "'self' opponents must act with the policy given to set_policy"
    assert not any(calls["grad"]), "opponent inference runs without grad"


@pytest.mark.slow
@pytest.mark.parametrize("mixed", [False, True])
def test_puffer_rl_on_league_vecenv(pr, tmp_path, mixed):
    """PuffeRL on LeagueVecEnv: finite losses; the buffer holds exactly one learner row per match;
    vs a noop-only pool no learner reward is ever negative."""
    import torch
    import pufferlib.pufferl as pufferl
    import pufferroyale
    import pufferroyale.torch as prt
    n_envs, horizon = 8, 16
    if mixed:
        ck = L.make_ckpt(tmp_path / "opp.pt")
        pool = L.make_pool(anchors=("bot:random", f"ckpt:{ck}"), self_play_frac=0.3, anchor_frac=0.5, seed=0)
    else:
        pool = noop_pool()
    lv = L.make_league(pool, num_envs=n_envs, seed=0, frame_skip=100)
    env = pufferroyale.Royale(num_envs=1, num_agents=2)
    policy = prt.Policy(env)
    env.close()
    lv.set_policy(policy)
    cfg = L.ppo_config(tmp_path, batch_size=n_envs * horizon, horizon=horizon, minibatch=n_envs * horizon,
                       total=n_envs * horizon * 6)
    trainer = pufferl.PuffeRL(cfg, lv, policy)
    try:
        assert trainer.total_agents == n_envs, "one learner row per match"
        assert tuple(trainer.observations.shape[:2]) == (n_envs, horizon)
        rewards = []
        while trainer.global_step < cfg["total_timesteps"]:
            trainer.evaluate()
            rewards.append(trainer.rewards.clone())
            trainer.train()
            for k, v in trainer.losses.items():
                if k != "explained_variance":
                    assert math.isfinite(float(v)), f"loss {k} = {v}"
        for n_, p in policy.named_parameters():
            assert torch.isfinite(p).all(), n_
        allr = torch.cat([r.flatten() for r in rewards])
        assert set(torch.unique(allr).tolist()) <= {-1.0, 0.0, 1.0}
        if not mixed:
            assert (allr >= 0).all(), "vs bot:noop the learner can never lose: no opponent rows in the buffer"
    finally:
        trainer.utilization.stop()
        lv.close()


# ==========================================================================================
# §15.4 scripts: league_train.py, best_response.py
# ==========================================================================================
LEAGUE_FLAGS = ["--total-timesteps", "--device", "--num-envs", "--anchors", "--pfsp", "--self-play-frac",
                "--anchor-frac", "--snapshot-interval", "--max-snapshots", "--mmd-coef",
                "--mmd-ref-interval", "--seed", "--data-dir", "--deck0", "--deck1", "--resume"]


def test_league_train_cli_flags(pr):
    flags = L.help_flags("scripts/league_train.py")  # §15.4 + §15.7.12
    missing = [f for f in LEAGUE_FLAGS if f not in flags]
    assert not missing, f"league_train.py lacks SPEC §15.4 flags {missing}"


def _snap_epochs(run_dir):
    return sorted(int(os.path.basename(p)[5:-3]) for p in glob.glob(os.path.join(run_dir, "snap_*.pt")))


@pytest.fixture(scope="module")
def league_run(tmp_path_factory):
    data = tmp_path_factory.mktemp("league")
    args = ["scripts/league_train.py", "--total-timesteps", "2048", "--device", "cpu", "--num-envs", "4",
            "--anchors", "bot:random,bot:noop", "--pfsp", "hard", "--self-play-frac", "0.3",
            "--anchor-frac", "0.4", "--snapshot-interval", "2", "--max-snapshots", "3", "--mmd-coef", "0.1",
            "--mmd-ref-interval", "2", "--seed", "1", "--data-dir", str(data), "--deck0", "hog26",
            "--deck1", "giant"]
    out = L.run_script(args)
    runs = sorted(glob.glob(os.path.join(str(data), "league", "*")))
    return dict(out=out, data=data, runs=runs)


def _resume(league_run, total):
    return L.run_script(["scripts/league_train.py", "--resume", league_run["runs"][0], "--total-timesteps",
                         str(total), "--device", "cpu", "--num-envs", "4", "--snapshot-interval", "2",
                         "--data-dir", str(league_run["data"]), "--seed", "1"])


@pytest.mark.slow
def test_league_train_snapshots_and_state(pr, league_run):
    out = league_run["out"]
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    assert len(league_run["runs"]) == 1, f"expected one <data-dir>/league/<run_id>/: {league_run['runs']}"
    run_dir = league_run["runs"][0]
    epochs = _snap_epochs(run_dir)
    assert epochs, "no snap_<epoch>.pt written"
    assert all(e >= 2 and e % 2 == 0 for e in epochs), \
        f"snapshot epochs {epochs}: PuffeRL epoch after train() increments, every --snapshot-interval 2"
    state = os.path.join(run_dir, "league_state.json")
    assert os.path.isfile(state), "league_state.json missing from the run dir"
    json.load(open(state))


@pytest.mark.slow
def test_league_train_resume_total_is_absolute(pr, league_run):
    """§15.7.12: with --resume, --total-timesteps is the absolute target: resuming with the already
    reached total adds nothing; a larger total continues the epoch count."""
    assert league_run["out"].returncode == 0 and league_run["runs"]
    run_dir = league_run["runs"][0]
    before = _snap_epochs(run_dir)
    out = _resume(league_run, 2048)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    assert _snap_epochs(run_dir) == before, "resuming at the reached absolute target must not train further"
    st0 = json.load(open(os.path.join(run_dir, "league_state.json")))
    out = _resume(league_run, 8192)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    after = _snap_epochs(run_dir)
    assert max(after) > max(before), f"resume must continue the epoch count: {before} -> {after}"
    st1 = json.load(open(os.path.join(run_dir, "league_state.json")))
    assert st1 != st0, "league_state.json must be updated by the resumed run"


BR_FLAGS = ["--target", "--total-timesteps", "--matches", "--device", "--seed", "--out"]


def test_best_response_cli_flags(pr):
    flags = L.help_flags("scripts/best_response.py")
    missing = [f for f in BR_FLAGS if f not in flags]
    assert not missing, f"best_response.py lacks SPEC §15.7.11 flags {missing}"


@pytest.mark.slow
def test_best_response_probe(pr, league_run, tmp_path):
    runs = league_run["runs"]
    if runs and _snap_epochs(runs[0]):
        target = os.path.join(runs[0], f"snap_{_snap_epochs(runs[0])[-1]}.pt")
    else:
        target = L.make_ckpt(tmp_path / "target.pt")
    out_file = tmp_path / "br.json"
    out = L.run_script(["scripts/best_response.py", "--target", target, "--total-timesteps", "2048",
                        "--matches", "4", "--device", "cpu", "--seed", "1", "--out", str(out_file)])
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    res = L.last_line_json(out.stdout)
    for k in ("br_score", "exploitability_proxy"):
        assert k in res, f"best_response JSON lacks {k!r}: {sorted(res)}"
    assert 0.0 <= res["br_score"] <= 1.0
    assert abs(res["exploitability_proxy"] - (2 * res["br_score"] - 1)) < 1e-9
    assert out_file.exists() and json.load(open(out_file)) == res, "--out writes the same JSON"
