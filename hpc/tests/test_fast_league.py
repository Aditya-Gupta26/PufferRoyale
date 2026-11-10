"""Checks of the opt-in throughput paths (pufferroyale.fast_league, DECISIONS D90) against the default
paths. Cluster-side (not under tests/spec, so not collected by `pytest`): run them explicitly,

    python -m pytest -q hpc/tests          (CPU tests; the CUDA tests skip without a GPU)

CPU: threaded stepping (binding.vec_step_range) is bit-identical to vec_step, FastLeagueVecEnv with
env_threads is bit-identical to LeagueVecEnv, and card_stats_update_rows counts exactly as
CardStats.update. CUDA: gpu_obs is bit-identical to LeagueVecEnv (bots), vmap / CUDA-graph forwards match
the eager ones, gpu_sample never draws an illegal action, and the device generator round-trips through
rng_state() / load_rng_state().
"""
import copy
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pufferroyale import binding                     # noqa: E402
from pufferroyale import royale as R                   # noqa: E402
from pufferroyale.league import LeagueVecEnv, OpponentPool   # noqa: E402
from pufferroyale.fast_league import FastLeagueVecEnv, card_stats_update_rows   # noqa: E402

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


def _actions(rng, obs, grid, p_play=0.35):
    """One action per row: a random legal fine play (grid 1) / random coarse play (grid > 1) w.p. p_play."""
    n = obs.shape[0]
    na = R.n_actions(grid)
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if rng.random() >= p_play:
            continue
        if grid == 1:
            legal = np.flatnonzero(obs[i, R.MASK_OFFSET + 1:R.MASK_OFFSET + na] > 0.5) + 1
            if legal.size:
                out[i] = int(rng.choice(legal))
        else:
            out[i] = int(rng.integers(1, na))
    return out


# ------------------------------------------------------------------------------------- CPU
@pytest.mark.parametrize("grid", [1, 2])
def test_vec_step_range_bit_identical(grid):
    n, steps = 48, 160
    a = R.Royale(num_envs=n, num_agents=2, seed=11, placement_grid=grid)
    b = R.Royale(num_envs=n, num_agents=2, seed=11, placement_grid=grid)
    a.reset(11)
    b.reset(11)
    rng = np.random.default_rng(3)
    edges = np.linspace(0, n, 13).astype(int)
    slices = [(int(x), int(y)) for x, y in zip(edges[:-1], edges[1:]) if y > x]
    with ThreadPoolExecutor(6) as pool:
        for _ in range(steps):
            act = _actions(rng, a.observations, grid)
            a.actions[:] = act
            b.actions[:] = act
            binding.vec_step(a.c_envs)
            list(pool.map(lambda s: binding.vec_step_range(b.c_envs, s[0], s[1]), slices))
            assert np.array_equal(a.observations, b.observations)
            assert np.array_equal(a.rewards, b.rewards)
            assert np.array_equal(a.terminals, b.terminals)
    assert binding.vec_log(a.c_envs) == binding.vec_log(b.c_envs)
    a.close()
    b.close()


def _league_pair(fast_kwargs, grid=1, n=32, seed=5, device="cpu"):
    mk = lambda: OpponentPool(anchors=("bot:heuristic", "bot:random", "bot:noop"), self_play_frac=0.0,
                              anchor_frac=1.0, seed=seed)
    base = LeagueVecEnv(mk(), num_envs=n, seed=seed, device="cpu", placement_grid=grid)
    fast = FastLeagueVecEnv(mk(), num_envs=n, seed=seed, device=device, placement_grid=grid, **fast_kwargs)
    return base, fast


def _obs_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _run_pair(base, fast, grid, steps=900):        # > one match (~360-570 steps), so results are compared
    rng = np.random.default_rng(9)
    base.reset(17)
    fast.reset(17)
    for t in range(steps):
        assert np.array_equal(base.observations, _obs_np(fast.observations)), f"observations differ at step {t}"
        act = _actions(rng, base.observations, grid)
        base.send(act)
        fast.send(act)
        assert np.array_equal(base.rewards, fast.rewards)
        assert np.array_equal(base.terminals, fast.terminals)
        assert np.array_equal(base.seats, fast.seats)
        assert base.opponents == fast.opponents
        assert [str(i) for i in base.infos] == [str(i) for i in fast.infos]
    assert base.drain_results() == fast.drain_results()
    for k in ("available", "played"):
        assert np.array_equal(getattr(base.card_stats, k), getattr(fast.card_stats, k))
    assert base.card_stats.decisions == fast.card_stats.decisions
    assert base.league_stats() == fast.league_stats()
    assert base.episodes == fast.episodes > 0


@pytest.mark.parametrize("grid", [1, 2])
def test_env_threads_bit_identical_to_league(grid):
    base, fast = _league_pair({"env_threads": 4}, grid=grid)
    try:
        _run_pair(base, fast, grid)
    finally:
        base.close()
        fast.close()


@pytest.mark.parametrize("grid", [1, 2, 4])
def test_card_stats_update_rows_matches_update(grid):
    env = R.Royale(num_envs=24, num_agents=2, seed=4, placement_grid=grid)
    env.reset(4)
    rng = np.random.default_rng(1)
    a, b = R.CardStats(grid), R.CardStats(grid)
    for _ in range(60):
        rows = 2 * np.arange(24) + rng.integers(0, 2, 24)
        act = _actions(rng, env.observations[rows], grid, p_play=0.6)
        a.update(env.observations[rows], act)
        card_stats_update_rows(b, env.observations, rows, act)
        env.actions[:] = _actions(rng, env.observations, grid)
        binding.vec_step(env.c_envs)
    assert np.array_equal(a.available, b.available) and np.array_equal(a.played, b.played)
    assert a.decisions == b.decisions and a.played.sum() > 0
    env.close()


def test_fast_flags_need_cuda_device():
    pool = OpponentPool(anchors=("bot:noop",), self_play_frac=0.0, anchor_frac=1.0)
    for kw in ({"gpu_obs": True}, {"gpu_sample": True}, {"vmap": True}, {"cuda_graphs": True}):
        with pytest.raises(ValueError):
            FastLeagueVecEnv(pool, num_envs=2, device="cpu", **kw)


# ------------------------------------------------------------------------------------- CUDA
@needs_cuda
@pytest.mark.parametrize("grid", [1, 2])
def test_gpu_obs_and_threads_bit_identical_to_league(grid):
    base, fast = _league_pair({"env_threads": 8, "gpu_obs": True}, grid=grid, device="cuda")
    try:
        _run_pair(base, fast, grid)
    finally:
        base.close()
        fast.close()


def _policies(k, grid=1, device="cuda"):
    from pufferroyale.league import _obs_stub
    from pufferroyale.torch import Policy
    out = []
    for i in range(k):
        torch.manual_seed(100 + i)
        out.append(Policy(_obs_stub(R.n_actions(grid)), placement_grid=grid).to(device).eval())
    return out


@needs_cuda
@pytest.mark.parametrize("grid", [1, 2])
def test_vmap_bank_matches_eager(tmp_path, grid):
    pols = _policies(6, grid)
    specs = []
    for i, p in enumerate(pols):
        path = str(tmp_path / f"snap_{i}.pt")
        torch.save({k: v.detach().cpu() for k, v in p.state_dict().items()}, path)
        specs.append(f"ckpt:{path}")
    pool = OpponentPool(anchors=tuple(specs), self_play_frac=0.0, anchor_frac=1.0, seed=2)
    env = FastLeagueVecEnv(pool, num_envs=64, seed=2, device="cuda", vmap=True, placement_grid=grid)
    try:
        env.reset(2)
        for s in specs:
            env._opponent_policy(s)                       # load every checkpoint into the cache
        opp_rows = 2 * np.arange(64) + (1 - env.seats)
        groups = {}
        for i, s in enumerate(env.opponents):
            groups.setdefault(s, []).append(i)
        vm = [(s, np.asarray(ix)) for s, ix in groups.items()]
        ix_all, got = env._vmap_logits(vm, opp_rows)[0]
        with torch.no_grad():
            ref = torch.cat([env._opponent_policy(s)[0].forward_eval(env._opp_x(opp_rows[ix], "cuda"), {})[0]
                             for s, ix in vm])
        legal = ref > torch.finfo(ref.dtype).min / 2
        assert torch.equal(legal, got > torch.finfo(got.dtype).min / 2)
        assert float((got - ref)[legal].abs().max()) < 1e-3
        kl = (torch.softmax(ref, -1) * (torch.log_softmax(ref, -1) - torch.log_softmax(got, -1))).sum(-1)
        assert float(kl.max()) < 1e-5
        assert torch.equal(got.argmax(-1), ref.argmax(-1))
    finally:
        env.close()


@needs_cuda
def test_cuda_graph_forward_matches_eager():
    from pufferroyale.fast_league import install_learner_graph
    pol = _policies(1)[0]
    eager = copy.deepcopy(pol)
    assert install_learner_graph(pol)
    for rows in (32, 96, 256, 32):
        x = torch.rand(rows, R.OBS_SIZE, device="cuda")
        with torch.no_grad():
            a = pol.forward_eval(x, {})
            b = eager.forward_eval(x, {})
        for u, v in zip(a, b):
            torch.testing.assert_close(u, v, rtol=1e-5, atol=1e-5)
    # weights updated in place are seen by the replay
    with torch.no_grad():
        for p, q in zip(pol.parameters(), eager.parameters()):
            p.add_(0.01)
            q.add_(0.01)
        x = torch.rand(96, R.OBS_SIZE, device="cuda")
        torch.testing.assert_close(pol.forward_eval(x, {})[0], eager.forward_eval(x, {})[0], rtol=1e-5, atol=1e-5)
    from pufferroyale.fast_league import learner_graph
    assert learner_graph(pol).captures == 3


@needs_cuda
def test_deepcopy_of_graphed_policy_runs_its_own_weights():
    """Review finding #1 (2026-10-04): MMDPuffeRL._frozen_copy deep-copies the learner; the copy (the MMD
    "magnet") must run the plain forward with ITS weights, not replay the learner's CUDA graph."""
    from pufferroyale.fast_league import install_learner_graph, learner_graph
    pol = _policies(1)[0]
    assert install_learner_graph(pol)
    x = torch.rand(128, R.OBS_SIZE, device="cuda")
    with torch.no_grad():
        pol.forward_eval(x, {})                              # a graph exists before the copy
    ref = copy.deepcopy(pol)                                 # exactly what _frozen_copy does
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    assert learner_graph(ref) is None
    with torch.no_grad():
        for p in ref.parameters():
            p.mul_(0.5).add_(0.01)                           # the magnet's weights now differ
        eager = copy.deepcopy(ref)
        eager.__dict__.pop("forward_eval", None)             # plain class method
        got = ref(x, dict(action=None, lstm_h=None, lstm_c=None))[0]     # the call _mmd_kl makes
        want = eager.forward_eval(x, {})[0]
        learner = pol.forward_eval(x, {})[0]
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert float((got - learner).abs().max()) > 1e-3, "the copy replayed the learner's graph"


@needs_cuda
def test_graph_cache_capture_after_all_graphs_freed():
    """Regression (2026-10-04): with one process-wide pool, capturing after every graph of the pool had been
    freed failed an allocator assert (use_count > 0), e.g. after a vmap-bank rebuild."""
    import gc
    from pufferroyale.fast_league import GraphCache
    pol = _policies(1)[0]
    for _ in range(3):                                    # create, use, drop -- repeatedly
        gcache = GraphCache(lambda x: pol.forward_eval(x, {}))
        x = torch.rand(64, R.OBS_SIZE, device="cuda")
        with torch.no_grad():
            out = gcache(x)[0]
            ref = pol.forward_eval(x, {})[0]
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        del gcache, out
        gc.collect()
        torch.cuda.synchronize()


@needs_cuda
def test_vmap_graph_bank_rebuilds(tmp_path):
    """The vmap bank is rebuilt (with new CUDA graphs) whenever the set of cached checkpoints changes; with
    no learner graph keeping any pool alive, every rebuild must still capture and match the eager forward."""
    pols = _policies(5)
    paths = []
    for i, p in enumerate(pols):
        paths.append(str(tmp_path / f"snap_{i}.pt"))
        torch.save({k: v.detach().cpu() for k, v in p.state_dict().items()}, paths[-1])
    pool = OpponentPool(anchors=tuple(f"ckpt:{p}" for p in paths[:2]), self_play_frac=0.0, anchor_frac=1.0, seed=4)
    env = FastLeagueVecEnv(pool, num_envs=32, seed=4, device="cuda", vmap=True, cuda_graphs=True)
    try:
        env.reset(4)
        for k in range(2, 6):                              # grow the cache: 2, 3, 4, 5 checkpoints
            for p in paths[:k]:
                env._opponent_policy(f"ckpt:{p}")
            for p in paths[k:]:
                env._ckpt_cache.pop(f"ckpt:{p}", None)
            specs = [f"ckpt:{p}" for p in paths[:k]]
            opp_rows = 2 * np.arange(32) + (1 - env.seats)
            vm = [(s, np.arange(32)[i::k]) for i, s in enumerate(specs)]
            ix_all, got = env._vmap_logits(vm, opp_rows)[0]
            with torch.no_grad():
                ref = torch.cat([env._opponent_policy(s)[0].forward_eval(env._opp_x(opp_rows[ix], "cuda"), {})[0]
                                 for s, ix in vm])
            assert env._vbank["K"] == k
            assert torch.equal(got.argmax(-1), ref.argmax(-1))
            legal = ref > torch.finfo(ref.dtype).min / 2
            assert float((got - ref)[legal].abs().max()) < 1e-3
    finally:
        env.close()


@needs_cuda
def test_policy_opponents_fast_path_bit_identical_greedy(tmp_path):
    """The policy-opponent path of gpu_obs + gpu_sample (group order, scatter of actions, device gathers),
    compared with LeagueVecEnv: greedy opponents make both deterministic, and without padding / vmap the
    forwards see identical batches, so whole trajectories must be identical."""
    pols = _policies(3)
    specs = []
    for i, p in enumerate(pols[:2]):
        path = str(tmp_path / f"snap_{i}.pt")
        torch.save({k: v.detach().cpu() for k, v in p.state_dict().items()}, path)
        specs.append(f"ckpt:{path}")
    mk = lambda: OpponentPool(anchors=tuple(specs), self_play_frac=0.4, anchor_frac=0.6, seed=8)
    base = LeagueVecEnv(mk(), num_envs=48, seed=8, device="cuda", opponent_greedy=True)
    fast = FastLeagueVecEnv(mk(), num_envs=48, seed=8, device="cuda", opponent_greedy=True, gpu_obs=True,
                            gpu_sample=True, env_threads=4)
    learner = pols[2]
    base.set_policy(learner)
    fast.set_policy(learner)
    try:
        _run_pair(base, fast, 1, steps=900)
        assert any(s.startswith("ckpt:") for s in base.opponents) and "self" in base.opponents
    finally:
        base.close()
        fast.close()


@needs_cuda
def test_opponent_actions_repeat_after_load_rng_state(tmp_path):
    pols = _policies(2)
    path = str(tmp_path / "snap.pt")
    torch.save({k: v.detach().cpu() for k, v in pols[0].state_dict().items()}, path)
    pool = OpponentPool(anchors=(f"ckpt:{path}",), self_play_frac=0.5, anchor_frac=0.5, seed=6)
    env = FastLeagueVecEnv(pool, num_envs=64, seed=6, device="cuda", gpu_obs=True, vmap=True, cuda_graphs=True)
    from pufferroyale.fast_league import install_learner_graph
    install_learner_graph(pols[1])
    env.set_policy(pols[1])
    try:
        env.reset(6)
        for _ in range(30):                        # past the start-of-match deploy lockout (90 ticks)
            env.send(np.zeros(64, dtype=np.int32))
        env._opponent_actions()                    # warm up (captures) before the comparison
        st = env.rng_state()
        first = [env._opponent_actions() for _ in range(12)]
        env.load_rng_state(st)
        again = [env._opponent_actions() for _ in range(12)]
        # a random-init policy mostly waits: count the plays over all calls so the check is not vacuous
        assert sum(int((x > 0).sum()) for x in first) > 8, "too few card plays for a meaningful comparison"
        assert all(np.array_equal(x, y) for x, y in zip(first, again))
        assert any(not np.array_equal(first[0], x) for x in first[1:])   # the stream moves on between calls
    finally:
        env.close()


@needs_cuda
def test_gpu_sampler_matches_cpu_sampler_distribution():
    """gpu_sample draws torch.multinomial(softmax(logits)) on the device; the default path draws the same on
    the CPU. On real masked logits (a policy on real game states, after the deploy lockout) the two empirical
    action distributions must differ no more than two CPU samplers with different seeds do."""
    env = R.Royale(num_envs=16, num_agents=2, seed=12)
    env.reset(12)
    rng = np.random.default_rng(12)
    for _ in range(40):
        env.actions[:] = _actions(rng, env.observations, 1)
        binding.vec_step(env.c_envs)
    pol = _policies(1)[0]
    with torch.no_grad():
        logits = pol.forward_eval(torch.as_tensor(env.observations, device="cuda"), {})[0].float()
    env.close()
    n, draws = logits.shape[0], 200_000
    probs = torch.softmax(logits, dim=-1)

    def hist(dev, seed):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        x = torch.multinomial(probs.to(dev), draws, replacement=True, generator=g)
        return torch.stack([torch.bincount(x[i].cpu(), minlength=probs.shape[1]) for i in range(n)]).double() / draws

    cpu_a, cpu_b, gpu = hist("cpu", 1), hist("cpu", 2), hist("cuda", 3)
    legal = (probs.cpu() > 0)
    assert float(gpu[~legal].sum()) == 0.0, "the GPU sampler drew a zero-probability (illegal) action"
    tv = lambda p, q: 0.5 * (p - q).abs().sum(-1)
    tv_cc, tv_gc = tv(cpu_a, cpu_b), tv(gpu, cpu_a)
    assert float(tv_gc.mean()) < 1.25 * float(tv_cc.mean()) + 1e-3, (float(tv_gc.mean()), float(tv_cc.mean()))
    # and both agree with the exact probabilities equally well
    p = probs.cpu().double()
    assert float(tv(gpu, p).mean()) < 1.25 * float(tv(cpu_a, p).mean()) + 1e-3


@needs_cuda
def test_gpu_sample_legal_and_rng_roundtrip(tmp_path):
    pol = _policies(1)[0]
    path = str(tmp_path / "snap.pt")
    torch.save({k: v.detach().cpu() for k, v in pol.state_dict().items()}, path)
    pool = OpponentPool(anchors=(f"ckpt:{path}",), self_play_frac=0.5, anchor_frac=0.5, seed=3)
    env = FastLeagueVecEnv(pool, num_envs=64, seed=3, device="cuda", gpu_obs=True, vmap=True,
                           cuda_graphs=True, env_threads=4)
    learner = _policies(1)[0]
    from pufferroyale.fast_league import install_learner_graph
    install_learner_graph(learner)
    env.set_policy(learner)
    try:
        env.reset(3)
        rng = np.random.default_rng(0)
        for _ in range(40):
            opp_rows = 2 * np.arange(64) + (1 - env.seats)
            opp = env._opponent_actions()
            mask = env.env.observations[opp_rows, R.MASK_OFFSET:R.MASK_OFFSET + R.n_actions(1)] > 0.5
            assert mask[np.arange(64), opp].all(), "an opponent drew an illegal action"
            env.send(_actions(rng, _obs_np(env.observations), 1))
        st = env.rng_state()
        assert "opponent_generator_device" in st
        x = torch.rand(256, 2305, device="cuda")
        draw = lambda: torch.multinomial(torch.softmax(x, -1), 1, generator=env._gen_dev)
        first = draw()
        env.load_rng_state(st)
        assert torch.equal(first, draw())
    finally:
        env.close()
