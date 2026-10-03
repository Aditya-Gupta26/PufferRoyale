"""SPEC §9 PufferLib 3.0 environment `pufferroyale.Royale`.

Mask correctness is validated against ENGINE BEHAVIOUR (accepted plays are counted by the
engine's own plays_0 / plays_1 / illegal_actions logs), never against the mask itself.
"""
import numpy as np
import pytest

import helpers as H

LOG_KEYS = ["episode_return", "episode_length", "score", "perf", "win_0", "win_1", "draw",
            "overtime", "tiebreak", "crowns_0", "crowns_1", "leaked_0", "leaked_1", "plays_0",
            "plays_1", "illegal_actions", "spawn_overflow", "n"]


def R():
    import pufferroyale.royale as mod
    return mod


def make(**kw):
    import pufferroyale
    args = dict(num_envs=1, num_agents=2, seed=0)
    args.update(kw)
    return pufferroyale.Royale(**args)


def mask_of(obs_row):
    m = R()
    return obs_row[m.MASK_OFFSET:m.MASK_OFFSET + m.MASK_SIZE]


def legal_actions(obs_row):
    return np.nonzero(mask_of(obs_row) > 0.5)[0]


def step(env, actions):
    out = env.step(np.asarray(actions, dtype=np.int32))
    assert len(out) == 5, "step returns (obs, rewards, terminals, truncations, infos)"
    obs, rew, term, trunc, infos = out
    assert isinstance(infos, list), "PufferLib 3.0 infos must be a list"
    return obs, rew, term, trunc, infos


def logs_in(infos):
    return [i for i in infos if isinstance(i, dict) and i]


# ------------------------------------------------------------------------------------------
# Spaces, constants, layout
# ------------------------------------------------------------------------------------------
def test_is_puffer_env_with_spaces(pr):
    import pufferlib
    env = make()
    assert isinstance(env, pufferlib.PufferEnv)
    m = R()
    a = env.single_action_space
    assert type(a).__name__ == "Discrete" and int(a.n) == H.N_ACTIONS
    o = env.single_observation_space
    assert type(o).__name__ == "Box" and o.dtype == np.float32 and tuple(o.shape) == (m.OBS_SIZE,)
    assert env.num_agents == 2
    env.close()


@pytest.mark.parametrize("n_envs,n_agents", [(1, 1), (1, 2), (3, 2), (4, 1)])
def test_num_agents_rows(pr, n_envs, n_agents):
    env = make(num_envs=n_envs, num_agents=n_agents, opponent="random")
    assert env.num_agents == n_envs * n_agents
    obs, infos = env.reset(seed=1)
    assert obs.shape == (n_envs * n_agents, R().OBS_SIZE) and obs.dtype == np.float32
    env.close()


def test_layout_constants_consistent(pr):
    m = R()
    C, h, w = m.SPATIAL_SHAPE
    assert (h, w) == (32, 18)
    assert C >= 25, "10 per-side channels x 2 + 5 shared (SPEC §9.1)"
    n_ent, F = m.ENTITY_SHAPE
    assert n_ent == 64 and F == 11, "SPEC §16.4: 32 own + 32 enemy entity rows of exactly 11 floats"
    assert m.SCALAR_SIZE == 306, "SPEC §16.4 / §16.6 + §19.3 (own_deck): exactly 306 scalar values"
    assert m.MASK_SIZE == H.N_ACTIONS
    sections = sorted([(m.SPATIAL_OFFSET, C * h * w), (m.ENTITY_OFFSET, n_ent * F),
                       (m.SCALAR_OFFSET, m.SCALAR_SIZE), (m.MASK_OFFSET, m.MASK_SIZE)])
    end = 0
    for off, size in sections:
        assert off >= end, f"observation sections overlap at offset {off}"
        end = off + size
    assert end <= m.OBS_SIZE and sections[0][0] >= 0
    import envkit as E
    lay = E.scalar_layout()
    assert sum(n for _, n in lay.values()) == E.SCALAR_TOTAL


# ------------------------------------------------------------------------------------------
# Observation content
# ------------------------------------------------------------------------------------------
def test_reset_obs_values_and_lockout_mask(pr):
    env = make()
    obs, _ = env.reset(seed=3)
    m = R()
    assert np.isfinite(obs).all()
    C, h, w = m.SPATIAL_SHAPE
    sp = obs[:, m.SPATIAL_OFFSET:m.SPATIAL_OFFSET + C * h * w]
    assert sp.min() >= 0.0 and sp.max() <= 1.0, "spatial planes in [0, 1]"
    for r in range(2):
        mk = mask_of(obs[r])
        assert set(np.unique(mk).tolist()) <= {0.0, 1.0}
        assert mk[0] == 1.0 and mk[1:].sum() == 0, "lockout at tick 0: only the no-op is legal"
    env.close()


def test_lockout_ends_after_nine_steps(pr):
    env = make()
    obs, _ = env.reset(seed=3)
    for k in range(1, 10):
        obs, *_ = step(env, [0, 0])
        if k < 9:
            assert mask_of(obs[0])[1:].sum() == 0, f"still locked out after {k} steps"
    for r in range(2):
        assert mask_of(obs[r])[1:].sum() > 0, "tick 90 >= deploy_lockout_ticks: cards playable"
    env.close()


def slot_patterns(team, deck):
    a = H.all_alive()
    pats = {}
    for c in H.DECKS[deck]:
        pats[c] = np.array([int(bool(H.tile_legal(c, team, tx, ty, a)))
                            for ty in range(32) for tx in range(18)], dtype=np.float32)
    return pats


@pytest.mark.parametrize("deck", ["hog26", "giant", "bait"])
def test_mask_blocks_are_own_frame_legality(pr, deck):
    """At tick 90 (21300 units: everything affordable) every slot block of BOTH agents must equal
    the own-frame legality pattern of one of the deck's cards (team-1 rotation included)."""
    env = make(deck0=deck, deck1=deck)
    obs, _ = env.reset(seed=5)
    for _ in range(9):
        obs, *_ = step(env, [0, 0])
    for r, team in ((0, 0), (1, 1)):
        pats = slot_patterns(team, deck)
        mk = mask_of(obs[r])
        for s in range(4):
            blk = mk[1 + s * 576:1 + (s + 1) * 576]
            assert any(np.array_equal(blk, p) for p in pats.values()), \
                f"agent {r} slot {s}: mask block matches no own-frame legality pattern of {deck}"
    env.close()


@pytest.mark.parametrize("steps", [0, 9])
def test_spatial_has_own_frame_tower_and_water_channels(pr, steps):
    """SPEC §9.1 + §13.14: the legal-tile channels show geometry only (also during lockout)."""
    env = make()
    obs, _ = env.reset(seed=0)
    for _ in range(steps):
        obs, *_ = step(env, [0, 0])
    m = R()
    C, h, w = m.SPATIAL_SHAPE
    water = np.array([[H.tile_is_water_engine(tx, ty) for tx in range(18)] for ty in range(32)])
    own_towers = np.zeros((32, 18), bool)
    enemy_towers = np.zeros((32, 18), bool)
    for i in range(3):
        cx, cy = H.TOWER_POS[0][i]
        F = H.TOWER_F[i]
        for ty in range(32):
            for tx in range(18):
                if H.overlap_pos((tx * 1000, ty * 1000, tx * 1000 + 1000, ty * 1000 + 1000),
                                 H.footprint(cx, cy, F)):
                    own_towers[ty, tx] = True          # team 0's towers in its own frame
                    enemy_towers[31 - ty, 17 - tx] = True
    troop_legal = np.array([[H.troop_tile_legal(0, tx, ty, H.all_alive()) for tx in range(18)]
                            for ty in range(32)])
    for r in range(2):
        planes = obs[r, m.SPATIAL_OFFSET:m.SPATIAL_OFFSET + C * h * w].reshape(C, h, w) > 0
        for name, want in (("water", water), ("own crown-tower footprint", own_towers),
                           ("enemy crown-tower footprint", enemy_towers),
                           ("own troop-legal tiles", troop_legal)):
            assert any(np.array_equal(planes[c], want) for c in range(C)), \
                f"agent {r}: no spatial channel equals the own-frame {name} map"
    env.close()


def entity_rows(obs_row):
    m = R()
    n, F = m.ENTITY_SHAPE
    return obs_row[m.ENTITY_OFFSET:m.ENTITY_OFFSET + n * F].reshape(n, F)


def play_first_legal_troop(obs_row, tile=(9, 21)):
    """Action placing a TROOP/BUILDING card at `tile` (own frame): a slot whose block allows the
    tile but forbids the own-frame river tile (9, 16) (spells would allow it)."""
    mk = mask_of(obs_row)
    for s in range(4):
        a = H.action_id(s, *tile)
        if mk[a] > 0.5 and mk[H.action_id(s, 9, 16)] < 0.5:
            return a
    return None


def test_entity_list_rotation_and_features(pr):
    """SPEC §16.4 (v0.3): team 0 plays at own tile (9,21). In team 0's entity list the unit is an OWN
    entity (slot 0) with id card+1 at [0] and position (9500/18000, 21500/32000); in team 1's it is
    an ENEMY entity (slot 32) at the rotated position."""
    env = make(deck0="giant", deck1="giant")
    obs, _ = env.reset(seed=2)
    for _ in range(9):
        obs, *_ = step(env, [0, 0])
    a = play_first_legal_troop(obs[0])
    assert a is not None
    obs, *_ = step(env, [a, 0])
    e0, e1 = entity_rows(obs[0]), entity_rows(obs[1])
    assert e0.shape == (64, 11)
    assert np.abs(e0[0]).sum() > 0, "team 0's own unit must occupy own entity slot 0"
    assert np.abs(e0[32:]).sum() == 0, "no enemy entities (towers are excluded)"
    assert np.abs(e1[:32]).sum() == 0 and np.abs(e1[32]).sum() > 0
    cid = float(e0[0, 0])
    assert cid == int(cid) and 1 <= cid <= 64, f"entity id feature {cid} must be card_id + 1"
    card = int(cid) - 1
    assert card in H.DECKS["giant"] and e1[32, 0] == cid
    # formation members may be offset from the tile centre; compare own vs rotated exactly
    for k in range(32):
        if np.abs(e0[k]).sum() == 0:
            break
        assert e0[k, 0] == cid and e1[32 + k, 0] == cid
        x0, y0 = e0[k, 1], e0[k, 2]
        x1, y1 = e1[32 + k, 1], e1[32 + k, 2]
        assert abs((x0 + x1) - 1.0) < 1e-5 and abs((y0 + y1) - 1.0) < 1e-5, \
            f"entity {k}: team0 ({x0},{y0}) vs team1 ({x1},{y1}) not 180-degree rotations"
    if card in (H.KNIGHT, H.GIANT, H.PRINCE, H.WIZARD, H.BABY_DRAGON):
        assert abs(e0[0, 1] - 9500 / 18000) < 1e-5 and abs(e0[0, 2] - 21500 / 32000) < 1e-5
    # SPEC §16.4 feature layout, for a unit 10 ticks into its 20-tick deploy
    st = H.expected_unit_stats(card)
    want = {3: 1.0, 4: min(1.0, st["hp"] / 2000), 5: float(st["flying"]), 6: 1.0, 7: 0.0,
            8: 0.0, 9: 0.0, 10: float(st["target_only_buildings"])}
    for row in (e0[0], e1[32]):
        for idx, v in want.items():
            assert abs(row[idx] - v) < 1e-6, f"entity feature {idx} = {row[idx]}, want {v}"
    env.close()


def test_scalars_contain_exact_elixir_and_clock(pr):
    """Scalar order is not pinned, so each required quantity is matched as a time series."""
    env = make()
    obs, _ = env.reset(seed=1)
    m = R()
    sc = lambda o: o[m.SCALAR_OFFSET:m.SCALAR_OFFSET + m.SCALAR_SIZE]
    hist = [sc(obs[0]).copy()]
    for _ in range(12):
        obs, *_ = step(env, [0, 0])
        hist.append(sc(obs[0]).copy())
    hist = np.array(hist)
    ks = np.arange(13)
    series = {
        "own elixir/10 (exact, from units)": np.minimum(16800 + 500 * ks, 28000) / 2800 / 10,
        "tick/6000": 10 * ks / 6000,
        "lockout active": (10 * ks < 90).astype(np.float64),
        "elixir rate/3": np.full(13, 1 / 3),
    }
    for name, want in series.items():
        want = want.astype(np.float32)
        assert any(np.allclose(hist[:, i], want, atol=1e-6) for i in range(hist.shape[1])), \
            f"no scalar equals {name} over the first 12 steps"
    env.close()


def test_no_leak_of_opponent_deck_or_hand(pr):
    """Swapping the opponent's (unrevealed) deck must not change a single observation value of
    the other agent while nobody has played."""
    for view, kw_a, kw_b in ((0, dict(deck0="hog26", deck1="giant"), dict(deck0="hog26", deck1="bait")),
                             (1, dict(deck0="giant", deck1="hog26"), dict(deck0="bait", deck1="hog26"))):
        ea, eb = make(**kw_a), make(**kw_b)
        oa, _ = ea.reset(seed=11)
        ob, _ = eb.reset(seed=11)
        assert np.array_equal(oa[view], ob[view]), f"agent {view} sees the opponent's deck at reset"
        for k in range(30):
            oa, *_ = step(ea, [0, 0])
            ob, *_ = step(eb, [0, 0])
            assert np.array_equal(oa[view], ob[view]), f"agent {view} obs differs at step {k + 1}"
        ea.close()
        eb.close()


# ------------------------------------------------------------------------------------------
# Rewards, terminals, auto-reset, logs
# ------------------------------------------------------------------------------------------
def run_episode(env, policy, max_steps=700):
    """Run until the first terminal. policy(obs) -> actions. Returns dict of traces."""
    obs, _ = env.reset(seed=env_seed(env))
    rews, logs = [], []
    for t in range(1, max_steps + 1):
        acts = policy(obs, t)
        obs, rew, term, trunc, infos = step(env, acts)
        rews.append(rew.copy())
        assert not trunc.any(), "truncations are always 0"
        logs += logs_in(infos)
        if term.any():
            assert term.all(), "all agents of a (single) env terminate together"
            return dict(steps=t, rews=np.array(rews), obs=obs.copy(), logs=logs)
    raise AssertionError(f"no terminal within {max_steps} steps")


_SEEDS = {}


def env_seed(env):
    return _SEEDS.get(id(env), 0)


def random_legal_policy(rng, p=0.3, illegal_every=None, counter=None):
    def pol(obs, t):
        acts = []
        for r in range(obs.shape[0]):
            leg = legal_actions(obs[r])
            a = 0
            if illegal_every and t % illegal_every == 0 and r == 0:
                bad = np.nonzero(mask_of(obs[r]) < 0.5)[0]
                if len(bad):
                    a = int(rng.choice(bad))
                    counter["illegal"] += 1
            elif len(leg) > 1 and rng.random() < p:
                a = int(rng.choice(leg[1:]))
                counter["plays"][r] += 1
            acts.append(a)
        return acts
    return pol


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_selfplay_episode_zero_sum_terminal_and_logs(pr, seed):
    env = make(log_interval=1, deck0="bait", deck1="hog26", seed=seed)
    _SEEDS[id(env)] = seed
    rng = np.random.default_rng(seed)
    cnt = {"plays": [0, 0], "illegal": 0}
    ep = run_episode(env, random_legal_policy(rng, 0.3, counter=cnt))
    rews = ep["rews"]
    assert np.allclose(rews.sum(axis=1), 0.0), "self-play rewards must be zero-sum every step"
    assert np.all(rews[:-1] == 0), "no shaping by default: reward only on the terminal step"
    assert rews[-1, 0] in (-1.0, 0.0, 1.0)
    assert ep["steps"] <= 600
    assert mask_of(ep["obs"][0])[1:].sum() == 0, "the returned obs belongs to the NEW match (lockout)"
    assert ep["logs"], "a finished episode must produce a log dict (log_interval=1)"
    lg = ep["logs"][-1]
    for k in LOG_KEYS:
        assert k in lg, f"log key {k!r} missing"
    assert lg["n"] == 1
    assert abs(lg["win_0"] + lg["win_1"] + lg["draw"] - 1.0) < 1e-6
    assert abs(lg["score"] - (lg["win_0"] + 0.5 * lg["draw"])) < 1e-6
    assert abs(lg["perf"] - lg["score"]) < 1e-6
    assert abs(lg["episode_return"] - (lg["win_0"] - lg["win_1"])) < 1e-6
    assert abs(lg["episode_return"] - rews[-1, 0]) < 1e-6
    assert lg["episode_length"] == ep["steps"]
    assert 0 <= lg["crowns_0"] <= 3 and 0 <= lg["crowns_1"] <= 3
    assert lg["leaked_0"] >= 0 and lg["leaked_1"] >= 0
    assert lg["plays_0"] == cnt["plays"][0] and lg["plays_1"] == cnt["plays"][1], \
        "every action the mask allowed must be accepted by the engine"
    assert lg["illegal_actions"] == 0
    env.close()


@pytest.mark.slow
def test_illegal_actions_are_noops_and_counted(pr):
    env = make(log_interval=1, seed=4)
    _SEEDS[id(env)] = 4
    rng = np.random.default_rng(4)
    cnt = {"plays": [0, 0], "illegal": 0}
    ep = run_episode(env, random_legal_policy(rng, 0.0, illegal_every=7, counter=cnt))
    lg = ep["logs"][-1]
    assert cnt["illegal"] > 0
    assert lg["illegal_actions"] == cnt["illegal"], "every masked-out action is counted as illegal"
    assert lg["plays_0"] == 0, "an illegal action is treated as a no-op"
    env.close()


@pytest.mark.parametrize("fs,steps", [(10, 600), (20, 300), (50, 120)])
def test_frame_skip_and_max_length(pr, fs, steps):
    """Nobody plays: the match reaches tick 6000 (exact tiebreak draw) after 6000/frame_skip steps."""
    env = make(frame_skip=fs, log_interval=1)
    obs, _ = env.reset(seed=0)
    for t in range(1, steps + 1):
        obs, rew, term, trunc, infos = step(env, [0, 0])
        if t < steps:
            assert not term.any(), f"terminal at step {t}, expected {steps}"
    assert term.all()
    assert np.all(rew == 0.0), "untouched towers: exact tie -> draw -> 0 reward"
    lg = logs_in(infos)[-1]
    assert lg["draw"] == 1.0 and lg["overtime"] == 1.0
    assert lg["tiebreak"] == 1.0, "SPEC §13.14: tiebreak = 1 iff the end-of-overtime comparison ran"
    assert lg["win_0"] == 0.0 and lg["win_1"] == 0.0 and lg["score"] == 0.5
    assert lg["episode_length"] == steps
    env.close()


def test_shaping_stays_zero_sum(pr):
    env = make(reward_tower=0.5, reward_crown=0.3, seed=9)
    obs, _ = env.reset(seed=9)
    rng = np.random.default_rng(9)
    cnt = {"plays": [0, 0], "illegal": 0}
    pol = random_legal_policy(rng, 0.5, counter=cnt)
    nonzero = 0
    for t in range(1, 400):
        obs, rew, term, trunc, _ = step(env, pol(obs, t))
        assert abs(float(rew[0] + rew[1])) < 1e-6, f"step {t}: rewards {rew} not zero-sum"
        nonzero += int(rew[0] != 0)
    assert nonzero > 0, "tower shaping should produce some nonzero rewards in 400 random steps"
    env.close()


@pytest.mark.parametrize("side", [0, 1])
def test_single_agent_vs_noop_bot(pr, side):
    env = make(num_agents=1, opponent="noop", learner_side=side, log_interval=1, frame_skip=50)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, R().OBS_SIZE)
    for t in range(1, 121):
        obs, rew, term, trunc, infos = step(env, [0])
    assert term.all()
    lg = logs_in(infos)[-1]
    assert lg["plays_0"] == 0 and lg["plays_1"] == 0, "noop bot never plays"
    env.close()


@pytest.mark.slow
@pytest.mark.parametrize("bot", ["random", "heuristic"])
def test_scripted_bots_play_only_legal_actions(pr, bot):
    env = make(num_agents=1, opponent=bot, learner_side=0, log_interval=1, bot_play_prob=0.5)
    obs, _ = env.reset(seed=0)
    for t in range(1, 601):
        obs, rew, term, trunc, infos = step(env, [0])
        if term.any():
            break
    assert term.all()
    lg = logs_in(infos)[-1]
    assert lg["plays_0"] == 0
    assert lg["plays_1"] > 0, f"{bot} bot never played in a whole match"
    assert lg["illegal_actions"] == 0, "scripted bots must only issue legal actions"
    env.close()


def test_learner_side_selects_rows(pr):
    """learner_side=1: the single row is TEAM 1's own-frame view (its troop-legal pattern)."""
    for side in (0, 1):
        env = make(num_agents=1, opponent="noop", learner_side=side)
        obs, _ = env.reset(seed=0)
        for _ in range(9):
            obs, *_ = step(env, [0])
        a = play_first_legal_troop(obs[0], (9, 21))
        assert a is not None, "hog26 always holds a troop or building card"
        obs, *_ = step(env, [a])
        rows = entity_rows(obs[0])
        assert np.abs(rows[0]).sum() > 0, f"learner_side={side}: own entity missing"
        assert rows[0, 2] > 0.5, "the learner's own unit is in the lower half of its own frame"
        env.close()


def test_env_determinism(pr):
    def rollout():
        env = make(num_envs=2, num_agents=2, seed=21)
        obs, _ = env.reset(seed=21)
        rng = np.random.default_rng(0)
        out = [obs.copy()]
        for t in range(120):
            acts = []
            for r in range(obs.shape[0]):
                leg = legal_actions(obs[r])
                acts.append(int(rng.choice(leg)) if rng.random() < 0.3 else 0)
            obs, rew, *_ = step(env, acts)
            out.append(obs.copy())
            out.append(rew.copy())
        env.close()
        return out
    a, b = rollout(), rollout()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_native_multi_env_rows_independent(pr):
    env = make(num_envs=4, num_agents=2, seed=3)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (8, R().OBS_SIZE)
    for t in range(30):
        obs, rew, term, trunc, _ = step(env, np.zeros(8, dtype=np.int32))
        for i in range(4):
            assert rew[2 * i] + rew[2 * i + 1] == 0
    env.close()


def test_pufferlib_serial_vectorization(pr):
    import pufferlib.vector
    import pufferroyale
    vec = pufferlib.vector.make(pufferroyale.Royale, backend=pufferlib.vector.Serial, num_envs=2,
                                env_kwargs=dict(num_envs=1, num_agents=2, frame_skip=10))
    obs, infos = vec.reset(seed=0)
    assert obs.shape == (4, R().OBS_SIZE)
    rng = np.random.default_rng(1)
    for t in range(40):
        acts = np.array([int(rng.choice(legal_actions(obs[r]))) for r in range(4)], dtype=np.int32)
        obs, rew, term, trunc, infos = vec.step(acts)
        assert np.isfinite(obs).all()
        assert abs(rew[0] + rew[1]) < 1e-6 and abs(rew[2] + rew[3]) < 1e-6
    vec.close()


def test_ansi_render(pr):
    env = make(render_mode="ansi")
    env.reset(seed=0)
    out = env.render()
    assert isinstance(out, str) and len(out) > 100, "render_mode='ansi' returns a text board"
    env.close()


def test_env_deploy_lockout_parameter(pr):
    env = make(deploy_lockout_ticks=0)
    obs, _ = env.reset(seed=0)
    assert mask_of(obs[0])[1:].sum() > 0, "deploy_lockout_ticks=0: cards playable at reset"
    env.close()


@pytest.mark.slow
def test_learner_side_random_varies_per_episode(pr):
    env = make(num_agents=1, opponent="noop", learner_side="random", frame_skip=50, log_interval=1,
               seed=13)
    obs, _ = env.reset(seed=13)
    rng = np.random.default_rng(13)
    sides = []
    for _ in range(10 * 121):
        leg = legal_actions(obs[0])
        a = int(rng.choice(leg[1:])) if len(leg) > 1 and rng.random() < 0.3 else 0
        obs, rew, term, trunc, infos = step(env, [a])
        for lg in logs_in(infos):
            if lg["plays_0"] > 0 and lg["plays_1"] == 0:
                sides.append(0)
            elif lg["plays_1"] > 0 and lg["plays_0"] == 0:
                sides.append(1)
        if len(sides) >= 8:
            break
    assert len(sides) >= 8, f"episodes attributed: {sides}"
    assert 0 in sides and 1 in sides, f"learner_side='random' never changed sides over {sides}"
    env.close()


@pytest.mark.slow
def test_illegal_actions_sum_over_both_agents(pr):
    """SPEC §13.14: illegal_actions = sum over both agents per episode."""
    env = make(log_interval=1, frame_skip=50, seed=2)
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(2)
    issued = 0
    for t in range(1, 200):
        acts = [0, 0]
        for r in range(2):
            if t % (3 + r) == 0:
                bad = np.nonzero(mask_of(obs[r]) < 0.5)[0]
                if len(bad):
                    acts[r] = int(rng.choice(bad))
                    issued += 1
        obs, rew, term, trunc, infos = step(env, acts)
        if term.any():
            break
    assert term.all()
    lg = logs_in(infos)[-1]
    assert lg["illegal_actions"] == issued, f"logged {lg['illegal_actions']}, issued {issued}"
    assert lg["plays_0"] == 0 and lg["plays_1"] == 0


def test_scalar_layout_values(pr):
    """Every v0.3 scalar (§9 order with the §16.4 encodings, §16.6.16 tower troops appended) at its
    SCALAR_INDEX position, during the first 12 noop steps."""
    import envkit as E
    lay = E.scalar_layout()
    env = make(deck0="bait", deck1="giant", seed=4)
    obs, _ = env.reset(seed=4)
    for k in range(13):
        if k:
            obs, *_ = step(env, [0, 0])
        tick = 10 * k
        units = min(16800 + 50 * tick, 28000)
        for r, deck in ((0, "bait"), (1, "giant")):
            row = obs[r]
            f = lambda name: E.scalar(row, lay, name)
            assert abs(f("own_elixir")[0] - units / 2800 / 10) < 1e-6
            hand = E.hand_of(row, lay)
            assert None not in hand and len(set(hand)) == 4 and all(c in H.DECKS[deck] for c in hand), hand
            assert np.allclose(f("hand_costs"), [H.COST[c] / 10 for c in hand], atol=1e-6)
            nxt = E.next_of(row, lay)
            assert nxt in H.DECKS[deck] and nxt not in hand
            assert np.array_equal(f("affordable"), np.array([float(H.COST[c] * 2800 <= units) for c in hand],
                                                            np.float32))
            assert abs(f("tick")[0] - tick / 6000) < 1e-6
            assert f("is_overtime")[0] == 0.0
            assert abs(f("elixir_rate")[0] - 1 / 3) < 1e-6
            assert f("lockout")[0] == float(tick < 90)
            assert np.array_equal(f("own_tower_hp"), np.ones(3, np.float32))
            assert np.array_equal(f("enemy_tower_hp"), np.ones(3, np.float32))
            assert np.array_equal(f("king_active"), np.zeros(2, np.float32))
            assert np.array_equal(f("crowns"), np.zeros(2, np.float32))
            for name in ("opp_seen", "opp_spent", "opp_last4", "opp_deduced"):
                assert not f(name).any(), f"{name} must be zero before any opponent play"
            ub = min(10.0, 6 + 50 * tick / 2800) / 10
            assert abs(f("opp_elixir_ub")[0] - ub) < 1e-5, "min(10, 6 + income(t) - spent)/10"
            assert np.array_equal(f("own_tower_troop"), E.tt_onehot("princess"))
            assert np.array_equal(f("enemy_tower_troop"), E.tt_onehot("princess"))
    env.close()
