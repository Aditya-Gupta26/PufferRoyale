"""SPEC §19.4 placement grid (v0.5-G).

- Royale(placement_grid=g), g in {1, 2, 4} (default 1; else ValueError); grid shape (ceil(32/g), ceil(18/g));
  B = rows*cols (576/144/40); action space Discrete(1 + 4B) = 2305/577/161; royale.grid_shape(g),
  royale.n_actions(g).
- Decoding of a coarse action a >= 1: s = (a-1)//B, b = (a-1)%B, by = b//cols, bx = b%cols; block tiles
  [g*bx, min(18, g*bx+g)) x [g*by, min(32, g*by+g)); representative = legal tile (exact fine mask)
  minimising (2tx+1-(x0+x1))^2 + (2ty+1-(y0+y1))^2, ties -> smaller ty, then tx; play = 1 + s*576 +
  ty*18 + tx; no legal tile -> illegal (no-op, counted in illegal_actions); g = 1 identity.
- Observation unchanged (fine 2305 mask); coarse mask derived: coarse[0] = 1, coarse[1+s*B+b] = max of
  the block's fine mask; exports royale.action_mask(obs, grid=1) (numpy bool) and
  torch.action_mask(observations, grid=1).
- Bots act in fine actions (single-agent env; LeagueVecEnv bot opponents exactly as in v0.4).
- Game.coarse_to_fine(team, action, grid) -> fine action (0 = no-op / illegal), same mapping.
- placement_grid is in league.MATCH_ENV_KEYS.
"""
import numpy as np
import pytest

import envkit as E
import gamekit as K
import helpers as H
import leaguekit as L
import v05kit as V


def R():
    return E.R()


# ==========================================================================================
# Constants, exports, validation
# ==========================================================================================
def test_grid_shape_and_n_actions_exports(pr):
    m = R()
    for g in V.GRIDS:
        assert tuple(int(x) for x in m.grid_shape(g)) == V.GRID_SHAPE[g], f"grid_shape({g}) = {m.grid_shape(g)}"
        assert int(m.n_actions(g)) == V.N_ACT[g], f"n_actions({g}) = {m.n_actions(g)}"
    assert [V.N_ACT[g] for g in V.GRIDS] == [2305, 577, 161]


@pytest.mark.parametrize("g", V.GRIDS)
def test_action_space_and_unchanged_observation_space(pr, g):
    env = E.make(placement_grid=g)
    assert int(env.single_action_space.n) == V.N_ACT[g]
    assert tuple(env.single_observation_space.shape) == (int(R().OBS_SIZE),)
    env.close()


def test_default_grid_is_one(pr):
    env = E.make()
    assert int(env.single_action_space.n) == 2305
    env.close()


@pytest.mark.parametrize("g", [0, 3, 8, -1, 5])
def test_invalid_grid_raises(pr, g):
    with pytest.raises(ValueError):
        E.make(placement_grid=g)


def test_match_env_keys_include_placement_grid(pr):
    import pufferroyale.league as lg
    assert "placement_grid" in tuple(lg.MATCH_ENV_KEYS), f"MATCH_ENV_KEYS {lg.MATCH_ENV_KEYS}"


# ==========================================================================================
# Coarse mask and decoding on real states
# ==========================================================================================
def _real_states():
    """A deterministic variety of real game states (both teams): random 64-card decks, random plays,
    full elixir, an opened pocket (destroyed Princess), buildings on the board, lockout."""
    rng = np.random.default_rng(2024)
    out = []
    for seed in range(14):
        g = K.new_game("random", "random", seed=seed, deploy_lockout_ticks=0 if seed % 3 else 90)
        if seed % 2 == 0:
            for t in (0, 1):
                g.set_elixir(t, H.MAX_ELIXIR)
        n_ticks = int(rng.integers(0, 400))
        if seed in (3, 6, 9):                              # open a pocket: kill a Princess tower
            enemy, idx = seed % 2, 1 + seed % 2
            x, y = H.TOWER_POS[enemy][idx]
            g.set_tower_hp(enemy, idx, 1)
            K.spawn(g, 1 - enemy, H.MUSKETEER, x, y + (-4000 if enemy == 0 else 4000))
            K.run_until(g, lambda gg: not K.tower_alive(gg, enemy, idx), 200, "pocket tower destruction")
        for k in range(n_ticks):
            if k % 10 == 0:
                for t in (0, 1):
                    m = np.asarray(g.legal_mask(t))
                    leg = np.nonzero(m[1:])[0] + 1
                    if len(leg) and rng.random() < 0.5:
                        s, tx, ty = H.decode_action(int(rng.choice(leg)))
                        g.play_tile(t, s, tx, ty)
            g.tick(1)
            if K.over(g):
                break
        if K.over(g):
            continue
        if seed % 4 != 1:                                  # mostly everything affordable: dense coarse masks
            for t in (0, 1):
                g.set_elixir(t, H.MAX_ELIXIR)
        out.append(g)
    return out


def test_coarse_mask_matches_block_max_on_real_states(pr):
    """§19.4: coarse[0] = 1, coarse[1+s*B+b] = max of the fine mask over the block (numpy export,
    bool, shape (..., 1+4B)); on Game.obs rows of real states, both teams."""
    m = R()
    for game in _real_states():
        for team in (0, 1):
            obs = np.asarray(game.obs(team), np.float32)
            fm = obs[m.MASK_OFFSET:m.MASK_OFFSET + m.MASK_SIZE]
            assert np.array_equal(fm > 0.5, np.asarray(game.legal_mask(team)) > 0), "obs mask == engine mask"
            for g in V.GRIDS:
                cm = np.asarray(m.action_mask(obs, grid=g))
                assert cm.dtype == np.bool_, f"action_mask dtype {cm.dtype} (SPEC: bool)"
                assert cm.shape == (V.N_ACT[g],), f"action_mask shape {cm.shape}"
                assert np.array_equal(cm, V.ref_coarse_mask(fm, g)), f"grid {g} team {team}: coarse mask"
            assert np.array_equal(np.asarray(m.action_mask(obs, grid=1)), fm > 0.5), "g = 1: the fine mask"
            assert np.array_equal(np.asarray(m.action_mask(obs)), fm > 0.5), "default grid = 1"


def test_action_mask_batched_shapes(pr):
    m = R()
    env = E.make(num_envs=3, num_agents=2, seed=1, deploy_lockout_ticks=0)
    obs, _ = env.reset(seed=1)
    obs, *_ = E.step(env, [0] * 6)
    batch = np.stack([obs, obs[::-1]])                                    # (2, 6, OBS)
    for g in V.GRIDS:
        cm = np.asarray(m.action_mask(batch, grid=g))
        assert cm.shape == (2, 6, V.N_ACT[g]) and cm.dtype == np.bool_
        assert np.array_equal(cm, V.ref_coarse_mask(batch[..., m.MASK_OFFSET:m.MASK_OFFSET + m.MASK_SIZE], g))
    env.close()


def test_torch_action_mask_equals_numpy(pr):
    import torch
    import pufferroyale.torch as prt
    m = R()
    env = E.make(num_envs=2, num_agents=2, seed=2, deploy_lockout_ticks=0)
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(2)
    rows = [obs.copy()]
    for _ in range(30):
        obs, *_ = E.step(env, [E.random_bot_action(rng, r, 0.5) for r in obs])
        rows.append(obs.copy())
    batch = np.stack(rows)                                                 # (31, 4, OBS)
    for g in V.GRIDS:
        want = np.asarray(m.action_mask(batch, grid=g))
        got = prt.action_mask(torch.as_tensor(batch), grid=g)
        assert isinstance(got, torch.Tensor)
        assert tuple(got.shape) == want.shape, f"torch action_mask shape {tuple(got.shape)} vs {want.shape}"
        assert np.array_equal(got.cpu().numpy().astype(bool), want), f"grid {g}: torch != numpy action_mask"
    assert np.array_equal(prt.action_mask(torch.as_tensor(batch)).numpy().astype(bool),
                          np.asarray(m.action_mask(batch))), "grid default = 1"
    env.close()


def test_coarse_to_fine_exhaustive_on_real_states(pr):
    """§19.4 + Game.coarse_to_fine: for every coarse action of every real state, both teams, g in
    {2, 4}: the result equals the SPEC representative rule computed from the exact fine mask; it is 0
    exactly when the coarse mask disallows the action (and for a = 0); every allowed one is accepted
    by the engine (played on a restored snapshot)."""
    checked, played = 0, 0
    for game in _real_states():
        snap = bytes(game.snapshot())
        for team in (0, 1):
            fm = np.asarray(game.legal_mask(team)) > 0
            for g in (2, 4):
                cm = V.ref_coarse_mask(fm, g)
                assert int(game.coarse_to_fine(team, 0, g)) == 0
                for a in range(1, V.N_ACT[g]):
                    want = V.ref_coarse_to_fine(fm, a, g)
                    got = int(game.coarse_to_fine(team, a, g))
                    assert got == want, f"grid {g} team {team} a={a}: coarse_to_fine {got}, SPEC rule {want}"
                    assert (got != 0) == bool(cm[a])
                    if got:
                        assert fm[got], "the representative must be a legal fine action"
                        checked += 1
                        if a % 7 == 0:                             # engine acceptance on a sample
                            s, tx, ty = H.decode_action(got)
                            code = game.play_tile(team, s, tx, ty)
                            assert code == H.OK, f"engine rejected decoded play {got}: code {code}"
                            game.restore(snap)
                            played += 1
    assert checked > 2000 and played > 200, f"setup: too few legal coarse actions ({checked}, {played})"


def test_coarse_to_fine_identity_for_grid_one(pr):
    for game in _real_states()[:4]:
        for team in (0, 1):
            fm = np.asarray(game.legal_mask(team)) > 0
            for a in range(2305):
                got = int(game.coarse_to_fine(team, a, 1))
                assert got == (a if fm[a] and a > 0 else 0), f"g=1 a={a}: {got} (identity on legal actions)"


def test_coarse_to_fine_partial_blocks_and_tie_breaks(pr):
    """Hand-checkable cases with a fully legal mask: g = 4, last column block bx = 4 spans tx 16..17 and
    every block row spans 4 tiles; the centre-nearest tile of a 4x4 block is a 4-way tie -> smaller ty,
    then smaller tx; of the 2-wide block (16..17 x 0..3) -> (16, 1)."""
    fm = np.ones(2305, bool)
    B, cols = 40, 5
    for s in range(4):
        a = 1 + s * B + 0 * cols + 0                       # block (by 0, bx 0): tiles 0..3 x 0..3
        assert V.ref_coarse_to_fine(fm, a, 4) == H.action_id(s, 1, 1)
        a = 1 + s * B + 0 * cols + 4                       # block (by 0, bx 4): tiles 16..17 x 0..3
        assert V.ref_coarse_to_fine(fm, a, 4) == H.action_id(s, 16, 1)
    g = K.new_game("hog26", "hog26", seed=0, deploy_lockout_ticks=0)
    g.set_elixir(0, H.MAX_ELIXIR)
    fm0 = np.asarray(g.legal_mask(0)) > 0
    hand = K.hand(g, 0)
    spell_slot = next((i for i, c in enumerate(hand) if c == H.FIREBALL), None)
    if spell_slot is not None:                             # Fireball: every tile legal (§3.1)
        for b in range(B):
            a = 1 + spell_slot * B + b
            assert int(g.coarse_to_fine(0, a, 4)) == V.ref_coarse_to_fine(fm0, a, 4)


# ==========================================================================================
# The env with a coarse grid
# ==========================================================================================
@pytest.mark.parametrize("g", [2, 4])
def test_coarse_env_equals_fine_env_with_representatives(pr, g):
    """A g-grid env stepped with coarse actions is identical (obs, rewards, terminals, logs) to a fine
    env stepped with their SPEC representatives (decoded from the observation's fine mask)."""
    m = R()
    kw = dict(num_envs=2, num_agents=2, seed=13, log_interval=1, frame_skip=20, deck0="random", deck1="random")
    coarse, fine = E.make(placement_grid=g, **kw), E.make(placement_grid=1, **kw)
    oc, _ = coarse.reset(seed=13)
    of, _ = fine.reset(seed=13)
    rng = np.random.default_rng(g)
    plays = 0
    for t in range(330):
        assert np.array_equal(oc, of), f"step {t}: observations diverged"
        ac, af = [], []
        for row in oc:
            cm = np.asarray(m.action_mask(row, grid=g))
            leg = np.nonzero(cm)[0]
            a = int(rng.choice(leg[1:])) if len(leg) > 1 and rng.random() < 0.4 else 0
            ac.append(a)
            af.append(V.ref_coarse_to_fine(V.fine_mask(row), a, g))
            plays += int(a != 0)
        oc, rc, tc, _, ic = E.step(coarse, ac)
        of, rf, tf, _, i_f = E.step(fine, af)
        assert np.array_equal(rc, rf) and np.array_equal(tc, tf)
        lc, lf = E.logs_in(ic), E.logs_in(i_f)
        assert len(lc) == len(lf)
        for a_, b_ in zip(lc, lf):
            for k in ("plays_0", "plays_1", "illegal_actions", "score", "episode_length"):
                assert a_[k] == b_[k], f"log {k}: coarse {a_[k]} vs fine {b_[k]}"
            assert a_["illegal_actions"] == 0
    assert plays > 50, "setup: the coarse env must play cards"
    coarse.close()
    fine.close()


@pytest.mark.parametrize("g", [2, 4])
def test_disallowed_coarse_actions_are_counted_noops(pr, g):
    """§19.4: a coarse action with no legal tile in its block is illegal: a no-op counted in
    illegal_actions (obs identical to sending 0)."""
    m = R()
    kw = dict(num_envs=1, num_agents=2, seed=3, log_interval=1, frame_skip=100, placement_grid=g)
    a_env, b_env = E.make(**kw), E.make(**kw)
    oa, _ = a_env.reset(seed=3)
    ob, _ = b_env.reset(seed=3)
    sent = 0
    for t in range(60):
        acts = []
        for row in oa:
            cm = np.asarray(m.action_mask(row, grid=g))
            bad = np.nonzero(~cm)[0]
            acts.append(int(bad[t % len(bad)]) if len(bad) else 0)
            sent += int(len(bad) > 0)
        oa, ra, ta, _, ia = E.step(a_env, acts)
        ob, rb, tb, _, ib = E.step(b_env, [0, 0])
        assert np.array_equal(oa, ob) and np.array_equal(ra, rb), "an illegal coarse action must be a no-op"
        if ta.any():
            lg = E.logs_in(ia)[-1]
            assert lg["illegal_actions"] == sent, f"illegal_actions {lg['illegal_actions']} != {sent} sent"
            assert lg["plays_0"] == 0 and lg["plays_1"] == 0
            break
    else:
        raise AssertionError("setup: the match must end (frame_skip 100)")
    a_env.close()
    b_env.close()


@pytest.mark.parametrize("bot", ["random", "heuristic"])
@pytest.mark.parametrize("g", [2, 4])
def test_single_agent_bot_acts_in_fine_actions(pr, bot, g):
    """§19.4: bots inside a single-agent env act in fine actions: with a no-op learner the g-grid env is
    identical to the g = 1 env (same bot plays, no illegal actions)."""
    kw = dict(num_envs=2, num_agents=1, opponent=bot, learner_side="random", bot_play_prob=0.5, log_interval=1,
              frame_skip=20, seed=9)
    ea, eb = E.make(placement_grid=g, **kw), E.make(placement_grid=1, **kw)
    oa, _ = ea.reset(seed=9)
    ob, _ = eb.reset(seed=9)
    bot_plays = 0
    for t in range(300):
        assert np.array_equal(oa, ob), f"step {t}: grid {g} env differs from the fine env"
        oa, ra, ta, _, ia = E.step(ea, [0, 0])
        ob, rb, tb, _, ib = E.step(eb, [0, 0])
        assert np.array_equal(ra, rb) and np.array_equal(ta, tb)
        for lg in E.logs_in(ia):
            assert lg["illegal_actions"] == 0
            bot_plays += lg["plays_0"] + lg["plays_1"]
    assert bot_plays > 0, "setup: the bot must play"
    ea.close()
    eb.close()


@pytest.mark.parametrize("g", [2, 4])
def test_league_bot_opponent_plays_as_in_v04(pr, g):
    """§19.4: in LeagueVecEnv a bot: opponent's play is applied exactly as in v0.4 when g > 1: with a
    no-op learner, the g-grid league produces the same observations as the g = 1 league."""
    def mk(grid):
        pool = L.make_pool(anchors=("bot:random",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
        return L.make_league(pool, num_envs=2, seed=5, log_interval=1, frame_skip=20, bot_play_prob=0.5,
                             placement_grid=grid)
    la, lb = mk(g), mk(1)
    assert int(la.single_action_space.n) == V.N_ACT[g]
    la.async_reset(5)
    lb.async_reset(5)
    plays = 0
    for t in range(300):
        oa, ra, ta, _, ia, _, _ = la.recv()
        ob, rb, tb, _, ib, _, _ = lb.recv()
        assert np.array_equal(oa, ob), f"step {t}: league obs differ between grid {g} and 1"
        assert np.array_equal(ra, rb) and np.array_equal(ta, tb)
        for i in ia:
            if isinstance(i, dict) and "illegal_actions" in i:
                assert i["illegal_actions"] == 0, "the bot's fine actions must not be decoded as coarse"
                plays += i["plays_0"] + i["plays_1"]
        la.send(np.zeros(2, np.int32))
        lb.send(np.zeros(2, np.int32))
    assert plays > 0, "setup: the bot must play"
    la.close()
    lb.close()


def test_coarse_mapping_deterministic(pr):
    """Same state, same call -> same fine action (pure function of the state)."""
    games = _real_states()[:3]
    for game in games:
        a = [int(game.coarse_to_fine(0, k, 4)) for k in range(161)]
        b = [int(game.coarse_to_fine(0, k, 4)) for k in range(161)]
        assert a == b
