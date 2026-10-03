"""SPEC §19.6 policy (`pufferroyale.torch`), v0.5-G.

- Policy(env, hidden_size=256, cnn_channels=64, entity_hidden=128, scalar_hidden=128, card_dim=16,
  head="conditional", card_stats=1, pos_channels=32, placement_grid=None); logits = env.single_action_space.n;
  placement_grid=None derives g from it (2305 -> 1, 577 -> 2, 161 -> 4).
- card_stat_table() -> (ndarray (CARD_SLOTS + 1, K), names): row card_id + 1 (row 0 / unused rows 0), built
  from binding.card_info for the 64 cards, every column scaled into [0, 1] by its maximum over the 64 cards
  (log1p first for HP/damage-like columns), columns in the SPEC order; non-persistent buffer.
- head="flat" = the v0.4 head; head="conditional": joint[0] = log P(wait), joint[1+sB+j] = log P(s) +
  log P(j|s); illegal = finfo.min; final layers std 0.01.
- Exactness (both heads where applicable): softmax(joint) = P(s) P(j|s) within 1e-5; illegal logits exactly
  finfo.min, legal probabilities > 0; wait-only rows -> probability 1, finite entropy; no NaN/inf in logits,
  values, log-probs, entropies or gradients; Policy and Recurrent (training forward on (segments, horizon,
  OBS) and forward_eval); no state persists between calls.
- Checkpoints: league.policy_kwargs_from_state_dict infers head, card_stats, pos_channels, sizes, grid;
  league.load_policy loads every v0.5 checkpoint; older checkpoints raise ValueError.
"""
import math
import os

import numpy as np
import pytest

import envkit as E
import helpers as H
import v05kit as V

HERE = os.path.dirname(os.path.abspath(__file__))
V04_CKPTS = [os.path.join(HERE, "data", "policy_v04_small.pt"), os.path.join(HERE, "data", "recurrent_v04_small.pt")]

STAT_COLUMNS = ["elixir", "is_troop", "is_building", "is_spell", "units", "hitpoints", "damage", "hit_speed",
                "dps", "range", "sight", "speed", "flying", "attacks_air", "attacks_ground", "tob", "splash",
                "ct_pct", "lifetime", "death_damage", "deploy_time", "jumps", "charges", "spawns"]
K = len(STAT_COLUMNS)                                     # 24 columns, §19.6 order


def prt():
    import pufferroyale.torch as mod
    return mod


def env_g(g, **kw):
    return E.make(num_envs=2, num_agents=2, seed=0, placement_grid=g, **kw)


def real_obs(g=1, steps=40, seed=0, deck="random"):
    """Real observations: lockout rows (wait-only) and later rows with full / partial masks."""
    env = E.make(num_envs=3, num_agents=2, seed=seed, deck0=deck, deck1=deck, placement_grid=g)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rows = [obs.copy()]
    for _ in range(steps):
        obs, *_ = E.step(env, [E.random_bot_action(rng, r, 0.5) for r in obs])
        rows.append(obs.copy())
    env.close()
    return np.concatenate(rows)


def fwd(policy, obs):
    import torch
    with torch.no_grad():
        return policy.forward_eval(torch.as_tensor(obs), dict(lstm_h=None, lstm_c=None))


# ==========================================================================================
# Construction
# ==========================================================================================
@pytest.mark.parametrize("g", V.GRIDS)
@pytest.mark.parametrize("head", ["flat", "conditional"])
def test_logit_width_and_grid_inference(pr, g, head):
    env = env_g(g)
    obs, _ = env.reset(seed=0)
    for kw in (dict(), dict(placement_grid=None), dict(placement_grid=g)):
        logits, value = fwd(prt().Policy(env, head=head, **kw), obs)
        assert tuple(logits.shape) == (4, V.N_ACT[g]), f"head {head} grid {g} {kw}: logits {tuple(logits.shape)}"
        assert value.numel() == 4
    env.close()


def test_default_kwargs(pr):
    """The default head is conditional with card stats (both distinguishable from the flat head via the
    state dict: the flat head has a (A, hidden) actor; the defaults equal the explicit SPEC defaults)."""
    import torch
    env = env_g(1)
    torch.manual_seed(0)
    a = prt().Policy(env).state_dict()
    torch.manual_seed(0)
    b = prt().Policy(env, hidden_size=256, cnn_channels=64, entity_hidden=128, scalar_hidden=128, card_dim=16,
                     head="conditional", card_stats=1, pos_channels=32, placement_grid=None).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "defaults != SPEC defaults"
    torch.manual_seed(0)
    c = prt().Policy(env, head="flat").state_dict()
    assert {k: tuple(v.shape) for k, v in a.items()} != {k: tuple(v.shape) for k, v in c.items()}
    env.close()


# ==========================================================================================
# Card-stat table
# ==========================================================================================
def card_infos():
    import pufferroyale as p
    return [p.card_info(c) for c in range(64)]


def _get(d, k):
    v = d.get(k)
    return 0 if v is None else v


def raw_columns():
    """Raw (pre-scaling) values per §19.6 + §19.9.7 for all 64 cards: spells are 0 in the unit-only
    columns (units summoned, hitpoints, hit speed, DPS, range, sight, speed, flying, attacks air/ground,
    buildings-only, lifetime, death damage, deploy time, jumps, charges); a spell's damage column is its
    damage, its splash column its `radius`, "spawns units" = 1 iff card_info["spawn"] is non-empty.
    Returns {col: (values over the 64 cards, log1p?)}."""
    infos = card_infos()
    kind = [d.get("kind") or d.get("card_kind") for d in infos]
    unit = [k in ("troop", "building") for k in kind]
    U = lambda f: [f(d) if u else 0 for d, u in zip(infos, unit)]           # unit-only column
    col = {}
    col["elixir"] = ([_get(d, "elixir") for d in infos], False)
    col["is_troop"] = ([float(k == "troop") for k in kind], False)
    col["is_building"] = ([float(k == "building") for k in kind], False)
    col["is_spell"] = ([float(k == "spell") for k in kind], False)
    col["units"] = (U(lambda d: _get(d, "count") + _get(d, "count2")), False)
    col["hitpoints"] = (U(lambda d: _get(d, "hitpoints")), True)
    col["damage"] = ([_get(d, "damage") for d in infos], True)
    col["hit_speed"] = (U(lambda d: _get(d, "hit_speed_ms")), False)
    col["dps"] = (U(lambda d: _get(d, "damage") * 1000.0 / d["hit_speed_ms"] if _get(d, "hit_speed_ms") > 0 else 0.0),
                  True)
    col["range"] = (U(lambda d: _get(d, "range_milli")), False)
    col["sight"] = (U(lambda d: _get(d, "sight_range_milli")), False)
    col["speed"] = (U(lambda d: _get(d, "speed")), False)
    col["flying"] = (U(lambda d: float(_get(d, "flying_height") > 0)), False)
    col["attacks_air"] = (U(lambda d: float(bool(_get(d, "attacks_air")))), False)
    col["attacks_ground"] = (U(lambda d: float(bool(_get(d, "attacks_ground")))), False)
    col["tob"] = (U(lambda d: float(bool(_get(d, "target_only_buildings")))), False)

    def splash(d, u):
        if u:
            pr_ = d.get("projectile") or {}
            return max(_get(d, "area_damage_radius_milli"), _get(pr_, "radius"))
        return _get(d, "radius")
    col["splash"] = ([splash(d, u) for d, u in zip(infos, unit)], False)
    col["ct_pct"] = ([_get(d, "crown_tower_damage_percent") for d in infos], False)
    col["lifetime"] = (U(lambda d: _get(d, "lifetime_ms")), False)
    col["death_damage"] = (U(lambda d: _get(d, "death_damage")), True)
    col["deploy_time"] = (U(lambda d: _get(d, "deploy_time_ms")), False)
    col["jumps"] = (U(lambda d: float(bool(_get(d, "jumps")))), False)
    col["charges"] = (U(lambda d: float(_get(d, "charge_range") > 0)), False)
    col["spawns"] = ([float(d.get("spawner") is not None or d.get("death_spawn") is not None) if u
                      else float(bool(d.get("spawn"))) for d, u in zip(infos, unit)], False)
    return col


def test_card_stat_table_shape_range_and_rows(pr):
    import pufferroyale as p
    S, names = prt().card_stat_table()
    S = np.asarray(S)
    assert S.shape == (int(p.CARD_SLOTS) + 1, K), f"card_stat_table shape {S.shape}, want (129, {K})"
    assert len(names) == K and all(isinstance(n, str) for n in names) and len(set(names)) == K
    assert np.isfinite(S).all() and S.min() >= 0.0 and S.max() <= 1.0, "every column scaled into [0, 1]"
    assert np.all(S[0] == 0) and np.all(S[65:] == 0), "row 0 (empty) and unused rows are 0"
    colmax = S[1:65].max(axis=0)
    assert np.all((colmax == 0) | (np.abs(colmax - 1) < 1e-6)), f"each column is scaled by its max: {colmax}"


def test_card_stat_table_columns_follow_card_info(pr):
    """Column order and formulas of §19.6 against binding.card_info: every checked column equals
    f(x) / max_c f(x) up to the common scale (checked as proportionality, robust to the IMPL-DEFINED
    values of unchecked cards), f = log1p for HP/damage-like columns; booleans exactly 0/1."""
    S, _ = prt().card_stat_table()
    S = np.asarray(S, np.float64)
    raw = raw_columns()
    for j, name in enumerate(STAT_COLUMNS):
        vals, is_log = raw[name]
        idx = [c for c in range(64) if vals[c] is not None]
        x = np.array([float(vals[c]) for c in idx])
        f = np.log1p(x) if is_log else x
        got = S[[c + 1 for c in idx], j]
        assert np.all((f > 0) | (got == 0)), f"column {j} ({name}): nonzero value where the SPEC gives 0"
        pos = f > 0
        if not pos.any():
            continue
        ratio = got[pos] / f[pos]
        assert np.allclose(ratio, ratio[0], rtol=1e-4), \
            f"column {j} ({name}) is not proportional to its §19.6 value: ratios {np.unique(np.round(ratio, 6))[:6]}"
        if name in ("elixir", "is_troop", "is_building", "is_spell", "flying", "ct_pct", "jumps", "charges",
                    "attacks_air", "attacks_ground", "tob", "spawns"):
            full = np.array([float(v) for v in vals])
            assert np.allclose(S[1:65, j], full / full.max(), atol=1e-6), f"column {j} ({name}) exact values"


def test_card_stats_buffer_not_in_state_dict(pr):
    env = env_g(1)
    sd = prt().Policy(env, card_stats=1).state_dict()
    assert not any(tuple(v.shape) == (129, K) for v in sd.values()), "S is a non-persistent buffer"
    sd0 = prt().Policy(env, card_stats=0).state_dict()
    assert len(sd0) < len(sd), "card_stats=0 drops the Linear(K, card_dim) stat encoder"
    env.close()


# ==========================================================================================
# Exactness of the heads on real observations
# ==========================================================================================
VARIANTS = [("flat", 1, 1), ("conditional", 1, 1), ("conditional", 2, 1), ("conditional", 4, 1),
            ("flat", 4, 1), ("conditional", 1, 0), ("conditional", 4, 0), ("flat", 2, 0)]


@pytest.fixture(scope="module")
def obs_by_grid():
    return {g: real_obs(g=1, steps=30, seed=3) for g in V.GRIDS}


@pytest.mark.parametrize("head,g,cs", VARIANTS)
def test_masks_probabilities_and_finiteness(pr, obs_by_grid, head, g, cs):
    import torch
    import pufferlib.pytorch
    m = E.R()
    obs = obs_by_grid[g]
    env = env_g(g)
    torch.manual_seed(1)
    policy = prt().Policy(env, head=head, card_stats=cs, pos_channels=16)
    logits, value = fwd(policy, obs)
    A = V.N_ACT[g]
    assert tuple(logits.shape) == (obs.shape[0], A)
    cm = torch.as_tensor(np.asarray(m.action_mask(obs, grid=g)))
    minv = torch.finfo(logits.dtype).min
    assert torch.all(logits[~cm] == minv), "illegal logits must be exactly finfo.min"
    assert torch.all(logits[cm] > minv) and torch.isfinite(logits[cm]).all()
    assert torch.isfinite(value).all()
    probs = torch.softmax(logits.double(), dim=-1)
    assert torch.all(probs[~cm] == 0) and torch.all(probs[cm] > 0), "legal probabilities > 0, illegal = 0"
    wait_only = ~cm[:, 1:].any(dim=1)
    assert wait_only.any() and (~wait_only).any(), "setup: lockout rows and playable rows"
    assert torch.all(probs[wait_only, 0] == 1.0)
    a, logp, ent = pufferlib.pytorch.sample_logits(logits)
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all()
    assert torch.all(cm[torch.arange(len(a)), a.long()]), "sampled actions are legal"
    assert torch.all(ent[wait_only].abs() < 1e-5)
    env.close()


@pytest.mark.parametrize("g", V.GRIDS)
def test_conditional_head_factorisation(pr, obs_by_grid, g):
    """§19.6(c): joint[0] = log P(wait), joint[1+sB+j] = log P(s) + log P(j|s), so exp(joint) is already
    normalised: softmax(joint) = P(s) P(j|s) within 1e-5; P(s) = 0 exactly for slots without a legal
    action; each P(.|s) sums to 1."""
    import torch
    m = E.R()
    obs = obs_by_grid[g]
    env = env_g(g)
    torch.manual_seed(2)
    policy = prt().Policy(env, head="conditional")
    with torch.no_grad():                                      # make the distribution far from uniform
        for p in policy.parameters():
            p.add_(torch.randn(p.shape) * 0.05)
    logits, _ = fwd(policy, obs)
    z = logits.double()
    sm = torch.softmax(z, dim=-1)
    ex = torch.exp(z)
    assert (sm - ex).abs().max().item() <= 1e-5, "softmax(joint) != exp(joint): joint is not log P(s) + log P(j|s)"
    B = V.N_BLOCKS[g]
    cm = np.asarray(m.action_mask(obs, grid=g))
    Ps = sm[:, 1:].reshape(-1, 4, B).sum(dim=2).numpy()
    seg = cm[:, 1:].reshape(-1, 4, B).any(axis=2)
    assert np.all(Ps[~seg] == 0.0) and np.all(Ps[seg] > 0)
    cond = sm[:, 1:].reshape(-1, 4, B).numpy()
    sums = cond.sum(axis=2)[seg] / Ps[seg]
    assert np.allclose(sums, 1.0, atol=1e-6)
    env.close()


def test_conditional_head_starts_near_uniform_over_cards(pr, obs_by_grid):
    """§19.6: final layers initialised with std 0.01 -> at initialisation P(wait) is close to
    1 / (1 + number of playable slots) (loose factor-3 band)."""
    import torch
    m = E.R()
    obs = obs_by_grid[1]
    env = env_g(1)
    torch.manual_seed(0)
    logits, _ = fwd(prt().Policy(env), obs)
    probs = torch.softmax(logits.double(), dim=-1).numpy()
    cm = np.asarray(m.action_mask(obs, grid=1))
    k = cm[:, 1:].reshape(-1, 4, 576).any(axis=2).sum(axis=1)
    rows = k > 0
    ratio = probs[rows, 0] * (1 + k[rows])
    assert np.all((ratio > 1 / 3) & (ratio < 3)), f"P(wait) * (1 + k) at init: {np.round(ratio[:8], 3)}"
    env.close()


@pytest.mark.parametrize("head,g", [("flat", 1), ("conditional", 1), ("conditional", 2), ("conditional", 4)])
def test_gradients_finite_on_train_path(pr, obs_by_grid, head, g):
    """No NaN/inf in gradients (incl. wait-only rows) through log-probs, entropy and value."""
    import torch
    import pufferlib.pytorch
    obs = torch.as_tensor(obs_by_grid[g])
    env = env_g(g)
    torch.manual_seed(3)
    policy = prt().Policy(env, head=head)
    logits, value = policy(obs, dict(action=None, lstm_h=None, lstm_c=None))
    assert logits.shape[-1] == V.N_ACT[g]
    a, logp, ent = pufferlib.pytorch.sample_logits(logits)
    loss = -logp.mean() - 0.01 * ent.mean() + value.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    for n, p in policy.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {n}"
    env.close()


@pytest.mark.parametrize("head,g", [("flat", 1), ("conditional", 1), ("conditional", 4)])
def test_recurrent_train_and_eval_paths(pr, head, g):
    """Recurrent(Policy): training forward on (segments, horizon, OBS) and forward_eval; masks exact,
    no persistent state (same input -> same output across calls)."""
    import torch
    m = E.R()
    env = E.make(num_envs=2, num_agents=2, seed=0, placement_grid=g)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    seq = []
    for _ in range(14):
        seq.append(obs.copy())
        obs, *_ = E.step(env, [E.random_bot_action(rng, r, 0.5) for r in obs])
    batch = torch.as_tensor(np.stack(seq, axis=1))                       # (4, 14, OBS)
    torch.manual_seed(4)
    policy = prt().Recurrent(env, prt().Policy(env, head=head))
    logits, value = policy(batch, dict(action=None, lstm_h=None, lstm_c=None))
    logits = logits.reshape(-1, V.N_ACT[g])
    cm = torch.as_tensor(np.asarray(m.action_mask(batch.reshape(-1, batch.shape[-1]).numpy(), grid=g)))
    assert logits.shape[0] == 56
    minv = torch.finfo(logits.dtype).min
    assert torch.all(logits[~cm] == minv) and torch.all(logits[cm] > minv) and torch.isfinite(value).all()
    (logits[cm].mean() + value.mean()).backward()
    for n, p in policy.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {n}"
    with torch.no_grad():
        again, _ = policy(batch, dict(action=None, lstm_h=None, lstm_c=None))
        assert torch.equal(again.reshape(-1, V.N_ACT[g]), logits.detach()), "state persisted between calls"
        h = torch.zeros(4, policy.hidden_size)
        st = dict(lstm_h=h.clone(), lstm_c=h.clone())
        e1, v1 = policy.forward_eval(batch[:, 5], st)
        st2 = dict(lstm_h=h.clone(), lstm_c=h.clone())
        policy.forward_eval(batch[:, 9], dict(lstm_h=h.clone(), lstm_c=h.clone()))
        e2, v2 = policy.forward_eval(batch[:, 5], st2)
        assert torch.equal(e1, e2) and torch.equal(v1, v2), "forward_eval depends on a previous call"
        cm5 = torch.as_tensor(np.asarray(m.action_mask(batch[:, 5].numpy(), grid=g)))
        assert torch.all(e1[~cm5] == minv) and torch.all(e1[cm5] > minv)
    env.close()


def test_policy_forward_has_no_hidden_state(pr, obs_by_grid):
    import torch
    env = env_g(2)
    policy = prt().Policy(env)
    a, _ = fwd(policy, obs_by_grid[2][:10])
    fwd(policy, obs_by_grid[2][10:30])
    b, _ = fwd(policy, obs_by_grid[2][:10])
    assert torch.equal(a, b)
    env.close()


# ==========================================================================================
# Checkpoints
# ==========================================================================================
CKPT_VARIANTS = [("flat", 1, 1, 32, False), ("conditional", 1, 1, 32, False), ("conditional", 2, 0, 16, False),
                 ("conditional", 4, 1, 8, True), ("flat", 4, 0, 32, True), ("conditional", 1, 1, 32, True)]


@pytest.mark.parametrize("head,g,cs,pc,rnn", CKPT_VARIANTS)
def test_load_policy_round_trips(pr, tmp_path, head, g, cs, pc, rnn):
    """§19.6: policy_kwargs_from_state_dict infers head, card_stats, pos_channels, sizes and grid;
    league.load_policy loads the checkpoint (returns (policy, recurrent)) with identical outputs."""
    import torch
    import pufferroyale.league as lg
    env = env_g(g)
    torch.manual_seed(5)
    policy = prt().Policy(env, head=head, card_stats=cs, pos_channels=pc, hidden_size=64, card_dim=8)
    if rnn:
        policy = prt().Recurrent(env, policy, input_size=64, hidden_size=64)
    path = str(tmp_path / "ckpt.pt")
    torch.save(policy.state_dict(), path)
    kw = lg.policy_kwargs_from_state_dict(torch.load(path))
    assert isinstance(kw, tuple) and len(kw) == 2, "§19.9.8: (policy_kwargs, rnn_kwargs | None)"
    pkw, rkw = kw
    assert (rkw is not None) == rnn
    assert pkw["head"] == head, f"inferred head {pkw['head']} != {head}"
    assert bool(pkw["card_stats"]) == bool(cs)
    if head == "conditional":
        assert int(pkw["pos_channels"]) == pc
    assert int(pkw["placement_grid"]) == g, f"inferred grid {pkw['placement_grid']} != {g}"
    loaded, recurrent = lg.load_policy(path)
    assert bool(recurrent) == rnn
    sd, sd2 = policy.state_dict(), loaded.state_dict()
    assert sd.keys() == sd2.keys() and all(torch.equal(sd[k], sd2[k].to(sd[k].device)) for k in sd)
    obs = real_obs(g=g, steps=12, seed=1)[:8]
    state = dict(lstm_h=None, lstm_c=None)
    with torch.no_grad():
        policy.eval()
        a, _ = policy.forward_eval(torch.as_tensor(obs), dict(state))
        b, _ = loaded.forward_eval(torch.as_tensor(obs), dict(state))
    assert torch.equal(a, b.to(a.device))
    env.close()


@pytest.mark.parametrize("path", V04_CKPTS, ids=["policy_v04", "recurrent_v04"])
def test_old_layout_checkpoints_raise_value_error(pr, path):
    """§19.6: older (v0.4 layout, 17,707-float observation) checkpoints raise a clear ValueError.
    The fixtures were written by the v0.4 code (small non-default sizes)."""
    import pufferroyale.league as lg
    assert os.path.isfile(path)
    with pytest.raises(ValueError):
        lg.load_policy(path)


def test_league_self_opponent_with_coarse_grid(pr):
    """§19.4/§19.6: a "self" opponent in a g = 4 league plays the policy's coarse actions through the
    grid: no illegal actions, cards are played."""
    import pufferroyale.league as lg
    import leaguekit as L
    pool = L.make_pool(anchors=(), self_play_frac=1.0, anchor_frac=0.0, seed=0)
    lv = L.make_league(pool, num_envs=2, seed=0, placement_grid=4, frame_skip=50, log_interval=1)
    policy = prt().Policy(lv.driver_env if hasattr(lv.driver_env, "single_action_space") else env_g(4))
    lv.set_policy(policy)
    lv.async_reset(0)
    m = E.R()
    rng = np.random.default_rng(0)
    plays, logs = 0, 0
    for _ in range(260):
        obs, rew, term, trunc, infos, _, _ = lv.recv()
        acts = []
        for row in obs:
            leg = np.nonzero(np.asarray(m.action_mask(row, grid=4)))[0]
            acts.append(int(rng.choice(leg[1:])) if len(leg) > 1 and rng.random() < 0.5 else 0)
        for i in infos:
            if isinstance(i, dict) and "illegal_actions" in i:
                assert i["illegal_actions"] == 0
                plays += i["plays_0"] + i["plays_1"]
                logs += 1
        lv.send(np.array(acts, np.int32))
    assert logs > 0 and plays > 0
    lv.close()


@pytest.mark.parametrize("head", ["flat", "conditional"])
def test_own_deck_is_encoded(pr, head):
    """§19.6: the card encoding is used for every card id in the observation, incl. own_deck
    (§19.3: CARD_ID_SCALARS gains own_deck): changing only the own_deck field changes the output."""
    import torch
    m = E.R()
    obs = real_obs(g=1, steps=12, seed=2, deck="hog26")[-6:].copy()
    off = m.SCALAR_OFFSET + m.SCALAR_INDEX["own_deck"][0]
    other = obs.copy()
    other[:, off:off + 8] = np.array(sorted(c + 1 for c in (21, 22, 23, 24, 25, 26, 27, 28)), np.float32)
    env = env_g(1)
    torch.manual_seed(6)
    policy = prt().Policy(env, head=head)
    la, va = fwd(policy, obs)
    lb, vb = fwd(policy, other)
    assert not (torch.equal(la, lb) and torch.equal(va, vb)), "own_deck does not reach the policy"
    env.close()
