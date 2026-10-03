"""Builder tests, SPEC §19.6 / §19.4 (v0.5): policy heads, card stats, placement-grid masks and
pooling, Recurrent, checkpoint inference, grid play in the tools (Game / LeagueVecEnv)."""
import numpy as np
import pytest
import torch

import pufferlib.pytorch
import pufferroyale
from pufferroyale import binding
from pufferroyale import royale as R
import pufferroyale.torch as prt

NEG = torch.finfo(torch.float32).min


def bot_states(grid=1, steps=70, seed=0, num_envs=3):
    """Observations of real states (heuristic bots on both seats, fine actions)."""
    env = pufferroyale.Royale(num_envs=num_envs, num_agents=2, seed=seed, placement_grid=grid, deck0="random",
                              deck1="random")
    obs, _ = env.reset(seed=seed)
    for i in range(num_envs):
        env.set_row_grid(i, 0, 1)
        env.set_row_grid(i, 1, 1)
    bots = [pufferroyale.Bot("heuristic", seed=seed * 10 + r) for r in range(2 * num_envs)]
    out = []
    for t in range(steps):
        acts = np.array([bots[r].act_env(env, r % 2, r // 2) for r in range(2 * num_envs)], np.int32)
        obs, *_ = env.step(acts)
        if t % 10 == 9:
            out.append(obs.copy())
    return env, np.concatenate(out)


@pytest.fixture(scope="module")
def states():
    out = {}
    for g in (1, 2, 4):
        env, obs = bot_states(g)
        out[g] = (env, obs)
    yield out
    for env, _ in out.values():
        env.close()


@pytest.mark.parametrize("grid", [1, 2, 4])
@pytest.mark.parametrize("head", ["flat", "conditional"])
def test_heads_masks_and_finiteness(states, grid, head):
    env, obs = states[grid]
    torch.manual_seed(0)
    pol = prt.Policy(env, head=head)
    assert pol.placement_grid == grid and pol.n_actions == R.n_actions(grid)
    x = torch.as_tensor(obs)
    logits, value = pol.forward_eval(x)
    mask = prt.action_mask(x, grid)
    assert torch.equal(mask, torch.as_tensor(R.action_mask(obs, grid)))
    assert logits.shape == (len(obs), R.n_actions(grid)) and value.shape == (len(obs), 1)
    assert (logits[~mask] == NEG).all(), "illegal logits are exactly finfo.min"
    p = torch.softmax(logits, -1)
    assert (p[mask] > 0).all() and (p[~mask] == 0).all()
    assert torch.allclose(p.sum(-1), torch.ones(len(obs)), atol=1e-5)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    a, lp, ent = pufferlib.pytorch.sample_logits(logits)
    assert mask[torch.arange(len(obs)), a].all()
    (lp.mean() + ent.mean() + value.mean()).backward()
    for n, q in pol.named_parameters():
        assert q.grad is None or torch.isfinite(q.grad).all(), n


@pytest.mark.parametrize("grid", [1, 2, 4])
def test_conditional_factorisation(states, grid):
    env, obs = states[grid]
    torch.manual_seed(1)
    pol = prt.Policy(env)
    with torch.no_grad():                                  # break the near-uniform init
        for q in pol.parameters():
            q.add_(torch.randn_like(q) * 0.05)
        x = torch.as_tensor(obs)
        parts = pol.conditional_parts(x)
        logits, _ = pol.forward_eval(x)
    assert torch.equal(parts["logits"], logits)
    N, B = len(obs), pol.n_blocks
    pc, pp = parts["card_logp"].exp(), parts["pos_logp"].exp()
    p = torch.softmax(logits, -1)
    assert torch.allclose(p[:, 0], pc[:, 0], atol=1e-5)
    joint = (pc[:, 1:, None] * pp).reshape(N, 4 * B)
    m = parts["mask"][:, 1:]
    assert (p[:, 1:] - torch.where(m, joint, 0)).abs().max() < 1e-5
    # a slot is masked in the card distribution iff its segment has no legal action
    seg = m.reshape(N, 4, B).any(-1)
    assert torch.equal(pc[:, 1:] > 0, seg)
    # P(j | s) sums to 1 over each legal slot's legal blocks
    assert torch.allclose((pp * m.reshape(N, 4, B)).sum(-1)[seg], torch.ones(int(seg.sum())), atol=1e-5)


def test_wait_only_rows():
    env = pufferroyale.Royale(num_envs=2, num_agents=2, seed=0)
    obs, _ = env.reset(seed=0)                             # deploy lockout: only the no-op is legal
    assert (R.mask(obs)[:, 1:] == 0).all()
    for head in ("flat", "conditional"):
        for grid in (1,):
            pol = prt.Policy(env, head=head)
            x = torch.as_tensor(obs)
            logits, _ = pol.forward_eval(x)
            p = torch.softmax(logits, -1)
            assert (p[:, 0] == 1).all() and (p[:, 1:] == 0).all()
            _, lp, ent = pufferlib.pytorch.sample_logits(logits)
            assert torch.isfinite(ent).all() and ent.abs().max() < 1e-6 and lp.abs().max() < 1e-6
            from pufferroyale.trainer import entropy_split
            hc, hp = entropy_split(logits)
            assert torch.isfinite(hc).all() and torch.isfinite(hp).all() and hc.abs().max() < 1e-6
    env.close()


@pytest.mark.parametrize("head", ["flat", "conditional"])
def test_recurrent_forward_and_eval(states, head):
    env, obs = states[2]
    torch.manual_seed(0)
    rec = prt.Recurrent(env, prt.Policy(env, head=head))
    assert prt.policy_grid(rec) == 2
    x = torch.as_tensor(obs[:12])
    st = {"lstm_h": None, "lstm_c": None}
    lg, v = rec.forward_eval(x, st)
    mask = prt.action_mask(x, 2)
    assert lg.shape == (12, R.n_actions(2)) and (lg[~mask] == NEG).all() and st["lstm_h"].shape == (12, 256)
    seq = x.reshape(3, 4, -1)                              # (segments, horizon, OBS)
    st2 = {"lstm_h": None, "lstm_c": None}
    lg2, v2 = rec.forward(seq, st2)
    assert lg2.shape == (12, R.n_actions(2)) and v2.shape == (3, 4)
    assert (lg2[~prt.action_mask(seq, 2).reshape(12, -1)] == NEG).all()
    _, lp, ent = pufferlib.pytorch.sample_logits(lg2)
    (lp.mean() + ent.mean() + v2.mean()).backward()
    assert all(q.grad is None or torch.isfinite(q.grad).all() for q in rec.parameters())
    # stateless: the same call twice gives the same logits
    lg3, _ = rec.forward(seq, {"lstm_h": None, "lstm_c": None})
    assert torch.equal(lg2, lg3)


def test_conditional_decode_needs_observations(states):
    env, obs = states[1]
    pol = prt.Policy(env)
    h = pol.encode_observations(torch.as_tensor(obs[:2]))
    with pytest.raises(RuntimeError):
        pol.decode_actions(h)
    flat = prt.Policy(env, head="flat")
    lg, _ = flat.decode_actions(flat.encode_observations(torch.as_tensor(obs[:2])))
    assert lg.shape == (2, 2305) and (lg > NEG).all(), "bare protocol: unmasked logits"


def test_heatmap_pooling_averages_existing_tiles():
    for g in (2, 4):
        cells, pool = prt._blocks(g)
        rows, cols = R.grid_shape(g)
        m = np.arange(576, dtype=np.float64)
        pooled = m @ pool
        for by in range(rows):
            for bx in range(cols):
                ty = range(g * by, min(32, g * by + g))
                tx = range(g * bx, min(18, g * bx + g))
                want = np.mean([y * 18 + x for y in ty for x in tx])
                assert abs(pooled[by * cols + bx] - want) < 1e-3
        assert np.allclose(pool.sum(0), 1.0)


def test_policy_rejects_inconsistent_grid_and_head(states):
    env, _ = states[2]
    with pytest.raises(ValueError):
        prt.Policy(env, placement_grid=1)
    with pytest.raises(ValueError):
        prt.Policy(env, head="joint")
    assert prt.Policy(env, placement_grid=2).placement_grid == 2


def test_card_stat_table_matches_card_info():
    S, names = prt.card_stat_table()
    assert S.shape == (R.CARD_SLOTS + 1, len(names)) == (129, 24) and names == list(prt.CARD_STAT_NAMES)
    assert np.isfinite(S).all() and S.min() >= 0 and S.max() <= 1
    assert (S[0] == 0).all() and (S[R.N_CARDS + 1:] == 0).all()
    raw = np.array([prt.card_stat_row(binding.card_info(c)) for c in range(R.N_CARDS)])
    for j, name in enumerate(names):
        col = np.log1p(raw[:, j]) if name in prt.CARD_STAT_LOG else raw[:, j]
        want = col / col.max() if col.max() > 0 else col
        assert np.allclose(S[1:R.N_CARDS + 1, j], want), name
        assert S[1:R.N_CARDS + 1, j].max() in (0.0, 1.0), name
    col = {n: j for j, n in enumerate(names)}
    info = {c: binding.card_info(c) for c in range(R.N_CARDS)}
    for c, d in info.items():
        row = S[c + 1]
        assert row[col["elixir"]] == pytest.approx(d["elixir"] / max(i["elixir"] for i in info.values()))
        assert row[col["is_spell"]] == float(d["card_kind"] == "spell")
        if d["card_kind"] == "spell":                       # SPEC §19.9.7
            for n in ("units", "hitpoints", "hit_speed_ms", "dps", "range_milli", "speed", "flying", "lifetime_ms",
                      "deploy_time_ms", "jumps", "charges"):
                assert row[col[n]] == 0, (d["name"], n)
            assert (row[col["spawns_units"]] == 1) == bool(d.get("spawn"))
    hog = info[4]
    assert S[5, col["jumps"]] == 1 and S[5, col["target_only_buildings"]] == 1 and hog["jumps"]
    # the table is a non-persistent buffer: not in the state dict, but used by enc()
    pol = prt.Policy(None)
    assert "card_stat_table" not in pol.state_dict() and torch.equal(pol.card_stat_table, torch.as_tensor(S).float())
    assert pol.encode_cards(torch.tensor([5])).shape == (1, 32)
    assert prt.Policy(None, card_stats=0).encode_cards(torch.tensor([5])).shape == (1, 16)


def _ckpt(tmp_path, name, grid=1, recurrent=False, **kw):
    from pufferroyale.league import _obs_stub
    torch.manual_seed(3)
    pol = prt.Policy(_obs_stub(R.n_actions(grid)), **kw)
    if recurrent:
        pol = prt.Recurrent(_obs_stub(R.n_actions(grid)), pol)
    path = str(tmp_path / name)
    torch.save(pol.state_dict(), path)
    return path, pol


@pytest.mark.parametrize("variant", [
    dict(), dict(head="flat"), dict(card_stats=0), dict(pos_channels=8, hidden_size=64), dict(grid=2),
    dict(grid=4, head="flat"), dict(grid=2, recurrent=True), dict(recurrent=True, head="flat", card_stats=0)])
def test_checkpoint_inference_round_trip(tmp_path, variant):
    from pufferroyale import league
    path, pol = _ckpt(tmp_path, "c.pt", **variant)
    kw, rnn = league.policy_kwargs_from_state_dict(torch.load(path, weights_only=True))
    assert kw["head"] == variant.get("head", "conditional")
    assert kw["card_stats"] == variant.get("card_stats", 1)
    assert kw["placement_grid"] == variant.get("grid", 1)
    assert (rnn is not None) == variant.get("recurrent", False)
    if kw["head"] == "conditional":
        assert kw["pos_channels"] == variant.get("pos_channels", 32)
    loaded, rec = league.load_policy(path)
    assert rec == variant.get("recurrent", False) and prt.policy_grid(loaded) == variant.get("grid", 1)
    ref = pol.state_dict()
    assert all(torch.equal(ref[k], v) for k, v in loaded.state_dict().items())


def test_grid_buffer_mismatch_and_old_layout_errors(tmp_path):
    from pufferroyale import league
    from pufferroyale.league import _obs_stub
    path, _ = _ckpt(tmp_path, "g2.pt", grid=2)
    with pytest.raises(RuntimeError, match="placement_grid"):
        prt.Policy(_obs_stub()).load_state_dict(torch.load(path, weights_only=True))
    # a v0.4-layout checkpoint: flat head, no card stats, no grid buffer, 17,707-wide observation
    sd = prt.Policy(None, head="flat", card_stats=0).state_dict()
    sd.pop("action_grid")
    w = sd["scalars.0.weight"]
    sd["scalars.0.weight"] = w[:, :w.shape[1] - 8 * 16]
    torch.save(sd, str(tmp_path / "v04.pt"))
    with pytest.raises(ValueError, match="v0.5"):
        league.load_policy(str(tmp_path / "v04.pt"))


def test_load_weights_checks_the_architecture(tmp_path):
    from pufferroyale import league
    from pufferroyale.league import _obs_stub
    path, src = _ckpt(tmp_path, "a.pt")
    dst = prt.Policy(_obs_stub())
    league.load_weights(dst, path)
    assert all(torch.equal(v, dst.state_dict()[k]) for k, v in src.state_dict().items())
    for other in (prt.Policy(_obs_stub(), head="flat"), prt.Policy(_obs_stub(R.n_actions(2))),
                  prt.Recurrent(_obs_stub(), prt.Policy(_obs_stub())), prt.Policy(_obs_stub(), card_stats=0)):
        with pytest.raises(ValueError, match="architecture mismatch"):
            league.load_weights(other, path)


def test_game_play_honours_the_checkpoint_grid(tmp_path):
    """metagame / tournament: a grid-4 checkpoint's coarse actions are mapped through
    Game.coarse_to_fine, so every play it makes is accepted and it plays at all."""
    from pufferroyale import metagame as mg
    path, _ = _ckpt(tmp_path, "g4.pt", grid=4)
    kind, (pol, rec) = mg.resolve_agent(path)
    player = mg._PolicyPlayer(pol, rec, greedy=False, seed=0)
    assert player.grid == 4
    g = pufferroyale.Game(seed=1)
    plays = 0
    for t in range(300):
        if g.state()["over"]:
            break
        a = player.act(g, 0)
        if a:
            assert a < R.MASK_SIZE and g.legal_mask(0)[a] == 1, "a mapped play must be legal"
            assert g.play_action(0, a) == 0
            plays += 1
        g.tick(10)
    assert plays > 0
    assert mg.play_match(path, "bot:random", "hog26", "hog26", seed=3, greedy=False) in (-1, 0, 1)


def test_league_ckpt_opponent_plays_on_its_own_grid(tmp_path):
    """A grid-2 checkpoint anchor next to a grid-1 learner: its env row decodes grid-2 actions."""
    from pufferroyale import league
    path, _ = _ckpt(tmp_path, "g2.pt", grid=2)
    pool = league.OpponentPool(anchors=(f"ckpt:{path}",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = league.LeagueVecEnv(pool, num_envs=2, seed=0, frame_skip=20, log_interval=10 ** 9)
    lv.reset(seed=0)
    for i in range(2):
        seat = int(lv.seats[i])
        grids = lv.env.env_info(i)["row_grids"]
        assert grids[seat] == 1 and grids[1 - seat] == 2
    for _ in range(40):
        lv.step(np.zeros(2, np.int32))
    lv.close()
