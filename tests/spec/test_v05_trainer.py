"""SPEC §19.2 trainer changes in `pufferroyale.trainer.MMDPuffeRL` (v0.5-G) and §19.8.

1. reward_clip (config float, default 1.0): evaluate() = PuffeRL 3.0 evaluate() except rewards are clamped
   to [-c, c] when c > 0 and not clamped when c <= 0; c = 1 -> stored rewards, actions, RNG draws and
   buffers identical to the base class.
2. Entropy decomposition for joint logits over A = 1 + 4B (slot-major): H_card over {wait, 0..3},
   H_pos = sum_s P(s) H(p(.|s)); H_joint = H_card + H_pos; every epoch logs losses/entropy_card and
   losses/entropy_pos (minibatch means) next to entropy.
3. ent_coef_card / ent_coef_pos (default -1 = unset): both >= 0 -> entropy term
   -(c_card mean(H_card) + c_pos mean(H_pos)); exactly one set -> ValueError at construction; both unset
   -> train() bit-identical to v0.4 (and to PuffeRL with mmd_coef = 0); both = ent_coef -> same loss
   within 1e-5 (relative).
4. obs_mask / masked_kl use the action mask matching the logits' width (§19.4).
5. royale.ini [train]: reward_clip = 1.0, ent_coef_card = -1.0, ent_coef_pos = -1.0.
§19.8: MMDPuffeRL with mmd_coef = 0, reward_clip = 1 and the split unset is PuffeRL's PPO bit for bit.

Rewards > 1 in magnitude are produced by a test-side vecenv (a Royale subclass whose recv() substitutes
seeded random rewards and records them), so the clip tests do not depend on the reward-v2 builder work.
The entropy-split tests use a small test-side policy with controlled, masked logits (wait-only rows
during the lockout, slots without any legal action) of width 2305 / 577 / 161.
"""
import configparser
import math
import os

import numpy as np
import pytest

import envkit as E
import helpers as H
import leaguekit as L
import v05kit as V


def trainer_mod():
    import pufferroyale.trainer as mod
    return mod


def rec_env_cls():
    import pufferroyale

    class RecRoyale(pufferroyale.Royale):
        """Royale whose recv() returns seeded N(0, scale^2) rewards (recorded)."""

        def __init__(self, *a, reward_scale=4.0, reward_seed=0, **kw):
            super().__init__(*a, **kw)
            self.recorded = []
            self._r_rng = np.random.default_rng(reward_seed)
            self._scale = reward_scale

        def recv(self):
            out = list(super().recv())
            r = (self._r_rng.standard_normal(np.shape(out[1])) * self._scale).astype(np.float32)
            out[1] = r
            self.recorded.append(r.copy())
            return tuple(out)
    return RecRoyale


def close(tr):
    try:
        tr.utilization.stop()
    except Exception:
        pass
    tr.vecenv.close()


def cfg(tmp_path, batch=64, horizon=16, minibatch=64, total=64 * 50, lr=3e-4, **extra):
    return L.ppo_config(tmp_path, batch_size=batch, horizon=horizon, minibatch=minibatch, total=total, lr=lr, **extra)


# ==========================================================================================
# 1. reward_clip
# ==========================================================================================
def _evaluate_rewards(tmp_path, cls, extra, seed=5):
    import torch
    import pufferroyale.torch as prt
    if cls is trainer_mod().MMDPuffeRL:
        extra = dict(dict(mmd_coef=0.0, mmd_ref_interval=0), **extra)
    env = rec_env_cls()(num_envs=2, num_agents=2, seed=0)
    torch.manual_seed(seed)
    policy = prt.Policy(env)
    tr = cls(cfg(tmp_path, **extra), env, policy)
    torch.manual_seed(seed)
    env.recorded.clear()
    tr.evaluate()
    rec = np.concatenate([r.reshape(-1) for r in env.recorded])
    bufs = {k: getattr(tr, k).detach().cpu().numpy().copy()
            for k in ("observations", "actions", "logprobs", "rewards", "terminals", "values")}
    close(tr)
    return rec, bufs


@pytest.mark.parametrize("c", [0.0, -1.0, 0.5, 1.0, 2.5, None])
def test_reward_clip_values(pr, tmp_path, c):
    """§19.2.1: stored rewards = clamp(env rewards, -c, c) for c > 0, unchanged for c <= 0;
    absent key = 1.0."""
    extra = {} if c is None else dict(reward_clip=c)
    rec, bufs = _evaluate_rewards(tmp_path, trainer_mod().MMDPuffeRL, extra)
    stored = np.sort(bufs["rewards"].reshape(-1))
    assert len(rec) == stored.size, f"{len(rec)} recorded rewards vs {stored.size} stored"
    eff = 1.0 if c is None else c
    want = np.sort(rec if eff <= 0 else np.clip(rec, -eff, eff)).astype(np.float32)
    assert np.abs(rec).max() > 3.0, "setup: the env must produce |r| > 1"
    assert np.array_equal(stored, want), f"reward_clip={c}: stored rewards are not clamp(env, -c, c)"


def test_reward_clip_one_is_puffer_rl_bit_for_bit(pr, tmp_path):
    """§19.2.1: with c = 1 the stored rewards, actions, RNG draws and buffers equal the base class."""
    import pufferlib.pufferl as pufferl
    _, base = _evaluate_rewards(tmp_path, pufferl.PuffeRL, {})
    for extra in ({}, dict(reward_clip=1.0)):
        _, mine = _evaluate_rewards(tmp_path, trainer_mod().MMDPuffeRL, dict(extra, mmd_coef=0.0, mmd_ref_interval=0))
        for k in base:
            assert np.array_equal(base[k], mine[k]), f"evaluate() buffer {k!r} differs from PuffeRL ({extra})"


# ==========================================================================================
# 2./3. Entropy split
# ==========================================================================================
def make_masked_policy(A, seed=0, scale=0.3):
    """Test-side policy with controlled, near-uniform masked logits over A actions (so H_card and H_pos
    and their gradients are substantial): logits = Linear(tanh(features)) with small weights; masks:
    - lockout rows: wait only;
    - otherwise slot s is legal iff (hand_id[s] + s + step) % 3 != 0 (a deterministic pattern that
      mixes empty and legal slots in the same row), and inside a legal slot a pseudo-random subset of
      tiles plus the slot's first tile is legal."""
    import torch
    import torch.nn as nn
    m = E.R()
    lay = E.scalar_layout()
    lock = m.SCALAR_OFFSET + lay["lockout"][0]
    hand = m.SCALAR_OFFSET + lay["hand"][0]
    tick = m.SCALAR_OFFSET + lay["tick"][0]
    idx = list(range(m.SCALAR_OFFSET, m.SCALAR_OFFSET + 18)) + [m.SPATIAL_OFFSET + 100, m.ENTITY_OFFSET + 1]
    B = (A - 1) // 4

    class MaskedPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            k = len(idx) + 1
            self.lin = nn.Linear(k, A)
            self.val = nn.Linear(k, 1)
            with torch.no_grad():
                self.lin.weight.copy_(torch.randn(A, k, generator=g) * scale)
                self.lin.bias.copy_(torch.randn(A, generator=g) * scale)
                self.val.weight.copy_(torch.randn(1, k, generator=g) * 0.1)
                self.val.bias.zero_()
            self.register_buffer("tile_u", torch.randn(k, A, generator=g))
            self.register_buffer("idx", torch.tensor(idx))
            self.is_continuous = False

        def feats(self, obs):
            x = torch.tanh(obs.reshape(-1, obs.shape[-1])[:, self.idx].float() / 8.0)
            return torch.cat([x, torch.ones_like(x[:, :1])], dim=1)

        def forward_eval(self, obs, state=None):
            flat = obs.reshape(-1, obs.shape[-1]).float()
            x = self.feats(obs)
            logits = self.lin(x)
            ids = torch.round(flat[:, hand:hand + 4]).long()
            step = torch.round(flat[:, tick] * 600).long()
            slot_ok = (ids + torch.arange(4)[None, :] + step[:, None]) % 3 != 0
            tile_ok = (x @ self.tile_u) > 0.0
            tile_ok[:, 1::B] = True                                       # first tile of every slot
            legal = torch.cat([torch.ones_like(tile_ok[:, :1]),
                               tile_ok[:, 1:] & slot_ok.repeat_interleave(B, dim=1)], dim=1)
            locked = flat[:, lock] > 0.5
            legal[locked, 1:] = False                                     # wait-only rows
            logits = torch.where(legal, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))
            return logits, self.val(x)

        def forward(self, obs, state=None):
            return self.forward_eval(obs, state)
    return MaskedPolicy()


def _split_epoch(tmp_path, A, extra, lr=0.0, seed=3, policy=None, second=True):
    """One logged epoch (losses of that epoch) and, with `second`, another one whose returned logs
    carry the first epoch's losses (PuffeRL publishes the previous losses in its logs)."""
    import torch
    import pufferroyale
    env = pufferroyale.Royale(num_envs=2, num_agents=2, seed=1)
    policy = policy or make_masked_policy(A, seed=seed)
    tr = trainer_mod().MMDPuffeRL(cfg(tmp_path, lr=lr, **extra), env, policy)
    torch.manual_seed(seed)
    tr.last_log_time = -1e18
    tr.evaluate()
    obs = tr.observations.detach().clone()
    tr.train()
    losses = dict(tr.losses)
    logs = None
    if second:
        tr.last_log_time = -1e18
        tr.evaluate()
        logs = tr.train()
    close(tr)
    return losses, obs, policy, logs


@pytest.mark.parametrize("A", [2305, 577, 161])
def test_entropy_split_logged_and_matches_reference(pr, tmp_path, A):
    """§19.2.2: losses/entropy_card and losses/entropy_pos are the minibatch means of the SPEC
    definitions (one minibatch = the whole batch, lr = 0, so they are the batch means of the
    reference values) and entropy = entropy_card + entropy_pos."""
    import torch
    losses, obs, policy, logs = _split_epoch(tmp_path, A, dict(mmd_coef=0.0, mmd_ref_interval=0))
    for k in ("entropy", "entropy_card", "entropy_pos"):
        assert k in losses, f"trainer.losses lacks {k!r}: {sorted(losses)}"
    assert logs is not None and "losses/entropy_card" in logs and "losses/entropy_pos" in logs
    with torch.no_grad():
        logits, _ = policy.forward_eval(obs.reshape(-1, obs.shape[-1]))
    hj, hc, hp = V.entropy_split(logits.numpy())
    assert np.allclose(hj, hc + hp, atol=1e-9), "tester reference identity"
    lg = logits.numpy()
    B = (A - 1) // 4
    legal = lg > np.finfo(np.float32).min
    wait_only = ~legal[:, 1:].any(axis=1)
    seg = legal[:, 1:].reshape(-1, 4, B).any(axis=2)
    assert wait_only.any(), "setup: wait-only rows (lockout) must be in the batch"
    assert ((~seg).any(axis=1) & seg.any(axis=1)).any(), "setup: rows with an empty slot next to a legal one"
    assert hc[~wait_only].mean() > 0.3 and hp[~wait_only].mean() > 0.3, \
        f"setup: the test policy must be far from deterministic (H_card {hc.mean():.3g}, H_pos {hp.mean():.3g})"
    for k, ref in (("entropy", hj.mean()), ("entropy_card", hc.mean()), ("entropy_pos", hp.mean())):
        assert abs(losses[k] - ref) <= 1e-4 * max(1.0, abs(ref)), f"A={A}: {k} {losses[k]} vs SPEC {ref}"
    assert abs(losses["entropy"] - losses["entropy_card"] - losses["entropy_pos"]) <= 1e-4 * max(1, losses["entropy"])


@pytest.mark.parametrize("card,pos", [(0.01, -1.0), (-1.0, 0.02), (0.0, -1.0), (0.3, -0.5), (-2.0, 0.0)])
def test_exactly_one_split_coefficient_is_an_error(pr, tmp_path, card, pos):
    import pufferroyale
    import pufferroyale.torch as prt
    env = pufferroyale.Royale(num_envs=2, num_agents=2)
    with pytest.raises(ValueError):
        trainer_mod().MMDPuffeRL(cfg(tmp_path, ent_coef_card=card, ent_coef_pos=pos), env, prt.Policy(env))
    env.close()


def test_any_negative_split_value_means_unset(pr, tmp_path):
    """§19.9.11: any negative value means unset: (-0.5, -2.0) constructs and trains like the default."""
    d_def = _update(tmp_path, 577, dict(ent_coef=0.2))
    d_neg = _update(tmp_path, 577, dict(ent_coef=0.2, ent_coef_card=-0.5, ent_coef_pos=-2.0))
    assert np.array_equal(d_def, d_neg)


def _update(tmp_path, A, extra, seed=4):
    """One epoch with a linear-in-gradient optimizer step (Adam with huge eps: delta ~ -lr g / eps), so
    parameter deltas compare the losses' gradients."""
    import torch
    policy = make_masked_policy(A, seed=seed)
    before = torch.cat([p.detach().flatten().clone() for p in policy.parameters()])
    base = dict(mmd_coef=0.0, mmd_ref_interval=0, adam_eps=1e4, max_grad_norm=1e12)
    base.update(extra)
    _split_epoch(tmp_path, A, base, lr=1e4, seed=seed, policy=policy, second=False)
    after = torch.cat([p.detach().flatten() for p in policy.parameters()])
    return (after - before).numpy().astype(np.float64)


@pytest.mark.parametrize("A", [2305, 577, 161])
def test_split_equal_to_ent_coef_reproduces_the_default_loss(pr, tmp_path, A):
    """§19.2.3: ent_coef_card = ent_coef_pos = ent_coef gives the default loss (within 1e-5 relative):
    observed through the gradient of one update; different split coefficients change it."""
    d_def = _update(tmp_path, A, dict(ent_coef=0.2))
    d_eq = _update(tmp_path, A, dict(ent_coef=0.2, ent_coef_card=0.2, ent_coef_pos=0.2))
    scale = np.abs(d_def).max()
    assert scale > 0
    d_none = _update(tmp_path, A, dict(ent_coef=0.0))
    assert np.abs(d_def - d_none).max() > 1e-2 * scale, "setup: the entropy term must matter in this update"
    assert np.abs(d_def - d_eq).max() <= 1e-4 * scale, \
        f"split = ent_coef changed the update by {np.abs(d_def - d_eq).max():.3e} (scale {scale:.3e})"
    d_card = _update(tmp_path, A, dict(ent_coef=0.2, ent_coef_card=0.4, ent_coef_pos=0.0))
    d_pos = _update(tmp_path, A, dict(ent_coef=0.2, ent_coef_card=0.0, ent_coef_pos=0.4))
    assert np.abs(d_card - d_pos).max() > 1e-3 * scale, "the split coefficients must change the loss"


def test_split_replaces_ent_coef(pr, tmp_path):
    """§19.2.3: with the split set, the entropy term is -(c_card H_card + c_pos H_pos) INSTEAD of
    -ent_coef H_joint: ent_coef no longer matters."""
    a = _update(tmp_path, 577, dict(ent_coef=0.0, ent_coef_card=0.0, ent_coef_pos=0.0))
    b = _update(tmp_path, 577, dict(ent_coef=0.5, ent_coef_card=0.0, ent_coef_pos=0.0))
    assert np.abs(a - b).max() <= 1e-7 * max(1.0, np.abs(a).max())


# ==========================================================================================
# §19.8 / §19.2.3 bit-identity with PuffeRL
# ==========================================================================================
def test_defaults_train_is_puffer_rl_bit_for_bit(pr, tmp_path):
    """§19.8: mmd_coef = 0, reward_clip = 1, split unset -> PuffeRL's PPO bit for bit (params, buffers,
    losses over 3 epochs; extra entropies computed without gradients and without RNG)."""
    import torch
    import pufferlib.pufferl as pufferl
    import pufferroyale
    import pufferroyale.torch as prt
    env0 = pufferroyale.Royale(num_envs=1, num_agents=2)
    torch.manual_seed(0)
    sd = {k: v.clone() for k, v in prt.Policy(env0).state_dict().items()}
    env0.close()
    runs = []
    for cls, extra in ((pufferl.PuffeRL, {}),
                       (trainer_mod().MMDPuffeRL, dict(mmd_coef=0.0, mmd_ref_interval=0)),
                       (trainer_mod().MMDPuffeRL, dict(mmd_coef=0.0, mmd_ref_interval=0, reward_clip=1.0,
                                                       ent_coef_card=-1.0, ent_coef_pos=-1.0))):
        env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0, reward_tower=2.0, reward_crown=2.0)
        policy = prt.Policy(env)
        policy.load_state_dict(sd)
        torch.manual_seed(123)
        tr = cls(cfg(tmp_path, batch=64, horizon=16, minibatch=32, **extra), env, policy)
        torch.manual_seed(123)
        hist = []
        for _ in range(3):
            tr.last_log_time = -1e18
            tr.evaluate()
            tr.train()
            hist.append({k: v for k, v in tr.losses.items() if k in ("policy_loss", "value_loss", "entropy",
                                                                      "approx_kl", "clipfrac", "importance")})
        params = torch.cat([p.detach().flatten() for p in policy.parameters()]).numpy().copy()
        rewards = tr.rewards.detach().numpy().copy()
        close(tr)
        runs.append((hist, params, rewards))
    for hist, params, rewards in runs[1:]:
        assert hist == runs[0][0], "losses differ from PuffeRL"
        assert np.array_equal(params, runs[0][1]), "parameters differ from PuffeRL after 3 epochs"
        assert np.array_equal(rewards, runs[0][2])


# ==========================================================================================
# 4. MMD with coarse-grid policies
# ==========================================================================================
@pytest.mark.parametrize("g", [2, 4])
def test_mmd_kl_with_grid_policies(pr, tmp_path, g):
    """§19.2.4: the KL uses the mask matching the logits' width: pi == pi_ref -> KL exactly ~0 and finite;
    after a perturbation KL > 0 and finite."""
    import torch
    import pufferroyale
    import pufferroyale.torch as prt
    env = pufferroyale.Royale(num_envs=2, num_agents=2, seed=0, placement_grid=g)
    policy = prt.Policy(env)
    assert int(env.single_action_space.n) == V.N_ACT[g]
    tr = trainer_mod().MMDPuffeRL(cfg(tmp_path, lr=0.0, mmd_coef=1.0, mmd_ref_interval=0), env, policy)
    torch.manual_seed(0)
    try:
        tr.last_log_time = -1e18
        tr.evaluate()
        assert int(tr.actions.max()) < V.N_ACT[g], "sampled actions must lie in the coarse action space"
        with torch.no_grad():
            lg, _ = policy.forward_eval(tr.observations[0, :4], dict(lstm_h=None, lstm_c=None))
        assert lg.shape[-1] == V.N_ACT[g], f"logits width {lg.shape[-1]} != {V.N_ACT[g]}"
        tr.train()
        kl0 = tr.losses["mmd_kl"]
        assert math.isfinite(kl0) and abs(kl0) < 1e-6, f"grid {g}: KL(pi || pi_ref) with pi == pi_ref = {kl0}"
        gen = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(torch.randn(p.shape, generator=gen) * 0.05)
        tr.last_log_time = -1e18
        tr.evaluate()
        tr.train()
        kl1 = tr.losses["mmd_kl"]
        assert math.isfinite(kl1) and kl1 > 1e-5, f"grid {g}: KL after perturbation {kl1}"
        for k, v in tr.losses.items():
            if k != "explained_variance":
                assert math.isfinite(float(v)), f"loss {k} = {v}"
    finally:
        close(tr)


# ==========================================================================================
# 5. royale.ini
# ==========================================================================================
def test_royale_ini_train_and_policy_keys(pr):
    cp = configparser.ConfigParser()
    cp.read(os.path.join(H.ROOT, "pufferroyale", "config", "royale.ini"))
    tr = cp["train"]
    for k, v in (("reward_clip", 1.0), ("ent_coef_card", -1.0), ("ent_coef_pos", -1.0)):
        assert k in tr, f"royale.ini [train] lacks {k} (§19.2.5)"
        assert float(tr[k]) == v, f"[train] {k} = {tr[k]}"
    pol = cp["policy"]
    assert pol.get("head", "").strip() == "conditional", "royale.ini [policy] head = conditional (§19.6)"
    assert int(pol.get("card_stats", "-1")) == 1 and int(pol.get("pos_channels", "-1")) == 32
