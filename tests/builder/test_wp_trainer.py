"""Builder tests, SPEC §19.2 (v0.5): MMDPuffeRL evaluate() copy with reward_clip, the entropy
decomposition and split coefficients, grid-aware MMD masks, and the §19.7.1 reward plumbing."""
import math

import numpy as np
import pytest
import torch

import pufferlib.pufferl as pufferl
import pufferlib.pytorch
import pufferroyale
import pufferroyale.torch as prt
from pufferroyale.trainer import MMDPuffeRL, entropy_split, obs_mask, shaping_env_kwargs

SHAPED = dict(reward_tower=0.5, reward_crown=0.5, reward_elixir=0.05, reward_gamma=0.9)


def cfg(tmp_path, **extra):
    c = dict(env="pufferroyale", seed=1, torch_deterministic=True, cpu_offload=False, device="cpu", optimizer="adam",
             precision="float32", total_timesteps=64 * 40, learning_rate=3e-4, anneal_lr=False, min_lr_ratio=0.0,
             gamma=0.9, gae_lambda=0.9, update_epochs=1, clip_coef=0.2, vf_coef=2.0, vf_clip_coef=0.2,
             max_grad_norm=1.5, ent_coef=0.01, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
             data_dir=str(tmp_path), checkpoint_interval=10 ** 9, batch_size=64, minibatch_size=32,
             max_minibatch_size=32, bptt_horizon=16, compile=False, compile_mode="default", vtrace_rho_clip=1.0,
             vtrace_c_clip=1.0, prio_alpha=0.8, prio_beta0=0.2, use_rnn=False, amp=False)
    c.update(extra)
    return c


def initial_state(grid=1, head="conditional"):
    env = pufferroyale.Royale(num_envs=1, num_agents=2, placement_grid=grid)
    torch.manual_seed(0)
    sd = {k: v.clone() for k, v in prt.Policy(env, head=head).state_dict().items()}
    env.close()
    return sd


def build(cls, tmp_path, sd, grid=1, head="conditional", env_kw=None, **extra):
    env = pufferroyale.Royale(num_envs=2, num_agents=2, seed=0, placement_grid=grid, **dict(dict(frame_skip=10),
                                                                                           **(env_kw or {})))
    policy = prt.Policy(env, head=head)
    policy.load_state_dict(sd)
    torch.manual_seed(123)
    quiet = type(cls.__name__, (cls,), {"print_dashboard": lambda self, *a, **k: None})
    tr = quiet(cfg(tmp_path, **extra), env, policy)
    torch.manual_seed(123)
    return tr, policy


def epochs(tr, n):
    out = []
    for _ in range(n):
        tr.last_log_time = -1e18
        tr.evaluate()
        bufs = [b.clone() for b in (tr.observations, tr.actions, tr.logprobs, tr.rewards, tr.terminals, tr.values)]
        tr.train()
        out.append((dict(tr.losses), bufs))
    return out


def close(tr):
    tr.utilization.stop()
    tr.vecenv.close()


def params(policy):
    return torch.cat([p.detach().flatten() for p in policy.parameters()])


@pytest.mark.parametrize("head", ["conditional", "flat"])
def test_defaults_are_puffer_rl_bit_for_bit(tmp_path, head):
    """mmd_coef 0, reward_clip 1, split unset: evaluate() buffers and train() losses / weights equal
    the base PuffeRL's exactly, on shaped rewards (some clamped by the base)."""
    sd = initial_state(head=head)
    runs = []
    for cls, kw in ((pufferl.PuffeRL, {}), (MMDPuffeRL, dict(mmd_coef=0.0, reward_clip=1.0, ent_coef_card=-1.0,
                                                               ent_coef_pos=-1.0))):
        tr, pol = build(cls, tmp_path, sd, head=head, env_kw=SHAPED, **kw)
        try:
            runs.append((epochs(tr, 3), params(pol)))
        finally:
            close(tr)
    (hb, pb), (hm, pm) = runs
    for (lb, bb), (lm, bm) in zip(hb, hm):
        for x, y in zip(bb, bm):
            assert torch.equal(x, y)
        for k, v in lb.items():
            assert v == lm[k] or (math.isnan(v) and math.isnan(lm[k])), k
        assert {"entropy_card", "entropy_pos", "mmd_kl"} <= set(lm)
        assert abs(lm["entropy_card"] + lm["entropy_pos"] - lm["entropy"]) < 1e-4
    assert torch.equal(pb, pm)


def test_reward_clip_values(tmp_path):
    sd = initial_state()
    raw = []
    for clip in (0.0, -1.0, 0.002):
        tr, _ = build(MMDPuffeRL, tmp_path, sd, env_kw=dict(SHAPED, reward_tower=20.0, reward_crown=20.0,
                                                            reward_elixir=5.0, frame_skip=50), reward_clip=clip)
        try:
            vals = []
            for _ in range(4):
                tr.evaluate()
                vals.append(tr.rewards.clone())
                tr.train()
            raw.append(torch.cat([v.flatten() for v in vals]))
        finally:
            close(tr)
    assert raw[0].abs().max() > 1.0, "setup: shaped rewards beyond [-1, 1]"
    assert torch.equal(raw[0], raw[1]), "c <= 0 stores the env rewards unchanged"
    assert raw[2].abs().max() <= 0.002 and torch.equal(raw[2], raw[0].clamp(-0.002, 0.002))


def test_entropy_split_identity_and_edge_rows():
    torch.manual_seed(0)
    for B in (576, 144, 40):
        x = torch.randn(6, 1 + 4 * B) * 3
        x[1, 1 + B:1 + 2 * B] = torch.finfo(torch.float32).min                                         # slot 1 illegal
        x[2, 1:] = torch.finfo(torch.float32).min                                                        # wait only
        x[3, 1 + 3 * B + 5] = 50.0                                                                       # near-deterministic
        _, _, ent = pufferlib.pytorch.sample_logits(x, action=torch.zeros(6, dtype=torch.long))
        hc, hp = entropy_split(x)
        assert torch.allclose(hc + hp, ent, atol=1e-4)
        assert hc[2].abs() < 1e-6 and hp[2].abs() < 1e-6
        # direct definition in float64
        p = torch.softmax(x.double(), -1)
        Ps = p[:, 1:].reshape(6, 4, B).sum(-1)
        Pc = torch.cat([p[:, :1], Ps], -1)
        hc64 = -(torch.where(Pc > 0, Pc * Pc.log(), 0)).sum(-1)
        cond = p[:, 1:].reshape(6, 4, B) / Ps.clamp(min=1e-300)[..., None]
        hs = -(torch.where(cond > 0, cond * cond.log(), 0)).sum(-1)
        hp64 = (Ps * hs).sum(-1)
        assert torch.allclose(hc.double(), hc64, atol=1e-4) and torch.allclose(hp.double(), hp64, atol=1e-4)
    with pytest.raises(ValueError):
        entropy_split(torch.zeros(2, 10))


def test_split_coefficients_rules(tmp_path):
    sd = initial_state()
    for kw in (dict(ent_coef_card=0.01), dict(ent_coef_pos=0.0), dict(ent_coef_card=0.01, ent_coef_pos=-1.0)):
        with pytest.raises(ValueError):
            build(MMDPuffeRL, tmp_path, sd, **kw)
    tr, _ = build(MMDPuffeRL, tmp_path, sd, ent_coef_card=-0.5, ent_coef_pos=-3.0)    # any negative = unset
    assert not tr.entropy_split
    close(tr)


def test_equal_split_coefficients_reproduce_the_default_loss(tmp_path):
    """Both coefficients = ent_coef: same losses within 1e-5 relative on the same batch (lr 0, so
    every epoch sees the same policy and identical buffers)."""
    sd = initial_state()
    res = []
    for kw in ({}, dict(ent_coef_card=0.01, ent_coef_pos=0.01)):
        tr, pol = build(MMDPuffeRL, tmp_path, sd, learning_rate=0.0, **kw)
        try:
            res.append(epochs(tr, 2))
        finally:
            close(tr)
    for (la, _), (lb, _) in zip(*res):
        for k in ("policy_loss", "value_loss", "entropy", "entropy_card", "entropy_pos"):
            assert abs(la[k] - lb[k]) <= 1e-5 * max(1.0, abs(la[k])), k


def test_split_coefficients_change_the_update(tmp_path):
    sd = initial_state()
    ps = []
    for kw in (dict(ent_coef_card=0.01, ent_coef_pos=0.01), dict(ent_coef_card=1.0, ent_coef_pos=0.0)):
        tr, pol = build(MMDPuffeRL, tmp_path, sd, learning_rate=1e-3, **kw)
        try:
            epochs(tr, 1)
            ps.append(params(pol))
        finally:
            close(tr)
    assert (ps[0] - ps[1]).abs().max() > 1e-7


def test_mmd_mask_matches_the_logits_grid(tmp_path):
    sd = initial_state(grid=2)
    tr, pol = build(MMDPuffeRL, tmp_path, sd, grid=2, mmd_coef=1.0, mmd_ref_interval=0, learning_rate=1e-3)
    try:
        hist = epochs(tr, 3)
        kls = [h[0]["mmd_kl"] for h in hist]
        assert all(math.isfinite(k) for k in kls) and max(kls) > 0
        assert all(torch.isfinite(q).all() for q in pol.parameters())
    finally:
        close(tr)
    x = torch.as_tensor(np.zeros((3, 5, pufferroyale.royale.OBS_SIZE), np.float32))
    assert obs_mask(x, 2).shape == (15, 577) and obs_mask(x).shape == (15, 2305)


def test_recurrent_training_with_the_conditional_head(tmp_path):
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0, placement_grid=4)
    torch.manual_seed(0)
    policy = prt.Recurrent(env, prt.Policy(env))
    quiet = type("Q", (MMDPuffeRL,), {"print_dashboard": lambda self, *a, **k: None})
    tr = quiet(cfg(tmp_path, use_rnn=True, mmd_coef=0.5, ent_coef_card=0.01, ent_coef_pos=0.002), env, policy)
    try:
        for losses, _ in epochs(tr, 2):
            assert all(math.isfinite(v) for k, v in losses.items() if k != "explained_variance"), losses
        assert all(torch.isfinite(q).all() for q in policy.parameters())
    finally:
        close(tr)


def test_shaping_env_kwargs_formula():
    kw = shaping_env_kwargs(0.999, 0.5, 1_000_000_000, 1024, 0)
    assert kw == {"reward_gamma": 0.999, "shaping_anneal_steps": math.ceil(0.5 * 1e9 / 1024), "shaping_step_offset": 0}
    assert shaping_env_kwargs(0.99, 0.0, 1000, 4, 999)["shaping_anneal_steps"] == 0
    assert shaping_env_kwargs(0.99, 0.3, 1001, 4, 999) == {"reward_gamma": 0.99, "shaping_anneal_steps": 76,
                                                           "shaping_step_offset": 249}
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            shaping_env_kwargs(0.99, bad, 100, 4)
