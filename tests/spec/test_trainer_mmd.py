"""SPEC §15.3 / §15.7.8 `pufferroyale.trainer.MMDPuffeRL`: PPO + mmd_coef * KL(pi_theta || pi_ref).

MMDPuffeRL(config, vecenv, policy, logger=None) reads config["mmd_coef"] and
config["mmd_ref_interval"]; the reference is refreshed after train() increments `epoch`, when
interval > 0 and epoch % interval == 0; losses/mmd_kl is the mean over the epoch's minibatches.

PuffeRL only copies the epoch's losses into `trainer.losses` when it logs (every 0.25 s); the tests
force a log on every epoch by rewinding the base-class `last_log_time`.
"""
import math

import numpy as np
import pytest

import leaguekit as L


def trainer_mod():
    import pufferroyale.trainer as mod
    return mod


def init_state(seed=0):
    import torch
    import pufferroyale
    import pufferroyale.torch as prt
    env = pufferroyale.Royale(num_envs=1, num_agents=2)
    torch.manual_seed(seed)
    sd = {k: v.clone() for k, v in prt.Policy(env).state_dict().items()}
    env.close()
    return sd


def build(cls, tmp_path, sd, mmd_coef=None, interval=None, lr=3e-4, seed=123):
    import torch
    import pufferroyale
    import pufferroyale.torch as prt
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0)
    policy = prt.Policy(env)
    policy.load_state_dict(sd)
    extra = {}
    if mmd_coef is not None:
        extra = dict(mmd_coef=mmd_coef, mmd_ref_interval=interval)
    cfg = L.ppo_config(tmp_path, batch_size=64, horizon=16, minibatch=32, total=64 * 50, lr=lr, **extra)
    torch.manual_seed(seed)
    tr = cls(cfg, env, policy)
    torch.manual_seed(seed)                    # identical RNG state from here on, whatever the ctor used
    return tr, policy


def epoch(tr):
    tr.last_log_time = -1e18                   # force PuffeRL to publish this epoch's losses
    tr.evaluate()
    logs = tr.train()
    return dict(tr.losses), logs


def perturb(policy, scale=0.05, seed=7):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)


def close(tr):
    tr.utilization.stop()
    tr.vecenv.close()


def test_mmd_is_a_puffer_rl_subclass(pr):
    import pufferlib.pufferl as pufferl
    assert issubclass(trainer_mod().MMDPuffeRL, pufferl.PuffeRL)


def test_kl_zero_when_policy_equals_reference(pr, tmp_path):
    tr, _ = build(trainer_mod().MMDPuffeRL, tmp_path, init_state(), mmd_coef=1.0, interval=0, lr=0.0)
    try:
        for _ in range(2):
            losses, _ = epoch(tr)
            assert "mmd_kl" in losses, f"losses/mmd_kl must be logged: {sorted(losses)}"
            assert abs(losses["mmd_kl"]) < 1e-6, f"KL(pi || pi_ref) with pi == pi_ref: {losses['mmd_kl']}"
    finally:
        close(tr)


def test_kl_positive_after_perturbation(pr, tmp_path):
    tr, policy = build(trainer_mod().MMDPuffeRL, tmp_path, init_state(), mmd_coef=1.0, interval=0, lr=0.0)
    try:
        perturb(policy)
        losses, _ = epoch(tr)
        assert losses["mmd_kl"] > 1e-4, f"KL must be > 0 once pi differs from pi_ref: {losses['mmd_kl']}"
        assert math.isfinite(losses["mmd_kl"])
    finally:
        close(tr)


def test_gradient_flows_through_kl(pr, tmp_path):
    """Identical runs except mmd_coef (0 vs 5): the updates must differ, so the KL term is part of
    the differentiated loss (not detached)."""
    import torch
    sd = init_state()
    params = []
    for coef in (0.0, 5.0):
        tr, policy = build(trainer_mod().MMDPuffeRL, tmp_path, sd, mmd_coef=coef, interval=0, lr=1e-3)
        try:
            perturb(policy)
            epoch(tr)
            params.append(torch.cat([p.detach().flatten() for p in policy.parameters()]))
        finally:
            close(tr)
    diff = (params[0] - params[1]).abs().max().item()
    assert diff > 1e-7, "mmd_coef had no effect on the update: no gradient through the KL term"


def test_coef_zero_matches_base_puffer_rl(pr, tmp_path):
    import torch
    import pufferlib.pufferl as pufferl
    sd = init_state()
    runs = []
    for cls, kw in ((pufferl.PuffeRL, {}), (trainer_mod().MMDPuffeRL, dict(mmd_coef=0.0, interval=0))):
        tr, policy = build(cls, tmp_path, sd, **kw)
        try:
            hist = [epoch(tr)[0] for _ in range(3)]
            runs.append((hist, torch.cat([p.detach().flatten() for p in policy.parameters()])))
        finally:
            close(tr)
    (hb, pb), (hm, pm) = runs
    for e, (lb, lm) in enumerate(zip(hb, hm)):
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "old_approx_kl", "clipfrac", "importance"):
            assert abs(lb[k] - lm[k]) <= 1e-6, f"epoch {e}: {k} base {lb[k]} vs mmd(0) {lm[k]}"
    assert (pb - pm).abs().max().item() <= 1e-6, "mmd_coef = 0 must train exactly like PuffeRL"


def _kl_series(tmp_path, interval, perturb_before, n):
    tr, policy = build(trainer_mod().MMDPuffeRL, tmp_path, init_state(), mmd_coef=1.0,
                       interval=interval, lr=0.0)
    out = []
    try:
        for t in range(1, n + 1):
            if t in perturb_before:
                perturb(policy, seed=100 + t)
            out.append(epoch(tr)[0]["mmd_kl"])
    finally:
        close(tr)
    return out


@pytest.mark.parametrize("interval,perturb_before,pattern", [
    (2, (1, 5), [1, 1, 0, 0, 1, 1, 0]),
    (3, (1,), [1, 1, 1, 0, 0]),
    (0, (1,), [1, 1, 1, 1, 1, 1]),
])
def test_reference_refresh_interval(pr, tmp_path, interval, perturb_before, pattern):
    """lr = 0, so only the explicit perturbations move pi. With refresh after every `interval`
    epochs, the KL is > 0 until the refresh and exactly 0 afterwards (interval 0: never refreshed)."""
    kl = _kl_series(tmp_path, interval, perturb_before, len(pattern))
    got = [int(v > 1e-5) for v in kl]
    assert got == pattern, f"interval {interval}: KL per epoch {np.round(kl, 6).tolist()} -> {got}, want {pattern}"


def test_mmd_kl_in_logs(pr, tmp_path):
    tr, policy = build(trainer_mod().MMDPuffeRL, tmp_path, init_state(), mmd_coef=0.5, interval=0)
    try:
        epoch(tr)
        _, logs = epoch(tr)
        assert logs is not None and "losses/mmd_kl" in logs, "SPEC §15.3: logs losses/mmd_kl"
    finally:
        close(tr)
