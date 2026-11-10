"""Checks of the placement-entropy term (2026-10-20, stage-4 ablation; pufferroyale.trainer.
placement_entropy_unweighted and the ent_coef_pos_unweighted config key). Cluster-side (not under
tests/spec): run with `python -m pytest -q hpc/tests`.

- H_u equals a float64 reference: per row, the mean over the slots with a legal tile of the entropy of the
  softmax over that slot's legal tiles; rows without a legal slot are flagged and give 0.
- H_u ignores the card's log-probability: adding any constant to a slot's logits leaves it unchanged and its
  gradient with respect to such a constant is zero.
- MMDPuffeRL: a negative coefficient is a ValueError; every epoch logs losses/entropy_pos_unweighted; with a
  linear-in-gradient optimizer step (Adam with a huge eps), the update with c > 0 minus the update with c = 0
  points up the gradient of H_u (it raises the placement entropy) and is otherwise identical.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "spec"))          # leaguekit.ppo_config (read-only use)

torch = pytest.importorskip("torch")


def _random_logits(n, B, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1 + 4 * B, generator=g, dtype=torch.float64) * 3.0
    mask = torch.rand(n, 1 + 4 * B, generator=g) < 0.3
    mask[:, 0] = True
    mask[0, 1:] = False                                  # row 0: only wait is legal
    mask[1, 1 + B:] = False                              # row 1: slot 0 only
    mask[1, 1:1 + B] = False
    mask[1, 5] = True                                    # ... with a single legal tile
    neg = torch.finfo(torch.float32).min
    return torch.where(mask, x, torch.tensor(neg, dtype=x.dtype)).float(), mask


def _reference(logits, mask):
    A = logits.shape[-1]
    B = (A - 1) // 4
    x, m = logits.double().numpy(), mask.numpy()
    rows, ok = [], []
    for i in range(x.shape[0]):
        hs = []
        for s in range(4):
            seg, leg = x[i, 1 + s * B:1 + (s + 1) * B], m[i, 1 + s * B:1 + (s + 1) * B]
            if not leg.any():
                continue
            z = seg[leg] - seg[leg].max()
            p = np.exp(z) / np.exp(z).sum()
            hs.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
        rows.append(sum(hs) / len(hs) if hs else 0.0)
        ok.append(bool(hs))
    return np.array(rows), np.array(ok)


@pytest.mark.parametrize("B", [576, 144, 40])
def test_matches_reference(B):
    from pufferroyale.trainer import placement_entropy_unweighted
    logits, mask = _random_logits(7, B, seed=B)
    h, ok = placement_entropy_unweighted(logits, mask)
    ref, ref_ok = _reference(logits, mask)
    assert ok.tolist() == ref_ok.tolist() and not ok[0] and ok[1]
    assert h[1].item() == 0.0                            # one legal tile: no uncertainty
    assert np.allclose(h.double().numpy(), ref, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(h).all()


def test_ignores_card_logits():
    from pufferroyale.trainer import placement_entropy_unweighted
    B = 576
    logits, mask = _random_logits(5, B, seed=11)
    shift = torch.zeros(4, requires_grad=True)
    full = torch.cat([torch.zeros(1), shift.repeat_interleave(B)])
    h, ok = placement_entropy_unweighted(torch.where(mask, logits + full, logits), mask)
    h0, _ = placement_entropy_unweighted(logits, mask)
    assert torch.allclose(h, h0, atol=1e-5)
    h[ok].sum().backward()
    assert shift.grad.abs().max().item() < 1e-4


def _one_update(tmp_path, coef, seed=3):
    import leaguekit as L
    import pufferroyale
    import pufferroyale.torch as prt
    from pufferroyale.trainer import MMDPuffeRL
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0)
    torch.manual_seed(seed)
    policy = prt.Policy(env)
    cfg = L.ppo_config(tmp_path, batch_size=64, horizon=16, minibatch=64, total=64 * 50, lr=1e4,
                       mmd_coef=0.0, mmd_ref_interval=0, adam_eps=1e4, max_grad_norm=1e12,
                       ent_coef_pos_unweighted=coef)
    tr = MMDPuffeRL(cfg, env, policy)
    try:
        torch.manual_seed(seed)
        tr.evaluate()
        obs = tr.observations.reshape(-1, tr.observations.shape[-1]).clone().float()
        before = [p.detach().clone() for p in policy.parameters()]
        tr.last_log_time = -1e18
        tr.train()
        delta = torch.cat([(p.detach() - b).flatten() for p, b in zip(policy.parameters(), before)])
        losses = dict(tr.losses)
    finally:
        try:
            tr.utilization.stop()
        except Exception:
            pass
        tr.vecenv.close()
    return delta, obs, before, losses


def test_trainer_term(tmp_path):
    from pufferroyale.trainer import MMDPuffeRL, obs_mask, placement_entropy_unweighted
    import leaguekit as L
    import pufferroyale
    import pufferroyale.torch as prt
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0)
    with pytest.raises(ValueError):
        MMDPuffeRL(L.ppo_config(tmp_path, batch_size=64, horizon=16, minibatch=64, total=64 * 50,
                                mmd_coef=0.0, mmd_ref_interval=0, ent_coef_pos_unweighted=-0.1), env, prt.Policy(env))
    env.close()
    d0, obs, before, losses0 = _one_update(tmp_path / "a", 0.0)
    d1, obs1, _, losses1 = _one_update(tmp_path / "b", 1.0)
    assert torch.equal(obs, obs1)                        # same rollout: the term uses no randomness
    assert "entropy_pos_unweighted" in losses0 and "entropy_pos_unweighted" in losses1
    assert losses0["entropy_pos_unweighted"] > 0
    # the extra update is (lr / eps) * c * grad(mean H_u over the minibatch): it must raise H_u
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0)
    pol = prt.Policy(env)
    env.close()
    with torch.no_grad():
        for p, b in zip(pol.parameters(), before):
            p.copy_(b)
    logits, _ = pol(obs, {})
    h, ok = placement_entropy_unweighted(logits, obs_mask(obs))
    g = torch.autograd.grad(h[ok].mean(), list(pol.parameters()), allow_unused=True)
    grad = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(g, pol.parameters())])
    diff = d1 - d0
    assert diff.norm() > 0
    cos = float((diff @ grad) / (diff.norm() * grad.norm()))
    assert cos > 0.9, cos
