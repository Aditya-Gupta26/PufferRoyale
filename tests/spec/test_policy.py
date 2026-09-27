"""SPEC §11 training integration: `pufferroyale.torch.Policy` (joint masked logits over 2305
actions) and its optional LSTM wrapper `pufferroyale.torch.Recurrent` with PufferLib 3.0 PuffeRL."""
import math

import numpy as np
import pytest

import envkit as E
import helpers as H


def policy_and_env(n_envs=2, **kw):
    import pufferroyale.torch as prt
    env = E.make(num_envs=n_envs, num_agents=2, seed=0, **kw)
    obs, _ = env.reset(seed=0)
    return prt.Policy(env), env, obs


def forward(policy, obs):
    import torch
    with torch.no_grad():
        return policy.forward_eval(torch.as_tensor(obs), dict(lstm_h=None, lstm_c=None))


def test_policy_forward_shapes(pr):
    policy, env, obs = policy_and_env()
    logits, value = forward(policy, obs)
    assert tuple(logits.shape) == (4, H.N_ACTIONS), f"joint logits shape {tuple(logits.shape)}"
    assert value.numel() == 4 and np.isfinite(value.numpy()).all()
    env.close()


def masked_probs_ok(policy, obs):
    import torch
    logits, _ = forward(policy, obs)
    probs = torch.softmax(logits.float(), dim=-1).numpy()
    masks = np.stack([E.mask_of(r) for r in obs])
    assert np.all(probs[masks < 0.5] == 0.0), "masked actions must have exactly zero probability"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    return logits, masks


def test_masked_actions_have_zero_probability(pr):
    import pufferlib.pytorch
    policy, env, obs = policy_and_env()
    rng = np.random.default_rng(0)
    for t in range(60):
        logits, masks = masked_probs_ok(policy, obs)
        a, logp, ent = pufferlib.pytorch.sample_logits(logits)
        assert np.isfinite(logp.numpy()).all() and np.isfinite(ent.numpy()).all()
        a = a.numpy().reshape(-1)
        assert all(masks[r, a[r]] > 0.5 for r in range(len(a)))
        acts = [E.random_bot_action(rng, r, 0.5) for r in obs]
        obs, *_ = E.step(env, acts)
    env.close()


def test_sparse_mask_finite_entropy(pr):
    """During the lockout only the no-op is legal: probability 1, entropy 0, all finite."""
    import pufferlib.pytorch
    policy, env, obs = policy_and_env()
    logits, masks = masked_probs_ok(policy, obs)
    assert masks[:, 1:].sum() == 0
    a, logp, ent = pufferlib.pytorch.sample_logits(logits)
    assert np.all(a.numpy() == 0)
    assert np.isfinite(ent.numpy()).all() and np.all(np.abs(ent.numpy()) < 1e-4)
    assert np.isfinite(logp.numpy()).all() and np.all(np.abs(logp.numpy()) < 1e-4)
    env.close()


def _config(tmp_path, use_rnn):
    return dict(env="pufferroyale", seed=1, torch_deterministic=True, cpu_offload=False, device="cpu",
                optimizer="adam", precision="float32", total_timesteps=4096, learning_rate=3e-4,
                anneal_lr=False, min_lr_ratio=0.0, gamma=0.995, gae_lambda=0.9, update_epochs=1,
                clip_coef=0.2, vf_coef=2.0, vf_clip_coef=0.2, max_grad_norm=1.5, ent_coef=0.001,
                adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, data_dir=str(tmp_path),
                checkpoint_interval=10_000, batch_size=512, minibatch_size=256,
                max_minibatch_size=256, bptt_horizon=32, compile=False, compile_mode="default",
                compile_fullgraph=False, vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.8,
                prio_beta0=0.2, use_rnn=use_rnn, name="smoke", project="smoke", amp=False)


@pytest.mark.slow
@pytest.mark.parametrize("use_rnn", [False, True])
def test_puffer_rl_smoke_feedforward_and_recurrent(pr, tmp_path, use_rnn):
    import torch
    import pufferlib.pufferl as pufferl
    import pufferlib.vector
    import pufferroyale
    import pufferroyale.torch as prt
    # Serial backend: PuffeRL's LSTM path reads vecenv.agents_per_batch, which PufferLib 3.0's
    # native PufferEnv spells agent_per_batch (a PufferLib quirk, not an env requirement)
    vecenv = pufferlib.vector.make(pufferroyale.Royale, backend=pufferlib.vector.Serial, num_envs=2,
                                   env_kwargs=dict(num_envs=4, num_agents=2))
    policy = prt.Policy(vecenv.driver_env)
    if use_rnn:
        policy = prt.Recurrent(vecenv.driver_env, policy)
    config = _config(tmp_path, use_rnn)
    trainer = pufferl.PuffeRL(config, vecenv, policy)
    try:
        while trainer.global_step < config["total_timesteps"]:
            trainer.evaluate()
            trainer.train()
            for k, v in trainer.losses.items():
                if k != "explained_variance":
                    assert math.isfinite(float(v)), f"loss {k} = {v}"
        for n, p in policy.named_parameters():
            assert torch.isfinite(p).all(), f"parameter {n} became non-finite"
    finally:
        trainer.utilization.stop()
        vecenv.close()
