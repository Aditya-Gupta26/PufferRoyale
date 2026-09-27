"""Helpers for the SPEC §15 (Phase D) tests, following the §15.7 (v0.3-D.1) pinned interfaces."""
import json
import os
import re
import subprocess
import sys

import numpy as np

import helpers as H

ROOT = H.ROOT


def league():
    import pufferroyale.league as mod
    return mod


def make_pool(**kw):
    """SPEC §15.1 constructor (fully specified)."""
    return league().OpponentPool(**kw)


def make_league(pool, num_envs=1, seed=0, opponent_greedy=False, **royale_kwargs):
    """SPEC §15.7.4: LeagueVecEnv(pool, num_envs=1, seed=0, device="cpu", opponent_greedy=False,
    **royale_kwargs), royale_kwargs forwarded to Royale(num_agents=2, ...)."""
    return league().LeagueVecEnv(pool, num_envs=num_envs, seed=seed, device="cpu",
                                 opponent_greedy=opponent_greedy, **royale_kwargs)


def policy_env():
    import pufferroyale
    return pufferroyale.Royale(num_envs=1, num_agents=2)


def make_ckpt(path, recurrent=False, noop_bias=None, seed=0):
    """SPEC §15.7.7: ckpt files are exactly torch.save(policy.state_dict()) of
    pufferroyale.torch.Policy (default sizes) or Recurrent(Policy).

    `noop_bias`: add this to the logit bias of action 0 (the unique 2305-long bias tensor of the
    action head), to craft a policy whose greedy choice is always the no-op while sampling still
    plays cards (used for the opponent_greedy test)."""
    import torch
    import pufferroyale.torch as prt
    env = policy_env()
    torch.manual_seed(seed)
    policy = prt.Policy(env)
    if recurrent:
        policy = prt.Recurrent(env, policy)
    sd = policy.state_dict()
    if noop_bias is not None:
        cands = [k for k, v in sd.items() if tuple(v.shape) == (H.N_ACTIONS,) and k.endswith("bias")]
        assert len(cands) == 1, f"tester cannot locate the action-logit bias (candidates {cands})"
        sd[cands[0]] = sd[cands[0]].clone()
        sd[cands[0]][0] += noop_bias
    torch.save(sd, str(path))
    env.close()
    return str(path)


def ppo_config(data_dir, batch_size, horizon, minibatch, total, lr=3e-4, seed=1, **extra):
    cfg = dict(env="pufferroyale", seed=seed, torch_deterministic=True, cpu_offload=False, device="cpu",
               optimizer="adam", precision="float32", total_timesteps=total, learning_rate=lr,
               anneal_lr=False, min_lr_ratio=0.0, gamma=0.995, gae_lambda=0.9, update_epochs=1,
               clip_coef=0.2, vf_coef=2.0, vf_clip_coef=0.2, max_grad_norm=1.5, ent_coef=0.001,
               adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, data_dir=str(data_dir),
               checkpoint_interval=10_000, batch_size=batch_size, minibatch_size=minibatch,
               max_minibatch_size=minibatch, bptt_horizon=horizon, compile=False,
               compile_mode="default", compile_fullgraph=False, vtrace_rho_clip=1.0,
               vtrace_c_clip=1.0, prio_alpha=0.8, prio_beta0=0.2, use_rnn=False, name="smoke",
               project="smoke", amp=False)
    cfg.update(extra)
    return cfg


def run_script(args, timeout=1200):
    out = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                         timeout=timeout)
    return out


def help_flags(script):
    """Flags advertised by `python <script> --help`."""
    out = run_script([script, "--help"], timeout=120)
    assert out.returncode == 0, f"{script} --help failed:\n{(out.stdout + out.stderr)[-2000:]}"
    return set(re.findall(r"(--[A-Za-z0-9][A-Za-z0-9_-]*)", out.stdout + out.stderr))


def last_line_json(text):
    """SPEC §15.7.11: the last (non-empty) stdout line is the JSON."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    assert lines, "no stdout"
    try:
        return json.loads(lines[-1])
    except ValueError as e:
        raise AssertionError(f"the last stdout line is not JSON: {lines[-1][:300]!r}") from e


def last_json(text):
    """Last line of `text` that parses as a JSON object."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return None


def assert_distribution(x, n=None, tol=1e-6):
    x = np.asarray(x, dtype=float).reshape(-1)
    if n is not None:
        assert len(x) == n, f"mixture has {len(x)} entries, want {n}"
    assert np.all(x >= -tol), f"negative probabilities {x}"
    assert abs(x.sum() - 1.0) < tol, f"mixture sums to {x.sum()}"
    return x


def assert_antisymmetric(P, tol=1e-6):
    P = np.asarray(P, dtype=float)
    assert P.ndim == 2 and P.shape[0] == P.shape[1]
    assert np.all((P >= -tol) & (P <= 1 + tol)), "scores must lie in [0, 1]"
    assert np.allclose(np.diag(P), 0.5, atol=tol), f"diagonal must be 0.5: {np.diag(P)}"
    off = P + P.T
    assert np.allclose(off, 1.0, atol=tol), f"P + P^T must be 1 off the diagonal:\n{P}"
    return P
