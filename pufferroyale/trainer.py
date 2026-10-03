"""pufferroyale.trainer -- PPO with a Magnetic-Mirror-Descent regulariser (SPEC §15.3, §15.7.8).

MMDPuffeRL is PufferLib 3.0's PuffeRL with one extra loss term,

    loss = PPO loss + mmd_coef * KL(pi_theta || pi_ref),

where pi_ref (the "magnet") is a frozen copy of the policy, refreshed every `mmd_ref_interval`
epochs (0 = keep the initial policy forever). Magnetic Mirror Descent (Sokota et al. 2022,
"A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player
Zero-Sum Games") shows that pulling each update towards a magnet policy gives last-iterate
convergence in two-player zero-sum games, where plain self-play / policy gradient tends to
cycle. KL(pi || pi_ref) is the reverse KL of the MMD proximal step, taken between the two
MASKED categorical distributions over the 2305 actions and averaged over the minibatch rows.

Config keys (read from the PuffeRL config dict): mmd_coef (default 0), mmd_ref_interval
(default 0). The reference is refreshed after train() increments `epoch`, when
interval > 0 and epoch % interval == 0. losses/mmd_kl is the mean over the epoch's minibatches.
With mmd_coef == 0 no reference is kept and train() computes exactly what PuffeRL.train() does
(bit-identical in float32), so all our scripts use this class, also for plain PPO.

Fixes over the PufferLib 3.0 base class (SPEC §18.6 / §18.7):
- Mixed precision: PuffeRL enters its autocast context in train() and never exits it, so
  autocast's weight-cast cache survives optimizer steps and later forwards reuse the FIRST bf16
  cast of the weights (the policy silently stops changing). Here the context is built with
  cache_enabled=False and entered/exited around each forward + loss only; precision="bfloat16"
  uses autocast on the configured device (cuda, cpu or mps), float32 uses no autocast.
- PuffeRL's utilization monitor thread is non-daemon, so any exception after the constructor
  left the process hanging; ours is a daemon thread.
- total_timesteps < batch_size (zero epochs) is rejected in the constructor instead of a
  ZeroDivisionError in train().
- load_training_state() restores optimizer state on resume and re-applies the configured
  learning rate / Adam betas / eps and the position on the cosine LR schedule.

v0.5 additions (SPEC §19.2):
- reward_clip (config, default 1.0): evaluate() is PuffeRL.evaluate() copied statement for
  statement, except that the stored rewards are clamped to [-c, c] when c > 0 and stored unchanged
  when c <= 0 (the base hard-codes [-1, 1]; potential-based shaped rewards can exceed it).
- Entropy decomposition of the joint logits over 1 + 4 B actions (entropy_split): H_card over
  {wait, slot 0..3} and H_pos = sum_s P(s) H(p(. | s)), with H_joint = H_card + H_pos. Every epoch
  logs losses/entropy_card and losses/entropy_pos (minibatch means; computed without grad and
  without RNG unless the split coefficients are set).
- ent_coef_card / ent_coef_pos (config, default -1.0 = unset): both >= 0 replace
  -ent_coef * mean(H_joint) by -(ent_coef_card * mean(H_card) + ent_coef_pos * mean(H_pos)); exactly
  one set is a ValueError. Unset, train() is exactly the v0.4 / base computation.
- obs_mask / masked_kl use the mask of the logits' placement grid (SPEC §19.4).
- WandbLogger: PufferLib's WandbLogger interface for --wandb in train.py / league_train.py, working
  with any wandb version (SPEC §19.7.5; PufferLib's own class needs wandb.util.generate_id).

v0.5-G.3 (SPEC §19.10): shaping_anneal_steps (exact rational N = ceil(F T / R), fixed per run by the
scripts), warn_shaping_clip (shaping weights > 0 with reward_clip > 0), and WandbLogger logs
global_step as its step metric.
"""
from __future__ import annotations

import contextlib
import copy
import math
import os
import time
from collections import defaultdict
from fractions import Fraction

import numpy as np
import torch

import pufferlib
import pufferlib.pytorch
import pufferlib.pufferl as pufferl
from pufferlib.pufferl import compute_puff_advantage
from torch.distributed.elastic.multiprocessing.errors import record

from . import royale as R


def masked_kl(logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over rows of KL(p || q), p = softmax(logits), q = softmax(ref_logits), both restricted
    to the legal actions of each row (`mask`, bool, same shape). Illegal entries contribute
    exactly 0 (the policies already give them probability 0 via finfo.min logits; masking here
    keeps the result exact and the gradient finite even if a policy did not)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    logq = torch.log_softmax(ref_logits.float(), dim=-1)
    terms = torch.where(mask, logp.exp() * (logp - logq), torch.zeros_like(logp))
    return terms.sum(-1).mean()


def obs_mask(observations: torch.Tensor, grid: int = 1) -> torch.Tensor:
    """Legality mask (rows, 1 + 4 B) of placement grid `grid` (SPEC §19.4; 2305 wide for grid 1) in
    the row order of the policies' logits (observations.reshape(-1, OBS_SIZE), also for the
    (segments, horizon, OBS) RNN batch)."""
    from .torch import action_mask
    x = observations.reshape(-1, R.OBS_SIZE)
    return action_mask(x, grid)


def entropy_split(logits: torch.Tensor):
    """SPEC §19.2.2: per-row (H_card, H_pos) of p = softmax(logits) over A = 1 + 4 B joint actions
    (slot-major, action 1 + s B + j): P(wait) = p_0, P(s) = sum_j p_{1 + s B + j},
    H_card = -sum_{c in wait, 0..3} P(c) log P(c), H_pos = sum_s P(s) H(p(. | s)). A slot with
    P(s) = 0 contributes 0 and 0 log 0 = 0, so H_card + H_pos = H_joint. Differentiable; finite
    for masked logits (illegal = finfo.min), including rows where only wait is legal."""
    A = logits.shape[-1]
    if (A - 1) % 4:
        raise ValueError(f"entropy_split needs 1 + 4 B joint logits, got {A}")
    B = (A - 1) // 4
    logp = torch.log_softmax(logits.float().reshape(-1, A), dim=-1)
    lp_pos = logp[:, 1:].reshape(-1, 4, B)
    lp_slot = torch.logsumexp(lp_pos, dim=-1)                               # log P(s), (N, 4)
    lp_card = torch.cat([logp[:, :1], lp_slot], dim=-1)                     # (N, 5)
    p_card = lp_card.exp()
    h_card = -torch.where(p_card > 0, p_card * lp_card, torch.zeros_like(p_card)).sum(-1)
    lp_cond = lp_pos - lp_slot.unsqueeze(-1)                                # log p(j | s)
    p_cond = lp_cond.exp()
    h_slot = -torch.where(p_cond > 0, p_cond * lp_cond, torch.zeros_like(p_cond)).sum(-1)    # H(p(. | s))
    p_slot = p_card[:, 1:]
    h_pos = torch.where(p_slot > 0, p_slot * h_slot, torch.zeros_like(p_slot)).sum(-1)
    return h_card, h_pos


def _coef(config, key):
    """An entropy-split coefficient: None when unset (any negative value, default -1.0; SPEC
    §19.9.11), else a finite float >= 0."""
    v = config.get(key, -1.0)
    if v is None:
        return None
    v = float(v)
    if v < 0.0:
        return None
    if not math.isfinite(v):
        raise ValueError(f"{key} must be finite, got {v!r}")
    return v


def shaping_anneal_steps(anneal_frac, total_timesteps, rows) -> int:
    """SPEC §19.10.1: the anneal length N = ceil(F * total_timesteps / R) in exact rational arithmetic
    on the decimal value of F (Fraction(str(F)): F = 0.07, T = 3e8, R = 3000 gives 7000, where the
    float product gives 7001). 0 when F = 0 (constant weights)."""
    try:
        F = Fraction(str(anneal_frac).strip())
    except (TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"--shaping-anneal-frac must be a finite number >= 0, got {anneal_frac!r}") from None
    rows = int(rows)
    if F < 0:
        raise ValueError(f"--shaping-anneal-frac must be a finite number >= 0, got {anneal_frac!r}")
    if rows < 1:
        raise ValueError("learner rows per step must be >= 1")
    return int(math.ceil(F * int(total_timesteps) / rows))


def shaping_env_kwargs(gamma, anneal_frac, total_timesteps, rows, global_step=0, anneal_steps=None) -> dict:
    """SPEC §19.7.1 / §19.10.1: the reward-v2 env keywords a training script injects -- reward_gamma =
    the trainer's gamma; shaping_anneal_steps = `anneal_steps` (the run's recorded N) when given, else
    shaping_anneal_steps(F, total_timesteps, R) for --shaping-anneal-frac F (0 = constant weights) with
    R learner rows per vector step; shaping_step_offset = global_step // R (the env steps a resumed run
    has already done)."""
    steps = shaping_anneal_steps(anneal_frac, total_timesteps, rows)      # validates F and R in any case
    if anneal_steps is not None:
        steps = int(anneal_steps)
        if steps < 0:
            raise ValueError(f"shaping_anneal_steps must be >= 0, got {anneal_steps!r}")
    return {"reward_gamma": float(gamma), "shaping_anneal_steps": steps,
            "shaping_step_offset": int(global_step) // int(rows)}


#: the reward-v2 shaping weights of the env (SPEC §19.1)
SHAPING_WEIGHTS = ("reward_tower", "reward_crown", "reward_elixir", "reward_play")


def warn_shaping_clip(env_kwargs, reward_clip, prog="train") -> bool:
    """SPEC §19.10.2: one warning on stderr when any shaping weight is > 0 while the trainer clamps
    rewards (reward_clip > 0) -- the clamp may clip shaped terminal rewards. Not an error. Returns
    whether it warned."""
    import sys
    clip = 1.0 if reward_clip is None else float(reward_clip)
    on = [k for k in SHAPING_WEIGHTS if float((env_kwargs or {}).get(k) or 0.0) > 0.0]
    if not (on and clip > 0.0):
        return False
    print(f"[{prog}] warning: reward shaping is on ({', '.join(on)}) while train.reward_clip = {clip:g} > 0: shaped "
          f"rewards (terminal steps included) may be clipped; use --train.reward-clip 0", file=sys.stderr, flush=True)
    return True


class WandbLogger:
    """Weights & Biases logger with the interface of PufferLib 3.0's pufferl.WandbLogger (run_id,
    log(logs, step), upload_model(path), close(model_path, early_stop); `wandb` = the module) used
    by train.py and league_train.py for --wandb (SPEC §19.7.5). PufferLib's own class calls
    wandb.util.generate_id(), which recent wandb (e.g. 0.30) no longer has; here the run id is
    wandb.init's own (or `load_id` when resuming a run), so it works across wandb versions. `args`
    needs the keys wandb_project, wandb_group, tag and no_model_upload and is the run's wandb
    config. wandb is imported here only, so a run without --wandb never imports it;
    WANDB_MODE=offline needs no network (offline runs land in $WANDB_DIR/wandb/offline-run-*).
    log(logs, step) records `step` as `global_step`, the declared step metric (SPEC §19.10.7), so the
    epochs a resumed run re-does are kept."""

    def __init__(self, args, load_id=None, resume="allow"):
        import wandb
        kw = dict(project=args.get("wandb_project"), group=args.get("wandb_group"), allow_val_change=True,
                  save_code=False, resume=resume, config=dict(args),
                  tags=[str(args["tag"])] if args.get("tag") is not None else [],
                  settings=wandb.Settings(console="off"))            # no dashboard text sent to wandb
        if load_id:
            kw["id"] = str(load_id)
        self.wandb = wandb
        self.run = wandb.init(**kw)
        self.run_id = self.run.id
        self.should_upload_model = not args.get("no_model_upload")
        # SPEC §19.10.7: global_step is the step metric of every series. Records carry it as a value and
        # are logged without wandb's own `step`, which must increase: after a resume the re-done epochs
        # (global steps already seen) would otherwise be dropped
        self.run.define_metric("global_step")
        self.run.define_metric("*", step_metric="global_step")

    def log(self, logs, step):
        rec = dict(logs)
        rec["global_step"] = int(step)
        self.run.log(rec)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

    def close(self, model_path, early_stop):
        self.run.summary["early_stop"] = bool(early_stop)
        if self.should_upload_model and model_path and os.path.isfile(model_path):
            self.upload_model(model_path)
        self.run.finish()

    def abort(self):
        """Finish the run with a failure exit code (never raises: used on error paths)."""
        try:
            self.run.finish(exit_code=1)
        except Exception:  # noqa: BLE001 -- never mask the original error
            pass


class _DaemonUtilization(pufferl.Utilization):
    """PuffeRL's CPU/GPU utilization monitor as a daemon thread: it can never keep the interpreter
    alive after an error (PuffeRL's own is non-daemon and only stops on trainer.utilization.stop())."""

    def start(self):
        self.daemon = True
        super().start()


def make_amp_context(config):
    """The autocast context for a PuffeRL config: none for float32 (or amp = False); for bfloat16
    autocast on the configured device type WITHOUT the weight-cast cache (SPEC §18.6), so every
    forward casts the current weights."""
    precision = config.get("precision", "float32")
    if precision == "float32" or not config.get("amp", True):
        return contextlib.nullcontext()
    if precision != "bfloat16":
        raise pufferlib.APIUsageError(f"Invalid precision: {precision}: use float32 or bfloat16")
    return torch.autocast(device_type=torch.device(config["device"]).type, dtype=torch.bfloat16,
                          cache_enabled=False)


class MMDPuffeRL(pufferl.PuffeRL):
    """PuffeRL + mmd_coef * KL(pi_theta || pi_ref). Same constructor as PuffeRL."""

    def __init__(self, config, vecenv, policy, logger=None):
        # SPEC §19.2: reward clip and entropy split, validated before the base class starts anything
        clip = config.get("reward_clip", 1.0)
        self.reward_clip = 1.0 if clip is None else float(clip)
        if not math.isfinite(self.reward_clip):
            raise ValueError(f"reward_clip must be finite, got {self.reward_clip!r}")
        self.ent_coef_card, self.ent_coef_pos = _coef(config, "ent_coef_card"), _coef(config, "ent_coef_pos")
        if (self.ent_coef_card is None) != (self.ent_coef_pos is None):
            raise ValueError("ent_coef_card and ent_coef_pos must be set together (both >= 0) or both left unset "
                             f"(negative): got {config.get('ent_coef_card', -1.0)!r} and {config.get('ent_coef_pos', -1.0)!r}")
        self.entropy_split = self.ent_coef_card is not None
        orig = pufferl.Utilization
        pufferl.Utilization = _DaemonUtilization          # the base constructor starts the monitor
        try:
            super().__init__(config, vecenv, policy, logger)
        finally:
            pufferl.Utilization = orig
        if self.total_epochs < 1:
            self.utilization.stop()
            raise ValueError(f"total_timesteps ({config['total_timesteps']}) must be >= batch_size "
                             f"({config['batch_size']}): PuffeRL trains in whole epochs of batch_size steps")
        self.amp_context = make_amp_context(config)
        self.mmd_coef = float(config.get("mmd_coef", 0.0) or 0.0)
        self.mmd_ref_interval = int(config.get("mmd_ref_interval", 0) or 0)
        if self.mmd_coef < 0:
            raise ValueError("mmd_coef must be >= 0")
        if self.mmd_ref_interval < 0:
            raise ValueError("mmd_ref_interval must be >= 0")
        self.ref_policy = self._frozen_copy() if self.mmd_coef > 0 else None
        self.mmd_ref_refreshes = 0

    # ------------------------------------------------------------------ the magnet
    def _frozen_copy(self):
        ref = copy.deepcopy(self.uncompiled_policy)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    def refresh_reference(self) -> None:
        """pi_ref <- pi_theta (a frozen copy of the current weights)."""
        if self.ref_policy is not None:
            self.ref_policy.load_state_dict(self.uncompiled_policy.state_dict())
            self.mmd_ref_refreshes += 1

    # ------------------------------------------------------------------ resume
    def load_training_state(self, optimizer_state, epoch, global_step):
        """Resume: load the optimizer state, set the counters, then re-apply the CONFIGURED
        learning rate, Adam betas and eps (load_state_dict would otherwise restore the saved run's
        values) and put the cosine schedule at `epoch` (continuing it rather than restarting)."""
        self.optimizer.load_state_dict(optimizer_state)
        self.epoch, self.global_step = int(epoch), int(global_step)
        self.apply_optimizer_config()

    def apply_optimizer_config(self):
        cfg = self.config
        lr = float(cfg["learning_rate"])
        groups = self.optimizer.param_groups
        for g in groups:
            g["initial_lr"] = lr
            if "betas" in g:
                g["betas"] = (float(cfg["adam_beta1"]), float(cfg["adam_beta2"]))
            if "eps" in g:
                g["eps"] = float(cfg["adam_eps"])
        sch = self.scheduler
        sch.base_lrs = [lr] * len(groups)
        sch.last_epoch = self.epoch
        cur = lr
        if cfg.get("anneal_lr"):
            eta_min, T = float(sch.eta_min), max(1, int(sch.T_max))
            cur = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * self.epoch / T)) / 2
        for g in groups:
            g["lr"] = cur
        sch._last_lr = [cur] * len(groups)

    def _mmd_kl(self, mb_obs, logits):
        with torch.no_grad():
            ref_state = dict(action=None, lstm_h=None, lstm_c=None)
            ref_logits, _ = self.ref_policy(mb_obs, ref_state)
        from .torch import grid_of_actions
        return masked_kl(logits, ref_logits, obs_mask(mb_obs, grid_of_actions(logits.shape[-1])))

    # ------------------------------------------------------------------ evaluate
    # SPEC §19.2.1: PufferLib 3.0's PuffeRL.evaluate() (pufferl.py, 3.0 branch commit 3b5c604) copied
    # statement for statement; the one change is the block marked "REWARD CLIP" (the base clamps
    # every stored reward to [-1, 1]). With reward_clip = 1 the stored rewards, actions, RNG draws and
    # buffers are exactly the base class's.
    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']

        if config['use_rnn']:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t # TODO: Handle truncations separately
            self.global_step += int(mask.sum())

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)#, non_blocking=True)
            r = torch.as_tensor(r).to(device)#, non_blocking=True)
            d = torch.as_tensor(d).to(device)#, non_blocking=True)

            profile('eval_forward', epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config['use_rnn']:
                    state['lstm_h'] = self.lstm_h[env_id.start]
                    state['lstm_c'] = self.lstm_c[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                # REWARD CLIP: [-c, c] for c > 0; c <= 0 stores the env rewards unchanged
                if self.reward_clip > 0:
                    r = torch.clamp(r, -self.reward_clip, self.reward_clip)

            profile('eval_copy', epoch)
            with torch.no_grad():
                if config['use_rnn']:
                    self.lstm_h[env_id.start] = state['lstm_h']
                    self.lstm_c[env_id.start] = state['lstm_c']

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1+self.ep_indices[env_id.stop - 1].item())

                if config['cpu_offload']:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l+1 >= config['bptt_horizon']:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config['device']).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile('eval_misc', epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile('env', epoch)
            self.vecenv.send(action)

        profile('eval_misc', epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    # ------------------------------------------------------------------ train
    # WHY A COPY: PuffeRL.train() builds and back-propagates the loss inside one method, with no
    # hook for an extra term. The body below is PufferLib 3.0's PuffeRL.train() (pufferl.py,
    # 3.0 branch commit 3b5c604) copied statement for statement (one commented-out block of the
    # original dropped); the additions are the three blocks marked "MMD", the two marked "ENTROPY
    # SPLIT" (SPEC §19.2.2-3) and the "AMP" fix (the autocast context wraps only forward + loss and
    # is exited before backward; the base entered it twice per minibatch and never exited). With
    # mmd_coef == 0 no MMD block runs, with the split unset the entropy term is the base's and the
    # logged H_card / H_pos are computed under no_grad from detached logits (no RNG), and in
    # float32 the AMP context is a no-op, so the losses, the RNG draws (multinomial minibatch
    # sampling) and the parameter updates are exactly the base class's.
    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile('train', epoch)
        profile('train_misc', epoch, nest=True)
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile('train_misc', epoch)

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(self.values, self.rewards,
                self.terminals, self.ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments*prio_probs[idx, None])**-anneal_beta

            profile('train_copy', epoch)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            # AMP: autocast (if any) around forward + loss only, exited before backward
            with self.amp_context:
                logits, newvalue = self.policy(mb_obs, state)
                actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)
                # ENTROPY SPLIT (1/2): H_card / H_pos per row; outside the graph unless they enter the loss
                if self.entropy_split:
                    h_card, h_pos = entropy_split(logits)
                else:
                    with torch.no_grad():
                        h_card, h_pos = entropy_split(logits.detach())

                profile('train_misc', epoch)
                newlogprob = newlogprob.reshape(mb_logprobs.shape)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                self.ratio[idx] = ratio.detach()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

                # Weight advantages by priority and normalize
                adv = mb_advantages
                adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

                # Losses
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(mb_returns.shape)
                v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()

                # ENTROPY SPLIT (2/2): -(c_card mean(H_card) + c_pos mean(H_pos)) replaces -ent_coef mean(H_joint)
                if self.entropy_split:
                    loss = pg_loss + config['vf_coef']*v_loss - (self.ent_coef_card*h_card.mean()
                                                                 + self.ent_coef_pos*h_pos.mean())
                else:
                    loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss

                # MMD (1/3): + mmd_coef * KL(pi_theta || pi_ref), differentiated through pi_theta only
                if self.mmd_coef > 0:
                    mmd_kl = self._mmd_kl(mb_obs, logits)
                    loss = loss + self.mmd_coef * mmd_kl
                    losses['mmd_kl'] += mmd_kl.item() / self.total_minibatches

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()

            # Logging
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.item() / self.total_minibatches
            losses['value_loss'] += v_loss.item() / self.total_minibatches
            losses['entropy'] += entropy_loss.item() / self.total_minibatches
            losses['entropy_card'] += h_card.mean().item() / self.total_minibatches
            losses['entropy_pos'] += h_pos.mean().item() / self.total_minibatches
            losses['old_approx_kl'] += old_approx_kl.item() / self.total_minibatches
            losses['approx_kl'] += approx_kl.item() / self.total_minibatches
            losses['clipfrac'] += clipfrac.item() / self.total_minibatches
            losses['importance'] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile('learn', epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # MMD (2/3): the logged KL is 0 when the term is disabled
        if self.mmd_coef <= 0:
            losses['mmd_kl'] = 0.0

        # Reprioritize experience
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config['total_timesteps']
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f'Checkpoint saved at update {self.epoch}'

        # MMD (3/3): refresh the magnet after the epoch counter moved (SPEC §15.7.8)
        if self.mmd_coef > 0 and self.mmd_ref_interval > 0 and self.epoch % self.mmd_ref_interval == 0:
            self.refresh_reference()

        return logs


__all__ = ["MMDPuffeRL", "masked_kl", "obs_mask", "make_amp_context", "entropy_split", "shaping_env_kwargs",
           "shaping_anneal_steps", "warn_shaping_clip", "SHAPING_WEIGHTS", "WandbLogger"]
