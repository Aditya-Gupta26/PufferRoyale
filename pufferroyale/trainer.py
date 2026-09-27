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
"""
from __future__ import annotations

import contextlib
import copy
import math
import time
from collections import defaultdict

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


def obs_mask(observations: torch.Tensor) -> torch.Tensor:
    """Legality mask (rows, 2305) in the row order of the policies' logits
    (observations.reshape(-1, OBS_SIZE), also for the (segments, horizon, OBS) RNN batch)."""
    x = observations.reshape(-1, R.OBS_SIZE)
    return x[:, R.MASK_OFFSET:R.MASK_OFFSET + R.MASK_SIZE] > 0.5


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
        return masked_kl(logits, ref_logits, obs_mask(mb_obs))

    # ------------------------------------------------------------------ train
    # WHY A COPY: PuffeRL.train() builds and back-propagates the loss inside one method, with no
    # hook for an extra term. The body below is PufferLib 3.0's PuffeRL.train() (pufferl.py,
    # 3.0 branch commit 3b5c604) copied statement for statement (one commented-out block of the
    # original dropped); the additions are the three blocks marked "MMD" and the "AMP" fix (the
    # autocast context wraps only forward + loss and is exited before backward; the base entered
    # it twice per minibatch and never exited). With mmd_coef == 0 no MMD block runs and in
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


__all__ = ["MMDPuffeRL", "masked_kl", "obs_mask", "make_amp_context"]
