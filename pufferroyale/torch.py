"""pufferroyale.torch -- policies for PufferLib 3.0 PuffeRL (SPEC §11, v0.3 observation §16.4).

Policy: a CNN over the spatial planes, a shared per-entity MLP with masked mean and max
pooling over the own and enemy entity slots, and an MLP over the scalars, concatenated into
an MLP trunk; joint logits over the 2305 actions and a value head.

Card identities (SPEC §16.4) arrive as integer ids card_id + 1 (0 = empty) in the entity rows
(feature 0) and in the hand (4), next-card (1) and opponent last-4 (4) scalars. They are embedded
with one shared nn.Embedding(CARD_SLOTS + 1, card_dim) (index 0 = empty); the embedded entity id
is concatenated with the row's other 10 features, the 9 embedded scalar ids with the remaining
scalars. The 128-wide multi-hots (opponent cards seen / deduced hand) stay plain inputs. An
entity slot is occupied iff its card id is non-zero.

Action masking: every forward pass (Policy.forward / forward_eval and Recurrent.forward /
forward_eval) slices the legality mask out of the observations it was given
(MASK_OFFSET : MASK_OFFSET + MASK_SIZE) and sets illegal logits to torch.finfo(dtype).min (not
-inf, so entropies stay finite). There is no state kept between calls and no silent skip: a
mask whose shape does not match the logits raises. PufferLib's LSTMWrapper calls
decode_actions(hidden) without the observations, so decode_actions returns UNMASKED logits
and Recurrent applies the mask after the wrapper's forward, where the flattened (B*T) row
order of the hidden batch equals observations.reshape(-1, OBS_SIZE).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import pufferlib.models
import pufferlib.pytorch

from . import royale as R


def _init(layer, std=np.sqrt(2)):
    return pufferlib.pytorch.layer_init(layer, std=std)


def _conv_out(n, k=3, s=2, p=1):
    return (n + 2 * p - k) // s + 1


def action_mask(observations):
    """Boolean legality mask (rows, 2305) of a batch of observations of any leading shape."""
    x = observations.reshape(-1, R.OBS_SIZE)
    return x[:, R.MASK_OFFSET:R.MASK_OFFSET + R.MASK_SIZE] > 0.5


def mask_logits(logits, observations):
    """Illegal actions of each row get torch.finfo(dtype).min (zero probability)."""
    mask = action_mask(observations)
    if mask.shape != logits.shape:
        raise ValueError(f"action mask {tuple(mask.shape)} does not match logits {tuple(logits.shape)}")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _card_id_positions():
    """Indices (inside the scalar section) of the integer card-id scalars, and of all the others."""
    ids = []
    for name in R.CARD_ID_SCALARS:
        off, n = R.SCALAR_INDEX[name]
        ids.extend(range(off, off + n))
    rest = [i for i in range(R.SCALAR_SIZE) if i not in set(ids)]
    return ids, rest


class Policy(nn.Module):
    def __init__(self, env=None, hidden_size=256, cnn_channels=64, entity_hidden=128, scalar_hidden=128, card_dim=16,
                 **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.is_multidiscrete = False
        C, H, W = R.SPATIAL_SHAPE
        self.spatial_shape = (C, H, W)
        self.n_ent, self.ent_f = R.ENTITY_SHAPE
        self.half = self.n_ent // 2
        self.card_dim = card_dim
        # SPEC §16.4: card ids card_id + 1 (0 = empty) -> one shared embedding table
        self.card_embed = nn.Embedding(R.CARD_SLOTS + 1, card_dim)
        ids, rest = _card_id_positions()
        self.register_buffer("sc_id_idx", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.register_buffer("sc_rest_idx", torch.tensor(rest, dtype=torch.long), persistent=False)
        h2, w2 = _conv_out(_conv_out(H)), _conv_out(_conv_out(W))
        self.cnn = nn.Sequential(
            _init(nn.Conv2d(C, cnn_channels, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(cnn_channels * h2 * w2, hidden_size)), nn.ReLU(),
        )
        self.entity = nn.Sequential(
            _init(nn.Linear(self.ent_f - 1 + card_dim, entity_hidden)), nn.ReLU(),
            _init(nn.Linear(entity_hidden, entity_hidden)), nn.ReLU(),
        )
        self.scalars = nn.Sequential(_init(nn.Linear(len(rest) + len(ids) * card_dim, scalar_hidden)), nn.ReLU())
        trunk_in = hidden_size + 4 * entity_hidden + scalar_hidden
        self.trunk = nn.Sequential(
            _init(nn.Linear(trunk_in, hidden_size)), nn.ReLU(),
            _init(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
        )
        self.actor = _init(nn.Linear(hidden_size, R.N_ACTIONS), std=0.01)
        self.value = _init(nn.Linear(hidden_size, 1), std=1.0)

    # -- PufferLib LSTMWrapper protocol ------------------------------------------------
    def encode_observations(self, observations, state=None):
        x = observations.reshape(-1, R.OBS_SIZE).float()
        B = x.shape[0]
        C, H, W = self.spatial_shape
        sp = x[:, R.SPATIAL_OFFSET:R.SPATIAL_OFFSET + C * H * W].reshape(B, C, H, W)
        ent = x[:, R.ENTITY_OFFSET:R.ENTITY_OFFSET + self.n_ent * self.ent_f].reshape(B, self.n_ent, self.ent_f)
        sc = x[:, R.SCALAR_OFFSET:R.SCALAR_OFFSET + R.SCALAR_SIZE]

        ent_id = self._ids(ent[:, :, R.ENTITY_CARD_FEATURE])                   # (B, 64) long
        valid = (ent_id > 0).float().unsqueeze(-1)                            # (B, 64, 1): occupied slots
        ent_in = torch.cat([self.card_embed(ent_id), ent[:, :, R.ENTITY_CARD_FEATURE + 1:]], dim=-1)
        emb = self.entity(ent_in) * valid                                     # ReLU >= 0, empty slots 0
        pooled = []
        for sl in (slice(0, self.half), slice(self.half, self.n_ent)):
            e, v = emb[:, sl], valid[:, sl]
            pooled.append(e.sum(1) / v.sum(1).clamp(min=1.0))                 # masked mean
            pooled.append(e.max(1).values)                                    # masked max (0 if empty)
        sc_ids = self.card_embed(self._ids(sc.index_select(1, self.sc_id_idx))).reshape(B, -1)
        sc_in = torch.cat([sc.index_select(1, self.sc_rest_idx), sc_ids], dim=-1)
        h = torch.cat([self.cnn(sp)] + pooled + [self.scalars(sc_in)], dim=-1)
        return self.trunk(h)

    @staticmethod
    def _ids(v):
        """Float card-id features -> embedding indices in [0, CARD_SLOTS]."""
        return v.round().long().clamp(0, R.CARD_SLOTS)

    def decode_actions(self, hidden):
        """UNMASKED logits and the value (the forward passes apply the mask)."""
        return self.actor(hidden), self.value(hidden)

    # -- PuffeRL entry points ------------------------------------------------------------
    def forward_eval(self, observations, state=None):
        logits, value = self.decode_actions(self.encode_observations(observations, state))
        return mask_logits(logits, observations), value

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class Recurrent(pufferlib.models.LSTMWrapper):
    """LSTM core between the Policy's encoder and decoder (PufferLib 3.0 LSTMWrapper:
    __init__(env, policy, input_size, hidden_size))."""

    def __init__(self, env, policy, input_size=256, hidden_size=256, **kwargs):
        # LSTMWrapper.__init__ re-initialises every parameter it can see -- including the wrapped
        # Policy's (biases to 0, matrices orthogonal gain 1), which throws away the Policy's
        # deliberate init (actor std 0.01, value std 1, the card embedding). Keep the Policy's
        # weights as they were; only the new LSTM uses the wrapper's init. (The wrapper still draws
        # its random numbers, so the global RNG stream is unchanged.)
        keep = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        super().__init__(env, policy, input_size=input_size, hidden_size=hidden_size)
        self.policy.load_state_dict(keep)

    def forward_eval(self, observations, state):
        logits, values = super().forward_eval(observations, state)
        return mask_logits(logits, observations), values

    def forward(self, observations, state):
        # training: observations (segments, horizon, OBS); logits come back as (segments *
        # horizon, 2305) in the same row order as observations.reshape(-1, OBS_SIZE)
        logits, values = super().forward(observations, state)
        return mask_logits(logits, observations), values
