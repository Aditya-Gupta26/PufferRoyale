"""pufferroyale.torch -- policies for PufferLib 3.0 PuffeRL (SPEC §11, v0.3 observation §16.4, v0.5 §19.6).

Policy: a CNN over the spatial planes, a shared per-entity MLP with masked mean and max
pooling over the own and enemy entity slots, and an MLP over the scalars, concatenated into
an MLP trunk; a value head and one of two action heads over Discrete(1 + 4 B) (B = 576 / 144 /
40 placement blocks for placement_grid 1 / 2 / 4, SPEC §19.4):

    head="flat"          the v0.4 head: actor = Linear(hidden, A)
    head="conditional"   card first, then position given the card (SPEC §19.6):
                         P(wait) and P(slot s) from the trunk vector h and an MLP of
                         [h, enc(hand_s)]; per slot a 32 x 18 heat map from a full-resolution
                         board CNN, FiLM-conditioned on that slot's [h, enc(hand_s)] and
                         mapped by a 1x1 conv, averaged to the grid's blocks when g > 1;
                         joint[0] = log P(wait), joint[1 + s B + j] = log P(s) + log P(j | s)

Card identities (SPEC §16.4) arrive as integer ids card_id + 1 (0 = empty) in the entity rows
(feature 0) and in the hand (4), next-card (1), opponent last-4 (4) and own-deck (8, §19.3)
scalars. Every id is encoded as enc(id) = [Embedding(CARD_SLOTS + 1, card_dim)(id),
Linear(K, card_dim)(S[id])] with card_stats on, else the embedding alone; S is the fixed
card-stat table of card_stat_table() (a non-persistent buffer). The encoded entity id is
concatenated with the row's other 10 features, the 17 encoded scalar ids with the remaining
scalars. The 128-wide multi-hots (opponent cards seen / deduced hand) stay plain inputs. An
entity slot is occupied iff its card id is non-zero.

Action masking: every forward pass (Policy.forward / forward_eval and Recurrent.forward /
forward_eval) builds the legality mask from the observations it was given -- the exact fine
mask, or its coarse version for g > 1 (action_mask) -- and sets illegal logits to
torch.finfo(dtype).min (not -inf, so entropies stay finite). There is no state kept between
calls and no silent skip: a mask whose shape does not match the logits raises.

Recurrent: PufferLib's LSTMWrapper hands only the LSTM output to decode_actions, but the
conditional head also needs the board planes, the hand and the mask of the same rows. Recurrent
therefore runs the wrapper's own encode -> LSTM -> decode steps (statement for statement) and
calls Policy.decode_actions(hidden, observations) with the observations of exactly those rows
(the flattened (B*T) row order of the hidden batch equals observations.reshape(-1, OBS_SIZE)).
Nothing is stashed on the module between the calls.

The placement grid of a policy is a persistent buffer (`action_grid`), so a checkpoint carries
it (the conditional head has no grid-dependent parameter); loading a state dict of another grid
raises.
"""
from __future__ import annotations

import numbers
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

import pufferlib.models
import pufferlib.pytorch

from . import royale as R

N_TILES = R.SPATIAL_SHAPE[1] * R.SPATIAL_SHAPE[2]
#: action-space size -> placement grid (2305 -> 1, 577 -> 2, 161 -> 4)
GRID_OF_ACTIONS = {R.n_actions(g): g for g in R.PLACEMENT_GRIDS}
HEADS = ("flat", "conditional")


def _init(layer, std=np.sqrt(2)):
    return pufferlib.pytorch.layer_init(layer, std=std)


def _conv_out(n, k=3, s=2, p=1):
    return (n + 2 * p - k) // s + 1


def _grid(g) -> int:
    if isinstance(g, bool) or not isinstance(g, numbers.Real) or g not in R.PLACEMENT_GRIDS:
        raise ValueError(f"placement_grid must be one of {R.PLACEMENT_GRIDS}, got {g!r}")
    return int(g)


def grid_of_actions(n: int) -> int:
    """The placement grid whose action space has n actions; ValueError otherwise."""
    if int(n) not in GRID_OF_ACTIONS:
        raise ValueError(f"{n} actions is no placement grid's action space ({sorted(GRID_OF_ACTIONS)})")
    return GRID_OF_ACTIONS[int(n)]


# ======================================================================================
# placement-grid masks and pooling (SPEC §19.4)
# ======================================================================================
@lru_cache(maxsize=None)
def _blocks(g: int):
    """(cells (B, g*g) int64 -- every block's fine cells ty*18 + tx, a partial block repeating its
    first cell --, pool (576, B) float32 -- column b averages block b over its existing tiles)."""
    rows, cols = R.grid_shape(g)
    H, W = R.SPATIAL_SHAPE[1], R.SPATIAL_SHAPE[2]
    cells = np.zeros((rows * cols, g * g), dtype=np.int64)
    pool = np.zeros((N_TILES, rows * cols), dtype=np.float32)
    for by in range(rows):
        for bx in range(cols):
            b = by * cols + bx
            cs = [ty * W + tx for ty in range(g * by, min(H, g * by + g)) for tx in range(g * bx, min(W, g * bx + g))]
            cells[b] = cs + [cs[0]] * (g * g - len(cs))
            pool[cs, b] = 1.0 / len(cs)
    return cells, pool


@lru_cache(maxsize=None)
def _blocks_on(g: int, device: str):
    cells, pool = _blocks(g)
    return torch.as_tensor(cells, device=device), torch.as_tensor(pool, device=device)


def action_mask(observations, grid=1):
    """Boolean legality mask (..., 1 + 4 B) of observations of any leading shape, equal to
    pufferroyale.royale.action_mask: the fine mask for grid 1; else coarse[0] = 1 and
    coarse[1 + s B + b] = any legal fine tile of block b for slot s (SPEC §19.4)."""
    g = _grid(grid)
    lead = tuple(observations.shape[:-1])
    fine = observations[..., R.MASK_OFFSET:R.MASK_OFFSET + R.MASK_SIZE] > 0.5
    if g == 1:
        return fine
    cells, _ = _blocks_on(g, str(fine.device))
    slots = fine[..., 1:].reshape(*lead, 4, N_TILES)
    coarse = slots[..., cells].any(-1).reshape(*lead, 4 * cells.shape[0])
    return torch.cat([torch.ones(*lead, 1, dtype=torch.bool, device=fine.device), coarse], dim=-1)


def mask_logits(logits, observations, grid=None):
    """Illegal actions of each row get torch.finfo(dtype).min (zero probability). The grid is
    taken from the logits' width unless given."""
    g = grid_of_actions(logits.shape[-1]) if grid is None else _grid(grid)
    mask = action_mask(observations, g).reshape(-1, R.n_actions(g))
    if mask.shape != logits.shape:
        raise ValueError(f"action mask {tuple(mask.shape)} does not match logits {tuple(logits.shape)}")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def policy_grid(policy) -> int:
    """The placement grid a policy acts on (unwraps Recurrent / DDP / torch.compile)."""
    for attr in ("module", "_orig_mod", "policy"):
        while hasattr(policy, attr) and not hasattr(policy, "placement_grid"):
            policy = getattr(policy, attr)
    return int(getattr(policy, "placement_grid", 1))


# ======================================================================================
# card-stat table (SPEC §19.6)
# ======================================================================================
#: column names of card_stat_table(), in order
CARD_STAT_NAMES = ("elixir", "is_troop", "is_building", "is_spell", "units", "hitpoints", "damage",
                   "hit_speed_ms", "dps", "range_milli", "sight_range_milli", "speed", "flying", "attacks_air",
                   "attacks_ground", "target_only_buildings", "splash_radius", "crown_tower_damage_percent",
                   "lifetime_ms", "death_damage", "deploy_time_ms", "jumps", "charges", "spawns_units")
#: columns scaled after log1p
CARD_STAT_LOG = ("hitpoints", "damage", "dps", "death_damage")
N_CARD_STATS = len(CARD_STAT_NAMES)


def _num(d, key) -> float:
    """A numeric card_info field; missing / None / non-numeric -> 0, booleans -> 0 / 1."""
    v = d.get(key) if isinstance(d, dict) else None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, numbers.Real):
        return float(v)
    return 0.0


def card_stat_row(info: dict) -> list:
    """The K raw (unscaled, before log1p) stat values of one card from its binding.card_info dict
    (SPEC §19.6, §19.9.7). Missing / None -> 0, booleans 0 / 1. Spells: the unit-only columns
    (units summoned, hitpoints, hit speed, DPS, range, sight, speed, flying, attacks air / ground,
    buildings-only, lifetime, death damage, deploy time, jumps, charges) are 0, damage = the spell's
    damage, splash = its radius, spawns units = 1 iff card_info["spawn"] is non-empty. Units: splash
    = max(area_damage_radius_milli, projectile radius), spawns units = spawner or death_spawn."""
    kind = info.get("card_kind")
    spell = kind == "spell"
    elixir, dmg, ctp = _num(info, "elixir"), _num(info, "damage"), _num(info, "crown_tower_damage_percent")
    head = [elixir, float(kind == "troop"), float(kind == "building"), float(spell)]
    if spell:
        row = dict(damage=dmg, splash_radius=_num(info, "radius"), crown_tower_damage_percent=ctp,
                   spawns_units=float(bool(info.get("spawn"))))
        return head + [row.get(name, 0.0) for name in CARD_STAT_NAMES[4:]]
    hs = _num(info, "hit_speed_ms")
    proj = info.get("projectile") if isinstance(info.get("projectile"), dict) else {}
    return head + [
        _num(info, "count") + _num(info, "count2"),
        _num(info, "hitpoints"),
        dmg,
        hs,
        dmg * 1000.0 / hs if hs > 0 else 0.0,
        _num(info, "range_milli"),
        _num(info, "sight_range_milli"),
        _num(info, "speed"),
        float(_num(info, "flying_height") > 0),
        _num(info, "attacks_air"),
        _num(info, "attacks_ground"),
        _num(info, "target_only_buildings"),
        max(_num(info, "area_damage_radius_milli"), _num(proj, "radius")),
        ctp,
        _num(info, "lifetime_ms"),
        _num(info, "death_damage"),
        _num(info, "deploy_time_ms"),
        _num(info, "jumps"),
        float(_num(info, "charge_range") > 0),
        float(bool(info.get("spawner")) or bool(info.get("death_spawn"))),
    ]


@lru_cache(maxsize=None)
def _card_stat_table():
    from . import binding
    raw = np.array([card_stat_row(binding.card_info(c)) for c in range(R.N_CARDS)], dtype=np.float64)
    raw = np.maximum(raw, 0.0)                                   # no card has a negative value; kept in [0, 1]
    for name in CARD_STAT_LOG:
        j = CARD_STAT_NAMES.index(name)
        raw[:, j] = np.log1p(raw[:, j])
    mx = raw.max(axis=0)
    scaled = np.divide(raw, mx, out=np.zeros_like(raw), where=mx > 0)
    table = np.zeros((R.CARD_SLOTS + 1, N_CARD_STATS), dtype=np.float64)
    table[1:R.N_CARDS + 1] = scaled                              # row card_id + 1; row 0 / unused ids = 0
    table.setflags(write=False)
    return table


def card_stat_table():
    """SPEC §19.6: (S, names) -- S is the (CARD_SLOTS + 1, K) float64 card-stat table, row
    card_id + 1 (row 0 and unused ids are 0), built from binding.card_info for the 64 cards; every
    column is scaled into [0, 1] by its maximum over the 64 cards (log1p first for hitpoints,
    damage, DPS and death damage). names = CARD_STAT_NAMES."""
    return _card_stat_table().copy(), list(CARD_STAT_NAMES)


def _card_id_positions():
    """Indices (inside the scalar section) of the integer card-id scalars, and of all the others."""
    ids = []
    for name in R.CARD_ID_SCALARS:
        off, n = R.SCALAR_INDEX[name]
        ids.extend(range(off, off + n))
    rest = [i for i in range(R.SCALAR_SIZE) if i not in set(ids)]
    return ids, rest


def _flag(v) -> bool:
    """A 0/1 config flag (ini / CLI ints, bools, or 'true'/'false' strings)."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        raise ValueError(f"expected a 0/1 flag, got {v!r}")
    return bool(v)


# ======================================================================================
# Policy
# ======================================================================================
class Policy(nn.Module):
    def __init__(self, env=None, hidden_size=256, cnn_channels=64, entity_hidden=128, scalar_hidden=128, card_dim=16,
                 head="conditional", card_stats=1, pos_channels=32, placement_grid=None, **kwargs):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        space = getattr(env, "single_action_space", None)
        n = int(space.n) if space is not None and hasattr(space, "n") else None
        if placement_grid is None:
            g = 1 if n is None else grid_of_actions(n)
        else:
            g = _grid(placement_grid)
            if n is not None and n != R.n_actions(g):
                raise ValueError(f"placement_grid {g} has {R.n_actions(g)} actions but the env has {n}")
        self.placement_grid = g
        self.n_actions = R.n_actions(g)
        self.n_blocks = (self.n_actions - 1) // 4
        self.head = head
        self.card_stats = _flag(card_stats)
        self.pos_channels = int(pos_channels)
        if self.pos_channels < 1:
            raise ValueError("pos_channels must be >= 1")
        self.policy_kwargs = dict(hidden_size=int(hidden_size), cnn_channels=int(cnn_channels),
                                  entity_hidden=int(entity_hidden), scalar_hidden=int(scalar_hidden),
                                  card_dim=int(card_dim), head=head, card_stats=int(self.card_stats),
                                  pos_channels=self.pos_channels, placement_grid=g)
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
        enc_dim = card_dim
        if self.card_stats:                       # SPEC §19.6: + Linear(K, card_dim)(S[id])
            self.card_stat_proj = _init(nn.Linear(N_CARD_STATS, card_dim), std=1.0)
            self.register_buffer("card_stat_table", torch.from_numpy(_card_stat_table().astype(np.float32)),
                                 persistent=False)
            enc_dim = 2 * card_dim
        self.enc_dim = enc_dim
        ids, rest = _card_id_positions()
        self.register_buffer("sc_id_idx", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.register_buffer("sc_rest_idx", torch.tensor(rest, dtype=torch.long), persistent=False)
        hand_off, _ = R.SCALAR_INDEX["hand"]
        self.register_buffer("sc_hand_idx", torch.arange(hand_off, hand_off + 4, dtype=torch.long), persistent=False)
        self.register_buffer("action_grid", torch.tensor(g, dtype=torch.long))      # persistent: in checkpoints
        h2, w2 = _conv_out(_conv_out(H)), _conv_out(_conv_out(W))
        self.cnn = nn.Sequential(
            _init(nn.Conv2d(C, cnn_channels, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(cnn_channels * h2 * w2, hidden_size)), nn.ReLU(),
        )
        self.entity = nn.Sequential(
            _init(nn.Linear(self.ent_f - 1 + enc_dim, entity_hidden)), nn.ReLU(),
            _init(nn.Linear(entity_hidden, entity_hidden)), nn.ReLU(),
        )
        self.scalars = nn.Sequential(_init(nn.Linear(len(rest) + len(ids) * enc_dim, scalar_hidden)), nn.ReLU())
        trunk_in = hidden_size + 4 * entity_hidden + scalar_hidden
        self.trunk = nn.Sequential(
            _init(nn.Linear(trunk_in, hidden_size)), nn.ReLU(),
            _init(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
        )
        if head == "flat":
            self.actor = _init(nn.Linear(hidden_size, self.n_actions), std=0.01)
        else:
            P = self.pos_channels
            # (a) card logits: wait from h; slot s from an MLP of [h, enc(hand_s)], shared over slots
            self.wait_logit = _init(nn.Linear(hidden_size, 1), std=0.01)
            self.slot_cond = nn.Sequential(_init(nn.Linear(hidden_size + enc_dim, hidden_size)), nn.ReLU())
            self.slot_logit = _init(nn.Linear(hidden_size, 1), std=0.01)
            # (b) full-resolution board features (receptive field 15 tiles: dilations 1, 2, 4), FiLM per
            # slot from the same [h, enc(hand_s)] conditioning, 1x1 conv to a 32 x 18 logit map
            self.pos_board = nn.Sequential(
                _init(nn.Conv2d(C, P, 3, padding=1)), nn.ReLU(),
                _init(nn.Conv2d(P, P, 3, padding=2, dilation=2)), nn.ReLU(),
                _init(nn.Conv2d(P, P, 3, padding=4, dilation=4)), nn.ReLU(),
            )
            self.film = _init(nn.Linear(hidden_size, 2 * P), std=0.01)
            self.pos_out = _init(nn.Conv2d(P, 1, 1), std=0.01)
        self.value = _init(nn.Linear(hidden_size, 1), std=1.0)

    # -- card encoding (SPEC §19.6) ----------------------------------------------------
    @staticmethod
    def _ids(v):
        """Float card-id features -> embedding indices in [0, CARD_SLOTS]."""
        return v.round().long().clamp(0, R.CARD_SLOTS)

    def encode_cards(self, ids):
        """enc(id) for a long tensor of card ids + 1 (any shape) -> (..., enc_dim)."""
        e = self.card_embed(ids)
        if self.card_stats:
            e = torch.cat([e, self.card_stat_proj(self.card_stat_table[ids])], dim=-1)
        return e

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
        ent_in = torch.cat([self.encode_cards(ent_id), ent[:, :, R.ENTITY_CARD_FEATURE + 1:]], dim=-1)
        emb = self.entity(ent_in) * valid                                     # ReLU >= 0, empty slots 0
        pooled = []
        for sl in (slice(0, self.half), slice(self.half, self.n_ent)):
            e, v = emb[:, sl], valid[:, sl]
            pooled.append(e.sum(1) / v.sum(1).clamp(min=1.0))                 # masked mean
            pooled.append(e.max(1).values)                                    # masked max (0 if empty)
        sc_ids = self.encode_cards(self._ids(sc.index_select(1, self.sc_id_idx))).reshape(B, -1)
        sc_in = torch.cat([sc.index_select(1, self.sc_rest_idx), sc_ids], dim=-1)
        h = torch.cat([self.cnn(sp)] + pooled + [self.scalars(sc_in)], dim=-1)
        return self.trunk(h)

    def decode_actions(self, hidden, observations=None):
        """(logits, value) from the trunk / LSTM vector `hidden`. With `observations` (the same rows,
        any leading shape) the logits are the masked joint logits of the head. Without them (the
        bare PufferLib protocol) the flat head returns UNMASKED logits; the conditional head needs
        the observations (board, hand, mask) and raises."""
        value = self.value(hidden)
        if self.head == "flat":
            logits = self.actor(hidden)
            if observations is None:
                return logits, value
            return mask_logits(logits, observations, self.placement_grid), value
        if observations is None:
            raise RuntimeError("the conditional head decodes from the observations too: call "
                               "decode_actions(hidden, observations) (Policy / Recurrent forward do)")
        return self._conditional(hidden, observations)["logits"], value

    def _conditional(self, hidden, observations):
        """The conditional head's parts for rows `hidden` (N, hidden) / observations (N rows):
        logits (N, A) masked joint logits; card_logp (N, 5) log P(wait), log P(slot s);
        pos_logp (N, 4, B) log P(j | s); mask (N, A) bool."""
        x = observations.reshape(-1, R.OBS_SIZE).float()
        N = x.shape[0]
        C, H, W = self.spatial_shape
        Bk, P = self.n_blocks, self.pos_channels
        mask = action_mask(x, self.placement_grid)                            # (N, 1 + 4B)
        slot_mask = mask[:, 1:].reshape(N, 4, Bk)
        neg = torch.finfo(torch.float32).min
        hand = self.encode_cards(self._ids(x[:, R.SCALAR_OFFSET:R.SCALAR_OFFSET + R.SCALAR_SIZE]
                                            .index_select(1, self.sc_hand_idx)))           # (N, 4, enc)
        cond = self.slot_cond(torch.cat([hidden.unsqueeze(1).expand(N, 4, hidden.shape[-1]), hand], dim=-1))
        # (a) card distribution: wait + 4 slots, a slot masked iff its segment has no legal action
        card = torch.cat([self.wait_logit(hidden), self.slot_logit(cond).squeeze(-1)], dim=-1).float()
        card_ok = torch.cat([mask[:, :1], slot_mask.any(-1)], dim=-1)
        card_logp = torch.log_softmax(torch.where(card_ok, card, neg), dim=-1)                 # (N, 5)
        # (b) position distribution per slot over the grid's blocks
        board = self.pos_board(x[:, R.SPATIAL_OFFSET:R.SPATIAL_OFFSET + C * H * W].reshape(N, C, H, W))
        gamma, beta = self.film(cond).reshape(N, 4, 2, P).unbind(2)                           # (N, 4, P) each
        feat = torch.relu(board.unsqueeze(1) * (1.0 + gamma)[..., None, None] + beta[..., None, None])
        pos = self.pos_out(feat.reshape(N * 4, P, H, W)).reshape(N, 4, H * W).float()       # ty*18 + tx
        # SPEC §19.10.6: pooling, masking and log-softmax in float32 even under bf16 autocast (an autocast
        # matmul would return bf16, where finfo(float32).min rounds to -inf: all -inf rows -> NaN)
        with torch.autocast(device_type=pos.device.type, enabled=False):
            if self.placement_grid > 1:                                       # block means, b = by*cols + bx
                _, pool = _blocks_on(self.placement_grid, str(pos.device))
                pos = pos @ pool
            pos_logp = torch.log_softmax(torch.where(slot_mask, pos, neg), dim=-1)            # (N, 4, B)
            # (c) joint logits; illegal entries exactly finfo.min (as v0.4)
            joint = torch.cat([card_logp[:, :1], (card_logp[:, 1:, None] + pos_logp).reshape(N, 4 * Bk)], dim=-1)
            logits = torch.where(mask, joint, neg)
        return {"logits": logits, "card_logp": card_logp, "pos_logp": pos_logp, "mask": mask}

    def conditional_parts(self, observations):
        """Diagnostics of the conditional head (no grad state is changed): the dict of _conditional
        for these observations (card_logp, pos_logp, logits, mask)."""
        if self.head != "conditional":
            raise ValueError("conditional_parts needs head='conditional'")
        return self._conditional(self.encode_observations(observations), observations)

    # -- checkpoints -------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        key = prefix + "action_grid"
        if key in state_dict and int(state_dict[key]) != self.placement_grid:
            error_msgs.append(f"{key}: the checkpoint acts on placement_grid {int(state_dict[key])}, this policy on "
                              f"{self.placement_grid} (SPEC §19.4)")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    # -- PuffeRL entry points ------------------------------------------------------------
    def forward_eval(self, observations, state=None):
        return self.decode_actions(self.encode_observations(observations, state), observations)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class Recurrent(pufferlib.models.LSTMWrapper):
    """LSTM core between the Policy's encoder and decoder (PufferLib 3.0 LSTMWrapper:
    __init__(env, policy, input_size, hidden_size)). forward / forward_eval are the wrapper's own
    (statement for statement), except that the decoder also receives the observations of the
    rows it decodes (SPEC §19.6; see the module docstring)."""

    def __init__(self, env, policy, input_size=256, hidden_size=256, **kwargs):
        # LSTMWrapper.__init__ re-initialises every parameter it can see -- including the wrapped
        # Policy's (biases to 0, matrices orthogonal gain 1), which throws away the Policy's
        # deliberate init (actor std 0.01, value std 1, the card embedding). Keep the Policy's
        # weights as they were; only the new LSTM uses the wrapper's init. (The wrapper still draws
        # its random numbers, so the global RNG stream is unchanged.)
        keep = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        super().__init__(env, policy, input_size=input_size, hidden_size=hidden_size)
        self.policy.load_state_dict(keep)
        self.placement_grid = getattr(policy, "placement_grid", 1)

    def forward_eval(self, observations, state):
        hidden = self.policy.encode_observations(observations, state=state)
        h = state.get('lstm_h')
        c = state.get('lstm_c')
        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0], 'LSTM state must be (h, c)'
            lstm_state = (h, c)
        else:
            lstm_state = None
        hidden, c = self.cell(hidden, lstm_state)
        state['hidden'] = hidden
        state['lstm_h'] = hidden
        state['lstm_c'] = c
        return self.policy.decode_actions(hidden, observations)

    def forward(self, observations, state):
        # training: observations (segments, horizon, OBS); logits come back as (segments * horizon, A)
        # in the same row order as observations.reshape(-1, OBS_SIZE)
        x = observations
        lstm_h = state['lstm_h']
        lstm_c = state['lstm_c']

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if lstm_h is not None:
            assert lstm_h.shape[1] == lstm_c.shape[1] == B, 'LSTM state must be (h, c)'
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        x = x.reshape(B*TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)

        hidden = hidden.transpose(0, 1)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        hidden = hidden.float()

        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(B*TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden, x)
        values = values.reshape(B, TT)
        state['hidden'] = hidden
        state['lstm_h'] = lstm_h.detach()
        state['lstm_c'] = lstm_c.detach()
        return logits, values


__all__ = ["Policy", "Recurrent", "action_mask", "mask_logits", "card_stat_table", "card_stat_row", "policy_grid",
           "grid_of_actions", "CARD_STAT_NAMES", "N_CARD_STATS", "HEADS"]
