"""pufferroyale.Royale -- the PufferLib 3.0 native environment (SPEC §9).

Follows the Ocean pattern (slimevolley): one C env per match, each initialised with its
slice of the shared buffers, then vectorised in C with ``binding.vectorize``.

Layout constants are read from the compiled binding, so Python and C cannot drift:
OBS_SIZE, SPATIAL_OFFSET, SPATIAL_SHAPE, ENTITY_OFFSET, ENTITY_SHAPE, SCALAR_OFFSET,
SCALAR_SIZE, MASK_OFFSET, MASK_SIZE, plus SCALAR_INDEX (name -> (offset, length) inside the
scalar section, SPEC §13.11), SPATIAL_CHANNELS and ENTITY_FEATURES.

Placement grid (SPEC §19.4): grid_shape(g), n_actions(g) and action_mask(obs, grid) describe the
coarse action space Discrete(1 + 4 B) of Royale(placement_grid=g); the observation keeps the
exact fine 2305-mask. CardStats implements the per-card play-rate logs (SPEC §19.7.6).
"""
from __future__ import annotations

import math
import numbers
from functools import lru_cache

import gymnasium
import numpy as np

import pufferlib

from . import binding
from .decks import deck_entries, parse_deck_set
from .game import CARD_KEYS, DECKS, N_TILES, TOWER_TROOPS, _deck_arg, tower_troop_index

_L = binding.royale_layout()

OBS_SIZE: int = _L["OBS_SIZE"]
SPATIAL_OFFSET: int = _L["SPATIAL_OFFSET"]
SPATIAL_SHAPE: tuple = tuple(_L["SPATIAL_SHAPE"])
ENTITY_OFFSET: int = _L["ENTITY_OFFSET"]
ENTITY_SHAPE: tuple = tuple(_L["ENTITY_SHAPE"])
SCALAR_OFFSET: int = _L["SCALAR_OFFSET"]
SCALAR_SIZE: int = _L["SCALAR_SIZE"]
MASK_OFFSET: int = _L["MASK_OFFSET"]
MASK_SIZE: int = _L["MASK_SIZE"]
#: name -> (offset, length) inside the scalar section, offsets relative to SCALAR_OFFSET,
#: in the SPEC §9.3 order (SPEC v0.2 §13.11)
SCALAR_INDEX: dict = dict(sorted(((k, tuple(v)) for k, v in _L["SCALAR_FIELDS"].items()), key=lambda kv: kv[1][0]))
SCALAR_FIELDS = SCALAR_INDEX  # alias
SPATIAL_CHANNELS: tuple = tuple(_L["SPATIAL_CHANNELS"])
#: SPEC §16.4 entity-row layout: card id + 1 [0], x/18000 [1], y/32000 [2], hp/max_hp [3],
#: min(1, hp/2000) [4], flying [5], deploying [6], stunned [7], slowed [8], is_building [9], tob [10]
ENTITY_FEATURES: tuple = tuple(_L["ENTITY_FEATURES"])
#: SPEC §16.4: width of the card multi-hots; card identities elsewhere are integer ids card_id + 1
CARD_SLOTS: int = _L["CARD_SLOTS"]
N_CARDS: int = _L["N_CARDS"]
#: the entity-row / scalar positions holding integer card ids (card_id + 1, 0 = empty)
ENTITY_CARD_FEATURE: int = 0
CARD_ID_SCALARS: tuple = ("hand", "next_card", "opp_last4", "own_deck")
N_ACTIONS: int = _L["N_ACTIONS"]
RAYLIB_AVAILABLE: bool = bool(_L["RAYLIB"])

OPPONENTS = {"noop": _L["BOT_NOOP"], "random": _L["BOT_RANDOM"], "heuristic": _L["BOT_HEURISTIC"]}
RENDER_MODES = {None: 0, "None": 0, "ansi": 1, "human": 2, "raylib": 2}


def scalar(obs: np.ndarray, name: str) -> np.ndarray:
    """The named field of the scalar section of one observation row (or a batch)."""
    off, n = SCALAR_INDEX[name]
    return obs[..., SCALAR_OFFSET + off:SCALAR_OFFSET + off + n]


def mask(obs: np.ndarray) -> np.ndarray:
    return obs[..., MASK_OFFSET:MASK_OFFSET + MASK_SIZE]


# ------------------------------------------------------------------ placement grid (SPEC §19.4)
PLACEMENT_GRIDS = (1, 2, 4)
DECK_DRAWS = ("independent", "mirror")
_TILES_Y, _TILES_X = SPATIAL_SHAPE[1], SPATIAL_SHAPE[2]


def _check_grid(g) -> int:
    if isinstance(g, bool) or not isinstance(g, numbers.Real) or g not in PLACEMENT_GRIDS:
        raise ValueError(f"placement_grid must be one of {PLACEMENT_GRIDS}, got {g!r}")
    return int(g)


def grid_shape(g: int = 1) -> tuple:
    """(rows, cols) = (ceil(32 / g), ceil(18 / g)): (32, 18), (16, 9) or (8, 5)."""
    g = _check_grid(g)
    return -(-_TILES_Y // g), -(-_TILES_X // g)


def n_actions(g: int = 1) -> int:
    """Size of the action space of placement grid g: 1 + 4 * rows * cols (2305 / 577 / 161)."""
    rows, cols = grid_shape(g)
    return 1 + 4 * rows * cols


@lru_cache(maxsize=None)
def _block_cells(g: int) -> np.ndarray:
    """(B, g*g) fine cells (ty*18 + tx) of every block b = by*cols + bx; a partial block repeats
    its first cell (harmless under max / any)."""
    rows, cols = grid_shape(g)
    out = np.zeros((rows * cols, g * g), dtype=np.int64)
    for by in range(rows):
        for bx in range(cols):
            cells = [ty * _TILES_X + tx for ty in range(g * by, min(_TILES_Y, g * by + g))
                     for tx in range(g * bx, min(_TILES_X, g * bx + g))]
            out[by * cols + bx] = cells + [cells[0]] * (g * g - len(cells))
    out.setflags(write=False)
    return out


def action_mask(obs: np.ndarray, grid: int = 1) -> np.ndarray:
    """Boolean legality mask (..., n_actions(grid)) of observations of any leading shape: the fine
    mask for grid 1; else coarse[0] = 1 and coarse[1 + s*B + b] = any legal fine tile of block b
    for slot s (SPEC §19.4)."""
    g = _check_grid(grid)
    fine = mask(obs) > 0.5
    if g == 1:
        return fine
    lead = fine.shape[:-1]
    slots = fine[..., 1:].reshape(*lead, 4, N_TILES)
    coarse = slots[..., _block_cells(g)].any(-1).reshape(*lead, -1)          # (..., 4 * B)
    return np.concatenate([np.ones((*lead, 1), dtype=bool), coarse], axis=-1)


class CardStats:
    """Per-card decision counters (SPEC §19.7.6), vectorised over agent rows.

    available[c]: decisions where card c is in hand, its slot is affordable and there is no lockout;
    played[c]: decisions whose action plays the slot holding c and is allowed by the action mask
    (the coarse mask for grid > 1). `emit(log)` adds cards/play_rate/<CARD_KEY> (cards with
    available > 0) and cards/decisions to `log`, then resets the counters."""

    def __init__(self, grid: int = 1):
        self.grid = _check_grid(grid)
        self.n_actions = n_actions(self.grid)
        self.n_blocks = (self.n_actions - 1) // 4
        self._cells = _block_cells(self.grid)
        off = lambda name: SCALAR_OFFSET + SCALAR_INDEX[name][0]
        self._hand = off("hand") + np.arange(4)
        self._afford = off("affordable") + np.arange(4)
        self._lockout = off("lockout")
        self.reset()

    def reset(self) -> None:
        self.available = np.zeros(len(CARD_KEYS) + 1, dtype=np.int64)   # index card_id + 1
        self.played = np.zeros(len(CARD_KEYS) + 1, dtype=np.int64)
        self.decisions = 0

    def update(self, obs: np.ndarray, actions) -> None:
        """Count one decision per row of `obs` (the observations the actions were chosen on)."""
        obs = obs.reshape(-1, OBS_SIZE)
        hand = obs[:, self._hand].astype(np.int64)                            # card id + 1, 0 = empty
        ok = (obs[:, self._afford] > 0.5) & (obs[:, self._lockout, None] < 0.5) & (hand > 0)
        self.available += np.bincount(hand[ok], minlength=len(self.available))
        a = np.asarray(actions).reshape(-1).astype(np.int64)
        rows = np.flatnonzero((a >= 1) & (a < self.n_actions))
        if rows.size:
            slot, block = np.divmod(a[rows] - 1, self.n_blocks)
            if self.grid == 1:
                legal = obs[rows, MASK_OFFSET + a[rows]] > 0.5
            else:
                cells = MASK_OFFSET + 1 + slot[:, None] * N_TILES + self._cells[block]
                legal = (obs[rows[:, None], cells] > 0.5).any(-1)
            card = hand[rows, slot]
            self.played += np.bincount(card[legal & (card > 0)], minlength=len(self.played))
        self.decisions += obs.shape[0]

    def emit(self, log: dict) -> dict:
        for c in np.flatnonzero(self.available[1:]):
            log[f"cards/play_rate/{CARD_KEYS[c]}"] = float(self.played[c + 1] / self.available[c + 1])
        log["cards/decisions"] = float(self.decisions)
        self.reset()
        return log


def _finite(name, v, lo=None, lo_open=False, hi=None, allow_inf=False) -> float:
    """A real-valued keyword checked against its SPEC §19 range (ValueError otherwise)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name} must be a number, got {v!r}")
    v = float(v)
    if math.isnan(v) or (math.isinf(v) and not (allow_inf and v > 0)):
        raise ValueError(f"{name} must be finite, got {v!r}")
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ValueError(f"{name} must be {'>' if lo_open else '>='} {lo}, got {v!r}")
    if hi is not None and v > hi:
        raise ValueError(f"{name} must be <= {hi}, got {v!r}")
    return v


def spatial(obs: np.ndarray) -> np.ndarray:
    lead = obs.shape[:-1]
    return obs[..., SPATIAL_OFFSET:SPATIAL_OFFSET + int(np.prod(SPATIAL_SHAPE))].reshape(*lead, *SPATIAL_SHAPE)


def entities(obs: np.ndarray) -> np.ndarray:
    lead = obs.shape[:-1]
    n, f = ENTITY_SHAPE
    return obs[..., ENTITY_OFFSET:ENTITY_OFFSET + n * f].reshape(*lead, n, f)


class Royale(pufferlib.PufferEnv):
    """Clash Royale 1v1 as a PufferLib 3.0 native env.

    num_agents=2: rows 2i / 2i+1 are team 0 / team 1 of env i (self-play).
    num_agents=1: the learner plays `learner_side` (0, 1, or 'random' per episode) against
    the scripted `opponent` ('noop', 'random', 'heuristic').

    SPEC §19 (v0.5) keywords, all defaulting to the v0.4 behaviour:
      reward v2 (§19.1): reward_tower, reward_crown, reward_elixir, reward_play (weights >= 0),
        reward_elixir_cap, reward_play_cap (> 0), reward_gamma (in (0, 1]), shaping_anneal_steps
        (N >= 0, 0 = constant weights), shaping_step_offset (n0 >= 0, env steps already done);
      placement_grid (§19.4): 1, 2 or 4 -> Discrete(n_actions(g)); policy rows decode coarse
        actions, the scripted opponent keeps fine actions (set_row_grid changes a row's grid);
      deck sampling (§19.5): deck_pool, heldout_decks (deck sets, see pufferroyale.decks),
        random_deck_frac in [0, 1], deck_draw 'independent' | 'mirror'. The sampler is active iff
        deck_pool is non-empty or random_deck_frac > 0; deck0 / deck1 are then ignored.
    """

    def __init__(self, num_envs=1, num_agents=2, frame_skip=10, deck0="hog26", deck1="hog26",
                 opponent="heuristic", learner_side="random", deploy_lockout_ticks=90,
                 tiebreak="absolute", reward_tower=0.0, reward_crown=0.0, bot_play_prob=0.2,
                 log_interval=128, render_mode=None, buf=None, seed=0, mask_check=False,
                 alternate_first=False, tower_troop0="princess", tower_troop1="princess",
                 reward_elixir=0.0, reward_play=0.0, reward_elixir_cap=20.0, reward_play_cap=20.0,
                 reward_gamma=1.0, shaping_anneal_steps=0, shaping_step_offset=0, placement_grid=1,
                 deck_pool="", random_deck_frac=0.0, heldout_decks="", deck_draw="independent"):
        if num_agents not in (1, 2):
            raise ValueError("num_agents must be 1 or 2")
        if opponent not in OPPONENTS:
            raise ValueError(f"opponent must be one of {sorted(OPPONENTS)}")
        if learner_side not in (0, 1, "random"):
            raise ValueError("learner_side must be 0, 1 or 'random'")
        if tiebreak not in ("absolute", "fraction"):
            raise ValueError("tiebreak must be 'absolute' or 'fraction'")
        if render_mode not in RENDER_MODES:
            raise ValueError(f"render_mode must be one of {list(RENDER_MODES)}")
        tt0, tt1 = tower_troop_index(tower_troop0), tower_troop_index(tower_troop1)  # SPEC §16.6.27
        self.tower_troops = (TOWER_TROOPS[tt0], TOWER_TROOPS[tt1])
        # SPEC §19.1 reward v2
        self.reward_weights = {k: _finite(k, v, lo=0.0) for k, v in (
            ("reward_tower", reward_tower), ("reward_crown", reward_crown),
            ("reward_elixir", reward_elixir), ("reward_play", reward_play))}
        rewards = dict(self.reward_weights,
                       reward_elixir_cap=_finite("reward_elixir_cap", reward_elixir_cap, lo=0.0, lo_open=True, allow_inf=True),
                       reward_play_cap=_finite("reward_play_cap", reward_play_cap, lo=0.0, lo_open=True, allow_inf=True),
                       reward_gamma=_finite("reward_gamma", reward_gamma, lo=0.0, lo_open=True, hi=1.0),
                       shaping_anneal_steps=_finite("shaping_anneal_steps", shaping_anneal_steps, lo=0.0),
                       shaping_step_offset=_finite("shaping_step_offset", shaping_step_offset, lo=0.0))
        # SPEC §19.4 placement grid
        self.placement_grid = _check_grid(placement_grid)
        # SPEC §19.5 deck sampling
        if deck_draw not in DECK_DRAWS:
            raise ValueError(f"deck_draw must be one of {DECK_DRAWS}, got {deck_draw!r}")
        pool = deck_entries(deck_pool)                             # [(deck, weight, installed order)]
        self.deck_pool = [(d, w) for d, w, _ in pool]
        self.heldout_decks = [d for d, _ in parse_deck_set(heldout_decks)]
        self.random_deck_frac = _finite("random_deck_frac", random_deck_frac, lo=0.0, hi=1.0)
        held = set(self.heldout_decks)
        clash = [list(d) for d, _ in self.deck_pool if d in held]
        if clash:
            raise ValueError(f"deck_pool decks {clash} are also held out (SPEC §19.5)")
        if 0.0 < self.random_deck_frac < 1.0 and not self.deck_pool:
            raise ValueError("random_deck_frac < 1 needs a non-empty deck_pool (SPEC §19.5)")
        if len(self.deck_pool) > 256 or len(self.heldout_decks) > 1024:
            raise ValueError("at most 256 deck_pool decks and 1024 heldout_decks")
        self.deck_sampler = bool(self.deck_pool) or self.random_deck_frac > 0.0
        self.deck_draw = deck_draw
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(n_actions(self.placement_grid))
        self.render_mode = render_mode
        self.num_envs_native = int(num_envs)
        self.agents_per_env = int(num_agents)
        self.num_agents = self.num_envs_native * self.agents_per_env
        # PuffeRL's LSTM path reads vecenv.agents_per_batch; PufferLib 3.0's native PufferEnv only
        # defines agent_per_batch, so expose the spelling the trainer uses.
        self.agents_per_batch = self.num_agents
        self.log_interval = int(log_interval)
        self.frame_skip = int(frame_skip)
        self.seed = 0 if seed is None else int(seed)

        super().__init__(buf)
        n = self.agents_per_env
        kwargs = dict(
            num_agents=n,
            frame_skip=self.frame_skip,
            opponent=OPPONENTS[opponent],
            learner_side=-1 if learner_side == "random" else int(learner_side),
            deploy_lockout_ticks=int(deploy_lockout_ticks),
            tiebreak=0 if tiebreak == "absolute" else 1,
            bot_play_prob=float(bot_play_prob),
            deck0=_deck_arg(deck0),
            deck1=_deck_arg(deck1),
            mask_check=int(bool(mask_check)),
            alternate_first=int(bool(alternate_first)),
            render_mode=RENDER_MODES[render_mode],
            tower_troop0=tt0,
            tower_troop1=tt1,
            placement_grid=self.placement_grid,
            deck_pool=[list(order) for _, _, order in pool],       # ruling v0.5-G.1: presets keep their order
            deck_weights=[w for _, w in self.deck_pool],
            heldout_decks=[list(d) for d in self.heldout_decks],
            random_deck_frac=self.random_deck_frac,
            deck_draw=DECK_DRAWS.index(deck_draw),
            **rewards,
        )
        self._c_env_list = []
        for i in range(self.num_envs_native):
            c_env = binding.env_init(
                self.observations[i * n:(i + 1) * n],
                self.actions[i * n:(i + 1) * n],
                self.rewards[i * n:(i + 1) * n],
                self.terminals[i * n:(i + 1) * n],
                self.truncations[i * n:(i + 1) * n],
                self.seed * self.num_envs_native + i,
                **kwargs,
            )
            self._c_env_list.append(c_env)
        self.c_envs = binding.vectorize(*self._c_env_list)
        self.tick = 0
        self.card_stats = CardStats(self.placement_grid)

    def reset(self, seed=None):
        """Re-deal every match. With `seed`, env i reseeds all its streams (game, bot, side, deck
        sampler) from seed * num_envs + i; without, every stream continues."""
        binding.vec_reset_seeded(self.c_envs, None if seed is None else int(seed))
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.card_stats.update(self.observations, self.actions)  # every row is a policy row
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(self.card_stats.emit(log))
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        """'ansi': the text board of env 0. 'human'/'raylib': the raylib window when the
        extension was built with PR_RAYLIB=1, else the text board printed to stdout."""
        if self.render_mode == "ansi":
            return binding.env_ansi(self._c_env_list[0])
        binding.vec_render(self.c_envs, 0)
        return None

    def ansi(self, env_index=0) -> str:
        return binding.env_ansi(self._c_env_list[env_index])

    def env_info(self, env_index=0) -> dict:
        """State of native match `env_index`: tick, learner, steps, hash, elixir, crowns, decks, and
        (SPEC §19) deck_sampler / deck0_ignored / deck1_ignored, placement_grid, row_grids, env_steps
        (c_steps since creation) and shaping_multiplier (the anneal factor of the next step)."""
        return binding.env_info(self._c_env_list[env_index])

    def set_row_grid(self, env_index: int, row: int, grid: int) -> None:
        """The placement grid agent row `row` (0/1) of native match `env_index` decodes its actions
        with from the next step on (1 = fine actions). LeagueVecEnv uses it so that a bot opponent
        keeps playing fine actions under placement_grid > 1."""
        binding.env_set_row_grid(self._c_env_list[env_index], int(row), _check_grid(grid))

    def deck_counts(self) -> dict:
        """SPEC §19.5 sampler counts, cumulative since construction and summed over the matches:
        {"pool": int64 array (len(deck_pool),) of seats dealt each pool deck (a mirror deal counts
        both seats), "random": seats dealt a random deck, "random_rejected": held-out redraws}."""
        pool = np.zeros(len(self.deck_pool), dtype=np.int64)
        rnd = rej = 0
        for h in self._c_env_list:
            p, r, j = binding.env_deck_counts(h)
            pool += np.asarray(p, dtype=np.int64)
            rnd, rej = rnd + r, rej + j
        return {"pool": pool, "random": int(rnd), "random_rejected": int(rej)}

    def close(self):
        if self.c_envs is not None:
            binding.vec_close(self.c_envs)
            self.c_envs = None


def make(**kwargs) -> Royale:
    return Royale(**kwargs)


__all__ = ["Royale", "OBS_SIZE", "SPATIAL_OFFSET", "SPATIAL_SHAPE", "ENTITY_OFFSET", "ENTITY_SHAPE",
           "SCALAR_OFFSET", "SCALAR_SIZE", "MASK_OFFSET", "MASK_SIZE", "SCALAR_INDEX", "SCALAR_FIELDS", "SPATIAL_CHANNELS",
           "ENTITY_FEATURES", "CARD_SLOTS", "N_CARDS", "ENTITY_CARD_FEATURE", "CARD_ID_SCALARS", "TOWER_TROOPS",
           "scalar", "mask", "spatial", "entities", "DECKS", "PLACEMENT_GRIDS", "DECK_DRAWS", "grid_shape",
           "n_actions", "action_mask", "CardStats"]
