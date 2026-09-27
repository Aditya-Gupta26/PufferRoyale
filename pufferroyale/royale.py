"""pufferroyale.Royale -- the PufferLib 3.0 native environment (SPEC §9).

Follows the Ocean pattern (slimevolley): one C env per match, each initialised with its
slice of the shared buffers, then vectorised in C with ``binding.vectorize``.

Layout constants are read from the compiled binding, so Python and C cannot drift:
OBS_SIZE, SPATIAL_OFFSET, SPATIAL_SHAPE, ENTITY_OFFSET, ENTITY_SHAPE, SCALAR_OFFSET,
SCALAR_SIZE, MASK_OFFSET, MASK_SIZE, plus SCALAR_INDEX (name -> (offset, length) inside the
scalar section, SPEC §13.11), SPATIAL_CHANNELS and ENTITY_FEATURES.
"""
from __future__ import annotations

import gymnasium
import numpy as np

import pufferlib

from . import binding
from .game import DECKS, TOWER_TROOPS, _deck_arg, tower_troop_index

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
CARD_ID_SCALARS: tuple = ("hand", "next_card", "opp_last4")
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
    """

    def __init__(self, num_envs=1, num_agents=2, frame_skip=10, deck0="hog26", deck1="hog26",
                 opponent="heuristic", learner_side="random", deploy_lockout_ticks=90,
                 tiebreak="absolute", reward_tower=0.0, reward_crown=0.0, bot_play_prob=0.2,
                 log_interval=128, render_mode=None, buf=None, seed=0, mask_check=False,
                 alternate_first=False, tower_troop0="princess", tower_troop1="princess"):
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
        self.single_observation_space = gymnasium.spaces.Box(
            low=0.0, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(N_ACTIONS)
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
            reward_tower=float(reward_tower),
            reward_crown=float(reward_crown),
            bot_play_prob=float(bot_play_prob),
            deck0=_deck_arg(deck0),
            deck1=_deck_arg(deck1),
            mask_check=int(bool(mask_check)),
            alternate_first=int(bool(alternate_first)),
            render_mode=RENDER_MODES[render_mode],
            tower_troop0=tt0,
            tower_troop1=tt1,
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

    def reset(self, seed=None):
        """Re-deal every match. With `seed`, env i reseeds all its streams (game, bot, side)
        from seed * num_envs + i; without, every stream continues."""
        binding.vec_reset_seeded(self.c_envs, None if seed is None else int(seed))
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
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
        return binding.env_info(self._c_env_list[env_index])

    def close(self):
        if self.c_envs is not None:
            binding.vec_close(self.c_envs)
            self.c_envs = None


def make(**kwargs) -> Royale:
    return Royale(**kwargs)


__all__ = ["Royale", "OBS_SIZE", "SPATIAL_OFFSET", "SPATIAL_SHAPE", "ENTITY_OFFSET", "ENTITY_SHAPE",
           "SCALAR_OFFSET", "SCALAR_SIZE", "MASK_OFFSET", "MASK_SIZE", "SCALAR_INDEX", "SCALAR_FIELDS", "SPATIAL_CHANNELS",
           "ENTITY_FEATURES", "CARD_SLOTS", "N_CARDS", "ENTITY_CARD_FEATURE", "CARD_ID_SCALARS", "TOWER_TROOPS",
           "scalar", "mask", "spatial", "entities", "DECKS"]
