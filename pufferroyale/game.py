"""pufferroyale.Game -- single-match, tick-level scenario / debug API (SPEC §10).

Everything here is a thin wrapper over the compiled engine (pufferroyale.binding);
no game logic lives in Python. Coordinates are engine-frame millitiles (1 tile =
1000, y = 0 is the top edge, team 0 is the bottom player) unless a method says it
takes an own-frame tile.
"""
from __future__ import annotations

import enum
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from . import binding as _b

_DB = _b.db_info()

#: The 64 card display names; index == stable card id (SPEC §0, §16.1).
CARD_NAMES: tuple = tuple(_DB["card_names"])
#: Source keys in data/source/cards-15.535.json, index == card id.
CARD_KEYS: tuple = tuple(_DB["card_keys"])
#: Elixir cost per card id.
CARD_COSTS: tuple = tuple(_DB["card_costs"])
#: Preset decks: name -> tuple of 8 card ids (SPEC §0, §16.4). 'random' is not a preset.
DECKS: dict = {k: tuple(v) for k, v in _DB["decks"].items()}

N_CARDS: int = _DB["n_cards"]
#: SPEC §16.3 tower troops, index order of §16.6.16 ('princess' is the default).
TOWER_TROOPS: tuple = tuple(_DB["tower_troops"])
CARD_SLOTS: int = _DB["card_slots"]
N_ACTIONS: int = _DB["n_actions"]
N_TILES: int = _DB["n_tiles"]
ELIXIR_UNIT = 2800

TIEBREAKS = {"absolute": 0, "fraction": 1}


def tower_troop_index(name) -> int:
    """SPEC §16.3 / §16.6.27: 'princess' | 'cannoneer' | 'dagger_duchess' | 'royal_chef' -> its index;
    anything else raises ValueError."""
    if isinstance(name, str) and name in TOWER_TROOPS:
        return TOWER_TROOPS.index(name)
    raise ValueError(f"tower troop must be one of {list(TOWER_TROOPS)}, got {name!r}")


class PlayError(enum.IntEnum):
    """Error codes returned by Game.play / Game.play_tile (SPEC §10)."""

    OK = 0
    NOT_IN_HAND = 1
    BAD_SLOT = 1  # alias of NOT_IN_HAND
    NOT_ENOUGH_ELIXIR = 2
    ILLEGAL_POSITION = 3
    LOCKOUT = 4
    GAME_OVER = 5


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


_NAME_TO_ID = {}
for _i, (_n, _k) in enumerate(zip(CARD_NAMES, CARD_KEYS)):
    _NAME_TO_ID[_norm(_n)] = _i
    _NAME_TO_ID[_norm(_k)] = _i
_NAME_TO_ID.setdefault("icespirits", _NAME_TO_ID["icespirit"])


def card_id(card: Union[int, str]) -> int:
    """Resolve a card id from an int id, a display name ("Hog Rider") or a source key."""
    if isinstance(card, (int, np.integer)) and not isinstance(card, bool):
        c = int(card)
        if not 0 <= c < N_CARDS:
            raise ValueError(f"card id {c} out of range [0, {N_CARDS})")
        return c
    if isinstance(card, str):
        key = _norm(card)
        if key in _NAME_TO_ID:
            return _NAME_TO_ID[key]
    raise ValueError(f"unknown card {card!r}")


def _deck_arg(deck) -> Optional[list]:
    """A preset name, 'random' (None: a random deck per reset), or 8 cards."""
    if isinstance(deck, str):
        if deck == "random":
            return None
        if deck in DECKS:
            return list(DECKS[deck])
        raise ValueError(f"unknown deck {deck!r}; presets: {sorted(DECKS)} or 'random'")
    cards = [card_id(c) for c in deck]
    if len(cards) != 8 or len(set(cards)) != 8:
        raise ValueError("a deck is 8 distinct cards")
    return cards


def card_info(card: Union[int, str]) -> dict:
    """Level-11 stats of a card exactly as compiled into the engine.

    Troop/building cards carry their summoned unit's stats flattened in (keys follow the
    source file: hitpoints, damage, hit_speed_ms, load_time_ms, speed, range_milli,
    sight_range_milli, collision_radius_milli, ...) plus a nested 'projectile' dict;
    spells carry damage, radius, crown_tower_damage_percent, crown_tower_damage and
    their projectile / area_effect / spawn details.
    """
    return _b.card_info(card_id(card))


class Game:
    """One match, tick-level access through the compiled engine (SPEC §10)."""

    def __init__(self, deck0="hog26", deck1="giant", seed: int = 0, deploy_lockout_ticks: int = 90,
                 tiebreak: str = "absolute", alternate_first: bool = False, tower_troop0: str = "princess",
                 tower_troop1: str = "princess"):
        """alternate_first (not in the SPEC, default off): when both teams play on the same
        tick, alternate which team's plays are applied first instead of always team 0
        (SPEC v0.2 §13.13 fixes team 0 first; see docs/FIDELITY.md).
        tower_troop0 / tower_troop1 (SPEC §16.3): the tower troop replacing both Princess towers of
        that team: 'princess' (default), 'cannoneer', 'dagger_duchess' or 'royal_chef'."""
        if tiebreak not in TIEBREAKS:
            raise ValueError(f"tiebreak must be one of {sorted(TIEBREAKS)}")
        tt0, tt1 = tower_troop_index(tower_troop0), tower_troop_index(tower_troop1)
        self.seed = int(seed)
        self.deploy_lockout_ticks = int(deploy_lockout_ticks)
        self.tiebreak = tiebreak
        self.tower_troops = (TOWER_TROOPS[tt0], TOWER_TROOPS[tt1])
        self._g = _b.game_new(_deck_arg(deck0), _deck_arg(deck1), self.seed, self.deploy_lockout_ticks,
                              TIEBREAKS[tiebreak], bool(alternate_first), tt0, tt1)

    # ------------------------------------------------------------------ flow
    def reset(self, seed: Optional[int] = None) -> None:
        """Re-deal a new match with the CURRENT decks (the constructor's, or the ones set_hand
        installed; a 'random' deck is redrawn). Configuration (lockout, tiebreak, play order)
        is kept. With `seed` the game RNG is reseeded first, so reset(seed=s) equals a fresh
        Game(seed=s) with the same decks and configuration; without, the RNG stream continues."""
        _b.game_reset(self._g, None if seed is None else int(seed))

    def tick(self, n: int = 1) -> None:
        """Advance n engine ticks (stops early if the match ends)."""
        _b.game_tick(self._g, int(n))

    # ------------------------------------------------------------------ plays
    def play(self, team: int, slot: int, x: int, y: int) -> PlayError:
        """Queue a play of hand slot `slot` for the next tick at the engine-frame tap point
        (x, y). Legality is that of the tile containing the tap in the acting team's OWN frame
        (rotation, then floor). Troops (their formation) and spells are placed at the tap
        point itself; buildings at that tile's anchor (tile centre, or its own-frame top-left
        corner for even footprints), exactly as play_tile would. Returns the error code
        (0 = OK)."""
        return PlayError(_b.game_play_xy(self._g, int(team), int(slot), int(x), int(y)))

    def play_tile(self, team: int, slot: int, tx: int, ty: int) -> PlayError:
        """Queue a play at own-frame tile (tx, ty) of `team` for the next tick."""
        return PlayError(_b.game_play_tile(self._g, int(team), int(slot), int(tx), int(ty)))

    def spawn(self, team: int, card_id_or_name, x: int, y: int, deployed: bool = True) -> list:
        """TEST HOOK: put a card on the board now at (x, y), bypassing elixir, hand and
        legality. Troops use the card's formation around (x, y); spells are cast at
        (x, y). deployed=True skips the deploy time (as if it had just completed).
        Returns the new entity ids."""
        return _b.game_spawn(self._g, int(team), card_id(card_id_or_name), int(x), int(y), bool(deployed))

    # ------------------------------------------------------------------ views
    def entities(self) -> list:
        return _b.game_entities(self._g)

    def entity(self, eid: int) -> Optional[dict]:
        for e in _b.game_entities(self._g):
            if e["id"] == eid:
                return e
        return None

    def projectiles(self) -> list:
        return _b.game_projectiles(self._g)

    def effects(self) -> list:
        return _b.game_effects(self._g)

    def state(self) -> dict:
        return _b.game_state(self._g)

    @property
    def tick_count(self) -> int:
        return _b.game_state(self._g)["tick"]

    # ------------------------------------------------------------------ setters
    def set_elixir(self, team: int, units: int) -> None:
        _b.game_set_elixir(self._g, int(team), int(units))

    def set_tower_hp(self, team: int, idx: int, hp: int) -> None:
        """Set crown tower hp (idx 0 King, 1 Left, 2 Right in engine x). hp <= 0
        destroys it now, with crowns / King activation / pocket side effects."""
        _b.game_set_tower_hp(self._g, int(team), int(idx), int(hp))

    def set_hand(self, team: int, deck8_order: Sequence) -> None:
        """hand = order[0:4], queue = order[4:8]; the team's deck becomes these 8 cards, also
        for later deals (reset re-deals them; a 'random' deck stops being redrawn). Queued plays
        of the team are discarded."""
        _b.game_set_hand(self._g, int(team), [card_id(c) for c in deck8_order])

    # ------------------------------------------------------------------ legality
    def legal_mask(self, team: int) -> np.ndarray:
        """uint8[2305]: mask[0] = 1; mask[1 + slot*576 + ty*18 + tx] = 1 iff
        play_tile(team, slot, tx, ty) would be accepted right now."""
        return _b.game_legal_mask(self._g, int(team))

    # ------------------------------------------------------------------ determinism
    def hash(self) -> int:
        return _b.game_hash(self._g)

    def snapshot(self) -> bytes:
        return _b.game_snapshot(self._g)

    def restore(self, snap: bytes) -> None:
        """Restore a snapshot() of this engine version. The bytes are validated first (pool
        counts, id order, every table index, card bookkeeping, counter ranges; SPEC §14.5):
        invalid input raises ValueError and leaves the game unchanged."""
        _b.game_restore(self._g, snap)

    # ------------------------------------------------------------------ actions / views
    def play_action(self, team: int, action: int) -> PlayError:
        """Queue a SPEC §9 action index (0 = no-op, own frame) for `team`."""
        d = decode_action(int(action))
        if d is None:
            return PlayError.OK
        slot, tx, ty = d
        return self.play_tile(team, slot, tx, ty)

    def coarse_to_fine(self, team: int, action: int, grid: int) -> int:
        """SPEC §19.4: the fine action (1 + slot*576 + ty*18 + tx) that action `action` of placement
        grid `grid` (1, 2 or 4) plays for `team` right now -- the legal tile of its block nearest
        the block centre (ties: smaller ty, then smaller tx) -- through the env's C mapping; 0 when
        `action` is the no-op, out of range or has no legal tile. grid=1: the identity on legal
        actions."""
        return _b.game_coarse_to_fine(self._g, int(team), int(action), int(grid))

    def obs(self, team: int) -> np.ndarray:
        """The SPEC §9 observation of `team` for the current state (float32[OBS_SIZE])."""
        return _b.game_obs(self._g, int(team))

    def ansi(self) -> str:
        """Text rendering of the board (team 0 at the bottom)."""
        return _b.game_ansi(self._g)

    # ------------------------------------------------------------------ builder hooks
    def _set_first_team(self, team: int) -> None:
        """Builder test hook: enable alternate_first and choose whose plays are applied first
        on the next tick where both teams play (it alternates after each such tick)."""
        _b.game_set_first_team(self._g, int(team))

    def _set_entity(self, eid: int, field: str, value: int) -> None:
        """Builder test hook (not SPEC): set hp / x / y / deploy_ms / load_ms of an entity."""
        _b.game_debug_set_entity(self._g, int(eid), field, int(value))

    def _path_stats(self) -> dict:
        return _b.game_path_stats(self._g)


BOT_KINDS = {"noop": 0, "random": 1, "heuristic": 2}


class Bot:
    """A scripted opponent (SPEC §8) with its own PCG32 stream: `act(game, team)` returns a
    legal action index for `team` (0 = no-op). Deterministic given (kind, seed)."""

    def __init__(self, kind: str = "heuristic", seed: int = 0, play_prob: float = 0.2):
        if kind not in BOT_KINDS:
            raise ValueError(f"bot kind must be one of {sorted(BOT_KINDS)}")
        self.kind = kind
        self._b = _b.bot_new(BOT_KINDS[kind], int(seed) & (2 ** 64 - 1), int(round(float(play_prob) * 1e6)))

    def act(self, game: "Game", team: int) -> int:
        return _b.bot_act(self._b, game._g, int(team))

    def act_env(self, env, team: int, env_index: int = 0) -> int:
        """Decide for `team` of native match `env_index` of a pufferroyale.Royale env. The
        action is in that team's own frame, i.e. ready for its agent row."""
        return _b.bot_act_env(self._b, env._c_env_list[env_index], int(team))


def bot_action(game: "Game", team: int, kind: str = "heuristic", seed: int = None, play_prob: float = 0.2) -> int:
    """One decision of a scripted bot for `team` (a legal action index, 0 = no-op). The bot
    (and its RNG stream) persists on the Game per (team, kind), seeded from the game's seed
    and the team unless `seed` is given."""
    bots = game.__dict__.setdefault("_bots", {})
    key = (int(team), kind)
    if key not in bots:
        bots[key] = Bot(kind, (game.seed * 2 + int(team)) if seed is None else seed, play_prob)
    return bots[key].act(game, team)


def action_index(slot: int, tx: int, ty: int) -> int:
    """SPEC §9 action encoding: 1 + slot*576 + ty*18 + tx (own-frame tile)."""
    return 1 + slot * N_TILES + ty * 18 + tx


def decode_action(a: int):
    """Inverse of action_index: None for the no-op, else (slot, tx, ty)."""
    if a == 0:
        return None
    slot, cell = divmod(a - 1, N_TILES)
    return slot, cell % 18, cell // 18
