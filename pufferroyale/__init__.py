"""PufferRoyale: a deterministic, integer, Clash Royale 1v1 battle simulator.

Public API (SPEC §10): Game, card_info, CARD_NAMES, DECKS, PlayError.
The PufferLib environment (pufferroyale.Royale) is imported lazily so that
`import pufferroyale` does not pull in pufferlib / torch.

Not affiliated with, endorsed, sponsored, or specifically approved by Supercell.

PUFFERROYALE_BINDING_DIR: when set, the compiled engine (binding*.so) is loaded from that
directory instead of the in-place build -- used for the sanitizer build (`make debug` writes
build/debug/pufferroyale/, `make asan-py` runs the Python suites against it), so a debug build
never replaces the release extension.
"""
import os as _os

if _os.environ.get("PUFFERROYALE_BINDING_DIR"):
    import glob as _glob
    import importlib.util as _ilu
    import sys as _sys

    _dir = _os.environ["PUFFERROYALE_BINDING_DIR"]
    _found = sorted(_glob.glob(_os.path.join(_dir, "binding*.so")) + _glob.glob(_os.path.join(_dir, "binding*.pyd")))
    if not _found:
        raise ImportError(f"PUFFERROYALE_BINDING_DIR={_dir!r} holds no binding*.so")
    _spec = _ilu.spec_from_file_location(__name__ + ".binding", _found[0])
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules[__name__ + ".binding"] = _mod
    _spec.loader.exec_module(_mod)

from .game import (  # noqa: E402,F401
    CARD_COSTS,
    CARD_KEYS,
    CARD_NAMES,
    CARD_SLOTS,
    BOT_KINDS,
    DECKS,
    N_ACTIONS,
    N_CARDS,
    N_TILES,
    TOWER_TROOPS,
    Bot,
    Game,
    PlayError,
    bot_action,
    action_index,
    card_id,
    card_info,
    decode_action,
    tower_troop_index,
)

__version__ = "0.3.0"

__all__ = ["Game", "card_info", "CARD_NAMES", "DECKS", "PlayError", "card_id", "action_index",
           "decode_action", "N_CARDS", "N_ACTIONS", "CARD_SLOTS", "TOWER_TROOPS", "Bot", "bot_action", "BOT_KINDS",
           "Royale"]


def __getattr__(name):  # lazy: the env needs pufferlib
    if name == "Royale":
        from .royale import Royale

        return Royale
    raise AttributeError(f"module 'pufferroyale' has no attribute {name!r}")
