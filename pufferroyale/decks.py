"""pufferroyale.decks -- deck sets for deck sampling (SPEC §19.5).

A *deck* is a tuple of 8 distinct card ids in ascending order (the deck as a set). A *deck set*
is a list of (deck, weight) pairs, written either as a string or as a Python list:

    string   items separated by ';' (surrounding whitespace ignored; empty items skipped):
               PRESET          a pufferroyale.DECKS name other than 'random', weight 1
               PRESET:WEIGHT   the same with a weight > 0
               random:N:SEED   the N decks of random_decks(N, SEED), weight 1 each
               file:PATH       a JSON list of elements (below)
    list     elements: a preset name or any string item above, a list of 8 card names / ids,
             {"deck": <preset | list of 8>, "weight": w}, or a (deck, weight) pair (so the
             output of parse_deck_set parses to itself)

Two equal decks (as sets) in one deck set raise ValueError, as does anything malformed.

    parse_deck_set(spec) -> [(deck, weight), ...]
    deck_entries(spec) -> [(deck, weight, order), ...]   `order` = the card order the env installs
                                           (SPEC §19.5 ruling v0.5-G.1): a deck named by a preset
                                           keeps the preset's order, every other deck is ascending
    random_decks(n, seed) -> [deck, ...]   n distinct uniformly random decks of all N_CARDS cards
    deck_key(deck) -> str                  canonical set key ("2-4-9-10-11-14-16-19")
    expanded_deck_sets(env_kwargs) -> dict deck_pool_decks / heldout_decks_decks for config.json

A weight is a finite number in (0, 1e9] (MAX_WEIGHT, SPEC §19.10.5).
"""
from __future__ import annotations

import json
import math
import numbers
from typing import List, Tuple

import numpy as np

from .game import DECKS, N_CARDS, card_id

Deck = Tuple[int, ...]
Entry = Tuple[Deck, float, Deck]   # (ascending deck, weight, installed card order)


def _deck(cards) -> Deck:
    """8 distinct cards (names or ids) -> the ascending id tuple."""
    if isinstance(cards, (str, bytes)) or not hasattr(cards, "__iter__"):
        raise ValueError(f"a deck is 8 distinct cards, got {cards!r}")
    ids = [card_id(c) for c in cards]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError(f"a deck is 8 distinct cards, got {list(cards)!r}")
    return tuple(sorted(ids))


def _preset(name: str) -> Deck:
    if name == "random" or name not in DECKS:
        raise ValueError(f"unknown deck preset {name!r}; presets: {sorted(DECKS)} ('random' is not one; "
                         f"use random:N:SEED)")
    return tuple(sorted(DECKS[name]))


def _preset_entry(name: str, w: float) -> Entry:
    """A deck named by a preset: installed in the preset's own card order (so a pool of 'hog26'
    deals exactly like deck0='hog26')."""
    return _preset(name), w, tuple(DECKS[name])


def _cards_entry(cards, w: float) -> Entry:
    d = _deck(cards)
    return d, w, d


#: largest deck weight (SPEC §19.10.5): keeps the env's weight total finite and exact enough
MAX_WEIGHT = 1e9


def _weight(w) -> float:
    try:
        w = float(w)
    except (TypeError, ValueError):
        raise ValueError(f"a deck weight is a number > 0, got {w!r}") from None
    if not (w > 0.0 and math.isfinite(w) and w <= MAX_WEIGHT):
        raise ValueError(f"a deck weight is a finite number in (0, {MAX_WEIGHT:g}], got {w!r}")
    return w


def _int(text: str, what: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        raise ValueError(f"{what} must be an integer, got {text!r}") from None


def deck_key(deck) -> str:
    """Canonical key of a deck as a set: its ascending card ids joined by '-'. Accepts a preset
    name or 8 card names / ids in any order."""
    return "-".join(str(c) for c in (_preset(deck) if isinstance(deck, str) else _deck(deck)))


def random_decks(n: int, seed: int) -> List[Deck]:
    """`n` distinct decks, each a uniformly random set of 8 of the N_CARDS cards, as a pure
    function of (n, seed): numpy PCG64 seeded from `seed`, decks drawn in order and repeats
    skipped, so random_decks(k, seed) is a prefix of random_decks(n, seed) for k <= n."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0:
        raise ValueError(f"random_decks: n must be an integer >= 0, got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f"random_decks: seed must be an integer, got {seed!r}")
    seed = int(seed)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1 if seed < 0 else 0, abs(seed)])))
    out, seen = [], set()
    while len(out) < n:
        d = tuple(sorted(int(c) for c in rng.choice(N_CARDS, size=8, replace=False)))
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _parse_item(item: str) -> List[Entry]:
    """One string item: PRESET, PRESET:WEIGHT, random:N:SEED or file:PATH."""
    head, sep, rest = item.partition(":")
    head = head.strip()
    if head == "file" and sep:
        path = rest.strip()
        if not path:
            raise ValueError("file: needs a path (file:PATH)")
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise ValueError(f"deck set file {path!r}: {e}") from None
        if not isinstance(data, list):
            raise ValueError(f"deck set file {path!r}: expected a JSON list")
        return [e for x in data for e in _parse_element(x, allow_strings=False)]
    if head == "random" and sep:
        parts = rest.split(":")
        if len(parts) != 2:
            raise ValueError(f"bad deck set item {item!r}: use random:N:SEED")
        n, seed = _int(parts[0], "random:N"), _int(parts[1], "random:SEED")
        if n < 1:
            raise ValueError(f"bad deck set item {item!r}: N must be >= 1")
        return [(d, 1.0, d) for d in random_decks(n, seed)]
    if sep:
        if ":" in rest:
            raise ValueError(f"bad deck set item {item!r}: use PRESET or PRESET:WEIGHT")
        return [_preset_entry(head, _weight(rest.strip()))]
    return [_preset_entry(head, 1.0)]


def _parse_element(x, allow_strings: bool = True) -> List[Entry]:
    """One list / JSON element -> its entries. In a JSON file a string is a preset name; in a
    Python list it is any string item (so it may expand to several decks)."""
    if isinstance(x, str):
        return _parse_item(x) if allow_strings else [_preset_entry(x.strip(), 1.0)]
    if isinstance(x, dict):
        if set(x) - {"deck", "weight"} or "deck" not in x:
            raise ValueError(f"a deck set element dict has keys 'deck' and optional 'weight', got {sorted(x)}")
        d, w = x["deck"], _weight(x.get("weight", 1.0))
        return [_preset_entry(d.strip(), w) if isinstance(d, str) else _cards_entry(d, w)]
    if isinstance(x, (list, tuple)) and len(x) == 2 and not isinstance(x[1], str):  # a (deck, weight) pair
        d, w = x[0], _weight(x[1])
        return [_preset_entry(d.strip(), w) if isinstance(d, str) else _cards_entry(d, w)]
    return [_cards_entry(x, 1.0)]


def deck_entries(spec) -> List[Entry]:
    """parse_deck_set with each deck's installed card order: [(deck, weight, order), ...]."""
    if spec is None:
        return []
    out: List[Entry] = []
    if isinstance(spec, str):
        for item in spec.split(";"):
            if item.strip():
                out.extend(_parse_item(item.strip()))
    elif isinstance(spec, (list, tuple)):
        for x in spec:
            out.extend(_parse_element(x))
    else:
        raise ValueError(f"a deck set is a string or a list, got {type(spec).__name__}")
    seen = {}
    for i, (d, _, _) in enumerate(out):
        if d in seen:
            raise ValueError(f"deck set lists the same deck twice (entries {seen[d]} and {i}: {list(d)})")
        seen[d] = i
    return out


def parse_deck_set(spec) -> List[Tuple[Deck, float]]:
    """A deck set (string, list, or None / '' = empty) -> [(deck, weight), ...] in the given order;
    ValueError on bad syntax, an unknown preset or card, a bad weight, or two equal decks."""
    return [(d, w) for d, w, _ in deck_entries(spec)]


def expanded_deck_sets(env) -> dict:
    """SPEC §19.10.5: the decks of an env-kwargs dict's deck sets, for config.json --
    {"deck_pool_decks": [[cards, weight], ...], "heldout_decks_decks": [[cards, weight], ...]} with
    the cards in the order the env installs them (deck_pool: deck_entries' order, so a preset keeps its
    own order; heldout_decks are installed ascending)."""
    env = env or {}
    return {"deck_pool_decks": [[list(order), w] for _, w, order in deck_entries(env.get("deck_pool") or "")],
            "heldout_decks_decks": [[list(d), w] for d, w, _ in deck_entries(env.get("heldout_decks") or "")]}


__all__ = ["parse_deck_set", "deck_entries", "random_decks", "deck_key", "expanded_deck_sets", "MAX_WEIGHT", "Deck"]
