"""HUD readers: hand cards, next card, elixir, princess-tower HP bars, in-battle check.

Algorithms are Build-A-Bot's (MIT; detectors/card_detector.py, number_detector.py, screen_detector.py). Each crop
is first resampled to the size it has in Build-A-Bot's 368x652 frames, so their pixel thresholds hold at any
resolution.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFilter
from scipy.optimize import linear_sum_assignment

from .. import engine_db as D
from .layout import Layout

# Build-A-Bot canonical crop sizes (368x652 frame pixels)
CARD_SIZE = (61, 73)
NEXT_SIZE = (26, 33)
ELIXIR_SIZE = (250, 15)
HP_SIZE = (40, 10)
ALLY_HP = ((111, 208, 252), (63, 79, 112))      # (left-of-change colour, right-of-change colour)
ENEMY_HP = ((224, 35, 93), (90, 49, 68))

# our card names -> Build-A-Bot card image file stems where the generic rule (lower case, '.' removed,
# ' ' and '-' -> '_') does not give the file name
CARD_IMAGE_OVERRIDES = {"Mini P.E.K.K.A": "minipekka"}


def card_image_stem(card: int) -> str:
    name = D.CARD_NAMES[card]
    if name in CARD_IMAGE_OVERRIDES:
        return CARD_IMAGE_OVERRIDES[name]
    return name.lower().replace(".", "").replace(" ", "_").replace("-", "_")


def _resize(img: Image.Image, rect, size) -> Image.Image:
    return img.crop(tuple(round(v) for v in rect)).resize(size, Image.Resampling.BILINEAR)


class CardReader:
    """Build-A-Bot CardDetector: 8x8 grey hashes (plus lighter / darker variants) of the next-card crop and the
    4 hand crops against the deck's card images and 'blank', matched with one Hungarian assignment."""
    MULTI_HASH_SCALE = 0.355
    MULTI_HASH_INTERCEPT = 163

    def __init__(self, deck: list, assets: str, hash_size: int = 8, grey_std_threshold: int = 5):
        self.deck = list(deck)
        self.hash_size, self.grey_std_threshold = hash_size, grey_std_threshold
        self.labels = self.deck + [-1] * 5      # -1 = blank
        hashes = []
        for c in self.labels:
            path = os.path.join(assets, "images", "cards", ("blank" if c < 0 else card_image_stem(c)) + ".jpg")
            hashes.append(self._multi_hash(Image.open(path)))
        self.card_hashes = np.stack(hashes)     # (n_labels, 3, hash*hash)

    def _hash(self, img: Image.Image) -> np.ndarray:
        return np.array(img.resize((self.hash_size, self.hash_size), Image.Resampling.BILINEAR).convert("L"),
                        dtype=np.float32).ravel()

    def _multi_hash(self, img: Image.Image) -> np.ndarray:
        g = self._hash(img)
        return np.vstack([g, self.MULTI_HASH_SCALE * g + self.MULTI_HASH_INTERCEPT,
                          (g - self.MULTI_HASH_INTERCEPT) / self.MULTI_HASH_SCALE]).astype(np.float32)

    def read(self, frame: Image.Image, layout: Layout):
        """(hand: 4 card ids (-1 = blank), next card id, ready slots) as Build-A-Bot defines them."""
        crops = [_resize(frame, layout.next_card, NEXT_SIZE)] + [_resize(frame, r, CARD_SIZE) for r in layout.hand]
        h = np.stack([self._hash(c) for c in crops])                     # (5, 64)
        # diff[label, crop] = mean over pixels of min over the 3 variants of |crop - card|
        diff = np.abs(h[None, None, :, :] - self.card_hashes[:, :, None, :]).min(axis=1).mean(axis=-1)
        rows, cols = linear_sum_assignment(diff.T)                       # crop -> label
        got = [-1] * 5
        for r_, c_ in zip(rows, cols):
            got[r_] = self.labels[c_]
        ready = [i for i, c in enumerate(crops[1:]) if np.mean(np.std(np.array(c), axis=2)) > self.grey_std_threshold]
        return got[1:], got[0], ready


def read_elixir(frame: Image.Image, layout: Layout, window_size: int = 10, threshold: float = 50) -> int:
    """Build-A-Bot NumberDetector._calculate_elixir: the filled part of the bar is colourful (high std across
    channels); the first low-std run marks the end of the fill."""
    crop = np.array(_resize(frame, layout.elixir_bar, ELIXIR_SIZE))
    std = crop.std(axis=(0, 2))
    rolling = np.convolve(std, np.ones(window_size) / window_size, mode="valid")
    change = np.nonzero(rolling < threshold)[0]
    if len(change) == 0:
        return 10
    return int((change[0] + window_size) * 10 // ELIXIR_SIZE[0])


def read_hp_bar(frame: Image.Image, rect, colours, threshold: float = 30) -> float:
    """Build-A-Bot NumberDetector._calculate_hp: HP fraction from where the bar changes from its 'full' colour to
    its 'empty' colour; 0.0 when the bar is absent (tower destroyed, or the bar not shown)."""
    crop = np.array(_resize(frame, rect, HP_SIZE).filter(ImageFilter.SMOOTH_MORE), dtype=np.float32)
    means = np.array([np.mean(np.abs(crop - np.array(c, np.float32)), axis=2) for c in colours])
    best_row = np.argmin(np.sum(np.min(means, axis=0), axis=1))
    means = means[:, best_row, :]
    sides = np.argmin(means, axis=0)
    if np.mean(np.where(sides, means[1], means[0])) > threshold:
        return 0.0
    return float(np.argmin(np.cumsum(2 * sides - 1)) / (HP_SIZE[0] - 1))


def read_tower_bar(frame: Image.Image, rect, colours, dist: float = 45.0, min_band: float = 0.3):
    """Princess-tower HP fraction for the current interface, where the HP number is drawn ON the bar (seen on
    KataCR's 2024 frames; Build-A-Bot's read_hp_bar assumes a clean bar and reads 0.0 there).

    Every pixel of the bar box is 'filled' (near the bar's full colour), 'empty' (near its empty colour) or
    neither (digits, outline, background). Rows where at least `min_band` of the pixels are bar-coloured form
    the bar band; each column is then filled / empty / unknown by majority over the band, and the fill end is
    the split that best separates filled columns (left) from empty ones (right). Returns (fraction, standing):
    standing = a bar band exists (a destroyed tower has no bar)."""
    a = np.array(frame.crop(tuple(round(v) for v in rect)), dtype=np.float32)
    if a.size == 0:
        return 0.0, False
    d_full = np.mean(np.abs(a - np.array(colours[0], np.float32)), axis=2)
    d_empty = np.mean(np.abs(a - np.array(colours[1], np.float32)), axis=2)
    full = (d_full < dist) & (d_full <= d_empty)
    empty = (d_empty < dist) & (d_empty < d_full)
    if not ((full | empty).mean(axis=1) >= min_band).any():
        return 0.0, False                 # no bar: the tower is down
    rowf = full.mean(axis=1)              # bar-body rows: the rows richest in the filled colour (the dark frame
    body = (rowf >= 0.2) & (rowf >= 0.75 * rowf.max())   # around the bar looks like the empty colour, and the
    if not body.any():                    # light digits of the own bar look like its filled colour)
        return 0.02, True                 # a bar with (almost) no fill
    nf, ne = full[body].mean(axis=0), empty[body].mean(axis=0)
    filled = np.flatnonzero(nf >= 0.5)
    bar = np.flatnonzero((nf >= 0.5) | (ne >= 0.5))
    if not len(filled):
        return 0.02, True
    return float((filled[-1] + 1) / (bar[-1] + 1)), True    # fill end / bar end (digits only hide the left part)


class BattleCheck:
    """Build-A-Bot ScreenDetector for the in-game screen: mean |8x8 hash difference| of a small patch below
    the threshold."""

    def __init__(self, assets: str, threshold: float = 30):
        self.template = self._hash(Image.open(os.path.join(assets, "images", "screen", "in_game.jpg")))
        self.threshold = threshold

    @staticmethod
    def _hash(img: Image.Image) -> np.ndarray:
        return np.array(img.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32).ravel()

    def in_battle(self, frame: Image.Image, layout: Layout) -> bool:
        patch = frame.crop(tuple(round(v) for v in layout.in_battle_patch))
        return float(np.mean(np.abs(self._hash(patch) - self.template))) < self.threshold
