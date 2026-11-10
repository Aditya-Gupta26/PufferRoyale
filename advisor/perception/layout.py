"""Screen layout: where the arena and the HUD are in a frame, and the pixel <-> arena mapping.

The arena mapping is linear: own-frame tile coordinate (tx, ty) (continuous; tile corners at integers) sits at
pixel (x0 + tx * sx, y0 + ty * sy). Checked on two independent sources: Build-A-Bot's 720x1280 constants give
sx / sy = 34 / 27.6 = 1.232, and the princess-tower boxes of KataCR's labelled 568x896 arena crops give
31.07 / 25.12 = 1.237 (docs/ADVISOR.md).

HUD rectangles are (left, top, right, bottom) pixels. Every reader resamples its crop to Build-A-Bot's canonical
size first, so their thresholds stay valid at any resolution.

The built-in profile `bluestacks_720x1280` is Build-A-Bot's (MIT): its arena constants are in display pixels
(720x1280) and its HUD constants in 368x652 screenshot pixels, converted here to display pixels.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# Build-A-Bot screenshot -> display scale (its frames are the 720x1280 display resized to 368x652)
_SX, _SY = 720 / 368, 1280 / 652


def _scr(l, t, r, b):
    return (l * _SX, t * _SY, r * _SX, b * _SY)


@dataclass
class Layout:
    name: str
    width: int
    height: int
    x0: float
    sx: float
    y0: float
    sy: float
    # region fed to the unit detector, in arena tiles (l, t, r, b); Build-A-Bot crops the display to
    # y in [5%, 80%] x full width, which is this rectangle in tiles for its 720x1280 layout
    detector_crop_tiles: tuple = (-52 / 34, (64 - 100.8) / 27.6, (720 - 52) / 34, (1024 - 100.8) / 27.6)
    hand: list = field(default_factory=list)          # 4 rects, slot 0..3 left to right
    next_card: tuple = ()
    elixir_bar: tuple = ()
    princess_bars: dict = field(default_factory=dict)  # own_left, own_right, enemy_left, enemy_right -> rect
    in_battle_patch: tuple = ()                        # small rect compared with the in_game template

    # ---------------------------------------------------------------- mapping
    def tile_to_px(self, tx: float, ty: float) -> tuple:
        return (self.x0 + tx * self.sx, self.y0 + ty * self.sy)

    def px_to_milli(self, px: float, py: float) -> tuple:
        """Pixel -> own-frame millitiles (not clamped)."""
        return ((px - self.x0) / self.sx * 1000.0, (py - self.y0) / self.sy * 1000.0)

    def milli_to_px(self, x: float, y: float) -> tuple:
        return self.tile_to_px(x / 1000.0, y / 1000.0)

    def detector_crop_px(self) -> tuple:
        l, t, r, b = self.detector_crop_tiles
        (pl, pt), (pr, pb) = self.tile_to_px(l, t), self.tile_to_px(r, b)
        return (pl, pt, pr, pb)

    # ---------------------------------------------------------------- scaling / io
    def scaled(self, width: int, height: int) -> "Layout":
        """This layout for a frame of another size with the SAME aspect ratio (e.g. a scaled window)."""
        if abs(width / height - self.width / self.height) > 0.01:
            raise ValueError(f"frame {width}x{height} has another aspect ratio than layout {self.name} "
                             f"({self.width}x{self.height}); calibrate this device (python -m advisor.calibrate)")
        k, j = width / self.width, height / self.height

        def r(rect):
            return tuple(v * (k if i % 2 == 0 else j) for i, v in enumerate(rect)) if rect else rect

        return Layout(name=f"{self.name}@{width}x{height}", width=width, height=height, x0=self.x0 * k, sx=self.sx * k,
                      y0=self.y0 * j, sy=self.sy * j, detector_crop_tiles=self.detector_crop_tiles,
                      hand=[r(h) for h in self.hand], next_card=r(self.next_card), elixir_bar=r(self.elixir_bar),
                      princess_bars={n: r(v) for n, v in self.princess_bars.items()},
                      in_battle_patch=r(self.in_battle_patch))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @staticmethod
    def load(path: str) -> "Layout":
        with open(path) as f:
            d = json.load(f)
        d["detector_crop_tiles"] = tuple(d["detector_crop_tiles"])
        d["hand"] = [tuple(h) for h in d["hand"]]
        for k in ("next_card", "elixir_bar", "in_battle_patch"):
            d[k] = tuple(d[k])
        d["princess_bars"] = {k: tuple(v) for k, v in d["princess_bars"].items()}
        return Layout(**d)


def bluestacks_720x1280() -> Layout:
    """Build-A-Bot's layout (BlueStacks / any Android display at 720x1280, portrait)."""
    hp_w, hp_h = 40, 10
    bars = {"own_left": (74, 404), "own_right": (266, 404), "enemy_left": (74, 95), "enemy_right": (266, 95)}
    return Layout(
        name="bluestacks_720x1280", width=720, height=1280, x0=52.0, sx=34.0, y0=1280 - 296 - 32 * 27.6, sy=27.6,
        hand=[_scr(84 + 69 * i, 543, 84 + 61 + 69 * i, 543 + 73) for i in range(4)],
        next_card=_scr(21, 609, 47, 642),
        elixir_bar=_scr(100, 628, 350, 643),
        princess_bars={n: _scr(x, y, x + hp_w, y + hp_h) for n, (x, y) in bars.items()},
        in_battle_patch=(148, 1254, 163, 1274),
    )


PROFILES = {"bluestacks_720x1280": bluestacks_720x1280}


def fit_arena(points_px: list, points_tiles: list) -> tuple:
    """Least-squares (x0, sx, y0, sy) from pixel <-> own-frame tile correspondences (at least 2 points with
    different x and different y). Returns (x0, sx, y0, sy, rms_px)."""
    import numpy as np
    P, T = np.asarray(points_px, float), np.asarray(points_tiles, float)
    ax = np.vstack([np.ones(len(T)), T[:, 0]]).T
    ay = np.vstack([np.ones(len(T)), T[:, 1]]).T
    (x0, sx), *_ = np.linalg.lstsq(ax, P[:, 0], rcond=None)
    (y0, sy), *_ = np.linalg.lstsq(ay, P[:, 1], rcond=None)
    res = np.hstack([ax @ [x0, sx] - P[:, 0], ay @ [y0, sy] - P[:, 1]])
    return float(x0), float(sx), float(y0), float(sy), float(np.sqrt(np.mean(res ** 2)))
