"""ScreenReader: one frame (PIL image of the game screen) -> FrameReading.

Reads: units (detector + side classifier), hand and next card, elixir (whole elixir), princess-tower HP
fractions, and whether the battle screen is showing. Does not read (docs/ADVISOR.md §5): spells, unit HP,
King-tower HP, the clock (the caller passes the time since the battle started).
"""
from __future__ import annotations

from PIL import Image

from ..reading import FrameReading
from .detector import UnitDetector, default_assets
from .hud import ALLY_HP, ENEMY_HP, BattleCheck, CardReader, read_elixir, read_tower_bar
from .layout import Layout


class ScreenReader:
    def __init__(self, layout: Layout, deck: list, assets: str = None, threads: int = 2):
        self.assets = assets or default_assets()
        self.base_layout = layout
        self.deck = list(deck)
        self.units = UnitDetector(self.assets, threads=threads)
        self.cards = CardReader(self.deck, self.assets)
        self.battle = BattleCheck(self.assets)
        self._layouts = {}

    def layout_for(self, frame: Image.Image) -> Layout:
        key = frame.size
        if key not in self._layouts:
            self._layouts[key] = self.base_layout if key == (self.base_layout.width, self.base_layout.height) \
                else self.base_layout.scaled(*key)
        return self._layouts[key]

    def read(self, frame: Image.Image, t: float, force: bool = False) -> FrameReading:
        """force=True reads everything even when the battle-screen check fails (testing on arena crops)."""
        frame = frame.convert("RGB")
        lay = self.layout_for(frame)
        in_battle = (not lay.in_battle_patch) or self.battle.in_battle(frame, lay)
        if not in_battle and not force:
            return FrameReading(t=t, in_battle=False)
        hand, nxt = [-1, -1, -1, -1], -1
        if lay.hand and lay.next_card:
            hand, nxt, _ready = self.cards.read(frame, lay)
        elixir = float(read_elixir(frame, lay)) if lay.elixir_bar else None
        bars = {}
        for n, r in lay.princess_bars.items():   # raw bar fraction (0.0 = no bar = tower down); the tracker scales it
            frac, standing = read_tower_bar(frame, r, ALLY_HP if n.startswith("own") else ENEMY_HP)
            bars[n] = frac if standing else 0.0
        return FrameReading(
            t=t, detections=self.units.detect(frame, lay, self.deck), hand=hand, next_card=nxt, elixir=elixir,
            own_towers=[None, bars.get("own_left"), bars.get("own_right")],
            enemy_towers=[None, bars.get("enemy_left"), bars.get("enemy_right")], in_battle=in_battle)
