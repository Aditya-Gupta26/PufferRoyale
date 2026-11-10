"""FrameReading: what the screen reader extracts from ONE frame, already in game terms (own-frame millitiles,
card ids, elixir), with no memory. The tracker (advisor.tracker) turns a sequence of readings into BoardStates.

Producers: advisor.perception (real pixels) and advisor.sim_perception (a simulated match, optionally with
the errors a real reader makes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    unit: Optional[str]        # engine unit name of the detected class; None = a unit outside our 64 cards
    label: str                 # the detector's class name (shown for unknown units)
    own: bool                  # True = the viewer's unit (bottom player)
    x: int                     # own-frame millitiles of the unit's ground point
    y: int
    conf: float = 1.0
    hp_frac: Optional[float] = None   # None = HP not read


@dataclass
class FrameReading:
    t: float                                   # game time in seconds since the battle started
    detections: list = field(default_factory=list)
    hand: list = field(default_factory=lambda: [-1, -1, -1, -1])   # card ids, -1 = not read
    next_card: int = -1
    elixir: Optional[float] = None             # own elixir 0..10 as the bar shows it; None = not read
    own_towers: list = field(default_factory=lambda: [None, None, None])     # HP fraction [King, left, right]
    enemy_towers: list = field(default_factory=lambda: [None, None, None])   # 0 = destroyed, None = not read
    in_battle: bool = True
