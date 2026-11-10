"""A simulated screen reader: FrameReadings from a simulated match, with what a real reader can and cannot see.

What it reports (like the real reader, advisor.perception): troops and buildings with their unit type, side and
ground position; the viewer's hand, next card and elixir as the bar shows it (whole elixir); crown-tower HP
fractions. What it never reports: spells, unit HP (unless read_hp), stun / slow, the opponent's hand or
elixir. Hidden Teslas and burrowing Miners are underground, so they are not visible (hide_underground).

Errors a real reader makes can be switched on (p_miss, pos_sigma, p_hand_unknown) to measure their cost.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .board import own_xy
from .reading import Detection, FrameReading
from . import engine_db as D


@dataclass
class SimPerceptionConfig:
    p_miss: float = 0.0           # each unit independently not detected in a frame
    pos_sigma: float = 0.0        # position noise (millitiles, Gaussian per axis)
    read_hp: bool = False         # report unit HP fractions
    p_hand_unknown: float = 0.0   # each hand slot independently not read in a frame
    hide_underground: bool = True
    read_king: bool = True        # False = King-tower HP not read (reported only as destroyed / not), as the
                                  # real reader (Build-A-Bot reads princess bars only)
    p_side_err: float = 0.0       # a detection's side flipped
    p_type_err: float = 0.0       # a detection reported as another unit of our 64 (uniformly drawn)
    fp_per_frame: float = 0.0     # mean number of phantom detections per frame (Poisson), random unit / side,
                                  # placed uniformly on the arena
    tower_hp_standing_only: bool = False   # crown-tower HP reported only as standing (1.0) / destroyed (0.0)
    tower_phantoms: bool = False  # each living crown tower shows a phantom in every frame (the real detector
                                  # reads the princess-tower archer / the King as a unit; docs/ADVISOR.md 6a)


# phantoms the real detector reports on crown towers (6a): princess-tower archer -> Princess / Bandit,
# King tower -> Baby Dragon (and Spear Goblin); own-frame tower centres
_TOWER_PHANTOMS = [("Princess", (3500, 6200)), ("Princess", (14500, 6200)), ("BabyDragon", (9000, 2800)),
                   ("Princess", (3500, 25200)), ("Princess", (14500, 25200)), ("BabyDragon", (9000, 28800))]
_PHANTOM_UNITS = None


def _phantom_units():
    global _PHANTOM_UNITS
    if _PHANTOM_UNITS is None:
        _PHANTOM_UNITS = sorted({u for comp in D.play_compositions().values() for u in comp})
    return _PHANTOM_UNITS


def sim_reading(g, team: int, cfg: SimPerceptionConfig = SimPerceptionConfig(), rng: random.Random = None) -> FrameReading:
    rng = rng or random.Random(0)
    st = g.state()
    dets = []
    for e in g.entities():
        if e["kind"] == "tower":
            continue
        if cfg.hide_underground and (e["hidden"] or e["burrowing"]):
            continue
        if cfg.p_miss and rng.random() < cfg.p_miss:
            continue
        x, y = own_xy(team, e["x"], e["y"])
        if cfg.pos_sigma:
            x = int(min(max(x + rng.gauss(0, cfg.pos_sigma), 0), D.ARENA_W - 1))
            y = int(min(max(y + rng.gauss(0, cfg.pos_sigma), 0), D.ARENA_H - 1))
        unit, own = e["unit"], e["team"] == team
        if cfg.p_type_err and rng.random() < cfg.p_type_err:
            unit = rng.choice(_phantom_units())
        if cfg.p_side_err and rng.random() < cfg.p_side_err:
            own = not own
        dets.append(Detection(unit=unit, label=unit, own=own, x=x, y=y,
                              hp_frac=(e["hp"] / e["max_hp"]) if cfg.read_hp else None))
    il, ir = (1, 2) if team == 0 else (2, 1)
    if cfg.fp_per_frame:
        k, p, L = 0, 1.0, pow(2.718281828459045, -cfg.fp_per_frame)   # Poisson draw (Knuth)
        while True:
            p *= rng.random()
            if p <= L:
                break
            k += 1
        for _ in range(k):
            u = rng.choice(_phantom_units())
            dets.append(Detection(unit=u, label=u, own=rng.random() < 0.5, x=rng.randrange(D.ARENA_W),
                                  y=rng.randrange(D.ARENA_H)))
    if cfg.tower_phantoms:
        alive = [st["towers"][1 - team][i]["alive"] for i in (il, ir, 0)] + [st["towers"][team][i]["alive"] for i in (il, ir, 0)]
        for (u, (x, y)), a in zip(_TOWER_PHANTOMS, alive):
            if a:
                dets.append(Detection(unit=u, label=u, own=False, x=x, y=y, conf=0.5))
    hand = [c if not (cfg.p_hand_unknown and rng.random() < cfg.p_hand_unknown) else -1 for c in st["hand"][team]]

    def fracs(t):
        tw = st["towers"][t]
        f = [(tw[i]["hp"] / tw[i]["max_hp"]) if tw[i]["alive"] else 0.0 for i in (0, il, ir)]
        if cfg.tower_hp_standing_only:
            f = [1.0 if v > 0 else 0.0 for v in f]
        if not cfg.read_king and tw[0]["alive"]:
            f[0] = None
        return f

    return FrameReading(t=st["tick"] * D.TICK_MS / 1000.0, detections=dets, hand=hand,
                        next_card=st["queue"][team][0], elixir=float(st["elixir"][team] // D.ELIXIR_UNIT),
                        own_towers=fracs(team), enemy_towers=fracs(1 - team))
