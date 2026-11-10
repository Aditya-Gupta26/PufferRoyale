"""BoardState: everything the policy's observation is built from, in the VIEWER'S OWN FRAME (the viewer is
always the bottom player, exactly as the real game shows it on the viewer's screen).

Two producers fill it:
  * from_game(game, team): perfect perception from a simulated match (the encoder's exactness test);
  * the screen reader + tracker (advisor.perception / advisor.tracker) from a live or recorded game.

Coordinates are own-frame millitiles (1 tile = 1000; x in [0, 18000), y in [0, 32000), y = 0 is the
enemy's back edge). Elixir is in engine units (2800 per elixir).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import engine_db as D


@dataclass
class Unit:
    """One non-tower entity (troop or building) on the board."""
    unit: str                 # engine unit name, e.g. "HogRider", "Skeleton", "Cannon"
    card: int                 # the card that created it (Witch's Skeletons -> Witch); -1 = unknown
    own: bool                 # True = the viewer's unit
    x: int                    # own-frame millitiles
    y: int
    hp: int
    max_hp: int
    inactive: bool = False    # deploying, or a burrowing Miner (the observation's "deploying" flag)
    stunned: bool = False     # stunned or frozen
    slowed: bool = False


@dataclass
class Tower:
    hp: int
    max_hp: int

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class SpellArea:
    """An active spell area as the observation draws it (pr_obs.h 'active spell areas')."""
    own: bool
    shape: str                # 'disc' (cx, cy, r) or 'rect' (x0, y0, x1, y1), own frame
    a: int
    b: int
    c: int
    d: int = 0


@dataclass
class BoardState:
    tick: int
    elixir: int                               # viewer's elixir, engine units
    hand: list                                # 4 card ids, -1 = unknown / empty
    next_card: int                            # -1 = unknown
    deck: list                                # 8 card ids in the order the deck was installed (see encoder)
    own_towers: list                          # [King, left, right] (own frame) Tower
    enemy_towers: list                        # [King, left, right] (own frame) Tower
    king_active: tuple = (False, False)       # (own, enemy)
    crowns: tuple = (0, 0)                    # (own, enemy)
    tower_troops: tuple = (0, 0)              # (own, enemy) index into TOWER_TROOPS
    units: list = field(default_factory=list)        # Unit, in entity-list order (oldest first)
    spells: list = field(default_factory=list)       # SpellArea
    opp_seen: set = field(default_factory=set)       # card ids the opponent has played
    opp_spent: int = 0                               # elixir units the opponent has spent
    opp_last4: list = field(default_factory=lambda: [-1, -1, -1, -1])  # most recent first
    lockout_ticks: int = 90
    over: bool = False


def own_xy(team: int, x: int, y: int) -> tuple:
    """Engine point -> `team`'s own frame (SPEC §1: team 1 sees the 180-degree rotation)."""
    return (x, y) if team == 0 else (D.ARENA_W - x, D.ARENA_H - y)


def from_game(g, team: int) -> BoardState:
    """Perfect perception: the BoardState of `team` in simulated match `g` (a pufferroyale.Game)."""
    st = g.state()
    opp = 1 - team
    # engine tower idx 1 / 2 are engine-left / engine-right; team 1's own-frame left is engine-right
    il, ir = (1, 2) if team == 0 else (2, 1)

    def towers(t):
        tw = st["towers"][t]
        return [Tower(max(0, tw[i]["hp"]) if tw[i]["alive"] else 0, tw[i]["max_hp"]) for i in (0, il, ir)]

    units = []
    for e in g.entities():  # engine array order == the observation's entity order; hp > 0 only
        if e["kind"] == "tower":
            continue
        x, y = own_xy(team, e["x"], e["y"])
        units.append(Unit(unit=e["unit"], card=e["card_id"], own=e["team"] == team, x=x, y=y, hp=e["hp"],
                          max_hp=e["max_hp"], inactive=e["deploying"] or e["burrowing"],
                          stunned=e["stunned"], slowed=e["slowed"]))

    spells = []
    for p in g.projectiles():  # spells in flight (Fireball, Goblin Barrel, ...): their landing area
        if p["target_id"] != -1 or p["card_id"] < 0 or D.card_kind(p["card_id"]) != "spell":
            continue
        cx, cy = own_xy(team, p["target_x"], p["target_y"])
        spells.append(SpellArea(own=p["team"] == team, shape="disc", a=cx, b=cy, c=max(p["radius"], 1000)))
    for f in g.effects():
        cx, cy = own_xy(team, f["x"], f["y"])
        own = f["team"] == team
        if f["kind"] == "rolling":  # the rest of the roll: current hitbox to the end of the roll
            y_end = cy + (-1 if own else 1) * f["remaining"]
            y0, y1 = min(cy, y_end) - f["radius_y"], max(cy, y_end) + f["radius_y"]
            spells.append(SpellArea(own=own, shape="rect", a=cx - f["radius"], b=y0, c=cx + f["radius"], d=y1))
        else:
            spells.append(SpellArea(own=own, shape="disc", a=cx, b=cy, c=f["radius"]))

    seen = {c for c in range(D.N_CARDS) if (st["seen_mask"][opp] >> c) & 1}
    return BoardState(
        tick=st["tick"], elixir=st["elixir"][team], hand=list(st["hand"][team]), next_card=st["queue"][team][0],
        deck=list(st["deck"][team]), own_towers=towers(team), enemy_towers=towers(opp),
        king_active=(bool(st["towers"][team][0]["active"]), bool(st["towers"][opp][0]["active"])),
        crowns=(st["crowns"][team], st["crowns"][opp]),
        tower_troops=(D.TOWER_TROOPS.index(st["tower_troops"][team]), D.TOWER_TROOPS.index(st["tower_troops"][opp])),
        units=units, spells=spells, opp_seen=seen, opp_spent=st["spent"][opp],
        opp_last4=list(st["last_played"][opp]), lockout_ticks=st["lockout_ticks"], over=bool(st["over"]))
