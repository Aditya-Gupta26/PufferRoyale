#!/usr/bin/env python3
"""Generate pufferroyale/csrc/pr_card_db.h from data/source/cards-15.535.json.

SPEC §2: every scaling stat at level 11 = floor(base * 256 / 100) (the Common ladder
index 10, RoyaleSim's measured unified level-11 multiplier); crown towers (and the SPEC §16.3
tower troops) use the compounding tower ladder (ledger combat.TOWER_HITPOINT_LADDER):
pct(1) = 100, pct(L) = floor(pct(L-1) * (100 + rate) / 100) with rate r for L <= 9 and 10
above, stat = floor(base * pct / 100).

The output is deterministic (fixed ordering, no timestamps): `make gen` twice gives
byte-identical files. Card ids follow SPEC §0 (0-20) and §16.1 (21-63) and are part of the
public API. Unit / projectile / area / buff indices of the v0.1 cards keep their v0.2 values
(the v0.1 cards and the crown towers are registered first; everything §16 adds comes after).

Every card's level-11 numbers are asserted at generation time against hand-checked literals
(`self_check`): the SPEC §2 examples, all 43 §16 cards (their units, spawned / death-spawned
units, death bombs, projectiles, spell areas and buffs) and the three tower troops.

Usage: python tools/gen_card_db.py [--check]
  --check  exit 1 if the committed header differs from what would be generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "source" / "cards-15.535.json"
OUT = ROOT / "pufferroyale" / "csrc" / "pr_card_db.h"

LEVEL = 11
LEVEL_PCT = 256  # Common ladder multiplier_percent_by_level[10]

# SPEC §0 (ids 0-20) and §16.1 (ids 21-63): stable card ids -> (display name, source key)
CARDS = [
    ("Knight", "Knight"),
    ("Archers", "Archer"),
    ("Musketeer", "Musketeer"),
    ("Giant", "Giant"),
    ("Hog Rider", "HogRider"),
    ("Minions", "Minions"),
    ("Baby Dragon", "BabyDragon"),
    ("Valkyrie", "Valkyrie"),
    ("Skeleton Army", "SkeletonArmy"),
    ("Skeletons", "Skeletons"),
    ("Ice Golem", "IceGolemite"),
    ("Ice Spirit", "IceSpirits"),
    ("Prince", "Prince"),
    ("Wizard", "Wizard"),
    ("Cannon", "Cannon"),
    ("Tesla", "Tesla"),
    ("Fireball", "Fireball"),
    ("Arrows", "Arrows"),
    ("Zap", "Zap"),
    ("The Log", "Log"),
    ("Goblin Barrel", "GoblinBarrel"),
    # ---- SPEC §16.1 (v0.3, Phase E) ----
    ("Barbarians", "Barbarians"),
    ("Mini P.E.K.K.A", "MiniPekka"),
    ("P.E.K.K.A", "Pekka"),
    ("Mega Minion", "MegaMinion"),
    ("Bats", "Bats"),
    ("Spear Goblins", "SpearGoblins"),
    ("Goblins", "Goblins"),
    ("Goblin Gang", "GoblinGang"),
    ("Royal Giant", "RoyalGiant"),
    ("Bomber", "Bomber"),
    ("Princess", "Princess"),
    ("Dart Goblin", "BlowdartGoblin"),
    ("Minion Horde", "MinionHorde"),
    ("Fire Spirit", "FireSpirits"),
    ("Ice Wizard", "IceWizard"),
    ("Golem", "Golem"),
    ("Lava Hound", "LavaHound"),
    ("Balloon", "Balloon"),
    ("Giant Skeleton", "GiantSkeleton"),
    ("Witch", "Witch"),
    ("Tombstone", "Tombstone"),
    ("Inferno Tower", "InfernoTower"),
    ("Inferno Dragon", "InfernoDragon"),
    ("Bandit", "Assassin"),
    ("Battle Ram", "BattleRam"),
    ("Royal Hogs", "RoyalHogs"),
    ("Miner", "Miner"),
    ("Mortar", "Mortar"),
    ("X-Bow", "Xbow"),
    ("Elixir Collector", "Elixir Collector"),
    ("Bomb Tower", "BombTower"),
    ("Rocket", "Rocket"),
    ("Poison", "Poison"),
    ("Freeze", "Freeze"),
    ("Earthquake", "Earthquake"),
    ("Lightning", "Lightning"),
    ("Giant Snowball", "Snowball"),
    ("Barbarian Barrel", "BarbLog"),
    ("Rascals", "Rascals"),
    ("Elite Barbarians", "AngryBarbarians"),
    ("Skeleton Dragons", "SkeletonDragons"),
    ("Wall Breakers", "Wallbreakers"),
    ("Night Witch", "DarkWitch"),
]
N_V01 = 21

# SPEC §0 / §16.4 preset decks (display names, in the order the SPEC lists them).
DECKS = [
    ("hog26", ["Hog Rider", "Musketeer", "Cannon", "Ice Golem", "Ice Spirit", "Skeletons",
               "Fireball", "The Log"]),
    ("giant", ["Giant", "Prince", "Baby Dragon", "Wizard", "Minions", "Knight", "Arrows", "Zap"]),
    ("bait", ["Goblin Barrel", "Skeleton Army", "Tesla", "Valkyrie", "Archers", "Knight",
              "The Log", "Fireball"]),
    ("golem", ["Golem", "Night Witch", "Baby Dragon", "Lightning", "Mega Minion", "Barbarian Barrel",
               "Mini P.E.K.K.A", "Zap"]),
    ("lavaloon", ["Lava Hound", "Balloon", "Minions", "Mega Minion", "Skeleton Dragons", "Arrows",
                  "Fireball", "Tombstone"]),
    ("xbow", ["X-Bow", "Tesla", "Archers", "Knight", "Skeletons", "Ice Spirit", "Fireball", "The Log"]),
    ("miner_poison", ["Miner", "Poison", "Goblin Gang", "Bats", "Inferno Tower", "Valkyrie",
                      "Spear Goblins", "The Log"]),
    ("pekka_bridge", ["P.E.K.K.A", "Battle Ram", "Bandit", "Minions", "Musketeer", "Zap", "Poison",
                      "Dart Goblin"]),
    ("royal_hogs", ["Royal Hogs", "Earthquake", "Fire Spirit", "Barbarian Barrel", "Goblin Gang",
                    "Mega Minion", "Zap", "Musketeer"]),
]

# SPEC §2 towers: (source name, hp ladder rate, damage ladder rate). SPEC §16.3: the tower troops
# replace both Princess towers and use the Princess rates.
TOWERS = [("PrincessTower", 8, 8), ("KingTower", 7, 8)]
TOWER_TROOPS = [  # SPEC §16.6.16 index order: (config name, data row)
    ("princess", "PrincessTower"),
    ("cannoneer", "Cannoneer"),
    ("dagger_duchess", "DaggerDuchess"),
    ("royal_chef", "ChefTower"),
]
TOWER_CAP_LEVEL = 9
TOWER_RATE_AFTER_CAP = 10

# SPEC §16.6.26: the Royal Chef's level-up period (the data carries no timing field) and reach.
CHEF_PERIOD_MS = 28000
CHEF_RANGE = 7500
# [IMPL-DEFINED] Dagger Duchess reload after her AttackSequence (docs/FIDELITY.md §12).
DUCHESS_RELOAD_MS = 1000

# Spell delivery (PR_SPELL_*), keyed by source card key. The rule data behind each lives in the
# card's `projectile` / `spell` blocks; this table only names the shape.
SPELL_TYPES = {
    "Fireball": "PR_SPELL_PROJECTILE",
    "Arrows": "PR_SPELL_WAVES",
    "Zap": "PR_SPELL_AREA",
    "Log": "PR_SPELL_ROLLING",
    "GoblinBarrel": "PR_SPELL_SPAWN_PROJECTILE",
    "Rocket": "PR_SPELL_PROJECTILE",
    "Snowball": "PR_SPELL_PROJECTILE",
    "Freeze": "PR_SPELL_AREA",
    "Poison": "PR_SPELL_PULSE",
    "Earthquake": "PR_SPELL_PULSE",
    "Lightning": "PR_SPELL_LIGHTNING",
    "BarbLog": "PR_SPELL_ROLLING",
}

# ANSI glyphs (pr_render.h): v0.1 letters unchanged; the rest is a readable best effort.
GLYPHS = {
    "Knight": "N", "Archer": "A", "Musketeer": "M", "Giant": "G", "HogRider": "H", "Minion": "I",
    "BabyDragon": "D", "Valkyrie": "V", "Skeleton": "S", "IceGolemite": "O", "IceSpirits": "E",
    "Prince": "R", "Wizard": "Z", "Cannon": "C", "Tesla": "T", "Goblin": "B", "PrincessTower": "P",
    "KingTower": "K", "Barbarian": "Y", "MiniPekka": "Q", "Pekka": "Q", "MegaMinion": "I", "Bat": "J",
    "SpearGoblin": "B", "Goblin_Stab": "B", "RoyalGiant": "G", "Bomber": "S", "Princess": "A",
    "BlowdartGoblin": "B", "FireSpirits": "E", "IceWizard": "Z", "Golem": "G", "Golemite": "O",
    "LavaHound": "L", "LavaPups": "L", "Balloon": "U", "GiantSkeleton": "S", "Witch": "W",
    "Tombstone": "X", "InfernoTower": "F", "InfernoDragon": "D", "Assassin": "N", "BattleRam": "H",
    "RoyalHog": "H", "Miner": "N", "Mortar": "X", "Xbow": "X", "ElixirCollector": "$",
    "BombTower": "X", "RascalBoy": "N", "RascalGirl": "A", "AngryBarbarian": "Y",
    "SkeletonDragon": "D", "Wallbreaker": "S", "DarkWitch": "W", "Cannoneer": "P",
    "DaggerDuchess": "P", "ChefTower": "P",
}


def scale(v):
    """Level-11 value of a scaling stat (SPEC §2)."""
    if v is None:
        return 0
    return (int(v) * LEVEL_PCT) // 100


def tower_pct(rate, level=LEVEL):
    pct = 100
    for lv in range(2, level + 1):
        r = rate if lv <= TOWER_CAP_LEVEL else TOWER_RATE_AFTER_CAP
        pct = pct * (100 + r) // 100
    return pct


def footprint_tiles(radius):
    """SPEC §3: F = ceil((2*radius + 1000) / 1000)."""
    return -(-(2 * radius + 1000) // 1000)


def ival(v, default=0):
    return default if v is None else int(v)


def bval(v):
    return 1 if v else 0


def enum_name(prefix, name):
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch.upper())
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return f"{prefix}_{s.strip('_')}"


def c_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def ring(radius, n, shift_deg=0.0):
    """n points on a circle in the OWN frame (x right, y DOWN = toward the own side), the first at
    `shift_deg` clockwise from "toward the enemy" (own -y), the rest 360/n apart. Rounded once,
    here, with Python's IEEE doubles; the engine only ever sees the integers."""
    out = []
    for k in range(n):
        a = math.radians(shift_deg) + 2.0 * math.pi * k / n
        out.append((int(round(radius * math.sin(a))), int(round(-radius * math.cos(a)))))
    return out


class DB:
    def __init__(self, data):
        self.data = data
        self.cards = {c["name"]: c for c in data["cards"]}
        self.units_src = data["units"]
        self.projs_src = data["projectiles"]
        self.areas_src = data["area_effect_objects"]
        self.buffs_src = data["buffs"]
        self.units = []      # list of dicts (C rows)
        self.unit_idx = {}
        self.projs = []
        self.proj_idx = {}
        self.areas = []
        self.area_idx = {}
        self.buffs = []
        self.buff_idx = {}
        self.formations = []  # list of (count, offsets)
        self.form_idx = {}
        self.spawn_offsets = []  # flat (x, y) table for spawner waves and death spawns

    # ---------------------------------------------------------------- buffs
    def buff(self, name):
        if name is None:
            return -1
        if name in self.buff_idx:
            return self.buff_idx[name]
        b = self.buffs_src[name]
        row = {
            "name": name,
            "speed_pct": ival(b.get("speed_multiplier_raw")),
            "hit_speed_pct": ival(b.get("hit_speed_multiplier_raw")),
            "spawn_speed_pct": ival(b.get("spawn_speed_multiplier_raw")),
            "dps": scale(b.get("damage_per_second")),
            "hit_freq_ms": ival(b.get("hit_frequency_ms")),
            "crown_pct": ival(b.get("crown_tower_damage_percent"), 100),
            "building_pct": ival(b.get("building_damage_percent"), 100),
            "stacking": bval(b.get("enable_stacking")),
        }
        self.buff_idx[name] = len(self.buffs)
        self.buffs.append(row)
        return self.buff_idx[name]

    # ---------------------------------------------------------------- areas
    def area(self, name):
        if name is None:
            return -1
        if name in self.area_idx:
            return self.area_idx[name]
        a = self.areas_src[name]
        buff = a.get("buff")
        proj = a.get("projectile")
        row = {
            "name": name,
            "life_ms": ival(a.get("life_duration_ms")),
            "radius": ival(a.get("radius_milli")),
            "damage": scale(a.get("damage")),
            "crown_pct": ival(a.get("crown_tower_damage_percent"), 100),
            "buff": self.buff(buff["name"]) if buff else -1,
            "hits_air": bval(a.get("hits_air")),
            "hits_ground": bval(a.get("hits_ground")),
            "affects_hidden": bval(a.get("affects_hidden")),
            "cap_buff": bval(a.get("cap_buff_time_to_area_effect_time")),
            "buff_ms": ival(a.get("buff_time_ms")),
            "hit_speed_ms": ival(a.get("hit_speed_ms")),
            "projectile": self.proj(proj) if proj else -1,
        }
        self.area_idx[name] = len(self.areas)
        self.areas.append(row)
        return self.area_idx[name]

    # ---------------------------------------------------------------- projectiles
    def proj(self, p):
        """p: an inline projectile block (dict with 'name') or a projectile name; returns the index."""
        if p is None:
            return -1
        name = p if isinstance(p, str) else p["name"]
        if name in self.proj_idx:
            return self.proj_idx[name]
        # the flat table row is authoritative; the inline block mirrors it
        src = self.projs_src.get(name, p if isinstance(p, dict) else None)
        buff = src.get("target_buff")
        spawn = src.get("spawn_character")
        row = {
            "name": name,
            "speed": ival(src.get("speed")),
            "damage": scale(src.get("damage")),
            "crown_pct": ival(src.get("crown_tower_damage_percent"), 100),
            "radius": ival(src.get("radius_milli")) or ival(src.get("projectile_radius_milli")),
            "radius_y": ival(src.get("radius_y_milli")) or ival(src.get("projectile_radius_y_milli")),
            "aoe_air": bval(src.get("aoe_to_air")),
            "aoe_ground": bval(src.get("aoe_to_ground")),
            "homing": bval(src.get("homing")),
            "pushback_all": bval(src.get("pushback_all")),
            "target_buff": self.buff(buff["name"]) if buff else -1,
            "spawn_unit": -1,  # resolved after units exist
            "_spawn_name": spawn,
            "buff_ms": ival(src.get("buff_time_ms")),
            "pushback": ival(src.get("pushback_milli")),
            "range": ival(src.get("projectile_range_milli")),
            # a spawn_character with no count spawns one unit (BarbLog's Barbarian)
            "spawn_count": ival(src.get("spawn_character_count"), 1 if spawn else 0),
            "spawn_deploy_ms": ival(src.get("spawn_character_deploy_time_ms")),
        }
        self.proj_idx[name] = len(self.projs)
        self.projs.append(row)
        return self.proj_idx[name]

    # ---------------------------------------------------------------- offsets
    def offsets(self, pts):
        first = len(self.spawn_offsets)
        self.spawn_offsets.extend(pts)
        return first

    # ---------------------------------------------------------------- units
    def unit_row(self, name, src, kind, tower_rates=None):
        hp = ival(src.get("hitpoints"))
        dmg = ival(src.get("damage"))
        if tower_rates is not None:
            hp = hp * tower_pct(tower_rates[0]) // 100
            dmg = dmg * tower_pct(tower_rates[1]) // 100
        else:
            hp = scale(hp)
            dmg = scale(dmg)
        raw = src.get("raw") or {}
        proj = src.get("projectile")
        # SPEC §16.6.1: a unit with a CustomFirstProjectile (the Princess) fires that projectile;
        # its 5 `PrincessProjectileDeco` volleys are cosmetic
        if raw.get("CustomFirstProjectile"):
            proj = raw["CustomFirstProjectile"]
            dmg = scale(self.projs_src[proj].get("damage"))
        charge = src.get("charge") or {}
        jump = src.get("jump") or {}
        dash = src.get("dash") or {}
        radius = ival(src.get("collision_radius_milli"))
        pidx = self.proj(proj) if proj else -1
        hit_speed = ival(src.get("hit_speed_ms"))
        # a unit with no damage and no projectile never attacks (Tombstone: HitSpeed 10000 only;
        # Elixir Collector); its hit speed stays in card_info as the data has it
        no_attack = 1 if (dmg <= 0 and pidx < 0) else 0
        # variable damage (SPEC §16.2, §16.6.9): stages from the raw columns; a stage lasts
        # VariableDamageTime / HitSpeed hits
        var2 = scale(raw.get("VariableDamage2")) if raw.get("VariableDamage2") else 0
        var3 = scale(raw.get("VariableDamage3")) if raw.get("VariableDamage3") else 0
        vh1 = ival(raw.get("VariableDamageTime1")) // hit_speed if var2 and hit_speed else 0
        vh2 = ival(raw.get("VariableDamageTime2")) // hit_speed if var3 and hit_speed else 0
        seq = [int(e["HitSpeedMultiplier"]) for e in ((src.get("list_columns") or {}).get("AttackSequenceList") or [])]
        row = {
            "name": name,
            "kind": kind,
            "attacks_air": bval(src.get("attacks_air")),
            "attacks_ground": bval(src.get("attacks_ground")),
            "only_buildings": bval(src.get("target_only_buildings")),
            "self_aoe": bval(src.get("self_as_aoe_center")),
            "kamikaze": 1 if (src.get("kamikaze") and not src.get("kamikaze_time_ms")) else 0,
            "ignore_pushback": bval(src.get("ignore_pushback")),
            "jumps": 1 if jump else 0,
            "hides": bval(src.get("hides_when_not_attacking")),
            "no_attack": no_attack,
            "glyph": GLYPHS.get(name, "?"),
            "projectile": pidx,
            "death_area": self.area(src.get("death_area_effect")),
            "hp": hp,
            "damage": dmg,
            "shield": scale(src.get("shield_hitpoints")),
            "hit_speed_ms": hit_speed,
            "load_time_ms": ival(src.get("load_time_ms")),
            "speed": ival(src.get("speed")),
            "range": ival(src.get("range_milli")),
            "min_range": ival(src.get("minimum_range_milli")),
            "sight": ival(src.get("sight_range_milli")),
            "radius": radius,
            "mass": ival(src.get("mass")),
            "deploy_ms": ival(src.get("deploy_time_ms")),
            "flying_height": ival(src.get("flying_height")),
            "area_radius": ival(src.get("area_damage_radius_milli")),
            "proj_start_radius": ival(src.get("projectile_start_radius_milli")),
            "crown_pct": ival(src.get("crown_tower_damage_percent"), 100),
            "lifetime_ms": ival(src.get("lifetime_ms")),
            "death_damage": scale(src.get("death_damage")),
            "death_radius": ival(src.get("death_damage_radius_milli")),
            # charge.CHARGE_RANGE_UNIT = centitiles: 250 raw -> 2500 millitiles
            "charge_range": ival(charge.get("charge_range_raw")) * 10,
            "charge_damage": scale(charge.get("damage_special")),
            "charge_speed_pct": ival(charge.get("charge_speed_multiplier_percent")),
            "jump_speed": ival(jump.get("speed")),
            "hide_ms": ival(src.get("hide_time_ms")),
            "up_ms": ival(src.get("up_time_ms")),
            "footprint_tiles": footprint_tiles(radius) if kind != "PR_KIND_TROOP" else 0,
            # SPEC §16 mechanics
            "var_damage2": var2,
            "var_damage3": var3,
            "var_hits1": vh1,
            "var_hits2": vh2,
            "dash_damage": scale(dash.get("damage")),
            "dash_min": ival(dash.get("min_range_milli")),
            "dash_max": ival(dash.get("max_range_milli")),
            "dash_speed": ival(dash.get("speed")),
            "dash_cooldown_ms": ival(dash.get("cooldown_ms")),
            "burrow_speed": ival((src.get("spawn_pathfind") or {}).get("speed")),
            "mana_amount": ival(raw.get("ManaCollectAmount")) * 2800,
            "mana_gen_ms": ival(raw.get("ManaGenerateTimeMs")),
            "mana_on_death": ival(raw.get("ManaOnDeath")) * 2800,
            "seq_n": len(seq),
            "seq_mult": (seq + [0, 0, 0, 0])[:4],
            "reload_ms": DUCHESS_RELOAD_MS if seq else 0,
            "chef_period_ms": CHEF_PERIOD_MS if name == "ChefTower" else 0,
            "chef_range": CHEF_RANGE if name == "ChefTower" else 0,
            # spawner / death spawn / death bomb: filled by resolve_spawns()
            "spawn_unit": -1, "spawn_number": 0, "spawn_interval_ms": 0, "spawn_start_ms": 0,
            "spawn_pause_ms": 0, "spawn_first": 0,
            "dspawn_unit": -1, "dspawn_count": 0, "dspawn_deploy_ms": 0, "dspawn_first": 0,
            "bomb_damage": 0, "bomb_radius": 0, "bomb_fuse_ms": 0, "bomb_air": 0, "bomb_ground": 0,
            "_src": src,
        }
        if row["mass"] == 0 and kind == "PR_KIND_TROOP":
            row["mass"] = 1
        return row

    def unit(self, name, kind=None, src=None, tower_rates=None):
        if name in self.unit_idx:
            return self.unit_idx[name]
        if src is None:
            src = self.units_src[name]
        if kind is None:
            kind = "PR_KIND_BUILDING" if src.get("source_table") == "buildings" else "PR_KIND_TROOP"
        row = self.unit_row(name, src, kind, tower_rates)
        self.unit_idx[name] = len(self.units)
        self.units.append(row)
        return self.unit_idx[name]

    def is_death_bomb(self, name):
        """A DeathSpawnCharacter with no hitpoints but DeployTime + DeathDamage: one timed area
        hit (RoyaleSim card.rs `convert_death_bomb`), not a unit (SPEC §16.2)."""
        u = self.units_src[name]
        return not u.get("hitpoints") and u.get("death_damage")

    def resolve_spawns(self, ui):
        """Spawner and death-spawn blocks of unit ui (registers the spawned units)."""
        row = self.units[ui]
        src = row["_src"]
        sp = src.get("spawner")
        if sp:
            su = self.unit(sp["character"])
            srow = self.units[su]
            n = int(sp["number"])
            interval = ival(sp.get("interval_ms"))
            radius = ival(sp.get("radius_milli"))
            if radius > 0 and interval == 0:
                # a wave that SETS SpawnRadius appears on a ring of that radius around the spawner
                # (ledger spawner.SPAWN_POINT), first unit at SpawnAngleShift
                pts = ring(radius, n, ival(src.get("spawn_angle_shift_deg")))
            else:
                # blank SpawnRadius: the emission point in front of the spawner at the two circles'
                # tangent (ledger spawner.SPAWN_POINT), one unit at a time
                pts = [(0, -(row["radius"] + srow["radius"]))] * n
            row.update(spawn_unit=su, spawn_number=n, spawn_interval_ms=interval,
                       spawn_start_ms=ival(sp.get("start_time_ms")), spawn_pause_ms=ival(sp.get("pause_time_ms")),
                       spawn_first=self.offsets(pts))
        ds = src.get("death_spawn")
        if ds:
            name = ds["character"]
            if self.is_death_bomb(name):
                b = self.units_src[name]
                row.update(bomb_damage=scale(b["death_damage"]), bomb_radius=ival(b.get("death_damage_radius_milli")),
                           bomb_fuse_ms=ival(b.get("deploy_time_ms")), bomb_air=bval(b.get("attacks_air")),
                           bomb_ground=bval(b.get("attacks_ground")))
            else:
                du = self.unit(name)
                drow = self.units[du]
                cnt = ival(ds.get("count"), 1)
                radius = ival(ds.get("radius_milli"))
                if radius > 0:
                    # ledger spawner.DEATH_SPAWN_LAYOUT: a ring of DeathSpawnRadius (facing rule
                    # [IMPL-DEFINED]: own-frame "toward the enemy" + SpawnAngleShift)
                    pts = ring(radius, cnt, ival(src.get("spawn_angle_shift_deg")))
                elif sp:
                    # ledger spawner.DEATH_SPAWN_AT_EMISSION_POINT (the Tombstone): all at the
                    # point the spawner emits at
                    pts = [(0, -(row["radius"] + drow["radius"]))] * cnt
                else:
                    pts = [(0, 0)] * cnt  # spawner.DEATH_SPAWN_RADIUS_DEFAULT = zero
                row.update(dspawn_unit=du, dspawn_count=cnt,
                           dspawn_deploy_ms=ival(ds.get("deploy_time_ms")), dspawn_first=self.offsets(pts))

    # ---------------------------------------------------------------- formations
    def formation(self, count):
        if count in self.form_idx:
            return self.form_idx[count]
        offs = formation_offsets(count)
        self.form_idx[count] = len(self.formations)
        self.formations.append((count, offs))
        return self.form_idx[count]


def formation_offsets(n):
    """SPEC §5.1 formation offsets, own frame (x right, y DOWN = toward own side).

    n=1: the point; n=2: 1000 apart horizontally; n=3: the triangle (0,-577),
    (+500,+289), (-500,+289) with the apex toward the enemy; n=4: a 2x2 square 1000 on a side;
    n=6: two rows of three (the front row is members 0-2: Goblin Gang's stab goblins); n=5 and
    n>=7: concentric rings, all members within 1600 of the point [IMPL-DEFINED]: n=5 a ring of
    radius 800; larger n one member on the point, then rings of radius 800 (6 members) and 1500
    (the rest, up to 12). Offsets are rounded once, here, with Python's IEEE doubles; the engine
    only ever sees the integers (no trigonometry at runtime).
    """
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(-500, 0), (500, 0)]
    if n == 3:
        return [(0, -577), (500, 289), (-500, 289)]
    if n == 4:
        return [(-500, -500), (500, -500), (-500, 500), (500, 500)]
    if n == 5:
        return ring(800, 5)
    if n == 6:
        return [(-1000, -500), (0, -500), (1000, -500), (-1000, 500), (0, 500), (1000, 500)]
    offs = [(0, 0)]
    rings = [(800, 6), (1500, 12)]
    left = n - 1
    for radius, cap in rings:
        k = min(cap, left)
        if k <= 0:
            break
        phase = 0.0 if radius == 800 else math.pi / k
        for i in range(k):
            a = phase + 2.0 * math.pi * i / k   # angle from "toward the enemy", clockwise
            offs.append((int(round(radius * math.sin(a))), int(round(-radius * math.cos(a)))))
        left -= k
    if left > 0:
        raise SystemExit(f"formation for n={n} does not fit the ring layout")
    return offs


def build(data):
    db = DB(data)
    card_rows = []
    towers_src = {t["name"]: t for t in data["towers"]}

    def add_card(display, key):
        c = db.cards[key]
        kind = c["kind"]
        count = ival(c.get("count"), 1)
        second = c.get("second_summon")
        delay = ival(c.get("summon_deploy_delay_ms")) or ival(c.get("summon_deploy_delay_second_ms"))
        row = {
            "name": display,
            "key": key,
            "elixir": int(c["elixir"]),
            "unit": -1,
            "count": count,
            "unit2": -1,
            "count2": 0,
            "formation": -1,
            "deploy_delay_ms": delay,
            "spell_type": "PR_SPELL_NONE",
            "spell_proj": -1,
            "spell_area": -1,
            "spell_radius": 0,
            "spell_damage": 0,
            "crown_pct": ival(c.get("crown_tower_damage_percent"), 100),
            "waves": 0,
            "wave_interval_ms": 0,
            "placement": "PR_PLACE_ANY",
        }
        if kind in ("troop", "building"):
            row["kind"] = "PR_CARD_KIND_TROOP" if kind == "troop" else "PR_CARD_KIND_BUILDING"
            ukind = "PR_KIND_TROOP" if kind == "troop" else "PR_KIND_BUILDING"
            row["unit"] = db.unit(c["summon_character"], ukind)
            if second:
                row["unit2"] = db.unit(second["character"], ukind)
                row["count2"] = int(second["count"])
            row["formation"] = db.formation(row["count"] + row["count2"])
            if kind == "building":
                row["placement"] = "PR_PLACE_BUILDING"
            elif db.units[row["unit"]]["burrow_speed"] > 0:
                row["placement"] = "PR_PLACE_MINER"   # SPEC §16.6.3
            else:
                row["placement"] = "PR_PLACE_TROOP"
        elif kind == "spell":
            row["kind"] = "PR_CARD_KIND_SPELL"
            row["spell_type"] = SPELL_TYPES[key]
            sp = c.get("spell") or {}
            proj = c.get("projectile")
            if key in ("Log", "BarbLog"):
                # the airborne LogProjectile only carries the rolling projectile (SPEC §7.2, §16.6.20)
                proj = proj["spawn_projectile"]
            if proj is not None:
                row["spell_proj"] = db.proj(proj)
            aeo = sp.get("area_effect_object")
            if aeo is not None:
                row["spell_area"] = db.area(aeo["name"])
            # placement (SPEC §3.1, v0.2; §16.1): troop territory for the Log / Barbarian Barrel
            # (data can_deploy_on_enemy_side = false), any non-water tile for Goblin Barrel
            if not sp.get("can_deploy_on_enemy_side", True):
                row["placement"] = "PR_PLACE_TERRITORY"
            elif key == "GoblinBarrel":
                row["placement"] = "PR_PLACE_LAND"
            st = row["spell_type"]
            if key == "Arrows":
                row["spell_radius"] = ival(sp.get("radius_milli"))
                row["waves"] = ival(sp.get("projectile_waves"))
                row["wave_interval_ms"] = ival(sp.get("projectile_wave_interval_ms"))
                row["spell_damage"] = scale(proj["damage"])
            elif st == "PR_SPELL_AREA":
                row["spell_radius"] = ival(aeo["radius_milli"])
                row["spell_damage"] = scale(aeo["damage"])
                row["crown_pct"] = ival(aeo.get("crown_tower_damage_percent"), 100)
            elif st in ("PR_SPELL_PROJECTILE", "PR_SPELL_ROLLING"):
                pr = db.projs[row["spell_proj"]]
                row["spell_radius"] = pr["radius"]
                row["spell_damage"] = pr["damage"]
                row["crown_pct"] = pr["crown_pct"]
            elif st == "PR_SPELL_PULSE":
                a = db.areas[row["spell_area"]]
                b = db.buffs[a["buff"]]
                row["spell_radius"] = a["radius"]
                row["spell_damage"] = b["dps"] * b["hit_freq_ms"] // 1000   # one event
                row["crown_pct"] = b["crown_pct"]
                row["waves"] = a["life_ms"] // b["hit_freq_ms"]            # number of events
                row["wave_interval_ms"] = b["hit_freq_ms"]
            elif st == "PR_SPELL_LIGHTNING":
                a = db.areas[row["spell_area"]]
                pr = db.projs[a["projectile"]]
                row["spell_proj"] = a["projectile"]
                row["spell_radius"] = a["radius"]
                row["spell_damage"] = pr["damage"]
                row["crown_pct"] = pr["crown_pct"]
                row["waves"] = 3                                           # SPEC §16.1: up to 3
                row["wave_interval_ms"] = a["hit_speed_ms"]
            elif key == "GoblinBarrel":
                spawn = sp["spawn"]
                db.unit(spawn["character"], "PR_KIND_TROOP")
                row["count"] = int(spawn["count"])
                row["formation"] = db.formation(row["count"])
        else:
            raise SystemExit(f"unknown card kind {kind} for {key}")
        card_rows.append(row)

    # 1. the v0.1 cards, then the crown towers (their indices are those of v0.2)
    for display, key in CARDS[:N_V01]:
        add_card(display, key)
    for name, hp_rate, dmg_rate in TOWERS:
        db.unit(name, "PR_KIND_TOWER", towers_src[name], (hp_rate, dmg_rate))
    # 2. the §16 cards, their spawned units and the tower troops
    for display, key in CARDS[N_V01:]:
        add_card(display, key)
    for _cfg, name in TOWER_TROOPS[1:]:
        db.unit(name, "PR_KIND_TOWER", db.units_src[name], (8, 8))
    # tower projectile damage follows the tower's damage ladder, not the card ladder
    for u in db.units:
        if u["kind"] == "PR_KIND_TOWER":
            db.projs[u["projectile"]]["damage"] = u["damage"]

    # spawners / death spawns / death bombs (registers spawned units until none is new)
    done = 0
    while done < len(db.units):
        db.resolve_spawns(done)
        done += 1

    # resolve projectile spawn units
    for pr in db.projs:
        sname = pr.pop("_spawn_name")
        if sname is not None:
            pr["spawn_unit"] = db.unit(sname, "PR_KIND_TROOP")
    for u in db.units:
        u.pop("_src", None)
    return db, card_rows


def self_check(db, cards):
    """Hand-checked level-11 numbers. A failure here means the data or the rule moved."""
    u = {r["name"]: r for r in db.units}
    p = {r["name"]: r for r in db.projs}
    a = {r["name"]: r for r in db.areas}
    b = {r["name"]: r for r in db.buffs}

    def eq(what, got, want):
        if got != want:
            raise SystemExit(f"self-check failed: {what} = {got}, expected {want}")

    def ct(d, pct):
        return -(-(d * pct) // 100)

    # ---- SPEC §2 examples
    eq("Knight hp", u["Knight"]["hp"], 1766)
    eq("Knight dmg", u["Knight"]["damage"], 202)
    eq("Hog hp", u["HogRider"]["hp"], 1697)
    eq("Fireball dmg", p["FireballSpell"]["damage"], 688)
    eq("Zap dmg", a["Zap"]["damage"], 192)
    eq("Log dmg", p["LogProjectileRolling"]["damage"], 268)
    eq("Arrows wave dmg", p["ArrowsSpell"]["damage"], 122)
    eq("Goblin hp", u["Goblin"]["hp"], 202)
    eq("Goblin dmg", u["Goblin"]["damage"], 125)
    eq("IceGolem death dmg", u["IceGolemite"]["death_damage"], 84)
    eq("Prince charge dmg", u["Prince"]["charge_damage"], 783)
    eq("Princess hp", u["PrincessTower"]["hp"], 3052)
    eq("Princess dmg", u["PrincessTower"]["damage"], 109)
    eq("King hp", u["KingTower"]["hp"], 4824)
    eq("King dmg", u["KingTower"]["damage"], 109)
    eq("Princess F", u["PrincessTower"]["footprint_tiles"], 3)
    eq("King F", u["KingTower"]["footprint_tiles"], 4)
    eq("Cannon F", u["Cannon"]["footprint_tiles"], 3)
    eq("Tesla F", u["Tesla"]["footprint_tiles"], 2)
    eq("Fireball vs tower", ct(688, p["FireballSpell"]["crown_pct"]), 172)
    eq("Zap vs tower", ct(192, a["Zap"]["crown_pct"]), 48)
    eq("Arrows vs tower", ct(122, p["ArrowsSpell"]["crown_pct"]), 25)
    eq("Log vs tower", ct(268, p["LogProjectileRolling"]["crown_pct"]), 35)

    # ---- ids and costs (SPEC §0, §4.1, §16.1)
    eq("card count", len(cards), 64)
    costs = [3, 3, 4, 5, 4, 3, 4, 4, 3, 1, 2, 1, 5, 5, 3, 4, 4, 3, 2, 2, 3,
             5, 4, 7, 3, 2, 2, 2, 3, 6, 2, 3, 3, 5, 1, 3, 8, 7, 5, 6, 5, 3, 5, 4, 3, 4, 5, 3,
             4, 6, 6, 4, 6, 4, 4, 3, 6, 2, 2, 5, 6, 4, 2, 4]
    eq("costs", [c["elixir"] for c in cards], costs)
    eq("card names", [c["name"] for c in cards], [n for n, _ in CARDS])

    # ---- SPEC §16 units at level 11: name -> (hp, damage, extra checks)
    units = {
        "Barbarian": (716, 192), "MiniPekka": (1390, 755), "Pekka": (3760, 842),
        "MegaMinion": (837, 312), "Bat": (81, 81), "SpearGoblin": (133, 81), "Goblin_Stab": (202, 125),
        "RoyalGiant": (3164, 307), "Bomber": (304, 225), "Princess": (261, 168),
        "BlowdartGoblin": (261, 151), "Minion": (230, 107), "FireSpirits": (217, 207),
        "IceWizard": (688, 89), "Golem": (5120, 312), "Golemite": (1039, 84), "LavaHound": (3581, 53),
        "LavaPups": (215, 81), "Balloon": (1676, 640), "GiantSkeleton": (3361, 276), "Witch": (839, 135),
        "Skeleton": (81, 81), "Tombstone": (529, 0), "InfernoTower": (1748, 43),
        "InfernoDragon": (1295, 35), "Assassin": (906, 194), "BattleRam": (967, 286),
        "RoyalHog": (837, 74), "Miner": (1210, 194), "Mortar": (1369, 266), "Xbow": (1600, 58),
        "ElixirCollector": (1070, 0), "BombTower": (1356, 222), "RascalBoy": (1832, 217),
        "RascalGirl": (261, 125), "AngryBarbarian": (1341, 384), "SkeletonDragon": (560, 151),
        "Wallbreaker": (330, 281), "DarkWitch": (906, 314),
    }
    for name, (hp, dmg) in units.items():
        eq(f"{name} hp", u[name]["hp"], hp)
        eq(f"{name} damage", u[name]["damage"], dmg)
    # death damage, death bombs, charge, dash, variable damage, burrow, collector, min range
    eq("Golem death damage", (u["Golem"]["death_damage"], u["Golem"]["death_radius"]), (225, 2000))
    eq("Golemite death damage", u["Golemite"]["death_damage"], 99)
    eq("Balloon bomb", (u["Balloon"]["bomb_damage"], u["Balloon"]["bomb_radius"], u["Balloon"]["bomb_fuse_ms"]), (240, 3000, 3000))
    eq("GiantSkeleton bomb", (u["GiantSkeleton"]["bomb_damage"], u["GiantSkeleton"]["bomb_radius"]), (688, 3000))
    eq("BombTower bomb", (u["BombTower"]["bomb_damage"], u["BombTower"]["bomb_radius"]), (222, 3000))
    eq("BattleRam charge", (u["BattleRam"]["charge_damage"], u["BattleRam"]["charge_range"], u["BattleRam"]["kamikaze"]), (573, 3000, 1))
    eq("Bandit dash", (u["Assassin"]["dash_damage"], u["Assassin"]["dash_min"], u["Assassin"]["dash_max"],
                       u["Assassin"]["dash_speed"], u["Assassin"]["dash_cooldown_ms"]), (389, 3500, 6000, 500, 800))
    eq("InfernoTower stages", (u["InfernoTower"]["damage"], u["InfernoTower"]["var_damage2"], u["InfernoTower"]["var_damage3"],
                               u["InfernoTower"]["var_hits1"], u["InfernoTower"]["var_hits2"]), (43, 158, 847, 5, 5))
    eq("InfernoDragon stages", (u["InfernoDragon"]["damage"], u["InfernoDragon"]["var_damage2"], u["InfernoDragon"]["var_damage3"],
                                u["InfernoDragon"]["var_hits1"], u["InfernoDragon"]["var_hits2"]), (35, 120, 422, 5, 5))
    eq("Miner", (u["Miner"]["burrow_speed"], u["Miner"]["crown_pct"]), (650, 20))
    eq("Collector", (u["ElixirCollector"]["mana_amount"], u["ElixirCollector"]["mana_gen_ms"],
                     u["ElixirCollector"]["mana_on_death"], u["ElixirCollector"]["lifetime_ms"]), (2800, 13000, 2800, 93000))
    eq("Mortar min range", (u["Mortar"]["min_range"], u["Mortar"]["range"]), (3500, 11500))
    eq("Tombstone no attack", (u["Tombstone"]["no_attack"], u["Tombstone"]["hit_speed_ms"]), (1, 10000))
    eq("RoyalHog jumps", u["RoyalHog"]["jumps"], 1)
    # spawners and death spawns (count, deploy)
    sk = db.unit_idx["Skeleton"]
    eq("Witch spawner", (u["Witch"]["spawn_unit"], u["Witch"]["spawn_number"], u["Witch"]["spawn_start_ms"],
                         u["Witch"]["spawn_pause_ms"]), (sk, 4, 1000, 7000))
    eq("Tombstone spawner", (u["Tombstone"]["spawn_unit"], u["Tombstone"]["spawn_number"], u["Tombstone"]["spawn_interval_ms"],
                             u["Tombstone"]["spawn_pause_ms"], u["Tombstone"]["dspawn_count"]), (sk, 2, 500, 3500, 4))
    eq("NightWitch spawner", (u["DarkWitch"]["spawn_unit"], u["DarkWitch"]["spawn_number"], u["DarkWitch"]["spawn_pause_ms"],
                              u["DarkWitch"]["dspawn_unit"], u["DarkWitch"]["dspawn_count"]),
       (db.unit_idx["Bat"], 2, 5000, db.unit_idx["Bat"], 1))
    eq("Golem death spawn", (u["Golem"]["dspawn_unit"], u["Golem"]["dspawn_count"]), (db.unit_idx["Golemite"], 2))
    eq("LavaHound death spawn", (u["LavaHound"]["dspawn_unit"], u["LavaHound"]["dspawn_count"]), (db.unit_idx["LavaPups"], 6))
    eq("BattleRam death spawn", (u["BattleRam"]["dspawn_unit"], u["BattleRam"]["dspawn_count"], u["BattleRam"]["dspawn_deploy_ms"]),
       (db.unit_idx["Barbarian"], 2, 1000))
    # projectiles
    eq("Princess projectile", (p["PrincessProjectile"]["damage"], p["PrincessProjectile"]["radius"],
                               p["PrincessProjectile"]["homing"]), (168, 2000, 0))
    for name in ("MortarProjectile", "BombTowerProjectile", "WallbreakerProjectile", "BombSkeletonProjectile"):
        eq(f"{name} non-homing", p[name]["homing"], 0)
    # spells
    eq("Rocket", (p["RocketSpell"]["damage"], p["RocketSpell"]["radius"], ct(1484, p["RocketSpell"]["crown_pct"]),
                  p["RocketSpell"]["pushback"]), (1484, 2000, 342, 1800))
    eq("Snowball", (p["SnowballSpell"]["damage"], p["SnowballSpell"]["radius"], ct(179, p["SnowballSpell"]["crown_pct"]),
                    p["SnowballSpell"]["pushback"], p["SnowballSpell"]["buff_ms"]), (179, 2500, 45, 1800, 3000))
    eq("Freeze", (a["Freeze"]["damage"], a["Freeze"]["radius"], ct(148, a["Freeze"]["crown_pct"]), a["Freeze"]["buff_ms"]),
       (148, 3000, 37, 4000))
    eq("Lightning", (p["LighningSpell"]["damage"], ct(1057, p["LighningSpell"]["crown_pct"]), a["Lightning"]["radius"],
                     a["Lightning"]["hit_speed_ms"]), (1057, 265, 3500, 460))
    eq("Poison buff", (b["Poison"]["dps"], b["Poison"]["hit_freq_ms"], ct(92, b["Poison"]["crown_pct"]), b["Poison"]["speed_pct"],
                       b["Poison"]["stacking"], a["Poison"]["life_ms"], a["Poison"]["radius"]), (92, 1000, 22, -15, 1, 8000, 3500))
    eq("Earthquake buff", (b["Earthquake"]["dps"], 81 * b["Earthquake"]["building_pct"] // 100, ct(81, b["Earthquake"]["crown_pct"]),
                           b["Earthquake"]["speed_pct"], a["Earthquake"]["life_ms"], a["Earthquake"]["hits_air"],
                           a["Earthquake"]["hit_speed_ms"], a["Earthquake"]["cap_buff"], a["Poison"]["hit_speed_ms"],
                           a["Poison"]["cap_buff"]),
       (81, 283, 49, -50, 3000, 0, 100, 1, 250, 0))
    eq("BarbLog", (p["BarbLogProjectileRolling"]["damage"], p["BarbLogProjectileRolling"]["range"],
                   p["BarbLogProjectileRolling"]["radius"], p["BarbLogProjectileRolling"]["radius_y"],
                   p["BarbLogProjectileRolling"]["spawn_count"], p["BarbLogProjectileRolling"]["spawn_deploy_ms"]),
       (232, 4500, 1300, 600, 1, 1000))
    # tower troops (SPEC §16.3: the tower ladder, Princess rates = 218 %)
    eq("tower pct", tower_pct(8), 218)
    eq("Cannoneer", (u["Cannoneer"]["hp"], u["Cannoneer"]["damage"], u["Cannoneer"]["hit_speed_ms"]), (2616, 272, 2200))
    eq("DaggerDuchess", (u["DaggerDuchess"]["hp"], u["DaggerDuchess"]["damage"], u["DaggerDuchess"]["seq_n"],
                         u["DaggerDuchess"]["seq_mult"]), (2768, 91, 4, [100, 100, 70, 90]))
    eq("ChefTower", (u["ChefTower"]["hp"], u["ChefTower"]["damage"], u["ChefTower"]["hit_speed_ms"]), (2703, 109, 1000))
    for name in ("Cannoneer", "DaggerDuchess", "ChefTower"):
        eq(f"{name} projectile damage", db.projs[u[name]["projectile"]]["damage"], u[name]["damage"])
        eq(f"{name} F", u[name]["footprint_tiles"], 3)
    # formations stay within the SPEC §5.1 bound
    for count, offs in db.formations:
        for (x, y) in offs:
            assert x * x + y * y <= 1600 * 1600, (count, x, y)


UNIT_FIELDS = [
    ("name", "s"), ("kind", "e"), ("attacks_air", "i"), ("attacks_ground", "i"),
    ("only_buildings", "i"), ("self_aoe", "i"), ("kamikaze", "i"), ("ignore_pushback", "i"),
    ("jumps", "i"), ("hides", "i"), ("no_attack", "i"), ("glyph", "c"), ("projectile", "proj"),
    ("death_area", "area"),
    ("hp", "i"), ("damage", "i"), ("shield", "i"), ("hit_speed_ms", "i"), ("load_time_ms", "i"),
    ("speed", "i"), ("range", "i"), ("min_range", "i"), ("sight", "i"), ("radius", "i"), ("mass", "i"),
    ("deploy_ms", "i"), ("flying_height", "i"), ("area_radius", "i"), ("proj_start_radius", "i"),
    ("crown_pct", "i"), ("lifetime_ms", "i"), ("death_damage", "i"), ("death_radius", "i"),
    ("charge_range", "i"), ("charge_damage", "i"), ("charge_speed_pct", "i"), ("jump_speed", "i"),
    ("hide_ms", "i"), ("up_ms", "i"), ("footprint_tiles", "i"),
    ("var_damage2", "i"), ("var_damage3", "i"), ("var_hits1", "i"), ("var_hits2", "i"),
    ("dash_damage", "i"), ("dash_min", "i"), ("dash_max", "i"), ("dash_speed", "i"), ("dash_cooldown_ms", "i"),
    ("burrow_speed", "i"), ("mana_amount", "i"), ("mana_gen_ms", "i"), ("mana_on_death", "i"),
    ("seq_n", "i"), ("seq_mult", "a4"), ("reload_ms", "i"), ("chef_period_ms", "i"), ("chef_range", "i"),
    ("spawn_unit", "unit"), ("spawn_number", "i"), ("spawn_interval_ms", "i"), ("spawn_start_ms", "i"),
    ("spawn_pause_ms", "i"), ("spawn_first", "i"),
    ("dspawn_unit", "unit"), ("dspawn_count", "i"), ("dspawn_deploy_ms", "i"), ("dspawn_first", "i"),
    ("bomb_damage", "i"), ("bomb_radius", "i"), ("bomb_fuse_ms", "i"), ("bomb_air", "i"), ("bomb_ground", "i"),
]
PROJ_FIELDS = [
    ("name", "s"), ("speed", "i"), ("damage", "i"), ("crown_pct", "i"), ("radius", "i"),
    ("radius_y", "i"), ("aoe_air", "i"), ("aoe_ground", "i"), ("homing", "i"),
    ("pushback_all", "i"), ("target_buff", "buff"), ("spawn_unit", "unit"), ("buff_ms", "i"),
    ("pushback", "i"), ("range", "i"), ("spawn_count", "i"), ("spawn_deploy_ms", "i"),
]
AREA_FIELDS = [
    ("name", "s"), ("life_ms", "i"), ("radius", "i"), ("damage", "i"), ("crown_pct", "i"),
    ("buff", "buff"), ("hits_air", "i"), ("hits_ground", "i"), ("affects_hidden", "i"), ("cap_buff", "i"),
    ("buff_ms", "i"), ("hit_speed_ms", "i"), ("projectile", "proj"),
]
BUFF_FIELDS = [("name", "s"), ("speed_pct", "i"), ("hit_speed_pct", "i"), ("spawn_speed_pct", "i"),
               ("dps", "i"), ("hit_freq_ms", "i"), ("crown_pct", "i"), ("building_pct", "i"), ("stacking", "i")]
CARD_FIELDS = [
    ("name", "s"), ("key", "s"), ("kind", "e"), ("elixir", "i"), ("spell_type", "e"),
    ("placement", "e"), ("unit", "unit"), ("count", "i"), ("unit2", "unit"), ("count2", "i"),
    ("formation", "form"), ("spell_proj", "proj"),
    ("spell_area", "area"), ("deploy_delay_ms", "i"), ("spell_radius", "i"),
    ("spell_damage", "i"), ("crown_pct", "i"), ("waves", "i"), ("wave_interval_ms", "i"),
]


def render(db, cards, sha):
    unit_enum = [enum_name("PR_UNIT", r["name"]) for r in db.units]
    proj_enum = [enum_name("PR_PROJ", r["name"]) for r in db.projs]
    area_enum = [enum_name("PR_AREA", r["name"]) for r in db.areas]
    buff_enum = [enum_name("PR_BUFF", r["name"]) for r in db.buffs]
    card_enum = [enum_name("PR_CARD", c["name"]) for c in cards]
    form_enum = [f"PR_FORMATION_N{count}" for count, _ in db.formations]
    refs = {"unit": unit_enum, "proj": proj_enum, "area": area_enum, "buff": buff_enum,
            "form": form_enum}

    def fmt(row, fields):
        parts = []
        for name, kind in fields:
            v = row[name]
            if kind == "s":
                parts.append(f".{name} = {c_str(v)}")
            elif kind == "c":
                parts.append(f".{name} = '{v}'")
            elif kind == "e":
                parts.append(f".{name} = {v}")
            elif kind == "i":
                parts.append(f".{name} = {int(v)}")
            elif kind == "a4":
                parts.append(f".{name} = {{{', '.join(str(int(x)) for x in v)}}}")
            else:
                parts.append(f".{name} = {refs[kind][v] if v >= 0 else -1}")
        lines = []
        cur = "    {"
        for i, p in enumerate(parts):
            piece = p + ("," if i < len(parts) - 1 else "")
            if len(cur) + len(piece) + 1 > 100:
                lines.append(cur.rstrip())
                cur = "      " + piece + " "
            else:
                cur += piece + " "
        lines.append(cur.rstrip() + "},")
        return "\n".join(lines)

    def enum_block(names, count_name):
        body = ",\n".join(f"    {n} = {i}" for i, n in enumerate(names))
        return f"enum {{\n{body},\n    {count_name} = {len(names)}\n}};\n"

    out = []
    out.append("/*")
    out.append(" * pr_card_db.h -- GENERATED by tools/gen_card_db.py. DO NOT EDIT BY HAND.")
    out.append(" *")
    out.append(" * Source: data/source/cards-15.535.json (RoyaleSim 72ed062, client 15.535.29)")
    out.append(f" * Source sha256: {sha}")
    out.append(" * Level 11: every scaling stat = floor(base * 256 / 100); crown towers and tower troops")
    out.append(" * use the compounding tower ladder (SPEC §2, §16.3, ledger combat.TOWER_HITPOINT_LADDER).")
    out.append(" * Distances millitiles, durations ms, speeds millitiles per 50 ms tick.")
    out.append(" */")
    out.append("#ifndef PR_CARD_DB_H")
    out.append("#define PR_CARD_DB_H")
    out.append("")
    out.append('#include "pr_defs.h"')
    out.append("")
    out.append(f'#define PR_DATA_SHA256 "{sha}"')
    out.append(f"#define PR_CARD_LEVEL {LEVEL}")
    out.append(f"#define PR_LEVEL_MULT_PCT {LEVEL_PCT}")
    out.append("")
    out.append("/* units: anything that becomes an entity */")
    out.append(enum_block(unit_enum, "PR_N_UNITS"))
    out.append("/* projectiles */")
    out.append(enum_block(proj_enum, "PR_N_PROJS"))
    out.append("/* area-effect objects */")
    out.append(enum_block(area_enum, "PR_N_AREAS"))
    out.append("/* character buffs (status effects) */")
    out.append(enum_block(buff_enum, "PR_N_BUFFS"))
    out.append("/* playable cards: ids are STABLE and part of the public API (SPEC §0, §16.1) */")
    out.append(enum_block(card_enum, "PR_N_CARDS"))
    out.append("/* formations (by member count) */")
    out.append(enum_block(form_enum, "PR_N_FORMATIONS"))

    out.append("static const PrUnitDef PR_UNITS[PR_N_UNITS] = {")
    for r in db.units:
        out.append(fmt(r, UNIT_FIELDS))
    out.append("};\n")
    out.append("static const PrProjDef PR_PROJS[PR_N_PROJS] = {")
    for r in db.projs:
        out.append(fmt(r, PROJ_FIELDS))
    out.append("};\n")
    out.append("static const PrAreaDef PR_AREAS[PR_N_AREAS] = {")
    for r in db.areas:
        out.append(fmt(r, AREA_FIELDS))
    out.append("};\n")
    out.append("static const PrBuffDef PR_BUFFS[PR_N_BUFFS] = {")
    for r in db.buffs:
        out.append(fmt(r, BUFF_FIELDS))
    out.append("};\n")
    out.append("static const PrCardDef PR_CARDS[PR_N_CARDS] = {")
    for r in cards:
        out.append(fmt(r, CARD_FIELDS))
    out.append("};\n")

    total = sum(len(o) for _, o in db.formations)
    out.append("/* member offsets in the acting team's OWN frame (-y = toward the enemy) */")
    out.append(f"static const int16_t PR_FORMATION_OFFSETS[{total}][2] = {{")
    first = []
    k = 0
    for count, offs in db.formations:
        first.append(k)
        items = ", ".join(f"{{{x}, {y}}}" for x, y in offs)
        out.append(f"    /* n={count} */ {items},")
        k += len(offs)
    out.append("};\n")
    out.append("static const PrFormationDef PR_FORMATIONS[PR_N_FORMATIONS] = {")
    for (count, _), f in zip(db.formations, first):
        out.append(f"    {{.count = {count}, .first = {f}}},")
    out.append("};\n")

    out.append("/* spawner-wave and death-spawn offsets in the OWNER's own frame (-y = toward the enemy);")
    out.append(" * PrUnitDef.spawn_first / dspawn_first index this table */")
    out.append(f"#define PR_N_SPAWN_OFFSETS {len(db.spawn_offsets)}")
    items = ", ".join(f"{{{x}, {y}}}" for x, y in db.spawn_offsets)
    out.append(f"static const int16_t PR_SPAWN_OFFSETS[PR_N_SPAWN_OFFSETS][2] = {{{items}}};\n")

    out.append("/* SPEC §16.3 tower troops, index order of SPEC §16.6.16 */")
    out.append(f"#define PR_N_TOWER_TROOPS {len(TOWER_TROOPS)}")
    out.append("static const char *const PR_TOWER_TROOP_NAMES[PR_N_TOWER_TROOPS] = {"
               + ", ".join(c_str(n) for n, _ in TOWER_TROOPS) + "};")
    out.append("static const int16_t PR_TOWER_TROOP_UNITS[PR_N_TOWER_TROOPS] = {"
               + ", ".join(unit_enum[db.unit_idx[r]] for _, r in TOWER_TROOPS) + "};\n")

    ladder = db.cards["Knight"]["level_scaling"]["multiplier_percent_by_level"]
    out.append("/* Common level ladder (percent of the level-1 stat), index = level - 1 (Royal Chef level-ups) */")
    out.append(f"#define PR_N_LEVELS {len(ladder)}")
    out.append("static const int16_t PR_LADDER[PR_N_LEVELS] = {" + ", ".join(str(int(x)) for x in ladder) + "};\n")

    name_to_id = {c["name"]: i for i, c in enumerate(cards)}
    out.append(f"#define PR_N_DECKS {len(DECKS)}")
    out.append("static const char *const PR_DECK_NAMES[PR_N_DECKS] = {"
               + ", ".join(c_str(n) for n, _ in DECKS) + "};")
    out.append("static const int8_t PR_DECKS[PR_N_DECKS][8] = {")
    for n, deck in DECKS:
        ids = [name_to_id[x] for x in deck]
        assert len(set(ids)) == 8
        out.append("    {" + ", ".join(card_enum[i] for i in ids) + "},  /* " + n + " */")
    out.append("};\n")
    out.append("#endif /* PR_CARD_DB_H */")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    db, cards = build(data)
    self_check(db, cards)
    text = render(db, cards, sha)
    if args.check:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != text:
            print(f"{OUT} is stale; run `make gen`", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} up to date")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(db.units)} units, {len(db.projs)} projectiles, "
          f"{len(db.areas)} areas, {len(db.buffs)} buffs, {len(cards)} cards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
