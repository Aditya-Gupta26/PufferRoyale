"""Builder tests: 180-degree seat symmetry, source rules (no floats in simulation code),
and deterministic code generation."""
import os
import re
import shutil
import subprocess
import sys

import pytest

from pufferroyale import Game, PlayError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSRC = os.path.join(ROOT, "pufferroyale", "csrc")
SIM_HEADERS = ["pr_math.h", "pr_defs.h", "pr_types.h", "pr_card_db.h", "pr_arena_db.h", "pr_arena.h",
               "pr_entity.h", "pr_rules.h", "pr_target.h", "pr_combat.h", "pr_path.h", "pr_move.h",
               "pr_spell.h", "pr_engine.h", "pr_bots.h"]

CARDS = list(range(64))

# a legal own-frame tile per card (troops own half / pocket-free, buildings odd F on the own half,
# the Miner and spells anywhere they may go)
TILES = {
    0: (5, 20), 1: (12, 24), 2: (8, 24), 3: (3, 17), 4: (14, 18), 5: (9, 22), 6: (2, 19), 7: (7, 21), 8: (9, 20),
    9: (16, 25), 10: (4, 18), 11: (10, 19), 12: (13, 23), 13: (6, 26), 14: (9, 19), 15: (11, 20), 16: (3, 6),
    17: (14, 7), 18: (9, 16), 19: (4, 18), 20: (15, 6),
    21: (9, 22), 22: (5, 20), 23: (12, 21), 24: (9, 24), 25: (4, 19), 26: (12, 26), 27: (6, 18), 28: (11, 22),
    29: (3, 20), 30: (14, 23), 31: (8, 26), 32: (10, 25), 33: (7, 22), 34: (15, 19), 35: (1, 24), 36: (5, 29),
    37: (13, 29), 38: (4, 18), 39: (9, 20), 40: (11, 27), 41: (9, 19), 42: (6, 22), 43: (12, 19), 44: (8, 21),
    45: (14, 18), 46: (3, 22), 47: (13, 12), 48: (5, 19), 49: (12, 18), 50: (9, 24), 51: (13, 21), 52: (14, 6),
    53: (3, 7), 54: (9, 10), 55: (14, 7), 56: (3, 6), 57: (9, 9), 58: (7, 20), 59: (10, 21), 60: (4, 21),
    61: (16, 24), 62: (13, 17), 63: (6, 25),
}


def rot(x, y):
    return 18000 - x, 32000 - y


def fingerprint(g, flip):
    """Units, towers and effects in team 0's frame (the mirror game's teams swapped and rotated)."""
    out = []
    for e in g.entities():
        team, x, y = e["team"], e["x"], e["y"]
        if flip:
            team, (x, y) = 1 - team, rot(x, y)
        out.append((team, e["unit"], x, y, e["hp"], e["max_hp"], e["deploying"], e["stunned"], e["slowed"],
                    e["hidden"], e["level"], e["dash_state"]))
    fx = []
    for f in g.effects():
        team, x, y = f["team"], f["x"], f["y"]
        if flip:
            team, (x, y) = 1 - team, rot(x, y)
        fx.append((f["kind"], team, x, y, f["timer_ms"], f["remaining"]))
    s = g.state()
    el = s["elixir"][::-1] if flip else s["elixir"]
    return sorted(out), sorted(fx), el


def _mirror_run(plays, ticks=700, alternate=False, troops=("princess", "princess")):
    """plays: [(tick, team, card, (tx, ty))] in game A (own-frame tiles). Game B plays the same with the
    teams swapped (and, with `alternate`, the other team first), so B must evolve as the 180-degree
    rotation of A."""
    kw = dict(deck0="hog26", deck1="hog26", seed=1, deploy_lockout_ticks=0, alternate_first=alternate)
    a = Game(tower_troop0=troops[0], tower_troop1=troops[1], **kw)
    b = Game(tower_troop0=troops[1], tower_troop1=troops[0], **kw)
    if alternate:
        a._set_first_team(0)
        b._set_first_team(1)
    by_tick = {}
    for p in plays:
        by_tick.setdefault(p[0], []).append(p)
    for t in range(ticks):
        for (_, team, card, (tx, ty)) in by_tick.get(t, []):
            for g, tm in ((a, team), (b, 1 - team)):
                hand = [card] + [c for c in CARDS if c != card][:7]
                g.set_hand(tm, hand)
                g.set_elixir(tm, 28000)
            # an earlier building of the same team may cover the tile: the nearest legal own-frame
            # tile (legality is itself rotation-symmetric, so both games agree on it)
            m = a.legal_mask(team)
            if not m[1 + ty * 18 + tx]:
                tx, ty = min(((abs(x - tx) + abs(y - ty), x, y) for y in range(32) for x in range(18)
                              if m[1 + y * 18 + x]))[1:]
            for g, tm in ((a, team), (b, 1 - team)):
                assert g.play_tile(tm, 0, tx, ty) == PlayError.OK, (card, tm, (tx, ty))
        a.tick()
        b.tick()
        if t % 5 == 0 or t == ticks - 1:
            fa, fb = fingerprint(a, False), fingerprint(b, True)
            assert fa == fb, f"diverged at tick {t}"
        if a.state()["over"]:
            break
    return a, b


@pytest.mark.parametrize("card", CARDS)
def test_single_seat_scenario_is_rotation_symmetric(card):
    """Team 0 plays a card; in the mirror game team 1 plays it at the same own-frame tile.
    The two matches must stay exact rotations of each other (all 64 cards, SPEC §13.13)."""
    _mirror_run([(0, 0, card, TILES[card])])


# two-seat interactions: every card meets another one on the same tick (alternating first team so
# the mirror is exact including creation order), 2 pairings per card
PAIRS = [(c, (c * 7 + 3) % 64) for c in CARDS if (c * 7 + 3) % 64 != c]


@pytest.mark.parametrize("x,y", PAIRS)
def test_two_seat_interaction_is_rotation_symmetric(x, y):
    _mirror_run([(0, 0, x, TILES[x]), (0, 1, y, TILES[y]), (60, 0, y, TILES[y]), (60, 1, x, TILES[x])],
                ticks=600, alternate=True)


@pytest.mark.parametrize("musk_tile", [(15, 21), (2, 21)])
def test_royal_chef_serving_order_is_rotation_symmetric(musk_tile):
    """SPEC §18.2: a Golem within 7500 of both Chef towers and a Musketeer near one of them at the first
    pancake (tick 560); own-left serves first for both seats, so the mirror stays exact, and the
    Musketeer is levelled only when it stands by the own-right tower."""
    a, b = _mirror_run([(540, 0, 36, (8, 21)), (545, 0, 2, musk_tile)], ticks=580,
                       troops=("royal_chef", "royal_chef"))
    lv = {e["unit"]: e["level"] for e in a.entities() if e["team"] == 0 and e["kind"] == "troop"}
    assert lv == {"Golem": 12, "Musketeer": 12 if musk_tile[0] > 9 else 11}, lv


@pytest.mark.parametrize("troop", ["cannoneer", "dagger_duchess", "royal_chef"])
def test_tower_troops_are_rotation_symmetric(troop):
    """Team 0 owns the tower troop: its towers shoot a Hog Rider and Minions, and the Chef levels up
    a Golem and a Musketeer standing near them."""
    _mirror_run([(0, 1, 4, (3, 17)), (0, 1, 5, (14, 17)), (400, 0, 36, (5, 29)), (420, 0, 2, (12, 26)),
                 (500, 1, 4, (14, 17))], ticks=900, troops=(troop, "princess"))


def test_no_floating_point_in_simulation_headers():
    pat = re.compile(r"\b(float|double)\b")
    for h in SIM_HEADERS:
        text = open(os.path.join(CSRC, h)).read()
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r'"(\\.|[^"\\])*"', '""', text)
        assert not pat.search(text), f"{h} uses floating point"


def test_codegen_is_deterministic_and_committed(tmp_path):
    for script in ("gen_card_db.py", "gen_arena.py"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", script), "--check"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
    # running twice into a scratch copy gives byte-identical files
    work = tmp_path / "proj"
    for d in ("tools", "data"):
        shutil.copytree(os.path.join(ROOT, d), work / d)
    (work / "pufferroyale" / "csrc").mkdir(parents=True)
    outs = []
    for _ in range(2):
        for script in ("gen_card_db.py", "gen_arena.py"):
            subprocess.run([sys.executable, str(work / "tools" / script)], check=True, cwd=work,
                           capture_output=True)
        outs.append({f: (work / "pufferroyale" / "csrc" / f).read_bytes()
                     for f in ("pr_card_db.h", "pr_arena_db.h")})
    assert outs[0] == outs[1]
    for f, data in outs[0].items():
        assert data == open(os.path.join(CSRC, f), "rb").read()
