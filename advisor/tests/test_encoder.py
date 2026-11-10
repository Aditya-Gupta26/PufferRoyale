"""The advisor's encoder must reproduce the engine's observation exactly.

For simulated bot-vs-bot matches (random decks, random tower troops, heuristic and random bots), at every
decision step and for both seats: BoardState from perfect perception (board.from_game) -> encoder.encode
must equal Game.obs(team) element for element. Run more matches from the command line:

    python -m advisor.tests.test_encoder --matches 200 --seed 1
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from pufferroyale import decks as _decks
from pufferroyale.game import TOWER_TROOPS, Game, bot_action

from advisor import board, encoder
from advisor import engine_db as D

SECTIONS = [("spatial:" + n, encoder.SPATIAL_OFFSET + i * encoder.PLANE, encoder.SPATIAL_OFFSET + (i + 1) * encoder.PLANE)
            for n, i in sorted(encoder.CH.items(), key=lambda kv: kv[1])]
SECTIONS.append(("entities", encoder.ENTITY_OFFSET, encoder.SCALAR_OFFSET))
SECTIONS += [("scalar:" + n, encoder.SCALAR_OFFSET + off, encoder.SCALAR_OFFSET + off + ln)
             for n, (off, ln) in sorted(encoder.SC.items(), key=lambda kv: kv[1][0])]
SECTIONS.append(("mask", encoder.MASK_OFFSET, encoder.OBS_SIZE))


def compare(a: np.ndarray, b: np.ndarray) -> list:
    """Names of the sections where the two observations differ (exact float32 equality)."""
    return [name for name, lo, hi in SECTIONS if not np.array_equal(a[lo:hi], b[lo:hi])]


def run(n_matches: int, seed: int, max_steps: int = 600, verbose: bool = False) -> dict:
    rng = random.Random(seed)
    deck_list = _decks.random_decks(2 * n_matches, seed)
    stats = {"states": 0, "mismatch_states": 0, "sections": {}, "first": None, "units_max": 0,
             "spells_seen": 0, "dead_towers_seen": 0}
    for m in range(n_matches):
        tt = (rng.choice(TOWER_TROOPS), rng.choice(TOWER_TROOPS))
        g = Game(deck0=list(deck_list[2 * m]), deck1=list(deck_list[2 * m + 1]), seed=seed * 100003 + m,
                 tower_troop0=tt[0], tower_troop1=tt[1])
        kinds = (rng.choice(["heuristic", "heuristic", "random"]), rng.choice(["heuristic", "random"]))
        for step in range(max_steps):
            if g.state()["over"]:
                break
            for team in (0, 1):
                bs = board.from_game(g, team)
                mine = encoder.encode(bs)
                ref = g.obs(team)
                stats["states"] += 1
                stats["units_max"] = max(stats["units_max"], len(bs.units))
                stats["spells_seen"] += bool(bs.spells)
                stats["dead_towers_seen"] += any(not t.alive for t in bs.own_towers + bs.enemy_towers)
                bad = compare(mine, ref)
                if bad:
                    stats["mismatch_states"] += 1
                    for s in bad:
                        stats["sections"][s] = stats["sections"].get(s, 0) + 1
                    if stats["first"] is None:
                        idx = int(np.flatnonzero(mine != ref)[0])
                        stats["first"] = dict(match=m, step=step, team=team, tick=bs.tick, sections=bad,
                                              index=idx, mine=float(mine[idx]), ref=float(ref[idx]))
            for team in (0, 1):
                a = bot_action(g, team, kinds[team])
                if a:
                    g.play_action(team, a)
            g.tick(10)
        if verbose:
            print(f"match {m}: tick {g.state()['tick']} states {stats['states']} mismatches {stats['mismatch_states']}",
                  flush=True)
    return stats


def test_units_db_matches_engine():
    """Every card's unit as parsed from the C header equals the engine's card_info."""
    units, proj_radius = D.tables()
    for c in range(D.N_CARDS):
        ci = D.card_info(c)
        if not ci["unit"]:
            continue
        u = units[ci["unit"]]
        assert (u.hp, u.range, u.flying, u.only_buildings, u.area_radius, u.footprint_tiles, u.shield) == (
            ci["hitpoints"], ci["range_milli"], ci["flying_height"] > 0, ci["target_only_buildings"],
            ci["area_damage_radius_milli"], ci["footprint_tiles"], ci["shield_hitpoints"]), ci["name"]
        assert (u.projectile >= 0) == (ci["projectile"] is not None), ci["name"]
        if ci["projectile"]:
            assert proj_radius[u.projectile] == ci["projectile"]["radius"], ci["name"]


def _midgame(seed: int = 7, steps: int = 150) -> Game:
    """A simulated match after `steps` decision steps (units, buildings and spells on the board)."""
    d = _decks.random_decks(2, seed)
    g = Game(deck0=list(d[0]), deck1=list(d[1]), seed=seed)
    for _ in range(steps):
        for team in (0, 1):
            a = bot_action(g, team, "heuristic")
            if a:
                g.play_action(team, a)
        g.tick(10)
    return g


def test_compare_detects_errors():
    """Mutation check: each deliberate error in the BoardState must change the encoded observation."""
    import copy
    g = _midgame()
    base = board.from_game(g, 0)
    assert base.units, "need units on the board"
    assert not compare(encoder.encode(base), g.obs(0))
    mutations = {
        "unit moved one tile": lambda b: setattr(b.units[0], "x", b.units[0].x + 1000 if b.units[0].x < 17000 else b.units[0].x - 1000),
        "unit hp changed": lambda b: setattr(b.units[0], "hp", max(1, b.units[0].hp // 2)),
        "unit dropped": lambda b: b.units.pop(0),
        "unit card changed": lambda b: setattr(b.units[0], "card", (b.units[0].card + 1) % D.N_CARDS),
        "elixir changed": lambda b: setattr(b, "elixir", b.elixir + D.ELIXIR_UNIT if b.elixir < 20000 else b.elixir - D.ELIXIR_UNIT),
        "tick changed": lambda b: setattr(b, "tick", b.tick + 1),
        "opp spent changed": lambda b: setattr(b, "opp_spent", b.opp_spent + D.ELIXIR_UNIT),
        "opp last4 changed": lambda b: b.opp_last4.__setitem__(0, (b.opp_last4[0] + 1) % D.N_CARDS),
        "enemy tower damaged": lambda b: setattr(b.enemy_towers[1], "hp", max(1, b.enemy_towers[1].hp - 500)),
        "hand slot unknown": lambda b: b.hand.__setitem__(0, -1),
    }
    for name, mutate in mutations.items():
        b = copy.deepcopy(base)
        mutate(b)
        assert compare(encoder.encode(b), g.obs(0)), f"mutation not detected: {name}"


def test_encoder_exact_small():
    s = run(n_matches=3, seed=12345)
    assert s["states"] > 0
    assert s["mismatch_states"] == 0, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    s = run(args.matches, args.seed, verbose=args.verbose)
    s["seconds"] = round(time.time() - t0, 1)
    print(s)
    return 0 if s["mismatch_states"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
