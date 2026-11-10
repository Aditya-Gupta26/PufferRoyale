"""Check 4 (docs/ADVISOR.md): the tracker on simulated readings vs the engine's truth.

For bot-vs-bot matches, the viewer's tracker gets a simulated reading every `every` ticks. Measured:
  * opponent plays: recall / precision of the tracker's play events against the engine's real plays,
    separately for cards that leave units (visible) and spells (invisible by construction);
  * card attribution: share of tracked units whose card equals the engine entity's card;
  * opponent history at the end of each match: seen set and elixir spent, against the engine's values
    restricted to visible cards.

    python -m advisor.tests.test_tracker --matches 100 --seed 1 [--p-miss 0.1 --pos-sigma 300]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np

from pufferroyale import decks as _decks
from pufferroyale.game import Game, bot_action

from advisor import engine_db as D
from advisor.board import own_xy
from advisor.sim_perception import SimPerceptionConfig, sim_reading
from advisor.tracker import Tracker, unit_key


def _visible(card: int) -> bool:
    return bool(D.play_compositions()[card])


def run(n_matches: int, seed: int, cfg: SimPerceptionConfig, every: int = 5, max_ticks: int = 6000) -> dict:
    rng = random.Random(seed)
    deck_list = _decks.random_decks(2 * n_matches, seed)
    S = dict(plays_vis=0, plays_spell=0, found_vis=0, found_spell=0, events=0, events_correct=0,
             units=0, units_card_ok=0, units_unmatched=0, seen_exact=0, seen_jaccard=[], spent_abs_err=[],
             matches=0)
    for m in range(n_matches):
        viewer = m % 2
        opp = 1 - viewer
        g = Game(deck0=list(deck_list[2 * m]), deck1=list(deck_list[2 * m + 1]), seed=seed * 7919 + m)
        tr = Tracker(deck=g.state()["deck"][viewer])
        kinds = (rng.choice(["heuristic", "random"]), rng.choice(["heuristic", "random"]))
        truth = []                    # (t, card) opponent plays
        prev_plays = 0
        tick = 0
        while not g.state()["over"] and tick < max_ticks:
            st = g.state()
            n_new = st["plays"][opp] - prev_plays
            for k in range(n_new - 1, -1, -1):
                truth.append((st["tick"] / 20.0, st["last_played"][opp][k]))
            prev_plays = st["plays"][opp]
            bs = tr.update(sim_reading(g, viewer, cfg, rng))
            # card attribution vs the engine entity at the same place
            ents = [e for e in g.entities() if e["kind"] != "tower"]
            for u in bs.units:
                best, bd = None, 600.0
                for e in ents:
                    if (e["team"] == viewer) != u.own or unit_key(e["unit"]) != unit_key(u.unit):
                        continue
                    x, y = own_xy(viewer, e["x"], e["y"])
                    d = float(np.hypot(x - u.x, y - u.y))
                    if d <= bd:
                        best, bd = e, d
                S["units"] += 1
                if best is None:
                    S["units_unmatched"] += 1
                elif best["card_id"] == u.card:
                    S["units_card_ok"] += 1
            if tick % 10 == 0:
                for team in (0, 1):
                    a = bot_action(g, team, kinds[team])
                    if a:
                        g.play_action(team, a)
            g.tick(every)
            tick += every
        inferred = [(ev.t, ev.card) for ev in tr.events if not ev.own and ev.play]
        S["events"] += len(inferred)
        used = set()
        for (tt, c) in truth:
            vis = _visible(c)
            S["plays_vis" if vis else "plays_spell"] += 1
            for i, (ti, ci) in enumerate(inferred):
                if i not in used and ci == c and abs(ti - tt) <= 1.5:
                    used.add(i)
                    S["found_vis" if vis else "found_spell"] += 1
                    break
        S["events_correct"] += len(used)
        seen_true = {c for (_, c) in truth if _visible(c)}
        seen_est = set(tr.opp_seen)
        S["seen_exact"] += seen_true == seen_est
        S["seen_jaccard"].append(len(seen_true & seen_est) / max(1, len(seen_true | seen_est)))
        spent_true = sum(D.CARD_COSTS[c] for (_, c) in truth if _visible(c))
        S["spent_abs_err"].append(abs(spent_true - tr.opp_spent / D.ELIXIR_UNIT))
        S["matches"] += 1
    out = dict(
        matches=S["matches"],
        opp_plays_visible=S["plays_vis"], opp_plays_spell=S["plays_spell"],
        recall_visible=round(S["found_vis"] / max(1, S["plays_vis"]), 4),
        recall_spell=round(S["found_spell"] / max(1, S["plays_spell"]), 4),
        play_events=S["events"], precision=round(S["events_correct"] / max(1, S["events"]), 4),
        unit_card_accuracy=round(S["units_card_ok"] / max(1, S["units"] - S["units_unmatched"]), 4),
        units_checked=S["units"], units_unmatched=S["units_unmatched"],
        seen_set_exact=round(S["seen_exact"] / max(1, S["matches"]), 4),
        seen_jaccard_mean=round(float(np.mean(S["seen_jaccard"])), 4),
        spent_abs_err_mean_elixir=round(float(np.mean(S["spent_abs_err"])), 2),
    )
    return out


def test_tracker_runs_on_perfect_readings():
    out = run(n_matches=2, seed=3, cfg=SimPerceptionConfig())
    assert out["matches"] == 2 and out["units_checked"] > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=5, help="ticks between readings (5 = 4 readings per second)")
    ap.add_argument("--p-miss", type=float, default=0.0)
    ap.add_argument("--pos-sigma", type=float, default=0.0)
    ap.add_argument("--p-side-err", type=float, default=0.0)
    ap.add_argument("--p-type-err", type=float, default=0.0)
    ap.add_argument("--fp-per-frame", type=float, default=0.0)
    ap.add_argument("--tower-phantoms", action="store_true")
    ap.add_argument("--real", action="store_true", help="the real detector's error rates (docs/ADVISOR.md 6a): "
                    "p_miss 0.25, pos_sigma 100, p_type_err 0.05, p_side_err 0.2, tower phantoms, 0.5 other phantoms/frame")
    args = ap.parse_args(argv)
    if args.real:
        args.p_miss, args.pos_sigma, args.p_type_err, args.p_side_err = 0.25, 100.0, 0.05, 0.2
        args.tower_phantoms, args.fp_per_frame = True, 0.5
    t0 = time.time()
    cfg = SimPerceptionConfig(p_miss=args.p_miss, pos_sigma=args.pos_sigma, p_side_err=args.p_side_err,
                              p_type_err=args.p_type_err, fp_per_frame=args.fp_per_frame,
                              tower_phantoms=args.tower_phantoms)
    out = run(args.matches, args.seed, cfg, args.every)
    out["seconds"] = round(time.time() - t0, 1)
    out["config"] = vars(args)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
