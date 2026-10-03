#!/usr/bin/env python3
"""Within-run transitivity of a league run's snapshots (SPEC §19.7.7; TRAINING_PLAN §9 E4).

    python scripts/transitivity.py --run experiments/league/<run> --snapshots 10 --matches 100 --out tr.json

Takes --snapshots evenly spaced snap_<epoch>.pt of the run (by epoch; all of them when there are
fewer), plays a round robin between them on the run's deck0 mirror -- --matches per pair, half
per seat, SAMPLED actions, each snapshot at its training settings (config.json) -- and reports
the payoff matrix, Elo, the meta-game Nash mixture and pufferroyale.metagame.transitivity:
later_beats_earlier (fraction of pairs where the later snapshot scores > 0.5 against the earlier
one; 1.0 = a transitive ladder) and cyclic_triads (triples forming a 3-cycle with every edge
> 0.5 + --margin; the signature of non-transitive dynamics). Strict JSON: the last stdout line,
and --out.
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="league run directory (holding snap_<epoch>.pt)")
    ap.add_argument("--snapshots", type=int, default=10, help="evenly spaced snapshots to compare (default 10)")
    ap.add_argument("--matches", type=int, default=100, help="matches per pair, half per seat (default 100)")
    ap.add_argument("--margin", type=float, default=0.05, help="cyclic-triad edge margin over 0.5 (default 0.05)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the result JSON here too")
    return ap


def pick_snapshots(run, k):
    """Evenly spaced snap_*.pt of `run` by epoch (all when there are at most k)."""
    import numpy as np
    from pufferroyale.league import _snap_epoch
    snaps = sorted(glob.glob(os.path.join(run, "snap_*.pt")), key=_snap_epoch)
    if len(snaps) <= k:
        return snaps
    idx = sorted({int(round(x)) for x in np.linspace(0, len(snaps) - 1, k)})
    return [snaps[i] for i in idx]


def run_deck(run):
    """The run's deck0 (config.json env, else the league args), default hog26."""
    path = os.path.join(run, "config.json")
    if os.path.isfile(path):
        with open(path) as f:
            cfg = json.load(f)
        for sec in ("env", "league"):
            d = (cfg.get(sec) or {}).get("deck0")
            if d:
                return d
    return "hog26"


def main():
    args = make_parser().parse_args()
    import numpy as np
    from pufferroyale import metagame as mg
    from pufferroyale.league import _snap_epoch, json_safe

    if args.matches < 1:
        raise SystemExit("transitivity.py: --matches must be >= 1")
    if args.snapshots < 2:
        raise SystemExit("transitivity.py: --snapshots must be >= 2")
    run = os.path.abspath(args.run)
    snaps = pick_snapshots(run, args.snapshots)
    if len(snaps) < 2:
        raise SystemExit(f"transitivity.py: {run} has {len(snaps)} snap_*.pt; need at least 2")
    deck = run_deck(run)
    n = len(snaps)
    t0 = time.time()
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            for m in range(args.matches):
                s = mg.match_seed(args.seed, i, j, m)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", mg.MatchSettingsWarning)
                    if m % 2 == 0:
                        r = mg.play_match(snaps[i], snaps[j], deck, deck, s, greedy=False, device=args.device)
                    else:
                        r = -mg.play_match(snaps[j], snaps[i], deck, deck, s, greedy=False, device=args.device)
                results.append((i, j, int(r)))
        print(f"[transitivity] {os.path.basename(snaps[i])} done ({i + 1}/{n})", flush=True)
    P, G = mg.payoff_matrix(results, n, return_counts=True)
    ratings, info = mg.elo(P, G, return_info=True)
    x, value = mg.metagame_nash(P)
    tr = mg.transitivity(P, margin=args.margin)
    out = {"run": run, "snapshots": [os.path.basename(p) for p in snaps], "epochs": [_snap_epoch(p) for p in snaps],
           "deck": deck, "matches": int(args.matches), "greedy": False, "seed": int(args.seed),
           "margin": float(args.margin), "payoff": P.tolist(), "games": G.tolist(), "elo": [float(r) for r in ratings],
           "elo_regularised": bool(info["regularised"]), "nash": [float(v) for v in x], "nash_value": float(value),
           "transitivity": tr, "elapsed_s": round(time.time() - t0, 2)}
    print(f"[transitivity] later beats earlier {tr['later_beats_earlier']:.3f} over {tr['pairs']} pairs; "
          f"{tr['cyclic_triads']} cyclic triads of {tr['triads']}", flush=True)
    line = json.dumps(json_safe(out), allow_nan=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
