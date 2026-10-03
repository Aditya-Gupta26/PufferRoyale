#!/usr/bin/env python3
"""Round-robin tournament + empirical meta-game (SPEC §15.4, §15.7.11).

    python scripts/tournament.py --agents bot:noop bot:random bot:heuristic \\
        experiments/league/<run>/snap_40.pt --decks hog26 giant --matches 4 --out t.json

Agents: bot specs (bot:noop, bot:random, bot:heuristic) and/or checkpoints (a PuffeRL model .pt,
"ckpt:<path>", or a run directory = its latest model_*.pt).

Schedule: for every pair of agents (i < j) and every deck pairing -- agent i with deck X, agent
j with deck Y, over all (X, Y) in --decks x --decks (--deck-mode all, default) or X = Y
(--deck-mode mirror) -- play --matches full matches, agent i in seat 0 for the even-numbered
ones and in seat 1 for the others (half per seat). Every match has its own deterministic seed
derived from --seed. Policies sample their actions from the masked policy (SPEC §19.11; seeded per
match, so the tournament is deterministic given --seed); --greedy plays the card-first greedy rule
instead (the most likely of wait / the 4 card slots by marginal probability, then that slot's most
likely tile). --sample is still accepted and does nothing (sampling is the default).

Output: the payoff matrix P[i][j] (agent i's mean score vs j, win 1 / draw 0.5 / loss 0),
maximum-likelihood Bradley-Terry Elo ratings (mean 1500) and the Nash equilibrium mixture of the
symmetric meta-game A = P - 0.5 (the population mixture no agent beats on average). --out writes
JSON with keys agents, payoff, elo, nash (+ games, nash_value, decks, results, ...).

Decks: presets, 'random', or 8 comma-separated card names / ids; repeated decks are played once.
Placement grid (SPEC §19.4): every checkpoint plays through its own grid (its coarse actions are
mapped to fine plays by Game.coarse_to_fine), so checkpoints of different grids can meet.
Match settings: each checkpoint decides at the frame_skip it was trained with and the game uses
the checkpoints' deploy lockout / tiebreak / tower troops (from their runs' config.json), with a
warning whenever a checkpoint plays under other settings; --frame-skip N (all agents) and
--ignore-train-config (defaults: frame_skip 10, lockout 90, absolute tiebreak, Princess towers)
override that.
"""
import argparse
import json
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", nargs="+", required=True, help="bot specs and/or checkpoint paths")
    ap.add_argument("--decks", nargs="+", default=["hog26"], help="deck presets (default hog26)")
    ap.add_argument("--matches", type=int, default=10, help="matches per agent pair and deck pairing (default 10)")
    ap.add_argument("--out", help="write the result JSON here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deck-mode", choices=["all", "mirror"], default="all",
                    help="all: every (X, Y) deck pairing; mirror: both agents use the same deck")
    ap.add_argument("--greedy", action="store_true",
                    help="policies play the card-first greedy rule (SPEC §19.11) instead of sampling")
    ap.add_argument("--sample", action="store_true",
                    help="accepted for compatibility; does nothing (policies sample by default)")
    ap.add_argument("--quiet", action="store_true", help="no per-pair progress lines")
    ap.add_argument("--frame-skip", type=int, help="decision cadence (ticks) for EVERY agent (default: each "
                                                   "checkpoint's training frame_skip, bots 10)")
    ap.add_argument("--ignore-train-config", action="store_true",
                    help="do not read the checkpoints' training env settings (use the defaults)")
    return ap


def deck_value(v):
    """A deck argument: a preset / 'random' string, or 8 comma-separated card names or ids."""
    if isinstance(v, str) and "," in v:
        return [int(c) if c.strip().lstrip("-").isdigit() else c.strip() for c in v.split(",")]
    return v


def label(agent, width=18):
    s = agent
    if not s.startswith("bot:"):
        p = s[5:] if s.startswith("ckpt:") else s
        s = os.path.basename(os.path.normpath(p))
    return s if len(s) <= width else "..." + s[-(width - 3):]


def main():
    args = make_parser().parse_args()
    import numpy as np
    from pufferroyale import metagame as mg

    if args.matches < 1:
        raise SystemExit("tournament.py: --matches must be >= 1")
    from pufferroyale.league import json_safe
    agents = list(args.agents)
    decks = []
    for d in args.decks:                                   # identical deck pairings are played once
        if d not in decks:
            decks.append(d)
    if len(decks) < len(args.decks):
        print(f"note: repeated decks dropped: {args.decks} -> {decks}", file=sys.stderr)
    deck_args = [deck_value(d) for d in decks]
    try:
        from pufferroyale.game import _deck_arg
        for d in deck_args:
            _deck_arg(d)
        for ag in agents:                                  # fail early on a bad spec / path
            mg.resolve_agent(ag, args.device)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"tournament.py: {e}")
    use_cfg = not args.ignore_train_config
    over = {"frame_skip": args.frame_skip} if args.frame_skip else {}
    ms = mg.match_settings(agents, over, use_cfg)
    print(f"match settings: {ms['game']}; decision cadence per agent (ticks): "
          f"{dict(zip([label(a) for a in agents], ms['frame_skip']))}")
    for w in ms["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    n = len(agents)
    if args.deck_mode == "all":
        pairings = [(x, y) for x in range(len(decks)) for y in range(len(decks))]
    else:
        pairings = [(x, x) for x in range(len(decks))]
    greedy = bool(args.greedy)                     # --sample is a no-op (SPEC §19.11)
    t0 = time.time()
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            tally = [0, 0, 0]
            for (x, y) in pairings:
                for m in range(args.matches):
                    s = mg.match_seed(args.seed, i, j, x, y, m)
                    kw = dict(greedy=greedy, device=args.device, match_config=dict(ms["game"], **over),
                              use_train_config=use_cfg)
                    with warnings.catch_warnings():            # already reported once above
                        warnings.simplefilter("ignore", mg.MatchSettingsWarning)
                        if m % 2 == 0:
                            r = mg.play_match(agents[i], agents[j], deck_args[x], deck_args[y], s, **kw)
                        else:
                            r = -mg.play_match(agents[j], agents[i], deck_args[y], deck_args[x], s, **kw)
                    results.append((i, j, int(r)))
                    tally[0 if r > 0 else (1 if r == 0 else 2)] += 1
            if not args.quiet:
                print(f"  {label(agents[i]):>18s} vs {label(agents[j]):<18s}  W {tally[0]:3d}  D {tally[1]:3d}  "
                      f"L {tally[2]:3d}", flush=True)
    P, G = mg.payoff_matrix(results, n, return_counts=True)
    ratings, info = mg.elo(P, G, return_info=True)
    x, value = mg.metagame_nash(P)

    names = [label(a) for a in agents]
    w = max(8, max(len(s) for s in names))
    print()
    print("payoff matrix P[i][j] = score of row agent vs column agent")
    print(" " * (w + 2) + " ".join(f"{k:>7d}" for k in range(n)))
    for i in range(n):
        print(f"{names[i]:>{w}s} {i:d} " + " ".join(f"{P[i, j]:7.3f}" for j in range(n)))
    print()
    print(f"{'agent':>{w}s}   {'elo':>7s}  {'nash':>6s}  {'mean score':>10s}")
    order = np.argsort(-ratings, kind="stable")
    for i in order:
        others = [P[i, j] for j in range(n) if j != i]
        mean = float(np.mean(others)) if others else 0.5
        print(f"{names[i]:>{w}s}   {ratings[i]:7.1f}  {x[i]:6.3f}  {mean:10.3f}")
    note = " (ML diverges for unbeaten agents: one virtual draw per pair added)" if info["regularised"] else ""
    print(f"\n{len(results)} matches ({'card-first greedy' if greedy else 'sampled'} policy actions) in "
          f"{time.time() - t0:.1f} s; meta-game value {value:+.2e}{note}")

    out = {"agents": agents, "payoff": P.tolist(), "elo": [float(r) for r in ratings], "nash": [float(v) for v in x],
           "nash_value": float(value), "games": G.tolist(), "elo_regularised": bool(info["regularised"]),
           "decks": decks, "deck_mode": args.deck_mode, "matches": int(args.matches), "greedy": greedy,
           "match_settings": {"game": ms["game"], "frame_skip": ms["frame_skip"]},
           "seed": int(args.seed), "results": [list(r) for r in results],
           "elapsed_s": round(time.time() - t0, 2)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(json_safe(out), f, indent=1, allow_nan=False)      # strict JSON (SPEC §18.8f)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
