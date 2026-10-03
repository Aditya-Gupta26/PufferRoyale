#!/usr/bin/env python3
"""Evaluate a policy checkpoint against the scripted bots from both seats (SPEC §11, §19.7.8).

    python scripts/eval.py --checkpoint experiments/pufferroyale_<run_id>        # latest model in the run
    python scripts/eval.py --checkpoint experiments/league/<run>/snap_40.pt --episodes 20
    python scripts/eval.py --checkpoint experiments/league/<run> --decks "hog26;giant;random:5:0"
    python scripts/eval.py --episodes 4                     # untrained (random-init) policy baseline
    python scripts/eval.py --episodes 4 --head flat --placement-grid 2   # ... of another architecture
    python scripts/eval.py --bot-policy heuristic           # a scripted bot in the learner's seat

--checkpoint takes anything pufferroyale.league.load_policy loads: a model .pt, a snapshot
snap_<epoch>.pt, 'ckpt:<path>', or a run directory (its newest model_*.pt, else learner.pt, else
its newest snapshot); recurrent or not, either head, any placement grid (the architecture is
read from the weights).

For every opponent in --bots and every seat in --seats, plays --episodes matches in a
single-agent env (learner_side fixed) and reports win/draw/loss for the evaluated side, taken
from the match OUTCOME (the env's win/draw record), never from reward signs. The matches are
spread evenly over the vectorised envs and each env contributes a fixed number of full
matches, so short matches are not over-sampled. Actions are sampled from the masked policy
(seeded; --greedy plays the card-first greedy rule of SPEC §19.11: the most likely of wait / the
4 card slots by marginal probability, then that slot's most likely tile). Every score (win 1 / draw 0.5 / loss 0) gets a Wilson 95% interval
(`ci95` = [lo, hi] in the JSON, p_hat = score, n = matches).

--decks DECKSET (the deck-set grammar of pufferroyale.decks: 'hog26;giant', 'random:N:SEED',
'file:PATH', ...) evaluates every deck as a mirror (both seats play it; overrides --deck0 /
--deck1): one row per deck, opponent and seat, plus a pooled row per opponent and seat
(deck "pooled").

The env uses the settings the checkpoint was trained with (frame_skip, deploy lockout, tiebreak,
tower troops, from the run's config.json) unless --ignore-train-config; --frame-skip overrides the
decision cadence. The placement grid is always the checkpoint's own (SPEC §19.4).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRAIN_ENV_KEYS = ("frame_skip", "deploy_lockout_ticks", "tiebreak", "tower_troop0", "tower_troop1", "placement_grid")


def load_policy(path, device, seed=0, head="conditional", placement_grid=1):
    """(policy, model_path, recurrent). `path` = anything league.load_policy loads; None = a
    random-init Policy (torch seed `seed`) with `head` and `placement_grid`."""
    import torch
    from pufferroyale.league import _obs_stub, load_policy as league_load, resolve_checkpoint_path
    from pufferroyale.royale import n_actions
    from pufferroyale.torch import Policy

    if not path:
        torch.manual_seed(int(seed))
        policy = Policy(_obs_stub(n_actions(placement_grid)), head=head, placement_grid=placement_grid)
        return policy.to(device).eval(), None, False
    try:
        model_path = resolve_checkpoint_path(path)
        policy, recurrent = league_load(model_path, device)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"eval.py: {e}")
    return policy, model_path, recurrent


def train_env_settings(path):
    """The env settings a checkpoint (file, snapshot or run directory) was trained with, from its
    run's config.json (SPEC §19.4: placement_grid included)."""
    if not path:
        return {}
    from pufferroyale.league import checkpoint_env_config
    return {k: v for k, v in (checkpoint_env_config(path) or {}).items() if k in TRAIN_ENV_KEYS}


def wilson(score, n):
    from pufferroyale.metagame import wilson_interval
    return wilson_interval(score, n)


def result_row(w, d, l, **keys):
    n = int(w + d + l)
    score = (w + 0.5 * d) / n if n else 0.0
    return dict(keys, wins=int(w), draws=int(d), losses=int(l), matches=n, score=score, ci95=wilson(score, n))


def play(policy, recurrent, bot, seat, episodes, deck0, deck1, device, greedy, seed, bot_policy=None,
         num_envs=4, env_settings=None):
    """(wins, draws, losses) of the evaluated side (team `seat`) over `episodes` full matches,
    read from each finished match's outcome."""
    import torch
    import pufferroyale
    from pufferroyale import binding
    from pufferroyale.league import select_actions

    # every env plays a fixed quota of FULL matches (episodes split evenly), so matches that end
    # early are not over-sampled; results past an env's quota are ignored
    num_envs = max(1, min(num_envs, episodes))
    quota = [episodes // num_envs + (1 if i < episodes % num_envs else 0) for i in range(num_envs)]
    env = pufferroyale.Royale(num_envs=num_envs, num_agents=1, opponent=bot, learner_side=seat, deck0=deck0,
                              deck1=deck1, seed=seed, log_interval=10 ** 9, **(env_settings or {}))
    obs, _ = env.reset(seed=seed)
    handles = env._c_env_list
    n = env.num_agents
    base = [binding.env_log_peek(h) for h in handles]       # outcome record (vec_log is never called here)
    wins = draws = losses = 0
    lstm = None
    bots = [pufferroyale.Bot(bot_policy, seed=seed * 100 + i) for i in range(n)] if bot_policy else None
    rng = torch.Generator().manual_seed(seed)
    finished = [0] * n
    while any(finished[i] < quota[i] for i in range(n)):
        if bots:
            acts = np.array([bots[i].act_env(env, seat, i) for i in range(n)], dtype=np.int32)
        else:
            with torch.no_grad():
                x = torch.as_tensor(obs, device=device)
                if recurrent:
                    state = {"lstm_h": None if lstm is None else lstm[0], "lstm_c": None if lstm is None else lstm[1]}
                    logits, _ = policy.forward_eval(x, state)
                    lstm = (state["lstm_h"], state["lstm_c"])
                else:
                    logits, _ = policy.forward_eval(x, {})
                acts = select_actions(logits, greedy, rng)        # card-first greedy or a seeded sample
        obs, rew, term, trunc, _ = env.step(acts)
        for i in np.flatnonzero(term):
            d = binding.env_log_peek(handles[i])
            w = (d["win_0"] - base[i]["win_0"], d["win_1"] - base[i]["win_1"])
            base[i] = d
            if finished[i] < quota[i]:
                finished[i] += 1
                if w[seat] > 0.5:
                    wins += 1
                elif w[1 - seat] > 0.5:
                    losses += 1
                else:
                    draws += 1
            if recurrent and lstm is not None:
                lstm[0][i].zero_()
                lstm[1][i].zero_()
    env.close()
    return wins, draws, losses


def deck_value(v):
    """A deck argument: a preset / 'random' string, or 8 comma-separated card names or ids."""
    if isinstance(v, str) and "," in v:
        return [int(c) if c.strip().lstrip("-").isdigit() else c.strip() for c in v.split(",")]
    return v


def deck_label(deck):
    """A preset name when the deck is one (as a set), else its canonical key."""
    from pufferroyale.decks import deck_key
    from pufferroyale.game import DECKS
    for name, cards in DECKS.items():
        if tuple(sorted(cards)) == tuple(sorted(deck)):
            return name
    return deck_key(deck)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="model / snapshot .pt, 'ckpt:<path>', or a run directory")
    ap.add_argument("--bot-policy", choices=["noop", "random", "heuristic"],
                    help="evaluate a scripted bot instead of a checkpoint")
    ap.add_argument("--bots", nargs="+", default=["noop", "random", "heuristic"])
    ap.add_argument("--seats", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--deck0", default="hog26")
    ap.add_argument("--deck1", default="hog26")
    ap.add_argument("--decks", help="deck set: evaluate each deck as a mirror (overrides --deck0/--deck1), "
                                    "per deck and pooled")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--greedy", action="store_true",
                    help="card-first greedy actions (SPEC §19.11) instead of seeded samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the result table here")
    ap.add_argument("--frame-skip", type=int, help="env frame_skip (default: the checkpoint's training value, else 10)")
    ap.add_argument("--ignore-train-config", action="store_true",
                    help="do not apply the checkpoint's training env settings from its config.json")
    ap.add_argument("--head", choices=["conditional", "flat"], default="conditional",
                    help="random-init baseline only (no --checkpoint): the policy head")
    ap.add_argument("--placement-grid", type=int, choices=[1, 2, 4], default=1,
                    help="random-init baseline only (no --checkpoint): the placement grid")
    args = ap.parse_args()
    if args.episodes < 1:
        raise SystemExit("eval.py: --episodes must be >= 1")
    policy, model_path, recurrent = (None, None, False)
    grid = 1
    if not args.bot_policy:
        policy, model_path, recurrent = load_policy(args.checkpoint, args.device, args.seed, args.head,
                                                    args.placement_grid)
        from pufferroyale.torch import policy_grid
        grid = policy_grid(policy)
    settings = {} if (args.bot_policy or args.ignore_train_config) else train_env_settings(args.checkpoint)
    if "placement_grid" in settings and int(settings["placement_grid"]) != grid:
        print(f"warning: config.json says placement_grid {settings['placement_grid']} but the checkpoint acts on "
              f"grid {grid}: using {grid}", file=sys.stderr)
    settings["placement_grid"] = grid
    if args.frame_skip:
        settings["frame_skip"] = int(args.frame_skip)
    decks = [(f"{args.deck0}/{args.deck1}" if args.deck0 != args.deck1 else args.deck0, deck_value(args.deck0),
              deck_value(args.deck1))]
    if args.decks:
        from pufferroyale.decks import deck_entries
        try:
            entries = deck_entries(args.decks)
        except (ValueError, OSError) as e:
            raise SystemExit(f"eval.py: --decks: {e}")
        if not entries:
            raise SystemExit("eval.py: --decks is empty")
        decks = [(deck_label(d), list(order), list(order)) for d, _, order in entries]
    print(f"env settings: {settings}")
    who = f"bot:{args.bot_policy}" if args.bot_policy else (model_path or "random-init policy")
    print(f"evaluating {who}  ({args.episodes} episodes per opponent and seat, decks "
          f"{', '.join(d[0] for d in decks)})")
    rows, t0 = [], time.time()
    pooled = {}
    for label, d0, d1 in decks:
        for bot in args.bots:
            for seat in args.seats:
                w, d, l = play(policy, recurrent, bot, seat, args.episodes, d0, d1, args.device,
                               args.greedy, args.seed + 17 * seat, bot_policy=args.bot_policy, env_settings=settings)
                row = result_row(w, d, l, opponent=bot, seat=seat)
                if args.decks:
                    row = dict({"deck": label}, **row)
                    acc = pooled.setdefault((bot, seat), [0, 0, 0])
                    acc[0], acc[1], acc[2] = acc[0] + w, acc[1] + d, acc[2] + l
                rows.append(row)
                lo, hi = row["ci95"]
                print(f"  {('[' + label + '] ') if args.decks else ''}vs {bot:9s} seat {seat}:  W {w:3d}  D {d:3d}  "
                      f"L {l:3d}   score {row['score']:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)
    for (bot, seat), (w, d, l) in pooled.items():        # SPEC §19.9.6: one pooled row per opponent and seat
        row = result_row(w, d, l, deck="pooled", opponent=bot, seat=seat)
        rows.append(row)
        print(f"  [pooled] vs {bot:9s} seat {seat}:  W {w:3d}  D {d:3d}  L {l:3d}   score {row['score']:.3f}  "
              f"[{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]", flush=True)
    print(f"done in {time.time() - t0:.1f} s")
    if args.json:
        from pufferroyale.league import json_safe
        out = {"policy": who, "env_settings": settings, "greedy": bool(args.greedy), "episodes": int(args.episodes),
               "decks": [d[0] for d in decks], "results": rows}
        with open(args.json, "w") as f:
            json.dump(json_safe(out), f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
