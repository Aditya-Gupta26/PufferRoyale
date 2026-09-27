#!/usr/bin/env python3
"""Evaluate a policy checkpoint against the scripted bots from both seats (SPEC §11).

    python scripts/eval.py --checkpoint experiments/pufferroyale_<run_id>        # latest model in the run
    python scripts/eval.py --checkpoint experiments/pufferroyale_<run_id>.pt --episodes 20
    python scripts/eval.py --episodes 4                     # untrained (random-init) policy baseline
    python scripts/eval.py --bot-policy heuristic           # a scripted bot in the learner's seat

For every opponent in --bots and every seat in --seats, plays --episodes matches in a
single-agent env (learner_side fixed) and reports win/draw/loss for the evaluated side. The
matches are spread evenly over the vectorised envs and each env contributes a fixed number of
full matches, so short matches are not over-sampled.
Actions are sampled from the masked policy (--greedy takes the argmax).

The env uses the settings the checkpoint was trained with (frame_skip, deploy lockout, tiebreak,
tower troops, from the run's config.json) unless --ignore-train-config; --frame-skip overrides the
decision cadence.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_policy(path, device):
    import torch
    import pufferroyale
    from pufferroyale.torch import Policy, Recurrent

    cfg, model_path = {}, path
    if path and os.path.isdir(path):
        models = sorted(glob.glob(os.path.join(path, "model_*.pt")))
        if not models:
            raise SystemExit(f"no model_*.pt in {path}")
        model_path = models[-1]
        cfg_path = os.path.join(path, "config.json")
    elif path:
        cfg_path = os.path.join(os.path.splitext(path)[0], "config.json")
    else:
        cfg_path = None
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    env = pufferroyale.Royale(num_envs=1, num_agents=1, opponent="noop")
    policy = Policy(env, **cfg.get("policy", {}))
    if cfg.get("rnn_name"):
        policy = Recurrent(env, policy, **cfg.get("rnn", {}))
    env.close()
    if model_path:
        sd = torch.load(model_path, map_location=device)
        policy.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    policy.to(device).eval()
    return policy, model_path, bool(cfg.get("rnn_name"))


TRAIN_ENV_KEYS = ("frame_skip", "deploy_lockout_ticks", "tiebreak", "tower_troop0", "tower_troop1")


def train_env_settings(path):
    """The env settings a checkpoint (file or run directory) was trained with, from config.json."""
    if not path:
        return {}
    cfg_path = os.path.join(path, "config.json") if os.path.isdir(path) else None
    if not cfg_path or not os.path.exists(cfg_path):
        for cand in (os.path.join(os.path.dirname(path), "config.json"),
                     os.path.join(os.path.splitext(path)[0], "config.json")):
            if os.path.exists(cand):
                cfg_path = cand
                break
    if not cfg_path or not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        env = json.load(f).get("env") or {}
    return {k: env[k] for k in TRAIN_ENV_KEYS if k in env}


def play(policy, recurrent, bot, seat, episodes, deck0, deck1, device, greedy, seed, bot_policy=None,
         num_envs=4, env_settings=None):
    import torch
    import pufferroyale

    # every env plays a fixed quota of FULL matches (episodes split evenly), so matches that end
    # early are not over-sampled; results past an env's quota are ignored
    num_envs = max(1, min(num_envs, episodes))
    quota = [episodes // num_envs + (1 if i < episodes % num_envs else 0) for i in range(num_envs)]
    env = pufferroyale.Royale(num_envs=num_envs, num_agents=1, opponent=bot, learner_side=seat, deck0=deck0,
                              deck1=deck1, seed=seed, log_interval=10 ** 9, **(env_settings or {}))
    obs, _ = env.reset(seed=seed)
    n = env.num_agents
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
                if greedy:
                    a = logits.argmax(-1)
                else:
                    a = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=rng).squeeze(-1)
            acts = a.cpu().numpy().astype(np.int32)
        obs, rew, term, trunc, _ = env.step(acts)
        for i in np.flatnonzero(term):
            if finished[i] < quota[i]:
                finished[i] += 1
                r = float(rew[i])
                wins += r > 0
                draws += r == 0
                losses += r < 0
            if recurrent and lstm is not None:
                lstm[0][i].zero_()
                lstm[1][i].zero_()
    env.close()
    return wins, draws, losses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="model .pt, or a run directory (latest model_*.pt)")
    ap.add_argument("--bot-policy", choices=["noop", "random", "heuristic"],
                    help="evaluate a scripted bot instead of a checkpoint")
    ap.add_argument("--bots", nargs="+", default=["noop", "random", "heuristic"])
    ap.add_argument("--seats", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--deck0", default="hog26")
    ap.add_argument("--deck1", default="hog26")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the result table here")
    ap.add_argument("--frame-skip", type=int, help="env frame_skip (default: the checkpoint's training value, else 10)")
    ap.add_argument("--ignore-train-config", action="store_true",
                    help="do not apply the checkpoint's training env settings from its config.json")
    args = ap.parse_args()
    policy, model_path, recurrent = (None, None, False)
    if not args.bot_policy:
        policy, model_path, recurrent = load_policy(args.checkpoint, args.device)
    settings = {} if (args.bot_policy or args.ignore_train_config) else train_env_settings(args.checkpoint)
    if args.frame_skip:
        settings["frame_skip"] = int(args.frame_skip)
    if settings:
        print(f"env settings: {settings}")
    who = f"bot:{args.bot_policy}" if args.bot_policy else (model_path or "random-init policy")
    print(f"evaluating {who}  ({args.episodes} episodes per opponent and seat, decks {args.deck0}/{args.deck1})")
    rows, t0 = [], time.time()
    for bot in args.bots:
        for seat in args.seats:
            w, d, l = play(policy, recurrent, bot, seat, args.episodes, args.deck0, args.deck1, args.device,
                           args.greedy, args.seed + 17 * seat, bot_policy=args.bot_policy, env_settings=settings)
            n = w + d + l
            rows.append(dict(opponent=bot, seat=seat, wins=int(w), draws=int(d), losses=int(l),
                             score=(w + 0.5 * d) / max(n, 1)))
            print(f"  vs {bot:9s} seat {seat}:  W {w:3d}  D {d:3d}  L {l:3d}   score {(w + 0.5 * d) / max(n, 1):.3f}",
                  flush=True)
    print(f"done in {time.time() - t0:.1f} s")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"policy": who, "env_settings": settings, "results": rows}, f, indent=2)


if __name__ == "__main__":
    main()
