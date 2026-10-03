#!/usr/bin/env python3
"""Watch a match: scripted bot vs bot, or a policy checkpoint vs a bot (SPEC §9 rendering).

    python scripts/watch.py                               # heuristic vs heuristic, text board
    python scripts/watch.py --p0 heuristic --p1 random --fps 20
    python scripts/watch.py --checkpoint experiments/pufferroyale_<run_id> --p1 heuristic
    python scripts/watch.py --raylib                      # window (needs a PR_RAYLIB=1 build)
    python scripts/watch.py --no-clear --every 100 --fps 0          # log-friendly
    python scripts/watch.py --checkpoint <run dir> --greedy         # card-first greedy, not sampled

A checkpoint samples its actions from the masked policy (SPEC §19.11), from a generator seeded by
--seed, so a given --seed replays the same match; --greedy plays the card-first greedy rule instead
(the most likely of wait / the 4 card slots by marginal probability, then that slot's most likely
tile; pufferroyale.league.greedy_actions). The last line reports the result and team 0's plays.

Text mode drives a pufferroyale.Game directly (the policy reads Game.obs, the same encoder
the env uses), so the final board is shown before anything resets. A checkpoint of placement grid
g > 1 plays through its own grid (Game.coarse_to_fine; the env's grid in --raylib mode). --raylib runs the env
with render_mode='human': a raylib window when the extension was built with
`PR_RAYLIB=1 make build`, otherwise the text board is printed.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


class PolicyPlayer:
    """A checkpoint (any file / snapshot / run directory league.load_policy loads) sampling its
    actions with a generator seeded by `seed` (default), or playing the card-first greedy rule
    (greedy=True; SPEC §19.11). act() returns an action of the policy's own placement grid
    (SPEC §19.4); fine() maps it to the fine play a Game accepts."""

    def __init__(self, checkpoint, greedy=False, seed=0):
        import importlib
        import torch
        from pufferroyale.torch import policy_grid

        ev = importlib.import_module("eval")
        self.policy, self.path, self.recurrent = ev.load_policy(checkpoint, "cpu")
        self.grid = policy_grid(self.policy)
        self.state = {"lstm_h": None, "lstm_c": None}
        self.greedy = bool(greedy)
        self.gen = torch.Generator().manual_seed((int(seed) * 2) & ((1 << 62) - 1))   # team 0's stream

    def act(self, obs_row):
        import torch
        from pufferroyale.league import select_actions

        with torch.no_grad():
            st = self.state if self.recurrent else {}
            logits, _ = self.policy.forward_eval(torch.as_tensor(obs_row[None, :]), st)
            return int(select_actions(logits, self.greedy, self.gen)[0])

    def fine(self, game, team, action):
        return game.coarse_to_fine(team, action, self.grid) if (self.grid > 1 and action) else action


def watch_text(args):
    import pufferroyale

    g = pufferroyale.Game(deck0=args.deck0, deck1=args.deck1, seed=args.seed)
    players = [pufferroyale.Bot(args.p0, seed=args.seed * 2), pufferroyale.Bot(args.p1, seed=args.seed * 2 + 1)]
    policy = PolicyPlayer(args.checkpoint, args.greedy, args.seed) if args.checkpoint else None
    step = plays = 0
    while not g.state()["over"] and step < args.max_steps:
        for team in (0, 1):
            a = policy.fine(g, 0, policy.act(g.obs(0))) if (policy and team == 0) else players[team].act(g, team)
            err = g.play_action(team, a)
            plays += int(team == 0 and a != 0 and err == pufferroyale.PlayError.OK)
        g.tick(args.frame_skip)
        step += 1
        if step % args.every == 0:
            if not args.no_clear:
                sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(g.ansi() + "\n")
            sys.stdout.flush()
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
    s = g.state()
    if not args.no_clear:
        sys.stdout.write("\x1b[2J\x1b[H")
    print(g.ansi())
    who = "policy (team 0)" if policy else f"team 0 ({args.p0})"
    mode = f", {'card-first greedy' if policy.greedy else 'sampled'} actions" if policy else ""
    print(f"after {step} steps ({s['tick']} ticks): result {s['result']} ({s['end_reason']}), "
          f"crowns {s['crowns'][0]}-{s['crowns'][1]}; {who} "
          f"{'won' if s['result'][0] > 0 else 'lost' if s['result'][0] < 0 else 'drew' if s['over'] else 'unfinished'} "
          f"({plays} plays by team 0{mode})")


def watch_raylib(args):
    import pufferroyale
    from pufferroyale import royale as R

    if not R.RAYLIB_AVAILABLE:
        print("note: extension built without raylib (rebuild with `PR_RAYLIB=1 make build`); printing text")
    policy = PolicyPlayer(args.checkpoint, args.greedy, args.seed) if args.checkpoint else None
    if policy:
        env = pufferroyale.Royale(num_envs=1, num_agents=1, opponent=args.p1, learner_side=0, deck0=args.deck0,
                                  deck1=args.deck1, seed=args.seed, render_mode="human", frame_skip=args.frame_skip,
                                  log_interval=10 ** 9, placement_grid=policy.grid)
    else:
        env = pufferroyale.Royale(num_envs=1, num_agents=2, deck0=args.deck0, deck1=args.deck1, seed=args.seed,
                                  render_mode="human", frame_skip=args.frame_skip, log_interval=10 ** 9)
        bots = [pufferroyale.Bot(args.p0, seed=args.seed * 2), pufferroyale.Bot(args.p1, seed=args.seed * 2 + 1)]
    obs, _ = env.reset(seed=args.seed)
    for step in range(1, args.max_steps + 1):
        if policy:
            acts = np.array([policy.act(obs[0])], dtype=np.int32)
        else:
            acts = np.array([bots[t].act_env(env, t) for t in (0, 1)], dtype=np.int32)
        obs, rew, term, trunc, _ = env.step(acts)
        if step % args.every == 0:
            env.render()
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
        if term.any():
            r = float(rew[0])
            print(f"match over after {step} steps: {'won' if r > 0 else 'lost' if r < 0 else 'drew'} (row 0)")
            break
    env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", default="heuristic", choices=["noop", "random", "heuristic"], help="team 0 bot")
    ap.add_argument("--p1", default="heuristic", choices=["noop", "random", "heuristic"], help="team 1 bot")
    ap.add_argument("--checkpoint", help="a policy plays team 0 (model .pt or run dir) against --p1")
    ap.add_argument("--greedy", action="store_true",
                    help="the policy plays the card-first greedy rule (SPEC §19.11) instead of sampling "
                         "(default: seeded samples)")
    ap.add_argument("--deck0", default="hog26")
    ap.add_argument("--deck1", default="giant")
    ap.add_argument("--seed", type=int, default=0, help="match seed; also seeds the policy's action samples")
    ap.add_argument("--frame-skip", type=int, default=10)
    ap.add_argument("--fps", type=float, default=10.0, help="frames shown per second (0 = no delay)")
    ap.add_argument("--every", type=int, default=1, help="draw every N decision steps")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--raylib", action="store_true")
    ap.add_argument("--no-clear", action="store_true")
    args = ap.parse_args()
    (watch_raylib if args.raylib else watch_text)(args)


if __name__ == "__main__":
    main()
