"""Check 5 (docs/ADVISOR.md): the advisor pipeline playing inside the simulator.

Player A sees the game through a VARIANT of the advisor pipeline; player B is the same checkpoint on the
engine's own observation (Game.obs). Both sample from their policy (SPEC §19.11) and act every 10 ticks,
as in training. Score = A's mean result (win 1, draw 0.5, loss 0); 0.5 = the pipeline costs nothing.

Variants (what A's observation is built from):
  oracle          board.from_game (perfect) -> encoder. Control: must be 0.5 up to sampling noise.
  no_spells       oracle minus everything about spells (areas, and spell plays in the opponent history)
  no_hp           oracle with every unit at full HP and no stun / slow (the reader sees neither)
  blind           no_spells + no_hp + no underground units (hidden Tesla, burrowing Miner)
  tracker         simulated perfect reader (4 readings / s) -> tracker -> encoder (all of 'blind', plus the
                  tracker's own errors: card attribution, clock, elixir between bar steps, deploy flags)
  tracker_noisy   tracker with 10% of units missed per frame and 300-millitile position noise
  delay1, delay2  tracker, with A's chosen play applied 1 / 2 decision steps (0.5 / 1.0 s) later
                  (reading + human reaction time); a play that is no longer legal is dropped
  noisy_delay2    tracker_noisy + delay2
  realistic       noisy_delay2 + King-tower HP not read (only destroyed / not), as the real reader
                  (added after array 21753108 started; run separately)
  real            the real detector's measured errors (docs/ADVISOR.md 6a): 25% units missed per frame,
                  0.1-tile position noise, 5% wrong type, 20% wrong side, phantoms on the crown towers and
                  0.5 other phantoms per frame; King HP not read
  real_delay1/2   real + the play applied 0.5 / 1.0 s later
  real_towers_standing  real, with crown-tower HP read only as standing / destroyed (own princess bars are
                  mostly hidden under the HP digits in the current interface, docs/ADVISOR.md 6b)

    python -m advisor.eval.in_loop --variant tracker --matches 200 --seed 1 --out runs/advisor/in_loop_tracker.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository

import numpy as np
import torch

from pufferroyale import decks as _decks
from pufferroyale import league
from pufferroyale.game import DECKS, Game

from advisor import board, encoder
from advisor import engine_db as D
from advisor.sim_perception import SimPerceptionConfig, sim_reading
from advisor.tracker import Tracker

VARIANTS = ("oracle", "no_spells", "no_hp", "blind", "tracker", "tracker_noisy", "delay1", "delay2", "noisy_delay2",
            "realistic", "real", "real_delay1", "real_delay2", "real_towers_standing")
REAL = dict(p_miss=0.25, pos_sigma=100.0, p_type_err=0.05, p_side_err=0.2, fp_per_frame=0.5, tower_phantoms=True,
            read_king=False)   # the real detector's errors (docs/ADVISOR.md 6a; the phantom rate is an assumption)
POOL = ("hog26", "giant", "bait", "golem", "lavaloon", "miner_poison", "pekka_bridge")   # S4_A_s1's deck_pool


class History:
    """The opponent's real plays (from engine state diffs), to rebuild a spell-free play history."""

    def __init__(self, opp: int):
        self.opp, self.prev, self.plays = opp, 0, []

    def update(self, st: dict) -> None:
        n = st["plays"][self.opp] - self.prev
        for k in range(n - 1, -1, -1):
            self.plays.append(st["last_played"][self.opp][k])
        self.prev = st["plays"][self.opp]

    def visible(self):
        vis = [c for c in self.plays if D.play_compositions()[c]]
        last4 = (vis[::-1] + [-1, -1, -1, -1])[:4]
        return set(vis), sum(D.CARD_COSTS[c] for c in vis) * D.ELIXIR_UNIT, last4


def _strip(bs: board.BoardState, g: Game, team: int, hist: History, spells: bool, hp: bool, underground: bool):
    bs = copy.deepcopy(bs)
    if spells:
        bs.spells = []
        bs.opp_seen, bs.opp_spent, bs.opp_last4 = hist.visible()
    if hp:
        for u in bs.units:
            u.hp, u.stunned, u.slowed = u.max_hp, False, False
    if underground:
        hidden = {(e["x"], e["y"]) for e in g.entities() if e["kind"] != "tower" and (e["hidden"] or e["burrowing"])}
        hidden = {board.own_xy(team, x, y) for x, y in hidden}
        bs.units = [u for u in bs.units if (u.x, u.y) not in hidden]
    return bs


def _sample(policy, obs: np.ndarray, gen: torch.Generator) -> int:
    with torch.no_grad():
        logits, _ = policy.forward_eval(torch.as_tensor(obs).unsqueeze(0), {})
    return int(league.select_actions(logits, False, gen)[0])


def _deck(rng: random.Random):
    if rng.random() < 0.5:
        return sorted(rng.sample(range(D.N_CARDS), 8))
    return list(DECKS[rng.choice(POOL)])


def play_match(policy, variant: str, seat_a: int, seed: int) -> float:
    rng = random.Random(seed)
    g = Game(deck0=_deck(rng), deck1=_deck(rng), seed=seed)
    gen_a = torch.Generator().manual_seed(seed * 2 + 1)
    gen_b = torch.Generator().manual_seed(seed * 2 + 2)
    seat_b = 1 - seat_a
    hist = History(opp=seat_b)
    uses_tracker = variant in ("tracker", "tracker_noisy", "delay1", "delay2", "noisy_delay2", "realistic", "real",
                               "real_delay1", "real_delay2", "real_towers_standing")
    noisy = variant in ("tracker_noisy", "noisy_delay2", "realistic")
    delay = {"delay1": 1, "delay2": 2, "noisy_delay2": 2, "realistic": 2, "real_delay1": 1, "real_delay2": 2}.get(variant, 0)
    if variant.startswith("real") and variant != "realistic":
        cfg = SimPerceptionConfig(**REAL, tower_hp_standing_only=variant == "real_towers_standing")
    else:
        cfg = SimPerceptionConfig(p_miss=0.1 if noisy else 0.0, pos_sigma=300.0 if noisy else 0.0,
                                  read_king=variant != "realistic")
    tracker = Tracker(deck=g.state()["deck"][seat_a]) if uses_tracker else None
    queue = deque([0] * delay)
    rrng = random.Random(seed + 99)
    tick = 0
    while not g.state()["over"]:
        st = g.state()
        hist.update(st)
        if uses_tracker:
            bs = tracker.update(sim_reading(g, seat_a, cfg, rrng))
        else:
            bs = board.from_game(g, seat_a)
            if variant != "oracle":
                bs = _strip(bs, g, seat_a, hist, spells=variant in ("no_spells", "blind"),
                            hp=variant in ("no_hp", "blind"), underground=variant == "blind")
        if tick % 10 == 0:
            a = _sample(policy, encoder.encode(bs), gen_a)
            if delay:
                queue.append(a)
                a = queue.popleft()
            b = _sample(policy, g.obs(seat_b), gen_b)
            if a:
                g.play_action(seat_a, a)     # an action that is no longer legal is refused (no-op)
            if b:
                g.play_action(seat_b, b)
        step = 5 if uses_tracker else 10
        g.tick(step)
        tick += step
    res = g.state()["result"]
    if res[seat_a] > res[seat_b]:
        return 1.0
    if res[seat_a] < res[seat_b]:
        return 0.0
    return 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "S4_A_s1_10B.pt"))
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    policy, recurrent = league.load_policy(args.ckpt)
    if recurrent:
        raise SystemExit("in_loop.py handles feed-forward checkpoints only")
    t0 = time.time()
    scores = []
    for m in range(args.matches):
        scores.append(play_match(policy, args.variant, seat_a=m % 2, seed=args.seed * 1_000_003 + m))
    s = np.array(scores)
    out = dict(variant=args.variant, ckpt=args.ckpt, matches=len(s), score=round(float(s.mean()), 4),
               ci95=round(float(1.96 * s.std(ddof=1) / np.sqrt(len(s))), 4) if len(s) > 1 else None,
               wins=int((s == 1).sum()), draws=int((s == 0.5).sum()), losses=int((s == 0).sum()),
               seed=args.seed, seconds=round(time.time() - t0, 1))
    print(json.dumps(out))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
