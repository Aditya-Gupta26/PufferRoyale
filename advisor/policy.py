"""The policy side of the advisor: load a checkpoint and turn its output for one observation into suggestions
a human can read ("play Hog Rider at tile (14, 15), 62%").

The numbers shown are the policy's own probabilities, split card first (SPEC §19.11): P(wait), P(slot s)
summed over its tiles, then P(tile | slot). The card-first greedy pick (league.greedy_actions) is reported
too, so the advisor's top suggestion is exactly what `--greedy` play would do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from pufferroyale import league
from pufferroyale.torch import policy_grid

from . import engine_db as D


@dataclass
class CardSuggestion:
    slot: int
    card: int                 # card id in that hand slot (-1 unknown)
    p_card: float             # P(play this slot now) = sum over its tiles
    tile: tuple               # (tx, ty) own-frame tile (grid 1) or the block's top-left tile (grid > 1)
    block: tuple              # (tx0, ty0, tx1, ty1) exclusive bounds of the chosen tile / block
    p_tile: float             # P(tile | this slot)
    tile_probs: np.ndarray    # P(tile | this slot) over the grid, shape (rows, cols)

    @property
    def name(self) -> str:
        return D.CARD_NAMES[self.card] if 0 <= self.card < D.N_CARDS else "?"


@dataclass
class Suggestion:
    p_wait: float
    cards: list                           # CardSuggestion, most likely first (legal slots only)
    greedy_action: int                    # league.greedy_actions on these logits (0 = wait)
    value: float                          # the critic's value estimate (own perspective)
    probs: np.ndarray = field(repr=False)  # softmax over all actions

    @property
    def top(self) -> Optional[CardSuggestion]:
        return self.cards[0] if self.cards else None


class PolicyAdvisor:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.checkpoint = league.resolve_checkpoint_path(checkpoint)
        self.policy, self.recurrent = league.load_policy(self.checkpoint, device)
        self.device = league.policy_device(self.policy)
        self.grid = policy_grid(self.policy)
        self.env_config = league.checkpoint_env_config(self.checkpoint) or {}
        self.rows, self.cols = (-(-32 // self.grid), -(-18 // self.grid))
        self.B = self.rows * self.cols
        self.reset()

    def reset(self) -> None:
        """Forget the recurrent state (call at the start of every match)."""
        self.state = {"lstm_h": None, "lstm_c": None} if self.recurrent else {}

    @torch.no_grad()
    def logits(self, obs: np.ndarray):
        x = torch.as_tensor(np.asarray(obs, np.float32)).unsqueeze(0).to(self.device)
        logits, value = self.policy.forward_eval(x, self.state)
        return logits, value

    def suggest(self, obs: np.ndarray, hand: list, k: int = 3) -> Suggestion:
        """Suggestions for one observation. `hand` = the 4 card ids of the observation's hand slots.
        Advances the recurrent state, so call exactly once per decision step (0.5 s of game time)."""
        logits, value = self.logits(obs)
        greedy = int(league.greedy_actions(logits)[0])
        x, floor = league._logits_f64(logits)
        x = x[0]
        legal = x > floor
        m = x[legal].max() if legal.any() else 0.0
        e = np.where(legal, np.exp(np.where(legal, x - m, 0.0)), 0.0)
        p = e / e.sum()
        cards = []
        for s in range(4):
            seg = p[1 + s * self.B:1 + (s + 1) * self.B]
            ps = float(seg.sum())
            if ps <= 0.0:
                continue
            seg_logits = np.where(legal[1 + s * self.B:1 + (s + 1) * self.B], x[1 + s * self.B:1 + (s + 1) * self.B], -np.inf)
            j = int(seg_logits.argmax())             # the greedy tile of this slot (ties -> smallest j)
            by, bx = divmod(j, self.cols)
            block = (bx * self.grid, by * self.grid, min(18, bx * self.grid + self.grid), min(32, by * self.grid + self.grid))
            cards.append(CardSuggestion(slot=s, card=int(hand[s]), p_card=ps, tile=(block[0], block[1]), block=block,
                                        p_tile=float(seg[j] / ps), tile_probs=(seg / ps).reshape(self.rows, self.cols)))
        cards.sort(key=lambda c: -c.p_card)
        return Suggestion(p_wait=float(p[0]), cards=cards[:k], greedy_action=greedy,
                          value=float(value.reshape(-1)[0]), probs=p)
