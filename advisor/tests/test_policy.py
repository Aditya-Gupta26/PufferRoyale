"""Check 3 (docs/ADVISOR.md): the advisor's policy wiring.

On simulated states, the advisor's greedy action on its own encoded observation equals league.greedy_actions
on Game.obs, and the suggestion probabilities are a proper card-first split of the policy's distribution.
Needs a checkpoint: ADVISOR_CKPT (default: the repo's models/S4_A_s1_10B.pt).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from pufferroyale import league

from advisor import board, encoder
from advisor.policy import PolicyAdvisor
from advisor.tests.test_encoder import _midgame

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.environ.get("ADVISOR_CKPT", os.path.join(ROOT, "models", "S4_A_s1_10B.pt"))


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="checkpoint not available")
def test_greedy_matches_engine_observation():
    adv = PolicyAdvisor(CKPT)
    n = 0
    for seed in (7, 8, 9):
        for steps in (20, 80, 160, 260):
            g = _midgame(seed=seed, steps=steps)
            if g.state()["over"]:
                continue
            for team in (0, 1):
                bs = board.from_game(g, team)
                s = adv.suggest(encoder.encode(bs), bs.hand)
                ref_logits, _ = adv.logits(g.obs(team))
                assert s.greedy_action == int(league.greedy_actions(ref_logits)[0])
                total = s.p_wait + sum(c.p_card for c in s.cards)
                assert total <= 1.0 + 1e-9
                for c in s.cards:
                    assert abs(c.tile_probs.sum() - 1.0) < 1e-6
                    assert 0 <= c.tile[0] < 18 and 0 <= c.tile[1] < 32
                    assert c.card == bs.hand[c.slot]
                assert abs(s.probs.sum() - 1.0) < 1e-9
                n += 1
    assert n >= 10
