"""SPEC §19.7.6 per-card statistics in the env / league logs (v0.5-G).

`Royale` (all policy rows: both rows in self-play, row 0 with a scripted opponent) and `LeagueVecEnv`
(learner rows only) count, per card c, available_c = decisions where c is in hand, its slot's
`affordable` flag is 1 and `lockout` is 0, and played_c = decisions whose chosen action plays the slot
holding c and is allowed by the action mask. Every emitted log adds cards/play_rate/<CARD_KEY> =
played_c / available_c for cards with available_c > 0, and cards/decisions; counters reset after each
emission.

The reference counts are computed from the acting row's own observation before the step (hand ids,
affordable flags, lockout flag, mask) and the action sent. <CARD_KEY> is read as
`pufferroyale.CARD_KEYS[c]` (the data key of card c); `cards/decisions` as the number of counted
decisions since the previous emission (see the report's SPEC questions).
"""
import numpy as np
import pytest

import envkit as E
import helpers as H
import leaguekit as L
import v05kit as V

PREFIX = "cards/play_rate/"


def card_keys():
    import pufferroyale
    return list(pufferroyale.CARD_KEYS)


class Counter:
    def __init__(self, grid=1):
        self.grid = grid
        self.reset()

    def reset(self):
        self.avail = np.zeros(64, np.int64)
        self.played = np.zeros(64, np.int64)
        self.decisions = 0

    def add(self, row, a, lay):
        m = E.R()
        hand = E.hand_of(row, lay)
        aff = V.sc(row, lay, "affordable")
        locked = V.sc(row, lay, "lockout")[0] > 0.5
        self.decisions += 1
        if not locked:
            for s, c in enumerate(hand):
                if c is not None and aff[s] > 0.5:
                    self.avail[c] += 1
        if a > 0:
            cm = np.asarray(m.action_mask(row, grid=self.grid)) if self.grid != 1 else (V.fine_mask(row) > 0.5)
            if cm[a]:
                B = V.N_BLOCKS[self.grid]
                s = (a - 1) // B
                self.played[hand[s]] += 1

    def check(self, lg, where):
        keys = card_keys()
        assert "cards/decisions" in lg, f"{where}: log lacks cards/decisions: {sorted(lg)[:12]}..."
        assert int(round(lg["cards/decisions"])) == self.decisions, \
            f"{where}: cards/decisions {lg['cards/decisions']} != {self.decisions} counted decisions"
        got = {k[len(PREFIX):]: v for k, v in lg.items() if k.startswith(PREFIX)}
        want = {keys[c]: self.played[c] / self.avail[c] for c in range(64) if self.avail[c] > 0}
        assert set(got) == set(want), f"{where}: play_rate keys {sorted(got)} != {sorted(want)}"
        for k, v in want.items():
            assert abs(got[k] - v) <= 1e-5, f"{where}: {PREFIX}{k} = {got[k]} != {v}"
        self.reset()


def _policy(rng, row, grid, p=0.5, illegal_p=0.05):
    m = E.R()
    cm = np.asarray(m.action_mask(row, grid=grid)) if grid != 1 else (V.fine_mask(row) > 0.5)
    r = rng.random()
    if r < illegal_p:
        bad = np.nonzero(~cm)[0]
        return int(rng.choice(bad)) if len(bad) else 0
    leg = np.nonzero(cm)[0]
    return int(rng.choice(leg[1:])) if len(leg) > 1 and r < p else 0


@pytest.mark.parametrize("grid", [1, 2])
def test_selfplay_counts_both_rows(pr, grid):
    lay = V.layout()
    env = E.make(num_envs=1, num_agents=2, deck0="random", deck1="random", frame_skip=50, log_interval=1, seed=3,
                 placement_grid=grid)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(grid)
    cnt = Counter(grid)
    emitted = 0
    for t in range(400):
        acts = [_policy(rng, r, grid) for r in obs]
        for r, a in zip(obs, acts):
            cnt.add(r, a, lay)
        obs, rew, term, trunc, infos = E.step(env, acts)
        for lg in E.logs_in(infos):
            cnt.check(lg, f"step {t}")
            emitted += 1
    assert emitted >= 2, "setup: logs must be emitted (frame_skip 50, log_interval 1)"
    env.close()


@pytest.mark.parametrize("side", [0, 1])
def test_scripted_opponent_counts_only_the_policy_row(pr, side):
    lay = V.layout()
    env = E.make(num_envs=1, num_agents=1, opponent="random", learner_side=side, bot_play_prob=0.6,
                 deck0="random", deck1="random", frame_skip=50, log_interval=1, seed=9)
    obs, _ = env.reset(seed=9)
    rng = np.random.default_rng(side)
    cnt = Counter(1)
    emitted = 0
    for t in range(300):
        a = _policy(rng, obs[0], 1)
        cnt.add(obs[0], a, lay)
        obs, rew, term, trunc, infos = E.step(env, [a])
        for lg in E.logs_in(infos):
            cnt.check(lg, f"side {side} step {t}")
            emitted += 1
    assert emitted >= 2
    env.close()


def test_counters_reset_after_each_emission_with_log_interval(pr):
    """log_interval > 1: the emitted log covers every decision since the previous emission."""
    lay = V.layout()
    env = E.make(num_envs=2, num_agents=2, deck0="random", deck1="random", frame_skip=50, log_interval=7, seed=5)
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    cnt = Counter(1)
    emitted = 0
    for t in range(500):
        acts = [_policy(rng, r, 1) for r in obs]
        for r, a in zip(obs, acts):
            cnt.add(r, a, lay)
        obs, rew, term, trunc, infos = E.step(env, acts)
        for lg in E.logs_in(infos):
            cnt.check(lg, f"step {t}")
            emitted += 1
    assert emitted >= 3
    env.close()


def test_league_counts_learner_rows_only(pr):
    lay = V.layout()
    pool = L.make_pool(anchors=("bot:random",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.make_league(pool, num_envs=2, seed=1, frame_skip=50, log_interval=1, bot_play_prob=0.6,
                       deck0="random", deck1="random")
    lv.async_reset(1)
    rng = np.random.default_rng(1)
    cnt = Counter(1)
    emitted = 0
    obs, rew, term, trunc, infos, _, _ = lv.recv()
    for t in range(400):
        acts = [_policy(rng, r, 1) for r in obs]
        for r, a in zip(obs, acts):
            cnt.add(r, a, lay)
        lv.send(np.array(acts, np.int32))
        obs, rew, term, trunc, infos, _, _ = lv.recv()
        for lg in infos:
            if isinstance(lg, dict) and ("n" in lg or "cards/decisions" in lg):     # a forwarded Royale log
                cnt.check(lg, f"league step {t}")
                emitted += 1
    assert emitted >= 2
    lv.close()
