"""SPEC §8 scripted bots, through the env (num_agents=1, the opponent is the bot).

Legality is checked behaviourally: SPEC §13.14 counts illegal actions of BOTH agents per episode,
so an episode where the learner only issues no-ops must log illegal_actions == 0 whatever the bot
did, and the bot's accepted plays appear in plays_{bot side}."""
import numpy as np
import pytest

import envkit as E

DECK_PAIRS = [("hog26", "giant"), ("giant", "bait"), ("bait", "hog26"), ("hog26", "hog26")]


def run_match(env, rng, learner, max_steps=700):
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    for _ in range(max_steps):
        a = learner(rng, obs[0])
        obs, rew, term, trunc, infos = E.step(env, [a])
        if term.any():
            logs = E.logs_in(infos)
            assert logs, "a finished episode must produce a log (log_interval=1)"
            return float(rew[0]), logs[-1]
    raise AssertionError("match did not end within 700 steps")


noop = lambda rng, row: 0
rand = lambda rng, row: E.random_bot_action(rng, row, 0.2)


@pytest.mark.slow
@pytest.mark.parametrize("bot", ["random", "heuristic"])
@pytest.mark.parametrize("side", [0, 1])
def test_bot_issues_only_legal_actions(pr, bot, side):
    rng = np.random.default_rng(10 + side)
    for d0, d1 in DECK_PAIRS[:2]:
        env = E.make(num_agents=1, opponent=bot, learner_side=side, deck0=d0, deck1=d1,
                     log_interval=1, bot_play_prob=0.5, seed=3)
        _, lg = run_match(env, rng, noop)
        assert lg["illegal_actions"] == 0, f"{bot} bot issued illegal actions"
        assert lg[f"plays_{side}"] == 0, "the no-op learner never plays"
        assert lg[f"plays_{1 - side}"] > 0, f"{bot} bot never played in a whole match"
        env.close()


@pytest.mark.parametrize("side", [0, 1])
def test_noop_bot_never_plays(pr, side):
    rng = np.random.default_rng(side)
    env = E.make(num_agents=1, opponent="noop", learner_side=side, log_interval=1, frame_skip=20,
                 seed=1)
    _, lg = run_match(env, rng, rand)
    assert lg[f"plays_{1 - side}"] == 0, "the noop bot never plays"
    assert lg[f"plays_{side}"] > 0
    assert lg["illegal_actions"] == 0
    env.close()


def win_rate(bot, learner, side, n, seed):
    rng = np.random.default_rng(seed)
    wins = 0
    for i in range(n):
        d0, d1 = DECK_PAIRS[i % len(DECK_PAIRS)]
        env = E.make(num_agents=1, opponent=bot, learner_side=side, deck0=d0, deck1=d1,
                     log_interval=1, seed=seed + i)
        r, lg = run_match(env, rng, learner)
        assert lg["illegal_actions"] == 0
        wins += int(r < 0)               # the learner lost -> the bot won
        env.close()
    return wins / n


@pytest.mark.slow
@pytest.mark.parametrize("side", [0, 1])
def test_heuristic_beats_noop(pr, side):
    """The heuristic must essentially always beat an idle opponent (at most 1 miss in 12)."""
    wr = win_rate("heuristic", noop, side, 12, 100 + side)
    assert wr >= 11 / 12, f"heuristic vs noop from the {'bottom' if side == 1 else 'top'} seat: {wr:.2f}"


@pytest.mark.slow
@pytest.mark.parametrize("side", [0, 1])
def test_heuristic_beats_random(pr, side):
    """Heuristic vs the SPEC §8 random policy (p = 0.2): a clear majority from both seats."""
    wr = win_rate("heuristic", rand, side, 20, 200 + side)
    assert wr >= 0.65, f"heuristic vs random with learner_side={side}: win rate {wr:.2f}"
