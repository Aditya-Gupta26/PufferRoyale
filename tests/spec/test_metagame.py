"""SPEC §15.5 `pufferroyale.metagame` and §15.4 `scripts/tournament.py`."""
import json
import os

import numpy as np
import pytest

import leaguekit as L


def mg():
    import pufferroyale.metagame as mod
    return mod


def solve(A):
    x, y, v = mg().solve_zero_sum(np.asarray(A, dtype=float))
    return np.asarray(x, float).reshape(-1), np.asarray(y, float).reshape(-1), float(v)


def check_equilibrium(A, x, y, v, tol=1e-6):
    A = np.asarray(A, dtype=float)
    L.assert_distribution(x, A.shape[0], tol)
    L.assert_distribution(y, A.shape[1], tol)
    assert abs(x @ A @ y - v) <= tol, f"x^T A y = {x @ A @ y} != value {v}"
    assert np.all(x @ A >= v - tol), f"a column deviation lowers the row payoff below v: {x @ A} vs {v}"
    assert np.all(A @ y <= v + tol), f"a row deviation raises the row payoff above v: {A @ y} vs {v}"


# ------------------------------------------------------------------------------------------
# solve_zero_sum
# ------------------------------------------------------------------------------------------
def test_rps_uniform_value_zero(pr):
    A = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    x, y, v = solve(A)
    assert np.allclose(x, 1 / 3, atol=1e-6) and np.allclose(y, 1 / 3, atol=1e-6) and abs(v) < 1e-6
    check_equilibrium(A, x, y, v)


def test_matching_pennies(pr):
    A = [[1, -1], [-1, 1]]
    x, y, v = solve(A)
    assert np.allclose(x, 0.5, atol=1e-6) and np.allclose(y, 0.5, atol=1e-6) and abs(v) < 1e-6


def test_strictly_dominated_get_zero_mass(pr):
    # row 2 < row 0 elementwise (dominated for the maximiser); column 2 > column 0 elementwise
    # (dominated for the minimiser). The reduced game [[3,2],[0,5]] has the unique equilibrium
    # x = (5/6, 1/6), y = (1/2, 1/2), value 2.5.
    A = [[3, 2, 4], [0, 5, 1], [2, 1, 3]]
    x, y, v = solve(A)
    check_equilibrium(A, x, y, v)
    assert x[2] <= 1e-6, f"strictly dominated row got mass {x[2]}"
    assert y[2] <= 1e-6, f"strictly dominated column got mass {y[2]}"
    assert np.allclose(x, [5 / 6, 1 / 6, 0], atol=1e-6) and np.allclose(y, [0.5, 0.5, 0], atol=1e-6)
    assert abs(v - 2.5) < 1e-6


def test_pure_saddle_point(pr):
    A = [[3, 2], [1, 0]]
    x, y, v = solve(A)
    assert np.allclose(x, [1, 0], atol=1e-6) and np.allclose(y, [0, 1], atol=1e-6) and abs(v - 2) < 1e-6


def test_one_by_one(pr):
    x, y, v = solve([[0.7]])
    assert np.allclose(x, [1]) and np.allclose(y, [1]) and abs(v - 0.7) < 1e-6


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (1, 4), (4, 1)])
def test_rectangular_games(pr, shape):
    rng = np.random.default_rng(sum(shape))
    A = rng.uniform(-1, 1, shape)
    x, y, v = solve(A)
    check_equilibrium(A, x, y, v)


def test_rectangular_known_value(pr):
    A = [[1, -1, 0.5], [-1, 1, 0.5]]
    x, y, v = solve(A)
    assert abs(v) < 1e-6 and np.allclose(x, 0.5, atol=1e-6)
    check_equilibrium(A, x, y, v)


@pytest.mark.parametrize("seed", range(6))
def test_random_games_equilibrium_property(pr, seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(2, 9, size=2)
    A = rng.normal(size=(n, m))
    x, y, v = solve(A)
    check_equilibrium(A, x, y, v)


def test_value_invariance_under_shift(pr):
    rng = np.random.default_rng(42)
    A = rng.normal(size=(5, 4))
    _, _, v1 = solve(A)
    _, _, v2 = solve(A + 3.0)
    assert abs(v2 - v1 - 3.0) < 1e-6


# ------------------------------------------------------------------------------------------
# payoff_matrix / metagame_nash / elo
# ------------------------------------------------------------------------------------------
def test_payoff_matrix_antisymmetric(pr):
    """§15.7.9: payoff_matrix(results, n_agents), results = (i, j, r) with r = agent i's result."""
    results = [(0, 1, 1), (0, 1, 1), (1, 0, -1), (0, 2, 0), (2, 1, 1), (1, 2, -1), (3, 0, -1), (0, 3, 0)]
    P = np.asarray(mg().payoff_matrix(results, 4), float)
    assert P.shape == (4, 4)
    L.assert_antisymmetric(P)
    assert abs(P[0, 1] - 1.0) < 1e-9 and abs(P[0, 2] - 0.5) < 1e-9 and abs(P[2, 1] - 1.0) < 1e-9
    assert abs(P[0, 3] - 0.75) < 1e-9, "P[0][3] = mean((r+1)/2) over (3,0,-1) and (0,3,0) from 0's view"


def _nash_x(P):
    out = mg().metagame_nash(np.asarray(P, float))
    assert isinstance(out, (tuple, list)) and len(out) == 2, "metagame_nash(P) -> (x, value) (§15.7.9)"
    x, value = out
    assert abs(float(value)) < 1e-6, "the symmetric game A = P - 0.5 has value 0"
    return L.assert_distribution(x, len(P))


def test_metagame_nash_cyclic_is_uniform(pr):
    P = [[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]]
    assert np.allclose(_nash_x(P), 1 / 3, atol=1e-6)


def test_metagame_nash_dominant_agent(pr):
    P = [[0.5, 0.8, 0.9], [0.2, 0.5, 0.6], [0.1, 0.4, 0.5]]
    assert np.allclose(_nash_x(P), [1, 0, 0], atol=1e-6)


def test_metagame_nash_matches_solve_zero_sum(pr):
    rng = np.random.default_rng(3)
    U = rng.uniform(size=(5, 5))
    P = np.triu(U, 1) + np.tril(1 - U.T, -1) + np.eye(5) * 0.5
    x = _nash_x(P)
    A = P - 0.5
    check_equilibrium(A, x, x, 0.0)


def test_elo_ordering_and_mean(pr):
    P = np.array([[0.5, 0.7, 0.8, 0.9], [0.3, 0.5, 0.6, 0.8], [0.2, 0.4, 0.5, 0.7], [0.1, 0.2, 0.3, 0.5]])
    r = np.asarray(mg().elo(P, np.full((4, 4), 20)), float)
    assert r.shape == (4,) and abs(r.mean() - 1500) < 1e-6
    assert np.all(np.diff(r) < 0), f"ratings must follow the score ordering: {r}"
    even = np.asarray(mg().elo(np.full((3, 3), 0.5), np.full((3, 3), 20)), float)
    assert np.allclose(even, 1500, atol=1e-6)


def bt_scores(R):
    R = np.asarray(R, float)
    return 1.0 / (1.0 + 10 ** ((R[None, :] - R[:, None]) / 400))


@pytest.mark.parametrize("R", [[1595.42, 1404.58], [1700, 1500, 1300], [1620, 1580, 1450, 1350]])
def test_elo_is_maximum_likelihood_bradley_terry(pr, R):
    """§15.7.9: ML Bradley-Terry on the 400*log10 scale, centred at 1500: scores generated exactly by
    a BT model with equal game counts are recovered exactly (e.g. p = 0.75 <-> 400*log10(3) apart)."""
    R = np.asarray(R, float)
    R = R - R.mean() + 1500
    got = np.asarray(mg().elo(bt_scores(R), np.full((len(R), len(R)), 50)), float)
    assert np.allclose(got, R, atol=1e-3), f"elo {got} vs generating ratings {R}"


def test_elo_two_agent_closed_form(pr):
    P = np.array([[0.5, 0.75], [0.25, 0.5]])
    r = np.asarray(mg().elo(P, np.full((2, 2), 10)), float)
    d = 400 * np.log10(3)
    assert np.allclose(r, [1500 + d / 2, 1500 - d / 2], atol=1e-3), r


# ------------------------------------------------------------------------------------------
# play_match / deck_metagame
# ------------------------------------------------------------------------------------------
def test_play_match_deterministic(pr):
    a = mg().play_match("bot:random", "bot:heuristic", "hog26", "giant", seed=5)
    b = mg().play_match("bot:random", "bot:heuristic", "hog26", "giant", seed=5, greedy=True)
    assert a == b, "play_match must be deterministic given the seed"
    assert a in (-1, 0, 1), "play_match -> +1/0/-1 for agent_a (§15.7.10)"


@pytest.mark.slow
def test_play_match_results(pr):
    assert mg().play_match("bot:noop", "bot:noop", "hog26", "hog26", seed=0) == 0, "untouched towers: draw"
    for seed in range(4):
        assert mg().play_match("bot:heuristic", "bot:noop", "hog26", "hog26", seed=seed) == 1
        assert mg().play_match("bot:noop", "bot:heuristic", "bait", "giant", seed=seed) == -1


@pytest.mark.slow
def test_deck_metagame(pr):
    """§15.7.10: deck_metagame(agent, decks, matches) -> (P, x), P antisymmetric deck-vs-deck."""
    decks = ["hog26", "giant", "bait"]
    P, x = mg().deck_metagame("bot:heuristic", decks, 2)
    P = L.assert_antisymmetric(P)
    assert P.shape == (3, 3)
    x = L.assert_distribution(x, 3)
    check_equilibrium(P - 0.5, x, x, 0.0)


# ------------------------------------------------------------------------------------------
# scripts/tournament.py
# ------------------------------------------------------------------------------------------
TOURNAMENT_FLAGS = ["--agents", "--decks", "--matches", "--out", "--seed", "--device"]


def test_tournament_cli_flags(pr):
    flags = L.help_flags("scripts/tournament.py")
    missing = [f for f in TOURNAMENT_FLAGS if f not in flags]
    assert not missing, f"tournament.py lacks SPEC §15.7.11 flags {missing}"


@pytest.mark.slow
def test_tournament_script(pr, tmp_path):
    """§15.7.11: --agents A.. --decks D.. --matches N --out FILE; JSON keys agents/payoff/elo/nash."""
    ck = L.make_ckpt(tmp_path / "agent.pt")
    agents = ["bot:noop", "bot:random", "bot:heuristic", ck]
    out_path = tmp_path / "t.json"
    out = L.run_script(["scripts/tournament.py", "--agents", *agents, "--decks", "hog26", "giant",
                        "--matches", "2", "--out", str(out_path), "--seed", "1", "--device", "cpu"])
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    assert out_path.exists(), "--out FILE must be written"
    res = json.load(open(out_path))
    for k in ("agents", "payoff", "elo", "nash"):
        assert k in res, f"tournament JSON lacks {k!r}: keys {sorted(res)}"
    assert list(res["agents"]) == agents
    P = L.assert_antisymmetric(res["payoff"])
    assert P.shape == (4, 4)
    nash = res["nash"]
    if isinstance(nash, dict):                  # only the key is pinned; accept {"x": [...], ...}
        nash = nash.get("x", nash.get("mixture"))
    x = L.assert_distribution(nash, 4)
    check_equilibrium(P - 0.5, x, x, 0.0)
    elo = np.asarray(res["elo"], float)
    assert elo.shape == (4,) and abs(elo.mean() - 1500) < 1e-3
    assert P[2, 0] > 0.5, "heuristic must beat noop in the tournament"
