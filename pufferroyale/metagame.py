"""pufferroyale.metagame -- empirical game-theoretic analysis tools (SPEC §15.5, §15.7.9-10).

    solve_zero_sum(A) -> (x, y, value)   Nash equilibrium of a two-player zero-sum matrix game
                                         (row player maximises x^T A y), by linear programming
    payoff_matrix(results, n_agents)     empirical score matrix P[i][j] in [0, 1], P + P^T = 1
    metagame_nash(P) -> (x, value)       Nash mixture of the symmetric meta-game A = P - 0.5
    elo(P, games) -> ratings             maximum-likelihood Bradley-Terry ratings, 400*log10
                                         scale, mean 1500
    play_match(a, b, deck_a, deck_b, seed, greedy=False) -> +1/0/-1 for a (team 0)
    deck_metagame(agent, decks, matches) -> (P, x)   deck-vs-deck scores of one agent + Nash
    transitivity(P, margin) -> {...}      later-beats-earlier fraction and cyclic triads (SPEC §19.7.7)
    wilson_interval(score, n) -> [lo, hi] Wilson 95% interval of a score

Policy agents sample their actions (seeded, so every match is deterministic given its seed);
greedy=True plays the card-first greedy rule (league.greedy_actions, SPEC §19.11).

A checkpoint of placement grid g > 1 plays through its own grid: its coarse actions are mapped to
fine plays by Game.coarse_to_fine (SPEC §19.4).

Agents (play_match / deck_metagame / scripts/tournament.py): a bot spec "bot:noop" |
"bot:random" | "bot:heuristic", a checkpoint "ckpt:<path>" or a plain path to a PuffeRL
checkpoint (or a run directory: its latest model_*.pt), or an already loaded policy module.
torch is only imported when a policy agent is used.
"""
from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# ======================================================================================
# zero-sum matrix games
# ======================================================================================
_HIGHS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def _maximin(A: np.ndarray) -> np.ndarray:
    """argmax_x min_j (x^T A)_j over the simplex: max v s.t. A^T x >= v 1, 1^T x = 1, x >= 0."""
    from scipy.optimize import linprog

    n, m = A.shape
    c = np.zeros(n + 1)
    c[-1] = -1.0                                             # maximise v
    A_ub = np.hstack([-A.T, np.ones((m, 1))])               # v - (A^T x)_j <= 0
    A_eq = np.concatenate([np.ones(n), [0.0]])[None, :]
    bounds = [(0.0, None)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(m), A_eq=A_eq, b_eq=[1.0], bounds=bounds,
                  method="highs", options=_HIGHS)
    if res.status != 0:
        raise RuntimeError(f"zero-sum LP failed: {res.message}")
    x = np.clip(res.x[:n], 0.0, None)
    return x / x.sum()


def solve_zero_sum(A) -> Tuple[np.ndarray, np.ndarray, float]:
    """Nash equilibrium (x, y, value) of the zero-sum game with row-payoff matrix A (n x m, the
    row player maximises x^T A y, the column player minimises it). Two LPs (scipy HiGHS, tight
    tolerances): the row player's maximin and the column player's minimax. Any shape >= 1x1.
    value = x^T A y (equal to both LP optima at an equilibrium)."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1 or not np.all(np.isfinite(A)):
        raise ValueError(f"A must be a finite 2-D matrix, got shape {A.shape}")
    x = _maximin(A)
    y = _maximin(-A.T)                                       # the column player maximises -A^T
    return x, y, float(x @ A @ y)


# ======================================================================================
# meta-game from match results
# ======================================================================================
def payoff_matrix(results: Iterable[Tuple[int, int, float]], n_agents: int, return_counts: bool = False):
    """P[i][j] = agent i's mean score (win 1 / draw 0.5 / loss 0) against agent j over all
    results (i, j, r) with r in {+1, 0, -1} = agent i's result (a result counts for both
    orientations). P + P^T = 1 off the diagonal, P[i][i] = 0.5, 0.5 for pairs never played.
    With return_counts=True also returns the symmetric game-count matrix."""
    n = int(n_agents)
    S = np.zeros((n, n))
    G = np.zeros((n, n))
    for i, j, r in results:
        i, j, r = int(i), int(j), float(r)
        if r not in (1.0, 0.0, -1.0):
            raise ValueError(f"result must be +1, 0 or -1, got {r}")
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"agent index out of range: ({i}, {j}) with n_agents={n}")
        if i == j:
            continue                                         # self-matches carry no information
        s = 0.5 * (r + 1.0)
        S[i, j] += s
        S[j, i] += 1.0 - s
        G[i, j] += 1.0
        G[j, i] += 1.0
    P = np.full((n, n), 0.5)
    np.divide(S, G, out=P, where=G > 0)
    np.fill_diagonal(P, 0.5)
    return (P, G) if return_counts else P


def metagame_nash(P) -> Tuple[np.ndarray, float]:
    """Nash mixture of the symmetric zero-sum meta-game A = P - 0.5 (value 0 when P + P^T = 1):
    the population mixture no agent can beat on average."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    x, _, v = solve_zero_sum(P - 0.5)
    return x, v


def _strongly_connected(adj: np.ndarray) -> bool:
    n = len(adj)
    reach = adj.astype(bool) | np.eye(n, dtype=bool)
    for _ in range(max(1, int(np.ceil(np.log2(max(n, 2)))) + 1)):
        reach = reach | ((reach.astype(np.int64) @ reach.astype(np.int64)) > 0)
    return bool(reach.all())


def elo(P, games, prior_games: Optional[float] = None, return_info: bool = False):
    """Maximum-likelihood Bradley-Terry ratings on the Elo scale: expected score of i vs j is
    1 / (1 + 10^((R_j - R_i) / 400)); ratings are centred at 1500 (SPEC §15.7.9).

    P: score matrix (P[i][j] = i's mean score vs j, draws count 0.5); games: count matrix
    (games between i and j; the diagonal is ignored, max(games_ij, games_ji) is used).
    Maximises sum_{i<j} n_ij [s_ij log E_ij + (1 - s_ij) log E_ji] by damped Newton steps.

    The ML estimate only exists when every group of agents scored at least a draw against the
    rest (the "scored against" graph is strongly connected); otherwise it diverges (an unbeaten
    agent's rating -> infinity). Only in that case `prior_games` virtual draws (default 1) are
    added between every pair (a standard Laplace-style fix), which keeps ratings finite and
    ordered; the exact ML solution is returned whenever it exists."""
    from scipy.special import expit, log_expit

    P = np.asarray(P, dtype=np.float64)
    G = np.asarray(games, dtype=np.float64)
    n = P.shape[0]
    if P.shape != (n, n) or G.shape != (n, n):
        raise ValueError("P and games must be square matrices of the same size")
    if n == 0:
        return np.zeros(0)
    N = np.maximum(G, G.T)
    np.fill_diagonal(N, 0.0)
    S = 0.5 * (P + 1.0 - P.T)                                # enforce s_ij + s_ji = 1
    S = np.clip(S, 0.0, 1.0)
    regularised = not _strongly_connected((N > 0) & (S > 0))
    if regularised:
        a = 1.0 if prior_games is None else float(prior_games)
        if a <= 0:
            raise ValueError("the ML ratings do not exist for these results; prior_games must be > 0")
        off = 1.0 - np.eye(n)
        W = N * S + 0.5 * a * off
        N = N + a * off
        S = np.divide(W, N, out=np.full((n, n), 0.5), where=N > 0)
    iu = np.triu_indices(n, 1)
    n_p, s_p = N[iu], S[iu]

    def loglik(t):
        d = t[iu[0]] - t[iu[1]]
        return float(np.sum(n_p * (s_p * log_expit(d) + (1.0 - s_p) * log_expit(-d))))

    theta = np.zeros(n)                                      # natural-log strength units
    ll = loglik(theta)
    ones = np.ones((n, n))
    for _ in range(200):
        E = expit(theta[:, None] - theta[None, :])           # E_ij = expected score of i vs j
        g = np.sum(N * (S - E), axis=1)
        if np.max(np.abs(g)) < 1e-11 * max(1.0, N.sum()):
            break
        w = N * E * (1.0 - E)
        H = w.copy()                                         # -Hessian = Laplacian of w
        np.fill_diagonal(H, 0.0)
        H = np.diag(H.sum(1)) - H
        step = np.linalg.solve(H + ones / n, g)              # gauge-fixed Newton step (1^T step = 0)
        t_new, ll_new, k = theta + step, loglik(theta + step), 0
        while ll_new < ll - 1e-12 and k < 40:                 # damping keeps the ascent monotone
            step *= 0.5
            t_new, ll_new, k = theta + step, loglik(theta + step), k + 1
        theta, ll = t_new, ll_new
    ratings = theta * 400.0 / np.log(10.0)
    ratings = ratings - ratings.mean() + 1500.0
    if return_info:
        return ratings, {"regularised": bool(regularised), "log_likelihood": ll}
    return ratings


# ======================================================================================
# agents and matches
# ======================================================================================
BOT_PLAY_PROB = 0.2
_POLICY_CACHE: dict = {}


def resolve_agent(agent, device="cpu"):
    """('bot', kind) | ('policy', (module, recurrent)) for a bot spec, "ckpt:<path>", a plain
    checkpoint path / run directory, or a torch module (loaded checkpoints are cached)."""
    from .league import BOT_KINDS, is_recurrent_policy, load_policy

    if isinstance(agent, tuple) and len(agent) == 2 and agent[0] in ("bot", "policy"):
        return agent
    if isinstance(agent, str):
        if agent.startswith("bot:"):
            kind = agent[4:]
            if kind not in BOT_KINDS:
                raise ValueError(f"unknown bot {agent!r}")
            return "bot", kind
        if agent == "self":
            raise ValueError("'self' is only meaningful inside a league")
        from .league import resolve_checkpoint_path
        path = resolve_checkpoint_path(agent)          # "ckpt:" prefix, run directories -> a file
        key = (os.path.abspath(path), str(device))
        if key not in _POLICY_CACHE:
            _POLICY_CACHE[key] = load_policy(path, device)
        return "policy", _POLICY_CACHE[key]
    if hasattr(agent, "forward_eval"):
        return "policy", (agent, is_recurrent_policy(agent))
    raise ValueError(f"cannot interpret agent {agent!r}")


def agent_name(agent) -> str:
    if isinstance(agent, str):
        return agent
    return type(agent).__name__


class _BotPlayer:
    def __init__(self, kind, seed):
        from .game import Bot
        self.bot = Bot(kind, seed=seed, play_prob=BOT_PLAY_PROB)

    def act(self, game, team):
        return self.bot.act(game, team)


class _PolicyPlayer:
    """Masked policy on Game observations; seeded sampling or the card-first greedy rule (greedy=True,
    league.greedy_actions, SPEC §19.11); recurrent state kept
    for the whole match. A policy of placement grid g > 1 acts in its coarse action space; act()
    returns the fine action that plays (Game.coarse_to_fine, the env's own C mapping; SPEC §19.4)."""

    def __init__(self, policy, recurrent, greedy, seed):
        import torch
        from .league import policy_device
        from .torch import policy_grid
        self.torch = torch
        self.policy, self.recurrent, self.greedy = policy, recurrent, greedy
        self.grid = policy_grid(policy)
        self.device = policy_device(policy)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed) & ((1 << 63) - 1))
        self.state = {"lstm_h": None, "lstm_c": None}

    def act(self, game, team):
        from .league import select_actions
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(game.obs(team)).unsqueeze(0).to(self.device)
            if self.recurrent:
                logits, _ = self.policy.forward_eval(x, self.state)
            else:
                logits, _ = self.policy.forward_eval(x, {})
            a = int(select_actions(logits, self.greedy, self.gen)[0])
        if self.grid > 1 and a:
            a = game.coarse_to_fine(team, a, self.grid)
        return a


def _player(agent, seed, team, greedy, device):
    kind, obj = resolve_agent(agent, device)
    sub = (int(seed) * 2 + team) & ((1 << 62) - 1)           # per-seat stream (as bot_action)
    if kind == "bot":
        return _BotPlayer(obj, sub)
    policy, recurrent = obj
    return _PolicyPlayer(policy, recurrent, greedy, sub)


# ------------------------------------------------------------------------------------ match settings
#: the settings of a match when nothing else is known (Royale / Game defaults)
DEFAULT_MATCH = {"frame_skip": 10, "deploy_lockout_ticks": 90, "tiebreak": "absolute", "tower_troop0": "princess",
                 "tower_troop1": "princess"}
GAME_KEYS = ("deploy_lockout_ticks", "tiebreak", "tower_troop0", "tower_troop1")
_ENV_CFG_CACHE: dict = {}


class MatchSettingsWarning(UserWarning):
    """A checkpoint is evaluated under env settings other than the ones it was trained with."""


def agent_train_config(agent) -> Optional[dict]:
    """The env settings (frame_skip, lockout, tiebreak, tower troops) a checkpoint agent was trained
    with, read from its run's config.json; None for bots, modules, or when no config is found."""
    if not isinstance(agent, str) or agent.startswith("bot:") or agent == "self":
        return None
    from .league import checkpoint_env_config, resolve_checkpoint_path
    try:
        key = os.path.abspath(resolve_checkpoint_path(agent))
    except FileNotFoundError:
        return None
    if key not in _ENV_CFG_CACHE:
        _ENV_CFG_CACHE[key] = checkpoint_env_config(key)
    return _ENV_CFG_CACHE[key]


def match_settings(agents: Sequence, overrides: Optional[dict] = None, use_train_config: bool = True) -> dict:
    """Settings for a match between `agents` (SPEC §18, audit L7): {"game": Game kwargs
    (deploy_lockout_ticks, tiebreak, tower_troop0/1), "frame_skip": [decision cadence per agent],
    "warnings": [...]}. With use_train_config, a checkpoint decides at the frame_skip it was
    trained with and the game uses the checkpoints' lockout / tiebreak / tower troops when they
    agree (a disagreement falls back to the default and warns); bots decide every 10 ticks.
    `overrides` (any DEFAULT_MATCH key; frame_skip applies to every agent) win; a checkpoint that
    ends up playing under settings other than its training ones is warned about."""
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(DEFAULT_MATCH)
    if unknown:
        raise ValueError(f"unknown match settings {sorted(unknown)}; use {sorted(DEFAULT_MATCH)}")
    cfgs = [agent_train_config(a) if use_train_config else None for a in agents]
    game, warns = {k: DEFAULT_MATCH[k] for k in GAME_KEYS}, []
    for k in GAME_KEYS:
        vals = sorted({str(c[k]) if k.startswith("tower") or k == "tiebreak" else int(c[k])
                       for c in cfgs if c and k in c}, key=str)
        if len(vals) == 1:
            game[k] = vals[0]
        elif len(vals) > 1:
            warns.append(f"checkpoints were trained with different {k} {vals}: the match uses the default "
                         f"{DEFAULT_MATCH[k]!r}")
    for k in GAME_KEYS:
        if k in overrides:
            game[k] = overrides[k]
    cad = []
    for a, c in zip(agents, cfgs):
        k = int(overrides["frame_skip"]) if "frame_skip" in overrides else (
            int(c["frame_skip"]) if c and "frame_skip" in c else DEFAULT_MATCH["frame_skip"])
        cad.append(max(1, k))
    for a, c, k in zip(agents, cfgs, cad):
        if not c:
            continue
        diff = [f"{key}={c[key]!r} (match: {game[key]!r})" for key in GAME_KEYS if key in c and c[key] != game[key]]
        if "frame_skip" in c and int(c["frame_skip"]) != k:
            diff.append(f"frame_skip={c['frame_skip']!r} (match: {k})")
        if diff:
            warns.append(f"{agent_name(a)} was trained with " + ", ".join(diff))
    return {"game": game, "frame_skip": cad, "warnings": warns}


def play_match(agent_a, agent_b, deck_a, deck_b, seed, greedy=False, frame_skip=None, device="cpu",
               return_info=False, match_config=None, use_train_config=True):
    """One full match through pufferroyale.Game: agent_a plays team 0 with deck_a, agent_b team 1
    with deck_b. Policy actions are seeded samples from the masked policy (default), or the card-first
    greedy rule with greedy=True (league.greedy_actions, SPEC §19.11). Deterministic given the
    arguments. Returns agent_a's result: +1 win, 0 draw, -1 loss.

    Settings (match_settings): each side decides at its own cadence -- a checkpoint at the
    frame_skip it was trained with (its run's config.json), a bot every 10 ticks -- and the game
    uses the checkpoints' lockout / tiebreak / tower troops, unless `frame_skip` (all agents),
    `match_config` (any of frame_skip, deploy_lockout_ticks, tiebreak, tower_troop0/1) or
    use_train_config=False say otherwise. Mismatches raise a MatchSettingsWarning."""
    from .game import Game

    over = dict(match_config or {})
    if frame_skip is not None:
        over["frame_skip"] = int(frame_skip)
    ms = match_settings([agent_a, agent_b], over, use_train_config)
    for w in ms["warnings"]:
        warnings.warn(w, MatchSettingsWarning, stacklevel=2)
    g = Game(deck0=deck_a, deck1=deck_b, seed=int(seed), **ms["game"])
    players = (_player(agent_a, seed, 0, greedy, device), _player(agent_b, seed, 1, greedy, device))
    cad = ms["frame_skip"]
    while True:
        st = g.state()
        if st["over"]:
            break
        t = int(st["tick"])
        for team in (0, 1):
            if t % cad[team] == 0:
                a = players[team].act(g, team)
                if a:
                    g.play_action(team, a)
        g.tick(min((t // k + 1) * k for k in cad) - t)
    result = int(st["result"][0])
    if return_info:
        return result, {"end_reason": st["end_reason"], "ticks": st["tick"], "crowns": st["crowns"],
                        "settings": {"game": ms["game"], "frame_skip": cad}}
    return result


def wilson_interval(score: float, n: int, z: float = 1.959963984540054) -> list:
    """Wilson 95% score interval [lo, hi] for a proportion p_hat = score (draws counted as 0.5) over
    n matches (SPEC §19.7.8, §19.9.6); [0, 1] when n = 0."""
    n = int(n)
    if n <= 0:
        return [0.0, 1.0]
    p = min(1.0, max(0.0, float(score)))
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def transitivity(P, margin: float = 0.05) -> dict:
    """Within-run transitivity of agents given in chronological order (SPEC §19.7.7, §19.9.5):
    later_beats_earlier = fraction of the pairs i < j with P[j][i] > 0.5 (None for fewer than 2
    agents); pairs = C(n, 2); cyclic_triads = number of unordered triples forming a directed 3-cycle
    a > b > c > a in which every edge's score is > 0.5 + margin; triads = C(n, 3)."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    n = P.shape[0]
    pairs = n * (n - 1) // 2
    later = sum(1 for i in range(n) for j in range(i + 1, n) if P[j, i] > 0.5)
    beats = P > 0.5 + float(margin)
    cyc = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if (beats[a, b] and beats[b, c] and beats[c, a]) or (beats[a, c] and beats[c, b] and beats[b, a]):
                    cyc += 1
    return {"later_beats_earlier": (later / pairs) if pairs else None, "pairs": int(pairs),
            "cyclic_triads": int(cyc), "triads": int(n * (n - 1) * (n - 2) // 6)}


def match_seed(base: int, *keys: int) -> int:
    """A deterministic per-match seed from a base seed and integer keys (splitmix-style mix)."""
    h = (int(base) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    for k in keys:
        h ^= (int(k) + 0x9E3779B97F4A7C15 + (h << 6) + (h >> 2)) & 0xFFFFFFFFFFFFFFFF
        h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 31
    return int(h & 0x7FFFFFFF)


def deck_metagame(agent, decks: Sequence, matches: int, seed: int = 0, greedy: bool = False, device="cpu",
                  return_counts: bool = False, **match_kwargs):
    """The deck meta-game of one agent: for every pair of decks (i < j) the agent plays itself
    `matches` times, deck i in seat 0 for the even-numbered matches and in seat 1 for the others
    (half per seat). P[i][j] = deck i's score vs deck j (antisymmetric, 0.5 on the diagonal);
    x = its Nash mixture over decks. Policy actions are sampled unless greedy=True (play_match).
    Returns (P, x)."""
    decks = list(decks)
    k = len(decks)
    results = []
    for i in range(k):
        for j in range(i + 1, k):
            for m in range(int(matches)):
                s = match_seed(seed, i, j, m)
                if m % 2 == 0:
                    r = play_match(agent, agent, decks[i], decks[j], s, greedy=greedy, device=device, **match_kwargs)
                else:
                    r = -play_match(agent, agent, decks[j], decks[i], s, greedy=greedy, device=device, **match_kwargs)
                results.append((i, j, r))
    P, G = payoff_matrix(results, k, return_counts=True)
    x, _ = metagame_nash(P)
    return (P, x, G) if return_counts else (P, x)


__all__ = ["solve_zero_sum", "payoff_matrix", "metagame_nash", "elo", "play_match", "deck_metagame",
           "resolve_agent", "match_seed", "match_settings", "agent_train_config", "MatchSettingsWarning", "DEFAULT_MATCH",
           "transitivity", "wilson_interval"]
