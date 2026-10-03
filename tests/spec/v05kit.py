"""Shared helpers for the SPEC §19 (v0.5-G "training work package") spec tests.

Everything here is derived from docs/SPEC.md §19 (and the earlier sections it builds on) only:
reference implementations of the placement-grid decoding (§19.4), the reward-v2 potential
(§19.1), the entropy split (§19.2) and small run/JSON utilities. Nothing reads the builder's code.
"""
import json
import math
import os

import numpy as np

import envkit as E
import helpers as H

ROOT = H.ROOT
SCRATCH = "/private/tmp/claude-501/-Users-aditya-Documents-NYU-Sem-3-ECE-GY-6263--Game-Theory--PufferRoyale/" \
          "e697885a-bf3e-410a-8521-230ac1befda5/scratchpad/tester"

# ---------------------------------------------------------------------------------------------
# §19.4 placement grid
# ---------------------------------------------------------------------------------------------
GRIDS = (1, 2, 4)
GRID_SHAPE = {1: (32, 18), 2: (16, 9), 4: (8, 5)}            # (ceil(32/g), ceil(18/g))
N_BLOCKS = {g: r * c for g, (r, c) in GRID_SHAPE.items()}     # 576 / 144 / 40
N_ACT = {g: 1 + 4 * b for g, b in N_BLOCKS.items()}           # 2305 / 577 / 161


def block_bounds(g, b):
    """§19.4: block b of grid g -> (x0, x1, y0, y1), half-open own-frame tile ranges."""
    rows, cols = GRID_SHAPE[g]
    by, bx = b // cols, b % cols
    x0, x1 = g * bx, min(18, g * bx + g)
    y0, y1 = g * by, min(32, g * by + g)
    return x0, x1, y0, y1


def ref_coarse_mask(fine_mask, g):
    """§19.4: coarse[0] = 1; coarse[1 + s*B + b] = max of the fine mask over the block's tiles.
    `fine_mask`: (..., 2305) array-like. Returns a bool array (..., 1 + 4B)."""
    fm = np.asarray(fine_mask) > 0.5
    lead = fm.shape[:-1]
    B = N_BLOCKS[g]
    out = np.zeros(lead + (1 + 4 * B,), dtype=bool)
    out[..., 0] = True
    tiles = fm[..., 1:].reshape(lead + (4, 32, 18))
    for b in range(B):
        x0, x1, y0, y1 = block_bounds(g, b)
        blk = tiles[..., :, y0:y1, x0:x1].reshape(lead + (4, -1)).any(axis=-1)
        for s in range(4):
            out[..., 1 + s * B + b] = blk[..., s]
    return out


def ref_coarse_to_fine(fine_mask, a, g):
    """§19.4 representative tile: among the block's tiles legal for the slot (exact fine mask),
    minimise (2tx+1-(x0+x1))^2 + (2ty+1-(y0+y1))^2, ties -> smaller ty, then smaller tx.
    Returns the fine action, or 0 for the no-op / no legal tile in the block."""
    fm = np.asarray(fine_mask).reshape(-1) > 0.5
    if a == 0:
        return 0
    B = N_BLOCKS[g]
    s, b = (a - 1) // B, (a - 1) % B
    x0, x1, y0, y1 = block_bounds(g, b)
    best = None
    for ty in range(y0, y1):
        for tx in range(x0, x1):
            f = 1 + s * 576 + ty * 18 + tx
            if not fm[f]:
                continue
            key = ((2 * tx + 1 - (x0 + x1)) ** 2 + (2 * ty + 1 - (y0 + y1)) ** 2, ty, tx)
            if best is None or key < best[0]:
                best = (key, f)
    return 0 if best is None else best[1]


# ---------------------------------------------------------------------------------------------
# Observation access (§9 / §16.4 / §19.3), through the tester's canonical layout (envkit)
# ---------------------------------------------------------------------------------------------
def layout():
    return E.scalar_layout()


def sc(row, lay, name):
    return E.scalar(row, lay, name)


def own_deck(row, lay=None):
    """§19.3: own_deck = 8 integer ids card_id + 1, ascending. Returns the tuple of card ids."""
    m = E.R()
    off, n = m.SCALAR_INDEX["own_deck"]
    v = row[m.SCALAR_OFFSET + off:m.SCALAR_OFFSET + off + n]
    ids = []
    for x in v:
        assert float(x) == int(round(float(x))), f"own_deck value {x} is not an integer"
        ids.append(int(round(float(x))) - 1)
    return tuple(ids)


def fine_mask(row):
    return E.mask_of(row)


# ---------------------------------------------------------------------------------------------
# §19.1 reward v2 reference
# ---------------------------------------------------------------------------------------------
def tower_sums(row, lay):
    """T_own, T_enemy from the acting row's own-frame tower HP fractions (§19.1: T_k = Σ
    pr_obs_tower_frac(s, k, i), the observation encoder's fractions)."""
    return (float(np.sum(sc(row, lay, "own_tower_hp").astype(np.float64))),
            float(np.sum(sc(row, lay, "enemy_tower_hp").astype(np.float64))))


def crowns_of(row, lay):
    c = sc(row, lay, "crowns").astype(np.float64) * 3.0
    return int(round(c[0])), int(round(c[1]))


def clip(x, c):
    return max(-c, min(c, x))


def phi_hat(row, lay, w, L_diff=0.0, P_diff=0.0, L_cap=20.0, P_cap=20.0):
    """§19.1 potential from the acting row's perspective (team 0's formula with own/enemy;
    team 1's potential is its negation, which is the same formula from team 1's own view).
    `w` = dict(t=, c=, e=, p=); L_diff = L_own - L_enemy (elixir); P_diff = P_own - P_enemy."""
    t_own, t_en = tower_sums(row, lay)
    c_own, c_en = crowns_of(row, lay)
    return (w.get("t", 0.0) * (t_own - t_en) + w.get("c", 0.0) * (c_own - c_en)
            - w.get("e", 0.0) * clip(L_diff, L_cap) + w.get("p", 0.0) * clip(P_diff, P_cap))


def anneal(n, N, n0):
    """§19.1: m_n = max(0, 1 - (n0 + n)/N) if N > 0 else 1."""
    if N > 0:
        return max(0.0, 1.0 - (n0 + n) / N)
    return 1.0


def elixir_income(T):
    """§4: total elixir regen (units) over ticks 0..T-1: 50/tick in [0,2400), 100 in [2400,4800),
    150 in [4800,6000)."""
    a = min(T, 2400)
    b = max(0, min(T, 4800) - 2400)
    c = max(0, T - 4800)
    return 50 * a + 100 * b + 150 * c


def passive_leak(T):
    """Cumulative leaked elixir (in elixir) of a team that never plays, after T processed ticks:
    start 16800 units, cap 28000 (§4)."""
    return max(0, 16800 + elixir_income(T) - 28000) / 2800.0


def match_result_from_log(lg, row_team=0):
    """+1/0/-1 for team `row_team` from a single-episode (n == 1) Royale log."""
    assert abs(float(lg["n"]) - 1.0) < 1e-9, f"expected a single-episode log, n={lg['n']}"
    r0 = int(round(float(lg["win_0"]) - float(lg["win_1"])))
    return r0 if row_team == 0 else -r0


# ---------------------------------------------------------------------------------------------
# §19.2 entropy split reference (float64)
# ---------------------------------------------------------------------------------------------
def entropy_split(logits):
    """Per-row (H_joint, H_card, H_pos) for joint logits over A = 1 + 4B (slot-major)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    A = z.shape[-1]
    B = (A - 1) // 4
    assert 1 + 4 * B == A

    def plogp(q):
        return np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0)

    hj = -np.sum(plogp(p), axis=-1)
    slots = p[..., 1:].reshape(p.shape[:-1] + (4, B))
    Ps = slots.sum(axis=-1)
    card = np.concatenate([p[..., :1], Ps], axis=-1)
    hc = -np.sum(plogp(card), axis=-1)
    cond = np.where(Ps[..., None] > 0, slots / np.where(Ps[..., None] > 0, Ps[..., None], 1.0), 0.0)
    hpos = np.sum(Ps * (-np.sum(plogp(cond), axis=-1)), axis=-1)
    return hj, hc, hpos


# ---------------------------------------------------------------------------------------------
# Wilson interval (§19.7.8)
# ---------------------------------------------------------------------------------------------
def wilson(p, n, z=1.959963984540054):
    if n <= 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / den
    return centre - half, centre + half


# ---------------------------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------------------------
def walk(obj, path=()):
    """Yield (path, dict) for every dict nested in a JSON object."""
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk(v, path + (k,))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, path + (i,))


def find_values(obj, key):
    """Every value stored under `key` anywhere in a nested JSON object."""
    return [d[key] for _, d in walk(obj) if key in d]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def strict_json_line(text):
    """Last stdout line parsed as strict JSON (no NaN / Infinity, §18.8(f))."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    assert lines, "no stdout"

    def bad(c):
        raise ValueError(f"non-standard JSON constant {c}")
    return json.loads(lines[-1], parse_constant=bad)


def scratch(*parts):
    p = os.path.join(SCRATCH, *parts)
    os.makedirs(p, exist_ok=True)
    return p
