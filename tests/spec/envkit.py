"""Shared helpers for the `pufferroyale.Royale` env tests (SPEC §9; v0.3 layout per §16.4 / §16.6:
integer card ids card_id + 1, 128-wide multi-hots, (64, 11) entity rows, tower-troop one-hots)."""
import numpy as np

import helpers as H

CARD_SLOTS = 128
TOWER_TROOPS = ["princess", "cannoneer", "dagger_duchess", "royal_chef"]      # §16.6 item 16 order

# SPEC §9 item 3 order with the §16.4 encodings; §16.6 item 16 appends the tower-troop one-hots
SCALAR_FIELDS = [
    ("own_elixir", 1), ("hand", 4), ("hand_costs", 4), ("next_card", 1), ("affordable", 4),
    ("tick", 1), ("is_overtime", 1), ("elixir_rate", 1), ("lockout", 1),
    ("own_tower_hp", 3), ("enemy_tower_hp", 3), ("king_active", 2), ("crowns", 2),
    ("opp_seen", CARD_SLOTS), ("opp_spent", 1), ("opp_last4", 4), ("opp_deduced", CARD_SLOTS),
    ("opp_elixir_ub", 1), ("own_tower_troop", 4), ("enemy_tower_troop", 4),
]
SCALAR_TOTAL = sum(n for _, n in SCALAR_FIELDS)          # 298
EXACT_KEYS = ("own_tower_troop", "enemy_tower_troop")    # §16.6 item 16 pins these key names

# SPEC §16.4 entity-row layout (F = 11); own slots 0-31, enemy slots 32-63
E_ID, E_X, E_Y, E_HPF, E_HP2K, E_FLY, E_DEPLOY, E_STUN, E_SLOW, E_BUILDING, E_TOB = range(11)
E_CARD = E_ID
N_OWN = 32


def R():
    import pufferroyale.royale as mod
    return mod


def make(**kw):
    import pufferroyale
    args = dict(num_envs=1, num_agents=2, seed=0)
    args.update(kw)
    return pufferroyale.Royale(**args)


def mask_of(row):
    m = R()
    return row[m.MASK_OFFSET:m.MASK_OFFSET + m.MASK_SIZE]


def legal_actions(row):
    return np.nonzero(mask_of(row) > 0.5)[0]


def step(env, actions):
    obs, rew, term, trunc, infos = env.step(np.asarray(actions, dtype=np.int32))
    assert isinstance(infos, list)
    return obs, rew, term, trunc, infos


def logs_in(infos):
    return [i for i in infos if isinstance(i, dict) and i]


def scalar_layout():
    """Map the canonical v0.3 scalar fields onto `SCALAR_INDEX` (sorted by offset; entries may be
    split more finely than the canonical list, but must be contiguous and in the listed order).
    The tower-troop one-hots must use the exact §16.6 key names."""
    m = R()
    assert hasattr(m, "SCALAR_INDEX"), "pufferroyale.royale.SCALAR_INDEX must be exported (SPEC §13.11)"
    items = sorted(((int(v[0]), int(v[1]), k) for k, v in m.SCALAR_INDEX.items()))
    pos = 0
    for off, n, k in items:
        assert off == pos, f"SCALAR_INDEX not contiguous at {k!r}: offset {off}, expected {pos}"
        pos += n
    assert pos <= m.SCALAR_SIZE
    out, i, off = {}, 0, 0
    for name, n in SCALAR_FIELDS:
        acc = 0
        start = off
        while acc < n:
            assert i < len(items), f"SCALAR_INDEX ends before canonical field {name!r}"
            acc += items[i][1]
            i += 1
        assert acc == n, f"SCALAR_INDEX entries do not align with canonical field {name!r} (len {n})"
        out[name] = (start, n)
        off += n
    for k in EXACT_KEYS:
        assert k in m.SCALAR_INDEX and tuple(int(v) for v in m.SCALAR_INDEX[k]) == out[k], \
            f"SCALAR_INDEX[{k!r}] must be {out[k]} (§16.6 item 16: appended after opp_elixir_ub)"
    return out


def scalar(row, layout, name):
    m = R()
    off, n = layout[name]
    return row[m.SCALAR_OFFSET + off:m.SCALAR_OFFSET + off + n]


def entity_rows(row):
    m = R()
    n, F = m.ENTITY_SHAPE
    return row[m.ENTITY_OFFSET:m.ENTITY_OFFSET + n * F].reshape(n, F)


def ids_to_cards(vals):
    """Integer-id floats (card_id + 1, 0 = empty) -> card ids (None for empty)."""
    out = []
    for v in vals:
        assert float(v) == int(round(float(v))), f"card id feature {v} is not an integer"
        k = int(round(float(v)))
        assert 0 <= k <= CARD_SLOTS, f"card id feature {k} outside 0..128"
        out.append(None if k == 0 else k - 1)
    return out


def hand_of(row, layout):
    return ids_to_cards(scalar(row, layout, "hand"))


def next_of(row, layout):
    return ids_to_cards(scalar(row, layout, "next_card"))[0]


def multihot(cards):
    """§16.6 item 17: multi-hot index = card_id; slots 64..127 stay 0."""
    v = np.zeros(CARD_SLOTS, np.float32)
    for c in cards:
        v[c] = 1.0
    return v


def id_vec(cards, n=4):
    v = np.zeros(n, np.float32)
    for i, c in enumerate(cards[:n]):
        v[i] = c + 1
    return v


def tt_onehot(name):
    v = np.zeros(4, np.float32)
    v[TOWER_TROOPS.index(name)] = 1.0
    return v


def troop_slot_action(row, tile=(9, 21)):
    """An action placing a troop/building card at `tile` (spells would also allow the river)."""
    mk = mask_of(row)
    for s in range(4):
        a = H.action_id(s, *tile)
        if mk[a] > 0.5 and mk[H.action_id(s, 9, 16)] < 0.5:
            return a
    return None


def random_bot_action(rng, row, p=0.2):
    """SPEC §8 'random': with probability p pick uniformly among legal non-noop actions."""
    leg = legal_actions(row)
    if len(leg) > 1 and rng.random() < p:
        return int(rng.choice(leg[1:]))
    return 0
