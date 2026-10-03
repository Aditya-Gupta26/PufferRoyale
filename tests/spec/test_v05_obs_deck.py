"""SPEC §19.3 own deck in the observation (v0.5-G).

- New scalar field `own_deck`: the acting team's 8 deck cards as integer ids card_id + 1, ascending by
  card id (the deck as a set; never the hand/queue order), appended after enemy_tower_troop:
  SCALAR_SIZE 298 -> 306, OBS_SIZE 17,707 -> 17,715, MASK_OFFSET + 8; existing fields keep their offsets;
  SCALAR_INDEX["own_deck"] = (298, 8); CARD_ID_SCALARS gains "own_deck"; C, binding.royale_layout(),
  Game.obs() and the Python exports agree.
- It depends only on the own deck; no-leak and seat-mirror properties continue to hold.
"""
import numpy as np
import pytest

import cardsv3 as C
import envkit as E
import gamekit as K
import helpers as H
import v05kit as V

# v0.4 SCALAR_INDEX (the published v0.3/v0.4 layout, §16.4 / §16.6): §19.3 keeps every offset.
V04_SCALAR_INDEX = {
    "elixir": (0, 1), "hand": (1, 4), "hand_cost": (5, 4), "next_card": (9, 1), "affordable": (10, 4),
    "tick": (14, 1), "overtime": (15, 1), "elixir_rate": (16, 1), "lockout": (17, 1), "own_towers": (18, 3),
    "enemy_towers": (21, 3), "king_active": (24, 2), "crowns": (26, 2), "opp_seen": (28, 128),
    "opp_spent": (156, 1), "opp_last4": (157, 4), "opp_deduced_hand": (161, 128), "opp_elixir_ub": (289, 1),
    "own_tower_troop": (290, 4), "enemy_tower_troop": (294, 4),
}
V04_OBS_SIZE, V04_MASK_OFFSET = 17707, 15402


def R():
    return E.R()


def preset_ids(name):
    return tuple(sorted(H.DECKS[name] if name in H.DECKS else C.PRESET_IDS[name]))


def test_own_deck_layout_constants(pr):
    m = R()
    assert int(m.SCALAR_SIZE) == 306, f"SCALAR_SIZE {m.SCALAR_SIZE} (§19.3: 298 -> 306)"
    assert int(m.OBS_SIZE) == 17715, f"OBS_SIZE {m.OBS_SIZE} (§19.3: 17,707 -> 17,715)"
    assert int(m.MASK_OFFSET) == V04_MASK_OFFSET + 8, f"MASK_OFFSET {m.MASK_OFFSET} (§19.3: + 8)"
    assert int(m.MASK_OFFSET) + int(m.MASK_SIZE) == int(m.OBS_SIZE)
    assert tuple(int(v) for v in m.SCALAR_INDEX["own_deck"]) == (298, 8), m.SCALAR_INDEX.get("own_deck")
    assert "own_deck" in tuple(m.CARD_ID_SCALARS), f"CARD_ID_SCALARS {m.CARD_ID_SCALARS} lacks own_deck"


def test_existing_scalar_fields_keep_their_offsets(pr):
    """§19.3: all existing fields keep their offsets (keys may be split more finely, but every v0.4
    entry that still exists must be unchanged, and the canonical layout must still map)."""
    m = R()
    for k, v in V04_SCALAR_INDEX.items():
        if k in m.SCALAR_INDEX:
            assert tuple(int(x) for x in m.SCALAR_INDEX[k]) == v, f"SCALAR_INDEX[{k!r}] moved: {m.SCALAR_INDEX[k]} vs {v}"
    lay = E.scalar_layout()                          # canonical fields incl. own_deck at (298, 8)
    assert lay["own_deck"] == (298, 8)
    assert lay["enemy_tower_troop"] == (294, 4) and lay["own_elixir"] == (0, 1)
    others = [int(v[0]) + int(v[1]) for k, v in m.SCALAR_INDEX.items() if k != "own_deck"]
    assert max(others) == 298, "own_deck is appended after enemy_tower_troop (nothing after it)"


def test_binding_layout_and_game_obs_agree(pr):
    """§19.3: binding.royale_layout(), Game.obs() and the Python exports agree."""
    import pufferroyale.binding as b
    m = R()
    lay = b.royale_layout()
    assert isinstance(lay, dict)
    for k in ("OBS_SIZE", "SCALAR_SIZE", "MASK_OFFSET", "SCALAR_OFFSET"):
        assert k in lay, f"royale_layout() lacks {k}"
        assert int(lay[k]) == int(getattr(m, k)), f"royale_layout()[{k}] = {lay[k]} vs royale.{k} = {getattr(m, k)}"
    found = V.find_values(lay, "own_deck")
    assert found, "royale_layout() must describe the own_deck field"
    assert all(tuple(int(x) for x in f) == (298, 8) for f in found), found
    g = K.new_game("golem", "xbow", seed=3)
    for team, name in ((0, "golem"), (1, "xbow")):
        o = np.asarray(g.obs(team))
        assert o.shape == (int(m.OBS_SIZE),), f"Game.obs shape {o.shape}"
        assert V.own_deck(o) == preset_ids(name)
        assert V.own_deck(o) == tuple(sorted(K.deck_of(g, team)))


@pytest.mark.parametrize("decks", [("hog26", "giant"), ("golem", "xbow"), ("royal_hogs", "royal_hogs"),
                                   ("pekka_bridge", "miner_poison")])
def test_own_deck_values_in_env(pr, decks):
    """own_deck = the row's deck ids + 1, ascending, unchanged by plays and by auto-resets."""
    env = E.make(num_envs=2, num_agents=2, deck0=decks[0], deck1=decks[1], frame_skip=100, seed=2)
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(0)
    want = {0: preset_ids(decks[0]), 1: preset_ids(decks[1])}
    lay = V.layout()
    ends = 0
    for t in range(150):
        for r in range(4):
            d = V.own_deck(obs[r])
            assert d == want[r % 2], f"step {t} row {r}: own_deck {d} != {want[r % 2]}"
            hand = E.hand_of(obs[r], lay)
            assert set(hand) <= set(d) and E.next_of(obs[r], lay) in d, "hand / next card are deck cards"
        obs, rew, term, *_ = E.step(env, [E.random_bot_action(rng, r, 0.5) for r in obs])
        ends += int(term.sum())
    assert ends > 0, "setup: auto-resets must occur (frame_skip 100)"
    env.close()


def test_own_deck_is_the_set_not_the_shuffle(pr):
    """Different seeds shuffle the deck differently (hand order changes) but own_deck is identical."""
    lay = V.layout()
    hands, decks = set(), set()
    for seed in range(12):
        env = E.make(deck0="bait", deck1="giant", seed=seed)
        obs, _ = env.reset(seed=seed)
        hands.add(tuple(E.hand_of(obs[0], lay)))
        decks.add(V.own_deck(obs[0]))
        env.close()
    assert len(hands) > 1, "setup: different seeds must deal different hands"
    assert decks == {preset_ids("bait")}


def test_own_deck_random_decks_consistent(pr):
    """deck 'random' (§16.4): own_deck is 8 distinct ascending ids of real cards that contains the
    current hand and next card, constant within a match."""
    lay = V.layout()
    env = E.make(num_envs=3, num_agents=2, deck0="random", deck1="random", frame_skip=100, seed=5)
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(1)
    prev = [V.own_deck(r) for r in obs]
    for t in range(120):
        for r in range(6):
            d = V.own_deck(obs[r])
            assert len(d) == 8 and list(d) == sorted(set(d)) and all(0 <= c < 64 for c in d), d
            assert set(E.hand_of(obs[r], lay)) <= set(d) and E.next_of(obs[r], lay) in d
        acts = [E.random_bot_action(rng, r, 0.5) for r in obs]
        obs, rew, term, *_ = E.step(env, acts)
        for r in range(6):
            if not term[r]:
                assert V.own_deck(obs[r]) == prev[r], "own_deck changed within a match"
        prev = [V.own_deck(r) for r in obs]
    env.close()


def test_own_deck_does_not_leak_the_opponent_deck(pr):
    """§19.3: own_deck depends only on the own deck; the opponent's deck (as a sorted id vector) never
    appears in the observation, and swapping it leaves the viewer's whole observation unchanged."""
    for view, kw_a, kw_b in ((0, dict(deck0="hog26", deck1="golem"), dict(deck0="hog26", deck1="lavaloon")),
                             (1, dict(deck0="xbow", deck1="giant"), dict(deck0="royal_hogs", deck1="giant"))):
        ea, eb = E.make(**kw_a), E.make(**kw_b)
        oa, _ = ea.reset(seed=4)
        ob, _ = eb.reset(seed=4)
        opp_a = kw_a["deck1"] if view == 0 else kw_a["deck0"]
        opp_vec = np.array([c + 1 for c in preset_ids(opp_a)], np.float32)
        for k in range(25):
            assert np.array_equal(oa[view], ob[view]), f"view {view} obs differs at step {k}"
            row = oa[view]
            assert not any(np.array_equal(row[i:i + 8], opp_vec) for i in range(len(row) - 7)), \
                "the opponent's deck (as a sorted id vector) appears in the observation"
            oa, *_ = E.step(ea, [0, 0])
            ob, *_ = E.step(eb, [0, 0])
        ea.close()
        eb.close()


def test_own_deck_seat_mirror(pr):
    """Seat mirror: swapping the decks between seats swaps the own_deck fields."""
    a = E.make(deck0="golem", deck1="xbow", seed=6)
    b = E.make(deck0="xbow", deck1="golem", seed=6)
    oa, _ = a.reset(seed=6)
    ob, _ = b.reset(seed=6)
    for _ in range(12):
        assert V.own_deck(oa[0]) == V.own_deck(ob[1]) == preset_ids("golem")
        assert V.own_deck(oa[1]) == V.own_deck(ob[0]) == preset_ids("xbow")
        oa, *_ = E.step(a, [0, 0])
        ob, *_ = E.step(b, [0, 0])
    g0 = K.new_game("golem", "xbow", seed=1)
    g1 = K.new_game("xbow", "golem", seed=1)
    assert V.own_deck(np.asarray(g0.obs(0))) == V.own_deck(np.asarray(g1.obs(1)))
    a.close()
    b.close()


def test_own_deck_single_agent_rows(pr):
    """1-agent mode: the learner row shows its own team's deck (either seat)."""
    for side in (0, 1):
        env = E.make(num_envs=2, num_agents=1, opponent="noop", learner_side=side, deck0="bait", deck1="golem")
        obs, _ = env.reset(seed=0)
        want = preset_ids("bait" if side == 0 else "golem")
        assert all(V.own_deck(r) == want for r in obs)
        env.close()
