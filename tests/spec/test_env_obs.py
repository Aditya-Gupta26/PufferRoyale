"""SPEC §9.3 scalars at their SCALAR_INDEX positions (§13.11), in the v0.3 encoding (§16.4 / §16.6:
card ids as card_id + 1, 128-wide multi-hots indexed by card_id, opponent spent / 200): opponent
modelling (cards seen, last-4 played, deduced hand, elixir spent, elixir upper bound), own hand /
next card, tower fractions in own frame, and hidden-information rules."""
import numpy as np
import pytest

import envkit as E
import helpers as H


multihot = E.multihot          # 128 wide, index = card_id (§16.6.17)
id_vec = E.id_vec              # card_id + 1 per slot, 0 = empty (§16.4)


def contains_slice(vec, target):
    n = len(target)
    return any(np.array_equal(vec[i:i + n], target) for i in range(len(vec) - n + 1))


def test_card_counting_features_exact_and_no_hand_leak(pr):
    lay = E.scalar_layout()
    deck1 = H.DECKS["giant"]
    env = E.make(deck0="hog26", deck1="giant", seed=6)
    obs, _ = env.reset(seed=6)
    for _ in range(9):
        obs, *_ = E.step(env, [0, 0])
    played, spent, k = [], 0, 9
    saw_all_at = None
    for _ in range(300):
        hand1 = E.hand_of(obs[1], lay)
        row0 = obs[0]
        f0 = lambda name: E.scalar(row0, lay, name)
        # --- exact opponent features in team 0's view --------------------------------------
        assert np.array_equal(f0("opp_seen"), multihot(set(played))), "opponent cards seen"
        last4 = played[::-1][:4]
        assert np.array_equal(f0("opp_last4"), id_vec(last4)), "opponent last-4 played (ids), most recent first"
        if len(set(played)) == 8:
            want_d = multihot([c for c in deck1 if c not in last4])
        else:
            want_d = np.zeros(E.CARD_SLOTS, np.float32)
        assert np.array_equal(f0("opp_deduced"), want_d), "deduced hand: zero until all 8 seen"
        assert abs(f0("opp_spent")[0] - spent / 200) < 1e-6, "opponent elixir spent / 200 (§16.4)"
        income = 50 * (10 * k) / 2800
        assert abs(f0("opp_elixir_ub")[0] - min(10.0, 6 + income - spent) / 10) < 1e-5
        # --- hidden information: team 1's ordered hand / next card never visible to team 0 -----
        ordered = id_vec(hand1)
        assert not contains_slice(row0, ordered), "team 0 observes team 1's hand"
        # --- team 1 plays a not-yet-seen card (then anything) at own tile (9,21) --------------
        a1 = 0
        mk = E.mask_of(obs[1])
        for s in range(4):
            c = hand1[s]
            if (c not in played or len(set(played)) == 8) and mk[H.action_id(s, 9, 21)] > 0.5:
                a1 = H.action_id(s, 9, 21)
                played.append(c)
                spent += H.COST[c]
                break
        obs, rew, term, trunc, _ = E.step(env, [0, a1])
        k += 1
        assert not term.any(), "setup: the match must not end during this test"
        if len(set(played)) == 8 and saw_all_at is None:
            saw_all_at = len(played)
        if saw_all_at is not None and len(played) >= saw_all_at + 3:
            break
    assert len(set(played)) == 8, f"only {len(set(played))} distinct opponent cards played"
    env.close()


def test_own_hand_costs_affordability_and_next_card(pr):
    lay = E.scalar_layout()
    env = E.make(deck0="bait", deck1="bait", seed=8)
    obs, _ = env.reset(seed=8)
    for _ in range(9):
        obs, *_ = E.step(env, [0, 0])
    btiles = [(2, 19), (5, 19), (12, 19), (15, 19), (2, 22), (5, 22)]
    plays = 0
    for k in range(150):                   # elixir-limited: ~1 play per 17 steps after the opening
        row = obs[0]
        hand = E.hand_of(row, lay)
        nxt = E.next_of(row, lay)
        el = E.scalar(row, lay, "own_elixir")[0] * 10 * 2800
        assert np.allclose(E.scalar(row, lay, "hand_costs"), [H.COST[c] / 10 for c in hand], atol=1e-6)
        assert np.array_equal(E.scalar(row, lay, "affordable"),
                              np.array([float(H.COST[c] * 2800 <= round(el)) for c in hand], np.float32))
        tile = (9, 21) if H.CARD_KIND[hand[0]] != "building" else btiles[plays % len(btiles)]
        a = H.action_id(0, *tile)
        if E.mask_of(row)[a] < 0.5:
            obs, *_ = E.step(env, [0, 0])
            continue
        obs, *_ = E.step(env, [a, 0])
        plays += 1
        new = E.hand_of(obs[0], lay)
        assert new[0] == nxt, "the next card replaces the played slot (SPEC §4.1)"
        assert new[1:] == hand[1:]
        if plays >= 6:
            break
    assert plays >= 6
    env.close()


def test_tower_fraction_order_is_own_frame(pr):
    """Team 1 Fireballs team 0's ABSOLUTE-RIGHT princess (172 of 3052). Team 0 lists it as its
    own-frame RIGHT tower, team 1 lists it as the enemy's own-frame LEFT tower."""
    lay = E.scalar_layout()
    for seed in range(40):                  # a fixed seed list; use the first deal with Fireball
        env = E.make(deck0="giant", deck1="hog26", seed=seed)
        obs, _ = env.reset(seed=seed)
        for _ in range(9):
            obs, *_ = E.step(env, [0, 0])
        hand1 = E.hand_of(obs[1], lay)
        if H.FIREBALL in hand1:
            break
        env.close()
    else:
        raise AssertionError("no seed in 0..39 dealt Fireball into team 1's opening hand")
    s = hand1.index(H.FIREBALL)
    a = H.action_id(s, 3, 6)                # own-frame (3,6) = engine (14500, 25500) for team 1
    assert E.mask_of(obs[1])[a] > 0.5
    obs, *_ = E.step(env, [0, a])
    for _ in range(8):
        obs, *_ = E.step(env, [0, 0])
    f = np.float32((3052 - 172) / 3052)
    g0 = lambda name: E.scalar(obs[0], lay, name)
    g1 = lambda name: E.scalar(obs[1], lay, name)
    assert np.allclose(g0("own_tower_hp"), [1.0, 1.0, f], atol=1e-6), g0("own_tower_hp")
    assert np.allclose(g0("enemy_tower_hp"), [1.0, 1.0, 1.0], atol=1e-6)
    assert np.allclose(g1("own_tower_hp"), [1.0, 1.0, 1.0], atol=1e-6)
    assert np.allclose(g1("enemy_tower_hp"), [1.0, f, 1.0], atol=1e-6), g1("enemy_tower_hp")
    assert np.array_equal(g0("king_active"), np.zeros(2, np.float32))
    env.close()


def test_king_active_and_crowns_scalars(pr):
    """Team 0 Zaps team 1's King (a trigger): 71 ticks later team 1's King is active. Team 1 sees
    it as its OWN King active, team 0 as the ENEMY King active."""
    lay = E.scalar_layout()
    for seed in range(40):
        env = E.make(deck0="giant", deck1="giant", seed=seed)
        obs, _ = env.reset(seed=seed)
        for _ in range(9):
            obs, *_ = E.step(env, [0, 0])
        hand0 = E.hand_of(obs[0], lay)
        if H.ZAP in hand0:
            break
        env.close()
    else:
        raise AssertionError("no seed dealt Zap into team 0's opening hand")
    a = H.action_id(hand0.index(H.ZAP), 8, 2)       # own tile (8,2): engine (8500, 2500), on the King
    obs, *_ = E.step(env, [a, 0])
    for _ in range(8):
        obs, *_ = E.step(env, [0, 0])
    assert np.array_equal(E.scalar(obs[0], lay, "king_active"), np.array([0, 1], np.float32)), \
        "team 0 view: [own King active, enemy King active] = [0, 1] after 81 ticks"
    assert np.array_equal(E.scalar(obs[1], lay, "king_active"), np.array([1, 0], np.float32))
    assert abs(E.scalar(obs[0], lay, "enemy_tower_hp")[0] - (4824 - 48) / 4824) < 1e-6
    env.close()
