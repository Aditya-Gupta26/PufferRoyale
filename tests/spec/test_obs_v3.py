"""SPEC §16.4 observation & API changes (v0.3) and §16.5 robustness/performance with all 64 cards.

- CARD_SLOTS = 128; card identities as integer ids card_id + 1 (0 = empty) in entity rows [0],
  hand (4), next card (1) and opponent last-4 (4); opponent seen / deduced stay 128-wide multi-hots;
  opponent elixir spent / 200; own + enemy tower-troop one-hots (4 each).
- ENTITY_SHAPE = (64, 11): own slots 0-31, enemy slots 32-63, row layout
  [id+1, x/18000, y/32000, hp/max_hp, min(1, hp/2000), flying, deploying, stunned/frozen, slowed,
  is_building, target_only_buildings].
- Mask exactness with the new cards (Miner, Barbarian Barrel, new building footprints).
"""
import math
import time

import numpy as np
import pytest

import cardsv3 as C
import envkit as E
import envkit3 as E3
import gamekit as K
import helpers as H

JUMPERS = (H.HOG, H.PRINCE, C.ROYAL_HOGS)


def R():
    return E.R()


def after_lockout(env, seed):
    obs, _ = env.reset(seed=seed)
    for _ in range(9):
        obs, *_ = E.step(env, [0] * obs.shape[0])
    return obs


# ------------------------------------------------------------------------------------------
# Constants and layout
# ------------------------------------------------------------------------------------------
def test_v3_layout_constants(pr):
    m = R()
    assert int(m.CARD_SLOTS) == 128
    assert tuple(m.ENTITY_SHAPE) == (64, 11), f"ENTITY_SHAPE {m.ENTITY_SHAPE}"
    assert int(pr.N_CARDS) == 64 and len(pr.CARD_NAMES) == 64
    Cc, h, w = m.SPATIAL_SHAPE
    assert (h, w) == (32, 18) and m.MASK_SIZE == H.N_ACTIONS
    sections = sorted([(m.SPATIAL_OFFSET, Cc * h * w), (m.ENTITY_OFFSET, 64 * 11),
                       (m.SCALAR_OFFSET, m.SCALAR_SIZE), (m.MASK_OFFSET, m.MASK_SIZE)])
    end = 0
    for off, size in sections:
        assert off >= end, f"observation sections overlap at offset {off}"
        end = off + size
    assert end <= m.OBS_SIZE
    lay = E3.v3_layout()
    assert sum(n for _, n in lay.values()) == E3.V3_TOTAL == 298
    assert m.SCALAR_SIZE >= 298


def test_v3_scalar_layout_values(pr):
    """Every v0.3 scalar at its SCALAR_INDEX position during the first 12 no-op steps."""
    lay = E3.v3_layout()
    decks = {0: "golem", 1: "xbow"}
    env = E.make(deck0=decks[0], deck1=decks[1], seed=4)
    obs, _ = env.reset(seed=4)
    for k in range(13):
        if k:
            obs, *_ = E.step(env, [0, 0])
        tick = 10 * k
        units = min(16800 + 50 * tick, 28000)
        for r in (0, 1):
            row = obs[r]
            deck = C.PRESET_IDS[decks[r]]
            f = lambda name: E3.scalar(row, lay, name)
            assert abs(f("own_elixir")[0] - units / 2800 / 10) < 1e-6
            hand = E3.hand_of(row, lay)
            assert None not in hand and len(set(hand)) == 4 and all(c in deck for c in hand), hand
            assert np.allclose(f("hand_costs"), [C.cost(c) / 10 for c in hand], atol=1e-6)
            nxt = E3.next_of(row, lay)
            assert nxt in deck and nxt not in hand
            assert np.array_equal(f("affordable"),
                                  np.array([float(C.cost(c) * 2800 <= units) for c in hand], np.float32))
            assert abs(f("tick")[0] - tick / 6000) < 1e-6
            assert f("is_overtime")[0] == 0.0 and abs(f("elixir_rate")[0] - 1 / 3) < 1e-6
            assert f("lockout")[0] == float(tick < 90)
            assert np.array_equal(f("own_tower_hp"), np.ones(3, np.float32))
            assert np.array_equal(f("enemy_tower_hp"), np.ones(3, np.float32))
            assert not f("king_active").any() and not f("crowns").any()
            for name in ("opp_seen", "opp_spent", "opp_last4", "opp_deduced"):
                assert not f(name).any(), f"{name} must be zero before any opponent play"
            ub = min(10.0, 6 + 50 * tick / 2800) / 10
            assert abs(f("opp_elixir_ub")[0] - ub) < 1e-5
            want_tt = np.array([1, 0, 0, 0], np.float32)          # default tower troop: princess
            assert np.array_equal(f("own_tower_troop"), want_tt)
            assert np.array_equal(f("enemy_tower_troop"), want_tt)
    env.close()


# ------------------------------------------------------------------------------------------
# Entity rows
# ------------------------------------------------------------------------------------------
SINGLE_TROOPS = (C.PEKKA, C.BANDIT, H.MUSKETEER, C.DART_GOBLIN, C.BATTLE_RAM)


def test_v3_entity_row_layout_and_rotation(pr):
    lay = E3.v3_layout()
    env = E.make(deck0="pekka_bridge", deck1="pekka_bridge")
    obs = after_lockout(env, 2)
    hand = E3.hand_of(obs[0], lay)
    a, card = None, None
    for s, c in enumerate(hand):
        if c in SINGLE_TROOPS and E.mask_of(obs[0])[H.action_id(s, 9, 21)] > 0.5:
            a, card = H.action_id(s, 9, 21), c
            break
    assert a is not None, f"no affordable single-unit troop in hand {hand}"
    obs, *_ = E.step(env, [a, 0])
    e0, e1 = E3.entity_rows(obs[0]), E3.entity_rows(obs[1])
    assert np.abs(e0[1:]).sum() == 0, "exactly one entity: own slot 0 (towers are excluded)"
    assert np.abs(e1[:32]).sum() == 0 and np.abs(e1[33:]).sum() == 0, "team 1 sees it in enemy slot 32"
    st = C.unit_stats(C.src(card)["summon_character"])
    want = [card + 1, 9500 / 18000, 21500 / 32000, 1.0, min(1.0, st["hp"] / 2000), float(st["flying"]),
            1.0, 0.0, 0.0, 0.0, float(st["tob"])]
    assert np.allclose(e0[0], want, atol=1e-6), f"own row {e0[0]} want {want}"
    want1 = list(want)
    want1[1], want1[2] = 1 - want[1], 1 - want[2]
    assert np.allclose(e1[32], want1, atol=1e-5), f"enemy row {e1[32]} want {want1}"
    env.close()


def _seed_with_card_in_both_hands(deck, card, lay, limit=400):
    env = E.make(deck0=deck, deck1=deck)
    for s in range(limit):
        obs, _ = env.reset(seed=s)
        if card in E3.hand_of(obs[0], lay) and card in E3.hand_of(obs[1], lay):
            env.close()
            return s
    env.close()
    raise AssertionError(f"no seed < {limit} deals {card} to both hands")


def test_v3_own_and_enemy_blocks_are_32_slots(pr):
    """Both teams play Skeleton Army (15 each): own ids 9 fill slots 0-14, enemy ids 9 fill 32-46."""
    lay = E3.v3_layout()
    s = _seed_with_card_in_both_hands("bait", H.SKARMY, lay)
    env = E.make(deck0="bait", deck1="bait")
    obs = after_lockout(env, s)
    acts = []
    for r in (0, 1):
        slot = E3.hand_of(obs[r], lay).index(H.SKARMY)
        acts.append(H.action_id(slot, 9, 21))
        assert E.mask_of(obs[r])[acts[-1]] > 0.5
    obs, *_ = E.step(env, acts)
    for r in (0, 1):
        rows = E3.entity_rows(obs[r])
        nz = [i for i in range(64) if np.abs(rows[i]).sum() > 0]
        assert nz == list(range(15)) + list(range(32, 47)), f"row {r}: filled slots {nz}"
        assert set(rows[nz, E3.F_ID]) == {H.SKARMY + 1}
    env.close()


PREFERRED_TILES = [(9, 21), (3, 22), (14, 22), (3, 20), (14, 20), (9, 24)]


def legal_action_for_slot(mask, slot, prefs=PREFERRED_TILES):
    """A mask-legal action for hand slot `slot` (deal-independent): the first legal preferred own
    tile, else the first legal tile of the slot's block; None if the slot is not playable now."""
    for t in prefs:
        a = H.action_id(slot, *t)
        if mask[a] > 0.5:
            return a
    base = 1 + slot * H.N_CELLS
    blk = np.nonzero(mask[base:base + H.N_CELLS] > 0.5)[0]
    return int(base + blk[0]) if len(blk) else None


def test_v3_spawned_members_carry_their_card_id(pr):
    """Goblin Gang's 3 Goblin_Stab + 3 SpearGoblin all show id 28 + 1 = 29. The deck is cycled with
    any mask-legal non-Goblin-Gang action until Goblin Gang is playable."""
    lay = E3.v3_layout()
    env = E.make(deck0="royal_hogs", deck1="royal_hogs", deploy_lockout_ticks=0)
    obs, _ = env.reset(seed=5)
    for _ in range(300):
        hand = E3.hand_of(obs[0], lay)
        mk = E.mask_of(obs[0])
        if C.GOBLIN_GANG in hand:
            a = legal_action_for_slot(mk, hand.index(C.GOBLIN_GANG))
            if a is not None:
                obs, *_ = E.step(env, [a, 0])
                rows = E3.entity_rows(obs[0])[:32]
                assert int((rows[:, E3.F_ID] == C.GOBLIN_GANG + 1).sum()) == 6, rows[:, E3.F_ID]
                assert np.abs(E3.entity_rows(obs[0])[32:]).sum() == 0
                env.close()
                return
        a = 0
        for s, c in enumerate(hand):                           # cycle with anything else that is legal
            if c != C.GOBLIN_GANG:
                a = legal_action_for_slot(mk, s) or 0
                if a:
                    break
        obs, *_ = E.step(env, [a, 0])
    raise AssertionError("Goblin Gang never became playable in 300 steps")


def test_v3_burrowing_miner_is_public(pr):
    """§16.6.29: from the play tick the burrowing Miner appears in BOTH observations (id 48,
    deploying = 1, hp fraction 1), at 180-degree-rotated positions, moving toward the tap; it stays
    deploying = 1 through its surfacing and 20-observation deploy."""
    lay = E3.v3_layout()
    env = E.make(deck0="miner_poison", deck1="hog26", deploy_lockout_ticks=0, frame_skip=1)
    for s in range(200):
        obs, _ = env.reset(seed=s)
        hand = E3.hand_of(obs[0], lay)
        if C.MINER in hand:
            break
    else:
        raise AssertionError("no seed dealt the Miner into team 0's opening hand")
    a = H.action_id(hand.index(C.MINER), 9, 10)
    assert E.mask_of(obs[0])[a] > 0.5
    obs, *_ = E.step(env, [a, 0])
    n = math.ceil(int(math.dist((9500, 10500), H.TOWER_POS[0][0])) / 650)
    prev_y = 2.0
    for k in range(n + 20):                  # N hidden observations, then 20 deploying at the tap
        own, enemy = E3.entity_rows(obs[0])[0], E3.entity_rows(obs[1])[32]
        for row, who in ((own, "own view"), (enemy, "opponent view")):
            assert row[E3.F_ID] == C.MINER + 1, f"obs {k + 1}, {who}: row {row}"
            assert row[E3.F_DEPLOY] == 1.0 and row[E3.F_HPF] == 1.0, f"obs {k + 1}, {who}: row {row}"
            assert abs(row[E3.F_HP2K] - 1210 / 2000) < 1e-6
        assert abs(own[E3.F_X] + enemy[E3.F_X] - 1) < 1e-5 and abs(own[E3.F_Y] + enemy[E3.F_Y] - 1) < 1e-5
        if k < n - 1:
            assert own[E3.F_Y] < prev_y, "the burrowing Miner moves toward the tap (own frame, upward)"
            prev_y = own[E3.F_Y]
        obs, *_ = E.step(env, [0, 0])
    own = E3.entity_rows(obs[0])[0]
    assert own[E3.F_ID] == C.MINER + 1 and own[E3.F_DEPLOY] == 0.0, "deploy over after N + 20 observations"
    env.close()


# ------------------------------------------------------------------------------------------
# Opponent modelling with integer ids, 128-wide multi-hots, spent / 200; hidden information
# ------------------------------------------------------------------------------------------
def test_v3_card_counting_features_and_no_hand_leak(pr):
    lay = E3.v3_layout()
    deck1 = C.PRESET_IDS["miner_poison"]
    env = E.make(deck0="royal_hogs", deck1="miner_poison", seed=6)
    obs = after_lockout(env, 6)
    played, spent, k = [], 0, 9
    saw_all_at = None
    for _ in range(400):
        hand1 = E3.hand_of(obs[1], lay)
        row0 = obs[0]
        f0 = lambda name: E3.scalar(row0, lay, name)
        assert np.array_equal(f0("opp_seen"), E3.multihot(set(played))), "opponent cards seen (128 multi-hot)"
        last4 = played[::-1][:4]
        assert np.array_equal(f0("opp_last4"), E3.id_vec(last4)), "last-4 ids (card+1), most recent first"
        want_d = E3.multihot([c for c in deck1 if c not in last4]) if len(set(played)) == 8 \
            else np.zeros(C.CARD_SLOTS, np.float32)
        assert np.array_equal(f0("opp_deduced"), want_d), "deduced hand: zero until all 8 seen"
        assert abs(f0("opp_spent")[0] - spent / 200) < 1e-6, "opponent elixir spent / 200"
        income = 50 * (10 * k) / 2800
        assert abs(f0("opp_elixir_ub")[0] - min(10.0, 6 + income - spent) / 10) < 1e-5
        hand_ids = np.array([c + 1 for c in hand1], np.float32)
        assert not any(np.array_equal(row0[i:i + 4], hand_ids) for i in range(len(row0) - 3)), \
            "team 0 observes team 1's ordered hand"
        a1 = 0
        mk = E.mask_of(obs[1])
        for s in range(4):                                      # any mask-legal tile (deal-independent)
            c = hand1[s]
            a = legal_action_for_slot(mk, s) if (c not in played or len(set(played)) == 8) else None
            if a is not None:
                a1 = a
                played.append(c)
                spent += C.cost(c)
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


def test_v3_no_leak_of_opponent_deck(pr):
    """Swapping the opponent's unrevealed new-card deck leaves the other agent's obs unchanged."""
    for view, kw_a, kw_b in ((0, dict(deck0="golem", deck1="lavaloon"), dict(deck0="golem", deck1="xbow")),
                             (1, dict(deck0="pekka_bridge", deck1="royal_hogs"),
                              dict(deck0="miner_poison", deck1="royal_hogs"))):
        ea, eb = E.make(**kw_a), E.make(**kw_b)
        oa, _ = ea.reset(seed=11)
        ob, _ = eb.reset(seed=11)
        assert np.array_equal(oa[view], ob[view]), f"agent {view} sees the opponent's deck at reset"
        for k in range(30):
            oa, *_ = E.step(ea, [0, 0])
            ob, *_ = E.step(eb, [0, 0])
            assert np.array_equal(oa[view], ob[view]), f"agent {view} obs differs at step {k + 1}"
        ea.close()
        eb.close()


# ------------------------------------------------------------------------------------------
# Mask exactness with the new cards
# ------------------------------------------------------------------------------------------
def _mask_mismatches(g, team, buildings=None):
    m = np.asarray(g.legal_mask(team))
    alive = K.alive_map(g)
    bl = K.living_buildings(g) if buildings is None else buildings
    el = K.elixir(g, team)
    bad = []
    for s, c in enumerate(K.hand(g, team)):
        for ty in range(32):
            for tx in range(18):
                want = C.tile_legal_v3(c, team, tx, ty, alive, bl)
                if want is None:
                    continue                                    # spec-silent (Miner) cell
                want = bool(want) and el >= C.cost(c) * H.ELIXIR_UNIT
                if bool(m[H.action_id(s, tx, ty)]) != want:
                    bad.append((C.ALL_NAMES[c], (tx, ty), want))
    return bad


@pytest.mark.parametrize("preset", sorted(C.PRESETS))
@pytest.mark.parametrize("team", [0, 1])
def test_v3_mask_matches_oracle_new_presets(pr, preset, team):
    g = K.new_game(preset, preset, deploy_lockout_ticks=0)
    g.set_elixir(team, H.MAX_ELIXIR)
    deck = K.deck_of(g, team)
    for start in (0, 4):
        g.set_hand(team, deck[start:] + deck[:start])
        bad = _mask_mismatches(g, team)
        assert not bad, f"{preset} team {team}: {len(bad)} mask/oracle mismatches, e.g. {bad[:6]}"


def _destroy_enemy_princess(g, team, idx):
    enemy = 1 - team
    deck = K.deck_of(g, team)
    spell = next(c for c in (H.ZAP, H.ARROWS, H.FIREBALL, C.POISON, C.EARTHQUAKE, C.LIGHTNING) if c in deck)
    g.set_tower_hp(enemy, idx, 1)
    assert K.cast(g, team, spell, *H.TOWER_POS[enemy][idx]) == H.OK
    K.run_until(g, lambda gg: not K.tower_alive(gg, enemy, idx), 120, "tower destruction")


@pytest.mark.parametrize("preset", ["miner_poison", "royal_hogs", "golem"])
@pytest.mark.parametrize("team", [0, 1])
def test_v3_mask_with_pocket_and_new_buildings(pr, preset, team):
    """Enemy left Princess destroyed (pocket opens, Miner may use its old footprint); one living
    Inferno Tower / X-Bow / Tombstone of each team block their footprints for troops/buildings."""
    g = K.new_game(preset, preset, deploy_lockout_ticks=0)
    _destroy_enemy_princess(g, team, 1)
    for t, cid, own_tile in ((team, C.INFERNO_TOWER, (6, 24)), (1 - team, C.XBOW, (11, 21)),
                             (team, C.TOMBSTONE, (14, 28))):
        x, y = H.own_to_engine_point(team, *H.building_centre_own(*own_tile, 3))
        K.spawn1(g, t, cid, x, y, deployed=True)
    g.set_elixir(team, H.MAX_ELIXIR)
    deck = K.deck_of(g, team)
    for start in (0, 4):
        g.set_hand(team, deck[start:] + deck[:start])
        bad = _mask_mismatches(g, team)
        assert not bad, f"{preset} team {team}: {len(bad)} mismatches, e.g. {bad[:6]}"


def test_v3_env_mask_matches_oracle(pr):
    lay = E3.v3_layout()
    env = E.make(deck0="miner_poison", deck1="golem")
    obs = after_lockout(env, 3)
    alive = H.all_alive()
    for r in (0, 1):
        hand = E3.hand_of(obs[r], lay)
        el = round(E3.scalar(obs[r], lay, "own_elixir")[0] * 10 * 2800)
        mk = E.mask_of(obs[r])
        assert mk[0] == 1.0
        bad = []
        for s, c in enumerate(hand):
            for ty in range(32):
                for tx in range(18):
                    want = C.tile_legal_v3(c, r, tx, ty, alive)
                    if want is None:
                        continue
                    want = bool(want) and el >= C.cost(c) * 2800
                    if bool(mk[H.action_id(s, tx, ty)] > 0.5) != want:
                        bad.append((C.ALL_NAMES[c], (tx, ty), want))
        assert not bad, f"row {r}: {bad[:6]}"
    env.close()


# ------------------------------------------------------------------------------------------
# Robustness: random 64-card decks
# ------------------------------------------------------------------------------------------
def random_play(g, rng, team, p=0.5):
    if rng.random() >= p:
        return
    m = np.asarray(g.legal_mask(team))
    leg = np.nonzero(m[1:])[0] + 1
    if len(leg):
        s, tx, ty = H.decode_action(int(rng.choice(leg)))
        code = g.play_tile(team, s, tx, ty)
        assert code == H.OK, f"legal_mask allowed slot {s} tile {(tx, ty)} but play returned {code}"


def check_invariants(g, decks):
    for t in (0, 1):
        assert 0 <= K.elixir(g, t) <= H.MAX_ELIXIR
        assert sorted(K.hand(g, t) + K.queue(g, t)) == decks[t]
        assert 0 <= K.crowns(g)[t] <= 3
        for i in range(3):
            tw = K.tower(g, t, i)
            assert 0 <= int(tw["hp"]) <= int(tw["max_hp"])
    es = K.ents(g)
    assert len(es) <= 256, "MAX_ENTITIES"
    ids = [int(e["id"]) for e in es]
    assert len(ids) == len(set(ids)), "duplicate entity ids"
    for e in es:
        assert 0 < int(e["hp"]) <= int(e["max_hp"]), f"{e['unit']} hp {e['hp']}/{e['max_hp']}"
        assert 0 <= int(e["card_id"]) < 64 or e["kind"] == "tower"
        x, y = int(e["x"]), int(e["y"])
        assert 0 <= x < H.ARENA_W and 0 <= y < H.ARENA_H, f"{e['unit']} outside the arena at {(x, y)}"
        if (e["kind"] == "troop" and not bool(e["flying"]) and not bool(e["hidden"])
                and int(e["card_id"]) not in JUMPERS):
            assert not H.point_in_water(x, y), f"ground troop {e['unit']} in water at {(x, y)}"


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(12))
def test_v3_soak_random_64_card_decks(pr, seed):
    g = K.new_game("random", "random", seed=seed)
    decks = {t: sorted(K.deck_of(g, t)) for t in (0, 1)}
    assert any(c >= 21 for c in decks[0] + decks[1]), f"random decks {decks} contain no §16 card"
    rng = np.random.default_rng(500 + seed)
    p = [0.6, 0.3, 0.15][seed % 3]
    for _ in range(6001):
        if K.over(g):
            break
        if K.tick_of(g) % 5 == 0:
            for t in (0, 1):
                random_play(g, rng, t, p)
        g.tick(1)
        check_invariants(g, decks)
    assert K.over(g) and K.tick_of(g) <= 6000
    r0 = K.result(g, 0)
    assert (K.end_reason(g) == "DRAW") == (r0 == 0)


def test_v3_random_decks_are_deterministic(pr):
    a, b = K.new_game("random", "random", seed=77), K.new_game("random", "random", seed=77)
    assert K.deck_of(a, 0) == K.deck_of(b, 0) and K.deck_of(a, 1) == K.deck_of(b, 1)
    ra, rb = np.random.default_rng(1), np.random.default_rng(1)
    for _ in range(600):
        if K.tick_of(a) % 5 == 0:
            for t in (0, 1):
                random_play(a, ra, t, 0.5)
                random_play(b, rb, t, 0.5)
        a.tick(1)
        b.tick(1)
        assert a.hash() == b.hash()


@pytest.mark.slow
@pytest.mark.parametrize("bot", ["random", "heuristic"])
@pytest.mark.parametrize("side", [0, 1])
def test_v3_bots_play_only_legal_actions_with_any_card(pr, bot, side):
    """The learner only no-ops, so illegal_actions counts the bot alone (§13.14); the bot's played
    cards (seen through opp_seen) must include §16 cards."""
    lay = E3.v3_layout()
    rng = np.random.default_rng(20 + side)
    plays, seen = 0, np.zeros(C.CARD_SLOTS, np.float32)
    for rep in range(3):
        env = E.make(num_agents=1, opponent=bot, learner_side=side, deck0="random", deck1="random",
                     log_interval=1, bot_play_prob=0.5, seed=rep)
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        for _ in range(700):
            seen = np.maximum(seen, E3.scalar(obs[0], lay, "opp_seen"))
            obs, rew, term, trunc, infos = E.step(env, [0])
            if term.any():
                lg = E.logs_in(infos)[-1]
                assert lg["illegal_actions"] == 0, f"{bot} bot issued illegal actions"
                plays += lg[f"plays_{1 - side}"]
                break
        else:
            raise AssertionError("match did not end within 700 steps")
        env.close()
    assert plays > 0
    assert seen[21:64].any(), f"the {bot} bot never played a §16 card in 3 random-deck matches"
    assert not seen[64:].any(), "opp_seen slots 64..127 are unused padding"


# ------------------------------------------------------------------------------------------
# Performance (§16.5)
# ------------------------------------------------------------------------------------------
@pytest.mark.perf
def test_v3_engine_ticks_per_second_random_decks(pr):
    rng = np.random.default_rng(0)
    best = 0.0
    for rep in range(5):
        g = K.new_game("random", "random", seed=100 + rep)
        assert any(c >= 21 for t in (0, 1) for c in K.deck_of(g, t)), "random decks must include §16 cards"
        engine_time, ticks = 0.0, 0
        while not K.over(g) and ticks < 6000:
            for t in (0, 1):
                random_play(g, rng, t, 0.6)
            t0 = time.perf_counter()
            g.tick(20)
            engine_time += time.perf_counter() - t0
            ticks += 20
        if rep > 0:
            best = max(best, ticks / engine_time)
    assert best >= 20000, f"engine throughput with random 64-card decks {best:.0f} ticks/s < 20,000"


def _stress_v3():
    g = K.new_game("hog26", "hog26", seed=0)
    for team in (0, 1):
        f = (lambda x, y: (x, y)) if team == 0 else (lambda x, y: (H.ARENA_W - x, H.ARENA_H - y))
        for x in (2500, 6500, 11500, 15500):
            K.spawn(g, team, H.SKARMY, *f(x, 19500), deployed=True)
        for x in (4500, 13500):
            K.spawn(g, team, C.MINION_HORDE, *f(x, 21000), deployed=True)
            K.spawn(g, team, C.GOBLIN_GANG, *f(x, 22500), deployed=True)
        K.spawn(g, team, C.WITCH, *f(9000, 23000), deployed=True)
    return g


def test_v3_stress_deterministic(pr):
    a, b = _stress_v3(), _stress_v3()
    for _ in range(400):
        a.tick(1)
        b.tick(1)
        assert a.hash() == b.hash()


@pytest.mark.perf
def test_v3_stress_throughput(pr):
    best = 0.0
    for rep in range(3):
        g = _stress_v3()
        g.tick(20)
        t0 = time.perf_counter()
        g.tick(600)
        best = max(best, 600 / (time.perf_counter() - t0))
    assert best >= 10000, f"new-card large battle runs at {best:.0f} ticks/s < 10,000"
