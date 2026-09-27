"""SPEC §4.1 hand & cycle: FIFO queue, slot replacement, permutation invariant, seeded shuffles."""
import random

import pytest

import gamekit as K
import helpers as H

ANY_TILE = (9, 21)   # own-frame tile legal for every card kind (troop/building/spell/barrel)
# mutually non-overlapping building tiles (Cannon F=3 centred, Tesla F=2 corner-anchored), none of
# whose footprints covers ANY_TILE (v0.2: troops may not be placed on building footprints)
BUILDING_TILES = [(2, 19), (5, 19), (12, 19), (15, 19), (2, 22), (5, 22), (13, 22), (16, 22),
                  (1, 28), (4, 28), (12, 28), (15, 28)]


def test_any_tile_is_legal_for_every_kind():
    a = H.all_alive()
    for c in range(21):
        assert H.tile_legal(c, 0, *ANY_TILE, a) is True
    btiles = BUILDING_TILES
    placed = []
    for bt in btiles:
        assert H.building_tile_legal(0, *bt, H.CANNON, a, placed) is True, bt
        assert H.building_tile_legal(0, *bt, H.TESLA, a, placed) is True, bt
        placed.append((*H.building_centre_engine(0, *bt, H.CANNON), 3))
        assert H.troop_tile_legal(0, *ANY_TILE, a, placed), f"{bt} covers ANY_TILE"


@pytest.mark.parametrize("deck", ["hog26", "giant", "bait"])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_initial_hand_queue_is_permutation_of_deck(pr, deck, seed):
    g = K.new_game(deck, deck, seed=seed)
    for t in (0, 1):
        h, q = K.hand(g, t), K.queue(g, t)
        assert len(h) == 4 and len(q) == 4
        assert sorted(h + q) == sorted(H.DECKS[deck])


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_play_replaces_slot_with_next_card(pr, team, slot):
    g = K.new_game("bait", "bait", seed=3, deploy_lockout_ticks=0)
    g.set_elixir(team, H.MAX_ELIXIR)
    h, q = K.hand(g, team), K.queue(g, team)
    c = h[slot]
    assert g.play_tile(team, slot, *ANY_TILE) == H.OK
    g.tick(1)
    h2, q2 = K.hand(g, team), K.queue(g, team)
    want_h = list(h)
    want_h[slot] = q[0]
    assert h2 == want_h, f"hand after playing slot {slot}: {h2}, want {want_h}"
    assert q2 == q[1:] + [c], f"queue after play: {q2}, want {q[1:] + [c]}"
    other = 1 - team
    assert sorted(K.hand(g, other) + K.queue(g, other)) == sorted(H.DECKS["bait"])


def test_set_hand_semantics(pr):
    g = K.new_game("giant", "giant")
    order = list(H.DECKS["giant"])[::-1]
    g.set_hand(0, order)
    assert K.hand(g, 0) == order[:4] and K.queue(g, 0) == order[4:]


@pytest.mark.parametrize("deck", ["hog26", "giant", "bait"])
def test_long_random_cycle_matches_fifo_model(pr, deck):
    rng = random.Random(1234)
    g = K.new_game(deck, deck, seed=11, deploy_lockout_ticks=0)
    model = {t: (K.hand(g, t), K.queue(g, t)) for t in (0, 1)}
    # mutually non-overlapping building tiles (3 apart), all legal for Cannon and Tesla
    btiles = BUILDING_TILES
    bnext = {0: 0, 1: 0}
    for step in range(40):
        plays = {}
        for t in (0, 1):
            if rng.random() < 0.7:
                slot = rng.randrange(4)
                g.set_elixir(t, H.MAX_ELIXIR)
                tile = ANY_TILE
                if H.CARD_KIND[K.hand(g, t)[slot]] == "building":
                    tile = btiles[bnext[t]]
                    bnext[t] += 1
                code = g.play_tile(t, slot, *tile)
                assert code == H.OK, f"step {step} team {t} slot {slot}: code {code}"
                plays[t] = slot
        g.tick(1)
        for t, slot in plays.items():
            h, q = model[t]
            c = h[slot]
            h = list(h)
            h[slot] = q[0]
            model[t] = (h, q[1:] + [c])
        for t in (0, 1):
            assert (K.hand(g, t), K.queue(g, t)) == model[t], f"step {step} team {t}"
            assert sorted(K.hand(g, t) + K.queue(g, t)) == sorted(H.DECKS[deck])
        if K.over(g):
            break


def test_shuffle_deterministic_by_seed(pr):
    for seed in (0, 5, 99):
        a = K.new_game("hog26", "bait", seed=seed)
        b = K.new_game("hog26", "bait", seed=seed)
        for t in (0, 1):
            assert K.deck_of(a, t) == K.deck_of(b, t)


def test_shuffle_varies_across_seeds(pr):
    orders = set()
    for seed in range(20):
        g = K.new_game("hog26", "hog26", seed=seed)
        orders.add(tuple(K.deck_of(g, 0)))
    assert len(orders) >= 15, f"only {len(orders)} distinct team-0 deck orders over 20 seeds"


def test_shuffle_is_roughly_uniform(pr):
    """Each card should open in slot 0 about 1/8 of the time (400 fixed seeds; bounds ~5 sigma)."""
    counts = {c: 0 for c in H.DECKS["giant"]}
    for seed in range(400):
        g = K.new_game("giant", "giant", seed=seed)
        counts[K.hand(g, 0)[0]] += 1
    for c, n in counts.items():
        assert 15 <= n <= 90, f"card {H.CARD_NAMES[c]} opened slot 0 in {n}/400 seeded deals"


def test_reset_with_seed_redeals_like_fresh_game(pr):
    g = K.new_game("hog26", "giant", seed=3, deploy_lockout_ticks=0)
    g.set_elixir(0, H.MAX_ELIXIR)
    g.play_tile(0, 0, *ANY_TILE)
    g.tick(50)
    g.reset(seed=8)
    f = K.new_game("hog26", "giant", seed=8, deploy_lockout_ticks=0)
    assert K.tick_of(g) == 0
    for t in (0, 1):
        assert K.deck_of(g, t) == K.deck_of(f, t)
        assert K.elixir(g, t) == H.START_ELIXIR
    assert len(K.ents(g)) == 6
    assert g.hash() == f.hash(), "reset(seed) must reproduce the fresh-game state bit-exactly"
