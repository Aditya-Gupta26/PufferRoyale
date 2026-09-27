"""SPEC §3 arena + §3.1 placement legality, for both seats (180-degree own frame), including
pockets, building rules, Goblin Barrel water rule and the exact legality mask (§9.4 / §10).

The reference oracle (helpers.troop_tile_legal / building_tile_legal) is built from the pinned
tilemap and the SPEC §3 tower geometry only.
"""
import numpy as np
import pytest

import gamekit as K
import helpers as H


# ------------------------------------------------------------------------------------------
# Tester self-checks on the data (not on the engine): these guard the oracle itself.
# ------------------------------------------------------------------------------------------
def test_tilemap_is_180_degree_symmetric():
    g = H.arena_grid()
    water = (g & H.WATER_BIT) > 0
    nod = (g & H.NODEPLOY_BIT) > 0
    assert (water == water[::-1, ::-1]).all()
    assert (nod == nod[::-1, ::-1]).all()


def test_river_and_bridges_match_spec_geometry():
    for x in range(0, 18000, 250):
        for y in (15000, 15999, 16500, 16999):
            on_bridge = 2500 <= x < 4500 or 13500 <= x < 15500
            assert H.point_in_water(x, y) == (not on_bridge), (x, y)
        assert not H.point_in_water(x, 14999) and not H.point_in_water(x, 17000)


def test_oracle_v02_anchor_and_log_rules():
    a = H.all_alive()
    # even-F (Tesla) footprint = own-frame tiles [tx-1, tx] x [ty-1, ty]
    assert H.building_centre_engine(0, 9, 21, H.TESLA) == (9000, 21000)
    assert H.building_centre_engine(1, 9, 21, H.TESLA) == (9000, 11000)
    assert H.building_centre_engine(1, 9, 21, H.CANNON) == (8500, 10500)
    assert H.building_tile_legal(0, 17, 19, H.TESLA, a) and not H.building_tile_legal(0, 17, 18, H.TESLA, a)
    # the Log: whole own half (incl. towers/nodeploy), never the enemy half without a pocket
    assert all(H.log_tile_legal(0, tx, ty, a) for tx in range(18) for ty in range(17, 32))
    assert not any(H.log_tile_legal(0, tx, ty, a) for tx in range(18) for ty in range(17))


def test_oracle_pockets_match_spec_consequence():
    a = H.all_alive()
    a[1][1] = False
    opened = {(tx, ty) for tx in range(18) for ty in range(17) if H.troop_tile_legal(0, tx, ty, a)}
    want = {(tx, ty) for tx in range(9) for ty in range(11, 15)} - {(0, 14)}
    assert opened == want


# ------------------------------------------------------------------------------------------
# Mask == reference oracle
# ------------------------------------------------------------------------------------------
def compare_mask(g, team, lockout=H.LOCKOUT_TICKS):
    got = np.asarray(g.legal_mask(team))
    assert got.shape == (H.N_ACTIONS,), f"legal_mask shape {got.shape}"
    assert set(np.unique(got).tolist()) <= {0, 1}, "mask values must be 0/1"
    hand = K.hand(g, team)
    exp = H.expected_mask(hand, K.elixir(g, team), K.tick_of(g), team, K.alive_map(g), lockout,
                          K.over(g), K.living_buildings(g))
    bad = np.nonzero(got.astype(np.int64) != exp.astype(np.int64))[0]
    if len(bad):
        lines = []
        for a in bad[:25]:
            if a == 0:
                lines.append(f"a=0 got {got[0]}")
                continue
            s, tx, ty = H.decode_action(int(a))
            lines.append(f"a={a} slot={s} card={H.CARD_NAMES[hand[s]]} own_tile=({tx},{ty}) "
                         f"got={int(got[a])} want={int(exp[a])}")
        raise AssertionError(f"team {team}: {len(bad)} mask entries differ from SPEC §3.1:\n" +
                             "\n".join(lines))
    return got


def ready_game(deck0, deck1, seed=0):
    g = K.new_game(deck0, deck1, seed=seed)
    g.tick(H.LOCKOUT_TICKS)
    g.set_elixir(0, H.MAX_ELIXIR)
    g.set_elixir(1, H.MAX_ELIXIR)
    return g


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("deck", ["hog26", "giant", "bait"])
def test_mask_equals_spec_legality_opening(pr, deck, team, seed):
    g = ready_game(deck, deck, seed)
    compare_mask(g, team)


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("deck", ["hog26", "bait"])
def test_mask_all_slots_every_card(pr, deck, team):
    """Rotate each of the 8 deck cards through slot 0 so every card's rule is compared."""
    g = ready_game(deck, deck)
    d = K.deck_of(g, team)
    for c in d:
        K.put_first(g, team, c)
        compare_mask(g, team)


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("deck", ["hog26", "giant", "bait"])
def test_mask_with_enemy_princesses_down(pr, deck, team):
    g = ready_game(deck, deck)
    enemy = 1 - team
    K.destroy_tower(g, enemy, 1)
    g.set_elixir(team, H.MAX_ELIXIR)
    compare_mask(g, team)
    K.destroy_tower(g, enemy, 2)
    g.set_elixir(team, H.MAX_ELIXIR)
    for c in K.deck_of(g, team):
        K.put_first(g, team, c)
        compare_mask(g, team)


@pytest.mark.parametrize("team", [0, 1])
def test_building_allowed_on_destroyed_own_princess(pr, team):
    """v0.2 §3.1 rule 6: a destroyed tower's footprint no longer blocks anything."""
    g = ready_game("hog26", "hog26")
    ox, oy = H.engine_to_own_point(team, *H.TOWER_POS[team][1])
    tile = (ox // 1000, oy // 1000)
    assert K.cast_tile(g, team, H.CANNON, *tile) == H.ILLEGAL_POSITION
    K.destroy_tower(g, team, 1)
    g.set_elixir(team, H.MAX_ELIXIR)
    assert K.cast_tile(g, team, H.CANNON, *tile) == H.OK
    compare_mask(g, team)


@pytest.mark.parametrize("team", [0, 1])
def test_mask_own_princess_down_frees_its_footprint_for_troops(pr, team):
    g = ready_game("giant", "giant")
    K.destroy_tower(g, team, 1)
    g.set_elixir(team, H.MAX_ELIXIR)
    got = compare_mask(g, team)
    # own-frame tiles of the destroyed (absolute-left) princess are now troop-legal
    K.put_first(g, team, H.KNIGHT)
    got = np.asarray(g.legal_mask(team))
    ox, oy = H.engine_to_own_point(team, *H.TOWER_POS[team][1])
    for tx in range(ox // 1000 - 1, ox // 1000 + 2):
        for ty in range(oy // 1000 - 1, oy // 1000 + 2):
            assert got[H.action_id(0, tx, ty)] == 1, (tx, ty)


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("deck,card", [("hog26", H.CANNON), ("bait", H.TESLA)])
def test_mask_with_own_building(pr, team, deck, card):
    g = ready_game(deck, deck)
    code = K.cast_tile(g, team, card, 9, 21)
    assert code == H.OK
    g.tick(1)
    bs = K.units(g, team=team, card=card)
    assert len(bs) == 1
    assert K.pos(bs[0]) == H.building_centre_engine(team, 9, 21, card), \
        "odd F: tile centre; even F (Tesla): the tapped tile's own-frame top-left corner (v0.2)"
    for c in K.deck_of(g, team):
        K.put_first(g, team, c)
        g.set_elixir(team, H.MAX_ELIXIR)
        compare_mask(g, team)


@pytest.mark.parametrize("team", [0, 1])
def test_mask_with_enemy_building_in_own_half(pr, team):
    """Troops may not be placed on ANY living building footprint, own or enemy (v0.2)."""
    g = ready_game("giant", "giant")
    ex, ey = H.tile_centre_engine(team, 12, 20)
    K.spawn1(g, 1 - team, H.CANNON, ex, ey, deployed=True)
    g.set_elixir(team, H.MAX_ELIXIR)
    got = compare_mask(g, team)
    K.put_first(g, team, H.KNIGHT)
    got = np.asarray(g.legal_mask(team))
    for tx in (11, 12, 13):
        for ty in (19, 20, 21):
            assert got[H.action_id(0, tx, ty)] == 0, (tx, ty)
    assert got[H.action_id(0, 14, 20)] == 1


# ------------------------------------------------------------------------------------------
# play_tile error codes (own-frame tiles; identical expectations for both seats)
# ------------------------------------------------------------------------------------------
TROOP_TILES = [
    ((9, 21), H.OK), ((6, 31), H.OK), ((0, 18), H.OK), ((17, 30), H.OK), ((4, 23), H.OK),
    ((5, 25), H.OK), ((1, 17), H.OK),
    ((9, 16), H.ILLEGAL_POSITION),   # river water
    ((3, 16), H.ILLEGAL_POSITION),   # own-frame bridge: river band is never troop territory
    ((3, 15), H.ILLEGAL_POSITION),
    ((9, 14), H.ILLEGAL_POSITION),   # enemy half, no pocket
    ((4, 12), H.ILLEGAL_POSITION),   # would-be pocket while the enemy princess lives
    ((3, 25), H.ILLEGAL_POSITION),   # own princess footprint
    ((9, 28), H.ILLEGAL_POSITION),   # own king footprint
    ((0, 17), H.ILLEGAL_POSITION),   # river-bank corner no-deploy
    ((1, 31), H.ILLEGAL_POSITION),   # back-corner no-deploy strip
    ((12, 31), H.ILLEGAL_POSITION),
    ((9, 3), H.ILLEGAL_POSITION),    # enemy king
]


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("tile,code", TROOP_TILES)
def test_troop_tile_codes(pr, team, tile, code):
    g = ready_game("giant", "giant")
    assert H.troop_tile_legal(team, *tile, H.all_alive()) == (code == H.OK)  # oracle agrees
    got = K.cast_tile(g, team, H.KNIGHT, *tile)
    assert got == code, f"team {team} Knight at own tile {tile}: got {got}, want {code}"


BUILDING_TILES = [
    ((9, 21), H.OK), ((9, 18), H.OK), ((1, 20), H.OK), ((6, 24), H.OK), ((1, 28), H.OK),
    ((9, 17), H.ILLEGAL_POSITION),   # footprint reaches into the river band
    ((0, 20), H.ILLEGAL_POSITION),   # footprint leaves the arena
    ((5, 24), H.ILLEGAL_POSITION),   # overlaps own princess footprint
    ((9, 26), H.ILLEGAL_POSITION),   # overlaps own king footprint
    ((6, 30), H.ILLEGAL_POSITION),   # touches back no-deploy strip / king
    ((9, 14), H.ILLEGAL_POSITION),   # enemy half
]


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("tile,code", BUILDING_TILES)
def test_cannon_tile_codes(pr, team, tile, code):
    g = ready_game("hog26", "hog26")
    assert bool(H.building_tile_legal(team, *tile, H.CANNON, H.all_alive())) == (code == H.OK)
    got = K.cast_tile(g, team, H.CANNON, *tile)
    assert got == code, f"team {team} Cannon at own tile {tile}: got {got}, want {code}"


TESLA_TILES = [
    ((9, 18), H.OK), ((9, 17), H.ILLEGAL_POSITION),     # footprint rows 16-17 reach the river
    ((1, 23), H.OK), ((2, 24), H.ILLEGAL_POSITION),     # columns 1-2 touch the princess (2-4)
    ((6, 25), H.OK), ((17, 19), H.OK), ((17, 18), H.ILLEGAL_POSITION),   # (17,17) is no-deploy
    ((0, 20), H.ILLEGAL_POSITION), ((7, 27), H.ILLEGAL_POSITION), ((6, 27), H.OK),
    ((9, 31), H.ILLEGAL_POSITION), ((1, 30), H.OK), ((1, 31), H.ILLEGAL_POSITION),
]


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("tile,code", TESLA_TILES)
def test_tesla_tile_codes(pr, team, tile, code):
    g = ready_game("bait", "bait")
    assert bool(H.building_tile_legal(team, *tile, H.TESLA, H.all_alive())) == (code == H.OK)
    got = K.cast_tile(g, team, H.TESLA, *tile)
    assert got == code, f"team {team} Tesla at own tile {tile}: got {got}, want {code}"
    if code == H.OK:
        g.tick(1)
        t = K.units(g, team=team, card=H.TESLA)
        assert [K.pos(e) for e in t] == [H.building_centre_engine(team, *tile, H.TESLA)]


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("tile", [(9, 2), (9, 16), (0, 0), (17, 31), (3, 6), (9, 28)])
def test_spells_legal_on_any_tile(pr, team, tile):
    g = ready_game("hog26", "giant")
    assert K.cast_tile(g, team, H.FIREBALL if team == 0 else H.ZAP, *tile) == H.OK
    g2 = ready_game("giant", "hog26")
    assert K.cast_tile(g2, team, H.ARROWS if team == 0 else H.FIREBALL, *tile) == H.OK


LOG_TILES = [
    ((9, 21), H.OK), ((9, 28), H.OK), ((3, 25), H.OK), ((1, 31), H.OK), ((0, 17), H.OK),
    ((9, 17), H.OK), ((17, 31), H.OK),
    ((9, 16), H.ILLEGAL_POSITION), ((3, 16), H.ILLEGAL_POSITION), ((3, 15), H.ILLEGAL_POSITION),
    ((9, 14), H.ILLEGAL_POSITION), ((4, 12), H.ILLEGAL_POSITION), ((9, 2), H.ILLEGAL_POSITION),
]


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("tile,code", LOG_TILES)
def test_log_tile_codes(pr, team, tile, code):
    """v0.2: the Log is troop territory only, not water, but may overlap towers/buildings/no-deploy."""
    g = ready_game("hog26", "hog26")
    assert H.log_tile_legal(team, *tile, H.all_alive()) == (code == H.OK)
    got = K.cast_tile(g, team, H.LOG, *tile)
    assert got == code, f"team {team} Log at own tile {tile}: got {got}, want {code}"


@pytest.mark.parametrize("team", [0, 1])
def test_log_allowed_in_opened_pocket(pr, team):
    g = ready_game("hog26", "hog26")
    K.destroy_tower(g, 1 - team, 1 if team == 0 else 2)
    g.set_elixir(team, H.MAX_ELIXIR)
    assert K.cast_tile(g, team, H.LOG, 4, 12) == H.OK
    g.tick(1)
    assert K.cast_tile(g, team, H.LOG, 13, 12) == H.ILLEGAL_POSITION
    assert K.cast_tile(g, team, H.LOG, 4, 10) == H.ILLEGAL_POSITION


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("tile,code", [((9, 16), H.ILLEGAL_POSITION), ((0, 15), H.ILLEGAL_POSITION),
                                       ((5, 16), H.ILLEGAL_POSITION), ((3, 16), H.OK), ((3, 15), H.OK),
                                       ((14, 15), H.OK), ((9, 3), H.OK), ((3, 6), H.OK), ((0, 0), H.OK),
                                       ((9, 14), H.OK)])
def test_goblin_barrel_not_on_water(pr, team, tile, code):
    g = ready_game("bait", "bait")
    got = K.cast_tile(g, team, H.GOBLIN_BARREL, *tile)
    assert got == code, f"team {team} Goblin Barrel at own tile {tile}: got {got}, want {code}"


@pytest.mark.parametrize("team", [0, 1])
def test_pocket_opens_only_for_troops_and_only_on_fallen_side(pr, team):
    g = ready_game("hog26", "hog26")
    enemy = 1 - team
    # The enemy tower on the acting team's own-frame LEFT is the enemy's absolute-left (idx 1)
    # for team 0 and the enemy's absolute-right (idx 2) for team 1.
    idx_own_left = 1 if team == 0 else 2
    assert K.cast_tile(g, team, H.HOG, 4, 12) == H.ILLEGAL_POSITION
    K.destroy_tower(g, enemy, idx_own_left)
    g.set_elixir(team, H.MAX_ELIXIR)
    assert K.cast_tile(g, team, H.HOG, 4, 12) == H.OK, "own-frame left pocket must be open"
    g.tick(1)
    assert K.cast_tile(g, team, H.HOG, 13, 12) == H.ILLEGAL_POSITION, "right side still closed"
    assert K.cast_tile(g, team, H.HOG, 4, 10) == H.ILLEGAL_POSITION, "inside the enemy king rect"
    assert K.cast_tile(g, team, H.HOG, 3, 16) == H.ILLEGAL_POSITION, "bridge stays closed"
    assert K.cast_tile(g, team, H.CANNON, 4, 12) == H.ILLEGAL_POSITION, "no pocket for buildings"


@pytest.mark.parametrize("team", [0, 1])
def test_troop_lands_on_tile_centre(pr, team):
    g = ready_game("giant", "giant")
    assert K.cast_tile(g, team, H.KNIGHT, 9, 21) == H.OK
    assert K.troops(g, team) == [], "a queued play takes effect on the NEXT tick"
    g.tick(1)
    ks = K.troops(g, team, H.KNIGHT)
    assert len(ks) == 1
    assert K.pos(ks[0]) == H.tile_centre_engine(team, 9, 21)
    assert bool(ks[0]["deploying"]) is True


def test_play_engine_coordinates_team1(pr):
    """play() takes ENGINE coordinates for both teams (SPEC §10)."""
    g = ready_game("giant", "giant")
    # engine (8500, 10500) is team 1's own tile (9, 21)
    assert K.cast(g, 1, H.KNIGHT, 8500, 10500) == H.OK
    g.tick(1)
    assert [K.pos(e) for e in K.troops(g, 1, H.KNIGHT)] == [(8500, 10500)]
    # engine (9500, 21500) is in team 0's half: illegal for team 1
    g2 = ready_game("giant", "giant")
    assert K.cast(g2, 1, H.KNIGHT, 9500, 21500) == H.ILLEGAL_POSITION


@pytest.mark.parametrize("slot", [-1, 4, 7])
def test_bad_slot(pr, slot):
    g = ready_game("giant", "giant")
    assert g.play_tile(0, slot, 9, 21) == H.NOT_IN_HAND
    assert g.play(0, slot, 9500, 21500) == H.NOT_IN_HAND


@pytest.mark.parametrize("team", [0, 1])
def test_building_on_building_refused(pr, team):
    g = ready_game("hog26", "hog26")
    assert K.cast_tile(g, team, H.CANNON, 9, 21) == H.OK
    g.tick(1)
    for tile, code in [((10, 21), H.ILLEGAL_POSITION), ((9, 23), H.ILLEGAL_POSITION),
                       ((11, 22), H.ILLEGAL_POSITION), ((12, 21), H.OK)]:
        got = K.cast_tile(g, team, H.CANNON, *tile)
        assert got == code, f"Cannon at {tile} next to a Cannon at (9,21): got {got} want {code}"
        if got == H.OK:
            g.tick(1)
    g.tick(1)
    assert len(K.units(g, team=team, card=H.CANNON)) == 2
    # v0.2: troops may not be placed on a living building footprint (Cannon at (9,21) covers
    # own tiles 8-10 x 20-22); the next row is fine
    K.put_first(g, team, H.HOG)
    g.set_elixir(team, H.MAX_ELIXIR)
    for tile in ((9, 21), (8, 20), (10, 22)):
        assert g.play_tile(team, 0, *tile) == H.ILLEGAL_POSITION, tile
    assert g.play_tile(team, 0, 9, 23) == H.OK
    g.tick(1)
    # ...while the Log (a spell) may overlap buildings (v0.2 §3.1 rule 5)
    assert K.cast_tile(g, team, H.LOG, 9, 21) == H.OK


# ------------------------------------------------------------------------------------------
# legal_mask == what play_tile accepts (all 2304 card actions), and accepted == applied
# ------------------------------------------------------------------------------------------
def cross_check_mask_vs_play(g, team):
    snap = g.snapshot()
    mask = np.asarray(g.legal_mask(team))
    bad = []
    for a in range(1, H.N_ACTIONS):
        g.restore(snap)
        s, tx, ty = H.decode_action(a)
        code = g.play_tile(team, s, tx, ty)
        if (code == H.OK) != bool(mask[a]):
            bad.append((a, s, tx, ty, int(mask[a]), code))
    g.restore(snap)
    assert not bad, f"team {team}: legal_mask disagrees with play_tile on {len(bad)} actions: {bad[:15]}"


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("deck", ["hog26", "giant", "bait"])
def test_legal_mask_equals_play_acceptance(pr, deck, team):
    g = ready_game(deck, deck, seed=4)
    cross_check_mask_vs_play(g, team)
    K.destroy_tower(g, 1 - team, 2)
    g.set_elixir(team, 9000)          # some cards unaffordable now
    cross_check_mask_vs_play(g, team)


@pytest.mark.parametrize("team", [0, 1])
def test_accepted_plays_are_applied(pr, team):
    """Behavioural check: a mask-legal action spends elixir and cycles the card on the next tick;
    a mask-illegal one changes nothing."""
    rng = np.random.default_rng(7 + team)
    g = ready_game("bait", "hog26", seed=1)
    snap = g.snapshot()
    mask = np.asarray(g.legal_mask(team))
    legal = np.nonzero(mask[1:])[0] + 1
    illegal = np.nonzero(mask[1:] == 0)[0] + 1
    for a in list(rng.choice(legal, 25, replace=False)) + list(rng.choice(illegal, 25, replace=False)):
        g.restore(snap)
        s, tx, ty = H.decode_action(int(a))
        hand, e0 = K.hand(g, team), K.elixir(g, team)
        card = hand[s]
        code = g.play_tile(team, s, tx, ty)
        g.tick(1)
        if mask[a]:
            assert code == H.OK
            assert K.elixir(g, team) == min(H.MAX_ELIXIR, e0 + 50) - H.COST[card] * H.ELIXIR_UNIT or \
                K.elixir(g, team) == e0 + 50 - H.COST[card] * H.ELIXIR_UNIT
            assert K.hand(g, team)[s] != card, "the played card leaves the hand"
        else:
            assert code != H.OK
            assert K.hand(g, team) == hand and K.elixir(g, team) == min(H.MAX_ELIXIR, e0 + 50)


def test_dead_building_frees_its_footprint(pr):
    g = K.new_game("hog26", "hog26", deploy_lockout_ticks=0)
    assert K.cast_tile(g, 0, H.CANNON, 9, 21) == H.OK
    g.tick(1)
    assert K.cast_tile(g, 0, H.CANNON, 10, 21) == H.ILLEGAL_POSITION
    K.run_until(g, lambda gg: not K.units(gg, team=0, card=H.CANNON), 700, "Cannon lifetime expiry")
    assert K.cast_tile(g, 0, H.CANNON, 10, 21) == H.OK, "a dead building no longer blocks placement"
