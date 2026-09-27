"""SPEC §0 (card ids/names/decks), §2 (level-11 derivation, towers), §10 (card_info, PlayError).

Expected values are computed independently from data/source/cards-15.535.json with
floor(base * 256 / 100) (helpers.L11); SPEC §2 examples are asserted literally as a cross-check.
"""
import pytest

import gamekit as K
import helpers as H

# Safe spawn point for stat inspection: deep in team 0's half, far outside team 1's tower reach.
SAFE0 = (9000, 21000)


def test_card_names_are_the_spec_display_names(pr):
    """v0.3 (§16.4): 64 entries; ids 0-20 keep the v0.1 display names (full table: test_cards_v3_db)."""
    names = tuple(pr.CARD_NAMES)
    assert len(names) == 64, f"CARD_NAMES must have 64 entries (N_CARDS, §16.4), got {len(names)}"
    assert names[:21] == H.CARD_NAMES, f"CARD_NAMES[:21] mismatch:\n got  {names[:21]}\n want {H.CARD_NAMES}"


def _deck_ids(pr, deck):
    out = []
    for c in deck:
        if isinstance(c, str):
            assert c in H.CARD_NAMES, f"unknown card name {c!r} in DECKS"
            out.append(H.CARD_NAMES.index(c))
        else:
            out.append(int(c))
    return out


@pytest.mark.parametrize("name", ["hog26", "giant", "bait"])
def test_preset_decks(pr, name):
    assert name in pr.DECKS, f"DECKS must contain preset {name!r}"
    ids = _deck_ids(pr, pr.DECKS[name])
    assert len(ids) == 8 and len(set(ids)) == 8, f"{name}: 8 distinct cards required, got {ids}"
    assert sorted(ids) == sorted(H.DECKS[name]), f"{name}: got {sorted(ids)} want {sorted(H.DECKS[name])}"


def test_play_error_codes(pr):
    E = pr.PlayError
    assert int(E.OK) == 0
    one = getattr(E, "NOT_IN_HAND", None)
    if one is None:
        one = getattr(E, "BAD_SLOT")
    assert int(one) == 1
    assert int(E.NOT_ENOUGH_ELIXIR) == 2
    assert int(E.ILLEGAL_POSITION) == 3
    assert int(E.LOCKOUT) == 4
    assert int(E.GAME_OVER) == 5


# ------------------------------------------------------------------------------------------
# card_info (SPEC §10, keys pinned by §13.9)
# ------------------------------------------------------------------------------------------
UNIT_KEYS = ["hitpoints", "damage", "hit_speed_ms", "load_time_ms", "speed", "range", "sight_range",
             "collision_radius", "deploy_time_ms", "count", "lifetime_ms", "crown_tower_damage_percent"]


@pytest.mark.parametrize("cid", range(21))
def test_card_info_identity_and_cost(pr, cid):
    info = pr.card_info(cid)
    assert isinstance(info, dict)
    for k in ("id", "name", "kind"):
        assert k in info, f"card_info({cid}) lacks {k!r} (SPEC §13.9)"
    assert int(info["id"]) == cid
    cost = info["cost"] if "cost" in info else info["elixir"]
    assert int(cost) == H.COST[cid], f"cost of {H.CARD_NAMES[cid]}"
    assert str(info["kind"]).lower() == H.CARD_KIND[cid], f"kind of {H.CARD_NAMES[cid]}"
    assert info["name"] == H.CARD_NAMES[cid]


@pytest.mark.parametrize("cid", H.TROOPS + H.BUILDINGS)
def test_card_info_unit_stats_level11(pr, cid):
    info = pr.card_info(cid)
    for k in UNIT_KEYS:
        assert k in info, f"card_info({H.CARD_NAMES[cid]}) lacks {k!r} (SPEC §13.9)"
    exp = H.expected_unit_stats(cid)
    name = H.CARD_NAMES[cid]
    assert int(info["hitpoints"]) == exp["hp"], f"{name} hp (floor(base*256/100))"
    assert int(info["damage"]) == exp["damage"], f"{name} damage"
    assert int(info["hit_speed_ms"]) == exp["hit_speed_ms"], f"{name} hit speed (ms)"
    assert int(info["load_time_ms"]) == exp["load_time_ms"], f"{name} load time (ms)"
    assert int(info["range"]) == exp["range"], f"{name} range"
    assert int(info["sight_range"]) == exp["sight"], f"{name} sight range"
    assert int(info["collision_radius"]) == exp["radius"], f"{name} collision radius"
    assert int(info["count"]) == exp["count"], f"{name} unit count"
    assert int(info["deploy_time_ms"]) == exp["deploy_time_ms"], f"{name} deploy time"
    assert int(info["speed"]) == exp["speed"], f"{name} speed (millitiles/tick)"
    assert int(info["crown_tower_damage_percent"]) == 100
    if cid in H.BUILDINGS:
        assert int(info["lifetime_ms"]) == exp["lifetime_ms"]


# SPEC §2 / §7.2 spell numbers: (damage, radius, crown_tower_damage)
SPELLS = {H.FIREBALL: (688, 2500, 172), H.ARROWS: (122, 3500, 25), H.ZAP: (192, 2500, 48),
          H.LOG: (268, None, 35)}


@pytest.mark.parametrize("cid", sorted(SPELLS))
def test_card_info_spell_numbers(pr, cid):
    info = pr.card_info(cid)
    dmg, radius, ctd = SPELLS[cid]
    assert int(info["damage"]) == dmg
    assert int(info["crown_tower_damage"]) == ctd, "crown_tower_damage = ceil(D*P/100) (SPEC §2)"
    if radius is not None:
        assert int(info["radius"]) == radius


def test_spec_examples_match_independent_derivation():
    """Guards the tester's own arithmetic against the SPEC §2 worked examples."""
    assert H.expected_unit_stats(H.KNIGHT)["hp"] == 1766
    assert H.expected_unit_stats(H.KNIGHT)["damage"] == 202
    assert H.expected_unit_stats(H.HOG)["hp"] == 1697
    assert H.L11(H.src_card(H.FIREBALL)["projectile"]["damage"]) == 688
    assert H.L11(H.src_card(H.ZAP)["damage"]) == 192
    assert H.L11(H.src_card(H.LOG)["projectile"]["spawn_projectile"]["damage"]) == 268
    assert H.GOBLIN["hp"] == 202 and H.GOBLIN["damage"] == 125
    assert H.L11(H.src_card(H.PRINCE)["charge"]["damage_special"]) == 783
    assert H.L11(H.src_card(H.ICE_GOLEM)["death_damage"]) == 84
    assert H.ct_damage(688, 25) == 172 and H.ct_damage(192, 25) == 48
    assert H.ct_damage(122, 20) == 25 and H.ct_damage(268, 13) == 35
    assert [H.F_of_radius(r) for r in (1000, 1400, 600, 500)] == [3, 4, 3, 2]


# ------------------------------------------------------------------------------------------
# Behavioural card DB: what actually spawns (entities() keys are pinned by SPEC §10)
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cid", H.TROOPS + H.BUILDINGS)
def test_spawned_units_have_level11_stats(pr, cid):
    g = K.new_game()
    exp = H.expected_unit_stats(cid)
    ids = K.spawn(g, 0, cid, *SAFE0, deployed=True)
    assert len(ids) == exp["count"], f"{H.CARD_NAMES[cid]}: spawn returned {len(ids)} ids, want {exp['count']}"
    for i in ids:
        e = K.ent(g, i)
        assert e is not None, f"spawned id {i} missing from entities()"
        assert int(e["team"]) == 0
        assert int(e["card_id"]) == cid
        assert e["kind"] == ("building" if cid in H.BUILDINGS else "troop")
        assert int(e["max_hp"]) == exp["hp"], f"{H.CARD_NAMES[cid]} max_hp"
        assert int(e["hp"]) == exp["hp"], f"{H.CARD_NAMES[cid]} spawns at full hp"
        assert int(e["radius"]) == exp["radius"], f"{H.CARD_NAMES[cid]} radius"
        assert bool(e["flying"]) == exp["flying"], f"{H.CARD_NAMES[cid]} flying flag"
        assert int(e["target_id"]) == -1, "nothing is in sight at the safe point"
        assert not bool(e["deploying"]), "spawn(deployed=True) must not be deploying"
        assert int(e.get("shield", 0)) == 0


def test_skeletons_count_follows_data():
    assert H.src_card(H.SKELETONS)["count"] == 3


@pytest.mark.parametrize("team", [0, 1])
def test_crown_towers_level11(pr, team):
    g = K.new_game()
    for idx in range(3):
        t = K.tower(g, team, idx)
        assert int(t["max_hp"]) == H.TOWER_MAX_HP[idx], f"team {team} tower {idx} max_hp"
        assert int(t["hp"]) == H.TOWER_MAX_HP[idx]
        assert bool(t["alive"]) is True
        e = K.tower_entity(g, team, idx)
        assert e is not None, f"tower {team}/{idx} must appear in entities() at {H.TOWER_POS[team][idx]}"
        assert int(e["radius"]) == H.TOWER_RADIUS[idx]
        assert int(e["max_hp"]) == H.TOWER_MAX_HP[idx]
        assert e["kind"] == "tower"
    assert K.tower_active(g, team, 0) is False, "King starts dormant"


def test_initial_board_has_only_the_six_towers(pr):
    g = K.new_game()
    es = K.ents(g)
    assert len(es) == 6, f"a fresh match has exactly 6 entities (towers), got {len(es)}"
    assert all(e["kind"] == "tower" for e in es)
    assert g.projectiles() == [] and g.effects() == []
