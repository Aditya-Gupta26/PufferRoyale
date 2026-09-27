"""Builder tests: the SPEC §10 debug API surface (names, types, keys, error codes)."""
import numpy as np
import pytest

import pufferroyale as pr
from pufferroyale import Game, PlayError, card_info

ENTITY_KEYS = {"id", "team", "card_id", "unit", "kind", "x", "y", "hp", "max_hp", "shield", "radius",
               "flying", "deploying", "stunned", "slowed", "hidden", "target_id", "charged"}
STATE_KEYS = {"tick", "elixir", "hand", "queue", "crowns", "towers", "over", "result", "end_reason", "overtime"}


def test_public_names():
    for name in ("Game", "card_info", "CARD_NAMES", "DECKS", "PlayError"):
        assert hasattr(pr, name)
    assert isinstance(pr.CARD_NAMES, tuple) and len(pr.CARD_NAMES) == 64 and pr.N_CARDS == 64
    assert pr.CARD_NAMES[0] == "Knight" and pr.CARD_NAMES[19] == "The Log" and pr.CARD_NAMES[20] == "Goblin Barrel"
    assert pr.CARD_NAMES[21] == "Barbarians" and pr.CARD_NAMES[44] == "Bandit" and pr.CARD_NAMES[63] == "Night Witch"
    assert pr.CARD_SLOTS == 128
    assert pr.TOWER_TROOPS == ("princess", "cannoneer", "dagger_duchess", "royal_chef")
    assert set(pr.DECKS) == {"hog26", "giant", "bait", "golem", "lavaloon", "xbow", "miner_poison", "pekka_bridge",
                             "royal_hogs"}
    names = {k: {pr.CARD_NAMES[c] for c in v} for k, v in pr.DECKS.items()}
    assert names["hog26"] == {"Hog Rider", "Musketeer", "Cannon", "Ice Golem", "Ice Spirit", "Skeletons",
                              "Fireball", "The Log"}
    assert names["giant"] == {"Giant", "Prince", "Baby Dragon", "Wizard", "Minions", "Knight", "Arrows", "Zap"}
    assert names["bait"] == {"Goblin Barrel", "Skeleton Army", "Tesla", "Valkyrie", "Archers", "Knight",
                             "The Log", "Fireball"}


def test_play_error_codes():
    assert PlayError.OK == 0
    assert PlayError.NOT_IN_HAND == 1 and PlayError.BAD_SLOT == 1
    assert PlayError.NOT_ENOUGH_ELIXIR == 2
    assert PlayError.ILLEGAL_POSITION == 3
    assert PlayError.LOCKOUT == 4
    assert PlayError.GAME_OVER == 5


def test_entities_and_state_shapes():
    g = Game()
    ents = g.entities()
    assert len(ents) == 6
    for e in ents:
        assert ENTITY_KEYS <= set(e)
        assert e["kind"] == "tower" and e["card_id"] == -1 and e["target_id"] == -1
    s = g.state()
    assert STATE_KEYS <= set(s)
    assert s["tick"] == 0 and s["elixir"] == [16800, 16800]
    assert len(s["hand"]) == 2 and all(len(h) == 4 for h in s["hand"])
    assert len(s["queue"]) == 2 and all(len(q) == 4 for q in s["queue"])
    assert s["crowns"] == [0, 0] and s["over"] is False and s["overtime"] is False
    assert s["end_reason"] is None
    for team in range(2):
        towers = s["towers"][team]
        assert [t["max_hp"] for t in towers] == [4824, 3052, 3052]
        assert [t["alive"] for t in towers] == [True, True, True]
        assert towers[0]["active"] is False
    # tower ids follow the index convention: 0 King, 1 left (x 3500), 2 right (x 14500)
    by_id = {e["id"]: e for e in ents}
    for team in range(2):
        xs = [by_id[t["id"]]["x"] for t in s["towers"][team]]
        assert xs == [9000, 3500, 14500]


def test_card_info_all_cards():
    for c in range(64):
        info = card_info(c)
        assert info["id"] == c and info["name"] == pr.CARD_NAMES[c]
        assert info["elixir"] == pr.CARD_COSTS[c]
    assert card_info(0)["hitpoints"] == 1766 and card_info(0)["damage"] == 202
    assert card_info("Hog Rider")["hitpoints"] == 1697
    assert card_info("Fireball")["damage"] == 688 and card_info("Fireball")["crown_tower_damage"] == 172
    assert card_info("Zap")["damage"] == 192 and card_info("Zap")["crown_tower_damage"] == 48
    assert card_info("Arrows")["damage"] == 122 and card_info("Arrows")["crown_tower_damage"] == 25
    assert card_info("The Log")["damage"] == 268 and card_info("The Log")["crown_tower_damage"] == 35
    gob = card_info("Goblin Barrel")["spawn"]
    assert gob["hitpoints"] == 202 and gob["damage"] == 125 and gob["speed"] == 120
    assert card_info("Ice Golem")["death_damage"] == 84
    assert card_info("Prince")["damage_special"] == 783
    assert card_info("Cannon")["lifetime_ms"] == 30000 and card_info("Tesla")["lifetime_ms"] == 25000
    # SPEC §16 cards (level 11)
    assert card_info("Golem")["hitpoints"] == 5120 and card_info("Golem")["death_spawn"]["count"] == 2
    assert card_info("Princess")["damage"] == 168 and card_info("Miner")["crown_tower_damage_percent"] == 20
    assert card_info("Goblin Gang")["second_unit"]["name"] == "SpearGoblin" and card_info("Goblin Gang")["count2"] == 3
    assert card_info("Inferno Tower")["variable_damage"] == (43, 158, 847)
    assert card_info("Bandit")["dash"]["damage"] == 389
    assert card_info("Balloon")["death_bomb"] == {"damage": 240, "radius": 3000, "fuse_ms": 3000}
    assert card_info("Rocket")["crown_tower_damage"] == 342 and card_info("Lightning")["damage"] == 1057
    assert card_info("Poison")["damage"] == 92 and card_info("Poison")["events"] == 8
    assert card_info("Earthquake")["building_damage"] == 283 and card_info("Earthquake")["crown_tower_damage"] == 49
    assert card_info("Witch")["spawner"]["number"] == 4 and card_info("Tombstone")["attacks"] is False
    with pytest.raises(ValueError):
        card_info(64)


def test_legal_mask_type():
    g = Game()
    m = g.legal_mask(0)
    assert isinstance(m, np.ndarray) and m.dtype == np.uint8 and m.shape == (2305,)
    assert m[0] == 1 and m.sum() == 1  # lockout
    g.tick(90)
    assert g.legal_mask(0).sum() > 1


def test_hash_snapshot_types():
    g = Game()
    h = g.hash()
    assert isinstance(h, int) and 0 <= h < 2 ** 64
    s = g.snapshot()
    assert isinstance(s, bytes)
    g.tick(50)
    assert g.hash() != h
    g.restore(s)
    assert g.hash() == h
    with pytest.raises(ValueError):
        g.restore(b"x" * 10)
    bad = bytearray(s)
    bad[0] ^= 0xFF
    with pytest.raises(ValueError):
        g.restore(bytes(bad))


def test_decks_by_name_list_and_random():
    g = Game(deck0=["Knight", "Archers", 2, 3, 4, 5, 6, 7], deck1="random", seed=3)
    s = g.state()
    assert sorted(s["hand"][0] + s["queue"][0]) == list(range(8))
    d1 = s["hand"][1] + s["queue"][1]
    assert len(set(d1)) == 8 and all(0 <= c < 64 for c in d1)
    with pytest.raises(ValueError):
        Game(deck0="nope")
    with pytest.raises(ValueError):
        Game(deck0=[0, 0, 1, 2, 3, 4, 5, 6])


def test_tower_troops_api():
    g = Game(tower_troop0="cannoneer", tower_troop1="royal_chef")
    s = g.state()
    assert s["tower_troops"] == ["cannoneer", "royal_chef"]
    assert [t["max_hp"] for t in s["towers"][0]] == [4824, 2616, 2616]
    assert [t["max_hp"] for t in s["towers"][1]] == [4824, 2703, 2703]
    units = {e["unit"] for e in g.entities()}
    assert {"Cannoneer", "ChefTower", "KingTower"} == units
    g.reset(seed=4)                                      # the configuration survives a reset
    assert g.state()["tower_troops"] == ["cannoneer", "royal_chef"]
    for bad in ("musketeer", "", None, 1):
        with pytest.raises(ValueError):
            Game(tower_troop0=bad)
