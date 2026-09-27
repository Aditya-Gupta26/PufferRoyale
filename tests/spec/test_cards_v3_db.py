"""SPEC §16.1 / §16.4 card database: 64 cards, stable ids, level-11 stats (floor(base*256/100)),
compositions (second_summon), presets and the all-64 random deck."""
import pytest

import cardsv3 as C
import gamekit as K
import helpers as H

SAFE0 = (9000, 21000)


def test_card_names_and_counts(pr):
    names = tuple(pr.CARD_NAMES)
    assert len(names) == 64 and int(pr.N_CARDS) == 64
    assert names[:21] == H.CARD_NAMES, "ids 0-20 are stable (SPEC §0)"
    assert names == C.ALL_NAMES, "ids 21-63 as in the §16.1 table"
    assert int(pr.CARD_SLOTS) == 128


@pytest.mark.parametrize("cid", range(21, 64))
def test_card_info_identity(pr, cid):
    info = pr.card_info(cid)
    assert int(info["id"]) == cid and info["name"] == C.ALL_NAMES[cid]
    assert int(info["cost"] if "cost" in info else info["elixir"]) == C.cost(cid)
    assert str(info["kind"]).lower() == C.kind(cid)


UNIT_CARDS = [cid for cid, _, _ in C.NEW_CARDS if C.kind(cid) in ("troop", "building")]


@pytest.mark.parametrize("cid", UNIT_CARDS)
def test_card_info_level11_unit_stats(pr, cid):
    c = C.src(cid)
    info = pr.card_info(cid)
    name = C.ALL_NAMES[cid]
    assert int(info["hitpoints"]) == H.L11(c["hitpoints"]), f"{name} hitpoints"
    if c.get("damage") is not None:
        assert int(info["damage"]) == H.L11(c["damage"]), f"{name} damage"
    for k_info, k_data in (("hit_speed_ms", "hit_speed_ms"), ("load_time_ms", "load_time_ms"),
                           ("collision_radius", "collision_radius_milli"), ("deploy_time_ms", "deploy_time_ms")):
        if c.get(k_data) is not None:
            assert int(info[k_info]) == c[k_data], f"{name} {k_info}"
    if c.get("range_milli") is not None:
        assert int(info["range"]) == c["range_milli"]
    if c.get("sight_range_milli") is not None:
        assert int(info["sight_range"]) == c["sight_range_milli"]
    if C.kind(cid) == "troop":
        assert int(info["speed"]) == c["speed"]
        assert int(info["count"]) == c["count"]
    if c.get("lifetime_ms"):
        assert int(info["lifetime_ms"]) == c["lifetime_ms"]
    assert int(info["crown_tower_damage_percent"]) == (c.get("crown_tower_damage_percent") or 100)


def test_princess_card_info_damage(pr):
    """§16.6.1: Princess fires PrincessProjectile, 66 -> 168 (the Deco volleys are cosmetic)."""
    assert int(pr.card_info(C.PRINCESS)["damage"]) == H.L11(66) == 168


def test_miner_crown_percent(pr):
    assert int(pr.card_info(C.MINER)["crown_tower_damage_percent"]) == 20


# spells: (damage, radius, crown_tower_damage) from the data; None = not pinned by §16
SPELL_NUMBERS = {
    C.ROCKET: (H.L11(580), 2000, H.ct_damage(H.L11(580), 23)),           # 1484 / 342
    C.FREEZE: (H.L11(58), 3000, H.ct_damage(H.L11(58), 25)),             # 148 / 37
    C.SNOWBALL: (H.L11(70), 2500, H.ct_damage(H.L11(70), 25)),           # 179 / 45
    C.LIGHTNING: (H.L11(413), 3500, H.ct_damage(H.L11(413), 25)),        # 1057 / 265
    C.BARB_BARREL: (H.L11(91), None, H.L11(91)),                          # 232
    C.POISON: (None, 3500, None),
    C.EARTHQUAKE: (None, 3500, None),
}


@pytest.mark.parametrize("cid", sorted(SPELL_NUMBERS))
def test_card_info_spell_numbers(pr, cid):
    dmg, radius, ctd = SPELL_NUMBERS[cid]
    info = pr.card_info(cid)
    if dmg is not None:
        assert int(info["damage"]) == dmg, f"{C.ALL_NAMES[cid]} damage"
    if radius is not None:
        assert int(info["radius"]) == radius
    if ctd is not None:
        assert int(info["crown_tower_damage"]) == ctd


def test_tester_spell_arithmetic():
    assert SPELL_NUMBERS[C.ROCKET] == (1484, 2000, 342)
    assert SPELL_NUMBERS[C.FREEZE] == (148, 3000, 37)
    assert SPELL_NUMBERS[C.SNOWBALL] == (179, 2500, 45)
    assert SPELL_NUMBERS[C.LIGHTNING] == (1057, 3500, 265)


@pytest.mark.parametrize("cid", UNIT_CARDS)
def test_spawned_composition_and_level11(pr, cid):
    """Game.spawn deploys the data composition (summon_character x count + second_summon) with
    level-11 hp and data radius / flying flag."""
    g = K.new_game()
    ids = K.spawn(g, 0, cid, *SAFE0, deployed=True)
    comp = C.composition(cid)
    assert len(ids) == sum(n for _, n in comp), f"{C.ALL_NAMES[cid]}: {len(ids)} entities, want {comp}"
    ents = [K.ent(g, i) for i in ids]
    want = sorted((C.unit_stats(u)["hp"], C.unit_stats(u)["radius"], C.unit_stats(u)["flying"])
                  for u, n in comp for _ in range(n))
    got = sorted((int(e["max_hp"]), int(e["radius"]), bool(e["flying"])) for e in ents)
    assert got == want, f"{C.ALL_NAMES[cid]}: (hp, radius, flying) {got} want {want}"
    for e in ents:
        assert int(e["card_id"]) == cid, "spawned members map to the card that created them"
        assert e["kind"] == ("building" if C.kind(cid) == "building" else "troop")


def test_composition_examples():
    assert C.composition(C.GOBLIN_GANG) == [("Goblin_Stab", 3), ("SpearGoblin", 3)]
    assert C.composition(C.RASCALS) == [("RascalBoy", 1), ("RascalGirl", 2)]
    assert [C.unit_stats(u)["hp"] for u, _ in C.composition(C.GOBLIN_GANG)] == [202, 133]
    assert [C.unit_stats(u)["hp"] for u, _ in C.composition(C.RASCALS)] == [1832, 261]
    assert C.unit_stats("Golemite")["hp"] == 1039 and C.unit_stats("LavaPups")["hp"] == 215
    assert C.unit_stats("Bat")["hp"] == 81 and C.unit_stats("Barbarian")["hp"] == 716


@pytest.mark.parametrize("name", sorted(C.PRESETS))
def test_new_presets(pr, name):
    assert name in pr.DECKS, f"DECKS must contain {name!r} (§16.4)"
    ids = [int(c) if not isinstance(c, str) else C.NAME_TO_ID[c] for c in pr.DECKS[name]]
    assert sorted(ids) == sorted(C.PRESET_IDS[name]), f"{name}: {sorted(ids)}"


def test_old_presets_unchanged(pr):
    for name in ("hog26", "giant", "bait"):
        ids = [int(c) if not isinstance(c, str) else C.NAME_TO_ID[c] for c in pr.DECKS[name]]
        assert sorted(ids) == sorted(H.DECKS[name])


def test_random_deck_draws_from_all_64(pr):
    seen = set()
    for seed in range(300):
        d = K.deck_of(K.new_game("random", "random", seed=seed), 0)
        assert len(d) == 8 and len(set(d)) == 8 and all(0 <= c < 64 for c in d)
        seen |= set(d)
    assert seen == set(range(64)), f"cards never drawn in 300 random decks: {sorted(set(range(64)) - seen)}"
