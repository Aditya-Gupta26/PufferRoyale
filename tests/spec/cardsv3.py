"""SPEC §16 (Phase E) reference data: the 43 new cards (ids 21-63), spawned units, tower troops,
the v0.3 presets, and oracle extensions (Miner / Barbarian Barrel territory, new footprints).
Everything is derived from data/source/cards-15.535.json and the SPEC text only."""
import helpers as H

# §16.1 table: (id, display name, cards-15.535.json key)
NEW_CARDS = [
    (21, "Barbarians", "Barbarians"), (22, "Mini P.E.K.K.A", "MiniPekka"), (23, "P.E.K.K.A", "Pekka"),
    (24, "Mega Minion", "MegaMinion"), (25, "Bats", "Bats"), (26, "Spear Goblins", "SpearGoblins"),
    (27, "Goblins", "Goblins"), (28, "Goblin Gang", "GoblinGang"), (29, "Royal Giant", "RoyalGiant"),
    (30, "Bomber", "Bomber"), (31, "Princess", "Princess"), (32, "Dart Goblin", "BlowdartGoblin"),
    (33, "Minion Horde", "MinionHorde"), (34, "Fire Spirit", "FireSpirits"), (35, "Ice Wizard", "IceWizard"),
    (36, "Golem", "Golem"), (37, "Lava Hound", "LavaHound"), (38, "Balloon", "Balloon"),
    (39, "Giant Skeleton", "GiantSkeleton"), (40, "Witch", "Witch"), (41, "Tombstone", "Tombstone"),
    (42, "Inferno Tower", "InfernoTower"), (43, "Inferno Dragon", "InfernoDragon"), (44, "Bandit", "Assassin"),
    (45, "Battle Ram", "BattleRam"), (46, "Royal Hogs", "RoyalHogs"), (47, "Miner", "Miner"),
    (48, "Mortar", "Mortar"), (49, "X-Bow", "Xbow"), (50, "Elixir Collector", "Elixir Collector"),
    (51, "Bomb Tower", "BombTower"), (52, "Rocket", "Rocket"), (53, "Poison", "Poison"),
    (54, "Freeze", "Freeze"), (55, "Earthquake", "Earthquake"), (56, "Lightning", "Lightning"),
    (57, "Giant Snowball", "Snowball"), (58, "Barbarian Barrel", "BarbLog"), (59, "Rascals", "Rascals"),
    (60, "Elite Barbarians", "AngryBarbarians"), (61, "Skeleton Dragons", "SkeletonDragons"),
    (62, "Wall Breakers", "Wallbreakers"), (63, "Night Witch", "DarkWitch"),
]
ID = {name: cid for cid, name, _ in NEW_CARDS}
KEY = {cid: key for cid, _, key in NEW_CARDS}
ALL_NAMES = tuple(list(H.CARD_NAMES) + [n for _, n, _ in NEW_CARDS])
NAME_TO_ID = {n: i for i, n in enumerate(ALL_NAMES)}
N_CARDS = 64
CARD_SLOTS = 128

(BARBARIANS, MINI_PEKKA, PEKKA, MEGA_MINION, BATS, SPEAR_GOBLINS, GOBLINS, GOBLIN_GANG, ROYAL_GIANT,
 BOMBER, PRINCESS, DART_GOBLIN, MINION_HORDE, FIRE_SPIRIT, ICE_WIZARD, GOLEM, LAVA_HOUND, BALLOON,
 GIANT_SKELETON, WITCH, TOMBSTONE, INFERNO_TOWER, INFERNO_DRAGON, BANDIT, BATTLE_RAM, ROYAL_HOGS, MINER,
 MORTAR, XBOW, ELIXIR_COLLECTOR, BOMB_TOWER, ROCKET, POISON, FREEZE, EARTHQUAKE, LIGHTNING, SNOWBALL,
 BARB_BARREL, RASCALS, ELITE_BARBARIANS, SKELETON_DRAGONS, WALL_BREAKERS, NIGHT_WITCH) = range(21, 64)

# §16.4 presets (display names)
PRESETS = {
    "golem": ["Golem", "Night Witch", "Baby Dragon", "Lightning", "Mega Minion", "Barbarian Barrel",
              "Mini P.E.K.K.A", "Zap"],
    "lavaloon": ["Lava Hound", "Balloon", "Minions", "Mega Minion", "Skeleton Dragons", "Arrows", "Fireball",
                 "Tombstone"],
    "xbow": ["X-Bow", "Tesla", "Archers", "Knight", "Skeletons", "Ice Spirit", "Fireball", "The Log"],
    "miner_poison": ["Miner", "Poison", "Goblin Gang", "Bats", "Inferno Tower", "Valkyrie", "Spear Goblins",
                     "The Log"],
    "pekka_bridge": ["P.E.K.K.A", "Battle Ram", "Bandit", "Minions", "Musketeer", "Zap", "Poison", "Dart Goblin"],
    "royal_hogs": ["Royal Hogs", "Earthquake", "Fire Spirit", "Barbarian Barrel", "Goblin Gang", "Mega Minion",
                   "Zap", "Musketeer"],
}
PRESET_IDS = {k: [NAME_TO_ID[n] for n in v] for k, v in PRESETS.items()}


def src(cid):
    if cid < 21:
        return H.src_card(cid)
    key = KEY[cid]
    for c in H.card_data()["cards"]:
        if c["name"] == key:
            return c
    raise KeyError(key)


def unit(name):
    return H.card_data()["units"][name]


def kind(cid):
    return src(cid)["kind"] if cid >= 21 else H.CARD_KIND[cid]


def cost(cid):
    return src(cid)["elixir"] if cid >= 21 else H.COST[cid]


def l11(x):
    return None if x is None else H.L11(x)


def composition(cid):
    """[(unit name, count)] a troop card deploys (summon + second_summon)."""
    c = src(cid)
    out = [(c["summon_character"], c["count"])]
    if c.get("second_summon"):
        out.append((c["second_summon"]["character"], c["second_summon"]["count"]))
    return out


def unit_stats(name):
    u = unit(name)
    return dict(hp=l11(u.get("hitpoints")), damage=l11(u.get("damage")), radius=u["collision_radius_milli"],
                flying=(u.get("flying_height") or 0) > 0, speed=u.get("speed") or 0,
                hit_speed_ms=u.get("hit_speed_ms"), load_time_ms=u.get("load_time_ms"),
                range=u.get("range_milli"), tob=bool(u.get("target_only_buildings")))


# Tower ladder at level 11, Princess rates (§2): Princess HP 1400 -> 3052 = 218 %
TOWER_PCT = 218


def tower_l11(x):
    return (int(x) * TOWER_PCT) // 100


TOWER_TROOPS = ["princess", "cannoneer", "dagger_duchess", "royal_chef"]
TOWER_ROW = {"cannoneer": "Cannoneer", "dagger_duchess": "DaggerDuchess", "royal_chef": "ChefTower"}


def tower_troop_stats(name):
    if name == "princess":
        return dict(hp=3052, damage=109, hit_speed_ms=800, load_time_ms=0)
    u = unit(TOWER_ROW[name])
    return dict(hp=tower_l11(u["hitpoints"]), damage=tower_l11(u["damage"]), hit_speed_ms=u["hit_speed_ms"],
                load_time_ms=u.get("load_time_ms") or 0, speed=(u.get("projectile") or {}).get("speed"),
                start_radius=u.get("projectile_start_radius_milli"))


# ------------------------------------------------------------------------------------------
# Oracle extensions (§16.1, §16.6): Miner and Barbarian Barrel territory, new building footprints
# ------------------------------------------------------------------------------------------
def miner_tile(team, tx, ty, alive):
    """§16.6 item 3: any non-water tile outside every LIVING crown-tower footprint (either team);
    building footprints and no-deploy cells are allowed."""
    etx, ety = H.own_to_engine_tile(team, tx, ty)
    if H.tile_is_water_engine(etx, ety):
        return False
    trect = H.tile_rect_engine(team, tx, ty)
    for t in (0, 1):
        for i in range(3):
            if alive[t][i] and H.overlap_pos(trect, H.footprint(*H.TOWER_POS[t][i], H.TOWER_F[i])):
                return False
    return True


def building_radius(cid):
    return src(cid)["collision_radius_milli"]


def building_legal_v3(team, tx, ty, cid, alive, buildings=()):
    """§3.1 rule 2 with the new building radii (all odd F -> tile-centred)."""
    if ty < 17:
        return False
    F = H.F_of_radius(building_radius(cid))
    orect = H.footprint(*H.building_centre_own(tx, ty, F), F)
    if orect[0] < 0 or orect[2] > H.ARENA_W or orect[1] < 17000 or orect[3] > H.ARENA_H:
        return False
    erect = H.rotate_rect(team, orect)
    g = H.arena_grid()
    for hr in range(erect[1] // 500, erect[3] // 500):
        for hc in range(erect[0] // 500, erect[2] // 500):
            if g[hr, hc] & (H.WATER_BIT | H.NODEPLOY_BIT):
                return False
    for t in (0, 1):
        for i in range(3):
            if alive[t][i] and H.overlap_pos(erect, H.footprint(*H.TOWER_POS[t][i], H.TOWER_F[i])):
                return False
    for (bx, by, bF) in buildings:
        if H.overlap_pos(erect, H.footprint(bx, by, bF)):
            return False
    return True


def tile_legal_v3(cid, team, tx, ty, alive, buildings=()):
    if cid < 21:
        return H.tile_legal(cid, team, tx, ty, alive, buildings)
    k = kind(cid)
    if cid == MINER:
        return miner_tile(team, tx, ty, alive)
    if cid == BARB_BARREL:
        return H.log_tile_legal(team, tx, ty, alive)
    if k == "troop":
        return H.troop_tile_legal(team, tx, ty, alive, buildings)
    if k == "building":
        return building_legal_v3(team, tx, ty, cid, alive, buildings)
    return True                                             # Rocket, Poison, Freeze, Earthquake, Lightning, Snowball


def game_with(team_cards, deck_other="random", max_seed=4000, **kw):
    """A Game whose team-0 'random' deck (§16.4: samples all 64) contains every card in `team_cards`;
    the first such seed in a fixed range (deterministic)."""
    import gamekit as K
    for seed in range(max_seed):
        g = K.new_game("random", deck_other, seed=seed, **kw)
        if set(team_cards) <= set(K.deck_of(g, 0)):
            return g
    raise AssertionError(f"no seed < {max_seed} deals {team_cards} to team 0")
