/*
 * pr_defs.h -- static definition types for the PufferRoyale card database.
 *
 * These are the row types of the tables that tools/gen_card_db.py writes into
 * pr_card_db.h. Everything here is compile-time constant data; nothing in this
 * file is simulation state. All distances are millitiles (1 tile = 1000), all
 * durations milliseconds, all speeds millitiles per 50 ms tick (SPEC §1).
 */
#ifndef PR_DEFS_H
#define PR_DEFS_H

#include <stdint.h>

/* Entity kinds (PrUnitDef.kind and PrEntity.kind). */
enum { PR_KIND_TROOP = 0, PR_KIND_BUILDING = 1, PR_KIND_TOWER = 2 };

/* Card kinds (PrCardDef.kind). */
enum { PR_CARD_KIND_TROOP = 0, PR_CARD_KIND_BUILDING = 1, PR_CARD_KIND_SPELL = 2 };

/* How a spell card is delivered (PrCardDef.spell_type). */
enum {
    PR_SPELL_NONE = 0,
    PR_SPELL_PROJECTILE = 1,       /* Fireball: point projectile from the King, splash on arrival */
    PR_SPELL_WAVES = 2,            /* Arrows: N damage waves at the tap point */
    PR_SPELL_AREA = 3,             /* Zap: instant area effect at the tap point */
    PR_SPELL_ROLLING = 4,          /* The Log: rolling rectangle hitbox */
    PR_SPELL_SPAWN_PROJECTILE = 5, /* Goblin Barrel: projectile that spawns units on landing */
    PR_SPELL_PULSE = 6,            /* Poison, Earthquake: an area dealing periodic damage events (SPEC §16.6.10-11) */
    PR_SPELL_LIGHTNING = 7         /* Lightning: bolts on the highest-hp enemies in the area (SPEC §16.6.13) */
};

/* Where a card may be placed (SPEC §3.1, §16.1, §16.6.3), PrCardDef.placement. */
enum {
    PR_PLACE_ANY = 0,        /* Fireball, Arrows, Zap, Rocket, Poison, Freeze, Earthquake, Lightning, Snowball */
    PR_PLACE_TROOP = 1,      /* troops: territory, not on water / no-deploy / structure footprints */
    PR_PLACE_BUILDING = 2,   /* buildings: own half, whole footprint legal */
    PR_PLACE_LAND = 3,       /* Goblin Barrel: any non-water tile */
    PR_PLACE_TERRITORY = 4,  /* The Log, Barbarian Barrel: troop territory, not water */
    PR_PLACE_MINER = 5       /* Miner: any non-water tile outside every living crown-tower footprint */
};

/* A unit: anything that becomes an entity (troops, buildings, crown towers). */
typedef struct PrUnitDef {
    const char *name;        /* source key, e.g. "Knight", "Skeleton", "PrincessTower" */
    uint8_t kind;            /* PR_KIND_* */
    uint8_t attacks_air;
    uint8_t attacks_ground;
    uint8_t only_buildings;  /* target_only_buildings */
    uint8_t self_aoe;        /* self_as_aoe_center (Valkyrie) */
    uint8_t kamikaze;        /* dies when it fires (Ice Spirit) */
    uint8_t ignore_pushback;
    uint8_t jumps;           /* jump-enabled: may cross water (Hog Rider, Prince) */
    uint8_t hides;           /* hides_when_not_attacking (Tesla) */
    uint8_t no_attack;       /* no damage and no projectile: never targets or attacks (Tombstone, Collector) */
    char glyph;              /* ANSI renderer letter */
    int16_t projectile;      /* PR_PROJ_* index, -1 = melee / instant hit */
    int16_t death_area;      /* PR_AREA_* index spawned on death, -1 = none */
    int32_t hp;              /* level-11 hitpoints */
    int32_t damage;          /* level-11 damage per hit (melee) / per projectile */
    int32_t shield;          /* level-11 shield hitpoints */
    int32_t hit_speed_ms;
    int32_t load_time_ms;
    int32_t speed;           /* millitiles per tick */
    int32_t range;           /* attack range, millitiles */
    int32_t min_range;       /* minimum range (Mortar, SPEC §16.2); 0 = none */
    int32_t sight;           /* sight range, millitiles */
    int32_t radius;          /* collision radius, millitiles */
    int32_t mass;
    int32_t deploy_ms;
    int32_t flying_height;   /* > 0 => flying */
    int32_t area_radius;     /* melee splash radius (projectile splash lives on the projectile) */
    int32_t proj_start_radius;
    int32_t crown_pct;       /* crown_tower_damage_percent of this unit's own hits */
    int32_t lifetime_ms;     /* buildings: linear HP drain over this time; 0 = none */
    int32_t death_damage;    /* level-11 */
    int32_t death_radius;
    int32_t charge_range;    /* millitiles (data charge_range_raw is centitiles) */
    int32_t charge_damage;   /* level-11 damage_special */
    int32_t charge_speed_pct;
    int32_t jump_speed;
    int32_t hide_ms;         /* hide_time_ms */
    int32_t up_ms;           /* up_time_ms */
    int32_t footprint_tiles; /* buildings/towers: F = ceil((2r + 1000) / 1000); troops: 0 */
    /* ---- SPEC §16 mechanics (v0.3) ---- */
    int32_t var_damage2;     /* variable damage (Inferno): stage 2 / 3 damage, level 11; 0 = none */
    int32_t var_damage3;
    int32_t var_hits1;       /* hits per stage: VariableDamageTime1 / HitSpeed, VariableDamageTime2 / HitSpeed */
    int32_t var_hits2;
    int32_t dash_damage;     /* dash (Bandit, ledger combat.DASH_ATTACK): level-11 dash hit, 0 = no dash */
    int32_t dash_min;        /* DashMinRange / DashMaxRange, millitiles */
    int32_t dash_max;
    int32_t dash_speed;      /* JumpSpeed, millitiles per tick */
    int32_t dash_cooldown_ms;/* the stand before the dash */
    int32_t burrow_speed;    /* spawn_pathfind.speed (Miner), millitiles per tick; 0 = none */
    int32_t mana_amount;     /* Elixir Collector: elixir units per production / on death */
    int32_t mana_gen_ms;
    int32_t mana_on_death;
    int32_t seq_n;           /* AttackSequence length (Dagger Duchess), 0 = none */
    int16_t seq_mult[4];     /* per-shot HitSpeedMultiplier percent */
    int32_t reload_ms;       /* [IMPL-DEFINED] reload after the sequence */
    int32_t chef_period_ms;  /* Royal Chef level-up period (SPEC §16.6.26), 0 = none */
    int32_t chef_range;
    int16_t spawn_unit;      /* periodic spawner (SPEC §16.2): PR_UNIT_* emitted, -1 = none */
    int16_t spawn_number;    /* units per wave */
    int32_t spawn_interval_ms; /* between the units of one wave (0 = all at once) */
    int32_t spawn_start_ms;  /* first wave (0 = pause_time), SPEC §16.6.5 / §16.6.24 */
    int32_t spawn_pause_ms;  /* wave period, start to start */
    int32_t spawn_first;     /* PR_SPAWN_OFFSETS index of the wave layout */
    int16_t dspawn_unit;     /* death spawn: PR_UNIT_*, -1 = none */
    int16_t dspawn_count;
    int32_t dspawn_deploy_ms;/* DeathSpawnDeployTime (0 = born deployed) */
    int32_t dspawn_first;    /* PR_SPAWN_OFFSETS index of the death-spawn layout */
    int32_t bomb_damage;     /* death bomb (Balloon, Giant Skeleton, Bomb Tower): level-11 damage, 0 = none */
    int32_t bomb_radius;
    int32_t bomb_fuse_ms;    /* the bomb row's DeployTime */
    uint8_t bomb_air;
    uint8_t bomb_ground;
} PrUnitDef;

/* A projectile row (unit shots, tower shots and spell projectiles). */
typedef struct PrProjDef {
    const char *name;
    int32_t speed;           /* millitiles per tick */
    int32_t damage;          /* level-11 */
    int32_t crown_pct;
    int32_t radius;          /* splash radius; 0 = single target */
    int32_t radius_y;        /* rolling hitbox half-depth (The Log) */
    uint8_t aoe_air;
    uint8_t aoe_ground;
    uint8_t homing;
    uint8_t pushback_all;
    int16_t target_buff;     /* PR_BUFF_* applied on impact, -1 = none */
    int16_t spawn_unit;      /* PR_UNIT_* spawned on landing, -1 = none */
    int32_t buff_ms;
    int32_t pushback;        /* knockback distance, millitiles */
    int32_t range;           /* rolling distance (The Log), millitiles */
    int32_t spawn_count;
    int32_t spawn_deploy_ms;
} PrProjDef;

/* An area-effect object (Zap, the Ice Golem's death slow). */
typedef struct PrAreaDef {
    const char *name;
    int32_t life_ms;
    int32_t radius;
    int32_t damage;          /* level-11 */
    int32_t crown_pct;
    int16_t buff;            /* PR_BUFF_* or -1 */
    uint8_t hits_air;
    uint8_t hits_ground;
    uint8_t affects_hidden;
    uint8_t cap_buff;        /* cap_buff_time_to_area_effect_time (Earthquake) */
    int32_t buff_ms;
    int32_t hit_speed_ms;    /* buff refresh / bolt period (Poison 250, Earthquake 100, Lightning 460); 0 = one-shot */
    int16_t projectile;      /* PR_PROJ_* of a bolt area (Lightning), -1 = none */
    int16_t pad2_;
} PrAreaDef;

/* A character buff (status effect). Multipliers are raw percent deltas: -100 = full stop. */
typedef struct PrBuffDef {
    const char *name;
    int32_t speed_pct;
    int32_t hit_speed_pct;
    int32_t spawn_speed_pct;
    int32_t dps;             /* damage per second, level 11 (Poison, Earthquake); 0 = none */
    int32_t hit_freq_ms;     /* damage period */
    int32_t crown_pct;       /* crown-tower percent of the damage */
    int32_t building_pct;    /* percent vs non-crown buildings (Earthquake 350) */
    int32_t stacking;        /* enable_stacking */
} PrBuffDef;

/* A playable card. */
typedef struct PrCardDef {
    const char *name;        /* display name, e.g. "Archers" */
    const char *key;         /* source key in cards-15.535.json, e.g. "Archer" */
    uint8_t kind;            /* PR_CARD_KIND_* */
    uint8_t elixir;
    uint8_t spell_type;      /* PR_SPELL_* */
    uint8_t placement;       /* PR_PLACE_* */
    int16_t unit;            /* PR_UNIT_* summoned (troops/buildings), -1 for spells */
    int16_t count;           /* members summoned */
    int16_t unit2;           /* second_summon unit (Goblin Gang, Rascals), -1 = none */
    int16_t count2;
    int16_t formation;       /* PR_FORMATION_* index */
    int16_t spell_proj;      /* PR_PROJ_* delivering the spell, -1 = none */
    int16_t spell_area;      /* PR_AREA_* created by the spell, -1 = none */
    int32_t deploy_delay_ms; /* per-member deploy stagger (summon_deploy_delay_ms) */
    int32_t spell_radius;    /* area radius of the spell's damage */
    int32_t spell_damage;    /* level-11 damage per hit/wave */
    int32_t crown_pct;
    int32_t waves;           /* Arrows waves; pulse events; Lightning bolts */
    int32_t wave_interval_ms;
} PrCardDef;

/* A deterministic multi-unit layout, in the acting team's OWN frame (-y = toward the enemy). */
typedef struct PrFormationDef {
    int16_t count;
    int16_t first;           /* index of member 0 in PR_FORMATION_OFFSETS */
} PrFormationDef;

#endif /* PR_DEFS_H */
