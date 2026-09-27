/*
 * pr_types.h -- constants and the plain-old-data game state (SPEC §1).
 *
 * PrState is the WHOLE simulation state: no pointers, fixed capacities, zeroed with
 * memset at reset so that padding bytes are deterministic. snapshot = memcpy,
 * restore = memcpy, state hash = FNV-1a over its bytes. Anything that is a pure
 * function of PrState (e.g. the path-field cache in pr_path.h) lives outside it.
 */
#ifndef PR_TYPES_H
#define PR_TYPES_H

#include <stdint.h>
#include <string.h>

#include "pr_math.h"
#include "pr_defs.h"
#include "pr_card_db.h"
#include "pr_arena_db.h"

/* ------------------------------------------------------------------ capacities */
#define PR_MAX_ENTITIES 256      /* troops + buildings + towers */
#define PR_MAX_PROJECTILES 256
#define PR_MAX_EFFECTS 64        /* spell / area objects */
#define PR_MAX_PENDING_SPAWNS 128 /* entities / death bombs queued for the next Spawn phase */
#define PR_MAX_PENDING_PLAYS 4   /* per team, per tick (debug API may queue several) */
#define PR_HIT_WORDS (PR_MAX_ENTITIES / 32) /* hit-once bitset of an effect: one bit per pool slot */

#define PR_NO_ID 0xFFFFFFFFu
#define PR_COUNTER_MAX 1000000000 /* statistics counters (leaked, spawn_overflow) saturate here (SPEC §18.5) */

/* ------------------------------------------------------------------ time & rules (SPEC §4) */
#define PR_TICK_MS 50
#define PR_TICKS_REGULATION 3600 /* ticks [0, 3600) */
#define PR_TICKS_MAX 6000        /* overtime ends at 6000 */
#define PR_TICK_2X 2400          /* elixir 2x from here */
#define PR_TICK_3X 4800          /* elixir 3x from here */
#define PR_ELIXIR_UNIT 2800      /* units per elixir */
#define PR_ELIXIR_START 16800    /* 6 elixir (ledger match.START_MANA) */
#define PR_ELIXIR_MAX 28000      /* 10 elixir */
#define PR_ELIXIR_RATE_1X 50     /* units per tick */
#define PR_KING_WAKE_TICKS 71    /* ledger match.KING_ACTIVATE_TIME_MS = 3550 */
#define PR_DEFAULT_LOCKOUT 90    /* ledger match.DEPLOY_LOCKOUT_TICKS */

/* ------------------------------------------------------------------ targeting (SPEC §6.1) */
#define PR_EXTRA_SIGHT_CROWN 2000 /* ledger targeting.EXTRA_SIGHT_RANGE_TO_CROWN_TOWERS */
#define PR_KEEP_TARGET_EXTRA 500  /* keep-hysteresis beyond the sight test */
#define PR_MELEE_RANGE 1900       /* ledger targeting.MELEE_RANGE_LIMIT: range <= this is melee */

/* ------------------------------------------------------------------ actions (SPEC §9) */
#define PR_N_TILES (PR_TILES_X * PR_TILES_Y) /* 576 */
#define PR_N_ACTIONS (1 + 4 * PR_N_TILES)    /* 2305 */
#define PR_CARD_SLOTS 128  /* SPEC §16.4: room for the full roster (multi-hot width) */

/* ------------------------------------------------------------------ enums */
enum { PR_TIEBREAK_ABSOLUTE = 0, PR_TIEBREAK_FRACTION = 1 };

enum {
    PR_END_NONE = 0,
    PR_END_KING = 1,
    PR_END_REGULATION_CROWNS = 2,
    PR_END_OVERTIME_CROWNS = 3,
    PR_END_TIEBREAK = 4,
    PR_END_DRAW = 5
};

/* play error codes (SPEC §10 PlayError) */
enum {
    PR_OK = 0,
    PR_ERR_BAD_SLOT = 1,         /* NOT_IN_HAND / BAD_SLOT */
    PR_ERR_NOT_ENOUGH_ELIXIR = 2,
    PR_ERR_ILLEGAL_POSITION = 3,
    PR_ERR_LOCKOUT = 4,
    PR_ERR_GAME_OVER = 5
};

/* Tesla hide state machine (Phase B, ledger hide.*) */
enum { PR_HIDE_UP = 0, PR_HIDE_HIDDEN = 1, PR_HIDE_RISING = 2 };

/* effect object types */
enum {
    PR_FX_NONE = 0,
    PR_FX_AREA = 1,     /* an area-effect object standing at a point (Zap, Ice Golem death slow) */
    PR_FX_WAVES = 2,    /* Arrows: damage waves at the tap point */
    PR_FX_ROLLING = 3,  /* The Log, Barbarian Barrel */
    PR_FX_PULSE = 4,    /* Poison, Earthquake: periodic damage events (SPEC §16.6.10-11) */
    PR_FX_LIGHTNING = 5,/* Lightning: up to 3 bolts on targets chosen at cast (SPEC §16.6.13) */
    PR_FX_BOMB = 6      /* a death bomb waiting for its fuse (SPEC §16.2, §16.6.8) */
};

/* Bandit dash (ledger combat.DASH_ATTACK) */
enum { PR_DASH_NONE = 0, PR_DASH_STAND = 1, PR_DASH_MOVE = 2, PR_DASH_HIT = 3 };

/* pending spawn kinds */
enum { PR_PEND_UNIT = 0, PR_PEND_BOMB = 1 };

/* ------------------------------------------------------------------ state records */

typedef struct PrEntity {
    uint32_t id;             /* creation id: unique in a match, never reused */
    int16_t unit;            /* PR_UNIT_* */
    int8_t card;             /* card id that created it; -1 for crown towers */
    uint8_t team;            /* 0 = bottom, 1 = top */
    uint8_t kind;            /* PR_KIND_* */
    int8_t tower_idx;        /* crown towers: 0 King, 1 Left, 2 Right (engine x); else -1 */
    uint8_t flying;          /* flying_height > 0 */
    uint8_t attacking;       /* set by the Attack phase: attacking this tick (holds movement) */
    int32_t x, y;            /* centre, millitiles, engine frame */
    int32_t radius;
    int32_t mass;
    int32_t hp, max_hp, shield;
    int32_t dmg_in;          /* damage buffered for the next Resolve (crown reduction applied) */
    int32_t drain_in;        /* lifetime drain buffered for Resolve (ignores hide) */
    int32_t spawn_tick;      /* index of the tick the entity was created for */
    int32_t deploy_ms;       /* remaining deploy time; > 0 = deploying */
    int32_t life_ticks;      /* lifetime drain ticks elapsed since deploy end */
    uint32_t target_id;      /* current attack target, PR_NO_ID = none */
    uint32_t prev_target_id; /* target seen by the previous Attack phase */
    int32_t load_ms;         /* ledger combat.ATTACK_CYCLE load timer */
    int32_t progress_ms;     /* ledger combat.ATTACK_CYCLE progress counter */
    int32_t buff_until[PR_N_BUFFS]; /* per buff row (status.BUFF_STACKING): the tick index from
                                     * which the buff is gone; 0 = inactive (SPEC §13.3) */
    /* movement cache: a pure function of the state, kept here so it snapshots */
    int32_t wp_x, wp_y;      /* waypoint */
    uint32_t wp_goal;        /* goal key the waypoint was computed for */
    uint32_t wp_sig;         /* occluder signature it was computed under */
    int16_t wp_cell;         /* the unit's cell when computed (-1 = invalid) */
    uint8_t wp_direct;       /* 1 = walk straight at the live goal point */
    uint8_t charged;         /* Prince: charge complete (Phase B) */
    int32_t charge_acc;      /* Prince: uninterrupted walking distance (Phase B) */
    uint8_t hide_state;      /* PR_HIDE_* (Tesla, Phase B) */
    uint8_t jumping;         /* over water on a river hop (Hog Rider / Prince) */
    uint8_t pad_[2];
    int32_t hide_ms;         /* Tesla hide/rise timer (Phase B) */
    int32_t kb_x, kb_y;      /* knockback displacement still to apply (Phase B) */
    int32_t kb_ticks;
    /* ---- SPEC §16 (v0.3, Phase E) ---- */
    uint8_t lvl;             /* Royal Chef level-ups received (0 = card level 11) */
    uint8_t burrow;          /* Miner travelling underground (hidden, SPEC §16.6.4 / §16.6.23) */
    uint8_t dash_state;      /* PR_DASH_* (Bandit) */
    uint8_t killed_by_dmg;   /* hp reached 0 through damage, not the lifetime drain (Collector) */
    uint8_t seq_idx;         /* Dagger Duchess: shots fired in the current AttackSequence */
    uint8_t pad2_[1];
    int16_t var_hits;        /* variable damage: hits landed on the current target */
    int16_t spawn_left;      /* periodic spawner: units still to emit in the current wave */
    int16_t pad3_;
    int32_t spawn_ms;        /* periodic spawner: time to the next emission */
    int32_t aux_ms;          /* Collector production / Chef cooking / Duchess reload / dash stand */
    int32_t tgt_x, tgt_y;    /* Miner: surfacing point (engine frame) */
} PrEntity;

typedef struct PrProjectile {
    uint32_t id;
    int16_t proj;            /* PR_PROJ_* */
    int8_t card;             /* firer's card (-1 crown tower), for observation/display */
    uint8_t team;
    uint8_t fresh;           /* created this tick: first step on the next tick */
    uint8_t has_target;      /* 1 = homing at target_id; 2 = aimed at target_id's launch position (non-homing,
                              * SPEC §16.6.2); 0 = flies to (tx, ty) (spells) */
    uint8_t pad_[2];
    int32_t x, y;
    int32_t tx, ty;          /* aim: the target's last known centre, or a fixed point */
    uint32_t target_id;
    uint32_t owner_id;
    int32_t damage;
    int32_t crown_pct;
    int32_t radius;          /* splash radius, 0 = single target */
    int32_t speed;
} PrProjectile;

typedef struct PrEffect {
    uint32_t id;
    uint8_t type;            /* PR_FX_* */
    uint8_t team;
    int8_t card;
    uint8_t flags;           /* PR_FX_BOMB: bit 0 hits air, bit 1 hits ground */
    int16_t area;            /* PR_AREA_* or -1 */
    int16_t proj;            /* PR_PROJ_* or -1 */
    int32_t x, y;
    int32_t radius;
    int32_t radius_y;
    int32_t damage;
    int32_t crown_pct;
    int32_t timer_ms;        /* time until the next event (wave, expiry) */
    int32_t remaining;       /* waves left / distance left to roll */
    int32_t dir_y;           /* roll direction for PR_FX_ROLLING: +1 or -1 (engine y) */
    uint32_t tgt[3];         /* PR_FX_LIGHTNING: bolt targets, highest hp first (PR_NO_ID = none) */
    int32_t life_ms;         /* PR_FX_PULSE: area life left */
    int32_t aux_ms;          /* PR_FX_PULSE: time to the next buff refresh */
    uint32_t hit_bits[PR_HIT_WORDS]; /* hit-once memory (SPEC §14.3): bit i = the entity in pool
                                      * slot i was hit; capacity = PR_MAX_ENTITIES, remapped by
                                      * pr_compact_entities, zero for slots >= n_ent */
} PrEffect;

typedef struct PrPendingSpawn {
    int16_t unit;            /* PR_PEND_UNIT: the unit to create; PR_PEND_BOMB: the dead unit (bomb row) */
    int8_t card;
    uint8_t team;
    uint8_t kind;            /* PR_PEND_* */
    uint8_t pad_[3];
    int32_t x, y;
    int32_t deploy_ms;
} PrPendingSpawn;

typedef struct PrPlay {
    int8_t slot;             /* hand slot 0..3 */
    int8_t card;             /* card in that slot when queued (re-checked in Upkeep) */
    int8_t tx, ty;           /* the acting team's OWN-frame tile (legality) */
    int32_t x, y;            /* engine-frame placement point (troops / spells) */
} PrPlay;

#define PR_STATE_MAGIC 0x50524F59u /* "PROY" */
#define PR_STATE_VERSION 4u /* v4: SPEC §16 (Phase E) entity / effect / state fields */

typedef struct PrState {
    uint32_t magic;
    uint32_t version;
    /* configuration (part of the state so snapshots carry it) */
    int32_t lockout_ticks;
    int32_t tiebreak;
    /* clock & randomness */
    int32_t tick;            /* ticks processed so far == index of the next tick */
    uint32_t next_id;        /* entity ids */
    uint32_t next_obj_id;    /* projectile / effect ids */
    PrRng rng;               /* game-logic PCG32 stream (deck shuffles, random decks) */
    int32_t random_deck[2];  /* 1 = redraw this team's deck at every reset */
    /* match */
    uint8_t over;
    int8_t result[2];        /* +1 win, -1 loss, 0 draw (valid once over) */
    uint8_t end_reason;      /* PR_END_* */
    int32_t crowns[2];
    int32_t king_wake[2];    /* ticks until the King activates; -1 = not triggered */
    uint8_t king_active[2];
    uint8_t first_team;      /* whose plays are applied first when both teams play on one tick */
    uint8_t alternate_first; /* config: 0 = team 0 always first (SPEC v0.2 §13.13), 1 = alternate */
    int8_t tower_troop[2];   /* config (SPEC §16.3): PR_TOWER_TROOP_UNITS index per team, 0 = princess */
    uint8_t pad_cfg_[2];
    /* economy & cards */
    int32_t elixir[2];       /* units (1 elixir = 2800) */
    int8_t deck[2][8];
    int8_t hand[2][4];
    int8_t queue[2][4];      /* queue[0] = next card */
    PrPlay pending[2][PR_MAX_PENDING_PLAYS];
    int32_t n_pending[2];
    /* public play history (observation deductions, Phase C) */
    int32_t spent[2];        /* elixir units spent */
    uint64_t seen_mask[2];   /* bit c = card c has been played by that team (64 cards) */
    int8_t last_played[2][4];/* most recent first, -1 = none */
    /* counters */
    int32_t leaked[2];       /* elixir units lost at the cap */
    int32_t plays[2];
    int32_t dropped_plays;   /* queued plays that failed validation at Upkeep (SPEC §14.1) */
    int32_t spawn_overflow;  /* entity/projectile/effect creations dropped at capacity */
    uint32_t tower_id[2][3]; /* ids of the crown towers [team][0 King, 1 Left, 2 Right] */
    /* pools (entities sorted by id; see pr_entity.h) */
    int32_t n_ent;
    int32_t n_proj;
    int32_t n_fx;
    int32_t n_spawn;
    PrEntity ent[PR_MAX_ENTITIES];
    PrProjectile proj[PR_MAX_PROJECTILES];
    PrEffect fx[PR_MAX_EFFECTS];
    PrPendingSpawn spawn[PR_MAX_PENDING_SPAWNS];
} PrState;

/* ------------------------------------------------------------------ small helpers */

static inline const PrUnitDef *pr_udef(const PrEntity *e) { return &PR_UNITS[e->unit]; }

/* Saturating add for a statistics counter (never multiplied; saturation is far beyond real play). */
static inline void pr_sat_add(int32_t *c, int64_t v) {
    int64_t r = (int64_t)*c + v;
    *c = (int32_t)(r > PR_COUNTER_MAX ? PR_COUNTER_MAX : r);
}

static inline int pr_is_structure(const PrEntity *e) {
    return e->kind == PR_KIND_BUILDING || e->kind == PR_KIND_TOWER;
}

/* Frame conversion (SPEC §1): team 1's own frame is the 180-degree rotation. Every coordinate
 * that reaches these is bounded (|x|, |y| <= 128000: debug inputs are clamped, snapshots are
 * validated), so W - x cannot overflow. */
static inline int32_t pr_own_x(int team, int32_t x) { return team == 0 ? x : PR_ARENA_W - x; }
static inline int32_t pr_own_y(int team, int32_t y) { return team == 0 ? y : PR_ARENA_H - y; }

#endif /* PR_TYPES_H */
