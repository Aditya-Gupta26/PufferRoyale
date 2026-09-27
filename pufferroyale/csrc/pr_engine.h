/*
 * pr_engine.h -- the PufferRoyale battle engine: reset and the tick pipeline (SPEC §5).
 *
 * Header-only C99; every function is static. Include this one header.
 *
 *   PrGame      = PrState (the hashed, snapshotted POD state) + PrPathCache (a pure
 *                 function of the state, see pr_path.h)
 *   pr_setup    configure decks / seed / rules and deal the first match
 *   pr_new_match  re-deal (continues the game RNG stream)
 *   pr_tick     advance one 50 ms tick through the 11 phases
 *   pr_hash     FNV-1a 64 over the PrState bytes
 *
 * Phase order (SPEC §5, ledger match.TICK_ORDER = client16402):
 *   1 Upkeep  2 Status  3 Spawn  4 Target  5 Attack  6 Path/Move  7 Collide
 *   8 Projectile (+ spell objects)  9 Resolve  10 Reap  11 Judge
 */
#ifndef PR_ENGINE_H
#define PR_ENGINE_H

#include "pr_types.h"
#include "pr_arena.h"
#include "pr_entity.h"
#include "pr_rules.h"
#include "pr_target.h"
#include "pr_combat.h"
#include "pr_path.h"
#include "pr_move.h"
#include "pr_spell.h"

#define PR_RNG_STREAM_GAME 0x5052u /* the game-logic PCG32 stream id */

typedef struct PrGame {
    PrState st;
    PrPathCache pc;
} PrGame;

/* ------------------------------------------------------------------ phases 1-3 */

static inline void pr_phase_upkeep(PrState *st) {
    pr_regen(st);
    /* Plays (SPEC v0.2.1 §14.1): all validated against the start-of-Upkeep state, then
     * applied; simultaneous plays team 0 first (SPEC v0.2 §13.13, so its units get the lower
     * creation ids). Opt-in alternative (config alternate_first = 1): the team applied first
     * alternates after every tick on which BOTH teams play (first_team starts at 0 each
     * match), which removes the systematic seat order. */
    pr_apply_tick_plays(st);
    for (int t = 0; t < 2; t++) {
        if (!st->king_active[t] && st->king_wake[t] > 0) {
            st->king_wake[t]--;
            if (st->king_wake[t] == 0) st->king_active[t] = 1;
        }
    }
}

/* Is any enemy that `e` could target inside its sight (Tesla rise trigger,
 * ledger hide.RISE_TRIGGER = enemy_in_sight_range)? */
static inline int pr_enemy_in_sight(const PrState *st, const PrEntity *e) {
    const PrUnitDef *d = pr_udef(e);
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *t = &st->ent[i];
        if (!pr_can_target(e, d, t)) continue;
        if (pr_within(e->x, e->y, t->x, t->y, pr_sight_reach(d, t))) return 1;
    }
    return 0;
}

/* Periodic spawner (SPEC §16.2, §16.6.5, §16.6.24): the timer runs on non-deploying, non-stunned
 * ticks; a wave of spawn_number units (all at once, or interval_ms apart), waves start to start
 * every pause_time_ms. Units are queued for this tick's Spawn phase (phase 3, right after this
 * one), born deployed, and map to the spawner's card. */
static inline void pr_spawner_tick(PrState *st, PrEntity *e, const PrUnitDef *d) {
    e->spawn_ms -= PR_TICK_MS;
    for (int guard = 0; e->spawn_ms <= 0 && guard < 64; guard++) {
        if (e->spawn_left <= 0) e->spawn_left = d->spawn_number; /* a new wave */
        if (d->spawn_interval_ms > 0) {
            int j = PR_CLAMP(d->spawn_number - e->spawn_left, 0, d->spawn_number - 1);
            pr_queue_layout(st, d->spawn_unit, e->team, e->card, e->x, e->y, d->spawn_first + j, 1, 1, 0);
            e->spawn_left--;
            e->spawn_ms += e->spawn_left > 0 ? d->spawn_interval_ms
                                             : PR_MAX(PR_TICK_MS, d->spawn_pause_ms - (d->spawn_number - 1) * d->spawn_interval_ms);
        } else {
            pr_queue_layout(st, d->spawn_unit, e->team, e->card, e->x, e->y, d->spawn_first, d->spawn_number,
                            d->spawn_number, 0);
            e->spawn_left = 0;
            e->spawn_ms += PR_MAX(PR_TICK_MS, d->spawn_pause_ms);
        }
    }
}

/* Elixir cost the Royal Chef ranks a troop by: its card's cost for the card's own members (summon
 * and second summon); 0 for units a spawner, a death or a spell released (a Witch's Skeletons,
 * Golemites, Goblin Barrel goblins) [IMPL-DEFINED]. */
static inline int pr_chef_cost(const PrEntity *u) {
    if (u->card < 0 || u->card >= PR_N_CARDS) return 0;
    const PrCardDef *c = &PR_CARDS[(int)u->card];
    return (c->unit == u->unit || c->unit2 == u->unit) ? c->elixir : 0;
}

/* Royal Chef (SPEC §16.3, §16.6.18, §16.6.26, §18.2): every chef_period_ms (counted from the match
 * start, paused while the tower is stunned) the Chef tower levels up the allied troop with the
 * highest elixir cost within chef_range (centre to centre) of it, ties to the lowest id; a troop is
 * levelled at most once [IMPL-DEFINED]; with nobody eligible the pancake waits. */
static inline void pr_chef_tick(PrState *st, PrEntity *tw, const PrUnitDef *d) {
    if (pr_is_stunned(tw)) return;
    if (tw->aux_ms < d->chef_period_ms) tw->aux_ms += PR_TICK_MS;
    if (tw->aux_ms < d->chef_period_ms) return;
    PrEntity *best = NULL;
    for (int i = 0; i < st->n_ent; i++) {
        PrEntity *u = &st->ent[i];
        if (u->hp <= 0 || u->team != tw->team || u->kind != PR_KIND_TROOP || u->lvl || u->burrow) continue;
        if (!pr_within(tw->x, tw->y, u->x, u->y, d->chef_range)) continue;
        if (!best || pr_chef_cost(u) > pr_chef_cost(best)) best = u;
    }
    if (!best) return;
    best->lvl = 1;
    int64_t num = PR_LADDER[PR_CARD_LEVEL], den = PR_LADDER[PR_CARD_LEVEL - 1];
    best->max_hp = (int32_t)(best->max_hp * num / den);
    best->hp = (int32_t)(best->hp * num / den);
    best->shield = (int32_t)(best->shield * num / den);
    tw->aux_ms = 0;
}

static inline void pr_phase_status(PrState *st) {
    for (int i = 0; i < st->n_ent; i++) {
        PrEntity *e = &st->ent[i];
        const PrUnitDef *d = pr_udef(e);
        /* the Miner surfaces at the tap point in tick P + ceil(d / speed) and deploys from there
         * (SPEC §16.6.23; its deploy timer does not run on the surfacing tick) */
        if (e->burrow) {
            if (st->tick - e->spawn_tick < pr_burrow_ticks(e)) continue;
            e->burrow = 0;
            e->hide_state = PR_HIDE_UP;
            e->x = e->tgt_x;
            e->y = e->tgt_y;
            e->deploy_ms = d->deploy_ms;
            e->spawn_tick = st->tick;
            e->load_ms = d->load_time_ms;
            e->wp_cell = -1;
        }
        /* buffs expire in the Status phase of their end tick (SPEC §13.3: a D ms buff applied
         * during tick t is active through tick t + D/50 - 1 and gone from t + D/50) */
        for (int b = 0; b < PR_N_BUFFS; b++)
            if (e->buff_until[b] != 0 && e->buff_until[b] <= st->tick) e->buff_until[b] = 0;
        /* deploy timer (not paused by stun: status.STUN_PAUSES_DEPLOY_TIMER = false);
         * it does not run on the tick the entity appeared, so a 1000 ms deploy keeps a
         * unit inactive for exactly 20 ticks (ledger movement.DEPLOY_TIMING) */
        if (e->deploy_ms > 0 && e->spawn_tick < st->tick) {
            e->deploy_ms = PR_MAX(0, e->deploy_ms - PR_TICK_MS);
            if (e->deploy_ms == 0 && d->hides) { /* hide.STARTS_HIDDEN */
                e->hide_state = PR_HIDE_HIDDEN;
                e->target_id = PR_NO_ID;
            }
        }
        if (e->deploy_ms > 0) continue;
        /* building lifetime (ledger lifetime.HP_DECAY = linear_drain, SPEC §6.5):
         * drain at elapsed tick k = floor(M(k+1)50/L) - floor(Mk50/L), so hp reaches 0
         * exactly L ms after deploy completes */
        if (d->lifetime_ms > 0) {
            int64_t k = e->life_ticks, M = e->max_hp, L = d->lifetime_ms;
            e->drain_in += (int32_t)((M * (k + 1) * PR_TICK_MS) / L - (M * k * PR_TICK_MS) / L);
            e->life_ticks++;
        }
        /* Tesla hide machine (ledger hide.*) */
        if (d->hides) {
            switch (e->hide_state) {
            case PR_HIDE_HIDDEN:
                if (!pr_is_stunned(e) && pr_enemy_in_sight(st, e)) {
                    e->hide_state = PR_HIDE_RISING;
                    e->hide_ms = d->up_ms;
                }
                break;
            case PR_HIDE_RISING:
                e->hide_ms -= PR_TICK_MS;
                if (e->hide_ms <= 0) {
                    e->hide_state = PR_HIDE_UP;
                    e->hide_ms = d->hide_ms;
                }
                break;
            default: /* UP: re-hide after hide_time_ms without a target */
                if (pr_get_c(st, e->target_id)) {
                    e->hide_ms = d->hide_ms;
                } else {
                    e->hide_ms -= PR_TICK_MS;
                    if (e->hide_ms <= 0) {
                        e->hide_state = PR_HIDE_HIDDEN;
                        e->hide_ms = 0; /* unused while hidden; kept in [0, max(hide, up)] */
                        e->target_id = PR_NO_ID;
                        e->progress_ms = 0;
                    }
                }
                break;
            }
        }
        if (d->spawn_unit >= 0 && !pr_is_stunned(e)) pr_spawner_tick(st, e, d);
        /* Elixir Collector (SPEC §16.6.15): +mana_amount at elapsed tick 259 after the deploy,
         * then every 260 ticks, capped at 10 elixir (the excess leaks) */
        if (d->mana_gen_ms > 0) {
            e->aux_ms += PR_TICK_MS;
            if (e->aux_ms >= d->mana_gen_ms) {
                e->aux_ms -= d->mana_gen_ms;
                pr_add_elixir(st, e->team, d->mana_amount);
            }
        }
    }
    /* Royal Chef towers (SPEC §18.2): after every entity's status, each team's Chef towers in
     * OWN-FRAME left-to-right order (engine left first for team 0, engine right first for team 1),
     * so a troop within reach of both is served by the same tower in a scenario and in its rotation
     * (a troop the first tower levels is no longer eligible for the second). */
    for (int team = 0; team < 2; team++) {
        int order[2] = {team == 0 ? 1 : 2, team == 0 ? 2 : 1}; /* own-left, own-right princess slots */
        for (int k = 0; k < 2; k++) {
            PrEntity *tw = pr_get(st, st->tower_id[team][order[k]]);
            if (!tw) continue;
            const PrUnitDef *d = pr_udef(tw);
            if (d->chef_period_ms > 0 && tw->kind == PR_KIND_TOWER) pr_chef_tick(st, tw, d);
        }
    }
}

static inline void pr_phase_spawn(PrState *st) {
    for (int i = 0; i < st->n_spawn; i++) {
        const PrPendingSpawn *s = &st->spawn[i];
        if (s->kind == PR_PEND_BOMB) { /* a death bomb: one timed area hit (SPEC §16.2, §16.6.8) */
            const PrUnitDef *d = &PR_UNITS[s->unit];
            PrEffect *f = pr_new_effect(st, PR_FX_BOMB, s->team, s->card, s->x, s->y);
            if (!f) continue;
            f->radius = d->bomb_radius;
            f->damage = d->bomb_damage;
            f->crown_pct = 100;
            f->flags = (uint8_t)((d->bomb_air ? 1 : 0) | (d->bomb_ground ? 2 : 0));
            f->timer_ms = s->deploy_ms;
            continue;
        }
        PrEntity *e = pr_new_entity(st, s->unit, s->team, s->card, s->x, s->y);
        if (e) e->deploy_ms = s->deploy_ms;
    }
    if (st->n_spawn) memset(st->spawn, 0, sizeof(PrPendingSpawn) * (size_t)st->n_spawn);
    st->n_spawn = 0;
}

/* ------------------------------------------------------------------ reset */

/* Deal a new match: keeps configuration, decks and the RNG stream; clears the rest. */
static inline void pr_new_match(PrState *st) {
    PrRng rng = st->rng;
    int32_t lockout = st->lockout_ticks, tiebreak = st->tiebreak;
    uint8_t alternate_first = st->alternate_first;
    int8_t troops[2] = {st->tower_troop[0], st->tower_troop[1]};
    int32_t rnd[2] = {st->random_deck[0], st->random_deck[1]};
    int8_t deck[2][8];
    memcpy(deck, st->deck, sizeof(deck));
    memset(st, 0, sizeof(*st));
    st->magic = PR_STATE_MAGIC;
    st->version = PR_STATE_VERSION;
    st->rng = rng;
    st->lockout_ticks = lockout;
    st->tiebreak = tiebreak;
    st->alternate_first = alternate_first;
    st->tower_troop[0] = troops[0];
    st->tower_troop[1] = troops[1];
    st->random_deck[0] = rnd[0];
    st->random_deck[1] = rnd[1];
    memcpy(st->deck, deck, sizeof(deck));
    for (int t = 0; t < 2; t++) {
        if (st->random_deck[t]) { /* 8 distinct of all the cards, uniformly (partial Fisher-Yates) */
            int8_t pool[PR_N_CARDS];
            for (int c = 0; c < PR_N_CARDS; c++) pool[c] = (int8_t)c;
            for (int k = 0; k < 8; k++) {
                int j = k + (int)pr_rng_below(&st->rng, (uint32_t)(PR_N_CARDS - k));
                int8_t tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp;
                st->deck[t][k] = pool[k];
            }
        }
    }
    for (int t = 0; t < 2; t++) { /* shuffle (Fisher-Yates), hand = [0:4], queue = [4:8] */
        int8_t order[8];
        memcpy(order, st->deck[t], 8);
        for (int k = 7; k > 0; k--) {
            int j = (int)pr_rng_below(&st->rng, (uint32_t)(k + 1));
            int8_t tmp = order[k]; order[k] = order[j]; order[j] = tmp;
        }
        memcpy(st->hand[t], order, 4);
        memcpy(st->queue[t], order + 4, 4);
        st->elixir[t] = PR_ELIXIR_START;
        memset(st->last_played[t], -1, 4);
    }
    pr_setup_towers(st);
}

static inline void pr_reseed(PrState *st, uint64_t seed) { pr_rng_seed(&st->rng, seed, PR_RNG_STREAM_GAME); }

/* Configure and deal. deck[t] = 8 card ids, or NULL for a random deck per reset; tower_troop[t] =
 * PR_TOWER_TROOP_UNITS index (0 = princess, SPEC §16.3). */
static inline void pr_setup_ex(PrGame *g, const int8_t *deck0, const int8_t *deck1, uint64_t seed,
                               int32_t lockout_ticks, int32_t tiebreak, int tower_troop0, int tower_troop1) {
    memset(g, 0, sizeof(*g));
    PrState *st = &g->st;
    st->tower_troop[0] = (int8_t)PR_CLAMP(tower_troop0, 0, PR_N_TOWER_TROOPS - 1);
    st->tower_troop[1] = (int8_t)PR_CLAMP(tower_troop1, 0, PR_N_TOWER_TROOPS - 1);
    const int8_t *decks[2] = {deck0, deck1};
    for (int t = 0; t < 2; t++) {
        if (decks[t]) memcpy(st->deck[t], decks[t], 8);
        else st->random_deck[t] = 1;
    }
    st->lockout_ticks = lockout_ticks;
    st->tiebreak = tiebreak;
    pr_reseed(st, seed);
    pr_new_match(st);
}

static inline void pr_setup(PrGame *g, const int8_t *deck0, const int8_t *deck1, uint64_t seed,
                     int32_t lockout_ticks, int32_t tiebreak) {
    pr_setup_ex(g, deck0, deck1, seed, lockout_ticks, tiebreak, 0, 0);
}

/* ------------------------------------------------------------------ the tick */

static inline void pr_tick(PrGame *g) {
    PrState *st = &g->st;
    if (st->over) return;
    int32_t t = st->tick;
    pr_phase_upkeep(st);          /* 1 */
    pr_phase_status(st);          /* 2 */
    pr_phase_spawn(st);           /* 3 */
    pr_phase_target(st);          /* 4 */
    pr_phase_attack(st);          /* 5 */
    pr_phase_move(st, &g->pc);    /* 6 */
    pr_phase_collide(st);         /* 7 */
    pr_phase_projectiles(st);     /* 8 */
    pr_phase_effects(st);
    pr_phase_resolve(st);         /* 9 */
    pr_phase_reap(st);            /* 10 */
    st->tick = t + 1;
    pr_judge(st, t, 0);           /* 11 */
}

static inline uint64_t pr_hash(const PrState *st) { return pr_fnv1a(PR_FNV_OFFSET, st, sizeof(*st)); }

/* ------------------------------------------------------------------ debug / test hooks */

/* TEST HOOK (SPEC §10 Game.spawn): put a card's units on the board now, bypassing
 * elixir, hand and legality. Spells are cast at (x, y). Returns the ids created. */
#define PR_DEBUG_COORD_MAX (4 * PR_ARENA_H) /* debug coordinates are clamped to +-this */

static inline int pr_debug_spawn(PrGame *g, int team, int card, int32_t x, int32_t y, int deployed,
                          uint32_t *out, int max_out) {
    if (card < 0 || card >= PR_N_CARDS || team < 0 || team > 1) return 0;
    /* keep every later frame conversion / distance free of signed overflow */
    x = PR_CLAMP(x, -PR_DEBUG_COORD_MAX, PR_DEBUG_COORD_MAX);
    y = PR_CLAMP(y, -PR_DEBUG_COORD_MAX, PR_DEBUG_COORD_MAX);
    return pr_deploy_card(&g->st, team, card, x, y, deployed, 0, out, max_out);
}

/* Set a crown tower's hp. hp <= 0 destroys it now, with all side effects (crowns,
 * King trigger, pocket), then the end conditions are checked immediately. */
static inline int pr_debug_set_tower_hp(PrGame *g, int team, int idx, int32_t hp) {
    PrState *st = &g->st;
    PrEntity *t = pr_get(st, st->tower_id[team][idx]);
    if (!t) return -1;
    if (hp > t->max_hp) hp = t->max_hp;
    if (hp > 0) {
        t->hp = hp;
        return 0;
    }
    t->hp = 0;
    pr_tower_destroyed(st, t);
    pr_compact_entities(st);
    pr_judge(st, st->tick - 1, 1);
    return 0;
}

/* Enable the alternating simultaneous-play order and choose which team goes first on the
 * next tick where both teams play (hook for exact mirror tests). */
static inline void pr_debug_set_first_team(PrGame *g, int team) {
    g->st.alternate_first = 1;
    g->st.first_team = (uint8_t)(team ? 1 : 0);
}

/* Set hand = order[0:4], queue = order[4:8]; the deck becomes these 8 cards, also for later
 * deals (a random-deck team stops redrawing): reset re-deals the CURRENT decks. */
static inline int pr_debug_set_hand(PrGame *g, int team, const int8_t order[8]) {
    PrState *st = &g->st;
    for (int i = 0; i < 8; i++) {
        if (order[i] < 0 || order[i] >= PR_N_CARDS) return -1;
        for (int j = 0; j < i; j++)
            if (order[j] == order[i]) return -1;
    }
    st->random_deck[team] = 0;
    memcpy(st->deck[team], order, 8);
    memcpy(st->hand[team], order, 4);
    memcpy(st->queue[team], order + 4, 4);
    st->n_pending[team] = 0;
    memset(st->pending[team], 0, sizeof(st->pending[team]));
    return 0;
}

/* ------------------------------------------------------------------ snapshot validation */

#define PR_CHK_BIG PR_COUNTER_MAX /* bound on counters / timers: keeps later arithmetic in range */

static inline int pr_chk_card(int c) { return c >= 0 && c < PR_N_CARDS; }
static inline int pr_chk_xy(int64_t x, int64_t y) {
    return x >= -PR_DEBUG_COORD_MAX && x <= PR_DEBUG_COORD_MAX && y >= -PR_DEBUG_COORD_MAX && y <= PR_DEBUG_COORD_MAX;
}
/* SPEC §18.5 bounds. Every per-tick growth below is a sound upper bound, so each tick-dependent
 * bound is preserved by pr_tick (a validated state stays validated): */
#define PR_CHK_PROG_STEP 10050   /* attack progress per tick: max(hit_speed, load_time + 50) of any attacker */
#define PR_CHK_WALK_STEP 1000    /* charge run-up per tick: >= any charged speed */
#define PR_CHK_DMG 100000        /* a projectile's / effect's damage (level-scaled data <= ~2,000) */
#define PR_CHK_NEW_PER_TICK 512  /* entity / object ids consumed per tick (pool capacities) */

/* max_hp / shield cap of a unit at Royal Chef level `lvl` (0 or 1): the level-11 value, times
 * ladder[12] / ladder[11] once levelled (SPEC §16.6.18). */
static inline int32_t pr_lvl_cap(int32_t v, int lvl) {
    return lvl ? (int32_t)(((int64_t)v * PR_LADDER[PR_CARD_LEVEL]) / PR_LADDER[PR_CARD_LEVEL - 1]) : v;
}

/* The largest aux_ms a unit's role uses (Collector production, Chef cooking (+ one tick past the
 * period while a pancake waits), Duchess reload, dash stand); 0 = the unit never uses aux_ms. */
static inline int32_t pr_aux_cap(const PrUnitDef *d) {
    int32_t m = PR_MAX(PR_MAX(d->mana_gen_ms, d->reload_ms), d->dash_cooldown_ms);
    if (d->chef_period_ms > 0) m = PR_MAX(m, d->chef_period_ms + PR_TICK_MS);
    return m;
}


/* Is `s` a structurally valid engine state (SPEC v0.2.1 §14.5)? Checks every index the engine
 * uses to address a table or pool, the pool counts and the id order, the card bookkeeping and
 * bounds on every counter, so that simulating, observing, masking or rendering a state that
 * passes can neither read out of bounds nor overflow. Returns 1, or 0 with a reason in *why. */
static inline int pr_state_check(const PrState *s, const char **why) {
#define PR_CHK(cond, msg) do { if (!(cond)) { if (why) *why = (msg); return 0; } } while (0)
    PR_CHK(s->magic == PR_STATE_MAGIC && s->version == PR_STATE_VERSION, "not a snapshot of this engine version");
    PR_CHK(s->lockout_ticks >= 0 && s->lockout_ticks <= PR_CHK_BIG, "lockout out of range");
    PR_CHK(s->tiebreak == PR_TIEBREAK_ABSOLUTE || s->tiebreak == PR_TIEBREAK_FRACTION, "bad tiebreak");
    PR_CHK(s->tick >= 0 && s->tick <= PR_TICKS_MAX, "tick outside [0, 6000]");
    PR_CHK(s->over <= 1 && s->end_reason <= PR_END_DRAW, "bad match result");
    PR_CHK(s->over || s->tick < PR_TICKS_MAX, "a running match past tick 5999");
    /* id headroom for the rest of the match (at most PR_CHK_NEW_PER_TICK new ids per tick) */
    PR_CHK((uint64_t)s->next_id + (uint64_t)(PR_TICKS_MAX - s->tick + 1) * PR_CHK_NEW_PER_TICK <= 0xFFFFFFFFull &&
               (uint64_t)s->next_obj_id + (uint64_t)(PR_TICKS_MAX - s->tick + 1) * PR_CHK_NEW_PER_TICK <= 0xFFFFFFFFull,
           "id counter out of range");
    PR_CHK(s->first_team <= 1 && s->alternate_first <= 1, "bad play-order flags");
    PR_CHK(s->tower_troop[0] >= 0 && s->tower_troop[0] < PR_N_TOWER_TROOPS && s->tower_troop[1] >= 0 &&
               s->tower_troop[1] < PR_N_TOWER_TROOPS && s->pad_cfg_[0] == 0 && s->pad_cfg_[1] == 0,
           "bad tower troop");
    for (int t = 0; t < 2; t++) {
        PR_CHK(s->random_deck[t] == 0 || s->random_deck[t] == 1, "bad random-deck flag");
        PR_CHK(s->result[t] >= -1 && s->result[t] <= 1, "bad result");
        PR_CHK(s->crowns[t] >= 0 && s->crowns[t] <= 3, "crowns out of range");
        PR_CHK(s->king_wake[t] >= -1 && s->king_wake[t] <= PR_KING_WAKE_TICKS, "king timer out of range");
        PR_CHK(s->king_active[t] <= 1, "bad king flag");
        PR_CHK(s->elixir[t] >= 0 && s->elixir[t] <= PR_ELIXIR_MAX, "elixir out of range");
        /* statistics counters saturate at PR_CHK_BIG (pr_sat_add); spent / plays grow by at most 4 plays
         * of at most 10 elixir per tick */
        PR_CHK(s->spent[t] >= 0 && (int64_t)s->spent[t] <= (int64_t)s->tick * 4 * 10 * PR_ELIXIR_UNIT &&
                   s->leaked[t] >= 0 && s->leaked[t] <= PR_CHK_BIG && s->plays[t] >= 0 &&
                   (int64_t)s->plays[t] <= (int64_t)s->tick * PR_MAX_PENDING_PLAYS, "counter out of range");
        PR_CHK((PR_N_CARDS >= 64 ? 0 : (s->seen_mask[t] >> (PR_N_CARDS & 63))) == 0, "bad seen-card mask");
        /* deck: 8 distinct cards; hand + queue: a permutation of the deck */
        uint64_t deck = 0, held = 0;
        for (int i = 0; i < 8; i++) {
            PR_CHK(pr_chk_card(s->deck[t][i]), "deck card out of range");
            PR_CHK(!(deck & ((uint64_t)1 << s->deck[t][i])), "deck cards not distinct");
            deck |= (uint64_t)1 << s->deck[t][i];
            int8_t c = i < 4 ? s->hand[t][i] : s->queue[t][i - 4];
            PR_CHK(pr_chk_card(c), "hand/queue card out of range");
            held |= (uint64_t)1 << c;
        }
        PR_CHK(held == deck, "hand + queue is not the deck");
        for (int i = 0; i < 4; i++) PR_CHK(s->last_played[t][i] >= -1 && s->last_played[t][i] < PR_N_CARDS, "bad play history");
        PR_CHK(s->n_pending[t] >= 0 && s->n_pending[t] <= PR_MAX_PENDING_PLAYS, "pending plays out of range");
        for (int i = 0; i < PR_MAX_PENDING_PLAYS; i++) {
            const PrPlay *p = &s->pending[t][i];
            if (i >= s->n_pending[t]) continue; /* unused: never read, zeroed by pr_state_canonicalize */
            PR_CHK(p->slot >= 0 && p->slot < 4 && pr_chk_card(p->card), "bad pending play");
            PR_CHK(p->tx >= 0 && p->tx < PR_TILES_X && p->ty >= 0 && p->ty < PR_TILES_Y, "bad pending tile");
            PR_CHK(p->x >= 0 && p->x <= PR_ARENA_W && p->y >= 0 && p->y <= PR_ARENA_H, "bad pending point");
        }
    }
    PR_CHK(s->dropped_plays >= 0 && (int64_t)s->dropped_plays <= (int64_t)s->tick * 2 * PR_MAX_PENDING_PLAYS,
           "counter out of range");
    PR_CHK(s->spawn_overflow >= 0 && s->spawn_overflow <= PR_CHK_BIG, "counter out of range");
    PR_CHK(s->n_ent >= 0 && s->n_ent <= PR_MAX_ENTITIES, "entity count out of range");
    PR_CHK(s->n_proj >= 0 && s->n_proj <= PR_MAX_PROJECTILES, "projectile count out of range");
    PR_CHK(s->n_fx >= 0 && s->n_fx <= PR_MAX_EFFECTS, "effect count out of range");
    PR_CHK(s->n_spawn >= 0 && s->n_spawn <= PR_MAX_PENDING_SPAWNS, "pending spawn count out of range");
    for (int i = 0; i < s->n_ent; i++) {
        const PrEntity *e = &s->ent[i];
        PR_CHK(e->id < s->next_id && (i == 0 || e->id > s->ent[i - 1].id), "entity ids not strictly increasing");
        PR_CHK(e->unit >= 0 && e->unit < PR_N_UNITS, "unit index out of range");
        PR_CHK(e->kind == PR_UNITS[e->unit].kind, "entity kind does not match its unit");
        PR_CHK(e->team <= 1 && e->flying <= 1 && e->attacking <= 1 && e->charged <= 1 && e->jumping <= 1 &&
                   e->wp_direct <= 1, "bad entity flag");
        if (e->kind == PR_KIND_TOWER) PR_CHK(e->card == -1 && e->tower_idx >= 0 && e->tower_idx <= 2, "bad tower");
        else PR_CHK(pr_chk_card(e->card) && e->tower_idx == -1, "entity card out of range");
        PR_CHK(pr_chk_xy(e->x, e->y) && pr_chk_xy(e->wp_x, e->wp_y), "entity position out of range");
        PR_CHK(e->radius >= 0 && e->radius <= 20000 && e->mass >= 0 && e->mass <= 1000000, "bad radius or mass");
        const PrUnitDef *ud = &PR_UNITS[e->unit];
        PR_CHK(e->lvl <= 1 && (e->lvl == 0 || e->kind == PR_KIND_TROOP), "bad level");
        /* SPEC §18.5: hp values are bounded by the unit's level-cap values, so nothing the engine
         * multiplies (lifetime drain, Chef level-up) can overflow */
        PR_CHK(e->max_hp >= 1 && e->max_hp <= pr_lvl_cap(ud->hp, e->lvl) && e->hp >= -PR_CHK_BIG &&
                   e->hp <= e->max_hp && e->shield >= 0 && e->shield <= pr_lvl_cap(ud->shield, e->lvl),
               "hp out of range");
        PR_CHK(e->dmg_in >= 0 && e->dmg_in <= PR_CHK_BIG && e->drain_in >= 0 && e->drain_in <= PR_CHK_BIG,
               "buffered damage out of range");
        PR_CHK(e->spawn_tick >= 0 && e->spawn_tick <= s->tick, "spawn tick out of range");
        PR_CHK(ud->lifetime_ms > 0 ? e->life_ticks >= 0 && e->life_ticks <= ud->lifetime_ms / PR_TICK_MS + 1
                                   : e->life_ticks == 0, "lifetime counter out of range");
        /* the load timer is set to load_time and only counts down (the fresh-cycle credit
         * load_time - load_ms is never negative) */
        PR_CHK(e->deploy_ms >= 0 && e->deploy_ms <= PR_CHK_BIG && e->load_ms >= 0 && e->load_ms <= ud->load_time_ms,
               "timer out of range");
        PR_CHK(e->progress_ms >= 0 && (int64_t)e->progress_ms <= (int64_t)s->tick * PR_CHK_PROG_STEP,
               "attack progress out of range");
        for (int b = 0; b < PR_N_BUFFS; b++) PR_CHK(e->buff_until[b] >= 0 && e->buff_until[b] <= PR_CHK_BIG, "buff out of range");
        PR_CHK(e->wp_cell >= -1 && e->wp_cell < PR_N_CELLS, "waypoint cell out of range");
        PR_CHK(ud->charge_range > 0 ? e->charge_acc >= 0 && (int64_t)e->charge_acc <= (int64_t)s->tick * PR_CHK_WALK_STEP
                                    : e->charge_acc == 0 && e->charged == 0, "charge out of range");
        if (ud->hides) /* hide_ms counts down from hide / up time and is reset on hiding */
            PR_CHK(e->hide_state <= PR_HIDE_RISING && e->hide_ms >= 0 && e->hide_ms <= PR_MAX(ud->hide_ms, ud->up_ms),
                   "bad hide state");
        else
            PR_CHK(e->hide_state == (e->burrow ? PR_HIDE_HIDDEN : PR_HIDE_UP) && e->hide_ms == 0, "bad hide state");
        PR_CHK(e->kb_x == 0 && e->kb_y == 0 && e->kb_ticks == 0, "bad knockback"); /* reserved, unused */
        /* SPEC §16 fields (state version 4) */
        PR_CHK(e->killed_by_dmg <= 1 && e->dash_state <= PR_DASH_HIT && e->pad2_[0] == 0 && e->pad3_ == 0,
               "bad entity flag");
        PR_CHK(e->burrow <= 1 && (!e->burrow || (ud->burrow_speed > 0 && e->hide_state == PR_HIDE_HIDDEN)),
               "bad burrow state");
        PR_CHK(e->dash_state == PR_DASH_NONE || ud->dash_damage > 0, "dash state on a unit without a dash");
        PR_CHK(e->seq_idx == 0 || e->seq_idx < ud->seq_n, "bad attack-sequence index");
        PR_CHK(e->var_hits >= 0 && e->var_hits <= 10000, "variable-damage counter out of range");
        if (ud->spawn_unit >= 0) /* a spawner's timer is > 0 after every tick */
            PR_CHK(e->spawn_left >= 0 && e->spawn_left <= ud->spawn_number && e->spawn_ms > 0 &&
                       e->spawn_ms <= PR_MAX(PR_MAX(ud->spawn_start_ms, ud->spawn_pause_ms), PR_TICK_MS),
                   "spawner timer out of range");
        else
            PR_CHK(e->spawn_left == 0 && e->spawn_ms == 0, "spawner timer on a non-spawner");
        PR_CHK(e->aux_ms >= 0 && e->aux_ms <= pr_aux_cap(ud), "role timer out of range");
        PR_CHK(pr_chk_xy(e->tgt_x, e->tgt_y), "entity target point out of range");
    }
    /* crown-tower slots: a living tower_id is the matching tower (team, index, unit), every tower entity
     * is its slot's, and a running match's crowns match the fallen enemy princess towers */
    int n_towers = 0;
    for (int i = 0; i < s->n_ent; i++) n_towers += s->ent[i].kind == PR_KIND_TOWER;
    int n_slots = 0;
    for (int t = 0; t < 2; t++) {
        int alive_princess = 0;
        for (int k = 0; k < 3; k++) {
            const PrEntity *e = pr_get_c(s, s->tower_id[t][k]);
            if (!e) continue;
            n_slots++;
            int want = k == 0 ? PR_UNIT_KINGTOWER : PR_TOWER_TROOP_UNITS[s->tower_troop[t]];
            PR_CHK(e->kind == PR_KIND_TOWER && e->team == t && e->tower_idx == k && e->unit == want, "bad tower slot");
            alive_princess += k > 0;
        }
        PR_CHK(s->over || s->crowns[1 - t] + alive_princess <= 2, "crowns do not match the fallen towers");
    }
    PR_CHK(n_slots == n_towers, "a tower entity without its slot");
    for (int i = 0; i < s->n_proj; i++) {
        const PrProjectile *p = &s->proj[i];
        PR_CHK(p->proj >= 0 && p->proj < PR_N_PROJS && p->card >= -1 && p->card < PR_N_CARDS, "projectile index out of range");
        PR_CHK(p->team <= 1 && p->fresh <= 1 && p->has_target <= 2, "bad projectile flag"); /* 2 = aimed (non-homing) */
        PR_CHK(pr_chk_xy(p->x, p->y) && pr_chk_xy(p->tx, p->ty), "projectile position out of range");
        PR_CHK(p->damage >= 0 && p->damage <= PR_CHK_DMG && p->crown_pct >= 0 && p->crown_pct <= 100 &&
                   p->radius >= 0 && p->radius <= 20000 && p->speed >= 0 && p->speed <= 100000, "bad projectile stats");
    }
    for (int i = 0; i < s->n_fx; i++) {
        const PrEffect *f = &s->fx[i];
        PR_CHK(f->team <= 1 && f->card >= -1 && f->card < PR_N_CARDS, "bad effect");
        PR_CHK(f->area >= -1 && f->area < PR_N_AREAS && f->proj >= -1 && f->proj < PR_N_PROJS, "effect index out of range");
        PR_CHK((f->type == PR_FX_AREA && f->area >= 0) || (f->type == PR_FX_WAVES && f->card >= 0) ||
                   (f->type == PR_FX_ROLLING && f->proj >= 0 && (f->dir_y == 1 || f->dir_y == -1)) ||
                   (f->type == PR_FX_PULSE && f->card >= 0 && f->area >= 0 && PR_AREAS[f->area].buff >= 0) ||
                   (f->type == PR_FX_LIGHTNING && f->area >= 0 && f->proj >= 0 && f->remaining >= 1 &&
                    f->remaining <= 3) ||
                   (f->type == PR_FX_BOMB && f->flags <= 3),
               "bad effect type");
        PR_CHK(pr_chk_xy(f->x, f->y), "effect position out of range");
        PR_CHK(f->life_ms >= -PR_CHK_BIG && f->life_ms <= PR_CHK_BIG && f->aux_ms >= -PR_CHK_BIG &&
                   f->aux_ms <= PR_CHK_BIG, "bad effect timer");
        PR_CHK(f->radius >= 0 && f->radius <= 20000 && f->radius_y >= 0 && f->radius_y <= 20000 && f->damage >= 0 &&
                   f->damage <= PR_CHK_DMG && f->crown_pct >= 0 && f->crown_pct <= 100, "bad effect stats");
        if (f->type == PR_FX_PULSE) { /* life counts down with the event timer: timer - life never falls */
            const PrAreaDef *ad = &PR_AREAS[f->area];
            int32_t iv = f->card >= 0 ? PR_CARDS[(int)f->card].wave_interval_ms : 0;
            PR_CHK(f->life_ms > 0 && f->life_ms <= ad->life_ms && f->timer_ms <= PR_MAX(iv, PR_TICK_MS) &&
                       (int64_t)f->timer_ms - f->life_ms >= -(int64_t)ad->life_ms - PR_MAX(iv, PR_TICK_MS) &&
                       f->aux_ms >= -PR_TICK_MS && f->aux_ms <= PR_MAX(ad->hit_speed_ms, PR_TICK_MS) &&
                       f->remaining >= 0 && f->remaining <= 1000, "bad pulse timers");
        }
        PR_CHK(f->timer_ms >= -PR_CHK_BIG && f->timer_ms <= PR_CHK_BIG && f->remaining >= -PR_CHK_BIG &&
                   f->remaining <= PR_CHK_BIG, "bad effect timer");
    }
    for (int i = 0; i < s->n_spawn; i++) {
        const PrPendingSpawn *p = &s->spawn[i];
        PR_CHK(p->unit >= 0 && p->unit < PR_N_UNITS && p->card >= -1 && p->card < PR_N_CARDS && p->team <= 1,
               "pending spawn index out of range");
        /* SPEC §18.5: a pending unit is a real non-tower unit of a valid card; a bomb names a unit with one */
        PR_CHK(pr_chk_card(p->card), "pending spawn card out of range");
        PR_CHK((p->kind == PR_PEND_UNIT && PR_UNITS[p->unit].kind != PR_KIND_TOWER) ||
                   (p->kind == PR_PEND_BOMB && PR_UNITS[p->unit].bomb_damage > 0),
               "bad pending spawn kind");
        PR_CHK(p->pad_[0] == 0 && p->pad_[1] == 0 && p->pad_[2] == 0, "bad pending spawn padding");
        PR_CHK(pr_chk_xy(p->x, p->y) && p->deploy_ms >= 0 && p->deploy_ms <= PR_CHK_BIG, "bad pending spawn");
    }
    return 1;
#undef PR_CHK
}

/* Zero everything of a checked state the engine never reads -- unused pool slots, stale pending
 * plays, hit-once bits of empty slots -- so that the hash depends only on live content. Restore
 * applies it after pr_state_check (SPEC §14.5: live content is validated; the rest is normalised). */
static inline void pr_state_canonicalize(PrState *s) {
    for (int t = 0; t < 2; t++)
        for (int i = s->n_pending[t]; i < PR_MAX_PENDING_PLAYS; i++) memset(&s->pending[t][i], 0, sizeof(PrPlay));
    memset(&s->ent[s->n_ent], 0, sizeof(PrEntity) * (size_t)(PR_MAX_ENTITIES - s->n_ent));
    memset(&s->proj[s->n_proj], 0, sizeof(PrProjectile) * (size_t)(PR_MAX_PROJECTILES - s->n_proj));
    memset(&s->fx[s->n_fx], 0, sizeof(PrEffect) * (size_t)(PR_MAX_EFFECTS - s->n_fx));
    memset(&s->spawn[s->n_spawn], 0, sizeof(PrPendingSpawn) * (size_t)(PR_MAX_PENDING_SPAWNS - s->n_spawn));
    for (int i = 0; i < s->n_fx; i++)
        for (int k = s->n_ent; k < PR_MAX_ENTITIES; k++) s->fx[i].hit_bits[k >> 5] &= ~(1u << (k & 31));
}

#endif /* PR_ENGINE_H */
