/*
 * pr_entity.h -- the entity pool: creation, lookup, removal and formations.
 *
 * Entities live in PrState.ent[0 .. n_ent) SORTED BY ID. New entities get the next id
 * and are appended; removal is a stable compaction. So iterating the array is
 * iterating in creation order -- every per-entity pass in the engine uses that order,
 * and "lowest entity id" tie-breaks are array order. Lookups by id are binary search.
 * Vacated slots are zeroed so the state hash depends only on live content.
 */
#ifndef PR_ENTITY_H
#define PR_ENTITY_H

#include "pr_arena.h"

/* Index of entity `id`, or -1. */
static inline int pr_find(const PrState *st, uint32_t id) {
    if (id == PR_NO_ID) return -1;
    int lo = 0, hi = st->n_ent - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t m = st->ent[mid].id;
        if (m == id) return mid;
        if (m < id) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* Live entity by id (hp > 0), or NULL. */
static inline PrEntity *pr_get(PrState *st, uint32_t id) {
    int i = pr_find(st, id);
    if (i < 0) return NULL;
    PrEntity *e = &st->ent[i];
    return e->hp > 0 ? e : NULL;
}

static inline const PrEntity *pr_get_c(const PrState *st, uint32_t id) {
    int i = pr_find(st, id);
    if (i < 0) return NULL;
    const PrEntity *e = &st->ent[i];
    return e->hp > 0 ? e : NULL;
}

/* Create an entity of `unit` at (x, y). Returns NULL (and counts the overflow) when
 * the pool is full (SPEC §1). The unit starts deploying for its data deploy time and
 * with load_timer = load_time (SPEC §6.2: the timer runs down during deploy). */
static inline PrEntity *pr_new_entity(PrState *st, int unit, int team, int card, int32_t x, int32_t y) {
    if (st->n_ent >= PR_MAX_ENTITIES) {
        pr_sat_add(&st->spawn_overflow, 1);
        return NULL;
    }
    const PrUnitDef *d = &PR_UNITS[unit];
    PrEntity *e = &st->ent[st->n_ent++];
    memset(e, 0, sizeof(*e));
    e->id = st->next_id++;
    e->unit = (int16_t)unit;
    e->card = (int8_t)card;
    e->team = (uint8_t)team;
    e->kind = d->kind;
    e->tower_idx = -1;
    e->flying = d->flying_height > 0;
    e->x = x;
    e->y = y;
    e->radius = d->radius;
    e->mass = d->mass;
    e->hp = d->hp;
    e->max_hp = d->hp;
    e->shield = d->shield;
    e->spawn_tick = st->tick;
    e->deploy_ms = d->deploy_ms;
    e->target_id = PR_NO_ID;
    e->prev_target_id = PR_NO_ID;
    e->load_ms = d->load_time_ms;
    e->wp_cell = -1;
    e->wp_goal = PR_NO_ID;
    /* a hiding building is visible while it deploys and goes under at deploy end
     * (ledger hide.STARTS_HIDDEN; the Status phase runs the machine) */
    e->hide_state = PR_HIDE_UP;
    /* periodic spawner (SPEC §16.6.5 / §16.6.24): the timer only runs on non-deploying ticks,
     * so the first wave lands at elapsed tick k = start/50 - 1 (pause/50 - 1 without a start) */
    if (d->spawn_unit >= 0) e->spawn_ms = d->spawn_start_ms > 0 ? d->spawn_start_ms : d->spawn_pause_ms;
    if (d->seq_n > 0) e->aux_ms = 0; /* Dagger Duchess: a full sequence ready */
    return e;
}

/* Royal Chef level-ups (SPEC §16.6.18): a stat of a levelled entity is the level-11 value times
 * ladder[L+1] / ladder[L], floored (one level-up at most, docs/FIDELITY.md §12). */
static inline int32_t pr_lvl(const PrEntity *e, int32_t v) {
    if (e->lvl == 0) return v;
    return (int32_t)(((int64_t)v * PR_LADDER[PR_CARD_LEVEL]) / PR_LADDER[PR_CARD_LEVEL - 1]);
}

static inline int pr_bit_get(const uint32_t *bits, int i) { return (int)((bits[i >> 5] >> (i & 31)) & 1u); }
static inline void pr_bit_set(uint32_t *bits, int i) { bits[i >> 5] |= 1u << (i & 31); }

/* Remove every entity with hp <= 0 (stable compaction, vacated slots zeroed). The effects'
 * hit-once bitsets are keyed by pool slot, so they are remapped to the new slots (bits of
 * removed entities are dropped). */
static inline void pr_compact_entities(PrState *st) {
    int16_t moved[PR_MAX_ENTITIES];
    int w = 0;
    for (int r = 0; r < st->n_ent; r++) {
        if (st->ent[r].hp > 0) {
            moved[r] = (int16_t)w;
            if (w != r) st->ent[w] = st->ent[r];
            w++;
        } else {
            moved[r] = -1;
        }
    }
    if (w == st->n_ent) return;
    for (int k = 0; k < st->n_fx; k++) {
        uint32_t *hb = st->fx[k].hit_bits, nb[PR_HIT_WORDS];
        int any = 0;
        for (int q = 0; q < PR_HIT_WORDS; q++) any |= hb[q] != 0;
        if (!any) continue;
        memset(nb, 0, sizeof(nb));
        for (int r = 0; r < st->n_ent; r++)
            if (moved[r] >= 0 && pr_bit_get(hb, r)) pr_bit_set(nb, moved[r]);
        memcpy(hb, nb, sizeof(nb));
    }
    memset(&st->ent[w], 0, sizeof(PrEntity) * (size_t)(st->n_ent - w));
    st->n_ent = w;
}

/* Move a GROUND point to legal land (SPEC §5.1 [IMPL-DEFINED]): clamp into the
 * arena, then off water to the nearest bank/bridge point. */
static inline void pr_to_land(int team, int32_t *x, int32_t *y) {
    pr_clamp_arena(x, y);
    pr_eject_water(team, x, y);
    pr_clamp_arena(x, y);
}

/* A ground formation member that landed in water goes to the tap's side of the river
 * (SPEC v0.2.1 §14.4): straight across to the own bank for a tap on the own half, to the far
 * bank for a tap in a pocket -- never across the river. Computed in the team's own frame. */
static inline void pr_to_land_side(int team, int tap_own_half, int32_t *x, int32_t *y) {
    pr_clamp_arena(x, y);
    if (pr_point_wet(*x, *y)) {
        int32_t bx = *x, by = pr_own_y(team, tap_own_half ? PR_RIVER_Y1 + 1 : PR_RIVER_Y0 - 1);
        if (!pr_point_wet(bx, by)) {
            *y = by;
        } else {
            pr_eject_water(team, x, y); /* not reachable on this arena; kept as a safe fallback */
        }
    }
    pr_clamp_arena(x, y);
}

/* Spawn `count` members of `unit` around (px, py) with formation `formation`
 * (offsets in the team's OWN frame, rotated for team 1). Member k deploys for
 * deploy_ms + k * delay_ms (SPEC §5 stagger). Ground members outside the arena are
 * clamped into it, and members in water go to the tap's bank (pr_to_land_side). Writes up
 * to `max_out` new ids to `out` (may be NULL); returns the number created. */
static inline int pr_spawn_formation2(PrState *st, int unit, int count, int unit2, int count2, int team, int card,
                                      int32_t px, int32_t py, int formation, int32_t delay_ms,
                                      uint32_t *out, int max_out) {
    const PrFormationDef *f = (formation >= 0 && formation < PR_N_FORMATIONS) ? &PR_FORMATIONS[formation] : NULL;
    int tap_own_half = pr_own_y(team, py) >= (PR_RIVER_Y0 + PR_RIVER_Y1) / 2;
    int made = 0;
    int total = count + (unit2 >= 0 ? count2 : 0);
    for (int k = 0; k < total; k++) {
        int u = k < count ? unit : unit2;   /* summon first, then second_summon (SPEC §16.1) */
        const PrUnitDef *d = &PR_UNITS[u];
        int32_t ox = 0, oy = 0;
        if (f && k < f->count) {
            ox = PR_FORMATION_OFFSETS[f->first + k][0];
            oy = PR_FORMATION_OFFSETS[f->first + k][1];
        }
        if (team == 1) { ox = -ox; oy = -oy; }
        int32_t x = px + ox, y = py + oy;
        if (d->flying_height > 0) pr_clamp_arena(&x, &y);
        else pr_to_land_side(team, tap_own_half, &x, &y);
        PrEntity *e = pr_new_entity(st, u, team, card, x, y);
        if (!e) break;
        e->deploy_ms = d->deploy_ms + k * delay_ms;
        if (out && made < max_out) out[made] = e->id;
        made++;
    }
    return made;
}

/* One-unit form (v0.2 signature, kept for callers/tests): `deploy_ms` is the unit's deploy time. */
static inline int pr_spawn_formation(PrState *st, int unit, int team, int card, int32_t px, int32_t py,
                              int count, int formation, int32_t deploy_ms, int32_t delay_ms,
                              uint32_t *out, int max_out) {
    int first = st->n_ent;
    int n = pr_spawn_formation2(st, unit, count, -1, 0, team, card, px, py, formation, delay_ms, out, max_out);
    for (int i = first; i < st->n_ent; i++) st->ent[i].deploy_ms = deploy_ms + (i - first) * delay_ms;
    return n;
}

/* Queue a spawn for the next Spawn phase (death spawns, spawner waves, Goblin Barrel / Barbarian
 * Barrel landings, death bombs). */
static inline void pr_queue_pending(PrState *st, int kind, int unit, int team, int card, int32_t x, int32_t y,
                                    int32_t deploy_ms) {
    if (st->n_spawn >= PR_MAX_PENDING_SPAWNS) {
        pr_sat_add(&st->spawn_overflow, 1);
        return;
    }
    PrPendingSpawn *s = &st->spawn[st->n_spawn++];
    memset(s, 0, sizeof(*s));
    s->unit = (int16_t)unit;
    s->team = (uint8_t)team;
    s->card = (int8_t)card;
    s->kind = (uint8_t)kind;
    s->x = x;
    s->y = y;
    s->deploy_ms = deploy_ms;
}

static inline void pr_queue_spawn(PrState *st, int unit, int team, int card, int32_t x, int32_t y, int32_t deploy_ms) {
    pr_queue_pending(st, PR_PEND_UNIT, unit, team, card, x, y, deploy_ms);
}

/* Queue `count` units of `unit` around (px, py) at the PR_SPAWN_OFFSETS layout starting at `first`
 * (the owner's own frame, rotated for team 1; offsets past the layout repeat its last point).
 * Ground units that would stand in water or outside the arena go to the nearest land point. */
static inline void pr_queue_layout(PrState *st, int unit, int team, int card, int32_t px, int32_t py, int first,
                                   int npts, int count, int32_t deploy_ms) {
    const PrUnitDef *d = &PR_UNITS[unit];
    for (int k = 0; k < count; k++) {
        int j = first + (npts > 0 ? PR_MIN(k, npts - 1) : 0);
        int32_t ox = 0, oy = 0;
        if (j >= 0 && j < PR_N_SPAWN_OFFSETS) {
            ox = PR_SPAWN_OFFSETS[j][0];
            oy = PR_SPAWN_OFFSETS[j][1];
        }
        if (team == 1) { ox = -ox; oy = -oy; }
        int32_t x = px + ox, y = py + oy;
        if (d->flying_height > 0) pr_clamp_arena(&x, &y);
        else pr_to_land(team, &x, &y);
        pr_queue_spawn(st, unit, team, card, x, y, deploy_ms);
    }
}

/* ------------------------------------------------------------------ status queries */

/* Most negative active multiplier of a buff field (status.BUFF_STACKING one slot per
 * row, movement.BUFF_SPEED_COMPOSITION strongest). 0 when no buff is active. */
static inline int32_t pr_buff_speed_pct(const PrEntity *e) {
    int32_t m = 0;
    for (int b = 0; b < PR_N_BUFFS; b++)
        if (e->buff_until[b] != 0 && PR_BUFFS[b].speed_pct < m) m = PR_BUFFS[b].speed_pct;
    return m;
}

static inline int32_t pr_buff_hit_pct(const PrEntity *e) {
    int32_t m = 0;
    for (int b = 0; b < PR_N_BUFFS; b++)
        if (e->buff_until[b] != 0 && PR_BUFFS[b].hit_speed_pct < m) m = PR_BUFFS[b].hit_speed_pct;
    return m;
}

/* Stunned/frozen: a full-stop buff is active (status.FULL_STOP_BUFF_IS_STUN). */
static inline int pr_is_stunned(const PrEntity *e) {
    for (int b = 0; b < PR_N_BUFFS; b++)
        if (e->buff_until[b] != 0 && PR_BUFFS[b].speed_pct <= -100) return 1;
    return 0;
}

static inline int pr_is_slowed(const PrEntity *e) {
    for (int b = 0; b < PR_N_BUFFS; b++)
        if (e->buff_until[b] != 0 && PR_BUFFS[b].speed_pct < 0 && PR_BUFFS[b].speed_pct > -100) return 1;
    return 0;
}

static inline int pr_is_hidden(const PrEntity *e) { return e->hide_state == PR_HIDE_HIDDEN; }

/* A Bandit moving in her dash, or on its arrival tick before Resolve (ledger combat.DASH_ATTACK,
 * SPEC §16.2, §18.3): no damage from any hit dealt now. */
static inline int pr_dash_immune(const PrEntity *e) { return e->dash_state >= PR_DASH_MOVE; }

static inline int pr_is_deploying(const PrEntity *e) { return e->deploy_ms > 0; }

#endif /* PR_ENTITY_H */
