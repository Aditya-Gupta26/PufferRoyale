/*
 * pr_move.h -- the Path/Move and Collide phases (SPEC §6.3, §6.4).
 */
#ifndef PR_MOVE_H
#define PR_MOVE_H

#include "pr_combat.h"
#include "pr_path.h"

/* SPEC §6.3: floor(speed * (100 + slow_pct) / 100), x2 (data percent) when charged,
 * 0 when stunned/frozen (ledger movement.BUFF_SPEED_RULE). */
static inline int32_t pr_effective_speed(const PrEntity *e) {
    const PrUnitDef *d = pr_udef(e);
    if (pr_is_stunned(e)) return 0;
    int32_t s = (int32_t)pr_floordiv((int64_t)d->speed * (100 + pr_buff_speed_pct(e)), 100);
    if (e->charged && d->charge_speed_pct > 0) s = s * d->charge_speed_pct / 100;
    return PR_MAX(s, 0);
}

/* Goal of a moving troop: its attack target if it has one (not yet in range --
 * units in range are attacking and do not move), else the default route tower. */
static inline const PrEntity *pr_move_goal(const PrState *st, const PrEntity *e) {
    const PrEntity *t = pr_get_c(st, e->target_id);
    return t ? t : pr_default_tower(st, e);
}

/* Goal key of a mover of `team`: the structure id, or the chase block (2x2 tiles) holding
 * the chased troop's cell under the mover's own cell convention. Quantising troop goals to
 * blocks lets every chaser of nearby targets share one field and keeps a field valid while
 * its target moves inside the block [IMPL-DEFINED]. */
static inline uint32_t pr_goal_key(int team, const PrEntity *g) {
    if (pr_is_structure(g)) return PR_GOAL_STRUCT | g->id;
    int c = pr_cell_of_t(team, g->x, g->y);
    if (c < 0) c = 0;
    int b = (c / PR_COLS / PR_BLOCK) * PR_BLOCK_COLS + (c % PR_COLS) / PR_BLOCK;
    return PR_GOAL_BLOCK | (uint32_t)b;
}

/* ------------------------------------------------------------------ Miner burrow (SPEC §16.6.4) */

/* Ticks the underground walk from the own King's centre to (tgt_x, tgt_y) takes: ceil(d / speed). */
static inline int32_t pr_burrow_ticks(const PrEntity *e) {
    const PrUnitDef *d = pr_udef(e);
    int64_t dist = pr_isqrt64(pr_dist2(PR_TOWER_POS[e->team][0][0], PR_TOWER_POS[e->team][0][1], e->tgt_x, e->tgt_y));
    if (d->burrow_speed <= 0 || dist <= 0) return 0;
    return (int32_t)pr_ceildiv(dist, d->burrow_speed);
}

/* Position after k whole ticks of the walk (exact interpolation on the straight line, so the Miner
 * stands on the tap after exactly ceil(d / speed) ticks whatever the rounding). */
static inline void pr_burrow_place(PrEntity *e, int32_t k) {
    int32_t n = pr_burrow_ticks(e);
    int32_t sx = PR_TOWER_POS[e->team][0][0], sy = PR_TOWER_POS[e->team][0][1];
    if (k >= n) {
        e->x = e->tgt_x;
        e->y = e->tgt_y;
        return;
    }
    int64_t dist = pr_isqrt64(pr_dist2(sx, sy, e->tgt_x, e->tgt_y));
    int64_t done = (int64_t)k * pr_udef(e)->burrow_speed;
    e->x = sx + (int32_t)(((int64_t)e->tgt_x - sx) * done / dist);
    e->y = sy + (int32_t)(((int64_t)e->tgt_y - sy) * done / dist);
}

/* ------------------------------------------------------------------ Bandit dash (combat.DASH_ATTACK) */

/* Is the straight segment from a to b free of water for a ground non-jumper? (A dash never
 * crosses the river [IMPL-DEFINED]; sampled every 250 millitiles.) */
static inline int pr_dry_segment(int32_t ax, int32_t ay, int32_t bx, int32_t by) {
    int64_t dx = (int64_t)bx - ax, dy = (int64_t)by - ay;
    int64_t n = PR_MAX(pr_abs64(dx), pr_abs64(dy)) / 250 + 1;
    for (int64_t i = 0; i <= n; i++)
        if (pr_point_wet(ax + (int32_t)(dx * i / n), ay + (int32_t)(dy * i / n))) return 0;
    return 1;
}

/* SPEC §16.6.14: a walking unit with a dash starts one (stands for DashCooldown) when its target
 * is not yet in attack range and min_range + r_a + r_t <= dist <= max_range + r_a + r_t. */
static inline int pr_dash_trigger(const PrEntity *e, const PrUnitDef *d, const PrEntity *t) {
    if (d->dash_damage <= 0 || pr_in_attack_range(e, d, t)) return 0;
    int64_t lo = (int64_t)d->dash_min + e->radius + t->radius, hi = (int64_t)d->dash_max + e->radius + t->radius;
    int64_t d2 = pr_dist2(e->x, e->y, t->x, t->y);
    if (d2 < lo * lo || d2 > hi * hi) return 0;
    return e->flying || pr_dry_segment(e->x, e->y, t->x, t->y);
}

/* One tick of a dash in progress; returns 1 while the unit is held by it (no walking). The move
 * is JumpSpeed per tick as two half-steps, stopping after the first whose edge gap to the target
 * is within Range: the dash hit (DashDamage) lands on that arrival tick. */
static inline int pr_dash_step(PrState *st, PrEntity *e, const PrUnitDef *d) {
    if (e->dash_state != PR_DASH_STAND && e->dash_state != PR_DASH_MOVE) return 0;
    PrEntity *t = pr_get(st, e->target_id);
    if (!t || !pr_can_act(st, e)) {
        e->dash_state = PR_DASH_NONE;
        return 0;
    }
    if (e->dash_state == PR_DASH_STAND) {
        e->aux_ms -= PR_TICK_MS;
        if (e->aux_ms > 0) return 1;
        e->dash_state = PR_DASH_MOVE;
        e->aux_ms = 0;
    }
    int32_t half = PR_MAX(1, d->dash_speed / 2);
    for (int k = 0; k < 2; k++) {
        pr_step_toward(&e->x, &e->y, t->x, t->y, half);
        int64_t gap = (int64_t)pr_dist(e->x, e->y, t->x, t->y) - e->radius - t->radius;
        if (gap <= d->range) {
            pr_hit(t, pr_lvl(e, d->dash_damage), d->crown_pct);
            e->dash_state = PR_DASH_HIT;  /* immune through this tick's Resolve */
            e->progress_ms = 0;
            e->load_ms = d->load_time_ms; /* the melee cycle starts over: first swing at arrival + 19 */
            e->prev_target_id = t->id;
            break;
        }
    }
    pr_clamp_arena(&e->x, &e->y);
    e->wp_cell = -1;
    return 1;
}

static inline void pr_phase_move(PrState *st, PrPathCache *pc) {
    pr_path_sync(st, pc);
    for (int i = 0; i < st->n_ent; i++) {
        PrEntity *e = &st->ent[i];
        if (e->kind != PR_KIND_TROOP || e->hp <= 0) continue;
        const PrUnitDef *d = pr_udef(e);
        if (e->burrow) { /* underground: k whole ticks walked after this one */
            pr_burrow_place(e, st->tick - e->spawn_tick + 1);
            continue;
        }
        if (pr_dash_step(st, e, d)) continue;
        int32_t speed = 0;
        const PrEntity *g = NULL;
        if (!pr_is_deploying(e) && !e->attacking && !pr_is_stunned(e)) {
            speed = pr_effective_speed(e);
            g = pr_move_goal(st, e);
            const PrEntity *t = pr_get_c(st, e->target_id);
            if (t && e->dash_state == PR_DASH_NONE && pr_dash_trigger(e, d, t)) {
                e->dash_state = PR_DASH_STAND; /* stands from this tick for DashCooldown */
                e->aux_ms = d->dash_cooldown_ms;
                speed = 0;
            }
        }
        int32_t moved = 0;
        if (speed > 0 && g) {
            int32_t wx = g->x, wy = g->y;
            if (!e->flying) {
                uint32_t key = pr_goal_key(e->team, g);
                int cell = pr_cell_of_t(e->team, e->x, e->y);
                if (e->wp_cell != cell || e->wp_goal != key || e->wp_sig != pr_sig32(pc->sig))
                    pr_plan(pc, e, g->x, g->y, key, pr_is_structure(g) ? -1 : pr_cell_of_t(e->team, g->x, g->y));
                if (!e->wp_direct) {
                    wx = e->wp_x;
                    wy = e->wp_y;
                }
            }
            moved = pr_step_toward(&e->x, &e->y, wx, wy, speed);
        }
        /* Prince charge (SPEC §6.2): an uninterrupted walk of charge_range charges the
         * unit; any tick without walking resets the run-up (charge.PROGRESS_ON_STOP). */
        if (d->charge_range > 0) {
            if (moved > 0) {
                e->charge_acc += moved;
                if (e->charge_acc >= d->charge_range) e->charged = 1;
            } else {
                e->charge_acc = 0;
                e->charged = 0;
            }
        }
    }
}

/* ------------------------------------------------------------------ Collide (SPEC §6.4) */

static inline int pr_collides(const PrEntity *e) {
    return e->kind == PR_KIND_TROOP && e->hp > 0 && !e->jumping && !e->burrow && e->dash_state != PR_DASH_MOVE;
}

/* Push ground troop u fully out of structure s's collision circle. */
static inline void pr_push_out_of(PrEntity *u, const PrEntity *s) {
    int64_t rs = (int64_t)u->radius + s->radius;
    int64_t dx = (int64_t)u->x - s->x, dy = (int64_t)u->y - s->y;
    if (pr_abs64(dx) >= rs || pr_abs64(dy) >= rs) return;
    int64_t d2 = dx * dx + dy * dy;
    if (d2 >= rs * rs) return;
    if (d2 == 0) { /* coincident: toward the unit's own side [IMPL-DEFINED] */
        u->y = s->y + (int32_t)(u->team == 0 ? rs : -rs);
        return;
    }
    int64_t d = pr_isqrt64(d2);
    if (d == 0) d = 1;
    /* scale (dx, dy) to length rs, rounding away from the centre */
    int64_t nx = dx * rs, ny = dy * rs;
    int64_t ox = nx >= 0 ? pr_ceildiv(nx, d) : -pr_ceildiv(-nx, d);
    int64_t oy = ny >= 0 ? pr_ceildiv(ny, d) : -pr_ceildiv(-ny, d);
    u->x = s->x + (int32_t)ox;
    u->y = s->y + (int32_t)oy;
}

static inline void pr_phase_collide(PrState *st) {
    int n = st->n_ent;
    int32_t dxs[PR_MAX_ENTITIES], dys[PR_MAX_ENTITIES];
    memset(dxs, 0, sizeof(int32_t) * (size_t)n);
    memset(dys, 0, sizeof(int32_t) * (size_t)n);
    /* 1. unit-unit separation: one simultaneous pass (displacements from start-of-pass
     *    positions, applied together), overlap split by inverse mass (SPEC §6.4). Each
     *    pair's contribution is independent and integer sums commute, so the pairs are
     *    enumerated by a sweep over tile rows (counting sort) instead of all n^2/2. */
    int16_t order[PR_MAX_ENTITIES], rowof[PR_MAX_ENTITIES];
    int cnt[PR_TILES_Y + 1];
    memset(cnt, 0, sizeof(cnt));
    int nc = 0;
    int32_t rmax = 0;
    for (int i = 0; i < n; i++) {
        const PrEntity *e = &st->ent[i];
        if (!pr_collides(e)) continue;
        rowof[i] = (int16_t)PR_CLAMP(e->y / 1000, 0, PR_TILES_Y - 1);
        cnt[rowof[i] + 1]++;
        if (e->radius > rmax) rmax = e->radius;
        nc++;
    }
    for (int r = 0; r < PR_TILES_Y; r++) cnt[r + 1] += cnt[r];
    for (int i = 0; i < n; i++)
        if (pr_collides(&st->ent[i])) order[cnt[rowof[i]]++] = (int16_t)i;
    int window = 2 * rmax / 1000 + 1; /* |dy| < r_a + r_b <= 2 rmax */
    for (int p = 0; p < nc; p++) {
        for (int q = p + 1; q < nc && rowof[order[q]] - rowof[order[p]] <= window; q++) {
            int i = PR_MIN(order[p], order[q]), j = PR_MAX(order[p], order[q]);
            const PrEntity *a = &st->ent[i];
            const PrEntity *b = &st->ent[j];
            if (a->flying != b->flying) continue;
            int64_t rs = (int64_t)a->radius + b->radius;
            int64_t dx = (int64_t)a->x - b->x, dy = (int64_t)a->y - b->y;
            if (pr_abs64(dx) >= rs || pr_abs64(dy) >= rs) continue;
            int64_t d2 = dx * dx + dy * dy;
            if (d2 >= rs * rs) continue;
            int64_t ma = PR_MAX(a->mass, 1), mb = PR_MAX(b->mass, 1);
            if (d2 == 0) {
                /* coincident centres [IMPL-DEFINED, seat-symmetric]: different teams go
                 * toward their own sides; same team: the lower id to its own-left, the
                 * higher to its own-right. */
                int64_t sa = rs * mb / (ma + mb), sb = rs * ma / (ma + mb);
                if (a->team != b->team) {
                    dys[i] += (int32_t)(a->team == 0 ? sa : -sa);
                    dys[j] += (int32_t)(b->team == 0 ? sb : -sb);
                } else {
                    int32_t left = a->team == 0 ? -1 : 1;
                    dxs[i] += (int32_t)(left * sa);
                    dxs[j] -= (int32_t)(left * sb);
                }
                continue;
            }
            int64_t d = pr_isqrt64(d2);
            if (d == 0) d = 1;
            int64_t overlap = rs - d, den = (ma + mb) * d;
            dxs[i] += (int32_t)(dx * overlap * mb / den);
            dys[i] += (int32_t)(dy * overlap * mb / den);
            dxs[j] -= (int32_t)(dx * overlap * ma / den);
            dys[j] -= (int32_t)(dy * overlap * ma / den);
        }
    }
    for (int i = 0; i < n; i++) {
        PrEntity *e = &st->ent[i];
        if (dxs[i] == 0 && dys[i] == 0) continue;
        /* [IMPL-DEFINED] a unit is displaced at most its own radius per tick, so a
         * dense spawn spreads over a few ticks instead of exploding */
        int64_t m2 = (int64_t)dxs[i] * dxs[i] + (int64_t)dys[i] * dys[i];
        int64_t lim = e->radius;
        if (m2 > lim * lim) {
            int64_t m = pr_isqrt64(m2);
            dxs[i] = (int32_t)((int64_t)dxs[i] * lim / m);
            dys[i] = (int32_t)((int64_t)dys[i] * lim / m);
        }
        e->x += dxs[i];
        e->y += dys[i];
    }
    /* 2. ground troops are pushed fully out of building/tower circles (structures never
     *    move), sequentially over structures in id order. */
    for (int i = 0; i < n; i++) {
        PrEntity *u = &st->ent[i];
        if (u->kind != PR_KIND_TROOP || u->hp <= 0 || u->flying || u->jumping || u->burrow) continue;
        for (int j = 0; j < n; j++) {
            const PrEntity *s = &st->ent[j];
            if (s->hp <= 0 || !pr_is_structure(s)) continue;
            pr_push_out_of(u, s);
        }
    }
    /* 3. arena bounds, water (ground non-jumpers never end a tick in water) and the
     *    river-hop state of jumpers. */
    for (int i = 0; i < n; i++) {
        PrEntity *u = &st->ent[i];
        if (u->kind != PR_KIND_TROOP || u->hp <= 0 || u->burrow) continue;
        pr_clamp_arena(&u->x, &u->y);
        if (u->flying) continue;
        if (PR_UNITS[u->unit].jumps) {
            u->jumping = (uint8_t)pr_point_wet(u->x, u->y);
        } else if (pr_point_wet(u->x, u->y)) {
            pr_to_land(u->team, &u->x, &u->y);
        }
    }
}

#endif /* PR_MOVE_H */
