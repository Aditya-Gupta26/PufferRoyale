/*
 * pr_spell.h -- spells and area-effect objects (SPEC §6.6, §7.2).
 *
 *   Fireball       point projectile from the caster's King centre to the tap point
 *                  (ledger spells.LAUNCH_POINT = caster_king_tower_centre); splash on
 *                  arrival with knockback away from the impact centre.
 *   Arrows         an effect at the tap point: the first wave lands when a projectile
 *                  at 1100/tick from the King would arrive, then every 200 ms, 3 waves;
 *                  each wave hits every enemy within 3500 + r once.
 *   Zap            applied at once during the play tick (damage buffered, stun now).
 *   The Log        a rolling rectangle (half-width 1950 x half-depth 600) moving toward
 *                  the enemy at 200/tick for 10100; each ground enemy hit at most once
 *                  (a bitset over all pool slots, so no entity can go unrecorded),
 *                  knockback 700 in the roll direction.
 *   Goblin Barrel  point projectile at 400/tick; the landing spawns 3 Goblins.
 *   Area effects   (Zap, the Ice Golem's death slow) hit once on creation (damage and
 *                  buff), ledger spells.ONE_SHOT_AREA_EFFECT_APPLICATION.
 */
#ifndef PR_SPELL_H
#define PR_SPELL_H

#include "pr_combat.h"

static inline PrEffect *pr_new_effect(PrState *st, int type, int team, int card, int32_t x, int32_t y) {
    if (st->n_fx >= PR_MAX_EFFECTS) {
        pr_sat_add(&st->spawn_overflow, 1);
        return NULL;
    }
    PrEffect *f = &st->fx[st->n_fx++];
    memset(f, 0, sizeof(*f));
    f->id = st->next_obj_id++;
    f->type = (uint8_t)type;
    f->team = (uint8_t)team;
    f->card = (int8_t)card;
    f->area = -1;
    f->proj = -1;
    f->x = x;
    f->y = y;
    return f;
}

static inline void pr_area_apply(PrState *st, const PrAreaDef *a, int team, int32_t x, int32_t y, int with_damage) {
    PrAreaHit h;
    memset(&h, 0, sizeof(h));
    h.damage = with_damage ? a->damage : 0;
    h.crown_pct = a->crown_pct;
    h.buff = a->buff;
    h.buff_ms = a->buff_ms;
    h.hits_air = a->hits_air;
    h.hits_ground = a->hits_ground;
    h.affects_hidden = a->affects_hidden;
    pr_area_hit(st, team, x, y, a->radius, &h);
}

/* An area-effect object with no hit speed (Zap, the Ice Golem's death slow) applies
 * exactly once, when it appears, whatever its life duration (ledger
 * spells.ONE_SHOT_AREA_EFFECT_APPLICATION = first_update_only). Pulsing areas (a hit
 * speed: Poison, Earthquake -- none in v0.1) would be PR_FX_AREA objects. */
static inline void pr_spawn_area_effect(PrState *st, int area, int team, int card, int32_t x, int32_t y) {
    (void)card;
    pr_area_apply(st, &PR_AREAS[area], team, x, y, 1);
}

/* A pulsing area (Poison, Earthquake; SPEC §16.1, §16.6.10-11). The area's buff carries the
 * damage: each damage EVENT (every hit_frequency_ms from the cast, life / hit_frequency events)
 * deals dps x period / 1000 to every enemy inside (radius + r, the area's air/ground filter);
 * crown towers take the buff's crown percent, non-crown buildings its building percent. The
 * buff itself (the slow) is (re)applied to everyone inside every area hit_speed_ms (250 / 100 ms)
 * while the area lives, for buff_time_ms -- capped at the area's remaining life when the data
 * says cap_buff_time_to_area_effect_time (Earthquake). */
static inline void pr_pulse_hit(PrState *st, const PrEffect *f, int damage) {
    const PrAreaDef *a = &PR_AREAS[f->area];
    const PrBuffDef *b = &PR_BUFFS[a->buff];
    PrAreaHit h;
    memset(&h, 0, sizeof(h));
    h.damage = damage ? f->damage : 0;
    h.crown_pct = b->crown_pct;
    h.building_pct = b->building_pct;
    h.buff = damage ? -1 : a->buff;
    h.buff_ms = a->cap_buff ? PR_MIN(a->buff_ms, f->life_ms) : a->buff_ms;
    h.hits_air = a->hits_air;
    h.hits_ground = a->hits_ground;
    h.affects_hidden = a->affects_hidden;
    pr_area_hit(st, f->team, f->x, f->y, f->radius, &h);
}

/* One Lightning bolt (SPEC §16.6.13): the bolt's projectile damage (crown %) and its 500 ms stun on
 * target k, skipped when the target died before its bolt. */
static inline void pr_lightning_bolt(PrState *st, const PrEffect *f, int k) {
    if (k < 0 || k > 2) return;
    PrEntity *t = pr_get(st, f->tgt[k]);
    if (!t || t->burrow) return;
    const PrProjDef *d = &PR_PROJS[f->proj];
    pr_hit(t, f->damage, f->crown_pct);
    if (d->target_buff >= 0) pr_apply_buff(t, d->target_buff, d->buff_ms, st->tick);
}

/* Lightning's targets at cast: the (up to) 3 highest CURRENT-hp valid enemies within radius + r
 * (air and ground, not hidden), ties to the lowest id; tgt[k] = the k-th highest (SPEC §16.6.25). */
static inline void pr_lightning_pick(const PrState *st, PrEffect *f, const PrAreaDef *a) {
    f->tgt[0] = f->tgt[1] = f->tgt[2] = PR_NO_ID;
    int32_t hp[3] = {0, 0, 0};
    for (int i = 0; i < st->n_ent; i++) { /* ascending id: a later equal hp never displaces */
        const PrEntity *v = &st->ent[i];
        if (v->hp <= 0 || v->team == f->team || v->burrow || pr_is_hidden(v)) continue;
        if (v->flying ? !a->hits_air : !a->hits_ground) continue;
        if (!pr_within(f->x, f->y, v->x, v->y, (int64_t)f->radius + v->radius)) continue;
        int k = 3;
        while (k > 0 && (f->tgt[k - 1] == PR_NO_ID || v->hp > hp[k - 1])) k--;
        if (k >= 3) continue;
        for (int j = 2; j > k; j--) { f->tgt[j] = f->tgt[j - 1]; hp[j] = hp[j - 1]; }
        f->tgt[k] = v->id;
        hp[k] = v->hp;
    }
}

/* Tick (relative to the cast tick P) of Lightning bolt k: ceil(hit_speed * k / 50). */
static inline int32_t pr_bolt_tick(const PrAreaDef *a, int k) {
    return (int32_t)pr_ceildiv((int64_t)a->hit_speed_ms * k, PR_TICK_MS);
}

/* Ticks until a projectile at `speed` from (x0, y0) reaches (x1, y1) under the
 * impact rule "remaining <= speed", counting the launch tick as tick 1. */
static inline int32_t pr_flight_ticks(int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t speed) {
    int64_t d = pr_isqrt64(pr_dist2(x0, y0, x1, y1));
    if (d <= speed) return 1;
    return (int32_t)pr_ceildiv(d, speed);
}

static inline void pr_cast_spell(PrState *st, int team, int card, int32_t x, int32_t y) {
    const PrCardDef *c = &PR_CARDS[card];
    int32_t kx = PR_TOWER_POS[team][0][0], ky = PR_TOWER_POS[team][0][1];
    switch (c->spell_type) {
    case PR_SPELL_PROJECTILE:
    case PR_SPELL_SPAWN_PROJECTILE: {
        PrProjectile *p = pr_new_projectile(st, c->spell_proj, team, card, kx, ky);
        if (!p) return;
        p->has_target = 0;
        p->tx = x;
        p->ty = y;
        break;
    }
    case PR_SPELL_WAVES: {
        PrEffect *f = pr_new_effect(st, PR_FX_WAVES, team, card, x, y);
        if (!f) return;
        f->proj = c->spell_proj;
        f->radius = c->spell_radius;
        f->damage = c->spell_damage;
        f->crown_pct = c->crown_pct;
        f->remaining = c->waves;
        f->timer_ms = (pr_flight_ticks(kx, ky, x, y, PR_PROJS[c->spell_proj].speed) - 1) * PR_TICK_MS;
        break;
    }
    case PR_SPELL_AREA:  /* Zap, Freeze: applied once, now (SPEC §13.12, §16.6.12) */
        pr_spawn_area_effect(st, c->spell_area, team, card, x, y);
        break;
    case PR_SPELL_PULSE: { /* Poison, Earthquake: first event now, then one every period */
        PrEffect *f = pr_new_effect(st, PR_FX_PULSE, team, card, x, y);
        if (!f) return;
        const PrAreaDef *a = &PR_AREAS[c->spell_area];
        f->area = c->spell_area;
        f->radius = c->spell_radius;
        f->damage = c->spell_damage;
        f->crown_pct = c->crown_pct;
        f->life_ms = a->life_ms;
        pr_pulse_hit(st, f, 0);          /* the buff first: a unit it kills is not buffed */
        pr_pulse_hit(st, f, 1);
        f->remaining = c->waves - 1;     /* damage events still to come */
        f->timer_ms = c->wave_interval_ms;
        f->aux_ms = PR_MAX(PR_TICK_MS, a->hit_speed_ms);
        break;
    }
    case PR_SPELL_LIGHTNING: { /* targets chosen now; bolt 0 now, bolt k at P + ceil(460 k / 50) */
        PrEffect *f = pr_new_effect(st, PR_FX_LIGHTNING, team, card, x, y);
        if (!f) return;
        const PrAreaDef *a = &PR_AREAS[c->spell_area];
        f->area = c->spell_area;
        f->proj = c->spell_proj;
        f->radius = c->spell_radius;
        f->damage = c->spell_damage;
        f->crown_pct = c->crown_pct;
        pr_lightning_pick(st, f, a);
        f->remaining = 1; /* index of the next bolt */
        f->timer_ms = 0;  /* ms since the cast */
        pr_lightning_bolt(st, f, 0);
        if (f->tgt[1] == PR_NO_ID) {
            memset(f, 0, sizeof(*f));
            st->n_fx--;
        }
        break;
    }
    case PR_SPELL_ROLLING: {
        const PrProjDef *d = &PR_PROJS[c->spell_proj];
        PrEffect *f = pr_new_effect(st, PR_FX_ROLLING, team, card, x, y);
        if (!f) return;
        f->proj = c->spell_proj;
        f->radius = d->radius;     /* half-width */
        f->radius_y = d->radius_y; /* half-depth */
        f->damage = d->damage;
        f->crown_pct = d->crown_pct;
        f->remaining = d->range;
        f->dir_y = team == 0 ? -1 : 1; /* toward the enemy side */
        break;
    }
    default:
        break;
    }
}


/* Circle vs axis-aligned rectangle (closed), ledger spells.ROLLING_HIT_SHAPE. */
static inline int pr_circle_rect(int32_t cx, int32_t cy, int32_t r, int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
    int32_t px = PR_CLAMP(cx, x0, x1), py = PR_CLAMP(cy, y0, y1);
    return pr_within(cx, cy, px, py, r);
}

/* Advance one effect; returns 0 when it expires. */
static inline int pr_effect_step(PrState *st, PrEffect *f) {
    switch (f->type) {
    case PR_FX_AREA:
        if (f->timer_ms <= 0) return 0;
        pr_area_apply(st, &PR_AREAS[f->area], f->team, f->x, f->y, 0);
        f->timer_ms -= PR_TICK_MS;
        return f->timer_ms > 0;
    case PR_FX_WAVES: {
        if (f->timer_ms <= 0) {
            PrAreaHit h;
            memset(&h, 0, sizeof(h));
            h.damage = f->damage;
            h.crown_pct = f->crown_pct;
            h.buff = -1;
            h.hits_air = 1;
            h.hits_ground = 1;
            pr_area_hit(st, f->team, f->x, f->y, f->radius, &h); /* one hit per unit per wave */
            f->remaining--;
            if (f->remaining <= 0) return 0;
            f->timer_ms = PR_CARDS[(int)f->card].wave_interval_ms;
        }
        f->timer_ms -= PR_TICK_MS;
        return 1;
    }
    case PR_FX_ROLLING: {
        int32_t x0 = f->x - f->radius, x1 = f->x + f->radius;
        int32_t y0 = f->y - f->radius_y, y1 = f->y + f->radius_y;
        const PrProjDef *d = &PR_PROJS[f->proj];
        for (int i = 0; i < st->n_ent; i++) {
            PrEntity *v = &st->ent[i];
            if (v->hp <= 0 || v->team == f->team || v->flying || pr_is_hidden(v) || v->burrow) continue;
            if (!pr_circle_rect(v->x, v->y, v->radius, x0, y0, x1, y1)) continue;
            if (pr_bit_get(f->hit_bits, i)) continue; /* each entity at most once (SPEC §14.3) */
            pr_bit_set(f->hit_bits, i);
            pr_hit(v, f->damage, f->crown_pct);
            if (d->pushback > 0 && v->kind == PR_KIND_TROOP && (d->pushback_all || !PR_UNITS[v->unit].ignore_pushback))
                pr_knockback(v, 0, f->dir_y * d->pushback);
        }
        int end = f->remaining <= 0;
        if (!end) {
            int32_t step = PR_MIN(d->speed, f->remaining);
            f->y += f->dir_y * step;
            f->remaining -= step;
            /* rolled to or past the arena edge: stops there [IMPL-DEFINED]; the closed test is its own
             * rotation (y <= 0 for a roll up, y >= H for a roll down: SPEC §13.13) */
            if (f->y <= 0 || f->y >= PR_ARENA_H) {
                f->y = PR_CLAMP(f->y, 1, PR_ARENA_H - 1);
                end = 1;
            }
        }
        if (!end) return 1;
        /* Barbarian Barrel (SPEC §16.6.20): the rolling projectile's spawn_character at the end
         * point, committed in the next Spawn phase */
        for (int k = 0; d->spawn_unit >= 0 && k < d->spawn_count; k++) {
            int32_t x = f->x, y = f->y;
            if (PR_UNITS[d->spawn_unit].flying_height <= 0) pr_to_land(f->team, &x, &y);
            else pr_clamp_arena(&x, &y);
            pr_queue_spawn(st, d->spawn_unit, f->team, f->card, x, y, d->spawn_deploy_ms);
        }
        return 0;
    }
    case PR_FX_PULSE: {
        const PrAreaDef *a = &PR_AREAS[f->area];
        if (f->aux_ms <= 0) { /* buff refresh every area hit speed */
            pr_pulse_hit(st, f, 0);
            f->aux_ms += PR_MAX(PR_TICK_MS, a->hit_speed_ms);
        }
        if (f->timer_ms <= 0 && f->remaining > 0) { /* damage event every buff hit frequency */
            pr_pulse_hit(st, f, 1);
            f->remaining--;
            f->timer_ms += PR_MAX(PR_TICK_MS, PR_CARDS[(int)f->card].wave_interval_ms);
        }
        f->aux_ms -= PR_TICK_MS;
        f->timer_ms -= PR_TICK_MS;
        f->life_ms -= PR_TICK_MS;
        return f->life_ms > 0;
    }
    case PR_FX_LIGHTNING: {
        const PrAreaDef *a = &PR_AREAS[f->area];
        int32_t j = f->timer_ms / PR_TICK_MS;
        while (f->remaining < 3 && pr_bolt_tick(a, f->remaining) <= j) {
            pr_lightning_bolt(st, f, f->remaining);
            f->remaining++;
        }
        if (f->remaining >= 3 || f->tgt[f->remaining] == PR_NO_ID) return 0;
        f->timer_ms += PR_TICK_MS;
        return 1;
    }
    case PR_FX_BOMB: /* a death bomb: one area hit when its fuse runs out (SPEC §16.6.8) */
        if (f->timer_ms <= 0) {
            PrAreaHit h;
            memset(&h, 0, sizeof(h));
            h.damage = f->damage;
            h.crown_pct = f->crown_pct;
            h.buff = -1;
            h.hits_air = (uint8_t)(f->flags & 1);
            h.hits_ground = (uint8_t)((f->flags >> 1) & 1);
            pr_area_hit(st, f->team, f->x, f->y, f->radius, &h);
            return 0;
        }
        f->timer_ms -= PR_TICK_MS;
        return 1;
    default:
        return 0;
    }
}

static inline void pr_phase_effects(PrState *st) {
    int w = 0;
    int n = st->n_fx;
    for (int r = 0; r < n; r++) {
        PrEffect f = st->fx[r];
        if (pr_effect_step(st, &f)) st->fx[w++] = f;
    }
    /* effects created during this pass (none today) would sit after index n */
    for (int r = n; r < st->n_fx; r++) st->fx[w++] = st->fx[r];
    if (w < st->n_fx) memset(&st->fx[w], 0, sizeof(PrEffect) * (size_t)(st->n_fx - w));
    st->n_fx = w;
}

#endif /* PR_SPELL_H */
