/*
 * pr_combat.h -- attack cycle, damage buffer, projectiles, status buffs,
 * knockback, Resolve and Reap (SPEC §6.2, §6.6, §6.7, §7.1, §5 phases 5/8/9/10).
 *
 * THE DAMAGE BUFFER. No attack, projectile or effect subtracts hp directly: every
 * hit adds to the victim's `dmg_in` (with the crown-tower reduction already applied)
 * and Resolve applies all of it at once, so two units that hit each other on the
 * same tick both land regardless of processing order (RoyaleSim combat.rs).
 */
#ifndef PR_COMBAT_H
#define PR_COMBAT_H

#include "pr_target.h"

/* ------------------------------------------------------------------ damage */

/* SPEC §2: a hit with crown_tower_damage_percent P < 100 deals ceil(D * P / 100) to
 * crown towers only (ledger combat.CROWN_TOWER_DAMAGE_ROUNDING = ceil_kept_share). */
static inline int32_t pr_damage_vs(const PrEntity *v, int32_t dmg, int32_t crown_pct) {
    if (v->kind == PR_KIND_TOWER && crown_pct < 100)
        return (int32_t)pr_ceildiv((int64_t)dmg * crown_pct, 100);
    return dmg;
}

/* Buffer one hit on v. A Bandit in the moving phase of her dash, or on her arrival tick before
 * Resolve, takes no damage from any source; this is decided when the hit is DEALT (SPEC §18.3), so
 * a spell whose stun or knockback cancels the dash in the same call still deals nothing (the
 * damage is buffered before the buff / push is applied). */
static inline void pr_hit(PrEntity *v, int32_t dmg, int32_t crown_pct) {
    if (dmg <= 0 || v->hp <= 0 || pr_dash_immune(v)) return;
    v->dmg_in += pr_damage_vs(v, dmg, crown_pct);
}

/* ------------------------------------------------------------------ status buffs (SPEC §6.7) */

/* Apply buff `b` for `ms` during tick `tick`: it is active through tick
 * tick + ceil(ms/50) - 1 (SPEC §13.3); re-application keeps the later end (the max of old
 * and new remaining, status.SAME_BUFF_REAPPLY = refresh_max). A full-stop buff
 * (stun/freeze) resets the attack cycle and the charge and clears the target (retarget on
 * resume). */
static inline void pr_apply_buff(PrEntity *v, int b, int32_t ms, int32_t tick) {
    if (b < 0 || b >= PR_N_BUFFS || ms <= 0 || v->hp <= 0) return;
    int32_t until = tick + (int32_t)pr_ceildiv(ms, PR_TICK_MS);
    if (until > v->buff_until[b]) v->buff_until[b] = until;
    if (PR_BUFFS[b].speed_pct <= -100) {
        v->progress_ms = 0;
        v->charged = 0;
        v->charge_acc = 0;
        v->target_id = PR_NO_ID;
        v->attacking = 0;
        v->var_hits = 0;                                     /* SPEC §16.6.9: back to stage 1 */
        if (v->dash_state == PR_DASH_STAND || v->dash_state == PR_DASH_MOVE) v->dash_state = PR_DASH_NONE;
    }
}

/* ------------------------------------------------------------------ knockback (SPEC §7.2) */

/* Instantaneous displacement by (dx, dy) [IMPL-DEFINED: applied at once], clamped to
 * the arena and, for ground non-jumpers, to land. Resets the attack cycle and the
 * charge, keeps the target (ledger knockback.ATTACK_RESET = reset_attack_keep_target). */
static inline void pr_knockback(PrEntity *v, int32_t dx, int32_t dy) {
    if (v->kind != PR_KIND_TROOP || v->hp <= 0 || v->burrow) return;
    if (v->dash_state == PR_DASH_STAND || v->dash_state == PR_DASH_MOVE) v->dash_state = PR_DASH_NONE;
    v->x += dx;
    v->y += dy;
    pr_clamp_arena(&v->x, &v->y);
    if (!v->flying && !PR_UNITS[v->unit].jumps) pr_to_land(v->team, &v->x, &v->y);
    v->progress_ms = 0;
    v->charged = 0;
    v->charge_acc = 0;
    v->wp_cell = -1; /* re-plan */
}

/* Push `v` by `dist` directly away from (cx, cy). A victim exactly on the centre is
 * pushed toward its own side [IMPL-DEFINED, seat-symmetric]. */
static inline void pr_knockback_from(PrEntity *v, int32_t cx, int32_t cy, int32_t dist) {
    int64_t dx = (int64_t)v->x - cx, dy = (int64_t)v->y - cy;
    int64_t d = pr_isqrt64(dx * dx + dy * dy);
    if (d == 0) {
        pr_knockback(v, 0, v->team == 0 ? dist : -dist);
        return;
    }
    pr_knockback(v, (int32_t)(dx * dist / d), (int32_t)(dy * dist / d));
}

/* ------------------------------------------------------------------ area hits */

typedef struct PrAreaHit {
    int32_t damage;
    int32_t crown_pct;
    int16_t buff;          /* -1 = none */
    uint8_t hits_air;
    uint8_t hits_ground;
    uint8_t affects_hidden;/* buff also lands on hidden entities */
    uint8_t troops_only_push;
    int32_t buff_ms;
    int32_t push;          /* knockback distance away from the centre, 0 = none */
    uint8_t push_all;      /* ignore the victim's ignore_pushback */
    int32_t building_pct;  /* percent of the damage vs non-crown buildings (Earthquake 350); 0 = 100 */
} PrAreaHit;

static inline int pr_area_victim(const PrEntity *v, int team, const PrAreaHit *h) {
    if (v->hp <= 0 || v->team == team || v->burrow) return 0; /* a burrowing Miner is out of reach */
    return v->flying ? h->hits_air : h->hits_ground;
}

/* Every enemy of `team` whose circle reaches within `radius` of (cx, cy)
 * (dist <= radius + r_victim, ledger spells.AOE_HIT_TEST = edge_inclusive).
 * Returns the number of victims. */
static inline int pr_area_hit(PrState *st, int team, int32_t cx, int32_t cy, int32_t radius, const PrAreaHit *h) {
    int n = 0;
    for (int i = 0; i < st->n_ent; i++) {
        PrEntity *v = &st->ent[i];
        if (!pr_area_victim(v, team, h)) continue;
        if (!pr_within(cx, cy, v->x, v->y, (int64_t)radius + v->radius)) continue;
        n++;
        int32_t dmg = h->damage;
        if (h->building_pct > 0 && v->kind == PR_KIND_BUILDING) dmg = (int32_t)(((int64_t)dmg * h->building_pct) / 100);
        pr_hit(v, dmg, h->crown_pct);
        if (h->buff >= 0 && (h->affects_hidden || !pr_is_hidden(v))) pr_apply_buff(v, h->buff, h->buff_ms, st->tick);
        if (h->push > 0 && v->kind == PR_KIND_TROOP && !pr_is_hidden(v) &&
            (h->push_all || !PR_UNITS[v->unit].ignore_pushback))
            pr_knockback_from(v, cx, cy, h->push);
    }
    return n;
}

/* ------------------------------------------------------------------ projectiles (SPEC §7.1) */

static inline PrProjectile *pr_new_projectile(PrState *st, int proj, int team, int card, int32_t x, int32_t y) {
    if (st->n_proj >= PR_MAX_PROJECTILES) {
        pr_sat_add(&st->spawn_overflow, 1);
        return NULL;
    }
    const PrProjDef *d = &PR_PROJS[proj];
    PrProjectile *p = &st->proj[st->n_proj++];
    memset(p, 0, sizeof(*p));
    p->id = st->next_obj_id++;
    p->proj = (int16_t)proj;
    p->card = (int8_t)card;
    p->team = (uint8_t)team;
    p->x = x;
    p->y = y;
    p->tx = x;
    p->ty = y;
    p->target_id = PR_NO_ID;
    p->owner_id = PR_NO_ID;
    p->damage = d->damage;
    p->crown_pct = d->crown_pct;
    p->radius = d->radius;
    p->speed = d->speed;
    return p;
}

/* A unit/tower shot: starts projectile_start_radius from the attacker toward the
 * target and takes its first step on the next tick (ledger combat.PROJECTILE_LAUNCH =
 * start_radius_next_tick). */
static inline void pr_launch(PrState *st, const PrEntity *a, const PrEntity *t, int proj) {
    int32_t x = a->x, y = a->y;
    int32_t r = PR_UNITS[a->unit].proj_start_radius;
    if (r > 0) {
        int64_t dx = (int64_t)t->x - a->x, dy = (int64_t)t->y - a->y;
        int64_t d = pr_isqrt64(dx * dx + dy * dy);
        if (d > 0) {
            x += (int32_t)(dx * r / d);
            y += (int32_t)(dy * r / d);
        }
    }
    PrProjectile *p = pr_new_projectile(st, proj, a->team, a->card, x, y);
    if (!p) return;
    p->fresh = 1;
    /* SPEC §16.6.2: a non-homing shot (Princess, Mortar, Bomb Tower, Bomber, Wall Breakers) flies to
     * the target's position AT LAUNCH and applies its effect there (it can miss a moving target) */
    p->has_target = PR_PROJS[proj].homing ? 1 : 2;
    p->target_id = t->id;
    p->owner_id = a->id;
    p->tx = t->x;
    p->ty = t->y;
    p->damage = pr_lvl(a, p->damage);
}

/* Impact of projectile p at its aim point. */
static inline void pr_projectile_impact(PrState *st, PrProjectile *p) {
    const PrProjDef *d = &PR_PROJS[p->proj];
    if (p->radius > 0 && (d->aoe_air || d->aoe_ground) && p->damage > 0) {
        PrAreaHit h;
        memset(&h, 0, sizeof(h));
        h.damage = p->damage;
        h.crown_pct = p->crown_pct;
        h.buff = d->target_buff;
        h.buff_ms = d->buff_ms;
        h.hits_air = d->aoe_air;
        h.hits_ground = d->aoe_ground;
        h.push = d->pushback;
        h.push_all = d->pushback_all;
        pr_area_hit(st, p->team, p->tx, p->ty, p->radius, &h);
    } else if (p->has_target) {
        PrEntity *t = pr_get(st, p->target_id);
        /* an aimed (non-homing) single-target shot only hits a target still under its landing point */
        if (t && p->has_target == 2 && !pr_within(p->tx, p->ty, t->x, t->y, t->radius)) t = NULL;
        if (t && !t->burrow) {
            pr_hit(t, p->damage, p->crown_pct);
            if (d->target_buff >= 0 && !pr_is_hidden(t)) pr_apply_buff(t, d->target_buff, d->buff_ms, st->tick);
        }
    }
    if (d->spawn_unit >= 0 && d->spawn_count > 0) {
        /* Goblin Barrel: the SPEC §5.1 triangle at the landing point, committed in the
         * next tick's Spawn phase (SPEC §5 phase 3). */
        int form = -1;
        for (int f = 0; f < PR_N_FORMATIONS; f++)
            if (PR_FORMATIONS[f].count == d->spawn_count) form = f;
        for (int k = 0; k < d->spawn_count; k++) {
            int32_t ox = 0, oy = 0;
            if (form >= 0) {
                ox = PR_FORMATION_OFFSETS[PR_FORMATIONS[form].first + k][0];
                oy = PR_FORMATION_OFFSETS[PR_FORMATIONS[form].first + k][1];
            }
            if (p->team == 1) { ox = -ox; oy = -oy; }
            int32_t x = p->tx + ox, y = p->ty + oy;
            pr_to_land(p->team, &x, &y);
            pr_queue_spawn(st, d->spawn_unit, p->team, p->card, x, y, d->spawn_deploy_ms);
        }
    }
}

/* Advance one projectile; returns 0 when it is consumed. */
static inline int pr_projectile_step(PrState *st, PrProjectile *p) {
    if (p->fresh) {
        p->fresh = 0;
        return 1;
    }
    if (p->has_target == 1) {
        const PrEntity *t = pr_get_c(st, p->target_id);
        if (t) {
            p->tx = t->x; /* homing: aim at the target's current centre */
            p->ty = t->y;
        } else if (p->radius <= 0) {
            return 0;     /* single-target shot at a dead target vanishes */
        } else {
            p->has_target = 0; /* splash shot flies on to the last known point [IMPL-DEFINED] */
        }
    }
    int64_t d = pr_isqrt64(pr_dist2(p->x, p->y, p->tx, p->ty));
    if (d <= p->speed) {
        p->x = p->tx;
        p->y = p->ty;
        pr_projectile_impact(st, p);
        return 0;
    }
    pr_step_toward(&p->x, &p->y, p->tx, p->ty, p->speed);
    return 1;
}

static inline void pr_phase_projectiles(PrState *st) {
    int w = 0;
    int n = st->n_proj; /* impacts never create projectiles */
    for (int r = 0; r < n; r++) {
        PrProjectile p = st->proj[r];
        if (pr_projectile_step(st, &p)) st->proj[w++] = p;
    }
    if (w < st->n_proj) memset(&st->proj[w], 0, sizeof(PrProjectile) * (size_t)(st->n_proj - w));
    st->n_proj = w;
}

/* ------------------------------------------------------------------ attack (SPEC §6.2) */

/* One landed hit of attacker a on target t. */
/* Damage of a direct hit: the charge hit, else the variable-damage stage (SPEC §16.2, §16.6.9: the
 * stage clock starts at the first hit on the current target, each stage lasts
 * VariableDamageTime / HitSpeed hits), else the unit's damage; all level-scaled (Royal Chef). */
static inline int32_t pr_hit_damage(const PrEntity *a, const PrUnitDef *ad) {
    int32_t dmg = ad->damage;
    if (a->charged && ad->charge_damage > 0) dmg = ad->charge_damage;
    else if (ad->var_damage2 > 0) {
        if (a->var_hits >= ad->var_hits1 + ad->var_hits2) dmg = ad->var_damage3;
        else if (a->var_hits >= ad->var_hits1) dmg = ad->var_damage2;
    }
    return pr_lvl(a, dmg);
}

static inline void pr_fire(PrState *st, PrEntity *a, PrEntity *t) {
    const PrUnitDef *ad = pr_udef(a);
    if (ad->projectile >= 0) {
        pr_launch(st, a, t, ad->projectile);
    } else {
        int32_t dmg = pr_hit_damage(a, ad);
        if (ad->var_damage2 > 0 && a->var_hits < 10000) a->var_hits++;
        if (ad->area_radius > 0) {
            PrAreaHit h;
            memset(&h, 0, sizeof(h));
            h.damage = dmg;
            h.crown_pct = ad->crown_pct;
            h.buff = -1;
            h.hits_air = ad->attacks_air;
            h.hits_ground = ad->attacks_ground;
            int32_t cx = ad->self_aoe ? a->x : t->x, cy = ad->self_aoe ? a->y : t->y;
            pr_area_hit(st, a->team, cx, cy, ad->area_radius, &h);
        } else {
            pr_hit(t, dmg, ad->crown_pct); /* melee, or an instant hit (Tesla) */
        }
    }
    /* charge.RESET_ON_ATTACK */
    a->charged = 0;
    a->charge_acc = 0;
    /* ledger combat.KAMIKAZE_DEATH = at_fire: a self-hit for its whole hp + shield */
    if (ad->kamikaze) a->dmg_in += a->hp + a->shield;
}

/* The progress-credit attack cycle (ledger combat.ATTACK_CYCLE = progress_credit):
 *   load = max(0, load - 50) every tick
 *   attack iff target in range || progress % hit_speed > 50 (a hit already started)
 *   fresh cycle (progress == 0): progress = load_time - load, load = load_time
 *   progress += 50 * m (m = hit-speed multiplier), a hit lands when progress crosses
 *   a multiple of hit_speed and sets load = load_time; otherwise progress = 0.
 * [IMPL-DEFINED] switching to a DIFFERENT target that is not in range cancels the
 * swing (progress = 0); switching to a new target already in range keeps the cadence
 * (ledger combat.RETARGET_PROGRESS = keep_when_dead_or_in_reach).
 * Mid-swing reach loss (SPEC v0.2.1 §14.2): a direct striker (no projectile) whose target
 * is out of reach while a hit is started switches to the nearest valid enemy in reach and
 * keeps its progress (the hit lands on schedule on the new target), else the swing is
 * cancelled -- a melee hit never lands out of reach; a projectile attacker keeps swinging
 * at a target within reach + 500 (and fires at it), beyond that the swing is cancelled.
 * A charged unit's progress snaps to the next multiple of hit_speed
 * (charge.CHARGED_HIT_TIMING = first_attack_pass_no_windup). */
static inline void pr_phase_attack(PrState *st) {
    int n = st->n_ent; /* entities are never created in this phase */
    for (int i = 0; i < n; i++) {
        PrEntity *a = &st->ent[i];
        const PrUnitDef *ad = pr_udef(a);
        a->attacking = 0;
        if (!pr_attacker(ad) || a->hp <= 0) continue;
        a->load_ms = PR_MAX(0, a->load_ms - PR_TICK_MS);
        if (!pr_can_act(st, a)) continue; /* deploying / stunned / hidden / dormant: held */
        int32_t m = 100 + pr_buff_hit_pct(a);
        int32_t adv = PR_TICK_MS * PR_MAX(m, 0) / 100;
        int reloading = ad->seq_n > 0 && a->aux_ms > 0; /* Duchess reload: runs with or without a target */
        if (reloading) a->aux_ms = PR_MAX(0, a->aux_ms - adv);
        if (a->dash_state == PR_DASH_STAND || a->dash_state == PR_DASH_MOVE) { /* the dash replaces the swing */
            a->progress_ms = 0;
            continue;
        }
        PrEntity *t = pr_get(st, a->target_id);
        if (!t) {
            a->progress_ms = 0;
            a->prev_target_id = PR_NO_ID;
            a->var_hits = 0;
            continue;
        }
        int in_range = pr_in_attack_range(a, ad, t);
        if (t->id != a->prev_target_id) {
            if (!in_range) a->progress_ms = 0;
            a->var_hits = 0; /* SPEC §16.6.9: a new target restarts the variable damage */
        }
        int started = ad->seq_n > 0 ? a->progress_ms > PR_TICK_MS : pr_hit_started(a, ad);
        if (started && !in_range) { /* SPEC §14.2 */
            if (ad->projectile >= 0) {
                if (!pr_mid_swing_hold(a, ad, t)) {
                    a->progress_ms = 0;
                    started = 0;
                }
            } else {
                const PrEntity *b = pr_scan(st, a, ad, 1);
                if (b) {
                    t = pr_get(st, b->id);
                    if (t->id != a->target_id) a->var_hits = 0;
                    a->target_id = t->id;
                    in_range = 1;
                } else {
                    a->progress_ms = 0;
                    started = 0;
                }
            }
        }
        a->prev_target_id = t->id;
        int32_t hs = ad->hit_speed_ms, lt = ad->load_time_ms;
        if (!(in_range || started)) {
            a->progress_ms = 0;
            continue;
        }
        a->attacking = 1;
        if (adv <= 0) continue;
        if (ad->seq_n > 0) {
            /* Dagger Duchess (SPEC §16.3, [IMPL-DEFINED] docs/FIDELITY.md §12): shot k of the
             * AttackSequence lands hit_speed x multiplier[k] / 100 after the previous one (the first
             * of a fresh engagement after its own cycle); after the last shot of the sequence she
             * reloads for reload_ms (counted down above, paused while stunned) before the next. */
            if (reloading) { /* the next sequence starts on the tick after the reload ends */
                a->progress_ms = 0;
                continue;
            }
            int k = PR_CLAMP(a->seq_idx, 0, ad->seq_n - 1);
            int32_t hk = PR_MAX(PR_TICK_MS, hs * ad->seq_mult[k] / 100);
            a->progress_ms += adv;
            if (a->progress_ms >= hk) {
                a->progress_ms = 0;
                pr_fire(st, a, t);
                a->seq_idx = (uint8_t)(k + 1);
                if (a->seq_idx >= ad->seq_n) {
                    a->seq_idx = 0;
                    a->aux_ms = ad->reload_ms;
                }
            }
            continue;
        }
        int32_t before = a->progress_ms / hs;
        int snap = a->charged && ad->charge_damage > 0;
        if (snap) {
            a->progress_ms += hs - (a->progress_ms % hs);
        } else if (a->progress_ms == 0) {
            if (lt <= hs) {
                a->progress_ms = lt - a->load_ms;
                a->load_ms = lt;
            } else if (a->load_ms > hs) {
                continue; /* LoadTime > HitSpeed: stand until the reload is within a cycle */
            } else {
                a->load_ms = 0;
            }
            a->progress_ms += adv;
        } else {
            a->progress_ms += adv;
        }
        if (a->progress_ms / hs > before) {
            a->load_ms = lt;
            pr_fire(st, a, t);
        }
    }
}

/* ------------------------------------------------------------------ Resolve (phase 9) */

static inline void pr_phase_resolve(PrState *st) {
    for (int i = 0; i < st->n_ent; i++) {
        PrEntity *e = &st->ent[i];
        /* ledger hide.HIDDEN_IMMUNE_TO_DAMAGE (Tesla, burrowing Miner); a dashing Bandit's immunity
         * is applied when each hit is dealt (pr_hit, SPEC §18.3), not here */
        int32_t dmg = pr_is_hidden(e) ? 0 : e->dmg_in;
        int32_t drain = e->drain_in;                     /* the lifetime drain ignores hide */
        e->dmg_in = 0;
        e->drain_in = 0;
        if (e->dash_state == PR_DASH_HIT) e->dash_state = PR_DASH_NONE;
        if (e->hp <= 0) continue;
        if (dmg > 0 && e->shield > 0) { /* shields first, overflow carries into hp */
            int32_t a = PR_MIN(e->shield, dmg);
            e->shield -= a;
            dmg -= a;
        }
        int32_t total = dmg + drain;
        if (total <= 0) continue;
        int32_t before = e->hp;
        e->hp -= total;
        if (e->hp < 0) e->hp = 0;
        /* destroyed by damage, not by the lifetime drain alone (Elixir Collector, SPEC §16.6.7) */
        if (e->hp == 0) e->killed_by_dmg = (uint8_t)(dmg > 0 && before > drain);
        if (e->kind == PR_KIND_TOWER && e->tower_idx == 0 && dmg > 0) pr_trigger_king(st, e->team);
    }
}

/* ------------------------------------------------------------------ Reap (phase 10) */

static inline void pr_spawn_area_effect(PrState *st, int area, int team, int card, int32_t x, int32_t y);

static inline void pr_phase_reap(PrState *st) {
    int n = st->n_ent;
    int any = 0;
    for (int i = 0; i < n; i++) {
        PrEntity *e = &st->ent[i];
        if (e->hp > 0) continue;
        any = 1;
        const PrUnitDef *d = pr_udef(e);
        if (d->death_damage > 0 && d->death_radius > 0) {
            /* SPEC §6.6: enemy ground+air within radius + r_target, buffered into the
             * victims' dmg_in, which the NEXT Resolve applies. */
            PrAreaHit h;
            memset(&h, 0, sizeof(h));
            h.damage = pr_lvl(e, d->death_damage);
            h.crown_pct = d->crown_pct;
            h.buff = -1;
            h.hits_air = 1;
            h.hits_ground = 1;
            pr_area_hit(st, e->team, e->x, e->y, d->death_radius, &h);
        }
        if (d->death_area >= 0) pr_spawn_area_effect(st, d->death_area, e->team, e->card, e->x, e->y);
        /* SPEC §16.2 / §16.6.7: death spawns and death bombs fire on ANY death (lifetime expiry
         * included) and are committed in the next Spawn phase */
        if (d->dspawn_unit >= 0 && d->dspawn_count > 0)
            pr_queue_layout(st, d->dspawn_unit, e->team, e->card, e->x, e->y, d->dspawn_first, d->dspawn_count,
                            d->dspawn_count, d->dspawn_deploy_ms);
        if (d->bomb_damage > 0) pr_queue_pending(st, PR_PEND_BOMB, e->unit, e->team, e->card, e->x, e->y, d->bomb_fuse_ms);
        /* Elixir Collector ManaOnDeath: only when destroyed by damage (SPEC §16.6.15) */
        if (d->mana_on_death > 0 && e->killed_by_dmg) pr_add_elixir(st, e->team, d->mana_on_death);
        if (e->kind == PR_KIND_TOWER) pr_tower_destroyed(st, e);
    }
    if (any) pr_compact_entities(st);
}

#endif /* PR_COMBAT_H */
