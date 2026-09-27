/*
 * pr_target.h -- target acquisition and retention (SPEC §6.1).
 *
 *   valid target  alive enemy, not hidden, layer matches attacks_air/attacks_ground,
 *                 buildings/towers only for target_only_buildings
 *   sight test    dist <= sight + r_target (+2000 when the target is a crown tower,
 *                 ledger targeting.EXTRA_SIGHT_RANGE_TO_CROWN_TOWERS)
 *   attack test   dist <= range + r_attacker + r_target
 *                 (ledger targeting.ATTACK_RANGE_RULE = range_plus_both_radii)
 *   preference    minimum (dist - r_target), ties to the lowest entity id -- except that two
 *                 tied crown towers go own-frame left first (SPEC §18.1, seat symmetry)
 *
 * Troops acquire by sight and keep a target while it stays valid and within
 * sight + r_target + 500 (+2000 for crown towers); a strictly closer candidate
 * replaces it only when the unit is not mid-cycle (progress == 0). Buildings and
 * towers acquire the nearest valid enemy inside ATTACK range and keep it while it
 * stays in range. Deploying, stunned and hidden units hold no target; a dormant
 * King does not target.
 *
 * Mid-swing (SPEC v0.2.1 §14.2, ledger targeting.LOGIC_PRESERVE_TARGET_IF_HIT_STARTED =
 * projectile_attackers_only): a projectile attacker whose hit has started holds its target
 * while it stays within attack reach + 500, until the shot is fired (here, and in the
 * Attack phase); a direct striker whose target leaves reach mid-swing switches to the
 * nearest enemy in reach keeping its progress, else the swing is cancelled (Attack phase).
 */
#ifndef PR_TARGET_H
#define PR_TARGET_H

#include "pr_rules.h"

static inline int pr_can_target(const PrEntity *a, const PrUnitDef *ad, const PrEntity *t) {
    if (t->hp <= 0 || t->team == a->team) return 0;
    if (pr_is_hidden(t)) return 0;
    if (ad->only_buildings && !pr_is_structure(t)) return 0;
    return t->flying ? ad->attacks_air : ad->attacks_ground;
}

static inline int64_t pr_sight_reach(const PrUnitDef *ad, const PrEntity *t) {
    return (int64_t)ad->sight + t->radius + (t->kind == PR_KIND_TOWER ? PR_EXTRA_SIGHT_CROWN : 0);
}

static inline int64_t pr_attack_reach(const PrEntity *a, const PrUnitDef *ad, const PrEntity *t) {
    return (int64_t)ad->range + a->radius + t->radius;
}

static inline int pr_in_attack_range(const PrEntity *a, const PrUnitDef *ad, const PrEntity *t) {
    return pr_within(a->x, a->y, t->x, t->y, pr_attack_reach(a, ad, t));
}

/* SPEC §16.2 minimum range (Mortar): a target closer than minimum_range + r_a + r_t is not valid
 * for acquisition and is dropped. */
static inline int pr_inside_min_range(const PrEntity *a, const PrUnitDef *ad, const PrEntity *t) {
    if (ad->min_range <= 0) return 0;
    int64_t r = (int64_t)ad->min_range + a->radius + t->radius;
    return pr_dist2(a->x, a->y, t->x, t->y) < r * r;
}

/* A unit that can attack at all (data: a damage or a projectile). */
static inline int pr_attacker(const PrUnitDef *ad) { return ad->hit_speed_ms > 0 && !ad->no_attack; }

/* A hit is under way: the attack progress is past the start of a cycle (SPEC §6.2). */
static inline int pr_hit_started(const PrEntity *a, const PrUnitDef *ad) {
    return ad->hit_speed_ms > 0 && (a->progress_ms % ad->hit_speed_ms) > PR_TICK_MS;
}

/* SPEC §14.2: a projectile attacker holds a target that left its reach mid-swing while it
 * stays within reach + 500. */
static inline int pr_mid_swing_hold(const PrEntity *a, const PrUnitDef *ad, const PrEntity *t) {
    return ad->projectile >= 0 && pr_hit_started(a, ad) &&
           pr_within(a->x, a->y, t->x, t->y, pr_attack_reach(a, ad, t) + PR_KEEP_TARGET_EXTRA);
}

/* Preference key: dist - r_target (edge distance). */
static inline int64_t pr_target_key(const PrEntity *a, const PrEntity *t) {
    return (int64_t)pr_dist(a->x, a->y, t->x, t->y) - t->radius;
}

/* Can this entity target anything this tick? */
static inline int pr_can_act(const PrState *st, const PrEntity *e) {
    if (e->hp <= 0 || pr_is_deploying(e) || pr_is_stunned(e)) return 0;
    if (e->hide_state != PR_HIDE_UP) return 0;
    if (e->kind == PR_KIND_TOWER && e->tower_idx == 0 && !st->king_active[e->team]) return 0;
    return 1;
}

/* Exact-tie order of two candidates for attacker a (SPEC §6.1, §18.1): two crown towers go by the
 * attacker's OWN-frame x, own-left first (tower ids follow the engine frame, so an id tie-break
 * between them would favour engine-left for both seats); every other tie by the lowest id (tower ids
 * are below every other id for both seats, and the remaining ids mirror). */
static inline int pr_tie_before(const PrEntity *a, const PrEntity *t, const PrEntity *u) {
    if (t->kind == PR_KIND_TOWER && u->kind == PR_KIND_TOWER) {
        int32_t xt = pr_own_x(a->team, t->x), xu = pr_own_x(a->team, u->x);
        if (xt != xu) return xt < xu;
    }
    return t->id < u->id;
}

/* Offer candidate t to a scan: valid, inside the sight test (troops) or the attack test
 * (structures), minimum (dist - r_target), ties by pr_tie_before. */
static inline void pr_scan_offer(const PrEntity *a, const PrUnitDef *ad, int by_attack_range, const PrEntity *t,
                                 const PrEntity **best, int64_t *bk) {
    if (!pr_can_target(a, ad, t)) return;
    int64_t reach = by_attack_range ? pr_attack_reach(a, ad, t) : pr_sight_reach(ad, t);
    int64_t dx = (int64_t)t->x - a->x, dy = (int64_t)t->y - a->y;
    if (dx > reach || -dx > reach || dy > reach || -dy > reach) return; /* cheap reject */
    int64_t d2 = dx * dx + dy * dy;
    if (d2 > reach * reach) return;
    if (ad->min_range > 0) { /* SPEC §16.2: inside the minimum range is not a valid target */
        int64_t mr = (int64_t)ad->min_range + a->radius + t->radius;
        if (d2 < mr * mr) return;
    }
    if (*best) { /* key = isqrt(d2) - r_t can only reach bk if isqrt(d2) <= bk + r_t: skip the sqrt */
        int64_t lim = *bk + t->radius + 1;
        if (lim <= 0 || d2 >= lim * lim) return;
    }
    int64_t k = pr_target_key(a, t);
    if (!*best || k < *bk || (k == *bk && pr_tie_before(a, t, *best))) {
        *best = t;
        *bk = k;
    }
}

/* Best candidate by the sight test (troops) or the attack test (structures). */
static inline const PrEntity *pr_scan(const PrState *st, const PrEntity *a, const PrUnitDef *ad, int by_attack_range) {
    const PrEntity *best = NULL;
    int64_t bk = 0;
    for (int i = 0; i < st->n_ent; i++) pr_scan_offer(a, ad, by_attack_range, &st->ent[i], &best, &bk);
    return best;
}

/* Per-team bucket grids over the living non-tower entities (2x2-tile blocks, counting-
 * sorted into contiguous lists), rebuilt at the start of the Target phase so that a scan only
 * looks at enemies in the blocks its reach can touch; crown towers (whose sight test has the
 * +2000 extra) are always offered. Candidate order does not matter: the scan's tie-break is
 * explicit, so the result equals the full scan's. */
#define PR_GRID 2000
#define PR_GRID_COLS (PR_ARENA_W / PR_GRID)
#define PR_GRID_ROWS (PR_ARENA_H / PR_GRID)
#define PR_GRID_N (PR_GRID_COLS * PR_GRID_ROWS)

typedef struct PrGrid {
    int16_t start[2][PR_GRID_N + 1]; /* team t, block b: list[t][start[t][b] .. start[t][b+1]) */
    int16_t list[2][PR_MAX_ENTITIES];
    int16_t tower[8];
    int n_tower;
    int32_t rmax; /* largest radius in the grids */
} PrGrid;

static inline int pr_grid_col(int64_t x) { return x < 0 ? 0 : (int)PR_MIN(x / PR_GRID, PR_GRID_COLS - 1); }
static inline int pr_grid_row(int64_t y) { return y < 0 ? 0 : (int)PR_MIN(y / PR_GRID, PR_GRID_ROWS - 1); }

static inline void pr_grid_build(const PrState *st, PrGrid *g) {
    int16_t blk[PR_MAX_ENTITIES];
    memset(g->start, 0, sizeof(g->start));
    g->n_tower = 0;
    g->rmax = 0;
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        blk[i] = -1;
        if (e->hp <= 0) continue;
        if (e->kind == PR_KIND_TOWER) {
            if (g->n_tower < 8) g->tower[g->n_tower++] = (int16_t)i;
            continue;
        }
        blk[i] = (int16_t)(pr_grid_row(e->y) * PR_GRID_COLS + pr_grid_col(e->x));
        g->start[e->team & 1][blk[i] + 1]++;
        if (e->radius > g->rmax) g->rmax = e->radius;
    }
    for (int t = 0; t < 2; t++)
        for (int b = 0; b < PR_GRID_N; b++) g->start[t][b + 1] = (int16_t)(g->start[t][b + 1] + g->start[t][b]);
    int16_t fill[2][PR_GRID_N];
    memcpy(fill[0], g->start[0], sizeof(fill[0]));
    memcpy(fill[1], g->start[1], sizeof(fill[1]));
    for (int i = 0; i < st->n_ent; i++) {
        if (blk[i] < 0) continue;
        int t = st->ent[i].team & 1;
        g->list[t][fill[t][blk[i]]++] = (int16_t)i;
    }
}

/* `seed` (may be NULL) starts the search as the incumbent (its key bounds the others, so
 * most candidates are rejected without a square root); the result is then the best of the
 * candidates and the seed. */
static inline const PrEntity *pr_scan_grid(const PrState *st, const PrGrid *g, const PrEntity *a, const PrUnitDef *ad,
                                           int by_attack_range, const PrEntity *seed) {
    const PrEntity *best = seed;
    int64_t bk = seed ? pr_target_key(a, seed) : 0;
    for (int k = 0; k < g->n_tower; k++) pr_scan_offer(a, ad, by_attack_range, &st->ent[g->tower[k]], &best, &bk);
    int enemy = 1 - (a->team & 1);
    int64_t r = (by_attack_range ? (int64_t)ad->range + a->radius : (int64_t)ad->sight) + g->rmax;
    int bx0 = pr_grid_col((int64_t)a->x - r), bx1 = pr_grid_col((int64_t)a->x + r);
    int by0 = pr_grid_row((int64_t)a->y - r), by1 = pr_grid_row((int64_t)a->y + r);
    const int16_t *start = g->start[enemy], *list = g->list[enemy];
    for (int by = by0; by <= by1; by++) {
        for (int bx = bx0; bx <= bx1; bx++) {
            int b = by * PR_GRID_COLS + bx;
            for (int k = start[b]; k < start[b + 1]; k++) pr_scan_offer(a, ad, by_attack_range, &st->ent[list[k]], &best, &bk);
        }
    }
    return best;
}

static inline void pr_phase_target(PrState *st) {
    PrGrid grid;
    pr_grid_build(st, &grid);
    for (int i = 0; i < st->n_ent; i++) {
        PrEntity *a = &st->ent[i];
        const PrUnitDef *ad = pr_udef(a);
        if (!pr_attacker(ad)) continue;
        if (!pr_can_act(st, a)) {
            a->target_id = PR_NO_ID;
            continue;
        }
        const PrEntity *cur = pr_get_c(st, a->target_id);
        if (cur && (!pr_can_target(a, ad, cur) || pr_inside_min_range(a, ad, cur))) cur = NULL;
        if (a->dash_state == PR_DASH_STAND || a->dash_state == PR_DASH_MOVE) {
            /* a dash keeps its target; it is cancelled when the target is lost (pr_phase_move) */
            if (!cur) a->target_id = PR_NO_ID;
            continue;
        }
        if (cur && pr_mid_swing_hold(a, ad, cur)) continue; /* SPEC §14.2 */
        if (pr_is_structure(a)) {
            if (cur && pr_in_attack_range(a, ad, cur)) continue; /* keep while in range */
            const PrEntity *b = pr_scan_grid(st, &grid, a, ad, 1, NULL);
            a->target_id = b ? b->id : PR_NO_ID;
            continue;
        }
        if (cur && !pr_within(a->x, a->y, cur->x, cur->y, pr_sight_reach(ad, cur) + PR_KEEP_TARGET_EXTRA))
            cur = NULL;
        if (cur && a->progress_ms != 0) continue; /* mid-cycle: kept (no closer-target switch) */
        /* seeded with the current target: b is the best of {cur} and the candidates, and a
         * strictly closer candidate replaces cur (progress == 0 here) */
        const PrEntity *b = pr_scan_grid(st, &grid, a, ad, 0, cur);
        if (cur) {
            if (b != cur && pr_target_key(a, b) < pr_target_key(a, cur))
                a->target_id = b->id;
            else
                a->target_id = cur->id;
        } else {
            a->target_id = b ? b->id : PR_NO_ID;
        }
    }
}

/* The default route (SPEC §6.1): the enemy crown tower of the unit's lane -- the
 * lane's Princess if alive, else the enemy King. Lane is decided in the unit's OWN
 * frame (SPEC v0.2 §13.6): own x < 9000 = own-left, else own-right, and exactly 9000
 * goes own-left (ledger targeting.CENTRE_LANE_FRAME = own_frame_tie_left). */
static inline const PrEntity *pr_default_tower(const PrState *st, const PrEntity *a) {
    int enemy = 1 - a->team;
    int own_left = pr_own_x(a->team, a->x) <= PR_ARENA_W / 2;
    /* own-left for team 0 is engine-left (idx 1); for team 1 it is engine-right (idx 2) */
    int idx = (a->team == 0) ? (own_left ? 1 : 2) : (own_left ? 2 : 1);
    const PrEntity *t = pr_get_c(st, st->tower_id[enemy][idx]);
    if (!t) t = pr_get_c(st, st->tower_id[enemy][0]);
    return t;
}

#endif /* PR_TARGET_H */
