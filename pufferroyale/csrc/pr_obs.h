/*
 * pr_obs.h -- the observation encoder (SPEC §9). Floats are allowed here (SPEC §1:
 * observation encoding); the encoder only READS the game state.
 *
 * Everything is in the viewing team's OWN frame (team 1 sees the 180-degree rotation):
 * own-frame tile (tx, ty) covers own x in [1000 tx, 1000 tx + 1000), own y likewise.
 *
 *   [SPATIAL  C x 32 x 18]  channel-major, values in [0, 1]
 *     per side (own = 0..9, enemy = 10..19):
 *       0 ground troops     count / 4 (clipped)     5 buildings-only targeters  count / 4
 *       1 air troops        count / 4               6 ranged (range > 1900)     count / 4
 *       2 building          1 on footprint tiles     7 splash                    count / 4
 *       3 crown tower       hp fraction on footprint 8 deploying                 count / 4
 *       4 hp sum            sum hp / 4000 (clipped)  9 debuffed (stun or slow)   count / 4
 *     shared: 20 water, 21 own troop-legal tiles, 22 own building-legal tiles (the own
 *     deck's first building, else Cannon), 23 own active spell areas, 24 enemy ones.
 *     Troops count on the tile containing their centre; buildings and towers cover
 *     every tile their footprint overlaps.
 *   [ENTITIES 64 x 11]  (SPEC §16.4, v0.3) 32 own + 32 enemy non-tower entities, each group in
 *     ascending entity id (more than 32 -> the 32 lowest ids), zero-padded: card id + 1 [0]
 *     (spawned / death-spawned units map to the card that created them: Goblins -> Goblin Barrel,
 *     Witch Skeletons -> Witch, ...), x/18000 [1], y/32000 [2], hp/max_hp [3], min(1, hp/2000)
 *     [4], flying [5], deploying [6] (also 1 for a burrowing Miner), stunned-or-frozen [7],
 *     slowed [8], is_building [9], target_only_buildings [10]
 *   [SCALARS 306]  the SPEC §9.3 list in its order with the §16.4 encodings (card ids as
 *     card_id + 1, 0 = empty; 128-wide multi-hots indexed by card_id; opponent spent / 200),
 *     the own / enemy tower-troop one-hots appended (§16.6.16) and the own deck as a set
 *     (§19.3: 8 ids card_id + 1, ascending) (PR_OBS_FIELDS / SCALAR_INDEX)
 *   [MASK 2305]  exactly pr_legal_mask (the engine's legality function)
 *
 * Hidden information (SPEC §9): the opponent's current elixir, hand, queue and unrevealed
 * deck never enter; only what the opponent has *played* (cards seen, last four, elixir
 * spent) and the deductions the SPEC allows.
 */
#ifndef PR_OBS_ENCODER_H
#define PR_OBS_ENCODER_H

#include "pr_engine.h"

enum {
    PR_CH_GROUND = 0, PR_CH_AIR, PR_CH_BUILDING, PR_CH_TOWER, PR_CH_HP,
    PR_CH_BT, PR_CH_RANGED, PR_CH_SPLASH, PR_CH_DEPLOYING, PR_CH_DEBUFFED,
    PR_CH_PER_SIDE,
    PR_CH_WATER = 2 * PR_CH_PER_SIDE,
    PR_CH_TROOP_LEGAL, PR_CH_BUILDING_LEGAL, PR_CH_OWN_SPELL, PR_CH_ENEMY_SPELL,
    PR_OBS_C
};

#define PR_OBS_H 32
#define PR_OBS_W 18
#define PR_OBS_PLANE (PR_OBS_H * PR_OBS_W)
#define PR_OBS_SPATIAL_SIZE (PR_OBS_C * PR_OBS_PLANE)
#define PR_OBS_ENT_N 64
#define PR_OBS_ENT_HALF 32
#define PR_OBS_ENT_F 11
#define PR_OBS_ENTITY_SIZE (PR_OBS_ENT_N * PR_OBS_ENT_F)

/* scalar section (SPEC §16.4: card identities are integer ids card_id + 1, 0 = empty) */
#define PR_SC_ELIXIR 0
#define PR_SC_HAND 1
#define PR_SC_HAND_COST (PR_SC_HAND + 4)
#define PR_SC_NEXT (PR_SC_HAND_COST + 4)
#define PR_SC_AFFORD (PR_SC_NEXT + 1)
#define PR_SC_TICK (PR_SC_AFFORD + 4)
#define PR_SC_OVERTIME (PR_SC_TICK + 1)
#define PR_SC_RATE (PR_SC_OVERTIME + 1)
#define PR_SC_LOCKOUT (PR_SC_RATE + 1)
#define PR_SC_OWN_TOWERS (PR_SC_LOCKOUT + 1)
#define PR_SC_ENEMY_TOWERS (PR_SC_OWN_TOWERS + 3)
#define PR_SC_KING_ACTIVE (PR_SC_ENEMY_TOWERS + 3)
#define PR_SC_CROWNS (PR_SC_KING_ACTIVE + 2)
#define PR_SC_OPP_SEEN (PR_SC_CROWNS + 2)
#define PR_SC_OPP_SPENT (PR_SC_OPP_SEEN + PR_CARD_SLOTS)
#define PR_SC_OPP_LAST4 (PR_SC_OPP_SPENT + 1)
#define PR_SC_OPP_HAND (PR_SC_OPP_LAST4 + 4)
#define PR_SC_OPP_ELIXIR_UB (PR_SC_OPP_HAND + PR_CARD_SLOTS)
#define PR_SC_OWN_TT (PR_SC_OPP_ELIXIR_UB + 1)
#define PR_SC_ENEMY_TT (PR_SC_OWN_TT + PR_N_TOWER_TROOPS)
#define PR_SC_OWN_DECK (PR_SC_ENEMY_TT + PR_N_TOWER_TROOPS) /* SPEC §19.3 */
#define PR_OBS_SCALAR_SIZE (PR_SC_OWN_DECK + 8)

#define PR_OBS_SPATIAL_OFFSET 0
#define PR_OBS_ENTITY_OFFSET (PR_OBS_SPATIAL_OFFSET + PR_OBS_SPATIAL_SIZE)
#define PR_OBS_SCALAR_OFFSET (PR_OBS_ENTITY_OFFSET + PR_OBS_ENTITY_SIZE)
#define PR_OBS_MASK_OFFSET (PR_OBS_SCALAR_OFFSET + PR_OBS_SCALAR_SIZE)
#define PR_OBS_MASK_SIZE PR_N_ACTIONS
#define PR_OBS_SIZE (PR_OBS_MASK_OFFSET + PR_OBS_MASK_SIZE)

typedef struct PrObsField { const char *name; int offset, len; } PrObsField;

/* Name -> (offset, length) inside the scalar section (exported to Python). */
static const PrObsField PR_OBS_FIELDS[] = {
    {"elixir", PR_SC_ELIXIR, 1},
    {"hand", PR_SC_HAND, 4},
    {"hand_cost", PR_SC_HAND_COST, 4},
    {"next_card", PR_SC_NEXT, 1},
    {"affordable", PR_SC_AFFORD, 4},
    {"tick", PR_SC_TICK, 1},
    {"overtime", PR_SC_OVERTIME, 1},
    {"elixir_rate", PR_SC_RATE, 1},
    {"lockout", PR_SC_LOCKOUT, 1},
    {"own_towers", PR_SC_OWN_TOWERS, 3},
    {"enemy_towers", PR_SC_ENEMY_TOWERS, 3},
    {"king_active", PR_SC_KING_ACTIVE, 2},
    {"crowns", PR_SC_CROWNS, 2},
    {"opp_seen", PR_SC_OPP_SEEN, PR_CARD_SLOTS},
    {"opp_spent", PR_SC_OPP_SPENT, 1},
    {"opp_last4", PR_SC_OPP_LAST4, 4},
    {"opp_deduced_hand", PR_SC_OPP_HAND, PR_CARD_SLOTS},
    {"opp_elixir_ub", PR_SC_OPP_ELIXIR_UB, 1},
    {"own_tower_troop", PR_SC_OWN_TT, PR_N_TOWER_TROOPS},
    {"enemy_tower_troop", PR_SC_ENEMY_TT, PR_N_TOWER_TROOPS},
    {"own_deck", PR_SC_OWN_DECK, 8},
};
#define PR_OBS_N_FIELDS ((int)(sizeof(PR_OBS_FIELDS) / sizeof(PR_OBS_FIELDS[0])))

static const char *const PR_OBS_CHANNELS[PR_OBS_C] = {
    "own_ground", "own_air", "own_building", "own_tower", "own_hp", "own_building_targeters",
    "own_ranged", "own_splash", "own_deploying", "own_debuffed",
    "enemy_ground", "enemy_air", "enemy_building", "enemy_tower", "enemy_hp",
    "enemy_building_targeters", "enemy_ranged", "enemy_splash", "enemy_deploying", "enemy_debuffed",
    "water", "own_troop_legal", "own_building_legal", "own_spells", "enemy_spells",
};

/* Total elixir units regenerated by the start of tick `t` (before tick t's regen). */
static inline int64_t pr_obs_income(int32_t t) {
    int64_t a = PR_MIN(t, PR_TICK_2X), b = PR_MAX(0, PR_MIN(t, PR_TICK_3X) - PR_TICK_2X), c = PR_MAX(0, t - PR_TICK_3X);
    return a * PR_ELIXIR_RATE_1X + b * 2 * PR_ELIXIR_RATE_1X + c * 3 * PR_ELIXIR_RATE_1X;
}

/* Own-frame tile index of an engine point for `team`. */
static inline int pr_obs_tile(int team, int32_t x, int32_t y) {
    int tx = PR_CLAMP(pr_own_x(team, x) / 1000, 0, PR_OBS_W - 1);
    int ty = PR_CLAMP(pr_own_y(team, y) / 1000, 0, PR_OBS_H - 1);
    return ty * PR_OBS_W + tx;
}

static inline void pr_obs_add(float *plane, int t, float v) { plane[t] += v; }

/* Set every tile of `plane` overlapped (positive area) by the own-frame rect to v. */
static inline void pr_obs_fill_rect(float *plane, int32_t x0, int32_t y0, int32_t x1, int32_t y1, float v) {
    int tx0 = PR_MAX(0, (int)pr_floordiv(x0, 1000)), tx1 = PR_MIN(PR_OBS_W - 1, (int)pr_floordiv(x1 - 1, 1000));
    int ty0 = PR_MAX(0, (int)pr_floordiv(y0, 1000)), ty1 = PR_MIN(PR_OBS_H - 1, (int)pr_floordiv(y1 - 1, 1000));
    for (int ty = ty0; ty <= ty1; ty++)
        for (int tx = tx0; tx <= tx1; tx++)
            if (plane[ty * PR_OBS_W + tx] < v) plane[ty * PR_OBS_W + tx] = v;
}

/* Mark tiles whose centre lies within `r` of own point (cx, cy). */
static inline void pr_obs_fill_disc(float *plane, int32_t cx, int32_t cy, int32_t r) {
    int tx0 = PR_MAX(0, (int)pr_floordiv(cx - r, 1000)), tx1 = PR_MIN(PR_OBS_W - 1, (int)pr_floordiv(cx + r, 1000));
    int ty0 = PR_MAX(0, (int)pr_floordiv(cy - r, 1000)), ty1 = PR_MIN(PR_OBS_H - 1, (int)pr_floordiv(cy + r, 1000));
    for (int ty = ty0; ty <= ty1; ty++)
        for (int tx = tx0; tx <= tx1; tx++)
            if (pr_within(cx, cy, tx * 1000 + 500, ty * 1000 + 500, r)) plane[ty * PR_OBS_W + tx] = 1.0f;
}

static inline float pr_obs_tower_frac(const PrState *st, int team, int idx) {
    const PrEntity *t = pr_get_c(st, st->tower_id[team][idx]);
    return t ? (float)t->hp / (float)t->max_hp : 0.0f;
}

/* SPEC §16.4: a card identity as the integer id card_id + 1 (0 = empty / unknown). */
static inline float pr_obs_card_id(int card) { return (card >= 0 && card < PR_N_CARDS) ? (float)(card + 1) : 0.0f; }

/* Write the full observation of `team` into out[PR_OBS_SIZE]. */
static inline void pr_obs_write(const PrState *st, int team, float *out) {
    memset(out, 0, sizeof(float) * PR_OBS_SIZE);
    int opp = 1 - team;
    float *sp = out + PR_OBS_SPATIAL_OFFSET;
#define PLANE(ch) (sp + (ch) * PR_OBS_PLANE)

    /* ---- entities: spatial counts and the entity list ---- */
    float *ent = out + PR_OBS_ENTITY_OFFSET;
    int n_own = 0, n_enemy = 0;
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        if (e->hp <= 0) continue;
        const PrUnitDef *d = pr_udef(e);
        int side = e->team == team ? 0 : PR_CH_PER_SIDE;
        int32_t ox = pr_own_x(team, e->x), oy = pr_own_y(team, e->y);
        if (e->kind == PR_KIND_TOWER) {
            int32_t h = d->footprint_tiles * 500;
            pr_obs_fill_rect(PLANE(side + PR_CH_TOWER), ox - h, oy - h, ox + h, oy + h, (float)e->hp / (float)e->max_hp);
            continue;
        }
        int t = pr_obs_tile(team, e->x, e->y);
        if (e->kind == PR_KIND_BUILDING) {
            int32_t h = d->footprint_tiles * 500;
            pr_obs_fill_rect(PLANE(side + PR_CH_BUILDING), ox - h, oy - h, ox + h, oy + h, 1.0f);
        } else {
            pr_obs_add(PLANE(side + (e->flying ? PR_CH_AIR : PR_CH_GROUND)), t, 0.25f);
        }
        pr_obs_add(PLANE(side + PR_CH_HP), t, (float)e->hp / 4000.0f);
        if (d->only_buildings) pr_obs_add(PLANE(side + PR_CH_BT), t, 0.25f);
        if (d->range > PR_MELEE_RANGE) pr_obs_add(PLANE(side + PR_CH_RANGED), t, 0.25f);
        if (d->area_radius > 0 || (d->projectile >= 0 && PR_PROJS[d->projectile].radius > 0))
            pr_obs_add(PLANE(side + PR_CH_SPLASH), t, 0.25f);
        int inactive = pr_is_deploying(e) || e->burrow;
        if (inactive) pr_obs_add(PLANE(side + PR_CH_DEPLOYING), t, 0.25f);
        if (pr_is_stunned(e) || pr_is_slowed(e)) pr_obs_add(PLANE(side + PR_CH_DEBUFFED), t, 0.25f);
        /* entity list: 32 per side in ascending id order (SPEC §16.6.22) */
        int slot;
        if (e->team == team) {
            if (n_own >= PR_OBS_ENT_HALF) continue;
            slot = n_own++;
        } else {
            if (n_enemy >= PR_OBS_ENT_HALF) continue;
            slot = PR_OBS_ENT_HALF + n_enemy++;
        }
        float *f = ent + slot * PR_OBS_ENT_F;
        f[0] = pr_obs_card_id(e->card);
        f[1] = (float)ox / (float)PR_ARENA_W;
        f[2] = (float)oy / (float)PR_ARENA_H;
        f[3] = (float)e->hp / (float)e->max_hp;
        f[4] = PR_MIN(1.0f, (float)e->hp / 2000.0f);
        f[5] = (float)e->flying;
        f[6] = (float)inactive;
        f[7] = (float)pr_is_stunned(e);
        f[8] = (float)pr_is_slowed(e);
        f[9] = (float)(e->kind == PR_KIND_BUILDING);
        f[10] = (float)d->only_buildings;
    }
    /* clip the count / hp planes to [0, 1] */
    for (int s = 0; s < 2; s++) {
        static const int clip_ch[] = {PR_CH_GROUND, PR_CH_AIR, PR_CH_HP, PR_CH_BT, PR_CH_RANGED,
                                      PR_CH_SPLASH, PR_CH_DEPLOYING, PR_CH_DEBUFFED};
        for (int k = 0; k < (int)(sizeof(clip_ch) / sizeof(clip_ch[0])); k++) {
            float *p = PLANE(s * PR_CH_PER_SIDE + clip_ch[k]);
            for (int t = 0; t < PR_OBS_PLANE; t++)
                if (p[t] > 1.0f) p[t] = 1.0f;
        }
    }

    /* ---- static / legality planes ---- */
    PrLegalCtx ctx;
    pr_legal_ctx(st, team, &ctx);
    int bunit = PR_UNIT_CANNON;
    for (int k = 0; k < 8; k++) {
        int c = st->deck[team][k];
        if (c >= 0 && c < PR_N_CARDS && PR_CARDS[c].kind == PR_CARD_KIND_BUILDING) { bunit = PR_CARDS[c].unit; break; }
    }
    for (int ty = 0; ty < PR_OBS_H; ty++) {
        for (int tx = 0; tx < PR_OBS_W; tx++) {
            int t = ty * PR_OBS_W + tx, etx, ety;
            pr_tile_to_engine(team, tx, ty, &etx, &ety);
            PLANE(PR_CH_WATER)[t] = (float)((PR_TILE_FLAGS[ety][etx] & PR_TF_WATER) != 0);
            PLANE(PR_CH_TROOP_LEGAL)[t] = (float)pr_troop_tile_legal_ctx(&ctx, team, tx, ty);
            PLANE(PR_CH_BUILDING_LEGAL)[t] = (float)pr_building_tile_legal_ctx(&ctx, team, bunit, tx, ty);
        }
    }

    /* ---- active spell areas ---- */
    for (int i = 0; i < st->n_proj; i++) { /* Fireball / Goblin Barrel in flight: the landing area */
        const PrProjectile *p = &st->proj[i];
        if (p->has_target || p->card < 0 || PR_CARDS[(int)p->card].kind != PR_CARD_KIND_SPELL) continue;
        float *pl = PLANE(p->team == team ? PR_CH_OWN_SPELL : PR_CH_ENEMY_SPELL);
        pr_obs_fill_disc(pl, pr_own_x(team, p->tx), pr_own_y(team, p->ty), PR_MAX(p->radius, 1000));
    }
    for (int i = 0; i < st->n_fx; i++) {
        const PrEffect *f = &st->fx[i];
        float *pl = PLANE(f->team == team ? PR_CH_OWN_SPELL : PR_CH_ENEMY_SPELL);
        int32_t cx = pr_own_x(team, f->x), cy = pr_own_y(team, f->y);
        if (f->type == PR_FX_ROLLING) { /* the rest of the roll: current hitbox to the end */
            int32_t dir = (f->team == team) ? -1 : 1; /* in the viewer's frame */
            int32_t y_end = cy + dir * f->remaining;
            int32_t y0 = PR_MIN(cy, y_end) - f->radius_y, y1 = PR_MAX(cy, y_end) + f->radius_y;
            pr_obs_fill_rect(pl, cx - f->radius, y0, cx + f->radius, y1, 1.0f);
        } else {
            pr_obs_fill_disc(pl, cx, cy, f->radius);
        }
    }
#undef PLANE

    /* ---- scalars ---- */
    float *sc = out + PR_OBS_SCALAR_OFFSET;
    sc[PR_SC_ELIXIR] = (float)st->elixir[team] / (float)PR_ELIXIR_MAX;
    for (int s = 0; s < 4; s++) {
        int c = st->hand[team][s];
        sc[PR_SC_HAND + s] = pr_obs_card_id(c);
        if (c >= 0 && c < PR_N_CARDS) {
            sc[PR_SC_HAND_COST + s] = (float)PR_CARDS[c].elixir / 10.0f;
            sc[PR_SC_AFFORD + s] = (float)(st->elixir[team] >= PR_CARDS[c].elixir * PR_ELIXIR_UNIT);
        }
    }
    sc[PR_SC_NEXT] = pr_obs_card_id(st->queue[team][0]);
    sc[PR_SC_TICK] = (float)st->tick / (float)PR_TICKS_MAX;
    sc[PR_SC_OVERTIME] = (float)(st->tick >= PR_TICKS_REGULATION);
    sc[PR_SC_RATE] = (float)pr_elixir_rate(st->tick) / (float)(3 * PR_ELIXIR_RATE_1X);
    sc[PR_SC_LOCKOUT] = (float)(st->tick < st->lockout_ticks);
    /* own-frame left/right -> engine index: team 0 left = 1; team 1 left = 2 */
    int il = team == 0 ? 1 : 2, ir = team == 0 ? 2 : 1;
    sc[PR_SC_OWN_TOWERS + 0] = pr_obs_tower_frac(st, team, 0);
    sc[PR_SC_OWN_TOWERS + 1] = pr_obs_tower_frac(st, team, il);
    sc[PR_SC_OWN_TOWERS + 2] = pr_obs_tower_frac(st, team, ir);
    sc[PR_SC_ENEMY_TOWERS + 0] = pr_obs_tower_frac(st, opp, 0);
    sc[PR_SC_ENEMY_TOWERS + 1] = pr_obs_tower_frac(st, opp, il);
    sc[PR_SC_ENEMY_TOWERS + 2] = pr_obs_tower_frac(st, opp, ir);
    sc[PR_SC_KING_ACTIVE + 0] = (float)st->king_active[team];
    sc[PR_SC_KING_ACTIVE + 1] = (float)st->king_active[opp];
    sc[PR_SC_CROWNS + 0] = (float)st->crowns[team] / 3.0f;
    sc[PR_SC_CROWNS + 1] = (float)st->crowns[opp] / 3.0f;
    int seen = 0; /* multi-hot index = card_id (SPEC §16.6.17); slots >= 64 stay 0 */
    for (int c = 0; c < PR_N_CARDS; c++)
        if (st->seen_mask[opp] & ((uint64_t)1 << c)) { sc[PR_SC_OPP_SEEN + c] = 1.0f; seen++; }
    sc[PR_SC_OPP_SPENT] = (float)st->spent[opp] / (float)PR_ELIXIR_UNIT / 200.0f; /* SPEC §16.4: / 200 */
    for (int k = 0; k < 4; k++) sc[PR_SC_OPP_LAST4 + k] = pr_obs_card_id(st->last_played[opp][k]);
    if (seen == 8) { /* all 8 revealed: the hand is the 4 cards not among the last 4 played */
        for (int c = 0; c < PR_N_CARDS; c++) {
            if (!(st->seen_mask[opp] & ((uint64_t)1 << c))) continue;
            int recent = 0;
            for (int k = 0; k < 4; k++) recent |= st->last_played[opp][k] == c;
            if (!recent) sc[PR_SC_OPP_HAND + c] = 1.0f;
        }
    }
    int64_t ub = PR_ELIXIR_START + pr_obs_income(st->tick) - st->spent[opp];
    ub = PR_CLAMP(ub, 0, PR_ELIXIR_MAX);
    sc[PR_SC_OPP_ELIXIR_UB] = (float)ub / (float)PR_ELIXIR_MAX;
    /* SPEC §16.3 / §16.6.16: both tower troops are public information */
    sc[PR_SC_OWN_TT + PR_CLAMP(st->tower_troop[team], 0, PR_N_TOWER_TROOPS - 1)] = 1.0f;
    sc[PR_SC_ENEMY_TT + PR_CLAMP(st->tower_troop[opp], 0, PR_N_TOWER_TROOPS - 1)] = 1.0f;
    /* SPEC §19.3: the own deck as a set -- ids card_id + 1 in ascending card-id order, never the
     * hand / queue order (that order is the hidden shuffle) */
    int8_t deck[8];
    memcpy(deck, st->deck[team], 8);
    for (int i = 1; i < 8; i++)
        for (int j = i; j > 0 && deck[j - 1] > deck[j]; j--) {
            int8_t tmp = deck[j]; deck[j] = deck[j - 1]; deck[j - 1] = tmp;
        }
    for (int k = 0; k < 8; k++) sc[PR_SC_OWN_DECK + k] = pr_obs_card_id(deck[k]);

    /* ---- the mask: exactly the engine's legality function ---- */
    uint8_t mask[PR_N_ACTIONS];
    pr_legal_mask(st, team, mask);
    float *m = out + PR_OBS_MASK_OFFSET;
    for (int a = 0; a < PR_N_ACTIONS; a++) m[a] = (float)mask[a];
}

/* Self-check: the number of actions where mask[a] disagrees with pr_check_play. */
static inline int pr_obs_mask_mismatches(const PrState *st, int team, const float *mask) {
    int bad = 0;
    if (mask[0] != 1.0f) bad++;
    for (int a = 1; a < PR_N_ACTIONS; a++) {
        int slot = (a - 1) / PR_N_TILES, cell = (a - 1) % PR_N_TILES;
        int ok = pr_check_play(st, team, slot, cell % PR_TILES_X, cell / PR_TILES_X) == PR_OK;
        if (ok != (mask[a] > 0.5f)) bad++;
    }
    return bad;
}

#endif /* PR_OBS_ENCODER_H */
