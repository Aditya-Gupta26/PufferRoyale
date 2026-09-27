/*
 * pr_rules.h -- match rules: elixir, hand & cycle, placement legality, card plays,
 * crown towers and the Judge (SPEC §3.1, §4, §4.1).
 */
#ifndef PR_RULES_H
#define PR_RULES_H

#include "pr_entity.h"

/* Defined in pr_spell.h (spells are cast from the play path). */
static inline void pr_cast_spell(PrState *st, int team, int card, int32_t x, int32_t y);

/* ------------------------------------------------------------------ elixir (SPEC §4) */

/* Regen units for tick t: 1x [0, 2400), 2x [2400, 4800), 3x [4800, 6000). */
static inline int32_t pr_elixir_rate(int32_t t) {
    if (t < PR_TICK_2X) return PR_ELIXIR_RATE_1X;
    if (t < PR_TICK_3X) return 2 * PR_ELIXIR_RATE_1X;
    return 3 * PR_ELIXIR_RATE_1X;
}

/* Regen for the tick being processed; anything past the cap is leaked. */
static inline void pr_regen(PrState *st) {
    int32_t r = pr_elixir_rate(st->tick);
    for (int t = 0; t < 2; t++) {
        int32_t e = st->elixir[t] + r;
        if (e > PR_ELIXIR_MAX) {
            pr_sat_add(&st->leaked[t], e - PR_ELIXIR_MAX);
            e = PR_ELIXIR_MAX;
        }
        st->elixir[t] = e;
    }
}

/* Add elixir outside the regen (Elixir Collector, SPEC §16.6.15): capped at 10, the excess leaks. */
static inline void pr_add_elixir(PrState *st, int team, int32_t units) {
    int32_t e = st->elixir[team] + units;
    if (e > PR_ELIXIR_MAX) {
        pr_sat_add(&st->leaked[team], e - PR_ELIXIR_MAX);
        e = PR_ELIXIR_MAX;
    }
    st->elixir[team] = e;
}

/* ------------------------------------------------------------------ hand & cycle (SPEC §4.1) */

/* Playing hand slot i: hand[i] = queue.pop_front(); queue.push_back(card). */
static inline void pr_cycle(int8_t hand[4], int8_t queue[4], int slot) {
    int8_t c = hand[slot];
    hand[slot] = queue[0];
    queue[0] = queue[1];
    queue[1] = queue[2];
    queue[2] = queue[3];
    queue[3] = c;
}

/* ------------------------------------------------------------------ placement (SPEC §3.1) */

static inline PrRect pr_entity_footprint(const PrEntity *e) {
    return pr_footprint_at(e->x, e->y, pr_udef(e)->footprint_tiles);
}

/* Pocket rects of a crown tower (RoyaleSim EnemyTowerRects): King 18x16 tiles,
 * Princess 11x21 tiles, centred on the tower, closed. */
static inline PrRect pr_pocket_rect(const PrEntity *t) {
    int32_t hw = t->tower_idx == 0 ? 9000 : 5500;
    int32_t hh = t->tower_idx == 0 ? 8000 : 10500;
    PrRect r = {t->x - hw, t->y - hh, t->x + hw, t->y + hh};
    return r;
}

/* The living structures a legality query needs, gathered once per query. */
typedef struct PrLegalCtx {
    int n_towers, n_structs;
    PrRect tower_fp[6];
    PrRect pocket[6];          /* enemy crown towers' pocket rects (valid when is_enemy) */
    uint8_t is_enemy[6];
    PrRect struct_fp[PR_MAX_ENTITIES];
} PrLegalCtx;

static inline void pr_legal_ctx(const PrState *st, int team, PrLegalCtx *c) {
    c->n_towers = 0;
    c->n_structs = 0;
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        if (e->hp <= 0 || !pr_is_structure(e)) continue;
        PrRect fp = pr_entity_footprint(e);
        c->struct_fp[c->n_structs++] = fp;
        if (e->kind == PR_KIND_TOWER && c->n_towers < 6) {
            c->tower_fp[c->n_towers] = fp;
            c->pocket[c->n_towers] = pr_pocket_rect(e);
            c->is_enemy[c->n_towers] = e->team != team;
            c->n_towers++;
        }
    }
}

/* Troop territory of own-frame tile (tx, ty): own half, or an opened pocket (tile centre not
 * in the river band and not inside the closed rect of any living enemy crown tower). */
static inline int pr_territory_ctx(const PrLegalCtx *c, int team, int tx, int ty) {
    if (ty >= 17) return 1;                  /* own half */
    if (ty == 15 || ty == 16) return 0;      /* the river band stays closed */
    int etx, ety;
    pr_tile_to_engine(team, tx, ty, &etx, &ety);
    int32_t cx = pr_tile_cx(etx), cy = pr_tile_cy(ety);
    for (int i = 0; i < c->n_towers; i++)
        if (c->is_enemy[i] && pr_rect_contains_closed(c->pocket[i], cx, cy)) return 0;
    return 1;                                /* an opened pocket */
}

/* Troop legality of own-frame tile (tx, ty) for `team` (SPEC §3.1 rule 1): not water, not
 * nodeploy, no positive-area overlap with any living tower OR building footprint (either
 * team, v0.2), and troop territory. */
static inline int pr_troop_tile_legal_ctx(const PrLegalCtx *c, int team, int tx, int ty) {
    int etx, ety;
    pr_tile_to_engine(team, tx, ty, &etx, &ety);
    if (PR_TILE_FLAGS[ety][etx] & (PR_TF_WATER | PR_TF_NODEPLOY)) return 0;
    PrRect tile = {etx * 1000, ety * 1000, etx * 1000 + 1000, ety * 1000 + 1000};
    for (int i = 0; i < c->n_structs; i++)
        if (pr_rect_overlap(tile, c->struct_fp[i])) return 0;
    return pr_territory_ctx(c, team, tx, ty);
}

/* The Log (SPEC §3.1 rule 5, v0.2): troop territory only, not water; it may overlap
 * buildings, towers and no-deploy cells. */
static inline int pr_log_tile_legal_ctx(const PrLegalCtx *c, int team, int tx, int ty) {
    int etx, ety;
    pr_tile_to_engine(team, tx, ty, &etx, &ety);
    if (PR_TILE_FLAGS[ety][etx] & PR_TF_WATER) return 0;
    return pr_territory_ctx(c, team, tx, ty);
}

/* Engine-frame centre of a building tapped on own-frame tile (tx, ty) (SPEC §3.1 rule 2,
 * v0.2): odd F -> the tile centre; even F (Tesla) -> the tile's own-frame top-left corner
 * (tx*1000, ty*1000), so the footprint covers own tiles [tx-F/2, tx+F/2-1] x [ty-F/2, ty+F/2-1]. */
static inline void pr_building_anchor(int team, int unit, int tx, int ty, int32_t *x, int32_t *y) {
    int32_t ox = tx * 1000 + 500, oy = ty * 1000 + 500;
    if (PR_UNITS[unit].footprint_tiles % 2 == 0) {
        ox = tx * 1000;
        oy = ty * 1000;
    }
    *x = pr_own_x(team, ox);
    *y = pr_own_y(team, oy);
}

/* Building legality (SPEC §3.1 rule 2): own half only (no pocket), the whole footprint
 * inside own half and the arena, no water / no-deploy half-cell under it, no
 * positive-area overlap with any living tower or building footprint. */
static inline int pr_building_tile_legal_ctx(const PrLegalCtx *c, int team, int unit, int tx, int ty) {
    if (ty < 17) return 0;
    int32_t bx, by;
    pr_building_anchor(team, unit, tx, ty, &bx, &by);
    PrRect fp = pr_footprint_at(bx, by, PR_UNITS[unit].footprint_tiles);
    if (fp.x0 < 0 || fp.y0 < 0 || fp.x1 > PR_ARENA_W || fp.y1 > PR_ARENA_H) return 0;
    if (team == 0 ? fp.y0 < PR_RIVER_Y1 : fp.y1 > PR_RIVER_Y0) return 0;
    for (int cy = fp.y0 / PR_CELL; cy <= (fp.y1 - 1) / PR_CELL; cy++)
        for (int cx = fp.x0 / PR_CELL; cx <= (fp.x1 - 1) / PR_CELL; cx++)
            if (PR_CELL_BITS[cy][cx] & (PR_CB_WATER | PR_CB_NODEPLOY)) return 0;
    for (int i = 0; i < c->n_structs; i++)
        if (pr_rect_overlap(fp, c->struct_fp[i])) return 0;
    return 1;
}

/* The Miner (SPEC §16.1, §16.6.3): any non-water tile outside every LIVING crown-tower footprint
 * (either team, positive-area overlap); building footprints and no-deploy cells are allowed. */
static inline int pr_miner_tile_legal_ctx(const PrLegalCtx *c, int team, int tx, int ty) {
    int etx, ety;
    pr_tile_to_engine(team, tx, ty, &etx, &ety);
    if (PR_TILE_FLAGS[ety][etx] & PR_TF_WATER) return 0;
    PrRect tile = {etx * 1000, ety * 1000, etx * 1000 + 1000, ety * 1000 + 1000};
    for (int i = 0; i < c->n_towers; i++)
        if (pr_rect_overlap(tile, c->tower_fp[i])) return 0;
    return 1;
}

static inline int pr_card_tile_legal_ctx(const PrLegalCtx *c, int team, int card, int tx, int ty) {
    if (tx < 0 || ty < 0 || tx >= PR_TILES_X || ty >= PR_TILES_Y) return 0;
    const PrCardDef *cd = &PR_CARDS[card];
    switch (cd->placement) {
    case PR_PLACE_TROOP: return pr_troop_tile_legal_ctx(c, team, tx, ty);
    case PR_PLACE_BUILDING: return pr_building_tile_legal_ctx(c, team, cd->unit, tx, ty);
    case PR_PLACE_MINER: return pr_miner_tile_legal_ctx(c, team, tx, ty);
    case PR_PLACE_TERRITORY: return pr_log_tile_legal_ctx(c, team, tx, ty); /* the Log, Barbarian Barrel */
    case PR_PLACE_LAND: {                                                   /* Goblin Barrel: any non-water tile */
        int etx, ety;
        pr_tile_to_engine(team, tx, ty, &etx, &ety);
        return (PR_TILE_FLAGS[ety][etx] & PR_TF_WATER) == 0;
    }
    default: return 1; /* Fireball, Arrows, Zap, Rocket, Poison, Freeze, Earthquake, Lightning, Snowball */
    }
}

/* Single-tile entry points (tests, pr_check_play). */
static inline int pr_troop_tile_legal(const PrState *st, int team, int tx, int ty) {
    PrLegalCtx c;
    pr_legal_ctx(st, team, &c);
    return pr_troop_tile_legal_ctx(&c, team, tx, ty);
}

static inline int pr_building_tile_legal(const PrState *st, int team, int unit, int tx, int ty) {
    PrLegalCtx c;
    pr_legal_ctx(st, team, &c);
    return pr_building_tile_legal_ctx(&c, team, unit, tx, ty);
}

static inline int pr_card_tile_legal(const PrState *st, int team, int card, int tx, int ty) {
    PrLegalCtx c;
    pr_legal_ctx(st, team, &c);
    return pr_card_tile_legal_ctx(&c, team, card, tx, ty);
}

/* All 576 own-frame tiles for one card (row-major cell = ty * 18 + tx). */
static inline void pr_legal_tiles_ctx(const PrLegalCtx *c, int team, int card, uint8_t *out) {
    for (int ty = 0; ty < PR_TILES_Y; ty++)
        for (int tx = 0; tx < PR_TILES_X; tx++)
            out[ty * PR_TILES_X + tx] = (uint8_t)pr_card_tile_legal_ctx(c, team, card, tx, ty);
}

static inline void pr_legal_tiles(const PrState *st, int team, int card, uint8_t *out) {
    PrLegalCtx c;
    pr_legal_ctx(st, team, &c);
    pr_legal_tiles_ctx(&c, team, card, out);
}

/* The engine's single legality function (SPEC §3.1 + play conditions), evaluated on the
 * CURRENT state (SPEC §13.1: a queued play is validated -- again -- and applied in the next
 * Upkeep). Error precedence [IMPL-DEFINED]: GAME_OVER, BAD_SLOT, LOCKOUT, NOT_ENOUGH_ELIXIR,
 * ILLEGAL_POSITION. */
static inline int pr_check_play(const PrState *st, int team, int slot, int tx, int ty) {
    if (st->over) return PR_ERR_GAME_OVER;
    if (slot < 0 || slot > 3 || team < 0 || team > 1) return PR_ERR_BAD_SLOT;
    if (st->n_pending[team] >= PR_MAX_PENDING_PLAYS) return PR_ERR_BAD_SLOT;
    if (st->tick < st->lockout_ticks) return PR_ERR_LOCKOUT;
    int card = st->hand[team][slot];
    if (card < 0 || card >= PR_N_CARDS) return PR_ERR_BAD_SLOT;
    if (st->elixir[team] < PR_CARDS[card].elixir * PR_ELIXIR_UNIT) return PR_ERR_NOT_ENOUGH_ELIXIR;
    if (!pr_card_tile_legal(st, team, card, tx, ty)) return PR_ERR_ILLEGAL_POSITION;
    return PR_OK;
}

/* Queue a play for the next tick: legality on own-frame tile (tx, ty), placement at
 * the engine point (x, y) for troops and spells; buildings always go to their tile anchor
 * (pr_building_anchor), so their footprint is exactly the one the legality check saw. */
static inline int pr_queue_play_full(PrState *st, int team, int slot, int tx, int ty, int32_t x, int32_t y) {
    int err = pr_check_play(st, team, slot, tx, ty);
    if (err != PR_OK) return err;
    PrPlay *p = &st->pending[team][st->n_pending[team]++];
    memset(p, 0, sizeof(*p));
    p->slot = (int8_t)slot;
    p->card = st->hand[team][slot];
    p->tx = (int8_t)tx;
    p->ty = (int8_t)ty;
    if (PR_CARDS[(int)p->card].kind == PR_CARD_KIND_BUILDING)
        pr_building_anchor(team, PR_CARDS[(int)p->card].unit, tx, ty, &x, &y);
    p->x = x;
    p->y = y;
    return PR_OK;
}

/* Queue a play at own-frame tile (tx, ty), placed at the tile centre (SPEC §3.1; the env
 * action and Game.play_tile). */
static inline int pr_queue_play(PrState *st, int team, int slot, int tx, int ty) {
    int etx = 0, ety = 0;
    if (tx >= 0 && ty >= 0 && tx < PR_TILES_X && ty < PR_TILES_Y) pr_tile_to_engine(team, tx, ty, &etx, &ety);
    return pr_queue_play_full(st, team, slot, tx, ty, pr_tile_cx(etx), pr_tile_cy(ety));
}

/* Queue a play at the engine-frame tap point (x, y) (Game.play): the tile is the one
 * containing the tap in the acting team's OWN frame (SPEC §1 rotation, then floor), so
 * the rule is seat-symmetric; troops and spells are placed at the tap point itself. */
static inline int pr_queue_play_at(PrState *st, int team, int slot, int64_t x, int64_t y) {
    int64_t ox = team == 0 ? x : PR_ARENA_W - x, oy = team == 0 ? y : PR_ARENA_H - y;
    int tx = (ox < 0 || ox >= PR_ARENA_W) ? -1 : (int)(ox / 1000);
    int ty = (oy < 0 || oy >= PR_ARENA_H) ? -1 : (int)(oy / 1000);
    if (tx < 0 || ty < 0) return pr_queue_play_full(st, team, slot, -1, -1, 0, 0);
    return pr_queue_play_full(st, team, slot, tx, ty, (int32_t)x, (int32_t)y);
}

/* Action mask (SPEC §9): mask[0] = 1; mask[1 + slot * 576 + ty * 18 + tx] = legal. */
static inline void pr_legal_mask(const PrState *st, int team, uint8_t *mask) {
    memset(mask, 0, PR_N_ACTIONS);
    mask[0] = 1;
    if (st->over || st->tick < st->lockout_ticks) return;
    if (st->n_pending[team] >= PR_MAX_PENDING_PLAYS) return;
    PrLegalCtx ctx;
    pr_legal_ctx(st, team, &ctx);
    /* one tile map per legality class: troops share one, buildings one per unit, spells */
    uint8_t troop[PR_N_TILES];
    int have_troop = 0;
    for (int slot = 0; slot < 4; slot++) {
        int card = st->hand[team][slot];
        if (card < 0 || card >= PR_N_CARDS) continue;
        if (st->elixir[team] < PR_CARDS[card].elixir * PR_ELIXIR_UNIT) continue;
        uint8_t *dst = mask + 1 + slot * PR_N_TILES;
        if (PR_CARDS[card].placement == PR_PLACE_TROOP) {
            if (!have_troop) {
                pr_legal_tiles_ctx(&ctx, team, card, troop);
                have_troop = 1;
            }
            memcpy(dst, troop, PR_N_TILES);
        } else {
            pr_legal_tiles_ctx(&ctx, team, card, dst);
        }
    }
}

/* A hand-played Miner (SPEC §16.6.4, §16.6.23): from the play tick an entity at its own King's
 * centre, hidden (untargetable, immune, inactive), travelling underground to the tap point in a
 * straight line at spawn_pathfind.speed; it surfaces and starts its deploy after ceil(d / speed)
 * ticks (pr_phase_status / pr_phase_move). */
static inline PrEntity *pr_start_burrow(PrState *st, int team, int card, int unit, int32_t x, int32_t y) {
    PrEntity *e = pr_new_entity(st, unit, team, card, PR_TOWER_POS[team][0][0], PR_TOWER_POS[team][0][1]);
    if (!e) return NULL;
    e->burrow = 1;
    e->hide_state = PR_HIDE_HIDDEN;
    e->deploy_ms = 0;
    e->tgt_x = x;
    e->tgt_y = y;
    return e;
}

/* Put a card on the board at an engine point (no checks). `from_play` = a hand play applied in
 * Upkeep (the Miner burrows from its King); the debug spawn hook puts every unit straight at
 * (x, y). Returns the number of entity ids written to `out`. */
static inline int pr_deploy_card(PrState *st, int team, int card, int32_t x, int32_t y, int deployed,
                          int from_play, uint32_t *out, int max_out) {
    const PrCardDef *c = &PR_CARDS[card];
    if (c->kind == PR_CARD_KIND_SPELL) {
        pr_cast_spell(st, team, card, x, y);
        return 0;
    }
    if (from_play && PR_UNITS[c->unit].burrow_speed > 0) {
        PrEntity *e = pr_start_burrow(st, team, card, c->unit, x, y);
        if (e && out && max_out > 0) out[0] = e->id;
        return e ? 1 : 0;
    }
    int first = st->n_ent;
    int n = pr_spawn_formation2(st, c->unit, c->count, c->unit2, c->count2, team, card, x, y, c->formation,
                                c->deploy_delay_ms, out, max_out);
    if (deployed) {
        /* TEST HOOK semantics: as if the deploy had just completed, including the
         * load timer having run down during it [IMPL-DEFINED]. */
        for (int i = first; i < st->n_ent; i++) {
            PrEntity *e = &st->ent[i];
            int32_t lt = PR_UNITS[e->unit].load_time_ms;
            e->load_ms = PR_MAX(0, lt - e->deploy_ms);
            e->deploy_ms = 0;
            if (PR_UNITS[e->unit].hides) e->hide_state = PR_HIDE_HIDDEN;
        }
    }
    return n;
}

static inline void pr_record_play(PrState *st, int team, int card) {
    int32_t cost = PR_CARDS[card].elixir * PR_ELIXIR_UNIT;
    st->elixir[team] -= cost;
    st->spent[team] += cost;
    st->seen_mask[team] |= (uint64_t)1 << card;
    for (int i = 3; i > 0; i--) st->last_played[team][i] = st->last_played[team][i - 1];
    st->last_played[team][0] = (int8_t)card;
    st->plays[team]++;
}

/* Upkeep (SPEC v0.2.1 §14.1): every play applied in this Upkeep is validated against the
 * state at the START of the Upkeep (after regen, before any play of this tick is applied): a
 * team's own queued plays in queue order against its running elixir and hand, and plays of
 * different teams never invalidate each other (a troop landing where the enemy places a
 * building this tick is resolved by collision). A play that fails is dropped and counted in
 * dropped_plays; since the legal mask is the same test on the same state, a mask-legal env
 * action is always applied. The accepted plays are then applied team by team: team 0 first
 * (§13.13), or, with alternate_first, the first team alternating after every tick on which
 * both teams play. */
static inline void pr_apply_tick_plays(PrState *st) {
    PrPlay acc[2][PR_MAX_PENDING_PLAYS];
    int nacc[2] = {0, 0};
    for (int team = 0; team < 2; team++) {
        int n = PR_CLAMP(st->n_pending[team], 0, PR_MAX_PENDING_PLAYS);
        if (n == 0) continue;
        PrLegalCtx ctx;
        pr_legal_ctx(st, team, &ctx);
        int32_t elixir = st->elixir[team];
        int8_t hand[4], queue[4];
        memcpy(hand, st->hand[team], 4);
        memcpy(queue, st->queue[team], 4);
        for (int i = 0; i < n; i++) {
            const PrPlay *p = &st->pending[team][i];
            int card = p->card, slot = p->slot;
            int ok = !st->over && st->tick >= st->lockout_ticks && slot >= 0 && slot < 4 && card >= 0 &&
                     card < PR_N_CARDS && hand[slot] == card &&
                     elixir >= PR_CARDS[card].elixir * PR_ELIXIR_UNIT &&
                     pr_card_tile_legal_ctx(&ctx, team, card, p->tx, p->ty);
            if (!ok) {
                st->dropped_plays++;
                continue;
            }
            acc[team][nacc[team]++] = *p;
            elixir -= PR_CARDS[card].elixir * PR_ELIXIR_UNIT;
            pr_cycle(hand, queue, slot);
        }
    }
    memset(st->pending, 0, sizeof(st->pending));
    st->n_pending[0] = st->n_pending[1] = 0;
    int first = 0;
    if (st->alternate_first && nacc[0] > 0 && nacc[1] > 0) {
        first = st->first_team & 1;
        st->first_team = (uint8_t)(1 - first);
    }
    for (int k = 0; k < 2; k++) {
        int team = k == 0 ? first : 1 - first;
        for (int i = 0; i < nacc[team]; i++) {
            const PrPlay *p = &acc[team][i];
            pr_record_play(st, team, p->card);
            pr_cycle(st->hand[team], st->queue[team], p->slot);
            pr_deploy_card(st, team, p->card, p->x, p->y, 0, 1, NULL, 0);
        }
    }
}

/* ------------------------------------------------------------------ crown towers */

static inline void pr_setup_towers(PrState *st) {
    for (int team = 0; team < 2; team++) {
        for (int idx = 0; idx < 3; idx++) {
            /* SPEC §16.3: the team's tower troop replaces BOTH Princess towers */
            int tt = PR_CLAMP(st->tower_troop[team], 0, PR_N_TOWER_TROOPS - 1);
            int unit = idx == 0 ? PR_UNIT_KINGTOWER : PR_TOWER_TROOP_UNITS[tt];
            PrEntity *e = pr_new_entity(st, unit, team, -1, PR_TOWER_POS[team][idx][0], PR_TOWER_POS[team][idx][1]);
            if (!e) { /* unreachable: called on a freshly cleared pool (6 of 256 slots) */
                st->tower_id[team][idx] = PR_NO_ID;
                continue;
            }
            e->tower_idx = (int8_t)idx;
            e->deploy_ms = 0;
            e->load_ms = 0; /* fully loaded at the start */
            st->tower_id[team][idx] = e->id;
        }
        st->king_wake[team] = -1;
        st->king_active[team] = 0;
    }
}

static inline void pr_trigger_king(PrState *st, int team) {
    if (!st->king_active[team] && st->king_wake[team] < 0) st->king_wake[team] = PR_KING_WAKE_TICKS;
}

/* Side effects of a crown tower's destruction (SPEC §6.6). */
static inline void pr_tower_destroyed(PrState *st, const PrEntity *t) {
    if (t->tower_idx == 1 || t->tower_idx == 2) {
        st->crowns[1 - t->team] = PR_MIN(st->crowns[1 - t->team] + 1, 3); /* 3 at most (a debug kill after the King) */
        pr_trigger_king(st, t->team);
    }
    /* A destroyed King is judged by pr_judge (the match ends). */
}

static inline int pr_tower_alive(const PrState *st, int team, int idx) {
    return pr_get_c(st, st->tower_id[team][idx]) != NULL;
}

/* ------------------------------------------------------------------ Judge (SPEC §4) */

static inline void pr_finish(PrState *st, int winner, int reason) {
    st->over = 1;
    if (winner < 0) {
        st->result[0] = st->result[1] = 0;
        st->end_reason = PR_END_DRAW;
    } else {
        st->result[winner] = 1;
        st->result[1 - winner] = -1;
        st->end_reason = (uint8_t)reason;
    }
}

/* Weakest surviving crown tower of a side, as hp and max_hp. */
static inline void pr_weakest_tower(const PrState *st, int team, int frac, int64_t *hp, int64_t *mx) {
    *hp = -1;
    *mx = 1;
    for (int idx = 0; idx < 3; idx++) {
        const PrEntity *e = pr_get_c(st, st->tower_id[team][idx]);
        if (!e) continue;
        int better;
        if (*hp < 0) better = 1;
        else if (frac) better = (int64_t)e->hp * *mx < *hp * (int64_t)e->max_hp;
        else better = e->hp < *hp;
        if (better) { *hp = e->hp; *mx = e->max_hp; }
    }
}

/* End conditions after tick t has been processed. `immediate` = called outside the
 * tick loop (a debug setter destroyed a tower), which never runs the tiebreak. */
static inline void pr_judge(PrState *st, int32_t t, int immediate) {
    if (st->over) return;
    int k0 = pr_tower_alive(st, 0, 0), k1 = pr_tower_alive(st, 1, 0);
    if (!k0 || !k1) {
        if (!k0 && !k1) {
            pr_finish(st, -1, PR_END_DRAW);
        } else {
            int winner = k0 ? 0 : 1;
            st->crowns[winner] = 3;
            pr_finish(st, winner, PR_END_KING);
        }
        return;
    }
    if (t >= PR_TICKS_REGULATION - 1 && st->crowns[0] != st->crowns[1]) {
        int winner = st->crowns[0] > st->crowns[1] ? 0 : 1;
        int reason = (t == PR_TICKS_REGULATION - 1 && !immediate) ? PR_END_REGULATION_CROWNS : PR_END_OVERTIME_CROWNS;
        pr_finish(st, winner, reason);
        return;
    }
    if (!immediate && t >= PR_TICKS_MAX - 1) {
        int frac = st->tiebreak == PR_TIEBREAK_FRACTION;
        int64_t h0, m0, h1, m1;
        pr_weakest_tower(st, 0, frac, &h0, &m0);
        pr_weakest_tower(st, 1, frac, &h1, &m1);
        int64_t a = frac ? h0 * m1 : h0, b = frac ? h1 * m0 : h1;
        if (a == b) pr_finish(st, -1, PR_END_DRAW);
        else pr_finish(st, a < b ? 1 : 0, PR_END_TIEBREAK); /* lower weakest tower loses */
    }
}

#endif /* PR_RULES_H */
