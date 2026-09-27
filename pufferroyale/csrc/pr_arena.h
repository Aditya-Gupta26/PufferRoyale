/*
 * pr_arena.h -- static arena queries over the generated half-tile bitmask (SPEC §3).
 *
 * SEAT SYMMETRY (SPEC §1: team 1's frame is the 180-degree rotation). Tile centres lie
 * exactly on half-cell boundaries, and a half-open interval [a, b) rotates into the
 * half-open-on-the-other-side (W-b, W-a]. So every point query that a MOVING unit
 * makes (its cell, "is this water?", the water ejection, the arena clamp) comes in a
 * team-oriented form: team 0 uses [a, b), team 1 uses (a, b]. Each seat then sees the
 * identical arena in its own frame and a scenario and its rotation evolve as exact
 * rotations of each other. (The arena bitmask itself is rotation-symmetric.)
 */
#ifndef PR_ARENA_QUERIES_H
#define PR_ARENA_QUERIES_H

#include "pr_types.h"

static inline int pr_in_arena(int32_t x, int32_t y) {
    return x >= 0 && y >= 0 && x < PR_ARENA_W && y < PR_ARENA_H;
}

/* Half-tile cell index of a point (engine convention [a, b)), or -1 outside. */
static inline int pr_cell_of(int32_t x, int32_t y) {
    if (!pr_in_arena(x, y)) return -1;
    return (y / PR_CELL) * PR_COLS + (x / PR_CELL);
}

/* Team-oriented cell lookup: team 0 [a, b), team 1 (a, b]. -1 outside. */
static inline int pr_cell_of_t(int team, int32_t x, int32_t y) {
    if (team == 0) return pr_cell_of(x, y);
    if (x <= 0 || y <= 0 || x > PR_ARENA_W || y > PR_ARENA_H) return -1;
    return ((y - 1) / PR_CELL) * PR_COLS + ((x - 1) / PR_CELL);
}

static inline int32_t pr_cell_cx(int cell) { return (cell % PR_COLS) * PR_CELL + PR_CELL / 2; }
static inline int32_t pr_cell_cy(int cell) { return (cell / PR_COLS) * PR_CELL + PR_CELL / 2; }

static inline uint8_t pr_cell_bits(int cell) { return PR_CELL_BITS[cell / PR_COLS][cell % PR_COLS]; }

static inline int pr_cell_water(int cell) { return (pr_cell_bits(cell) & PR_CB_WATER) != 0; }

/* Is the point on a water cell (engine convention)? Outside the arena: not water. */
static inline int pr_point_water(int32_t x, int32_t y) {
    int c = pr_cell_of(x, y);
    return c >= 0 && pr_cell_water(c);
}

/* Team-oriented water test for units. */
static inline int pr_point_water_t(int team, int32_t x, int32_t y) {
    int c = pr_cell_of_t(team, x, y);
    return c >= 0 && pr_cell_water(c);
}

/* Clamp a unit centre into the arena interior [1, W-1] x [1, H-1] (a range that is
 * its own rotation, so the clamp is seat-symmetric). */
static inline void pr_clamp_arena(int32_t *x, int32_t *y) {
    *x = PR_CLAMP(*x, 1, PR_ARENA_W - 1);
    *y = PR_CLAMP(*y, 1, PR_ARENA_H - 1);
}

/* "Wet" for a ground unit: water under EITHER seat's convention (so also on the
 * water's boundary lines). The union is rotation-symmetric, and a unit that is never
 * wet is never on water whichever half-open reading of SPEC §3 one applies. */
static inline int pr_point_wet(int32_t x, int32_t y) {
    return pr_point_water_t(0, x, y) || pr_point_water_t(1, x, y);
}

/* Nearest dry point for a ground unit of `team` that is wet [IMPL-DEFINED]. Computed
 * in the team's OWN frame: candidates are the own bank straight across (first, so it
 * wins ties), the far bank, and the nearest point strictly inside each bridge deck at
 * the same y; the closest candidate that is not wet wins. No-op when not wet. */
static inline void pr_eject_water(int team, int32_t *x, int32_t *y) {
    if (!pr_point_wet(*x, *y)) return;
    int32_t ox = pr_own_x(team, *x), oy = pr_own_y(team, *y);
    int32_t cand[4][2] = {
        {ox, PR_RIVER_Y1 + 1},                                           /* own bank */
        {ox, PR_RIVER_Y0 - 1},                                           /* far bank */
        {PR_CLAMP(ox, PR_BRIDGE_L_X0 + 1, PR_BRIDGE_L_X1 - 1), oy},      /* own-left bridge */
        {PR_CLAMP(ox, PR_BRIDGE_R_X0 + 1, PR_BRIDGE_R_X1 - 1), oy},      /* own-right bridge */
    };
    int best = -1;
    int64_t bd = 0;
    for (int i = 0; i < 4; i++) {
        if (pr_point_wet(cand[i][0], cand[i][1])) continue;
        int64_t d = pr_dist2(ox, oy, cand[i][0], cand[i][1]);
        if (best < 0 || d < bd) { best = i; bd = d; }
    }
    if (best >= 0) {
        *x = pr_own_x(team, cand[best][0]);
        *y = pr_own_y(team, cand[best][1]);
    }
}

/* Axis-aligned footprint of a building/tower: side F tiles centred on it (SPEC §3).
 * Half-open [x0, x1) x [y0, y1). */
typedef struct PrRect { int32_t x0, y0, x1, y1; } PrRect;

static inline PrRect pr_footprint_at(int32_t x, int32_t y, int f_tiles) {
    int32_t h = f_tiles * 500;
    PrRect r = {x - h, y - h, x + h, y + h};
    return r;
}

static inline int pr_rect_overlap(PrRect a, PrRect b) { /* positive-area overlap */
    return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
}

/* Closed-rect point test (pocket rule, RoyaleSim EnemyTowerRects). */
static inline int pr_rect_contains_closed(PrRect r, int32_t x, int32_t y) {
    return x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1;
}

/* Own-frame tile -> engine tile (SPEC §1: (tx, ty) <-> (17 - tx, 31 - ty)). */
static inline void pr_tile_to_engine(int team, int tx, int ty, int *etx, int *ety) {
    if (team == 0) { *etx = tx; *ety = ty; }
    else { *etx = PR_TILES_X - 1 - tx; *ety = PR_TILES_Y - 1 - ty; }
}

static inline int32_t pr_tile_cx(int etx) { return etx * 1000 + 500; }
static inline int32_t pr_tile_cy(int ety) { return ety * 1000 + 500; }

#endif /* PR_ARENA_QUERIES_H */
