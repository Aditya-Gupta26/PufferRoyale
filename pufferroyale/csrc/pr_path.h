/*
 * pr_path.h -- deterministic ground pathing over the 36x64 half-tile grid (SPEC §6.3).
 *
 * [IMPL-DEFINED] algorithm: Dijkstra distance fields ("flow fields") toward a goal,
 * plus line-of-sight smoothing.
 *
 *   grid      8-connected half-tile cells; orthogonal step 10, diagonal 14
 *             (ledger pathfinding.DIAGONAL_COST_RATIO 1414/1000); a diagonal may not
 *             cut the corner of an impassable cell.
 *   water     impassable for ground units except jump-enabled ones (Hog Rider, Prince).
 *   occlusion a cell whose centre lies inside the footprint square of a living tower or
 *             building costs 6x to ENTER (ledger pathfinding.OCCLUDED_CELL_TREATMENT =
 *             cost_50 over default 8), except the goal's own footprint. It is a cost,
 *             never a wall, so every dry cell stays reachable.
 *   field     dist[c] = cheapest cost from cell c into the goal set (reverse Dijkstra
 *             with an indexed binary heap). The goal set is the goal structure's
 *             footprint, or the passable cells of the chased troop's CHASE BLOCK (2x2 tiles,
 *             so chasers of nearby targets share a field and a field stays valid while its
 *             target moves inside the block); inside the block without sight of the target
 *             a unit routes to the target's own cell.
 *   waypoint  if the straight segment to the goal point is clear (no impassable or
 *             foreign-occluded cell) the unit walks straight at the goal; otherwise it
 *             follows the field's steepest-descent chain for up to 8 cells and aims at
 *             the farthest chain cell it can see.
 *
 * NEVER STUCK: every dry cell has a finite field value (the two halves connect over
 * the bridges and occlusion is only a cost), each chain step strictly decreases it,
 * and a unit always steps toward a cell it can reach in a straight line.
 *
 * DETERMINISM & SNAPSHOTS. The occluder grid and the fields are cached in PrPathCache,
 * which lives OUTSIDE PrState. Every cached object is a pure function of its key
 * (goal key, jumper flag) and of the occluder signature -- an FNV hash of (id, unit, x, y)
 * of every living structure -- so a cache hit and a rebuild give identical results and
 * a restored snapshot can never read a stale field. The per-entity waypoint cache
 * (wp_* fields) lives inside PrState and is keyed the same way: it is recomputed when
 * the unit's cell, its goal key or the occluder signature changes.
 */
#ifndef PR_PATH_H
#define PR_PATH_H

#include "pr_entity.h"

#define PR_N_FIELDS 24
#define PR_DIST_INF 0xFFFFu
#define PR_OCC_MULT 6
#define PR_CHAIN_LEN 8
#define PR_GOAL_STRUCT 0x80000000u /* goal key flag: low bits = structure id */
#define PR_GOAL_BLOCK 0x40000000u  /* goal key flag: low bits = chase block index */
#define PR_BLOCK 4                 /* chase block side in cells (2x2 tiles) */
#define PR_BLOCK_COLS (PR_COLS / PR_BLOCK)
#define PR_BLOCK_ROWS (PR_ROWS / PR_BLOCK)
#define PR_LOS_STEP 125            /* line-of-sight sample spacing, millitiles */

/* ---- indexed binary min-heap over cells, keyed by (dist, cell) ---- */
typedef struct PrHeap {
    int n;
    uint16_t h[PR_N_CELLS];
    int16_t pos[PR_N_CELLS];
} PrHeap;

typedef struct PrField {
    uint32_t key;      /* PR_GOAL_STRUCT | id, or a cell index */
    uint64_t sig;      /* occluder signature it was built under */
    uint32_t stamp;    /* LRU */
    uint8_t jumper;
    uint8_t valid;
    uint16_t dist[PR_N_CELLS];
} PrField;

typedef struct PrPathCache {
    uint64_t sig;              /* signature the occluder grid below was built for (64-bit:
                                * the cache must never confuse two structure sets) */
    uint32_t stamp;
    int32_t built;
    uint32_t occ[PR_N_CELLS];  /* id of the structure covering the cell, PR_NO_ID = none */
    uint64_t field_builds;     /* statistics only */
    PrField f[PR_N_FIELDS];
} PrPathCache;

/* 8 neighbours: 4 orthogonal first (fixed order = deterministic tie-break). */
static const int8_t PR_NB_DX[8] = {1, -1, 0, 0, 1, 1, -1, -1};
static const int8_t PR_NB_DY[8] = {0, 0, 1, -1, 1, -1, 1, -1};
static const int8_t PR_NB_COST[8] = {10, 10, 10, 10, 14, 14, 14, 14};

static inline uint64_t pr_occluder_sig(const PrState *st) {
    uint64_t h = PR_FNV_OFFSET;
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        if (e->hp <= 0 || !pr_is_structure(e)) continue;
        int32_t rec[4] = {(int32_t)e->id, e->unit, e->x, e->y};
        h = pr_fnv1a(h, rec, sizeof(rec));
    }
    return h;
}

/* The per-entity copy of the signature (a change detector stored in the state). */
static inline uint32_t pr_sig32(uint64_t sig) { return (uint32_t)(sig ^ (sig >> 32)); }

/* Rebuild the occluder grid if the set of living structures changed. */
static inline void pr_path_sync(const PrState *st, PrPathCache *pc) {
    uint64_t sig = pr_occluder_sig(st);
    if (pc->built && pc->sig == sig) return;
    for (int c = 0; c < PR_N_CELLS; c++) pc->occ[c] = PR_NO_ID;
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        if (e->hp <= 0 || !pr_is_structure(e)) continue;
        PrRect r = pr_footprint_at(e->x, e->y, pr_udef(e)->footprint_tiles);
        int cx0 = PR_MAX(0, r.x0 / PR_CELL), cx1 = PR_MIN(PR_COLS - 1, (r.x1 - 1) / PR_CELL);
        int cy0 = PR_MAX(0, r.y0 / PR_CELL), cy1 = PR_MIN(PR_ROWS - 1, (r.y1 - 1) / PR_CELL);
        for (int cy = cy0; cy <= cy1; cy++) {
            for (int cx = cx0; cx <= cx1; cx++) {
                int32_t px = cx * PR_CELL + PR_CELL / 2, py = cy * PR_CELL + PR_CELL / 2;
                /* cell centre strictly inside the footprint: the open square is its own 180-degree
                 * rotation (a half-open test is not, for an off-grid debug building whose edge falls
                 * on a cell centre; grid-anchored footprints never touch a centre, so nothing changes
                 * for placed buildings and towers) */
                if (px <= r.x0 || px >= r.x1 || py <= r.y0 || py >= r.y1) continue;
                int c = cy * PR_COLS + cx;
                if (pc->occ[c] == PR_NO_ID) pc->occ[c] = e->id; /* first (lowest id) wins */
            }
        }
    }
    pc->sig = sig;
    pc->built = 1;
}

static inline int pr_cell_passable(int c, int jumper) { return jumper || !pr_cell_water(c); }

/* Cost of ENTERING cell c by a step of base cost `base`. */
static inline int pr_enter_cost(const PrPathCache *pc, int c, int base, uint32_t goal_id) {
    uint32_t o = pc->occ[c];
    return (o != PR_NO_ID && o != goal_id) ? base * PR_OCC_MULT : base;
}

/* Is the move from cell a by neighbour k legal? Returns the target cell or -1. The
 * neighbour table is read in the mover's OWN frame (negated for team 1) so that the
 * fixed tie-break order is itself rotation-symmetric. */
static inline int pr_nb(int a, int k, int jumper, int team) {
    int dx = team ? -PR_NB_DX[k] : PR_NB_DX[k], dy = team ? -PR_NB_DY[k] : PR_NB_DY[k];
    int cx = a % PR_COLS + dx, cy = a / PR_COLS + dy;
    if (cx < 0 || cy < 0 || cx >= PR_COLS || cy >= PR_ROWS) return -1;
    int b = cy * PR_COLS + cx;
    if (!pr_cell_passable(b, jumper)) return -1;
    if (k >= 4) { /* no corner cutting past an impassable cell */
        int o1 = (a / PR_COLS) * PR_COLS + cx;
        int o2 = cy * PR_COLS + (a % PR_COLS);
        if (!pr_cell_passable(o1, jumper) || !pr_cell_passable(o2, jumper)) return -1;
    }
    return b;
}

static inline int pr_heap_less(const uint16_t *dist, int a, int b) {
    return dist[a] < dist[b] || (dist[a] == dist[b] && a < b);
}

static inline void pr_heap_up(PrHeap *hp, const uint16_t *dist, int i) {
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!pr_heap_less(dist, hp->h[i], hp->h[p])) break;
        uint16_t t = hp->h[i]; hp->h[i] = hp->h[p]; hp->h[p] = t;
        hp->pos[hp->h[i]] = (int16_t)i;
        hp->pos[hp->h[p]] = (int16_t)p;
        i = p;
    }
}

static inline void pr_heap_down(PrHeap *hp, const uint16_t *dist, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < hp->n && pr_heap_less(dist, hp->h[l], hp->h[m])) m = l;
        if (r < hp->n && pr_heap_less(dist, hp->h[r], hp->h[m])) m = r;
        if (m == i) break;
        uint16_t t = hp->h[i]; hp->h[i] = hp->h[m]; hp->h[m] = t;
        hp->pos[hp->h[i]] = (int16_t)i;
        hp->pos[hp->h[m]] = (int16_t)m;
        i = m;
    }
}

static inline void pr_heap_push_or_decrease(PrHeap *hp, const uint16_t *dist, int c) {
    if (hp->pos[c] >= 0) {
        pr_heap_up(hp, dist, hp->pos[c]);
        return;
    }
    hp->h[hp->n] = (uint16_t)c;
    hp->pos[c] = (int16_t)hp->n;
    hp->n++;
    pr_heap_up(hp, dist, hp->n - 1);
}

static inline int pr_heap_pop(PrHeap *hp, const uint16_t *dist) {
    int c = hp->h[0];
    hp->pos[c] = -2; /* closed */
    hp->n--;
    if (hp->n > 0) {
        hp->h[0] = hp->h[hp->n];
        hp->pos[hp->h[0]] = 0;
        pr_heap_down(hp, dist, 0);
    }
    return c;
}

/* Build field f for (key, jumper) on the current occluder grid. */
static inline void pr_field_build(PrPathCache *pc, PrField *f, uint32_t key, int jumper) {
    PrHeap heap; /* Dijkstra scratch (9 KB of stack), fully initialised here */
    PrHeap *hp = &heap;
    uint16_t *dist = f->dist;
    for (int c = 0; c < PR_N_CELLS; c++) {
        dist[c] = PR_DIST_INF;
        hp->pos[c] = -1;
    }
    hp->n = 0;
    uint32_t goal_id = (key & PR_GOAL_STRUCT) ? (key & ~PR_GOAL_STRUCT) : PR_NO_ID;
    int ngoal = 0;
    if (goal_id != PR_NO_ID) {
        for (int c = 0; c < PR_N_CELLS; c++) {
            if (pc->occ[c] == goal_id && pr_cell_passable(c, jumper)) {
                dist[c] = 0;
                pr_heap_push_or_decrease(hp, dist, c);
                ngoal++;
            }
        }
    } else if (key & PR_GOAL_BLOCK) { /* every passable cell of a chase block */
        int b = (int)(key & ~PR_GOAL_BLOCK);
        int bx = b % PR_BLOCK_COLS, by = b / PR_BLOCK_COLS;
        for (int cy = by * PR_BLOCK; cy < by * PR_BLOCK + PR_BLOCK; cy++) {
            for (int cx = bx * PR_BLOCK; cx < bx * PR_BLOCK + PR_BLOCK; cx++) {
                int c = cy * PR_COLS + cx;
                if (!pr_cell_passable(c, jumper)) continue;
                dist[c] = 0;
                pr_heap_push_or_decrease(hp, dist, c);
                ngoal++;
            }
        }
    } else if (key < (uint32_t)PR_N_CELLS) {
        dist[key] = 0;
        pr_heap_push_or_decrease(hp, dist, (int)key);
        ngoal++;
    }
    while (hp->n > 0) {
        int c = pr_heap_pop(hp, dist);
        uint32_t dc = dist[c];
        for (int k = 0; k < 8; k++) {
            /* a walker at n steps into c: n = c - offset(k) */
            int cx = c % PR_COLS - PR_NB_DX[k], cy = c / PR_COLS - PR_NB_DY[k];
            if (cx < 0 || cy < 0 || cx >= PR_COLS || cy >= PR_ROWS) continue;
            int n = cy * PR_COLS + cx;
            if (hp->pos[n] == -2) continue;
            if (pr_nb(n, k, jumper, 0) != c) continue; /* passability + corner rule, walker's view */
            uint32_t nd = dc + (uint32_t)pr_enter_cost(pc, c, PR_NB_COST[k], goal_id);
            if (nd >= PR_DIST_INF) continue;
            if (nd < dist[n]) {
                dist[n] = (uint16_t)nd;
                pr_heap_push_or_decrease(hp, dist, n);
            }
        }
    }
    f->key = key;
    f->jumper = (uint8_t)jumper;
    f->sig = pc->sig;
    f->valid = 1;
    pc->field_builds++;
    (void)ngoal;
}

static inline PrField *pr_field_get(PrPathCache *pc, uint32_t key, int jumper) {
    PrField *slot = NULL;
    for (int i = 0; i < PR_N_FIELDS; i++) {
        PrField *f = &pc->f[i];
        if (f->valid && f->sig == pc->sig && f->key == key && f->jumper == jumper) {
            f->stamp = ++pc->stamp;
            return f;
        }
    }
    for (int i = 0; i < PR_N_FIELDS; i++) { /* a free or stale slot, else the LRU one */
        PrField *f = &pc->f[i];
        if (!f->valid || f->sig != pc->sig) { slot = f; break; }
        if (!slot || f->stamp < slot->stamp) slot = f;
    }
    pr_field_build(pc, slot, key, jumper);
    slot->stamp = ++pc->stamp;
    return slot;
}

/* Next cell of the steepest-descent chain from c, or -1 at a goal / dead end. Ties go
 * to the first neighbour in the mover's own-frame order. */
static inline int pr_field_next(const PrPathCache *pc, const PrField *f, int c, int jumper, int team) {
    if (f->dist[c] == 0) return -1;
    uint32_t goal_id = (f->key & PR_GOAL_STRUCT) ? (f->key & ~PR_GOAL_STRUCT) : PR_NO_ID;
    int best = -1;
    uint32_t bv = 0;
    for (int k = 0; k < 8; k++) {
        int n = pr_nb(c, k, jumper, team);
        if (n < 0 || f->dist[n] == PR_DIST_INF) continue;
        uint32_t v = (uint32_t)f->dist[n] + (uint32_t)pr_enter_cost(pc, n, PR_NB_COST[k], goal_id);
        if (best < 0 || v < bv) { best = n; bv = v; }
    }
    if (best >= 0 && f->dist[c] != PR_DIST_INF && f->dist[best] >= f->dist[c]) return -1;
    return best;
}

/* Straight-segment walkability: every sampled cell passable and not occluded by a
 * structure other than `goal_id`; the start cell and `goal_cell` are exempt. */
static inline int pr_los(const PrPathCache *pc, int team, int32_t ax, int32_t ay, int32_t bx, int32_t by,
                         int jumper, uint32_t goal_id, int goal_cell) {
    int start = pr_cell_of_t(team, ax, ay);
    int64_t dx = (int64_t)bx - ax, dy = (int64_t)by - ay;
    int64_t span = PR_MAX(pr_abs64(dx), pr_abs64(dy));
    int64_t n = span / PR_LOS_STEP + 1;
    for (int64_t i = 1; i <= n; i++) {
        int32_t x = ax + (int32_t)(dx * i / n), y = ay + (int32_t)(dy * i / n);
        int c = pr_cell_of_t(team, x, y);
        if (c < 0) return 0;
        if (c == start || c == goal_cell) continue;
        if (!pr_cell_passable(c, jumper)) return 0;
        uint32_t o = pc->occ[c];
        if (o != PR_NO_ID && o != goal_id) return 0;
    }
    return 1;
}

/* Recompute entity e's waypoint toward goal point (gx, gy). `key` is the goal key (a
 * structure, or the chase block of a troop target) and `goal_cell` the chased troop's own
 * cell (-1 for structures). */
static inline void pr_plan(PrPathCache *pc, PrEntity *e, int32_t gx, int32_t gy, uint32_t key, int goal_cell) {
    int jumper = PR_UNITS[e->unit].jumps;
    int team = e->team;
    int cell = pr_cell_of_t(team, e->x, e->y);
    uint32_t goal_id = (key & PR_GOAL_STRUCT) ? (key & ~PR_GOAL_STRUCT) : PR_NO_ID;
    e->wp_cell = (int16_t)cell;
    e->wp_goal = key;
    e->wp_sig = pr_sig32(pc->sig);
    e->wp_direct = 1;
    if (cell < 0) return;
    if (pr_los(pc, team, e->x, e->y, gx, gy, jumper, goal_id, goal_cell)) return;
    PrField *f = pr_field_get(pc, key, jumper);
    if (f->dist[cell] == 0 && (key & PR_GOAL_BLOCK) && goal_cell >= 0 && goal_cell != cell) {
        /* inside the chase block but the target is not in sight: route to its own cell */
        key = (uint32_t)goal_cell;
        f = pr_field_get(pc, key, jumper);
    }
    if (f->dist[cell] == 0 || f->dist[cell] == PR_DIST_INF) return;
    int chain[PR_CHAIN_LEN];
    int n = 0, c = cell;
    while (n < PR_CHAIN_LEN) {
        int nx = pr_field_next(pc, f, c, jumper, team);
        if (nx < 0) break;
        chain[n++] = nx;
        c = nx;
        if (f->dist[nx] == 0) break;
    }
    if (n == 0) return;
    int pick = 0;
    for (int j = n - 1; j >= 1; j--) {
        if (pr_los(pc, team, e->x, e->y, pr_cell_cx(chain[j]), pr_cell_cy(chain[j]), jumper, goal_id, goal_cell)) {
            pick = j;
            break;
        }
    }
    e->wp_direct = 0;
    e->wp_x = pr_cell_cx(chain[pick]);
    e->wp_y = pr_cell_cy(chain[pick]);
}

#endif /* PR_PATH_H */
