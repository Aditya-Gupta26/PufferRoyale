/*
 * pr_test.h -- a tiny test harness for the PufferRoyale C unit tests (builder-owned).
 */
#ifndef PR_TEST_H
#define PR_TEST_H

#include <stdio.h>
#include <stdlib.h>

#include "pr_engine.h"

static int pr_t_fail = 0, pr_t_checks = 0;

/* Referencing the counters keeps gcc's -Wunused-variable quiet in files that include this
 * header without running checks (bench.c). */
static inline int pr_t_failures(void) { return pr_t_fail + 0 * pr_t_checks; }

#define CHECK(cond)                                                                  \
    do {                                                                             \
        pr_t_checks++;                                                               \
        if (!(cond)) {                                                               \
            pr_t_fail++;                                                             \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                            \
    } while (0)

#define CHECK_EQ(a, b)                                                                            \
    do {                                                                                          \
        long long a_ = (long long)(a), b_ = (long long)(b);                                       \
        pr_t_checks++;                                                                            \
        if (a_ != b_) {                                                                           \
            pr_t_fail++;                                                                          \
            fprintf(stderr, "%s:%d: CHECK_EQ failed: %s == %s (%lld vs %lld)\n", __FILE__, __LINE__, \
                    #a, #b, a_, b_);                                                              \
        }                                                                                         \
    } while (0)

#define RUN(fn)                                                               \
    do {                                                                      \
        int f0_ = pr_t_fail;                                                  \
        fn();                                                                 \
        printf("  %-56s %s\n", #fn, pr_t_fail == f0_ ? "ok" : "FAIL");        \
    } while (0)

#define TEST_END()                                                         \
    do {                                                                   \
        printf("%d checks, %d failures\n", pr_t_checks, pr_t_fail);        \
        return pr_t_fail ? 1 : 0;                                          \
    } while (0)

/* A fresh game (hog26 vs giant unless decks given), lockout 0 unless stated. */
static inline PrGame *t_game_decks(uint64_t seed, const int8_t *d0, const int8_t *d1, int lockout) {
    PrGame *g = (PrGame *)calloc(1, sizeof(PrGame));
    if (!g) { fprintf(stderr, "out of memory\n"); exit(2); }
    pr_setup(g, d0, d1, seed, lockout, PR_TIEBREAK_ABSOLUTE);
    return g;
}

static inline PrGame *t_game(uint64_t seed) { return t_game_decks(seed, PR_DECKS[0], PR_DECKS[1], PR_DEFAULT_LOCKOUT); }

static inline PrEntity *t_ent(PrGame *g, uint32_t id) { return pr_get(&g->st, id); }

/* Place one unit directly (no formation), optionally already deployed. */
static inline uint32_t t_place(PrGame *g, int team, int unit, int card, int32_t x, int32_t y, int deployed) {
    PrEntity *e = pr_new_entity(&g->st, unit, team, card, x, y);
    if (!e) return PR_NO_ID;
    if (deployed) {
        e->load_ms = PR_MAX(0, PR_UNITS[unit].load_time_ms - e->deploy_ms);
        e->deploy_ms = 0;
    }
    return e->id;
}

static inline void t_ticks(PrGame *g, int n) {
    for (int i = 0; i < n; i++) pr_tick(g);
}

/* Remove every non-tower entity (for clean scenarios). */
static inline void t_clear_units(PrGame *g) {
    for (int i = 0; i < g->st.n_ent; i++)
        if (g->st.ent[i].kind != PR_KIND_TOWER) g->st.ent[i].hp = 0;
    pr_compact_entities(&g->st);
    g->st.n_proj = 0;
    g->st.n_fx = 0;
}

/* Make a crown tower inert for scenario tests: it neither targets nor is targeted. */
static inline void t_mute_tower(PrGame *g, int team, int idx) {
    PrEntity *t = pr_get(&g->st, g->st.tower_id[team][idx]);
    if (t) t->hide_state = PR_HIDE_HIDDEN;
}

static inline void t_mute_all_towers(PrGame *g) {
    for (int t = 0; t < 2; t++)
        for (int i = 0; i < 3; i++) t_mute_tower(g, t, i);
}

#endif /* PR_TEST_H */
