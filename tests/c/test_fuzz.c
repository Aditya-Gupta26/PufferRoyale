/* test_fuzz.c -- snapshot validation (SPEC v0.2.1 §14.5): random byte flips of real snapshots
 * are either rejected by pr_state_check or simulate, observe, mask, render and bot-play without
 * any out-of-bounds access or undefined behaviour (run under `make asan` as well). */
#include "royale.h"
#include "pr_test.h"

#define N_SNAPS 11
#ifndef PR_FUZZ_ITERS
#define PR_FUZZ_ITERS 3000
#endif
#ifndef PR_FUZZ_REPORT
#define PR_FUZZ_REPORT 1
#endif

/* Real snapshots at interesting moments: fresh deal, mid-match with pending plays queued and a
 * Log, Arrows and a Goblin Barrel in flight, and the finished match. */
static int collect(PrState *snaps) {
    int n = 0;
    PrGame *g = t_game_decks(77, PR_DECKS[1], PR_DECKS[2], 0);
    snaps[n++] = g->st;
    PrBot b0, b1;
    pr_bot_init(&b0, PR_BOT_RANDOM, 400000, 1);
    pr_bot_init(&b1, PR_BOT_HEURISTIC, 0, 2);
    int want[] = {100, 400, 900, 1500};
    int k = 0;
    while (!g->st.over && n < N_SNAPS - 1) {
        for (int team = 0; team < 2; team++) {
            int a = pr_bot_act(team ? &b1 : &b0, &g->st, team);
            if (a > 0) pr_queue_play(&g->st, team, (a - 1) / PR_N_TILES, (a - 1) % PR_N_TILES % 18, (a - 1) % PR_N_TILES / 18);
        }
        if (k < 4 && g->st.tick >= want[k]) {
            pr_debug_spawn(g, 0, PR_CARD_THE_LOG, 9000, 20000, 1, NULL, 0);
            pr_debug_spawn(g, 1, PR_CARD_ARROWS, 9000, 20000, 1, NULL, 0);
            pr_debug_spawn(g, 1, PR_CARD_GOBLIN_BARREL, 6000, 24000, 1, NULL, 0);
            snaps[n++] = g->st; /* with pending plays queued, effects and projectiles live */
            k++;
        }
        pr_tick(g);
    }
    while (!g->st.over) pr_tick(g);
    snaps[n++] = g->st;
    free(g);
    /* SPEC §16 state (version 4): random 64-card decks, a Dagger Duchess against a Royal Chef, and
     * mid-flight Phase E mechanics -- a burrowing Miner, spawners, a death bomb on its fuse, a Poison
     * pulse, Lightning bolts pending, a Barbarian Barrel rolling, a dashing Bandit, an Inferno ramp */
    g = (PrGame *)calloc(1, sizeof(PrGame));
    pr_setup_ex(g, NULL, NULL, 91, 0, PR_TIEBREAK_ABSOLUTE, 2, 3);
    pr_bot_init(&b0, PR_BOT_RANDOM, 500000, 7);
    pr_bot_init(&b1, PR_BOT_HEURISTIC, 0, 8);
    int want2[] = {60, 300, 700, 1200, 2000};
    k = 0;
    while (!g->st.over && k < 5) {
        for (int team = 0; team < 2; team++) {
            int a = pr_bot_act(team ? &b1 : &b0, &g->st, team);
            if (a > 0) pr_queue_play(&g->st, team, (a - 1) / PR_N_TILES, (a - 1) % PR_N_TILES % 18, (a - 1) % PR_N_TILES / 18);
        }
        if (g->st.tick == want2[k] - 30) {
            int8_t order[8] = {PR_CARD_MINER, 0, 1, 2, 3, 4, 5, 6};
            pr_debug_set_hand(g, 0, order);
            g->st.elixir[0] = PR_ELIXIR_MAX;
            pr_queue_play(&g->st, 0, 0, 9, 9);
            pr_debug_spawn(g, 1, PR_CARD_WITCH, 9000, 9000, 1, NULL, 0);
            pr_debug_spawn(g, 0, PR_CARD_TOMBSTONE, 9000, 22000, 1, NULL, 0);
            pr_debug_spawn(g, 1, PR_CARD_INFERNO_TOWER, 6000, 12000, 1, NULL, 0);
            pr_debug_spawn(g, 0, PR_CARD_BANDIT, 6000, 17500, 1, NULL, 0);
            uint32_t bl[2];
            if (pr_debug_spawn(g, 1, PR_CARD_BALLOON, 12000, 20000, 1, bl, 2) > 0) {
                PrEntity *e = pr_get(&g->st, bl[0]);
                if (e) e->dmg_in = 99999; /* dies now: its bomb is on the fuse at the snapshot */
            }
        }
        if (g->st.tick == want2[k] - 2) {
            pr_debug_spawn(g, 0, PR_CARD_POISON, 9000, 9000, 1, NULL, 0);
            pr_debug_spawn(g, 1, PR_CARD_LIGHTNING, 9000, 22000, 1, NULL, 0);
            pr_debug_spawn(g, 0, PR_CARD_BARBARIAN_BARREL, 9000, 20000, 1, NULL, 0);
        }
        if (g->st.tick >= want2[k]) snaps[n++] = g->st, k++;
        pr_tick(g);
    }
    free(g);
    return n;
}

/* SPEC §18.5: a validated state never evolves into an invalid one. Returns the first failing check. */
static const char *exercise(PrGame *g) {
    const char *bad = NULL;
    static float obs[PR_OBS_SIZE];
    static char buf[PR_ANSI_BUF];
    uint8_t mask[PR_N_ACTIONS];
    PrBot bot;
    pr_bot_init(&bot, PR_BOT_HEURISTIC, 0, 5);
    for (int t = 0; t < 25; t++) {
        for (int team = 0; team < 2; team++) {
            pr_obs_write(&g->st, team, obs);
            pr_legal_mask(&g->st, team, mask);
            int a = pr_bot_act(&bot, &g->st, team);
            if (a > 0) pr_queue_play(&g->st, team, (a - 1) / PR_N_TILES, (a - 1) % PR_N_TILES % 18, (a - 1) % PR_N_TILES / 18);
        }
        pr_render_ansi(&g->st, buf, PR_ANSI_BUF);
        (void)pr_hash(&g->st);
        pr_tick(g);
        const char *why = NULL;
        if (!bad && !pr_state_check(&g->st, &why)) bad = why;
    }
    return bad;
}

static void test_real_snapshots_valid(void) {
    static PrState snaps[N_SNAPS];
    int n = collect(snaps);
    CHECK_EQ(n, N_SNAPS);
    for (int i = 0; i < n; i++) {
        const char *why = NULL;
        CHECK(pr_state_check(&snaps[i], &why));
        if (why) fprintf(stderr, "snapshot %d rejected: %s\n", i, why);
    }
    /* a valid state stays valid through real play */
    PrGame *g = t_game(78);
    PrBot b0, b1;
    pr_bot_init(&b0, PR_BOT_RANDOM, 300000, 3);
    pr_bot_init(&b1, PR_BOT_HEURISTIC, 0, 4);
    int bad = 0;
    while (!g->st.over) {
        for (int team = 0; team < 2; team++) {
            int a = pr_bot_act(team ? &b1 : &b0, &g->st, team);
            if (a > 0) pr_queue_play(&g->st, team, (a - 1) / PR_N_TILES, (a - 1) % PR_N_TILES % 18, (a - 1) % PR_N_TILES / 18);
        }
        pr_tick(g);
        bad += !pr_state_check(&g->st, NULL);
    }
    CHECK_EQ(bad, 0);
    free(g);
}

static void test_byte_flips(void) {
    static PrState snaps[N_SNAPS];
    int n = collect(snaps);
    PrGame *g = (PrGame *)calloc(1, sizeof(PrGame));
    PrRng r;
    pr_rng_seed(&r, 2026, 9);
    int accepted = 0, rejected = 0, invalidated = 0;
    const int iters = PR_FUZZ_ITERS;
    for (int it = 0; it < iters; it++) {
        PrState s = snaps[it % n];
        size_t live = offsetof(PrState, ent) + sizeof(PrEntity) * (size_t)PR_CLAMP(s.n_ent, 1, PR_MAX_ENTITIES);
        int flips = 1 + (int)pr_rng_below(&r, 4);
        for (int f = 0; f < flips; f++) {
            size_t pos;
            if (pr_rng_below(&r, 2)) { /* anywhere */
                pos = (size_t)pr_rng_below(&r, (uint32_t)sizeof(PrState));
            } else { /* inside the header and live entities, where flips are most likely accepted */
                pos = (size_t)pr_rng_below(&r, (uint32_t)live);
            }
            ((unsigned char *)&s)[pos] ^= (unsigned char)(1u << pr_rng_below(&r, 8));
        }
        if (!pr_state_check(&s, NULL)) {
            rejected++;
            continue;
        }
        accepted++;
        memset(g, 0, sizeof(*g));
        pr_state_canonicalize(&s); /* as Game.restore does */
        g->st = s;
        const char *bad = exercise(g);
        if (bad) {
            if (invalidated < PR_FUZZ_REPORT) fprintf(stderr, "  mutated snapshot %d (from %d) became invalid: %s\n", it, it % n, bad);
            invalidated++;
        }
    }
    printf("    %d mutated snapshots accepted and exercised (%d became invalid), %d rejected\n", accepted,
           invalidated, rejected);
    CHECK(accepted > 100 && rejected > 100);
    CHECK_EQ(invalidated, 0);
    free(g);
}

int main(void) {
    printf("test_fuzz\n");
    RUN(test_real_snapshots_valid);
    RUN(test_byte_flips);
    TEST_END();
}
