/* test_soak.c -- random full matches with invariants, determinism, snapshot/restore. */
#include "pr_test.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define PR_SANITIZED 1
#endif
#endif
#ifndef PR_SANITIZED
#define PR_SANITIZED 0
#endif

/* Pick and queue a random legal action for `team` with probability num/den. */
static void random_play(PrState *st, PrRng *bot, int team, uint32_t num, uint32_t den) {
    if (pr_rng_below(bot, den) >= num) return;
    static uint8_t mask[PR_N_ACTIONS];
    pr_legal_mask(st, team, mask);
    int legal[PR_N_ACTIONS], n = 0;
    for (int a = 1; a < PR_N_ACTIONS; a++)
        if (mask[a]) legal[n++] = a;
    if (n == 0) return;
    int a = legal[pr_rng_below(bot, (uint32_t)n)];
    int slot = (a - 1) / PR_N_TILES, cell = (a - 1) % PR_N_TILES;
    int err = pr_queue_play(st, team, slot, cell % PR_TILES_X, cell / PR_TILES_X);
    if (err != PR_OK) {
        CHECK_EQ(err, PR_OK); /* the mask promised it */
    }
}

static int invariants(const PrState *st) {
    int bad = 0;
    uint32_t prev = 0;
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        if (i > 0 && e->id <= prev) bad |= 1;
        prev = e->id;
        if (e->hp <= 0 || e->hp > e->max_hp) bad |= 2;
        if (!pr_in_arena(e->x, e->y)) bad |= 4;
        /* ground troops never stand in water (jumpers may; a burrowing Miner passes under the river) */
        if (e->kind == PR_KIND_TROOP && !e->flying && !PR_UNITS[e->unit].jumps && !e->burrow && pr_point_wet(e->x, e->y)) {
            if (!(bad & 8)) fprintf(stderr, "    wet: %s at (%d, %d) dash %d\n", PR_UNITS[e->unit].name, e->x, e->y, e->dash_state);
            bad |= 8;
        }
        if (e->shield < 0) bad |= 16;
    }
    for (int t = 0; t < 2; t++) {
        if (st->elixir[t] < 0 || st->elixir[t] > PR_ELIXIR_MAX) bad |= 32;
        int count[PR_N_CARDS] = {0};
        for (int k = 0; k < 4; k++) {
            count[st->hand[t][k]]++;
            count[st->queue[t][k]]++;
        }
        for (int k = 0; k < 8; k++)
            if (count[st->deck[t][k]] != 1) bad |= 64;
    }
    if (st->n_ent > PR_MAX_ENTITIES || st->n_proj > PR_MAX_PROJECTILES || st->n_fx > PR_MAX_EFFECTS) bad |= 128;
    for (int i = 0; i < st->n_proj; i++)
        if (st->proj[i].speed <= 0) bad |= 256;
    return bad;
}

/* Play one random match; returns the final hash and fills the per-tick hash trail. */
static uint64_t play_match(uint64_t seed, uint64_t bot_seed, uint32_t num, uint32_t den, uint64_t *trail,
                           int trail_len, int check_inv, PrState *final_out) {
    PrGame *g = t_game_decks(seed, NULL, NULL, PR_DEFAULT_LOCKOUT);
    PrRng bot;
    pr_rng_seed(&bot, bot_seed, 7);
    int reported = 0;
    while (!g->st.over) {
        random_play(&g->st, &bot, 0, num, den);
        random_play(&g->st, &bot, 1, num, den);
        pr_tick(g);
        if (trail && g->st.tick - 1 < trail_len) trail[g->st.tick - 1] = pr_hash(&g->st);
        if (check_inv) {
            int bad = invariants(&g->st);
            if (bad && !reported) {
                CHECK_EQ(bad, 0);
                fprintf(stderr, "  invariant 0x%x broken at tick %d (seed %llu)\n", bad, g->st.tick,
                        (unsigned long long)seed);
                reported = 1;
            }
        }
    }
    CHECK(g->st.tick <= PR_TICKS_MAX);
    CHECK(g->st.end_reason != PR_END_NONE);
    CHECK_EQ(g->st.result[0] + g->st.result[1], 0);
    if (g->st.end_reason == PR_END_DRAW) CHECK_EQ(g->st.result[0], 0);
    else CHECK(g->st.result[0] == 1 || g->st.result[0] == -1);
    uint64_t h = pr_hash(&g->st);
    if (final_out) memcpy(final_out, &g->st, sizeof(PrState));
    free(g);
    return h;
}

static void test_soak_invariants(void) {
    int matches = PR_SANITIZED ? 6 : 60;
    int reasons[6] = {0};
    long long ticks = 0, plays = 0, overflow = 0;
    static PrState fin;
    for (int m = 0; m < matches; m++) {
        /* play rates from sparse to spammy */
        uint32_t num = (uint32_t)(1 + m % 4), den = 20;
        play_match(1000 + (uint64_t)m, 5000 + (uint64_t)m, num, den, NULL, 0, 1, &fin);
        reasons[fin.end_reason]++;
        ticks += fin.tick;
        plays += fin.plays[0] + fin.plays[1];
        overflow += fin.spawn_overflow;
    }
    printf("    %d matches, %lld ticks, %lld plays, overflow %lld; ends KING %d REG %d OT %d TB %d DRAW %d\n",
           matches, ticks, plays, overflow, reasons[1], reasons[2], reasons[3], reasons[4], reasons[5]);
}

static void test_determinism_same_seed(void) {
    enum { N = 6000 };
    static uint64_t a[N], b[N];
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    uint64_t ha = play_match(42, 4242, 3, 20, a, N, 0, NULL);
    uint64_t hb = play_match(42, 4242, 3, 20, b, N, 0, NULL);
    CHECK_EQ(ha, hb);
    int diff = -1;
    for (int i = 0; i < N; i++)
        if (a[i] != b[i]) { diff = i; break; }
    CHECK_EQ(diff, -1);
    uint64_t hc = play_match(43, 4242, 3, 20, NULL, 0, 0, NULL);
    CHECK(hc != ha);
}

static void test_snapshot_restore(void) {
    PrGame *g = t_game_decks(77, NULL, NULL, PR_DEFAULT_LOCKOUT);
    PrRng bot;
    pr_rng_seed(&bot, 99, 7);
    while (g->st.tick < 900) {
        random_play(&g->st, &bot, 0, 2, 20);
        random_play(&g->st, &bot, 1, 2, 20);
        pr_tick(g);
    }
    static PrState snap;
    memcpy(&snap, &g->st, sizeof(snap));
    PrRng bot_snap = bot;
    enum { K = 1200 };
    static uint64_t h1[K], h2[K];
    for (int i = 0; i < K && !g->st.over; i++) {
        random_play(&g->st, &bot, 0, 2, 20);
        random_play(&g->st, &bot, 1, 2, 20);
        pr_tick(g);
        h1[i] = pr_hash(&g->st);
    }
    /* restore into a DIFFERENT game object whose path cache is cold / stale */
    PrGame *g2 = t_game_decks(1, PR_DECKS[2], PR_DECKS[2], 0);
    t_ticks(g2, 300);
    memcpy(&g2->st, &snap, sizeof(snap));
    bot = bot_snap;
    int diff = -1;
    for (int i = 0; i < K && !g2->st.over; i++) {
        random_play(&g2->st, &bot, 0, 2, 20);
        random_play(&g2->st, &bot, 1, 2, 20);
        pr_tick(g2);
        h2[i] = pr_hash(&g2->st);
        if (h2[i] != h1[i] && diff < 0) diff = i;
    }
    CHECK_EQ(diff, -1);
    free(g);
    free(g2);
}

static void test_path_cache_is_pure(void) {
    /* the same match with the path cache wiped every tick gives the same hashes */
    PrGame *a = t_game_decks(88, NULL, NULL, PR_DEFAULT_LOCKOUT);
    PrGame *b = t_game_decks(88, NULL, NULL, PR_DEFAULT_LOCKOUT);
    PrRng ba, bb;
    pr_rng_seed(&ba, 5, 7);
    pr_rng_seed(&bb, 5, 7);
    int diff = -1;
    for (int t = 0; t < 2500 && !a->st.over; t++) {
        random_play(&a->st, &ba, 0, 2, 20);
        random_play(&a->st, &ba, 1, 2, 20);
        random_play(&b->st, &bb, 0, 2, 20);
        random_play(&b->st, &bb, 1, 2, 20);
        memset(&b->pc, 0, sizeof(b->pc));
        pr_tick(a);
        pr_tick(b);
        if (pr_hash(&a->st) != pr_hash(&b->st) && diff < 0) diff = t;
    }
    CHECK_EQ(diff, -1);
    free(a);
    free(b);
}

static void test_capacity_overflow_is_safe(void) {
    PrGame *g = t_game_decks(9, PR_DECKS[2], PR_DECKS[2], 0);
    t_mute_all_towers(g);
    for (int i = 0; i < 40; i++) pr_debug_spawn(g, i & 1, PR_CARD_SKELETON_ARMY, 9000, i & 1 ? 12000 : 20000, 1, NULL, 0);
    CHECK_EQ(g->st.n_ent, PR_MAX_ENTITIES);
    CHECK(g->st.spawn_overflow > 0);
    t_ticks(g, 300);
    CHECK_EQ(invariants(&g->st), 0);
    free(g);
}

int main(void) {
    printf("test_soak%s\n", PR_SANITIZED ? " (sanitized: fewer matches)" : "");
    RUN(test_determinism_same_seed);
    RUN(test_snapshot_restore);
    RUN(test_path_cache_is_pure);
    RUN(test_capacity_overflow_is_safe);
    RUN(test_soak_invariants);
    TEST_END();
}
