/* test_special.c -- Phase-B mechanics: spells, Tesla hide, Prince charge, river hop. */
#include "pr_test.h"

static uint32_t dummy(PrGame *g, int team, int unit, int32_t x, int32_t y) {
    uint32_t id = t_place(g, team, unit, -1, x, y, 0);
    PrEntity *e = t_ent(g, id);
    e->deploy_ms = 100000000;
    e->hp = e->max_hp = 100000000;
    return id;
}

static PrGame *arena(uint64_t seed) {
    PrGame *g = t_game_decks(seed, PR_DECKS[0], PR_DECKS[1], 0);
    t_mute_all_towers(g);
    return g;
}

static void test_fireball_timing_and_knockback(void) {
    PrGame *g = arena(1);
    int32_t tx = 9000, ty = 11000; /* 18000 from team 0's King: lands on the 30th Projectile phase */
    uint32_t k = dummy(g, 1, PR_UNIT_KNIGHT, tx + 1000, ty);
    uint32_t gi = dummy(g, 1, PR_UNIT_GIANT, tx - 1000, ty);
    uint32_t mi = dummy(g, 1, PR_UNIT_MINION, tx, ty + 1500);
    int32_t P = g->st.tick;
    pr_debug_spawn(g, 0, PR_CARD_FIREBALL, tx, ty, 1, NULL, 0);
    int32_t landed = -1;
    for (int i = 0; i < 40 && landed < 0; i++) {
        int32_t t = g->st.tick;
        pr_tick(g);
        if (t_ent(g, k)->hp < 100000000) landed = t;
    }
    CHECK_EQ(landed, P + 29);
    CHECK_EQ(t_ent(g, k)->hp, 100000000 - 688);
    CHECK_EQ(t_ent(g, gi)->hp, 100000000 - 688);
    CHECK_EQ(t_ent(g, mi)->hp, 100000000 - 688);
    CHECK_EQ(t_ent(g, k)->x, tx + 2000);  /* pushed 1000 radially */
    CHECK_EQ(t_ent(g, k)->y, ty);
    CHECK_EQ(t_ent(g, gi)->x, tx - 1000); /* ignore_pushback */
    CHECK_EQ(t_ent(g, mi)->y, ty + 2500); /* air troops are pushed too */
    free(g);
}

static void test_arrows_waves(void) {
    PrGame *g = arena(2);
    int32_t tx = 9000, ty = 12000; /* 17000 from the King: first wave on phase ceil(17000/1100) = 16 */
    uint32_t c = dummy(g, 1, PR_UNIT_CANNON, tx + 3500 + 600, ty); /* edge-inclusive: 4100 */
    uint32_t far = dummy(g, 1, PR_UNIT_CANNON, tx - 4101 - 0, ty);
    int32_t P = g->st.tick, hits[4], nh = 0, hp = 100000000;
    pr_debug_spawn(g, 0, PR_CARD_ARROWS, tx, ty, 1, NULL, 0);
    for (int i = 0; i < 40; i++) {
        int32_t t = g->st.tick;
        pr_tick(g);
        PrEntity *e = t_ent(g, c);
        if (e->hp < hp) {
            if (nh < 4) hits[nh] = t;
            nh++;
            CHECK_EQ(hp - e->hp, 122);
            hp = e->hp;
        }
    }
    CHECK_EQ(nh, 3);
    CHECK_EQ(hits[0], P + 15);
    CHECK_EQ(hits[1], P + 19);
    CHECK_EQ(hits[2], P + 23);
    CHECK_EQ(t_ent(g, far)->hp, 100000000);
    CHECK_EQ(g->st.n_fx, 0);
    free(g);
}

static void test_log_roll(void) {
    PrGame *g = arena(3);
    /* team 0 casts at (9000, 22000): rolls toward -y, final centre 11900 */
    uint32_t near = dummy(g, 1, PR_UNIT_CANNON, 9000, 11900 - 600 - 600 + 1); /* touches */
    uint32_t beyond = dummy(g, 1, PR_UNIT_CANNON, 9000, 11900 - 600 - 600 - 1);
    uint32_t side_in = dummy(g, 1, PR_UNIT_KNIGHT, 9000 + 1950 + 500, 15000 - 2000);
    uint32_t side_out = dummy(g, 1, PR_UNIT_KNIGHT, 9000 - 1950 - 501, 15000 - 2000);
    uint32_t air = dummy(g, 1, PR_UNIT_MINION, 9000, 16000);
    uint32_t heavy = dummy(g, 1, PR_UNIT_GIANT, 7000, 19000);
    int32_t gy = t_ent(g, heavy)->y;
    pr_debug_spawn(g, 0, PR_CARD_THE_LOG, 9000, 22000, 1, NULL, 0);
    t_ticks(g, 60);
    CHECK_EQ(t_ent(g, near)->hp, 100000000 - 268);
    CHECK_EQ(t_ent(g, beyond)->hp, 100000000);
    CHECK_EQ(t_ent(g, side_in)->hp, 100000000 - 268);
    CHECK_EQ(t_ent(g, side_out)->hp, 100000000);
    CHECK_EQ(t_ent(g, air)->hp, 100000000);
    CHECK_EQ(t_ent(g, heavy)->hp, 100000000 - 268); /* hit exactly once */
    CHECK_EQ(t_ent(g, heavy)->y, gy - 700);          /* pushback_all: the Giant too */
    CHECK_EQ(t_ent(g, side_in)->y, 13000 - 700);
    CHECK_EQ(g->st.n_fx, 0);
    free(g);
}

static void test_goblin_barrel(void) {
    PrGame *g = arena(4);
    int32_t P = g->st.tick;
    pr_debug_spawn(g, 0, PR_CARD_GOBLIN_BARREL, 9000, 13000, 1, NULL, 0); /* 16000 at 400/tick: 40 */
    int32_t seen = -1;
    for (int i = 0; i < 60 && seen < 0; i++) {
        int32_t t = g->st.tick;
        pr_tick(g);
        int n = 0;
        for (int j = 0; j < g->st.n_ent; j++) n += g->st.ent[j].unit == PR_UNIT_GOBLIN;
        if (n) {
            CHECK_EQ(n, 3);
            seen = t;
        }
    }
    CHECK_EQ(seen, P + 39 + 1); /* lands on phase 40 (tick P+39), goblins committed in P+40's Spawn */
    int k = 0;
    for (int j = 0; j < g->st.n_ent; j++) {
        PrEntity *e = &g->st.ent[j];
        if (e->unit != PR_UNIT_GOBLIN) continue;
        CHECK_EQ(e->deploy_ms, 1100);
        CHECK_EQ(e->card, PR_CARD_GOBLIN_BARREL);
        int32_t ox = PR_FORMATION_OFFSETS[PR_FORMATIONS[PR_FORMATION_N3].first + k][0];
        int32_t oy = PR_FORMATION_OFFSETS[PR_FORMATIONS[PR_FORMATION_N3].first + k][1];
        CHECK_EQ(e->x, 9000 + ox);
        CHECK_EQ(e->y, 13000 + oy);
        k++;
    }
    free(g);
}

static void test_tesla_hide_cycle(void) {
    PrGame *g = arena(5);
    uint32_t t = t_place(g, 0, PR_UNIT_TESLA, -1, 9500, 20500, 0);
    t_ticks(g, 20); /* the creation tick does not count: deployed after the 21st */
    CHECK_EQ(t_ent(g, t)->deploy_ms, 50);
    CHECK_EQ(t_ent(g, t)->hide_state, PR_HIDE_UP); /* visible while deploying */
    pr_tick(g);
    CHECK_EQ(t_ent(g, t)->deploy_ms, 0);
    CHECK_EQ(t_ent(g, t)->hide_state, PR_HIDE_HIDDEN);
    /* hidden: immune to damage (except the drain), untargetable */
    int32_t hp = t_ent(g, t)->hp;
    t_ent(g, t)->dmg_in += 500;
    pr_tick(g);
    CHECK(t_ent(g, t)->hp >= hp - 2);
    uint32_t k = dummy(g, 1, PR_UNIT_KNIGHT, 9500, 20500 - 5900); /* inside sight 5500 + 500 */
    pr_tick(g);
    CHECK_EQ(t_ent(g, t)->hide_state, PR_HIDE_RISING);
    int n = 1;
    while (t_ent(g, t)->hide_state == PR_HIDE_RISING && n < 40) { pr_tick(g); n++; }
    CHECK_EQ(n, 17); /* rising for 16 ticks (up_time 800 ms), up in the 17th */
    CHECK_EQ(t_ent(g, t)->target_id, k); /* targets on the tick it is up */
    /* remove the enemy: it re-hides after hide_time (800 ms) without a target */
    t_ent(g, k)->hp = 0;
    pr_compact_entities(&g->st);
    int m = 0;
    while (t_ent(g, t)->hide_state == PR_HIDE_UP && m < 40) { pr_tick(g); m++; }
    CHECK(m >= 15 && m <= 17);
    CHECK_EQ(t_ent(g, t)->hide_state, PR_HIDE_HIDDEN);
    free(g);
}

static void test_prince_charge(void) {
    PrGame *g = arena(6);
    uint32_t p = t_place(g, 0, PR_UNIT_PRINCE, -1, 9000, 26000, 1);
    uint32_t d = dummy(g, 1, PR_UNIT_KNIGHT, 9000, 26000 - 6500); /* sight 5500 + 500: not yet */
    int charged_at = -1;
    for (int i = 0; i < 120; i++) {
        pr_tick(g);
        PrEntity *e = t_ent(g, p);
        if (charged_at < 0 && e->charged) charged_at = i + 1;
        if (t_ent(g, d)->hp < 100000000) break;
    }
    CHECK_EQ(charged_at, 42); /* 42 x 60 >= 2500 */
    CHECK_EQ(t_ent(g, d)->hp, 100000000 - 783);
    CHECK(!t_ent(g, p)->charged);
    free(g);
}

static void test_hog_hops_river(void) {
    PrGame *g = t_game_decks(7, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t h = t_place(g, 0, PR_UNIT_HOGRIDER, -1, 8500, 18500, 1);
    int wet = 0, attacked = 0;
    for (int i = 0; i < 200 && !attacked; i++) {
        pr_tick(g);
        PrEntity *e = t_ent(g, h);
        if (!e) break;
        if (pr_point_wet(e->x, e->y)) { wet++; CHECK(e->jumping); }
        attacked = e->attacking;
    }
    CHECK(attacked);
    CHECK(wet > 0); /* crossed the water off the bridges */
    free(g);
}

/* SPEC v0.2.1 §14.3: the Log hits every ground enemy on its path exactly once, however many
 * there are (the hit memory covers the whole entity pool), also when entities in front of or
 * behind it die mid-roll and the pool is compacted under it. */
static void test_log_hits_each_once_at_scale(void) {
    PrGame *g = arena(9);
    uint32_t ids[200];
    int n = 0;
    for (int row = 0; row < 20; row++)      /* 200 static Cannons in the roll corridor */
        for (int col = 0; col < 10; col++)
            ids[n++] = dummy(g, 1, PR_UNIT_CANNON, 7300 + col * 380, 20500 - row * 450);
    uint32_t doomed[20];
    for (int k = 0; k < 20; k++) doomed[k] = dummy(g, 1, PR_UNIT_KNIGHT, 1000 + k * 100, 3000); /* off path */
    pr_debug_spawn(g, 0, PR_CARD_THE_LOG, 9000, 22000, 1, NULL, 0);
    for (int t = 0; t < 60; t++) {
        if (t % 7 == 3 && t / 7 < 20) t_ent(g, doomed[t / 7])->hp = 0; /* deaths -> compaction mid-roll */
        pr_tick(g);
    }
    int once = 0, other = 0;
    for (int i = 0; i < n; i++) {
        int32_t lost = 100000000 - t_ent(g, ids[i])->hp;
        lost -= (int32_t)t_ent(g, ids[i])->life_ticks * 0; /* deploying dummies do not drain */
        if (lost == 268) once++;
        else other++;
    }
    CHECK_EQ(once, n);
    CHECK_EQ(other, 0);
    free(g);
}

int main(void) {
    printf("test_special\n");
    RUN(test_fireball_timing_and_knockback);
    RUN(test_arrows_waves);
    RUN(test_log_roll);
    RUN(test_log_hits_each_once_at_scale);
    RUN(test_goblin_barrel);
    RUN(test_tesla_hide_cycle);
    RUN(test_prince_charge);
    RUN(test_hog_hops_river);
    TEST_END();
}
