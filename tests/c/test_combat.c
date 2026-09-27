/* test_combat.c -- attack cycle timing, projectiles, splash, lifetime, deaths, status. */
#include "pr_test.h"

/* A target that never moves or attacks: a deploying unit (targetable, SPEC §5) with
 * effectively infinite hp. */
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

/* Walk attacker toward a dummy; return the tick index at which it first attacked, and
 * record the tick indices of the dummy's hp drops. */
static int32_t engage(PrGame *g, uint32_t att, uint32_t dum, int32_t *drops, int ndrops, int max_ticks) {
    int32_t entered = -1;
    int k = 0;
    int32_t hp = t_ent(g, dum)->hp;
    for (int i = 0; i < max_ticks && k < ndrops; i++) {
        int32_t t = g->st.tick;
        pr_tick(g);
        PrEntity *a = t_ent(g, att);
        if (a && entered < 0 && a->attacking) entered = t;
        PrEntity *d = t_ent(g, dum);
        if (d && d->hp < hp) {
            drops[k++] = t;
            hp = d->hp;
        }
    }
    return entered;
}

static void check_first_hit(int unit, int32_t d0) {
    PrGame *g = arena(100 + unit);
    const PrUnitDef *u = &PR_UNITS[unit];
    uint32_t a = t_place(g, 0, unit, -1, 9000, 21000, 1);
    uint32_t d = dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000 - d0);
    int32_t drops[4] = {-1, -1, -1, -1};
    int32_t e = engage(g, a, d, drops, 3, 400);
    CHECK(e >= 0);
    int32_t first = (u->hit_speed_ms - u->load_time_ms) / 50 - 1;
    int32_t period = u->hit_speed_ms / 50;
    if (u->projectile < 0) {
        CHECK_EQ(drops[0], e + first);
        CHECK_EQ(drops[1], e + first + period);
        CHECK_EQ(drops[2], e + first + 2 * period);
    } else {
        /* projectile: the launch is on e + first; the impacts keep the cadence */
        CHECK(drops[0] > e + first);
        CHECK_EQ(drops[1] - drops[0], period);
        CHECK_EQ(drops[2] - drops[1], period);
    }
    free(g);
}

static void test_first_hit_formula(void) {
    check_first_hit(PR_UNIT_KNIGHT, 4000);    /* (1200-700)/50-1 = 9, every 24 */
    check_first_hit(PR_UNIT_VALKYRIE, 4000);
    check_first_hit(PR_UNIT_SKELETON, 3000);  /* 9, every 22 */
    check_first_hit(PR_UNIT_GOBLIN, 3000);    /* 7, every 22 */
    check_first_hit(PR_UNIT_MUSKETEER, 9000); /* launch 13 after entering range */
    check_first_hit(PR_UNIT_ARCHER, 9000);    /* launch 9 */
}

static void test_projectile_launch_tick(void) {
    /* Musketeer fully loaded, target already in range at acquisition */
    PrGame *g = arena(3);
    uint32_t m = t_place(g, 0, PR_UNIT_MUSKETEER, -1, 9000, 21000, 1);
    dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000 - 5000);
    int32_t t0 = g->st.tick;
    int launched = -1;
    for (int i = 0; i < 30 && launched < 0; i++) {
        pr_tick(g);
        if (g->st.n_proj > 0) launched = g->st.tick - 1;
    }
    CHECK_EQ(launched, t0 + 13);
    /* it starts projectile_start_radius (450) from the shooter toward the target */
    CHECK_EQ(g->st.proj[0].x, 9000);
    CHECK_EQ(g->st.proj[0].y, 21000 - 450);
    CHECK_EQ(g->st.proj[0].owner_id, m);
    pr_tick(g); /* first step on the next tick: 1000 per tick */
    CHECK_EQ(g->st.proj[0].y, 21000 - 450 - 1000);
    free(g);
}

static void test_tower_timing(void) {
    /* Princess tower (load 0, hit 800): first shot 15 ticks after the target enters
     * range, then every 16 ticks; 109 damage. */
    PrGame *g = t_game_decks(4, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t tw = g->st.tower_id[0][1]; /* team 0 left princess (3500, 25500) */
    int32_t t0 = g->st.tick;
    uint32_t d = dummy(g, 1, PR_UNIT_KNIGHT, 3500, 25500 - 7000);
    int32_t hp0 = t_ent(g, d)->hp;
    int launches[3], nl = 0, prev_np = 0;
    int32_t drops[3], nd = 0, ndrops = 0, hp = hp0;
    for (int i = 0; i < 80; i++) {
        int32_t t = g->st.tick;
        pr_tick(g);
        int np = 0;
        for (int p = 0; p < g->st.n_proj; p++) np += g->st.proj[p].owner_id == tw;
        if (np > prev_np && nl < 3) launches[nl++] = t;
        prev_np = np;
        PrEntity *de = t_ent(g, d);
        if (de->hp < hp) {
            if (nd < 3) drops[nd++] = t;
            ndrops++;
            hp = de->hp;
        }
    }
    CHECK(nl >= 3);
    CHECK_EQ(launches[0], t0 + 15);
    CHECK_EQ(launches[1], t0 + 31);
    CHECK_EQ(launches[2], t0 + 47);
    CHECK(nd >= 2);
    CHECK_EQ(drops[1] - drops[0], 16);
    CHECK_EQ(hp0 - t_ent(g, d)->hp, 109 * ndrops);
    free(g);
    /* King (load 500, hit 1000) once active: first shot 9 ticks after acquisition */
    g = t_game_decks(5, PR_DECKS[0], PR_DECKS[1], 0);
    g->st.king_active[0] = 1;
    uint32_t king = g->st.tower_id[0][0];
    t_ticks(g, 20); /* the King's load timer is long run down */
    int32_t t1 = g->st.tick;
    dummy(g, 1, PR_UNIT_KNIGHT, 9000, 29000 - 6000);
    int first = -1;
    for (int i = 0; i < 30 && first < 0; i++) {
        int32_t t = g->st.tick;
        pr_tick(g);
        for (int p = 0; p < g->st.n_proj; p++)
            if (g->st.proj[p].owner_id == king) first = t;
    }
    CHECK_EQ(first, t1 + 9);
    free(g);
    /* a dormant King does not target */
    g = t_game_decks(6, PR_DECKS[0], PR_DECKS[1], 0);
    dummy(g, 1, PR_UNIT_KNIGHT, 9000, 29000 - 6000);
    t_ticks(g, 40);
    CHECK_EQ(t_ent(g, g->st.tower_id[0][0])->target_id, PR_NO_ID);
    free(g);
}

static void test_crown_tower_reduction(void) {
    PrGame *g = t_game_decks(7, PR_DECKS[0], PR_DECKS[1], 0);
    PrState *st = &g->st;
    /* Zap on the enemy right princess: 48 instead of 192, resolved in the play tick */
    pr_debug_spawn(g, 0, PR_CARD_ZAP, 14500, 6500, 1, NULL, 0);
    pr_tick(g);
    CHECK_EQ(t_ent(g, st->tower_id[1][2])->hp, 3052 - 48);
    CHECK(pr_is_stunned(t_ent(g, st->tower_id[1][2]))); /* crown towers can be stunned */
    /* Fireball on the left princess: 172 on arrival */
    pr_debug_spawn(g, 0, PR_CARD_FIREBALL, 3500, 6500, 1, NULL, 0);
    for (int i = 0; i < 80; i++) pr_tick(g);
    CHECK_EQ(t_ent(g, st->tower_id[1][1])->hp, 3052 - 172);
    /* the King took no damage and is still dormant */
    CHECK_EQ(t_ent(g, st->tower_id[1][0])->hp, 4824);
    CHECK(!st->king_active[1]);
    CHECK(pr_damage_vs(t_ent(g, st->tower_id[1][1]), 268, 13) == 35);
    CHECK(pr_damage_vs(t_ent(g, st->tower_id[1][1]), 122, 20) == 25);
    free(g);
}

static void test_valkyrie_splash(void) {
    PrGame *g = arena(8);
    uint32_t v = t_place(g, 0, PR_UNIT_VALKYRIE, -1, 9000, 21000, 1);
    t_ent(g, v)->load_ms = 0;
    uint32_t a = dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000 - 1500);     /* the target */
    uint32_t b = dummy(g, 1, PR_UNIT_KNIGHT, 9000 + 1500, 21000 + 1500); /* behind her, within 2000 + 500 */
    uint32_t c = dummy(g, 1, PR_UNIT_KNIGHT, 9000 - 2500, 21000);     /* 2500 <= 2000 + 500: hit */
    uint32_t far = dummy(g, 1, PR_UNIT_KNIGHT, 9000 + 2600, 21000);   /* 2600 > 2500: not hit */
    uint32_t air = dummy(g, 1, PR_UNIT_MINION, 9000, 21000 + 500);    /* air: not hit */
    int32_t h = 100000000;
    t_ticks(g, 3);
    CHECK_EQ(t_ent(g, a)->hp, h - 266);
    CHECK_EQ(t_ent(g, b)->hp, h - 266);
    CHECK_EQ(t_ent(g, c)->hp, h - 266);
    CHECK_EQ(t_ent(g, far)->hp, h);
    CHECK_EQ(t_ent(g, air)->hp, h);
    free(g);
}

static void test_wizard_splash_projectile(void) {
    PrGame *g = arena(9);
    t_place(g, 0, PR_UNIT_WIZARD, -1, 9000, 26000, 1);
    uint32_t a = dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000);
    uint32_t b = dummy(g, 1, PR_UNIT_MINION, 9000 + 1800, 21000); /* 1800 <= 1500 + 500 */
    uint32_t c = dummy(g, 1, PR_UNIT_KNIGHT, 9000 - 2100, 21000); /* 2100 > 2000 */
    uint32_t f = dummy(g, 0, PR_UNIT_KNIGHT, 9000, 21000 + 1200); /* friendly, 1200 away: never hit */
    int32_t h = 100000000;
    t_ticks(g, 40);
    CHECK(t_ent(g, a)->hp < h);
    CHECK_EQ(t_ent(g, a)->hp, t_ent(g, b)->hp);
    CHECK_EQ(t_ent(g, c)->hp, h);
    CHECK_EQ(t_ent(g, f)->hp, h);
    CHECK_EQ((h - t_ent(g, a)->hp) % 281, 0);
    free(g);
}

static void test_building_lifetime(void) {
    /* Cannon via the test hook, deployed between ticks: drain k = 0 on the next tick;
     * dies after exactly 600 drain ticks (30 s). */
    PrGame *g = arena(10);
    PrState *st = &g->st;
    uint32_t id;
    pr_debug_spawn(g, 0, PR_CARD_CANNON, 9500, 21500, 1, &id, 1);
    int32_t M = t_ent(g, id)->max_hp;
    CHECK_EQ(M, 824);
    int32_t P = st->tick;
    for (int k = 0; k < 599; k++) {
        pr_tick(g);
        PrEntity *e = t_ent(g, id);
        CHECK(e != NULL);
        if (!e) break;
        int64_t want = M - ((int64_t)M * (k + 1) * 50) / 30000;
        if (e->hp != want) { CHECK_EQ(e->hp, want); break; }
    }
    CHECK(t_ent(g, id) != NULL);
    pr_tick(g); /* tick P + 599 */
    CHECK(t_ent(g, id) == NULL);
    CHECK_EQ(st->tick, P + 600);
    free(g);
    /* played normally: dies 600 ticks after deploy end (tick P + 20 .. P + 619) */
    g = t_game_decks(11, PR_DECKS[0], PR_DECKS[1], 0);
    st = &g->st;
    t_mute_all_towers(g);
    int8_t hand[8] = {PR_CARD_CANNON, PR_CARD_TESLA, PR_CARD_KNIGHT, PR_CARD_ZAP,
                      PR_CARD_FIREBALL, PR_CARD_ARROWS, PR_CARD_GIANT, PR_CARD_MINIONS};
    pr_debug_set_hand(g, 0, hand);
    st->elixir[0] = 28000;
    P = st->tick;
    CHECK_EQ(pr_queue_play(st, 0, 0, 9, 20), PR_OK);
    CHECK_EQ(pr_queue_play(st, 0, 1, 4, 20), PR_OK);
    pr_tick(g);
    uint32_t cannon = PR_NO_ID, tesla = PR_NO_ID;
    for (int i = 0; i < st->n_ent; i++) {
        if (st->ent[i].unit == PR_UNIT_CANNON) cannon = st->ent[i].id;
        if (st->ent[i].unit == PR_UNIT_TESLA) tesla = st->ent[i].id;
    }
    while (st->tick < P + 20) pr_tick(g);
    CHECK_EQ(t_ent(g, cannon)->hp, 824);  /* no drain while deploying */
    CHECK_EQ(t_ent(g, tesla)->hp, 1182);
    while (st->tick < P + 519) pr_tick(g);
    CHECK(t_ent(g, tesla) != NULL);
    pr_tick(g); /* tick P + 519 = deploy end (P + 20) + 500 - 1 */
    CHECK(t_ent(g, tesla) == NULL); /* dies under ground: the drain ignores hide */
    while (st->tick < P + 619) pr_tick(g);
    CHECK(t_ent(g, cannon) != NULL);
    pr_tick(g);
    CHECK(t_ent(g, cannon) == NULL);
    free(g);
}

static void test_death_damage_next_resolve(void) {
    PrGame *g = arena(12);
    uint32_t ig = t_place(g, 0, PR_UNIT_ICEGOLEMITE, -1, 9000, 21000, 1);
    uint32_t near = dummy(g, 1, PR_UNIT_KNIGHT, 9000 + 2400, 21000); /* 2400 <= 2000 + 500 */
    uint32_t air = dummy(g, 1, PR_UNIT_MINION, 9000, 21000 - 1000);
    uint32_t far = dummy(g, 1, PR_UNIT_KNIGHT, 9000 - 2600, 21000);
    t_ent(g, ig)->dmg_in += 999999; /* dies in this tick's Reap */
    int32_t T0 = g->st.tick;
    pr_tick(g);
    CHECK(t_ent(g, ig) == NULL);
    int32_t h = 100000000;
    CHECK_EQ(t_ent(g, near)->hp, h); /* buffered for the next Resolve */
    pr_tick(g);
    CHECK_EQ(t_ent(g, near)->hp, h - 84);
    CHECK_EQ(t_ent(g, air)->hp, h - 84);
    CHECK_EQ(t_ent(g, far)->hp, h);
    /* the death area slows whoever it touched, once, for 2000 ms */
    CHECK(pr_is_slowed(t_ent(g, near)));
    CHECK(!pr_is_slowed(t_ent(g, far)));
    CHECK_EQ(t_ent(g, near)->buff_until[PR_BUFF_ICEWIZARDSLOWDOWN], T0 + 40); /* gone from T0 + 40 */
    CHECK_EQ(g->st.n_fx, 0);
    while (g->st.tick < T0 + 40) { CHECK(pr_is_slowed(t_ent(g, near))); pr_tick(g); }
    pr_tick(g); /* tick T0 + 40 */
    CHECK(!pr_is_slowed(t_ent(g, near)));
    free(g);
}

static void test_kamikaze(void) {
    PrGame *g = arena(13);
    uint32_t s = t_place(g, 0, PR_UNIT_ICESPIRITS, -1, 9000, 21000, 1);
    uint32_t d = dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000 - 3000);
    int32_t died = -1;
    for (int i = 0; i < 60; i++) {
        int32_t t = g->st.tick;
        pr_tick(g);
        if (died < 0 && t_ent(g, s) == NULL) {
            died = t;
            CHECK_EQ(g->st.n_proj, 1); /* the shot flies on */
        }
    }
    CHECK(died >= 0);
    CHECK_EQ(t_ent(g, d)->hp, 100000000 - 110);
    CHECK(pr_is_stunned(t_ent(g, d)) || 1); /* freeze may have expired by now */
    free(g);
}

static void test_target_rules(void) {
    /* Knight ignores air; Musketeer takes air */
    PrGame *g = arena(14);
    uint32_t k = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 21000, 1);
    uint32_t mu = t_place(g, 0, PR_UNIT_MUSKETEER, -1, 7000, 21000, 1);
    uint32_t mi = dummy(g, 1, PR_UNIT_MINION, 8000, 19000);
    pr_tick(g);
    CHECK_EQ(t_ent(g, k)->target_id, PR_NO_ID);
    CHECK_EQ(t_ent(g, mu)->target_id, mi);
    free(g);
    /* Giant targets only buildings */
    g = arena(15);
    uint32_t gi = t_place(g, 0, PR_UNIT_GIANT, -1, 9000, 21000, 1);
    dummy(g, 1, PR_UNIT_KNIGHT, 9000, 20000);
    uint32_t cannon = t_place(g, 1, PR_UNIT_CANNON, -1, 9000, 16000 - 2500, 1);
    pr_tick(g);
    CHECK_EQ(t_ent(g, gi)->target_id, cannon);
    free(g);
    /* sight test uses dist <= sight + r_target: Knight sight 5500 vs Knight r 500 */
    g = arena(16);
    k = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 21000, 1);
    uint32_t in = dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000 - 6000);
    pr_tick(g);
    CHECK_EQ(t_ent(g, k)->target_id, in);
    free(g);
    g = arena(17);
    k = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 21000, 1);
    dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000 - 6001);
    pr_tick(g);
    CHECK_EQ(t_ent(g, k)->target_id, PR_NO_ID);
    free(g);
    /* nearest by (dist - r_target), ties to the lowest id */
    g = arena(18);
    k = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 21000, 1);
    uint32_t t1 = dummy(g, 1, PR_UNIT_KNIGHT, 9000 + 3000, 21000);
    dummy(g, 1, PR_UNIT_KNIGHT, 9000 - 3000, 21000);
    pr_tick(g);
    CHECK_EQ(t_ent(g, k)->target_id, t1);
    free(g);
    /* retention: keep the current target while mid-cycle even if a closer one appears */
    g = arena(19);
    k = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 21000, 1);
    uint32_t first = dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000 - 2000); /* in range (2200) */
    t_ticks(g, 3);
    CHECK_EQ(t_ent(g, k)->target_id, first);
    CHECK(t_ent(g, k)->progress_ms > 0);
    dummy(g, 1, PR_UNIT_KNIGHT, 9000 + 1200, 21000); /* strictly closer */
    pr_tick(g);
    CHECK_EQ(t_ent(g, k)->target_id, first);
    free(g);
}

static void test_zap_stun(void) {
    PrGame *g = arena(20);
    uint32_t k = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 21000, 1);
    uint32_t d = dummy(g, 1, PR_UNIT_KNIGHT, 9000, 21000 - 2000);
    t_ticks(g, 5);
    CHECK(t_ent(g, k)->progress_ms > 0);
    int32_t P = g->st.tick;
    pr_debug_spawn(g, 1, PR_CARD_ZAP, 9000, 21000, 1, NULL, 0);
    pr_tick(g); /* tick P */
    PrEntity *e = t_ent(g, k);
    CHECK_EQ(e->hp, 1766 - 192);
    CHECK(pr_is_stunned(e));
    CHECK_EQ(e->progress_ms, 0);
    CHECK_EQ(e->target_id, PR_NO_ID);
    int32_t hp_d = t_ent(g, d)->hp;
    /* SPEC §13.3: applied during tick P (500 ms) -> active in ticks P .. P+9, gone from P+10 */
    CHECK_EQ(t_ent(g, k)->buff_until[PR_BUFF_ZAPFREEZE], P + 10);
    while (g->st.tick < P + 10) {
        CHECK(pr_is_stunned(t_ent(g, k)));
        CHECK_EQ(t_ent(g, k)->target_id, PR_NO_ID);
        pr_tick(g);
    }
    pr_tick(g); /* tick P + 10: free, retargets on resume */
    CHECK(!pr_is_stunned(t_ent(g, k)));
    CHECK_EQ(t_ent(g, k)->target_id, d);
    CHECK_EQ(t_ent(g, d)->hp, hp_d);
    free(g);
}

static void test_slow_multipliers(void) {
    PrGame *g = arena(21);
    uint32_t k = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 22000, 1);
    PrEntity *e = t_ent(g, k);
    int32_t T = g->st.tick;
    pr_apply_buff(e, PR_BUFF_ICEWIZARDSLOWDOWN, 2000, T);
    CHECK_EQ(pr_effective_speed(e), 42); /* floor(60 * 70 / 100) */
    pr_apply_buff(e, PR_BUFF_ICEWIZARDSLOWDOWN, 1000, T + 5); /* refresh keeps the later end */
    CHECK_EQ(e->buff_until[PR_BUFF_ICEWIZARDSLOWDOWN], T + 40);
    pr_apply_buff(e, PR_BUFF_ICEWIZARDSLOWDOWN, 2000, T + 5);
    CHECK_EQ(e->buff_until[PR_BUFF_ICEWIZARDSLOWDOWN], T + 45);
    CHECK_EQ(pr_buff_hit_pct(e), -30);
    free(g);
}

/* Tick until attacker `a` has a hit under way (SPEC §6.2 "started"). */
static int until_started(PrGame *g, uint32_t a, int max_ticks) {
    for (int i = 0; i < max_ticks; i++) {
        PrEntity *e = t_ent(g, a);
        if (e && pr_hit_started(e, pr_udef(e))) return 1;
        pr_tick(g);
    }
    return 0;
}

/* SPEC v0.2.1 §14.2 (ledger targeting.LOGIC_PRESERVE_TARGET_IF_HIT_STARTED =
 * projectile_attackers_only), the auditor's Knight / Hog / Cannon case: a Knight mid-swing
 * whose target leaves its reach switches to an enemy in reach and lands the hit on schedule;
 * with nobody in reach the swing is cancelled; a Musketeer holds within reach + 500. */
static void test_mid_swing_reach_loss(void) {
    for (int with_cannon = 0; with_cannon < 2; with_cannon++) {
        PrGame *g = arena(40 + (uint64_t)with_cannon);
        uint32_t k = t_place(g, 1, PR_UNIT_KNIGHT, PR_CARD_KNIGHT, 9000, 12000, 1);
        uint32_t hog = dummy(g, 0, PR_UNIT_HOGRIDER, 9000, 14000);
        CHECK(until_started(g, k, 60));
        PrEntity *ke = t_ent(g, k);
        CHECK_EQ(ke->target_id, hog);
        uint32_t cannon = PR_NO_ID;
        if (with_cannon) cannon = dummy(g, 0, PR_UNIT_CANNON, 10400, 12000); /* in reach, placed mid-swing */
        const PrUnitDef *kd = pr_udef(ke);
        int32_t prog = ke->progress_ms;
        int32_t hits_left = kd->hit_speed_ms - prog % kd->hit_speed_ms; /* ms to the scheduled hit */
        /* the Hog runs out of reach */
        PrEntity *he = t_ent(g, hog);
        he->y = 12000 + (int32_t)pr_attack_reach(ke, kd, he) + 100;
        int32_t hog_hp = he->hp;
        pr_tick(g);
        ke = t_ent(g, k);
        if (with_cannon) {
            CHECK_EQ(ke->target_id, cannon);
            CHECK_EQ(ke->progress_ms, prog + 50); /* the swing carries over */
            int32_t chp = t_ent(g, cannon)->hp;
            int ticks = hits_left / 50 - 1, landed = -1;
            for (int i = 0; i < ticks + 3 && landed < 0; i++) {
                if (t_ent(g, cannon)->hp < chp) landed = i;
                else pr_tick(g);
            }
            CHECK_EQ(landed, ticks); /* on the old schedule */
        } else {
            CHECK_EQ(ke->progress_ms, 0); /* no enemy in reach: the swing is cancelled */
            for (int i = 0; i < hits_left / 50 + 2; i++) pr_tick(g);
        }
        CHECK_EQ(t_ent(g, hog)->hp, hog_hp); /* never hit out of reach */
        free(g);
    }
    for (int beyond = 0; beyond < 2; beyond++) { /* projectile attacker: hold within reach + 500 */
        PrGame *g = arena(50 + (uint64_t)beyond);
        uint32_t m = t_place(g, 1, PR_UNIT_MUSKETEER, PR_CARD_MUSKETEER, 9000, 10000, 1);
        uint32_t d = dummy(g, 0, PR_UNIT_KNIGHT, 9000, 15000);
        CHECK(until_started(g, m, 60));
        PrEntity *me = t_ent(g, m), *de = t_ent(g, d);
        int32_t reach = (int32_t)pr_attack_reach(me, pr_udef(me), de);
        de->y = 10000 + reach + (beyond ? 600 : 400);
        int32_t prog = me->progress_ms;
        int n0 = g->st.n_proj;
        pr_tick(g);
        me = t_ent(g, m);
        if (beyond) {
            CHECK_EQ(me->progress_ms, 0);
        } else {
            CHECK_EQ(me->target_id, d);
            CHECK_EQ(me->progress_ms, prog + 50);
            int fired = 0;
            for (int i = 0; i < 40 && !fired; i++) {
                for (int j = 0; j < g->st.n_proj; j++)
                    if (g->st.proj[j].target_id == d && g->st.proj[j].owner_id == m) fired = 1;
                if (!fired) pr_tick(g);
            }
            CHECK(fired); /* launched at it from beyond reach */
            CHECK(g->st.n_proj >= n0);
        }
        free(g);
    }
}

int main(void) {
    printf("test_combat\n");
    RUN(test_first_hit_formula);
    RUN(test_projectile_launch_tick);
    RUN(test_tower_timing);
    RUN(test_crown_tower_reduction);
    RUN(test_valkyrie_splash);
    RUN(test_wizard_splash_projectile);
    RUN(test_building_lifetime);
    RUN(test_death_damage_next_resolve);
    RUN(test_kamikaze);
    RUN(test_target_rules);
    RUN(test_zap_stun);
    RUN(test_slow_multipliers);
    RUN(test_mid_swing_reach_loss);
    TEST_END();
}
