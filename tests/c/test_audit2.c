/* test_audit2.c -- the second audit's engine items (SPEC §18, v0.4.1): Royal Chef serving order,
 * crown-tower tie-break, bot seat symmetry and spell valuation, the rolling-spell arena edge, the
 * path-occluder test, and snapshot-validation bounds. (Dash immunity: tests/c/test_cards_v3.c.) */
#include "royale.h"
#include "pr_test.h"

static uint32_t spawn1(PrGame *g, int team, int card, int32_t x, int32_t y, int deployed) {
    uint32_t ids[16];
    int n = pr_debug_spawn(g, team, card, x, y, deployed, ids, 16);
    CHECK(n >= 1);
    return n >= 1 ? ids[0] : PR_NO_ID;
}

/* own-frame point -> engine point of `team` */
static int32_t ex_(int team, int32_t ox) { return team == 0 ? ox : PR_ARENA_W - ox; }
static int32_t ey_(int team, int32_t oy) { return team == 0 ? oy : PR_ARENA_H - oy; }

/* M1 / §18.2: a Golem within reach of both Chef towers and a Musketeer within reach of one. Own-left
 * serves first, for both seats. */
static void test_chef_serving_order(void) {
    for (int variant = 0; variant < 2; variant++) {       /* Musketeer near own-right / own-left */
        int levelled[2][2], waiting_right[2];
        for (int team = 0; team < 2; team++) {
            PrGame *g = (PrGame *)calloc(1, sizeof(PrGame));
            pr_setup_ex(g, PR_DECKS[0], PR_DECKS[1], 1, 0, PR_TIEBREAK_ABSOLUTE, team == 0 ? 3 : 0, team == 1 ? 3 : 0);
            uint32_t golem = spawn1(g, team, PR_CARD_GOLEM, ex_(team, 9000), ey_(team, 24000), 0);
            int32_t mx = variant == 0 ? 14500 : 3500;
            uint32_t musk = spawn1(g, team, PR_CARD_MUSKETEER, ex_(team, mx), ey_(team, 21500), 0);
            for (int k = 0; k < 565; k++) {
                pr_get(&g->st, golem)->deploy_ms = 1000; /* keep both standing */
                pr_get(&g->st, musk)->deploy_ms = 1000;
                pr_tick(g);
            }
            levelled[team][0] = pr_get(&g->st, golem)->lvl;
            levelled[team][1] = pr_get(&g->st, musk)->lvl;
            int own_right = team == 0 ? 2 : 1;
            waiting_right[team] = pr_get(&g->st, g->st.tower_id[team][own_right])->aux_ms >= 28000;
            free(g);
        }
        for (int team = 0; team < 2; team++) {
            CHECK_EQ(levelled[team][0], 1);                /* own-left tower serves the Golem first */
            CHECK_EQ(levelled[team][1], variant == 0 ? 1 : 0);
            CHECK_EQ(waiting_right[team], variant == 0 ? 0 : 1);
        }
    }
}

/* L3 / §18.1: a unit exactly between the two enemy princess towers takes its own-left one. */
static void test_crown_tower_tie_own_left(void) {
    for (int team = 0; team < 2; team++) {
        PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[0], 0);
        uint32_t k = spawn1(g, team, PR_CARD_KNIGHT, 9000, ey_(team, 12000), 1);
        pr_tick(g);
        int own_left = team == 0 ? 1 : 2;
        CHECK_EQ(pr_get(&g->st, k)->target_id, g->st.tower_id[1 - team][own_left]);
        free(g);
    }
}

/* L4 / §18.4: with both enemy princess towers killable, the heuristic bot finishes its own-left one
 * from both seats. */
static void test_bot_finish_tower_own_frame(void) {
    int cells[2];
    for (int team = 0; team < 2; team++) {
        PrGame *g = t_game_decks(3, PR_DECKS[0], PR_DECKS[0], 0);
        pr_debug_set_tower_hp(g, 1 - team, 1, 100);
        pr_debug_set_tower_hp(g, 1 - team, 2, 100);
        int8_t order[8] = {PR_CARD_FIREBALL, 0, 1, 2, 3, 4, 5, 6};
        pr_debug_set_hand(g, team, order);
        g->st.elixir[team] = PR_ELIXIR_MAX;
        PrBot b;
        pr_bot_init(&b, PR_BOT_HEURISTIC, 0, 7);
        int a = pr_bot_act(&b, &g->st, team);
        CHECK(a > 0);
        cells[team] = (a - 1) % PR_N_TILES;
        CHECK(cells[team] % 18 < 9); /* own-left */
        free(g);
    }
    CHECK_EQ(cells[0], cells[1]);
}

/* NIT: the bot values Lightning on the targets its bolts would strike (the 3 highest-hp enemies in
 * range) and a rolling spell from the tap it really uses. */
static void test_bot_spell_valuation(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[3], PR_DECKS[0], 0);
    PrBotView v;
    uint8_t mask[PR_N_ACTIONS];
    pr_legal_mask(&g->st, 0, mask);
    v.st = &g->st;
    v.mask = mask;
    v.team = 0;
    v.elixir = 10;
    /* 5 Skeletons in a Lightning disc: 3 bolts kill 3 of them -> 9 (not 15) */
    for (int k = 0; k < 5; k++) spawn1(g, 1, PR_CARD_KNIGHT, 7000 + 1000 * k, 20000, 0);
    for (int i = 0; i < g->st.n_ent; i++)
        if (g->st.ent[i].unit == PR_UNIT_KNIGHT) g->st.ent[i].hp = 50; /* killable by a bolt */
    CHECK_EQ(pr_bot_spell_value(&v, PR_CARD_LIGHTNING, 9000, 20000), 9);
    CHECK_EQ(pr_bot_spell_value(&v, PR_CARD_ZAP, 9000, 20000), 5 * 3); /* Zap hits all within 2500 + r */
    /* a Golem among them: bolt 1 on the Golem (survives, 1), bolts 2-3 on two Knights (3 + 3) */
    spawn1(g, 1, PR_CARD_GOLEM, 9000, 21000, 0);
    CHECK_EQ(pr_bot_spell_value(&v, PR_CARD_LIGHTNING, 9000, 20000), 7);
    free(g);
    /* rolling spells are scored over the corridor from the tap: [tap - min(roll, 5000), tap + 600] */
    g = t_game_decks(1, PR_DECKS[0], PR_DECKS[0], 0);
    pr_legal_mask(&g->st, 0, mask);
    v.st = &g->st;
    spawn1(g, 1, PR_CARD_KNIGHT, 9500, 20000, 0);
    spawn1(g, 1, PR_CARD_KNIGHT, 9500, 21000, 0);  /* behind the lead: in the corridor of a tap at 22000 */
    spawn1(g, 1, PR_CARD_KNIGHT, 9500, 17200, 0);  /* 4800 ahead of the tap (dry land) */
    CHECK_EQ(pr_bot_spell_value(&v, PR_CARD_THE_LOG, 9500, 22000), 3);            /* 5000 scored */
    CHECK_EQ(pr_bot_spell_value(&v, PR_CARD_BARBARIAN_BARREL, 9500, 22000), 2);   /* a 4500 roll */
    CHECK_EQ(pr_bot_spell_value(&v, PR_CARD_THE_LOG, 9500, 20000), 2); /* tapped at the lead: misses 21000 */
    free(g);
}

/* NIT: a rolling spell stops at the arena edge in both directions after the same number of ticks. */
static void test_rolling_edge_symmetric(void) {
    int ended[2];
    for (int team = 0; team < 2; team++) {
        PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[0], 0);
        pr_debug_spawn(g, team, PR_CARD_THE_LOG, 9000, ey_(team, 5000), 1, NULL, 0);
        ended[team] = -1;
        for (int k = 0; k < 60 && ended[team] < 0; k++) {
            pr_tick(g);
            if (g->st.n_fx == 0) ended[team] = k;
        }
        free(g);
    }
    CHECK(ended[0] > 0);
    CHECK_EQ(ended[0], ended[1]);
}

/* NIT: the occluder grid of an off-grid (debug) building is the rotation of its mirror's. */
static void test_occluder_rotation(void) {
    static PrGame a, b;
    pr_setup(&a, PR_DECKS[0], PR_DECKS[0], 1, 0, PR_TIEBREAK_ABSOLUTE);
    pr_setup(&b, PR_DECKS[0], PR_DECKS[0], 1, 0, PR_TIEBREAK_ABSOLUTE);
    pr_debug_spawn(&a, 0, PR_CARD_CANNON, 9250, 20250, 1, NULL, 0); /* footprint edges on cell centres */
    pr_debug_spawn(&b, 1, PR_CARD_CANNON, PR_ARENA_W - 9250, PR_ARENA_H - 20250, 1, NULL, 0);
    memset(&a.pc, 0, sizeof(a.pc));
    memset(&b.pc, 0, sizeof(b.pc));
    pr_path_sync(&a.st, &a.pc);
    pr_path_sync(&b.st, &b.pc);
    int bad = 0, covered = 0;
    for (int c = 0; c < PR_N_CELLS; c++) {
        int cx = c % PR_COLS, cy = c / PR_COLS, rc = (PR_ROWS - 1 - cy) * PR_COLS + (PR_COLS - 1 - cx);
        int oa = a.pc.occ[c] != PR_NO_ID, ob = b.pc.occ[rc] != PR_NO_ID;
        bad += oa != ob;
        covered += oa;
    }
    CHECK_EQ(bad, 0);
    CHECK(covered > 0);
}

/* L2 / §18.5: the audit's invalid states are rejected, and real states pass. */
static void test_snapshot_bounds(void) {
    static PrGame g;
    const char *why = NULL;
    pr_setup(&g, PR_DECKS[0], PR_DECKS[0], 1, 0, PR_TIEBREAK_ABSOLUTE);
    uint32_t id = spawn1(&g, 0, PR_CARD_CANNON, 9000, 24500, 1);
    for (int k = 0; k < 30; k++) pr_tick(&g);
    CHECK(pr_state_check(&g.st, &why));
    static PrState s;
    int i = pr_find(&g.st, id);
    /* (a) values the lifetime drain multiplies */
    s = g.st; s.ent[i].max_hp = s.ent[i].hp = 1000000000;
    CHECK(!pr_state_check(&s, &why));
    s = g.st; s.ent[i].life_ticks = 30000 / 50 + 2;
    CHECK(!pr_state_check(&s, &why));
    s = g.st; s.ent[i].life_ticks = 30000 / 50 + 1;
    CHECK(pr_state_check(&s, &why));
    s = g.st; s.ent[i].hp = s.ent[i].max_hp + 1;
    CHECK(!pr_state_check(&s, &why));
    s = g.st; s.ent[i].load_ms = PR_UNITS[PR_UNIT_CANNON].load_time_ms + 1;
    CHECK(!pr_state_check(&s, &why));
    s = g.st; s.ent[i].progress_ms = s.tick * PR_CHK_PROG_STEP + 1;
    CHECK(!pr_state_check(&s, &why));
    s = g.st; s.ent[i].lvl = 1; /* buildings are never levelled */
    CHECK(!pr_state_check(&s, &why));
    /* (b) pending unit spawns: a valid card and a non-tower unit */
    PrPendingSpawn p = {PR_UNIT_KNIGHT, 0, 0, PR_PEND_UNIT, {0, 0, 0}, 9000, 20000, 0};
    s = g.st; s.spawn[0] = p; s.n_spawn = 1;
    CHECK(pr_state_check(&s, &why));
    s.spawn[0].card = -1;
    CHECK(!pr_state_check(&s, &why));
    s.spawn[0].card = 0;
    s.spawn[0].unit = PR_UNIT_KINGTOWER;
    CHECK(!pr_state_check(&s, &why));
    /* crowns, tower slots, counters */
    s = g.st; s.crowns[0] = 1;             /* no enemy princess has fallen */
    CHECK(!pr_state_check(&s, &why));
    s = g.st; s.ent[1].unit = PR_UNIT_CANNONEER; /* the Princess slot holds the configured troop */
    CHECK(!pr_state_check(&s, &why));
    s = g.st; s.spent[0] = s.tick * 4 * 10 * PR_ELIXIR_UNIT + 1;
    CHECK(!pr_state_check(&s, &why));
    s = g.st; s.tick = PR_TICKS_MAX;       /* a running match cannot be at tick 6000 */
    CHECK(!pr_state_check(&s, &why));
    /* statistics counters saturate instead of overflowing */
    s = g.st;
    s.leaked[0] = PR_COUNTER_MAX - 10;
    s.elixir[0] = PR_ELIXIR_MAX;
    CHECK(pr_state_check(&s, &why));
    g.st = s;
    for (int k = 0; k < 5; k++) pr_tick(&g);
    CHECK_EQ(g.st.leaked[0], PR_COUNTER_MAX);
    CHECK(pr_state_check(&g.st, &why));
    /* the step bounds really cover every attacker */
    for (int u = 0; u < PR_N_UNITS; u++) {
        const PrUnitDef *d = &PR_UNITS[u];
        if (!pr_attacker(d)) continue;
        CHECK(d->hit_speed_ms <= PR_CHK_PROG_STEP && d->load_time_ms + PR_TICK_MS <= PR_CHK_PROG_STEP);
        CHECK((int64_t)d->speed * PR_MAX(d->charge_speed_pct, 100) / 100 <= PR_CHK_WALK_STEP);
    }
}

int main(void) {
    printf("test_audit2\n");
    RUN(test_chef_serving_order);
    RUN(test_crown_tower_tie_own_left);
    RUN(test_bot_finish_tower_own_frame);
    RUN(test_bot_spell_valuation);
    RUN(test_rolling_edge_symmetric);
    RUN(test_occluder_rotation);
    RUN(test_snapshot_bounds);
    TEST_END();
}
