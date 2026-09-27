/* test_path.c -- SPEC §6.3: a lone ground unit placed on ANY legal tile of its half
 * with no enemies must reach attack range of its default tower; ground non-jumpers
 * never end a tick in water; everyone stays in the arena. */
#include "pr_test.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define PR_SANITIZED 1
#endif
#endif
#ifndef PR_SANITIZED
#define PR_SANITIZED 0
#endif

static int reached_count = 0, trials = 0;
static long long total_ticks = 0;

/* Returns ticks to reach, or -1. */
static int lone_walk(int team, int unit, int tx, int ty, int max_ticks, int *water_violation) {
    PrGame *g = t_game_decks(1234, PR_DECKS[0], PR_DECKS[1], 0);
    int etx, ety;
    pr_tile_to_engine(team, tx, ty, &etx, &ety);
    uint32_t id = t_place(g, team, unit, -1, pr_tile_cx(etx), pr_tile_cy(ety), 1);
    PrEntity *e = t_ent(g, id);
    e->hp = e->max_hp = 1000000000; /* survive the towers */
    const PrUnitDef *d = &PR_UNITS[unit];
    int reached = -1;
    for (int t = 0; t < max_ticks; t++) {
        pr_tick(g);
        e = t_ent(g, id);
        if (!e) break;
        if (!pr_in_arena(e->x, e->y)) *water_violation = 2;
        if (!e->flying && !d->jumps && pr_point_wet(e->x, e->y)) *water_violation = 1;
        if (e->attacking) {
            const PrEntity *tg = pr_get_c(&g->st, e->target_id);
            if (tg && tg->kind == PR_KIND_TOWER) { reached = t + 1; break; }
        }
        if (g->st.over) break;
    }
    free(g);
    return reached;
}

static void sweep_unit(int unit, int max_ticks) {
    int stride = PR_SANITIZED ? 7 : 1, k = 0;
    int worst = 0;
    for (int team = 0; team < 2; team++) {
        PrGame *g = t_game(1);
        for (int ty = 17; ty < 32; ty++) {
            for (int tx = 0; tx < 18; tx++) {
                if (!pr_troop_tile_legal(&g->st, team, tx, ty)) continue;
                if ((k++ % stride) != 0) continue;
                int wv = 0;
                int r = lone_walk(team, unit, tx, ty, max_ticks, &wv);
                trials++;
                if (r < 0 || wv) {
                    CHECK(0);
                    fprintf(stderr, "  %s team %d tile (%d,%d): reached=%d water/arena violation=%d\n",
                            PR_UNITS[unit].name, team, tx, ty, r, wv);
                } else {
                    reached_count++;
                    total_ticks += r;
                    if (r > worst) worst = r;
                }
            }
        }
        free(g);
    }
    printf("    %-14s worst %4d ticks\n", PR_UNITS[unit].name, worst);
}

static void test_lone_units_reach_default_tower(void) {
    static const int units[] = {PR_UNIT_KNIGHT, PR_UNIT_GIANT, PR_UNIT_SKELETON, PR_UNIT_GOBLIN,
                                PR_UNIT_VALKYRIE, PR_UNIT_MUSKETEER, PR_UNIT_ARCHER, PR_UNIT_WIZARD,
                                PR_UNIT_ICEGOLEMITE, PR_UNIT_HOGRIDER, PR_UNIT_PRINCE, PR_UNIT_MINION,
                                PR_UNIT_BABYDRAGON, PR_UNIT_ICESPIRITS,
                                /* SPEC §16 ground troops (the Battle Ram is left out: its charged
                                 * kamikaze hit ends the walk on the tick it arrives) */
                                PR_UNIT_BARBARIAN, PR_UNIT_MINIPEKKA, PR_UNIT_PEKKA, PR_UNIT_SPEARGOBLIN,
                                PR_UNIT_GOBLIN_STAB, PR_UNIT_ROYALGIANT, PR_UNIT_BOMBER, PR_UNIT_PRINCESS,
                                PR_UNIT_BLOWDARTGOBLIN, PR_UNIT_FIRESPIRITS, PR_UNIT_ICEWIZARD, PR_UNIT_GOLEM,
                                PR_UNIT_GOLEMITE, PR_UNIT_GIANTSKELETON, PR_UNIT_WITCH, PR_UNIT_ASSASSIN,
                                PR_UNIT_ROYALHOG, PR_UNIT_MINER, PR_UNIT_RASCALBOY, PR_UNIT_RASCALGIRL,
                                PR_UNIT_ANGRYBARBARIAN, PR_UNIT_WALLBREAKER, PR_UNIT_DARKWITCH};
    for (size_t i = 0; i < sizeof(units) / sizeof(units[0]); i++) sweep_unit(units[i], 1500);
    printf("    %d/%d walks reached, mean %.1f ticks\n", reached_count, trials,
           reached_count ? (double)total_ticks / reached_count : 0.0);
}

/* After the enemy princess of its lane falls, a unit walks on to the King. */
static void test_default_route_falls_back_to_king(void) {
    for (int team = 0; team < 2; team++) {
        PrGame *g = t_game_decks(55, PR_DECKS[0], PR_DECKS[1], 0);
        int enemy = 1 - team;
        pr_debug_set_tower_hp(g, enemy, 1, 0);
        pr_debug_set_tower_hp(g, enemy, 2, 0);
        int etx, ety;
        pr_tile_to_engine(team, 3, 20, &etx, &ety);
        uint32_t id = t_place(g, team, PR_UNIT_GIANT, -1, pr_tile_cx(etx), pr_tile_cy(ety), 1);
        t_ent(g, id)->hp = 1000000000;
        int ok = 0;
        for (int t = 0; t < 1500 && !ok; t++) {
            pr_tick(g);
            PrEntity *e = t_ent(g, id);
            if (e->attacking && e->target_id == g->st.tower_id[enemy][0]) ok = 1;
        }
        CHECK(ok);
        free(g);
    }
}

/* Lane choice: own-frame x < 9000 goes own-left (engine left for team 0, engine right
 * for team 1); the two seats are rotations of each other. */
static void test_lane_choice_rotation(void) {
    PrGame *g = t_game(56);
    PrState *st = &g->st;
    uint32_t a = t_place(g, 0, PR_UNIT_KNIGHT, -1, 8500, 22000, 1);
    uint32_t b = t_place(g, 1, PR_UNIT_KNIGHT, -1, 18000 - 8500, 32000 - 22000, 1);
    uint32_t c = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9500, 22000, 1);
    uint32_t d = t_place(g, 1, PR_UNIT_KNIGHT, -1, 18000 - 9500, 32000 - 22000, 1);
    CHECK_EQ(pr_default_tower(st, t_ent(g, a))->id, st->tower_id[1][1]);
    CHECK_EQ(pr_default_tower(st, t_ent(g, b))->id, st->tower_id[0][2]);
    CHECK_EQ(pr_default_tower(st, t_ent(g, c))->id, st->tower_id[1][2]);
    CHECK_EQ(pr_default_tower(st, t_ent(g, d))->id, st->tower_id[0][1]);
    /* exactly on the centre line: own-left for both seats (SPEC v0.2 §13.6) */
    uint32_t e = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 22000, 1);
    uint32_t f = t_place(g, 1, PR_UNIT_KNIGHT, -1, 9000, 32000 - 22000, 1);
    CHECK_EQ(pr_default_tower(st, t_ent(g, e))->id, st->tower_id[1][1]);
    CHECK_EQ(pr_default_tower(st, t_ent(g, f))->id, st->tower_id[0][2]);
    free(g);
}

/* Rotational symmetry of a lone walk: team 1's rotated twin traces the rotated path. */
static void test_walk_rotation_symmetry(void) {
    int tiles[][2] = {{5, 20}, {12, 27}, {0, 18}, {16, 30}, {7, 22}};
    for (size_t i = 0; i < sizeof(tiles) / sizeof(tiles[0]); i++) {
        PrGame *g0 = t_game_decks(57, PR_DECKS[0], PR_DECKS[1], 0);
        PrGame *g1 = t_game_decks(57, PR_DECKS[0], PR_DECKS[1], 0);
        int tx = tiles[i][0], ty = tiles[i][1];
        uint32_t a = t_place(g0, 0, PR_UNIT_KNIGHT, -1, tx * 1000 + 500, ty * 1000 + 500, 1);
        uint32_t b = t_place(g1, 1, PR_UNIT_KNIGHT, -1, 18000 - (tx * 1000 + 500), 32000 - (ty * 1000 + 500), 1);
        int mism = 0;
        for (int t = 0; t < 300; t++) {
            pr_tick(g0);
            pr_tick(g1);
            PrEntity *ea = t_ent(g0, a), *eb = t_ent(g1, b);
            if (!ea || !eb) break;
            if (ea->x != 18000 - eb->x || ea->y != 32000 - eb->y) { mism = t + 1; break; }
        }
        if (mism) fprintf(stderr, "  rotation mismatch from tick %d at tile (%d,%d)\n", mism, tx, ty);
        CHECK_EQ(mism, 0);
        free(g0);
        free(g1);
    }
}

/* A unit knocked or pushed into the river is put back on land the same tick. */
static void test_never_in_water_after_push(void) {
    PrGame *g = t_game_decks(58, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t id = t_place(g, 0, PR_UNIT_KNIGHT, -1, 9000, 17200, 1);
    PrEntity *e = t_ent(g, id);
    pr_knockback(e, 0, -1500); /* into the river */
    CHECK(!pr_point_wet(e->x, e->y));
    e->y = 16000; /* teleported into the water: the Collide phase ejects it */
    pr_tick(g);
    e = t_ent(g, id);
    CHECK(!pr_point_wet(e->x, e->y));
    free(g);
}

/* A crowd funnelling over one bridge still gets through. */
static void test_crowd_crosses_bridge(void) {
    PrGame *g = t_game_decks(59, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t ids[15];
    int n = pr_spawn_formation(&g->st, PR_UNIT_SKELETON, 0, PR_CARD_SKELETON_ARMY, 3500, 20500, 15,
                               PR_FORMATION_N15, 0, 0, ids, 15);
    CHECK_EQ(n, 15);
    for (int i = 0; i < n; i++) t_ent(g, ids[i])->hp = 1000000000;
    t_ticks(g, 400);
    int across = 0;
    for (int i = 0; i < n; i++) {
        PrEntity *e = t_ent(g, ids[i]);
        if (e && e->y < 15000) across++;
        if (e) CHECK(!pr_point_wet(e->x, e->y));
    }
    CHECK_EQ(across, 15);
    free(g);
}

int main(void) {
    printf("test_path%s\n", PR_SANITIZED ? " (sanitized: sampled sweep)" : "");
    RUN(test_lane_choice_rotation);
    RUN(test_walk_rotation_symmetry);
    RUN(test_never_in_water_after_push);
    RUN(test_default_route_falls_back_to_king);
    RUN(test_crowd_crosses_bridge);
    RUN(test_lone_units_reach_default_tower);
    TEST_END();
}
