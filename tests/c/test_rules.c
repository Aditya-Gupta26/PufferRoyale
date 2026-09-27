/* test_rules.c -- match rules: elixir, lockout, cycle, legality, mask, Judge, King. */
#include "pr_test.h"

static const int8_t DECK_TEST[8] = {PR_CARD_KNIGHT, PR_CARD_ARCHERS, PR_CARD_CANNON, PR_CARD_TESLA,
                                    PR_CARD_FIREBALL, PR_CARD_GOBLIN_BARREL, PR_CARD_SKELETON_ARMY, PR_CARD_ZAP};

static void test_elixir_schedule(void) {
    PrGame *g = t_game(1);
    PrState *st = &g->st;
    CHECK_EQ(st->elixir[0], 16800);
    CHECK_EQ(st->elixir[1], 16800);
    pr_tick(g);
    CHECK_EQ(st->elixir[0], 16850);
    t_ticks(g, 55); /* 56 ticks = exactly one elixir at 1x */
    CHECK_EQ(st->elixir[0], 16800 + 2800);
    /* cap at 28000, the excess is leaked */
    t_ticks(g, 400);
    CHECK_EQ(st->elixir[0], 28000);
    CHECK_EQ(st->leaked[0], (456 - 224) * 50);
    /* rates by tick: 1x before 2400, 2x from 2400, 3x from 4800 */
    CHECK_EQ(pr_elixir_rate(0), 50);
    CHECK_EQ(pr_elixir_rate(2399), 50);
    CHECK_EQ(pr_elixir_rate(2400), 100);
    CHECK_EQ(pr_elixir_rate(4799), 100);
    CHECK_EQ(pr_elixir_rate(4800), 150);
    CHECK_EQ(pr_elixir_rate(5999), 150);
    while (st->tick < 2400) pr_tick(g);
    st->elixir[0] = 0;
    pr_tick(g); /* tick 2400 */
    CHECK_EQ(st->elixir[0], 100);
    while (st->tick < 4800) pr_tick(g);
    st->elixir[1] = 0;
    pr_tick(g); /* tick 4800 */
    CHECK_EQ(st->elixir[1], 150);
    free(g);
}

static void test_lockout(void) {
    PrGame *g = t_game(2);
    PrState *st = &g->st;
    uint8_t mask[PR_N_ACTIONS];
    for (int t = 0; t < 90; t++) {
        CHECK_EQ(pr_check_play(st, 0, 0, 9, 25), PR_ERR_LOCKOUT);
        pr_legal_mask(st, 0, mask);
        int s = 0;
        for (int a = 0; a < PR_N_ACTIONS; a++) s += mask[a];
        CHECK_EQ(s, 1);
        CHECK_EQ(mask[0], 1);
        pr_tick(g);
    }
    CHECK_EQ(st->tick, 90);
    pr_debug_set_hand(g, 0, DECK_TEST);
    CHECK_EQ(pr_queue_play(st, 0, 0, 9, 25), PR_OK); /* Knight at own tile (9, 25) */
    pr_tick(g);
    CHECK_EQ(st->plays[0], 1);
    free(g);
}

static void test_play_cycle_and_spawn(void) {
    PrGame *g = t_game_decks(3, DECK_TEST, PR_DECKS[1], 0);
    PrState *st = &g->st;
    pr_debug_set_hand(g, 0, DECK_TEST);
    /* hand = Knight Archers Cannon Tesla, queue = Fireball Barrel SkArmy Zap */
    CHECK_EQ(pr_queue_play(st, 0, 1, 5, 20), PR_OK); /* Archers */
    CHECK_EQ(st->elixir[0], 16800); /* charged at application, not at queue time */
    pr_tick(g);
    CHECK_EQ(st->elixir[0], 16800 + 50 - 3 * 2800);
    CHECK_EQ(st->hand[0][1], PR_CARD_FIREBALL);
    CHECK_EQ(st->queue[0][0], PR_CARD_GOBLIN_BARREL);
    CHECK_EQ(st->queue[0][3], PR_CARD_ARCHERS);
    CHECK_EQ(st->last_played[0][0], PR_CARD_ARCHERS);
    CHECK_EQ(st->spent[0], 3 * 2800);
    CHECK(st->seen_mask[0] & (1u << PR_CARD_ARCHERS));
    /* two Archers 1000 apart horizontally around the tile centre (5500, 20500) */
    int n = 0;
    for (int i = 0; i < st->n_ent; i++) {
        PrEntity *e = &st->ent[i];
        if (e->unit != PR_UNIT_ARCHER) continue;
        CHECK_EQ(e->y, 20500);
        CHECK(e->x == 5000 || e->x == 6000);
        CHECK_EQ(e->card, PR_CARD_ARCHERS);
        CHECK_EQ(e->deploy_ms, e->x == 5000 ? 1000 : 1100); /* member 1 staggered by 100 ms */
        n++;
    }
    CHECK_EQ(n, 2);
    /* the hand/queue is a permutation of the deck */
    int count[PR_N_CARDS] = {0};
    for (int i = 0; i < 4; i++) { count[st->hand[0][i]]++; count[st->queue[0][i]]++; }
    for (int i = 0; i < 8; i++) CHECK_EQ(count[DECK_TEST[i]], 1);
    /* a team-1 play of the same own-frame tile lands rotated */
    pr_debug_set_hand(g, 1, DECK_TEST);
    CHECK_EQ(pr_queue_play(st, 1, 0, 5, 20), PR_OK); /* Knight */
    pr_tick(g);
    int found = 0;
    for (int i = 0; i < st->n_ent; i++)
        if (st->ent[i].unit == PR_UNIT_KNIGHT && st->ent[i].team == 1) {
            CHECK_EQ(st->ent[i].x, 18000 - 5500);
            CHECK_EQ(st->ent[i].y, 32000 - 20500);
            found = 1;
        }
    CHECK(found);
    free(g);
}

static void test_play_errors(void) {
    PrGame *g = t_game_decks(4, DECK_TEST, PR_DECKS[1], 0);
    PrState *st = &g->st;
    pr_debug_set_hand(g, 0, DECK_TEST);
    CHECK_EQ(pr_check_play(st, 0, 4, 9, 25), PR_ERR_BAD_SLOT);
    CHECK_EQ(pr_check_play(st, 0, -1, 9, 25), PR_ERR_BAD_SLOT);
    CHECK_EQ(pr_check_play(st, 0, 0, 9, 16), PR_ERR_ILLEGAL_POSITION); /* river */
    CHECK_EQ(pr_check_play(st, 0, 0, 18, 25), PR_ERR_ILLEGAL_POSITION); /* off the grid */
    st->elixir[0] = 3 * 2800 - 1;
    CHECK_EQ(pr_check_play(st, 0, 0, 9, 25), PR_ERR_NOT_ENOUGH_ELIXIR);
    st->elixir[0] = 3 * 2800;
    CHECK_EQ(pr_check_play(st, 0, 0, 9, 25), PR_OK);
    /* SPEC §13.1: validation uses the current state; queued plays are re-validated in order
     * in Upkeep, where a play that no longer fits is dropped */
    st->elixir[0] = 6 * 2800;
    CHECK_EQ(pr_queue_play(st, 0, 0, 9, 25), PR_OK);    /* Knight (3) */
    CHECK_EQ(pr_check_play(st, 0, 0, 9, 25), PR_OK);    /* still the current state */
    CHECK_EQ(pr_queue_play(st, 0, 1, 9, 25), PR_OK);    /* Archers (3) */
    CHECK_EQ(st->elixir[0], 6 * 2800);                  /* nothing applied yet */
    pr_tick(g);
    CHECK_EQ(st->plays[0], 2);
    CHECK_EQ(st->elixir[0], 50);
    CHECK_EQ(st->dropped_plays, 0);
    /* two plays of the same slot: the second finds another card there and is dropped */
    pr_debug_set_hand(g, 0, DECK_TEST);
    st->elixir[0] = 28000;
    CHECK_EQ(pr_queue_play(st, 0, 0, 9, 25), PR_OK);
    CHECK_EQ(pr_queue_play(st, 0, 0, 9, 24), PR_OK);
    pr_tick(g);
    CHECK_EQ(st->plays[0], 3);
    CHECK_EQ(st->dropped_plays, 1);
    st->over = 1;
    CHECK_EQ(pr_check_play(st, 0, 0, 9, 25), PR_ERR_GAME_OVER);
    free(g);
}

/* Count legal own-frame troop tiles with ty < 17. */
static int enemy_side_troop_tiles(const PrState *st, int team, int *minx, int *maxx, int *miny, int *maxy) {
    int n = 0;
    *minx = 99; *maxx = -1; *miny = 99; *maxy = -1;
    for (int ty = 0; ty < 17; ty++)
        for (int tx = 0; tx < 18; tx++)
            if (pr_troop_tile_legal(st, team, tx, ty)) {
                n++;
                if (tx < *minx) *minx = tx;
                if (tx > *maxx) *maxx = tx;
                if (ty < *miny) *miny = ty;
                if (ty > *maxy) *maxy = ty;
            }
    return n;
}

static void test_troop_legality_and_pockets(void) {
    for (int team = 0; team < 2; team++) {
        PrGame *g = t_game(5);
        PrState *st = &g->st;
        int a, b, c, d;
        CHECK_EQ(enemy_side_troop_tiles(st, team, &a, &b, &c, &d), 0);
        /* own half: legal except tower footprints and no-deploy cells */
        CHECK(pr_troop_tile_legal(st, team, 9, 20));
        CHECK(pr_troop_tile_legal(st, team, 0, 18));
        CHECK(!pr_troop_tile_legal(st, team, 0, 17));  /* river-bank corner no-deploy */
        CHECK(!pr_troop_tile_legal(st, team, 3, 25));  /* own princess (own-frame left) */
        CHECK(!pr_troop_tile_legal(st, team, 2, 24));
        CHECK(pr_troop_tile_legal(st, team, 1, 24));
        CHECK(pr_troop_tile_legal(st, team, 5, 25));
        CHECK(!pr_troop_tile_legal(st, team, 8, 28));  /* King block */
        CHECK(!pr_troop_tile_legal(st, team, 3, 31));  /* back-corner strip */
        CHECK(pr_troop_tile_legal(st, team, 8, 31));
        /* the enemy tower on the acting team's OWN-frame left falls: engine idx 1 for
         * team 0, engine idx 2 for team 1 */
        int enemy = 1 - team;
        pr_debug_set_tower_hp(g, enemy, team == 0 ? 1 : 2, 0);
        CHECK_EQ(st->crowns[team], 1);
        int n = enemy_side_troop_tiles(st, team, &a, &b, &c, &d);
        CHECK_EQ(a, 0);
        CHECK_EQ(b, 8);
        CHECK_EQ(c, 11);
        CHECK_EQ(d, 14);
        CHECK_EQ(n, 9 * 4 - 1); /* minus the (0, 14) river-bank no-deploy corner */
        CHECK(!pr_troop_tile_legal(st, team, 3, 15)); /* the bridge stays closed */
        /* the other one falls too */
        pr_debug_set_tower_hp(g, enemy, team == 0 ? 2 : 1, 0);
        n = enemy_side_troop_tiles(st, team, &a, &b, &c, &d);
        CHECK_EQ(n, 18 * 4 - 2);
        CHECK_EQ(c, 11);
        CHECK(!st->over); /* two crowns in regulation do not end the match */
        CHECK_EQ(st->crowns[team], 2);
        /* a dead own princess no longer blocks its footprint */
        pr_debug_set_tower_hp(g, team, team == 0 ? 1 : 2, 0);
        CHECK(pr_troop_tile_legal(st, team, 3, 25));
        free(g);
    }
}

static void test_building_legality(void) {
    for (int team = 0; team < 2; team++) {
        PrGame *g = t_game_decks(6, DECK_TEST, DECK_TEST, 0);
        PrState *st = &g->st;
        int cannon = PR_UNIT_CANNON, tesla = PR_UNIT_TESLA;
        CHECK(!pr_building_tile_legal(st, team, cannon, 9, 17)); /* footprint pokes into the river */
        CHECK(pr_building_tile_legal(st, team, cannon, 9, 18));
        CHECK(!pr_building_tile_legal(st, team, cannon, 0, 20)); /* outside the arena */
        CHECK(!pr_building_tile_legal(st, team, cannon, 1, 18)); /* covers a no-deploy corner */
        CHECK(pr_building_tile_legal(st, team, cannon, 1, 19));
        CHECK(pr_building_tile_legal(st, team, cannon, 5, 22));  /* touches the princess footprint */
        CHECK(!pr_building_tile_legal(st, team, cannon, 5, 23)); /* overlaps it */
        CHECK(!pr_building_tile_legal(st, team, cannon, 9, 14)); /* never in a pocket */
        CHECK(pr_building_tile_legal(st, team, tesla, 9, 18));
        CHECK(!pr_building_tile_legal(st, team, tesla, 9, 17));
        /* building vs building (any team) */
        pr_debug_set_hand(g, team, DECK_TEST);
        CHECK_EQ(pr_queue_play(st, team, 2, 9, 20), PR_OK); /* Cannon */
        pr_tick(g);
        CHECK(!pr_building_tile_legal(st, team, cannon, 11, 20));
        CHECK(pr_building_tile_legal(st, team, cannon, 12, 20));
        /* Tesla (F=2) anchors on the own-frame top-left corner of the tile (v0.2): tapped at
         * own (11, 20) it covers own x [10000, 12000), overlapping the Cannon's [8000, 11000) */
        CHECK(!pr_building_tile_legal(st, team, tesla, 11, 20));
        CHECK(pr_building_tile_legal(st, team, tesla, 12, 20));  /* [11000, 13000): touches only */
        /* v0.2: troops may not be placed on a tile overlapping a living building footprint */
        CHECK(!pr_troop_tile_legal(st, team, 9, 20));
        CHECK(!pr_troop_tile_legal(st, team, 10, 21));
        CHECK(pr_troop_tile_legal(st, team, 11, 20));
        CHECK(pr_troop_tile_legal(st, team, 9, 22));
        /* the enemy cannot drop troops on it through a pocket either */
        PrLegalCtx c;
        pr_legal_ctx(st, 1 - team, &c);
        CHECK(!pr_troop_tile_legal_ctx(&c, 1 - team, 17 - 9, 31 - 20));
        free(g);
    }
}

static void test_spell_legality(void) {
    PrGame *g = t_game(7);
    PrState *st = &g->st;
    for (int team = 0; team < 2; team++) {
        int nfb = 0, nbarrel = 0, nlog = 0;
        for (int ty = 0; ty < 32; ty++)
            for (int tx = 0; tx < 18; tx++) {
                nfb += pr_card_tile_legal(st, team, PR_CARD_FIREBALL, tx, ty);
                nbarrel += pr_card_tile_legal(st, team, PR_CARD_GOBLIN_BARREL, tx, ty);
                nlog += pr_card_tile_legal(st, team, PR_CARD_THE_LOG, tx, ty);
            }
        CHECK_EQ(nfb, 576);
        CHECK_EQ(nbarrel, 576 - 32); /* 2 river rows x 16 water tiles */
        CHECK(pr_card_tile_legal(st, team, PR_CARD_GOBLIN_BARREL, 3, 15));
        CHECK(!pr_card_tile_legal(st, team, PR_CARD_GOBLIN_BARREL, 4, 15));
        /* v0.2 The Log: troop territory only (all of the own half, towers and no-deploy
         * cells included), not water, not the enemy side until a pocket opens */
        CHECK_EQ(nlog, 15 * 18);
        CHECK(pr_card_tile_legal(st, team, PR_CARD_THE_LOG, 3, 25));  /* on its own princess */
        CHECK(pr_card_tile_legal(st, team, PR_CARD_THE_LOG, 0, 17));  /* no-deploy corner */
        CHECK(!pr_card_tile_legal(st, team, PR_CARD_THE_LOG, 3, 15)); /* the bridge (river band) */
        CHECK(!pr_card_tile_legal(st, team, PR_CARD_THE_LOG, 9, 10));
    }
    pr_debug_set_tower_hp(g, 1, 1, 0); /* team 0 opens its own-left pocket */
    int npocket = 0;
    for (int ty = 0; ty < 17; ty++)
        for (int tx = 0; tx < 18; tx++) npocket += pr_card_tile_legal(st, 0, PR_CARD_THE_LOG, tx, ty);
    CHECK_EQ(npocket, 9 * 4); /* tx 0..8 x ty 11..14, the no-deploy corner included */
    free(g);
}

static void test_mask_equals_check(void) {
    PrGame *g = t_game_decks(8, NULL, NULL, 0);
    PrState *st = &g->st;
    PrRng r;
    pr_rng_seed(&r, 99, 1);
    uint8_t mask[PR_N_ACTIONS];
    for (int round = 0; round < 40; round++) {
        for (int team = 0; team < 2; team++) {
            pr_legal_mask(st, team, mask);
            CHECK_EQ(mask[0], 1);
            for (int a = 1; a < PR_N_ACTIONS; a++) {
                int slot = (a - 1) / 576, cell = (a - 1) % 576;
                int ok = pr_check_play(st, team, slot, cell % 18, cell / 18) == PR_OK;
                if (ok != mask[a]) {
                    CHECK(0);
                    fprintf(stderr, "mask mismatch team %d a %d\n", team, a);
                    break;
                }
            }
            /* random legal play now and then */
            if (pr_rng_below(&r, 3) == 0) {
                int legal[PR_N_ACTIONS], nl = 0;
                for (int a = 1; a < PR_N_ACTIONS; a++)
                    if (mask[a]) legal[nl++] = a;
                if (nl) {
                    int a = legal[pr_rng_below(&r, (uint32_t)nl)];
                    int slot = (a - 1) / 576, cell = (a - 1) % 576;
                    CHECK_EQ(pr_queue_play(st, team, slot, cell % 18, cell / 18), PR_OK);
                }
            }
        }
        t_ticks(g, 15);
        if (round == 20) pr_debug_set_tower_hp(g, 1, 1, 0); /* open a pocket */
    }
    free(g);
}

static void test_king_activation(void) {
    /* (a) damage to the King */
    PrGame *g = t_game(9);
    PrState *st = &g->st;
    t_ticks(g, 10);
    PrEntity *k = t_ent(g, st->tower_id[1][0]);
    k->dmg_in += 1; /* resolved in the next tick's Resolve (tick T) */
    int32_t T = st->tick;
    pr_tick(g);
    CHECK_EQ(st->king_wake[1], PR_KING_WAKE_TICKS);
    CHECK(!st->king_active[1]);
    while (st->tick < T + 71) {
        CHECK(!st->king_active[1]);
        pr_tick(g);
    }
    CHECK_EQ(st->tick, T + 71);
    CHECK(!st->king_active[1]); /* ticks T+1 .. T+70 processed */
    pr_tick(g);                  /* tick T + 71 */
    CHECK(st->king_active[1]);
    CHECK(!st->king_active[0]);
    free(g);
    /* (b) own princess destroyed */
    g = t_game(10);
    st = &g->st;
    pr_debug_set_tower_hp(g, 0, 2, 0);
    CHECK_EQ(st->crowns[1], 1);
    CHECK_EQ(st->king_wake[0], PR_KING_WAKE_TICKS);
    t_ticks(g, 70);
    CHECK(!st->king_active[0]);
    pr_tick(g);
    CHECK(st->king_active[0]);
    free(g);
}

static void test_judge(void) {
    /* King destroyed: immediate, destroyer's crowns become 3 */
    PrGame *g = t_game(11);
    PrState *st = &g->st;
    t_ticks(g, 5);
    pr_debug_set_tower_hp(g, 1, 0, 0);
    CHECK(st->over);
    CHECK_EQ(st->result[0], 1);
    CHECK_EQ(st->result[1], -1);
    CHECK_EQ(st->crowns[0], 3);
    CHECK_EQ(st->end_reason, PR_END_KING);
    int32_t t = st->tick;
    pr_tick(g); /* a finished match does not tick */
    CHECK_EQ(st->tick, t);
    free(g);
    /* both Kings on the same tick: draw */
    g = t_game(12);
    st = &g->st;
    t_ent(g, st->tower_id[0][0])->dmg_in += 99999;
    t_ent(g, st->tower_id[1][0])->dmg_in += 99999;
    pr_tick(g);
    CHECK(st->over);
    CHECK_EQ(st->result[0], 0);
    CHECK_EQ(st->result[1], 0);
    CHECK_EQ(st->end_reason, PR_END_DRAW);
    free(g);
    /* regulation: more crowns wins at the end of tick 3599 */
    g = t_game(13);
    st = &g->st;
    pr_debug_set_tower_hp(g, 0, 1, 0); /* team 1 leads */
    while (st->tick < 3599) pr_tick(g);
    CHECK(!st->over);
    pr_tick(g); /* tick 3599 */
    CHECK(st->over);
    CHECK_EQ(st->result[1], 1);
    CHECK_EQ(st->end_reason, PR_END_REGULATION_CROWNS);
    CHECK_EQ(st->tick, 3600);
    free(g);
    /* overtime: the first crown difference ends it */
    g = t_game(14);
    st = &g->st;
    while (st->tick < 3600) pr_tick(g);
    CHECK(!st->over);
    t_ticks(g, 100);
    t_ent(g, st->tower_id[1][2])->dmg_in += 99999;
    pr_tick(g);
    CHECK(st->over);
    CHECK_EQ(st->result[0], 1);
    CHECK_EQ(st->end_reason, PR_END_OVERTIME_CROWNS);
    free(g);
    /* tiebreak (absolute): the weakest surviving tower loses */
    g = t_game(15);
    st = &g->st;
    pr_debug_set_tower_hp(g, 0, 1, 1000);
    pr_debug_set_tower_hp(g, 1, 0, 1001); /* team 1's weakest is its King at 1001 */
    while (!st->over) pr_tick(g);
    CHECK_EQ(st->tick, 6000);
    CHECK_EQ(st->result[0], -1);
    CHECK_EQ(st->end_reason, PR_END_TIEBREAK);
    free(g);
    /* exact tie: draw */
    g = t_game(16);
    st = &g->st;
    while (!st->over) pr_tick(g);
    CHECK_EQ(st->result[0], 0);
    CHECK_EQ(st->end_reason, PR_END_DRAW);
    free(g);
    /* fraction mode: 1000/3052 (32.8 %) vs 1600/4824 (33.2 %): team 0 loses */
    g = (PrGame *)calloc(1, sizeof(PrGame));
    pr_setup(g, PR_DECKS[0], PR_DECKS[1], 17, 90, PR_TIEBREAK_FRACTION);
    st = &g->st;
    pr_debug_set_tower_hp(g, 0, 2, 1000);
    pr_debug_set_tower_hp(g, 1, 0, 1600);
    while (!st->over) pr_tick(g);
    CHECK_EQ(st->result[0], -1);
    CHECK_EQ(st->end_reason, PR_END_TIEBREAK);
    free(g);
}

static void test_deploy_timing(void) {
    PrGame *g = t_game_decks(18, DECK_TEST, PR_DECKS[1], 0);
    PrState *st = &g->st;
    pr_debug_set_hand(g, 0, DECK_TEST);
    t_ticks(g, 3);
    int32_t P = st->tick;
    CHECK_EQ(pr_queue_play(st, 0, 0, 9, 22), PR_OK); /* Knight */
    pr_tick(g);                                      /* tick P */
    PrEntity *k = NULL;
    for (int i = 0; i < st->n_ent; i++)
        if (st->ent[i].unit == PR_UNIT_KNIGHT) k = &st->ent[i];
    CHECK(k != NULL);
    if (!k) { free(g); return; }
    uint32_t id = k->id;
    CHECK_EQ(k->spawn_tick, P);
    CHECK_EQ(k->deploy_ms, 1000);
    int32_t x0 = k->x, y0 = k->y;
    for (int i = 1; i < 20; i++) { /* ticks P+1 .. P+19: deploying, not moving */
        pr_tick(g);
        k = t_ent(g, id);
        CHECK(k->deploy_ms > 0);
        CHECK_EQ(k->x, x0);
        CHECK_EQ(k->y, y0);
    }
    CHECK_EQ(k->deploy_ms, 50);
    pr_tick(g); /* tick P+20: deployed, takes its first step */
    k = t_ent(g, id);
    CHECK_EQ(k->deploy_ms, 0);
    CHECK(k->x != x0 || k->y != y0);
    CHECK_EQ(k->load_ms, 0); /* the 700 ms load timer ran down during the deploy */
    free(g);
}

static void test_reset_determinism(void) {
    PrGame *a = t_game_decks(77, NULL, NULL, 90), *b = t_game_decks(77, NULL, NULL, 90);
    CHECK_EQ(pr_hash(&a->st), pr_hash(&b->st));
    pr_new_match(&a->st);
    CHECK(pr_hash(&a->st) != pr_hash(&b->st)); /* the RNG stream continued: a new deal */
    pr_reseed(&a->st, 77);
    pr_new_match(&a->st);
    CHECK_EQ(pr_hash(&a->st), pr_hash(&b->st));
    /* random decks: 8 distinct cards */
    for (int t = 0; t < 2; t++) {
        int seen[PR_N_CARDS] = {0};
        for (int i = 0; i < 8; i++) seen[a->st.deck[t][i]]++;
        for (int c = 0; c < PR_N_CARDS; c++) CHECK(seen[c] <= 1);
    }
    free(a);
    free(b);
}

static void test_tesla_anchor_both_seats(void) {
    PrGame *g = t_game_decks(19, DECK_TEST, DECK_TEST, 0);
    PrState *st = &g->st;
    pr_debug_set_hand(g, 0, DECK_TEST);
    pr_debug_set_hand(g, 1, DECK_TEST);
    st->elixir[0] = 28000;
    CHECK_EQ(pr_queue_play(st, 0, 3, 9, 20), PR_OK); /* Tesla */
    CHECK_EQ(pr_queue_play(st, 1, 3, 9, 20), PR_OK);
    CHECK_EQ(pr_queue_play_at(st, 0, 2, 4600, 21900), PR_OK); /* Cannon via a raw tap: tile centre */
    pr_tick(g);
    int seen = 0;
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        if (e->unit == PR_UNIT_TESLA) { /* own-frame top-left corner of own tile (9, 20) */
            CHECK_EQ(e->x, e->team == 0 ? 9000 : 18000 - 9000);
            CHECK_EQ(e->y, e->team == 0 ? 20000 : 32000 - 20000);
            seen++;
        }
        if (e->unit == PR_UNIT_CANNON) {
            CHECK_EQ(e->x, 4500);
            CHECK_EQ(e->y, 21500);
            seen++;
        }
    }
    CHECK_EQ(seen, 3);
    /* the placed Tesla's footprint is exactly what legality saw: own tiles 8..9 x 19..20 */
    CHECK(!pr_troop_tile_legal(st, 0, 8, 19));
    CHECK(!pr_troop_tile_legal(st, 0, 9, 20));
    CHECK(pr_troop_tile_legal(st, 0, 10, 20));
    CHECK(pr_troop_tile_legal(st, 0, 9, 21));
    CHECK(!pr_troop_tile_legal(st, 1, 8, 19));
    CHECK(pr_troop_tile_legal(st, 1, 10, 20));
    free(g);
}

static void test_simultaneous_order(void) {
    /* SPEC v0.2 §13.13: team 0's queued plays are applied first (lower ids) */
    PrGame *g = t_game_decks(20, DECK_TEST, DECK_TEST, 0);
    PrState *st = &g->st;
    pr_debug_set_hand(g, 0, DECK_TEST);
    pr_debug_set_hand(g, 1, DECK_TEST);
    for (int round = 0; round < 2; round++) {
        CHECK_EQ(pr_queue_play(st, 1, 0, 5, 20), PR_OK);
        CHECK_EQ(pr_queue_play(st, 0, 0, 5, 20), PR_OK);
        int first = st->n_ent;
        pr_tick(g);
        CHECK_EQ(st->ent[first].team, 0);
        pr_debug_set_hand(g, 0, DECK_TEST);
        pr_debug_set_hand(g, 1, DECK_TEST);
        st->elixir[0] = st->elixir[1] = 20000;
    }
    free(g);
    /* opt-in alternation: first_team flips after each tick where both teams play */
    g = t_game_decks(21, DECK_TEST, DECK_TEST, 0);
    st = &g->st;
    pr_debug_set_first_team(g, 0);
    for (int round = 0; round < 4; round++) {
        pr_debug_set_hand(g, 0, DECK_TEST);
        pr_debug_set_hand(g, 1, DECK_TEST);
        st->elixir[0] = st->elixir[1] = 20000;
        CHECK_EQ(pr_queue_play(st, 0, 0, 5, 20), PR_OK);
        CHECK_EQ(pr_queue_play(st, 1, 0, 5, 20), PR_OK);
        int first = st->n_ent;
        pr_tick(g);
        CHECK_EQ(st->ent[first].team, round % 2);
    }
    free(g);
}

/* With alternation, a two-seat scenario and its mirror (roles swapped, the other team first)
 * evolve as exact rotations of each other, ids included. */
static void test_alternating_order_mirror_is_exact(void) {
    PrGame *a = t_game_decks(22, DECK_TEST, DECK_TEST, 0), *b = t_game_decks(22, DECK_TEST, DECK_TEST, 0);
    int8_t h0[8] = {PR_CARD_KNIGHT, PR_CARD_ARCHERS, PR_CARD_CANNON, PR_CARD_TESLA, PR_CARD_FIREBALL,
                    PR_CARD_GOBLIN_BARREL, PR_CARD_SKELETON_ARMY, PR_CARD_ZAP};
    int8_t h1[8] = {PR_CARD_SKELETON_ARMY, PR_CARD_ARCHERS, PR_CARD_CANNON, PR_CARD_TESLA, PR_CARD_FIREBALL,
                    PR_CARD_GOBLIN_BARREL, PR_CARD_KNIGHT, PR_CARD_ZAP};
    pr_debug_set_hand(a, 0, h0);
    pr_debug_set_hand(a, 1, h1);
    pr_debug_set_hand(b, 1, h0);
    pr_debug_set_hand(b, 0, h1);
    pr_debug_set_first_team(a, 0);
    pr_debug_set_first_team(b, 1);
    CHECK_EQ(pr_queue_play(&a->st, 0, 0, 5, 18), PR_OK); /* Knight near the river */
    CHECK_EQ(pr_queue_play(&a->st, 1, 0, 6, 12), PR_ERR_ILLEGAL_POSITION);
    CHECK_EQ(pr_queue_play(&a->st, 1, 0, 11, 18), PR_OK); /* Skeleton Army */
    CHECK_EQ(pr_queue_play(&b->st, 1, 0, 5, 18), PR_OK);
    CHECK_EQ(pr_queue_play(&b->st, 0, 0, 11, 18), PR_OK);
    int diverged = -1;
    for (int t = 0; t < 500 && diverged < 0; t++) {
        pr_tick(a);
        pr_tick(b);
        if (a->st.n_ent != b->st.n_ent) { diverged = t; break; }
        for (int i = 0; i < a->st.n_ent; i++) {
            const PrEntity *x = &a->st.ent[i], *y = &b->st.ent[i];
            if (x->kind == PR_KIND_TOWER) continue; /* towers keep fixed per-team ids: see below */
            if (x->id != y->id || x->unit != y->unit || x->team != 1 - y->team || x->x != 18000 - y->x ||
                x->y != 32000 - y->y || x->hp != y->hp) { diverged = t; break; }
        }
        for (int team = 0; team < 2 && diverged < 0; team++) {
            static const int mirror_idx[3] = {0, 2, 1}; /* engine left <-> right under rotation */
            for (int k = 0; k < 3; k++) {
                const PrEntity *x = pr_get_c(&a->st, a->st.tower_id[team][k]);
                const PrEntity *y = pr_get_c(&b->st, b->st.tower_id[1 - team][mirror_idx[k]]);
                if ((x == NULL) != (y == NULL) || (x && x->hp != y->hp)) diverged = t;
            }
        }
    }
    CHECK_EQ(diverged, -1);
    free(a);
    free(b);
}

/* SPEC v0.2.1 §14.1: all of a tick's plays are validated against the start-of-Upkeep state;
 * plays of different teams never invalidate each other (the auditor's pocket case: a team 0
 * Cannon and a team 1 pocket Knight on the same tile in the same tick), and a team's own
 * plays run against its own running elixir / hand only. */
static void test_same_tick_validation(void) {
    PrGame *g = t_game_decks(30, DECK_TEST, DECK_TEST, 0);
    PrState *st = &g->st;
    pr_debug_set_hand(g, 0, DECK_TEST);
    pr_debug_set_hand(g, 1, DECK_TEST);
    CHECK_EQ(pr_debug_set_tower_hp(g, 0, 1, 0), 0); /* team 0 left princess: team 1 pocket opens */
    st->elixir[0] = st->elixir[1] = 28000;
    uint8_t m0[PR_N_ACTIONS], m1[PR_N_ACTIONS];
    pr_legal_mask(st, 0, m0);
    pr_legal_mask(st, 1, m1);
    int a0 = 1 + 2 * PR_N_TILES + 18 * 18 + 6;                 /* team 0 Cannon (slot 2), engine tile (6, 18) */
    int a1 = 1 + 0 * PR_N_TILES + (31 - 18) * 18 + (17 - 6);   /* team 1 Knight, same engine tile */
    CHECK(m0[a0] && m1[a1]);
    CHECK_EQ(pr_queue_play(st, 0, 2, 6, 18), PR_OK);
    CHECK_EQ(pr_queue_play(st, 1, 0, 11, 13), PR_OK);
    pr_tick(g);
    CHECK_EQ(st->dropped_plays, 0);
    CHECK_EQ(st->plays[0], 1);
    CHECK_EQ(st->plays[1], 1);
    CHECK_EQ(st->elixir[1], 28000 - 3 * 2800); /* at the cap, the regen leaked */
    int knights = 0, cannons = 0;
    for (int i = 0; i < st->n_ent; i++) {
        knights += st->ent[i].unit == PR_UNIT_KNIGHT && st->ent[i].team == 1;
        cannons += st->ent[i].unit == PR_UNIT_CANNON && st->ent[i].team == 0;
    }
    CHECK_EQ(knights, 1);
    CHECK_EQ(cannons, 1);
    /* the same team's own Cannon + Knight on one tile in one tick: both judged on the
     * start-of-Upkeep board, so both are applied */
    pr_debug_set_hand(g, 0, DECK_TEST);
    st->elixir[0] = 28000;
    CHECK_EQ(pr_queue_play(st, 0, 2, 12, 22), PR_OK);
    CHECK_EQ(pr_queue_play(st, 0, 0, 12, 22), PR_OK);
    pr_tick(g);
    CHECK_EQ(st->dropped_plays, 0);
    CHECK_EQ(st->plays[0], 3);
    /* ... but the running elixir is per team: a second play the team cannot afford drops */
    pr_debug_set_hand(g, 0, DECK_TEST);
    st->elixir[0] = 6 * 2800 - 100;                        /* 5.96 elixir + 50 regen */
    CHECK_EQ(pr_queue_play(st, 0, 0, 3, 25), PR_OK);       /* Knight 3 */
    CHECK_EQ(pr_queue_play(st, 0, 1, 4, 25), PR_OK);       /* Archers 3: 2.98 left, dropped */
    pr_tick(g);
    CHECK_EQ(st->dropped_plays, 1);
    CHECK_EQ(st->plays[0], 4);
    free(g);
}

/* SPEC v0.2.1 §14.4: members of a hand-played troop that land in water go to the tap's side
 * of the river, never across: a raw tap on the own half's river edge (Game.play) puts every
 * Skeleton on the own bank, a pocket tap every Skeleton on the enemy bank; both seats agree
 * exactly (own frame). */
static void test_formation_stays_on_tap_side(void) {
    static const int8_t SKARMY_FIRST[8] = {PR_CARD_SKELETON_ARMY, PR_CARD_KNIGHT, PR_CARD_ARCHERS, PR_CARD_CANNON,
                                           PR_CARD_TESLA, PR_CARD_FIREBALL, PR_CARD_GOBLIN_BARREL, PR_CARD_ZAP};
    int32_t pos[2][2][32][2];
    int cnt[2][2] = {{0, 0}, {0, 0}};
    for (int team = 0; team < 2; team++) {
        for (int pocket = 0; pocket < 2; pocket++) {
            PrGame *g = t_game_decks(31, DECK_TEST, DECK_TEST, 0);
            PrState *st = &g->st;
            pr_debug_set_hand(g, team, SKARMY_FIRST);
            st->elixir[team] = 28000;
            int32_t ox = 7000, oy = 17000;                        /* own tile (7, 17), on the river edge,
                                                                   * clear of the bridges */
            if (pocket) {
                /* the enemy princess of the team's own-left lane falls: own tiles x 0-8, y 11-14 open */
                CHECK_EQ(pr_debug_set_tower_hp(g, 1 - team, team == 0 ? 1 : 2, 0), 0);
                oy = 14999;                                       /* own tile (7, 14), on the river edge */
            }
            int64_t x = team == 0 ? ox : PR_ARENA_W - ox, y = team == 0 ? oy : PR_ARENA_H - oy;
            CHECK_EQ(pr_queue_play_at(st, team, 0, x, y), PR_OK);
            int first = st->n_ent;
            pr_phase_upkeep(st); /* the play is applied (formation placed) in Upkeep */
            for (int i = first; i < st->n_ent; i++) {
                const PrEntity *e = &st->ent[i];
                int32_t ey = pr_own_y(team, e->y);
                CHECK(!pr_point_wet(e->x, e->y));
                if (pocket) CHECK(ey < PR_RIVER_Y0);
                else CHECK(ey >= PR_RIVER_Y1);
                pos[team][pocket][cnt[team][pocket]][0] = pr_own_x(team, e->x);
                pos[team][pocket][cnt[team][pocket]][1] = ey;
                cnt[team][pocket]++;
            }
            CHECK_EQ(cnt[team][pocket], 15);
            free(g);
        }
    }
    for (int pocket = 0; pocket < 2; pocket++)
        for (int i = 0; i < cnt[0][pocket]; i++) {
            CHECK_EQ(pos[0][pocket][i][0], pos[1][pocket][i][0]);
            CHECK_EQ(pos[0][pocket][i][1], pos[1][pocket][i][1]);
        }
}

int main(void) {
    printf("test_rules\n");
    RUN(test_elixir_schedule);
    RUN(test_lockout);
    RUN(test_play_cycle_and_spawn);
    RUN(test_play_errors);
    RUN(test_troop_legality_and_pockets);
    RUN(test_building_legality);
    RUN(test_spell_legality);
    RUN(test_mask_equals_check);
    RUN(test_king_activation);
    RUN(test_judge);
    RUN(test_deploy_timing);
    RUN(test_reset_determinism);
    RUN(test_tesla_anchor_both_seats);
    RUN(test_simultaneous_order);
    RUN(test_alternating_order_mirror_is_exact);
    RUN(test_same_tick_validation);
    RUN(test_formation_stays_on_tap_side);
    TEST_END();
}
