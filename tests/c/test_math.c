/* test_math.c -- integer helpers, PCG32 and FNV-1a against reference values. */
#include "pr_test.h"

static void test_isqrt_exact(void) {
    for (int64_t n = 0; n < 200000; n++) {
        int64_t r = pr_isqrt64(n);
        if (!(r * r <= n && (r + 1) * (r + 1) > n)) {
            CHECK(0);
            fprintf(stderr, "isqrt(%lld) = %lld\n", (long long)n, (long long)r);
            return;
        }
    }
    for (int64_t k = 1; k < 3000000000LL; k = k * 3 + 7) {
        CHECK_EQ(pr_isqrt64(k * k), k);
        CHECK_EQ(pr_isqrt64(k * k - 1), k - 1);
        CHECK_EQ(pr_isqrt64(k * k + 1), k);
    }
    CHECK_EQ(pr_isqrt64(-5), 0);
    int64_t big = 9223372036854775807LL;
    int64_t r = pr_isqrt64(big);
    CHECK((uint64_t)r * (uint64_t)r <= (uint64_t)big);
    CHECK((uint64_t)(r + 1) * (uint64_t)(r + 1) > (uint64_t)big);
}

static void test_divs(void) {
    CHECK_EQ(pr_floordiv(7, 2), 3);
    CHECK_EQ(pr_floordiv(-7, 2), -4);
    CHECK_EQ(pr_floordiv(-8, 2), -4);
    CHECK_EQ(pr_ceildiv(7, 2), 4);
    CHECK_EQ(pr_ceildiv(-7, 2), -3);
    CHECK_EQ(pr_ceildiv(8, 2), 4);
    CHECK_EQ(pr_ceildiv(688 * 25, 100), 172);
    CHECK_EQ(pr_ceildiv(122 * 20, 100), 25);
    CHECK_EQ(pr_ceildiv(268 * 13, 100), 35);
    CHECK_EQ(pr_truncdiv(-7, 2), -3);
}

static void test_step_toward(void) {
    int32_t x = 0, y = 0;
    CHECK_EQ(pr_step_toward(&x, &y, 3000, 4000, 1000), 1000);
    CHECK_EQ(x, 600);
    CHECK_EQ(y, 800);
    /* never overshoots: a long step lands exactly on the target */
    CHECK_EQ(pr_step_toward(&x, &y, 3000, 4000, 99999), 4000);
    CHECK_EQ(x, 3000);
    CHECK_EQ(y, 4000);
    CHECK_EQ(pr_step_toward(&x, &y, 3000, 4000, 60), 0); /* already there */
    x = 0; y = 0;
    pr_step_toward(&x, &y, -3, -70, 50); /* d = 70: (-3*50/70, -70*50/70) truncated toward zero */
    CHECK_EQ(x, -2);
    CHECK_EQ(y, -50);
}

static void test_pcg32_reference(void) {
    /* pcg32-demo: pcg32_srandom_r(&rng, 42u, 54u) */
    PrRng r;
    pr_rng_seed(&r, 42u, 54u);
    const uint32_t want[6] = {0xa15c02b7u, 0x7b47f409u, 0xba1d3330u, 0x83d2f293u, 0xbfa4784bu, 0xcbed606eu};
    for (int i = 0; i < 6; i++) CHECK_EQ(pr_rng_next(&r), want[i]);
    /* bounded draws stay in range and hit every value */
    int seen[21] = {0};
    for (int i = 0; i < 10000; i++) {
        uint32_t v = pr_rng_below(&r, 21);
        CHECK(v < 21);
        seen[v]++;
    }
    for (int i = 0; i < 21; i++) CHECK(seen[i] > 300);
}

static void test_fnv1a(void) {
    CHECK_EQ(pr_fnv1a(PR_FNV_OFFSET, "", 0), (long long)0xcbf29ce484222325ULL);
    CHECK_EQ(pr_fnv1a(PR_FNV_OFFSET, "a", 1), (long long)0xaf63dc4c8601ec8cULL);
    CHECK_EQ(pr_fnv1a(PR_FNV_OFFSET, "foobar", 6), (long long)0x85944171f73967e8ULL);
}

static void test_card_db_examples(void) {
    /* SPEC §2 worked examples, as compiled */
    CHECK_EQ(PR_UNITS[PR_UNIT_KNIGHT].hp, 1766);
    CHECK_EQ(PR_UNITS[PR_UNIT_KNIGHT].damage, 202);
    CHECK_EQ(PR_UNITS[PR_UNIT_HOGRIDER].hp, 1697);
    CHECK_EQ(PR_PROJS[PR_PROJ_FIREBALLSPELL].damage, 688);
    CHECK_EQ(PR_AREAS[PR_AREA_ZAP].damage, 192);
    CHECK_EQ(PR_PROJS[PR_PROJ_LOGPROJECTILEROLLING].damage, 268);
    CHECK_EQ(PR_UNITS[PR_UNIT_GOBLIN].hp, 202);
    CHECK_EQ(PR_UNITS[PR_UNIT_GOBLIN].damage, 125);
    CHECK_EQ(PR_UNITS[PR_UNIT_PRINCESSTOWER].hp, 3052);
    CHECK_EQ(PR_UNITS[PR_UNIT_PRINCESSTOWER].damage, 109);
    CHECK_EQ(PR_UNITS[PR_UNIT_KINGTOWER].hp, 4824);
    CHECK_EQ(PR_UNITS[PR_UNIT_KINGTOWER].damage, 109);
    CHECK_EQ(PR_PROJS[PR_PROJ_TOWERPRINCESSPROJECTILE].damage, 109);
    CHECK_EQ(PR_UNITS[PR_UNIT_ICEGOLEMITE].death_damage, 84);
    CHECK_EQ(PR_UNITS[PR_UNIT_PRINCE].charge_damage, 783);
    CHECK_EQ(PR_UNITS[PR_UNIT_PRINCE].charge_range, 2500);
    CHECK_EQ(PR_N_CARDS, 64); /* SPEC §16.4 */
    static const int costs[64] = {3, 3, 4, 5, 4, 3, 4, 4, 3, 1, 2, 1, 5, 5, 3, 4, 4, 3, 2, 2, 3,
                                  5, 4, 7, 3, 2, 2, 2, 3, 6, 2, 3, 3, 5, 1, 3, 8, 7, 5, 6, 5, 3, 5,
                                  4, 3, 4, 5, 3, 4, 6, 6, 4, 6, 4, 4, 3, 6, 2, 2, 5, 6, 4, 2, 4};
    for (int c = 0; c < PR_N_CARDS; c++) CHECK_EQ(PR_CARDS[c].elixir, costs[c]);
    /* SPEC §16 level-11 examples (the generator asserts every card; these pin the compiled table) */
    CHECK_EQ(PR_UNITS[PR_UNIT_GOLEM].hp, 5120);
    CHECK_EQ(PR_UNITS[PR_UNIT_GOLEM].death_damage, 225);
    CHECK_EQ(PR_UNITS[PR_UNIT_GOLEMITE].hp, 1039);
    CHECK_EQ(PR_UNITS[PR_UNIT_PEKKA].hp, 3760);
    CHECK_EQ(PR_UNITS[PR_UNIT_INFERNOTOWER].damage, 43);
    CHECK_EQ(PR_UNITS[PR_UNIT_INFERNOTOWER].var_damage2, 158);
    CHECK_EQ(PR_UNITS[PR_UNIT_INFERNOTOWER].var_damage3, 847);
    CHECK_EQ(PR_UNITS[PR_UNIT_BALLOON].bomb_damage, 240);
    CHECK_EQ(PR_UNITS[PR_UNIT_ASSASSIN].dash_damage, 389);
    CHECK_EQ(PR_UNITS[PR_UNIT_MINER].crown_pct, 20);
    CHECK_EQ(PR_UNITS[PR_UNIT_MORTAR].min_range, 3500);
    CHECK_EQ(PR_UNITS[PR_UNIT_PRINCESS].damage, 168);
    CHECK_EQ(PR_PROJS[PR_PROJ_ROCKETSPELL].damage, 1484);
    CHECK_EQ(PR_PROJS[PR_PROJ_LIGHNINGSPELL].damage, 1057);
    CHECK_EQ(PR_BUFFS[PR_BUFF_POISON].dps, 92);
    CHECK_EQ(PR_BUFFS[PR_BUFF_EARTHQUAKE].building_pct, 350);
    CHECK_EQ(PR_UNITS[PR_UNIT_CANNONEER].hp, 2616);
    CHECK_EQ(PR_UNITS[PR_UNIT_CANNONEER].damage, 272);
    CHECK_EQ(PR_UNITS[PR_UNIT_DAGGERDUCHESS].hp, 2768);
    CHECK_EQ(PR_UNITS[PR_UNIT_CHEFTOWER].hp, 2703);
    CHECK_EQ(PR_CARDS[PR_CARD_GOBLIN_GANG].count + PR_CARDS[PR_CARD_GOBLIN_GANG].count2, 6);
    CHECK_EQ(PR_CARDS[PR_CARD_MINER].placement, PR_PLACE_MINER);
    CHECK_EQ(PR_CARDS[PR_CARD_BARBARIAN_BARREL].placement, PR_PLACE_TERRITORY);
    CHECK_EQ(PR_N_DECKS, 9);
    CHECK_EQ(PR_CARDS[PR_CARD_SKELETON_ARMY].count, 15);
    CHECK_EQ(PR_CARDS[PR_CARD_ARCHERS].count, 2);
    CHECK_EQ(PR_CARDS[PR_CARD_MINIONS].count, 3);
    /* formation offsets stay within 1600 */
    for (int f = 0; f < PR_N_FORMATIONS; f++)
        for (int k = 0; k < PR_FORMATIONS[f].count; k++) {
            int32_t ox = PR_FORMATION_OFFSETS[PR_FORMATIONS[f].first + k][0];
            int32_t oy = PR_FORMATION_OFFSETS[PR_FORMATIONS[f].first + k][1];
            CHECK((int64_t)ox * ox + (int64_t)oy * oy <= 1600LL * 1600LL);
        }
}

static void test_arena_db(void) {
    /* river y in [15000, 17000) is water except the bridges */
    CHECK(pr_point_water(1000, 15000));
    CHECK(pr_point_water(9000, 16999));
    CHECK(!pr_point_water(9000, 17000));
    CHECK(!pr_point_water(9000, 14999));
    CHECK(!pr_point_water(3500, 16000));
    CHECK(!pr_point_water(2500, 16000));
    CHECK(pr_point_water(2499, 16000));
    CHECK(!pr_point_water(4499, 16000));
    CHECK(pr_point_water(4500, 16000));
    CHECK(!pr_point_water(14500, 15500));
    /* tower positions */
    CHECK_EQ(PR_TOWER_POS[0][0][1], 29000);
    CHECK_EQ(PR_TOWER_POS[1][2][0], 14500);
    /* water ejection lands on dry ground */
    int32_t x = 9000, y = 16100;
    pr_eject_water(0, &x, &y);
    CHECK(!pr_point_wet(x, y));
    CHECK_EQ(y, 17001);
    x = 9000; y = 15100;
    pr_eject_water(0, &x, &y);
    CHECK_EQ(y, 14999);
    x = 9000; y = 15900; /* team 1: own bank is the top one */
    pr_eject_water(1, &x, &y);
    CHECK_EQ(y, 14999);
    x = 4700; y = 16000;
    pr_eject_water(1, &x, &y);
    CHECK_EQ(x, 4499);
    CHECK_EQ(y, 16000);
    /* the boundary lines are wet for both seats; seat conventions are rotations */
    CHECK(pr_point_wet(9000, 17000));
    CHECK(pr_point_wet(9000, 15000));
    CHECK(!pr_point_wet(9000, 17001));
    CHECK(!pr_point_wet(9000, 14999));
    CHECK(pr_point_wet(4500, 16000));
    CHECK(pr_point_wet(2500, 16000));
    CHECK(!pr_point_wet(2501, 16000));
    for (int32_t py = 14900; py <= 17100; py += 50)
        for (int32_t px = 2000; px <= 5000; px += 50)
            CHECK_EQ(pr_point_water_t(0, px, py), pr_point_water_t(1, 18000 - px, 32000 - py));
}

int main(void) {
    printf("test_math\n");
    RUN(test_isqrt_exact);
    RUN(test_divs);
    RUN(test_step_toward);
    RUN(test_pcg32_reference);
    RUN(test_fnv1a);
    RUN(test_card_db_examples);
    RUN(test_arena_db);
    TEST_END();
}
