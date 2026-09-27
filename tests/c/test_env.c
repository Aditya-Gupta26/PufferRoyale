/* test_env.c -- the PufferLib env wrapper (royale.h), observation encoder and bots in C. */
#include "royale.h"
#include "pr_test.h"

typedef struct {
    Royale env;
    float obs[2 * PR_OBS_SIZE];
    int acts[2];
    float rew[2];
    unsigned char term[2];
} TEnv;

static TEnv *t_env(int num_agents, int opponent, int learner, uint64_t seed) {
    TEnv *t = (TEnv *)calloc(1, sizeof(TEnv));
    Royale *e = &t->env;
    e->observations = t->obs;
    e->actions = t->acts;
    e->rewards = t->rew;
    e->terminals = t->term;
    e->num_agents = num_agents;
    e->frame_skip = 10;
    e->opponent = opponent;
    e->learner_cfg = learner;
    e->bot_ppm = 200000;
    e->mask_check = 1;
    pr_setup(&e->game, PR_DECKS[0], PR_DECKS[1], seed, PR_DEFAULT_LOCKOUT, PR_TIEBREAK_ABSOLUTE);
    royale_seed(e, seed);
    c_reset(e);
    return t;
}

static void test_layout(void) {
    /* SPEC §16.4 (v0.3): 64 x 11 entity rows, integer card ids, 128-wide multi-hots, tower troops */
    CHECK_EQ(PR_OBS_C, 25);
    CHECK_EQ(PR_OBS_ENT_N, 64);
    CHECK_EQ(PR_OBS_ENT_F, 11);
    CHECK_EQ(PR_CARD_SLOTS, 128);
    CHECK_EQ(PR_OBS_SCALAR_SIZE, 1 + 4 + 4 + 1 + 4 + 4 + 6 + 2 + 2 + 128 + 1 + 4 + 128 + 1 + 4 + 4);
    CHECK_EQ(PR_OBS_SCALAR_SIZE, 298);
    CHECK_EQ(PR_OBS_SIZE, 25 * 576 + 64 * 11 + 298 + 2305);
    CHECK_EQ(PR_SC_OWN_TT, PR_SC_OPP_ELIXIR_UB + 1);
    CHECK_EQ(PR_SC_ENEMY_TT, PR_SC_OWN_TT + 4);
    int end = 0;
    for (int i = 0; i < PR_OBS_N_FIELDS; i++) { /* contiguous, in the SPEC §9.3 order */
        CHECK_EQ(PR_OBS_FIELDS[i].offset, end);
        end += PR_OBS_FIELDS[i].len;
    }
    CHECK_EQ(end, PR_OBS_SCALAR_SIZE);
}

static void test_reset_obs_and_lockout(void) {
    TEnv *t = t_env(2, PR_BOT_NOOP, 0, 1);
    for (int r = 0; r < 2; r++) {
        const float *o = t->obs + r * PR_OBS_SIZE;
        const float *m = o + PR_OBS_MASK_OFFSET;
        CHECK(m[0] == 1.0f);
        float s = 0;
        for (int a = 1; a < PR_N_ACTIONS; a++) s += m[a];
        CHECK(s == 0.0f);
        CHECK(o[PR_OBS_SCALAR_OFFSET + PR_SC_ELIXIR] == 16800.0f / 28000.0f);
        CHECK(o[PR_OBS_SCALAR_OFFSET + PR_SC_LOCKOUT] == 1.0f);
        for (int i = 0; i < PR_OBS_SPATIAL_SIZE; i++)
            if (o[i] < 0.0f || o[i] > 1.0f) { CHECK(0); break; }
    }
    for (int k = 0; k < 9; k++) c_step(&t->env);
    float s = 0;
    for (int a = 1; a < PR_N_ACTIONS; a++) s += t->obs[PR_OBS_MASK_OFFSET + a];
    CHECK(s > 0.0f); /* tick 90: playable */
    free(t);
}

static void test_mask_equals_engine_and_selfcheck(void) {
    TEnv *t = t_env(2, PR_BOT_NOOP, 0, 2);
    PrRng r;
    pr_rng_seed(&r, 5, 1);
    uint8_t m[PR_N_ACTIONS];
    for (int step = 0; step < 400; step++) {
        for (int row = 0; row < 2; row++) {
            const float *o = t->obs + row * PR_OBS_SIZE + PR_OBS_MASK_OFFSET;
            pr_legal_mask(&t->env.game.st, row, m);
            for (int a = 0; a < PR_N_ACTIONS; a++)
                if ((o[a] > 0.5f) != (m[a] != 0)) { CHECK(0); break; }
            t->acts[row] = 0;
            if (pr_rng_below(&r, 4) == 0) {
                int n = 0;
                for (int a = 1; a < PR_N_ACTIONS; a++) n += m[a];
                if (n) {
                    int k = (int)pr_rng_below(&r, (uint32_t)n);
                    for (int a = 1; a < PR_N_ACTIONS; a++)
                        if (m[a] && k-- == 0) { t->acts[row] = a; break; }
                }
            }
        }
        c_step(&t->env);
        CHECK(t->rew[0] + t->rew[1] == 0.0f);
    }
    CHECK_EQ(t->env.ep_mask_bad, 0);
    CHECK_EQ(t->env.log.illegal_actions + t->env.ep_illegal, 0);
    /* the self-check does detect a corrupted entry */
    float *o = t->obs + PR_OBS_MASK_OFFSET;
    o[7] = 1.0f - o[7];
    CHECK_EQ(pr_obs_mask_mismatches(&t->env.game.st, 0, o), 1);
    free(t);
}

static void test_illegal_counted_and_noop(void) {
    TEnv *t = t_env(2, PR_BOT_NOOP, 0, 3);
    t->acts[0] = 1 + 0 * 576 + 16 * 18 + 9; /* locked out anyway */
    t->acts[1] = 5000;                       /* out of range */
    c_step(&t->env);
    CHECK_EQ(t->env.ep_illegal, 2);
    CHECK_EQ(t->env.game.st.plays[0], 0);
    free(t);
}

static void test_terminal_autoreset_and_log(void) {
    TEnv *t = t_env(2, PR_BOT_NOOP, 0, 4);
    int steps = 0;
    t->acts[0] = t->acts[1] = 0;
    while (!t->term[0] && steps < 700) {
        c_step(&t->env);
        steps++;
    }
    CHECK_EQ(steps, 600);
    CHECK(t->term[1] == 1);
    CHECK(t->rew[0] == 0.0f && t->rew[1] == 0.0f);
    Log *l = &t->env.log;
    CHECK(l->n == 1.0f && l->draw == 1.0f && l->tiebreak == 1.0f && l->overtime == 1.0f);
    CHECK(l->episode_length == 600.0f && l->score == 0.5f);
    CHECK_EQ(t->env.game.st.tick, 0); /* the returned obs belong to the new match */
    CHECK(t->obs[PR_OBS_SCALAR_OFFSET + PR_SC_TICK] == 0.0f);
    free(t);
}

static void test_single_agent_bots_and_sides(void) {
    for (int bot = PR_BOT_NOOP; bot <= PR_BOT_HEURISTIC; bot++) {
        for (int side = 0; side < 2; side++) {
            TEnv *t = t_env(1, bot, side, 10 + (uint64_t)bot);
            int steps = 0;
            while (!t->term[0] && steps < 700) {
                t->acts[0] = 0;
                c_step(&t->env);
                steps++;
            }
            Log *l = &t->env.log;
            CHECK(l->n == 1.0f);
            CHECK(l->illegal_actions == 0.0f);
            float learner_plays = side == 0 ? l->plays_0 : l->plays_1;
            float bot_plays = side == 0 ? l->plays_1 : l->plays_0;
            CHECK(learner_plays == 0.0f);
            if (bot == PR_BOT_NOOP) CHECK(bot_plays == 0.0f);
            else CHECK(bot_plays > 0.0f);
            if (bot == PR_BOT_HEURISTIC) CHECK(l->learner_score == 0.0f); /* a passive learner loses */
            free(t);
        }
    }
}

static void test_obs_rotation(void) {
    /* team 0's own view of its unit equals team 1's view of the rotated enemy */
    TEnv *t = t_env(2, PR_BOT_NOOP, 0, 5);
    uint32_t ids[4];
    pr_debug_spawn(&t->env.game, 0, PR_CARD_KNIGHT, 5500, 20500, 1, ids, 4);
    royale_write_obs(&t->env);
    const float *e0 = t->obs + PR_OBS_ENTITY_OFFSET;
    const float *e1 = t->obs + PR_OBS_SIZE + PR_OBS_ENTITY_OFFSET + PR_OBS_ENT_HALF * PR_OBS_ENT_F;
    CHECK(e0[0] == (float)(PR_CARD_KNIGHT + 1) && e1[0] == (float)(PR_CARD_KNIGHT + 1)); /* id + 1 */
    CHECK(e0[1] + e1[1] == 1.0f);
    CHECK(e0[2] + e1[2] == 1.0f);
    CHECK(e0[3] == 1.0f && e0[5] == 0.0f && e0[9] == 0.0f);
    for (int k = 1; k < PR_OBS_ENT_N; k++) /* one entity only: own slot 0 / enemy slot 32 */
        if (k != PR_OBS_ENT_HALF) CHECK(t->obs[PR_OBS_SIZE + PR_OBS_ENTITY_OFFSET + k * PR_OBS_ENT_F] == 0.0f);
    /* spatial: own ground count on own tile (5, 20) for team 0, enemy ground on (12, 11) for team 1 */
    CHECK(t->obs[PR_CH_GROUND * 576 + 20 * 18 + 5] == 0.25f);
    CHECK(t->obs[PR_OBS_SIZE + (PR_CH_PER_SIDE + PR_CH_GROUND) * 576 + 11 * 18 + 12] == 0.25f);
    free(t);
}

static void test_tower_troop_scalars(void) {
    /* SPEC §16.6.16: own / enemy tower-troop one-hots (princess, cannoneer, dagger_duchess, royal_chef) */
    TEnv *t = (TEnv *)calloc(1, sizeof(TEnv));
    Royale *e = &t->env;
    e->observations = t->obs;
    e->actions = t->acts;
    e->rewards = t->rew;
    e->terminals = t->term;
    e->num_agents = 2;
    e->frame_skip = 10;
    pr_setup_ex(&e->game, PR_DECKS[3], PR_DECKS[4], 3, PR_DEFAULT_LOCKOUT, PR_TIEBREAK_ABSOLUTE, 1, 3);
    royale_seed(e, 3);
    c_reset(e);
    for (int r = 0; r < 2; r++) {
        const float *sc = t->obs + r * PR_OBS_SIZE + PR_OBS_SCALAR_OFFSET;
        int own = r == 0 ? 1 : 3, enemy = r == 0 ? 3 : 1;
        for (int k = 0; k < 4; k++) {
            CHECK(sc[PR_SC_OWN_TT + k] == (k == own ? 1.0f : 0.0f));
            CHECK(sc[PR_SC_ENEMY_TT + k] == (k == enemy ? 1.0f : 0.0f));
        }
        for (int s2 = 0; s2 < 4; s2++) { /* hand ids: card + 1 of the dealt hand */
            int c = e->game.st.hand[r][s2];
            CHECK(sc[PR_SC_HAND + s2] == (float)(c + 1));
        }
        CHECK(sc[PR_SC_NEXT] == (float)(e->game.st.queue[r][0] + 1));
    }
    /* the Cannoneer / Chef towers carry their own hp (tower ladder, SPEC §16.3) */
    CHECK_EQ(pr_get(&e->game.st, e->game.st.tower_id[0][1])->max_hp, 2616);
    CHECK_EQ(pr_get(&e->game.st, e->game.st.tower_id[1][2])->max_hp, 2703);
    CHECK_EQ(pr_get(&e->game.st, e->game.st.tower_id[1][0])->max_hp, 4824);
    free(t);
}

static void test_heuristic_vs_random(void) {
    /* the heuristic bot is a real sparring partner: it beats the random bot */
    int wins = 0, games = 6;
    for (int gi = 0; gi < games; gi++) {
        PrGame *g = t_game_decks(100 + (uint64_t)gi, PR_DECKS[gi % 3], PR_DECKS[(gi + 1) % 3], PR_DEFAULT_LOCKOUT);
        PrBot h, rnd;
        int ht = gi % 2;
        pr_bot_init(&h, PR_BOT_HEURISTIC, 0, 7 + (uint64_t)gi);
        pr_bot_init(&rnd, PR_BOT_RANDOM, 200000, 9 + (uint64_t)gi);
        while (!g->st.over) {
            for (int team = 0; team < 2; team++) {
                int a = pr_bot_act(team == ht ? &h : &rnd, &g->st, team);
                if (a) {
                    int slot = (a - 1) / PR_N_TILES, cell = (a - 1) % PR_N_TILES;
                    CHECK_EQ(pr_queue_play(&g->st, team, slot, cell % 18, cell / 18), PR_OK);
                }
            }
            t_ticks(g, 10);
        }
        wins += g->st.result[ht] > 0;
        free(g);
    }
    printf("    heuristic beat random %d/%d\n", wins, games);
    CHECK(wins >= games - 1);
}

/* SPEC v0.2.1 §14.6: tiebreak = 1 only if the end-of-overtime comparison ran (the auditor's
 * tick-5999 King kill and crown lead are not tiebreaks); learner_win / dropped_plays logged. */
static void test_log_tiebreak_and_learner_keys(void) {
    TEnv *t = t_env(2, PR_BOT_NOOP, 0, 6);
    PrState *st = &t->env.game.st;
    t->env.frame_skip = 1;
    st->tick = 5999;
    PrEntity *k1 = pr_get(st, st->tower_id[1][0]);
    k1->dmg_in = k1->hp; /* team 1's King falls in this tick's Resolve */
    c_step(&t->env);
    Log *l = &t->env.log;
    CHECK(t->term[0] == 1 && l->n == 1.0f);
    CHECK(l->tiebreak == 0.0f && l->overtime == 1.0f && l->win_0 == 1.0f);
    CHECK(l->learner_win == 1.0f && l->learner_score == 1.0f && l->dropped_plays == 0.0f);
    free(t);
    t = t_env(2, PR_BOT_NOOP, 0, 7); /* crowns differ at the last tick: OVERTIME_CROWNS */
    t->env.frame_skip = 1;
    t->env.game.st.tick = 5999;
    t->env.game.st.crowns[1] = 1;
    c_step(&t->env);
    CHECK(t->env.log.n == 1.0f && t->env.log.tiebreak == 0.0f && t->env.log.win_1 == 1.0f);
    CHECK(t->env.log.learner_win == 0.0f && t->env.log.learner_score == 0.0f);
    free(t);
    /* the comparison itself: a decisive tiebreak and an exact tie both count */
    for (int tie = 0; tie < 2; tie++) {
        t = t_env(2, PR_BOT_NOOP, 0, 8);
        t->env.frame_skip = 1;
        t->env.game.st.tick = 5999;
        if (!tie) pr_get(&t->env.game.st, t->env.game.st.tower_id[0][1])->hp -= 100;
        c_step(&t->env);
        CHECK(t->env.log.n == 1.0f && t->env.log.tiebreak == 1.0f);
        CHECK(t->env.log.draw == (tie ? 1.0f : 0.0f));
        free(t);
    }
    /* 1-agent mode: the learner keys follow the policy's team */
    t = t_env(1, PR_BOT_NOOP, 1, 9);
    t->env.frame_skip = 1;
    t->env.game.st.tick = 5999;
    PrEntity *k0 = pr_get(&t->env.game.st, t->env.game.st.tower_id[0][0]);
    k0->dmg_in = k0->hp; /* team 0 (the bot) loses: the learner (team 1) wins */
    t->acts[0] = 0;
    c_step(&t->env);
    CHECK(t->env.log.win_0 == 0.0f && t->env.log.learner_win == 1.0f && t->env.log.learner_score == 1.0f);
    CHECK(t->env.log.learner_return == 1.0f && t->env.log.episode_return == -1.0f);
    free(t);
}

int main(void) {
    printf("test_env\n");
    RUN(test_layout);
    RUN(test_reset_obs_and_lockout);
    RUN(test_mask_equals_engine_and_selfcheck);
    RUN(test_illegal_counted_and_noop);
    RUN(test_terminal_autoreset_and_log);
    RUN(test_single_agent_bots_and_sides);
    RUN(test_obs_rotation);
    RUN(test_tower_troop_scalars);
    RUN(test_heuristic_vs_random);
    RUN(test_log_tiebreak_and_learner_keys);
    TEST_END();
}
