/* test_wp_env.c -- the SPEC §19 (v0.5 work package) env items in C: reward v2, the own-deck
 * observation, placement-grid decoding and the deck sampler (royale.h, pr_obs.h). */
#include <math.h>

#include "royale.h"
#include "pr_test.h"

typedef struct {
    Royale env;
    float obs[2 * PR_OBS_SIZE];
    int acts[2];
    float rew[2];
    unsigned char term[2];
} TEnv;

/* A self-play env (both rows are policies; bots drive them below), like binding.c's my_init with
 * the given reward config. The deck sampler stays inactive unless the caller fills env.decks
 * before t_env_start. */
static TEnv *t_env_new(void) {
    TEnv *t = (TEnv *)calloc(1, sizeof(TEnv));
    if (!t) { fprintf(stderr, "out of memory\n"); exit(2); }
    Royale *e = &t->env;
    e->observations = t->obs;
    e->actions = t->acts;
    e->rewards = t->rew;
    e->terminals = t->term;
    e->num_agents = 2;
    e->frame_skip = 10;
    e->opponent = PR_BOT_NOOP;
    e->learner_cfg = 0;
    e->reward_gamma = 1.0;
    e->cap_elixir = e->cap_play = 20.0;
    e->grid = e->row_grid[0] = e->row_grid[1] = 1;
    return t;
}

static void t_env_start(TEnv *t, const int8_t *d0, const int8_t *d1, uint64_t seed) {
    Royale *e = &t->env;
    e->shaping = e->w_tower != 0.0 || e->w_crown != 0.0 || e->w_elixir != 0.0 || e->w_play != 0.0;
    int8_t s0[8], s1[8];
    if (e->decks.active) {
        pr_rng_seed(&e->decks.rng, seed, PR_DECK_STREAM);
        royale_draw_pair(&e->decks, s0, s1);
        d0 = s0;
        d1 = s1;
    }
    pr_setup(&e->game, d0, d1, seed, PR_DEFAULT_LOCKOUT, PR_TIEBREAK_ABSOLUTE);
    royale_seed(e, seed);
    c_reset(e);
}

/* Both rows act with scripted bots (fine actions). */
static void t_bot_actions(TEnv *t, PrBot bots[2]) {
    for (int row = 0; row < 2; row++) t->acts[row] = pr_bot_act(&bots[row], &t->env.game.st, row);
}

/* ------------------------------------------------------------------ reward v2 (SPEC §19.1) */

/* The potential of team 0, written independently of royale_potential from the SPEC text. */
static double t_potential(const Royale *e) {
    const PrState *st = &e->game.st;
    double T[2] = {0, 0}, L[2], P[2], C[2];
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) T[k] += (double)pr_obs_tower_frac(st, k, i);
        C[k] = st->crowns[k];
        L[k] = st->leaked[k] / 2800.0;
        P[k] = st->plays[k];
    }
    double dl = fmin(fmax(L[0] - L[1], -e->cap_elixir), e->cap_elixir);
    double dp = fmin(fmax(P[0] - P[1], -e->cap_play), e->cap_play);
    return e->w_tower * (T[0] - T[1]) + e->w_crown * (C[0] - C[1]) - e->w_elixir * dl + e->w_play * dp;
}

/* Shaped bot matches: every reward equals the SPEC formula bit for bit, r1 = -r0 exactly, and each
 * finished match telescopes: sum_t gamma^t r0_t = gamma^(T-1) result0 (|.| <= 1e-4), also while
 * the weights anneal through 0 within a match. */
static void run_shaped(double gamma, double N, double n0, int matches, uint64_t seed, int *clipped) {
    TEnv *t = t_env_new();
    Royale *e = &t->env;
    e->w_tower = 0.3;
    e->w_crown = 0.2;
    e->w_elixir = 0.05;
    e->w_play = 0.02;
    e->cap_elixir = 2.0; /* small caps: the clip binds in bot matches */
    e->cap_play = 3.0;
    e->reward_gamma = gamma;
    e->anneal_steps = N;
    e->step_offset = n0;
    t_env_start(t, PR_DECKS[0], PR_DECKS[1], seed);
    PrBot bots[2];
    pr_bot_init(&bots[0], PR_BOT_HEURISTIC, 0, seed + 11);
    pr_bot_init(&bots[1], PR_BOT_RANDOM, 300000, seed + 12);
    double prev = 0.0, disc = 1.0, sum = 0.0;
    int64_t n = 0;
    int done = 0, len = 0, bad_formula = 0, bad_neg = 0;
    while (done < matches) {
        t_bot_actions(t, bots);
        c_step(e);
        double m = N > 0 ? fmax(0.0, 1.0 - (n0 + (double)n) / N) : 1.0;
        n++;
        CHECK_EQ(e->env_steps, n);
        float r0 = t->rew[0], r1 = t->rew[1];
        if (!(r1 == -r0) || (r0 == 0.0f && signbit(r1))) bad_neg++;
        if (!t->term[0]) { /* the state is s' (no re-deal): recompute the reward */
            double phi = m * t_potential(e);
            if ((float)(gamma * phi - prev) != r0) bad_formula++;
            double l = (e->game.st.leaked[0] - e->game.st.leaked[1]) / 2800.0;
            if (fabs(l) > e->cap_elixir || abs(e->game.st.plays[0] - e->game.st.plays[1]) > (int)e->cap_play) (*clipped)++;
            prev = phi;
        }
        sum += disc * (double)r0;
        disc *= gamma;
        len++;
        if (t->term[0]) {
            /* the terminal step t = T-1 carries result0 - Phi_prev: sum = gamma^(T-1) result0 */
            const Log *l = &e->log;
            double res = l->win_0 > 0.5f ? 1.0 : (l->win_1 > 0.5f ? -1.0 : 0.0);
            CHECK(fabs(sum - pow(gamma, len - 1) * res) <= 1e-4);
            memset(&e->log, 0, sizeof(e->log));
            CHECK(e->phi_prev == 0.0);
            prev = sum = 0.0;
            disc = 1.0;
            len = 0;
            done++;
        }
    }
    CHECK_EQ(bad_formula, 0);
    CHECK_EQ(bad_neg, 0);
    free(t);
}

static void test_reward_v2_formula_and_telescoping(void) {
    int clipped = 0;
    run_shaped(1.0, 0.0, 0.0, 3, 21, &clipped);       /* constant weights, gamma 1 */
    run_shaped(0.999, 0.0, 0.0, 3, 22, &clipped);     /* the trainer's gamma */
    run_shaped(0.99, 700.0, 0.0, 4, 23, &clipped);    /* anneals to 0 inside the second match */
    run_shaped(0.999, 900.0, 300.0, 3, 24, &clipped); /* resumed: offset n0 */
    CHECK(clipped > 0);                               /* the clip terms were exercised */
}

/* All weights 0: the rewards are exactly the v0.4 ones (terminal result only, +0 otherwise), with
 * any gamma and annealing. */
static void test_reward_zero_weights_bit_identical(void) {
    for (int variant = 0; variant < 2; variant++) {
        TEnv *t = t_env_new();
        t->env.reward_gamma = variant ? 0.9 : 1.0;
        t->env.anneal_steps = variant ? 50.0 : 0.0;
        t_env_start(t, PR_DECKS[2], PR_DECKS[3], 31 + (uint64_t)variant);
        PrBot bots[2];
        pr_bot_init(&bots[0], PR_BOT_HEURISTIC, 0, 5);
        pr_bot_init(&bots[1], PR_BOT_HEURISTIC, 0, 6);
        int ends = 0, bad = 0;
        for (int s = 0; s < 1500; s++) {
            t_bot_actions(t, bots);
            int8_t res[2] = {0, 0};
            c_step(&t->env);
            if (t->term[0]) {
                ends++;
                res[0] = t->env.log.win_0 > 0.5f ? 1 : (t->env.log.win_1 > 0.5f ? -1 : 0);
                res[1] = (int8_t)-res[0];
                memset(&t->env.log, 0, sizeof(t->env.log));
            }
            float v04[2] = {0.0f, 0.0f}; /* v0.4: r = 0.0f, += (float)result on the ending step */
            if (t->term[0]) {
                v04[0] += (float)res[0];
                v04[1] += (float)res[1];
            }
            if (memcmp(v04, t->rew, sizeof(v04)) != 0) bad++;
        }
        CHECK(ends >= 2);
        CHECK_EQ(bad, 0);
        free(t);
    }
}

/* The anneal counter: m_n from n0 + n, n counted over c_steps since creation (never reset). */
static void test_anneal_counter_never_resets(void) {
    TEnv *t = t_env_new();
    t->env.w_tower = 1.0;
    t->env.anneal_steps = 100.0;
    t->env.step_offset = 20.0;
    t_env_start(t, PR_DECKS[0], PR_DECKS[0], 3);
    for (int i = 0; i < 30; i++) c_step(&t->env);
    c_reset(&t->env);
    CHECK_EQ(t->env.env_steps, 30);
    CHECK(royale_anneal(&t->env, t->env.env_steps) == 1.0 - 50.0 / 100.0);
    CHECK(royale_anneal(&t->env, 80) == 0.0);
    CHECK(royale_anneal(&t->env, 500) == 0.0);
    t->env.anneal_steps = 0.0;
    CHECK(royale_anneal(&t->env, 500) == 1.0);
    free(t);
}

/* ------------------------------------------------------------------ own deck (SPEC §19.3) */

static void test_own_deck_field(void) {
    TEnv *t = t_env_new();
    t_env_start(t, PR_DECKS[5], PR_DECKS[6], 4);
    for (int row = 0; row < 2; row++) {
        const float *sc = t->obs + row * PR_OBS_SIZE + PR_OBS_SCALAR_OFFSET;
        const int8_t *deck = t->env.game.st.deck[row];
        float prev = 0.0f;
        for (int k = 0; k < 8; k++) {
            float v = sc[PR_SC_OWN_DECK + k];
            int found = 0;
            for (int j = 0; j < 8; j++) found |= v == (float)(deck[j] + 1);
            CHECK(found && v > prev); /* the own deck, ascending */
            prev = v;
        }
    }
    /* the hand / queue order (the hidden shuffle) and the opponent's deck do not enter */
    float before[8];
    memcpy(before, t->obs + PR_OBS_SCALAR_OFFSET + PR_SC_OWN_DECK, sizeof(before));
    int8_t order[8], other[8];
    for (int k = 0; k < 8; k++) {
        order[k] = PR_DECKS[5][7 - k];
        other[k] = PR_DECKS[2][k];
    }
    CHECK_EQ(pr_debug_set_hand(&t->env.game, 0, order), 0);
    CHECK_EQ(pr_debug_set_hand(&t->env.game, 1, other), 0);
    royale_write_obs(&t->env);
    CHECK(memcmp(before, t->obs + PR_OBS_SCALAR_OFFSET + PR_SC_OWN_DECK, sizeof(before)) == 0);
    free(t);
}

/* ------------------------------------------------------------------ placement grid (SPEC §19.4) */

/* Brute force over every action of grid g in the current state, against the exact fine mask. */
static void check_grid_state(const PrState *st, int team, int g, int *n_legal) {
    uint8_t mask[PR_N_ACTIONS];
    pr_legal_mask(st, team, mask);
    int cols = ROYALE_GRID_COLS(g), nb = ROYALE_GRID_ROWS(g) * cols, bad = 0;
    CHECK_EQ(royale_coarse_to_fine(st, team, 0, g), 0);
    CHECK_EQ(royale_coarse_to_fine(st, team, ROYALE_GRID_ACTIONS(g), g), -1);
    CHECK_EQ(royale_coarse_to_fine(st, team, -3, g), -1);
    for (int a = 1; a < ROYALE_GRID_ACTIONS(g); a++) {
        int s = (a - 1) / nb, b = (a - 1) % nb, by = b / cols, bx = b % cols;
        int x0 = g * bx, x1 = PR_MIN(18, x0 + g), y0 = g * by, y1 = PR_MIN(32, y0 + g);
        int best = -1, bd = 0;
        for (int ty = y0; ty < y1; ty++)
            for (int tx = x0; tx < x1; tx++) {
                if (!mask[1 + s * 576 + ty * 18 + tx]) continue;
                int dx = 2 * tx + 1 - x0 - x1, dy = 2 * ty + 1 - y0 - y1;
                if (best < 0 || dx * dx + dy * dy < bd) { /* row-major scan = the tie order */
                    best = 1 + s * 576 + ty * 18 + tx;
                    bd = dx * dx + dy * dy;
                }
            }
        int f = royale_coarse_to_fine(st, team, a, g);
        if (f != best) bad++;
        if (best > 0) {
            (*n_legal)++;
            int slot = (f - 1) / 576, cell = (f - 1) % 576;
            if (pr_check_play(st, team, slot, cell % 18, cell / 18) != PR_OK) bad++; /* accepted */
        }
        if (g == 1 && f != (mask[a] ? a : -1)) bad++; /* g = 1: the identity on legal actions */
    }
    CHECK_EQ(bad, 0);
}

static void test_grid_decoding_exhaustive_on_real_states(void) {
    TEnv *t = t_env_new();
    t_env_start(t, PR_DECKS[7], PR_DECKS[8], 41); /* pekka_bridge vs royal_hogs: every placement class */
    PrBot bots[2];
    pr_bot_init(&bots[0], PR_BOT_HEURISTIC, 0, 1);
    pr_bot_init(&bots[1], PR_BOT_RANDOM, 400000, 2);
    int legal = 0, states = 0;
    for (int s = 0; s < 900; s++) {
        if (s % 9 == 0) {
            for (int team = 0; team < 2; team++)
                for (int gi = 0; gi < 3; gi++) check_grid_state(&t->env.game.st, team, gi == 0 ? 1 : 2 * gi, &legal);
            states++;
        }
        t_bot_actions(t, bots);
        c_step(&t->env);
    }
    printf("    %d states, %d coarse-legal actions checked\n", states, legal);
    CHECK(legal > 10000);
    CHECK_EQ(ROYALE_GRID_ACTIONS(1), 2305);
    CHECK_EQ(ROYALE_GRID_ACTIONS(2), 577);
    CHECK_EQ(ROYALE_GRID_ACTIONS(4), 161);
    free(t);
}

/* A coarse row plays the representative tile; a refused coarse action is a counted no-op. */
static void test_grid_rows_in_the_env(void) {
    TEnv *t = t_env_new();
    t->env.grid = t->env.row_grid[0] = t->env.row_grid[1] = 4;
    t_env_start(t, PR_DECKS[0], PR_DECKS[0], 5);
    for (int s = 0; s < 9; s++) c_step(&t->env); /* past the lockout */
    PrState *st = &t->env.game.st;
    int a = -1, fine = -1;
    for (int c = 1; c < ROYALE_GRID_ACTIONS(4) && a < 0; c++) { /* a legal block whose corner tile is not */
        int f = royale_coarse_to_fine(st, 0, c, 4);
        int b = (c - 1) % 40, slot = (c - 1) / 40;
        if (f > 0 && pr_check_play(st, 0, slot, 4 * (b % 5), 4 * (b / 5)) != PR_OK) {
            a = c;
            fine = f;
        }
    }
    CHECK(a > 0);
    PrState *copy = (PrState *)malloc(sizeof(PrState));
    memcpy(copy, st, sizeof(PrState));
    CHECK(royale_queue_action(copy, 0, a, 4));
    const PrPlay *p = &copy->pending[0][copy->n_pending[0] - 1];
    CHECK_EQ(1 + p->slot * 576 + p->ty * 18 + p->tx, fine);
    free(copy);
    t->acts[0] = a;
    t->acts[1] = ROYALE_GRID_ACTIONS(4); /* out of range for grid 4 */
    int plays = st->plays[0];
    c_step(&t->env);
    CHECK_EQ(st->plays[0], plays + 1);
    CHECK(t->env.ep_illegal == 1.0f);
    /* a row switched to fine actions (row_grid 1) decodes 2305-actions as in v0.4 */
    t->env.row_grid[1] = 1;
    CHECK_EQ(royale_queue_action(st, 1, 2304, 1), pr_check_play(st, 1, 3, 17, 31) == PR_OK);
    free(t);
}

/* ------------------------------------------------------------------ deck sampler (SPEC §19.5) */

static void t_pool(RoyaleDecks *d, int n, const double *w, double frac) {
    double total = 0, acc = 0;
    for (int i = 0; i < n; i++) total += w[i];
    for (int i = 0; i < n; i++) {
        memcpy(d->pool[i], PR_DECKS[i], 8);
        for (int a = 1; a < 8; a++) /* ascending */
            for (int b = a; b > 0 && d->pool[i][b - 1] > d->pool[i][b]; b--) {
                int8_t x = d->pool[i][b]; d->pool[i][b] = d->pool[i][b - 1]; d->pool[i][b - 1] = x;
            }
        acc += w[i];
        d->pool_cum[i] = i == n - 1 ? (uint64_t)1 << 32 : (uint64_t)(acc / total * 4294967296.0);
    }
    d->n_pool = n;
    d->random_thr = (uint64_t)(frac * 4294967296.0 + 0.5);
    d->active = n > 0 || frac > 0;
}

static void test_deck_sampler_statistics(void) {
    RoyaleDecks *d = (RoyaleDecks *)calloc(1, sizeof(RoyaleDecks));
    double w[3] = {1.0, 2.0, 5.0};
    t_pool(d, 3, w, 0.25);
    d->heldout[0] = royale_deck_mask(PR_DECKS[5]); /* xbow held out */
    d->n_heldout = 1;
    pr_rng_seed(&d->rng, 7, PR_DECK_STREAM);
    int n = 40000;
    for (int i = 0; i < n / 2; i++) {
        int8_t a[8], b[8];
        royale_draw_pair(d, a, b);
        for (int k = 1; k < 8; k++) CHECK(a[k - 1] < a[k] && b[k - 1] < b[k]); /* 8 distinct, ascending */
        CHECK(royale_deck_mask(a) != d->heldout[0] && royale_deck_mask(b) != d->heldout[0]);
    }
    CHECK_EQ(d->pool_count[0] + d->pool_count[1] + d->pool_count[2] + d->random_count, n);
    double expect[4] = {0.75 * 1 / 8 * n, 0.75 * 2 / 8 * n, 0.75 * 5 / 8 * n, 0.25 * n};
    double got[4] = {(double)d->pool_count[0], (double)d->pool_count[1], (double)d->pool_count[2], (double)d->random_count};
    for (int i = 0; i < 4; i++) CHECK(fabs(got[i] - expect[i]) < 5.0 * sqrt(expect[i]));
    /* mirror: team 1 copies team 0, both seats counted */
    memset(d->pool_count, 0, sizeof(d->pool_count));
    d->random_count = 0;
    d->mirror = 1;
    for (int i = 0; i < 1000; i++) {
        int8_t a[8], b[8];
        royale_draw_pair(d, a, b);
        CHECK(memcmp(a, b, 8) == 0);
    }
    CHECK_EQ(d->pool_count[0] + d->pool_count[1] + d->pool_count[2] + d->random_count, 2000);
    CHECK(d->pool_count[0] % 2 == 0 && d->random_count % 2 == 0);
    free(d);
}

/* A random draw equal to a held-out deck is redrawn (and counted). */
static void test_deck_sampler_heldout_rejection(void) {
    RoyaleDecks *d = (RoyaleDecks *)calloc(1, sizeof(RoyaleDecks));
    t_pool(d, 0, NULL, 1.0);
    pr_rng_seed(&d->rng, 99, PR_DECK_STREAM);
    int8_t first[8], again[8];
    royale_draw_deck(d, first);
    d->heldout[d->n_heldout++] = royale_deck_mask(first);
    d->random_count = 0;
    pr_rng_seed(&d->rng, 99, PR_DECK_STREAM);
    royale_draw_deck(d, again);
    CHECK(memcmp(first, again, 8) != 0);
    CHECK_EQ(d->rejected_count, 1);
    CHECK_EQ(d->random_count, 1);
    free(d);
}

/* The sampler never touches the game RNG: a one-deck pool equals fixed (ascending) decks bit for
 * bit, and the same seed reproduces the same deals. */
static void test_deck_sampler_uses_its_own_stream(void) {
    int8_t asc[8];
    memcpy(asc, PR_DECKS[0], 8);
    for (int a = 1; a < 8; a++)
        for (int b = a; b > 0 && asc[b - 1] > asc[b]; b--) { int8_t x = asc[b]; asc[b] = asc[b - 1]; asc[b - 1] = x; }
    TEnv *fixed = t_env_new(), *pool = t_env_new();
    double w = 1.0;
    t_pool(&pool->env.decks, 1, &w, 0.0);
    t_env_start(fixed, asc, asc, 8);
    t_env_start(pool, NULL, NULL, 8);
    PrBot b1[2], b2[2];
    for (int k = 0; k < 2; k++) {
        pr_bot_init(&b1[k], PR_BOT_HEURISTIC, 0, 70 + (uint64_t)k);
        pr_bot_init(&b2[k], PR_BOT_HEURISTIC, 0, 70 + (uint64_t)k);
    }
    int diff = 0;
    for (int s = 0; s < 1300; s++) {
        t_bot_actions(fixed, b1);
        t_bot_actions(pool, b2);
        c_step(&fixed->env);
        c_step(&pool->env);
        diff += pr_hash(&fixed->env.game.st) != pr_hash(&pool->env.game.st);
        diff += memcmp(fixed->obs, pool->obs, sizeof(fixed->obs)) != 0;
    }
    CHECK_EQ(diff, 0);
    CHECK(pool->env.decks.pool_count[0] >= 6); /* construction + reset + 2 re-deals, both seats */
    free(fixed);
    free(pool);
    /* determinism: same seed, same random decks */
    TEnv *a = t_env_new(), *b = t_env_new();
    t_pool(&a->env.decks, 0, NULL, 1.0);
    t_pool(&b->env.decks, 0, NULL, 1.0);
    t_env_start(a, NULL, NULL, 12);
    t_env_start(b, NULL, NULL, 12);
    for (int k = 0; k < 5; k++) {
        CHECK(memcmp(a->env.game.st.deck, b->env.game.st.deck, 16) == 0);
        c_reset(&a->env);
        c_reset(&b->env);
    }
    free(a);
    free(b);
}

int main(void) {
    printf("test_wp_env\n");
    RUN(test_reward_v2_formula_and_telescoping);
    RUN(test_reward_zero_weights_bit_identical);
    RUN(test_anneal_counter_never_resets);
    RUN(test_own_deck_field);
    RUN(test_grid_decoding_exhaustive_on_real_states);
    RUN(test_grid_rows_in_the_env);
    RUN(test_deck_sampler_statistics);
    RUN(test_deck_sampler_heldout_rejection);
    RUN(test_deck_sampler_uses_its_own_stream);
    TEST_END();
}
