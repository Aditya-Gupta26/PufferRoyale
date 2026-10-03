/*
 * royale.h -- the PufferLib 3.0 environment around the engine (SPEC §9, §19).
 *
 * One Royale = one match. Agent rows:
 *   num_agents == 2  row 0 = team 0, row 1 = team 1 (self-play)
 *   num_agents == 1  row 0 = the learner's team (learner_side 0 / 1, or drawn per episode
 *                    from the env's side stream); the other team is the scripted opponent.
 * One c_step = frame_skip engine ticks. Actions (own frame) are queued on the first tick of the
 * step; an action the engine refuses is a no-op and is counted in illegal_actions. Each row
 * decodes its action through its placement grid (SPEC §19.4; grid 1 = the 2305 fine actions,
 * grid g > 1 = Discrete(1 + 4 B) coarse blocks, played on the block's representative tile).
 * On the step a match ends: terminals = 1 for every row, the per-episode Log is accumulated, and
 * the env re-deals within the same c_step, so the observations returned belong to the new
 * match. truncations are never set.
 *
 * Rewards (SPEC §19.1, reward v2): r_0 = F + result_0 on the ending step, F = gamma * Phi_new -
 * Phi_prev, with the potential Phi of team 0 (towers, crowns, leaked elixir, plays; annealed by
 * a step counter that is never reset) stored from the previous step of the same match and 0 at
 * the terminal state; r_1 = -r_0. All weights 0 (the default) = the terminal +1/-1/0 only.
 *
 * Randomness: the game RNG (deck shuffles, random decks) lives in the engine state; the
 * scripted bot, the learner-side draw and the deck sampler (SPEC §19.5) use their own PCG32
 * streams. An inactive sampler draws nothing.
 */
#ifndef ROYALE_H
#define ROYALE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pr_engine.h"
#include "pr_bots.h"
#include "pr_obs.h"
#include "pr_render.h"

/* Log: floats only (vec_log sums it float-wise and divides by n); `n` must be last. */
typedef struct Log {
    float episode_return;   /* team 0's summed reward over the episode */
    float episode_length;   /* env steps */
    float score;            /* team 0: win 1 / draw 0.5 / loss 0 */
    float perf;             /* = score */
    float win_0;
    float win_1;
    float draw;
    float overtime;         /* the match entered overtime (went past tick 3600) */
    float tiebreak;         /* the end-of-overtime comparison ran (SPEC §14.6) */
    float crowns_0;
    float crowns_1;
    float leaked_0;         /* elixir */
    float leaked_1;
    float plays_0;
    float plays_1;
    float illegal_actions;
    float spawn_overflow;
    float learner_return;   /* SPEC §14.6: the learner's summed reward (1-agent mode: the
                             * policy's team; self-play: team 0) */
    float learner_score;    /* the learner's win 1 / draw 0.5 / loss 0 (sweep metric) */
    float learner_win;      /* the learner won */
    float dropped_plays;    /* queued plays dropped at Upkeep validation (0 in env play) */
    float match_ticks;
    float mask_mismatch;    /* mask self-check failures (only when mask_check is on) */
    float n;
} Log;

enum { PR_RENDER_NONE = 0, PR_RENDER_ANSI = 1, PR_RENDER_HUMAN = 2 };

#define PR_SIDE_STREAM 0x534944u /* learner-side draws */
#define PR_DECK_STREAM 0x44434Bu /* deck-sampler draws (SPEC §19.5) */

/* ------------------------------------------------------------------ placement grid (SPEC §19.4) */

#define ROYALE_GRID_ROWS(g) ((PR_TILES_Y + (g) - 1) / (g))
#define ROYALE_GRID_COLS(g) ((PR_TILES_X + (g) - 1) / (g))
#define ROYALE_GRID_ACTIONS(g) (1 + 4 * ROYALE_GRID_ROWS(g) * ROYALE_GRID_COLS(g))

static inline int royale_grid_ok(int g) { return g == 1 || g == 2 || g == 4; }

/* The fine action a grid-g action `a` plays for `team` in the current state: 0 for the no-op,
 * -1 when `a` is out of range or no tile of its block is legal (the play is refused). Block b of
 * slot s covers own-frame tiles [g bx, min(18, g bx + g)) x [g by, min(32, g by + g)); the
 * representative is its tile legal for (team, s) -- exactly the fine mask, pr_check_play --
 * nearest the block centre (squared doubled distance), ties -> smaller ty, then smaller tx.
 * g = 1: the identity on legal actions. */
static inline int royale_coarse_to_fine(const PrState *st, int team, int a, int g) {
    if (a == 0) return 0;
    if (!royale_grid_ok(g) || a < 0 || a >= ROYALE_GRID_ACTIONS(g)) return -1;
    int cols = ROYALE_GRID_COLS(g), nb = ROYALE_GRID_ROWS(g) * cols;
    int slot = (a - 1) / nb, b = (a - 1) % nb, by = b / cols, bx = b % cols;
    int x0 = g * bx, x1 = PR_MIN(PR_TILES_X, x0 + g), y0 = g * by, y1 = PR_MIN(PR_TILES_Y, y0 + g);
    /* the refusals that do not depend on the tile (game over, slot, lockout, elixir) */
    int err = pr_check_play(st, team, slot, x0, y0);
    if (err != PR_OK && err != PR_ERR_ILLEGAL_POSITION) return -1;
    PrLegalCtx ctx; /* the rest of pr_check_play: the card's tile legality */
    pr_legal_ctx(st, team, &ctx);
    int card = st->hand[team][slot], best = -1, best_d = 0;
    for (int ty = y0; ty < y1; ty++)
        for (int tx = x0; tx < x1; tx++) {
            if (!pr_card_tile_legal_ctx(&ctx, team, card, tx, ty)) continue;
            int dx = 2 * tx + 1 - (x0 + x1), dy = 2 * ty + 1 - (y0 + y1), d = dx * dx + dy * dy;
            if (best < 0 || d < best_d) {
                best = ty * PR_TILES_X + tx;
                best_d = d;
            }
        }
    return best < 0 ? -1 : 1 + slot * PR_N_TILES + best;
}

/* ------------------------------------------------------------------ deck sampler (SPEC §19.5) */

#define PR_DECK_POOL_MAX 256
#define PR_DECK_HELDOUT_MAX 1024

/* card sets are 64-bit masks (bit c = card c), as the engine's seen_mask */
typedef char pr_deck_mask_fits[PR_N_CARDS <= 64 ? 1 : -1];

typedef struct RoyaleDecks {
    int active;                            /* deck_pool non-empty or random_deck_frac > 0 */
    int mirror;                            /* deck_draw "mirror": team 1 takes team 0's deck */
    int n_pool, n_heldout;
    uint64_t random_thr;                   /* P(random deck) = random_thr / 2^32 */
    uint64_t pool_cum[PR_DECK_POOL_MAX];   /* cumulative pool weights scaled to 2^32 (last = 2^32) */
    int8_t pool[PR_DECK_POOL_MAX][8];      /* installed card order (a preset's own order, else ascending) */
    uint64_t heldout[PR_DECK_HELDOUT_MAX]; /* held-out card sets */
    int64_t pool_count[PR_DECK_POOL_MAX];  /* seats dealt pool deck i (since construction) */
    int64_t random_count, rejected_count;  /* seats dealt a random deck; held-out redraws */
    PrRng rng;
} RoyaleDecks;

static inline uint64_t royale_deck_mask(const int8_t d[8]) {
    uint64_t m = 0;
    for (int k = 0; k < 8; k++) m |= (uint64_t)1 << d[k];
    return m;
}

static inline int royale_is_heldout(const RoyaleDecks *d, uint64_t m) {
    for (int i = 0; i < d->n_heldout; i++)
        if (d->heldout[i] == m) return 1;
    return 0;
}

/* One seat's deck into out; counted. Returns the pool index, or -1 for a random deck: with
 * probability random_deck_frac 8 distinct cards uniformly (partial Fisher-Yates; installed
 * ascending), redrawn while the set is held out; otherwise a pool deck with probability ~ weight,
 * in its stored order. */
static inline int royale_draw_deck(RoyaleDecks *d, int8_t out[8]) {
    if ((uint64_t)pr_rng_next(&d->rng) < d->random_thr) {
        uint64_t m;
        for (;;) {
            int8_t pool[PR_N_CARDS];
            for (int c = 0; c < PR_N_CARDS; c++) pool[c] = (int8_t)c;
            m = 0;
            for (int k = 0; k < 8; k++) {
                int j = k + (int)pr_rng_below(&d->rng, (uint32_t)(PR_N_CARDS - k));
                int8_t tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp;
                m |= (uint64_t)1 << pool[k];
            }
            if (!royale_is_heldout(d, m)) break;
            d->rejected_count++;
        }
        for (int c = 0, k = 0; c < PR_N_CARDS; c++)
            if ((m >> c) & 1u) out[k++] = (int8_t)c;
        d->random_count++;
        return -1;
    }
    uint64_t u = pr_rng_next(&d->rng);
    int i = 0;
    while (i < d->n_pool - 1 && u >= d->pool_cum[i]) i++;
    memcpy(out, d->pool[i], 8);
    d->pool_count[i]++;
    return i;
}

/* Both seats' decks for one deal: team 0 draws, then team 1 (mirror: a copy, counted again). */
static inline void royale_draw_pair(RoyaleDecks *d, int8_t d0[8], int8_t d1[8]) {
    int i = royale_draw_deck(d, d0);
    if (!d->mirror) {
        royale_draw_deck(d, d1);
        return;
    }
    memcpy(d1, d0, 8);
    if (i < 0) d->random_count++;
    else d->pool_count[i]++;
}

/* ------------------------------------------------------------------ the env */

typedef struct Royale {
    Log log;
    float *observations;
    int *actions;
    float *rewards;
    unsigned char *terminals;
    int num_agents;
    int frame_skip;
    int learner_cfg;          /* 0, 1, or -1 = random per episode */
    int learner;              /* team of row 0 (single-agent mode) */
    int opponent;             /* PR_BOT_* */
    uint32_t bot_ppm;
    int mask_check;
    int render_mode;
    int raylib_open;
    int steps;
    float ep_illegal;
    float ep_mask_bad;
    float ep_return[2];
    /* reward v2 (SPEC §19.1); all zero (calloc) = the terminal result only */
    double w_tower, w_crown, w_elixir, w_play;
    double cap_elixir, cap_play;  /* L_cap, P_cap */
    double reward_gamma;
    double anneal_steps;          /* N (0 = constant weights) */
    double step_offset;           /* n0 */
    int shaping;                  /* some weight is non-zero */
    int64_t env_steps;            /* c_steps since creation: never reset */
    double phi_prev;              /* Phi_new of the previous c_step of this match */
    /* placement grid (SPEC §19.4): the env's grid and each row's (<= 1 = fine actions) */
    int grid;
    int row_grid[2];
    RoyaleDecks decks;
    PrRng side_rng;
    PrBot bot;
    PrGame game;
} Royale;

static inline int royale_row_team(const Royale *env, int row) {
    return env->num_agents == 2 ? row : env->learner;
}

static inline void royale_write_obs(Royale *env) {
    for (int row = 0; row < env->num_agents; row++) {
        float *o = env->observations + (size_t)row * PR_OBS_SIZE;
        int team = royale_row_team(env, row);
        pr_obs_write(&env->game.st, team, o);
        if (env->mask_check) env->ep_mask_bad += (float)pr_obs_mask_mismatches(&env->game.st, team, o + PR_OBS_MASK_OFFSET);
    }
}

/* Start a new match (continuing every RNG stream); the sampler, when active, deals the decks. */
static inline void royale_new_episode(Royale *env) {
    if (env->decks.active) royale_draw_pair(&env->decks, env->game.st.deck[0], env->game.st.deck[1]);
    pr_new_match(&env->game.st);
    if (env->learner_cfg < 0) env->learner = (int)pr_rng_below(&env->side_rng, 2);
    else env->learner = env->learner_cfg;
    env->bot.last_play_tick = -1000000;
    env->steps = 0;
    env->ep_illegal = 0.0f;
    env->ep_mask_bad = 0.0f;
    env->ep_return[0] = env->ep_return[1] = 0.0f;
    env->phi_prev = 0.0;
}

/* Reseed every stream (game, bot, side, deck sampler) from `seed`. */
static inline void royale_seed(Royale *env, uint64_t seed) {
    pr_reseed(&env->game.st, seed);
    pr_bot_init(&env->bot, env->opponent, env->bot_ppm, seed);
    pr_rng_seed(&env->side_rng, seed, PR_SIDE_STREAM);
    pr_rng_seed(&env->decks.rng, seed, PR_DECK_STREAM);
}

static inline void c_reset(Royale *env) {
    royale_new_episode(env);
    for (int row = 0; row < env->num_agents; row++) {
        env->rewards[row] = 0.0f;
        env->terminals[row] = 0;
    }
    royale_write_obs(env);
}

static inline void royale_log_episode(Royale *env) {
    const PrState *st = &env->game.st;
    Log *l = &env->log;
    float r0 = (float)st->result[0], r1 = (float)st->result[1];
    float score0 = r0 > 0 ? 1.0f : (r0 == 0 ? 0.5f : 0.0f);
    int lt = royale_row_team(env, 0);
    float rl = (float)st->result[lt];
    l->episode_return += env->ep_return[0];
    l->episode_length += (float)env->steps;
    l->score += score0;
    l->perf += score0;
    l->win_0 += (float)(r0 > 0);
    l->win_1 += (float)(r1 > 0);
    l->draw += (float)(r0 == 0);
    l->overtime += (float)(st->tick > PR_TICKS_REGULATION);
    /* the end-of-overtime comparison ran: a TIEBREAK result, or a DRAW with both Kings
     * standing (the other DRAW is both Kings falling on one tick) */
    l->tiebreak += (float)(st->end_reason == PR_END_TIEBREAK ||
                           (st->end_reason == PR_END_DRAW && pr_tower_alive(st, 0, 0) && pr_tower_alive(st, 1, 0)));
    l->crowns_0 += (float)st->crowns[0];
    l->crowns_1 += (float)st->crowns[1];
    l->leaked_0 += (float)st->leaked[0] / (float)PR_ELIXIR_UNIT;
    l->leaked_1 += (float)st->leaked[1] / (float)PR_ELIXIR_UNIT;
    l->plays_0 += (float)st->plays[0];
    l->plays_1 += (float)st->plays[1];
    l->illegal_actions += env->ep_illegal;
    l->spawn_overflow += (float)st->spawn_overflow;
    l->learner_return += env->ep_return[lt];
    l->learner_score += rl > 0 ? 1.0f : (rl == 0 ? 0.5f : 0.0f);
    l->learner_win += (float)(rl > 0);
    l->dropped_plays += (float)st->dropped_plays;
    l->match_ticks += (float)st->tick;
    l->mask_mismatch += env->ep_mask_bad;
    l->n += 1.0f;
}

static inline double royale_clip(double v, double cap) { return v < -cap ? -cap : (v > cap ? cap : v); }

/* SPEC §19.1: the un-annealed potential of team 0 in the current state (team 1's = -this). */
static inline double royale_potential(const Royale *env) {
    const PrState *st = &env->game.st;
    double tw[2] = {0.0, 0.0}, leak[2];
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 3; i++) tw[t] += (double)pr_obs_tower_frac(st, t, i); /* destroyed = 0 */
        leak[t] = (double)st->leaked[t] / (double)PR_ELIXIR_UNIT;
    }
    return env->w_tower * (tw[0] - tw[1]) + env->w_crown * (double)(st->crowns[0] - st->crowns[1])
         - env->w_elixir * royale_clip(leak[0] - leak[1], env->cap_elixir)
         + env->w_play * royale_clip((double)st->plays[0] - (double)st->plays[1], env->cap_play);
}

/* SPEC §19.1: the anneal multiplier m_n of the env's n-th c_step. */
static inline double royale_anneal(const Royale *env, int64_t n) {
    if (env->anneal_steps <= 0.0) return 1.0;
    double m = 1.0 - (env->step_offset + (double)n) / env->anneal_steps;
    return m > 0.0 ? m : 0.0;
}

/* Queue one row's action; returns 0 when the engine refuses it. */
static inline int royale_queue_action(PrState *st, int team, int a, int grid) {
    if (grid > 1) { /* SPEC §19.4: play the block's representative tile */
        a = royale_coarse_to_fine(st, team, a, grid);
        if (a < 0) return 0;
    } else if (a < 0 || a >= PR_N_ACTIONS) {
        return 0;
    }
    int slot = (a - 1) / PR_N_TILES, cell = (a - 1) % PR_N_TILES;
    return pr_queue_play(st, team, slot, cell % PR_TILES_X, cell / PR_TILES_X) == PR_OK;
}

static inline void c_step(Royale *env) {
    PrState *st = &env->game.st;
    /* 1. actions (own-frame, queued for the first tick of this step) */
    for (int row = 0; row < env->num_agents; row++) {
        int a = env->actions[row];
        if (a == 0) continue;
        if (!royale_queue_action(st, royale_row_team(env, row), a, env->row_grid[row])) env->ep_illegal += 1.0f;
    }
    if (env->num_agents == 1) { /* the scripted opponent plays fine actions */
        int bt = 1 - env->learner;
        int a = pr_bot_act(&env->bot, st, bt);
        if (a > 0) {
            int slot = (a - 1) / PR_N_TILES, cell = (a - 1) % PR_N_TILES;
            pr_queue_play(st, bt, slot, cell % PR_TILES_X, cell / PR_TILES_X);
        }
    }
    /* 2. simulate */
    for (int k = 0; k < env->frame_skip && !st->over; k++) pr_tick(&env->game);
    env->steps++;
    /* 3. rewards (SPEC §19.1), in double: F = gamma Phi_new - Phi_prev with Phi_new = 0 at the
     * terminal state, so a finished match's discounted shaping sums to 0 even while annealing */
    int64_t n = env->env_steps++;
    int over = st->over;
    double phi = 0.0;
    if (env->shaping && !over) phi = royale_anneal(env, n) * royale_potential(env);
    double f = env->reward_gamma * phi - env->phi_prev;
    env->phi_prev = phi;
    float r[2];
    r[0] = (float)(f + (over ? (double)st->result[0] : 0.0));
    r[1] = 0.0f - r[0]; /* the exact negation, +0 (as in v0.4) for a zero reward */
    env->ep_return[0] += r[0];
    env->ep_return[1] += r[1];
    for (int row = 0; row < env->num_agents; row++) {
        env->rewards[row] = r[royale_row_team(env, row)];
        env->terminals[row] = (unsigned char)over;
    }
    /* 4. auto-reset within the same step (PufferLib 3.0 convention) */
    if (over) {
        royale_log_episode(env);
        royale_new_episode(env);
    }
    royale_write_obs(env);
}

static inline int royale_ansi(const Royale *env, char *buf, int cap) { return pr_render_ansi(&env->game.st, buf, cap); }

static inline void c_render(Royale *env) {
#ifdef PR_RAYLIB
    if (env->render_mode == PR_RENDER_HUMAN && pr_raylib_draw(&env->game.st, &env->raylib_open) >= 0) return;
    /* no display available: fall through to the text board */
#endif
    static char buf[PR_ANSI_BUF];
    royale_ansi(env, buf, PR_ANSI_BUF);
    fputs(buf, stdout);
    fflush(stdout);
}

static inline void c_close(Royale *env) {
#ifdef PR_RAYLIB
    pr_raylib_close(&env->raylib_open);
#else
    (void)env;
#endif
}

#endif /* ROYALE_H */
