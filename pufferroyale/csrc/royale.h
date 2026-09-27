/*
 * royale.h -- the PufferLib 3.0 environment around the engine (SPEC §9).
 *
 * One Royale = one match. Agent rows:
 *   num_agents == 2  row 0 = team 0, row 1 = team 1 (self-play)
 *   num_agents == 1  row 0 = the learner's team (learner_side 0 / 1, or drawn per episode
 *                    from the env's side stream); the other team is the scripted opponent.
 * One c_step = frame_skip engine ticks. Actions (Discrete 2305, own frame) are queued on
 * the first tick of the step; an action the engine refuses is a no-op and is counted in
 * illegal_actions. On the step a match ends: terminal reward +1/-1/0, terminals = 1 for
 * every row, the per-episode Log is accumulated, and the env re-deals within the same
 * c_step, so the observations returned belong to the new match. truncations are never set.
 *
 * Randomness: the game RNG (deck shuffles, random decks) lives in the engine state; the
 * scripted bot and the learner-side draw use their own PCG32 streams.
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
    float reward_tower;
    float reward_crown;
    int mask_check;
    int render_mode;
    int raylib_open;
    int steps;
    float ep_illegal;
    float ep_mask_bad;
    float ep_return[2];
    float prev_frac[2][3];
    int prev_crowns[2];
    PrRng side_rng;
    PrBot bot;
    PrGame game;
} Royale;

static inline int royale_row_team(const Royale *env, int row) {
    return env->num_agents == 2 ? row : env->learner;
}

static inline void royale_snapshot_towers(Royale *env) {
    const PrState *st = &env->game.st;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 3; i++) env->prev_frac[t][i] = pr_obs_tower_frac(st, t, i);
        env->prev_crowns[t] = st->crowns[t];
    }
}

static inline void royale_write_obs(Royale *env) {
    for (int row = 0; row < env->num_agents; row++) {
        float *o = env->observations + (size_t)row * PR_OBS_SIZE;
        int team = royale_row_team(env, row);
        pr_obs_write(&env->game.st, team, o);
        if (env->mask_check) env->ep_mask_bad += (float)pr_obs_mask_mismatches(&env->game.st, team, o + PR_OBS_MASK_OFFSET);
    }
}

/* Start a new match (continuing every RNG stream). */
static inline void royale_new_episode(Royale *env) {
    pr_new_match(&env->game.st);
    if (env->learner_cfg < 0) env->learner = (int)pr_rng_below(&env->side_rng, 2);
    else env->learner = env->learner_cfg;
    env->bot.last_play_tick = -1000000;
    env->steps = 0;
    env->ep_illegal = 0.0f;
    env->ep_mask_bad = 0.0f;
    env->ep_return[0] = env->ep_return[1] = 0.0f;
    royale_snapshot_towers(env);
}

/* Reseed every stream (game, bot, side) from `seed`. */
static inline void royale_seed(Royale *env, uint64_t seed) {
    pr_reseed(&env->game.st, seed);
    pr_bot_init(&env->bot, env->opponent, env->bot_ppm, seed);
    pr_rng_seed(&env->side_rng, seed, PR_SIDE_STREAM);
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

static inline void c_step(Royale *env) {
    PrState *st = &env->game.st;
    /* 1. actions (own-frame, queued for the first tick of this step) */
    for (int row = 0; row < env->num_agents; row++) {
        int a = env->actions[row];
        if (a == 0) continue;
        int team = royale_row_team(env, row);
        int ok = 0;
        if (a > 0 && a < PR_N_ACTIONS) {
            int slot = (a - 1) / PR_N_TILES, cell = (a - 1) % PR_N_TILES;
            ok = pr_queue_play(st, team, slot, cell % PR_TILES_X, cell / PR_TILES_X) == PR_OK;
        }
        if (!ok) env->ep_illegal += 1.0f;
    }
    if (env->num_agents == 1) {
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
    /* 3. rewards: terminal result + optional antisymmetric shaping */
    float r[2] = {0.0f, 0.0f};
    if (env->reward_tower != 0.0f || env->reward_crown != 0.0f) {
        float lost[2] = {0.0f, 0.0f};
        for (int t = 0; t < 2; t++)
            for (int i = 0; i < 3; i++) {
                float f = pr_obs_tower_frac(st, t, i);
                lost[t] += env->prev_frac[t][i] - f;
            }
        int dc0 = st->crowns[0] - env->prev_crowns[0], dc1 = st->crowns[1] - env->prev_crowns[1];
        r[0] = env->reward_tower * (lost[1] - lost[0]) + env->reward_crown * (float)(dc0 - dc1);
        r[1] = -r[0];
        royale_snapshot_towers(env);
    }
    int over = st->over;
    if (over) {
        r[0] += (float)st->result[0];
        r[1] += (float)st->result[1];
    }
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
