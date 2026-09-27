/* bench.c -- engine throughput (SPEC §11: >= 20,000 ticks/s on one core).
 *
 * Pass 1 records, per match, the plays of two random bots that pick a legal action
 * with probability p per tick (masks are not engine time). Pass 2 replays the recorded
 * plays on fresh games and times only the engine: pr_queue_play + pr_tick. */
/* clock_gettime / CLOCK_MONOTONIC under -std=c99 (glibc hides POSIX otherwise). 200112L
 * rather than 199309L: macOS's headers then still declare C99's snprintf. */
#define _POSIX_C_SOURCE 200112L
#include <time.h>

#include "pr_test.h"

typedef struct { int32_t tick; int8_t team, slot, tx, ty; } Rec;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    int matches = argc > 1 ? atoi(argv[1]) : 40;
    uint32_t num = 1, den = argc > 2 ? (uint32_t)atoi(argv[2]) : 40; /* 40: ~one play per 2 s per team */
    static Rec recs[40000];
    static uint8_t mask[PR_N_ACTIONS];
    static int legal[PR_N_ACTIONS];
    PrGame *g = (PrGame *)calloc(1, sizeof(PrGame));
    long long total_ticks = 0, total_ent = 0, total_plays = 0;
    double engine_s = 0.0;
    int max_ent = 0;
    for (int m = 0; m < matches; m++) {
        int nrec = 0;
        pr_setup(g, NULL, NULL, 100 + (uint64_t)m, PR_DEFAULT_LOCKOUT, PR_TIEBREAK_ABSOLUTE);
        PrRng bot;
        pr_rng_seed(&bot, 900 + (uint64_t)m, 3);
        while (!g->st.over) {
            for (int team = 0; team < 2; team++) {
                if (pr_rng_below(&bot, den) >= num) continue;
                pr_legal_mask(&g->st, team, mask);
                int n = 0;
                for (int a = 1; a < PR_N_ACTIONS; a++)
                    if (mask[a]) legal[n++] = a;
                if (!n) continue;
                int a = legal[pr_rng_below(&bot, (uint32_t)n)];
                int slot = (a - 1) / PR_N_TILES, cell = (a - 1) % PR_N_TILES;
                pr_queue_play(&g->st, team, slot, cell % 18, cell / 18);
                if (nrec < 40000) {
                    Rec r = {g->st.tick, (int8_t)team, (int8_t)slot, (int8_t)(cell % 18), (int8_t)(cell / 18)};
                    recs[nrec++] = r;
                }
            }
            pr_tick(g);
        }
        uint64_t want = pr_hash(&g->st);
        /* pass 2: timed replay */
        pr_setup(g, NULL, NULL, 100 + (uint64_t)m, PR_DEFAULT_LOCKOUT, PR_TIEBREAK_ABSOLUTE);
        int ri = 0;
        double t0 = now_s();
        while (!g->st.over) {
            while (ri < nrec && recs[ri].tick == g->st.tick) {
                pr_queue_play(&g->st, recs[ri].team, recs[ri].slot, recs[ri].tx, recs[ri].ty);
                ri++;
            }
            pr_tick(g);
            total_ent += g->st.n_ent;
            if (g->st.n_ent > max_ent) max_ent = g->st.n_ent;
        }
        engine_s += now_s() - t0;
        if (pr_hash(&g->st) != want) {
            fprintf(stderr, "replay diverged in match %d\n", m);
            return 1;
        }
        total_ticks += g->st.tick;
        total_plays += g->st.plays[0] + g->st.plays[1];
    }
    printf("bench: %d matches, %lld ticks, %lld plays, mean entities %.1f (max %d)\n", matches, total_ticks,
           total_plays, (double)total_ent / (double)total_ticks, max_ent);
    printf("bench: engine %.0f ticks/s (%.2f us/tick), path fields built %llu\n",
           (double)total_ticks / engine_s, 1e6 * engine_s / (double)total_ticks,
           (unsigned long long)g->pc.field_builds);
    free(g);
    return 0;
}
