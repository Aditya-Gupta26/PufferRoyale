/* bench_stress.c -- worst-case engine throughput, best of 9 runs with a cold path cache each
 * time, per-phase breakdown. The target is >= 10,000 ticks/s (audit Phase C.1, SPEC §16.5):
 *   variant 0  the auditor's 216-Skeleton fight across the river (7 Skeleton Armies per side)
 *   variant 1  the same with 8 Cannons in the middle (non line-of-sight paths)
 *   variant 2  the SPEC §16 large battle: per side 4 Skeleton Armies, 2 Minion Hordes, 2 Goblin
 *              Gangs and a Witch (tests/spec/test_obs_v3.py::_stress_v3), 600 ticks */
#define _POSIX_C_SOURCE 200112L
#include <time.h>

#include "pr_test.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void scenario_v3(PrGame *g) {
    pr_setup(g, PR_DECKS[0], PR_DECKS[0], 0, 90, PR_TIEBREAK_ABSOLUTE);
    for (int team = 0; team < 2; team++) {
#define PX(x) (team == 0 ? (x) : PR_ARENA_W - (x))
#define PY(y) (team == 0 ? (y) : PR_ARENA_H - (y))
        const int32_t xs[4] = {2500, 6500, 11500, 15500};
        for (int k = 0; k < 4; k++) pr_debug_spawn(g, team, PR_CARD_SKELETON_ARMY, PX(xs[k]), PY(19500), 1, NULL, 0);
        for (int k = 0; k < 2; k++) {
            int32_t x = k ? 13500 : 4500;
            pr_debug_spawn(g, team, PR_CARD_MINION_HORDE, PX(x), PY(21000), 1, NULL, 0);
            pr_debug_spawn(g, team, PR_CARD_GOBLIN_GANG, PX(x), PY(22500), 1, NULL, 0);
        }
        pr_debug_spawn(g, team, PR_CARD_WITCH, PX(9000), PY(23000), 1, NULL, 0);
#undef PX
#undef PY
    }
    memset(&g->pc, 0, sizeof(g->pc));
}

static void scenario(PrGame *g, int buildings) {
    if (buildings == 2) {
        scenario_v3(g);
        return;
    }
    pr_setup(g, PR_DECKS[2], PR_DECKS[2], 1, 0, PR_TIEBREAK_ABSOLUTE);
    for (int k = 0; k < 7; k++) {
        pr_debug_spawn(g, 0, PR_CARD_SKELETON_ARMY, 3000 + (k % 4) * 4000, 20000 + (k / 4) * 3000, 1, NULL, 0);
        pr_debug_spawn(g, 1, PR_CARD_SKELETON_ARMY, 3000 + (k % 4) * 4000, 12000 - (k / 4) * 3000, 1, NULL, 0);
    }
    for (int k = 0; buildings && k < 4; k++) { /* buildings in the middle: non-LOS paths */
        pr_debug_spawn(g, 0, PR_CARD_CANNON, 6500 + k * 1500, 18500, 1, NULL, 0);
        pr_debug_spawn(g, 1, PR_CARD_CANNON, 6500 + k * 1500, 13500, 1, NULL, 0);
    }
    memset(&g->pc, 0, sizeof(g->pc));
}

int main(void) {
    PrGame *g = (PrGame *)calloc(1, sizeof(PrGame));
    if (!g) return 2;
    for (int variant = 0; variant < 3; variant++) {
        int ticks = variant == 2 ? 600 : 300;
        double best[6] = {1e9, 1e9, 1e9, 1e9, 1e9, 1e9}, tot_best = 1e9;
        unsigned long long builds = 0;
        int n0 = 0;
        for (int rep = 0; rep < 9; rep++) {
            scenario(g, variant);
            n0 = g->st.n_ent;
            PrState *st = &g->st;
            double T[6] = {0}, t0 = now_s();
            if (variant == 2) for (int i = 0; i < 20; i++) pr_tick(g); /* as the tester's warm-up */
            for (int i = 0; i < ticks && !st->over; i++) {
                double t = now_s(), u;
                int32_t tk = st->tick;
                pr_phase_upkeep(st); pr_phase_status(st); pr_phase_spawn(st);
                u = now_s(); T[0] += u - t; t = u;
                pr_phase_target(st); u = now_s(); T[1] += u - t; t = u;
                pr_phase_attack(st); u = now_s(); T[2] += u - t; t = u;
                pr_phase_move(st, &g->pc); u = now_s(); T[3] += u - t; t = u;
                pr_phase_collide(st); u = now_s(); T[4] += u - t; t = u;
                pr_phase_projectiles(st); pr_phase_effects(st); pr_phase_resolve(st); pr_phase_reap(st);
                st->tick = tk + 1;
                pr_judge(st, tk, 0);
                u = now_s(); T[5] += u - t;
            }
            double tot = now_s() - t0;
            for (int k = 0; k < 6; k++) if (T[k] < best[k]) best[k] = T[k];
            if (tot < tot_best) tot_best = tot;
            builds = (unsigned long long)g->pc.field_builds;
        }
        printf("stress variant %d (%d entities%s): best %.0f ticks/s (%.1f us/tick), field builds %llu\n", variant,
               n0, variant == 1 ? ", 8 Cannons" : variant == 2 ? ", SPEC §16 cards" : "", ticks / tot_best,
               tot_best / ticks * 1e6, builds);
        const char *nm[6] = {"upkeep+status+spawn", "target", "attack", "move", "collide", "rest"};
        for (int k = 0; k < 6; k++) printf("    %-20s %6.1f us/tick\n", nm[k], best[k] / ticks * 1e6);
    }
    free(g);
    return pr_t_failures();
}
