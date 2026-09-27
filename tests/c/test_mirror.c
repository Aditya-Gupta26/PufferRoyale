/* test_mirror.c -- bot-driven rotation-symmetry fuzz over all 64 cards and the four tower troops
 * (SPEC §13.13, §18; after the second audit's mirror.c). Game B is game A with the seats swapped:
 * decks, hands, tower troops, bots, the alternating first team and mirrored debug spawns. After every
 * tick B must be the exact 180-degree rotation of A, entity ids included (towers compared by slot). */
#include "royale.h"
#include "pr_test.h"

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define PR_SANITIZED 1
#endif
#endif
#ifndef PR_SANITIZED
#define PR_SANITIZED 0
#endif

static int mir_idx(int i) { return i == 0 ? 0 : 3 - i; }

static int same_rotated(const PrEntity *a, const PrEntity *b) {
    return a->unit == b->unit && a->team == 1 - b->team && a->x == PR_ARENA_W - b->x && a->y == PR_ARENA_H - b->y &&
           a->hp == b->hp && a->max_hp == b->max_hp && a->shield == b->shield && a->deploy_ms == b->deploy_ms &&
           a->lvl == b->lvl && a->dash_state == b->dash_state && a->burrow == b->burrow &&
           a->hide_state == b->hide_state && a->progress_ms == b->progress_ms && a->var_hits == b->var_hits &&
           a->spawn_ms == b->spawn_ms && a->aux_ms == b->aux_ms && a->seq_idx == b->seq_idx &&
           memcmp(a->buff_until, b->buff_until, sizeof(a->buff_until)) == 0;
}

static const char *mirror_diff(const PrState *A, const PrState *B) {
    if (A->n_ent != B->n_ent) return "entity count";
    if (A->elixir[0] != B->elixir[1] || A->elixir[1] != B->elixir[0]) return "elixir";
    if (A->leaked[0] != B->leaked[1] || A->leaked[1] != B->leaked[0]) return "leaked";
    if (A->crowns[0] != B->crowns[1] || A->crowns[1] != B->crowns[0]) return "crowns";
    if (A->over != B->over || A->result[0] != B->result[1]) return "result";
    for (int t = 0; t < 2; t++)
        for (int i = 0; i < 3; i++) {
            const PrEntity *x = pr_get_c(A, A->tower_id[t][i]), *y = pr_get_c(B, B->tower_id[1 - t][mir_idx(i)]);
            if (!x != !y) return "tower alive";
            if (x && !same_rotated(x, y)) return "tower state";
        }
    int ia = 0, ib = 0;
    for (;;) {
        while (ia < A->n_ent && A->ent[ia].kind == PR_KIND_TOWER) ia++;
        while (ib < B->n_ent && B->ent[ib].kind == PR_KIND_TOWER) ib++;
        if (ia >= A->n_ent || ib >= B->n_ent) break;
        if (A->ent[ia].id != B->ent[ib].id) return "entity id order";
        if (!same_rotated(&A->ent[ia], &B->ent[ib])) return "entity state";
        ia++;
        ib++;
    }
    if (A->n_fx != B->n_fx || A->n_proj != B->n_proj) return "effect / projectile count";
    for (int i = 0; i < A->n_fx; i++) {
        const PrEffect *f = &A->fx[i], *g = &B->fx[i];
        if (f->type != g->type || f->team != 1 - g->team || f->x != PR_ARENA_W - g->x || f->y != PR_ARENA_H - g->y ||
            f->timer_ms != g->timer_ms || f->remaining != g->remaining)
            return "effect";
    }
    /* projectiles as a multiset (tower shots are created in tower-id order, which differs by seat) */
    static int64_t ka[PR_MAX_PROJECTILES], kb[PR_MAX_PROJECTILES];
    for (int i = 0; i < A->n_proj; i++) {
        const PrProjectile *p = &A->proj[i], *q = &B->proj[i];
        ka[i] = ((int64_t)p->proj << 50) ^ ((int64_t)p->x << 25) ^ p->y ^ ((int64_t)p->team << 60) ^ ((int64_t)p->damage << 40);
        kb[i] = ((int64_t)q->proj << 50) ^ ((int64_t)(PR_ARENA_W - q->x) << 25) ^ (PR_ARENA_H - q->y) ^
                ((int64_t)(1 - q->team) << 60) ^ ((int64_t)q->damage << 40);
    }
    for (int i = 0; i < A->n_proj; i++)
        for (int j = i + 1; j < A->n_proj; j++) {
            if (ka[j] < ka[i]) { int64_t t = ka[i]; ka[i] = ka[j]; ka[j] = t; }
            if (kb[j] < kb[i]) { int64_t t = kb[i]; kb[i] = kb[j]; kb[j] = t; }
        }
    for (int i = 0; i < A->n_proj; i++)
        if (ka[i] != kb[i]) return "projectile";
    return NULL;
}

/* `matches` random 64-card matches; force_tt >= 0 gives both teams that tower troop. Returns the
 * number of matches whose mirror broke. */
static int mirror_fuzz(int matches, int force_tt, uint64_t seed, long *ticks_out) {
    PrGame *A = (PrGame *)calloc(1, sizeof(PrGame)), *B = (PrGame *)calloc(1, sizeof(PrGame));
    PrRng R;
    pr_rng_seed(&R, seed, 5);
    int fails = 0;
    long ticks = 0;
    for (int m = 0; m < matches; m++) {
        int8_t d[2][8];
        for (int t = 0; t < 2; t++) {
            int8_t pool[PR_N_CARDS];
            for (int c = 0; c < PR_N_CARDS; c++) pool[c] = (int8_t)c;
            for (int k = 0; k < 8; k++) {
                int j = k + (int)pr_rng_below(&R, (uint32_t)(PR_N_CARDS - k));
                int8_t tmp = pool[k]; pool[k] = pool[j]; pool[j] = tmp;
                d[t][k] = pool[k];
            }
        }
        int tt0 = force_tt >= 0 ? force_tt : (int)pr_rng_below(&R, 4);
        int tt1 = force_tt >= 0 ? force_tt : (int)pr_rng_below(&R, 4);
        pr_setup_ex(A, d[0], d[1], 7 + (uint64_t)m, 0, PR_TIEBREAK_ABSOLUTE, tt0, tt1);
        pr_setup_ex(B, d[1], d[0], 7 + (uint64_t)m, 0, PR_TIEBREAK_ABSOLUTE, tt1, tt0);
        pr_debug_set_hand(A, 0, d[0]);
        pr_debug_set_hand(A, 1, d[1]);
        pr_debug_set_hand(B, 1, d[0]);
        pr_debug_set_hand(B, 0, d[1]);
        pr_debug_set_first_team(A, 0);
        pr_debug_set_first_team(B, 1);
        int kx = m % 3 == 0 ? PR_BOT_HEURISTIC : PR_BOT_RANDOM, ky = m % 2 ? PR_BOT_HEURISTIC : PR_BOT_RANDOM;
        PrBot xA, xB, yA, yB;
        pr_bot_init(&xA, kx, 300000, (uint64_t)m);
        pr_bot_init(&xB, kx, 300000, (uint64_t)m);
        pr_bot_init(&yA, ky, 300000, (uint64_t)m + 99);
        pr_bot_init(&yB, ky, 300000, (uint64_t)m + 99);
        const char *why = NULL;
        while (!A->st.over && !why) {
            if (A->st.tick % 2 == 0) {
                int ax = pr_bot_act(&xA, &A->st, 0), bx = pr_bot_act(&xB, &B->st, 1);
                int ay = pr_bot_act(&yA, &A->st, 1), by = pr_bot_act(&yB, &B->st, 0);
                if (ax != bx || ay != by) { why = "bot actions"; break; }
                if (ax > 0) {
                    int s = (ax - 1) / PR_N_TILES, c = (ax - 1) % PR_N_TILES;
                    pr_queue_play(&A->st, 0, s, c % 18, c / 18);
                    pr_queue_play(&B->st, 1, s, c % 18, c / 18);
                }
                if (ay > 0) {
                    int s = (ay - 1) / PR_N_TILES, c = (ay - 1) % PR_N_TILES;
                    pr_queue_play(&A->st, 1, s, c % 18, c / 18);
                    pr_queue_play(&B->st, 0, s, c % 18, c / 18);
                }
            }
            if (pr_rng_below(&R, 60) == 0) { /* mirrored debug spawns of any card anywhere */
                int team = (int)pr_rng_below(&R, 2), card = (int)pr_rng_below(&R, PR_N_CARDS);
                int32_t x = 1000 + (int32_t)pr_rng_below(&R, 16000), y = 1000 + (int32_t)pr_rng_below(&R, 30000);
                if (PR_CARDS[card].kind == PR_CARD_KIND_BUILDING) { x = x / 1000 * 1000 + 500; y = y / 1000 * 1000 + 500; }
                pr_debug_spawn(A, team, card, x, y, 1, NULL, 0);
                pr_debug_spawn(B, 1 - team, card, PR_ARENA_W - x, PR_ARENA_H - y, 1, NULL, 0);
            }
            pr_tick(A);
            pr_tick(B);
            ticks++;
            why = mirror_diff(&A->st, &B->st);
        }
        if (why) {
            fails++;
            fprintf(stderr, "  match %d tick %d: mirror broken (%s), tower troops %d/%d\n", m, A->st.tick, why, tt0, tt1);
        }
    }
    free(A);
    free(B);
    if (ticks_out) *ticks_out += ticks;
    return fails;
}

static void test_bot_driven_mirror_fuzz(void) {
    int n = PR_SANITIZED ? 6 : 60, total = n;
    long ticks = 0;
    int fails = mirror_fuzz(n, -1, 4242, &ticks);
    for (int tt = 0; tt < PR_N_TOWER_TROOPS; tt++) { /* the Royal Chef (3) gets the most: its serving order */
        int k = PR_SANITIZED ? 2 : (tt == 3 ? 90 : 15);
        fails += mirror_fuzz(k, tt, 77 + (uint64_t)tt, &ticks);
        total += k;
    }
    printf("    %d matches, %ld mirrored ticks, %d divergences\n", total, ticks, fails);
    CHECK_EQ(fails, 0);
}

int main(void) {
    printf("test_mirror\n");
    RUN(test_bot_driven_mirror_fuzz);
    TEST_END();
}
