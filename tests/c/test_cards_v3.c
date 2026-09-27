/* test_cards_v3.c -- SPEC §16 (Phase E) mechanics as implemented: spawners, death spawns and
 * bombs, variable damage, minimum range, the Bandit's dash, the Miner's burrow, the Elixir
 * Collector, the new spells, the tower troops, and state-version-4 snapshot validation. */
#include "pr_test.h"

static uint32_t spawn1(PrGame *g, int team, int card, int32_t x, int32_t y, int deployed) {
    uint32_t ids[16];
    int n = pr_debug_spawn(g, team, card, x, y, deployed, ids, 16);
    CHECK(n >= 1);
    return n >= 1 ? ids[0] : PR_NO_ID;
}

static int count_unit(const PrState *st, int team, int unit) {
    int n = 0;
    for (int i = 0; i < st->n_ent; i++)
        if (st->ent[i].hp > 0 && st->ent[i].team == team && st->ent[i].unit == unit) n++;
    return n;
}

static int32_t hp_of(PrGame *g, uint32_t id) {
    const PrEntity *e = pr_get(&g->st, id);
    return e ? e->hp : -1;
}

/* Queue a play of `card` for `team` at own-frame tile (tx, ty) with full elixir (debug set_hand). */
static int play_card(PrGame *g, int team, int card, int tx, int ty) {
    int8_t order[8];
    order[0] = (int8_t)card;
    for (int c = 0, k = 1; k < 8; c++)
        if (c != card) order[k++] = (int8_t)c;
    pr_debug_set_hand(g, team, order);
    g->st.elixir[team] = PR_ELIXIR_MAX;
    return pr_queue_play(&g->st, team, 0, tx, ty);
}

static void test_witch_and_tombstone_waves(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    spawn1(g, 0, PR_CARD_WITCH, 2000, 30000, 1);
    int births[8], nb = 0, prev = 0;
    for (int k = 0; k < 320 && nb < 8; k++) {
        pr_tick(g);
        int n = count_unit(&g->st, 0, PR_UNIT_SKELETON);
        if (n > prev) {
            CHECK_EQ(n - prev, 4); /* a whole wave on one tick */
            births[nb++] = k;
        }
        prev = n;
    }
    CHECK(nb >= 3);
    CHECK_EQ(births[0], 19); /* elapsed tick k = start/50 - 1 (SPEC §16.6.24) */
    CHECK_EQ(births[1], 19 + 140);
    CHECK_EQ(births[2], 19 + 280);
    for (int i = 0; i < g->st.n_ent; i++) {
        const PrEntity *e = &g->st.ent[i];
        if (e->unit != PR_UNIT_SKELETON) continue;
        CHECK_EQ(e->card, PR_CARD_WITCH);
        CHECK_EQ(e->deploy_ms, 0);
    }
    free(g);
    /* Tombstone: singles 500 ms apart, waves 3500 ms start to start, the first at k = 69 */
    g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    spawn1(g, 0, PR_CARD_TOMBSTONE, 9000, 21000, 1);
    int want[6] = {69, 79, 139, 149, 209, 219}, got = 0;
    prev = 0;
    for (int k = 0; k < 230; k++) {
        pr_tick(g);
        int n = count_unit(&g->st, 0, PR_UNIT_SKELETON);
        if (n > prev) {
            CHECK_EQ(n - prev, 1);
            if (got < 6) CHECK_EQ(k, want[got]);
            got++;
        }
        prev = n;
    }
    CHECK_EQ(got, 6);
    free(g);
}

static void test_spawner_pauses_while_stunned(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t w = spawn1(g, 0, PR_CARD_NIGHT_WITCH, 9000, 22000, 1);
    for (int k = 0; k < 10; k++) pr_tick(g);
    pr_apply_buff(pr_get(&g->st, w), PR_BUFF_ZAPFREEZE, 500, g->st.tick); /* 10 ticks of stun from now */
    int first = -1;
    for (int k = 10; k < 60 && first < 0; k++) {
        pr_tick(g);
        if (count_unit(&g->st, 0, PR_UNIT_BAT) >= 2) first = k;
    }
    CHECK_EQ(first, 19 + 10);
    free(g);
}

static void test_death_spawn_and_bomb(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t golem = spawn1(g, 0, PR_CARD_GOLEM, 9000, 14000, 1);
    pr_get(&g->st, golem)->hp = 1;
    pr_get(&g->st, golem)->dmg_in = 5;  /* dies in this tick's Resolve */
    pr_tick(g);
    CHECK(pr_get(&g->st, golem) == NULL);
    CHECK_EQ(count_unit(&g->st, 0, PR_UNIT_GOLEMITE), 0);  /* committed in the NEXT Spawn phase */
    CHECK_EQ(g->st.n_spawn, 2);
    pr_tick(g);
    CHECK_EQ(count_unit(&g->st, 0, PR_UNIT_GOLEMITE), 2);
    free(g);
    /* Balloon bomb: committed at T+1, detonates at T+61 (SPEC §16.6.8) for 240 */
    g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t ball = spawn1(g, 0, PR_CARD_BALLOON, 9000, 11000, 0);
    uint32_t ec = spawn1(g, 1, PR_CARD_ELIXIR_COLLECTOR, 10500, 11000, 1);
    pr_get(&g->st, ball)->hp = 1;
    pr_get(&g->st, ball)->dmg_in = 5;
    pr_tick(g);                               /* death tick T (index 0) */
    CHECK(pr_get(&g->st, ball) == NULL);
    pr_tick(g);                               /* T+1: the bomb effect appears */
    int bombs = 0;
    for (int i = 0; i < g->st.n_fx; i++) bombs += g->st.fx[i].type == PR_FX_BOMB;
    CHECK_EQ(bombs, 1);
    int32_t before = hp_of(g, ec), lost_at = -1, lost = 0;
    for (int k = 2; k < 70; k++) {
        int32_t h0 = hp_of(g, ec);
        pr_tick(g);
        if (h0 - hp_of(g, ec) > 50) { lost_at = k; lost = h0 - hp_of(g, ec); }
    }
    CHECK_EQ(lost_at, 61);
    CHECK(lost == 240 || lost == 241);        /* + at most one drain tick */
    CHECK(before - hp_of(g, ec) >= 240);
    free(g);
}

static void test_variable_damage_and_reset(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[1], PR_DECKS[0], 0);
    uint32_t it = spawn1(g, 1, PR_CARD_INFERNO_TOWER, 9000, 18000, 1);
    uint32_t golem = spawn1(g, 0, PR_CARD_GOLEM, 9000, 21600, 0);
    int32_t dmg[16];
    int n = 0;
    for (int k = 0; k < 140 && n < 11; k++) { /* 11 hits: 5 x 43 + 5 x 158 + 847 (the Golem survives) */
        int32_t h0 = hp_of(g, golem);
        pr_tick(g);
        if (hp_of(g, golem) < h0) dmg[n++] = h0 - hp_of(g, golem);
    }
    CHECK_EQ(n, 11);
    for (int i = 0; i < n; i++) CHECK_EQ(dmg[i], i < 5 ? 43 : i < 10 ? 158 : 847);
    /* a stun (Zap) resets to stage 1 */
    pr_apply_buff(pr_get(&g->st, it), PR_BUFF_ZAPFREEZE, 500, g->st.tick);
    CHECK_EQ(pr_get(&g->st, it)->var_hits, 0);
    int32_t next = 0;
    for (int k = 0; k < 60 && !next; k++) {
        int32_t h0 = hp_of(g, golem);
        pr_tick(g);
        if (hp_of(g, golem) < h0) next = h0 - hp_of(g, golem);
    }
    CHECK_EQ(next, 43);
    free(g);
}

static void test_mortar_minimum_range(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t m = spawn1(g, 1, PR_CARD_MORTAR, 6000, 18000, 1);
    uint32_t near = spawn1(g, 0, PR_CARD_ELIXIR_COLLECTOR, 6000, 21000, 1);
    for (int k = 0; k < 30; k++) {
        pr_tick(g);
        CHECK(pr_get(&g->st, m)->target_id != near);
    }
    CHECK(pr_get(&g->st, m)->target_id == g->st.tower_id[0][1]); /* the Princess beyond the minimum */
    free(g);
}

static void test_bandit_dash(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    uint32_t golem = spawn1(g, 1, PR_CARD_GOLEM, 9000, 18000, 0);
    uint32_t b = spawn1(g, 0, PR_CARD_BANDIT, 9000, 23000, 1);
    int stand = 0, moved_fast = 0, hit_at = -1;
    int32_t hit = 0;
    for (int k = 0; k < 60 && hit_at < 0; k++) {
        const PrEntity *e = pr_get(&g->st, b);
        int32_t y0 = e->y, h0 = hp_of(g, golem);
        pr_tick(g);
        e = pr_get(&g->st, b);
        if (e->dash_state == PR_DASH_STAND) stand++;
        if (y0 - e->y >= 250) moved_fast++;
        if (hp_of(g, golem) < h0) { hit_at = k; hit = h0 - hp_of(g, golem); }
    }
    CHECK_EQ(stand, 16);            /* stands DashCooldown 800 ms */
    CHECK(moved_fast >= 4);         /* then 500 per tick */
    CHECK_EQ(hit, 389);             /* the dash hit */
    /* the next swing lands 19 ticks after the arrival (ledger combat.DASH_ATTACK) */
    int next = -1;
    for (int k = 1; k < 30 && next < 0; k++) {
        int32_t h0 = hp_of(g, golem);
        pr_tick(g);
        if (hp_of(g, golem) < h0) { next = k; CHECK_EQ(h0 - hp_of(g, golem), 194); }
    }
    CHECK_EQ(next, 19);
    /* immune while dashing: hits dealt in the moving phase buffer nothing (SPEC §18.3) */
    PrEntity *e = pr_get(&g->st, b);
    e->dash_state = PR_DASH_MOVE;
    pr_hit(e, 100, 100);
    CHECK_EQ(e->dmg_in, 0);
    e->dash_state = PR_DASH_HIT;
    pr_hit(e, 100, 100);
    CHECK_EQ(e->dmg_in, 0);
    e->dash_state = PR_DASH_STAND; /* the stand is not protected (ledger combat.DASH_ATTACK) */
    pr_hit(e, 100, 100);
    CHECK_EQ(e->dmg_in, 100);
    free(g);
}

/* SPEC §18.3: a spell whose stun (Zap, Freeze) or knockback (Snowball) cancels a moving dash deals no
 * damage in that tick either; the same spell on a standing Bandit does. */
static void test_dash_immunity_vs_cancelling_spells(void) {
    const int spells[3] = {PR_CARD_ZAP, PR_CARD_FREEZE, PR_CARD_GIANT_SNOWBALL};
    for (int si = 0; si < 3; si++) {
        for (int moving = 0; moving < 2; moving++) {
            PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
            spawn1(g, 1, PR_CARD_GOLEM, 9000, 18000, 0);
            uint32_t b = spawn1(g, 0, PR_CARD_BANDIT, 9000, 23000, 1);
            int want = moving ? PR_DASH_MOVE : PR_DASH_STAND, found = 0;
            for (int k = 0; k < 60 && !found; k++) {
                pr_tick(g);
                const PrEntity *e = pr_get(&g->st, b);
                found = e->dash_state == want && (moving || e->aux_ms <= 400);
            }
            CHECK(found);
            PrEntity *e = pr_get(&g->st, b);
            int32_t hp = e->hp;
            if (spells[si] == PR_CARD_GIANT_SNOWBALL) {
                /* the Snowball lands on her now (its impact, called directly between the phases): damage,
                 * slow and a 1800 knockback that cancels a moving dash */
                PrProjectile *p = pr_new_projectile(&g->st, PR_PROJ_SNOWBALLSPELL, 1, PR_CARD_GIANT_SNOWBALL, e->x, e->y);
                CHECK(p != NULL);
                if (p) {
                    p->has_target = 0;
                    PrProjectile q = *p;
                    g->st.n_proj--;
                    memset(p, 0, sizeof(*p));
                    pr_projectile_impact(&g->st, &q);
                }
                CHECK_EQ(pr_get(&g->st, b)->dash_state, moving ? PR_DASH_NONE : PR_DASH_NONE);
                pr_phase_resolve(&g->st);
            } else {
                pr_debug_spawn(g, 1, spells[si], e->x, e->y, 1, NULL, 0); /* applied now, before Resolve */
                pr_tick(g);
            }
            e = pr_get(&g->st, b);
            CHECK(e != NULL);
            if (!e) { free(g); continue; }
            if (moving) {
                CHECK_EQ(e->hp, hp);                       /* no damage */
                CHECK_EQ(e->dash_state, PR_DASH_NONE);     /* but the stun / knockback did cancel the dash */
            } else {
                CHECK(e->hp < hp);                         /* the stand is not protected */
            }
            free(g);
        }
    }
}

static void test_miner_burrow(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    CHECK_EQ(play_card(g, 0, PR_CARD_MINER, 9, 10), PR_OK); /* tap (9500, 10500) */
    int64_t d = pr_isqrt64(pr_dist2(9000, 29000, 9500, 10500));
    int n = (int)pr_ceildiv(d, 650);
    uint32_t id = PR_NO_ID;
    int hidden = 0, deploying = 0, surfaced = 0;
    for (int k = 0; k < n + 30; k++) {
        pr_tick(g);
        if (id == PR_NO_ID)
            for (int i = 0; i < g->st.n_ent; i++)
                if (g->st.ent[i].unit == PR_UNIT_MINER) id = g->st.ent[i].id;
        const PrEntity *e = pr_get(&g->st, id);
        CHECK(e != NULL);
        if (!e) break;
        if (pr_is_hidden(e)) {
            hidden++;
            CHECK(e->burrow && !pr_is_deploying(e));
        } else {
            if (!surfaced++) { /* it surfaces exactly on the tap point */
                CHECK_EQ(e->x, 9500);
                CHECK_EQ(e->y, 10500);
            }
            deploying += pr_is_deploying(e);
        }
    }
    CHECK_EQ(hidden, n);     /* SPEC §16.6.23 */
    CHECK_EQ(deploying, 20);
    free(g);
}

static void test_elixir_collector(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    g->st.elixir[0] = 0;
    uint32_t ec = spawn1(g, 0, PR_CARD_ELIXIR_COLLECTOR, 9000, 22000, 0);
    int prod[3], np = 0;
    for (int k = 0; k < 560; k++) {
        int32_t e0 = g->st.elixir[0];
        if (e0 > 20000) g->st.elixir[0] = e0 = 0;
        pr_tick(g);
        if (g->st.elixir[0] - e0 == 2850 && np < 3) prod[np++] = k;
    }
    CHECK_EQ(np, 2);
    CHECK_EQ(prod[0], 20 + 259);
    CHECK_EQ(prod[1], 20 + 519);
    /* destroyed by damage: +2800 once, capped (the excess leaks) */
    g->st.elixir[0] = PR_ELIXIR_MAX - 1000;
    int32_t leaked = g->st.leaked[0];
    pr_get(&g->st, ec)->dmg_in = 5000;
    pr_tick(g);
    CHECK(pr_get(&g->st, ec) == NULL);
    CHECK_EQ(g->st.elixir[0], PR_ELIXIR_MAX);
    CHECK_EQ(g->st.leaked[0] - leaked, 50 + 2800 - 1000);
    free(g);
}

static void test_pulse_lightning_freeze(void) {
    /* Poison: 8 events of 92 at P + 20 i; crown towers 22 */
    PrGame *g = t_game_decks(1, PR_DECKS[6], PR_DECKS[1], 0);
    uint32_t golem = spawn1(g, 1, PR_CARD_GOLEM, 9000, 11000, 0);
    CHECK_EQ(play_card(g, 0, PR_CARD_POISON, 9, 10), PR_OK);
    int ev[10], ne = 0;
    for (int k = 0; k < 170; k++) {
        int32_t h0 = hp_of(g, golem);
        pr_tick(g);
        if (hp_of(g, golem) < h0) { CHECK_EQ(h0 - hp_of(g, golem), 92); if (ne < 10) ev[ne++] = k; }
    }
    CHECK_EQ(ne, 8);
    for (int i = 0; i < ne && i < 8; i++) CHECK_EQ(ev[i], 20 * i);
    free(g);
    /* Earthquake vs a building 283, vs a crown tower 49, never air */
    g = t_game_decks(1, PR_DECKS[8], PR_DECKS[1], 0);
    uint32_t xb = spawn1(g, 1, PR_CARD_X_BOW, 3500, 9000, 1);
    uint32_t mm = spawn1(g, 1, PR_CARD_MEGA_MINION, 4500, 9000, 0);
    int32_t hx = hp_of(g, xb), hm = hp_of(g, mm), ht = pr_get(&g->st, g->st.tower_id[1][1])->hp;
    CHECK_EQ(play_card(g, 0, PR_CARD_EARTHQUAKE, 3, 8), PR_OK);
    pr_tick(g);
    CHECK(hx - hp_of(g, xb) >= 283 && hx - hp_of(g, xb) <= 285);
    CHECK_EQ(hm, hp_of(g, mm));
    CHECK_EQ(ht - pr_get(&g->st, g->st.tower_id[1][1])->hp, 49);
    free(g);
    /* Lightning: the 3 highest-hp targets, highest first, at P, P+10, P+19 */
    g = t_game_decks(1, PR_DECKS[3], PR_DECKS[1], 0);
    uint32_t a = spawn1(g, 1, PR_CARD_GOLEM, 9000, 11000, 0);
    uint32_t b = spawn1(g, 1, PR_CARD_P_E_K_K_A, 7000, 11000, 0);
    uint32_t c = spawn1(g, 1, PR_CARD_KNIGHT, 11000, 11000, 0);
    spawn1(g, 1, PR_CARD_SKELETONS, 9000, 12500, 0);
    CHECK_EQ(play_card(g, 0, PR_CARD_LIGHTNING, 9, 11), PR_OK);
    uint32_t order[3] = {a, b, c};
    int when[3] = {-1, -1, -1};
    for (int k = 0; k < 30; k++) {
        int32_t h[3] = {hp_of(g, a), hp_of(g, b), hp_of(g, c)};
        pr_tick(g);
        for (int j = 0; j < 3; j++)
            if (hp_of(g, order[j]) < h[j]) {
                when[j] = k;
                CHECK_EQ(h[j] - hp_of(g, order[j]), 1057);
                CHECK(pr_is_stunned(pr_get(&g->st, order[j])));
            }
    }
    CHECK_EQ(when[0], 0);
    CHECK_EQ(when[1], 10);
    CHECK_EQ(when[2], 19);
    free(g);
    /* Freeze: the crown tower and a hidden Tesla frozen 80 ticks; the Tesla takes no damage */
    g = t_game_decks(1, PR_DECKS[3], PR_DECKS[1], 0);
    uint32_t tesla = spawn1(g, 1, PR_CARD_TESLA, 6500, 9500, 1);
    int8_t order8[8] = {PR_CARD_FREEZE, 0, 1, 2, 3, 4, 5, 6};
    pr_debug_set_hand(g, 0, order8);
    g->st.elixir[0] = PR_ELIXIR_MAX;
    CHECK_EQ(pr_queue_play_at(&g->st, 0, 0, 3500, 9500), PR_OK);
    int32_t htes = hp_of(g, tesla);
    int frozen = 0;
    for (int k = 0; k < 100; k++) {
        pr_tick(g);
        frozen += pr_is_stunned(pr_get(&g->st, g->st.tower_id[1][1]));
        if (k == 0) CHECK(pr_is_stunned(pr_get(&g->st, tesla)));
    }
    CHECK_EQ(frozen, 80);
    CHECK(htes - hp_of(g, tesla) <= 3 * 100); /* only the lifetime drain (2.4 per tick) */
    free(g);
}

static void test_pulse_slow_refresh(void) {
    /* the slow is refreshed every area HitSpeed while the area lives: Poison (250 ms, 8 s) slows a
     * unit standing in it through P+174; Earthquake (100 ms, 3 s, capped buff) through P+59 */
    for (int eq = 0; eq < 2; eq++) {
        PrGame *g = t_game_decks(1, PR_DECKS[eq ? 8 : 6], PR_DECKS[1], 0);
        uint32_t golem = spawn1(g, 1, PR_CARD_GOLEM, 9000, 11000, 0);
        CHECK_EQ(play_card(g, 0, eq ? PR_CARD_EARTHQUAKE : PR_CARD_POISON, 9, 10), PR_OK);
        int last = -1;
        for (int k = 0; k < 200; k++) {
            pr_get(&g->st, golem)->deploy_ms = 1000; /* keep it standing inside */
            pr_tick(g);
            if (pr_is_slowed(pr_get(&g->st, golem))) last = k;
        }
        CHECK_EQ(last, eq ? 59 : 174);
        free(g);
    }
}

static void test_rolling_barrel_and_knockback(void) {
    PrGame *g = t_game_decks(1, PR_DECKS[8], PR_DECKS[1], 0);
    CHECK_EQ(play_card(g, 0, PR_CARD_BARBARIAN_BARREL, 9, 18), PR_OK); /* rolls from (9500, 18500) */
    int born = -1;
    for (int k = 0; k < 40 && born < 0; k++) {
        pr_tick(g);
        for (int i = 0; i < g->st.n_ent; i++)
            if (g->st.ent[i].unit == PR_UNIT_BARBARIAN) {
                born = k;
                CHECK_EQ(g->st.ent[i].card, PR_CARD_BARBARIAN_BARREL);
                CHECK_EQ(g->st.ent[i].x, 9500);
                CHECK_EQ(g->st.ent[i].y, 14000);
                CHECK_EQ(g->st.ent[i].deploy_ms, 1000);
            }
    }
    CHECK_EQ(born, 24); /* 23 rolling steps, the end-point hit test, then the next Spawn phase */
    free(g);
    /* Rocket: 1800 knockback away from the impact point */
    g = t_game_decks(1, PR_DECKS[0], PR_DECKS[1], 0);
    spawn1(g, 0, PR_CARD_GOLEM, 9000, 19300, 0);
    uint32_t kn = spawn1(g, 1, PR_CARD_KNIGHT, 9000, 18000, 1);
    int8_t order8[8] = {PR_CARD_ROCKET, 0, 1, 2, 3, 4, 5, 6};
    pr_debug_set_hand(g, 0, order8);
    g->st.elixir[0] = PR_ELIXIR_MAX;
    CHECK_EQ(pr_queue_play_at(&g->st, 0, 0, 8000, 18000), PR_OK);
    for (int k = 0; k < 60 && hp_of(g, kn) == 1766; k++) pr_tick(g);
    CHECK_EQ(1766 - hp_of(g, kn), 1484);
    CHECK_EQ(pr_get(&g->st, kn)->x, 10800);
    CHECK_EQ(pr_get(&g->st, kn)->y, 18000);
    free(g);
}

static void test_tower_troops(void) {
    /* Dagger Duchess: shots 10 / 7 / 9 ticks apart inside the sequence, then the 1000 ms reload (30) */
    PrGame *g = (PrGame *)calloc(1, sizeof(PrGame));
    pr_setup_ex(g, PR_DECKS[0], PR_DECKS[1], 1, 0, PR_TIEBREAK_ABSOLUTE, 0, 2);
    for (int k = 0; k < 100; k++) pr_tick(g);
    uint32_t golem = spawn1(g, 0, PR_CARD_GOLEM, 3500, 9000, 1);
    int hits[12], nh = 0;
    for (int k = 0; k < 200 && nh < 12; k++) {
        int32_t h0 = hp_of(g, golem);
        pr_tick(g);
        if (h0 - hp_of(g, golem) == 91) hits[nh++] = k;
    }
    CHECK(nh >= 9);
    int want_gap[8] = {10, 7, 9, 30, 10, 7, 9, 30};
    for (int i = 1; i < nh && i < 9; i++) CHECK_EQ(hits[i] - hits[i - 1], want_gap[i - 1]);
    free(g);
    /* Royal Chef: the first level-up after the first full 28 s period, the highest-cost troop within
     * 7500 of that tower, once per troop */
    g = (PrGame *)calloc(1, sizeof(PrGame));
    pr_setup_ex(g, PR_DECKS[0], PR_DECKS[1], 1, 0, PR_TIEBREAK_ABSOLUTE, 3, 0);
    uint32_t kn = spawn1(g, 0, PR_CARD_KNIGHT, 3500, 23000, 0);
    uint32_t pk = spawn1(g, 0, PR_CARD_P_E_K_K_A, 5000, 23000, 0);
    int lvl_at = -1;
    for (int k = 0; k < 600 && lvl_at < 0; k++) {
        PrEntity *e = pr_get(&g->st, pk), *f = pr_get(&g->st, kn);
        e->deploy_ms = f->deploy_ms = 1000; /* keep them standing */
        pr_tick(g);
        if (pr_get(&g->st, pk)->lvl) lvl_at = k;
    }
    CHECK_EQ(lvl_at, 559);
    CHECK_EQ(pr_get(&g->st, pk)->max_hp, 3760 * 281 / 256);
    CHECK_EQ(pr_get(&g->st, kn)->lvl, 0);      /* the cheaper Knight waits for the next pancake */
    CHECK_EQ(pr_lvl(pr_get(&g->st, pk), 842), 842 * 281 / 256);
    free(g);
}

static void test_snapshot_v4_validation(void) {
    PrGame *g = t_game_decks(7, PR_DECKS[6], PR_DECKS[7], 0);
    CHECK_EQ(play_card(g, 0, PR_CARD_MINER, 9, 10), PR_OK);
    spawn1(g, 1, PR_CARD_BANDIT, 9000, 16000, 1);
    spawn1(g, 1, PR_CARD_WITCH, 9000, 12000, 1);
    uint32_t ball = spawn1(g, 1, PR_CARD_BALLOON, 3000, 20000, 1);
    pr_get(&g->st, ball)->dmg_in = 99999;
    for (int k = 0; k < 3; k++) pr_tick(g);
    CHECK_EQ(play_card(g, 1, PR_CARD_POISON, 9, 20), PR_OK);
    pr_tick(g);
    const char *why = NULL;
    CHECK(pr_state_check(&g->st, &why));
    static PrState bad;
    memcpy(&bad, &g->st, sizeof(bad));
    int mi = -1;
    for (int i = 0; i < bad.n_ent; i++)
        if (bad.ent[i].unit == PR_UNIT_MINER) mi = i;
    CHECK(mi >= 0 && bad.ent[mi].burrow);
    bad.ent[mi].unit = PR_UNIT_KNIGHT; /* a burrowing Knight is not a state this engine makes */
    CHECK(!pr_state_check(&bad, &why));
    memcpy(&bad, &g->st, sizeof(bad));
    bad.tower_troop[1] = 9;
    CHECK(!pr_state_check(&bad, &why));
    memcpy(&bad, &g->st, sizeof(bad));
    bad.ent[0].dash_state = PR_DASH_MOVE; /* a King tower cannot dash */
    CHECK(!pr_state_check(&bad, &why));
    free(g);
}

static void test_capacity_with_spawners(void) {
    /* spawners and death spawns never overflow the pools silently: drops are counted */
    PrGame *g = t_game_decks(3, PR_DECKS[0], PR_DECKS[1], 0);
    for (int k = 0; k < 40; k++) spawn1(g, k % 2, PR_CARD_WITCH, 2000 + 350 * (k / 2), k % 2 ? 4000 : 28000, 1);
    for (int k = 0; k < 400; k++) pr_tick(g);
    CHECK(g->st.n_ent <= PR_MAX_ENTITIES);
    CHECK(g->st.spawn_overflow > 0);
    const char *why = NULL;
    CHECK(pr_state_check(&g->st, &why));
    free(g);
}

int main(void) {
    printf("test_cards_v3\n");
    RUN(test_witch_and_tombstone_waves);
    RUN(test_spawner_pauses_while_stunned);
    RUN(test_death_spawn_and_bomb);
    RUN(test_variable_damage_and_reset);
    RUN(test_mortar_minimum_range);
    RUN(test_bandit_dash);
    RUN(test_dash_immunity_vs_cancelling_spells);
    RUN(test_miner_burrow);
    RUN(test_elixir_collector);
    RUN(test_pulse_lightning_freeze);
    RUN(test_pulse_slow_refresh);
    RUN(test_rolling_barrel_and_knockback);
    RUN(test_tower_troops);
    RUN(test_snapshot_v4_validation);
    RUN(test_capacity_with_spawners);
    TEST_END();
}
