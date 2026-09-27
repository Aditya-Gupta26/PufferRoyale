/*
 * pr_bots.h -- scripted opponents (SPEC §8). Integer-only and deterministic given the
 * bot's own PCG32 stream (never the game RNG). Every action a bot returns is checked
 * against the engine's exact legality mask; 0 is the no-op.
 *
 *   noop       never plays.
 *   random     each decision, with probability play_ppm / 1e6, a uniformly random legal
 *              non-noop action from the mask.
 *   heuristic  a rule bot, reasoning in its own frame (rules in docs/FIDELITY.md §8):
 *                1. finish an enemy crown tower a damage spell can kill;
 *                2. defend when enemy troops threaten its half: a damage spell on a
 *                   cluster it pays for, a building in the centre against a
 *                   buildings-only targeter, otherwise the cheapest capable troop
 *                   placed in front of the threat (melee) or behind the lane's tower
 *                   (ranged);
 *                3. otherwise push the weaker enemy lane with a win condition at the
 *                   bridge (Goblin Barrel / Miner on the tower, slow tanks further back,
 *                   siege buildings at the river), support an advancing win condition
 *                   from behind, and never sit at full elixir.
 *              Every card is handled by its kind and data (SPEC §16.4): buildings-only troops,
 *              the Miner, Goblin Barrel and siege buildings are win conditions; every spell with
 *              damage is a damage spell (rolling ones never hit air, Earthquake neither).
 */
#ifndef PR_BOTS_H
#define PR_BOTS_H

#include "pr_engine.h"

enum { PR_BOT_NOOP = 0, PR_BOT_RANDOM = 1, PR_BOT_HEURISTIC = 2 };

#define PR_BOT_STREAM 0x424F54u /* "BOT": bot PCG32 stream id (the game uses 0x5052) */

typedef struct PrBot {
    int32_t kind;
    uint32_t play_ppm;        /* random bot: play probability in millionths */
    PrRng rng;
    int32_t last_play_tick;   /* heuristic: tick of its last play */
    int32_t pad_;
} PrBot;

static inline void pr_bot_init(PrBot *b, int kind, uint32_t play_ppm, uint64_t seed) {
    memset(b, 0, sizeof(*b));
    b->kind = kind;
    b->play_ppm = play_ppm > 1000000u ? 1000000u : play_ppm;
    pr_rng_seed(&b->rng, seed, PR_BOT_STREAM);
    b->last_play_tick = -1000000;
}

static inline int pr_bot_action_of(int slot, int tx, int ty) { return 1 + slot * PR_N_TILES + ty * PR_TILES_X + tx; }

/* ------------------------------------------------------------------ random */

static inline int pr_bot_random(PrBot *b, const PrState *st, int team, const uint8_t *mask) {
    (void)st;
    (void)team;
    if (pr_rng_below(&b->rng, 1000000u) >= b->play_ppm) return 0;
    int n = 0;
    for (int a = 1; a < PR_N_ACTIONS; a++) n += mask[a];
    if (n == 0) return 0;
    int k = (int)pr_rng_below(&b->rng, (uint32_t)n);
    for (int a = 1; a < PR_N_ACTIONS; a++)
        if (mask[a] && k-- == 0) return a;
    return 0;
}

/* ------------------------------------------------------------------ heuristic helpers */

typedef struct PrBotView {
    const PrState *st;
    const uint8_t *mask;
    int team;
    int32_t elixir;           /* whole elixir */
    int8_t hand[4];
} PrBotView;

static inline int pr_bot_slot_ok(const PrBotView *v, int slot) {
    const uint8_t *blk = v->mask + 1 + slot * PR_N_TILES;
    for (int i = 0; i < PR_N_TILES; i++)
        if (blk[i]) return 1;
    return 0;
}

/* Nearest legal tile to (tx, ty) for `slot` (square rings, fixed scan order). */
static inline int pr_bot_find(const PrBotView *v, int slot, int tx, int ty, int radius) {
    for (int r = 0; r <= radius; r++) {
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                if (PR_MAX(pr_abs32(dx), pr_abs32(dy)) != r) continue;
                int x = tx + dx, y = ty + dy;
                if (x < 0 || y < 0 || x >= PR_TILES_X || y >= PR_TILES_Y) continue;
                int a = pr_bot_action_of(slot, x, y);
                if (v->mask[a]) return a;
            }
        }
    }
    return 0;
}

static inline int pr_bot_tile_x(int32_t own_x) { return PR_CLAMP(own_x / 1000, 0, PR_TILES_X - 1); }
static inline int pr_bot_tile_y(int32_t own_y) { return PR_CLAMP(own_y / 1000, 0, PR_TILES_Y - 1); }

/* A siege building: a building whose range reaches across the river (X-Bow, Mortar). */
static inline int pr_bot_is_siege(int card) {
    const PrCardDef *c = &PR_CARDS[card];
    return c->kind == PR_CARD_KIND_BUILDING && PR_UNITS[c->unit].range >= 10000;
}

static inline int pr_bot_is_win_condition(int card) {
    if (card < 0 || card >= PR_N_CARDS) return 0;
    const PrCardDef *c = &PR_CARDS[card];
    if (card == PR_CARD_PRINCE || c->spell_type == PR_SPELL_SPAWN_PROJECTILE) return 1; /* Prince, Goblin Barrel */
    if (c->kind == PR_CARD_KIND_TROOP)
        return PR_UNITS[c->unit].only_buildings || PR_UNITS[c->unit].burrow_speed > 0; /* Giant..Wall Breakers, Miner */
    return pr_bot_is_siege(card);
}

static inline int pr_bot_is_damage_spell(int card) {
    if (card < 0 || card >= PR_N_CARDS) return 0;
    const PrCardDef *c = &PR_CARDS[card];
    return c->kind == PR_CARD_KIND_SPELL && c->spell_damage > 0;
}

/* A spell rolling along the ground (The Log, Barbarian Barrel). */
static inline int pr_bot_is_rolling(int card) { return PR_CARDS[card].spell_type == PR_SPELL_ROLLING; }

/* Total crown-tower damage of a damage spell (Arrows: all three waves; a pulsing area: all its
 * events; Lightning: a single bolt on a lone tower). */
static inline int32_t pr_bot_spell_tower_damage(int card) {
    const PrCardDef *c = &PR_CARDS[card];
    int32_t d = (int32_t)pr_ceildiv((int64_t)c->spell_damage * c->crown_pct, 100);
    if (c->spell_type == PR_SPELL_WAVES || c->spell_type == PR_SPELL_PULSE) return d * PR_MAX(c->waves, 1);
    return d;
}

static inline int pr_bot_spell_hits_air(int card) {
    const PrCardDef *c = &PR_CARDS[card];
    if (c->spell_type == PR_SPELL_ROLLING) return 0;
    if (c->spell_area >= 0) return PR_AREAS[c->spell_area].hits_air;
    return 1;
}

/* Damage a spell deals to one troop over its whole life (for the bot's kill estimate). */
static inline int32_t pr_bot_spell_total(int card) {
    const PrCardDef *c = &PR_CARDS[card];
    if (c->spell_type == PR_SPELL_WAVES || c->spell_type == PR_SPELL_PULSE) return c->spell_damage * PR_MAX(c->waves, 1);
    return c->spell_damage;
}

/* Can the troop/building of `card` attack entity e? */
static inline int pr_bot_can_hit(int card, const PrEntity *e) {
    const PrCardDef *c = &PR_CARDS[card];
    if (c->kind == PR_CARD_KIND_SPELL) return 0;
    const PrUnitDef *u = &PR_UNITS[c->unit];
    if (u->no_attack) return u->spawn_unit >= 0 && !e->flying; /* Tombstone: its Skeletons defend the ground */
    if (u->only_buildings && !pr_is_structure(e)) return 0;
    return e->flying ? u->attacks_air : u->attacks_ground;
}

/* Lightning's bolts (SPEC §16.6.13): would entity e be one of the 3 highest current-hp enemies the
 * spell tapped at own point (cx, cy) picks (every valid enemy in the radius counts, crown towers
 * included; ties to the lower rank as the engine's lowest id -- crown towers first, as their ids are
 * the lowest for both seats)? */
static inline int pr_bot_lightning_picks(const PrBotView *v, const PrCardDef *c, const PrEntity *e, int32_t cx,
                                         int32_t cy) {
    int above = 0;
    for (int i = 0; i < v->st->n_ent; i++) {
        const PrEntity *o = &v->st->ent[i];
        if (o == e || o->hp <= 0 || o->team == v->team || o->burrow || pr_is_hidden(o)) continue;
        if (o->flying ? !PR_AREAS[c->spell_area].hits_air : !PR_AREAS[c->spell_area].hits_ground) continue;
        if (!pr_within(cx, cy, pr_own_x(v->team, o->x), pr_own_y(v->team, o->y), (int64_t)c->spell_radius + o->radius))
            continue;
        if (o->hp > e->hp || (o->hp == e->hp && o->id < e->id)) above++;
    }
    return above < PR_MAX(c->waves, 1);
}

/* Value of casting damage spell `card` at own point (cx, cy) -- the tap it would really use: 3 per
 * enemy troop it kills, 1 per other enemy it hits. A rolling spell (the Log, Barbarian Barrel) is
 * scored over its roll corridor from the tap toward the enemy (up to 5 tiles); Lightning only on
 * the targets its bolts would strike. */
static inline int pr_bot_spell_value(const PrBotView *v, int card, int32_t cx, int32_t cy) {
    const PrCardDef *c = &PR_CARDS[card];
    int32_t dmg = pr_bot_spell_total(card);
    int32_t roll = c->spell_proj >= 0 ? PR_PROJS[c->spell_proj].range : 0;
    int value = 0;
    for (int i = 0; i < v->st->n_ent; i++) {
        const PrEntity *e = &v->st->ent[i];
        if (e->hp <= 0 || e->team == v->team || e->kind == PR_KIND_TOWER) continue;
        if (e->flying && !pr_bot_spell_hits_air(card)) continue;
        int32_t ex = pr_own_x(v->team, e->x), ey = pr_own_y(v->team, e->y);
        int hit;
        if (pr_bot_is_rolling(card)) /* rolls toward the enemy (own -y); the bot scores up to 5 tiles */
            hit = pr_abs32(ex - cx) <= c->spell_radius + e->radius && ey <= cy + 600 && ey >= cy - PR_MIN(roll, 5000);
        else
            hit = pr_within(cx, cy, ex, ey, (int64_t)c->spell_radius + e->radius);
        if (!hit) continue;
        if (c->spell_type == PR_SPELL_LIGHTNING && !pr_bot_lightning_picks(v, c, e, cx, cy)) continue;
        value += (e->hp <= dmg) ? 3 : 1;
    }
    return value;
}

typedef struct PrBotThreat {
    int n;                    /* enemy troops/buildings on or approaching our half */
    int32_t hp;
    int air;
    int building_targeter;
    int32_t lead_x, lead_y;   /* own frame position of the most advanced threat */
    const PrEntity *lead;
    int32_t own_def_hp;       /* our troops standing in our half */
    const PrEntity *own_win;  /* our most advanced win-condition troop, if any */
} PrBotThreat;

static inline void pr_bot_assess(const PrBotView *v, PrBotThreat *t) {
    memset(t, 0, sizeof(*t));
    int32_t best_y = -1, win_y = 1 << 30;
    for (int i = 0; i < v->st->n_ent; i++) {
        const PrEntity *e = &v->st->ent[i];
        if (e->hp <= 0 || e->kind == PR_KIND_TOWER) continue;
        int32_t ex = pr_own_x(v->team, e->x), ey = pr_own_y(v->team, e->y);
        if (e->team != v->team) {
            if (e->kind != PR_KIND_TROOP || ey < 13000) continue;
            t->n++;
            t->hp += e->hp;
            if (e->flying) t->air = 1;
            if (pr_udef(e)->only_buildings) t->building_targeter = 1;
            if (ey > best_y) { best_y = ey; t->lead_x = ex; t->lead_y = ey; t->lead = e; }
        } else if (e->kind == PR_KIND_TROOP) {
            if (ey >= 15000) t->own_def_hp += e->hp;
            if (e->card >= 0 && pr_bot_is_win_condition(e->card) && ey < win_y) { win_y = ey; t->own_win = e; }
        }
    }
}

/* Own-frame x of the enemy lane to attack: the weaker surviving enemy princess (own-frame
 * left = 3500, right = 14500), ties broken by the bot RNG; the centre if both are down. */
static inline int32_t pr_bot_attack_lane_x(PrBot *b, const PrBotView *v) {
    int enemy = 1 - v->team;
    /* own-frame left/right -> engine tower index for the enemy */
    int idx_left = v->team == 0 ? 1 : 2, idx_right = v->team == 0 ? 2 : 1;
    const PrEntity *l = pr_get_c(v->st, v->st->tower_id[enemy][idx_left]);
    const PrEntity *r = pr_get_c(v->st, v->st->tower_id[enemy][idx_right]);
    if (!l && !r) return 9000;
    if (!l) return 3500;   /* its pocket: walk down the fallen lane toward the King */
    if (!r) return 14500;
    if (l->hp != r->hp) return l->hp < r->hp ? 3500 : 14500;
    return pr_rng_below(&b->rng, 2) ? 3500 : 14500;
}

static inline int pr_bot_heuristic(PrBot *b, const PrState *st, int team, const uint8_t *mask) {
    PrBotView v;
    v.st = st;
    v.mask = mask;
    v.team = team;
    v.elixir = st->elixir[team] / PR_ELIXIR_UNIT;
    memcpy(v.hand, st->hand[team], 4);
    int any = 0;
    for (int s = 0; s < 4; s++) any |= pr_bot_slot_ok(&v, s);
    if (!any) return 0;
    PrBotThreat th;
    pr_bot_assess(&v, &th);
    int enemy = 1 - team;

    /* 1. finish a crown tower with a damage spell; the enemy towers in the OWN frame's order (King,
     * own-left, own-right: SPEC §18.4), so both seats pick the same tower of a mirrored board */
    for (int k = 0; k < 3; k++) {
        int idx = k == 0 ? 0 : (team == 0 ? k : 3 - k);
        const PrEntity *tw = pr_get_c(st, st->tower_id[enemy][idx]);
        if (!tw) continue;
        for (int s = 0; s < 4; s++) {
            int c = v.hand[s];
            if (!pr_bot_is_damage_spell(c) || pr_bot_is_rolling(c) || !pr_bot_slot_ok(&v, s)) continue;
            if (tw->hp > pr_bot_spell_tower_damage(c)) continue;
            int a = pr_bot_find(&v, s, pr_bot_tile_x(pr_own_x(team, tw->x)), pr_bot_tile_y(pr_own_y(team, tw->y)), 1);
            if (a) return a;
        }
    }

    /* 2. defend */
    if (th.n > 0 && th.hp > th.own_def_hp / 2) {
        /* 2a. a damage spell on the best cluster, if it pays */
        int best_a = 0, best_val = 0;
        for (int s = 0; s < 4; s++) {
            int c = v.hand[s];
            if (!pr_bot_is_damage_spell(c) || !pr_bot_slot_ok(&v, s)) continue;
            for (int i = 0; i < st->n_ent; i++) {
                const PrEntity *e = &st->ent[i];
                if (e->hp <= 0 || e->team == team || e->kind != PR_KIND_TROOP) continue;
                int32_t ex = pr_own_x(team, e->x), ey = pr_own_y(team, e->y);
                if (ey < 13000) continue;
                int32_t cy = pr_bot_is_rolling(c) ? ey + 1500 : ey;
                int a = pr_bot_find(&v, s, pr_bot_tile_x(ex), pr_bot_tile_y(cy), 1);
                if (!a) continue;
                /* valued at the tap the action really uses: the chosen tile's centre */
                int cell = (a - 1) % PR_N_TILES;
                int val = pr_bot_spell_value(&v, c, (cell % PR_TILES_X) * 1000 + 500, (cell / PR_TILES_X) * 1000 + 500);
                int need = 2 + PR_CARDS[c].elixir; /* worth at least its cost in value */
                if (val < need || val <= best_val) continue;
                best_a = a;
                best_val = val;
            }
        }
        if (best_a) { b->last_play_tick = st->tick; return best_a; }
        /* 2b. a building in the centre against a buildings-only targeter */
        if (th.building_targeter) {
            for (int s = 0; s < 4; s++) {
                int c = v.hand[s];
                if (PR_CARDS[c].kind != PR_CARD_KIND_BUILDING || !pr_bot_slot_ok(&v, s)) continue;
                if (pr_bot_is_siege(c) || PR_UNITS[PR_CARDS[c].unit].mana_gen_ms > 0) continue; /* not a decoy */
                int a = pr_bot_find(&v, s, th.lead_x < 9000 ? 8 : 9, 21, 2);
                if (a) { b->last_play_tick = st->tick; return a; }
            }
        }
        /* 2c. the cheapest troop that can hit the lead threat (a bigger one vs tanks) */
        int pick = -1;
        for (int s = 0; s < 4; s++) {
            int c = v.hand[s];
            if (PR_CARDS[c].kind != PR_CARD_KIND_TROOP || !pr_bot_slot_ok(&v, s)) continue;
            if (PR_UNITS[PR_CARDS[c].unit].only_buildings || PR_UNITS[PR_CARDS[c].unit].burrow_speed > 0) continue;
            if (!th.lead || !pr_bot_can_hit(c, th.lead)) continue;
            if (pick < 0) { pick = s; continue; }
            int cp = v.hand[pick];
            int better = th.hp >= 2000 ? PR_CARDS[c].elixir > PR_CARDS[cp].elixir
                                       : PR_CARDS[c].elixir < PR_CARDS[cp].elixir;
            if (better) pick = s;
        }
        if (pick >= 0) {
            const PrUnitDef *u = &PR_UNITS[PR_CARDS[(int)v.hand[pick]].unit];
            int tx, ty;
            if (u->range > PR_MELEE_RANGE) { /* ranged: behind the threatened lane's tower */
                tx = th.lead_x < 9000 ? 3 : 14;
                ty = 27;
            } else {                          /* melee: in the threat's path */
                tx = pr_bot_tile_x(th.lead_x);
                ty = PR_CLAMP(pr_bot_tile_y(th.lead_y) + 3, 17, 31);
            }
            int a = pr_bot_find(&v, pick, tx, ty, 3);
            if (a) { b->last_play_tick = st->tick; return a; }
        }
    }

    /* 3. offence and elixir management */
    int32_t full = st->elixir[team] + 2 * pr_elixir_rate(st->tick) >= PR_ELIXIR_MAX;
    if (st->tick - b->last_play_tick < 40 && !full) return 0;
    int32_t lane_x = pr_bot_attack_lane_x(b, &v);
    int lane_tx = pr_bot_tile_x(lane_x);
    if (lane_tx != 3 && lane_tx != 14) lane_tx = 9;
    /* 3a. support an advancing win condition from behind */
    if (th.own_win && v.elixir >= 4) {
        int32_t wx = pr_own_x(team, th.own_win->x), wy = pr_own_y(team, th.own_win->y);
        for (int s = 0; s < 4; s++) {
            int c = v.hand[s];
            if (PR_CARDS[c].kind != PR_CARD_KIND_TROOP || pr_bot_is_win_condition(c) || !pr_bot_slot_ok(&v, s)) continue;
            int a = pr_bot_find(&v, s, pr_bot_tile_x(wx), PR_CLAMP(pr_bot_tile_y(wy) + 2, 17, 31), 2);
            if (a) { b->last_play_tick = st->tick; return a; }
        }
    }
    /* 3b. a win condition at the bridge of the weaker lane */
    if (v.elixir >= 7 || full) {
        for (int s = 0; s < 4; s++) {
            int c = v.hand[s];
            if (!pr_bot_is_win_condition(c) || !pr_bot_slot_ok(&v, s)) continue;
            int a;
            const PrCardDef *cd = &PR_CARDS[c];
            if (cd->spell_type == PR_SPELL_SPAWN_PROJECTILE || cd->placement == PR_PLACE_MINER) {
                /* Goblin Barrel / Miner: on (next to) the target tower */
                int enemy_idx = lane_tx == 3 ? (team == 0 ? 1 : 2) : lane_tx == 14 ? (team == 0 ? 2 : 1) : 0;
                const PrEntity *tw = pr_get_c(st, st->tower_id[enemy][enemy_idx]);
                if (!tw) continue;
                a = pr_bot_find(&v, s, pr_bot_tile_x(pr_own_x(team, tw->x)), pr_bot_tile_y(pr_own_y(team, tw->y)) + 2, 2);
            } else if (cd->kind == PR_CARD_KIND_BUILDING) { /* siege: at the river of the lane */
                a = pr_bot_find(&v, s, lane_tx == 9 ? 9 : lane_tx, 18, 3);
            } else {
                int jitter = (int)pr_rng_below(&b->rng, 3) - 1;
                const PrUnitDef *u = &PR_UNITS[cd->unit];
                /* slow tanks start further back (Golem behind the King), fast ones at the bridge */
                int ty = u->speed <= 45 ? (u->hp >= 5000 ? 29 : 19) : 17;
                a = pr_bot_find(&v, s, PR_CLAMP(lane_tx + jitter, 0, 17), ty, 3);
            }
            if (a) { b->last_play_tick = st->tick; return a; }
        }
    }
    /* 3c. never sit at full elixir: the cheapest card, placed safely */
    if (full) {
        int pick = -1;
        for (int s = 0; s < 4; s++) {
            int c = v.hand[s];
            if (!pr_bot_slot_ok(&v, s) || PR_CARDS[c].kind == PR_CARD_KIND_SPELL) continue;
            if (PR_CARDS[c].placement == PR_PLACE_MINER) continue; /* not a safe spender */
            if (pick < 0 || PR_CARDS[c].elixir < PR_CARDS[(int)v.hand[pick]].elixir) pick = s;
        }
        if (pick >= 0) {
            int c = v.hand[pick];
            int a = PR_CARDS[c].kind == PR_CARD_KIND_BUILDING ? pr_bot_find(&v, pick, 9, 22, 3)
                                                              : pr_bot_find(&v, pick, lane_tx, 26, 4);
            if (a) { b->last_play_tick = st->tick; return a; }
        }
    }
    return 0;
}

/* One decision for `team`: an action index that is legal right now (0 = no-op). */
static inline int pr_bot_act(PrBot *b, const PrState *st, int team) {
    if (b->kind == PR_BOT_NOOP || st->over) return 0;
    uint8_t mask[PR_N_ACTIONS];
    pr_legal_mask(st, team, mask);
    int a = b->kind == PR_BOT_RANDOM ? pr_bot_random(b, st, team, mask) : pr_bot_heuristic(b, st, team, mask);
    if (a < 0 || a >= PR_N_ACTIONS || !mask[a]) return 0; /* belt and braces: never illegal */
    return a;
}

#endif /* PR_BOTS_H */
