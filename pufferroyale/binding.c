/*
 * binding.c -- the pufferroyale.binding extension module.
 *
 * Two surfaces share the one compiled engine:
 *   1. the PufferLib 3.0 env protocol (env_init / vec_step / ...) from the vendored
 *      env_binding.h, driving the Env in csrc/royale.h;
 *   2. the scenario / debug API behind pufferroyale.Game (SPEC §10): game_* functions
 *      on a capsule that owns one PrGame, plus card_info / db_info.
 */
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

#include "csrc/royale.h"

#define Env Royale

/* ---- prototypes of the extra module methods (defined below env_binding.h) ---- */
static PyObject *py_game_new(PyObject *self, PyObject *args);
static PyObject *py_game_reset(PyObject *self, PyObject *args);
static PyObject *py_game_tick(PyObject *self, PyObject *args);
static PyObject *py_game_play_tile(PyObject *self, PyObject *args);
static PyObject *py_game_play_xy(PyObject *self, PyObject *args);
static PyObject *py_game_spawn(PyObject *self, PyObject *args);
static PyObject *py_game_entities(PyObject *self, PyObject *args);
static PyObject *py_game_projectiles(PyObject *self, PyObject *args);
static PyObject *py_game_effects(PyObject *self, PyObject *args);
static PyObject *py_game_state(PyObject *self, PyObject *args);
static PyObject *py_game_set_elixir(PyObject *self, PyObject *args);
static PyObject *py_game_set_tower_hp(PyObject *self, PyObject *args);
static PyObject *py_game_set_hand(PyObject *self, PyObject *args);
static PyObject *py_game_legal_mask(PyObject *self, PyObject *args);
static PyObject *py_game_hash(PyObject *self, PyObject *args);
static PyObject *py_game_snapshot(PyObject *self, PyObject *args);
static PyObject *py_game_restore(PyObject *self, PyObject *args);
static PyObject *py_game_debug_set_entity(PyObject *self, PyObject *args);
static PyObject *py_game_path_stats(PyObject *self, PyObject *args);
static PyObject *py_card_info(PyObject *self, PyObject *args);
static PyObject *py_db_info(PyObject *self, PyObject *args);
static PyObject *py_game_set_first_team(PyObject *self, PyObject *args);
static PyObject *py_game_ansi(PyObject *self, PyObject *args);
static PyObject *py_game_obs(PyObject *self, PyObject *args);
static PyObject *py_bot_new(PyObject *self, PyObject *args);
static PyObject *py_bot_act(PyObject *self, PyObject *args);
static PyObject *py_bot_act_env(PyObject *self, PyObject *args);
static PyObject *py_vec_reset_seeded(PyObject *self, PyObject *args);
static PyObject *py_env_ansi(PyObject *self, PyObject *args);
static PyObject *py_env_info(PyObject *self, PyObject *args);
static PyObject *py_royale_layout(PyObject *self, PyObject *args);
static PyObject *py_env_log_peek(PyObject *self, PyObject *args); /* Phase D */

#define MY_METHODS \
    {"game_new", py_game_new, METH_VARARGS, "game_new(deck0, deck1, seed, lockout_ticks, tiebreak[, alternate, tower_troop0, tower_troop1]) -> capsule"}, \
    {"game_reset", py_game_reset, METH_VARARGS, "game_reset(g, seed_or_None)"}, \
    {"game_tick", py_game_tick, METH_VARARGS, "game_tick(g, n)"}, \
    {"game_play_tile", py_game_play_tile, METH_VARARGS, "game_play_tile(g, team, slot, tx, ty) -> err"}, \
    {"game_play_xy", py_game_play_xy, METH_VARARGS, "game_play_xy(g, team, slot, x, y) -> err"}, \
    {"game_spawn", py_game_spawn, METH_VARARGS, "game_spawn(g, team, card, x, y, deployed) -> [ids]"}, \
    {"game_entities", py_game_entities, METH_VARARGS, "game_entities(g) -> [dict]"}, \
    {"game_projectiles", py_game_projectiles, METH_VARARGS, "game_projectiles(g) -> [dict]"}, \
    {"game_effects", py_game_effects, METH_VARARGS, "game_effects(g) -> [dict]"}, \
    {"game_state", py_game_state, METH_VARARGS, "game_state(g) -> dict"}, \
    {"game_set_elixir", py_game_set_elixir, METH_VARARGS, "game_set_elixir(g, team, units)"}, \
    {"game_set_tower_hp", py_game_set_tower_hp, METH_VARARGS, "game_set_tower_hp(g, team, idx, hp)"}, \
    {"game_set_hand", py_game_set_hand, METH_VARARGS, "game_set_hand(g, team, order8)"}, \
    {"game_legal_mask", py_game_legal_mask, METH_VARARGS, "game_legal_mask(g, team) -> ndarray[2305] uint8"}, \
    {"game_hash", py_game_hash, METH_VARARGS, "game_hash(g) -> int"}, \
    {"game_snapshot", py_game_snapshot, METH_VARARGS, "game_snapshot(g) -> bytes"}, \
    {"game_restore", py_game_restore, METH_VARARGS, "game_restore(g, bytes)"}, \
    {"game_debug_set_entity", py_game_debug_set_entity, METH_VARARGS, "game_debug_set_entity(g, id, field, value)"}, \
    {"game_path_stats", py_game_path_stats, METH_VARARGS, "game_path_stats(g) -> dict"}, \
    {"card_info", py_card_info, METH_VARARGS, "card_info(card_id) -> dict"}, \
    {"db_info", py_db_info, METH_NOARGS, "db_info() -> dict"}, \
    {"game_set_first_team", py_game_set_first_team, METH_VARARGS, "game_set_first_team(g, team)"}, \
    {"game_ansi", py_game_ansi, METH_VARARGS, "game_ansi(g) -> str"}, \
    {"game_obs", py_game_obs, METH_VARARGS, "game_obs(g, team) -> ndarray[OBS_SIZE] float32"}, \
    {"bot_new", py_bot_new, METH_VARARGS, "bot_new(kind, seed, play_ppm) -> capsule"}, \
    {"bot_act", py_bot_act, METH_VARARGS, "bot_act(bot, g, team) -> action"}, \
    {"bot_act_env", py_bot_act_env, METH_VARARGS, "bot_act_env(bot, env_handle, team) -> action"}, \
    {"vec_reset_seeded", py_vec_reset_seeded, METH_VARARGS, "vec_reset_seeded(vec, seed_or_None)"}, \
    {"env_ansi", py_env_ansi, METH_VARARGS, "env_ansi(env) -> str"}, \
    {"env_info", py_env_info, METH_VARARGS, "env_info(env) -> dict"}, \
    {"royale_layout", py_royale_layout, METH_NOARGS, "royale_layout() -> dict"}, \
    {"env_log_peek", py_env_log_peek, METH_VARARGS, "env_log_peek(env) -> dict of raw Log sums (not cleared)"}

/* The vendored PufferLib header is not ours to edit; silence its -Wextra noise only (clang and
 * gcc spell the pragmas differently; each block is guarded so neither warns about the other's). */
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"
#pragma clang diagnostic ignored "-Wcast-function-type"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "env_binding.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/* ================================================================== env protocol */

static double kw_num(PyObject *kwargs, const char *key, double dflt) {
    PyObject *v = PyDict_GetItemString(kwargs, key);
    if (!v || v == Py_None) return dflt;
    if (PyLong_Check(v)) return (double)PyLong_AsLongLong(v);
    if (PyFloat_Check(v)) return PyFloat_AsDouble(v);
    PyErr_Format(PyExc_TypeError, "keyword %s must be a number", key);
    return dflt;
}

/* A deck keyword: None -> random per reset (returns 1), or 8 distinct card ids. */
static int kw_deck(PyObject *kwargs, const char *key, int8_t out[8]) {
    PyObject *v = PyDict_GetItemString(kwargs, key);
    if (!v || v == Py_None) return 1;
    PyObject *seq = PySequence_Fast(v, "deck must be None or 8 card ids");
    if (!seq) return -1;
    if (PySequence_Fast_GET_SIZE(seq) != 8) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "a deck has exactly 8 cards");
        return -1;
    }
    for (int i = 0; i < 8; i++) {
        long c = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if ((c == -1 && PyErr_Occurred()) || c < 0 || c >= PR_N_CARDS) {
            Py_DECREF(seq);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "card id out of range");
            return -1;
        }
        for (int j = 0; j < i; j++)
            if (out[j] == (int8_t)c) {
                Py_DECREF(seq);
                PyErr_SetString(PyExc_ValueError, "a deck has 8 distinct cards");
                return -1;
            }
        out[i] = (int8_t)c;
    }
    Py_DECREF(seq);
    return 0;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    (void)args;
    env->num_agents = (int)kw_num(kwargs, "num_agents", 2);
    env->frame_skip = (int)kw_num(kwargs, "frame_skip", 10);
    env->opponent = (int)kw_num(kwargs, "opponent", PR_BOT_HEURISTIC);
    env->learner_cfg = (int)kw_num(kwargs, "learner_side", -1);
    int lockout = (int)kw_num(kwargs, "deploy_lockout_ticks", PR_DEFAULT_LOCKOUT);
    int tiebreak = (int)kw_num(kwargs, "tiebreak", PR_TIEBREAK_ABSOLUTE);
    env->reward_tower = (float)kw_num(kwargs, "reward_tower", 0.0);
    env->reward_crown = (float)kw_num(kwargs, "reward_crown", 0.0);
    double prob = kw_num(kwargs, "bot_play_prob", 0.2);
    env->mask_check = (int)kw_num(kwargs, "mask_check", 0);
    env->render_mode = (int)kw_num(kwargs, "render_mode", PR_RENDER_NONE);
    long long seed = (long long)kw_num(kwargs, "seed", 0);
    int tt0 = (int)kw_num(kwargs, "tower_troop0", 0), tt1 = (int)kw_num(kwargs, "tower_troop1", 0);
    if (PyErr_Occurred()) return -1;
    if (tt0 < 0 || tt0 >= PR_N_TOWER_TROOPS || tt1 < 0 || tt1 >= PR_N_TOWER_TROOPS) {
        PyErr_SetString(PyExc_ValueError, "tower_troop0/1 must be a tower-troop index 0..3 (SPEC §16.3)");
        return -1;
    }
    int8_t d0[8], d1[8];
    int r0 = kw_deck(kwargs, "deck0", d0), r1 = kw_deck(kwargs, "deck1", d1);
    if (r0 < 0 || r1 < 0) return -1;
    if (env->num_agents < 1 || env->num_agents > 2) {
        PyErr_SetString(PyExc_ValueError, "num_agents must be 1 or 2");
        return -1;
    }
    if (env->opponent < PR_BOT_NOOP || env->opponent > PR_BOT_HEURISTIC) {
        PyErr_SetString(PyExc_ValueError, "bad opponent");
        return -1;
    }
    if (env->learner_cfg < -1 || env->learner_cfg > 1) {
        PyErr_SetString(PyExc_ValueError, "learner_side must be 0, 1 or -1 (random)");
        return -1;
    }
    if (env->frame_skip < 1) env->frame_skip = 1;
    if (prob < 0.0) prob = 0.0;
    if (prob > 1.0) prob = 1.0;
    env->bot_ppm = (uint32_t)(prob * 1000000.0 + 0.5);
    pr_setup_ex(&env->game, r0 ? NULL : d0, r1 ? NULL : d1, (uint64_t)seed, lockout,
                tiebreak ? PR_TIEBREAK_FRACTION : PR_TIEBREAK_ABSOLUTE, tt0, tt1);
    env->game.st.alternate_first = (uint8_t)(kw_num(kwargs, "alternate_first", 0) != 0);
    royale_seed(env, (uint64_t)seed);
    env->learner = env->learner_cfg < 0 ? 0 : env->learner_cfg;
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "win_0", log->win_0);
    assign_to_dict(dict, "win_1", log->win_1);
    assign_to_dict(dict, "draw", log->draw);
    assign_to_dict(dict, "overtime", log->overtime);
    assign_to_dict(dict, "tiebreak", log->tiebreak);
    assign_to_dict(dict, "crowns_0", log->crowns_0);
    assign_to_dict(dict, "crowns_1", log->crowns_1);
    assign_to_dict(dict, "leaked_0", log->leaked_0);
    assign_to_dict(dict, "leaked_1", log->leaked_1);
    assign_to_dict(dict, "plays_0", log->plays_0);
    assign_to_dict(dict, "plays_1", log->plays_1);
    assign_to_dict(dict, "illegal_actions", log->illegal_actions);
    assign_to_dict(dict, "spawn_overflow", log->spawn_overflow);
    assign_to_dict(dict, "learner_return", log->learner_return);
    assign_to_dict(dict, "learner_score", log->learner_score);
    assign_to_dict(dict, "learner_win", log->learner_win);
    assign_to_dict(dict, "dropped_plays", log->dropped_plays);
    assign_to_dict(dict, "match_ticks", log->match_ticks);
    assign_to_dict(dict, "mask_mismatch", log->mask_mismatch);
    return 0;
}

/* ================================================================== helpers */

#define GAME_CAPSULE "pufferroyale.Game"

static void game_capsule_free(PyObject *cap) {
    void *p = PyCapsule_GetPointer(cap, GAME_CAPSULE);
    free(p);
}

static PrGame *get_game(PyObject *cap) {
    PrGame *g = (PrGame *)PyCapsule_GetPointer(cap, GAME_CAPSULE);
    return g; /* NULL with an exception set on a bad object */
}

static int dset(PyObject *d, const char *k, PyObject *v) {
    if (!v) return -1;
    int r = PyDict_SetItemString(d, k, v);
    Py_DECREF(v);
    return r;
}
#define DSET_INT(d, k, v) dset((d), (k), PyLong_FromLongLong((long long)(v)))
#define DSET_BOOL(d, k, v) dset((d), (k), PyBool_FromLong((long)((v) != 0)))
#define DSET_STR(d, k, v) dset((d), (k), PyUnicode_FromString((v)))

static PyObject *int_list(const int32_t *v, int n) {
    PyObject *l = PyList_New(n);
    if (!l) return NULL;
    for (int i = 0; i < n; i++) PyList_SET_ITEM(l, i, PyLong_FromLong(v[i]));
    return l;
}

static PyObject *i8_list(const int8_t *v, int n) {
    PyObject *l = PyList_New(n);
    if (!l) return NULL;
    for (int i = 0; i < n; i++) PyList_SET_ITEM(l, i, PyLong_FromLong(v[i]));
    return l;
}

/* Parse an optional 8-card deck: None -> random (returns 0 and *is_random = 1). */
static int parse_deck(PyObject *o, int8_t out[8], int *is_random) {
    *is_random = 0;
    if (o == Py_None) {
        *is_random = 1;
        return 0;
    }
    PyObject *seq = PySequence_Fast(o, "deck must be a sequence of 8 card ids or None");
    if (!seq) return -1;
    if (PySequence_Fast_GET_SIZE(seq) != 8) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "a deck has exactly 8 cards");
        return -1;
    }
    for (int i = 0; i < 8; i++) {
        long c = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, i));
        if (c == -1 && PyErr_Occurred()) { Py_DECREF(seq); return -1; }
        if (c < 0 || c >= PR_N_CARDS) {
            Py_DECREF(seq);
            PyErr_Format(PyExc_ValueError, "card id %ld out of range [0, %d)", c, PR_N_CARDS);
            return -1;
        }
        out[i] = (int8_t)c;
    }
    Py_DECREF(seq);
    return 0;
}

static const char *kind_name(int kind) {
    switch (kind) {
    case PR_KIND_TROOP: return "troop";
    case PR_KIND_BUILDING: return "building";
    default: return "tower";
    }
}

static const char *end_reason_name(int r) {
    switch (r) {
    case PR_END_KING: return "KING";
    case PR_END_REGULATION_CROWNS: return "REGULATION_CROWNS";
    case PR_END_OVERTIME_CROWNS: return "OVERTIME_CROWNS";
    case PR_END_TIEBREAK: return "TIEBREAK";
    case PR_END_DRAW: return "DRAW";
    default: return NULL;
    }
}

/* ================================================================== game API */

static PyObject *py_game_new(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *d0, *d1, *seed_o;
    int lockout, tiebreak, alternate = 0, tt0 = 0, tt1 = 0;
    if (!PyArg_ParseTuple(args, "OOOii|pii", &d0, &d1, &seed_o, &lockout, &tiebreak, &alternate, &tt0, &tt1)) return NULL;
    if (tt0 < 0 || tt0 >= PR_N_TOWER_TROOPS || tt1 < 0 || tt1 >= PR_N_TOWER_TROOPS) {
        PyErr_SetString(PyExc_ValueError, "tower troop index must be 0..3 (SPEC §16.3)");
        return NULL;
    }
    int8_t deck0[8], deck1[8];
    int r0, r1;
    if (parse_deck(d0, deck0, &r0) < 0 || parse_deck(d1, deck1, &r1) < 0) return NULL;
    unsigned long long seed = PyLong_AsUnsignedLongLongMask(seed_o);
    if (PyErr_Occurred()) return NULL;
    if (tiebreak != PR_TIEBREAK_ABSOLUTE && tiebreak != PR_TIEBREAK_FRACTION) {
        PyErr_SetString(PyExc_ValueError, "bad tiebreak");
        return NULL;
    }
    PrGame *g = (PrGame *)calloc(1, sizeof(PrGame));
    if (!g) return PyErr_NoMemory();
    pr_setup_ex(g, r0 ? NULL : deck0, r1 ? NULL : deck1, (uint64_t)seed, lockout, tiebreak, tt0, tt1);
    g->st.alternate_first = (uint8_t)(alternate != 0);
    PyObject *cap = PyCapsule_New(g, GAME_CAPSULE, game_capsule_free);
    if (!cap) free(g);
    return cap;
}

static PyObject *py_game_reset(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seed_o;
    if (!PyArg_ParseTuple(args, "OO", &cap, &seed_o)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (seed_o != Py_None) {
        unsigned long long seed = PyLong_AsUnsignedLongLongMask(seed_o);
        if (PyErr_Occurred()) return NULL;
        pr_reseed(&g->st, (uint64_t)seed);
    }
    pr_new_match(&g->st);
    Py_RETURN_NONE;
}

static PyObject *py_game_tick(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    long n;
    if (!PyArg_ParseTuple(args, "Ol", &cap, &n)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    for (long i = 0; i < n && !g->st.over; i++) pr_tick(g);
    Py_RETURN_NONE;
}

static PyObject *py_game_play_tile(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    int team, slot, tx, ty;
    if (!PyArg_ParseTuple(args, "Oiiii", &cap, &team, &slot, &tx, &ty)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (team < 0 || team > 1) {
        PyErr_SetString(PyExc_ValueError, "team must be 0 or 1");
        return NULL;
    }
    return PyLong_FromLong(pr_queue_play(&g->st, team, slot, tx, ty));
}

/* (x, y) engine millitiles = the tap point (see pr_queue_play_at). */
static PyObject *py_game_play_xy(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    int team, slot;
    long long x, y;
    if (!PyArg_ParseTuple(args, "OiiLL", &cap, &team, &slot, &x, &y)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (team < 0 || team > 1) {
        PyErr_SetString(PyExc_ValueError, "team must be 0 or 1");
        return NULL;
    }
    if (x < -100000000LL || x > 100000000LL || y < -100000000LL || y > 100000000LL) { x = -1; y = -1; }
    return PyLong_FromLong(pr_queue_play_at(&g->st, team, slot, (int64_t)x, (int64_t)y));
}

static PyObject *py_game_spawn(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    int team, card, deployed;
    long long x, y;
    if (!PyArg_ParseTuple(args, "OiiLLp", &cap, &team, &card, &x, &y, &deployed)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (team < 0 || team > 1 || card < 0 || card >= PR_N_CARDS) {
        PyErr_SetString(PyExc_ValueError, "bad team or card id");
        return NULL;
    }
    uint32_t ids[64];
    x = PR_CLAMP(x, -PR_DEBUG_COORD_MAX, PR_DEBUG_COORD_MAX); /* no signed overflow downstream */
    y = PR_CLAMP(y, -PR_DEBUG_COORD_MAX, PR_DEBUG_COORD_MAX);
    int n = pr_debug_spawn(g, team, card, (int32_t)x, (int32_t)y, deployed, ids, 64);
    PyObject *l = PyList_New(n);
    if (!l) return NULL;
    for (int i = 0; i < n; i++) PyList_SET_ITEM(l, i, PyLong_FromUnsignedLong(ids[i]));
    return l;
}

static PyObject *entity_dict(const PrState *st, const PrEntity *e) {
    const PrUnitDef *d = pr_udef(e);
    PyObject *o = PyDict_New();
    if (!o) return NULL;
    DSET_INT(o, "id", e->id);
    DSET_INT(o, "team", e->team);
    DSET_INT(o, "card_id", e->card);
    DSET_STR(o, "unit", d->name);
    DSET_STR(o, "kind", kind_name(e->kind));
    DSET_INT(o, "x", e->x);
    DSET_INT(o, "y", e->y);
    DSET_INT(o, "hp", e->hp);
    DSET_INT(o, "max_hp", e->max_hp);
    DSET_INT(o, "shield", e->shield);
    DSET_INT(o, "radius", e->radius);
    DSET_BOOL(o, "flying", e->flying);
    DSET_BOOL(o, "deploying", pr_is_deploying(e));
    DSET_BOOL(o, "stunned", pr_is_stunned(e));
    DSET_BOOL(o, "slowed", pr_is_slowed(e));
    DSET_BOOL(o, "hidden", pr_is_hidden(e));
    const PrEntity *t = pr_get_c(st, e->target_id);
    DSET_INT(o, "target_id", t ? (long long)t->id : -1LL);
    DSET_BOOL(o, "charged", e->charged);
    /* extras (not in SPEC §10, stable nonetheless) */
    DSET_INT(o, "tower_idx", e->tower_idx);
    DSET_INT(o, "deploy_ms", e->deploy_ms);
    DSET_INT(o, "load_ms", e->load_ms);
    DSET_INT(o, "progress_ms", e->progress_ms);
    DSET_BOOL(o, "attacking", e->attacking);
    DSET_BOOL(o, "jumping", e->jumping);
    DSET_INT(o, "hide_state", e->hide_state);
    DSET_INT(o, "mass", e->mass);
    DSET_INT(o, "speed", d->speed);
    DSET_INT(o, "charge_acc", e->charge_acc);
    DSET_INT(o, "spawn_tick", e->spawn_tick);
    /* SPEC §16 extras */
    DSET_INT(o, "level", PR_CARD_LEVEL + e->lvl);
    DSET_BOOL(o, "burrowing", e->burrow);
    DSET_INT(o, "dash_state", e->dash_state);
    DSET_INT(o, "var_hits", e->var_hits);
    return o;
}

static PyObject *py_game_entities(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    const PrState *st = &g->st;
    PyObject *l = PyList_New(0);
    if (!l) return NULL;
    for (int i = 0; i < st->n_ent; i++) {
        if (st->ent[i].hp <= 0) continue;
        PyObject *o = entity_dict(st, &st->ent[i]);
        if (!o || PyList_Append(l, o) < 0) { Py_XDECREF(o); Py_DECREF(l); return NULL; }
        Py_DECREF(o);
    }
    return l;
}

static PyObject *py_game_projectiles(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    const PrState *st = &g->st;
    PyObject *l = PyList_New(0);
    if (!l) return NULL;
    for (int i = 0; i < st->n_proj; i++) {
        const PrProjectile *p = &st->proj[i];
        PyObject *o = PyDict_New();
        if (!o) { Py_DECREF(l); return NULL; }
        DSET_INT(o, "id", p->id);
        DSET_STR(o, "name", PR_PROJS[p->proj].name);
        DSET_INT(o, "team", p->team);
        DSET_INT(o, "card_id", p->card);
        DSET_INT(o, "x", p->x);
        DSET_INT(o, "y", p->y);
        DSET_INT(o, "target_x", p->tx);
        DSET_INT(o, "target_y", p->ty);
        DSET_INT(o, "target_id", p->has_target ? (long long)p->target_id : -1LL);
        DSET_INT(o, "owner_id", p->owner_id == PR_NO_ID ? -1LL : (long long)p->owner_id);
        DSET_INT(o, "damage", p->damage);
        DSET_INT(o, "radius", p->radius);
        DSET_INT(o, "speed", p->speed);
        if (PyList_Append(l, o) < 0) { Py_DECREF(o); Py_DECREF(l); return NULL; }
        Py_DECREF(o);
    }
    return l;
}

static PyObject *py_game_effects(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    const PrState *st = &g->st;
    PyObject *l = PyList_New(0);
    if (!l) return NULL;
    for (int i = 0; i < st->n_fx; i++) {
        const PrEffect *f = &st->fx[i];
        PyObject *o = PyDict_New();
        if (!o) { Py_DECREF(l); return NULL; }
        const char *kind = f->type == PR_FX_AREA ? "area" : f->type == PR_FX_WAVES ? "waves"
                         : f->type == PR_FX_ROLLING ? "rolling" : f->type == PR_FX_PULSE ? "pulse"
                         : f->type == PR_FX_LIGHTNING ? "lightning" : "bomb";
        DSET_INT(o, "id", f->id);
        DSET_STR(o, "kind", kind);
        DSET_STR(o, "name", f->area >= 0 ? PR_AREAS[f->area].name : (f->proj >= 0 ? PR_PROJS[f->proj].name : ""));
        DSET_INT(o, "team", f->team);
        DSET_INT(o, "card_id", f->card);
        DSET_INT(o, "x", f->x);
        DSET_INT(o, "y", f->y);
        DSET_INT(o, "radius", f->radius);
        DSET_INT(o, "radius_y", f->radius_y);
        DSET_INT(o, "damage", f->damage);
        DSET_INT(o, "timer_ms", f->timer_ms);
        DSET_INT(o, "remaining", f->remaining);
        if (PyList_Append(l, o) < 0) { Py_DECREF(o); Py_DECREF(l); return NULL; }
        Py_DECREF(o);
    }
    return l;
}

static PyObject *py_game_state(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    const PrState *st = &g->st;
    PyObject *d = PyDict_New();
    if (!d) return NULL;
    DSET_INT(d, "tick", st->tick);
    dset(d, "elixir", int_list(st->elixir, 2));
    PyObject *hand = PyList_New(2), *queue = PyList_New(2), *deck = PyList_New(2);
    PyObject *towers = PyList_New(2), *lastp = PyList_New(2);
    for (int t = 0; t < 2; t++) {
        PyList_SET_ITEM(hand, t, i8_list(st->hand[t], 4));
        PyList_SET_ITEM(queue, t, i8_list(st->queue[t], 4));
        PyList_SET_ITEM(deck, t, i8_list(st->deck[t], 8));
        PyList_SET_ITEM(lastp, t, i8_list(st->last_played[t], 4));
        PyObject *tl = PyList_New(3);
        for (int idx = 0; idx < 3; idx++) {
            const PrEntity *e = pr_get_c(st, st->tower_id[t][idx]);
            PyObject *td = PyDict_New();
            DSET_INT(td, "hp", e ? e->hp : 0);
            DSET_INT(td, "max_hp", e ? e->max_hp : PR_UNITS[idx == 0 ? PR_UNIT_KINGTOWER : PR_TOWER_TROOP_UNITS[st->tower_troop[t]]].hp);
            DSET_BOOL(td, "alive", e != NULL);
            DSET_BOOL(td, "active", e != NULL && (idx != 0 || st->king_active[t]));
            DSET_INT(td, "id", st->tower_id[t][idx]);
            PyList_SET_ITEM(tl, idx, td);
        }
        PyList_SET_ITEM(towers, t, tl);
    }
    dset(d, "hand", hand);
    dset(d, "queue", queue);
    dset(d, "deck", deck);
    dset(d, "crowns", int_list(st->crowns, 2));
    dset(d, "towers", towers);
    PyObject *tts = PyList_New(2);
    PyList_SET_ITEM(tts, 0, PyUnicode_FromString(PR_TOWER_TROOP_NAMES[st->tower_troop[0]]));
    PyList_SET_ITEM(tts, 1, PyUnicode_FromString(PR_TOWER_TROOP_NAMES[st->tower_troop[1]]));
    dset(d, "tower_troops", tts);
    DSET_BOOL(d, "over", st->over);
    PyObject *res = PyList_New(2);
    PyList_SET_ITEM(res, 0, PyLong_FromLong(st->result[0]));
    PyList_SET_ITEM(res, 1, PyLong_FromLong(st->result[1]));
    dset(d, "result", res);
    const char *er = end_reason_name(st->end_reason);
    if (er) DSET_STR(d, "end_reason", er);
    else PyDict_SetItemString(d, "end_reason", Py_None);
    DSET_BOOL(d, "overtime", st->tick >= PR_TICKS_REGULATION);
    /* extras */
    DSET_INT(d, "elixir_rate", pr_elixir_rate(st->tick));
    dset(d, "king_wake", int_list(st->king_wake, 2));
    dset(d, "leaked", int_list(st->leaked, 2));
    dset(d, "plays", int_list(st->plays, 2));
    dset(d, "spent", int_list(st->spent, 2));
    dset(d, "last_played", lastp);
    PyObject *seen = PyList_New(2);
    PyList_SET_ITEM(seen, 0, PyLong_FromUnsignedLongLong((unsigned long long)st->seen_mask[0]));
    PyList_SET_ITEM(seen, 1, PyLong_FromUnsignedLongLong((unsigned long long)st->seen_mask[1]));
    dset(d, "seen_mask", seen);
    DSET_INT(d, "dropped_plays", st->dropped_plays);
    DSET_INT(d, "spawn_overflow", st->spawn_overflow);
    DSET_INT(d, "lockout_ticks", st->lockout_ticks);
    PyObject *pend = PyList_New(2);
    PyList_SET_ITEM(pend, 0, PyLong_FromLong(st->n_pending[0]));
    PyList_SET_ITEM(pend, 1, PyLong_FromLong(st->n_pending[1]));
    dset(d, "pending_plays", pend);
    if (PyErr_Occurred()) { Py_DECREF(d); return NULL; }
    return d;
}

static PyObject *py_game_set_elixir(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    int team;
    long units;
    if (!PyArg_ParseTuple(args, "Oil", &cap, &team, &units)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (team < 0 || team > 1 || units < 0 || units > PR_ELIXIR_MAX) {
        PyErr_Format(PyExc_ValueError, "team must be 0/1 and units in [0, %d]", PR_ELIXIR_MAX);
        return NULL;
    }
    g->st.elixir[team] = (int32_t)units;
    Py_RETURN_NONE;
}

static PyObject *py_game_set_tower_hp(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    int team, idx;
    long hp;
    if (!PyArg_ParseTuple(args, "Oiil", &cap, &team, &idx, &hp)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (team < 0 || team > 1 || idx < 0 || idx > 2) {
        PyErr_SetString(PyExc_ValueError, "team must be 0/1 and idx 0..2");
        return NULL;
    }
    if (pr_debug_set_tower_hp(g, team, idx, (int32_t)PR_CLAMP(hp, -1000000L, 1000000L)) < 0) {
        PyErr_SetString(PyExc_ValueError, "that tower is already destroyed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_game_set_hand(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *order;
    int team;
    if (!PyArg_ParseTuple(args, "OiO", &cap, &team, &order)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (team < 0 || team > 1) {
        PyErr_SetString(PyExc_ValueError, "team must be 0 or 1");
        return NULL;
    }
    int8_t o[8];
    int rnd;
    if (order == Py_None || parse_deck(order, o, &rnd) < 0) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "set_hand needs 8 card ids");
        return NULL;
    }
    if (pr_debug_set_hand(g, team, o) < 0) {
        PyErr_SetString(PyExc_ValueError, "set_hand needs 8 distinct valid card ids");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_game_legal_mask(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    int team;
    if (!PyArg_ParseTuple(args, "Oi", &cap, &team)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (team < 0 || team > 1) {
        PyErr_SetString(PyExc_ValueError, "team must be 0 or 1");
        return NULL;
    }
    npy_intp dims[1] = {PR_N_ACTIONS};
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_UINT8);
    if (!arr) return NULL;
    pr_legal_mask(&g->st, team, (uint8_t *)PyArray_DATA((PyArrayObject *)arr));
    return arr;
}

static PyObject *py_game_hash(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    return PyLong_FromUnsignedLongLong((unsigned long long)pr_hash(&g->st));
}

static PyObject *py_game_snapshot(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    return PyBytes_FromStringAndSize((const char *)&g->st, (Py_ssize_t)sizeof(PrState));
}

static PyObject *py_game_restore(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    Py_buffer buf;
    if (!PyArg_ParseTuple(args, "Oy*", &cap, &buf)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) { PyBuffer_Release(&buf); return NULL; }
    if (buf.len != (Py_ssize_t)sizeof(PrState)) {
        PyBuffer_Release(&buf);
        PyErr_Format(PyExc_ValueError, "snapshot has %zd bytes, expected %zu", buf.len, sizeof(PrState));
        return NULL;
    }
    /* validate a private copy first (SPEC §14.5): an invalid snapshot raises ValueError and
     * leaves the game unchanged */
    PrState *tmp = (PrState *)malloc(sizeof(PrState));
    if (!tmp) { PyBuffer_Release(&buf); return PyErr_NoMemory(); }
    memcpy(tmp, buf.buf, sizeof(PrState));
    PyBuffer_Release(&buf);
    const char *why = "invalid snapshot";
    if (!pr_state_check(tmp, &why)) {
        free(tmp);
        PyErr_Format(PyExc_ValueError, "invalid snapshot: %s", why);
        return NULL;
    }
    pr_state_canonicalize(tmp); /* unused slots zeroed: the hash depends only on live content */
    memcpy(&g->st, tmp, sizeof(PrState));
    free(tmp);
    Py_RETURN_NONE;
}

/* Builder test hook (not in SPEC §10): poke one field of an entity. */
static PyObject *py_game_debug_set_entity(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    unsigned long id;
    const char *field;
    long long v;
    if (!PyArg_ParseTuple(args, "OksL", &cap, &id, &field, &v)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    PrEntity *e = pr_get(&g->st, (uint32_t)id);
    if (!e) {
        PyErr_SetString(PyExc_KeyError, "no live entity with that id");
        return NULL;
    }
    /* clamped to the ranges pr_state_check accepts, so a poked state stays well-formed */
    if (!strcmp(field, "hp")) e->hp = (int32_t)PR_CLAMP(v, -PR_CHK_BIG, e->max_hp); /* never above max_hp (SPEC §18.5) */
    else if (!strcmp(field, "x")) e->x = (int32_t)PR_CLAMP(v, -PR_DEBUG_COORD_MAX, PR_DEBUG_COORD_MAX);
    else if (!strcmp(field, "y")) e->y = (int32_t)PR_CLAMP(v, -PR_DEBUG_COORD_MAX, PR_DEBUG_COORD_MAX);
    else if (!strcmp(field, "deploy_ms")) e->deploy_ms = (int32_t)PR_CLAMP(v, 0, PR_CHK_BIG);
    else if (!strcmp(field, "load_ms")) e->load_ms = (int32_t)PR_CLAMP(v, 0, PR_CHK_BIG);
    else {
        PyErr_SetString(PyExc_KeyError, "field must be hp, x, y, deploy_ms or load_ms");
        return NULL;
    }
    e->wp_cell = -1;
    Py_RETURN_NONE;
}

static PyObject *py_game_path_stats(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    PyObject *d = PyDict_New();
    DSET_INT(d, "field_builds", g->pc.field_builds);
    dset(d, "sig", PyLong_FromUnsignedLongLong((unsigned long long)g->pc.sig));
    return d;
}

/* ================================================================== static data */

static PyObject *proj_dict(int p) {
    const PrProjDef *d = &PR_PROJS[p];
    PyObject *o = PyDict_New();
    DSET_STR(o, "name", d->name);
    DSET_INT(o, "speed", d->speed);
    DSET_INT(o, "damage", d->damage);
    DSET_INT(o, "crown_tower_damage_percent", d->crown_pct);
    DSET_INT(o, "radius", d->radius);
    DSET_INT(o, "radius_y", d->radius_y);
    DSET_BOOL(o, "aoe_to_air", d->aoe_air);
    DSET_BOOL(o, "aoe_to_ground", d->aoe_ground);
    DSET_BOOL(o, "homing", d->homing);
    DSET_INT(o, "pushback", d->pushback);
    DSET_BOOL(o, "pushback_all", d->pushback_all);
    DSET_INT(o, "range", d->range);
    DSET_STR(o, "target_buff", d->target_buff >= 0 ? PR_BUFFS[d->target_buff].name : "");
    DSET_INT(o, "buff_time_ms", d->buff_ms);
    DSET_STR(o, "spawn_unit", d->spawn_unit >= 0 ? PR_UNITS[d->spawn_unit].name : "");
    DSET_INT(o, "spawn_count", d->spawn_count);
    DSET_INT(o, "spawn_deploy_time_ms", d->spawn_deploy_ms);
    return o;
}

static PyObject *unit_dict(int u) {
    const PrUnitDef *d = &PR_UNITS[u];
    PyObject *o = PyDict_New();
    DSET_STR(o, "name", d->name);
    DSET_STR(o, "kind", kind_name(d->kind));
    DSET_INT(o, "hitpoints", d->hp);
    DSET_INT(o, "damage", d->damage);
    DSET_INT(o, "shield_hitpoints", d->shield);
    DSET_INT(o, "hit_speed_ms", d->hit_speed_ms);
    DSET_INT(o, "load_time_ms", d->load_time_ms);
    DSET_INT(o, "speed", d->speed);
    DSET_INT(o, "range", d->range);
    DSET_INT(o, "sight_range", d->sight);
    DSET_INT(o, "collision_radius", d->radius);
    DSET_INT(o, "mass", d->mass);
    DSET_INT(o, "deploy_time_ms", d->deploy_ms);
    DSET_INT(o, "flying_height", d->flying_height);
    DSET_BOOL(o, "attacks_air", d->attacks_air);
    DSET_BOOL(o, "attacks_ground", d->attacks_ground);
    DSET_BOOL(o, "target_only_buildings", d->only_buildings);
    DSET_INT(o, "area_damage_radius", d->area_radius);
    DSET_BOOL(o, "self_as_aoe_center", d->self_aoe);
    DSET_INT(o, "projectile_start_radius", d->proj_start_radius);
    DSET_INT(o, "crown_tower_damage_percent", d->crown_pct);
    DSET_INT(o, "lifetime_ms", d->lifetime_ms);
    DSET_INT(o, "death_damage", d->death_damage);
    DSET_INT(o, "death_damage_radius", d->death_radius);
    DSET_STR(o, "death_area_effect", d->death_area >= 0 ? PR_AREAS[d->death_area].name : "");
    DSET_BOOL(o, "kamikaze", d->kamikaze);
    DSET_BOOL(o, "ignore_pushback", d->ignore_pushback);
    DSET_BOOL(o, "jumps", d->jumps);
    DSET_INT(o, "charge_range", d->charge_range);
    DSET_INT(o, "damage_special", d->charge_damage);
    DSET_INT(o, "charge_speed_multiplier_percent", d->charge_speed_pct);
    DSET_BOOL(o, "hides_when_not_attacking", d->hides);
    DSET_INT(o, "hide_time_ms", d->hide_ms);
    DSET_INT(o, "up_time_ms", d->up_ms);
    DSET_INT(o, "footprint_tiles", d->footprint_tiles);
    /* SPEC §16 mechanics */
    DSET_INT(o, "minimum_range", d->min_range);
    DSET_BOOL(o, "attacks", !d->no_attack);
    if (d->var_damage2 > 0) {
        PyObject *vd = Py_BuildValue("(iii)", d->damage, d->var_damage2, d->var_damage3);
        dset(o, "variable_damage", vd);
        DSET_INT(o, "variable_damage_hits", d->var_hits1);
    }
    if (d->dash_damage > 0) {
        PyObject *dd = PyDict_New();
        DSET_INT(dd, "damage", d->dash_damage);
        DSET_INT(dd, "min_range", d->dash_min);
        DSET_INT(dd, "max_range", d->dash_max);
        DSET_INT(dd, "speed", d->dash_speed);
        DSET_INT(dd, "cooldown_ms", d->dash_cooldown_ms);
        dset(o, "dash", dd);
    }
    if (d->burrow_speed > 0) DSET_INT(o, "burrow_speed", d->burrow_speed);
    if (d->mana_gen_ms > 0) {
        DSET_INT(o, "mana_collect_amount", d->mana_amount);
        DSET_INT(o, "mana_generate_time_ms", d->mana_gen_ms);
        DSET_INT(o, "mana_on_death", d->mana_on_death);
    }
    if (d->spawn_unit >= 0) {
        PyObject *sp = PyDict_New();
        DSET_STR(sp, "character", PR_UNITS[d->spawn_unit].name);
        DSET_INT(sp, "number", d->spawn_number);
        DSET_INT(sp, "interval_ms", d->spawn_interval_ms);
        DSET_INT(sp, "start_time_ms", d->spawn_start_ms);
        DSET_INT(sp, "pause_time_ms", d->spawn_pause_ms);
        dset(o, "spawner", sp);
    }
    if (d->dspawn_unit >= 0) {
        PyObject *ds = PyDict_New();
        DSET_STR(ds, "character", PR_UNITS[d->dspawn_unit].name);
        DSET_INT(ds, "count", d->dspawn_count);
        DSET_INT(ds, "deploy_time_ms", d->dspawn_deploy_ms);
        dset(o, "death_spawn", ds);
    }
    if (d->bomb_damage > 0) {
        PyObject *db = PyDict_New();
        DSET_INT(db, "damage", d->bomb_damage);
        DSET_INT(db, "radius", d->bomb_radius);
        DSET_INT(db, "fuse_ms", d->bomb_fuse_ms);
        dset(o, "death_bomb", db);
    }
    if (d->seq_n > 0) {
        PyObject *sq = PyList_New(d->seq_n);
        for (int k = 0; k < d->seq_n; k++) PyList_SET_ITEM(sq, k, PyLong_FromLong(d->seq_mult[k]));
        dset(o, "attack_sequence", sq);
        DSET_INT(o, "reload_ms", d->reload_ms);
    }
    if (d->chef_period_ms > 0) {
        DSET_INT(o, "level_up_period_ms", d->chef_period_ms);
        DSET_INT(o, "level_up_range", d->chef_range);
    }
    /* millitile aliases in the source file's vocabulary */
    DSET_INT(o, "range_milli", d->range);
    DSET_INT(o, "sight_range_milli", d->sight);
    DSET_INT(o, "collision_radius_milli", d->radius);
    DSET_INT(o, "area_damage_radius_milli", d->area_radius);
    if (d->projectile >= 0) dset(o, "projectile", proj_dict(d->projectile));
    else PyDict_SetItemString(o, "projectile", Py_None);
    return o;
}

static PyObject *py_card_info(PyObject *self, PyObject *args) {
    (void)self;
    int card;
    if (!PyArg_ParseTuple(args, "i", &card)) return NULL;
    if (card < 0 || card >= PR_N_CARDS) {
        PyErr_Format(PyExc_ValueError, "card id must be in [0, %d)", PR_N_CARDS);
        return NULL;
    }
    const PrCardDef *c = &PR_CARDS[card];
    PyObject *o;
    if (c->kind != PR_CARD_KIND_SPELL) {
        o = unit_dict(c->unit);           /* the summoned unit's level-11 stats, flattened */
        DSET_STR(o, "unit", PR_UNITS[c->unit].name);
        if (c->unit2 >= 0) {              /* second_summon (Goblin Gang, Rascals) */
            dset(o, "second_unit", unit_dict(c->unit2));
            DSET_INT(o, "count2", c->count2);
        }
    } else {
        o = PyDict_New();
        PyDict_SetItemString(o, "unit", Py_None);
        DSET_INT(o, "damage", c->spell_damage);
        DSET_INT(o, "radius", c->spell_radius);
        DSET_INT(o, "area_damage_radius", c->spell_radius);
        DSET_INT(o, "crown_tower_damage_percent", c->crown_pct);
        DSET_INT(o, "crown_tower_damage", (int)pr_ceildiv((int64_t)c->spell_damage * c->crown_pct, 100));
        DSET_INT(o, "waves", c->waves);
        DSET_INT(o, "wave_interval_ms", c->wave_interval_ms);
        if (c->spell_type == PR_SPELL_PULSE) {   /* damage = one event (SPEC §16.6.10-11) */
            const PrBuffDef *b = &PR_BUFFS[PR_AREAS[c->spell_area].buff];
            DSET_INT(o, "events", c->waves);
            DSET_INT(o, "event_interval_ms", c->wave_interval_ms);
            DSET_INT(o, "damage_per_second", b->dps);
            DSET_INT(o, "building_damage_percent", b->building_pct);
            DSET_INT(o, "building_damage", (int)((int64_t)c->spell_damage * b->building_pct / 100));
        }
        if (c->spell_type == PR_SPELL_LIGHTNING) DSET_INT(o, "bolts", c->waves);
        if (c->spell_proj >= 0) dset(o, "projectile", proj_dict(c->spell_proj));
        else PyDict_SetItemString(o, "projectile", Py_None);
        if (c->spell_area >= 0) {
            const PrAreaDef *a = &PR_AREAS[c->spell_area];
            PyObject *ad = PyDict_New();
            DSET_STR(ad, "name", a->name);
            DSET_INT(ad, "radius", a->radius);
            DSET_INT(ad, "damage", a->damage);
            DSET_INT(ad, "life_duration_ms", a->life_ms);
            DSET_STR(ad, "buff", a->buff >= 0 ? PR_BUFFS[a->buff].name : "");
            DSET_INT(ad, "buff_time_ms", a->buff_ms);
            dset(o, "area_effect", ad);
        }
        if (c->spell_proj >= 0 && PR_PROJS[c->spell_proj].spawn_unit >= 0)
            dset(o, "spawn", unit_dict(PR_PROJS[c->spell_proj].spawn_unit));
    }
    DSET_INT(o, "id", card);
    DSET_STR(o, "name", c->name);
    DSET_STR(o, "key", c->key);
    DSET_STR(o, "card_kind", c->kind == PR_CARD_KIND_TROOP ? "troop" : c->kind == PR_CARD_KIND_BUILDING ? "building" : "spell");
    DSET_STR(o, "kind", c->kind == PR_CARD_KIND_TROOP ? "troop" : c->kind == PR_CARD_KIND_BUILDING ? "building" : "spell");
    DSET_INT(o, "elixir", c->elixir);
    DSET_INT(o, "cost", c->elixir);
    DSET_INT(o, "count", c->count);
    DSET_STR(o, "spell_type", c->spell_type == PR_SPELL_PROJECTILE ? "projectile" : c->spell_type == PR_SPELL_WAVES ? "waves"
                            : c->spell_type == PR_SPELL_AREA ? "area" : c->spell_type == PR_SPELL_ROLLING ? "rolling"
                            : c->spell_type == PR_SPELL_SPAWN_PROJECTILE ? "spawn_projectile"
                            : c->spell_type == PR_SPELL_PULSE ? "pulse" : c->spell_type == PR_SPELL_LIGHTNING ? "lightning" : "none");
    DSET_INT(o, "summon_deploy_delay_ms", c->deploy_delay_ms);
    DSET_INT(o, "level", PR_CARD_LEVEL);
    if (PyErr_Occurred()) { Py_DECREF(o); return NULL; }
    return o;
}

static PyObject *py_db_info(PyObject *self, PyObject *noargs) {
    (void)self;
    (void)noargs;
    PyObject *d = PyDict_New();
    PyObject *names = PyTuple_New(PR_N_CARDS), *keys = PyTuple_New(PR_N_CARDS), *costs = PyTuple_New(PR_N_CARDS);
    for (int c = 0; c < PR_N_CARDS; c++) {
        PyTuple_SET_ITEM(names, c, PyUnicode_FromString(PR_CARDS[c].name));
        PyTuple_SET_ITEM(keys, c, PyUnicode_FromString(PR_CARDS[c].key));
        PyTuple_SET_ITEM(costs, c, PyLong_FromLong(PR_CARDS[c].elixir));
    }
    dset(d, "card_names", names);
    dset(d, "card_keys", keys);
    dset(d, "card_costs", costs);
    PyObject *decks = PyDict_New();
    for (int k = 0; k < PR_N_DECKS; k++) {
        PyObject *t = PyTuple_New(8);
        for (int i = 0; i < 8; i++) PyTuple_SET_ITEM(t, i, PyLong_FromLong(PR_DECKS[k][i]));
        dset(decks, PR_DECK_NAMES[k], t);
    }
    dset(d, "decks", decks);
    DSET_INT(d, "n_cards", PR_N_CARDS);
    PyObject *tts = PyTuple_New(PR_N_TOWER_TROOPS);
    for (int k = 0; k < PR_N_TOWER_TROOPS; k++) PyTuple_SET_ITEM(tts, k, PyUnicode_FromString(PR_TOWER_TROOP_NAMES[k]));
    dset(d, "tower_troops", tts);
    PyObject *tthp = PyTuple_New(PR_N_TOWER_TROOPS);
    for (int k = 0; k < PR_N_TOWER_TROOPS; k++) PyTuple_SET_ITEM(tthp, k, unit_dict(PR_TOWER_TROOP_UNITS[k]));
    dset(d, "tower_troop_units", tthp);
    DSET_INT(d, "card_slots", PR_CARD_SLOTS);
    DSET_INT(d, "n_actions", PR_N_ACTIONS);
    DSET_INT(d, "n_tiles", PR_N_TILES);
    DSET_INT(d, "max_entities", PR_MAX_ENTITIES);
    DSET_INT(d, "max_projectiles", PR_MAX_PROJECTILES);
    DSET_INT(d, "max_effects", PR_MAX_EFFECTS);
    DSET_INT(d, "state_size", (long long)sizeof(PrState));
    DSET_STR(d, "data_sha256", PR_DATA_SHA256);
    DSET_STR(d, "arena_sha256", PR_ARENA_SHA256);
    DSET_INT(d, "obs_size", PR_OBS_SIZE);
    if (PyErr_Occurred()) { Py_DECREF(d); return NULL; }
    return d;
}


/* ================================================================== Phase C additions */

static PyObject *py_game_set_first_team(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    int team;
    if (!PyArg_ParseTuple(args, "Oi", &cap, &team)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    pr_debug_set_first_team(g, team);
    Py_RETURN_NONE;
}

static PyObject *py_game_ansi(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    char buf[PR_ANSI_BUF];
    int n = pr_render_ansi(&g->st, buf, PR_ANSI_BUF);
    return PyUnicode_FromStringAndSize(buf, n);
}

static PyObject *py_game_obs(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    int team;
    if (!PyArg_ParseTuple(args, "Oi", &cap, &team)) return NULL;
    PrGame *g = get_game(cap);
    if (!g) return NULL;
    if (team < 0 || team > 1) {
        PyErr_SetString(PyExc_ValueError, "team must be 0 or 1");
        return NULL;
    }
    npy_intp dims[1] = {PR_OBS_SIZE};
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!arr) return NULL;
    pr_obs_write(&g->st, team, (float *)PyArray_DATA((PyArrayObject *)arr));
    return arr;
}

#define BOT_CAPSULE "pufferroyale.Bot"

static void bot_capsule_free(PyObject *cap) { free(PyCapsule_GetPointer(cap, BOT_CAPSULE)); }

static PyObject *py_bot_new(PyObject *self, PyObject *args) {
    (void)self;
    int kind;
    unsigned long long seed;
    unsigned long ppm;
    if (!PyArg_ParseTuple(args, "iKk", &kind, &seed, &ppm)) return NULL;
    if (kind < PR_BOT_NOOP || kind > PR_BOT_HEURISTIC) {
        PyErr_SetString(PyExc_ValueError, "bot kind must be 0 (noop), 1 (random) or 2 (heuristic)");
        return NULL;
    }
    PrBot *b = (PrBot *)calloc(1, sizeof(PrBot));
    if (!b) return PyErr_NoMemory();
    pr_bot_init(b, kind, (uint32_t)ppm, (uint64_t)seed);
    PyObject *cap = PyCapsule_New(b, BOT_CAPSULE, bot_capsule_free);
    if (!cap) free(b);
    return cap;
}

static PyObject *py_bot_act(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *bcap, *gcap;
    int team;
    if (!PyArg_ParseTuple(args, "OOi", &bcap, &gcap, &team)) return NULL;
    PrBot *b = (PrBot *)PyCapsule_GetPointer(bcap, BOT_CAPSULE);
    if (!b) return NULL;
    PrGame *g = get_game(gcap);
    if (!g) return NULL;
    if (team < 0 || team > 1) {
        PyErr_SetString(PyExc_ValueError, "team must be 0 or 1");
        return NULL;
    }
    return PyLong_FromLong(pr_bot_act(b, &g->st, team));
}

static PyObject *py_bot_act_env(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *bcap, *handle;
    int team;
    if (!PyArg_ParseTuple(args, "OOi", &bcap, &handle, &team)) return NULL;
    PrBot *b = (PrBot *)PyCapsule_GetPointer(bcap, BOT_CAPSULE);
    if (!b) return NULL;
    Env *env = (Env *)PyLong_AsVoidPtr(handle);
    if (!env || team < 0 || team > 1) {
        PyErr_SetString(PyExc_ValueError, "invalid env handle or team");
        return NULL;
    }
    return PyLong_FromLong(pr_bot_act(b, &env->game.st, team));
}

static PyObject *py_vec_reset_seeded(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *handle, *seed_o;
    if (!PyArg_ParseTuple(args, "OO", &handle, &seed_o)) return NULL;
    VecEnv *vec = (VecEnv *)PyLong_AsVoidPtr(handle);
    if (!vec || vec->num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    int have_seed = seed_o != Py_None;
    unsigned long long seed = 0;
    if (have_seed) {
        seed = PyLong_AsUnsignedLongLongMask(seed_o);
        if (PyErr_Occurred()) return NULL;
    }
    for (int i = 0; i < vec->num_envs; i++) {
        if (have_seed) royale_seed(vec->envs[i], (uint64_t)seed * (uint64_t)vec->num_envs + (uint64_t)i);
        c_reset(vec->envs[i]);
    }
    Py_RETURN_NONE;
}

static PyObject *py_env_ansi(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *handle;
    if (!PyArg_ParseTuple(args, "O", &handle)) return NULL;
    Env *env = (Env *)PyLong_AsVoidPtr(handle);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "invalid env handle");
        return NULL;
    }
    char buf[PR_ANSI_BUF];
    int n = royale_ansi(env, buf, PR_ANSI_BUF);
    return PyUnicode_FromStringAndSize(buf, n);
}

static PyObject *py_env_info(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *handle;
    if (!PyArg_ParseTuple(args, "O", &handle)) return NULL;
    Env *env = (Env *)PyLong_AsVoidPtr(handle);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "invalid env handle");
        return NULL;
    }
    PyObject *d = PyDict_New();
    DSET_INT(d, "tick", env->game.st.tick);
    DSET_INT(d, "learner", env->learner);
    DSET_INT(d, "steps", env->steps);
    dset(d, "hash", PyLong_FromUnsignedLongLong((unsigned long long)pr_hash(&env->game.st)));
    dset(d, "elixir", int_list(env->game.st.elixir, 2));
    dset(d, "crowns", int_list(env->game.st.crowns, 2));
    PyObject *decks = PyList_New(2);
    PyList_SET_ITEM(decks, 0, i8_list(env->game.st.deck[0], 8));
    PyList_SET_ITEM(decks, 1, i8_list(env->game.st.deck[1], 8));
    dset(d, "decks", decks);
    if (PyErr_Occurred()) { Py_DECREF(d); return NULL; }
    return d;
}

static PyObject *py_royale_layout(PyObject *self, PyObject *noargs) {
    (void)self;
    (void)noargs;
    PyObject *d = PyDict_New();
    DSET_INT(d, "OBS_SIZE", PR_OBS_SIZE);
    DSET_INT(d, "SPATIAL_OFFSET", PR_OBS_SPATIAL_OFFSET);
    dset(d, "SPATIAL_SHAPE", Py_BuildValue("(iii)", PR_OBS_C, PR_OBS_H, PR_OBS_W));
    DSET_INT(d, "ENTITY_OFFSET", PR_OBS_ENTITY_OFFSET);
    dset(d, "ENTITY_SHAPE", Py_BuildValue("(ii)", PR_OBS_ENT_N, PR_OBS_ENT_F));
    DSET_INT(d, "SCALAR_OFFSET", PR_OBS_SCALAR_OFFSET);
    DSET_INT(d, "SCALAR_SIZE", PR_OBS_SCALAR_SIZE);
    DSET_INT(d, "MASK_OFFSET", PR_OBS_MASK_OFFSET);
    DSET_INT(d, "MASK_SIZE", PR_OBS_MASK_SIZE);
    PyObject *fields = PyDict_New();
    for (int i = 0; i < PR_OBS_N_FIELDS; i++)
        dset(fields, PR_OBS_FIELDS[i].name, Py_BuildValue("(ii)", PR_OBS_FIELDS[i].offset, PR_OBS_FIELDS[i].len));
    dset(d, "SCALAR_FIELDS", fields);
    PyObject *ch = PyTuple_New(PR_OBS_C);
    for (int c = 0; c < PR_OBS_C; c++) PyTuple_SET_ITEM(ch, c, PyUnicode_FromString(PR_OBS_CHANNELS[c]));
    dset(d, "SPATIAL_CHANNELS", ch);
    PyObject *ef = Py_BuildValue("(sssssssssss)", "card_id", "x", "y", "hp_frac", "hp_2000", "flying", "deploying",
                                 "stunned", "slowed", "is_building", "target_only_buildings");
    dset(d, "ENTITY_FEATURES", ef);
    DSET_INT(d, "CARD_SLOTS", PR_CARD_SLOTS);
    DSET_INT(d, "N_CARDS", PR_N_CARDS);
    PyObject *tts = PyTuple_New(PR_N_TOWER_TROOPS);
    for (int k = 0; k < PR_N_TOWER_TROOPS; k++) PyTuple_SET_ITEM(tts, k, PyUnicode_FromString(PR_TOWER_TROOP_NAMES[k]));
    dset(d, "TOWER_TROOPS", tts);
    DSET_INT(d, "N_ACTIONS", PR_N_ACTIONS);
#ifdef PR_DEBUG
    DSET_INT(d, "DEBUG_BUILD", 1); /* the -O0 sanitizer build (make debug) */
#else
    DSET_INT(d, "DEBUG_BUILD", 0);
#endif
    DSET_INT(d, "BOT_NOOP", PR_BOT_NOOP);
    DSET_INT(d, "BOT_RANDOM", PR_BOT_RANDOM);
    DSET_INT(d, "BOT_HEURISTIC", PR_BOT_HEURISTIC);
#ifdef PR_RAYLIB
    DSET_BOOL(d, "RAYLIB", 1);
#else
    DSET_BOOL(d, "RAYLIB", 0);
#endif
    if (PyErr_Occurred()) { Py_DECREF(d); return NULL; }
    return d;
}


/* ================================================================== Phase D additions */

/* env_log_peek(env_handle) -> dict: ONE native env's raw Log accumulators -- the sums over the
 * episodes it finished since the last vec_log, plus "n" -- read WITHOUT clearing them (vec_log
 * still sees and clears them as before). pufferroyale.league.LeagueVecEnv uses it to attribute
 * each finished match's result to the opponent that played it: vec_log only returns the average
 * over all envs, and the env has already re-dealt by the time Python sees the terminal. */
static PyObject *py_env_log_peek(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *handle;
    if (!PyArg_ParseTuple(args, "O", &handle)) return NULL;
    Env *env = (Env *)PyLong_AsVoidPtr(handle);
    if (!env) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "invalid env handle");
        return NULL;
    }
    PyObject *d = PyDict_New();
    if (!d) return NULL;
    Log copy = env->log;
    my_log(d, &copy);
    assign_to_dict(d, "n", copy.n);
    if (PyErr_Occurred()) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}
