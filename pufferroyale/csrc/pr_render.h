/*
 * pr_render.h -- human-facing renderers (SPEC §9: render_mode='ansi', optional raylib).
 * Rendering only reads the state (floats allowed here, SPEC §1).
 *
 * ANSI board: engine frame (team 0 at the bottom), one character per tile, UPPER case =
 * team 0, lower case = team 1. Units: N Knight, A Archer, M Musketeer, G Giant, H Hog Rider,
 * I Minion, D Baby Dragon, V Valkyrie, S Skeleton, O Ice Golem, E Ice Spirit, R Prince,
 * Z Wizard, C Cannon, T Tesla, B Goblin; towers K (King) and P (Princess / tower troop); the
 * §16 units' letters are in tools/gen_card_db.py GLYPHS ('$' Elixir Collector); ~ water,
 * = bridge, : no-deploy, . land; * marks a spell area / flying spell target.
 */
#ifndef PR_RENDER_H
#define PR_RENDER_H

#include <stdio.h>

#include "pr_engine.h"

static inline char pr_unit_glyph(int unit) {
    return (unit >= 0 && unit < PR_N_UNITS) ? PR_UNITS[unit].glyph : '?'; /* generated (tools/gen_card_db.py) */
}

#define PR_ANSI_BUF 8192

/* Render the board and a status header into buf (NUL-terminated). Returns the length. */
static inline int pr_render_ansi(const PrState *st, char *buf, int cap) {
    char grid[PR_TILES_Y][PR_TILES_X + 1];
    for (int ty = 0; ty < PR_TILES_Y; ty++) {
        for (int tx = 0; tx < PR_TILES_X; tx++) {
            uint8_t f = PR_TILE_FLAGS[ty][tx];
            char c = '.';
            if (f & PR_TF_WATER) c = '~';
            else if (ty == 15 || ty == 16) c = '=';
            else if (f & PR_TF_NODEPLOY) c = ':';
            grid[ty][tx] = c;
        }
        grid[ty][PR_TILES_X] = 0;
    }
    for (int i = 0; i < st->n_fx; i++) { /* spell areas */
        const PrEffect *fx = &st->fx[i];
        int tx = PR_CLAMP(fx->x / 1000, 0, PR_TILES_X - 1), ty = PR_CLAMP(fx->y / 1000, 0, PR_TILES_Y - 1);
        grid[ty][tx] = '*';
    }
    for (int i = 0; i < st->n_proj; i++) {
        const PrProjectile *p = &st->proj[i];
        if (p->has_target) continue;
        int tx = PR_CLAMP(p->tx / 1000, 0, PR_TILES_X - 1), ty = PR_CLAMP(p->ty / 1000, 0, PR_TILES_Y - 1);
        grid[ty][tx] = '*';
    }
    for (int i = 0; i < st->n_ent; i++) { /* structures: footprint; troops: centre tile */
        const PrEntity *e = &st->ent[i];
        if (e->hp <= 0) continue;
        char g = pr_unit_glyph(e->unit);
        if (e->team == 1 && g >= 'A' && g <= 'Z') g = (char)(g - 'A' + 'a');
        if (pr_is_structure(e)) {
            int h = pr_udef(e)->footprint_tiles * 500;
            for (int ty = PR_MAX(0, (e->y - h) / 1000); ty <= PR_MIN(PR_TILES_Y - 1, (e->y + h - 1) / 1000); ty++)
                for (int tx = PR_MAX(0, (e->x - h) / 1000); tx <= PR_MIN(PR_TILES_X - 1, (e->x + h - 1) / 1000); tx++)
                    grid[ty][tx] = g;
        }
    }
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        if (e->hp <= 0 || pr_is_structure(e)) continue;
        char g = pr_unit_glyph(e->unit);
        if (e->team == 1 && g >= 'A' && g <= 'Z') g = (char)(g - 'A' + 'a');
        grid[PR_CLAMP(e->y / 1000, 0, PR_TILES_Y - 1)][PR_CLAMP(e->x / 1000, 0, PR_TILES_X - 1)] = g;
    }
    int n = 0;
#define PR_EMIT(...) do { if (n < cap) n += snprintf(buf + n, (size_t)(cap - n), __VA_ARGS__); } while (0)
    PR_EMIT("tick %d/%d%s  elixir %.1f | %.1f  crowns %d-%d%s\n", st->tick, PR_TICKS_MAX,
            st->tick >= PR_TICKS_REGULATION ? " (OT)" : "", st->elixir[0] / 2800.0, st->elixir[1] / 2800.0,
            st->crowns[0], st->crowns[1], st->over ? "  [OVER]" : "");
    for (int team = 1; team >= 0; team--) {
        PR_EMIT("team %d towers K/L/R:", team);
        for (int idx = 0; idx < 3; idx++) {
            const PrEntity *t = pr_get_c(st, st->tower_id[team][idx]);
            PR_EMIT(" %4d", t ? t->hp : 0);
        }
        PR_EMIT("   hand:");
        for (int s = 0; s < 4; s++) PR_EMIT(" %s%s", PR_CARDS[(int)st->hand[team][s]].name, s < 3 ? "," : "");
        PR_EMIT("\n");
    }
    for (int ty = 0; ty < PR_TILES_Y; ty++) PR_EMIT("  %2d %s\n", ty, grid[ty]);
#undef PR_EMIT
    if (n >= cap) n = cap - 1;
    return n;
}

#ifdef PR_RAYLIB
#include <stdlib.h>
#include "raylib.h"

#define PR_RL_SCALE 24 /* pixels per tile */

#ifdef __APPLE__
/* CoreGraphics, declared here to keep MacTypes out of raylib's namespace */
extern int32_t CGGetActiveDisplayList(uint32_t max_displays, uint32_t *displays, uint32_t *count);
#endif

/* Is there a display to open a window on? raylib/GLFW crash inside InitWindow without one
 * (e.g. an SSH or agent session on macOS), so check first and fall back to text. */
static inline int pr_display_available(void) {
#ifdef __APPLE__
    uint32_t n = 0;
    return CGGetActiveDisplayList(0, NULL, &n) == 0 && n > 0;
#else
    return getenv("DISPLAY") != NULL || getenv("WAYLAND_DISPLAY") != NULL;
#endif
}

static inline Color pr_rl_team(int team, unsigned char a) {
    return team == 0 ? (Color){60, 110, 230, a} : (Color){220, 60, 60, a};
}

/* Draw one frame (opens the window on first use). Returns 1 when a frame was drawn, 0 when
 * the window was closed, and -1 when no window can be opened (no display): the caller then
 * falls back to the text board. *opened: 0 = not tried, 1 = open, -1 = unavailable. */
static inline int pr_raylib_draw(const PrState *st, int *opened) {
    const int W = PR_TILES_X * PR_RL_SCALE, H = PR_TILES_Y * PR_RL_SCALE, PANEL = 90;
    if (*opened < 0) return -1;
    if (*opened == 0) {
        if (!pr_display_available()) {
            *opened = -1;
            return -1;
        }
        SetTraceLogLevel(LOG_WARNING);
        InitWindow(W, H + PANEL, "PufferRoyale");
        if (!IsWindowReady()) {
            *opened = -1;
            return -1;
        }
        SetTargetFPS(60);
        *opened = 1;
    }
    if (WindowShouldClose()) return 0;
    const float s = PR_RL_SCALE / 1000.0f;
    BeginDrawing();
    ClearBackground((Color){34, 40, 49, 255});
    for (int ty = 0; ty < PR_TILES_Y; ty++) {
        for (int tx = 0; tx < PR_TILES_X; tx++) {
            uint8_t f = PR_TILE_FLAGS[ty][tx];
            Color c = ((tx + ty) & 1) ? (Color){96, 150, 72, 255} : (Color){104, 160, 78, 255};
            if (f & PR_TF_WATER) c = (Color){60, 130, 200, 255};
            else if (ty == 15 || ty == 16) c = (Color){150, 110, 70, 255};
            DrawRectangle(tx * PR_RL_SCALE, ty * PR_RL_SCALE, PR_RL_SCALE, PR_RL_SCALE, c);
        }
    }
    for (int i = 0; i < st->n_fx; i++) {
        const PrEffect *fx = &st->fx[i];
        Color c = pr_rl_team(fx->team, 70);
        if (fx->type == PR_FX_ROLLING)
            DrawRectangle((int)((fx->x - fx->radius) * s), (int)((fx->y - fx->radius_y) * s),
                          (int)(2 * fx->radius * s), (int)(2 * fx->radius_y * s), c);
        else
            DrawCircle((int)(fx->x * s), (int)(fx->y * s), fx->radius * s, c);
    }
    for (int i = 0; i < st->n_ent; i++) {
        const PrEntity *e = &st->ent[i];
        if (e->hp <= 0) continue;
        float x = e->x * s, y = e->y * s, r = e->radius * s;
        if (pr_is_structure(e)) {
            int h = pr_udef(e)->footprint_tiles * 500;
            unsigned char a = pr_is_hidden(e) ? 90 : 230;
            DrawRectangle((int)((e->x - h) * s) + 2, (int)((e->y - h) * s) + 2, (int)(2 * h * s) - 4,
                          (int)(2 * h * s) - 4, pr_rl_team(e->team, a));
        } else {
            DrawCircle((int)x, (int)y, r, pr_rl_team(e->team, pr_is_deploying(e) ? 120 : 255));
            if (e->flying) DrawCircleLines((int)x, (int)y, r + 2, RAYWHITE);
            if (pr_is_stunned(e)) DrawCircleLines((int)x, (int)y, r + 4, SKYBLUE);
        }
        char g[2] = {pr_unit_glyph(e->unit), 0};
        DrawText(g, (int)x - 4, (int)y - 6, 12, RAYWHITE);
        float frac = (float)e->hp / (float)e->max_hp;
        DrawRectangle((int)(x - 10), (int)(y - r - 6), 20, 3, (Color){30, 30, 30, 255});
        DrawRectangle((int)(x - 10), (int)(y - r - 6), (int)(20 * frac), 3, (Color){80, 230, 90, 255});
    }
    for (int i = 0; i < st->n_proj; i++) {
        const PrProjectile *p = &st->proj[i];
        DrawCircle((int)(p->x * s), (int)(p->y * s), 3, pr_rl_team(p->team, 255));
    }
    DrawRectangle(0, H, W, PANEL, (Color){20, 22, 28, 255});
    DrawText(TextFormat("tick %d   crowns %d-%d%s", st->tick, st->crowns[0], st->crowns[1], st->over ? "  OVER" : ""),
             8, H + 6, 16, RAYWHITE);
    for (int team = 0; team < 2; team++) {
        DrawText(TextFormat("T%d %.1f  %s, %s, %s, %s", team, st->elixir[team] / 2800.0,
                            PR_CARDS[(int)st->hand[team][0]].name, PR_CARDS[(int)st->hand[team][1]].name,
                            PR_CARDS[(int)st->hand[team][2]].name, PR_CARDS[(int)st->hand[team][3]].name),
                 8, H + 30 + 24 * (1 - team), 12, pr_rl_team(team, 255));
    }
    EndDrawing();
    return 1;
}

static inline void pr_raylib_close(int *opened) {
    if (*opened > 0) CloseWindow();
    *opened = 0;
}
#endif /* PR_RAYLIB */

#endif /* PR_RENDER_H */
