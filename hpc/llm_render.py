#!/usr/bin/env python3
"""Animated GIF of one recorded Claude-vs-policy game (DECISIONS D101; generalised from the route-1 renderer):
the arena with towers, units, projectiles, spell areas and labelled plays, a header with the clock, crowns and
elixir, and a footer with Claude's latest decision. Claude's side is always drawn at the bottom. GIF because W&B
videos take gif / mp4 / webm and the image has no ffmpeg; a fixed palette keeps it small.

    python hpc/llm_render.py GAME_DIR OUT_GIF [--every 4] [--title TEXT]

GAME_DIR is what hpc/llm_game.py writes: meta.json, frames.jsonl.gz, plays.jsonl, decisions.jsonl.
"""
import argparse
import gzip
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

TILE = 16
W, ARENA_H = 18 * TILE, 32 * TILE
HEAD, FOOT = 60, 84
H = HEAD + ARENA_H + FOOT
CLAUDE, POLICY = (42, 98, 201), (194, 51, 74)
TURF_A, TURF_B, WATER, BRIDGE = (169, 201, 154), (162, 194, 147), (121, 174, 205), (168, 134, 94)
BG, FG, MUTED, DEAD, ICE, FIRE = (238, 240, 236), (29, 36, 32), (93, 106, 98), (138, 143, 140), (127, 216, 255), \
    (240, 122, 26)
WHITE = (255, 255, 255)
PALETTE = [BG, FG, MUTED, DEAD, CLAUDE, POLICY, TURF_A, TURF_B, WATER, BRIDGE, ICE, FIRE, WHITE,
           tuple((c + 255) // 2 for c in CLAUDE), tuple((c + 255) // 2 for c in POLICY), (214, 220, 211)]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


F_SM, F_MD, F_LG = _font(10), _font(12), _font(20)


def clock(t):
    if t < 3600:
        s = -(-(3600 - t) // 20)
        return f"{s // 60}:{s % 60:02d}"
    s = max(0, -(-(6000 - t) // 20))
    return f"OT {s // 60}:{s % 60:02d}"


def label(unit):
    """Two-letter label of a unit name: its capitals (HogRider -> HR, Xbow -> X), else its first letters."""
    caps = "".join(c for c in unit if c.isupper())
    return (caps or unit[:2])[:2]


def _palette_image():
    pal = Image.new("P", (1, 1))
    flat = [v for c in PALETTE for v in c]
    pal.putpalette(flat + [0] * (768 - len(flat)))
    return pal


def render(game_dir, out, every=4, title=""):
    """Write the GIF; returns the number of frames."""
    from pufferroyale.game import card_info               # building cards are drawn as squares
    meta = json.load(open(os.path.join(game_dir, "meta.json")))
    me = int(meta["claude_team"])
    flip = me == 1                                        # Claude at the bottom (team 0's frame is drawn as is)

    def xy(x, y):                                         # engine milli-tiles (global frame) -> pixels
        if flip:
            x, y = 18000 - x, 32000 - y
        return x / 1000 * TILE, HEAD + y / 1000 * TILE

    def gtile(gx, gy):                                    # a global tile -> its pixel centre
        if flip:
            gx, gy = 17 - gx, 31 - gy
        return (gx + .5) * TILE, HEAD + (gy + .5) * TILE

    plays = [json.loads(ln) for ln in open(os.path.join(game_dir, "plays.jsonl"))]
    decs = [json.loads(ln) for ln in open(os.path.join(game_dir, "decisions.jsonl"))]
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    for y in range(32):
        for x in range(18):
            d.rectangle([x * TILE, HEAD + y * TILE, x * TILE + TILE, HEAD + y * TILE + TILE],
                        fill=TURF_A if (x + y) % 2 else TURF_B)
    d.rectangle([0, HEAD + 15 * TILE, W, HEAD + 17 * TILE], fill=WATER)
    for bx in (3, 14):
        d.rectangle([bx * TILE, HEAD + 15 * TILE - 3, bx * TILE + TILE, HEAD + 17 * TILE + 3], fill=BRIDGE)
    names = (f"Claude ({meta['claude_deck_name']})", f"Policy ({meta['policy_deck_name']})")
    building = {}
    pal = _palette_image()
    frames, di = [], -1
    with gzip.open(os.path.join(game_dir, "frames.jsonl.gz"), "rt") as fh:
        raw = [json.loads(ln) for ln in fh]
    keep = [f for i, f in enumerate(raw) if f["t"] % every == 0 or i == len(raw) - 1]
    for f in keep:
        t = f["t"]
        im = base.copy()
        d = ImageDraw.Draw(im)
        d.text((W / 2, 6), clock(t), fill=FG, font=F_LG, anchor="mt")
        d.text((W / 2, 30), title or f"tick {t}", fill=MUTED, font=F_SM, anchor="mt")
        cl, po = f["cr"][me], f["cr"][1 - me]
        d.text((6, 4), names[0], fill=CLAUDE, font=F_MD)
        d.text((6, 20), f"crowns {cl}", fill=FG, font=F_MD)
        d.text((W - 6, 4), names[1], fill=POLICY, font=F_MD, anchor="rt")
        d.text((W - 6, 20), f"crowns {po}", fill=FG, font=F_MD, anchor="rt")
        for x0, col, el in ((6, CLAUDE, f["el"][me]), (W - 106, POLICY, f["el"][1 - me])):
            d.rectangle([x0, 42, x0 + 100, 50], fill=PALETTE[-1])
            d.rectangle([x0, 42, x0 + 10 * el, 50], fill=col)
        for team, _unit, x, y, hp, mx in f["tw"]:
            half = (2.0 if mx > 4000 else 1.5) * TILE
            cx, cy = xy(x, y)
            col = (CLAUDE if team == me else POLICY) if hp > 0 else DEAD
            d.rounded_rectangle([cx - half + 2, cy - half + 2, cx + half - 2, cy + half - 2], radius=4, fill=col)
            if hp > 0:
                d.text((cx, cy), str(hp), fill=WHITE, font=F_SM, anchor="mm")
        for fx in f["f"]:                                 # spell areas
            cx, cy = xy(fx[2], fx[3])
            r = max(.5, fx[4] / 1000) * TILE
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ICE, width=2)
        for e in sorted(f["e"], key=lambda e: -e[11]):    # units, largest first
            _id, team, unit, card, x, y, hp, mx, fly, dep, stun, rad = e
            cx, cy = xy(x, y)
            r = max(.3, rad / 1000) * TILE
            col = CLAUDE if team == me else POLICY
            if dep:
                col = PALETTE[13] if team == me else PALETTE[14]
            if card not in building:
                building[card] = card >= 0 and card_info(card)["kind"] == "building"
            box = [cx - r, cy - r, cx + r, cy + r]
            if building[card]:
                d.rounded_rectangle(box, radius=3, fill=col, outline=BG)
            else:
                d.ellipse(box, fill=col, outline=BG)
            if stun:
                d.ellipse([cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], outline=ICE, width=2)
            if r > 6:
                d.text((cx, cy), label(unit), fill=WHITE, font=F_SM, anchor="mm")
            if 0 < hp < (mx or hp):
                bw = max(12, 2 * r)
                d.rectangle([cx - bw / 2, cy - r - 4, cx + bw / 2, cy - r - 2], fill=BG)
                d.rectangle([cx - bw / 2, cy - r - 4, cx - bw / 2 + bw * hp / mx, cy - r - 2], fill=col)
        for p in f["p"]:                                  # projectiles
            cx, cy = xy(p[2], p[3])
            big = "Spell" in p[1]
            r = 7 if big else 3
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FIRE if big else (CLAUDE if p[0] == me else POLICY))
        for pl in plays:                                  # the last 2 s of plays, labelled
            if pl["err"] or pl["t"] > t or t - pl["t"] > 40:
                continue
            col = CLAUDE if pl["team"] == me else POLICY
            cx, cy = gtile(pl["gx"], pl["gy"])
            r = TILE * (.6 + (t - pl["t"]) / 40 * 1.4)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)
            tw = d.textlength(pl["card"], font=F_MD) + 8
            lx, ly = min(max(cx, tw / 2 + 2), W - tw / 2 - 2), max(HEAD + 12, cy - TILE * 1.2)
            d.rounded_rectangle([lx - tw / 2, ly - 8, lx + tw / 2, ly + 8], radius=4, fill=col)
            d.text((lx, ly), pl["card"], fill=WHITE, font=F_MD, anchor="mm")
        while di + 1 < len(decs) and decs[di + 1]["t"] <= t:
            di += 1
        if di >= 0:                                       # footer: Claude's latest decision
            dec = decs[di]
            d.text((6, HEAD + ARENA_H + 5), f"#{dec['n']} {clock(dec['t'])}  {dec.get('decoded') or 'WAIT'}",
                   fill=CLAUDE if dec["action"] else FG, font=F_MD)
            why = " ".join(ln.strip() for ln in (dec.get("reply") or "").strip().splitlines()[:-1]).strip()
            for k, ln in enumerate(textwrap.wrap(why, 52)[:4]):
                d.text((6, HEAD + ARENA_H + 22 + k * 14), ln, fill=MUTED, font=F_SM)
        frames.append(im.quantize(palette=pal, dither=Image.Dither.NONE))
    frames += [frames[-1]] * 15                           # hold the end ~3 s
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(50 * every), loop=0, optimize=False)
    return len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game_dir")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=4, help="ticks per frame (default 4 = 5 fps, real time)")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    n = render(a.game_dir, a.out, a.every, a.title)
    print(a.out, n, "frames", os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
