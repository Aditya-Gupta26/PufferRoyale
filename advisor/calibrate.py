"""Calibrate the screen layout for a device whose game frame is not 720x1280 (9:16).

Take one screenshot of a battle with all four princess towers standing, then either click in a window:

    python -m advisor.calibrate --image shot.png --out mydevice.json

or give the four princess-tower centres directly (pixels; enemy-left, enemy-right, own-left, own-right):

    python -m advisor.calibrate --image shot.png --out mydevice.json --points "112,195;455,195;112,680;455,680"

"Centre" = the middle of the tower's square base. The arena mapping is fitted to these four points (each tower
footprint is centred on tile (3.5 | 14.5, 6.5 | 25.5)); the princess HP bars are placed relative to the arena
(they are drawn above the towers); the hand / next card / elixir bar are placed by scaling Build-A-Bot's
720x1280 layout to the frame width, anchored to the bottom edge, unless you drag them yourself (--drag-hud).
A check image with every region drawn is written next to the JSON: look at it before playing.
"""
from __future__ import annotations

import argparse
import sys

from PIL import Image, ImageDraw

from .perception.layout import Layout, bluestacks_720x1280, fit_arena

TOWER_TILES = [(3.5, 6.5), (14.5, 6.5), (3.5, 25.5), (14.5, 25.5)]
TOWER_NAMES = ["enemy LEFT princess tower", "enemy RIGHT princess tower", "your LEFT princess tower",
               "your RIGHT princess tower"]


def _bottom_anchored(rect, ref: Layout, w: int, h: int):
    k = w / ref.width
    l, t, r, b = rect
    return (l * k, h - (ref.height - t) * k, r * k, h - (ref.height - b) * k)


def _arena_relative(rect, ref: Layout, new: Layout):
    l, t, r, b = rect
    (tl, tt), (tr, tb) = ((l - ref.x0) / ref.sx, (t - ref.y0) / ref.sy), ((r - ref.x0) / ref.sx, (b - ref.y0) / ref.sy)
    (nl, nt), (nr, nb) = new.tile_to_px(tl, tt), new.tile_to_px(tr, tb)
    return (nl, nt, nr, nb)


def build_layout(name: str, w: int, h: int, points: list, hud: dict = None) -> tuple:
    x0, sx, y0, sy, rms = fit_arena(points, TOWER_TILES)
    ref = bluestacks_720x1280()
    lay = Layout(name=name, width=w, height=h, x0=x0, sx=sx, y0=y0, sy=sy)
    lay.princess_bars = {n: _arena_relative(r, ref, lay) for n, r in ref.princess_bars.items()}
    hud = hud or {}
    lay.hand = hud.get("hand") or [_bottom_anchored(r, ref, w, h) for r in ref.hand]
    lay.next_card = hud.get("next_card") or _bottom_anchored(ref.next_card, ref, w, h)
    lay.elixir_bar = hud.get("elixir_bar") or _bottom_anchored(ref.elixir_bar, ref, w, h)
    lay.in_battle_patch = _bottom_anchored(ref.in_battle_patch, ref, w, h)
    return lay, rms


def draw_check(img: Image.Image, lay: Layout) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for tx in range(19):
        d.line([lay.tile_to_px(tx, 0), lay.tile_to_px(tx, 32)], fill=(255, 255, 255), width=1)
    for ty in range(33):
        d.line([lay.tile_to_px(0, ty), lay.tile_to_px(18, ty)], fill=(255, 255, 255), width=1)
    d.rectangle([lay.tile_to_px(0, 15), lay.tile_to_px(18, 17)], outline=(0, 160, 255), width=3)   # the river
    for i, r in enumerate(lay.hand):
        d.rectangle(r, outline=(0, 255, 0), width=3)
        d.text((r[0] + 3, r[1] + 3), f"slot {i}", fill=(0, 255, 0))
    d.rectangle(lay.next_card, outline=(0, 255, 255), width=3)
    d.rectangle(lay.elixir_bar, outline=(255, 0, 255), width=3)
    for n, r in lay.princess_bars.items():
        d.rectangle(r, outline=(255, 255, 0), width=2)
    d.rectangle(lay.detector_crop_px(), outline=(255, 128, 0), width=2)
    return out


def _click_points(img: Image.Image) -> list:
    import cv2
    import numpy as np
    pts = []
    win = "advisor calibration"
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while len(pts) < 4:
        view = bgr.copy()
        for p in pts:
            cv2.circle(view, p, 6, (0, 255, 0), -1)
        cv2.putText(view, f"click the centre of the {TOWER_NAMES[len(pts)]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        cv2.imshow(win, view)
        if cv2.waitKey(20) & 0xFF == 27:
            raise SystemExit("cancelled")
    cv2.destroyWindow(win)
    return pts


def _drag_hud(img: Image.Image) -> dict:
    import cv2
    import numpy as np
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = {"hand": []}
    for i in range(4):
        x, y, w, h = cv2.selectROI(f"drag hand card slot {i} (left to right), then ENTER", bgr, False)
        out["hand"].append((x, y, x + w, y + h))
    x, y, w, h = cv2.selectROI("drag the NEXT card (small, left), then ENTER", bgr, False)
    out["next_card"] = (x, y, x + w, y + h)
    x, y, w, h = cv2.selectROI("drag the elixir bar (the purple bar only), then ENTER", bgr, False)
    out["elixir_bar"] = (x, y, x + w, y + h)
    cv2.destroyAllWindows()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", help="x,y;x,y;x,y;x,y (enemy-left, enemy-right, own-left, own-right)")
    ap.add_argument("--drag-hud", action="store_true", help="drag the hand / next / elixir boxes yourself")
    args = ap.parse_args(argv)
    img = Image.open(args.image)
    w, h = img.size
    if args.points:
        pts = [tuple(float(v) for v in p.split(",")) for p in args.points.split(";")]
        if len(pts) != 4:
            raise SystemExit("--points needs 4 points")
    else:
        pts = _click_points(img)
    hud = _drag_hud(img) if args.drag_hud else None
    lay, rms = build_layout(f"calibrated_{w}x{h}", w, h, pts, hud)
    lay.save(args.out)
    check = args.out.rsplit(".", 1)[0] + "_check.png"
    draw_check(img, lay).save(check)
    print(f"layout {args.out}: x0={lay.x0:.1f} sx={lay.sx:.2f} y0={lay.y0:.1f} sy={lay.sy:.2f} "
          f"(tile {lay.sx:.1f} x {lay.sy:.1f} px, ratio {lay.sx / lay.sy:.3f}; 720x1280 reference 1.232); "
          f"fit rms {rms:.1f} px. Check image: {check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
