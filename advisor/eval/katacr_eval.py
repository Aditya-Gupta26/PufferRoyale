"""Check 6 (docs/ADVISOR.md): unit-detector accuracy on KataCR's hand-labelled frames.

KataCR frames (wty-yy/Clash-Royale-Detection-Dataset, MIT) are 568x896 crops of the arena from phone recordings,
labelled with Labelme: one box per object, label = name + side digit (0 = bottom / viewer, 1 = top) + optional
state suffix ("king-tower1_freeze"). For every frame:

  * the arena mapping of the episode is fitted from the labelled crown-tower boxes (box bottom-centre <->
    footprint bottom-centre), so positions can be compared in tiles;
  * the detector (Build-A-Bot model, advisor.perception.detector) runs on the frame with that mapping;
  * labelled units and detections are matched one-to-one by IoU (Hungarian, IoU >= 0.3, class-agnostic).

Reported: for labelled units of OUR 64 cards -- recall, unit-type accuracy among matches, side accuracy of the
side classifier among type-correct matches, ground-point error in tiles; for detections -- the share with no
labelled unit (false positives); per-class recall; ms per frame.

    python -m advisor.eval.katacr_eval --data $PR_ROOT/research/ext/crds/images/part2 --out runs/advisor/katacr_eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from advisor.perception.detector import CLASS_TO_UNIT, UnitDetector
from advisor.perception.layout import Layout, fit_arena
from advisor.tracker import Tracker, unit_key

LABEL_RE = re.compile(r"^([a-z\-]+?)(\d)(?:_.*)?$")
NON_UNITS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "dagger-duchess-tower-bar",
             "tower-bar", "king-tower-bar", "bar", "bar-level", "clock", "emote", "text", "elixir", "selected",
             "skeleton-king-bar", "evolution-symbol", "ice-spirit-evolution-symbol", "skeleton-king-skill", "dirt"}
SPELLS = {"zap", "giant-snowball", "barbarian-barrel", "rage", "the-log", "arrows", "earthquake", "goblin-barrel",
          "clone", "tornado", "fireball", "freeze", "poison", "graveyard", "rocket", "lightning", "mirror", "bomb",
          "axe", "goblin-ball", "royal-delivery", "tesla-evolution-shock", "zap-evolution", "skeleton-barrel"}
KATACR_OVERRIDES = {"mini-pekka": "MiniPekka", "baby-dragon": "BabyDragon"}
# own-frame tile of each crown tower's footprint bottom-centre (x centre, bottom edge): engine towers at
# King (9000, 29000) F=4 / (9000, 3000); Princess (3500 | 14500, 25500) F=3 / (.., 6500)
TOWER_REF = {("king-tower", 0): [(9.0, 31.0)], ("king-tower", 1): [(9.0, 5.0)],
             ("queen-tower", 0): [(3.5, 27.0), (14.5, 27.0)], ("queen-tower", 1): [(3.5, 8.0), (14.5, 8.0)]}


def katacr_unit(name: str):
    """KataCR object name -> engine unit name of OUR cards, None for units of other cards."""
    if name in KATACR_OVERRIDES:
        return KATACR_OVERRIDES[name]
    return CLASS_TO_UNIT.get(name.replace("-", "_"))


def parse(json_path: str):
    d = json.load(open(json_path))
    objs = []
    for s in d["shapes"]:
        m = LABEL_RE.match(s["label"])
        if not m or s.get("shape_type", "rectangle") != "rectangle":
            continue
        (x1, y1), (x2, y2) = s["points"]
        objs.append(dict(name=m.group(1), side=int(m.group(2)),
                         box=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))))
    return d["imageWidth"], d["imageHeight"], objs


def fit_episode(files: list):
    """Arena mapping from crown-tower boxes of the episode's frames (all frames pooled)."""
    px, tl = [], []
    for f in files:
        w, _, objs = parse(f)
        for o in objs:
            key = (o["name"], o["side"])
            if key not in TOWER_REF:
                continue
            x1, y1, x2, y2 = o["box"]
            cx = (x1 + x2) / 2
            refs = TOWER_REF[key]
            ref = refs[0] if len(refs) == 1 else refs[0 if cx < w / 2 else 1]   # left / right princess tower
            px.append((cx, y2))
            tl.append(ref)
    return fit_arena(px, tl), len(px)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="KataCR images/part2 directory")
    ap.add_argument("--episodes", nargs="*", default=None, help="episode dirs relative to --data (default: all)")
    ap.add_argument("--every", type=int, default=1, help="use every n-th labelled frame")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    det = UnitDetector(threads=args.threads)
    tf = Tracker(deck=list(range(8)))       # only its tower filter is used
    eps = args.episodes or sorted({os.path.relpath(os.path.dirname(p), args.data)
                                   for p in glob.glob(os.path.join(args.data, "**", "*.json"), recursive=True)})
    S = Counter()
    per_class = defaultdict(Counter)
    pos_err, ms = [], []
    confusion = Counter()
    fp_classes, fp_on = Counter(), Counter()
    fits = {}
    for ep in eps:
        files = sorted(glob.glob(os.path.join(args.data, ep, "*.json")))[::args.every]
        files = [f for f in files if os.path.isfile(f[:-5] + ".jpg")]
        if not files:
            continue
        (x0, sx, y0, sy, rms), n_pts = fit_episode(files)
        fits[ep] = dict(x0=round(x0, 2), sx=round(sx, 3), y0=round(y0, 2), sy=round(sy, 3), rms_px=round(rms, 2),
                        tower_points=n_pts, frames=len(files))
        for f in files:
            w, h, objs = parse(f)
            lay = Layout(name="katacr", width=w, height=h, x0=x0, sx=sx, y0=y0, sy=sy)
            img = Image.open(f[:-5] + ".jpg").convert("RGB")
            t0 = time.perf_counter()
            boxes = det.detect_boxes(img, lay)
            ms.append((time.perf_counter() - t0) * 1000)
            gts = [o for o in objs if o["name"] not in NON_UNITS and o["name"] not in SPELLS]
            S["frames"] += 1
            S["gt_all_units"] += len(gts)
            S["pred"] += len(boxes)
            if gts and boxes:
                cost = np.array([[1.0 - iou(g["box"], b[:4]) for b in boxes] for g in gts])
                ri, ci = linear_sum_assignment(cost)
                pairs = [(i, j) for i, j in zip(ri, ci) if 1.0 - cost[i, j] >= args.iou]
            else:
                pairs = []
            matched_g = {i for i, _ in pairs}
            matched_p = {j for _, j in pairs}
            S["pred_unmatched"] += len(boxes) - len(matched_p)
            # the tracker's tower filter (advisor.tracker: ground point inside a standing tower's footprint,
            # shrunk by tower_margin; Balloons kept), with the towers standing in this frame per the labels
            standing = {(o["name"], o["side"], (o["box"][0] + o["box"][2]) / 2 < w / 2) for o in objs
                        if o["name"] in ("queen-tower", "king-tower")}
            tf.own_tower_frac = [float(("king-tower", 0, True) in standing or ("king-tower", 0, False) in standing),
                                 float(("queen-tower", 0, True) in standing), float(("queen-tower", 0, False) in standing)]
            tf.enemy_tower_frac = [float(("king-tower", 1, True) in standing or ("king-tower", 1, False) in standing),
                                   float(("queen-tower", 1, True) in standing), float(("queen-tower", 1, False) in standing)]
            for j, b in enumerate(boxes):
                gx, gy = lay.px_to_milli((b[0] + b[2]) / 2, b[3])
                u = CLASS_TO_UNIT.get(b[5])
                dropped = tf._in_standing_tower(unit_key(u) if u else "label:" + b[5], gx, gy)
                S["tf_true" if j in matched_p else "tf_unmatched"] += 1
                S["tf_true_dropped" if j in matched_p else "tf_unmatched_dropped"] += dropped
            others = [o for o in objs if o["name"] in NON_UNITS or o["name"] in SPELLS]
            for j, b in enumerate(boxes):
                if j in matched_p:
                    continue
                hit = max(others, key=lambda o: iou(o["box"], b[:4]), default=None)
                what = hit["name"] if hit is not None and iou(hit["box"], b[:4]) >= args.iou else "nothing labelled"
                fp_classes[b[5]] += 1
                fp_on[what] += 1
            for i, g in enumerate(gts):
                u = katacr_unit(g["name"])
                if u is None:
                    continue
                S["gt_ours"] += 1
                per_class[u]["gt"] += 1
                if i not in matched_g:
                    continue
                j = next(j for ii, j in pairs if ii == i)
                S["found"] += 1
                per_class[u]["found"] += 1
                pu = CLASS_TO_UNIT.get(boxes[j][5])
                confusion[(u, pu or boxes[j][5])] += 1
                if pu is not None and unit_key(pu) == unit_key(u):
                    S["type_ok"] += 1
                    per_class[u]["type_ok"] += 1
                    own = det.side_of(img, boxes[j])
                    S["side_ok"] += own == (g["side"] == 0)
                    gx, gy = lay.px_to_milli((g["box"][0] + g["box"][2]) / 2, g["box"][3])
                    bx, by = lay.px_to_milli((boxes[j][0] + boxes[j][2]) / 2, boxes[j][3])
                    pos_err.append(np.hypot(gx - bx, gy - by) / 1000.0)
    out = dict(
        episodes=fits, frames=S["frames"], labelled_units_all=S["gt_all_units"], labelled_units_ours=S["gt_ours"],
        recall_ours=round(S["found"] / max(1, S["gt_ours"]), 4),
        type_accuracy_of_matched=round(S["type_ok"] / max(1, S["found"]), 4),
        side_accuracy_of_type_ok=round(S["side_ok"] / max(1, S["type_ok"]), 4),
        recall_ours_type_correct=round(S["type_ok"] / max(1, S["gt_ours"]), 4),
        pos_err_tiles_median=round(float(np.median(pos_err)), 3) if pos_err else None,
        pos_err_tiles_p90=round(float(np.percentile(pos_err, 90)), 3) if pos_err else None,
        detections=S["pred"], detections_without_label=S["pred_unmatched"],
        false_positive_share=round(S["pred_unmatched"] / max(1, S["pred"]), 4),
        unmatched_detections_by_class=dict(fp_classes.most_common(15)),
        unmatched_detections_overlap=dict(fp_on.most_common(15)),
        tower_filter=dict(unmatched_dropped=S["tf_unmatched_dropped"], unmatched=S["tf_unmatched"],
                          matched_dropped=S["tf_true_dropped"], matched=S["tf_true"],
                          unmatched_after=S["tf_unmatched"] - S["tf_unmatched_dropped"],
                          share_unmatched_after=round((S["tf_unmatched"] - S["tf_unmatched_dropped"]) /
                                                      max(1, S["pred"] - S["tf_unmatched_dropped"] - S["tf_true_dropped"]), 4),
                          unmatched_after_per_frame=round((S["tf_unmatched"] - S["tf_unmatched_dropped"]) / max(1, S["frames"]), 3)),
        ms_per_frame_median=round(float(np.median(ms)), 1) if ms else None,
        per_class={u: dict(gt=c["gt"], recall=round(c["found"] / c["gt"], 3), type_ok=round(c["type_ok"] / c["gt"], 3))
                   for u, c in sorted(per_class.items(), key=lambda kv: -kv[1]["gt"])},
        top_confusions=[(a, b, n) for (a, b), n in confusion.most_common(40) if unit_key(a) != unit_key(CLASS_TO_UNIT.get(b, b) or b)][:20],
    )
    print(json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
