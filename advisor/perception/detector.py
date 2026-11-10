"""Unit detector: Build-A-Bot's YOLOv10m ONNX model (97 classes) plus its ally / enemy classifier, with the
pre- and post-processing of Build-A-Bot's UnitDetector / OnnxDetector / SideDetector (MIT), except that the
image region fed to the model is defined in arena tiles (Layout.detector_crop_tiles), so any calibrated
screen gives the model the view it was trained on.

Model facts (read from the file, docs/ADVISOR.md §6): input `images` 1x3x480x352 float16 (H=480, W=352),
output `output0` 1x300x6 = (x1, y1, x2, y2, confidence, class) in input pixels, NMS-free (YOLOv10).
Metadata licence: AGPL-3.0 (Ultralytics) -- see docs/ADVISOR.md §5.
"""
from __future__ import annotations

import ast
import os

import numpy as np
from PIL import Image

from .. import engine_db as D
from ..reading import Detection
from .layout import Layout

# detector class -> engine unit name; classes of units outside our 64 cards map to None
CLASS_TO_UNIT = {
    "archer": "Archer", "balloon": "Balloon", "bandit": "Assassin", "barbarian": "Barbarian", "bat": "Bat",
    "battle_ram": "BattleRam", "bomb_tower": "BombTower", "bomber": "Bomber", "cannon": "Cannon",
    "dart_goblin": "BlowdartGoblin", "elite_barbarian": "AngryBarbarian", "elixir_collector": "ElixirCollector",
    "fire_spirit": "FireSpirits", "giant": "Giant", "giant_skeleton": "GiantSkeleton", "goblin": "Goblin_Stab",
    "golem": "Golem", "golemite": "Golemite", "hog_rider": "HogRider",
    "hungry_dragon": "BabyDragon",   # Build-A-Bot's DETECTOR_UNITS[46] = Units.BABY_DRAGON
    "ice_golem": "IceGolemite", "ice_spirit": "IceSpirits", "ice_wizard": "IceWizard",
    "inferno_dragon": "InfernoDragon", "inferno_tower": "InfernoTower", "knight": "Knight",
    "lava_hound": "LavaHound", "lava_pup": "LavaPups", "mega_minion": "MegaMinion", "miner": "Miner",
    "minion": "Minion", "minipekka": "MiniPekka", "mortar": "Mortar", "musketeer": "Musketeer",
    "night_witch": "DarkWitch", "pekka": "Pekka", "prince": "Prince", "princess": "Princess",
    "rascal_boy": "RascalBoy", "rascal_girl": "RascalGirl", "royal_giant": "RoyalGiant", "royal_hog": "RoyalHog",
    "skeleton": "Skeleton", "skeleton_dragon": "SkeletonDragon", "spear_goblin": "SpearGoblin", "tesla": "Tesla",
    "tombstone": "Tombstone", "valkyrie": "Valkyrie", "wall_breaker": "Wallbreaker", "witch": "Witch",
    "wizard": "Wizard", "x_bow": "Xbow",
}
NOT_A_UNIT = {"giant_snowball"}      # a spell projectile in flight, not a board unit
MIN_CONF = 0.3                       # Build-A-Bot UnitDetector.MIN_CONF
SIDE_SIZE = 16                       # Build-A-Bot SideDetector.SIDE_SIZE


def default_assets() -> str:
    return os.environ.get("ADVISOR_ASSETS", os.path.expandvars("/scratch/$USER/pufferroyale/research/ext/buildabot/clashroyalebuildabot"))


def own_deck_units(deck: list) -> set:
    """Engine units the viewer's deck can put on the board (plays, spawner and death-spawn products)."""
    out = set()
    for c in deck:
        for u in D.play_compositions()[c]:
            out.add(u)
            ud = D.unit(u)
            if ud.spawn_unit:
                out.add(ud.spawn_unit)
            if ud.dspawn_unit:
                out.add(ud.dspawn_unit)
    return {"Goblin_Stab" if u == "Goblin" else u for u in out}


class UnitDetector:
    def __init__(self, assets: str = None, threads: int = 2):
        import onnxruntime as ort
        assets = assets or default_assets()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads    # also avoids onnxruntime's thread-affinity errors on shared nodes
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(assets, "models", "units_M_480x352.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.side = ort.InferenceSession(os.path.join(assets, "models", "side.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"])
        self.class_names = [names[i] for i in range(len(names))]
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_dtype = np.float16 if "float16" in inp.type else np.float32
        self.model_h, self.model_w = inp.shape[2:]
        self.side_input = self.side.get_inputs()[0].name

    # Build-A-Bot OnnxDetector.resize / pad / resize_pad_transpose_and_scale
    def _prepare(self, img: Image.Image):
        ratio = img.height / img.width
        if ratio > self.model_h / self.model_w:
            h, w = self.model_h, int(self.model_h / ratio)
        else:
            w, h = self.model_w, int(self.model_w * ratio)
        x = np.array(img.resize((w, h)), dtype=np.float16)
        dx, dy = self.model_w - w, self.model_h - h
        pad_right, pad_bottom = dx // 2, dy // 2
        pad_left, pad_top = dx - pad_right, dy - pad_bottom
        x = np.pad(x, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=114)
        x = x.transpose(2, 0, 1) / 255
        return np.expand_dims(x, 0).astype(self.input_dtype), (pad_left, pad_right, pad_top, pad_bottom)

    def detect_boxes(self, frame: Image.Image, layout: Layout) -> list:
        """[(x1, y1, x2, y2, conf, class_name)] in frame pixels."""
        l, t, r, b = layout.detector_crop_px()
        crop = frame.crop((round(l), round(t), round(r), round(b)))   # PIL pads outside the frame with black
        x, (pl, pr, pt, pb) = self._prepare(crop)
        pred = self.sess.run(None, {self.input_name: x})[0][0]
        pred = pred[pred[:, 4] > MIN_CONF].astype(np.float64)
        # Build-A-Bot fix_bboxes: undo the padding and the resize
        pred[:, [0, 2]] -= pl
        pred[:, [1, 3]] -= pt
        pred[:, [0, 2]] *= crop.width / (self.model_w - pl - pr)
        pred[:, [1, 3]] *= crop.height / (self.model_h - pt - pb)
        pred[:, [0, 2]] += round(l)
        pred[:, [1, 3]] += round(t)
        return [(float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), self.class_names[int(p[5])])
                for p in pred]

    def side_of(self, frame: Image.Image, box) -> bool:
        """True = ally (Build-A-Bot SideDetector: 16x16 bicubic crop, argmax over (ally, enemy))."""
        crop = frame.crop(tuple(round(v) for v in box[:4])).resize((SIDE_SIZE, SIDE_SIZE), Image.Resampling.BICUBIC)
        x = np.expand_dims(np.array(crop, dtype=np.float32) / 255, 0)
        return int(np.argmax(self.side.run(None, {self.side_input: x})[0][0])) == 0

    def detect(self, frame: Image.Image, layout: Layout, deck: list) -> list:
        """Detections in game terms. A unit the viewer's deck cannot produce is an enemy; otherwise the side
        classifier decides (Build-A-Bot UnitDetector._calculate_side)."""
        possible_own = own_deck_units(deck)
        out = []
        for (x1, y1, x2, y2, conf, name) in self.detect_boxes(frame, layout):
            if name in NOT_A_UNIT:
                continue
            unit = CLASS_TO_UNIT.get(name)
            own = unit in possible_own and self.side_of(frame, (x1, y1, x2, y2))
            mx, my = layout.px_to_milli((x1 + x2) / 2, y2)       # ground point: box bottom centre
            mx = int(min(max(mx, 0), D.ARENA_W - 1))
            my = int(min(max(my, 0), D.ARENA_H - 1))
            out.append(Detection(unit=unit, label=name, own=own, x=mx, y=my, conf=conf))
        return out
