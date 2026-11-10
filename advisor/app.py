"""The advisor program: screen -> suggestions in a separate window. It never sends input to the game.

Live (phone mirrored with scrcpy, or the PC client), on the player's own computer:

    python -m advisor.app --ckpt S4_A_s1/snap_476523.pt --deck "Hog Rider,Musketeer,Cannon,Ice Golem,Ice Spirit,Skeletons,Fireball,The Log" \
        --source screen --region LEFT,TOP,WIDTH,HEIGHT [--layout-file mydevice.json]

Offline, on a recorded match (writes an annotated video, no window):

    python -m advisor.app --ckpt ... --deck ... --source video --video match.mp4 --out annotated.mp4

Keys in the window: s = the battle starts now (re-sync the clock), r = reset for a new match, q = quit.
The clock starts at the first frame showing the battle screen; press s at the "3:00" mark if it drifted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from pufferroyale.game import TOWER_TROOPS, card_id

from . import encoder
from .perception.layout import PROFILES, Layout
from .perception.reader import ScreenReader
from .policy import PolicyAdvisor
from .tracker import Tracker
from . import overlay

DECISION_S = 0.5     # the policy acts every 10 ticks x 50 ms


class Session:
    """One advisor session: reader + tracker + policy, fed one frame at a time."""

    def __init__(self, ckpt: str, deck: list, layout: Layout, tower_troops=(0, 0), assets: str = None,
                 threads: int = 2, top_k: int = 3, assume_battle: bool = False):
        self.deck = deck
        self.tower_troops = tower_troops
        self.reader = ScreenReader(layout, deck, assets=assets, threads=threads)
        self.policy = PolicyAdvisor(ckpt)
        self.top_k = top_k
        self.assume_battle = assume_battle
        self.reset()

    def reset(self) -> None:
        self.tracker = Tracker(deck=self.deck, tower_troops=self.tower_troops)
        self.policy.reset()
        self.start = None             # source time of the battle start
        self.next_decision = 0.0
        self.suggestion = None
        self.board = None

    def sync_now(self, t_src: float) -> None:
        self.start = t_src

    def step(self, frame, t_src: float) -> dict:
        """Process one frame. Returns a dict for logging (and keeps the latest suggestion on self)."""
        c0 = time.perf_counter()
        reading = self.reader.read(frame, 0.0 if self.start is None else t_src - self.start, force=self.assume_battle)
        c1 = time.perf_counter()
        if not reading.in_battle and not self.assume_battle:
            return {"t": 0.0 if self.start is None else round(t_src - self.start, 3), "in_battle": False,
                    "ms_read": round(1000 * (c1 - c0), 1)}
        if self.start is None:        # the first frame showing the battle: the clock starts here
            self.start = t_src
        t = reading.t = t_src - self.start
        self.board = self.tracker.update(reading)
        c2 = time.perf_counter()
        decided = False
        c3 = c4 = c2
        if t >= self.next_decision:
            obs = encoder.encode(self.board)
            c3 = time.perf_counter()
            self.suggestion = self.policy.suggest(obs, self.board.hand, k=self.top_k)
            c4 = time.perf_counter()
            self.next_decision = max(self.next_decision + DECISION_S, t)
            decided = True
        s = self.suggestion
        return {"t": round(t, 3), "in_battle": True, "decided": decided, "tick": self.board.tick,
                "ms_read": round(1000 * (c1 - c0), 1), "ms_track": round(1000 * (c2 - c1), 1),
                "ms_encode": round(1000 * (c3 - c2), 1), "ms_policy": round(1000 * (c4 - c3), 1),
                "detections": len(reading.detections),
                "elixir": self.board.elixir, "hand": self.board.hand, "units": len(self.board.units),
                "opp_seen": sorted(self.board.opp_seen), "opp_spent": self.board.opp_spent,
                "p_wait": None if s is None else round(s.p_wait, 4),
                "top": None if (s is None or not s.cards) else
                [(c.name, c.tile, round(c.p_card, 4), round(c.p_tile, 4)) for c in s.cards]}


def parse_deck(text: str) -> list:
    cards = [card_id(x.strip()) for x in text.split(",") if x.strip()]
    if len(cards) != 8 or len(set(cards)) != 8:
        raise SystemExit(f"--deck needs 8 distinct cards of the simulator's 64, got {len(cards)}: {text}")
    return cards


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="policy checkpoint (snap_*.pt / model_*.pt / run dir)")
    ap.add_argument("--deck", required=True, help="your 8 cards, comma separated (names of the simulator's 64)")
    ap.add_argument("--tower-troop", default="princess", choices=TOWER_TROOPS, help="yours")
    ap.add_argument("--opp-tower-troop", default="princess", choices=TOWER_TROOPS)
    ap.add_argument("--source", choices=("screen", "video", "images"), required=True)
    ap.add_argument("--region", help="screen: LEFT,TOP,WIDTH,HEIGHT of the game in desktop pixels")
    ap.add_argument("--video", help="video: path of a recorded match")
    ap.add_argument("--images", help="images: folder of frames")
    ap.add_argument("--fps", type=float, default=8.0, help="frames read per second")
    ap.add_argument("--layout", default="bluestacks_720x1280", choices=sorted(PROFILES))
    ap.add_argument("--layout-file", help="a calibrated layout (python -m advisor.calibrate)")
    ap.add_argument("--assets", help="Build-A-Bot clashroyalebuildabot/ directory (models/, images/)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", help="video/images: write the annotated frames to this .mp4 instead of a window")
    ap.add_argument("--log", help="write one JSON line per processed frame")
    ap.add_argument("--assume-battle", action="store_true", help="skip the in-battle screen check (testing)")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args(argv)

    from . import capture
    import cv2
    layout = Layout.load(args.layout_file) if args.layout_file else PROFILES[args.layout]()
    deck = parse_deck(args.deck)
    tt = (TOWER_TROOPS.index(args.tower_troop), TOWER_TROOPS.index(args.opp_tower_troop))
    sess = Session(args.ckpt, deck, layout, tower_troops=tt, assets=args.assets, threads=args.threads,
                   assume_battle=args.assume_battle)
    if sess.policy.env_config.get("tower_troop0", "princess") != args.tower_troop or \
            sess.policy.env_config.get("tower_troop1", "princess") != args.opp_tower_troop:
        print(f"warning: the checkpoint was trained with tower troops {sess.policy.env_config.get('tower_troop0')} / "
              f"{sess.policy.env_config.get('tower_troop1')}", file=sys.stderr)
    if args.source == "screen":
        if not args.region:
            raise SystemExit("--source screen needs --region LEFT,TOP,WIDTH,HEIGHT")
        src = capture.ScreenRegion(*[int(v) for v in args.region.split(",")], fps=args.fps)
    elif args.source == "video":
        src = capture.VideoFile(args.video, fps=args.fps)
    else:
        src = capture.ImageFolder(args.images, fps=args.fps)
    writer = None
    log = open(args.log, "w", buffering=1) if args.log else None    # line-buffered: partial runs keep their lines
    window = not args.out and not args.no_window
    t_proc = []
    try:
        for frame, t_src in src:
            t0 = time.perf_counter()
            rec = sess.step(frame, t_src)
            t_proc.append(time.perf_counter() - t0)
            rec["proc_ms"] = round(1000 * t_proc[-1], 1)
            if window or args.out:
                status = "battle" if rec["in_battle"] else "waiting for the battle screen"
                t1 = time.perf_counter()
                img = overlay.render(np.asarray(frame), sess.reader.layout_for(frame), sess.suggestion, sess.board,
                                     sess.tracker, status=f"{status}  t={rec['t']:.1f}s  {rec['proc_ms']:.0f} ms")
                rec["ms_render"] = round(1000 * (time.perf_counter() - t1), 1)
            if log:
                log.write(json.dumps(rec) + "\n")
            if not window and not args.out:
                continue
            if args.out:
                if writer is None:
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                             (img.shape[1], img.shape[0]))
                writer.write(img)
            else:
                cv2.imshow("PufferRoyale advisor", img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("r"):
                    sess.reset()
                if k == ord("s"):
                    sess.sync_now(t_src)
    finally:
        if writer is not None:
            writer.release()
        if log:
            log.close()
        if window:
            cv2.destroyAllWindows()
    if t_proc:
        print(f"frames {len(t_proc)}, processing ms median {1000 * float(np.median(t_proc)):.1f}, "
              f"p90 {1000 * float(np.percentile(t_proc, 90)):.1f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
