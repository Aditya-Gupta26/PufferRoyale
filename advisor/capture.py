"""Frame sources. Each yields (frame: PIL.Image RGB, t: seconds on the source's own clock).

  ScreenRegion  a rectangle of the desktop (the scrcpy window of a phone, or the PC client window); needs `mss`
  VideoFile     a recorded match (t = the video timestamp, so analysis runs faster or slower than real time)
  ImageFolder   numbered images at a fixed rate

Nothing here, or anywhere in the advisor, sends input to the game.
"""
from __future__ import annotations

import glob
import os
import time

from PIL import Image


class ScreenRegion:
    def __init__(self, left: int, top: int, width: int, height: int, fps: float = 10.0):
        import mss   # pip install mss
        self.sct = mss.mss()
        self.box = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        self.period = 1.0 / fps

    def __iter__(self):
        next_t = time.monotonic()
        while True:
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += self.period
            shot = self.sct.grab(self.box)
            yield Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"), time.monotonic()


class VideoFile:
    def __init__(self, path: str, fps: float = 10.0):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.step = max(1, round(self.src_fps / fps))

    def __iter__(self):
        i = 0
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                return
            if i % self.step == 0:
                t = self.cap.get(self.cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield Image.fromarray(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)), t
            i += 1


class ImageFolder:
    def __init__(self, path: str, fps: float = 2.0, pattern: str = "*.jpg"):
        self.files = sorted(glob.glob(os.path.join(path, pattern)))
        if not self.files:
            raise FileNotFoundError(f"no {pattern} in {path}")
        self.period = 1.0 / fps

    def __iter__(self):
        for i, f in enumerate(self.files):
            yield Image.open(f).convert("RGB"), i * self.period
