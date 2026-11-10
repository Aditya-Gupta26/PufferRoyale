#!/usr/bin/env python3
"""A self-contained HTML player for a recorded Claude-vs-policy game (DECISIONS D101): the GIF's frames as embedded
WebP images on a canvas, with play / pause, speed, a frame slider and keyboard controls (space, arrows). For
viewers that show an animated GIF as a stack of still images. No network access needed.

    python hpc/llm_player.py GAME_GIF OUT_HTML [--title TEXT]
"""
import argparse
import base64
import io
import json

from PIL import Image

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #eef0ec; --fg: #1d2420; --muted: #5d6a62; --accent: #2a62c9; --panel: #ffffff; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161a18; --fg: #e8ece9; --muted: #9aa69f; --accent: #6f9cf0;
    --panel: #212724; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.4 system-ui, sans-serif; }
  main { max-width: 420px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 16px; margin: 0 0 4px; } p { margin: 0 0 12px; color: var(--muted); font-size: 13px; }
  canvas { width: 100%; height: auto; display: block; border-radius: 8px; background: var(--panel); }
  .bar { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  button, select { font: inherit; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--muted);
    background: var(--panel); color: var(--fg); cursor: pointer; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; min-width: 76px; }
  input[type=range] { flex: 1 1 100%; accent-color: var(--accent); }
  .count { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 13px; }
</style></head>
<body><main>
  <h1>__TITLE__</h1>
  <p>Space = play / pause, arrow keys = one frame back / forward. 1x = real time.</p>
  <canvas id="c" width="__W__" height="__H__"></canvas>
  <div class="bar">
    <button class="primary" id="play">Play</button>
    <button id="restart">Restart</button>
    <select id="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option>
      <option value="2">2x</option><option value="4">4x</option></select>
    <span class="count" id="count"></span>
    <input type="range" id="seek" min="0" value="0">
  </div>
</main>
<script>
const FRAMES = __FRAMES__, DELAYS = __DELAYS__;
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
const playBtn = document.getElementById("play"), seek = document.getElementById("seek");
const speed = document.getElementById("speed"), count = document.getElementById("count");
const imgs = FRAMES.map(src => { const im = new Image(); im.src = "data:image/webp;base64," + src; return im; });
seek.max = imgs.length - 1;
let i = 0, timer = null;
function show(k) {
  i = Math.max(0, Math.min(imgs.length - 1, k));
  const im = imgs[i];
  const draw = () => ctx.drawImage(im, 0, 0);
  if (im.complete) draw(); else im.onload = draw;
  seek.value = i; count.textContent = (i + 1) + " / " + imgs.length;
}
function stop() { clearTimeout(timer); timer = null; playBtn.textContent = "Play"; }
function tick() {
  if (i >= imgs.length - 1) { stop(); return; }
  show(i + 1); timer = setTimeout(tick, DELAYS[i] / parseFloat(speed.value));
}
function play() { if (i >= imgs.length - 1) show(0); playBtn.textContent = "Pause"; tick(); }
playBtn.onclick = () => timer ? stop() : play();
document.getElementById("restart").onclick = () => { stop(); show(0); };
seek.oninput = () => { stop(); show(parseInt(seek.value)); };
document.addEventListener("keydown", e => {
  if (e.code === "Space") { e.preventDefault(); timer ? stop() : play(); }
  else if (e.code === "ArrowRight") { stop(); show(i + 1); }
  else if (e.code === "ArrowLeft") { stop(); show(i - 1); }
});
imgs[0].onload = () => show(0);
if (imgs[0].complete) show(0);
</script></body></html>
"""


def build(gif, out, title):
    im = Image.open(gif)
    frames, delays = [], []                          # per frame: the GIF writer merges identical consecutive
    for k in range(im.n_frames):                     # frames into one longer one
        im.seek(k)
        delays.append(int(im.info.get("duration", 200)))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "WEBP", quality=80, method=4)
        frames.append(base64.b64encode(buf.getvalue()).decode())
    html = (PAGE.replace("__TITLE__", title).replace("__W__", str(im.size[0])).replace("__H__", str(im.size[1]))
            .replace("__DELAYS__", json.dumps(delays)).replace("__FRAMES__", json.dumps(frames)))
    with open(out, "w") as f:
        f.write(html)
    return len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gif")
    ap.add_argument("out")
    ap.add_argument("--title", default="PufferRoyale game replay")
    a = ap.parse_args()
    n = build(a.gif, a.out, a.title)
    print(a.out, n, "frames")


if __name__ == "__main__":
    main()
