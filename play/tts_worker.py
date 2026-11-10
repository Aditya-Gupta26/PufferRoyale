#!/usr/bin/env python3
"""Neural voice for the play site's commentary, two engines (weights in $PR_ROOT/tools/hf_cache):
  * kokoro (default): Kokoro-82M (hexgrad, Apache 2.0) + misaki; installed by play/kokoro_install.sh into
    $PR_ROOT/tools/kokoro_py. Natural speech in about a second per line on CPU; expression comes from punctuation
    and --speed. Voices: am_fenrir, am_michael, am_puck (American male), bm_george (British male), af_heart, ...
  * chatterbox: Chatterbox TTS (Resemble AI, MIT); installed by play/tts_install.sh into $PR_ROOT/tools/tts_py.
    More dramatic (--exaggeration) but 8-12 s per 10-word line on a CPU node (measured 2026-11-06): GPU only.
Runs in the image:

    apptainer exec ... --env PYTHONPATH=$PR_ROOT/tools/kokoro_py,HF_HOME=$PR_ROOT/tools/hf_cache,HF_HUB_OFFLINE=1 $SIF \
        python play/tts_worker.py --dir $DATA_DIR/human_play/voice [--engine kokoro] [--voice am_fenrir] [--threads 4]
    ... python play/tts_worker.py --bench "Your Hog Rider just met a wall. Tragic!"      # time one line

Protocol (files, like hpc/llm_relay.py): the game server writes <dir>/req/<id>.txt (one commentary line); this
worker writes <dir>/<id>.wav (16-bit mono PCM at the model's rate, written atomically) and removes the request.
It touches <dir>/HEARTBEAT every 2 s while alive (the server offers the voice only then) and appends one line per
clip to <dir>/clips.jsonl (id, words, seconds of audio, seconds to make it). Stops on SIGTERM or <dir>/STOP.

Delivery: --exaggeration (0.5 neutral .. 1.0 very dramatic; default 0.75 for the roast announcer), --cfg (lower =
slower, more deliberate pacing; default 0.4), --temperature 0.8. Chatterbox marks its audio with an inaudible
watermark (Resemble's Perth), which is fine here.
"""
import argparse
import glob
import json
import os
import re
import signal
import sys
import time
import wave

import numpy as np

ID_RE = re.compile(r"^[A-Za-z0-9_\-]{6,80}$")
VOICES = ("am_fenrir", "am_michael", "am_puck", "bm_george", "af_heart")   # the Kokoro voices the site offers


def load(args):
    """A callable text -> float waveform (numpy), and its sample rate."""
    import torch
    torch.set_num_threads(max(1, int(args.threads)))
    t0 = time.time()
    if args.engine == "chatterbox":
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device=args.device)

        def run(text, voice=None):
            with torch.inference_mode():
                return model.generate(text, exaggeration=args.exaggeration, cfg_weight=args.cfg,
                                      temperature=args.temperature).detach().float().cpu().numpy().reshape(-1)
        sr = int(model.sr)
    else:
        from kokoro import KPipeline
        pipes = {}

        def run(text, voice=None):
            voice = voice if voice in VOICES else args.voice
            lang = "b" if voice.startswith("b") else "a"              # British voices need the British phonemizer
            if lang not in pipes:
                pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=args.device)
            parts = []
            with torch.inference_mode():
                for _, _, audio in pipes[lang](text, voice=voice, speed=args.speed):
                    if audio is not None:
                        parts.append(audio.detach().float().cpu().numpy().reshape(-1))
            return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        sr = 24000
        for v in VOICES:                                              # load every offered voice once, up front
            run("Warming up.", v)
    print(f"[tts] {args.engine} ({args.voice if args.engine == 'kokoro' else 'default voice'}) loaded on {args.device} "
          f"in {time.time() - t0:.1f} s (sample rate {sr})", flush=True)
    return run, sr


def synth(model, text, args, voice=None):
    run, sr = model
    a = run(text, voice)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 0.99:
        a = a * (0.99 / peak)
    return (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16), sr


def write_wav(path, pcm, sr):
    tmp = path + ".tmp"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="the voice directory shared with the game server")
    ap.add_argument("--bench", help="synthesize this line a few times, print the timing, write bench.wav, exit")
    ap.add_argument("--engine", default="kokoro", choices=("kokoro", "chatterbox"))
    ap.add_argument("--voice", default="am_fenrir", help="Kokoro voice (am_*, bm_*, af_*, bf_*)")
    ap.add_argument("--speed", type=float, default=1.08, help="Kokoro speaking speed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--exaggeration", type=float, default=0.75)
    ap.add_argument("--cfg", type=float, default=0.4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--keep-hours", type=float, default=72.0, help="clips older than this are deleted")
    args = ap.parse_args()
    model = load(args)

    if args.bench:
        for i in range(3):
            t0 = time.time()
            pcm, sr = synth(model, args.bench, args)
            dt = time.time() - t0
            print(f"[tts] run {i + 1}: {len(pcm) / sr:.2f} s of audio in {dt:.2f} s (real-time factor {dt / max(1e-6, len(pcm) / sr):.2f})",
                  flush=True)
        write_wav(os.path.join(args.dir or ".", "bench.wav"), pcm, sr)
        return

    if not args.dir:
        raise SystemExit("--dir is required (or --bench)")
    req = os.path.join(args.dir, "req")
    os.makedirs(req, exist_ok=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    hb = os.path.join(args.dir, "HEARTBEAT")
    last_hb = last_clean = 0.0
    with open(os.path.join(args.dir, "clips.jsonl"), "a") as log:
        while not stop["flag"] and not os.path.exists(os.path.join(args.dir, "STOP")):
            now = time.time()
            if now - last_hb > 2:
                with open(hb, "w") as f:
                    f.write(str(now))
                last_hb = now
            if now - last_clean > 600:                              # old clips go after --keep-hours
                for f in glob.glob(os.path.join(args.dir, "*.wav")):
                    if now - os.path.getmtime(f) > args.keep_hours * 3600:
                        os.remove(f)
                last_clean = now
            todo = sorted(glob.glob(os.path.join(req, "*.txt")), key=os.path.getmtime)
            if not todo:
                time.sleep(0.1)
                continue
            if len(todo) > 3:                                       # behind: drop the oldest, the match moved on
                for f in todo[:-3]:
                    os.remove(f)
                todo = todo[-3:]
            path = todo[0]
            cid = os.path.basename(path)[:-4]
            try:
                raw = open(path).read().strip()
            except OSError:
                continue
            os.remove(path)
            voice = None
            if raw.startswith("{"):                                    # {"text": ..., "voice": ...}
                try:
                    req_ = json.loads(raw)
                    raw, voice = str(req_.get("text", "")), req_.get("voice")
                except ValueError:
                    raw = ""
            text = raw[:300]
            if not ID_RE.match(cid) or not text:
                continue
            t0 = time.time()
            try:
                pcm, sr = synth(model, text, args, voice)
            except Exception as e:                                  # one bad line must not stop the voice
                log.write(json.dumps({"id": cid, "error": repr(e)[:300]}) + "\n")
                log.flush()
                continue
            write_wav(os.path.join(args.dir, cid + ".wav"), pcm, sr)
            dt = time.time() - t0
            log.write(json.dumps({"id": cid, "voice": voice, "words": len(text.split()), "audio_s": round(len(pcm) / sr, 2),
                                  "make_s": round(dt, 2), "time": time.strftime("%H:%M:%S")}) + "\n")
            log.flush()
    print("[tts] stopped", flush=True)


if __name__ == "__main__":
    main()
