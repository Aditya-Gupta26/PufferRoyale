#!/usr/bin/env python3
"""Relay for hpc/llm_game.py (DECISIONS D101): answers each decision of Claude's side with ONE headless Claude
Code call on the HOST (outside the image: the CLI and its claude.ai login live in $HOME), no SDK and no API key:

    claude -p <turn text> --model M --system-prompt-file <relay>/RULES_PROMPT.txt --tools "" --strict-mcp-config
           --no-session-persistence --disable-slash-commands --setting-sources "" --output-format json [--effort E]

No tools, no MCP servers, no settings, hooks, CLAUDE.md or memory: the model sees the game's rules (the system
prompt) and the turn text only, and every call is stateless. Stdlib only (the host's python3).

Every <relay>/g*/turn_<n>.txt without a reply gets <relay>/g*/reply_<n>.json = {"text": reply, "meta": {latency_s,
usage, cost_usd_estimate, rc, error, attempts}} (written atomically). A failed call (non-zero exit, is_error, no
JSON, timeout -- e.g. a usage limit) is retried with back-off; after --max-wait seconds of failures the reply is
{"text": "WAIT", "meta": {"fatal": true, ...}} and the game host aborts that game (retried later). Every call is
appended to <relay>/calls.jsonl. Stops when <relay>/STOP exists or on SIGTERM.

    python3 hpc/llm_relay.py --relay-dir DIR [--model claude-opus-5-5] [--effort low] [--claude PATH]
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

BACKOFF = (20, 60, 120, 300, 600, 900)        # seconds between retries of a failed call (then 900 each)
FLAGS = ["--tools", "", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands",
         "--setting-sources", "", "--output-format", "json"]


def claude_cmd(claude, model, effort, system_file, prompt):
    cmd = [claude, "-p", prompt, "--model", model, "--system-prompt-file", system_file] + FLAGS
    return cmd + (["--effort", effort] if effort else [])


def call(cmd, timeout, cwd):
    """(reply text or None, meta)."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, {"error": f"timeout after {timeout:.0f} s", "latency_s": round(time.time() - t0, 3)}
    except OSError as e:
        return None, {"error": f"cannot run {cmd[0]}: {e}", "latency_s": round(time.time() - t0, 3)}
    meta = {"rc": p.returncode, "latency_s": round(time.time() - t0, 3)}
    try:
        d = json.loads(p.stdout)
    except ValueError:
        meta["error"] = f"rc {p.returncode}, no JSON: {(p.stdout or p.stderr or '').strip()[-300:]}"
        return None, meta
    u = d.get("usage") or {}
    meta.update(is_error=bool(d.get("is_error")), subtype=d.get("subtype"), duration_api_ms=d.get("duration_api_ms"),
                cost_usd_estimate=d.get("total_cost_usd"),
                usage={k: u.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                             "cache_creation_input_tokens")})
    if p.returncode != 0 or d.get("is_error") or not isinstance(d.get("result"), str):
        meta["error"] = str(d.get("result") or d.get("subtype") or f"rc {p.returncode}")[-300:]
        return None, meta
    return d["result"], meta


def answer(turn, reply, a, log):
    """Call until it works or --max-wait has passed; write the reply file."""
    prompt = open(turn).read()
    cmd = claude_cmd(a.claude, a.model, a.effort, os.path.join(a.relay_dir, "RULES_PROMPT.txt"), prompt)
    t_first, attempt = time.time(), 0
    while True:
        attempt += 1
        text, meta = call(cmd, a.timeout, os.path.dirname(turn))
        meta["attempts"] = attempt
        log.write(json.dumps({"turn": os.path.relpath(turn, a.relay_dir), "time": time.strftime("%H:%M:%S"),
                              **meta}) + "\n")
        log.flush()
        if text is not None:
            break
        waited = time.time() - t_first
        if waited >= a.max_wait:
            text, meta["fatal"] = "WAIT", True
            break
        delay = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)] * a.backoff_scale
        print(f"[llm_relay] {os.path.relpath(turn, a.relay_dir)}: {meta.get('error')} -- retry in {delay} s",
              flush=True)
        time.sleep(min(delay, max(1.0, a.max_wait - waited)))
    with open(reply + ".tmp", "w") as f:
        json.dump({"text": text, "meta": meta}, f)
    os.replace(reply + ".tmp", reply)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--relay-dir", required=True)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", help="--effort of the CLI ('' = the CLI's default)")
    ap.add_argument("--claude", default=shutil.which("claude") or os.path.expanduser("~/.local/bin/claude"))
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per call")
    ap.add_argument("--max-wait", type=float, default=10800.0, help="give up on a decision after this long")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--backoff-scale", type=float, default=1.0, help="multiplies the retry delays (tests)")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    os.makedirs(a.relay_dir, exist_ok=True)
    print(f"[llm_relay] {a.relay_dir}: {a.claude} --model {a.model} --effort {a.effort or '(default)'}", flush=True)
    with open(os.path.join(a.relay_dir, "calls.jsonl"), "a") as log:
        while not os.path.exists(os.path.join(a.relay_dir, "STOP")):
            busy = False
            for turn in sorted(glob.glob(os.path.join(a.relay_dir, "g*", "turn_*.txt"))):
                reply = os.path.join(os.path.dirname(turn), "reply_" + os.path.basename(turn)[5:-4] + ".json")
                if not os.path.exists(reply) and os.path.isfile(os.path.join(a.relay_dir, "RULES_PROMPT.txt")):
                    try:
                        answer(turn, reply, a, log)
                    except FileNotFoundError:           # the game ended and its directory went away meanwhile
                        continue
                    busy = True
            if not busy:
                time.sleep(a.poll)


if __name__ == "__main__":
    main()
