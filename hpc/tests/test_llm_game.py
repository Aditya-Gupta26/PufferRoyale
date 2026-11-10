"""Checks of the Claude-vs-policy games (DECISIONS D101, 2026-10-09). Cluster-side (not under tests/spec): run with
`python -m pytest -q hpc/tests`. No test calls Claude: a stub `claude` executable stands in for the CLI, and the
end-to-end games are answered by the repo's text-only mock_first_legal.

- hpc/llm_game.py: the game schedule (decks, mirror / pool deck, seats), the saved policy picked per game, two
  recorded games end to end through the relay files (meta, frames, plays, decisions, result, GIF, state.json,
  W&B offline rows with llm/score and the replay), a resumed run that plays the last game against the final
  model and writes DONE.
- hpc/llm_relay.py: the exact CLI flags (no tools, no MCP, no settings, the rules as system prompt), the reply
  file, a failed call retried, and a model that stays unavailable giving a fatal reply.
"""
import glob
import json
import os
import stat
import subprocess
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hpc"))
sys.path.insert(0, os.path.join(ROOT, "hpc", "tests"))

torch = pytest.importorskip("torch")

STUB = r"""#!/bin/bash
# stand-in for the Claude Code CLI: records its argv, fails the first $STUB_FAIL calls, answers like `-p --output-format json`
mkdir -p "$STUB_DIR"
printf '%s\0' "$@" > "$STUB_DIR/argv_$(date +%s%N)"
n=$(ls "$STUB_DIR" | grep -c '^argv_')
if [ "$n" -le "${STUB_FAIL:-0}" ]; then echo '{"is_error": true, "subtype": "error", "result": "usage limit reached"}'; exit 1; fi
echo '{"is_error": false, "subtype": "success", "result": "Defend the left lane.\nPLAY 0 AT 9,20", "total_cost_usd": 0.02, "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 3000, "cache_creation_input_tokens": 0}}'
"""


def test_game_schedule():
    import llm_game as G
    cfg = [G.game_config(k) for k in range(1, 9)]
    assert [c["claude_deck"] for c in cfg] == ["xbow", "royal_hogs"] * 4
    assert [c["mirror"] for c in cfg] == [True, True, False, False] * 2
    assert [c["policy_deck"] for c in cfg] == ["xbow", "royal_hogs", "hog26", "hog26", "xbow", "royal_hogs",
                                               "giant", "giant"]
    assert [c["claude_team"] for c in cfg] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert {G.game_config(k)["policy_deck"] for k in range(1, 41)} >= set(G.POOL)          # every pool deck in 40


def test_pick_policy(tmp_path):
    import llm_game as G
    for n in ("snap_477.pt", "snap_954.pt", "snap_1431.pt", "model_001500.pt", "learner.pt"):
        (tmp_path / n).write_bytes(b"x")
    pick = lambda step: (G.pick_policy(str(tmp_path), 1000, step) or {}).get("epoch")     # noqa: E731
    assert pick(1) == 477 and pick(477_000) == 477 and pick(477_001) == 954 and pick(1_431_001) == 1500
    assert pick(1_500_001) is None
    assert G.pick_policy(str(tmp_path), 1000, 900_000)["global_step"] == 954_000


def _stub(tmp_path):
    p = tmp_path / "claude"
    p.write_text(STUB)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _relay(tmp_path, relay, env_extra, *args):
    env = dict(os.environ, STUB_DIR=str(tmp_path / "stub"), **env_extra)
    return subprocess.Popen([sys.executable, os.path.join(ROOT, "hpc", "llm_relay.py"), "--relay-dir", str(relay),
                             "--claude", _stub(tmp_path), "--poll", "0.05", "--backoff-scale", "0.01", *args],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait(path, timeout=60):
    t0 = time.time()
    while not os.path.exists(path):
        assert time.time() - t0 < timeout, f"{path} never appeared"
        time.sleep(0.05)
    return json.load(open(path))


def test_relay_flags_retry_and_reply(tmp_path):
    relay = tmp_path / "relay"
    (relay / "g001").mkdir(parents=True)
    (relay / "RULES_PROMPT.txt").write_text("RULES")
    (relay / "g001" / "turn_0000.txt").write_text("=== DECISION 0 ===\nSTATE")
    p = _relay(tmp_path, relay, {"STUB_FAIL": "1"}, "--model", "claude-opus-5-5", "--effort", "low")
    try:
        rep = _wait(relay / "g001" / "reply_0000.json")
    finally:
        (relay / "STOP").write_text("")
        p.wait(timeout=30)
    assert rep["text"] == "Defend the left lane.\nPLAY 0 AT 9,20"
    assert rep["meta"]["attempts"] == 2 and not rep["meta"].get("fatal")
    assert rep["meta"]["usage"]["output_tokens"] == 20 and rep["meta"]["cost_usd_estimate"] == 0.02
    argv = [open(f).read().split("\0")[:-1] for f in sorted(glob.glob(str(tmp_path / "stub" / "argv_*")))]
    assert len(argv) == 2 and argv[0] == argv[1]
    a = argv[0]
    assert a[0] == "-p" and a[1] == "=== DECISION 0 ===\nSTATE"
    assert a[a.index("--model") + 1] == "claude-opus-5-5" and a[a.index("--effort") + 1] == "low"
    assert a[a.index("--system-prompt-file") + 1] == str(relay / "RULES_PROMPT.txt")
    assert a[a.index("--tools") + 1] == "" and a[a.index("--setting-sources") + 1] == ""
    for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in a
    assert a[a.index("--output-format") + 1] == "json"
    calls = [json.loads(ln) for ln in open(relay / "calls.jsonl")]
    assert [c["attempts"] for c in calls] == [1, 2] and "usage limit" in calls[0]["error"]


def test_relay_fatal_when_model_stays_unavailable(tmp_path):
    relay = tmp_path / "relay"
    (relay / "g001").mkdir(parents=True)
    (relay / "RULES_PROMPT.txt").write_text("RULES")
    (relay / "g001" / "turn_0000.txt").write_text("STATE")
    p = _relay(tmp_path, relay, {"STUB_FAIL": "1000"}, "--max-wait", "1")
    try:
        rep = _wait(relay / "g001" / "reply_0000.json")
    finally:
        (relay / "STOP").write_text("")
        p.wait(timeout=30)
    assert rep["text"] == "WAIT" and rep["meta"]["fatal"] and rep["meta"]["attempts"] >= 2


def _answer_turns(relay, stop):
    """The test's player: mock_first_legal on every turn file (it reads only the rendered text)."""
    from pufferroyale.llm import mock_first_legal
    while not stop.is_set():
        for turn in glob.glob(os.path.join(relay, "g*", "turn_*.txt")):
            reply = os.path.join(os.path.dirname(turn), "reply_" + os.path.basename(turn)[5:-4] + ".json")
            if not os.path.exists(reply):
                try:
                    text = "Mock reasoning line.\n" + mock_first_legal("", open(turn).read())
                    with open(reply + ".tmp", "w") as f:
                        json.dump({"text": text, "meta": {"latency_s": 0.01, "cost_usd_estimate": 0.01,
                                                          "usage": {"output_tokens": 5}}}, f)
                    os.replace(reply + ".tmp", reply)
                except FileNotFoundError:               # a finished game's directory went away meanwhile
                    continue
        time.sleep(0.02)


def _game(run, relay, wd, *extra):
    env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR=str(wd), WANDB_SILENT="true", WANDB_API_KEY="")
    cmd = [sys.executable, os.path.join(ROOT, "hpc", "llm_game.py"), str(run), "--relay-dir", str(relay),
           "--every-steps", "1000", "--max-ticks", "200", "--poll", "0.2", "--render-every", "8",
           "--wandb", "--wandb-group", "g", *extra]
    stop = threading.Event()
    th = threading.Thread(target=_answer_turns, args=(str(relay), stop), daemon=True)
    th.start()
    try:
        return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    finally:
        stop.set()
        th.join(timeout=10)


def test_llm_game_end_to_end(tmp_path):
    pytest.importorskip("wandb")
    from test_stage4 import _random_policy, _wandb_history
    run, relay, wd = tmp_path / "run", tmp_path / "relay", tmp_path / "wandb"
    run.mkdir()
    wd.mkdir()
    json.dump({"env": {"frame_skip": 10}, "league": {"run_id": "fake", "total_timesteps": 3000, "batch_size": 1000,
                                                      "num_envs": 8}}, open(run / "config.json", "w"))
    _random_policy(run / "snap_1.pt", 1)
    _random_policy(run / "snap_2.pt", 2)
    r = _game(run, relay, wd, "--max-games", "2")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    st = json.load(open(run / "llm" / "state.json"))
    assert [st["games"][k]["status"] for k in ("1", "2")] == ["done", "done"] and "3" not in st["games"]
    assert [st["games"][k]["policy"]["epoch"] for k in ("1", "2")] == [1, 2]
    assert st["games"]["1"]["wandb"] is True and not (run / "llm" / "DONE").exists()
    for k in ("1", "2"):
        g = run / "llm" / f"game_{int(k):03d}"
        res = json.load(open(g / "result.json"))
        meta = json.load(open(g / "meta.json"))
        assert res["ticks"] == 200 and res["decisions"] >= 1 and res["plays"] >= 1 and res["parse_errors"] == 0
        assert meta["claude_deck"] == ("xbow" if k == "1" else "royal_hogs") and meta["mirror"]
        decs = [json.loads(ln) for ln in open(g / "decisions.jsonl")]
        assert len(decs) == res["decisions"] and len(glob.glob(str(g / "turn_*.txt"))) == len(decs)
        assert decs[0]["reply"].startswith("Mock reasoning line.")
        assert (g / f"game_{int(k):03d}.gif").stat().st_size > 1000
        assert not (relay / f"g{int(k):03d}").exists()                  # the live files are removed
    files = glob.glob(str(wd / "**" / "offline-run-*" / "*.wandb"), recursive=True)
    assert len(files) == 1
    rows = [x for x in _wandb_history(files[0]) if "llm/score" in x]
    assert [x["global_step"] for x in rows] == [1000, 2000]
    # media are stored as flattened sub-keys of the logged key
    assert all(x["llm/replay/_type"] == "video-file" and x["llm/replay/path"].endswith(".gif") for x in rows)
    assert [x["llm/decision_log/_type"] for x in rows] == ["table-file"] * 2
    assert [x["llm/decision_log/nrows"] for x in rows] == [st["games"][k]["decisions"] for k in ("1", "2")]
    assert glob.glob(str(wd / "**" / "media" / "videos" / "**" / "*.gif"), recursive=True)
    # the run finishes: game 3 is played against the final model, then DONE
    _random_policy(run / "model_000003.pt", 3)
    json.dump({"global_step": 3000}, open(run / "league_state.json", "w"))
    r = _game(run, relay, wd)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    st = json.load(open(run / "llm" / "state.json"))
    assert st["games"]["3"]["status"] == "done" and st["games"]["3"]["policy"]["kind"] == "model"
    assert st["games"]["3"]["config"]["policy_deck"] == "hog26" and (run / "llm" / "DONE").exists()
    assert "2 already done" in r.stdout
