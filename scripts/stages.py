#!/usr/bin/env python3
"""Bot-ladder stage runner (SPEC §19.7.4, §19.9.10; TRAINING_PLAN §8.1): chained league_train.py
runs, one scripted (or checkpoint) anchor per rung, each gated on the learner's score vs that anchor.

    python scripts/stages.py --run-prefix S1main_s0 --data-dir experiments --device cuda --num-envs 1024 \\
        --snapshot-interval 50 --seed 0 --env.reward-tower 0.3 --env.reward-crown 0.2 --train.reward-clip 0

--rungs "ANCHOR:BUDGET:THRESHOLD;..." (default
"bot:noop:5000000:0.95;bot:random:50000000:0.90;bot:heuristic:300000000:0.55"): rung i is the league
run <data-dir>/league/<P>_<i>_<anchor kind>, started as

    league_train.py --anchors ANCHOR --self-play-frac 0 --anchor-frac 1.0 --total-timesteps BUDGET
                    --early-stop-score THRESHOLD --early-stop-window N --run-id ... --data-dir ...
                    [--init-from <rung i-1's final model> for i > 0] <every other flag given here>

(N = --gate-window, default 2000). The run finishes at its budget or early, at the first snapshot
epoch where the learner scored >= THRESHOLD over its last N matches vs the anchor. Gate: the
learner's score vs the anchor over its last N finished matches >= THRESHOLD, from the early-stop
window persisted in the rung's final league_state.json (SPEC §19.10.4; never from records a
preempted process wrote after its last save); fewer than N matches FAILS the gate ("insufficient
matches"). The ladder stops at the first failed gate. A rung whose run directory already holds a
run is resumed (league_train.py --resume, e.g. after a preemption) instead of restarted. Rungs run
in the caller's working directory (league_train.py is invoked by its absolute path), so relative
paths in forwarded flags (e.g. --env.deck-pool file:decks.json) are the caller's.

Output: <data-dir>/stages_<P>.json (strict JSON; also the last stdout line): per rung the anchor,
budget, threshold, run dir, steps, score, matches, passed (+ early_stopped, model, reason). Exit
status 0 when every gate passed, 1 when a gate failed, 2 when a rung's league_train.py failed.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from pufferroyale.league import json_safe  # noqa: E402  (numpy only)

DEFAULT_RUNGS = "bot:noop:5000000:0.95;bot:random:50000000:0.90;bot:heuristic:300000000:0.55"
#: league_train.py flags stages.py sets per rung (forwarding them is an error)
RESERVED = ("--anchors", "--self-play-frac", "--anchor-frac", "--early-stop-score", "--early-stop-window",
            "--init-from", "--resume", "--run-id", "--total-timesteps", "--data-dir")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rungs", default=DEFAULT_RUNGS, help=f"ANCHOR:BUDGET:THRESHOLD items separated by ';' "
                                                           f"(default {DEFAULT_RUNGS})")
    ap.add_argument("--run-prefix", required=True, help="rung i runs as <prefix>_<i>_<anchor kind>")
    ap.add_argument("--gate-window", type=int, default=2000, help="matches per gate (also --early-stop-window; "
                                                                  "default 2000)")
    ap.add_argument("--data-dir", default="experiments", help="league runs go to <data-dir>/league/ (default "
                                                              "experiments)")
    return ap


def parse_rungs(text):
    """[(anchor, budget, threshold)] from 'ANCHOR:BUDGET:THRESHOLD;...' (the anchor may contain ':')."""
    from pufferroyale.league import parse_spec
    out = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(f"rung {item!r}: expected ANCHOR:BUDGET:THRESHOLD")
        anchor, budget, thr = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if parse_spec(anchor)[0] == "self":
            raise ValueError(f"rung {item!r}: 'self' is not an anchor")
        try:
            b = float(budget.replace("_", ""))
            t = float(thr)
        except ValueError:
            raise ValueError(f"rung {item!r}: BUDGET must be an integer and THRESHOLD a number") from None
        if not (b >= 1 and b == int(b)) or not (t == t and abs(t) != float("inf")):
            raise ValueError(f"rung {item!r}: BUDGET must be an integer >= 1 and THRESHOLD finite")
        out.append((anchor, int(b), t))
    if not out:
        raise ValueError("no rungs")
    return out


def anchor_label(anchor):
    return anchor.split(":", 1)[1] if anchor.startswith("bot:") else "ckpt"


def _anchor_key(anchor):
    """An anchor spec in comparable form (a ckpt path made absolute, relative to the cwd)."""
    return f"ckpt:{os.path.abspath(anchor[5:])}" if anchor.startswith("ckpt:") else anchor


def gate(run_dir, anchor, window):
    """(score, matches) of the learner vs `anchor` over its last `window` finished matches (score None
    when there are none). SPEC §19.10.4: the source is the early-stop window persisted in the rung's
    final league_state.json (early_stop.recent[anchor], the outcomes +1/0/-1 at the final save), so
    records a preempted process wrote after its last save never count."""
    path = os.path.join(run_dir, "league_state.json")
    recent = {}
    if os.path.isfile(path):
        with open(path) as f:
            recent = ((json.load(f).get("early_stop") or {}).get("recent")) or {}
    outcomes = recent.get(anchor)
    if outcomes is None:
        outcomes = next((v for k, v in recent.items() if _anchor_key(k) == _anchor_key(anchor)), [])
    last = [int(r) for r in outcomes][-int(window):]
    if not last:
        return None, 0
    return sum(0.5 * (r + 1) for r in last) / len(last), len(last)


def run_rung(cmd):
    """Run league_train.py in the caller's working directory (SPEC §19.10.4: relative paths in forwarded
    flags are the caller's), echoing its output; (return code, its last-line JSON summary or None)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
    last = None
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.strip():
            last = line.strip()
    rc = p.wait()
    try:
        summary = json.loads(last) if last else None
    except ValueError:
        summary = None
    return rc, summary


def main():
    known, rest = make_parser().parse_known_args()
    for t in rest:
        if str(t).split("=", 1)[0] in RESERVED:
            raise SystemExit(f"stages.py: {t.split('=', 1)[0]} is set per rung by stages.py; do not pass it")
    if known.gate_window < 1:
        raise SystemExit("stages.py: --gate-window must be >= 1")
    try:
        rungs = parse_rungs(known.rungs)
    except ValueError as e:
        raise SystemExit(f"stages.py: --rungs: {e}")
    data_dir = os.path.abspath(known.data_dir)
    out_path = os.path.join(data_dir, f"stages_{known.run_prefix}.json")
    os.makedirs(data_dir, exist_ok=True)
    t0 = time.time()
    report = {"run_prefix": known.run_prefix, "rungs_spec": known.rungs, "gate_window": int(known.gate_window),
              "forwarded": list(rest), "rungs": [], "passed": False, "failed_rung": None}
    status, prev_model = 0, None
    for i, (anchor, budget, thr) in enumerate(rungs):
        run_id = f"{known.run_prefix}_{i}_{anchor_label(anchor)}"
        run_dir = os.path.join(data_dir, "league", run_id)
        common = ["--total-timesteps", str(budget), "--early-stop-score", repr(float(thr)),
                  "--early-stop-window", str(known.gate_window), "--data-dir", data_dir]
        if os.path.isfile(os.path.join(run_dir, "league_state.json")):
            cmd = [sys.executable, os.path.join(HERE, "league_train.py"), "--resume", run_dir] + common + list(rest)
        else:
            cmd = [sys.executable, os.path.join(HERE, "league_train.py"), "--anchors", anchor, "--self-play-frac", "0",
                   "--anchor-frac", "1.0", "--run-id", run_id] + common + list(rest)
            if prev_model:
                cmd += ["--init-from", prev_model]
        print(f"[stages] rung {i}: {anchor} budget {budget} gate {thr:g} over {known.gate_window} matches -> {run_dir}",
              flush=True)
        rc, summary = run_rung(cmd)
        row = {"rung": i, "anchor": anchor, "budget": int(budget), "threshold": float(thr), "run_dir": run_dir,
               "init_from": prev_model, "exit_code": int(rc)}
        if rc != 0:
            row.update(passed=False, steps=None, score=None, matches=0, reason=f"league_train.py exited {rc}")
            report["rungs"].append(row)
            report["failed_rung"], status = i, 2
            break
        score, matches = gate(run_dir, anchor, known.gate_window)
        models = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")))
        passed = matches >= known.gate_window and score is not None and score >= thr
        reason = None if passed else ("insufficient matches" if matches < known.gate_window else "score below threshold")
        row.update(steps=int((summary or {}).get("global_step", 0)), score=score, matches=int(matches), passed=passed,
                   early_stopped=bool((summary or {}).get("early_stopped", False)),
                   model=models[-1] if models else None, reason=reason)
        report["rungs"].append(row)
        print(f"[stages] rung {i} {anchor}: score {score if score is not None else float('nan'):.3f} over {matches} "
              f"matches -> {'PASS' if passed else 'FAIL (' + reason + ')'}", flush=True)
        if not passed:
            report["failed_rung"], status = i, 1
            break
        prev_model = row["model"]
    report["passed"] = status == 0
    report["elapsed_s"] = round(time.time() - t0, 2)
    line = json.dumps(json_safe(report), allow_nan=False)
    with open(out_path, "w") as f:
        f.write(line + "\n")
    print(f"[stages] wrote {out_path}", flush=True)
    print(line)
    sys.exit(status)


if __name__ == "__main__":
    main()
