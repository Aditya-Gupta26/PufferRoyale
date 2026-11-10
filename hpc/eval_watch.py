#!/usr/bin/env python3
"""Evaluation sidecar for a league run (DECISIONS D91; the user's request of 2026-10-04: a wandb "eval"
section with the learned policy vs the heuristic, random and noop bots every ~100 epochs).

scripts/league_train.py --eval-export-interval N writes the learner's weights every N epochs to
<run>/eval_ckpt/policy_<epoch>.pt (+ policy_<epoch>.json = {epoch, global_step}, written last). This
process, started next to the training by hpc/league.sbatch (EVAL_EVERY=N), evaluates each export with
scripts/eval.py on the CPU -- one process per bot, in parallel, sampled actions, both seats, the same
--seed at every point (common random numbers make the curves smoother) -- and logs

    eval/vs_heuristic, eval/vs_random, eval/vs_noop        learner score (wins + draws/2, both seats)
    eval_detail/<bot>_seat0|seat1, eval_detail/<bot>_lo|hi  per-seat scores and the 95% Wilson interval

against global_step to a wandb run "<run_id>-eval" in the training run's group (job_type "eval"), and
every point to <run>/eval_history.jsonl. Evaluation only: nothing here reaches the training, and the bots
stay unseen by the stage-3 learners. If it falls behind by `--max-backlog` exports, it jumps to the
newest one (the skipped epochs are recorded).

Only committed points are recorded: a result is held until the run's league_state.json has saved an epoch
>= the export's (or the training has finished), and the export file must still be the one evaluated (an
export is identified by epoch + the .json's mtime). A requeued run restarts at its last save and re-exports
the epochs after it; evaluations of the discarded trajectory are therefore never recorded (review finding
#7, 2026-10-04), and history lines beyond the saved epoch are dropped at start-up. It stops after
<run>/eval_ckpt/DONE appears and every remaining export is done, or on SIGTERM (which also terminates the
eval.py processes). Restarted it skips the epochs already recorded and resumes the same wandb run
(<run>/eval_ckpt/wandb_id); a wandb error disables wandb for the rest of the job, never the evaluation.

    python hpc/eval_watch.py RUN_DIR [--episodes 50] [--bots heuristic random noop] [--wandb --wandb-group G]

Stage 4 (D99), both optional:
  --decks NAME=DECKSET ...   the bots are played on each deck set instead (scripts/eval.py --decks: every deck
                             a mirror, --episodes per seat and deck, the pooled rows), logged as
                             eval/vs_<bot>_<NAME> (e.g. pool=hog26;giant;...  heldout=xbow;royal_hogs;random:50:S)
  --ckpt NAME=PATH ...       the export also plays these frozen checkpoints (scripts/tournament.py, --ckpt-matches
                             per opponent, half per seat, the --ckpt-deck mirror, sampled actions, CPU), logged as
                             eval/vs_<NAME>; e.g. the stage-3 Hog 2.6 specialists (the D93 specialist check)
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = False
CHILDREN = []          # running eval.py processes (terminated on SIGTERM)


def _stop(signum, frame):
    global STOP
    STOP = True
    for p in list(CHILDREN):
        try:
            p.terminate()
        except OSError:
            pass


def saved_epoch(run_dir):
    """The epoch of the run's last save (league_state.json), -1 when there is none yet."""
    try:
        with open(os.path.join(run_dir, "league_state.json")) as f:
            return int(json.load(f)["epoch"])
    except (OSError, ValueError, KeyError):
        return -1


def append_line(path, rec):
    """Append one JSON line, first ending a line a killed writer may have left unterminated."""
    with open(path, "a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((json.dumps(rec) + "\n").encode())


def wilson(k, n, z=1.959964):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def exports(d):
    """[(epoch, mtime_ns of the .json, json path, pt path)] of the complete exports in d, oldest first."""
    out = []
    for n in os.listdir(d):
        if n.startswith("policy_") and n.endswith(".json"):
            js, pt = os.path.join(d, n), os.path.join(d, n[:-5] + ".pt")
            try:
                out.append((int(n[7:-5]), os.stat(js).st_mtime_ns, js, pt))
            except (ValueError, OSError):
                continue
    return sorted(e for e in out if os.path.isfile(e[3]))


def _summary(seat_wdl):
    """{score, lo, hi, matches, seat0, seat1} from {seat: (wins, draws, matches)}."""
    w = sum(v[0] for v in seat_wdl.values())
    dr = sum(v[1] for v in seat_wdl.values())
    n = sum(v[2] for v in seat_wdl.values())
    k = w + 0.5 * dr
    lo, hi = wilson(k, n)
    seat = {s: (v[0] + 0.5 * v[1]) / v[2] if v[2] else None for s, v in seat_wdl.items()}
    return {"score": k / n, "lo": lo, "hi": hi, "matches": n, "seat0": seat.get(0), "seat1": seat.get(1)}


def _parse_eval(js, bot, pooled):
    rows = [r for r in json.load(open(js))["results"] if r["opponent"] == bot
            and (r.get("deck") == "pooled" if pooled else "deck" not in r)]
    return _summary({r["seat"]: (r["wins"], r["draws"], r["matches"]) for r in rows})


def _parse_tournament(js):
    """The export (agent 0) vs the checkpoint (agent 1): tournament.py puts agent 0 in seat 0 for the
    even-numbered matches of the pair and in seat 1 for the others; r is agent 0's result."""
    seat_wdl = {0: [0, 0, 0], 1: [0, 0, 0]}
    for m, (i, j, r) in enumerate(json.load(open(js))["results"]):
        assert (i, j) == (0, 1), (i, j)
        v = seat_wdl[m % 2]
        v[0] += r > 0
        v[1] += r == 0
        v[2] += 1
    return _summary(seat_wdl)


def evaluate(pt, bots, episodes, seed, threads, decks=(), ckpts=(), ckpt_matches=200, ckpt_deck="hog26"):
    """{key: {score, lo, hi, seat0, seat1, matches}}, one CPU process per item, all in parallel:
    key = bot (scripts/eval.py on the default deck), or <bot>_<name> per --decks set (pooled rows), and <name>
    per --ckpt opponent (scripts/tournament.py, the --ckpt-deck mirror)."""
    procs = {}
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads))
    jobs = []                                  # (key, cmd, json path, parser)
    with tempfile.TemporaryDirectory(prefix="eval_watch_", dir=os.environ.get("TMPDIR")) as tmp:
        for b in bots:
            for name, deckset in (decks or [(None, None)]):
                key = b if name is None else f"{b}_{name}"
                js = os.path.join(tmp, f"{key}.json")
                cmd = [sys.executable, os.path.join(ROOT, "scripts", "eval.py"), "--checkpoint", pt, "--bots", b,
                       "--seats", "0", "1", "--episodes", str(episodes), "--device", "cpu", "--seed", str(seed),
                       "--json", js] + ([] if name is None else ["--decks", deckset])
                jobs.append((key, cmd, js, lambda js, b=b, pooled=name is not None: _parse_eval(js, b, pooled)))
        for name, path in ckpts:
            js = os.path.join(tmp, f"ckpt_{name}.json")
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "tournament.py"), "--agents", pt, path,
                   "--decks", ckpt_deck, "--deck-mode", "mirror", "--matches", str(ckpt_matches), "--device", "cpu",
                   "--seed", str(seed), "--quiet", "--out", js]
            jobs.append((name, cmd, js, _parse_tournament))
        try:
            for key, cmd, js, parse in jobs:
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     text=True)
                CHILDREN.append(p)
                procs[key] = (p, js, parse)
            out = {}
            for key, (p, js, parse) in procs.items():
                _, err = p.communicate()
                if p.returncode != 0 or not os.path.isfile(js):
                    raise RuntimeError(f"evaluation {key} failed (exit {p.returncode}): {err.strip()[-400:]}")
                out[key] = parse(js)
        finally:                              # never leave eval.py processes behind
            for p, _, _ in procs.values():
                if p.poll() is None:
                    p.kill()
                    p.wait()
                if p in CHILDREN:
                    CHILDREN.remove(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--bots", nargs="+", default=["heuristic", "random", "noop"])
    ap.add_argument("--episodes", type=int, default=50, help="matches per seat and bot at every point (default 50)")
    ap.add_argument("--seed", type=int, default=0, help="eval.py --seed, the same at every point (default 0)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per eval.py process (default 1)")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between directory scans")
    ap.add_argument("--max-backlog", type=int, default=3, help="jump to the newest export beyond this many pending")
    ap.add_argument("--wandb", action="store_true", help="log to wandb (WANDB_* from the environment)")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "pufferroyale"))
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--decks", nargs="+", default=[], metavar="NAME=DECKSET",
                    help="play the bots on these deck sets (eval.py --decks; logged as eval/vs_<bot>_<NAME>)")
    ap.add_argument("--ckpt", nargs="+", default=[], metavar="NAME=PATH",
                    help="also play these frozen checkpoints (tournament.py; logged as eval/vs_<NAME>)")
    ap.add_argument("--ckpt-matches", type=int, default=200, help="matches per --ckpt opponent, half per seat")
    ap.add_argument("--ckpt-deck", default="hog26", help="deck preset of the --ckpt mirror (default hog26)")
    a = ap.parse_args()

    def pairs(items, what):
        out = []
        for it in items:
            name, sep, val = it.partition("=")
            if not sep or not name or not val or not all(c.isalnum() or c in "_-" for c in name):
                raise SystemExit(f"--{what} wants NAME=VALUE with NAME of letters, digits, _ or -; got {it!r}")
            out.append((name, val))
        return out
    decks, ckpts = pairs(a.decks, "decks"), pairs(a.ckpt, "ckpt")
    for _, path in ckpts:
        if not os.path.isfile(path):
            raise SystemExit(f"--ckpt: no such file {path}")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    run_dir = os.path.abspath(a.run_dir)
    run_id = os.path.basename(run_dir)
    d = os.path.join(run_dir, "eval_ckpt")
    hist = os.path.join(run_dir, "eval_history.jsonl")
    # recorded points; lines beyond the run's saved epoch (a discarded trajectory) are dropped
    done_epochs, keep = set(), []
    e0 = saved_epoch(run_dir)
    if os.path.isfile(hist):
        for line in open(hist):
            try:
                r = json.loads(line)
                ep = int(r["epoch"])
            except (ValueError, KeyError):
                continue
            if e0 >= 0 and ep > e0:
                continue
            keep.append(line if line.endswith("\n") else line + "\n")
            done_epochs.add(ep)
        with open(hist + ".tmp", "w") as f:
            f.writelines(keep)
        os.replace(hist + ".tmp", hist)
    wb = None
    if a.wandb:
        try:
            import wandb
            os.makedirs(d, exist_ok=True)
            idf = os.path.join(d, "wandb_id")
            wid = open(idf).read().strip() if os.path.isfile(idf) else None
            wb = wandb.init(project=a.wandb_project, entity=os.environ.get("WANDB_ENTITY") or None,
                            group=a.wandb_group, name=f"{run_id}-eval", job_type="eval", id=wid, resume="allow",
                            config={"run_id": run_id, "run_dir": run_dir, "bots": a.bots,
                                    "episodes_per_seat": a.episodes, "seed": a.seed, "decks": dict(decks),
                                    "ckpt": dict(ckpts), "ckpt_matches": a.ckpt_matches,
                                    "ckpt_deck": a.ckpt_deck})
            if not wid:
                with open(idf, "w") as f:
                    f.write(wb.id)
            wb.define_metric("global_step")
            wb.define_metric("eval/*", step_metric="global_step")
            wb.define_metric("eval_detail/*", step_metric="global_step")
        except Exception as e:  # noqa: BLE001 -- evaluation goes on without wandb
            print(f"[eval_watch] wandb unavailable ({e!r}): eval_history.jsonl only", file=sys.stderr, flush=True)
            wb = None
    print(f"[eval_watch] {run_dir}: bots {a.bots}, {a.episodes} matches per seat"
          + (f" on deck sets {[n for n, _ in decks]}" if decks else "")
          + (f", checkpoints {[n for n, _ in ckpts]} ({a.ckpt_matches} matches, {a.ckpt_deck} mirror)" if ckpts else "")
          + ", "
          f"{len(done_epochs)} points already recorded (saved epoch {e0})", flush=True)

    evaluated = set()            # (epoch, mtime) keys evaluated in this process
    staged = []                  # [(epoch, mtime, json path, record)] waiting for the run to save past them

    def commit(final):
        nonlocal wb
        se = saved_epoch(run_dir)
        for item in sorted(staged, key=lambda t: t[0]):
            ep, mt, js, rec = item
            if not final and ep > se:
                continue
            staged.remove(item)
            try:
                current = os.stat(js).st_mtime_ns == mt
            except OSError:
                current = False
            if not current or ep in done_epochs:     # re-exported / removed: the evaluated weights are gone
                continue
            append_line(hist, rec)
            done_epochs.add(ep)
            print(f"[eval_watch] epoch {ep} step {rec['global_step']}: "
                  + " ".join(f"{b}={r['score']:.3f}" for b, r in rec["results"].items())
                  + f" ({rec['seconds']} s" + (f", skipped {len(rec['skipped_epochs'])}" if rec["skipped_epochs"] else "")
                  + ")", flush=True)
            if wb is not None:
                log = {"global_step": rec["global_step"]}
                for b, r in rec["results"].items():
                    log[f"eval/vs_{b}"] = r["score"]
                    log[f"eval_detail/{b}_lo"] = r["lo"]
                    log[f"eval_detail/{b}_hi"] = r["hi"]
                    if r["seat0"] is not None:
                        log[f"eval_detail/{b}_seat0"] = r["seat0"]
                    if r["seat1"] is not None:
                        log[f"eval_detail/{b}_seat1"] = r["seat1"]
                try:
                    wb.log(log)
                except Exception as e:  # noqa: BLE001
                    print(f"[eval_watch] wandb log failed ({e!r}): wandb off", file=sys.stderr, flush=True)
                    wb = None

    while not STOP:
        finishing = os.path.isfile(os.path.join(d, "DONE"))
        if os.path.isdir(d):
            pending = [e for e in exports(d) if e[0] not in done_epochs and (e[0], e[1]) not in evaluated]
        else:
            pending = []
        if not pending:
            commit(final=finishing)
            if finishing and not staged:
                break
            if finishing:                       # staged results whose exports vanished: drop them
                staged.clear()
                break
            time.sleep(a.poll)
            continue
        skipped = []
        if len(pending) > a.max_backlog:
            skipped = [e[0] for e in pending[:-1]]
            for e in pending[:-1]:
                evaluated.add((e[0], e[1]))
            pending = pending[-1:]
        epoch, mt, js, pt = pending[0]
        evaluated.add((epoch, mt))
        try:
            meta = json.load(open(js))
            t0 = time.time()
            res = evaluate(pt, a.bots, a.episodes, a.seed, a.threads, decks, ckpts, a.ckpt_matches, a.ckpt_deck)
        except FileNotFoundError:          # pruned or re-exported meanwhile: take the next one
            continue
        except Exception as e:  # noqa: BLE001 -- never take the training down; log and move on
            if not STOP:
                print(f"[eval_watch] epoch {epoch}: {e}", file=sys.stderr, flush=True)
            continue
        rec = {"epoch": int(meta["epoch"]), "global_step": int(meta["global_step"]), "time": time.time(),
               "seconds": round(time.time() - t0, 1), "results": res, "skipped_epochs": skipped}
        staged.append((epoch, mt, js, rec))
        commit(final=False)
    if wb is not None:
        try:
            wb.finish()
        except Exception:  # noqa: BLE001
            pass
    print("[eval_watch] stopped", flush=True)


if __name__ == "__main__":
    main()
