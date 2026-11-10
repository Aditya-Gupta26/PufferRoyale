#!/usr/bin/env python3
"""Start a league run, or resume it if its run directory already holds one.

    python hpc/league_run.py <scripts/league_train.py arguments, incl. --run-id R [--data-dir D]>

hpc/pack.sbatch replays its command lines verbatim after a requeue or a re-submission; a fresh
`league_train.py --run-id R` then refuses to start ("already holds a run"). This wrapper checks
<D>/league/<R>/league_state.json (D = --data-dir, default experiments): if it exists the run is resumed
with the same arguments minus those only a new run accepts (--init-from, --run-id), so pool settings,
--rnn and the --section.key overrides behave exactly as league_train.py's own --resume documents;
otherwise league_train.py starts the run with the arguments unchanged. Run from the checkout (the
working directory), inside our image.
"""
import os
import sys

ROOT = os.getcwd()
NEW_RUN_ONLY = ("--init-from", "--run-id")        # flags that take one value and that --resume rejects / ignores


def value(args, flag, default=None):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def without(args, flags):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        if any(a.startswith(f + "=") for f in flags):
            continue
        out.append(a)
    return out


def main():
    args = sys.argv[1:]
    run_id = value(args, "--run-id")
    if not run_id:
        raise SystemExit("hpc/league_run.py needs --run-id (the run directory to start or resume)")
    run_dir = os.path.join(value(args, "--data-dir", "experiments"), "league", run_id)
    script = os.path.join(ROOT, "scripts", "league_train.py")
    if os.path.isfile(os.path.join(run_dir, "league_state.json")):
        cmd = [script, "--resume", run_dir] + without(args, NEW_RUN_ONLY)
        print(f"[league_run] resuming {run_dir}", flush=True)
    else:
        cmd = [script] + args
        print(f"[league_run] starting {run_dir}", flush=True)
    os.execv(sys.executable, [sys.executable] + cmd)


if __name__ == "__main__":
    main()
