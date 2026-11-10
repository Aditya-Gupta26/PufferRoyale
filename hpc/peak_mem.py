"""Run a PufferRoyale script unchanged and report its peak GPU memory (docs/HPC_HANDOFF.md rule 7:
measure torch.cuda.max_memory_allocated before choosing minibatch sizes).

    run python hpc/peak_mem.py scripts/league_train.py --device cuda ...      (inside the container)

At interpreter exit (normal end, sys.exit, an error or SIGTERM) it prints one line to stderr:
    [peak_mem] {"script": ..., "max_memory_allocated": bytes, "max_memory_reserved": bytes, "device": ...}
and, if $PR_PEAK_MEM_JSON is set, writes the same JSON there. The script runs as __main__ via runpy,
so it behaves exactly as `python <script> <args>` (use only with in-process vectorisation: a
multiprocessing child would not report its own memory).
"""
import atexit
import json
import os
import runpy
import signal
import sys


def report(script):
    try:
        import torch
        if not torch.cuda.is_available():
            return
        rec = {"script": script, "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
               "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
               "device": torch.cuda.get_device_name(0)}
    except Exception as e:  # noqa: BLE001 -- reporting must never mask the script's own exit
        rec = {"script": script, "error": repr(e)}
    print("[peak_mem] " + json.dumps(rec), file=sys.stderr, flush=True)
    out = os.environ.get("PR_PEAK_MEM_JSON")
    if out:
        with open(out, "w") as f:
            json.dump(rec, f)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python hpc/peak_mem.py <script.py> [args...]")
    script = sys.argv[1]
    atexit.register(report, script)
    # SIGTERM (scancel, or hpc/calibrate.py stopping a measurement) normally skips atexit: turn it into
    # a SystemExit in the main thread so the script's own error handling runs and the report is written
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
