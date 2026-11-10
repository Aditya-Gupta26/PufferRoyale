#!/usr/bin/env python3
"""Branch a league run at one of its snapshots into a NEW, self-contained run directory that hpc/league.sbatch
resumes (2026-10-20, the stage-4 collapse ablations). The branch reproduces the source run's state at that
snapshot as closely as the saved files allow:

  learner.pt          the snapshot's weights (a copy)
  mmd_reference.pt    the same weights (the magnet is refreshed every mmd_ref_interval epochs; a copy)
  snap_<e>.pt         COPIES of the source's 64 most recent snapshots up to and including this one, which form
                      the branch's opponent pool (the source's pool at that epoch; snapshot_interval spacing)
  trainer_state.pt    epoch / global_step of the snapshot, so every step-based schedule (card floor fade,
                      shaping offset, prioritised-replay beta) continues exactly; the optimizer's param groups
                      from the source with an EMPTY state (Adam's moments are not saved with snapshots)
  league_state.json   the source's args and ini overrides with --set changes, the pool above with fresh stats
                      (PFSP p = 0.5 until played) and a fresh RNG; no wandb id (a new W&B run)
  BRANCH.json         the source, snapshot, overrides and the sha256 of every copied file

Nothing in the source run directory is written. The branch never refers to the source's files (copies only),
so later changes to the source cannot reach it. Refuses to overwrite an existing directory.

    python hpc/branch_run.py --source S4_C_s0 --at 1.55 --run-id S4x_nommd_s0 --set mmd_coef=0 --set mmd_ref_interval=0
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_set(items, args):
    """--set key=value with the type of the source arg (bool / int / float / str; 'none' -> None)."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set {it!r}: use key=value")
        k, v = it.split("=", 1)
        if k not in args:
            raise SystemExit(f"--set {k}: not an arg of the source run ({sorted(args)})")
        old = args[k]
        if v.lower() == "none":
            val = None
        elif isinstance(old, bool):
            val = v.lower() in ("1", "true", "yes")
        elif isinstance(old, int):
            val = int(v)
        elif isinstance(old, float) or old is None:
            val = float(v)
        else:
            val = v
        out[k] = val
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="source run id (a directory under DATA_DIR/league)")
    ap.add_argument("--at", type=float, required=True, help="branch point, billions of learner steps")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--set", action="append", default=[], help="key=value: change a saved arg of the run")
    ap.add_argument("--ini-override", action="append", default=[],
                    help="extra --section.key=value ini override stored with the run (repeatable)")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", os.path.join(ROOT, "experiments")))
    a = ap.parse_args()

    import numpy as np
    import torch
    from pufferroyale.league import OpponentPool

    src = os.path.join(a.data_dir, "league", a.source)
    dst = os.path.join(a.data_dir, "league", a.run_id)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists: refusing to overwrite")
    st = json.load(open(os.path.join(src, "league_state.json")))
    args = dict(st["args"])
    batch, interval, k = int(args["batch_size"]), int(args["snapshot_interval"]), int(args["max_snapshots"])
    e = int(round(a.at * 1e9 / batch / interval)) * interval
    pool_eps = [x for x in range(e - interval * (k - 1), e + 1, interval) if x > 0]
    missing = [x for x in pool_eps if not os.path.isfile(os.path.join(src, f"snap_{x}.pt"))]
    if missing:
        raise SystemExit(f"{a.source}: snapshots missing for epochs {missing[:5]}...")
    changes = parse_set(a.set, args)
    args.update(changes)
    args["run_id"] = a.run_id
    args["init_from"] = None
    ts_src = torch.load(os.path.join(src, "trainer_state.pt"), map_location="cpu", weights_only=False)
    groups = ts_src["optimizer_state_dict"]["param_groups"]

    os.makedirs(dst)
    copied = {}
    try:
        for x in pool_eps:
            name = f"snap_{x}.pt"
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
            copied[name] = sha256(os.path.join(dst, name))
            assert copied[name] == sha256(os.path.join(src, name)), name
        start = os.path.join(src, f"snap_{e}.pt")
        for name in ("learner.pt", "mmd_reference.pt"):
            shutil.copy2(start, os.path.join(dst, name))
            copied[name] = sha256(os.path.join(dst, name))
        sd = torch.load(os.path.join(dst, "learner.pt"), map_location="cpu", weights_only=True)
        sd0 = torch.load(start, map_location="cpu", weights_only=True)
        assert sd.keys() == sd0.keys() and all(torch.equal(sd[n], sd0[n]) for n in sd)
        torch.save({"optimizer_state_dict": {"state": {}, "param_groups": groups}, "global_step": e * batch,
                    "epoch": e, "run_id": a.run_id, "torch_rng": None}, os.path.join(dst, "trainer_state.pt"))
        pool = OpponentPool(anchors=(), max_snapshots=k, pfsp=args["pfsp"], pfsp_eps=float(args["pfsp_eps"]),
                            self_play_frac=float(args["self_play_frac"]), anchor_frac=float(args["anchor_frac"]),
                            seed=int(args["seed"]))
        for x in pool_eps:
            pool.add_snapshot(f"snap_{x}.pt")                      # relative: resolved inside the branch dir
        ps = pool.state_dict()
        ps["stats"] = {}
        state = {"version": 2, "run_id": a.run_id, "args": args,
                 "ini_overrides": list(st.get("ini_overrides", [])) + list(a.ini_override),
                 "epoch": e, "global_step": e * batch, "pool": ps, "rng": {}, "learner": "learner.pt",
                 "snapshot_files": [f"snap_{x}.pt" for x in pool_eps], "episodes": 0,
                 "early_stop": {"recent": {}}, "early_stopped": False}
        json.dump(state, open(os.path.join(dst, "league_state.json"), "w"), indent=1)
        manifest = {"source": src, "source_snapshot": f"snap_{e}.pt", "epoch": e, "global_step": e * batch,
                    "set": changes, "ini_override": a.ini_override, "pool_epochs": pool_eps,
                    "optimizer": "param groups of the source, empty state (fresh Adam moments)",
                    "sha256": copied}
        json.dump(manifest, open(os.path.join(dst, "BRANCH.json"), "w"), indent=1)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    print(f"{dst}: branch of {a.source} at snap_{e}.pt ({e * batch:,} steps); pool {len(pool_eps)} snapshots "
          f"({pool_eps[0]}..{pool_eps[-1]}); set {changes}; ini {a.ini_override}")


if __name__ == "__main__":
    main()
