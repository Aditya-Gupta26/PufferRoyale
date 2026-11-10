#!/usr/bin/env python3
"""Stage-4 collapse diagnostics, GPU part (2026-10-16; see hpc/collapse_diag.py). Does the league's training
path play its frozen snapshots exactly as a plain evaluation does, and what does the D100 card floor cost the
learner against them? One learner snapshot plays a pool of earlier snapshots (uniform sampling, no self-play,
no anchors) in the training env (the run's own config.json env settings: decks, rewards, frame skip) through:

  fast        FastLeagueVecEnv on the GPU exactly as league_train builds it with --fast-opponents
              --cuda-graphs --env-threads (gpu_obs, gpu_sample, vmap bank of the snapshots, CUDA graphs; the
              learner's forward_eval replayed from its own graphs)
  fast_floor  the same with the learner's card floor on (eps of the run's schedule at the learner's step), the
              asymmetry of training: the learner is floored, the snapshots never are
  plain       LeagueVecEnv on the CPU (eager per-checkpoint forwards, CPU sampling)

and reports the learner's score per snapshot and seat. In the fast arm it also checks, at sampled steps, that
the vmap bank's logits for every opponent row equal the eager forward of the snapshot that row is assigned
(and differ from a wrong snapshot's), and that the learner's graph replay equals its eager forward.

    python hpc/collapse_gpu_check.py --run S4_C_s0 --learner 1.5 --opponents 1.0 1.1 1.2 1.3 1.4 \\
        --out $DATA_DIR/eval/collapse_diag/gpu_check_S4_C_s0.json
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hpc"))

from collapse_diag import snapshot  # noqa: E402


def env_kwargs(run_dir):
    env = json.load(open(os.path.join(run_dir, "config.json")))["env"]
    return {k: v for k, v in env.items() if not k.endswith("_decks")}


def _legal_max_diff(a, b):
    import torch
    legal = b > torch.finfo(torch.float32).min
    return float((a.float() - b.float()).abs()[legal].max().item())


def bank_check(lv):
    """Bank logits vs the eager forward of the assigned snapshot (and of a wrong one) for every opponent row."""
    import torch
    n = lv.num_envs
    opp_rows = 2 * np.arange(n) + (1 - lv.seats)
    groups = collections.OrderedDict()
    for i, spec in enumerate(lv.opponents):
        groups.setdefault(spec, []).append(i)
    vm = [(spec, np.asarray(idx)) for spec, idx in groups.items()]
    with torch.no_grad():
        for spec, _ in vm:
            lv._opponent_policy(spec)
        got = lv._vmap_logits(vm, opp_rows)
        if got is None:
            return {"bank": False}
        _, logits = got[0]
        specs = list(lv._ckpt_cache)
        out = {"bank": True, "K": lv._vbank["K"], "rows": 0, "max_abs_diff": 0.0, "min_wrong_diff": float("inf")}
        pos = 0
        for spec, ix in vm:
            pol, _, dev = lv._opponent_policy(spec)
            x = lv._opp_x(opp_rows[ix], dev)
            ref, _ = type(pol).forward_eval(pol, x, {})
            mine = logits[pos:pos + len(ix)]
            pos += len(ix)
            out["max_abs_diff"] = max(out["max_abs_diff"], _legal_max_diff(mine, ref))
            other = next((s for s in specs if s != spec), None)
            if other is not None:
                wp = lv._ckpt_cache[other][0]
                wrong, _ = type(wp).forward_eval(wp, x, {})
                out["min_wrong_diff"] = min(out["min_wrong_diff"], _legal_max_diff(mine, wrong))
            out["rows"] += len(ix)
    return out


def play_arm(arm, a, learner_path, opp_paths, kw, floor):
    import torch
    from pufferroyale.league import LeagueVecEnv, OpponentPool, load_policy
    fast = arm.startswith("fast")
    device = a.device if fast else "cpu"
    pool = OpponentPool(anchors=(), max_snapshots=64, pfsp="uniform", self_play_frac=0.0, anchor_frac=0.0,
                        seed=a.seed)
    for p in opp_paths:
        pool.add_snapshot(p)
    if fast:
        from pufferroyale.fast_league import FastLeagueVecEnv, install_learner_graph
        lv = FastLeagueVecEnv(pool, num_envs=a.num_envs, seed=a.seed, device=device, env_threads=a.env_threads,
                              gpu_obs=True, gpu_sample=True, vmap=True, cuda_graphs=True, **kw)
    else:
        lv = LeagueVecEnv(pool, num_envs=a.num_envs, seed=a.seed, device=device, **kw)
    learner, _ = load_policy(learner_path, device)
    if floor > 0:
        learner.enable_card_floor(floor)
    if fast:
        install_learner_graph(learner)
    lv.set_policy(learner)
    gen = torch.Generator(device=device).manual_seed(a.seed + 7)
    episodes = a.episodes if fast else a.cpu_episodes
    obs, _ = lv.reset(a.seed)
    checks, learner_diff, steps, t0 = [], 0.0, 0, time.time()
    try:
        while lv.episodes < episodes:
            with torch.no_grad():
                x = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
                x = x.to(device)
                logits, _ = learner.forward_eval(x, {})
                if fast and steps % 101 == 7 and len(checks) < 20:
                    eager, _ = type(learner).forward_eval(learner, x, {})
                    learner_diff = max(learner_diff, _legal_max_diff(logits, eager))
                    checks.append(bank_check(lv))
                act = torch.multinomial(torch.softmax(logits.float(), dim=-1), 1, generator=gen).squeeze(-1)
            lv.send(act.to(torch.int32).cpu().numpy())
            obs = lv.recv()[0]
            steps += 1
        res = lv.drain_results()
    finally:
        lv.close()
    per = {}
    for spec, r, seat in res:
        d = per.setdefault(spec[len("ckpt:"):], {"n": 0, "pts": 0.0, "seat": {"0": [0, 0.0], "1": [0, 0.0]}})
        d["n"] += 1
        d["pts"] += (r + 1) / 2
        d["seat"][str(seat)][0] += 1
        d["seat"][str(seat)][1] += (r + 1) / 2
    n = sum(d["n"] for d in per.values())
    out = {"arm": arm, "device": str(device), "floor": floor, "episodes": n, "steps": steps,
           "seconds": round(time.time() - t0, 1),
           "score": sum(d["pts"] for d in per.values()) / max(1, n),
           "per_opponent": {p: {"n": d["n"], "score": d["pts"] / d["n"],
                                "seat0": d["seat"]["0"][1] / max(1, d["seat"]["0"][0]),
                                "seat1": d["seat"]["1"][1] / max(1, d["seat"]["1"][0])} for p, d in per.items()}}
    if fast:
        out["learner_graph_max_abs_diff"] = learner_diff
        out["bank_checks"] = checks
    print(f"[gpu_check] {arm}: score {out['score']:.3f} over {n} episodes ({out['seconds']} s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--learner", type=float, required=True, help="learner snapshot, billions of steps")
    ap.add_argument("--opponents", type=float, nargs="+", required=True, help="opponent snapshots, billions")
    ap.add_argument("--arms", nargs="+", default=["fast", "fast_floor", "plain"])
    ap.add_argument("--episodes", type=int, default=6000, help="per GPU arm")
    ap.add_argument("--cpu-episodes", type=int, default=3000, help="for the plain (CPU) arm")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--env-threads", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", os.path.join(ROOT, "experiments")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_dir = os.path.join(a.data_dir, "league", a.run)
    lg = json.load(open(os.path.join(run_dir, "config.json")))["league"]
    name, lpath, _, lstep = snapshot(a.data_dir, a.run, a.learner)
    opps = [snapshot(a.data_dir, a.run, s) for s in a.opponents]
    eps0, n = float(lg.get("card_floor") or 0.0), int(lg.get("card_floor_steps") or 0)
    floor = eps0 * max(0.0, 1.0 - lstep / n) if (eps0 > 0 and n > 0) else eps0
    kw = env_kwargs(run_dir)
    out = {"run": a.run, "learner": {"name": name, "path": lpath, "global_step": lstep},
           "opponents": [{"name": o[0], "path": o[1], "global_step": o[3]} for o in opps], "floor_at_learner": floor,
           "num_envs": a.num_envs, "seed": a.seed, "arms": []}
    for arm in a.arms:
        out["arms"].append(play_arm(arm, a, lpath, [o[1] for o in opps], kw, floor if arm == "fast_floor" else 0.0))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({r["arm"]: round(r["score"], 4) for r in out["arms"]}))


if __name__ == "__main__":
    main()
