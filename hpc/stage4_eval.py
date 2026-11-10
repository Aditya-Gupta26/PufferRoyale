#!/usr/bin/env python3
"""Stage-4 evaluation at an analysis point (EXPERIMENTS "Stage-4 pre-registration", items 1-4; TRAINING_PLAN §8.4),
split into independent single-CPU tasks so a Slurm array (hpc/stage4_eval.sbatch) finishes in about an hour:

  gen   scripts/eval.py: each run's checkpoint vs the heuristic and random bots on the pool decks (72 matches per seat
        and deck = 504 per seat) and on the held-out decks (10 per seat and deck over 52 decks = 520 per seat); every
        deck a mirror; pooled per deck set (the pre-registered >= 500 per seat and deck set)
  meta  each checkpoint's deck meta-game over the 9 presets (the 7 pool decks + xbow + royal_hogs): 200 self-play
        matches per deck pair -- exactly the matches and seeds of pufferroyale.metagame.deck_metagame, one task per
        pair; deck-vs-deck matrix and its Nash mixture
  spec  each checkpoint vs the stage-3 Hog 2.6 specialists S3_A_s1 and S3_C_s0 (best final per method by E2 Elo) in
        the Hog 2.6 mirror, 500 matches each (5 tasks of 100), half per seat (the D93 specialist check)
  rr    the four runs' checkpoints in a round robin on the 9 presets, each deck a mirror, 200 matches per pair and
        deck, half per seat; payoff, Elo, Nash, and C against A pooled over the four cross pairs

Policies sample their actions (SPEC §19.11), on the CPU, without the D100 card floor (checkpoints never carry it).
The checkpoint of a run at a point is the snapshot nearest the step count (2B: snap_122112 = 2.0007B steps, 5B:
snap_305280), or the run's final model for --point 10B.

    python hpc/stage4_eval.py plan --point 2B --out DIR [--runs S4_C_s0 ...]   # DIR/plan.json, DIR/tasks.jsonl
    python hpc/stage4_eval.py run DIR --task 17                                 # one task -> DIR/out/<id>.json
    python hpc/stage4_eval.py merge DIR                                         # DIR/summary.json, DIR/summary.md

A task whose output exists is skipped, so re-running the array resumes it. Per-match seeds:
meta = match_seed(seed, i, j, m) as deck_metagame; spec = match_seed(seed, 1000 + s, m); rr = match_seed(seed,
2000 + 10 i + j, d, m); for an even m the first agent plays team 0, else team 1 (results are always the first
agent's: +1 / 0 / -1).
"""
import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RUNS = ("S4_C_s0", "S4_A_s0", "S4_C_s1", "S4_A_s1")
POOL = "hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge"
HELDOUT = "xbow;royal_hogs;random:50:20261003"
PRESETS = "hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge;xbow;royal_hogs"
POINTS = {"2B": 2_000_000_000, "5B": 5_000_000_000, "10B": 10_000_000_000}


def _deck_list(spec):
    return [d.strip() for d in spec.split(";") if d.strip()]


def checkpoint(run_dir, steps, final):
    """(path, epoch, global_step) of the run's checkpoint for an analysis point, or None if it is not there yet."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))["league"]
    batch, interval = int(cfg["batch_size"]), int(cfg["snapshot_interval"])
    if final:
        models = sorted(n for n in os.listdir(run_dir) if re.fullmatch(r"model_\d+\.pt", n))
        if not models:
            return None
        e = int(models[-1][6:-3])
        return os.path.join(run_dir, models[-1]), e, e * batch
    e = int(round(steps / batch / interval)) * interval
    p = os.path.join(run_dir, f"snap_{e}.pt")
    return (p, e, e * batch) if os.path.isfile(p) else None


def plan(a):
    data = a.data_dir
    final = a.point == "10B"
    steps = POINTS[a.point]
    ckpts = {}
    for r in a.runs:
        c = checkpoint(os.path.join(data, "league", r), steps, final)
        if c is None:
            raise SystemExit(f"{r}: no checkpoint for {a.point} yet")
        ckpts[r] = {"path": c[0], "epoch": c[1], "global_step": c[2]}
    specs = dict(kv.split("=", 1) for kv in a.specialists)
    presets = _deck_list(a.presets)
    tasks = []

    def add(**t):
        t["id"] = len(tasks)
        tasks.append(t)
    if "gen" in a.parts:
        for r in a.runs:
            for bot in ("heuristic", "random"):
                for name, deckset, eps in (("pool", a.pool, a.gen_pool_episodes), ("heldout", a.heldout,
                                                                                   a.gen_heldout_episodes)):
                    add(kind="gen", run=r, bot=bot, deckset_name=name, deckset=deckset, episodes=eps)
    if "meta" in a.parts:
        for r in a.runs:
            for i, j in itertools.combinations(range(len(presets)), 2):
                add(kind="meta", run=r, i=i, j=j, m0=0, m1=a.meta_matches)
    if "spec" in a.parts:
        for r in a.runs:
            for s, name in enumerate(sorted(specs)):
                for m0 in range(0, a.spec_matches, a.spec_chunk):
                    add(kind="spec", run=r, spec=name, s=s, m0=m0, m1=min(a.spec_matches, m0 + a.spec_chunk))
    if "rr" in a.parts and len(a.runs) > 1:
        for (i, ri), (j, rj) in itertools.combinations(enumerate(a.runs), 2):
            for d in range(len(presets)):
                add(kind="rr", i=i, j=j, a=ri, b=rj, d=d, m0=0, m1=a.rr_matches)
    os.makedirs(os.path.join(a.out, "out"), exist_ok=True)
    meta = {"point": a.point, "runs": list(a.runs), "checkpoints": ckpts, "specialists": specs, "presets": presets,
            "pool": a.pool, "heldout": a.heldout, "seed": a.seed, "parts": list(a.parts), "n_tasks": len(tasks),
            "matches": {"gen_pool_episodes": a.gen_pool_episodes, "gen_heldout_episodes": a.gen_heldout_episodes,
                        "meta": a.meta_matches, "spec": a.spec_matches, "rr": a.rr_matches},
            "planned": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(meta, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
    with open(os.path.join(a.out, "tasks.jsonl"), "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"{a.out}: {len(tasks)} tasks ({', '.join(f'{k} {sum(t['kind'] == k for t in tasks)}' for k in a.parts)})")


def _play(agent_a, agent_b, deck, seeds):
    """[(m, r)]: for an even m agent_a plays team 0, else team 1; r is agent_a's result."""
    from pufferroyale.metagame import play_match
    out = []
    for m, s in seeds:
        r = play_match(agent_a, agent_b, deck, deck, s) if m % 2 == 0 else -play_match(agent_b, agent_a, deck, deck, s)
        out.append((m, int(r)))
    return out


def run_task(d, tid):
    out = os.path.join(d, "out", f"{tid:04d}.json")
    if os.path.exists(out):
        return
    meta = json.load(open(os.path.join(d, "plan.json")))
    t = [json.loads(ln) for ln in open(os.path.join(d, "tasks.jsonl"))][tid]
    from pufferroyale.metagame import match_seed
    seed, presets, ck = int(meta["seed"]), meta["presets"], meta["checkpoints"]
    t0 = time.time()
    if t["kind"] == "gen":
        js = out[:-5] + ".eval.json"
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "eval.py"), "--checkpoint", ck[t["run"]]["path"],
               "--bots", t["bot"], "--decks", t["deckset"], "--episodes", str(t["episodes"]), "--seed", str(seed),
               "--json", js]
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        rows = [r for r in json.load(open(js))["results"] if r["opponent"] == t["bot"] and r.get("deck") == "pooled"]
        res = {"seat": {str(r["seat"]): [r["wins"], r["draws"], r["matches"]] for r in rows}}
    elif t["kind"] == "meta":
        from pufferroyale.metagame import play_match
        agent, i, j = ck[t["run"]]["path"], t["i"], t["j"]
        rs = []
        for m in range(t["m0"], t["m1"]):                      # exactly deck_metagame's loop body
            s = match_seed(seed, i, j, m)
            if m % 2 == 0:
                r = play_match(agent, agent, presets[i], presets[j], s)
            else:
                r = -play_match(agent, agent, presets[j], presets[i], s)
            rs.append((m, int(r)))
        res = {"results": rs}
    elif t["kind"] == "spec":
        seeds = [(m, match_seed(seed, 1000 + t["s"], m)) for m in range(t["m0"], t["m1"])]
        res = {"results": _play(ck[t["run"]]["path"], meta["specialists"][t["spec"]], "hog26", seeds)}
    elif t["kind"] == "rr":
        seeds = [(m, match_seed(seed, 2000 + 10 * t["i"] + t["j"], t["d"], m)) for m in range(t["m0"], t["m1"])]
        res = {"results": _play(ck[t["a"]]["path"], ck[t["b"]]["path"], presets[t["d"]], seeds)}
    else:
        raise ValueError(t["kind"])
    res.update(task=t, seconds=round(time.time() - t0, 1))
    with open(out + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out + ".tmp", out)


def _score(wdl):
    w, dr, n = wdl
    return (w + 0.5 * dr) / n if n else float("nan")


def merge(d):
    import numpy as np
    from pufferroyale.metagame import elo, metagame_nash, payoff_matrix, wilson_interval
    meta = json.load(open(os.path.join(d, "plan.json")))
    tasks = [json.loads(ln) for ln in open(os.path.join(d, "tasks.jsonl"))]
    res = {}
    for t in tasks:
        p = os.path.join(d, "out", f"{t['id']:04d}.json")
        if os.path.exists(p):
            res[t["id"]] = json.load(open(p))
    missing = [t["id"] for t in tasks if t["id"] not in res]
    presets, runs = meta["presets"], meta["runs"]
    S = {"point": meta["point"], "checkpoints": meta["checkpoints"], "missing_tasks": missing}

    def ci(score, n):
        return [round(v, 4) for v in wilson_interval(score, n)]
    gen = {}
    for t in tasks:
        if t["kind"] != "gen" or t["id"] not in res:
            continue
        seat = res[t["id"]]["seat"]
        tot = [sum(seat[s][k] for s in seat) for k in range(3)]
        sc = _score(tot)
        gen.setdefault(t["run"], {}).setdefault(t["bot"], {})[t["deckset_name"]] = {
            "score": round(sc, 4), "ci": ci(sc, tot[2]), "matches": tot[2],
            "seat0": round(_score(seat["0"]), 4), "seat1": round(_score(seat["1"]), 4)}
    for r in gen:
        for bot, v in gen[r].items():
            if "pool" in v and "heldout" in v:
                p, h = v["pool"], v["heldout"]
                se = math.sqrt(p["score"] * (1 - p["score"]) / p["matches"] + h["score"] * (1 - h["score"]) / h["matches"])
                v["gap_pool_minus_heldout"] = {"value": round(p["score"] - h["score"], 4),
                                               "ci95": [round(p["score"] - h["score"] - 1.96 * se, 4),
                                                        round(p["score"] - h["score"] + 1.96 * se, 4)]}
    S["generalisation"] = gen
    metas = {}
    for r in runs:
        trip = [(t["i"], t["j"], rr) for t in tasks if t["kind"] == "meta" and t["run"] == r and t["id"] in res
                for _, rr in res[t["id"]]["results"]]
        if trip:
            P, G = payoff_matrix(trip, len(presets), return_counts=True)
            x, v = metagame_nash(P)
            metas[r] = {"P": np.round(P, 4).tolist(), "games": np.asarray(G).tolist(),
                        "nash": {presets[k]: round(float(x[k]), 4) for k in range(len(presets)) if x[k] > 1e-4},
                        "deck_mean_score": {presets[k]: round(float(np.mean([P[k][q] for q in range(len(presets))
                                                                             if q != k])), 4)
                                            for k in range(len(presets))}}
    S["deck_metagame"] = metas
    spec = {}
    for t in tasks:
        if t["kind"] != "spec" or t["id"] not in res:
            continue
        v = spec.setdefault(t["run"], {}).setdefault(t["spec"], {"0": [0, 0, 0], "1": [0, 0, 0]})
        for m, r in res[t["id"]]["results"]:
            s = v[str(m % 2)]
            s[0] += r > 0
            s[1] += r == 0
            s[2] += 1
    for r in spec:
        for name, v in list(spec[r].items()):
            tot = [v["0"][k] + v["1"][k] for k in range(3)]
            sc = _score(tot)
            spec[r][name] = {"score": round(sc, 4), "ci": ci(sc, tot[2]), "matches": tot[2],
                             "seat0": round(_score(v["0"]), 4), "seat1": round(_score(v["1"]), 4)}
    S["specialists"] = spec
    rr_trip = [(t["i"], t["j"], r) for t in tasks if t["kind"] == "rr" and t["id"] in res
               for _, r in res[t["id"]]["results"]]
    if rr_trip:
        P, G = payoff_matrix(rr_trip, len(runs), return_counts=True)
        x, _ = metagame_nash(P)
        ratings = elo(P, G)
        method = [r.split("_")[1] for r in runs]                     # S4_C_s0 -> C
        cross = [(i, j) for i, j in itertools.combinations(range(len(runs)), 2) if method[i] != method[j]]
        cvsa = []
        for t in tasks:
            if t["kind"] == "rr" and t["id"] in res and (t["i"], t["j"]) in cross:
                c_first = method[t["i"]] == "C"
                cvsa += [(r if c_first else -r) for _, r in res[t["id"]]["results"]]
        sc = sum(0.5 * (r + 1) for r in cvsa) / len(cvsa) if cvsa else float("nan")
        per_deck = {}
        for dk, deck in enumerate(presets):
            trip = [(t["i"], t["j"], r) for t in tasks if t["kind"] == "rr" and t["d"] == dk and t["id"] in res
                    for _, r in res[t["id"]]["results"]]
            if trip:
                per_deck[deck] = np.round(payoff_matrix(trip, len(runs)), 4).tolist()
        S["round_robin"] = {"agents": runs, "P": np.round(P, 4).tolist(), "games": np.asarray(G).tolist(),
                            "elo": {runs[k]: round(float(ratings[k]), 1) for k in range(len(runs))},
                            "nash": {runs[k]: round(float(x[k]), 4) for k in range(len(runs))},
                            "C_vs_A": {"score": round(sc, 4), "ci": ci(sc, len(cvsa)), "matches": len(cvsa)},
                            "per_deck": per_deck}
    json.dump(S, open(os.path.join(d, "summary.json"), "w"), indent=1)
    lines = [f"# Stage-4 evaluation at {meta['point']}", "",
             "checkpoints: " + ", ".join(f"{r} `{os.path.basename(c['path'])}` ({c['global_step']:,} steps)"
                                         for r, c in meta["checkpoints"].items()),
             f"missing tasks: {len(missing)}", "", "## Generalisation (score, 95% Wilson)", "",
             "| run | bot | pool decks | held-out decks | pool − held-out |", "|---|---|---|---|---|"]
    for r, bots in gen.items():
        for bot, v in bots.items():
            f = lambda k: f"{v[k]['score']:.3f} [{v[k]['ci'][0]:.3f}, {v[k]['ci'][1]:.3f}] n={v[k]['matches']}" \
                if k in v else "–"                                                    # noqa: E731
            g = v.get("gap_pool_minus_heldout")
            lines.append(f"| {r} | {bot} | {f('pool')} | {f('heldout')} | "
                         + (f"{g['value']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]" if g else "–") + " |")
    lines += ["", "## Specialist check (Hog 2.6 mirror)", "", "| run | specialist | score | n |", "|---|---|---|---|"]
    for r, v in spec.items():
        for name, s in v.items():
            lines.append(f"| {r} | {name} | {s['score']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}] | {s['matches']} |")
    lines += ["", "## Deck meta-game (Nash mixture over the 9 presets)", ""]
    for r, v in metas.items():
        lines.append(f"- {r}: " + ", ".join(f"{k} {p:.2f}" for k, p in sorted(v["nash"].items(), key=lambda kv: -kv[1])))
    if "round_robin" in S:
        rr = S["round_robin"]
        lines += ["", "## Round robin (9 presets, mirror)", "",
                  "| | " + " | ".join(runs) + " | Elo | Nash |", "|---" * (len(runs) + 3) + "|"]
        for k, r in enumerate(runs):
            lines.append(f"| {r} | " + " | ".join("–" if q == k else f"{rr['P'][k][q]:.3f}" for q in range(len(runs)))
                         + f" | {rr['elo'][r]:.0f} | {rr['nash'][r]:.2f} |")
        c = rr["C_vs_A"]
        lines += ["", f"C vs A (four cross pairs pooled): {c['score']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}], "
                      f"n = {c['matches']}"]
    open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--point", choices=sorted(POINTS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--runs", nargs="+", default=list(RUNS))
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", os.path.join(ROOT, "experiments")))
    p.add_argument("--parts", nargs="+", default=["gen", "meta", "spec", "rr"], choices=["gen", "meta", "spec", "rr"])
    p.add_argument("--pool", default=POOL)
    p.add_argument("--heldout", default=HELDOUT)
    p.add_argument("--presets", default=PRESETS)
    p.add_argument("--specialists", nargs="+", default=None, help="NAME=PATH ... (default: S3_A_s1, S3_C_s0 finals)")
    p.add_argument("--gen-pool-episodes", type=int, default=72)
    p.add_argument("--gen-heldout-episodes", type=int, default=10)
    p.add_argument("--meta-matches", type=int, default=200)
    p.add_argument("--spec-matches", type=int, default=500)
    p.add_argument("--spec-chunk", type=int, default=100)
    p.add_argument("--rr-matches", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("dir")
    r.add_argument("--task", type=int, required=True)
    m = sub.add_parser("merge")
    m.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "plan":
        if a.specialists is None:
            lg = os.path.join(a.data_dir, "league")
            a.specialists = [f"S3_A_s1={lg}/S3_A_s1/model_061036.pt", f"S3_C_s0={lg}/S3_C_s0/model_061036.pt"]
        plan(a)
    elif a.cmd == "run":
        run_task(a.dir, a.task)
    else:
        merge(a.dir)


if __name__ == "__main__":
    main()
