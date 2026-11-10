#!/usr/bin/env python3
"""Stage-4 collapse diagnostics (2026-10-15). In both league runs (S4_C_s0, S4_C_s1) the learner scores
below 0.5 against its own pool snapshots and, after about 1.9-2.1B steps, collapses against the heuristic
bot while its position entropy goes to ~0; the self-play runs (S4_A_*) do not. Two CPU task kinds, run by
hpc/collapse_diag.sbatch exactly as hpc/stage4_eval.py runs its tasks (one single-threaded process per
CPU, finished tasks skipped):

  ladder  snapshots of ONE run at fixed step counts play each other on the 9 presets, each deck a mirror,
          `matches` per pair and deck, half per seat (match seeds as stage4_eval's round robin). Policies
          sample, on the CPU, without the D100 card floor (checkpoints never carry it). Answers: does a
          later snapshot beat an earlier one (later_beats_earlier, metagame.transitivity)?
  place   one snapshot against the heuristic bot on the 9 presets (mirror), `matches` per deck, half per
          seat, recording how the policy places its cards: plays per match, distinct tiles per match, the
          share of a match's plays on its most used tile, and per decision P(play) = 1 - P(wait), the
          UNWEIGHTED position entropy H(pos | slot) averaged over the legal slots and max_j p(j | slot).
          (The training log's entropy_pos is weighted by P(slot), so it also falls when the policy waits.)

    python hpc/collapse_diag.py plan-ladder --run S4_C_s0 --steps 0.25 0.5 1.0 --out DIR [--matches 100]
    python hpc/collapse_diag.py plan-place --snap S4_C_s0:1.0:2.4:0.1 S4_A_s1:2.0 --out DIR [--matches 10]
    python hpc/collapse_diag.py run DIR --task 17                     # one task -> DIR/out/<id>.json
    python hpc/collapse_diag.py merge DIR                             # DIR/summary.json, DIR/summary.md

A snapshot "RUN:STEPS" (billions) resolves to the run's snap_<e>.pt with e the multiple of its
snapshot_interval nearest STEPS (2.0 -> snap_122112 = 2.0007B steps, as stage4_eval)."""
import argparse
import collections
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PRESETS = "hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge;xbow;royal_hogs"


def _deck_list(s):
    return [d for d in s.split(";") if d]


def snapshot(data, run, steps_b):
    """(name, path, epoch, global_step) of run's snapshot nearest steps_b billion steps."""
    rd = os.path.join(data, "league", run)
    lg = json.load(open(os.path.join(rd, "config.json")))["league"]
    batch, interval = int(lg["batch_size"]), int(lg["snapshot_interval"])
    e = int(round(steps_b * 1e9 / batch / interval)) * interval
    p = os.path.join(rd, f"snap_{e}.pt")
    if not os.path.isfile(p):
        raise SystemExit(f"{run}: no snapshot for {steps_b}B ({p})")
    return f"{run}@{steps_b:g}B", p, e, e * batch


def _snap_specs(items):
    """'RUN:B' or 'RUN:B0:B1:STEP' -> [(run, steps_b)]."""
    out = []
    for it in items:
        parts = it.split(":")
        if len(parts) == 2:
            out.append((parts[0], float(parts[1])))
        elif len(parts) == 4:
            b0, b1, st = map(float, parts[1:])
            n = int(math.floor((b1 - b0) / st + 1e-9)) + 1
            out += [(parts[0], round(b0 + k * st, 6)) for k in range(n)]
        else:
            raise SystemExit(f"bad --snap {it!r}: use RUN:B or RUN:B0:B1:STEP")
    return out


def _write_plan(out, meta, tasks):
    os.makedirs(os.path.join(out, "out"), exist_ok=True)
    meta.update(n_tasks=len(tasks), planned=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(meta, open(os.path.join(out, "plan.json"), "w"), indent=1)
    with open(os.path.join(out, "tasks.jsonl"), "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"{out}: {len(tasks)} tasks")


def plan_ladder(a):
    if len(a.steps) > 10:
        raise SystemExit("at most 10 snapshots (the per-pair seed key 10 i + j must stay unique)")
    snaps = [snapshot(a.data_dir, a.run, s) for s in a.steps]
    presets = _deck_list(a.presets)
    tasks = []
    for i in range(len(snaps)):
        for j in range(i + 1, len(snaps)):
            for d in range(len(presets)):
                tasks.append({"id": len(tasks), "kind": "ladder", "i": i, "j": j, "d": d, "m0": 0, "m1": a.matches})
    meta = {"kind": "ladder", "run": a.run, "seed": a.seed, "presets": presets, "matches": a.matches,
            "snapshots": [{"name": n, "path": p, "epoch": e, "global_step": g} for n, p, e, g in snaps]}
    _write_plan(a.out, meta, tasks)


def plan_place(a):
    snaps = [snapshot(a.data_dir, r, s) for r, s in _snap_specs(a.snap)]
    meta = {"kind": "place", "seed": a.seed, "presets": _deck_list(a.presets), "matches": a.matches,
            "bot": a.bot, "snapshots": [{"name": n, "path": p, "epoch": e, "global_step": g} for n, p, e, g in snaps]}
    tasks = [{"id": k, "kind": "place", "k": k} for k in range(len(snaps))]
    _write_plan(a.out, meta, tasks)


# ------------------------------------------------------------------------------------------------ tasks
def _ladder(meta, t):
    from pufferroyale.metagame import match_seed, play_match
    sn, presets, seed = meta["snapshots"], meta["presets"], int(meta["seed"])
    a, b, deck = sn[t["i"]]["path"], sn[t["j"]]["path"], presets[t["d"]]
    rs = []
    for m in range(t["m0"], t["m1"]):                          # as stage4_eval's rr: even m -> a on team 0
        s = match_seed(seed, 2000 + 10 * t["i"] + t["j"], t["d"], m)
        r = play_match(a, b, deck, deck, s) if m % 2 == 0 else -play_match(b, a, deck, deck, s)
        rs.append((m, int(r)))
    return {"results": rs}


def _decision_stats(logits):
    """(P(play), mean over legal slots of H(pos | slot), mean over legal slots of max_j p(j | slot)) of
    one row of masked joint logits; (P(play), None, None) when no slot is legal."""
    import numpy as np
    from pufferroyale.league import _logits_f64
    x, floor = _logits_f64(logits)
    x = x.reshape(-1)
    A = x.shape[0]
    B = (A - 1) // 4
    legal = x > floor
    m = x[legal].max()
    e = np.where(legal, np.exp(np.where(legal, x - m, 0.0)), 0.0)
    p_play = 1.0 - e[0] / e.sum()
    hs, mx = [], []
    for s in range(4):
        seg, ok = x[1 + s * B:1 + (s + 1) * B], legal[1 + s * B:1 + (s + 1) * B]
        if not ok.any():
            continue
        z = seg[ok] - seg[ok].max()
        q = np.exp(z) / np.exp(z).sum()                         # p(j | slot) from the slot's own logits
        hs.append(float(-(q[q > 0] * np.log(q[q > 0])).sum()))
        mx.append(float(q.max()))
    if not hs:
        return float(p_play), None, None
    return float(p_play), sum(hs) / len(hs), sum(mx) / len(mx)


def _place(meta, t):
    import torch
    from pufferroyale.game import Game
    from pufferroyale.league import load_policy, select_actions
    from pufferroyale.metagame import _player, match_seed, match_settings
    sn = meta["snapshots"][t["k"]]
    path, presets, seed, bot = sn["path"], meta["presets"], int(meta["seed"]), meta["bot"]
    policy, rec = load_policy(path, "cpu")
    if rec:
        raise SystemExit("place: recurrent checkpoints are not supported")
    per_deck = {}
    tiles_all = collections.Counter()
    acc = collections.defaultdict(float)
    for d, deck in enumerate(presets):
        for m in range(meta["matches"]):
            s = match_seed(seed, 3000 + t["k"], d, m)
            team = m % 2
            agents = [path, bot] if team == 0 else [bot, path]
            ms = match_settings(agents, None, True)
            g = Game(deck0=deck, deck1=deck, seed=int(s), **ms["game"])
            opp = _player(bot, s, 1 - team, False, "cpu")
            gen = torch.Generator(device="cpu").manual_seed((int(s) * 2 + team) & ((1 << 62) - 1))
            cad = ms["frame_skip"]
            plays = []
            while True:
                st = g.state()
                if st["over"]:
                    break
                tk = int(st["tick"])
                for tm in (0, 1):
                    if tk % cad[tm]:
                        continue
                    if tm == team:
                        with torch.no_grad():
                            logits, _ = policy.forward_eval(torch.as_tensor(g.obs(tm)).unsqueeze(0), {})
                        pp, h, mx = _decision_stats(logits)
                        acc["decisions"] += 1
                        acc["p_play"] += pp
                        if h is not None:
                            acc["slot_decisions"] += 1
                            acc["h_pos"] += h
                            acc["max_p_pos"] += mx
                        a = int(select_actions(logits, False, gen)[0])
                        if a and int(g.play_action(tm, a)) == 0:
                            plays.append((a - 1) % 576)         # tile index ty * 18 + tx, own frame
                    else:
                        a = opp.act(g, tm)
                        if a:
                            g.play_action(tm, a)
                g.tick(min((tk // k + 1) * k for k in cad) - tk)
            r0 = int(st["result"][0])
            r = r0 if team == 0 else -r0
            c = collections.Counter(plays)
            tiles_all.update(c)
            row = per_deck.setdefault(deck, {"w": 0, "d": 0, "n": 0})
            row["w"] += r > 0
            row["d"] += r == 0
            row["n"] += 1
            acc["matches"] += 1
            acc["plays"] += len(plays)
            acc["distinct"] += len(c)
            acc["top_share"] += (c.most_common(1)[0][1] / len(plays)) if plays else 0.0
    n, nd, ns = acc["matches"], acc["decisions"], max(1.0, acc["slot_decisions"])
    tot = [sum(v[k] for v in per_deck.values()) for k in ("w", "d", "n")]
    top = tiles_all.most_common(1)
    return {"score": (tot[0] + 0.5 * tot[1]) / tot[2], "wdl": tot, "per_deck": per_deck,
            "plays_per_match": acc["plays"] / n, "distinct_tiles_per_match": acc["distinct"] / n,
            "top_tile_share": acc["top_share"] / n, "distinct_tiles_total": len(tiles_all),
            "top_tile_overall": [top[0][0], top[0][1] / max(1, sum(tiles_all.values()))] if top else None,
            "p_play": acc["p_play"] / nd, "h_pos_unweighted": acc["h_pos"] / ns, "max_p_pos": acc["max_p_pos"] / ns,
            "decisions": nd}


def run_task(d, tid):
    out = os.path.join(d, "out", f"{tid:04d}.json")
    if os.path.exists(out):
        return
    meta = json.load(open(os.path.join(d, "plan.json")))
    t = [json.loads(ln) for ln in open(os.path.join(d, "tasks.jsonl"))][tid]
    t0 = time.time()
    res = {"ladder": _ladder, "place": _place}[t["kind"]](meta, t)
    res.update(task=t, seconds=round(time.time() - t0, 1))
    with open(out + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(out + ".tmp", out)


# ------------------------------------------------------------------------------------------------ merge
def merge(d):
    import numpy as np
    from pufferroyale.metagame import elo, transitivity, wilson_interval
    meta = json.load(open(os.path.join(d, "plan.json")))
    tasks = [json.loads(ln) for ln in open(os.path.join(d, "tasks.jsonl"))]
    res = {t["id"]: json.load(open(p)) for t in tasks
           for p in [os.path.join(d, "out", f"{t['id']:04d}.json")] if os.path.exists(p)}
    missing = [t["id"] for t in tasks if t["id"] not in res]
    names = [s["name"] for s in meta["snapshots"]]
    lines = [f"# Collapse diagnostics: {meta['kind']}", "", f"missing tasks: {len(missing)}", ""]
    S = {"kind": meta["kind"], "missing_tasks": missing}
    if meta["kind"] == "ladder":
        n = len(names)
        W = np.zeros((n, n))
        G = np.zeros((n, n))
        for t in tasks:
            if t["id"] not in res:
                continue
            for _, r in res[t["id"]]["results"]:
                i, j = t["i"], t["j"]
                W[i, j] += (r + 1) / 2
                W[j, i] += (1 - r) / 2
                G[i, j] += 1
                G[j, i] += 1
        P = np.where(G > 0, W / np.maximum(G, 1), 0.5)
        np.fill_diagonal(P, 0.5)
        tr = transitivity(P)
        ratings = elo(P, G)
        S.update(names=names, payoff=P.round(4).tolist(), games=G.astype(int).tolist(), transitivity=tr,
                 elo=[round(float(x)) for x in ratings])
        lines += [f"Row snapshot's score against the column snapshot ({len(meta['presets'])} presets, mirror, pooled; "
                  f"{int(G[0, 1]) if n > 1 else 0} matches per pair).", "",
                  "| | " + " | ".join(names) + " | Elo |", "|---" * (n + 2) + "|"]
        for i in range(n):
            cells = ["–" if i == j else f"{P[i, j]:.2f}" for j in range(n)]
            lines.append(f"| {names[i]} | " + " | ".join(cells) + f" | {ratings[i]:.0f} |")
        lines += ["", f"later beats earlier: {tr['later_beats_earlier']:.2f} of {tr['pairs']} pairs; cyclic triads "
                  f"(every edge > 0.55): {tr['cyclic_triads']} of {tr['triads']}"]
        nxt = [(names[i + 1], P[i + 1, i], wilson_interval(P[i + 1, i], int(G[i + 1, i]))) for i in range(n - 1)]
        lines += ["", "Each snapshot against the one before it:", ""]
        lines += [f"- {nm}: {p:.3f} [{lo:.3f}, {hi:.3f}]" for nm, p, (lo, hi) in nxt]
    else:
        rows = []
        lines += [f"Each snapshot against {meta['bot']} on {len(meta['presets'])} presets (mirror), {meta['matches']} "
                  "match(es) per deck; "
                  "placement statistics of the snapshot's own plays and decisions.", "",
                  "| snapshot | score | plays / match | distinct tiles / match | top-tile share | tiles used (all) "
                  "| P(play) | H(pos\\|slot) | max p(pos\\|slot) |", "|---" * 9 + "|"]
        for t in tasks:
            if t["id"] not in res:
                continue
            r = res[t["id"]]
            rows.append(dict(name=names[t["k"]], **{k: v for k, v in r.items() if k not in ("task",)}))
            lines.append(f"| {names[t['k']]} | {r['score']:.3f} | {r['plays_per_match']:.1f} | "
                         f"{r['distinct_tiles_per_match']:.1f} | {r['top_tile_share']:.2f} | {r['distinct_tiles_total']} | "
                         f"{r['p_play']:.3f} | {r['h_pos_unweighted']:.3f} | {r['max_p_pos']:.3f} |")
        S["rows"] = rows
    json.dump(S, open(os.path.join(d, "summary.json"), "w"), indent=1)
    open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    data = os.environ.get("DATA_DIR", os.path.join(ROOT, "experiments"))
    p = sub.add_parser("plan-ladder")
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=float, nargs="+", required=True, help="snapshot step counts in billions")
    p.add_argument("--out", required=True)
    p.add_argument("--matches", type=int, default=100, help="per pair and deck, half per seat")
    q = sub.add_parser("plan-place")
    q.add_argument("--snap", nargs="+", required=True, help="RUN:B or RUN:B0:B1:STEP (billions of steps)")
    q.add_argument("--out", required=True)
    q.add_argument("--matches", type=int, default=10, help="per deck, half per seat")
    q.add_argument("--bot", default="bot:heuristic")
    for x in (p, q):
        x.add_argument("--data-dir", default=data)
        x.add_argument("--presets", default=PRESETS)
        x.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("dir")
    r.add_argument("--task", type=int, required=True)
    m = sub.add_parser("merge")
    m.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "plan-ladder":
        plan_ladder(a)
    elif a.cmd == "plan-place":
        plan_place(a)
    elif a.cmd == "run":
        run_task(a.dir, a.task)
    else:
        merge(a.dir)


if __name__ == "__main__":
    main()
