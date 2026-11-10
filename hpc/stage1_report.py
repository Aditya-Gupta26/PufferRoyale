#!/usr/bin/env python3
"""Stage-1 report (docs/HPC_HANDOFF.md §8, TRAINING_PLAN §8.1): the numbers the decision rule needs, for
every ladder (scripts/stages.py prefix), from the files the runs wrote. Read-only.

    python hpc/stage1_report.py --data-dir $DATA_DIR S1main_s0 S1main_s1 S1noshape_s0 ... [--png curves.png]

Per ladder (prefix P): rung runs are $DATA_DIR/league/P_<i>_<bot>/; the gate verdicts come from
$DATA_DIR/stages_P.json when the ladder finished (scripts/stages.py's own verdict, never recomputed).
Reported:
  * per rung: learner steps, the learner's score over its last --window finished matches vs the rung's
    bot (from history.jsonl "outcomes"), and the stages.py verdict;
  * steps to G1 / G2: cumulative learner steps (rungs start from 0, so earlier rungs' steps are added)
    at the end of the rung that passed;
  * the heuristic rung's score over the last --window matches before 100M / 200M / 300M learner steps
    of that rung, and whether it is still rising (score over the last 20% of the rung's steps minus the
    score over the 20% before that, in points);
  * --png: the learner's score vs bot:heuristic against the heuristic rung's steps, one line per
    ladder (rolling over --window matches). Configurations share a colour, seeds differ by line style.
Fixed-budget heuristic phase (DECISIONS D86): when $DATA_DIR/league/P_H300/ exists (300M steps vs
bot:heuristic from the ladder's random-rung model, no early stop), a second table gives its score over
the last --window matches before each mark and its rising points, and a third the mean over the seeds
of each configuration (prefix without _s<seed>), sorted by the last mark every run has reached; the
TRAINING_PLAN §8.1 decision rule reads that table. --marks changes the marks (e.g. 11M for an interim
look); --png-h300 plots those runs' curves.
Prints markdown tables; --json writes everything.
"""
import argparse
import glob
import json
import os
import re

MARKS = (100_000_000, 200_000_000, 300_000_000)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#8a8a85", "#00a3c4"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_history(run_dir):
    out = []
    path = os.path.join(run_dir, "history.jsonl")
    if not os.path.isfile(path):
        return out
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                         # a line cut short by a killed job
    return out


def outcomes(hist, spec):
    """[(global_step at the end of the epoch, result in {+1, 0, -1})] vs `spec`, in order."""
    out = []
    for r in hist:
        for res in (r.get("outcomes") or {}).get(spec, []):
            out.append((int(r["global_step"]), int(res)))
    return out


def score(results):
    return sum(0.5 * (r + 1) for _, r in results) / len(results) if results else None


def last_before(res, step, window):
    sub = [x for x in res if x[0] <= step]
    return score(sub[-window:]) if sub else None


def parse_marks(text):
    out = []
    for t in text.split(","):
        t = t.strip().upper()
        out.append(int(float(t[:-1]) * 1_000_000) if t.endswith("M") else int(t))
    return tuple(out)


def mark_name(m):
    return f"{m / 1e6:g}M"


def rising(res, steps):
    a = [x for x in res if 0.6 * steps < x[0] <= 0.8 * steps]
    b = [x for x in res if x[0] > 0.8 * steps]
    return round(100 * (score(b) - score(a)), 1) if a and b else None


def fixed_budget(data_dir, prefix, window):
    """The D86 run P_H300 (None when absent): steps, score at each mark, last score, rising points."""
    run = os.path.join(data_dir, "league", f"{prefix}_H300")
    hist = read_history(run)
    if not hist:
        return None
    res = outcomes(hist, "bot:heuristic")
    steps = int(hist[-1]["global_step"])
    return {"run": run, "steps": steps, "matches": len(res), "score_last": score(res[-window:]),
            "at": {mark_name(m): (last_before(res, m, window) if steps >= m else None) for m in MARKS},
            "rising_pts": rising(res, steps) if steps > 0 and res else None, "outcomes": res}


def ladder(data_dir, prefix, window):
    runs = sorted(glob.glob(os.path.join(data_dir, "league", f"{prefix}_*_*")),
                  key=lambda p: int(re.match(r".*_(\d+)_[^_/]+$", p).group(1)))
    verdicts = {}
    sj = os.path.join(data_dir, f"stages_{prefix}.json")
    if os.path.isfile(sj):
        for r in json.load(open(sj)).get("rungs", []):
            verdicts[os.path.basename(str(r.get("run_dir", "")).rstrip("/"))] = r
    rungs, offset, steps_to = [], 0, {}
    heur = None
    for run in runs:
        bot = run.rsplit("_", 1)[1]
        spec = f"bot:{bot}"
        hist = read_history(run)
        res = outcomes(hist, spec)
        steps = int(hist[-1]["global_step"]) if hist else 0
        v = verdicts.get(os.path.basename(run), {})
        rung = {"run": run, "bot": bot, "steps": steps, "matches": len(res), "score_last": score(res[-window:]),
                "passed": v.get("passed"), "stages_score": v.get("score"), "reason": v.get("reason")}
        if v.get("passed"):
            steps_to[bot] = offset + steps
        offset += steps
        if bot == "heuristic":
            heur = res
            rung["at"] = {mark_name(m): (last_before(res, m, window) if steps >= m else None) for m in MARKS}
            if steps > 0 and res:
                rung["rising_pts"] = rising(res, steps)
        rungs.append(rung)
    return {"prefix": prefix, "rungs": rungs, "steps_to": steps_to, "finished": os.path.isfile(sj),
            "heuristic_outcomes": heur, "fixed_budget": fixed_budget(data_dir, prefix, window)}


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(ladders):
    head = ("| ladder | noop: steps / score / gate | random: steps / score / gate | steps to G2 | heuristic @"
            + " / ".join(mark_name(m) for m in MARKS) + " | heuristic last | rising (pts) |")
    out = [head, "|" + "---|" * 7]
    for L in ladders:
        cells = {}
        h = {}
        for r in L["rungs"]:
            gate = {True: "PASS", False: "FAIL", None: "…"}[r["passed"]]
            cells[r["bot"]] = f"{r['steps'] / 1e6:.1f}M / {fmt(r['score_last'])} / {gate}"
            if r["bot"] == "heuristic":
                h = r
        at = h.get("at", {})
        g2 = L["steps_to"].get("random")
        out.append(f"| {L['prefix']} | {cells.get('noop', '-')} | {cells.get('random', '-')} | "
                   f"{fmt(g2 / 1e6, 1) + 'M' if g2 else '-'} | {' / '.join(fmt(at.get(mark_name(m))) for m in MARKS)} | "
                   f"{fmt(h.get('score_last'))} ({h.get('steps', 0) / 1e6:.0f}M) | {fmt(h.get('rising_pts'), 1)} |")
    return "\n".join(out)


def config_of(prefix):
    return re.sub(r"_s\d+$", "", prefix)


def fixed_tables(ladders):
    """(per-run table, per-config table) for the D86 runs, or None when there are none."""
    runs = [(L["prefix"], L["fixed_budget"]) for L in ladders if L.get("fixed_budget")]
    if not runs:
        return None
    names = [mark_name(m) for m in MARKS]
    t1 = ["| run | steps | " + " | ".join(f"@{n}" for n in names) + " | last | rising (pts) |",
          "|" + "---|" * (len(names) + 4)]
    for p, fb in runs:
        t1.append(f"| {p}_H300 | {fb['steps'] / 1e6:.0f}M | " + " | ".join(fmt(fb["at"].get(n)) for n in names)
                  + f" | {fmt(fb['score_last'])} | {fmt(fb['rising_pts'], 1)} |")
    groups = {}
    for p, fb in runs:
        groups.setdefault(config_of(p), []).append((p, fb))
    reached = [n for n in names if all(fb["at"].get(n) is not None for _, fb in runs)]
    key = reached[-1] if reached else None
    rows = []
    for c, members in groups.items():
        means = []
        for n in names:
            vals = [fb["at"].get(n) for _, fb in members]
            means.append(sum(vals) / len(vals) if vals and all(v is not None for v in vals) else None)
        seeds = " / ".join(fmt(fb["at"].get(key)) if key else "-" for _, fb in members)
        rows.append((means[names.index(key)] if key else -1, c, means, seeds, len(members)))
    rows.sort(key=lambda r: -(r[0] if r[0] is not None else -1))
    t2 = [f"| config | seeds | " + " | ".join(f"mean @{n}" for n in names) + f" | per seed @{key or '-'} |",
          "|" + "---|" * (len(names) + 3)]
    for _, c, means, seeds, k in rows:
        t2.append(f"| {c} | {k} | " + " | ".join(fmt(m) for m in means) + f" | {seeds} |")
    return "\n".join(t1), "\n".join(t2)


def plot(ladders, path, window, fixed=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configs = []
    for L in ladders:
        c = re.sub(r"_s\d+$", "", L["prefix"])
        if c not in configs:
            configs.append(c)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for L in ladders:
        res = ((L.get("fixed_budget") or {}).get("outcomes") if fixed else L["heuristic_outcomes"]) or []
        if len(res) < window:
            continue
        c = re.sub(r"_s\d+$", "", L["prefix"])
        xs, ys = [], []
        for i in range(window, len(res) + 1, max(1, window // 4)):
            xs.append(res[i - 1][0])
            ys.append(score(res[i - window:i]))
        seed = re.search(r"_s(\d+)$", L["prefix"])
        ax.plot(xs, ys, color=SERIES[configs.index(c) % len(SERIES)], linewidth=1.0,
                linestyle="-" if not seed or seed.group(1) == "0" else "--",
                label=L["prefix"] + ("_H300" if fixed else ""))
    ax.axhline(0.5, color=INK_2, linewidth=0.5, alpha=0.6)
    ax.axhline(0.55, color=INK_2, linewidth=0.5, linestyle=":", alpha=0.8)
    ax.set_ylim(-0.02, 1.02)
    what = "fixed-budget runs (D86, from each ladder's random-rung model)" if fixed else "heuristic rung"
    ax.set_title(f"stage 1, {what}: learner score vs bot:heuristic (last {window} matches); dotted = G3 threshold 0.55",
                 loc="left", fontsize=9, color=INK)
    ax.set_xlabel("learner steps " + ("in the fixed-budget run" if fixed else "in the heuristic rung"), fontsize=8,
                  color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=7, length=0)
    if ax.lines:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=7, labelcolor=INK_2)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="+", help="stages.py --run-prefix values")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--window", type=int, default=2000, help="matches per score (default 2000, the gate window)")
    ap.add_argument("--png", help="write the heuristic-score comparison plot here")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--marks", default="100M,200M,300M", help="comma-separated step marks (default 100M,200M,300M)")
    ap.add_argument("--png-h300", help="write the fixed-budget runs' heuristic-score plot here")
    a = ap.parse_args()
    global MARKS
    MARKS = parse_marks(a.marks)
    ladders = [ladder(a.data_dir, p, a.window) for p in a.prefixes]
    print(table(ladders))
    ft = fixed_tables(ladders)
    if ft:
        print("\nfixed-budget runs (D86):\n" + ft[0] + "\n\nby configuration (mean over seeds):\n" + ft[1])
    if a.png:
        plot(ladders, a.png, a.window)
        print(f"plot: {a.png}")
    if a.png_h300:
        plot(ladders, a.png_h300, a.window, fixed=True)
        print(f"plot: {a.png_h300}")
    if a.json:
        def strip(L):
            out = {k: v for k, v in L.items() if k != "heuristic_outcomes"}
            if out.get("fixed_budget"):
                out["fixed_budget"] = {k: v for k, v in out["fixed_budget"].items() if k != "outcomes"}
            return out
        with open(a.json, "w") as f:
            json.dump([strip(L) for L in ladders], f, indent=1)


if __name__ == "__main__":
    main()
