"""Regenerate the README figures of PufferRoyale (docs/figures/*.png).

The figures are drawn from experiment results that are NOT in this repository: they live in the
project's data directory, $PR_ROOT/runs (league runs, best-response results, round robins, human-play
records). Without that directory the script cannot run.

How to run (on the Torch cluster, from the repository root; one process, a few seconds):

    source /scratch/$USER/pufferroyale/env.sh          # sets PR_ROOT and SIF (the project's Apptainer image)
    apptainer exec --no-home --bind $PR_ROOT:$PR_ROOT $SIF \
        python docs/figures/make_figures.py

Options:
    --runs DIR    results directory (default: $PR_ROOT/runs)
    --doc FILE    selection log the numbers are checked against (default: docs/STAGE4_SELECTION.md)
    --out DIR     where the PNGs go (default: this script's directory)
    --only NAME   make only these figures (e.g. --only training_curve human_play)

Needs numpy and matplotlib (both in the project image). Every PNG is written next to this script.
At the end the script prints a check of every plotted number that docs/STAGE4_SELECTION.md also
states (data file vs document); a mismatch is printed, it does not stop the run.

Figures (data sources in docs/figures/FIGURES.md):
    training_curve.png       S4_A_s1's evaluation scores over its 10B training steps
    selection_br.png         best-response exploitability of the selection candidates (stages 2 and 3b)
    stage1_round_robin.png   stage-1 round robin of the 9 candidates: payoff matrix and Elo
    floor_ablation.png       stage-4a card-floor arms, evaluation over their +1.0B steps
    stage5_round_robin.png   stage-5 round robin of the 4 stage-4a arms: payoff matrix and Elo
    human_play.png           ranked human matches against the policy (aggregates only, no player data)
    claude_games.png         Claude models against the policy (also writes claude_games.json, the per-game table)
"""
import argparse
import glob
import json
import math
import os
import re
import textwrap
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- style
# Okabe-Ito hues (colour-blind safe). Roles are fixed across all figures:
#   BLUE = the selected policy (S4_A_s1) and its no-floor lineage; VERM = the other side / contrast.
BLUE, SKY, VERM, GREEN = "#0072B2", "#56B4E9", "#D55E00", "#009E73"
# Card-floor arms: the control keeps BLUE; the floors are one orange hue, light -> dark with the floor
# (OKLCH L 0.77 / 0.60 / 0.43, hue 55; checked: monotone, adjacent dL >= 0.17, min CVD dE vs each other
# and BLUE >= 15 (Machado 2009 protan/deutan, OKLab x100)).
ARM_COLOR = {"ctrl": BLUE, "f05": "#f39d5e", "f10": "#c66000", "f20": "#7c3c00"}
ARM_MARKER = {"ctrl": "o", "f05": "s", "f10": "^", "f20": "D"}
ARM_LABEL = {"ctrl": "control (no floor)", "f05": "floor 0.05", "f10": "floor 0.10", "f20": "floor 0.20"}
INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#767676"
GRID, AXIS, SHADE = "#e4e4e4", "#a8a8a8", "#f1f1f1"
# Diverging map for "row's score vs column": vermillion (row loses) - light gray (0.5, even) - blue (row wins).
DIVERGING = LinearSegmentedColormap.from_list("pr_div", [VERM, "#f0ab7d", "#f3f3f3", "#7fb2d6", BLUE])

# Short names of the selection candidates (as in the stage-1 table of docs/STAGE4_SELECTION.md); used by both
# selection figures so a model has one name everywhere.
SHORT = {"S4_C_s0@1.55B": "C@1.55B", "S4x_selfonly_s0": "selfonly", "S4x_nommd_s0": "nommd",
         "S4x_posent_s0": "posent", "S4df20_A_s0": "S4df20"}
WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
W = 8.0  # figure width, inches (1200 px at 150 dpi; ~900 px in a README column -> 11 pt reads as ~14 px)
DPI = 150

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def frame(h_plot, title, subtitle, foot=None, w=W):
    """A white figure that holds only the plot. The figure's context (its takeaway, what is measured, caveats) is
    NOT drawn into the image: it lives in the README text around the figure (user, 2026-11-14: "The context for
    figures is in the text - not in the figures itself"). The title, subtitle and footnote are still computed from
    the data and printed, so the README caption can be checked against them. Returns (fig, top, bottom): the
    figure fractions left for the plot area."""
    for tag, txt in (("takeaway", title), ("measured", subtitle), ("note", foot)):
        if txt:
            print(f"  {tag}: {txt}")
    head, tail = 0.16, 0.04
    H = head + h_plot + tail
    fig = plt.figure(figsize=(w, H), dpi=DPI)
    return fig, (H - head) / H, tail / H


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path}")


def repel(ys, gap):
    """Spread label y positions so neighbours are at least `gap` apart (keeps the order)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


# ----------------------------------------------------------------------------- data helpers
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def wilson(k, n, z=1.959964):
    """95% Wilson interval of k successes in n (k may be fractional: wins + draws/2), as hpc/eval_watch.py."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rolling_mean(y, k=5):
    """Centred moving average over k points; the window shrinks at both ends."""
    y = np.asarray(y, dtype=float)
    h = k // 2
    return np.array([y[max(0, i - h):i + h + 1].mean() for i in range(len(y))])


# ----------------------------------------------------------------------------- check against the selection log
class DocCheck:
    """Compares numbers from the data files with the ones docs/STAGE4_SELECTION.md states. A doc number with d
    decimals matches when the data value rounds to it (|data - doc| <= half a unit in the last place)."""

    def __init__(self, path):
        self.path = path
        self.lines = open(path).read().splitlines() if os.path.isfile(path) else []
        self.ok, self.bad, self.missing = 0, [], []
        if not self.lines:
            self.missing.append(f"document not found: {path}")

    def table(self, header_prefix):
        for i, line in enumerate(self.lines):
            line = line.strip()  # some tables sit inside list items (indented)
            if line.startswith(header_prefix):
                rows = []
                for row in self.lines[i + 1:]:
                    row = row.strip()
                    if not row.startswith("|"):
                        break
                    if set(row.replace("|", "").strip()) <= set("-: "):
                        continue
                    rows.append([c.strip() for c in row.strip().strip("|").split("|")])
                return [c.strip() for c in line.strip().strip("|").split("|")], rows
        self.missing.append(f"table not found in doc: '{header_prefix}'")
        return None, []

    def find(self, pattern):
        m = re.search(pattern, " ".join(" ".join(self.lines).split()))  # sentences wrap across lines
        if m:
            return m
        self.missing.append(f"sentence not found in doc: /{pattern}/")
        return None

    @staticmethod
    def nums(s):
        return re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", " ").replace("**", ""))

    def cmp(self, what, data, doc_str):
        dec = len(doc_str.split(".")[1]) if "." in doc_str else 0
        doc = float(doc_str)
        if abs(data - doc) <= 0.5 * 10 ** (-dec) + 1e-9:
            self.ok += 1
        else:
            self.bad.append(f"{what}: data {data:.{dec + 2}f} vs doc {doc_str}")

    def report(self):
        print(f"\n=== check against {os.path.relpath(self.path)}: {self.ok} numbers match, "
              f"{len(self.bad)} mismatch(es), {len(self.missing)} not checked")
        for b in self.bad:
            print("  MISMATCH", b)
        for m in self.missing:
            print("  NOT CHECKED", m)


# ----------------------------------------------------------------------------- 1. training curve
def fig_training_curve(runs, out, chk):
    run = "S4_A_s1"
    rows = load_jsonl(f"{runs}/league/{run}/eval_history.jsonl")
    x = np.array([r["global_step"] for r in rows]) / 1e9
    res = [r["results"] for r in rows]
    series = [  # key, direct label, colour
        ("heuristic_pool", "bot, training decks", SKY),
        ("heuristic_heldout", "bot, held-out decks", BLUE),
        ("S3_A_s1", "Hog specialist 1", VERM),
        ("S3_C_s0", "Hog specialist 2", GREEN),
    ]
    rnd = min(min(r["random_pool"]["score"], r["random_heldout"]["score"]) for r in res)
    hh = [r["heuristic_heldout"] for r in res]
    first, last = hh[0], hh[-1]
    fig, top, bottom = frame(
        4.3,
        f"The agent's score against the scripted bot climbs from {first['score']:.2f} to {last['score']:.2f} "
        "during training",
        "Every 0.1B steps the agent played the scripted bot on its 7 training decks and on 52 decks it never "
        "trained with (held-out). The two low lines are agents trained only on the Hog deck, met in a Hog "
        "mirror: on their own deck they still beat it easily.",
        foot=f"Score = (wins + ½ draws) / matches. Dots are single evaluations; lines average 5 neighbouring "
             f"evaluations. Against the random bot the agent scored at least {math.floor(rnd * 1000) / 1000:.3f} "
             f"every time (not drawn).")
    ax = fig.add_axes([0.085, bottom + 0.62 / fig.get_figheight(), 0.645,
                       top - bottom - 0.70 / fig.get_figheight()])
    ax.grid(axis="y")
    ends = []
    for key, lab, col in series:
        y = np.array([r[key]["score"] for r in res])
        ys = rolling_mean(y, 5)
        ax.scatter(x, y, s=10, color=col, alpha=0.35, linewidths=0, zorder=2)
        ax.plot(x, ys, color=col, lw=2.2, zorder=3, label=lab)
        ends.append((ys[-1], col, f"{ys[-1]:.2f}  {lab}"))
    # label every line at its smoothed end value (no legend)
    ly = repel([e[0] for e in ends], 0.055)
    for (yv, col, txt), yl in zip(ends, ly):
        ax.text(x[-1] + 0.17, yl, txt, color=INK, fontsize=11, va="center", ha="left")
        ax.plot([x[-1] + 0.03, x[-1] + 0.14], [yv, yl], color=MUTED, lw=0.8, clip_on=False)
    # call out the first held-out evaluation (raw)
    ax.annotate(f"{first['score']:.2f} at the first check", xy=(x[0], first["score"]), xytext=(0.6, 0.70),
                fontsize=11, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, shrinkA=0, shrinkB=3))
    # the checkpoint that entered the selection (league/S4_A_s1/snap_476523.pt, 7,807,352,832 steps)
    sel = 7.807352832
    ax.axvline(sel, color=AXIS, lw=1, zorder=1)
    ax.text(sel - 0.12, 0.45, "selection\ncheckpoint\n(7.81B)", fontsize=11, color=INK2, ha="right", va="center")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 11))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("training steps (billions)")
    ax.set_ylabel("the agent's score")
    save(fig, out, "training_curve.png")
    s3a, s3c = [r["S3_A_s1"] for r in res], [r["S3_C_s0"] for r in res]
    print(f"  {run}: {len(rows)} evaluations {x[0]:.3f}B-{x[-1]:.3f}B; held-out first {first['score']:.4f} "
          f"[{first['lo']:.3f},{first['hi']:.3f}] last {last['score']:.4f} [{last['lo']:.3f},{last['hi']:.3f}]; "
          f"pool first/last {res[0]['heuristic_pool']['score']:.3f}/{res[-1]['heuristic_pool']['score']:.3f}; "
          f"S3_A_s1 first/last {s3a[0]['score']:.3f}/{s3a[-1]['score']:.3f}; "
          f"S3_C_s0 first/last {s3c[0]['score']:.3f}/{s3c[-1]['score']:.3f}; smoothed ends "
          + ", ".join(f"{k} {v[2]}" for (k, _, _), v in zip(series, ends)) + f"; random min {rnd:.4f}")


# ----------------------------------------------------------------------------- 2. selection: best response
def _br(runs, prefix, agent_file, seed):
    d = load_json(f"{runs}/eval/br_{prefix}_{agent_file}_s{seed}.json")
    k = d["wins"] + 0.5 * d["draws"]
    lo, hi = wilson(k, d["matches"])
    return dict(score=d["br_score"], lo=lo, hi=hi, w=d["wins"], d=d["draws"], l=d["losses"], n=d["matches"],
                steps=d["total_timesteps"], init=d["init_from_target"])


def fig_selection_br(runs, out, chk):
    rr = load_json(f"{runs}/eval/stage4_rr_20261026/summary.json")["round_robin"]
    elo = rr["elo"]
    agents = list(rr["agents"])
    ref = "S4_C_s0@1.55B"  # measured, not eligible (selection rule 3)
    fname = {a: a.replace("@", "_at") for a in agents}
    copy = {a: [_br(runs, "S4sel", fname[a], s) for s in (0, 1)] for a in agents}
    scratch = {}
    for a in agents:
        if os.path.isfile(f"{runs}/eval/br_S4scr_{fname[a]}_s0.json"):
            scratch[a] = [_br(runs, "S4scr", fname[a], s) for s in (0, 1)]
    # rule 3 with deviation 2: a br_score below 0.5 counts as 0.5; score = mean of the two seeds
    score = {a: float(np.mean([max(0.5, b["score"]) for b in copy[a]])) for a in agents}
    raw = {a: float(np.mean([b["score"] for b in copy[a]])) for a in agents}
    elig = sorted([a for a in agents if a != ref], key=lambda a: (round(score[a], 6), -elo[a]))
    best = min(score[a] for a in elig)
    tied = [a for a in elig if score[a] <= best + 0.05 + 1e-12]
    winner = max(tied, key=lambda a: elo[a])
    order = elig + [ref]
    ypos = {a: i for i, a in enumerate(elig)}
    ypos[ref] = len(elig) + 0.6

    # checks against the doc (stage-2 table, stage-3b table)
    _, t2 = chk.table("| target | seed 0: W-D-L, br_score")
    for row in t2:
        a = row[0].split(" ")[0]
        if a not in copy:
            chk.missing.append(f"stage-2 doc row '{row[0]}' has no data file")
            continue
        for s in (0, 1):
            w, d_, l_ = re.search(r"(\d+)-(\d+)-(\d+)", row[1 + s]).groups()  # "W-D-L, br_score"
            sc = row[1 + s].split(",")[-1].strip()
            b = copy[a][s]
            for nm, dv, ds in (("W", b["w"], w), ("D", b["d"], d_), ("L", b["l"], l_), ("br_score", b["score"], sc)):
                chk.cmp(f"stage 2 {a} seed {s} {nm}", dv, ds)
        chk.cmp(f"stage 2 {a} mean (rule 3 as written)", raw[a], chk.nums(row[3])[0])
        chk.cmp(f"stage 2 {a} mean (<0.5 read as 0.5)", score[a], chk.nums(row[4])[0])
        chk.cmp(f"stage 2 {a} Elo", elo[a], chk.nums(row[5])[0])
    _, t3 = chk.table("| target | seed 0 | seed 1 | raw mean")
    for row in t3:
        a = row[0].split(" ")[0]
        if a not in scratch:
            chk.missing.append(f"stage-3b doc row '{row[0]}' has no data file")
            continue
        chk.cmp(f"stage 3b {a} seed 0", scratch[a][0]["score"], chk.nums(row[1])[0])
        chk.cmp(f"stage 3b {a} seed 1", scratch[a][1]["score"], chk.nums(row[2])[0])
        chk.cmp(f"stage 3b {a} raw mean", np.mean([b["score"] for b in scratch[a]]), chk.nums(row[3])[0])
        chk.cmp(f"stage 3b {a} score", np.mean([max(0.5, b["score"]) for b in scratch[a]]),
                chk.nums(row[4])[0])
    m = chk.find(r"\*\*Winner: (\S+)\*\*")
    if m and m.group(1) != winner:
        chk.bad.append(f"stage 3 winner: data rule gives {winner}, doc says {m.group(1)}")
    elif m:
        chk.ok += 1

    n_rows = len(order)
    fig, top, bottom = frame(
        0.42 * n_rows + 1.25,
        f"Attackers found no clear weakness in {WORDS.get(len(tied), str(len(tied))).lower()} candidates, and "
        f"{SHORT.get(winner, winner)} won the tie on Elo",
        "For each candidate we trained a fresh attacker (a best response) against a frozen copy of it, then "
        "played them 1,000 times on random decks. An attacker score above 0.5 means it found a weakness, so "
        "lower is better. A candidate's score is the mean of its 2 attackers, each counted as at least 0.5.",
        foot="Bars are 95% intervals. An attacker that starts as a copy of its target begins at 0.5, so a score "
             "below 0.5 means it found nothing and drifted.")
    H = fig.get_figheight()
    ax = fig.add_axes([0.205, bottom + 0.58 / H, 0.47, top - bottom - 1.05 / H])
    ax.axvspan(0, 0.5, color=SHADE, zorder=0, lw=0)
    ax.axvline(0.5, color=AXIS, lw=1.2, zorder=1)
    ax.text(0.25, -0.75, "no weakness found", ha="center", va="center", fontsize=11, color=INK2)
    ax.text(0.75, -0.75, "weakness found", ha="center", va="center", fontsize=11, color=INK2)
    # winner row band
    yw = ypos[winner]
    ax.add_patch(Rectangle((0, yw - 0.45), 1, 0.9, color="#e3eef7", zorder=0, lw=0))
    for a in order:
        y = ypos[a]
        pts = [(b, BLUE, "o") for b in copy[a]] + [(b, VERM, "s") for b in scratch.get(a, [])]
        offs = np.linspace(-0.22, 0.22, len(pts)) if len(pts) > 2 else [-0.09, 0.09]
        for (b, col, mk), dy in zip(pts, offs):
            ax.plot([b["lo"], b["hi"]], [y + dy, y + dy], color=col, lw=1.6, zorder=3)
            ax.plot(b["score"], y + dy, marker=mk, ms=6.5, color=col, mec="white", mew=1.0, zorder=4,
                    ls="none")
    ax.set_ylim(len(elig) + 1.2, -1.1)
    ax.set_xlim(0, 1)
    ax.set_yticks([ypos[a] for a in order])
    ax.set_yticklabels([SHORT.get(a, a) + ("\n(not eligible)" if a == ref else "") for a in order])
    for t, a in zip(ax.get_yticklabels(), order):
        if a == winner:
            t.set_fontweight("bold")
            t.set_color(INK)
        if a == ref:
            t.set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.1)])
    ax.set_xlabel("attacker's score against the candidate")
    ax.spines["left"].set_visible(False)
    ax.axhline(len(elig) - 0.2, color=GRID, lw=1)
    # right-hand table: selection score and Elo
    tr = ax.get_yaxis_transform()
    ax.text(1.09, -0.75, "score", ha="center", va="center", fontsize=11, color=INK2, transform=tr,
            fontweight="bold")
    ax.text(1.29, -0.75, "Elo", ha="center", va="center", fontsize=11, color=INK2, transform=tr,
            fontweight="bold")
    for a in order:
        wgt = "bold" if a == winner else "normal"
        col = MUTED if a == ref else INK
        ax.text(1.09, ypos[a], f"{score[a]:.3f}", ha="center", va="center", fontsize=11, color=col,
                transform=tr, fontweight=wgt)
        ax.text(1.29, ypos[a], f"{elo[a]:.0f}", ha="center", va="center", fontsize=11, color=col,
                transform=tr, fontweight=wgt)
    # bracket for the tie group
    y0, y1 = ypos[tied[0]] - 0.3, ypos[tied[-1]] + 0.3
    ax.plot([1.39, 1.42, 1.42, 1.39], [y0, y0, y1, y1], color=INK2, lw=1, transform=tr, clip_on=False)
    ax.text(1.45, (y0 + y1) / 2, "tied", ha="left", va="center", fontsize=11, color=INK2, transform=tr)
    handles = [Line2D([], [], color=BLUE, marker="o", ms=6.5, lw=1.6, mec="white"),
               Line2D([], [], color=VERM, marker="s", ms=6.5, lw=1.6, mec="white")]
    ax.legend(handles, ["attacker started as a copy (150M steps)", "attacker from scratch (1B steps)"],
              loc="lower left", bbox_to_anchor=(-0.40, 1.01), ncol=2, handlelength=1.8, columnspacing=1.2,
              borderaxespad=0)
    save(fig, out, "selection_br.png")
    for a in order:
        print(f"  {a}: copy " + " / ".join(f"{b['score']:.4f} [{b['lo']:.3f},{b['hi']:.3f}]" for b in copy[a])
              + (" | scratch " + " / ".join(f"{b['score']:.4f} [{b['lo']:.3f},{b['hi']:.3f}]" for b in scratch[a])
                 if a in scratch else "")
              + f" | score {score[a]:.4f} raw {raw[a]:.4f} Elo {elo[a]:.1f}")
    print(f"  tie group {tied}; winner {winner}")


# ----------------------------------------------------------------------------- 3 + 5. round-robin heatmaps
def _rr_heatmap(rr, names, order, title, subtitle, foot, out, fname, highlight=None, colors=None, ci=False,
                cell=0.62, label_w=1.55):
    idx = [rr["agents"].index(a) for a in order]
    P = np.array(rr["P"])[np.ix_(idx, idx)]
    G = np.array(rr["games"])[np.ix_(idx, idx)]
    n = len(order)
    hm_w, gap, elo_w = cell * n, 0.3, 1.55
    assert label_w + hm_w + gap + elo_w <= W - 0.15, "heatmap + Elo panel wider than the figure"
    xlab = 1.02 if n > 5 else 0.72  # room under the heatmap for (rotated) column labels and the axis label
    fig, top, bottom = frame(cell * n + xlab + 0.15, title, subtitle, foot)
    H = fig.get_figheight()
    left = label_w / W
    ax = fig.add_axes([left, bottom + xlab / H, hm_w / W, cell * n / H])
    ax.imshow(P, cmap=DIVERGING, vmin=0, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # a model never plays itself: the cell stays blank
            v = P[i, j]
            r, g, b = to_rgb(DIVERGING(v))
            txt = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK
            if ci and i < j:
                lo, hi = wilson(v * G[i, j], G[i, j])
                ax.text(j, i - 0.13, f"{v:.3f}", ha="center", va="center", color=txt, fontsize=12.5,
                        fontweight="bold")
                ax.text(j, i + 0.22, f"[{lo:.3f}, {hi:.3f}]", ha="center", va="center", color=txt, fontsize=10.5)
            else:
                ax.text(j, i, f"{v:.3f}" if ci else f"{v:.2f}", ha="center", va="center", color=txt,
                        fontsize=12.5 if ci else 11, fontweight="bold" if ci else "normal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([names[a] for a in order], rotation=35 if n > 5 else 0, ha="right" if n > 5 else "center",
                       rotation_mode="anchor")
    ax.set_yticklabels([names[a] for a in order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("opponent (column)")
    ax.set_ylabel("model (row)", labelpad=6)
    if highlight:
        k = order.index(highlight)
        for t in (ax.get_xticklabels()[k], ax.get_yticklabels()[k]):
            t.set_fontweight("bold")
            t.set_color(INK)
        ax.add_patch(Rectangle((-0.5, k - 0.5), n, 1, fill=False, ec=INK, lw=1.6, zorder=5, clip_on=False))
    # Elo dot plot, rows aligned with the heatmap
    ex = fig.add_axes([(label_w + hm_w + gap) / W, bottom + xlab / H, elo_w / W, cell * n / H], sharey=ax)
    e = np.array([rr["elo"][a] for a in order])
    lo_e, hi_e = math.floor((e.min() - 40) / 50) * 50, math.ceil((e.max() + 40) / 50) * 50
    ex.axvline(1500, color=GRID, lw=1, zorder=0)
    for i, a in enumerate(order):
        col = colors[a] if colors else (BLUE if a == highlight else INK2)
        ex.axhline(i, color=GRID, lw=1, zorder=1)  # row guide (no stem: Elo has no zero)
        ex.plot(e[i], i, marker="o", ms=8, color=col, mec="white", mew=1.2, zorder=3)
        ex.text(e[i] + (hi_e - lo_e) * 0.07, i, f"{e[i]:.0f}", va="center", ha="left", fontsize=11,
                color=INK, fontweight="bold" if a == highlight else "normal")
    ex.set_xlim(lo_e, hi_e + (hi_e - lo_e) * 0.28)
    step = 200 if hi_e - lo_e > 300 else 100
    ex.set_xticks([t for t in range(int(math.ceil(lo_e / step) * step), int(hi_e) + 1, step)])
    ex.tick_params(axis="y", which="both", length=0, labelleft=False)  # shares the heatmap's minor ticks
    ex.spines["left"].set_visible(False)
    ex.set_xlabel("Elo (mean 1500)")
    ex.set_ylim(n - 0.5, -0.5)
    save(fig, out, fname)
    return P


def fig_stage1_rr(runs, out, chk):
    d = load_json(f"{runs}/eval/stage4_rr_20261026/summary.json")
    rr = d["round_robin"]
    names = {a: SHORT.get(a, a) for a in rr["agents"]}
    order = sorted(rr["agents"], key=lambda a: -rr["elo"][a])
    games = {int(g) for row in rr["games"] for g in row if g}
    # doc check: stage-1 table and the candidates' checkpoint steps
    hdr, rows = chk.table("| row vs col | C@1.55B")
    if hdr:
        inv = {v: k for k, v in names.items()}
        cols = [inv.get(c, c) for c in hdr[1:1 + len(order)]]
        for row in rows:
            a = inv.get(row[0], row[0])
            i = rr["agents"].index(a)
            for c, cellv in zip(cols, row[1:1 + len(cols)]):
                if cellv in ("–", "-"):
                    continue
                chk.cmp(f"stage 1 {names[a]} vs {names[c]}", rr["P"][i][rr["agents"].index(c)], cellv)
            chk.cmp(f"stage 1 Elo {names[a]}", rr["elo"][a], row[-2])
            chk.cmp(f"stage 1 Nash {names[a]}", rr["nash"][a], row[-1])
    _, crow = chk.table("| agent | checkpoint | steps |")
    for row in crow:
        a = row[0]
        if a in d["checkpoints"]:
            ck = d["checkpoints"][a]
            if not ck["path"].endswith(row[1]):
                chk.bad.append(f"stage 1 checkpoint {a}: data {ck['path']} vs doc {row[1]}")
            else:
                chk.ok += 1
            chk.cmp(f"stage 1 steps {a}", ck["global_step"] / 1e9, chk.nums(row[2])[0])
    top3 = order[:3]
    # the top-rated model's worst opponent (a model that beats it)
    i0 = rr["agents"].index(order[0])
    worst = min((a for a in order[1:]), key=lambda a: rr["P"][i0][rr["agents"].index(a)])
    worst_score = rr["P"][i0][rr["agents"].index(worst)]
    unbeaten = [a for a in order if all(rr["P"][rr["agents"].index(a)][rr["agents"].index(b)] > 0.5
                                        for b in order if b != a)]
    assert not unbeaten, f"{unbeaten} beats every other candidate: the title below would be wrong"
    fig_title = (f"No candidate beats all the others; even the top-rated {names[order[0]]} scores only "
                 f"{worst_score:.2f} against {names[worst]}")
    sub = (f"Each of the 9 candidates (checkpoints from 1.55B to 7.81B steps) played every other one "
           f"{max(games):,} times, in mirrors of the 9 preset decks. A cell is the row model's score against the "
           f"column model (95% interval about ±0.023). Rows are sorted by Elo.")
    nash = ", ".join(f"{names[a]} {rr['nash'][a]:.2f}" for a in order if rr["nash"][a] > 0.005)
    foot = (f"The hardest mix of these models to beat (the Nash mixture) is {nash}. In the selection, Elo was "
            f"only used to break a tie.")
    _rr_heatmap(rr, names, order, fig_title, sub, foot, out, "stage1_round_robin.png", highlight="S4_A_s1",
                cell=0.52, label_w=1.25)
    i = {a: rr["agents"].index(a) for a in top3}
    print(f"  games per pair {sorted(games)}; Elo " + ", ".join(f"{names[a]} {rr['elo'][a]:.1f}" for a in order)
          + f"; cycle: {names[top3[0]]} vs {names[top3[2]]} {rr['P'][i[top3[0]]][i[top3[2]]]:.3f}, "
            f"{names[top3[2]]} vs {names[top3[1]]} {rr['P'][i[top3[2]]][i[top3[1]]]:.3f}, "
            f"{names[top3[1]]} vs {names[top3[0]]} {rr['P'][i[top3[1]]][i[top3[0]]]:.3f}; Nash {nash}")


def fig_stage5_rr(runs, out, chk):
    d = load_json(f"{runs}/stage4/S5a/sub_20261109_113404/rr/summary.json")
    rr = d["round_robin"]
    arm = {a: a.rsplit("_", 1)[1] for a in rr["agents"]}
    names = {a: ARM_LABEL[arm[a]].replace(" (no floor)", "") for a in rr["agents"]}
    order = [a for k in ("ctrl", "f05", "f10", "f20") for a in rr["agents"] if arm[a] == k]
    games = {int(g) for row in rr["games"] for g in row if g}
    steps = {c["global_step"] for c in d["checkpoints"].values()}
    hdr, rows = chk.table("| row vs col | ctrl | f05")
    if hdr:
        by = {arm[a]: a for a in rr["agents"]}
        cols = hdr[1:5]
        for row in rows:
            a = by[row[0]]
            ia = rr["agents"].index(a)
            for c, cellv in zip(cols, row[1:5]):
                if cellv in ("–", "-"):
                    continue
                ib = rr["agents"].index(by[c])
                nums = chk.nums(cellv)
                v = rr["P"][ia][ib]
                chk.cmp(f"stage 5 {row[0]} vs {c}", v, nums[0])
                if len(nums) == 3:
                    lo, hi = wilson(v * rr["games"][ia][ib], rr["games"][ia][ib])
                    chk.cmp(f"stage 5 {row[0]} vs {c} CI lo", lo, nums[1])
                    chk.cmp(f"stage 5 {row[0]} vs {c} CI hi", hi, nums[2])
            chk.cmp(f"stage 5 Elo {row[0]}", rr["elo"][a], row[5])
            chk.cmp(f"stage 5 Nash {row[0]}", rr["nash"][a], row[6])
    colors = {a: ARM_COLOR[arm[a]] for a in rr["agents"]}
    ctrl = order[0]
    fig_title = "Head to head, the copy trained without a card floor beats every floored copy"
    sub = (f"The four copies of the agent at {min(steps) / 1e9:.1f}B steps (the control and floors 0.05, 0.10 and "
           f"0.20) played each other {max(games):,} times per pair in 9 deck mirrors. A cell is the row model's "
           f"score against the column model, with its 95% interval shown once per pair.")
    br5 = {}
    for f in sorted(glob.glob(f"{runs}/eval/br_S5a_S4L_A_s1_*_s[01].json")):
        br5.setdefault(re.search(r"_(ctrl|f05|f10|f20)_s[01]\.json$", f).group(1), []).append(load_json(f)["br_score"])
    if len(br5) == 4 and all(len(v) == 2 for v in br5.values()):
        m5 = {k: float(np.mean(v)) for k, v in br5.items()}
        assert min(m5, key=m5.get) == "ctrl", m5
        foot = (f"The deciding test, exploitability, agreed: 2 trained attackers per copy scored "
                f"{m5['ctrl']:.2f} on average against the control and {m5['f05']:.2f}, {m5['f10']:.2f} and "
                f"{m5['f20']:.2f} against floors 0.05, 0.10 and 0.20 (lower is harder to exploit).")
    else:
        foot = "The deciding test between the copies is their exploitability; this round robin only breaks ties."
    _rr_heatmap(rr, names, order, fig_title, sub, foot, out, "stage5_round_robin.png", highlight=ctrl,
                colors=colors, ci=True, cell=1.15, label_w=1.3)
    ia = {arm[a]: rr["agents"].index(a) for a in rr["agents"]}
    keys = ("ctrl", "f05", "f10", "f20")
    wins = [f"{r} beats {c} {rr['P'][ia[r]][ia[c]]:.4f}" for r in keys for c in keys
            if r != c and rr["P"][ia[r]][ia[c]] > 0.5]
    print("  " + "; ".join(wins) + "; Elo " + ", ".join(f"{arm[a]} {rr['elo'][a]:.1f}" for a in order)
          + f"; games {sorted(games)}")


# ----------------------------------------------------------------------------- 4. floor ablation
def fig_floor_ablation(runs, out, chk):
    arms = ["ctrl", "f05", "f10", "f20"]
    data = {a: load_jsonl(f"{runs}/league/S4L_A_s1_{a}/eval_history.jsonl") for a in arms}
    steps = {a: [r["global_step"] for r in data[a]] for a in arms}
    assert all(steps[a] == steps["ctrl"] for a in arms), "arms were evaluated at different steps"
    branch = load_json(f"{runs}/league/S4L_A_s1_ctrl/BRANCH.json") \
        if os.path.isfile(f"{runs}/league/S4L_A_s1_ctrl/BRANCH.json") else {}
    # doc check: sidecar table (10.70/10.80/10.90B) and the f20 / control sentence at 10.90B
    hdr, rows = chk.table("| step | control pool / held-out")
    col_arm = {1: "ctrl", 2: "f05", 3: "f10", 4: "f20"}
    for row in rows:
        step = float(chk.nums(row[0])[0])
        for ci, a in col_arm.items():
            nums = chk.nums(row[ci])
            if len(nums) < 2:
                continue
            r = min(data[a], key=lambda r: abs(r["global_step"] / 1e9 - step))
            if abs(r["global_step"] / 1e9 - step) > 0.006:
                chk.bad.append(f"stage 4a {a}: no evaluation at {step}B")
                continue
            res = r["results"]
            chk.cmp(f"stage 4a {a} {step}B pool", res["heuristic_pool"]["score"], nums[0])
            chk.cmp(f"stage 4a {a} {step}B held-out", res["heuristic_heldout"]["score"], nums[1])
            if len(nums) >= 4:
                chk.cmp(f"stage 4a {a} {step}B held-out lo", res["heuristic_heldout"]["lo"], nums[2])
                chk.cmp(f"stage 4a {a} {step}B held-out hi", res["heuristic_heldout"]["hi"], nums[3])
    m = chk.find(r"Last sidecar eval at 10\.90B: pool ([\d.]+) \[([\d.]+),([\d.]+)\],\s*held-out ([\d.]+) "
                 r"\[([\d.]+),([\d.]+)\] \(control ([\d.]+) \[([\d.]+),([\d.]+)\] / ([\d.]+) \[([\d.]+),([\d.]+)\]")
    if m:
        g = m.groups()
        for a, (p, plo, phi, h, hlo, hhi) in (("f20", g[0:6]), ("ctrl", g[6:12])):
            r = min(data[a], key=lambda r: abs(r["global_step"] / 1e9 - 10.90))["results"]
            for nm, dv, ds in (("pool", r["heuristic_pool"]["score"], p), ("pool lo", r["heuristic_pool"]["lo"], plo),
                               ("pool hi", r["heuristic_pool"]["hi"], phi),
                               ("held-out", r["heuristic_heldout"]["score"], h),
                               ("held-out lo", r["heuristic_heldout"]["lo"], hlo),
                               ("held-out hi", r["heuristic_heldout"]["hi"], hhi)):
                chk.cmp(f"stage 4a {a} 10.90B {nm} (text)", dv, ds)

    # Which floor arms are separated from the control on held-out decks: 95% intervals disjoint (arm below) at
    # each of the last three evaluations (10.7-10.9B, the window the selection log compares).
    def last3(a):
        return [r["results"]["heuristic_heldout"] for r in data[a][-3:]]
    sep = [a for a in arms[1:] if all(f["hi"] < c["lo"] for f, c in zip(last3(a), last3("ctrl")))]
    near = [a for a in arms[1:] if a not in sep]
    # pool decks: does any arm's interval ever fall entirely outside the control's (same evaluation)?
    pool_disjoint = [a for a in arms[1:] for f, c in zip(data[a], data["ctrl"])
                     if f["results"]["heuristic_pool"]["hi"] < c["results"]["heuristic_pool"]["lo"]
                     or f["results"]["heuristic_pool"]["lo"] > c["results"]["heuristic_pool"]["hi"]]
    pool_txt = ("On the training decks (140 matches) every interval overlaps the control's at every check."
                if not pool_disjoint else
                f"On the training decks {sorted(set(pool_disjoint))} separate from the control at least once.")
    fl = [ARM_LABEL[a].split()[-1] for a in sep]

    def floors(arms_):  # ["f10", "f20"] -> "floors 0.10 and 0.20"
        v = [ARM_LABEL[a].split()[-1] for a in arms_]
        return "" if not v else f"floor {v[0]}" if len(v) == 1 else f"floors {', '.join(v[:-1])} and {v[-1]}"
    what = ("Card floors of " + " and ".join(fl)) if len(fl) > 1 else (f"A card floor of {fl[0]}" if fl else "")
    title = (f"{what} made the agent weaker against the scripted bot on held-out decks" if fl
             else "No card floor separates from the control against the scripted bot")
    fig, top, bottom = frame(
        4.1,
        title,
        "We copied the agent at 10.0B steps and trained four copies for 1.0B more steps: one as before (the "
        "control) and three with a card floor, where every legal card keeps at least a 5%, 10% or 20% chance "
        "of being picked. Scores are against the scripted bot, played without the floor; bands are 95% "
        "intervals.",
        foot=f"Over the last three checks the held-out intervals of {floors(sep) or 'no floor'} sit below the "
             f"control's" + (f"; {floors(near)} {'overlaps' if len(near) == 1 else 'overlap'} it at least once"
                              if near else "")
             + f". {pool_txt}")
    H = fig.get_figheight()
    y0, h = bottom + 0.62 / H, top - bottom - 1.02 / H
    axL = fig.add_axes([0.085, y0, 0.445, h])
    axR = fig.add_axes([0.765, y0, 0.21, h], sharey=axL)
    panels = [(axL, "heuristic_heldout", "held-out decks (1,040 matches per point)"),
              (axR, "heuristic_pool", "training decks (140)")]
    for ax, key, ttl in panels:
        ax.grid(axis="y")
        ends = []
        for a in arms:
            rows_ = data[a]
            x = np.array([r["global_step"] for r in rows_]) / 1e9
            y = np.array([r["results"][key]["score"] for r in rows_])
            lo = np.array([r["results"][key]["lo"] for r in rows_])
            hi = np.array([r["results"][key]["hi"] for r in rows_])
            col = ARM_COLOR[a]
            ax.fill_between(x, lo, hi, color=col, alpha=0.10 if key == "heuristic_heldout" else 0.06, lw=0,
                            zorder=1)
            ax.plot(x, y, color=col, lw=2, marker=ARM_MARKER[a], ms=5.5, mec="white", mew=0.8, zorder=3,
                    label=ARM_LABEL[a])
            ends.append((y[-1], a, x[-1]))
        ax.set_title(ttl, loc="left", fontsize=12, color=INK, pad=6)
        ax.set_xlim(9.97, 10.93)
        ax.set_xticks([10.0, 10.2, 10.4, 10.6, 10.8] if ax is axL else [10.0, 10.4, 10.8])
        ax.set_xlabel("training steps (billions)")
        if ax is axL:
            ly = repel([e[0] for e in ends], 0.0115)
            for (yv, a, xe), yl in zip(ends, ly):
                ax.text(xe + 0.045, yl, f"{yv:.3f} {ARM_LABEL[a].split(' (')[0]}", fontsize=11, va="center",
                        ha="left", color=INK)
                ax.plot([xe + 0.01, xe + 0.04], [yv, yl], color=MUTED, lw=0.8, clip_on=False)
    axL.set_ylim(0.78, 1.0)
    axL.set_yticks(np.arange(0.80, 1.001, 0.05))
    axL.set_ylabel("score against the scripted bot")
    plt.setp(axR.get_yticklabels(), visible=False)
    axL.legend(loc="lower left", ncol=2, handlelength=2.2, columnspacing=1.0, borderaxespad=0.3)
    save(fig, out, "floor_ablation.png")
    for a in arms:
        r = data[a][-1]["results"]
        s0, s1 = data[a][0]["global_step"] / 1e9, data[a][-1]["global_step"] / 1e9
        print(f"  {a}: {len(data[a])} evals {s0:.4f}-{s1:.4f}B;"
              f" last held-out {r['heuristic_heldout']['score']:.4f} [{r['heuristic_heldout']['lo']:.3f},"
              f"{r['heuristic_heldout']['hi']:.3f}], pool {r['heuristic_pool']['score']:.4f} "
              f"[{r['heuristic_pool']['lo']:.3f},{r['heuristic_pool']['hi']:.3f}]")
    print(f"  separated from control at 10.7-10.9B (held-out): {sep}; overlapping at least once: {near}")
    if branch:
        print(f"  branch: {json.dumps({k: branch[k] for k in list(branch)[:6]})[:300]}")


# ----------------------------------------------------------------------------- 6. human play
def fig_human_play(runs, out, chk):
    model = "league/S4_A_s1/model_610352.pt"
    files = sorted(glob.glob(f"{runs}/human_play/matches/*.json"))
    recs, other = [], Counter()
    for f in files:
        d = load_json(f)
        s = d["summary"]
        if not d["opponent"]["checkpoint"].endswith(model):
            other[os.path.basename(os.path.dirname(d["opponent"]["checkpoint"]))] += 1
            continue
        # Only the fields needed for aggregates are kept; names and ids never leave this function.
        recs.append(dict(t=datetime.fromisoformat(s["started"]), result=s["result"], ranked=s["ranked"],
                         abandoned=s["abandoned"], advisor=s["advisor_used"], crowns=s["crowns"],
                         end=s["end_reason"], player=s["player_id"], greedy=d["opponent"]["greedy"]))
    ranked = sorted([r for r in recs if r["ranked"] and r["result"] in (-1, 0, 1)], key=lambda r: r["t"])
    n = len(ranked)
    if n == 0:
        print("  no ranked matches; human_play.png skipped")
        return
    wins = sum(r["result"] == 1 for r in ranked)
    draws = sum(r["result"] == 0 for r in ranked)
    losses = n - wins - draws
    k = wins + 0.5 * draws
    lo, hi = wilson(k, n)
    players = len({r["player"] for r in ranked})
    top2 = sum(c for _, c in Counter(r["player"] for r in ranked).most_common(2))  # an aggregate count only
    king = sum(r["result"] == -1 and r["end"] == "KING" for r in ranked)  # King Tower destroyed = 3 crowns
    unranked = len(recs) - n
    abandoned = sum(r["abandoned"] for r in recs)
    advisor = sum(r["advisor"] and not r["abandoned"] for r in recs)
    rest = unranked - abandoned - advisor
    days = sorted({r["t"].date() for r in ranked})
    span = (f"{days[0].day}-{days[-1].day} {days[-1]:%b %Y}" if days[0].month == days[-1].month
            else f"{days[0]:%d %b} - {days[-1]:%d %b %Y}")

    fig, top, bottom = frame(
        3.6,
        f"Humans won {wins} of {n} ranked matches against the agent",
        f"People played the final agent in the browser, {span}: {n} ranked matches (finished, at real-time "
        f"speed, with no hints) by {players} players. The agent won {losses}, {king} of them by destroying the "
        f"King Tower.",
        foot=f"Not counted: {unranked} unranked matches ({abandoned} abandoned, {advisor} with hints on"
             + (f", {rest} other" if rest else "") + f"). The interval treats matches as independent, although "
             f"the {min(2, players)} most active players played {top2} of the {n}.")
    H = fig.get_figheight()
    y0, h = bottom + 0.55 / H, top - bottom - 0.98 / H
    # left: overall human score with interval
    a1 = fig.add_axes([0.10, y0, 0.13, h])
    a1.axhline(0.5, color=AXIS, lw=1.2)
    a1.text(0.0, 0.53, "even", ha="center", va="bottom", fontsize=11, color=INK2)
    a1.plot([0, 0], [lo, hi], color=VERM, lw=2.2, zorder=3)
    a1.plot(0, k / n, marker="o", ms=10, color=VERM, mec="white", mew=1.2, zorder=4)
    a1.text(0.13, k / n, f"{k / n:.2f}", ha="left", va="center", fontsize=12.5, fontweight="bold", color=INK)
    a1.text(0.13, k / n - 0.085, f"[{lo:.2f}, {hi:.2f}]", ha="left", va="center", fontsize=11, color=INK2)
    a1.set_xlim(-0.35, 0.75)
    a1.set_ylim(0, 1)
    a1.set_xticks([0])
    a1.set_xticklabels(["all\nranked"])
    a1.tick_params(axis="x", length=0)
    a1.set_ylabel("human score")
    a1.set_title("score, 95% CI", loc="left", fontsize=12, color=INK, pad=6)
    # right: crown difference per ranked match, in time order
    a2 = fig.add_axes([0.355, y0, 0.625, h])
    xs = np.arange(1, n + 1)
    diff = np.array([r["crowns"][0] - r["crowns"][1] for r in ranked])
    cols = [VERM if r["result"] == 1 else (MUTED if r["result"] == 0 else BLUE) for r in ranked]
    a2.axhline(0, color=AXIS, lw=1)
    a2.grid(axis="y")
    a2.bar(xs, diff, width=0.62, color=cols, zorder=3)
    # day separators and labels (centred over each day's matches)
    groups = []
    for i, r in enumerate(ranked):
        if not groups or groups[-1][0] != r["t"].date():
            groups.append([r["t"].date(), i + 1, i + 1])
        groups[-1][2] = i + 1
    for g, (dd, i0, i1) in enumerate(groups):
        if g:
            a2.axvline(i0 - 0.5, color=AXIS, lw=0.8, zorder=1)
        a2.text((i0 + i1) / 2, 3.35, f"{dd:%b} {dd.day}", ha="center", va="bottom", fontsize=10, color=INK2)
    a2.set_xlim(0.3, n + 0.7)
    a2.set_ylim(-3.4, 3.9)
    a2.set_yticks(range(-3, 4))
    a2.set_xticks([1] + list(range(5, n + 1, 5)))
    a2.set_xlabel("ranked match, in the order played")
    a2.set_ylabel("crowns: human minus agent")
    a2.set_title("each ranked match", loc="left", fontsize=12, color=INK, pad=6)
    handles = [Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=VERM)]
    a2.legend(handles, [f"agent won ({losses})", f"human won ({wins})"], loc="upper left",
              bbox_to_anchor=(0.0, 0.90), ncol=1, handlelength=1.0, borderaxespad=0.2)
    save(fig, out, "human_play.png")
    print(f"  {len(files)} match files, {len(recs)} vs {model} ({sum(other.values())} other opponents "
          f"{dict(other)}); ranked {n}: W {wins} D {draws} L {losses}; human score {k / n:.4f} "
          f"[{lo:.4f},{hi:.4f}]; players {players}; policy King-Tower wins {king}; unranked {unranked} "
          f"(abandoned {abandoned}, advisor {advisor}, other {rest}); greedy "
          f"{sorted({r['greedy'] for r in recs})}; days {[str(d) for d in days]}")



CLAUDE_MODELS = (("claude-fable-5-2", "Fable 5.2"), ("claude-opus-5-5", "Opus 5.5"), ("claude-sonnet-5-5", "Sonnet 5.5"),
                 ("claude-haiku-5-5", "Haiku 5.5"))
DRAW_GRAY = "#c4c4c4"  # lighter than MUTED so a draw cell separates from BLUE and VERM by lightness too
SETUPS = (("xbow", True, "X-Bow\nmirror"), ("xbow", False, "X-Bow\nvs other"),
          ("royal_hogs", True, "Royal Hogs\nmirror"), ("royal_hogs", False, "Royal Hogs\nvs other"))


def claude_games(runs):
    """Every finished Claude-vs-policy game against the released model, as one row per game (Claude's view).
    Folders: runs/claude_games/S4_A_s1_10B/<model>[_<suffix>]/llm/ (one model may use several folders)."""
    base = os.path.join(runs, "claude_games", "S4_A_s1_10B")
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "claude-*", "llm", "state.json"))):
        folder = os.path.basename(os.path.dirname(os.path.dirname(d)))
        m = re.match(r"(claude-[a-z]+-\d+-\d+)", folder)
        slug = m.group(1)
        for k, g in load_json(d)["games"].items():
            if g.get("status") != "done":
                continue
            gdir = os.path.join(os.path.dirname(d), f"game_{int(k):03d}")
            meta = load_json(os.path.join(gdir, "meta.json"))
            cc, pc = int(g["claude_crowns"]), int(g["policy_crowns"])
            c = g["config"]
            rows.append(dict(
                model=slug, k=int(k), claude_deck=c["claude_deck"], policy_deck=c["policy_deck"],
                mirror=bool(c["mirror"]), claude_team=int(c["claude_team"]), seed=int(meta["seed"]),
                claude_crowns=cc, policy_crowns=pc,
                # the draw rule: equal crowns at the end = draw (the engine's tower-HP tiebreak is ignored)
                result="W" if cc > pc else "L" if cc < pc else "D",
                engine_result={1: "claude", -1: "policy", 0: "draw"}[int(g["result"])],
                end_reason=g["end_reason"], end_tick=int(g["ticks"]), decisions=int(g["decisions"]),
                plays=int(g["plays"]), illegal=int(g["illegal"]), policy_step=int(g["policy"]["global_step"]),
                replay=os.path.relpath(os.path.join(gdir, f"game_{int(k):03d}.gif"), base)))
    return rows


def fig_claude_games(runs, out, chk):
    rows = claude_games(runs)
    if not rows:
        print("  no Claude games found; claude_games.png skipped")
        return
    names = dict(CLAUDE_MODELS)
    models = [m for m, _ in CLAUDE_MODELS if any(r["model"] == m for r in rows)]
    by = {(r["model"], r["k"]): r for r in rows}
    assert len(by) == len(rows), "a game appears twice"
    ks = sorted({r["k"] for r in rows})
    per = len(ks)
    assert all((m, k) in by for m in models for k in ks), "every model must have played the same games"
    for k in ks:  # the same setup for every model: seed, decks, side
        assert len({(by[m, k]["seed"], by[m, k]["claude_deck"], by[m, k]["policy_deck"], by[m, k]["claude_team"])
                    for m in models}) == 1, f"game {k} differs between models"
    steps = {r["policy_step"] for r in rows}
    assert len(steps) == 1, steps
    n = len(rows)
    tot = Counter(r["result"] for r in rows)
    score = {}
    for m in models:
        c = Counter(by[m, k]["result"] for k in ks)
        s = c["W"] + 0.5 * c["D"]
        score[m] = (c, s / per, *wilson(s, per))
    wins = [r for r in rows if r["result"] == "W"]
    mirror_x = [r for r in wins if r["claude_deck"] == "xbow" and r["mirror"]]
    one_nil = all((r["claude_crowns"], r["policy_crowns"]) == (1, 0) for r in wins)

    best = max(models, key=lambda m: score[m][1])
    fig, top, bottom = frame(
        3.15,
        f"The agent outscored all {WORDS.get(len(models), len(models)).lower()} Claude models; "
        f"{len(mirror_x)} of Claude's {len(wins)} wins came in X-Bow mirrors",
        f"{n} games against the final agent ({next(iter(steps)) / 1e9:.1f}B steps), {per} per model, and every "
        f"model got the same {per} setups (seed, decks, side). Claude reads a text description of the board and "
        f"answers once per game second while the game waits; the agent acts every half second.",
        foot=f"Claude's score = (wins + ½ draws) / games, with 95% intervals. A draw means equal crowns at the "
             f"end (the engine's tower-HP tiebreak is ignored)."
             + (f" Every Claude win was 1-0." if one_nil else "")
             + " Claude ran through the Claude Code command line at low effort.")
    H = fig.get_figheight()
    y0, h = bottom + 0.88 / H, top - bottom - 1.56 / H
    ys = {m: len(models) - 1 - i for i, m in enumerate(models)}
    # left: Claude's score per model, with interval
    a1 = fig.add_axes([0.155, y0, 0.215, h])
    a1.axvline(0.5, color=AXIS, lw=1.2)
    for m in models:
        c, sc, lo, hi = score[m]
        a1.plot([lo, hi], [ys[m]] * 2, color=VERM, lw=2.2, zorder=3)
        a1.plot(sc, ys[m], marker="o", ms=9, color=VERM, mec="white", mew=1.2, zorder=4)
        a1.text(hi + 0.03, ys[m], f"{sc:.2f}", ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)
    a1.set_xlim(0, 1)
    a1.set_ylim(-0.6, len(models) - 0.4)
    a1.set_yticks([ys[m] for m in models])
    a1.set_yticklabels([names[m] for m in models])
    a1.tick_params(axis="y", length=0)
    a1.set_xticks([0, 0.5, 1])
    a1.set_xlabel("Claude's score")
    a1.set_title("score, 95% CI", loc="left", fontsize=12, color=INK, pad=6)
    # right: every game, grouped by setup (columns = games in order within a group)
    a2 = fig.add_axes([0.43, y0, 0.47, h])
    col = {"W": VERM, "L": BLUE, "D": DRAW_GRAY}
    x, gap, centres, block = 0.0, 0.9, [], []
    for deck, mir, label in SETUPS:
        gk = [k for k in ks if by[models[0], k]["claude_deck"] == deck and by[models[0], k]["mirror"] == mir]
        if not gk:
            continue
        x0 = x
        for k in gk:
            for m in models:
                a2.add_patch(Rectangle((x + 0.07, ys[m] - 0.36), 0.86, 0.72, color=col[by[m, k]["result"]], lw=0))
            x += 1
        sub = [by[m, k] for m in models for k in gk]
        cs = Counter(r["result"] for r in sub)
        centres.append(((x0 + x) / 2, label, (cs["W"] + 0.5 * cs["D"]) / len(sub), len(gk)))
        block.append((deck, mir, cs, len(sub)))
        x += gap
    xmax = x - gap
    for cx, label, s, ng in centres:
        a2.text(cx, len(models) - 0.45, label, ha="center", va="bottom", fontsize=10.5, color=INK2, linespacing=1.1)
        a2.text(cx, -0.75, f"{math.floor(s * 100 + 0.5) / 100:.2f}", ha="center", va="top", fontsize=11,
                fontweight="bold", color=INK)  # half up: 3/24 = 0.125 -> 0.13
    a2.text(xmax / 2, -1.28, "Claude's score in each setup", ha="center", va="top", fontsize=10.5, color=INK2)
    for m in models:
        c = score[m][0]
        a2.text(xmax + 0.35, ys[m], f"{c['W']}-{c['D']}-{c['L']}", ha="left", va="center", fontsize=10.5,
                color=INK2)
    a2.text(xmax + 0.35, len(models) - 0.45, "Claude\nW-D-L", ha="left", va="bottom", fontsize=10.5, color=INK2,
            linespacing=1.1)
    a2.set_xlim(-0.1, xmax + 0.1)
    a2.set_ylim(-0.6, len(models) - 0.4)
    a2.axis("off")
    handles = [Rectangle((0, 0), 1, 1, color=VERM), Rectangle((0, 0), 1, 1, color=DRAW_GRAY),
               Rectangle((0, 0), 1, 1, color=BLUE)]
    a2.legend(handles, [f"Claude won ({tot['W']})", f"draw ({tot['D']})", f"agent won ({tot['L']})"],
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, handlelength=1.0, borderaxespad=0.2,
              columnspacing=1.2)
    a2.set_title("every game, grouped by setup", loc="left", fontsize=12, color=INK, pad=36)
    save(fig, out, "claude_games.png")
    # the per-game table behind the figure (also read by the play site's "Games against language models")
    summ = {m: dict(name=names[m], games=per, w=score[m][0]["W"], d=score[m][0]["D"], l=score[m][0]["L"],
                    score=round(score[m][1], 4), ci=[round(score[m][2], 4), round(score[m][3], 4)]) for m in models}
    s_all = tot["W"] + 0.5 * tot["D"]
    lo, hi = wilson(s_all, n)
    data = dict(policy=dict(run="S4_A_s1", file="model_610352.pt", global_step=next(iter(steps))),
                draw_rule="equal crowns at the end = draw (the engine's tower-HP tiebreak is ignored)",
                score="Claude's (wins + draws/2) / games, 95% Wilson interval",
                models=summ, total=dict(games=n, w=tot["W"], d=tot["D"], l=tot["L"], score=round(s_all / n, 4),
                                        ci=[round(lo, 4), round(hi, 4)]),
                setups=[dict(claude_deck=d_, mirror=mi, games=ng_, w=cs_["W"], d=cs_["D"], l=cs_["L"],
                             score=round((cs_["W"] + 0.5 * cs_["D"]) / ng_, 4)) for d_, mi, cs_, ng_ in block],
                games=sorted(rows, key=lambda r: (models.index(r["model"]), r["k"])))
    with open(os.path.join(out, "claude_games.json"), "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print(f"wrote {os.path.join(out, 'claude_games.json')}")
    print(f"  {n} games, {len(models)} models x {per}; Claude W {tot['W']} D {tot['D']} L {tot['L']} = "
          f"{s_all / n:.4f} [{lo:.4f},{hi:.4f}]; " + "; ".join(
              f"{names[m]} {score[m][0]['W']}-{score[m][0]['D']}-{score[m][0]['L']} {score[m][1]:.2f} "
              f"[{score[m][2]:.2f},{score[m][3]:.2f}]" for m in models)
          + f"; wins in X-Bow mirrors {len(mirror_x)}/{len(wins)}; all 1-0: {one_nil}; best {names[best]}")
    # the selection log states each model's policy-side W-D-L and score for the 100 games
    m_ = chk.find(r"All 100 games \(25 per model\): (\d+) W (\d+) D (\d+) L, ([\d.]+)")
    if m_:
        for i, key in enumerate(("L", "D", "W")):  # the log counts from the policy's side
            chk.cmp(f"claude games: policy {'WDL'[i]}", tot[key], m_.group(i + 1))
        chk.cmp("claude games: policy score", 1 - s_all / n, m_.group(4))


FIGURES = {
    "training_curve": fig_training_curve,
    "selection_br": fig_selection_br,
    "stage1_round_robin": fig_stage1_rr,
    "floor_ablation": fig_floor_ablation,
    "stage5_round_robin": fig_stage5_rr,
    "human_play": fig_human_play,
    "claude_games": fig_claude_games,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pr = os.environ.get("PR_ROOT")
    ap.add_argument("--runs", default=os.path.join(pr, "runs") if pr else None)
    ap.add_argument("--doc", default=os.path.join(HERE, "..", "STAGE4_SELECTION.md"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=None)
    a = ap.parse_args()
    if not a.runs or not os.path.isdir(a.runs):
        raise SystemExit("results directory not found: pass --runs or set PR_ROOT (results live in $PR_ROOT/runs "
                         "on the cluster; they are not part of the repository)")
    chk = DocCheck(os.path.normpath(a.doc))
    for name in a.only or FIGURES:
        print(f"[{name}]")
        FIGURES[name](a.runs, a.out, chk)
    chk.report()


if __name__ == "__main__":
    main()
