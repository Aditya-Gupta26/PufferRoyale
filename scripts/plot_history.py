#!/usr/bin/env python3
"""Plot a league run's history.jsonl (SPEC §19.7.10): the learner's score per opponent and the
training losses against the global (learner) step.

    python scripts/plot_history.py experiments/league/<run>                # -> <run>/history.png
    python scripts/plot_history.py experiments/league/<run> --out curves.png --window 20

Score panel: per opponent, the learner's score over the last --window epochs (games-weighted
rolling mean; opponents are labelled like the league's progress lines). Loss panels: policy and
value loss, the entropies (joint, card, position), approx KL / clip fraction, the MMD KL and the
explained variance. Needs matplotlib (pip install matplotlib, or the 'plots' extra:
pip install -e '.[plots]').
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LOSS_PANELS = (("policy / value loss", ("policy_loss", "value_loss")),
               ("entropy", ("entropy", "entropy_card", "entropy_pos")),
               ("KL / clipping", ("approx_kl", "clipfrac")),
               ("MMD KL", ("mmd_kl",)),
               ("explained variance", ("explained_variance",)))


def read_history(run):
    path = os.path.join(run, "history.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"plot_history.py: no history.jsonl in {run}")
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"plot_history.py: {path} is empty")
    return recs


def short(spec):
    return f"ckpt:{os.path.basename(spec[5:])}" if spec.startswith("ckpt:") else spec


def score_series(recs, window):
    """{opponent: (steps, rolling score)} -- games-weighted mean over the last `window` epochs that
    played that opponent."""
    out = {}
    hist = {}
    for r in recs:
        for spec, v in (r.get("results") or {}).items():
            h = hist.setdefault(spec, [])
            h.append((int(v["games"]), float(v["score"]) * int(v["games"])))
            del h[:-window]
            g = sum(x for x, _ in h)
            steps, vals = out.setdefault(spec, ([], []))
            steps.append(int(r["global_step"]))
            vals.append(sum(s for _, s in h) / g if g else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", metavar="RUN_DIR", help="league run directory (holding history.jsonl)")
    ap.add_argument("--out", help="PNG path (default RUN_DIR/history.png)")
    ap.add_argument("--window", type=int, default=10, help="epochs in the rolling score (default 10)")
    args = ap.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("plot_history.py needs matplotlib: pip install matplotlib "
                         "(or the 'plots' extra: pip install -e '.[plots]')") from None
    run = os.path.abspath(args.run)
    recs = read_history(run)
    out = args.out or os.path.join(run, "history.png")
    steps = [int(r["global_step"]) for r in recs]
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(grid[0, :])
    for spec, (xs, ys) in sorted(score_series(recs, max(1, args.window)).items()):
        ax.plot(xs, ys, label=short(spec), linewidth=1.5)
    ax.axhline(0.5, color="0.6", linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("global step (learner)")
    ax.set_ylabel(f"learner score (last {args.window} epochs)")
    ax.set_title(f"{os.path.basename(run)}: learner score per opponent")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8, ncol=2)
    for k, (title, keys) in enumerate(LOSS_PANELS):
        a = fig.add_subplot(grid[1 + k // 3, k % 3])
        for key in keys:
            ys = [(r.get("losses") or {}).get(key) for r in recs]
            pts = [(x, y) for x, y in zip(steps, ys) if y is not None]
            if pts:
                a.plot([p[0] for p in pts], [p[1] for p in pts], label=key, linewidth=1.0)
        a.set_title(title, fontsize=10)
        a.set_xlabel("global step", fontsize=8)
        a.tick_params(labelsize=8)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out} ({len(recs)} epochs)")


if __name__ == "__main__":
    main()
