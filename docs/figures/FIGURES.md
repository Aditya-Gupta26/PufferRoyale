# README figures

All six PNGs are made by `make_figures.py` (next to this file) from result files in `$PR_ROOT/runs`
(`/scratch/$USER/pufferroyale/runs`). Those results are not in the repository.

```bash
source /scratch/$USER/pufferroyale/env.sh       # PR_ROOT, SIF
apptainer exec --no-home --bind $PR_ROOT:$PR_ROOT $SIF python docs/figures/make_figures.py
# one figure only:  ... make_figures.py --only human_play
```

It runs in about 10 s in one process on a login node, using the image's matplotlib 3.11.2 and numpy.
At the end it prints a check of every number the figures share with `docs/STAGE4_SELECTION.md`.

**Last run (2026-11-09):** 304 numbers match, 0 mismatches, 0 tables or sentences not found
(stage 1: 108 values incl. checkpoint paths and steps; stage 2: 99; stage 3b: 8; stage 4a: 56; stage 5: 32;
the stage-3 winner: 1).

**Style.** All figures share one style:
- 8 in wide at 150 dpi (1200 px), on a white background.
- Text is 11 pt or larger; the title states the takeaway.
- Colours are Okabe-Ito. Blue `#0072B2` always means the selected policy (S4_A_s1) and its no-floor
  line; vermillion `#D55E00` is the other side.
- The card-floor arms use one orange hue that darkens as the floor grows (`#f39d5e`, `#c66000`, `#7c3c00`).
- The colours were checked with a Python port of the dataviz validator (Machado 2009 CVD simulation, OKLab),
  over every pair of series within each figure:
  - smallest colour-blind difference ΔE 11.0 (training curve, vermillion vs green; target ≥ 8);
  - smallest normal-vision difference ΔE 17.0 (floor arms; floor ≥ 15);
  - the floor ramp also passes the ordinal checks (monotone lightness, one hue).
  Sky blue and the lightest orange are below 3:1 contrast, so those lines are also labelled directly.
- Intervals are 95% Wilson intervals on (wins + ½ draws) / matches, as in `hpc/eval_watch.py`.

---

## training_curve.png

**Caption:** Against the scripted bot, S4_A_s1's score rises from 0.74 at 0.1B steps to 0.94 at 9.9B. Against
the two stage-3 Hog-mirror specialists it stays far below even.

- **Data:**
  - `runs/league/S4_A_s1/eval_history.jsonl`: 99 evaluations, one every 0.1B steps from 0.100B to 9.901B,
    written by the eval sidecar (`hpc/eval_watch.py`).
  - Opponents and match counts come from `runs/league/S4_A_s1/eval_watch.log`:
    - heuristic bot, 7 pool decks: 140 matches;
    - heuristic bot, held-out decks (xbow, royal_hogs, 50 random): 1,040 matches;
    - S3_A_s1 and S3_C_s0: 200 matches each, in the hog26 mirror.
- **Key numbers:**
  - Held-out, vs the heuristic bot: 0.741 [0.714, 0.767] at 0.1B → 0.939 [0.923, 0.952] at 9.9B.
  - Pool decks: 0.757 → 0.950.
  - S3_A_s1: 0.105 → 0.210. S3_C_s0: 0.035 → 0.075.
  - End values of the smoothed lines (5-evaluation centred mean): 0.95, 0.94, 0.18, 0.06.
  - Random bot: ≥ 0.992 at every evaluation (not drawn).
  - A vertical line marks the selection checkpoint, snap_476523, at 7.81B.
- **Notes:**
  - There is no evaluation at step 0; the first one is at 0.1B.
  - `history.jsonl` (610,352 epoch records) holds no anchor scores: 610,342 records have only self-play
    results (opponent `self`) and 10 have empty results. The curve comes from `eval_history.jsonl`.
  - S4_A_s1's training-curve numbers do not appear in STAGE4_SELECTION.md, so they could not be checked there.

## selection_br.png

**Caption:** Five candidates showed no clear exploit (selection score ≤ 0.55). S4_A_s1 won that tie on
round-robin Elo (1588), and from-scratch attackers trained for 1B steps did not exploit it either.

- **Data:**
  - Copy-start best responses (stage 2): `runs/eval/br_S4sel_<agent>_s{0,1}.json`, 18 files: W/D/L, br_score,
    1,000 matches, 150,011,904 steps, init_from_target true.
  - From-scratch best responses (stage 3b): `runs/eval/br_S4scr_{S4_A_s1,S4_C_s0_at1.55B}_s{0,1}.json`,
    1,000,013,824 steps.
  - Elo: `runs/eval/stage4_rr_20261026/summary.json`.
- **How the selection score is computed:** each seed's br_score counts as at least 0.5 (deviation 2), and the
  two seeds are averaged. Candidates within 0.05 of the lowest score are tied, and the highest Elo among them
  wins (rule 3).
- **Key numbers** (seed 0 / seed 1 → selection score, Elo):

  | candidate | seed 0 | seed 1 | score | Elo |
  |---|---|---|---|---|
  | **S4_A_s1** | 0.4845 | 0.4565 | 0.500 | 1588 |
  | S4d_A_s1 | 0.440 | 0.372 | 0.500 | 1582 |
  | S4d_A_s0 | 0.3375 | 0.3615 | 0.500 | 1533 |
  | nommd | 0.495 | 0.4655 | 0.500 | 1511 |
  | posent | 0.5045 | 0.5235 | 0.514 | 1371 |
  | S4_A_s0 | 0.4525 | 0.6605 | 0.580 | 1276 |
  | S4df20 | 0.7105 | 0.653 | 0.682 | 1367 |
  | selfonly | 0.751 | 0.6595 | 0.705 | 1623 |
  | C@1.55B (not eligible) | 0.298 | 0.3355 | 0.500 | 1648 |

  - Tie group: S4_A_s1, S4d_A_s1, S4d_A_s0, nommd, posent. Winner: **S4_A_s1**.
  - From-scratch BR scores: S4_A_s1 0.0745 / 0.166; C@1.55B 0.3485 / 0.2295.
  - Each seed's 95% interval is about ±0.03.
- **Names:**
  - nommd = S4x_nommd_s0, posent = S4x_posent_s0, selfonly = S4x_selfonly_s0, S4df20 = S4df20_A_s0, and
    C@1.55B = S4_C_s0 at snap_94446.
  - Figure 3 uses the same short names, which are the ones in the doc's stage-1 table.
- **Doc check:** the stage-2 table (W/D/L, br_score, both means, Elo), the stage-3b table and the stage-3 winner
  all match.
- **Not shown:** stage 5's best responses on the 4a arms (job 22422166). No `runs/eval/br_S5a_*.json` exists yet;
  the doc lists that job as PENDING (2026-11-09 18:51).

## stage1_round_robin.png

**Caption:** In the stage-1 round robin of the 9 candidates, no model beats all the others. Even the top-rated
C@1.55B scores only 0.38 against selfonly.

- **Data:** `runs/eval/stage4_rr_20261026/summary.json` (`round_robin`: agents, P, games, elo, nash;
  `checkpoints`).
- **What the cells show:** the row model's score against the column model, 1,800 matches per pair over the
  9 preset mirrors. Rows and columns are sorted by Elo.
- **Key numbers:**
  - Elo: C@1.55B 1648, selfonly 1623, S4_A_s1 1588, S4d_A_s1 1582, S4d_A_s0 1533, nommd 1511, posent 1371,
    S4df20 1367, S4_A_s0 1276.
  - Cycle: C@1.55B vs S4_A_s1 0.615; S4_A_s1 vs selfonly 0.507 (even); selfonly vs C@1.55B 0.624.
  - Nash mixture: S4_A_s1 0.51, selfonly 0.47, C@1.55B 0.03.
  - Cells are printed with 2 decimals; the doc's table has 3.
- **Doc check:** all 72 cells, 9 Elo values, 9 Nash weights, and the 9 checkpoint paths and step counts in the
  candidates table match.
- **Note:** "stage 1" here means stage 1 of the selection (STAGE4_SELECTION.md). It is not the Stage-1 training
  sweep in `runs/reports/stage1/`, which has no round robin.

## floor_ablation.png

**Caption:** In the stage-4a arms (S4_A_s1 branched at 10.0B, +1.0B steps each), card floors of 0.10 and 0.20
cost strength against the scripted bot on held-out decks; a floor of 0.05 is within noise of the control.

- **Data:**
  - `runs/league/S4L_A_s1_{ctrl,f05,f10,f20}/eval_history.jsonl`: 10 evaluations per arm, at the same steps,
    10.0008B to 10.9009B.
  - Branch point: `runs/league/S4L_A_s1_ctrl/BRANCH.json` (snap_610083, 9,995,599,872 steps).
- **Panels:** the left panel shows the held-out decks (1,040 matches per point); the right panel shows the
  pool decks (140 matches per point).
- **Key numbers** at the last evaluation (10.90B), held-out [95% interval] and pool:

  | arm | held-out | pool |
  |---|---|---|
  | control | 0.951 [0.936, 0.963] | 0.957 |
  | floor 0.05 | 0.944 [0.928, 0.956] | 0.943 |
  | floor 0.10 | 0.889 [0.869, 0.907] | 0.907 |
  | floor 0.20 | 0.901 [0.882, 0.918] | 0.871 |

  - At 10.7, 10.8 and 10.9B, the held-out intervals of floors 0.10 and 0.20 lie entirely below the control's.
    Floor 0.05 overlaps the control's interval at least once.
  - On the pool decks, every arm's interval overlaps the control's at every evaluation.
- **Doc check:** the sidecar table (10.70, 10.80, 10.90B) and the 05:05 entry (f20 and control at 10.90B,
  including the pool intervals) all match.
- **Not checkable:** the points at 10.0-10.6B are not stated in the doc.
- **Note:** the arms end at 10,995,941,376 steps, but the last sidecar evaluation is at 10.90B. There is no
  evaluation at 11.0B.

## stage5_round_robin.png

**Caption:** Head to head, the control without a card floor beats every floor arm. Its Elo is 1634, against
1523 (0.10), 1493 (0.05) and 1349 (0.20).

- **Data:** `runs/stage4/S5a/sub_20261109_113404/rr/summary.json`. The checkpoints are
  `league/S4L_A_s1_<arm>/snap_671139.pt`, all at 10,995,941,376 steps.
- **What the cells show:** the row arm's score, 1,800 matches per pair over 9 deck mirrors. Each pair's
  95% Wilson interval is printed once, in the upper triangle.
- **Key numbers:**
  - control vs floor 0.05: 0.679 [0.657, 0.701]
  - control vs floor 0.10: 0.686 [0.665, 0.707]
  - control vs floor 0.20: 0.819 [0.801, 0.836]
  - floor 0.10 vs floor 0.05: 0.573 (that is, floor 0.05 vs floor 0.10 is 0.427 [0.404, 0.450])
  - floor 0.10 vs floor 0.20: 0.734 [0.713, 0.754]
  - floor 0.05 vs floor 0.20: 0.713 [0.692, 0.734]
  - Elo: control 1634, floor 0.10 1523, floor 0.05 1493, floor 0.20 1349. Nash: control 1.00.
- **Doc check:** every value matches the doc's stage-5 table (cells, Wilson intervals, Elo, Nash) and the
  expected values given in the request.
- **Note:** under rule 5 this round robin is only the tie-break. The deciding stage-5 BR results did not exist
  when the figure was made, and its footnote says so; that sentence is computed from `runs/eval/br_S5a_*.json`.

## human_play.png

**Caption:** In browser play against S4_A_s1's final model (10.0B steps), humans won 3 of 20 ranked matches,
a human score of 0.15 [0.05, 0.36]. The policy won 17, 8 of them by destroying the King Tower.

- **Data:**
  - `runs/human_play/matches/*.json`: 27 files, all against `league/S4_A_s1/model_610352.pt` with sampled
    actions (greedy false).
  - Only these `summary` fields are used: started, result, ranked, abandoned, advisor_used, crowns,
    end_reason. `player_id` is used only to count distinct players.
- **What counts as ranked:** `summary.ranked`, which means finished, 1× speed and the advisor never on
  (HUMAN_PLAY.md, "Leaderboard").
- **Key numbers:**
  - 20 ranked matches from 6 to 9 Nov 2026, by 4 players. The two most active players played 16 of the 20.
  - Results: 3 human wins, 0 draws, 17 losses. Crown differences: +2, +1, +2 for the wins; −3 seven times.
  - Not counted: 7 unranked matches (4 abandoned, 3 with the advisor on).
- **Privacy:** no name, player id or per-player result appears in the figure.
- **Caveats:**
  - The interval treats matches as independent, although a few players played most of them.
  - The figure shows whatever is in `matches/` when the script runs, and the live site keeps adding matches.
  - These numbers are not in STAGE4_SELECTION.md. HUMAN_PLAY.md confirms that the site serves
    model_610352.pt (10.0B).

---

## Which checkpoint is which

These are all the same run (S4_A_s1, seed 1) at different steps:

| use | checkpoint | steps |
|---|---|---|
| Selection (figures 2, 3) | snap_476523 | 7.81B |
| Branch point of stage 4a (figure 4) | snap_610083 | 9.9956B |
| Human-play opponent (figure 6) | model_610352 | 10.0B, end of training |
| Stage-5 arms (figure 5) | snap_671139 | 10.996B |

## Data not used

- `runs/stage3/` holds the earlier Hog-mirror study (E1). It is not part of the selection pipeline, so no figure
  uses it.
- The selection's stage 3 is a decision, not a data set. It is computed from the stage-2 files (figure 2).
- Stage 4b (the from-scratch floor runs S4f05/S4f10/S4f20) is still training and is not plotted.
