# Stage-4 model selection and floor ablation (started 2026-10-26, user's request)

The user's plan (2026-10-26): a round robin of the current Stage-4 models, then a best response (BR) against
each one to measure its exploitability, then pick the single best model by exploitability and ablate ONLY that
one (continue from its latest snapshot, retrain from scratch, and vary the card-probability floor). One metric
decides everything: exploitability. Each stage starts only after the previous one's results are written here.

Everything in "Rules" was written on 2026-10-26 before any result of stages 2-5 existed. A change after that is
logged under "Deviations" with its reason.

## Stages and status

| stage | what | jobs | status |
|---|---|---|---|
| 1 | Round robin of the 9 current models | 21002053 (CPU array 0-3) | **done** 2026-10-26 |
| 2 | Exploitability: 2 BR seeds per candidate | 21005306 (GPU array 0-17%4); its tasks 0-5 failed and rerun as 21343356 (array 0-5%2, deviation 1) | **done** 2026-10-31 (18 of 18) |
| 3 | Select the winner (rule below) | – | **done** 2026-11-01: **S4_A_s1** (deviation 2) |
| 3b | From-scratch attackers (1B) vs S4_A_s1 and C@1.55B | 21630359 (GPU array 0-3%4, H100/H200); A100 copy 21778800 (array 0-3%4, deviation 3) | **done** 2026-11-05: S4_A_s1 0.0745 / 0.166 (score 0.500), C@1.55B 0.3485 / 0.2295 (0.500, reported only): **S4_A_s1 stays the winner** |
| 4a | Winner's config, branched from its latest snapshot, 4 floors | S4L_A_s1_{ctrl,f05,f10,f20}: 22014154/56, 22014157/58, 22014161/62, 22014164/65 (2 links each, deviation 4) | **done** 2026-11-09 (all four at snap_671139, 10.996B) |
| 4b | Winner's config from scratch, floors not yet covered | S4f05_A_s1 22014171/78/80/83, S4f10_A_s1 22014197/98, 22014202/03, S4f20_A_s1 22014245, 22014284/85/86 (4 links each) | **done** 2026-11-12 11:18 (all three at snap_152640, 2,500,853,760 steps) |
| 5 | Exploitability (and a round robin) of every stage-4 arm; verdict | 4a: BR 22422166 (GPU array 0-7%4), round robin 22422168 (CPU array 0-1); 4b: dry run `runs/stage4/S5b/dryrun/` | **done** 2026-11-14: **control in both families** (4a S4L_A_s1_ctrl snap_671139; 4b S4_A_s1 snap_152640); 4b: BR 22933924, round robin 22933925 |

## Candidates (the stage-1 checkpoints; stage 2 uses the same files)

| agent | checkpoint | steps |
|---|---|---|
| S4_A_s0 | league/S4_A_s0/snap_404496.pt | 6.63B |
| S4_A_s1 | league/S4_A_s1/snap_476523.pt | 7.81B |
| S4d_A_s0 | league/S4d_A_s0/snap_325314.pt | 5.33B |
| S4d_A_s1 | league/S4d_A_s1/snap_321498.pt | 5.27B |
| S4df20_A_s0 | league/S4df20_A_s0/snap_140715.pt | 2.31B |
| S4x_nommd_s0 | league/S4x_nommd_s0/snap_166473.pt | 2.73B |
| S4x_selfonly_s0 | league/S4x_selfonly_s0/snap_176490.pt | 2.89B |
| S4x_posent_s0 | league/S4x_posent_s0/snap_159318.pt | 2.61B |
| S4_C_s0@1.55B | league/S4_C_s0/snap_94446.pt | 1.55B (reference only, see rule 3) |

The models were trained for different lengths (1.55B to 7.81B steps): the selection ranks the models as they are
now, not the methods at equal budgets.

## Rules

1. **Round robin (stage 1).** `hpc/stage4_eval.py`'s rr part with the checkpoints above fixed by
   `tmp/rr_s4/plan_rr.py`: the 9 presets, each a mirror, 200 matches per pair and deck, half per seat, sampled
   actions, no card floor. Output: payoff matrix, Elo, Nash mixture (`runs/eval/stage4_rr_20261026`). The C-vs-A
   line of that summary is meaningless here (it pools mixed pairs) and is ignored.
2. **Exploitability (stage 2).** `scripts/best_response.py` as E1 (stage 3) except the decks: `--init-from-target`,
   150M BR steps (E1's calibrated budget; not re-calibrated), 256 envs, 1000 evaluation matches half per seat,
   sampled actions, BR seeds 0 and 1, flags `--train.torch-deterministic= --env.reward-tower 0.3
   --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0 --shaping-anneal-frac 0.5 --env-threads 16
   --fast-opponents --cuda-graphs`. **Decks: `--deck0 random --deck1 random`** (a fresh random deck per side and
   match, in training and evaluation). Reason: the candidates trained on 7 pool decks plus 50% random decks;
   a single mirror would measure one matchup only, and in the Hog mirror the stage-3 specialists S3_A_s1 / S3_C_s0
   already score 0.78-0.90 / 0.61-0.97 against the candidates (their latest sidecar evals, 200 matches each), so a
   Hog-mirror BR would likely saturate near 1 for all of them. The BR learner
   has no card floor (league_train DEFAULTS card_floor = 0). Check, not a rule: from W&B, whether each BR's
   training score has levelled off by 150M.
3. **Selection (stage 3).** Score of a candidate = the mean of its two BRs' `br_score` (lower = less exploitable).
   Candidates within 0.05 of the lowest score are tied; among tied candidates the highest round-robin Elo wins.
   S4_C_s0@1.55B is measured but not eligible: the C configuration collapsed in all four C runs and in the S4x
   control, so it cannot be trained further or from scratch.
4. **Ablation (stage 4), on the winner only.** Arms: a control and constant floors 0.05, 0.10, 0.20
   (`card_floor_steps = 0`; the floor must be < 0.25). The control keeps the winner's own floor schedule unchanged
   (S4_A_*: 0.005 faded to 0 by 5B, so 0 after 5B; S4d_*: 0.005 constant; S4x_*: 0.005 fading to 0 by 5B;
   S4df20: 0.20 constant, which then also stands for the 0.20 arm).
   - 4a, from latest: `hpc/branch_run.py` at the winner's checkpoint, `--set card_floor=X --set
     card_floor_steps=0` (the control branch has no `--set`), one 20 h link each.
   - 4b, from scratch: the winner's run configuration with floor X from random init (same seed), to 2.5B steps; the control is the winner's own run up to 2.5B (for a branched winner, S4x_*, which did not
     start from scratch: a new from-scratch run of its configuration with its own floor schedule). An
     existing run with exactly that configuration and floor counts as the arm (e.g. S4d_A_s0 for 0.005 and
     S4df20_A_s0 for 0.20 if the winner is S4d_A_s0's configuration).
5. **Verdict (stage 5).** Each arm's checkpoint at the end of its budget (4a: its last snapshot; 4b: the snapshot
   nearest 2.5B) gets the stage-2 BR protocol (2 seeds) and a round robin among the arms of its family; within
   4a and within 4b the rule-3 score and tie-break pick the best floor.

## Stage 3b: from-scratch attackers (added 2026-11-01, user's request; written before any result)
Why: in stage 2 every BR started as a copy of its target (`--init-from-target`); 11 of 18 found no exploit and
drifted below 0.5. A BR from random weights is not tied to the target's own policy and may find a weakness the
copy-start BR missed. Stage 3 (E1) probes: a fresh learner (LSTM, 150M) failed against S3_A_s0 / S3_C_s0 (0.013,
0.016) and beat S3_B_s0 (0.8955).
- **Targets (user's choice):** S4_A_s1 (snap_476523, the stage-3 winner) and S4_C_s0@1.55B (snap_94446). The S4d
  models are not tested (user: "lets not use s4d").
- **Protocol:** stage 2's rule 2 except: no `--init-from-target` (a fresh feed-forward policy of royale.ini's
  architecture, no `--rnn`, so not the E1 probes' LSTM confound), **1B BR steps** (user's choice), H100/H200 only
  (a 1B BR needs about 17 h on an H100; an A100 would not fit the 24 h job), BR seeds 0 and 1, 1000 matches, random
  decks, shaping annealed over the first half (500M). W&B offline (group S4sel_BRscratch).
- **Reading:** a fresh BR starts near 0, so a score below 0.5 is the expected outcome and counts as 0.5 (no
  exploit found), as in deviation 2. Score = mean of the 2 seeds.
- **Rule for the winner:** if S4_A_s1's score is above 0.55 it leaves the stage-3 tie group and the winner becomes
  the highest-Elo remaining tied model (S4d_A_s1, 1582, not re-tested here). Otherwise S4_A_s1 stays the winner.
  C@1.55B's result is reported only (it stays not eligible). Stage 4 starts after this result.
- **Setup (2026-11-01):** smoke test 21629934 (CPU, 100K steps, 4 matches, vs C@1.55B): COMPLETED in 4 min;
  init_from_target false, init_from None, rnn False, policy = both targets' architecture (hidden 256, cnn 64,
  conditional head, pos_channels 32, placement_grid 1), decks random with no pool, card_floor 0; training score
  0.015 over 407 games (a fresh learner loses, a copy starts at 0.50). Submission: `runs/stage4/S4sel/`
  `brs_manifest.txt` (task 0 S4_A_s1 s0, 1 C@1.55B s0, 2 S4_A_s1 s1, 3 C@1.55B s1), `brs_array.sbatch` (=
  br_array.sbatch without --init-from-target, job name pr-s4brs, h100|h200, W&B group S4sel_BRscratch), rule-2
  flags, BR_STEPS=1000000000, checkout pinned/s4x_3cd765c; `$PR_ROOT/env.sh` logs pr-s4brs offline too (backup
  tmp/env.sh.bak_20261101). Outputs `runs/eval/br_S4scr_<agent>_s<seed>.json`.

## Results

### Stage 1: round robin (job 21002053, 2026-10-26, 4 x 27 min; all 324 tasks)
`runs/eval/stage4_rr_20261026/summary.json`. Row agent's score against the column agent over the 9 preset mirrors,
1,800 matches per pair (95% CI about +/- 0.023); rows by Elo.

| row vs col | C@1.55B | selfonly | S4_A_s1 | S4d_A_s1 | S4d_A_s0 | nommd | posent | S4df20 | S4_A_s0 | Elo | Nash |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C@1.55B | – | 0.376 | 0.615 | 0.597 | 0.676 | 0.723 | 0.853 | 0.877 | 0.910 | 1648 | 0.03 |
| selfonly | 0.624 | – | 0.493 | 0.649 | 0.602 | 0.636 | 0.754 | 0.765 | 0.835 | 1623 | 0.47 |
| S4_A_s1 | 0.385 | 0.507 | – | 0.443 | 0.567 | 0.618 | 0.785 | 0.777 | 0.896 | 1588 | 0.51 |
| S4d_A_s1 | 0.403 | 0.351 | 0.557 | – | 0.545 | 0.614 | 0.764 | 0.835 | 0.840 | 1582 | 0.00 |
| S4d_A_s0 | 0.324 | 0.398 | 0.433 | 0.455 | – | 0.522 | 0.694 | 0.722 | 0.806 | 1533 | 0.00 |
| nommd | 0.278 | 0.364 | 0.382 | 0.386 | 0.477 | – | 0.739 | 0.667 | 0.800 | 1511 | 0.00 |
| posent | 0.147 | 0.246 | 0.215 | 0.236 | 0.306 | 0.261 | – | 0.530 | 0.605 | 1371 | 0.00 |
| S4df20 | 0.123 | 0.235 | 0.223 | 0.165 | 0.278 | 0.333 | 0.470 | – | 0.669 | 1367 | 0.00 |
| S4_A_s0 | 0.090 | 0.165 | 0.104 | 0.160 | 0.194 | 0.200 | 0.395 | 0.331 | – | 1276 | 0.00 |

- Elo order: C@1.55B 1648, selfonly 1623, S4_A_s1 1588, S4d_A_s1 1582, S4d_A_s0 1533, nommd 1511, posent 1371,
  S4df20 1367, S4_A_s0 1276.
- A cycle: C@1.55B beats S4_A_s1 (0.615), S4_A_s1 and selfonly are even (0.507), selfonly beats C@1.55B (0.624).
  The Nash mixture is S4_A_s1 0.51, selfonly 0.47, C@1.55B 0.03.
- Against their common ancestor C@1.55B the branches score: selfonly 0.624, nommd 0.278, posent 0.147 (and the
  collapsed control was not entered). The A runs score 0.385 (S4_A_s1), 0.324 (S4d_A_s0), 0.403 (S4d_A_s1),
  0.090 (S4_A_s0), 0.123 (S4df20) against it.
- S4df20 (floor 0.20) is 8th of 9: it loses to every A run except S4_A_s0 (0.669 against it).
- The summary's "C vs A" line pools mixed pairs and is ignored (rule 1).
- Stage 1 decides nothing by itself: Elo is only the stage-3 tie-break.

### Stage 2: exploitability (all 18 BRs; written 2026-10-31 21:15)
`runs/eval/br_S4sel_<agent>_s<seed>.json`; all 18 checked against `br_manifest.txt` (target, seed, 1000 matches,
>= 150M steps, random decks, init_from_target, br_score = (W + D/2)/1000): no mismatch; no error in any task log.
Jobs: 21005306 tasks 6-17 (2026-10-29/30) and rerun 21343356 tasks 0-5 (2026-10-31, 2.6-3.3 h each).
br_score = the BR's score against the frozen target (SE about 0.015 per seed). Rows by the literal rule-3 mean.

| target | seed 0: W-D-L, br_score | seed 1: W-D-L, br_score | mean (rule 3 as written) | mean with < 0.5 read as 0.5 | rr Elo |
|---|---|---|---|---|---|
| S4_C_s0@1.55B (not eligible) | 274-48-678, 0.2980 | 298-75-627, 0.3355 | 0.3167 | 0.5000 | 1648 |
| S4d_A_s0 | 118-439-443, 0.3375 | 127-469-404, 0.3615 | 0.3495 | 0.5000 | 1533 |
| S4d_A_s1 | 423-34-543, 0.4400 | 362-20-618, 0.3720 | 0.4060 | 0.5000 | 1582 |
| S4_A_s1 | 477-15-508, 0.4845 | 437-39-524, 0.4565 | 0.4705 | 0.5000 | 1588 |
| S4x_nommd_s0 | 454-82-464, 0.4950 | 437-57-506, 0.4655 | 0.4803 | 0.5000 | 1511 |
| S4x_posent_s0 | 500-9-491, 0.5045 | 507-33-460, 0.5235 | 0.5140 | 0.5140 | 1371 |
| S4_A_s0 | 427-51-522, 0.4525 | 656-9-335, 0.6605 | 0.5565 | 0.5802 | 1276 |
| S4df20_A_s0 | 695-31-274, 0.7105 | 630-46-324, 0.6530 | 0.6818 | 0.6818 | 1367 |
| S4x_selfonly_s0 | 645-212-143, 0.7510 | 604-111-285, 0.6595 | 0.7052 | 0.7052 | 1623 |

- 11 of the 18 BRs score below 0.5, 9 of them by more than 2 SE (below 0.47; the other two are 0.4845 and
  0.495). Each BR started as an exact copy of its target, and a copy scores 0.5 against
  it by symmetry (random decks, both seats), so these BRs ended weaker than their starting point: no exploit found
  in 150M steps. In stage 3 (E1, Hog mirror) every BR scored 0.64-0.99. Cause not established (the deck setting is
  the main difference; rule 2 kept E1's 150M without re-calibrating it for random decks).
- Exploits found in both seeds: S4x_selfonly_s0 (0.751, 0.660) and S4df20_A_s0 (0.711, 0.653). S4_A_s0: one seed
  (0.661; the other 0.453). S4x_posent_s0 sits at 0.5 (0.505, 0.524).
- Seed spread up to 0.21 (S4_A_s0: 0.453 vs 0.661), like stage 3's up to 0.32.
- S4d_A_s0's two BRs draw 439 and 469 of 1000 matches (all others 9-212) and win only 118 and 127.
- Rule 3 as written (lowest mean wins, ties within 0.05, C@1.55B not eligible): **S4d_A_s0** (0.3495; next is
  S4d_A_s1 at 0.406, 0.057 away, so no tie). With scores below 0.5 read as 0.5: S4_A_s1, S4d_A_s0, S4d_A_s1,
  S4x_nommd_s0 (all 0.5) and S4x_posent_s0 (0.514) tie, and the Elo tie-break gives **S4_A_s1** (1588).
- **The two readings pick different winners. The user deferred this choice on 2026-10-31 ("We'll punt on
  this"); stage 3 is not decided until the user chooses.**

### Stage 3: selection (2026-11-01)
- The user chose (2026-11-01, "below 0.5 is 0.5") to read a br_score below 0.5 as 0.5 (deviation 2).
- Scores: S4_A_s1, S4d_A_s0, S4d_A_s1, S4x_nommd_s0 0.500; S4x_posent_s0 0.514; S4_A_s0 0.580;
  S4df20_A_s0 0.682; S4x_selfonly_s0 0.705 (C@1.55B 0.500, not eligible). Lowest 0.500, tie band <= 0.550:
  S4_A_s1, S4d_A_s0, S4d_A_s1, S4x_nommd_s0, S4x_posent_s0.
- Tie-break, highest round-robin Elo: S4_A_s1 1588, S4d_A_s1 1582, S4d_A_s0 1533, S4x_nommd_s0 1511, S4x_posent_s0
  1371. **Winner: S4_A_s1** (league/S4_A_s1/snap_476523.pt, 7.81B steps).
- Caveats: S4d_A_s1 is 6 Elo behind and beats S4_A_s1 head to head 0.557 (Nash weight 0.51 vs 0.00); the
  same configuration with seed 0 (S4_A_s0, the only config difference is the seed) is last by Elo (1276) and
  exploited in one BR seed (0.661).
- 2026-11-01, open: the user asked to train "the top 3 models with score < 0.5" from scratch as well
  (S4_A_s1, S4d_A_s1, S4d_A_s0: the tie group ranked by Elo, C@1.55B excluded). All three already started from
  random weights (config init_from None, TOTAL 10B), so this means new seeds; which runs and how long is not
  decided yet. (An earlier version of this line read the user's "No" as declining a seed-2 run; the user meant
  this request.) Stage 4b is separate and uses seed 1 as rule 4 says.
- Stage 4b facts for S4_A_s1 (its config.json): method A (self_play_frac 1, no MMD), seed 1, 256 envs, deck pool
  hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge + random_deck_frac 0.5, held-out xbow;royal_hogs;random,
  shaping annealed over the first 0.5B learner steps (shaping_anneal_frac 0.05 of TOTAL 10B), card floor 0.005
  faded to 0 over 5B. Control at 2.5B: snap_152640.pt (2.501B). `hpc/stage4_submit.sh` keeps the 0.5B shaping
  fade for any TOTAL (SHAPING_STEPS=500000000) and takes CARD_FLOOR / CARD_FLOOR_STEPS=0.

#### Stage 2: why BRs end below 0.5 (analysis 2026-11-01)
Source: the 18 offline W&B histories, read with `runs/stage4/S4sel/br_hist.py` (LevelDB log reader; 9,156 rows =
one per epoch of 16,384 steps per run) into `runs/stage4/S4sel/br_hist_all.json`. (A first regex scan of the raw
file mis-aligned keys and is discarded.)
- In the first 0.5M steps every BR scores 0.47-0.53 against its target: the copy starts even, so there is no
  setup mismatch.
- The BRs that end below 0.5 drift down gradually, e.g. S4d_A_s0 seed 1 by window (0-0.5, 0.5-2, 2-10, 10-40,
  40-75, 75-113, 113-150M steps): 0.49 0.49 0.45 0.43 0.41 0.39 0.35; C@1.55B seed 0: 0.47 0.45 0.40 0.38 0.33 0.35
  0.32. The exploiting BRs climb: S4x_selfonly_s0 seed 0: 0.50 0.53 0.50 0.60 0.71 0.71 0.74.
- In the drifting BRs the policy entropy rises (S4d_A_s0: 0.10-0.12 -> 0.19-0.20; C@1.55B: 0.17 -> 0.26-0.27),
  and S4d_A_s0's BRs drift into draws (draw rate 0.09-0.15 -> 0.43-0.48).
- Ruled out: different PPO settings. Targets and BRs both use royale.ini's ent_coef 0.01, learning_rate 3e-4,
  update_epochs 1, gamma 0.999, and the same reward flags (stage4_submit.sh COMMON = rule-2 flags).
- Differences that remain (not tested): BR decks are 100% random (targets: 7-deck pool + 50% random); the BR's
  shaping starts at full weight and anneals over 75M steps (targets: S4_A / S4_C / S4x annealed theirs over the
  first 0.5B steps, shaping_anneal_frac 0.05 of 10B; S4d / S4df20 never anneal, shaping_anneal_frac 0; corrected
  2026-11-01, an earlier version of this line said "by 5B");
  frozen opponent instead of self-play; fresh optimizer state.
- Reading (not proven): when the target has no clear weakness, the BR's learning signal is weak and its noisy
  updates move the tuned policy downhill; when there is a weakness (selfonly, S4df20) the signal wins.

### Stage 2 setup (2026-10-26)
- Smoke test 21003843 (CPU, 100K steps, 4 matches, S4d_A_s0's snap_325314, random decks, `--init-from-target`):
  COMPLETED; the BR's env had deck0 = deck1 = random with no deck pool, card_floor 0, the target as its only
  opponent (self_play_frac 0, mmd 0); training score vs the target 0.50 (it starts as a copy); evaluation ran.
- Submission: `runs/stage4/S4sel/` (`br_manifest.txt`: line k = array task k - 1, the 9 candidates with seed 0,
  then with seed 1; `br_array.sbatch` -> `hpc/br.sbatch` on `pinned/s4x_3cd765c`), `BR_STEPS=150000000`,
  `--array=0-17%4`. Outputs `runs/eval/br_S4sel_<agent>_s<seed>.json`, W&B group `S4sel_BR`.

## Deviations

1. **2026-10-29, W&B offline for the stage-2 BR tasks (logging only, results unchanged).** Tasks 0-5 of 21005306
   started 11:21-11:36 on A100 nodes ga042/ga024/ga016 and all failed after about 20 s in `wandb.init`:
   `CommError: ... tls: failed to verify certificate: x509: certificate signed by unknown authority`. Nothing
   was trained, no result file was written. The certificate chain of api.wandb.ai checks out from the login node, and an
   S4d_A_s1 link on ga035 (2026-10-28) and the S4df20 link on gh010 (2026-10-29 04:54) connected to W&B
   normally, so the failure is new or node-specific. Fix (11:50): `$PR_ROOT/env.sh` sets `WANDB_MODE=offline`
   only when `SLURM_JOB_NAME=pr-s4br` (backup `tmp/env.sh.bak_20261029`); the queued tasks 6-17 read env.sh
   at start, so they log offline and are synced from a login node afterwards. `hpc/tests/test_best_response_fast.py`
   checks that offline W&B gives the same weights and W/D/L as no W&B. Tasks 0-5 = manifest lines 1-6 = seed 0
   of S4_A_s0, S4_A_s1, S4d_A_s0, S4d_A_s1, S4df20_A_s0, S4x_nommd_s0. Rerun 2026-10-29 as **21343356**
   (`br_array.sbatch --array=0-5%2`, same manifest, checkout, BR_STEPS and the rule-2 flags; Slurm does not keep
   the original FLAGS, so a rerun task's config.json is compared with an original task's). `%2` keeps stage 2 at
   most 6 GPUs at once; total GPU-hours are unchanged.
   Check of the fix: task 6 (S4x_selfonly_s0, seed 0) started 11:50 on ga033 and logged offline
   (`offline-run-20261029_115056-h1qc6qsu`), no error after 8 min. Its config.json: deck0 = deck1 = random,
   deck_pool empty, random_deck_frac 0, card_floor 0, mmd 0, self_play_frac 0, 150M steps, 256 envs, seed 0,
   env_threads 16, fast_opponents, cuda_graphs, shaping_anneal_frac 0.5; its W&B config: reward_clip 0,
   torch_deterministic false, init_from_target true, matches 1000 (the weights are loaded from the target
   with `load_state_dict`, so league `init_from` is None by design). Rerun tasks are compared with it by
   `runs/stage4/S4sel/compare_br_config.py`.
   **Duplicate:** a second identical rerun array, **21343360** (same submit line, submitted 11:41:57, 10 s
   after 21343356), is also queued. My submit command printed only 21343356; the cause of the second submission
   is unknown. If both ran, the 6 seed-0 BRs would run twice and the second to finish would overwrite
   `br_S4sel_<agent>_s0.json`. Cancelled 2026-10-31 08:05 on the user's instruction (still pending, nothing
   had run); 21343356 remains the only rerun.
   Rerun task 0 (S4_A_s0, seed 0) started 2026-10-31 10:58 on gh011, offline W&B, no error after 8 min;
   `compare_br_config.py` against original task 6: 82 fields compared (config.json without target/seed/run id,
   plus the W&B train flags and br block), no differences; its W&B record shows target snap_404496, seed 0,
   init_from_target true, 1000 matches.

2. **2026-11-01, rule 3: a br_score below 0.5 counts as 0.5 (user's decision).** Decided after the results:
   11 of 18 BRs ended below 0.5 although each started as a copy of its target (0.47-0.53 in its first 0.5M
   steps), i.e. they found no exploit and drifted; exploitability cannot be negative. Taken literally, rule 3
   would have picked S4d_A_s0 (0.350).

3. **2026-11-03, stage 3b: a second copy on A100s (user's decision; written before either copy started).**
   21630359 (H100/H200 only) had waited about 40.5 h without a start. User: "Launch second copy on a100 - look at
   the fairshare and use the optimal accoutn". Checks before submitting:
   - Accounts: only the two tandon accounts get A100s (`sbatch --test-only --constraint=a100` with
     torch_pr_355_general or torch_pr_159_general: "allocation failure"). Fair-share (`sshare`): tandon_priority
     0.0582, tandon_advanced 0.0487; test-only start estimates 2026-11-06 02:51 vs 12:52. So **tandon_priority**.
   - The per-user cap of QOS gpu48 (2 GPUs; 4 in use, 2 of them g1v_* jobs on pr_159) does not hold these jobs: the
     partition QOS a100_tandon / h100_tandon has no per-user cap, and another user started 4 gpu48 jobs at 16:32.
     What holds them is the tandon group pools: a100_tandon 60 of 60 GPUs in use, h100_tandon 57 of 60.
   - Time: a 1B BR is about 17.5 h on an H100 (stage 2: 2:37-2:41 per 150M, x 6.67); on an A100, 3:01-3:31 per
     150M -> about 20-23.5 h, too close to 24 h, so the A100 copy gets 36 h (gpu48 allows 2 days).
   Copy **21778800** (submitted 2026-11-03 16:37, partition a100_tandon, account tandon_priority, QOS gpu48):
   `runs/stage4/S4sel/brs_a100_array.sbatch` (= brs_array.sbatch with constraint a100, 36 h, and a twin
   check) and `brs_a100_manifest.txt` (= brs_manifest.txt with `_a100` added to every name, so data dirs, result
   files `runs/eval/br_S4scr_<agent>_a100_s<seed>.json` and W&B names never collide). Same checkout
   (pinned/s4x_3cd765c), the same flags as 21630359 (taken from its submit command), BR_STEPS 1B, 1000 matches,
   256 envs, array 0-3%4. Task k of the copy is the twin of 21630359 task k (same target and seed). When a copy
   task starts it asks `sacct` for its twin's state and exits without training if the twin is RUNNING,
   COMPLETING or COMPLETED (tested on a pending, a running and a completed task).
   **Which result counts (fixed now, before any result):** for each target and seed, the copy that starts first.
   If that copy fails, the other one counts. The other copy's result, if it runs, is reported only. The 3b rule
   itself (S4_A_s1's mean over the 2 seeds above 0.55 -> S4d_A_s1 becomes the winner) is unchanged.
   Not covered by the twin check: a copy task starts first and its H100 twin starts later. Both then run;
   cancelling the later one needs the user's OK (standing rule: no cancelling queued jobs).
   2026-11-05 19:30: all 4 H100 tasks COMPLETED, none of the copy's tasks had started; the user chose to cancel
   the copy ("Cancel it"), so 21778800 was cancelled while pending (sacct: CANCELLED 19:30:45; nothing of it ran).
4. **2026-11-05, stage 4a: a fixed step budget instead of "one 20 h link each" (user's decision, before any
   stage-4 job started).** Rule 4 gives each 4a arm one 20 h link. The S4x branches (4 arms, one 20 h link each,
   all on H100 nodes) trained 1.06B, 1.17B, 1.18B and 1.34B steps, so a time budget gives arms unequal training
   for reasons unrelated to the floor. User (2026-11-05): "Fixed +1.0B steps". Each 4a arm trains from
   snap_610083 (9,995,599,872 steps) to epoch 671139 = 10,995,941,376 steps (+1,000,341,504; 671139 = 1407 x 477,
   so the last snapshot is the final model), in at most 2 links of 20 h (S4_A_s1's A100 rate, 10B in 188.7 h,
   gives about 19 h per arm; a link that finds the run finished does nothing). Also decided 2026-11-05: all 7
   stage-4 runs are submitted at once (user: "All 7 now", above the usual 4-GPU start).

## Notes

- 2026-10-26: the queued GPU links of the A runs wait with Slurm reason `QOSMaxGRESPerUser`, but the QOS
  counter shows 2 of 16 GPUs in use (`scontrol show assoc_mgr flags=qos`): the cap is not what holds them;
  the h100/a100 partitions are busy.
- 2026-10-26 23:40: stage 2 (21005306) still pending, no task started. `sprio`: it is behind four of our own
  A-chain links (S4_A_s0, S4d_A_s0, S4d_A_s1, S4df20; older, so higher age priority) and 928 pending jobs in
  a100_tandon/h100_tandon. A same-shaped `sbatch --test-only` job gets an estimated start of 2026-10-28 16:45. Stage-3 E1 BR tasks
  took 2.3-4.6 h each, so the 18 tasks need about 60 GPU-hours (about 15 h of wall time with 4 running at once).
- 2026-10-27 11:20: stage 2 still pending. The Slurm reason on our waiting links is now `QOSGrpGRES`: the
  a100_tandon partition QOS caps the whole tandon group at 60 GPUs and all 60 are in use. Priority order of our
  waiting jobs: S4_A_s0 link 12283, s20btf (not ours to touch) 12179, S4d_A_s1 link 12133, S4df20 link 12111,
  **stage-2 array 12072**, S4_A_s1 next link 11946.
- 2026-11-02 11:30: stage 3b (21630359) still pending, no task started (Slurm reason `None`, priority 11719,
  partition h100_tandon). 44 jobs run on the partition's 15 nodes; 62 eligible pending jobs there have higher
  priority. A same-shaped `sbatch --test-only` job gets an estimated start of 2026-11-06 10:44 (an upper bound:
  Slurm assumes every running job uses its full time limit). The A chains: S4d_A_s0 9.32B and S4df20 6.22B
  running; S4d_A_s1 at 8.21B, its next link pending.
- 2026-11-03 10:40: stage 3b (21630359) still pending after about 35 h (reason `None`, priority 11858, no Slurm
  start estimate). S4d_A_s0 finished its 10B run (model_610352.pt; its leftover chain links exited in about 30 s
  each). S4d_A_s1 8.31B (link running on an A100), S4df20 7.68B (link running).
- 2026-11-04 04:43: stage 3b task 0 (S4_A_s1, seed 0) of the H100 copy 21630359 started on gh013 (Slurm job
  21837856), about 52 h after submission; under deviation 3 it is the copy that counts for this target and seed.
  Checks at 04:49: offline W&B (`offline-run-20261104_044407-ur3823jv`), isolation OK, env settings as stage 2.
  `compare_br_config.py` against stage 2's BR on the same target (S4sel_S4_A_s1 seed 0, run 20261031_121735_s0):
  82 fields, exactly the expected differences: total_timesteps 1B vs 150M, shaping_anneal_steps (env and league)
  1,953,125 vs 292,969 (half of each run, as shaping_anneal_frac 0.5), init_from_target false vs true. Its config:
  init_from None, deck0 = deck1 = random, no deck pool, card_floor 0, self_play_frac 0, hidden 256 / cnn 64 /
  conditional head / pos_channels 32. Progress: 5.65M steps after about 5 min (about 18K steps/s, so about 16 h
  for 1B); training score vs the target 0.014 (a fresh learner; the smoke test showed 0.015). Still pending:
  21630359 tasks 1-3 (H100) and the A100 copy 21778800 tasks 0-3.
- 2026-11-05 01:25: stage 3b **task 0 done** (21630359_0, COMPLETED 2026-11-04 18:36 after 13:52:53 on gh013):
  `runs/eval/br_S4scr_S4_A_s1_s0.json`: br_score **0.0745** (W 59, D 31, L 910 of 1000), 1,000,013,824 steps,
  2,527,164 training games vs the target (training score 0.083). Below 0.5, so it counts as 0.5 (deviation 2): no
  exploit found. For the 3b rule (S4_A_s1's mean above 0.55) seed 1 would now need a br_score above 0.60.
  Training score by window (from its W&B history with br_hist.py; windowed = change in score x games / games):
  0-10M 0.026, 10-50M 0.083, 50-100M 0.154, 100-200M 0.133, 200-300M 0.088, 300-400M 0.077, 400-500M 0.068,
  500-600M 0.066, 600-700M 0.068, 700-800M 0.070, 800-900M 0.077, 900M-1B 0.082; entropy 0.25-0.35, draws ~0.05.
  The peak at 50-100M and the fall until about 500M coincide with the shaping weight annealing to 0 over the first
  500M (not tested as a cause); after that it climbs slowly (0.066 -> 0.082).
  **Task 1 started** (C@1.55B seed 0; 21630359_1 = job 21931168, gh014, 2026-11-05 01:23): offline W&B
  (`offline-run-20261105_012332-xapncpyg`); `compare_br_config.py` against stage 2's C@1.55B seed 0 (run
  20261029_120049_s0): 82 fields, the same 4 expected differences as task 0. Tasks 2-3 (H100) and the A100 copy
  21778800 (all 4; its task 0 and 1 will skip, their twins have run) still pending. Chains: S4d_A_s1 9.89B (link
  19555047_3 running), S4df20 8.65B (next link pending).
- 2026-11-05 05:28: stage 3b **task 2 started** (S4_A_s1 seed 1; 21630359_2 = job 21951088, gh008, 05:23): offline
  W&B (`offline-run-20261105_052407-b7ntsp8z`), isolation OK, league seed 1; `compare_br_config.py` against
  stage 2's S4_A_s1 seed 1 (run 20261030_010401_s1): 82 fields, the same 4 expected differences. Its br_score
  decides the 3b rule (above 0.60 -> S4_A_s1's mean above 0.55). Task 1 still logging (W&B file updated 05:27).
  Task 3 (C@1.55B seed 1) pending (Resources); A100 copy 21778800 pending.
- 2026-11-05 05:33: stage 3b **task 3 started** (C@1.55B seed 1; 21630359_3, gh015, 05:28): offline W&B
  (`offline-run-20261105_052830-6hu2r6gq`), isolation OK, league seed 1; `compare_br_config.py` against stage 2's
  C@1.55B seed 1 (run 20261030_054700_s1, W&B name S4sel_S4_C_s0_at1.55B_br_s1): 82 fields, the same 4 expected
  differences. All 4 tasks of the H100 copy have now started first, so under deviation 3 they are the results
  that count. Every task of the A100 copy 21778800 (still pending) will find its twin running or done and exit
  without training; if a twin has FAILED by then, that task trains as a backup.
- 2026-11-05 14:30: stage 3b progress (training win rate vs the target, read from each run's offline W&B log; the
  quick reader matches br_hist.py on task 0 in all 5 windows): task 1 C@1.55B s0 at 974M, last 50M 0.324
  (0-100M 0.110, 300-500M 0.265, 700M+ 0.312); task 2 S4_A_s1 s1 at 671M, last 50M 0.168 (0-100M 0.088,
  300-500M 0.135, 500-700M 0.158); task 3 C@1.55B s1 at 671M, last 50M 0.240. For comparison task 0 (S4_A_s1 s0)
  ended training at 0.076-0.082 and scored 0.0745 in its 1000-match evaluation. **S4d_A_s1 finished its 10B
  run** (link 19555047_3 COMPLETED 2026-11-05 03:43; model_610352.pt, 10,000,007,168 steps); its leftover
  link 19555049_3 exited in 31 s, 19555050/51 still queued (they will do the same). S4df20 at 9.03B (link
  20446670_0 running). The A100 copy 21778800 and 19555050 wait with reason QOSMaxGRESPerUser; gpu48's cap is
  16 GPUs per user and 4 are in use (`scontrol show assoc_mgr`), so the reason is not explained by that number.
- 2026-11-05 15:28: stage 3b **task 1 done** (C@1.55B seed 0; 21630359_1, COMPLETED 2026-11-05 15:24 after
  14:01:03 on gh014): `runs/eval/br_S4scr_S4_C_s0_at1.55B_s0.json`: br_score **0.3485** (W 343, D 11, L 646 of
  1000), 1,000,013,824 steps, 2,185,725 training games vs the target (training score 0.259), init_from_target
  false, random decks. Below 0.5, so it counts as 0.5: no exploit found (reported only; C@1.55B is not eligible).
  Training score by window (br_hist.py, 61,036 history rows): 0-10M 0.019, 10-50M 0.084, 50-100M 0.165, 100-200M
  0.202, 200-300M 0.229, 300-400M 0.253, 400-500M 0.277, 500-600M 0.297, 600-700M 0.309, 700-800M 0.305,
  800-900M 0.316, 900M-1B 0.323; entropy 0.11-0.66, draws 0.02-0.06. Unlike task 0 (S4_A_s1 seed 0: peak 0.154 at
  50-100M, then down to 0.066 and back to 0.082) it rises in every window up to about 600M and then flattens near
  0.31-0.32. Same seed, same protocol: the attacker ends about 4 times higher against C@1.55B (0.3485) than
  against S4_A_s1 (0.0745). Running (br_hist.py at 15:28): task 2 (S4_A_s1 s1) at 750M, 700-750M 0.165
  (0-100M 0.087, 300-500M 0.135, 600-700M 0.166); task 3 (C@1.55B s1) at 751M, last 50M 0.236. The A100 copy
  21778800 is still pending; its task 1 will find its twin COMPLETED and exit.
- 2026-11-05 19:25: stage 3b **task 3 done** (C@1.55B seed 1; 21630359_3, COMPLETED 2026-11-05 19:21 after
  13:52:49 on gh015): `runs/eval/br_S4scr_S4_C_s0_at1.55B_s1.json`: br_score **0.2295** (W 191, D 77, L 732 of
  1000), 1,000,013,824 steps, 2,312,181 training games vs the target (training score 0.203), init_from_target
  false, random decks, seed 1. Below 0.5: no exploit found (reported only). Training score by window (br_hist.py,
  61,036 rows): 0-10M 0.019, 10-50M 0.053, 50-100M 0.118, 100-200M 0.184, 200-300M 0.193, 300-400M 0.204,
  400-500M 0.215, 500-600M 0.230, 600-700M 0.240, 700-800M 0.237, 800-900M 0.239, 900M-1B 0.255; draws about
  0.05. **C@1.55B from-scratch result: 0.3485 and 0.2295, mean 0.289 (raw), 0.500 under the below-0.5 rule.**
  Task 2 (S4_A_s1 seed 1) finished its 1B training steps (61,036 rows; training score by window 0-10M 0.018,
  10-50M 0.071, 50-100M 0.124, 100-200M 0.152, 200-300M 0.108, 300-400M 0.134, 400-500M 0.136, 500-600M 0.152,
  600-700M 0.166, 700-800M 0.172, 800-900M 0.183, 900M-1B 0.201) and is in its 1000-match evaluation.
- 2026-11-05 19:30: stage 3b **task 2 done** (S4_A_s1 seed 1; 21630359_2, COMPLETED 2026-11-05 19:25 after
  14:01:55 on gh008, exit 0:0): `runs/eval/br_S4scr_S4_A_s1_s1.json`: br_score **0.166** (W 163, D 6, L 831 of
  1000), 1,000,013,824 steps, 2,326,346 training games vs the target (training score 0.147), init_from_target
  false, random decks, seed 1. Training score by window: see the 19:25 entry (900M-1B 0.201).
  **Stage 3b result (all 4 tasks of the H100 copy 21630359 COMPLETED; under deviation 3 these are the results
  that count):**

  | target | seed 0 | seed 1 | raw mean | score (below 0.5 -> 0.5) |
  |---|---|---|---|---|
  | S4_A_s1 (snap_476523, 7.81B) | 0.0745 | 0.166 | 0.120 | **0.500** |
  | S4_C_s0@1.55B (snap_94446), reported only | 0.3485 | 0.2295 | 0.289 | 0.500 |

  **Rule applied:** S4_A_s1's score 0.500 is not above 0.55, so **S4_A_s1 stays the winner**; stage 4 runs on it.
  No from-scratch attacker found an exploit of either target in 1B steps. Observation (not part of the rule):
  with the same protocol and seeds the attackers ended about 2.4 times higher against C@1.55B (raw mean 0.289)
  than against S4_A_s1 (0.120), while C@1.55B is higher by round-robin Elo (1648 vs 1588) and beats S4_A_s1
  0.615 head to head. The A100 copy 21778800 (still pending, QOSGrpGRES) now has all 4 twins COMPLETED, so each
  of its tasks will exit within about a minute of starting without training; it is left queued (no cancelling
  without the user's instruction).
- 2026-11-05 19:28: **stage 4 dry run (nothing submitted; waits for the user).** Checkout `pinned/s4x_3cd765c` for
  every arm (S4x and S4df20 used it; it differs from S4_A_s1's `s4_d70c336` by `hpc/branch_run.py`, the stage-4
  eval/diagnostic scripts and `ent_coef_pos_unweighted`, default 0 = off, which leaves the update unchanged).
  - **4a** (`runs/stage4/S4L/dryrun/`: 4 env files, manifest, commands.txt): `hpc/branch_run.py --source S4_A_s1
    --at 9.995` -> snap_610083.pt (9,995,599,872 steps; the run's latest snapshot; its final model_610352.pt
    (10,000,007,168) is not a snapshot epoch, 610352 is not a multiple of 477), pool = its 64 latest snapshots
    (580032..610083). Arms S4L_A_s1_ctrl (no floor `--set`: 0.005 faded over 5B, so 0 here), S4L_A_s1_f05 / f10 /
    f20 (`--set card_floor=X --set card_floor_steps=0`, the same in EXTRA_ARGS so a resume cannot undo it). Env file
    = S4_A_s1.env with: checkout, RUN_ID, W&B group S4L, TOTAL 10,995,941,376 (epoch 671139 = 1407 x 477, so the
    last snapshot is the final model; +1,000,341,504 steps), `--shaping-anneal-frac` dropped (a resume then keeps
    the stored anneal length 1,953,125; the shaping weight is 0 after about 0.5B either way). Verified on a
    throwaway copy under tmp/ (source files unchanged; branch args: total, floor, floor steps, anneal length as
    intended; removed after). **Proposed deviation 4:** a fixed step budget (+1.0B, up to 2 links of 20 h; a link
    that finds the run finished does nothing) instead of "one 20 h link each": the S4x branches got 1.06-1.34B in
    one 20 h link on the same GPU type, so a time budget would give the arms unequal training.
  - **4b** (`runs/stage4/S4f05|S4f10|S4f20/dryrun/`, `hpc/stage4_submit.sh` with PREFIX=S4fXX, RUNS="A:1:a100|h100|h200",
    TOTAL=2,500,853,760 (= the control snap_152640's step), LINKS=4, CARD_FLOOR=X, CARD_FLOOR_STEPS=0, LLM_RUNS=""):
    env file = S4_A_s1.env except checkout, RUN_ID, TOTAL, W&B group, floor and the shaping fraction
    0.19993172251703353 (anneal length ceil(F x TOTAL / 256) = 1,953,125, the same as S4_A_s1's). Runs S4f05_A_s1,
    S4f10_A_s1, S4f20_A_s1, seed 1.
  - Cost (S4_A_s1's A100 rate, 10B in 188.7 h of links): 4a about 19 h per arm, 4b about 47 h per arm (3 links);
    about 220 GPU-hours, 7 GPUs at once if submitted together (plus the S4df20 chain's 1).
- 2026-11-05 19:35: **stage 4 submitted** (user, 2026-11-05: "Fixed +1.0B steps", "All 7 now", A100 copy "Cancel it").
  - **4a:** the 4 branches were made with `hpc/branch_run.py` exactly as in the dry run (commands in
    `runs/stage4/S4L/dryrun/commands.txt`, plus `--data-dir $PR_ROOT/runs`); checked after: each BRANCH.json names
    snap_610083.pt at 9,995,599,872 steps, pool 64 snapshots (580032..610083), learner.pt sha256 = snap_610083.pt's,
    args total 10,995,941,376, seed 1, anneal length 1,953,125, floor ctrl 0.005 over 5B (0 here) / 0.05, 0.10,
    0.20 with steps 0; S4_A_s1's league_state / trainer_state / learner / snap_610083 unchanged (mtime and size).
    Env files `runs/stage4/S4L/sub_20261105_193343/` (identical to the dry run's apart from the comment line),
    stage3_array.sbatch task i of its manifest, account tandon_priority, constraint a100|h100|h200, 24 CPUs, 96G,
    20 h; link 2 `--dependency=afterany:<link 1>`. Chains (`chains.txt`): ctrl 22014154 22014156, f05 22014157
    22014158, f10 22014161 22014162, f20 22014164 22014165. W&B group S4L (training online, like S4_A_s1).
  - **4b:** `hpc/stage4_submit.sh` with SUBMIT=1 and the dry run's variables; env files identical to the dry run's
    apart from the comment line. Chains: S4f05_A_s1 22014171 22014178 22014180 22014183
    (`runs/stage4/S4f05/sub_20261105_193406/`), S4f10_A_s1 22014197 22014198 22014202 22014203
    (`S4f10/sub_20261105_193421/`), S4f20_A_s1 22014245 22014284 22014285 22014286 (`S4f20/sub_20261105_193456/`).
    No Claude-game chains (LLM_RUNS="").
  - Queue at 19:35: every first link PENDING (reason QOSMaxGRESPerUser on 6 of 7, None on one). Limits read with
    `scontrol show assoc_mgr`: QOS gpu48 per-user cap 16 GPUs, 1 in use (the S4df20 link); partition QOS a100_tandon
    60 of 60 GPUs in use, h100_tandon 54 of 60, neither with a per-user cap. The reason does not match these
    numbers (as on 2026-11-05 14:30); the full a100 pool is the visible constraint.
- 2026-11-06 14:15: no stage-4 job has started (all 7 first links PENDING, reason None, priority 12042, 18.7 h
  after submission). Queue: a100_tandon 610 pending / 55 of 60 GPUs in use, h100_tandon 465 pending / 57 of 60; 288
  (a100) and 240 (h100) pending jobs have a higher priority than ours. `sbatch --test-only` (24 CPUs, 96G, 1 GPU,
  20 h) estimates for a new job: a100 2026-11-08 15:43 (tandon_priority) / 12:24 (tandon_advanced), h100
  2026-11-16, h200 and the general account: allocation failure. Fair-share: tandon_advanced 0.094, tandon_priority
  0.086. S4df20: link 20446670 TIMEOUT 2026-11-06 04:27 at 9,925,263,360 steps, next link 20446673 pending.
  The start watcher of 2026-11-05 died with the session; nothing was missed (no job started).
- 2026-11-07 11:30: **stage 4a control started**: S4L_A_s1_ctrl, link 1 (22014154_0) on gh013 (H100) at 11:20, about
  40 h after submission. Checks: resumed from the branch (first step 9,995,616,256 = epoch 610084, right after
  snap_610083), total 10,995,941,376 (deviation 4), floor 0.0000 (0.005 faded to 0 over 5B), checkout 3cd765c, pool
  line `pool 1 | self=...` as in S4_A_s1's own log (pure self-play), eval sidecar and GPU heartbeat started, W&B online
  (run wo1hz1c2, group S4L). 18,498 steps/s after 5 min (S4_A_s1 on an A100: 16,155), so +1.0B is about 15 h, within
  link 1; link 2 (22014156) should find the run finished. Still pending (reason None/Priority): 4a f05/f10/f20 and
  4b S4f05/S4f10/S4f20 first links.
- 2026-11-08 01:45: **stage 4a control done**: S4L_A_s1_ctrl link 1 (22014154_0) COMPLETED 2026-11-08 01:29 after
  14:09:42 on gh013. league_state: epoch 671139, global step 10,995,941,376 (= the deviation-4 target exactly), not
  early-stopped. Final model `model_671139.pt` and the same epoch as `snap_671139.pt` (the stage-5 checkpoint for this
  arm), 192 snapshot files (the 64 pool copies + 128 new). Floor 0.0000 throughout; 19,634 steps/s at the end. Eval
  sidecar (vs the heuristic bot, pool decks, 140 matches): 0.936 at 10.80B, 0.957 at 10.90B. Link 2 (22014156) is
  still queued; it will find the run finished and exit (left queued: no cancelling queued jobs). Still pending: 4a
  f05/f10/f20 and 4b S4f05/S4f10/S4f20 first links.
- 2026-11-08 02:00: **stage 4a floor 0.05 started**: S4L_A_s1_f05 link 1 (22014157_1) on ga013 (A100) at 01:53. Checks:
  resumed at epoch 610084 (step 9,995,616,256), total 10,995,941,376, floor 0.0500 from the first step (constant), checkout
  3cd765c, `pool 1 | self=...` (pure self-play), eval sidecar and W&B online (dry-sun-173). Entropy 0.26-0.32 on the
  first epochs vs 0.10-0.20 for the control (the floor spreads probability over the legal cards, as intended).
  13,768 steps/s after 4 min and rising; at S4_A_s1's A100 average (16,155) +1.0B is about 17 h, inside link 1 (link 2,
  22014158, finishes any rest). Still pending: 4a f10/f20, 4b S4f05/S4f10/S4f20.
- 2026-11-08 08:45: **stage 4a floor 0.10 started**: S4L_A_s1_f10 link 1 (22014161_2) on ga008 (A100) at 08:30. Checks:
  resumed at epoch 610084 (step 9,995,616,256), total 10,995,941,376, floor 0.1000 from the first step (constant),
  checkout 3cd765c, `pool 1 | self=...` (pure self-play), eval sidecar, GPU heartbeat (80-82% utilization) and W&B
  online (clean-snowflake-175, xcpz7j4s). Entropy 0.22-0.35 on the first epochs. 14,685 steps/s after 12 min, so +1.0B
  is about 19 h, inside link 1's 20 h limit (link 2, 22014162, finishes any rest). **Correction to the f05 estimate:**
  f05 has held 14,500 steps/s (step 10,352,214,016 at 08:43, 6 h 50 min in), not S4_A_s1's 16,155, so it ends about
  21:00, near its 20 h limit (21:53); if it does not finish, link 2 (22014158) does. Still pending: 4a f20, 4b
  S4f05/S4f10/S4f20.
- 2026-11-08 09:25: **stage 4a floor 0.20 started**: S4L_A_s1_f20 link 1 (22014164_3) on ga011 (A100) at 09:11. Checks:
  resumed at epoch 610084 (step 9,995,616,256), total 10,995,941,376, floor 0.2000 from the first step (constant),
  checkout 3cd765c, `pool 1 | self=...` (pure self-play), eval sidecar, GPU heartbeat (80-82%) and W&B online
  (curious-monkey-177, 5y15d38t). 13,778 steps/s after 12 min, so +1.0B is about 20 h: at or just past link 1's 20 h
  limit (05:11 on 11-09); link 2 (22014165) finishes the rest. All four 4a arms are now started. Logged entropy over
  each arm's first 599 epochs (epochs 610085-610683; median, p10-p90): control 0.203 (0.172-0.250), f05 0.264
  (0.243-0.292), f10 0.300 (0.276-0.326), f20 0.310 (0.279-0.348); it rises with the floor. At 09:25 4b S4f05_A_s1
  (22014171_0) and S4f10_A_s1 (22014197_0) were also running (checked next); S4f20_A_s1 still pending.
- 2026-11-08 09:40: **stage 4b floors 0.05 and 0.10 started**: S4f05_A_s1 link 1 (22014171_0) on ga017 (A100) at 09:15
  and S4f10_A_s1 link 1 (22014197_0) on ga031 (A100) at 09:18. Checks, both: fresh start (no resume; first line step
  16,384, epoch 1), total 2,500,853,760, floor 0.0500 / 0.1000 constant (card_floor_steps 0), checkout 3cd765c, pure
  self-play, eval sidecar and W&B online (dauntless-star-179, charmed-salad-181). Config compared key by key with
  S4_A_s1's config.json: the only differences are run_id, total_timesteps, card_floor, card_floor_steps,
  shaping_anneal_frac (0.19993 vs 0.05, same N: shaping_anneal_steps 1,953,125 in both, as planned) and
  env.shaping_step_offset (0 vs 37,213,632). The offset is rewritten at every resume (= global step // 256):
  37,213,632 x 256 = 9,526,689,792, and S4_A_s1's link 19508322 resumed there (first logged step 9,526,706,176), so its
  original from-scratch start used 0, as these do. 14,216 and 13,809 steps/s after 8 and 5 min (still rising); 2.5B
  takes about 2 days, i.e. 3 links of 20 h (each chain has 4). Still pending: 4b S4f20_A_s1 (22014245).
- 2026-11-08 09:45: **stage 4b floor 0.20 started; all 7 stage-4 arms now started**: S4f20_A_s1 link 1 (22014245_0) on
  ga019 (A100) at 09:31. Checks: fresh start (step 16,384, epoch 1), total 2,500,853,760, floor 0.2000 constant,
  checkout 3cd765c, pure self-play, eval sidecar, GPU heartbeat (82%) and W&B online (twilight-water-183, a7bmcbxb).
  Config vs S4_A_s1: the same six differences as S4f05/S4f10 (run_id, total, floor, floor steps, anneal frac with the
  same N 1,953,125, resume offset 0). 14,603 steps/s after 8 min. Six GPUs in use (4a f05/f10/f20, 4b
  S4f05/S4f10/S4f20); the control's link 2 (22014156) and S4df20_A_s0 / S4d_A_s1 are still queued.
- 2026-11-08 23:20: **stage 4a floor 0.05 done**: S4L_A_s1_f05 link 1 (22014157_1) COMPLETED 20:43 after 18:50:06 on
  ga013. league_state: epoch 671139, global step 10,995,941,376 (the target exactly), not early-stopped. Final
  `model_671139.pt` and `snap_671139.pt` (the stage-5 checkpoint), 192 snapshot files, floor 0.0500 to the last step
  (entropy 0.283 at the end), 14,759 steps/s at the end. Link 2 (22014158) is queued; it will find the run finished and
  exit. Eval sidecar (heuristic pool / held-out) at 10.70/10.80/10.90B: f05 0.964/0.925,
  0.950/0.930, 0.943/0.944; control 0.971/0.955, 0.936/0.940, 0.957/0.951. [Corrected 11-09 02:30: pool is 140 matches
  (95% interval about +-0.04, within noise), but held-out is 1,040 matches (about +-0.015); see the next entry.] Also: S4df20_A_s0 (the stage-3 floor-0.20 arm) finished its 10B on 11-08 11:10 (link 20446673_0,
  step 10,000,007,168, epoch 610352); its next links will exit. Progress at 23:16: 4a f10 10.83B, f20 10.71B; 4b
  S4f05 0.736B, S4f10 0.751B, S4f20 0.717B of 2.50B.
- 2026-11-09 02:30: **stage 4a floor 0.10 done**: S4L_A_s1_f10 link 1 (22014161_2) COMPLETED 02:15 after 17:44:49 on
  ga008. league_state: epoch 671139, global step 10,995,941,376, not early-stopped; `model_671139.pt`,
  `snap_671139.pt`, 192 snapshot files, floor 0.1000 to the end, 15,665 steps/s at the end. Link 2 (22014162) will
  exit. **Eval sidecar vs the heuristic bot, with 95% intervals** (pool 140 matches, held-out 1,040; the sidecar
  loads plain checkpoints, and a loaded checkpoint plays without the floor, `pufferroyale/torch.py`, so these measure
  what the floor did to training, not forced plays at test time):

  | step | control pool / held-out | f05 | f10 | f20 |
  |---|---|---|---|---|
  | 10.70B | 0.971 / 0.955 [0.940,0.966] | 0.964 / 0.925 [0.907,0.939] | 0.950 / 0.900 [0.880,0.916] | 0.907 / 0.900 [0.881,0.917] |
  | 10.80B | 0.936 / 0.940 [0.924,0.953] | 0.950 / 0.930 [0.913,0.944] | 0.943 / 0.898 [0.878,0.915] | 0.879 / 0.904 [0.885,0.921] |
  | 10.90B | 0.957 / 0.951 [0.936,0.963] | 0.943 / 0.944 [0.928,0.956] | 0.907 / 0.889 [0.869,0.907] | (not yet) |

  On the held-out decks the floor costs strength against the heuristic bot: f10 and f20 sit 0.036-0.062 below the
  control, with no interval overlap; f05 is 0.007-0.030 below (overlapping at 10.80B and 10.90B). The pool decks (140
  matches) cannot separate any arm (every interval overlaps the control's). This is the sanity eval, not the stage-5 test
  (exploitability), which decides. f20 at 10.863B (14,110 steps/s), about 2.6 h left, inside link 1 (ends 05:11).
- 2026-11-09 05:05: **stage 4a floor 0.20 done; the whole 4a family is done**: S4L_A_s1_f20 link 1 (22014164_3)
  COMPLETED 04:50 after 19:38:51 on ga011 (inside its 20 h limit). league_state: epoch 671139, global step
  10,995,941,376, not early-stopped; `model_671139.pt`, `snap_671139.pt`, 192 snapshot files, floor 0.2000 to the end,
  14,150 steps/s at the end. Link 2 (22014165) will exit. Last sidecar eval at 10.90B: pool 0.871 [0.806,0.917],
  held-out 0.901 [0.882,0.918] (control 0.957 [0.910,0.980] / 0.951 [0.936,0.963]). All four 4a arms (ctrl, f05,
  f10, f20) stand at global step 10,995,941,376 with `snap_671139.pt`, the stage-5 checkpoints. 4b at 05:03: S4f05
  1.035B, S4f10 1.061B, S4f20 1.019B of 2.50B; their link 1 reaches the 20 h limit at about 05:16-05:32 (TIMEOUT
  expected; link 2 of each chain resumes).
- 2026-11-09 05:45: **stage 5 dry run, 4a family (nothing submitted)**, `runs/stage4/S5a/dryrun/`, rule 5 as written:
  - **Targets:** `league/S4L_A_s1_{ctrl,f05,f10,f20}/snap_671139.pt` (10,995,941,376 steps). Checked: each loads,
    40 tensors with exactly S4_A_s1 snap_476523's shapes, each equal tensor for tensor to the run's
    `model_671139.pt`, four different files (sha256 6357161, d6ab903, 683b295, afe815e; pairwise L1 weight distance
    163-175K).
  - **BR (rule 2):** `br_manifest.txt` (tasks 0-3 = ctrl, f05, f10, f20 with seed 0; 4-7 the same with seed 1; names
    `S5a_S4L_A_s1_<arm>`, outputs `runs/eval/br_S5a_S4L_A_s1_<arm>_s<seed>.json`, none of these names exists yet);
    `br_array.sbatch` = stage 2's `runs/stage4/S4sel/br_array.sbatch` except the comment, job name pr-s5br and W&B
    group S5a_BR (diff checked); BR_STEPS 150,000,000, the rule-2 FLAGS (the same as stage 2's stored BR config:
    deck0 = deck1 = random, no pool, card_floor 0, anchor_frac 1, shaping_anneal_frac 0.5, env_threads 16,
    fast_opponents, cuda_graphs), checkout pinned/s4x_3cd765c, `--array=0-7%4`. W&B: online (env.sh's offline rule
    covers only pr-s4br/pr-s4brs; the 4a/4b training links connected online this week). Cost from stage 2 (2.6-3.5 h
    per BR): about 25 GPU-hours, 2 waves of 4.
  - **Round robin:** `plan_rr_s5a.py` (= stage 1's `tmp/rr_s4/plan_rr.py` with the 4 checkpoints fixed; diff checked)
    -> `rr/plan.json`, `rr/tasks.jsonl`: 6 pairs x 9 presets = 54 tasks, 200 matches each, seed 0;
    `hpc/stage4_eval.sbatch --array=0-1` on the pinned checkout (CPU).
  - `sbatch --test-only` accepted both (estimates: BR 11-11 13:03 on an A100; round robin 12-02 on cs750, the CPU
    partition being in security patching). Commands in `commands.txt`. Waits for the user's go and their choice of
    attacker (copy-start 150M as rule 5, or from-scratch 1B as stage 3b).
- 2026-11-09 11:35: **stage 5, 4a family, submitted** (user, 2026-11-09: "Copy-start now", i.e. rule 5 as written:
  copy-start BR, 150M steps). `runs/stage4/S5a/sub_20261109_113404/` = the dry run copied (its `rr/out` empty):
  BR array **22422166** (`--array=0-7%4`, pr-s5br, the dry run's manifest, FLAGS, BR_STEPS 150,000,000 and checkout)
  and round robin **22422168** (`hpc/stage4_eval.sbatch --array=0-1`, EVAL_DIR = the submission's `rr/`, outputs to
  `rr/out/`; neither plan file refers to the dry-run directory). Job ids in `jobids.txt`. Both PENDING (reason None)
  at 11:35. With the 3 running 4b links, at most 7 GPUs at once (as stage 4).
- 2026-11-09 11:40: **4b link 1 -> link 2**: all three link 1s ended TIMEOUT at the 20 h limit (05:16, 05:19, 05:31; A100,
  at 1.047B / 1.075B / 1.044B), and link 2 started on H100s (22014178_0 gh007 06:48, 22014198_0 gh006 06:49,
  22014284_0 gh008 07:38). Each resumed from its last snapshot (epochs 63918 / 65349 / 63441 = 134 / 137 / 133 x
  477; up to 477 epochs of link 1 are redone, as in every chain), same total 2,500,853,760, same floor, checkout
  3cd765c. At 11:34: 1.361B / 1.381B / 1.292B at 17,900-18,300 steps/s, so each needs about 17-19 h more: past link
  2's limit (02:48-03:38 on 11-10) by a few hours, so link 3 finishes them (after its queue wait).
- 2026-11-09 15:55: **stage 5 (4a) round robin started; my error in its setup**. 22422168_0 (cs777) and _1 (cs779)
  started about 15:45 on the pinned checkout 3cd765c, 32 processes each, EVAL_DIR the submission's `rr/`. When I copied
  the dry run into `sub_20261109_113404/` I deleted `rr/out` (meant: start with no outputs), but `stage4_eval.py run`
  does not create that folder, so a finished task could not save: task 24 failed at 15:5x with FileNotFoundError on
  `rr/out/0024.json.tmp`. Folder recreated at 15:51:55; the tasks still running save normally. After the array ends,
  the same array is submitted again for the missing task(s) (finished tasks are skipped by design). The BR array
  22422166 is still PENDING (reason None).
- 2026-11-09 16:05: **stage 5 (4a) round robin: 52 of 54 tasks done**. 22422168_0/_1 COMPLETED at 15:57-16:00 (11:39
  and 11:46). Missing: tasks 24 and 42, both of element 0, both with the FileNotFoundError above (42 finished just
  before the folder was recreated; the first saved output is 15:52:26). Rerun **22449446** (`--array=0`, so one
  element walks all 54 and skips the 52 present; 2 CPUs, 8G, 1 h; same EVAL_DIR and checkout). The merge waits for it.
- 2026-11-09 16:15: **stage 5 (4a) round robin done** (rerun 22449446 COMPLETED 16:10 in 5:52, tasks 24 and 42; 54 of
  54 outputs; merged with the pinned checkout: `runs/stage4/S5a/sub_20261109_113404/rr/summary.{json,md}`). The 4
  arms at snap_671139, the 9 presets as mirrors, 1,800 matches per pair (row's score vs column, 95% Wilson):

  | row vs col | ctrl | f05 | f10 | f20 | Elo | Nash |
  |---|---|---|---|---|---|---|
  | ctrl | – | 0.679 [0.657,0.701] | 0.686 [0.665,0.707] | 0.819 [0.801,0.836] | 1634 | 1.00 |
  | f05 | 0.321 | – | 0.427 [0.404,0.450] | 0.713 [0.692,0.734] | 1493 | 0.00 |
  | f10 | 0.314 | 0.573 | – | 0.734 [0.713,0.754] | 1523 | 0.00 |
  | f20 | 0.181 | 0.287 | 0.266 | – | 1349 | 0.00 |

  The control beats every floor arm (no interval touches 0.5) and is the whole Nash mixture; f20 loses to all. Under
  rule 5 the round robin is only the tie-break: the verdict is the BR score (22422166, still PENDING). The "C vs A"
  line of the summary is meaningless here (no C runs), as in stage 1.
- 2026-11-09 18:51: **status**. Stage 5 (4a) BR array 22422166 still PENDING (reason None, priority 11639; waiting
  since 11:35). 4b link 2s running on H100s: S4f05 1.833B (18,140 steps/s), S4f10 1.845B (17,888), S4f20 1.750B
  (17,626) of 2,500,853,760. At those rates link 2 ends (20 h limit: 02:48 / 02:49 / 03:38 on 11-10) at about 2.35B /
  2.36B / 2.31B, so each link 3 needs about 0.14B / 0.14B / 0.19B (2-3 h plus its queue wait). 19555051_[3]
  (pr-S4d_A_s1) is a leftover link of a run that finished 2026-11-05 (10,000,007,168 steps): it will exit at once if it
  starts, as 19555049_3 did; left queued (no-cancel rule).
- 2026-11-09 19:20: **requested follow-up (user, 2026-11-09; "need not happen right now")**: on the best model, 60
  Claude games: 20 vs Claude Fable (claude-fable-5-1), 20 vs Claude Opus (claude-opus-5-5), 20 vs Claude Haiku
  (claude-haiku-5-5). Not started. Existing harness (D101): `hpc/llm.sbatch` + `hpc/llm_game.py` + `hpc/llm_relay.py`
  (the Claude Code CLI on the owner's login, `--model` / `--effort`; no API key, $0 extra). Earlier Claude games: 63
  (S4_C_s0 18, S4d_A_s0 34, S4d_C_s0 10, S4pilot_C_s1 1), all claude-opus-5-5 at effort low (job logs), 1 s decision
  interval, Claude decks xbow / royal_hogs. To settle when it runs: which model is "the best" (the stage 5 verdict;
  confirm with the user), one setup for all three Claude models (decks, seats, effort, decision interval), and a
  one-call check that Fable answers on the owner's login before the 60 games.
- 2026-11-09 19:30: **follow-up amended (user, 2026-11-09: "and 20 on Sonnet - so 80 total")**: add 20 games vs Claude
  Sonnet (claude-sonnet-5-5): 80 games, 20 each for Fable, Opus, Sonnet and Haiku. Still not started.
- 2026-11-09 19:40: **follow-up corrected (user, 2026-11-09: "No - against fable-5-2")**: the Fable games use Fable 5.2,
  not claude-fable-5-1 as written at 19:20. CLI name assumed `claude-fable-5-2` (not yet checked: the one-call test
  before the games confirms it). Games: 20 each for Fable 5.2, Opus 5.5, Sonnet 5.5, Haiku 5.5 = 80. Still not started.
- 2026-11-10 03:00: **4b link 2 -> link 3 (S4f05, S4f10)**: S4f05 (22014178_0) and S4f10 (22014198_0) link 2 ended
  TIMEOUT at the 20 h limit (02:49:03 / 02:49:33) at 2,345,648,128 / 2,355,462,144 steps (last log lines); last
  snapshots `snap_143100.pt` (02:48, 2,344,550,400 steps) / `snap_143577.pt` (02:46, 2,352,365,568 steps), the
  resume points of link 3. Left to 2,500,853,760: 0.156B / 0.148B (about 2.3-2.4 h at 18,000 steps/s). Link 3
  (22014180_0 / 22014202_0) is PENDING with reason QOSMaxGRESPerUser: the per-user GPU cap of QOS gpu48 reads
  gres/gpu=0 now (Torch has shown it at 2, 0 and 16 before), with 1 GPU of ours in use (S4f20 link 2, ends 03:38).
  The stage 5 (4a) BR array 22422166 waits for the same reason. Nothing to change on our side (no-cancel rule; Torch
  does not allow edits to pending jobs); they start when the cap rises.
- 2026-11-10 03:45: **4b S4f20 link 2 ended; GPU cap back**: S4f20 (22014284_0) link 2 ended TIMEOUT at 03:39:13 at
  2,318,008,320 steps (last log line); last snapshot `snap_141192.pt` (03:35, 2,313,289,728 steps), 0.188B left
  (about 2.9 h at 17,750 steps/s). At 03:41 the gpu48 per-user cap reads gres/gpu=16 again; the three link 3s
  (22014180_0, 22014202_0, 22014285_0) and the BR array 22422166 are PENDING with reason None (waiting for a node).
- 2026-11-10 17:10: **Claude games started (user, 2026-11-10: "On our best model - lets start the claude games -> play 3
  Fable 5-2 based games")**. Model: the current best, S4_A_s1 `model_610352.pt` (10,000,007,168 steps; sha256 46c62cdd...,
  = the released `models/S4_A_s1_10B.pt`); the stage 5 verdicts are still pending. One-call check (login node, 17:00,
  the relay's exact flags, effort low): `claude-fable-5-2` answers (modelUsage `claude-fable-5-2`, 7.8 s, a valid
  `PLAY 3 AT 5,25`). Setup = the D101 harness, the same as the earlier 63 Opus games: `hpc/llm.sbatch` (now takes
  `LLM_RUN_DIR`, a folder with the run's config.json and a link to the checkpoint) + `hpc/llm_game.py` +
  `hpc/llm_relay.py` (Claude Code CLI on the owner's login, no API key), checkout 3cd765c, effort low, decisions every
  20 ticks (1 s), the policy every 10 ticks, sampled actions; the 20-game cycle of `llm_game.py` (LLM_EVERY 500M of
  the 10B total, every game against the same file): games 1-3 = Claude xbow vs xbow, royal_hogs vs royal_hogs, xbow vs
  hog26, Claude team 0, seeds crc32("S4_A_s1:k") (the same seeds for every Claude model). Folder
  `runs/claude_games/S4_A_s1_10B/claude-fable-5-2/` (`job.env`, `llm/state.json`, `llm/game_00k/`). Job 22595211
  (pr-cg-fable52, cs629, cpu48, 4 h limit, `--max-games 3`); game 1 started 17:05, about 4.5 s per decision.
- 2026-11-10 18:10: **Claude games, Fable 5.2, games 1-3 done** (job 22595211 COMPLETED 18:04 after 1:02:01; 497 CLI
  calls, 0 failed, 0 illegal plays, 0 parse errors; mean 6.5-7.0 s per call; the CLI's list-price estimate $18.64 in
  total, on the owner's plan, no API key). Claude's result (team 0; tiebreak = the weakest surviving crown tower,
  absolute HP, SPEC 4):

  | game | Claude deck vs policy deck | result (Claude) | crowns | end | weakest tower Claude / policy | decisions (plays) |
  |---|---|---|---|---|---|---|
  | 1 | xbow vs xbow | **win** | 0-0 | tiebreak at 6000 | 961 / 757 | 186 (81) |
  | 2 | royal_hogs vs royal_hogs | loss | 0-1 | regulation crowns at 3600 | 515 / 509 (one Princess tower lost) | 97 (34) |
  | 3 | xbow vs hog26 | loss | 0-0 | tiebreak at 6000 | 688 / 2,173 | 214 (78) |

  Fable 5.2 score 1.0 / 3 = 0.333 (1 W, 0 D, 2 L); n = 3, no interval worth stating. Replays
  `runs/claude_games/S4_A_s1_10B/claude-fable-5-2/llm/game_00{1,2,3}/game_00k.gif`, every tick in
  `frames.jsonl.gz`, Claude's replies in `decisions.jsonl`. Games 4-20 not started.
- 2026-11-10 18:20: **counting rule for the Claude games (user, 2026-11-10: "from now on - when there's a draw - lets
  count it as a draw not a win/loss")**: a game that ends with equal crowns counts as a draw (0.5), whatever the
  engine's tiebreak says (the engine still plays to the tiebreak and records it; only the count changes). Recount of
  Fable 5.2 games 1-3 from the policy's side: game 1 (0-0, tiebreak) draw, game 2 (policy 1-0) win, game 3 (0-0,
  tiebreak) draw = **policy 1 W, 2 D, 0 L, score 0.667; Fable 0 W, 2 D, 1 L, 0.333** (the same 0.333 as the tiebreak
  count above, by coincidence: 1 W 2 L then, 2 D 1 L now). The tiebreak result stays in each `result.json`.
  Not applied to the stage 1-5 selection scores (rules written before the results; changing them is the user's call).
- 2026-11-10 19:45: **Claude games, Fable 5.2, games 4-10 started (user, 2026-11-10: "now lets play 7 more matches
  against Fable 5.2")**: job 22609722 (pr-cg-fable52, cs608, 6 h limit), env `job_games4-10.env` (= `job.env` with
  `--max-games 7`), resuming from `llm/state.json` (games 1-3 done). Schedule (Claude deck vs policy deck, Claude's
  team): 4 royal_hogs vs hog26 (0); 5 xbow vs xbow (1); 6 royal_hogs vs royal_hogs (1); 7 xbow vs giant (1); 8
  royal_hogs vs giant (1); 9 xbow vs xbow (0); 10 royal_hogs vs royal_hogs (0). Counted with the draw rule (equal
  crowns = draw). Submitted 19:41:30, game 4 started 19:41. (Times corrected at 20:10: first written as 18:35 /
  18:31 by my mistake.)
- 2026-11-10 20:10: **Fable 5.2 game 4 done**: Claude royal_hogs (team 0) vs policy hog26: policy 1-0 in overtime
  (OVERTIME_CROWNS, 861 s, 112 decisions). Running count (policy's side, draw rule), games 1-4: 2 W, 2 D, 0 L. Game 5
  (xbow mirror, Claude team 1) in progress: 0-0 with 0:31 of regulation left, Claude's weakest tower 3,017, the
  policy's 850.
- 2026-11-10 21:45: **Claude games, Fable 5.2, games 4-10 done** (job 22609722 COMPLETED 21:37 after 1:55:38; 902 CLI
  calls, 0 failed; 1 illegal play (game 9), 0 parse errors; mean 6.7-7.8 s per call; list-price estimate $31.56).
  All 10 games, counted from the policy's side with the draw rule (equal crowns = draw; the engine's tiebreak winner
  is in `result.json` but not counted):

  | game | Claude deck vs policy deck | Claude team | crowns Claude-policy | end (tick) | policy's result | weakest tower Claude / policy |
  |---|---|---|---|---|---|---|
  | 1 | xbow vs xbow | 0 | 0-0 | tiebreak (6000) | draw | 961 / 757 |
  | 2 | royal_hogs vs royal_hogs | 0 | 0-1 | regulation crowns (3600) | win | 515 / 509 |
  | 3 | xbow vs hog26 | 0 | 0-0 | tiebreak (6000) | draw | 688 / 2,173 |
  | 4 | royal_hogs vs hog26 | 0 | 0-1 | overtime crowns (4337) | win | 3,052 / 1,757 |
  | 5 | xbow vs xbow | 1 | 1-0 | regulation crowns (3600) | **loss** | 2,982 / 678 |
  | 6 | royal_hogs vs royal_hogs | 1 | 0-1 | regulation crowns (3600) | win | 1,927 / 591 |
  | 7 | xbow vs giant | 1 | 0-0 | tiebreak (6000) | draw | 485 / 2,100 |
  | 8 | royal_hogs vs giant | 1 | 1-3 | King destroyed (3599) | win | 278 / 1,507 |
  | 9 | xbow vs xbow | 0 | 0-1 | overtime crowns (3650) | win | 2,569 / 1,993 |
  | 10 | royal_hogs vs royal_hogs | 0 | 0-1 | overtime crowns (4421) | win | 3,052 / 1,557 |

  **Policy vs Fable 5.2 after 10 games: 6 W, 3 D, 1 L, score 0.75** (Fable 1 W, 3 D, 6 L = 0.25). By Claude's deck:
  royal_hogs games (2, 4, 6, 8, 10) policy 5 W of 5; xbow games (1, 3, 5, 7, 9) 1 W, 3 D, 1 L. Wilson 95% interval on
  the score with n = 10 is wide (draws make it inexact; treating 7.5/10 as a proportion gives [0.44, 0.92]). Games 11-20 not started.
- 2026-11-10 22:35: **Claude games, Haiku 5.5, games 1-10 started (user, 2026-11-10: "lets do 10 games against Haiku
  nlw")**. One-call check (login node, 22:2x, the relay's flags, effort low): `claude-haiku-5-5` answers (modelUsage
  `claude-haiku-5-5`, 5.5 s, a valid `PLAY 3 AT 6,22`). Folder `runs/claude_games/S4_A_s1_10B/claude-haiku-5-5/`:
  the same config.json, checkpoint link (sha256 46c62cdd...), checkout 3cd765c, effort low, LLM_EVERY 500M and seeds as
  the Fable 5.2 games (`job.env` differs only in LLM_RUN_DIR, LLM_MODEL and `--max-games 10`), so games 1-10 repeat
  Fable's 10 setups exactly. Job 22623622 (pr-cg-haiku55, cs680, 6 h limit) started 22:31; first decisions 4-5.5 s.
  Counted with the draw rule.
- 2026-11-10 23:35: **Claude games, Haiku 5.5, games 1-10 done** (job 22623622 COMPLETED 23:26 after 0:54:35; 633 CLI
  calls, 0 failed; 0 illegal plays, 0 parse errors; mean 4.5 s per call; list-price estimate $0.37). Policy's result
  per game (draw rule), next to Fable 5.2 in the same setup (same seed, decks, seats):

  | game | Claude deck vs policy deck | Claude team | Haiku: crowns Claude-policy, end | policy vs Haiku | policy vs Fable |
  |---|---|---|---|---|---|
  | 1 | xbow vs xbow | 0 | 0-1, regulation | win | draw |
  | 2 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win | win |
  | 3 | xbow vs hog26 | 0 | 0-1, regulation | win | draw |
  | 4 | royal_hogs vs hog26 | 0 | 0-1, regulation | win | win |
  | 5 | xbow vs xbow | 1 | 0-1, regulation | win | loss |
  | 6 | royal_hogs vs royal_hogs | 1 | 0-3, King destroyed at tick 3,231 | win | win |
  | 7 | xbow vs giant | 1 | 1-2, regulation | win | draw |
  | 8 | royal_hogs vs giant | 1 | 1-2, regulation | win | win |
  | 9 | xbow vs xbow | 0 | 0-0, tiebreak (6000) | draw | win |
  | 10 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win | win |

  **Policy vs Haiku 5.5: 9 W, 1 D, 0 L, score 0.95** (proportion-style Wilson 95% about [0.66, 0.99]); vs Fable 5.2:
  6 W, 3 D, 1 L, 0.75 ([0.44, 0.92]). Style difference (from `result.json`): Haiku plays a card on 72% of its turns
  (453 of 633), Fable on 36% (506 of 1,399); a turn is skipped when no card is affordable, so Haiku's games have
  fewer turns (34-126 vs 88-214). Haiku games 11-20 not started.
- 2026-11-11 14:22: **Claude games, Sonnet 5.5, games 1-10 started (user, 2026-11-11: "Lets play 10 Sonnet games
  now")**. One-call check (login node, 14:20, the relay's flags, effort low): `claude-sonnet-5-5` answers (modelUsage
  `claude-sonnet-5-5`, 6.1 s, `WAIT`). Folder `runs/claude_games/S4_A_s1_10B/claude-sonnet-5-5/`: the same config.json,
  checkpoint link (sha256 46c62cdd...), checkout 3cd765c, effort low, LLM_EVERY 500M, seeds and `--max-games 10` as the
  Haiku games (`job.env` differs only in LLM_RUN_DIR and LLM_MODEL). Job 22718409 (pr-cg-sonnet55, cs660, 6 h limit)
  started 14:19:57; first decisions 4-8.6 s. Counted with the draw rule. Queue at 14:18: the 4b link 3s and the BR
  array 22422166 PENDING with reason QOSGrpGRES (the QOS's group GPU total is in use; per-user cap 16).
- 2026-11-11 16:40: **Claude games, Sonnet 5.5, games 1-10 done** (job 22718409 COMPLETED 16:34 after 2:14:05; 1,522
  CLI calls, 0 failed; 0 parse errors; 11 illegal plays; mean about 4.7 s per call; list-price estimate $14.37).
  Illegal plays were checked in `decisions.jsonl` and the turn files: in each case Sonnet picked a tile missing from the
  LEGAL TILES list it was given, e.g. game 3 turn 13 "PLAY 1 AT 15,19" (Ice Spirit) while its own X-Bow at (14,18)
  covered tx 13-15 of row 19. The harness then plays WAIT for that turn. Policy's result per game (draw rule), next to
  Fable 5.2 and Haiku 5.5 in the same setup (same seed, decks, seats):

  | game | Claude deck vs policy deck | Claude team | Sonnet: crowns Claude-policy, end | policy vs Sonnet | vs Fable | vs Haiku |
  |---|---|---|---|---|---|---|
  | 1 | xbow vs xbow | 0 | 1-0, overtime (tick 4,437) | loss | draw | win |
  | 2 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win | win | win |
  | 3 | xbow vs hog26 | 0 | 0-0, tiebreak (6000) | draw | draw | win |
  | 4 | royal_hogs vs hog26 | 0 | 0-1, regulation | win | win | win |
  | 5 | xbow vs xbow | 1 | 1-0, overtime (tick 4,995) | loss | loss | win |
  | 6 | royal_hogs vs royal_hogs | 1 | 0-1, regulation | win | win | win |
  | 7 | xbow vs giant | 1 | 0-1, regulation | win | draw | win |
  | 8 | royal_hogs vs giant | 1 | 1-3, King destroyed in overtime (tick 4,592) | win | win | win |
  | 9 | xbow vs xbow | 0 | 0-0, tiebreak (6000) | draw | win | draw |
  | 10 | royal_hogs vs royal_hogs | 0 | 0-1, overtime (tick 4,813) | win | win | win |

  **Policy vs Sonnet 5.5: 6 W, 2 D, 2 L, score 0.70** (proportion-style Wilson 95% about [0.40, 0.89]). Fable 0.75
  ([0.44, 0.92]), Haiku 0.95 ([0.66, 0.99]); all 30 games 21 W 6 D 3 L, 0.80 ([0.63, 0.91]). By Claude's deck: Royal
  Hogs 15 of 15 policy wins across the three models; X-Bow 6 W 6 D 3 L (0.60, [0.36, 0.80]). Both Sonnet wins are X-Bow
  mirrors decided in overtime. Game 5: at tick 4,460 the policy's towers were 2,255 and 2,870 and Sonnet's right tower
  790; Sonnet's Archers plus a Fireball (tick 4,940, tower to 80 HP) took the policy's tower at tick 4,995. Game 8: 1-1
  at the end of regulation, Sonnet's King at 36 HP from about tick 3,780, destroyed at 4,592. Style: Sonnet plays a card
  on 31% of its calls (479 of 1,522), Fable 36%, Haiku 72%. Sonnet games 11-20 not started. Queue at 16:35: BR array
  22422166 and 4b link 3s PENDING; squeue --start estimate 2026-11-12 20:17 for all four (scheduler estimate only).
- 2026-11-12 21:20 (time corrected; first written as 21:40): **stage 5 (4a) BR array done; verdict for the 4a family: the control.** Job 22422166, all 8 tasks
  COMPLETED 2026-11-12 between 05:18 and 10:12 (started 02:43-07:39, 2.5-2.7 h each, A100/H100 nodes gh003-gh015). All 8
  outputs `runs/eval/br_S5a_S4L_A_s1_<arm>_s<seed>.json` checked against `br_manifest.txt` (target, seed, 1000 matches,
  150,011,904 >= 150M steps, random decks both sides, init_from_target, br_score = (W + D/2)/1000, W+D+L = 1000): no
  mismatch; no "error"/"traceback" in the task logs. The W&B levelling-off check (rule 2, not a rule) is not done yet.

  | arm (snap_671139) | seed 0: W-D-L, br_score | seed 1: W-D-L, br_score | mean (as written) | mean, < 0.5 read as 0.5 (dev. 2) | rr Elo |
  |---|---|---|---|---|---|
  | ctrl | 436-0-564, 0.4360 | 470-0-530, 0.4700 | 0.4530 | **0.5000** | 1634 |
  | f10 | 456-11-533, 0.4615 | 541-34-425, 0.5580 | 0.5098 | 0.5290 | 1523 |
  | f05 | 582-140-278, 0.6520 | 614-84-302, 0.6560 | 0.6540 | 0.6540 | 1493 |
  | f20 | 697-2-301, 0.6970 | 708-2-290, 0.7080 | 0.7025 | 0.7025 | 1349 |

  - Rule 3 with deviation 2: lowest 0.500 (ctrl), tie band <= 0.550 holds ctrl and f10 (0.529); Elo tie-break ctrl
    1634 > f10 1523: **ctrl**. Rule 3 as written (no deviation): ctrl 0.453 lowest, f10 0.057 away, no tie: **ctrl**.
    Both readings agree.
  - Exploits found: f05 and f20 in both seeds (0.652 / 0.656, 0.697 / 0.708); f10 in one seed (0.558; other 0.4615);
    the control in neither (0.436, 0.470). Order of exploitability = order of the round robin (ctrl best, f20 worst).
  - Training score vs target at the end (from the same files): ctrl 0.520 / 0.512, f05 0.643 / 0.645, f10 0.545 / 0.528,
    f20 0.713 / 0.704.
  - Meaning for 4a: on S4_A_s1 at 10B, a constant card floor (0.05 / 0.10 / 0.20) for 1B more steps made the model more
    exploitable and weaker in the round robin than continuing its own schedule (floor 0 after 5B).
  - Not measured: the control at 11.0B (S4L_A_s1_ctrl/snap_671139) against the released S4_A_s1 10B
    (model_610352); which file is "the best model" (Claude games, release) is the user's call after 4b.
- 2026-11-12 21:20 (time corrected; first written as 21:40): **4b runs done.** Link 3s 22014180_0 (gh006, 08:01-10:30), 22014202_0 (gh009, 08:01-10:28),
  22014285_0 (gh015, 08:11-11:17) COMPLETED; each resumed one epoch after the snapshot recorded on 11-10 (first log
  lines: epoch 143101 / 143578 / 141193, floor 0.0500 / 0.1000 / 0.2000), checkout 3cd765c, and stopped at global step
  2,500,853,760 (epoch 152640, `summary_<job>.json`). Configs: seed 1, init_from None, total 2,500,853,760,
  card_floor 0.05 / 0.1 / 0.2, card_floor_steps 0. Stage-5 checkpoints: `snap_152640.pt` of each (= its
  `model_152640.pt` tensor for tensor). Link 4s 22014183 / 22014203 / 22014286 still PENDING (Priority): they will start
  and exit at once (the runs are complete), as 19555051_3 did on 2026-11-11 23:30 (COMPLETED in 36 s); left queued
  (no-cancel rule).
- 2026-11-12 21:22 (time corrected; first written as 21:45): **stage 5 dry run, 4b family (nothing submitted)**, `runs/stage4/S5b/dryrun/`, rule 5 as written,
  built from the 4a dry run (diffs checked):
  - **Targets:** `league/{S4_A_s1,S4f05_A_s1,S4f10_A_s1,S4f20_A_s1}/snap_152640.pt` (2,500,853,760 steps; the control
    is the winner's own run at 2.5B, rule 4). Each loads, 40 tensors with S4_A_s1 snap_476523's shapes; four different
    files (sha256 9124642, 18eb67e, a3e43d6, b3ddf2b).
  - **BR:** `br_manifest.txt` (tasks 0-3 = S4_A_s1, S4f05, S4f10, S4f20 with seed 0; 4-7 seed 1; names `S5b_<run>`,
    outputs `runs/eval/br_S5b_<run>_s<seed>.json`, none exists yet); `br_array.sbatch` = the 4a one except the comment,
    W&B group S5b_BR and the tag prefix; BR_STEPS 150M, the rule-2 FLAGS, checkout pinned/s4x_3cd765c, `--array=0-7%4`.
  - **Round robin:** `plan_rr_s5b.py` (= `plan_rr_s5a.py` with the 4 checkpoints) -> `rr/plan.json`, `rr/tasks.jsonl`
    (54 tasks = 6 pairs x 9 presets, 200 matches each, seed 0); `rr/out/` created (the 2026-11-09 error).
  - `sbatch --test-only` accepted both (estimates: BR 2026-11-15 12:48 on an A100; round robin 2026-11-20 on cs613;
    estimates only). Commands in `commands.txt`. Waits for the user's go.
- 2026-11-12 21:24: **stage 5, 4b family, submitted** (user, 2026-11-12: "4b is a go"). `runs/stage4/S5b/sub_20261112_212401/`
  = the dry run copied (its `rr/out` exists and is empty; no plan or manifest file refers to the dry-run folder;
  `commands.txt` paths updated): BR array **22933924** (`--array=0-7%4`, pr-s5br, the dry run's manifest, FLAGS,
  BR_STEPS 150,000,000, checkout pinned/s4x_3cd765c) and round robin **22933925** (`hpc/stage4_eval.sbatch --array=0-1`,
  EVAL_DIR = the submission's `rr/`). Job ids in `jobids.txt`. At 21:24: BR PENDING (reason None), round robin PENDING
  (Priority). At most 4 GPUs at once (the 4b training links are done).
- 2026-11-14 08:55: **stage 5 (4b) done; verdict for the 4b family: the control** (S4_A_s1 at snap_152640, the winner's
  own run at 2.5B). BR array 22933924: all 8 tasks COMPLETED 2026-11-13 between 10:42 and 23:46 (2.6-3.4 h each,
  gh004-gh014 and ga021). All 8 outputs `runs/eval/br_S5b_<run>_s<seed>.json` checked against `br_manifest.txt` as for
  4a: no mismatch; no "error"/"traceback" in the BR or round-robin logs. Round robin 22933925_0/_1 COMPLETED 2026-11-13
  09:38 / 09:39 (12-13 min, cs702 / cs704), 54 of 54 outputs, merged with the pinned checkout:
  `runs/stage4/S5b/sub_20261112_212401/rr/summary.{json,md}`. The 4b link 4s (22014183_0 / 22014203_0 / 22014286_0)
  started 2026-11-13 07:59-08:02 and exited in 26-45 s, as expected.

  | arm (snap_152640) | seed 0: W-D-L, br_score | seed 1: W-D-L, br_score | mean (as written) | mean, < 0.5 read as 0.5 (dev. 2) | rr Elo |
  |---|---|---|---|---|---|
  | ctrl = S4_A_s1 | 422-138-440, 0.4910 | 365-253-382, 0.4915 | 0.4912 | **0.5000** | 1566 |
  | f05 | 680-0-320, 0.6800 | 622-0-378, 0.6220 | 0.6510 | 0.6510 | 1491 |
  | f10 | 688-86-226, 0.7310 | 595-127-278, 0.6585 | 0.6947 | 0.6947 | 1520 |
  | f20 | 735-3-262, 0.7365 | 692-0-308, 0.6920 | 0.7143 | 0.7143 | 1423 |

  Round robin (9 preset mirrors, 1,800 matches per pair, row's score vs column, 95% Wilson):

  | row vs col | ctrl | f05 | f10 | f20 | Elo | Nash |
  |---|---|---|---|---|---|---|
  | ctrl | – | 0.628 [0.605,0.650] | 0.614 [0.591,0.636] | 0.623 [0.600,0.645] | 1566 | 1.00 |
  | f05 | 0.372 | – | 0.492 [0.469,0.515] | 0.586 [0.563,0.609] | 1491 | 0.00 |
  | f10 | 0.386 | 0.508 | – | 0.718 [0.697,0.738] | 1520 | 0.00 |
  | f20 | 0.378 | 0.414 | 0.282 | – | 1423 | 0.00 |

  - Rule 3 with deviation 2: ctrl 0.500, next f05 0.651 (0.151 away): no tie, **ctrl**. As written: ctrl 0.491, the
    same. Both readings agree.
  - Exploits found in both seeds of every floor arm (0.62-0.74); none against the control (0.491, 0.4915; its BRs draw
    138 and 253 of 1000). The control also beats each floor arm head to head (0.61-0.63, no interval touches 0.5).
  - **Stage 5 overall: in both families (from 10B continued, and from scratch to 2.5B) the control, i.e. S4_A_s1's own
    floor schedule (0.005 faded to 0 by 5B), is the least exploitable and the highest-Elo arm. A constant floor of
    0.05-0.20 never helped.**
  - Still open (user): whether the 4a control at 11.0B (S4L_A_s1_ctrl/snap_671139) replaces the released S4_A_s1 10B
    (model_610352) as "the best model"; the two have not been compared directly.
- 2026-11-14 08:54: **Claude games, Sonnet 5.5, games 11-20 started** (user, 2026-11-14: "please run 10 more Sonnet
  games"). One-call check (login node, 08:53, the relay's flags, effort low, RULES_PROMPT extracted from the pinned
  checkout, Sonnet game 10's turn 5): `claude-sonnet-5-5` answers (modelUsage `claude-sonnet-5-5`, 10.0 s, `WAIT`).
  `job_games11-20.env` = `job.env` except the comment (same model, run dir, checkpoint link, LLM_EVERY 500M,
  `--max-games 10`, which resumes from `llm/state.json` with games 1-10 done). Job 23119369 (pr-cg-sonnet55, cs601,
  6 h limit) running at 08:54; game 11 = Claude xbow vs policy bait, Claude team 0. Counted with the draw rule.
- 2026-11-14 09:05: **Claude games, Opus 5.5, games 1-10 started, in parallel with Sonnet 11-20** (user, 2026-11-14:
  "in parallel - run Opus 1-10 too"). Checks first: (1) the relay retries a failed call (e.g. a usage limit) with
  back-off and after 3 h of failures (`--max-wait` 10800) aborts the game, which `llm_game.py` retries later (up to 3
  tries), so a limit stalls games instead of turning Claude's moves into WAITs; (2) the game waits for each reply
  (`FileRelay`, up to 14400 s), so call speed cannot change a result; (3) the 63 earlier `claude-opus-5-5` games
  (D101, league/*/llm) cost a list-price estimate of $2.21 per game (130 calls per game, 0 failed calls). One-call
  check (login node, 09:03, the relay's flags, effort low): `claude-opus-5-5` answers (modelUsage `claude-opus-5-5`,
  7.3 s, `WAIT`). Folder `runs/claude_games/S4_A_s1_10B/claude-opus-5-5/`: config.json identical to Sonnet's,
  checkpoint link (sha256 46c62cdd...), `job.env` = Sonnet's except comment, LLM_RUN_DIR and LLM_MODEL. Job 23119788
  (pr-cg-opus55, cs621, 6 h limit) running at 09:04; game 1 has Sonnet game 1's seed, decks and seat. Counted with the
  draw rule.
- 2026-11-14 10:50: **Claude games, Opus 5.5, games 1-10 done** (job 23119788 COMPLETED 10:47 after 1:43:21, in parallel
  with Sonnet 11-20 the whole time; 1,213 CLI calls, 0 failed, 0 retries; 0 parse errors; 3 illegal plays; mean
  4.5 s per call; list-price estimate $19.83). Policy's result per game (draw rule), same setups as the other models:

  | game | Claude deck vs policy deck | Claude team | Opus: crowns Claude-policy, end | vs Opus | vs Sonnet | vs Fable | vs Haiku |
  |---|---|---|---|---|---|---|---|
  | 1 | xbow vs xbow | 0 | 1-0, regulation | loss | loss | draw | win |
  | 2 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win | win | win | win |
  | 3 | xbow vs hog26 | 0 | 0-0, tiebreak (6000) | draw | draw | draw | win |
  | 4 | royal_hogs vs hog26 | 0 | 0-1, regulation | win | win | win | win |
  | 5 | xbow vs xbow | 1 | 0-1, regulation | win | loss | loss | win |
  | 6 | royal_hogs vs royal_hogs | 1 | 0-1, regulation | win | win | win | win |
  | 7 | xbow vs giant | 1 | 0-1, regulation | win | win | draw | win |
  | 8 | royal_hogs vs giant | 1 | 0-2, regulation | win | win | win | win |
  | 9 | xbow vs xbow | 0 | 0-1, regulation | win | draw | win | draw |
  | 10 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win | win | win | win |

  **Policy vs Opus 5.5: 8 W, 1 D, 1 L, score 0.85** (Wilson about [0.54, 0.96]); Sonnet 0.70, Fable 0.75, Haiku 0.95
  (games 1-10). All 40 games 1-10: 29 W 7 D 4 L, 0.81. Royal Hogs: 20 of 20 policy wins across the four models.
  Opus plays a card on 33% of its calls (396 of 1,213; Sonnet 31%, Fable 36%, Haiku 72%). Game 3 (xbow vs hog26) is a
  0-0 draw for Opus, Sonnet and Fable. Opus games 11-20 not started.
- 2026-11-14 11:00: **Claude games, Sonnet 5.5, games 11-20 done** (job 23119369 COMPLETED 10:55 after 2:00:28, in
  parallel with Opus 1-10 until 10:47; 1,250 CLI calls, 0 failed, 0 retries; 0 parse errors; 21 illegal plays; mean
  5.2 s per call; list-price estimate $11.58). Policy's result per game (draw rule):

  | game | Claude deck vs policy deck | Claude team | Sonnet: crowns Claude-policy, end | policy vs Sonnet |
  |---|---|---|---|---|
  | 11 | xbow vs bait | 0 | 0-1, regulation | win |
  | 12 | royal_hogs vs bait | 0 | 0-1, regulation | win |
  | 13 | xbow vs xbow | 1 | 0-0, tiebreak (6000) | draw |
  | 14 | royal_hogs vs royal_hogs | 1 | 0-3, King destroyed at tick 2,164 | win |
  | 15 | xbow vs golem | 1 | 0-1, regulation | win |
  | 16 | royal_hogs vs golem | 1 | 1-3, King destroyed at tick 1,065 | win |
  | 17 | xbow vs xbow | 0 | 0-1, regulation | win |
  | 18 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win |
  | 19 | xbow vs lavaloon | 0 | 0-0, tiebreak (6000) | draw |
  | 20 | royal_hogs vs lavaloon | 0 | 0-1, regulation | win |

  **Games 11-20: 8 W, 2 D, 0 L (0.90, Wilson about [0.60, 0.98]). Sonnet games 1-20: 14 W, 4 D, 2 L, score 0.80
  ([0.58, 0.92])**; 2,772 calls, 32 illegal plays, $25.95 in all. Game 16: crowns 1-1 at tick 900 (45 s); Sonnet's
  King went from 4,656 HP (tick 900) to destroyed at tick 1,065 (`frames.jsonl.gz`). Royal Hogs: 25 of 25 policy wins
  over all models so far. All 50 games so far (Fable 10, Haiku 10, Sonnet 20, Opus 10): 37 W 9 D 4 L, 0.83.
  Remaining of the 80: Fable 11-20, Haiku 11-20, Opus 11-20.
- 2026-11-14 14:41: **Claude games, Fable 5.2 games 11-20 and Haiku 5.5 games 11-20 started, in parallel** (user,
  2026-11-14: "Run fable 11-20 / Also run haiku 11-20 IN PARALLEL"). One-call checks (login node, 14:39, the relay's
  flags, effort low, Sonnet game 10's turn 5): `claude-fable-5-2` (6.2 s, `PLAY 1 AT 13,19`) and `claude-haiku-5-5`
  (6.3 s, `PLAY 2 AT 11,21`) answer, modelUsage naming each model. Env files `claude-fable-5-2/job_games11-20.env`
  (= `job_games4-10.env` except the first comment line and `--max-games 10`) and `claude-haiku-5-5/job_games11-20.env`
  (= its `job.env` except comments); both resume from `llm/state.json` with games 1-10 done. Jobs 23146536
  (pr-cg-fable52, cs642) and 23146538 (pr-cg-haiku55, cs661), 6 h limit each, running at 14:40; both game 11s have
  Sonnet game 11's seed, decks and seat. Counted with the draw rule.
- 2026-11-14 15:38: **Claude games, Haiku 5.5, games 11-20 done** (job 23146538 COMPLETED 15:37 after 0:56:58, in
  parallel with Fable 11-20; 638 CLI calls, 0 failed, 0 retries; 0 parse errors; 0 illegal plays; mean 4.6 s per
  call; Haiku plays a card on 70% of its calls (448 of 638); list-price estimate $0.38). Policy's result per game
  (draw rule):

  | game | Claude deck vs policy deck | Claude team | Haiku: crowns Claude-policy, end | policy vs Haiku |
  |---|---|---|---|---|
  | 11 | xbow vs bait | 0 | 0-1, regulation | win |
  | 12 | royal_hogs vs bait | 0 | 0-1, regulation | win |
  | 13 | xbow vs xbow | 1 | 0-1, overtime crown at tick 4,597 | win |
  | 14 | royal_hogs vs royal_hogs | 1 | 0-1, regulation | win |
  | 15 | xbow vs golem | 1 | 1-3, King destroyed at tick 1,743 | win |
  | 16 | royal_hogs vs golem | 1 | 0-1, regulation | win |
  | 17 | xbow vs xbow | 0 | 0-0, tiebreak (6000) | draw |
  | 18 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win |
  | 19 | xbow vs lavaloon | 0 | 1-2, regulation | win |
  | 20 | royal_hogs vs lavaloon | 0 | 0-1, regulation | win |

  **Games 11-20: 9 W, 1 D, 0 L (0.95, Wilson about [0.66, 0.99]). Haiku games 1-20: 18 W, 2 D, 0 L, score 0.95
  ([0.76, 0.99])**; $0.75 in all. Royal Hogs: 30 of 30 policy wins over all models so far. Completed model-sets so far
  (Fable 1-10, Haiku 1-20, Sonnet 1-20, Opus 1-10; 60 games): 46 W 10 D 4 L, 0.85 ([0.74, 0.92]). Fable 11-20 still
  running (job 23146536): games 11-13 done at 15:37 (11: 0-0 tiebreak, draw; 12: 0-2, policy win; 13: 1-0 overtime,
  **Fable win**, xbow mirror). All 5 Claude wins so far are X-Bow mirrors.
- 2026-11-14 17:33: **Claude games, Fable 5.2, games 11-20 done** (job 23146536 COMPLETED 17:31 after 2:51:33, in
  parallel with Haiku 11-20 until 15:37; 1,448 CLI calls, 0 failed, 0 retries; 0 parse errors; 0 illegal plays; mean
  6.5 s per call, max 16.1 s; Fable plays a card on 35% of its calls (510 of 1,448); list-price estimate $51.52;
  Fable 1-20 $101.72). Policy's result per game (draw rule; crown ticks and tower HP from `frames.jsonl.gz`):

  | game | Claude deck vs policy deck | Claude team | Fable: crowns Claude-policy, end | policy vs Fable | engine HP tiebreak |
  |---|---|---|---|---|---|
  | 11 | xbow vs bait | 0 | 0-0, tiebreak (6000) | draw | policy |
  | 12 | royal_hogs vs bait | 0 | 0-2, regulation | win | - |
  | 13 | xbow vs xbow | 1 | 1-0, Fable crown in overtime at tick 4,423 | **loss** | - |
  | 14 | royal_hogs vs royal_hogs | 1 | 0-1, regulation | win | - |
  | 15 | xbow vs golem | 1 | 1-1 (crowns at ticks 1,533 and 1,558), tiebreak (6000) | draw | policy |
  | 16 | royal_hogs vs golem | 1 | 1-3, King destroyed at tick 1,007 | win | - |
  | 17 | xbow vs xbow | 0 | 0-0, tiebreak (6000) | draw | Fable |
  | 18 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win | - |
  | 19 | xbow vs lavaloon | 0 | 1-0, Fable crown in overtime at tick 5,779 | **loss** | - |
  | 20 | royal_hogs vs lavaloon | 0 | 1-0, Fable crown in overtime at tick 4,201 | **loss** | - |

  **Games 11-20: 4 W, 3 D, 3 L (0.55, Wilson about [0.27, 0.80]). Fable games 1-20: 10 W, 6 D, 4 L, score 0.65
  ([0.43, 0.82])**, the lowest of the four models so far. In games 13, 19 and 20 the tower Fable took in overtime was
  already low at the end of regulation (tick 3600: 1,731, 987 and 830 HP of 3,052). Game 20 is the first Claude win
  with Royal Hogs (Royal Hogs now 34 W 0 D 1 L for the policy) and games 19-20 are the first Claude wins outside an
  X-Bow mirror. All 7 Claude wins so far are 1-0. **70 of 80 games: 50 W 13 D 7 L, 0.81 ([0.70, 0.88])**: Fable 0.65
  (20), Sonnet 0.80 (20), Opus 0.85 (10), Haiku 0.95 (20). Remaining: Opus 11-20, only when the user says.
- 2026-11-14 18:21: **Claude games, Opus 5.5 games 11-20 started** (user, 2026-11-14: "Do the last opus games (10)").
  One-call check (login node, 18:19, the relay's flags, effort low, Sonnet game 10's turn 5): `claude-opus-5-5`
  answers in 7.7 s (`WAIT`), modelUsage naming the model. Env file `claude-opus-5-5/job_games11-20.env` (= its
  `job.env` except comment lines 1 and 3); resumes from `llm/state.json` with games 1-10 done. Job 23173061
  (pr-cg-opus55, cs607), 6 h limit, running at 18:20, alone (no other pr-cg job). Game 11 has Sonnet game 11's seed
  (424444787), decks (xbow vs bait) and seat (team 0); policy step 10,000,007,168. Counted with the draw rule. Last 10
  of the 80.
- 2026-11-14 20:00: **Claude games 21-25 for all four models started, in parallel** (user, 2026-11-14: "Play 5 more OF
  EACH - parallely !"). Games 21-25 continue `game_config(k)` and seed = crc32("S4_A_s1:k") unchanged: 21 xbow mirror
  (team 1), 22 royal_hogs mirror (team 1), 23 xbow vs miner_poison (team 1), 24 royal_hogs vs miner_poison (team 1),
  25 xbow mirror (team 0). `hpc/llm_game.py` schedules total // LLM_EVERY games, so the new env files
  (`job_games21-25.env`) set LLM_EVERY 400M (25 games; every target k * 400M <= 10B picks model_610352.pt) and
  `--max-games 5`; nothing else changed. Fable, Sonnet and Haiku play in their own folders (llm/DONE renamed
  llm/DONE_games1-20 so the job runs). Opus games 11-20 (job 23173061) were still using `claude-opus-5-5/`, and two
  jobs must not share a state.json, so Opus 21-25 play in a new folder `claude-opus-5-5_g21-25/` (config.json copy,
  model link sha256 46c62cddee6a, llm/state.json marking games 1-20 'skipped' = played in `claude-opus-5-5/`).
  One-call check of `claude-sonnet-5-5` at 19:58 (7.1 s, `WAIT`; Fable and Haiku checked 14:39, Opus 18:19). Jobs
  23183973 (pr-cg-fable52, cs612), 23183974 (pr-cg-sonnet55, cs648), 23183975 (pr-cg-haiku55, cs701), 23183979
  (pr-cg-opus55b, cs694), 6 h limit each, running at 19:59 alongside 23173061 (five Claude streams at once). All four
  game 21s: seed 845271216, xbow vs xbow, Claude team 1, policy step 10,000,007,168. Counted with the draw rule.
- 2026-11-14 20:07: **Claude games, Opus 5.5, games 11-20 done: all 80 planned games are played** (job 23173061
  COMPLETED 20:04 after 1:44:04; from 19:59 in parallel with the four games 21-25 jobs; 1,253 CLI calls, 0 failed, 0
  retries; 0 parse errors; 4 illegal plays; mean 4.4 s per call; Opus plays a card on 31% of its calls (394 of 1,253);
  list-price estimate $20.98; Opus 1-20 $40.81). Policy's result per game (draw rule; crowns from `frames.jsonl.gz`):

  | game | Claude deck vs policy deck | Claude team | Opus: crowns Claude-policy, end | policy vs Opus |
  |---|---|---|---|---|
  | 11 | xbow vs bait | 0 | 0-1, overtime crown at tick 4,311 | win |
  | 12 | royal_hogs vs bait | 0 | 1-2 (policy 709, Opus 2,616, policy 2,943), regulation | win |
  | 13 | xbow vs xbow | 1 | 1-0, Opus crown in overtime at tick 4,961 | **loss** |
  | 14 | royal_hogs vs royal_hogs | 1 | 0-1, regulation | win |
  | 15 | xbow vs golem | 1 | 0-1, regulation | win |
  | 16 | royal_hogs vs golem | 1 | 1-3, King destroyed at tick 883 | win |
  | 17 | xbow vs xbow | 0 | 1-0, Opus crown at tick 1,349, regulation | **loss** |
  | 18 | royal_hogs vs royal_hogs | 0 | 0-1, regulation | win |
  | 19 | xbow vs lavaloon | 0 | 0-0, tiebreak (6000; engine HP tiebreak: policy) | draw |
  | 20 | royal_hogs vs lavaloon | 0 | 0-2, regulation | win |

  **Games 11-20: 7 W, 1 D, 2 L (0.75, Wilson about [0.44, 0.92]). Opus games 1-20: 15 W, 2 D, 3 L, score 0.80
  ([0.58, 0.92]).** **All 80 games (20 per model): 57 W 14 D 9 L, 0.80 ([0.70, 0.87])**: Fable 0.65 (10-6-4), Opus
  0.80 (15-2-3), Sonnet 0.80 (14-4-2), Haiku 0.95 (18-2-0). Policy result by game, columns Fable / Opus / Sonnet / Haiku:

  | game | decks | F | O | S | H | | game | decks | F | O | S | H |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 1 | xbow mirror | D | L | L | W | | 11 | xbow vs bait | D | W | W | W |
  | 2 | royal_hogs mirror | W | W | W | W | | 12 | royal_hogs vs bait | W | W | W | W |
  | 3 | xbow vs hog26 | D | D | D | W | | 13 | xbow mirror | L | L | D | W |
  | 4 | royal_hogs vs hog26 | W | W | W | W | | 14 | royal_hogs mirror | W | W | W | W |
  | 5 | xbow mirror | L | W | L | W | | 15 | xbow vs golem | D | W | W | W |
  | 6 | royal_hogs mirror | W | W | W | W | | 16 | royal_hogs vs golem | W | W | W | W |
  | 7 | xbow vs giant | D | W | W | W | | 17 | xbow mirror | D | L | W | D |
  | 8 | royal_hogs vs giant | W | W | W | W | | 18 | royal_hogs mirror | W | W | W | W |
  | 9 | xbow mirror | W | W | D | D | | 19 | xbow vs lavaloon | L | D | D | W |
  | 10 | royal_hogs mirror | W | W | W | W | | 20 | royal_hogs vs lavaloon | L | W | W | W |

  Royal Hogs: 39 W 0 D 1 L for the policy (the loss: Fable game 20). X-Bow: 18 W 14 D 8 L. All 9 Claude wins are 1-0;
  7 of the 9 are X-Bow mirrors. Games 21-25 of all four models are running (entry 19:59).
- 2026-11-14 21:18: **Claude games 21-25 done for all four models: 100 games in all** (jobs 23183973 Fable COMPLETED
  1:09:41, 23183974 Sonnet 1:16:29, 23183975 Haiku 0:40:29, 23183979 Opus 1:04:41; last ended 21:15; five Claude
  streams at once until 20:04, four after). Calls: Fable 654, Sonnet 799, Haiku 413, Opus 723; **0 failed, 0 retries**
  in all four; 0 parse errors; illegal plays Fable 0, Opus 2, Sonnet 14, Haiku 0. List-price estimates: Fable $22.78,
  Opus $12.11, Sonnet $7.48, Haiku $0.25 ($42.62). Every game's meta.json: policy step 10,000,007,168, the same seed
  per game for all four models, decision interval 20 ticks, opponent interval 10. Every llm/DONE rewritten (all 25
  done; Opus folder: 1-20 skipped + 21-25 done). Policy's result (draw rule; Claude-policy crowns; F/O/S/H):

  | game | decks (Claude team) | Fable | Opus | Sonnet | Haiku |
  |---|---|---|---|---|---|
  | 21 | xbow mirror (1) | 1-0 at tick 3,398: **L** | 1-1, tiebreak: D | 0-0: D | 0-0: D |
  | 22 | royal_hogs mirror (1) | 0-1: W | 0-1: W | 0-2: W | 0-1: W |
  | 23 | xbow vs miner_poison (1) | 0-1 overtime (5,939): W | 1-0 at 2,909: **L** | 0-1: W | 0-0: D |
  | 24 | royal_hogs vs miner_poison (1) | 1-0 at 2,877: **L** | 1-0 at 2,041: **L** | 0-1: W | 0-1: W |
  | 25 | xbow mirror (0) | 1-0 at 3,280: **L** | 1-0 overtime (4,568): **L** | 0-0: D | 0-1: W |

  Games 21-25: Fable 2-0-3 (0.40), Opus 1-1-3 (0.30), Sonnet 3-2-0 (0.80), Haiku 3-2-0 (0.80); all 20: 9 W 5 D 6 L,
  **0.575** (Wilson [0.36, 0.76]), below games 1-20 (0.80). Part of the gap is the mix: 2 of these 5 games are X-Bow
  mirrors (5 of 20 before), the policy's weakest setup (games 1-20: 0.50; here 0.375); with the games 1-20 rate of each
  setup type the batch would expect about 0.74, inside the interval. Games 23-24 bring a deck the policy had not used
  here, miner_poison: 4-1-3 (0.56, n=8). Engine HP tiebreak differs from the draw rule in Opus 21 and Sonnet 21
  (engine: Claude) and Haiku 21, 23, Sonnet 25 (engine: policy).

  **All 100 games (25 per model): 66 W 19 D 15 L, 0.755 (Wilson [0.66, 0.83])**: Fable 12-6-7 **0.60** ([0.41, 0.77]),
  Opus 16-3-6 0.70 ([0.50, 0.84]), Sonnet 17-6-2 0.80 ([0.61, 0.91]), Haiku 21-4-0 0.92 ([0.75, 0.98]). By setup:
  Royal Hogs mirror 24-0-0 (1.00), Royal Hogs vs other 21-0-3 (0.875), X-Bow vs other 13-9-2 (0.73), X-Bow mirror
  8-10-10 (0.46). All 15 Claude wins are 1-0 (Fable 7, Opus 6, Sonnet 2, Haiku 0); 10 are X-Bow mirrors.
- 2026-11-14 22:34: **Claude-game results added to the README (release tree, not pushed yet) and to the play site**
  (user, 2026-11-14: "add these results and a plot for the same in the README ... and also add this to the website
  ... a separate section of games against LLMs and show the statistics").
  - **Figure:** `docs/figures/make_figures.py --only claude_games` (release tree; new function
    `fig_claude_games`) writes `docs/figures/claude_games.png` and `claude_games.json` (per-game table, no
    costs, no Claude text). It asserts that every model played the same 25 games with the same seed, decks and
    side, and one policy step (10,000,007,168); the doc check of the 100-game W-D-L and score: 4 numbers match.
    Claude's score: Fable 0.40 [0.23, 0.59], Opus 0.30 [0.16, 0.50], Sonnet 0.20 [0.09, 0.39], Haiku 0.08
    [0.02, 0.25], all 0.245 [0.17, 0.34]; by setup X-Bow mirror 0.54 (28), X-Bow vs other 0.27 (24), Royal Hogs
    mirror 0.00 (24), Royal Hogs vs other 0.125 (24). Colours: smallest CVD ΔE 21.9, normal 26.2.
  - **README:** results item 7 (table, figure, three findings), "In short" finding 6, a Limitations bullet;
    FIGURES.md entry; release `hpc/llm.sbatch` gets LLM_RUN_DIR (as used for these games); release
    STAGE4_SELECTION.md synced with this file.
  - **Play site:** new lobby panel "Games against language models" (`scripts/play_human.{py,html}`, same change in
    release `play/play_server.py`, `play/play_page.html`): `GET /api/llm_games`, `GET /api/llm_replay` (GIFs
    named in the JSON and inside `--llm-replays`, default `<data-dir>/../claude_games/S4_A_s1_10B`). Checked on a
    private localhost copy (bad model / game / traversal / no token refused; headless Chromium at 1100 px and
    390 px, no console errors). Tests: live `tests/builder/test_play_human.py` 28 passed; release
    `test_play_site.py` + `test_play_semantics.py` 39 passed. Live site (job 22719628) had 0 players, restarted
    with `hpc/play_restart.sh` at about 22:30 (same link): 100 games, 100 replays, a 6.0 MB replay served through
    the tunnel. The queued successor 23027985 starts from the same checkout. Backup of the old page and server:
    `/scratch/$USER/pufferroyale/tmp_site_backup_20261114_222645/`.
