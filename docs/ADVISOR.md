# Advisor: the trained policy as a live coach for real Clash Royale

Started 2026-11-03 on the user's request ("We'll do the advisor thing - go build the entire thing - BE SUPER
METICULOUS"), parallel to the Stage-4 experiments (docs/STAGE4_SELECTION.md on `hpc-training`).
Built on branch `advisor` (from `hpc-training` 1214564); part of the public release since 2026-11-09. Engine and
game code are not changed; everything lives in `advisor/` and this file.

> **Terms of Service warning.** Read §1 "Rules" before using the live part. Using a screen-reading advisor in the
> real game can get an account banned; never use it in ranked play or against people who have not agreed.

## 1. What it is and is not

- **Is:** a program on the player's own computer that reads the game screen (a phone mirrored with scrcpy,
  or the PC client), rebuilds the policy's observation, runs the policy, and draws its suggestions
  (card, tile, probability) in a separate window. The human decides and taps.
- **Is not:** a bot. It sends no input to the game (no taps, no ADB input), by design and in code.
- **Rules:** Supercell's Terms of Service (effective 2024-11-06) prohibit "automation software, emulators,
  bots, hacks, mods or any unauthorized third-party software designed to modify or interfere with the
  Service" and getting information "using any method not expressly permitted". A screen-reading advisor is
  still third-party software reading the game; the user chose this mode knowing that (2026-11-03). Use it
  outside ranked play (friendly battles with a consenting friend, Training Camp) and on an account you
  accept the risk for.

## 2. Pipeline

```
frame (pixels) --perception--> FrameReading --tracker--> BoardState --encoder--> obs[17715] --policy--> suggestions --overlay
                (units, hand,                (tracks, card       (exact SPEC §9   (S4 checkpoint,
                 elixir, towers)              attribution, clock,  vector, engine   card-first
                                              opponent history)    legality)        probabilities)
```

| Layer | File | Verified how |
|---|---|---|
| engine facts | `advisor/engine_db.py` | unit table parsed from `pr_card_db.h`; every card's unit equals `card_info` |
| board state | `advisor/board.py` | `from_game` = perfect perception of a simulated match |
| encoder | `advisor/encoder.py` | bit-identical to `Game.obs` on simulated matches (both seats) |
| policy | `advisor/policy.py` | its greedy pick = `league.greedy_actions` on the engine's observation |
| tracker | `advisor/tracker.py` | on simulated perfect detections: opponent history and card attribution vs engine truth |
| perception | `advisor/perception/` | on KataCR's hand-labelled frames (accuracy), on recorded matches |
| advisor-in-the-loop | `advisor/eval/` | the full pipeline playing in the simulator against the clean policy |

## 3. Checks written before their results

1. **Encoder exactness.** On 600 simulated bot matches (random decks, random tower troops, heuristic and
   random bots; seeds 101/202/303) every decision state, for both seats, encodes to exactly `Game.obs`.
   Pass = 0 mismatching states.
2. **Mutation check.** Each of 10 deliberate BoardState errors (unit moved one tile, hp halved, unit
   dropped, card changed, elixir, tick, opponent spent, opponent last-4, tower hp, unknown hand slot)
   changes the encoded vector. Pass = all 10 detected.
3. **Policy wiring.** For simulated states, the advisor's greedy action equals `league.greedy_actions` on
   `Game.obs` of the same state.
4. **Tracker** (on simulated perfect detections, without spells, which the detector cannot see): report the
   share of opponent troop/building plays recovered and the card-attribution accuracy of units. No pass
   threshold; these numbers set expectations.
5. **Advisor-in-the-loop** (simulator): the policy fed by the full pipeline (simulated detections ->
   tracker -> encoder) plays the same policy fed by the engine's observation. Score 0.5 = no loss from the
   pipeline. Reported per degradation (no unit HP, no spells, detection misses, position noise, latency).
6. **Perception accuracy** on KataCR's labelled frames: unit detection recall/precision per class and side
   accuracy, for the classes our 64 cards use.

## 4. Results

| # | Check | Result |
|---|---|---|
| 1 | encoder exactness | **pass**: job 21751882, 600 matches, 409,566 states (both seats), 0 mismatching states; up to 41 units on the board, 43,444 states with live spells, 237,754 with a destroyed tower |
| 2 | mutation check | **pass**: 10 of 10 deliberate errors change the vector (pytest, 2026-11-03) |
| 3 | policy wiring | **pass**: S4_A_s1 snap_476523, 22 states (3 matches x 4 moments x 2 seats): advisor greedy = `league.greedy_actions(Game.obs)`; probabilities sum to 1 |
| 4 | tracker | 4 matches (perfect detections, 4 readings/s): see 4a below |
| 5 | advisor-in-the-loop | arrays 21753108 (9 variants) + 21753509 (`realistic`), 400 matches each: see 5a |
| 6 | perception accuracy | job 21753708, 2,610 KataCR frames: see 6a |

**4b. Tracker against the real detector's error rates** (`python -m advisor.tests.test_tracker --matches 6
--seed 1 [--real]`; `--real` = 25% units missed per frame, 0.1-tile position noise, 5% wrong type, 20% wrong
side, a phantom on every standing crown tower, 0.5 other phantoms per frame -- the first four measured in 6a,
the phantom rate assumed). The first tracker collapsed on these errors, exactly as it did on the real KataCR
episode (21 "cards seen", 381 elixir "spent" in 2 minutes). Tracker v2 adds, one at a time (each measured):

| Tracker | Perfect input: precision / spent err | Real errors: recall / precision / card acc / seen Jaccard / spent err |
|---|---|---|
| v1 | 1.00 / 0.0 | 0.79 / 0.04 / 0.66 / 0.12 / 1302.7 |
| + tower filter, position matching with type/side penalties, 3-hit confirmation, majority vote, 8-card cap, deploy-zone rule | 0.91 / 7.2 | 0.78 / 0.44 / 0.77 / 0.69 / 49.3 |
| + re-identification of lost tracks (3 s) | 0.97 / 2.2 | 0.78 / 0.54 / 0.80 / 0.67 / 23.0 |
| + late joiners (units of a recent play first seen late) | 0.97 / 2.2 | 0.78 / 0.59 / 0.80 / 0.67 / 17.5 |
| + side override (spawner owner; own hand change), deploy-anywhere penalty, re-identification 5 s | 0.95 / 2.5 | 0.85 / 0.72 / 0.84 / 0.72 / 11.2 |

(spent err = mean absolute error of the opponent's elixir spent per match, in elixir, unit-leaving plays only.)
Each error alone, measured with the tracker of the second row: missed detections cost the most (precision
0.51 alone), then wrong sides (0.88), phantoms (0.91 for either kind) and wrong types (0.94); position noise
0.92 -- which is why re-identification and late joiners came next. On perfect input the last row loses a little (0.97 -> 0.95): the
simulator's random bot throws Goblin Barrels into its own half, which the deploy-anywhere penalty reads as
Goblins; real players throw them at towers.

**5a. Advisor in the loop** (S4_A_s1 vs itself; A sees the game through the variant, B through `Game.obs`;
400 matches per variant, the same 400 deck / seat draws for every variant, seed 1; ± = 95% interval).
Code: snapshot `runs/advisor/code_21753108.tar.gz` (the `realistic` variant ran the same code plus the
`read_king` switch).

| Variant | What A misses | Score of A | ± |
|---|---|---|---|
| oracle | nothing (control) | 0.4525 | 0.049 |
| no_spells | spell areas, spell plays in the opponent history | 0.4575 | 0.049 |
| no_hp | unit HP, stun, slow | 0.385 | 0.048 |
| blind | no_spells + no_hp + underground units | 0.415 | 0.048 |
| tracker | blind + tracker (cards, clock, elixir between bar steps, deploy flags) | 0.390 | 0.048 |
| tracker_noisy | tracker + 10% missed units per frame + 0.3-tile position noise | 0.320 | 0.046 |
| delay1 | tracker + its play applied 0.5 s late | 0.260 | 0.043 |
| delay2 | tracker + 1.0 s late | 0.240 | 0.042 |
| noisy_delay2 | tracker_noisy + 1.0 s late | 0.2025 | 0.039 |
| realistic | noisy_delay2 + King HP not read | 0.1675 | 0.037 |

No draws in any variant. The control (oracle) is 0.4525 ± 0.049, i.e. 0.5 is just inside its interval;
the comparisons below are read against the oracle row, not against 0.5.
- Spells cost nothing measurable (0.4575 vs 0.4525).
- Not reading unit HP / stun / slow costs about 0.07; the tracker adds little on top of what it cannot see
  (blind 0.415 vs tracker 0.390).
- Detection errors at the sizes of 6a cost about 0.07 (0.390 -> 0.320).
- **Latency is the largest cost**: half a second between the frame and the play drops 0.390 to 0.260;
  one second to 0.240. A human following a suggestion adds reaction time on top of the reader's.

**5b. Round 2: tracker v2 and the real detector's error rates** (array 21756300 + job 21756605, code
snapshot `runs/advisor/code_in_loop_v2.tar.gz`; same 400 deck / seat draws as 5a, seed 1; the control with
1,000 new matches, seed 2):

| Variant | What A misses | Score of A | ± |
|---|---|---|---|
| oracle (1,000 matches) | nothing (control) | 0.501 | 0.031 |
| tracker (v2) | perfect detections, tracker v2 | 0.3275 | 0.046 |
| real | 25% missed, 0.1-tile noise, 5% wrong type, 20% wrong side, tower + 0.5 other phantoms per frame, King HP not read | 0.250 | 0.043 |
| real_towers_standing | real, tower HP read only as standing / destroyed | 0.2325 | 0.041 |
| real_delay1 | real + play 0.5 s late | 0.190 | 0.039 |
| real_delay2 | real + play 1.0 s late | 0.1875 | 0.038 |

- The control is 0.501 ± 0.031 over 1,000 matches (501 wins, 499 losses): the harness itself has no bias,
  so 5a's 0.4525 ± 0.049 was sampling noise, and every row reads directly against 0.5.
- Tracker v2 costs about 0.06 on perfect input (0.390 in 5a -> 0.3275): a unit enters the observation only
  after 3 detections (0.5 s at 4 readings per second) and deploy-anywhere / zone rules occasionally misfile a
  play. It is what makes the real errors survivable (the v1 tracker's play history was unusable under them, 4b).
- Tower HP beyond standing / destroyed is worth about 0.02 (within noise): reading the digits is not a priority.
- With the real errors, half a second of delay costs another 0.06; a second costs about the same.

**6a. Detector on KataCR's labelled frames** (11 episodes, 2,610 frames, 11,178 labelled units of which 6,988
are units of our 64 cards; arena mapping fitted per episode from the crown-tower boxes, rms 0.95-6.6 px):

| Measure | Value |
|---|---|
| labelled units of our cards found (IoU >= 0.3) | 74.7% |
| right unit type among found | 95.1% |
| right side (side classifier alone, no deck knowledge) | 78.6% |
| ground-point error (tiles) | median 0.086, 90th percentile 0.275 |
| detections with no labelled unit | 50.3% (8,518 of 16,937) |
| ms per frame (8 CPU threads) | 97.5 |

Weak classes (labelled count, found): Skeleton (1,361, 50%), Royal Hog (332, 45%), Ice Spirit (202, 36%),
Bat (117, 51%), Barbarian (84, 46%), Lava Pup (61, 15%). Strong: Musketeer 89%, Cannon 90%, Giant 94%,
Archer 100%, Balloon 100%, Prince 99%. Looking at frames: most detections with no label sit on the crown
towers themselves -- the princess-tower archer read as "bandit" / "princess", the King tower as
"hungry_dragon" (Baby Dragon) and "spear_goblin". Unfiltered, they would put phantom enemy units on the
towers from the first second (and phantom plays into the opponent history).

**6b. Tower filter, HP bars** (job 21756332 for the filter; the bar readings on frames of WTY_20240305 /
WTY_20240307 with the arena fitted at rms 0.96 px):
- The tracker's tower filter (a detection whose ground point lies inside a standing crown tower's footprint
  shrunk by 0.3 tiles is dropped; Balloons kept) removes 5,134 of the 8,518 unmatched detections and 61 of
  the 8,419 matched ones (0.7%). Left: 3,384 unmatched detections, 1.30 per frame (28.8% of the detections
  kept) -- some may be units KataCR did not label. The tracker check at 1.3 random phantoms per frame
  (instead of 0.5) gives the same quality (precision 0.74, card accuracy 0.89, spent error 12.3).
- Princess-tower HP-bar boxes placed from Build-A-Bot's layout through the arena mapping sit on the bars
  (check image `runs/advisor/inspect/calib_test2_check.png`); KataCR's `tower-bar` labels agree (enemy
  left label box tiles (1.99, 2.90)-(4.98, 4.04) vs Build-A-Bot's bar box (2.73, 3.11)-(5.03, 3.82)).
- In the current interface the HP number is drawn ON the bar. Build-A-Bot's bar reader reads 0.0 ("no bar")
  for every standing tower. `perception.hud.read_tower_bar` reads the bar body rows instead, and the tracker
  scales each bar by its reading in the first 10 s (all towers full): full towers read 1.0, damaged ones less.
  **Limit:** on the viewer's own (blue) bars the digits have the fill colour and cover the left ~60% of the bar,
  so below about half HP the reading stops at about 0.5 (a tower showing "976" of 3052 = 0.32 read 0.49).
  Enemy (red) bars show the bar under the digits and are not affected. Reading the digits (OCR) would fix it;
  check 5's `real_towers_standing` variant measures what tower HP is worth to the policy first.

**6c. The whole app on a real recorded match** (job 21756856: KataCR episode WTY_20240307/1, 342 frames at
2 per second = 171 s, arena-only crops so no hand / elixir; the viewer's deck given as hog26, which this
player uses). Enemy units KataCR labelled in that episode, mapped to our 64 cards: Prince, Tesla, Rascals,
Skeleton Dragons; outside our 64: Sparky, Skeleton King, Skeleton Barrel, Phoenix (and an Evolved Tesla).
- Tracker v2 ends with 8 cards seen and 61 elixir spent in 171 s. All 4 real cards of our 64 are among the
  8; the other 4 are wrong: Hog Rider (most likely the viewer's own Hog Rider read as an enemy) and Baby
  Dragon, Bats, Fire Spirit (most likely Phoenix / Sparky / Skeleton-King units read as units of ours).
- The first tracker on episode WTY_20240301 (job 21755597): 21 cards seen, 381 elixir spent in 2 minutes.
- Opponents with cards outside our 64 will always add wrong cards: their units are detected as the nearest
  class we know, or as an unknown class (shown as "not in our 64" in the window, never encoded).
- **The suggestion step was not exercised in this run** (found 2026-11-03 re-reading its log
  `runs/advisor/plumbing/WTY_20240307_1.jsonl`): the job used layout `inspect/calib_test2.json`, whose hand /
  elixir boxes sit where the HUD is on a full phone screen; on these arena-only crops they cover arena pixels.
  The "hand" read there is noise (e.g. cards 9, 2, 4 in 261 of 342 decisions), and the elixir estimate stays
  at 2,799 units (just under 1 elixir) in 336 of 342. The policy chose "wait" with probability 1.0 at every
  decision, and no card was ever suggested. Suggestions have been checked only in the simulator (checks 3 and 5);
  a real full-screen recording is needed.
- **Time per decision** (same log, 4 CPU threads): median 395 ms, 90th percentile 559 ms; 87 of 342 decisions
  over the 500 ms budget. Reading the frame took 174 ms median; the policy took 214 ms median here vs 4 ms in
  check 3 (2 torch threads). The app does not set torch's thread count; whether that explains the gap is
  not yet checked. Related measurement (2026-11-06, human-play work, S4_A_s1 on a 32-core login node): torch's
  default of 28 threads took 3,030 ms per decision vs 6.0 ms with `torch.set_num_threads(2)` and 6.6 ms with 1, so
  an unset thread count can cost far more than this gap on a busy machine; the app should pin it (not yet done).

**4a. Tracker on perfect detections** (4 matches, seed 1; 110 opponent plays: 88 that leave units, 22
spells): recall of unit-leaving plays 1.0, precision 1.0, spell plays found 0 (by construction); unit card
attribution 0.9948 over 16,794 matched unit-readings; opponent seen-set exact in 4/4 matches and elixir spent
error 0.0 (both restricted to unit-leaving plays). 781 of 17,575 tracked unit-readings (4.4%) had no live engine
unit nearby: tracks are kept 0.6 s after their last detection so a missed frame does not end them, and a
unit that really died stays that long.

Timing (cluster CPU, 2 torch threads, S4_A_s1): encode 0.47 ms median, policy 4.0 ms median after a
one-time warm-up of about 0.3-5 s. The decision period is 500 ms (frame_skip 10 x 50 ms), so perception
has about 490 ms.

## 5. Known gaps (each one is measured in check 5 before anything is built for it)

- **Cards outside our 64** (e.g. Mega Knight, Evolutions, Champions): the detector sees them but the policy
  has no id for them. They are shown as "unknown" and left out of the observation.
- **Spells are invisible** to the unit detector (Fireball, Zap, Log, Poison, ...): the spell-area planes
  stay empty and the opponent's spell plays are missing from "cards seen / last 4 / elixir spent".
- **Unit HP, stun and slow** are not read (Build-A-Bot reads none of them): units are encoded at full HP.
- **King-tower HP and the match clock** are not read by Build-A-Bot; the clock comes from the match start
  time, King HP from its bar once it is visible.
- **Tower troops:** S4_A_s1 trained with Princess towers on both sides only (config.json); an opponent
  with Cannoneer / Dagger Duchess / Royal Chef is outside its training.
- **Card levels:** the simulator uses level 11 for every card; real matches have other levels.
- **Arena geometry and starting elixir** follow the simulator's ledger (2018 tilemap, 6 starting elixir).
- **Detector licence:** Build-A-Bot's repository is MIT, but its unit model's metadata says it was trained
  with Ultralytics YOLOv10 and carries "AGPL-3.0". Fine for private research use; check before sharing
  a build that includes it.

## 6. Third-party material (downloaded as data; their code is not run)

- `research/ext/buildabot`: Pbatch/ClashRoyaleBuildABot @ main (MIT): `units_M_480x352.onnx` (YOLOv10m, 97
  classes, input 1x3x480x352 float16, output 1x300x6 = x1, y1, x2, y2, conf, class), `side.onnx`
  (ally / enemy from a 16x16 crop), card images, screen layout constants for a 720x1280 display.
- `research/ext/katacr`: wty-yy/KataCR @ master (MIT): code only, no weights. Its labelled dataset
  (wty-yy/Clash-Royale-Detection-Dataset, MIT, about 600 MB) is for check 6.
- Python packages for the advisor live in `$PR_ROOT/envs/advisor_py` (onnxruntime 1.20.1,
  opencv-python-headless 4.10.0.84, onnx 1.17.0), installed with `--no-deps` so the training container's
  numpy 1.26.4 / torch 2.6.0 are untouched.

## 7. Running it on your own computer

The live part needs a screen, so it runs on your machine, not on the cluster. Nothing in it sends input
to the game.

1. **Code.** This repository, with the engine built (`make build`; on macOS this needs the
   Xcode command-line tools, on Windows MSVC) and its Python requirements (torch, pufferlib 3.0, numpy, scipy).
2. **Advisor packages.** `pip install onnxruntime opencv-python pillow mss` (opencv-python, not the headless
   build: the window needs its GUI).
3. **Detector assets.** Copy `clashroyalebuildabot/models/` and `clashroyalebuildabot/images/` from
   https://github.com/Pbatch/ClashRoyaleBuildABot (MIT; the unit model's metadata says AGPL-3.0, §5) into one
   folder and pass it as `--assets FOLDER` (or set `ADVISOR_ASSETS`).
4. **Checkpoint.** The repository's `models/S4_A_s1_10B.pt` works as is (its `config.json` is in
   `models/S4_A_s1_10B/`). The checks above used the same run's 7.81B snapshot `snap_476523.pt`; any checkpoint
   with its `config.json` next to it (or in `<file stem>/`) works.
5. **Game on screen.** Phone: mirror it with scrcpy (`scrcpy --max-size 1280`) and keep the window fixed.
   PC: the Google Play Games client (Windows), window fixed. Note the game area's desktop rectangle
   LEFT,TOP,WIDTH,HEIGHT.
6. **Layout.** If the game area is 9:16 (e.g. 720x1280 or any scaled copy), the built-in
   `bluestacks_720x1280` layout applies. Otherwise take one screenshot during a battle with all four
   princess towers standing and run `python -m advisor.calibrate --image shot.png --out mydevice.json`
   (click the 4 princess-tower centres; add `--drag-hud` to mark the hand, next card and elixir bar
   yourself), then open `mydevice_check.png` and check that the grid, hand boxes and elixir bar sit right.
7. **Run.** Your deck must be 8 of the simulator's 64 cards, with no Evolutions, and your tower troop the
   Princess (what S4_A_s1 trained with):

       python -m advisor.app --ckpt models/S4_A_s1_10B.pt --assets FOLDER --source screen --region L,T,W,H \
           --deck "Hog Rider,Musketeer,Cannon,Ice Golem,Ice Spirit,Skeletons,Fireball,The Log" \
           [--layout-file mydevice.json] [--opp-tower-troop princess]

   Keys: `s` = the battle starts now (re-sync the clock at 3:00), `r` = new match, `q` = quit. Use it in
   friendly battles or Training Camp (§1).
8. **Offline first (recommended).** Screen-record one of your matches and run
   `python -m advisor.app ... --source video --video match.mp4 --out annotated.mp4 --log match.jsonl`;
   watch the annotated video before trusting the live window.

## 8. Log

- 2026-11-03: worktree + branch `advisor`; extension built in the worktree (data / arena sha256 and the
  tracked `pufferroyale/` files identical to the pinned training checkout s4x_3cd765c). engine_db, board,
  encoder, policy written; checks 1-3 passed; tracker v1 + simulated reader, check 4a; check 5 round 1
  (5a). Perception (Build-A-Bot detector / HUD, layout, calibration), app, overlay, capture written; check 6
  (6a) on 2,610 KataCR frames; the app ran end to end on a recorded match (job 21755597) and exposed the
  phantom-play problem; tracker v2 (4b), tower filter and HP-bar reader for the current interface (6b);
  the app with tracker v2 on a second recorded match (6c); check 5 round 2 (5b).
  Tests (advisor/tests, 5): pass (jobs 21756344, 21756779).
