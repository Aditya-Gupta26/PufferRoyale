# PufferRoyale: master plan

> **Status update (2026-09-27):** W0–W4-level engine work, the Puffer env, league/MMD/exploitability/meta-game tooling (Phase D) and a 64-card batch (Phase E) plus an LLM play interface were built and verified locally during the handoff session — see `docs/STATUS.md`. Decisions made autonomously are in `docs/DECISIONS.md` for review.


_Planning doc, 2026-09-25. No code gets written until the decisions it depends on (§6) are resolved; we'll resolve them one at a time._

---

## 0. Context

**Goal:** the strongest Clash Royale 1v1 bot we can build, in three steps:
1. **PufferRoyale**: a high-fidelity battle simulator in C that runs as a PufferLib environment.
2. Train a policy with self-play RL plus game-theoretic machinery: a league, regularization, and exploitability probes.
3. Later, connect to the real client.

**Constraints (confirmed by you):**
- Team: solo, with Claude writing most of the code. Your C skill: moderate.
- Deadline: the full project is due **early Dec 2026, about 10 weeks** from 2026-09-25.
- Compute: NYU HPC (Torch cluster, H200 and L40S GPUs), account active. Dev machine is an Apple Silicon Mac with no CUDA.
- Fidelity: staged up to the full game.
- Deployment: sim first, then the real client.

**Honest framing:**
- **The hard part is not RL throughput.** It's (a) simulator fidelity and (b) training pathologies that the engine, the reward, or the action space quietly create. Every prior project that got far enough says this.
- **No open project has shown a sim-trained Clash Royale bot beating a competent human.** "Best in the world" means being first to do it.
- **What the course grades is the game-theory analysis.** So the RL and evaluation pipeline has to run early; it can't wait behind the engine.

---

## 1. What's already settled

1. **The game-theoretic object.** A two-player, zero-sum, partially observable stochastic game (POSG): simultaneous moves, a finite horizon, and a regime that changes with the clock. The target is Nash, with **exploitability** as the metric. Against humans, we exploit safely on top of equilibrium play.
2. **Modeling level 2.**
   - Known: own deck, hand, and exact elixir, plus the whole board.
   - Hidden: the opponent's hand, elixir, unrevealed deck, and evo counter.
   - The opponent's deck is revealed as they play. We train across a deck pool.
   - Memory is required, via a recurrent core and/or derived belief features.
3. **Learning happens in the simulator, never on live servers.**
4. **Fixed decision cadence** (about 0.5 s) with an explicit **wait** action. Prior art measured agents waiting on 80–96% of decisions.
5. **Training direction.** PPO self-play → a PFSP league (frozen checkpoints plus scripted bots) → last-iterate regularization (MMD). We estimate exploitability with best-response probes.
6. **Fidelity is staged.** Architecture that can hold every feature, then a verified thin slice, then cards added in gated batches.
7. **Deployment is staged.** Sim first; the real client comes late (its scope is decided in D11).

**Game-theory insight to use.** The opponent's cycle is FIFO. Once all 8 of their cards are revealed and they've made at least 4 plays, their hand and next card are *deducible*. Their hand is the deck minus the last four cards played, and the next card is the oldest of those four. Mirror is an edge case. The evo counter is also derivable, once we know which card is their Evo. From then on, **the only material hidden variable is the opponent's elixir**, which is bounded by spend accounting and uncertain only through leakage at the cap. So the information asymmetry collapses over the course of a match. It's a report finding, and it justifies giving the policy derived belief features (ClashRoyaleEnv's LSTM never learned card counting on its own).

---

## 2. The game we simulate (Sept 2026 ruleset)

**Match flow**
- **Clock:** 3:00 of regulation, then 2:00 of overtime with sudden death (the first tower destroyed wins).
- **Tiebreaker:** troops vanish, towers drain, the lowest tower loses; exact equal HP is a draw.
- **Elixir windows** (best consensus): 1× from 0:00, 2× from 2:00, 3× from 4:00 to 5:00. Rates are 1 per 2.8 s / 1.4 s / 0.933 s, capped at 10.
- **Instant win:** killing the King (3 crowns). Otherwise a crown lead at 3:00 wins, or the first crown in overtime.
- **King tower** wakes when hit or when one of its own Princess towers falls (RoyaleSim measured a delay of about 3.55 s).
- **Deck and cycle:** 8 cards, 4 in hand plus a FIFO queue. Mirror (+1 elixir) is a special case.
- **Placement:** own half, minus tower footprints. A pocket on the enemy side opens after that lane's Princess tower falls. Spells go anywhere; Miner and Drill almost anywhere. Buildings snap to tiles, and illegal building taps get relocated.
- **Special slots (since March 2026):** 1 Evolution slot, 1 Hero slot (shared with Champions), 1 Wild slot.
  - Heroes (Dec 2025+, 13+) and Champions (8) have ability buttons. **Since Aug 26 2026 each ability is single-use per deployment** (Boss Bandit excepted).
  - Evolutions (41+) deploy evolved every N cycles.
- **Tower troops:** Tower Princess (the default), Cannoneer, Dagger Duchess, Royal Chef.
- **Levels:** tournament standard is **level 11** for all cards. Stats scale as `floor(base×table[L]/100)` with the table `[100,110,121,…,409]`; towers use their own ladder.
- **Spells vs. towers:** spells deal reduced crown-tower damage (Fireball does about 1/4). ClashRoyaleEnv omits this, and that made spells 4× too strong against towers.
- **Arena:** 18×32 tiles, a two-row river, 2 bridges, Princess towers 3×3, King 4×4. Units are **millitiles** (1000 per tile) and **milliseconds**, the client's own format.

**Mechanics families** (everything the engine eventually needs)
- **Core stats:** HP and shield, damage, hit speed and first hit, range (edge to edge), sight range, speed tiers, targets (ground/air/buildings), flying, mass, collision radius, deploy time, lifetime/decay.
- **Movement:** A* pathing, bridges, river hop, charge, dash/jump, burrow, hook/pull.
- **Combat:** projectiles (homing, splash, pierce, boomerang, chain, rolling), ramp damage, line/cone splash, spawn and death damage, death spawns, spawners.
- **Status effects:** stun/reset, freeze, slow, rage, heal, knockback, tornado pull, invisibility, clone, mirror. Elixir effects: Collector, and Elixir Golem feeding the enemy.
- **Special systems:** Evo counters, Champion/Hero abilities, tower-troop behaviors, King activation.
- **Collision/body-blocking:** the single most policy-relevant fidelity item.

**Observation contract.** Visible to the agent:
- the whole board, all tower HPs, the clock and elixir phase
- every opponent play, with timestamps
- the opponent's tower troop
- evolved units as they appear

Hidden: the opponent's elixir, hand, next card, unrevealed deck, and evo counter (the last three become deducible, per §1).

**Facts where the references disagree** (settled in D3; RS = RoyaleSim, CRE = ClashRoyaleEnv, HC = Hasty-CR)

| Fact | RS (measured on the 2026 client) | CRE | HC / others | Working answer |
|---|---|---|---|---|
| Engine tick | 50 ms (20 Hz) | 100 ms (10 Hz, derived from its own cooldown table) | HC 50 ms; ClashAI: tick 4800 = 240 s at 20 Hz | **20 Hz** |
| Speed units | millitiles per 50 ms tick (Medium = 1.2 tiles/s) | about 1.33 tiles/s (calibrated) | HC/crforge read tiles/min (1.0); HC's video analysis found Hog **27% too slow** | Prior = RS; confirm on video |
| Starting elixir | **6** (measured) | 5 | HC 5, crforge 5 | Verify on a recording |
| Triple elixir | missing | from 3:00 | HC from 240 s; web and ClashAI agree | from 4:00 |
| Tiebreak | weakest tower, absolute HP | weakest tower, absolute HP | HC: princess HP fraction; crforge: total HP | Verify |
| Tower HP at level 11 | 3052 / 4824 | 2534 / 4008 | HC 3584 / 6144 | Decode from the pinned client |
| Bridge width | 2018 tilemap | 2 tiles (`WWBBWW…`) | HC 3 tiles | Verify on video |
| Deploy lockout | first 4.5 s | none | – | Verify |
| Spell tower damage | client columns | missing | HC per-spell % | Client columns |
| River hop | modeled | modeled | HC: not modeled | Model it |

---

## 3. What already exists, and how we use it

| Asset | Summary | License | Use |
|---|---|---|---|
| **RoyaleGym/RoyaleSim** (Rust, 4 days old, changes daily) | 22k lines. Integer subtiles, no floats, 20 Hz, a **measured tick-phase order**, SoA entities, A* (743 of 744 real routes reproduced), a measured contact law. 2026 client card table: 144 cards; 101 load, 18 behavior-checked. No Evos, Champions, Heroes, or tower troops. | MIT (Supercell data excluded) | **The spec source:** `data/calibration.json` (173 constants with evidence and status), the phase order, the contact law. The committed `cards-15.535.json` covers Tier A data. **Our W1–W4 differential oracle, at a pinned commit.** Also the decode tooling. |
| **itzik123/ClashRoyaleAi** ("CRE", C++17 header-only) | 132 cards, 41 Evos, 8 Champions, Heroes, 4 tower troops. 715 tests, about 5 µs/tick, `snapshot()`, PFSP league, perception bridge. A 190 KB report plus a 580 KB decision log. Floats, 10 Hz, data hardcoded in C++, spells hit towers at 100%, no levels, some stats unsourced, Windows-first build. | MIT | **Backend 0** (D1): the RL pipeline gets debugged on it. The reference for mechanic breadth (Champions, Heroes, Evos, tower troops). The lessons bible (§8). |
| **hastylmao/Hasty-CR** (Python 3.12) | Integer millitiles, 50 ms. About 119 cards, about 20 ability types. Data is decoded from your own APK and not committed. **RL:** 8×32×18 planes + 47 scalars, 2,321 masked actions, BC then PPO: 18% → 93% against its rule bot, 87% against meta decks, 6.1M steps, 2.6 only. **Bugs:** its opponent acts twice on even steps; the robustness probe is a no-op; pathing breaks at the river edge; charging units deadlock. | MIT | Its RL recipe and measured traps (§8). A later oracle, once our APK decode exists (W5+). Its numbers are weaker evidence than advertised. |
| **crforge** (Java) | 113 cards, level-scaling formulas, docs on reverse-engineering stats from video, "secret stats". | Apache-2.0 | Cross-check. |
| **PufferLib** | 5.0 (default): pure CUDA, built-in per-head masks, 8-slot self-play pool, but custom nets need a hand-written CUDA backward pass and there's no resume. **3.0 (PyPI): PyTorch, flexible policies, no masking or opponent pool.** | MIT | The trainer (D6). |
| **PufferDrive** | Built on Puffer 3.0. **Documents an NYU HPC Singularity setup.** | MIT | HPC recipe. |
| **KataCR** | YOLOv8 with 150 classes; open detection and 105-episode expert-replay datasets. | open | Perception (late). |
| **ClashAI** | Its RL on a hand-written sim failed (26% outcome match). Measured a 1.2–1.35 s tap-to-land delay. | – | Negative evidence, sim-to-real numbers. |
| **Card data** | cr-api-data is stale (2023). Current data only via APK decode (`csv_logic` + TOML overlays; `*_evo.toml` and `actions.toml` have no public extractor). | Supercell's | D2. |
| **Meta decks** | HuggingFace: 1.45M Path of Legends battles (MIT); RoyaleAPI popular decks. | MIT | Deck pool (D9). |

---

## 4. Proposed architecture (subject to D1 and D6)

```
[pinned client tables] → data pipeline (Python) → cards.json (+provenance) → codegen card_db.h
                                                                                │
              pr_env.h  — ONE backend-neutral C interface —────────────────────┤
              reset(seed,decks) · step(actions[2]) · state view (entity table) ·
              legal-mask · snapshot/restore · state_hash · event log
                 │                                   │
   Backend 0: CRE C++ behind extern "C" shim   Backend 1: PufferRoyale engine (C99)
   (pipeline bring-up only; no reward tuning,  20 Hz · int32 millitiles/ms · SoA fixed pools ·
    no claims)                                 no malloc after init · PCG32 · RS phase order ·
                                               A* half-tile grid · snapshot = memcpy (zeroed padding)
                 └──────────────┬──────────────────┘
      our obs encoder + masks + reward (identical for both backends) · decision cadence · scripted C bots
                                │
      Puffer 3.0 adapter → PuffeRL (PyTorch) + league fork (frozen/scripted opponent rows excluded from loss)
                                │
      eval harness: fixed anchor pool · paired seat-swapped A/B · conditionals · BR probes · meta-game
      raylib viewer + human-play mode · (late) passive video measurement / real-client bridge
```

**Design principles (from prior-art failures):**
- One source of truth per fact.
- Integer client units.
- Snapshots as plain-old-data.
- Instruments that measure behavior and data, not only unit tests.
- Every gameplay-affecting change bumps the engine version and starts a new result lineage.

**Cross-platform determinism rules** (the Mac and Linux must produce the same state hash):
- no float `sqrt`, `atan2`, `sin`, or `cos` in simulation code; use integer `isqrt` or lookup tables
- no `qsort`, since tie order differs between libcs; use a stable in-house sort
- floor-division helpers, because C truncates where Python floors
- zeroed struct padding
- a seeded PCG32 per env
- a golden-hash test run in CI on both platforms

---

## 5. Blockers

| # | Blocker | Resolution path |
|---|---|---|
| B0 | **Course deliverables unknown**: proposal/milestone dates, rubric, report format, code-release rules, whether human-subject play needs IRB | D0, first thing |
| B1 | **Can't train at scale on the Mac.** 3.0 runs on the CPU for smoke tests only. | Torch via SLURM + Singularity (PufferDrive's recipe). **W0 spike: Puffer 3.0 + a toy 1v1 env + the league fork + mask-in-observation, trained on Torch.** |
| B2 | **Ground truth.** The references disagree (§2 table); RoyaleSim's recordings are private. | Ledger v0 reconciled across the sims and client columns. Measure the disputed facts from **passive** recordings (screen-record our own friendly battles and replays; no automation). About 10 hand-measured calibration scenarios, e.g. Hog vs. Cannon pull, Log width, walk speed per tier. |
| B3 | **Current card data** (Evos, Heroes, tower troops, newer cards) | Tier A uses RoyaleSim's committed `cards-15.535.json`. W5+: get the current APK (Android emulator, `adb pull`), decode it with the RS or HC tools, and extend the decoding to evo and hero TOMLs. Never commit or redistribute raw client files. |
| B4 | **Trainer gaps.** 3.0 has no masking and no opponent pool. | Put the mask in the observation and apply it in our policy at rollout *and* update time. Use `finfo.min`, not `-inf`, to avoid NaN entropy. Fork the `pufferl` rollout so opponent rows run frozen or scripted policies and are excluded from the buffer and loss. |
| B5 | **Scope vs. 10 weeks solo** | Tiered definition of done (§9). Pin one game version. **Freeze the engine for experiments at W7.** |
| B6 | **ToS / legal.** Automation *and* live advisor assistance are third-party software under Supercell's policies, and bots get permanently banned. Fan Content Policy limits apply. | D11: December scope = **passive video measurement plus human evaluation inside our own sim**. No Supercell art in our renderer; add the unofficial disclaimer. |
| B7 | **HPC friction**: GPU-hour quota and fairshare, H200 queue waits, wall-time limits, storage quota, container CUDA vs. host driver | Default to L40S. Checkpoint *and* resume the full league state (pool, PFSP stats, optimizer). Keep a checkpoint retention policy. Pin the Singularity image. |

---

## 6. Open decisions, in resolution order

We'll take one per discussion. Each has a recommendation.

**D0 · Course deliverables and research questions** _(blocker)._ Get the rubric and dates, and fix the game-theory research questions now; that's the grade. Candidates:
- **RQ1:** Does a PFSP league reduce exploitability compared with naive self-play in this POSG? Measured with best-response probes.
- **RQ2:** Does last-iterate regularization (MMD, a KL term toward a reference policy) lower exploitability at equal compute?
- **RQ3:** The empirical meta-game: a payoff matrix over deck archetypes (or checkpoints) and its Nash, compared with real ladder usage. This is the deck-selection layer.
- **RQ4:** How fast does hidden information collapse (the cycle deduction in §1), and do derived belief features beat a pure LSTM?

Also confirm the tiered definition of done (§9) against your "full project by the deadline" goal.

**D1 + D6 · Engine strategy and trainer, decided together** (they define `pr_env.h`)
- **D1 recommendation: two backends behind one interface.**
  - Backend 0 wraps CRE (week 1). The trainer, eval harness, league, and probes get debugged on it by W2–W3.
  - Backend 1 is our fresh C engine (RS architecture and ledger as the spec, CRE for breadth). It swaps in at M1, around W4.
  - **The bailout costs nothing:** if backend 1 slips, we keep training on backend 0 after patching its worst gaps (spell tower damage, the triple-elixir window).
  - Results on backend 0 are pipeline tests only.
  - Rejected alternatives: fresh-only (RL waits 4+ weeks); wrap-only (we never own fidelity); porting RS's Rust (thin coverage, changes daily).
- **D6 recommendation: Puffer 3.0 (PyTorch)**, with the known traps (B4):
  - The ability head is always present and masked, since tensor shapes are fixed.
  - Check that the LSTM wrapper and BPTT work with the masked observation.
  - Pin the commit.
  - A 5.0 port is out of the 10-week scope (Tier C).

**D12 (minimum) · Repo and infra.**
- Mono-repo; C99 built with clang on the Mac and gcc on Torch.
- ASan/UBSan debug builds; CI on Linux and macOS with the golden-hash test.
- `uv` for Python; a Singularity image.
- Experiment tracking (wandb).
- `docs/DECISIONS.md` and `docs/PLAN.md`.

**D11 · Real-client scope** _(decide early, because it changes W7–W9)._ Recommendation for December: passive video measurement plus human play in our sim. Advisor or actuation modes come only after the course, knowing the ToS risk.

**D8 · Reward** _(the zero-sum rule is fixed early)._
- Terminal: **+1 / −1 / 0 for a draw, for both seats**. Anything else, like a draw counted as a loss, makes the game general-sum and voids the Nash and exploitability framing.
- Shaping must be **antisymmetric**: `γΦ(s′) − Φ(s)` with `Φ = f(own) − f(enemy)`, e.g. the tower-HP differential. It's annealed toward zero.
- γ ≈ 0.999, so the horizon covers the match.
- **No safe null action that earns a guaranteed zero.**
- Protein/wandb sweeps for the weights.

**D2 · Data pipeline and version pin.** Tier A uses RS's `cards-15.535.json` pinned by hash, with codegen into `card_db.h`. W5+: our own APK decode for Evos, Heroes, and newer cards. Pin *that* client version and ignore later patches.

**D3 · Fidelity spec.**
- Ledger v0: each constant gets a status (guess → measured) and evidence.
- Settle the §2 table.
- The recording workflow.
- Tier A "faithful enough" = the 10 hand-measured scenarios pass, plus differential agreement with RS on shared cards within tolerance.

**D4 · Tick and cadence.** A 20 Hz engine; a decision every 10 ticks (0.5 s); an action-latency model. Finer or event-triggered cadence is a later study.

**D10 · Evaluation protocol** (built *before* the main training runs):
- a fixed anchor pool: scripted bots plus frozen checkpoints
- paired, seat-swapped fixed seeds
- per-card conditionals (P(play | in hand), placement modal share)
- per-deck win rates
- **3 budgeted best-response runs**
- meta-game payoff matrices
- human play in the viewer
- regime tags on every number

Win rate against a fixed anchor pool replaces "monotone Elo", since Elo is meaningless in non-transitive games.

**D5 · Observation.**
- **Spatial planes:** own/enemy attributes, summed HP, deploy timers, status effects.
- **Flat features:**
  - own elixir, hand, next card
  - clock and elixir phase
  - tower HPs
  - opponent's cumulative spend
  - opponent card history
  - **derived beliefs:** revealed deck, deduced hand and next card, opponent elixir lower and upper bounds
- Mirror per seat by position, then truncate.
- Decide when needed (around W2).

**D7 · Action space.** A card head (4 slots + wait) plus a **joint placement head conditioned on the chosen card**, with affordability and per-card legality masks; or a flattened joint `1+4×576`. Factorized x/y heads plateaued in prior work. Decide around W2.

**D9 · Algorithm and opponents.**
1. Optional BC warm start from a scripted C bot (needs value warmup and a KL target if used).
2. Shared-weights self-play.
3. A PFSP league: frozen checkpoints plus scripted bots (rusher, defender, cycler, counter) kept forever.
4. MMD regularization.

Deck curriculum: one matchup → 2–3 decks → pool. Decide around W4.

---

## 7. Risks and mitigations

| Risk | Evidence | Mitigation |
|---|---|---|
| The policy exploits sim bugs; behavior is authored by physics or reward | CRE §5: Cannon parked behind the King, Fireball never cast, "apathy" | Exploit probes that actually perturb (HC's didn't). Engine-scored counterfactuals. Per-card conditionals from day one. Play the sim ourselves weekly. |
| Silent engine defects (absorbing states, bad data, duplicated facts) | CRE: bridge traps stalled about 20% of lone crossings; speeds were 4–5× off with tests green | Instruments: soak, trajectory sweeps, spawn-speed and sight audits. A data round trip against the client table. One arena layout. |
| Body-block miscalibration decides whether defense works | HC's permanent block; RS's summed push-outs cause a wedge | RS's measured contact law as the prior. Dedicated pull-timing scenarios. |
| Wrong economy or rules shift every card's value | CRE ran flat 1× elixir for 7 weeks | Rules layer verified against recordings first. |
| Training pathologies | CRE §4 and ClashAI: wait-collapse, per-head entropy mis-scaling, mask noise, γ too short, mirror off-by-one | Masks recomputed at update time. Entropy normalized by log(n_legal). γ ≥ 0.999. Seat-swapped null test. Advantages normalized over decision rows. |
| Non-transitive cycling or forgetting | PSRO theory; CRE phase-2 specialization | PFSP with permanent anchors, MMD, BR probes. |
| Engine churn invalidates experiments | CRE crossed 8 comparability boundaries in 6 weeks | **Engine freeze at W7.** Version tags on every result. |
| Throughput over-estimated | A match is 2 seats × 360–600 decisions | Budget **50–100k agent-steps/s**, i.e. about 4–9M matches/day, still about 200–400× CRE's rate. Check CPU cores per GPU on Torch. |
| Backend 0 builds poorly on Linux/Mac (Windows-first) | CRE's roadmap lists Linux/macOS wheels; tests were built with g++ under WSL | W0 compile check. It's header-only C++17, so the risk is low. |
| Puffer API churn | 3.0/4.0/5.0 are incompatible | Pin a commit; thin adapter. |
| Sim-to-real gap | RS: 76% outcome agreement; ClashAI: latency costs 2–16 percentage points | Latency and noise injection (Tier B). Report measured gaps honestly. |

---

## 8. Lessons we carry forward

**From ClashRoyaleEnv:**
1. Build the evaluation harness before the trainer.
2. Log conditionals, not only aggregates.
3. Tag results with the engine version.
4. Seed everything; do paired A/B tests on snapshots.
5. **"If doing nothing is a guaranteed zero, the learner will find it."**
6. Test mechanisms in unit tests, and measure data and trajectories with instruments.
7. Masks are computed from the stored observation and validated against engine *behavior*.
8. Their scale was a CPU laptop at about 943 episodes/hour. **We use our scale to buy correctness, not to hide bugs.**

**From Hasty-CR:**
- Opponent diversity is not optional: a meta-only diet gave 17% against the rule engine; a mixed diet gave 93%.
- Choose checkpoints against the right opponent set.
- A BC warm start needs value warmup and a KL target.
- Humans find bugs that 30M training steps miss.
- Verify the env's turn ordering, and verify that probes actually perturb something.

**From ClashAI:** a sim that matches real outcomes only 26% of the time makes RL worse than imitation. Fidelity gates come before RL claims.

---

## 9. Roadmap (10 weeks) and definition of done

| Week | Dates | Engine track (backend 1) | RL and eval track | Gate |
|---|---|---|---|---|
| W0 | Sep 26 – Oct 4 | Decide D0, D1+D6, D12-min. Repo scaffold. `pr_env.h`. | **Torch spike:** Puffer 3.0 + a toy 1v1 env + league fork + mask-in-observation. Backend 0 compiles on the Mac and Linux. | **M0:** a league spike trains on Torch |
| W1–W2 | Oct 5 – 18 | Rules, economy, cycle, placement, towers, entity pools, targeting, A*, contact. RS JSON → `card_db.h`. Ledger v0. Passive recordings of the disputed facts. | Backend 0 wired. Observation, mask, and reward v0. Eval harness v0. **First PPO plus league on backend 0** (pipeline test). | M1a: the pipeline learns on backend 0 |
| W3–W4 | Oct 19 – Nov 1 | Thin slice, **about 20 cards over 2–3 decks**: projectiles, spells, splash, death effects, charge, river hop, knockback, stun/slow. Viewer plus human play. Instruments. Differential tests against RS. | Scripted C bots. Eval harness complete. | **M1: fidelity gate → swap to backend 1** (or keep backend 0) |
| W5–W6 | Nov 2 – 15 | If on track: Tier B batches, APK decode for Evos and Heroes, more tower troops. | League plus MMD on backend 1. First BR probe. Pre-register the experiments. **Start the report.** | M2: beats scripted bots; seat-swapped null holds; probes clean |
| W7 | Nov 16 – 22 | **Engine freeze (v1.0).** | Main experiment launches: self-play vs. PFSP vs. MMD. | Freeze |
| W8–W9 | Nov 23 – Dec 1 | Bug fixes only, each one a new lineage. | BR probes (3 total), meta-game analysis, human play tests in the sim. | M3: RQ results |
| W10 | Dec 1 – 4 | – | Report, demo, final freeze. | Final |

**Critical path:** interface → Torch league spike → PPO on backend 0 (W2) → backend 1 swap (W4) → freeze (W7) → probes and meta-game (W8–9) → report (W6–10).

**Tiered definition of done (to confirm in D0):**
- **Tier A (must):**
  - Full match rules.
  - About 20 cards over 2–3 decks.
  - Princess tower.
  - The mechanic families those cards need.
  - A verified engine: tests, instruments, differential checks, a cross-platform determinism test.
  - Puffer env, league plus MMD, 3 BR exploitability probes, meta-game analysis.
  - A playable viewer (humans against the bot).
  - The report.
- **Tier B (should):**
  - A 16-deck pool (about 60–80 cards).
  - All 4 tower troops.
  - One each of the Evo, Hero, and Champion frameworks.
  - Latency and noise injection.
  - Sim-to-real numbers from recordings.
- **Tier C (stretch, likely after December):**
  - Full card coverage (the "ALL features" end state).
  - A Puffer 5.0 port.
  - A real-client bridge (advisor or actuation).

---

## 10. Verification strategy

**Engine**
- Unit tests per mechanic.
- Golden traces: seed plus script gives a fixed state hash, on the Mac and in Linux CI.
- Snapshot round trip.
- 180° rotation symmetry test.
- ASan/UBSan builds.
- Instruments: soak, trajectory sweeps, spawn-speed and sight audits, King activation.
- Data round trip against `cards.json`.
- Differential tests against pinned RoyaleSim on shared cards: the same scripts, compare trajectories, and triage every divergence into the ledger.

**Fidelity**
- The 10 hand-measured scenarios from our recordings.
- Disputed facts are measured, never assumed.

**Env / RL**
- The mask is validated against engine behavior (was the elixir actually spent?).
- Observation mirror test.
- Seat-swapped paired null test.
- Per-card conditional probes.
- Exploit probes (with a check that the perturbation took effect).
- Best-response exploitability runs.
- Human play tests.

---

## 11. Immediate next steps (after approval; still no engine code)

1. **Set up the repo.** `https://github.com/Aditya-Gupta26/PufferRoyale` is empty, so connect it to the existing project folder rather than cloning into a subfolder: `git init -b main` and `git remote add origin …`. Keep the empty `ideas.txt`. Then add:
   - `README.md`: what the project is, status, the unofficial Supercell disclaimer, a no-bots-on-live-servers note.
   - `docs/PLAN.md`: this plan.
   - `docs/DECISIONS.md`: a decision log. Each entry records the decision, the options, the evidence, the date, and what would reverse it. D0–D12 are listed as open.
   - `docs/REFERENCES.md`: links to the reference sims, PufferLib, datasets, and papers.
   - `.gitignore`: C/Python build outputs, `checkpoints/`, `data/raw/` (never commit client files), `*.apk`.

   Save the project context to memory. **Commit and push only when you say so.**
2. Resolve **D0** (you bring the course rubric and dates; we fix the RQs and tiers).
3. Resolve **D1 + D6** together, then D12-min and D11. That unblocks W0: the scaffold, `pr_env.h`, and the Torch league spike.
4. Resolve D8 → D2/D3/D4 → D10 before the matching work starts. D5/D7 around W2 and D9 around W4.
