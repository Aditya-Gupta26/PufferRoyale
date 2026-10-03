# PufferRoyale — status & handoff (2026-09-27)

Everything below was built locally on the Mac during the handoff session and pushed as commit
`b6b63da`; no HPC was used. **The training plan of record is `docs/TRAINING_PLAN.md` (2026-10-02).** `docs/SPEC.md` (v0.4.2) is the binding contract.
`docs/DECISIONS.md` records every decision, and `docs/FIDELITY.md` lists every modelling choice and
known divergence from the real game.

## What exists

| Layer | What | Where |
|---|---|---|
| Data | 2026 client card tables (15.535.29, decoded by RoyaleSim), arena tilemap, rules ledger; all pinned, with provenance | `data/source/` |
| Codegen | Level-11 integer stats for 64 cards, spawned units, projectiles, areas and buffs; tower ladder; arena bitmask. Deterministic, with asserted reference values | `tools/gen_*.py` → `pufferroyale/csrc/pr_*_db.h` |
| Engine | Header-only C99, integer-only, 20 Hz (details below) | `pufferroyale/csrc/` |
| Bots | `noop`, `random` and `heuristic`. All seat-symmetric and legal-only | `pr_bots.h`, `pufferroyale.Bot`, `bot_action` |
| PufferLib env | `pufferroyale.Royale(PufferEnv)` (details below) | `royale.h`, `binding.c`, `royale.py`, `config/royale.ini` |
| Policy | CNN over board planes + pooled entity MLP + card-id embedding (+ fixed card-stat table) → value and a conditional head (card, then position given the card; SPEC §19.6) or the v0.4 flat head, masked joint logits on any placement grid; `Recurrent` (LSTM) | `pufferroyale/torch.py` |
| Training | `train.py` (PuffeRL; `MMDPuffeRL` with coefficient 0 = plain PPO; reward clip and entropy split, `--resume` / `--init-from`, wandb), `stages.py` bot ladder, `plot_history.py` | `scripts/`, `pufferroyale/trainer.py` |
| Game-theory tooling | League with PFSP opponent pool (frozen checkpoints, scripted anchors, self), MMD regulariser, best-response exploitability probe, tournament with Elo and meta-game Nash solver, deck meta-game | `pufferroyale/{league,trainer,metagame}.py`, `scripts/{league_train,best_response,tournament}.py` |
| LLM play (your `ideas.txt`) | Text state renderer, action parser, `LLMAgent`, match harness, offline mocks, optional Anthropic adapter (off unless `--allow-network`) | `pufferroyale/llm.py`, `scripts/llm_match.py` |
| Tools | Text and raylib viewer, benchmarks, eval (per deck, Wilson intervals), snapshot transitivity, 21-check end-to-end ladder | `scripts/{watch,bench,eval,transitivity,e2e_check}.py` |
| HPC | Apptainer definition + SLURM scripts for NYU Torch. **Written but never run** | `hpc/` |

**Engine details:**
- Full match rules:
  - 3:00 regulation plus 2:00 sudden-death overtime and a tiebreak.
  - 1×/2×/3× elixir, starting at 6.
  - A 4.5 s deploy lockout; King activation after 3.55 s.
  - Placement pockets; card cycle.
- Mechanics:
  - A measured-order tick pipeline and progress-credit attack cycle.
  - A* pathing with no stuck units, and mass collision.
  - Projectiles (homing and non-homing), splash, spells, status effects, knockback.
  - Spawners, death spawns and bombs, Inferno ramp, dash, charge, burrow, river jump.
  - 3 tower troops.
- State and speed: plain-data state; `memcpy` snapshot and restore with validation; FNV hash. About 500k ticks/s.

**Env details:**
- 1 or 2 agents (self-play or vs bot).
- Discrete(2305) own-frame actions (or a coarse `placement_grid` of 2 / 4: Discrete(577) / Discrete(161),
  SPEC §19.4); v0.5 observation of 17,715 floats (the v0.3 layout + the own deck, SPEC §19.3) with the
  exact legality mask.
- Zero-sum reward; optional potential-based shaping (reward v2, SPEC §19.1); deck sampling from pools /
  random decks with held-out decks (SPEC §19.5); auto-reset; logs incl. per-card play rates.
- About 26k steps/s per core.

## Cards (64) and presets
- **v0.1 (0–20):** Knight, Archers, Musketeer, Giant, Hog Rider, Minions, Baby Dragon, Valkyrie, Skeleton Army, Skeletons, Ice Golem, Ice Spirit, Prince, Wizard, Cannon, Tesla, Fireball, Arrows, Zap, The Log, Goblin Barrel.
- **v0.3 (21–63):** Barbarians, Mini P.E.K.K.A, P.E.K.K.A, Mega Minion, Bats, Spear Goblins, Goblins, Goblin Gang, Royal Giant, Bomber, Princess, Dart Goblin, Minion Horde, Fire Spirit, Ice Wizard, Golem, Lava Hound, Balloon, Giant Skeleton, Witch, Tombstone, Inferno Tower, Inferno Dragon, Bandit, Battle Ram, Royal Hogs, Miner, Mortar, X-Bow, Elixir Collector, Bomb Tower, Rocket, Poison, Freeze, Earthquake, Lightning, Giant Snowball, Barbarian Barrel, Rascals, Elite Barbarians, Skeleton Dragons, Wall Breakers, Night Witch.
- **Tower troops:** Tower Princess (default), Cannoneer, Dagger Duchess, Royal Chef.
- **Deck presets:** `hog26`, `giant`, `bait`, `golem`, `lavaloon`, `xbow`, `miner_poison`, `pekka_bridge`, `royal_hogs`, `random`.
- **Not yet:** Evolutions, Heroes, Champions, and the other roughly 60 cards (e.g. Electro Wizard, Mega Knight, Tornado, Graveyard, Executioner, Bowler, Hunter, Royal Ghost, Clone, Mirror, Rage). Most newer cards need the client's "action graph" logic, which has no public extractor.

## How it was verified
**How the work was split:**
- **Simulator (v0.1–v0.4):** an orchestrator, three builder agents, an independent black-box
  **tester** (it never read the implementation) and **two independent code audits**.
- **Training work package (v0.5, SPEC §19, 2026-10-02/03):** the same process, with builders for the
  env side and for the policy/trainer/tools side, the same independent tester, and one more
  independent audit (its findings became SPEC §19.10).

Every audit finding was fixed and got a regression test.

Final independent run, on a clean rebuild (2026-10-03):

| Check | Result |
|---|---|
| C unit/scenario tests (`make test-c`) | 184,647 checks, 0 failures, warning-free (clang; v0.4 was also checked with gcc-15) |
| AddressSanitizer + UBSan (`make asan`) | clean |
| Golden hashes (`scripts/golden_hashes.py`) | PASS: the engine is unchanged by v0.5 |
| Tester spec tests (`pytest tests/spec`) | **1,455 passed** (incl. 286 for v0.5) |
| Builder tests (`pytest tests/builder`) | **416 passed** |
| End-to-end ladder (`scripts/e2e_check.py`) | **21/21 PASS** |

**What the e2e ladder covers:**
- build, codegen, C tests, ASan, pytest;
- determinism, soak, mask, zero-sum, no-leak, auto-reset;
- the v0.5 checks: `reward_v2`, `policy_heads`, `decks`, `grid`, `ops`;
- perf, llm, train, eval and league.

**CPU learning check (v0.5, TRAINING_PLAN §7):** the default policy learned to beat the random bot
(0.94 without shaping, 0.98 with shaping; 100 matches each) within about 0.6M learner steps on the
Mac.

Other checks:
- **Mirror fuzz:** 1,000 bot-driven matches, each checked against its 180° seat-rotation, 1.15M ticks, **0 divergences**.
- **Seat bias:** none measurable (300-seed mirror matchups, within 2σ).
- **Pathing:** 16k lone-unit walks from every legal tile all reach their target.

## Quick start
```bash
source .venv/bin/activate
make build && make test                    # build and run all tests
python scripts/e2e_check.py                # the whole verification ladder
python scripts/watch.py --deck0 hog26 --deck1 golem                          # watch bots (text board)
python scripts/train.py --total-timesteps 100000 --device cpu                  # local smoke training
python scripts/league_train.py --total-timesteps 150000 --anchors bot:heuristic --device cpu
python scripts/tournament.py --agents bot:noop bot:random bot:heuristic --decks hog26 --matches 4 --out t.json
python scripts/llm_match.py --model mock_first_legal --opponent bot:heuristic --matches 4 --out llm.json
```
`README.md` has the full command reference, including the PufferLib 3.0 source-install fix.

## Known approximations (details in FIDELITY.md)
- **Where the numbers come from:** the stats are the client's own, but the behaviour rules follow RoyaleSim's measured ledger where it exists and documented choices where it doesn't.
- **Simplified vs the real game:**
  - collision/contact is simpler than the measured contact law;
  - knockback is instantaneous;
  - Poison and Earthquake damage are area events, not per-unit pulses;
  - the Dagger Duchess reload and the Royal Chef rules are invented within the data;
  - Log knockback follows the roll direction.
- **Settings still open for review** (all configurable): start elixir 6, the 4.5 s lockout, and the tiebreak on absolute HP.
- **Not calibrated:** nothing has been checked against recordings of real matches yet (see PLAN §5, B2).
- **Training:** only local CPU smoke runs (100–150k steps) have been done. They prove the pipeline, not strength. The heuristic bot still beats every trained checkpoint.

## Suggested next steps
1. **Review the autonomous decisions** in `docs/DECISIONS.md` (D1–D9, then D13+).
2. **Training:** follow `docs/TRAINING_PLAN.md`. The work package (reward v2, conditional policy head, card stats, own deck, deck sampling, `placement_grid`, operations) is built on the Mac first. In parallel, hand `docs/HPC_HANDOFF.md` to the HPC agent for cluster bring-up (isolated environment, G0, throughput), then stages 1, 3 and 4. Keep float32 unless you're testing bf16; the bf16 bug is fixed but only CPU-verified.
3. **Course research questions:** self-play (A) vs PFSP league (B) vs league + MMD (C), from random init with no scripted bots in training, compared by exploitability (`best_response.py`); the deck meta-game Nash (`deck_metagame`). Details in TRAINING_PLAN §2 and §8.
4. **Fidelity:** record a few real matches and measure the disputed facts (start elixir, lockout, tiebreak, speeds).
5. **LLM matches:** `llm_match.py --model anthropic:claude-opus-5-5 --allow-network ...`. Check `cache_read_input_tokens` in the transcripts, because the rules prompt (~2.8k tokens) may be below some models' caching minimum.
6. **More cards:** Evolutions, Heroes and Champions need their data decoded from a current APK (PLAN B3).

## Where things are
- **Design and decisions:** `docs/PLAN.md`, `docs/SPEC.md`, `docs/DECISIONS.md`, `docs/FIDELITY.md`, `docs/STATUS.md` (this file).
- **Training:** `docs/TRAINING_PLAN.md` (what and why), `docs/HPC_HANDOFF.md` (how, on the cluster).
- **Tests:** `tests/spec/` (tester-owned), `tests/builder/`, `tests/c/`.
- **Experiment outputs** (gitignored): `experiments/` — smoke runs, the league demo, LLM demos.
