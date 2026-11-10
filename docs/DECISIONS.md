# Decision log

Each entry records what was decided, the options, the evidence, and what would reverse it.
Entries marked **(autonomous)** were made by Claude during the 2026-09-26 handoff while the
project owner was away; they are open for review.

---

### D1 — Engine strategy: fresh C engine, header-only, as a PufferLib 3.0 env (autonomous, 2026-09-26)
- **Options:** (a) fresh C engine; (b) wrap ClashRoyaleEnv (C++); (c) port RoyaleSim (Rust).
- **Decision:** (a), matching the owner's instruction ("a new env like PufferDrive ... ClashRoyale
  specific"). The backend-0 wrapper of ClashRoyaleEnv from the plan was **not** built: the owner
  asked for the simulator itself, and one engine is simpler to verify.
- **Evidence:** ClashRoyaleEnv's own report shows float/10 Hz/hard-coded-data designs accumulate
  silent defects; RoyaleSim's measured, integer, data-driven design is the better spec.
- **Reverse if:** the fresh engine cannot reach the thin-slice fidelity gate.

### D6 — Trainer: PufferLib 3.0 (PyTorch) (autonomous, 2026-09-26)
- **Options:** 3.0 (PyPI/PyTorch), 4.0 (unpublished), 5.0 (pure CUDA; no Mac training; custom nets need hand-written CUDA).
- **Decision:** 3.0 — the version PufferDrive uses; custom policies in PyTorch; runs on CPU/Mac for smoke tests; masks applied inside our policy (mask is part of the observation).
- **Local install note:** PufferLib 3.0's `setup.py` has a bug under `NO_OCEAN=1` (`c_extension_paths` undefined); installed from the 3.0 branch (commit `3b5c604`) with a one-line local fix.
- **Reverse if:** throughput on the HPC becomes the bottleneck → port the frozen policy to 5.0.

### D2 — Card data: RoyaleSim's decoded 15.535.29 table, level 11 (autonomous)
- Pinned `data/source/cards-15.535.json` (RoyaleSim `72ed062`). Level-11 multiplier 256% for all
  base objects (measured by RoyaleSim: Knight 690→1766, Hog 663→1697). Towers use the measured
  compounding tower ladder (Princess 3052/109, King 4824/109).
- **Reverse if:** we decode a newer APK ourselves (needed for Evos/Heroes/tower troops).

### D3 — Rules constants from the RoyaleSim ledger (autonomous)
- 20 Hz ticks; start elixir **6**; 1×/2×/3× at 0:00 / 2:00 / 4:00; overtime 120 s sudden death;
  deploy lockout 4.5 s (90 ticks, configurable); King wakes 3550 ms after trigger; tiebreak =
  weakest tower absolute HP (configurable); crown-tower damage = `ceil(D·P/100)`.
- Disputed facts from the plan's §2 table are resolved in favour of RoyaleSim's *measured* values
  and made configurable where the evidence is MEDIUM or lower.

### D4 — Decision cadence: one decision per 10 ticks (0.5 s), configurable `frame_skip` (autonomous)

### D5/D7 — Observation & action (autonomous)
- Action: joint `Discrete(2305)` = no-op + 4 hand slots × 576 own-frame tiles (native to
  PufferLib's sampler; card→tile conditioning is inside the joint head).
- Observation: spatial planes + 48-slot entity list + scalars (incl. deduced opponent cycle) +
  the exact legality mask. Opponent hidden info never leaks.

### D8 — Reward (autonomous)
- Terminal +1/−1/0 (draw 0 for both → zero-sum). Optional antisymmetric shaping (tower HP
  fraction, crowns), default 0.

### D9 — Scope of this pass (autonomous)
- 21 cards (3 decks), default tower troop. Evolutions/Heroes/Champions/tower troops deferred.

### Future — LLM opponents (from `ideas.txt`)
- "Play against LLMs and compare performance across Opus 5.5, Fable, Sonnet, and other accessible
  models." The text renderer + `Game` scenario API are designed so an LLM agent can read a text
  state and emit a (slot, tile) action; to be built after the simulator is verified.

---

## Builder decisions, Phase A (2026-09-26)

### D13 — Engine layout: header-only C99, one POD state, entities sorted by id (builder)
- **Decision:** `pufferroyale/csrc/pr_*.h`, all functions `static inline`; `PrState` holds
  everything (fixed pools: 256 entities, 256 projectiles, 64 effects, 64 pending spawns).
  Entities are kept sorted by creation id (append + stable compaction), so every pass runs
  in creation order and "lowest id" tie-breaks are array order; lookups are binary search.
- **Why:** memcpy snapshots, a hash that depends only on content, no pointer fix-ups.
- **Reverse if:** profiling shows compaction or O(n^2) scans dominate at larger unit counts
  (measured: 2.5-3.3 us/tick in random matches, ~300-400k ticks/s).

### D14 — Ground pathing: Dijkstra flow fields + line-of-sight smoothing (builder)
- **Options:** per-unit A*, bridge-waypoint heuristics, flow fields.
- **Decision:** reverse Dijkstra fields per (goal, jumper) over the 36x64 grid, occlusion as
  a 6x entry cost (never a wall), waypoint = farthest visible cell on the descent chain;
  field cache outside the state but a pure function of it (keyed by an FNV signature of
  the living structures); per-entity waypoint cache inside the state.
- **Evidence:** 6216/6216 lone walks from every legal tile reach their tower; cold-cache
  replays reproduce hashes; cheap (a few dozen field builds per match).
- **Reverse if:** fidelity work needs RoyaleSim's measured 16.402 search (costs, goal rule).

### D15 — Seat symmetry by team-oriented point queries (builder)
- **Decision:** a moving unit's cell / water / clamp queries use its own frame's half-open
  convention; "wet" = water under either convention. A scenario and its rotation evolve as
  exact rotations (tested for all 21 cards).
- **Why:** both seats share one policy; a colour-dependent board is a silent bias.

### D16 — Debug API semantics the SPEC left open (builder; open for review)
- `play(x, y)` = the literal tap point: legality of the own-frame tile containing it, troops
  and spells placed at (x, y), buildings at that tile's anchor (v0.2: even footprints on the
  tile's own-frame top-left corner); queued plays are validated at call time against the
  current state (superseded in Phase C, see D20) and re-validated when applied; `spawn(deployed=True)` = as if the deploy just completed; `reset()` without a
  seed continues the RNG stream; `set_tower_hp(<=0)` destroys at once and judges King/crown
  conditions immediately; `end_reason = "DRAW"` for every draw. Details in FIDELITY.md.

---

## Builder decisions, Phase C (2026-09-26)

### D17 — Env = one match per native env, obs written by the engine's own legality code (builder)
- **Decision:** `royale.h` holds one `PrGame` per native env plus a bot and a learner-side
  RNG; `num_agents` 2 = self-play rows (team 0, team 1), 1 = learner row + scripted bot. The
  observation encoder (`pr_obs.h`) is C, own frame, float32, and its mask section is written
  by `pr_legal_mask` -- the function `play` uses -- so mask == legality by construction; a
  `mask_check` flag re-validates every action each step as a runtime self-check.
- **Why:** PufferLib's zero-copy native path; a Python-side mask could drift from the engine.
- **Measured:** ~23-25k env steps/s single process (frame_skip 10, 2 agents, obs), 11x the
  SPEC §11 floor; 76.4 KB per agent row (FIDELITY §9 lists what could shrink).
- **Reverse if:** rollout memory becomes the bottleneck on a cluster (then uint8 planes /
  factored mask, which need a SPEC change).

### D18 — Heuristic bot: fixed-priority rules, integer-only, legal by construction (builder)
- **Decision:** finish towers with spells > defend (cluster spells, central building vs
  buildings-only targeters, cheapest capable troop placed by range) > rate-limited offence
  (win condition at the weaker lane's bridge with >= 7 elixir, support troops, spend before
  capping). Chooses only from the exact mask; own PCG32 stream.
- **Why:** a cheap, deterministic sparring partner that beats random play (6/6) and punishes
  passive policies, without leaking engine internals to the learner.
- **Reverse if:** agents saturate it quickly -- then add league / past-checkpoint opponents.

### D19 — Policy: CNN + pooled entity MLP + scalar MLP, joint masked logits (builder)
- **Decision:** `pufferroyale/torch.py` `Policy` (~1.7M params): 3-layer CNN over the 25
  spatial planes, a shared entity MLP with masked mean + max pooling per side, a scalar MLP,
  a trunk, 2305 logits with illegal actions set to `finfo.min`, and a value head. `Recurrent`
  wraps it in `pufferlib.models.LSTMWrapper` (the mask is stashed at encode time because the
  wrapper's decode only sees the hidden state).
- **Evidence:** masked actions are never sampled (tested); 100k-step CPU smokes, self-play and
  vs heuristic, ~1k SPS, finite losses.
- **Reverse if:** the joint 2305 head learns card choice poorly (then a factored slot -> tile
  head).

### D20 — Plays validated on the current state; simultaneous order team 0 first (SPEC v0.2)
- **Decision:** `play` and the mask are the same function of the current state (§13.1);
  queued plays are re-validated in order at Upkeep and dropped (counted) if they no longer fit.
  Same-tick plays of both teams: team 0 first (§13.13). Alternating the first team per tick is
  implemented as an opt-in flag (`alternate_first`) and tested as an exact mirror.
- **Why:** SPEC v0.2 pins both; the virtual-hand model of Phase A made the mask disagree with
  `play` after a pending play.
- **Reverse if:** the orchestrator adopts alternation as the default (one flag flip).

### D21 — raylib window only when a display exists (builder)
- **Decision:** `PR_RAYLIB=1` builds the window renderer; before `InitWindow` the env checks
  for an active display (CoreGraphics on macOS, `DISPLAY` / `WAYLAND_DISPLAY` on Linux) and
  otherwise prints the text board.
- **Why:** GLFW segfaults inside `InitWindow` in headless sessions (seen in this session:
  "Failed to determine Monitor").

---

## Builder decisions, Phase C.1 (audit fixes, SPEC v0.2.1, 2026-09-26)

### D22 — Same-tick plays validated on the start-of-Upkeep state (SPEC §14.1)
- **Decision:** Upkeep first validates every queued play of both teams against the state after
  regen (a team's own plays in order on its running elixir / hand; the board as it stood), then
  applies the accepted ones team 0 first. Cross-team plays never invalidate each other.
- **Why:** a mask-legal action could be dropped when the enemy's same-tick building covered a
  pocket tile (auditor's `pocket_drop.py`); now `dropped_plays` is 0 in env play by construction.

### D23 — Hit-once memory as a slot bitset remapped on compaction (SPEC §14.3)
- **Options:** a 256-id list per effect (+48 KB of state), pruning dead ids, a bitset.
- **Decision:** `uint32_t hit_bits[8]` per effect keyed by pool slot; `pr_compact_entities`
  remaps the bits (dead slots dropped). The state shrank by 14.6 KB (68.2 -> 53.6 KB).
- **Reverse if:** entities ever get inserted mid-array (the pool is append + stable compaction).

### D24 — Chase-field storm: block goal keys, exact scans made cheap (audit perf item)
- **Decision:** troop-chase goals are quantised to 2x2-tile chase blocks (exact-cell fallback
  inside the block); Target-phase scans use per-team bucket grids, a sqrt-free bound and an
  explicit id tie-break; Collide enumerates pairs by a row sweep. Results are bit-identical to
  the plain loops where the rewrite is only an acceleration (same state hashes), and the cache
  stays a pure function of the state (auditor's `cachepure.c` identical).
- **Evidence:** auditor's `stress2.c` (216 Skeletons fighting across the river) 2.2k -> 12-14k
  ticks/s, field builds 713 -> 42 per 300 ticks; typical play unchanged (21 builds / 40 matches).
- **Not done:** capping builds per tick (it would make movement depend on how many units replan
  in the same tick); shrinking the field cache below 24 slots (the stress thrashes at 16).

### D25 — Melee mid-swing retarget per ledger (SPEC §14.2)
- **Decision:** direct strikers switch to the nearest enemy in reach keeping progress, else cancel;
  projectile attackers hold within reach + 500 until firing (troops and structures alike, as
  §14.2 names projectile attackers without restriction).
- **Open:** the ledger's control run has the Knight completing a hit 473 beyond reach when no
  other enemy is in sight; SPEC §14.2 cancels it (followed).

### D26 — Snapshot validation, debug-input clamps, packaging (SPEC §14.5, audit nits)
- `pr_state_check` validates restores (ValueError, state untouched); byte-flip fuzzing in C
  (ASan/UBSan) and Python. Debug coordinates clamped to +-128000.
- Packaging: `install_requires = numpy, gymnasium`; `[train]` extra = torch + PufferLib 3.0
  (from source with NO_OCEAN=1 and the D6 one-line fix); `PUFFERLIB_LICENSE` shipped;
  `MANIFEST.in` (absolute source paths are not added to sdists automatically); raylib via
  `tools/get_raylib.sh` or `RAYLIB_DIR`.
- `set_hand` makes the given deck the team's deck for later deals too (reset re-deals the
  current decks).

### D27 — Policy masking is stateless (audit nit, training correctness)
- **Decision:** every forward pass slices the mask from its own observations; `decode_actions`
  returns unmasked logits and `Recurrent` masks after the LSTM wrapper's forward (flattened row
  order = `observations.reshape(-1, OBS_SIZE)`); mismatches raise. Verified inside PuffeRL LSTM
  epochs.


---

## Builder decisions, Phase D (training & evaluation tooling, SPEC §15 / §15.7, 2026-09-26)

### D28 — LeagueVecEnv drives the native self-play env itself; one small read-only binding helper
- **Decision:** `LeagueVecEnv` owns a `Royale(num_agents=2)` and calls `binding.vec_step` /
  `vec_log` directly (the same three lines as `Royale.step`), writing the learner's action into row
  `2i + seat_i` and the opponent's into the other row, and returns only the learner rows. Match
  results are attributed per match with a new **read-only** binding function
  `env_log_peek(env_handle) -> dict` (one env's raw Log sums since the last `vec_log`, not
  cleared): on a terminal the wrapper diffs `win_0` / `win_1` against its baseline.
- **Why:** the env re-deals inside `c_step`, `vec_log` only returns an average over all envs, and
  with reward shaping the terminal reward is not ±1, so neither source identifies each match's
  result. The helper adds no state, changes no struct and no existing function (binding.c, "Phase D
  additions" block + one `MY_METHODS` entry).
- **Learner keys:** `learner_score` / `learner_return` / `learner_win` in the forwarded Royale log are
  replaced by the means over the same episodes from the learner's seat (§15.7.5); everything else is
  forwarded unchanged; `league/*` keys are added to every forwarded log.
- **Reverse if:** the env grows a per-episode result buffer (then read it instead of peeking).

### D29 — Opponent inference: grouped per spec, no grad, own RNG; bots re-created per episode
- **Decision:** each step the matches are grouped by opponent spec (insertion order = match order,
  deterministic); a policy group is one batched `forward_eval` under `torch.no_grad()` on the
  opponent rows' observations, actions sampled with the wrapper's CPU `torch.Generator`
  (argmax with `opponent_greedy`). Frozen checkpoints are loaded once (`load_policy`, sizes
  inferred from the tensor shapes, recurrent iff `lstm.*` keys) and dropped from the cache once they
  left the pool and no match plays them. Recurrent opponents (checkpoints and a recurrent `"self"`)
  keep a per-match `(h, c)` zeroed at every episode start. A `bot:*` opponent gets a fresh C bot per
  episode (`bot_new`, seed drawn from the wrapper RNG), so the heuristic's play-rate memory never
  leaks across matches.
- **Seeds:** seats and bot seeds come from a PCG64 stream seeded by `SeedSequence([seed, reset_seed])`
  (reset by `async_reset(seed)`), policy-opponent sampling from a torch generator from the same
  sequence; the pool has its own PCG64 stream. Verified: identical runs (bots + feed-forward and
  recurrent checkpoints + self) produce identical observations, rewards, seats and pool states.
- **Pool corner cases (not pinned by the SPEC):** `weights()` lists only specs with non-zero
  probability; a result against a checkpoint evicted while its match was running is ignored
  (its stats stay dropped); `add_snapshot` of a path that is already an anchor/snapshot is a no-op;
  with `pfsp_eps = 0` and all PFSP weights 0 the snapshot mass is spread uniformly.

### D30 — MMDPuffeRL copies PuffeRL 3.0's train() and adds three marked blocks
- **Decision:** `train()` is the 3.0 body statement for statement plus (1) `+ mmd_coef * KL` after the
  PPO loss, (2) `losses['mmd_kl'] = 0` when disabled, (3) the reference refresh after `epoch += 1`.
  The KL is the reverse KL(π_θ ‖ π_ref) between the masked categoricals over the 2305 actions
  (illegal entries excluded explicitly with `torch.where`, so it is exact and the gradient finite),
  averaged over the minibatch rows; π_ref runs under `no_grad` on the same minibatch (a fresh LSTM
  state for recurrent policies, as the learner's own training forward). With `mmd_coef = 0` no
  reference is kept and no extra computation or RNG draw happens, so losses and weights equal
  PuffeRL's exactly (tested bitwise to 1e-6).
- **Why a copy:** PuffeRL builds and back-propagates the loss inside `train()` with no hook; a
  gradient-injecting autograd function would avoid the copy but be far less readable.
- **Reverse if:** PufferLib exposes a loss hook (then subclass without copying).

### D31 — league_train.py: run-directory format, resume, CPU defaults
- **Decision:** `batch_size = num_envs × bptt_horizon` by default (one rollout of the horizon per
  epoch; `minibatch = min(1024, batch)`), so short CPU runs have many epochs (snapshots and PFSP
  actually happen in a 2k-step test). LR annealing is off by default (`--anneal-lr` to enable)
  because its schedule would restart at `--resume`. PuffeRL's own checkpoint writer is replaced by the
  run's `save()` (learner.pt, trainer_state.pt with the optimizer, league_state.json, the MMD
  reference). On `--resume`, pool-defining options and `--rnn` always come from the saved run
  (warning if the CLI differs); the resumed run reseeds matches with `seed + 7919·epoch` so it does
  not replay the start's match seeds (still deterministic). Evicted snapshot files stay on disk (the
  pool never touches the filesystem; they remain usable for tournaments).
- **Progress:** PuffeRL publishes an epoch's losses only when it logs (≥ 0.25 s apart); the script
  rewinds `last_log_time` each epoch so every epoch's losses reach `history.jsonl`.

### D32 — Meta-game tools
- **solve_zero_sum:** two LPs (row maximin, column minimax) with scipy HiGHS at 1e-10 feasibility
  tolerances; tiny negative LP noise is clipped and renormalised; `value = xᵀAy`. With multiple
  equilibria HiGHS returns a (deterministic) vertex — no max-entropy selection.
- **elo:** maximum-likelihood Bradley–Terry by damped, gauge-fixed Newton steps (exact recovery of
  generating ratings). The ML estimate does not exist when some group of agents never conceded a
  point to the rest (e.g. heuristic vs noop 100%); only then one virtual draw per pair is added
  (reported as `elo_regularised` by `tournament.py`), which keeps ratings finite and ordered.
- **play_match** runs through `Game` at the env's cadence (one decision per `frame_skip = 10` ticks;
  the action is queued and applied on the first tick, as in the env); bots use `Bot(kind,
  seed*2 + team)`; policies see `Game.obs(team)` (the env's encoder) and sample with a per-seat
  generator. **tournament.py** plays `--matches` per agent pair *and deck pairing* (all `decks ×
  decks` by default, `--deck-mode mirror` for X = X), alternating seats with the agents keeping
  their decks; policies are greedy unless `--sample` (play_match's default). **best_response.py**
  evaluates with sampled actions by default (the distribution it was trained against; `--greedy`
  to change) and keeps BR weights only with `--data-dir`.

### D33 — HPC templates target the NYU Torch cluster without having been run there
- `hpc/pufferroyale.def` (CUDA 12.4 devel base so PufferLib's CUDA advantage kernel compiles;
  deadsnakes Python 3.12; cu124 torch; PufferLib 3.0 @ 3b5c604 with `NO_OCEAN=1` and the D6 fix
  applied by `sed`, verified by `grep`), `train.sbatch`, `league.sbatch` (resumes when the run dir
  exists, `--requeue`; then a tournament of the newest snapshots), `hpc/README.md`. Account,
  partition and GPU type are placeholders (`YOUR_ACCOUNT`, `YOUR_GPU_PARTITION`); every script flag
  they use is in the scripts' `--help` (tested).

---

## Builder decisions, Phase E (card batch 2 + tower troops, SPEC §16 / v0.3-E.1, 2026-09-26)

### D34 — All 64 cards from one data-driven generator; hand-checked literals as the gate
- **Decision:** `tools/gen_card_db.py` registers the v0.1 cards and crown towers first (their unit /
  projectile / area / buff indices are unchanged), then the §16 cards, the units they spawn or
  death-spawn, and the tower troops. Every mechanic is a column of a generated row (spawner,
  death spawn, death bomb, variable damage, minimum range, dash, burrow, mana, attack sequence,
  Chef period, placement class); the engine never tests a card id. `self_check` asserts
  hand-checked level-11 literals for every §16 unit, bomb, projectile, area, buff and tower troop.
- **Why:** one reading of the data, and a data or rule drift fails `make gen` loudly.
- **Reverse if:** a card needs behaviour no column describes (then a named mechanic flag, still
  generated).

### D35 — Death bombs are timed effects; pulsing areas deal damage on their own events
- **Decision:** a death bomb is a `PR_FX_BOMB` effect committed in the next Spawn phase and fired
  after the bomb row's `DeployTime` (RoyaleSim `convert_death_bomb`), never an entity. Poison /
  Earthquake are `PR_FX_PULSE` effects: their per-second events deal the buff's damage to whoever
  is inside, and the slow is refreshed every area `HitSpeed` (250 / 100 ms) while the area lives
  (Earthquake's capped at the area's life).
- **Evidence:** SPEC §16.6.8 / §16.6.10-11 pin the event ticks and values exactly; the tester's
  "slow ends within 1 s of the area" holds.
- **Reverse if:** a measured trace shows damage after leaving the area (then per-entity DoT
  buffs with their own pulse clocks, RoyaleSim `status.BUFF_PULSE_TIMING`).

### D36 — Bandit dash and Miner burrow as small per-entity state machines
- **Decision:** `dash_state` (stand 800 ms -> two 250 half-steps a tick with the Range test ->
  hit + immunity through the arrival Resolve), and `burrow` (hidden, interpolated straight walk,
  surfacing in tick P + ceil(d/650)); both inside `PrState`, validated by `pr_state_check`.
- **Why:** the ledger's measured dash (`combat.DASH_ATTACK`) and the SPEC's exact Miner counts
  (§16.6.23) are reproducible to the tick with integer state and no heap.

### D37 — Dagger Duchess: the data sequence plus a 1000 ms reload; Royal Chef once per troop
- **Decision:** DD throws her 4-entry `AttackSequence` at the data's multipliers and reloads
  1000 ms (long-run 130 dmg/s vs the Princess's 136); the Chef serves the highest-cost troop within
  7500 every 28 s (SPEC §16.6.26), a troop at most once, instantly, spawned units ranked as cost 0.
- **Reverse if:** a measurement of the DD's ammo / reload or of the Chef's serving rule appears.

### D38 — v0.3 observation: integer card ids, 64 x 11 entities, 298 scalars; policy embeds ids
- **Decision:** exactly SPEC §16.4 / §16.6.16-17 (OBS_SIZE 17,707, down from 19,098). The policy
  embeds every id with one shared `nn.Embedding(129, 16)` (entity ids + the 9 scalar ids) and
  still masks from the observation slice. v0.2 checkpoints are incompatible (expected).

### D39 — MAX_ENTITIES stays 256
- **Options:** 256 or 512. **Decision:** 256 — random 64-card matches peak at ~36 live entities
  and the §16 large battle at 176; overflow drops and counts spawns (tested with 40 Witches). A
  512 pool would double the state (65 KB -> ~115 KB) and every snapshot / hash for no measured
  benefit, and the tester's soak asserts <= 256. `MAX_PENDING_SPAWNS` 64 -> 128.
- **Reverse if:** `spawn_overflow` becomes non-zero in training logs.

### D40 — Bots: kind-generic rules
- **Decision:** win condition / damage spell / placement predicates read card data (buildings-only,
  burrow, siege range, spell damage and hit layers), so every present and future card is handled.
  The heuristic still beats the random bot in every C-test game.

---

## Builder decisions, Phase F (LLM play interface, SPEC §17, 2026-09-26)

### D41 — The state text is built from the viewer's observation + the public board
- **Decision:** `render_state` reads only the viewer's own hand / queue / elixir from `Game.state()`;
  every opponent fact (cards seen, last four played, the deduced hand, the elixir upper bound) is
  decoded from the viewer's own observation vector (`Game.obs(team)`), whose no-leak property is
  already tested, and the rest comes from the public board (units, towers, spells in flight, crowns,
  tower troops). Tested by changing the opponent's hand order, queue, unrevealed deck and elixir with
  the debug API: the text is byte-identical (and changing the viewer's own elixir does change it).
- **Format:** own frame throughout; clock as time left (m:ss) with the phase; elixir floored to one
  decimal (never shows a card as affordable when it is not); units `Card @ (tx,ty) hp/max [flags]`
  sorted by id, consecutive identical lines merged as `Nx ...` (a Skeleton Army is one line), spawned
  units named by their card plus the unit when it differs (`Goblin Gang (SpearGoblin)`); legal tiles
  per slot as row runs with identical consecutive rows merged (`ty=18-23: tx 0-17`), so a full hand
  costs ~10 lines. The parse of that text reproduces the engine mask exactly (tested).
- **RULES_PROMPT** is generated at import from the compiled card data (cost, HP, damage, hit speed,
  range, targets, speed, lifetime, spell radius / crown damage) plus a hand-written one-line role per
  card, so it cannot drift from the engine and is identical on every call (prompt-cacheable).

### D42 — Match harness choices the SPEC leaves open
- Cadence (SPEC §17.4.10): `play_llm_match(decision_interval=None)` uses `LLMAgent.decision_interval`;
  an `LLMAgent` *opponent* acts at its own; bots every 10 ticks (the env's cadence); checkpoints at the
  frame_skip they were trained with (their run's config.json), else 10. `crowns` = `[agent, opponent]`.
  `decisions` counts model calls; `skip_idle=True` (off by default; `--skip-idle`) skips the call when
  the agent has no legal play. History (§17.4.7): the last k (state text, reply) pairs are embedded
  before the current state. `mock_random` (§17.4.8) is uniform over WAIT and every listed command.
- Error classes: `parse` (no command line, unknown card, card not in hand, slot outside 0-3),
  `illegal` (tile outside the arena or rejected by the mask, with the reason), `model` (the model
  function raised, or the adapter recorded an API error / refusal). All are WAITs.
- The Anthropic adapter reports per-call metadata through `fn.last_call` (stop reason, usage incl.
  cache reads, refusal category, API error), copied into each transcript entry. It is tested offline
  with a fake `anthropic` module and a fake client (request shape: cached system block, single user
  turn, `output_config.effort` except Haiku, beta fallbacks for Fable 5.1 / Opus 5; refusal and each
  exception class -> WAIT); the real SDK is never imported by the tests.
- `league.load_policy` now raises a clear `ValueError` for checkpoints that predate the v0.3
  observation (size mismatch), instead of a bare torch traceback.

---

## Builder decisions, second audit (SPEC v0.4.1 §18.6-18.7, 2026-09-27)

### D43 — One trainer class for every script; AMP without the weight-cast cache
- **Decision:** `train.py`, `league_train.py` and `best_response.py` all build `MMDPuffeRL` (with
  `mmd_coef = 0` it is PuffeRL's PPO bit for bit in float32). Its autocast context is built with
  `cache_enabled=False` and wraps only forward + loss (exited before backward); bf16 uses autocast on the
  configured device (cuda, cpu, mps). PuffeRL's context was entered and never exited, so autocast's
  weight-cast cache survived optimizer steps and every later forward reused the first bf16 cast of the
  weights (audit M2: KL(live || initial) = 0 after training). Its utilization thread is a daemon and
  `total_timesteps < batch_size` is rejected up front (all three scripts check before starting anything).
- **Resume (§18.7):** `MMDPuffeRL.load_training_state` restores the optimizer moments, then re-applies
  the invocation's learning rate / Adam betas / eps and puts the cosine schedule at the saved epoch
  (continued, not restarted). `--section.key` overrides of the original run are persisted and re-used.
  Turning MMD on at a resume sets the reference to the loaded learner (and warns about the coefficient
  change); the reference file is loaded only if the saved run already used MMD. `lv.episodes`, the
  wrapper RNG and the torch RNG are restored; in-progress matches are re-dealt (documented: deterministic,
  not bit-identical). Pool snapshot paths are stored relative to the run directory.
- **Failure handling:** losses and weights are checked every epoch BEFORE any snapshot or save; a
  non-finite epoch writes `nonfinite.json`, keeps the last good save and exits 1 (PuffeRL's own in-train
  checkpoint writes are disabled; `train.py` refuses to write a non-finite checkpoint). Any exception after
  start-up stops the monitor thread, closes the envs, saves nothing, and exits non-zero. Summaries are
  valid JSON (non-finite numbers as strings). `snap_*`, `learner.pt`, `mmd_reference.pt` and the state
  files are written atomically.
- **Evaluation uses the training env settings (audit L7):** `play_match` / `tournament.py` /
  `best_response.py` / `eval.py` read frame_skip, deploy lockout, tiebreak and tower troops from the
  checkpoint's run `config.json`: each checkpoint decides at its own frame_skip (bots at 10), the game
  uses the checkpoints' settings when they agree (defaults + a warning when they do not), and any
  checkpoint playing under other settings is warned about. `--frame-skip` / `--ignore-train-config`
  override. The best-response learner trains and is evaluated under its target's settings.
- **Smaller fixes:** `Recurrent` keeps the wrapped Policy's init (LSTMWrapper re-initialised it);
  checkpoint arguments accept `ckpt:` and run directories everywhere (`resolve_checkpoint_path`); decks
  accept comma-separated names or ids; tournaments drop repeated decks; `LeagueVecEnv.results` is a
  bounded deque drained each epoch; `bot_play_prob` is clamped; the wrapper's SeedSequence is injective
  for negative / huge seeds; unbuffered `np.take(mode="clip")`; `best_response --matches 0` is an error.


---

## Builder-3 decisions, second audit, engine items (SPEC §18 / v0.4.2, 2026-09-26)

### D44 — Snapshot validation: evolution-invariant bounds; unused content normalised, not rejected
- **Decision:** `pr_state_check` bounds everything the engine multiplies or accumulates by values the
  engine itself preserves. Examples: per-unit caps (max_hp, load, role timers); per-tick growth bounds
  (attack progress, charge, spent, plays); consistency of crowns and tower slots. A validated state
  therefore never evolves into an invalid or overflowing one (fuzz-checked after every tick). Pure
  statistics counters (`leaked`, `spawn_overflow`) saturate at 10^9 instead of being tick-bounded.
  Unused pool slots and stale pending plays are never read by the engine, so they are zeroed after
  validation instead of rejected.
- **Why:** the old "unused slots must be zero" rule rejected nearly every random byte flip, so fuzzing
  rarely exercised accepted states (the tester's §18.5 fuzz requires more than 50). Zeroing keeps the
  hash a function of live content only.
- **Reverse if:** snapshots must round-trip byte-for-byte even with garbage in unused slots.

### D45 — Seat-symmetry fixes are order rules, not id rules
- **Decision:**
  - Crown-tower ties go own-left (§18.1).
  - Chef towers are served own-left first (§18.2).
  - Bot tower scans run in own-frame order (§18.4).
  - The open-square occluder and the closed rolling-edge test are each their own rotation.
  - Dash immunity is decided when a hit is dealt (§18.3), so it cannot depend on whether the stun or
    the damage comes first.
- **Evidence:** the audit's bot-driven mirror fuzz found Chef divergences in 10 of 300 matches. It now
  reports 0 divergences over 1000 random matches and 300 matches per tower troop. A port of it
  (`tests/c/test_mirror.c`) fails 5 of 195 matches when the Chef order is reverted.

---

## Builder-1 decisions, training work package, env side (SPEC §19 / v0.5-G, 2026-10-02)

### D46 — Reward v2 lives in the C env, in double, with a step counter that never resets
- **Decision:** `royale.h` computes the potential, the anneal multiplier and `F` per `c_step`, stores
  `Phi_prev` per match and counts `c_step`s in an int64 that no reset touches (SPEC §19.1). `r_1` is
  `0.0f - r_0`, which is the exact negation and never produces -0, so all-zero weights stay
  bit-identical to v0.4. The v0.4 tower / crown difference shaping is gone (same keyword names).
- **Why:** the per-env counter works unchanged for in-process vectorisation, the Multiprocessing
  backend and `LeagueVecEnv` without any cross-process setter (TRAINING_PLAN §3.3).
- **Reverse if:** annealing must follow the learner's global step exactly across envs with different
  step rates (then pass a setter, and reach the Multiprocessing workers).

### D47 — `own_deck` appended after the tower troops; the policy picks it up through CARD_ID_SCALARS
- **Decision:** exactly SPEC §19.3 (offset 298, OBS_SIZE 17,715). Existing offsets are unchanged.
  Adding `own_deck` to `CARD_ID_SCALARS` makes the current `torch.Policy` embed it without code
  changes. Pre-v0.5 checkpoints no longer match the scalar layer, which is expected.

### D48 — One coarse-to-fine decoder in C, shared by the env and `Game.coarse_to_fine`
- **Decision:** `royale_coarse_to_fine` checks the tile-independent refusals once through
  `pr_check_play`, then tests each block tile with `pr_card_tile_legal_ctx` on one legality context.
  That is the code path of the fine mask, so the candidate set is the fine mask by construction, at
  one context build per action instead of up to 16. Grid-1 rows keep the v0.4 decode path.
- **Evidence:** exhaustive checks on bot-driven states: about 99k coarse-legal actions in C, all three
  grids and both teams, plus the Python brute force against `legal_mask`. Mutating the tie rule
  fails them.

### D49 — "Fine actions" as a per-row grid
- **Options:** a per-row fine flag (the SPEC's example), or a per-row grid. **Decision:** a per-row
  grid `row_grid[row]` in {1, 2, 4} (`Royale.set_row_grid`, `binding.env_set_row_grid`). Grid 1
  means fine actions. `LeagueVecEnv` sets the opponent row to 1 for `bot:` opponents and to the env's
  grid otherwise, at every episode start.
- **Why:** it costs the same as a flag, and it also lets a tool play a checkpoint whose grid differs
  from the env's (e.g. a g = 1 checkpoint as a league anchor next to a g = 2 learner), which SPEC
  §19.4 asks of the tools.

### D50 — The deck sampler is fixed-capacity C state with its own stream
- **Decision:** pool decks (ids + cumulative 2^-32 weight thresholds), held-out card-set masks and
  counters live inline in the env (256 / 1024 capacity, about 14 KB per match; no allocation).
  Python parses deck-set strings (`pufferroyale.decks`) and passes id lists in their installed card
  order: a preset keeps its own order, any other deck is ascending (ruling v0.5-G.1). The deal draws from
  `PR_DECK_STREAM`. Construction draws, then reseeds every stream, as v0.4 does for the game stream.
- **Why:** deals happen inside `c_step` (auto-reset), so the sampler must be in C to stay
  deterministic and cheap. Masks make the held-out test a 64-bit compare.

### D51 — Deck-set grammar choices
- **Decision:**
  - Empty items are skipped.
  - Whitespace is ignored around items and `:`, but `file:PATH` keeps spaces inside the path (the
    project path itself has spaces).
  - `random:N:SEED` needs N >= 1.
  - Python lists accept every string item plus `(deck, weight)` pairs, so `parse_deck_set` is
    idempotent.
  - `deck_key` is a string (`"2-4-9-..."`), usable as a JSON key.
  - `random_decks` is a prefix-stable numpy PCG64 draw.
- **Reverse if:** the tester or orchestrator wants a stricter grammar (e.g. rejecting a trailing `;`).

### D52 — Per-card play rates are counted in Python from the observations
- **Decision:** `royale.CardStats` updates vectorised numpy counters from each decision's observation
  and action. `Royale.step` counts every row (before `vec_step`); `LeagueVecEnv.send` counts the
  learner rows. The counters are emitted into the log dict.
- **Why:** the C `Log` is averaged over episodes by `vec_log`, while play rates are ratios of decision
  counts. Putting 128 counters into the Log would distort both. The update costs a few small numpy
  operations per step.

## Builder-2 decisions, training work package, trainer / policy / tools side (SPEC §19 / v0.5-G.2, 2026-10-03)

(D53–D59 are not used.)

### D60 — Conditional head: a card MLP plus a full-resolution FiLM board map
- **Decision:** (a) `wait_logit = Linear(h, 1)`; slot `s` = `slot_logit(ReLU(Linear([h, enc(hand_s)])))`,
  one MLP shared by the four slots. (b) `pos_board` = three 3x3 convs at 32 x 18 with dilations
  1, 2, 4 (receptive field 15 tiles, `pos_channels` wide). Per slot it is modulated by FiLM
  `ReLU(board * (1 + γ_s) + β_s)`, with `(γ_s, β_s) = Linear(cond_s)`, where `cond_s` is the slot MLP's
  hidden vector. A 1x1 conv then gives the 32 x 18 logit map. For g > 1 a fixed (576, B) matrix
  averages each block over the tiles it really has. (c) Both factors are `log_softmax` over
  masked logits (`finfo.min`), in float32: card over {wait, slot 0..3} with a slot masked iff its
  segment has no legal action, and position per slot. Then `joint = log P(s) + log P(j | s)`, and
  illegal entries are set to `finfo.min` again. `wait_logit`, `slot_logit`, `film` and `pos_out`
  are initialised with std 0.01, so the head starts near uniform per factor and FiLM starts near
  the identity. The flat head is the v0.4 `actor` unchanged. `decode_actions(hidden)` without the
  observations returns the flat head's unmasked logits (the bare PufferLib protocol), and the
  conditional head raises there instead of guessing.
- **Why:** FiLM conditions four slot maps on one shared board pass, which costs much less than
  concatenating per-slot condition planes. Full resolution keeps a tile-exact map for g = 1. The
  head has no grid-dependent parameter, so the grid is a persistent `action_grid` buffer (D63).
- **Reverse if:** profiling on the GPU shows the board pass dominates; then lower `pos_channels`
  or pool before the 1x1 conv.

### D61 — Recurrent: the wrapper's forward copied, with the observations passed to the decoder
- **Decision:** `Recurrent.forward` / `forward_eval` copy PufferLib 3.0's `LSTMWrapper` methods
  statement for statement. The one change is `decode_actions(hidden, observations)`, which passes
  the observations of exactly the rows being decoded. The flattened `(B*T)` order of the LSTM
  output equals `observations.reshape(-1, OBS_SIZE)`. Nothing is stored on the module between
  calls. `LSTMWrapper.__init__` re-initialises every parameter it can see, so `Recurrent` saves
  the wrapped Policy's state dict first and restores it afterwards. The wrapper still makes its
  random draws, so the global RNG stream is unchanged.
- **Why:** the conditional head needs the board planes, the hand and the mask of the same rows,
  and the wrapper only forwards the LSTM output (SPEC §19.6 [IMPL-DEFINED]).
- **Reverse if:** PufferLib gains a decoder hook that carries the observations.

### D62 — Card-stat table details
- **Decision:** the columns are those of SPEC §19.6 / §19.9.7 (`CARD_STAT_NAMES`), with these
  details:
  - Raw values come from `binding.card_info`. A missing, `None` or non-numeric field is 0, and
    booleans are 0/1.
  - DPS = `damage * 1000 / hit_speed_ms`, or 0 when the hit speed is 0.
  - Unit splash = `max(area_damage_radius_milli, projectile.radius)`.
  - Flying = `flying_height > 0`, charges = `charge_range > 0`, and "spawns units" =
    `spawner or death_spawn`.
  - Negative values are clamped to 0 (no card has one).
  - `log1p` is applied to hitpoints, damage, DPS and death damage before scaling.
  - Every column is divided by its maximum over the 64 cards. An all-zero column stays 0.
  
  The float64 table becomes a float32 non-persistent buffer. `card_stat_proj = Linear(K,
  card_dim)` (orthogonal init, gain 1) and the embedding are concatenated, so `enc_dim =
  2 * card_dim`. `enc` is used for the 17 card-id scalars, the entity ids and the conditional
  head's hand cards. `card_stats` accepts 0/1, booleans and "true"/"false".
- **Why:** this is a deterministic, fixed prior that also covers cards an agent rarely sees.
  Because it is not in the state dict, a checkpoint never carries a stale table.

### D63 — Checkpoints carry their architecture and grid; every tool plays them on their own grid
- **Decision:** `policy_kwargs_from_state_dict` reads the following from the tensors:
  - the sizes (`hidden_size` from `value.weight`, so both heads work);
  - head = flat iff `actor.*` is present;
  - `card_stats` iff `card_stat_proj.*` is present;
  - `pos_channels` from `pos_out.weight`;
  - the grid from the `action_grid` buffer (else from the flat actor's width, else 1).
  
  Loading a state dict of another grid raises. `league.load_weights` (`--init-from`) requires the
  whole architecture to be equal: the sizes, head, card stats, pos channels, grid, and recurrence
  with its LSTM sizes. The error names each difference. At every episode start `LeagueVecEnv` sets
  both rows' grids through D49's per-row grid: the learner and self-play use the env grid, a bot
  uses 1, and a `ckpt:` opponent uses its own grid. `eval.py` / `watch.py` / `tournament.py` /
  `metagame` play a checkpoint on the grid its weights carry. When `config.json` disagrees,
  `eval.py` warns and uses the weights' grid.
- **Why:** the weights are the ground truth, and a grid mismatch would silently misread every
  action.

### D64 — Early stop and the stage ladder
- **Decision:** `league_train.py` keeps one deque per anchor of the last N outcomes (+1/0/-1).
  It stores them in `league_state.json` (`early_stop.recent`), so a resume keeps them. The test
  runs only at snapshot epochs, after the snapshot and before the save. The save, the
  `history.jsonl` record and the summary then carry `early_stopped`. The run finishes normally
  (final model; wandb `early_stop = true`). An early-stopped run is finished, so a later
  `--resume` reports `nothing_to_do`. `--early-stop-score` without an anchor, or with
  `--snapshot-interval 0`, is an error. `stages.py` works as follows:
  - Each rung is a new run `<P>_<i>_<anchor kind>` with an absolute budget.
  - The flags it sets per rung are refused when forwarded.
  - A rung whose directory already holds a run is resumed (preemption).
  - Each gate is recomputed from `history.jsonl`'s per-match outcomes, not from the summary, so a
    rung that ran to its budget can still pass.
  - Rung i > 0 starts from rung i-1's newest `model_*.pt`.
  - The exit code is 0 (all passed), 1 (a gate failed) or 2 (a rung's process failed).
- **Why:** with the outcomes persisted, a preempted ladder behaves like an uninterrupted one. A
  gate read from the run's own record cannot disagree with what the run logged.

### D65 — Entropy split
- **Decision:** `entropy_split` works from `log_softmax` and `logsumexp`: `log P(s)` is the
  logsumexp of the slot's joint log-probs, and `log p(j|s)` is the difference. Every `p log p`
  term is guarded by `p > 0`, so it stays finite with `finfo.min` logits and with wait-only rows.
  When the split is unset it runs on detached logits under `no_grad` and uses no RNG, so `train()`
  stays identical to the base class. `losses/entropy_card` / `entropy_pos` are logged in both
  modes. Any negative coefficient means unset. Exactly one coefficient set, or a non-finite one,
  is a `ValueError` at construction.

### D66 — `reward_clip` through a copied `evaluate()`
- **Decision:** `MMDPuffeRL.evaluate()` is PufferLib 3.0's `PuffeRL.evaluate()` (commit
  `3b5c604`) copied with one marked block: clamp to `[-c, c]` when `c > 0`, no clamp when
  `c <= 0`. `None` means 1.0, and a non-finite value raises.
- **Why:** the base hard-codes `clamp(-1, 1)` inside `evaluate()` with no hook. Potential-based
  shaping can exceed 1, and the copy keeps every other statement (RNG, buffers) identical.

### D67 — `train.py --resume` / `--init-from`
- **Decision:** `--resume` takes a `pufferroyale_<run_id>` directory that holds `config.json` +
  `trainer_state.pt`. It loads the save named in the trainer state: the weights, optimizer
  (`load_training_state`), epoch, global step and the torch RNG state. `train.py` adds that RNG
  state to `trainer_state.pt` at every save. The run's `config.json` settings are reused, and
  flags given again override them. The architecture always comes from the run. Matches in progress
  are re-dealt from `seed + 7919 * epoch` (as in the league), and a fixed-id logger keeps the run
  directory. `shaping_step_offset = global_step // R`. `--init-from` is `league.load_weights`
  (D63) into a new run and cannot be combined with `--resume`.

### D68 — Our own wandb logger
- **Decision:** `pufferroyale.trainer.WandbLogger` has PufferLib's `WandbLogger` interface
  (`run_id`, `log(logs, step)`, `upload_model`, `close(model_path, early_stop)`, plus `abort()`
  for error paths). It calls `wandb.init(project, group, tags, config, resume="allow",
  settings=Settings(console="off"))` and passes `id` only when resuming. `train.py` passes the
  `pufferroyale_<run_id>` id; `league_train.py` passes the `wandb_id` saved in `league_state.json`.
  Both scripts use it. wandb is imported only inside it, so without `--wandb` nothing imports
  wandb. The league logs one call per `history.jsonl` record at `step = global_step`.
- **Why:** PufferLib 3.0's class calls `wandb.util.generate_id()`, which recent wandb (0.30)
  removed, and we never edit the installed PufferLib. Letting `wandb.init` choose the id works
  across versions.

### D69 — `eval.py` results and JSON
- **Decision:** results come from the match outcomes. `eval.py` diffs each env's cumulative
  `env_log_peek` win counters and never calls `vec_log`. Each env plays a fixed quota of full
  matches. A row has `wins`, `draws`, `losses`, `matches`, `score` and `ci95`, the Wilson interval
  (`metagame.wilson_interval`, p̂ = score, n = matches; `[0, 1]` for n = 0). With `--decks` each
  deck is a mirror in its installed order (a preset keeps its order). Its label is the preset name,
  else `deck_key`. Every deck uses the same seeds (`seed + 17 * seat`), which makes the per-deck
  rows paired comparisons. The pooled rows follow the per-deck rows. The output is strict JSON
  (`json_safe`).

## Builder-3 decisions, audit amendments (SPEC §19.10 / v0.5-G.3, 2026-10-03)

### D70 — The anneal length N is a recorded property of the run
- **Decision:** `trainer.shaping_anneal_steps(F, T, R)` computes `ceil(Fraction(str(F)) * T / R)`.
  So F = 0.07, T = 3e8, R = 3000 gives 7000, where the float product gave 7001.
  `shaping_env_kwargs(..., anneal_steps=N)` passes a recorded N through unchanged; the offset is
  still `global_step // R`.
  - `league_train.py` keeps N in the run's args as `shaping_anneal_steps`, so `league_state.json`
    and `config.json` `league` both carry it. A new run computes it. A `--resume` keeps it unless
    `--shaping-anneal-frac` is given again; then it is recomputed from the new F and the new
    absolute total.
  - `train.py` records N as `config.json` `train.shaping_anneal_steps`. "Given again" means
    `--shaping-anneal-frac` or `--train.shaping-anneal-frac` on the command line.
  - `best_response.py` (always a new run) records N in its `config.json` `league` args.
  - A run saved before N was recorded keeps the N its last invocation used. For the league that
    is `ceil(F * saved total / saved num_envs)`; for `train.py` it is `config.json`
    `env.shaping_anneal_steps`.
- **Why:** with N recomputed from a grown `--total-timesteps`, a resume would stretch an anneal
  that is already under way. Exact rationals make N independent of float rounding.

### D71 — The shaping / clamp warning
- **Decision:** `trainer.warn_shaping_clip(env_kwargs, reward_clip, prog)` prints one stderr line
  `[<prog>] warning: reward shaping is on (<weights>) while train.reward_clip = c > 0 ...` when
  any of `reward_tower`, `reward_crown`, `reward_elixir` or `reward_play` is > 0 and c > 0. Each
  script calls it once per process: after the env kwargs are final (`train.py`), and after the
  trainer is built (`league_train.py`, `best_response.py`). A run that has nothing left to do
  (league `--resume` at its target) does not warn.

### D72 — League history after a preemption
- **Decision:** on every `--resume`, a "nothing to do" one included, `history.jsonl` is rewritten
  atomically. It keeps the records with `epoch <= league_state.json["epoch"]` and drops later
  records and unparsable lines (a torn last line). This happens before the build, so a resume that
  then fails (for example D77) has already dropped records that every later resume would drop too. A new run in
  a directory holding `history.jsonl` but no `league_state.json` renames it to
  `history.jsonl.stale-<k>`, with the smallest free k >= 1. Other leftovers (`snap_*.pt`) are not
  touched. Both actions add a note to the summary. The early-stop window was already written by
  every `save()`, the final one in `finish()` included (D64), so nothing changed there.

### D73 — `stages.py` gates on the persisted window and runs in the caller's directory
- **Decision:** this replaces D64's "gate recomputed from `history.jsonl`". `gate(run_dir,
  anchor, N)` reads `league_state.json` `early_stop.recent[anchor]` and scores its last <= N
  outcomes. The lookup is exact; a `ckpt:` anchor also matches after both paths are made absolute.
  A missing state or anchor gives `(None, 0)`, which fails the gate as insufficient matches.
  `league_train.py` is started by its absolute path, with no `cwd`, so it inherits the caller's
  working directory. `--data-dir`, `--init-from` and `--resume` are already absolute.
- **Why:** `history.jsonl` can hold epochs a preempted process wrote after its last save. The
  saved window is exactly what the run's own early-stop test saw.

### D74 — Recorded deck sets and the weight cap
- **Decision:** `decks.expanded_deck_sets(env)` returns `deck_pool_decks` (`deck_entries`
  order, so a preset keeps its own order, as installed) and `heldout_decks_decks` (ascending,
  which is how `Royale` installs held-out decks), each a list of `[cards, weight]`. Both are
  always written, as `[]` when empty, into `config.json` `env` by `train.py`, `league_train.py`
  and `best_response.py`. They are records only. `train.py --resume` never passes them, or the
  derived reward keys, back to the env (`DERIVED_ENV_KEYS`), and the spec strings stay the
  source. `decks._weight` enforces `0 < w <= MAX_WEIGHT = 1e9`, and every weight path (string,
  list, pair, dict, file) goes through it, so `Royale` and the C binding never see a larger one.

### D75 — The conditional head's position factor in float32 under autocast
- **Decision:** `Policy._conditional` runs the block pooling matmul, the slot masking, the position
  log-softmax and the joint assembly inside `torch.autocast(device_type=<device>, enabled=False)`.
  The input (the 1x1 conv output) is cast with `.float()`. The card factor was already float32
  (`.float()` before masking). Outside autocast the context is a no-op, so float32 results are
  bit-identical. The flat head keeps v0.4's `finfo(logits.dtype).min` masking. Its bf16 logits
  are finite and were not part of the finding.

### D76 — wandb step metric
- **Decision:** `WandbLogger.__init__` declares `define_metric("global_step")` and
  `define_metric("*", step_metric="global_step")`. `log(logs, step)` logs `{**logs,
  "global_step": step}` without wandb's `step=`, so wandb's own counter keeps increasing and
  records of re-done epochs are kept. wandb is still imported only inside the logger.

### D77 — Clean `--resume` architecture errors
- **Decision:** `league.architecture_diff(policy, state_dict)` lists the differences (sizes, head,
  card stats, pos channels, grid, LSTM). `load_weights` now uses it as well.
  - `league_train.py` checks the built learner against `learner.pt` in `_restore`. Any
    difference, or a `RuntimeError` from the strict load, raises `SystemExit("--resume <dir>:
    this invocation builds a different policy than the saved run (...)")`. The constructor's
    `abort()` then releases the envs and threads.
  - `train.py` makes the same check right after building the vecenv and policy, before the
    trainer exists. It also turns a `RuntimeError` from the later load into the same exit.
  - A corrupt file still fails in `torch.load`, as before.
  - `train.py` takes `policy`, `rnn_name` and `rnn` from the run, so in practice only
    `--env.placement-grid` can differ there. The league takes `--rnn` from the run (POOL_KEYS),
    but `--policy.*`, `--rnn.*` and `--env.placement-grid` overrides can differ.

## Builder-4 decisions, sampling by default (SPEC §19.11 / v0.5-G.5, 2026-10-03)

### D78 — The card-first greedy rule and sampling defaults
- **Decision:** `league.greedy_actions(logits)` is the only greedy choice. `select_actions(...,
  greedy=True)` calls it, so league `--opponent-greedy`, `eval.py --greedy`, `best_response.py
  --greedy`, `tournament.py --greedy`, `watch.py --greedy`, `metagame` (`_PolicyPlayer`,
  `play_match`, `deck_metagame`) and the LLM-match policy opponents all use it. `eval.py` and
  `watch.py` no longer take their own argmax.
  - It takes torch or numpy logits of any dtype and batch shape `(..., 1 + 4B)`, and returns int32
    of shape `logits.shape[:-1]` (0-d for one row). Other widths raise `ValueError`.
  - The math is float64 numpy: `exp(x - row max)` over the legal entries, summed per segment.
    These are the softmax marginals up to one row constant, so their order and ties are the same.
    `argmax` takes the first maximum, which gives the tie orders of the spec (wait, slot 0..3;
    smallest `j`).
  - An entry is illegal when it is `<= max(finfo(input dtype).min, finfo(float32).min)`; NaN is
    illegal too. So float32-masked logits passed as float64 still read as masked, and bf16
    logits work. Illegal entries get probability 0 and are never chosen. A row with no legal
    entry gives 0. The env never makes one, since wait is always legal.
  - Sampling is unchanged (`torch.multinomial` on `softmax(logits.float())`). `eval.py` now
    calls `select_actions`, which draws the same samples from the same generator as before.
- **Defaults:** `play_match`, `deck_metagame` and `play_llm_match` default to `greedy=False`.
  - `tournament.py` samples. `--greedy` switches, and `--sample` is a no-op kept for old
    command lines (with both flags, greedy). The JSON `greedy` field is `false` by default. The
    summary line names the mode.
  - `watch.py` samples with `torch.Generator().manual_seed(2 * seed)` (team 0's stream in the
    `metagame` convention), so a given `--seed` replays the match. `--greedy` switches. The last
    line keeps `policy (team 0) won/lost/drew` and adds `(<n> plays by team 0, <mode> actions)`.
  - `llm_match.py` gains `--opponent-greedy` and records `opponent_greedy` in its summary.
  - `hpc/league.sbatch` never passed `--sample`, so its end-of-job tournament now samples.
- **Why:** with the joint card × tile action the probability of a card is spread over many
  tiles, so the plain argmax nearly always waits. The CPU learning-check policy won 0/20 greedy
  matches vs `bot:random` with the argmax, and 18/20 with the card-first rule (20/20 sampled).
  Over 4 matches the argmax chose a card 3 times in 1,440 decisions; the card-first rule chose
  one 187 times in 1,471.

---

## HPC agent decisions (NYU Torch, docs/HPC_HANDOFF.md, 2026-10-03)

### D79 — Isolated Torch environment: Path A image, binds limited to $PR_ROOT, 75% GPU floor
- **Path A (our Apptainer image).** `--fakeroot` builds work on Torch compute nodes (checked: an image
  of the pre-v0.5 definition built there on 2026-09-27). The hosts (RHEL 10) have no gcc and no
  CUDA/Python modules, so Path C is impractical. Path B, PufferDrive's pattern (a shared
  `/scratch/$USER` miniforge + venv + overlay), would mean sharing tooling locations with
  PufferDrive. Image: `$PR_ROOT/containers/pufferroyale.sif`, built by `hpc/build_image.sbatch` in
  the job's node-local temp directory.
- **Two cluster-only build fixes in `hpc/pufferroyale.def`:**
  - `%setup` creates the site bind points (`/scratch`, `/projects`, `/share/apps`, `/mnt`,
    `/state/partition1`). The writable build sandbox cannot create bind destinations, and `%post`
    failed on `/scratch` on 2026-09-27 even with `APPTAINER_BINDPATH` unset;
  - apt's `_apt` sandbox is off: with no `/etc/subuid` entry, `--fakeroot` uses a root-mapped
    namespace. Nothing else changed: torch is still the unpinned newest cu124 wheel (recorded per
    run).
- **Isolation.**
  - `hpc/env.sh` (installed as `$PR_ROOT/env.sh`) unsets the login profile's
    `APPTAINER_BINDPATH=/scratch,/projects,/share/apps,/mnt,/state/partition1`. Every `apptainer exec`
    of ours binds only `$PR_ROOT` and uses `--no-home`, so `$PD_REF` is not even visible inside our
    containers.
  - `hpc/isolation_check.py` (run in every job) adds two checks to HANDOFF §4d: `PYTHONPATH` is empty,
    and the imported `pufferroyale` is the job's checkout, not the image's `/opt/pufferroyale`.
- **wandb:** the key is copied (read-only) from `~/.netrc` into `$PR_ROOT/secrets/wandb_key` (mode
  600, directory 700) at the user's request. `env.sh` exports it as `WANDB_API_KEY`, plus
  `WANDB_ENTITY=<netid>` and `WANDB_PROJECT=pufferroyale`, with every wandb directory under `$PR_ROOT`.
- **GPU utilisation floor (user's instructions, 2026-10-03).**
  - **What:** our GPU jobs keep the device at **≥ 75% utilization at all times**, by running
    `hpc/gpu_heartbeat.py` next to the job.
  - **Why:** Torch reclaims GPUs whose jobs stay at low utilization; the user's PufferDrive runs a
    heartbeat for this reason. Pre-v0.5 calibration showed `league_train.py` at 2–13%.
  - **How it works:** an integral controller on NVML device utilization (setpoint 80%), small
    2048² matmuls, at most 95% of the time, nice 19. `HEARTBEAT=0` disables it.
  - **Stacking:** several runs may share one GPU whenever that gives more total training throughput
    (user: "whatever might lead to faster training overall"). Phase 3 measures 1/2/4 runs per GPU
    with the heartbeat running. The user also set **one GPU per job**: no multi-GPU jobs.
- **Accounts and GPU types:** `torch_pr_355_tandon_priority` / `_advanced`, `--gres=gpu:1
  --constraint=a100|h100|h200`, and no `--partition` (Torch's job-submit filter routes). PufferDrive's
  compute configs use the same accounts and GPU types. Concurrency: start with ≤ 4 GPUs, more as
  needed (user).
- **Reverse if:** Torch publishes a utilization policy that makes the floor unnecessary, or a
  stacked layout keeps real utilization above it on its own.

### D80 — Two Linux fixes found by the first G0 attempt: `-lm`, and the site's BASH_ENV
- **`make test-c` did not link on Linux.** `tests/c/test_wp_env.c` (from the work package) calls `pow()`.
  macOS links libm implicitly; Linux does not (`undefined reference to 'pow'`, CPU G0 job 19136106).
  **Fix:** the Makefile appends `$(LDLIBS)` (default `-lm`) to every C test/bench link line. It changes
  build tooling only: no engine code and no test changed. A HANDOFF §5 portability fix; the golden
  hashes are the proof. **To merge on the Mac side.**
- **Our isolation was being undone.** Torch's site profile `/share/apps/profile.d/nyu.sh` exports
  `BASH_ENV=/etc/profile.d/cluster.sh`, so every non-interactive bash re-sources it. That includes
  the bash launcher behind the `apptainer` command (`/share/apps/apptainer/bin/apptainer`), and it
  re-exported `APPTAINER_BINDPATH=/scratch,/projects,/share/apps,/mnt,/state/partition1` and
  `APPTAINER_CACHEDIR`/`SINGULARITY_CACHEDIR=/scratch/$USER/...-cache` after `env.sh` had changed
  them.
  - **Effects:** `/scratch` (incl. `$PD_REF`) was visible inside our containers (only reported by the
    isolation check; nothing was used from it). The first image build (job 19133793) also wrote its
    Docker layers to the user's general `/scratch/$USER/apptainer-cache` instead of
    `$PR_ROOT/cache/apptainer`.
  - **Fix:** `env.sh` unsets `BASH_ENV` (and `ENV`) for our processes only; no dotfile or site file is
    touched.
  - **Verified on a compute node:** `PD_REF visible in container: False`, Apptainer no longer receives
    the site cache or bind variables, and `make test-c` gives 184,647 checks, 0 failures.
  - It also explains why unsetting `APPTAINER_BINDPATH` "did not work" on 2026-09-27 (D79's `%setup`
    bind points stay as a harmless safety net).

### D81 — G0 passed with four explained, platform-only deviations (user's decision, 2026-10-04)
CPU part of G0 on a Xeon Platinum 8592+ node (job 19136476, commit 50999ea):
- **Passed:** golden hashes (all 6 matches bit-identical to the macOS reference), the isolation
  check, `make test-c` (184,647 checks, 0 failures), `make asan` (184,431 checks, clean).
- **pytest:** 1,867 passed, 4 failed. The user decided to treat G0 as passed and to continue. The
  tests and thresholds are untouched; fixes go to the Mac side.
  1. `tests/spec/test_v05_policy.py::test_recurrent_train_and_eval_paths[flat-1]` and
     `[conditional-1]` require bitwise-equal logits between the training forward (autograd on) and
     a `no_grad` forward of `Recurrent`.
     - **What differs:** on x86 CPU they differ by at most 5.2e-10 (flat) and 4.8e-7 (conditional).
     - **Why:** with `torch.backends.mkldnn.enabled = False` (oneDNN off) both are bitwise equal, and
       two `no_grad` forwards are always bitwise equal with identical masks. So no state persists:
       the difference is oneDNN choosing different CPU kernels for the two paths.
     - **Impact on training:** none (it runs on CUDA).
     - **Suggested fix (tester / SPEC process):** compare with `torch.allclose(atol=1e-5)`, or disable
       mkldnn in that test.
  2. `tests/builder/test_wp_trainer.py::test_reward_clip_values` assumes the clip-0, clip −1 and
     clip-0.002 runs see the same rollouts for 4 epochs, but each run trains between rollouts on
     differently scaled rewards.
     - **What differs:** epochs 0–2 matched exactly (clamp exact; c ≤ 0 unchanged). In epoch 3, 4 of
       64 rewards differ with opposite signs, i.e. different actions were sampled after 3 updates.
     - **Suggested fix:** check only the rollout before the first `train()`, or use learning rate 0.
  3. `tests/spec/test_nonfunctional.py::test_e2e_check_prints_pass`: e2e 19/21. One failing check is
     pytest (item 2); the other is item 4.
  4. The e2e `perf` stress check: 6,467 ticks/s against the 10,000 threshold set on the M4.
     - On these CPUs the cold-start stress window runs at 6–9k ticks/s; with a warm-up it runs at
       18–20k (EXPERIMENTS, pre-v0.5 notes).
     - **Suggested fix:** a warm-up as in the spec test, or a per-platform threshold.
- **Non-gating `make asan-py`:** 1,797 passed and 7 failed, with no sanitizer report from our extension.
  - Three failures are items 1–2 above.
  - Four are new v0.5 subprocess tests the Makefile's ASan skip list doesn't know yet
    (`test_wp_audit.py::test_stages_after_a_preemption_and_from_another_cwd`,
    `test_v05_tools.py::test_stages_relative_paths_are_the_callers`, and `test_plot_history` in
    `test_wp_tools.py` and `test_v05_tools.py`):
    - the `stages` children inherit the relative `PUFFERROYALE_BINDING_DIR=build/debug/pufferroyale` in
      another working directory and cannot import the binding;
    - the `plot_history` children run matplotlib under the preloaded ASan runtime.
  - **Suggested fix (Mac side):** add them to `ASAN_PY_SKIP_SUBPROC`.
- **Reverse if:** the GPU node's run of G0 shows anything else, or the Mac team decides one of
  these is a real defect.

### D82 — GPU training runs with non-deterministic cuDNN (`--train.torch-deterministic=`)
- **Finding (H100, 2026-10-04).** The v0.5 league trained at only ~1,160–1,190 learner steps/s
  (calibration job 19134430, any num_envs and batch), 4–5× below the pre-v0.5 league on the same GPU.
  - **Where the time went:** a cProfile of a short stage-1-shaped run spent 15.9 of 37.7 s in
    `torch.conv2d` (~17 ms per call), versus 2.9 s in the C env.
  - **Micro-benchmark** (default conditional policy, `forward_eval` of 256 rows, H100, heartbeat on):
    1.6 ms with cuDNN's default algorithms, **73 ms with `cudnn.deterministic=True`**, 2.7 ms with
    `benchmark=True`, 95 ms with cuDNN off, 202 ms on 16 CPU threads.
  - **Cause:** `royale.ini [train] torch_deterministic = True`, which PuffeRL turns into
    `torch.backends.cudnn.deterministic = True`.
- **Decision:** every GPU training run passes `--train.torch-deterministic=` in `EXTRA_ARGS`, under the
  user's "whatever might lead to faster training overall" (2026-10-03). This is a run flag only: no
  code or config file changes. `hpc/calibrate.py` uses it by default (`--deterministic` for the
  reference mode).
  - The value must be EMPTY: PufferLib's config parser builds boolean options with `type=bool`, so
    `False` and `0` are non-empty strings and parse as **True** (checked; it affects every boolean
    `--train.*` / `--env.*` override, e.g. `--train.anneal-lr`). **For the Mac side:** consider
    a proper bool parser.
- **Cost:** GPU training is no longer bit-for-bit reproducible run to run. Training is still seeded,
  league resumes were already "deterministic, not bit-identical", and every determinism test (CPU)
  is unaffected. Evaluation tools (`eval.py`, `tournament.py`, `best_response.py`) also run under
  PuffeRL / torch defaults: the tournament and `eval.py` do not set deterministic cuDNN, while
  `best_response.py` trains through PuffeRL and needs the same flag.
- **Reverse if:** exact GPU reproducibility becomes a requirement (then accept ~5× slower runs, or
  ask the Mac team for a deterministic-but-fast decoder).

### D83 — wandb entity `<netid>-new-york-university`
- **What failed:** the user asked for wandb "on my <netid> account (not emerge_org)". From a compute
  node (2026-10-04), `wandb.init(entity="<netid>")` fails with `CommError: entity <netid> not found
  during upsertBucket`. The bare username entity exists (GraphQL: personal, not a team), but it cannot
  hold new projects.
- **What we use:** the account's personal workspace, `<netid>-new-york-university` (a team in the
  user's own organisation, holding their other personal projects). Online logging works there:
  `https://wandb.ai/<netid>-new-york-university/pufferroyale` (test run `compute-node-connectivity`,
  group `setup-checks`). `env.sh` sets `WANDB_ENTITY` to it; the default entity `emerge_` is never
  used.
- **Network:** compute nodes reach wandb directly; no TLS workaround is needed (PufferDrive's
  `WANDB_INSECURE_DISABLE_SSL` is not adopted).

### D84 — Stage-1 sizes: league defaults (batch = num_envs × 64, minibatch 1,024), not a throughput-picked minibatch
- **Evidence.** The first §6b smoke (job 19150297: 512 envs, batch 65,536, minibatch 8,192) was still at
  0.25 vs `bot:random` after 1.1M learner steps.
  - Training was healthy: explained variance 0.17 → 0.57, losses finite.
  - But `approx_kl` stayed at ~0.0003 and `clipfrac` near 0: few, small updates.
  - That configuration makes one gradient step per 8,192 learner steps. The Mac CPU learning check
    (TRAINING_PLAN §7: batch 2,048, minibatch 1,024) made one per 1,024 and passed 0.9 at ~0.4M steps,
    i.e. after ~390 updates. At equal updates the two curves agree so far (~0.25–0.3 after ~136).
- **Throughput does not decide it.** Phase 3 shows SPS is set by env stepping: minibatch 4,096 vs
  16,384 gives the same SPS. So HANDOFF §6's minibatch menu (4,096–16,384) only trades learning per
  step for nothing.
- **Decision.** Stage 1 uses `league_train.py`'s documented defaults (DECISIONS D31): batch = num_envs ×
  bptt_horizon (256 × 64 = 16,384) and minibatch = `royale.ini`'s 1,024. That is TRAINING_PLAN §8.1's "royale.ini
  hyperparameters" and the Mac check's update density. It also keeps GPU memory small: ~1 GB of
  activations, so stacking is limited by CPU and GPU time, not memory.
  - Confirmed by a second smoke with exactly these sizes (job 19150430) before Stage 1 is launched.
- **Snapshot interval 200 epochs** (3.3M learner steps, ~10 min at 5.4k SPS) for Stage 1:
  - stage-1 snapshots only make resume possible (their pool mass is 0);
  - 5.2 MB each, interval 50 would write ~366 per heuristic rung (~34 GB over 18 ladders);
  - early stopping is then checked every 3.3M steps. `hpc/stage1_report.py` measures steps to
    G2 from the match outcomes, not the snapshot grid.
- **Note for the plan.** Batch and minibatch are learning hyperparameters here, not only throughput
  settings. HANDOFF §6/§8 should say so; the Stage-3 pre-registration must fix them explicitly.

### D85 — Stage 1 stacks 7 ladders per GPU (3 GPUs in total)
- **Evidence:** stage-1 pack 0 (job 19150509, H100, 4 ladders, league default sizes):
  - 6.5–7.1k learner SPS per ladder;
  - 16 GB of the 80 GB used;
  - the heartbeat busy 47% of the time to hold 80% device utilization, i.e. the 4 ladders' training
    uses about a third of the GPU's time;
  - each ladder needs ~1 CPU core.
- **Decision:** the 14 ladders not yet started go into two 7-ladder jobs (`stage1_packA/B.txt`, 16 CPUs,
  128 GB host memory), replacing four pending 4-ladder jobs (cancelled while pending).
  - Expected at 7 ladders: ~60% GPU time, ~28 GB, 7 cores plus the heartbeat (H200 jobs allow 16 CPUs).
  - Stage 1 then needs 3 GPUs instead of 5 (within the user's starting limit of 4), and the smaller
    request should start sooner in the congested queue.
  - Per-ladder SPS may drop somewhat at 7. The comparison is in learner steps, so speed does not
    bias it.
- **Reverse if:** the 7-ladder jobs run below ~4k SPS per ladder (then split them).

### D86 — Stage 1 adds a fixed-budget heuristic phase (the plan's decision rule needs it)
- **The disagreement** (HANDOFF/tools vs TRAINING_PLAN; the plan wins, reported to the user):
  - `scripts/stages.py` passes `--early-stop-score <threshold>` on every rung, so the heuristic rung ends
    as soon as the learner scores ≥ 0.55 over 2,000 matches at a snapshot epoch.
  - TRAINING_PLAN §8.1's decision rule compares configurations by "the heuristic score at 100M / 200M
    / 300M steps", and the plan budgeted G3 at 100–300M.
  - On the cluster, `S1main_s1` passed G3 (0.656) at the rung's first gate check, 3.28M steps, so the
    fixed-budget scores never exist for ladders that pass.
- **Decision.** The ladders run as specified: their verdicts are G1–G3, plus steps-to-gate at
  the 3.28M-step check resolution. In addition, every ladder gets a **fixed-budget heuristic run**:
  - `hpc/league_run.py --run-id <prefix>_H300 --init-from <prefix>_1_random --anchors bot:heuristic
    --self-play-frac 0 --anchor-frac 1.0 --total-timesteps 300000000`;
  - the ladder's own config flags, no early stop, wandb group `S1H<config>`;
  - 18 runs, 6 per GPU (`$DATA_DIR/packs/stage1_H300_pack{H,I,J}.txt`).
  - Same start for every configuration (its random-rung model), the same 300M budget, and no optimizer
    reset in the middle. The 100M / 200M / 300M scores (last 2,000 matches before each mark, from
    `history.jsonl` outcomes) feed the decision rule.
- **Cost:** 18 × 300M = 5.4B learner steps, ~17 h on 3 GPUs at ~5k SPS per run.
- **Reverse if:** the user prefers ranking by steps-to-G3 or by the score at a smaller budget (then
  stop the runs at that budget).

### D87 — GPU heartbeat: double-buffered at nice 0; packs cap threads per line
- **What went wrong.** Every stacked pack started with the GPU below the 75% floor (D79):
  - first 5-min window: Stage-1 pack B (19150614) 7.4%, pack H (19152998) 64%, pack I (19152999) 34%;
  - pack J (19153517, gh002, node load 88 of 96 cores) read 5%, 0.4% and 1.7% for 15 min. Its 6 runs
    needed 13–16 min to start (about 2 min elsewhere), each burning 2–3 cores of user CPU meanwhile.
  - Throughout, the heartbeat reported itself busy 56–99% of the time.
- **Causes (measured, jobs 19154132 / 19155336, H100, 24 busy loops on the job's 16 cores):**
  - the heartbeat ran at nice 19 and synchronized after every 4 small matmuls (~1.4 ms of GPU work),
    so each time it lost the CPU the GPU went idle; at nice 19 under full CPU load it never even
    started within 130 s, in either version;
  - pack lines ran with no thread limit, so torch and numpy started one thread per CPU in every
    process (6 lines × 16 threads on 16 CPUs). That oversubscription is the likely cause of the slow
    starts. It is not proven: the slow phase was over before it could be profiled.
- **Decision.**
  - `hpc/gpu_heartbeat.py` queues ~20 ms batches of matmuls (count measured at start) and keeps two
    batches in flight, waiting on blocking CUDA events (no spinning core).
  - `hpc/lib.sh` starts it at nice 0 (`HEARTBEAT_NICE`).
  - Measured under the same CPU load: new heartbeat at nice 0 held 80.0% with 0 of 30 one-second samples
    below 75% in every window; the old one at nice 0 held 54–73%; the new one at nice 10 held 79–82%,
    with up to 10 of 30 samples below 75%.
  - `hpc/pack.sbatch` defaults `PACK_THREADS` (OMP / MKL / OpenBLAS) to the job's CPUs ÷ the lines
    running at once, at least 1 (6 lines on 16 CPUs → 2).
- **Scope.** New jobs only. The running packs (B, H, I, J) keep the pinned old heartbeat. After their
  startup windows, every 5-min window has been 75.7–87.3% (mostly 79.5–80.5%), so they are not
  restarted.
- **Reverse if:** a pack's SPS drops with the thread cap (then raise `PACK_THREADS`), or the nice-0
  heartbeat measurably slows training.

### D88 — The fixed-budget phase stops at 100M, not 300M (user's decision, 2026-10-04)
- **Deviation** from D86 and TRAINING_PLAN §8.1 ("heuristic score at 100M / 200M / 300M"), as D86's
  "reverse if" anticipated.
- **Why** (what the user was shown):
  - Ceiling: at 75M, 7 of the 9 configurations had a seed-mean score vs `bot:heuristic` within 0.987–0.995.
  - The uncertainty is about ±0.004 (95%) per run at p ≈ 0.99 over 2,000 matches. The two seeds of
    one configuration differ by up to 0.014 (grid4 at 50M: 0.971 vs 0.993).
  - The order reshuffled between marks (noshape joint 1st at 50M, 5th at 75M), so 200M/300M would
    most likely give a tighter tie, which the plan's rule already breaks toward the simpler option.
  - Continuing would have cost ~27 GPU-hours and ~8.5 h before stage 1 could be frozen.
- **Procedure.**
  - Each run's last snapshot past 100M is `snap_6200.pt` (epoch 6,200 × 16,384 = 101.6M steps).
    Once all 18 had written it, jobs 19152998 / 19152999 / 19153517 were cancelled by job id.
  - Scores at 100M:
    - the training window (last 2,000 matches before 100M, `hpc/stage1_report.py`);
    - `eval.py` on `snap_6200.pt` vs `bot:heuristic`, 500 matches per seat, sampled
      (`$DATA_DIR/packs/h300_100M_eval.txt`).
- **Lost:** the 200M / 300M marks and longer stability evidence against a fixed bot. The latter is a
  weak proxy for stage 3, which is self-play. The one late decline (lr1e-3_s0, 0.990 at 50M to
  0.920 at 78M) was already visible.

### D89 — Stage-1 configuration: `main` as the default, `grid2` kept ready (user's decision, 2026-10-04)
- **Data:** EXPERIMENTS "Stage-1 results". All 18 ladders passed G1–G3. At the fixed budget (100M,
  D88), every configuration except lr1e-3 and flat scores 0.988–0.997 vs `bot:heuristic` (`eval.py`,
  1,000 matches per seed): a ceiling tie.
- **The plan's rule** (TRAINING_PLAN §8.1): best heuristic score at the fixed budget, ties broken
  toward the simpler option (feed-forward, grid 1). Among the tied feed-forward grid-1
  configurations (main, noshape, lr1e-4, ent), main learns fastest: 0.90 at 11.5M vs 15.2–19.3M.
- **Priority (looking at everything: final score, learning speed, stability, seed spread, risk):**
  1. **main** (recommended freeze).
     - Tied best at 100M (0.990).
     - Fastest grid-1 learner, with the smallest seed spread in speed (11.4M / 11.5M).
     - Stable.
     - No placement-precision risk.
  2. **grid2** (best measured performer; the alternative).
     - Fastest to 0.90 (7.4M) and ≥ main at every mark on both seats; 0.997 at 100M.
     - Not chosen first: the margin is within the ceiling tie at the budget, and 4M steps is < 1% of
       a 500M–1B stage-3 budget.
     - Coarse placement (144 blocks instead of 576 tiles) changes the game that stages 3–4 would
       solve and measure exploitability in. The heuristic bot cannot show whether that costs skill
       against strong opponents.
  3. No gain over main; each learns slower:
     - noshape (15.2M);
     - lstm (16.2M, adds recurrence);
     - lr1e-4 (17.2M);
     - ent split (19.3M);
     - grid4 (16.0M, largest seed spread).
  4. Rejected:
     - flat: 3× slower (34.1M), lowest healthy final score (0.980);
     - lr1e-3: unstable; s1 collapsed to 0.45 at 100M.
- **Frozen settings if approved** (stages 3–4):
  - conditional head, feed-forward, `placement_grid 1`;
  - shaping `reward_tower 0.3`, `reward_crown 0.2`, `reward_elixir 0.01` with `--train.reward-clip 0`
    (stage 3 anneals it to 0 over the first 50%, TRAINING_PLAN §8.3);
  - `royale.ini` hyperparameters: lr 3e-4, γ 0.999, λ 0.95, clip 0.2, vf_coef 2.0, ent_coef 0.01,
    bptt 64, 1 update epoch;
  - `--train.torch-deterministic=` (D82).
  - Sizes (num_envs, batch, minibatch) are a stage-3 pre-registration item: the SPS research may
    change them.
- **Reverse if:** the user prefers grid2's speed (accepting the coarser game), or stage-3
  throughput forces a budget so small that the 4M-step head start matters.
- **User's decision (2026-10-04):** "Lets keep main as the default - BUT lets keep everything ready
  for Grid 2 as well. This is a VERY useful ablation down the line as things scale up. I want to keep
  both."
  - Default (stages 3–4): **main**, as frozen above.
  - **grid2 is a standing ablation, ready to launch at any stage:**
    - `hpc/stage3_submit.sh` with `GRID=2` prints / submits the same A / B / C runs with
      `--env.placement-grid 2` (run ids `S3g2_<run>_s<seed>`, wandb groups `S3g2_<run>`);
    - the evaluation tools already handle it: `eval.py` and `best_response.py` take each checkpoint's
      grid from its `config.json`, and `tournament.py` lets checkpoints of different grids meet, so
      main-vs-grid2 head-to-head works.
  - The stage-1 grid2 runs are kept, never deleted: ladders `S1grid2_s{0,1}_{0_noop,1_random,2_heuristic}`
    and fixed-budget runs `S1grid2_s{0,1}_H300` (snapshots to 101.6M) under `$DATA_DIR/league/`.


### D90 — Throughput package for league training (user-approved 2026-10-04: "I agree with your changes")
- **Why.**
  - Stage-3 league shapes were CPU-bound: the C engine stepped every match on one core (74–80% of a
    step), and B/C paid ~1.4 ms of host time per opponent group (~65 groups per step at 64 snapshots).
  - Stage-3 baselines at the frozen sizes (H100, 256 envs / batch 16,384 / minibatch 1,024): A 4,745 SPS,
    ckpt64 1,224 SPS. At 1B steps that is ~2.4 days per A run and ~9.5 days per B/C run.
  - Findings and measurements: `research/REPORT.md` on branch `sps-research`; EXPERIMENTS "SPS research".
- **What.** All opt-in; with none of these the default paths are unchanged.
  - `pufferroyale/fast_league.py` (`FastLeagueVecEnv`, enabled by `--env-threads K`, `--fast-opponents`,
    `--cuda-graphs`):
    - threaded engine stepping through a new binding function `vec_step_range` (GIL released; no game
      code touched);
    - observations copied to the GPU once per step;
    - opponents sampled on the GPU;
    - frozen checkpoints in one vmapped forward;
    - CUDA graphs for the no-grad forwards.
  - `LeagueVecEnv.send()` calls two hooks whose base versions are exactly the statements they replaced.
  - `PR_OPT_FLAGS` (setup.py; `-O3 -march=x86-64-v3` for every job via env.sh).
  - `OMP_NUM_THREADS=1` default in league.sbatch.
- **Measured** (research job 19164429, H100). Full package, frozen sizes: A 13,194 SPS (2.8×), ckpt64
  10,703 (8.7×). At 512 envs / batch 65,536: A 18,888, ckpt64 14,802.
- **Semantics.**
  - Bit-identical to the default path:
    - env stepping, observations, rewards, results and card statistics (checked at grids 1 and 2);
    - greedy policy-opponent trajectories.
  - Differences:
    - GPU sampling uses a different random stream than the CPU generator (same distribution; saved in
      `rng_state()` for exact resume);
    - vmapped logits differ by float rounding (max |Δ| 8.4e-5, KL ≤ 2.3e-7, same argmax);
    - padding "self" rows to a multiple of 32 may change logits at rounding level.
  - All stage-3 runs use the same code, so the A/B/C comparison is unaffected.
- **Bugs found before any real run, and fixed.**
  1. A process-wide CUDA-graph memory pool failed an allocator assert once all its graphs had been freed
     (hpc/tests on GPU). Fix: one pool per graph cache.
  2. **The learner's graph wrapper, a plain closure, was copied into MMD's deep-copied reference, so the
     magnet replayed the learner's graph and the MMD KL was 0** (independent code review). Run C would
     silently have trained like B. Fixes:
     - the wrapper is a bound method that replays only for its own policy object;
     - it is installed after the trainer;
     - a test deep-copies a graphed policy exactly as `_frozen_copy` does.
  3. Further review hardening (league_train.py, eval_watch.py, league.sbatch, stage3_submit.sh):
     - eager fallbacks if a capture or the vmapped forward fails;
     - `thread_local` graph capture;
     - an event sync before the C step overwrites the pinned buffer;
     - `--no-fast-opponents` / `--no-cuda-graphs`;
     - eval exports newer than the resumed epoch are deleted;
     - one live job per run directory;
     - immutable, self-contained stage-3 submissions.
- **Tests:** `hpc/tests/test_fast_league.py` (CPU + CUDA) and the `fast-tests` step of hpc/verify.sbatch.
  Results of the final verification: EXPERIMENTS "Stage-3 verification".

### D91 — Evaluation sidecar: wandb "eval" section vs heuristic / random / noop (user's request, 2026-10-04)
- **What.**
  - `league_train.py --eval-export-interval N` writes the learner's weights every N epochs to
    `<run>/eval_ckpt/`. It touches no run state or RNG.
  - `hpc/eval_watch.py`, started next to the training by league.sbatch (`EVAL_EVERY=N`), evaluates each
    export with `scripts/eval.py` on the CPU:
    - one process per bot;
    - sampled actions, both seats, `EVAL_EPISODES` matches per seat and bot;
    - the same seed at every point.
  - It logs `eval/vs_heuristic`, `eval/vs_random`, `eval/vs_noop` (plus `eval_detail/*` intervals and
    seats) against global_step to a wandb run `<RUN_ID>-eval` in the training run's group, and every
    point to `<run>/eval_history.jsonl`.
- **Stage 3.**
  - Every 100 epochs = 1.64M learner steps, about every 2–3 min of training.
  - 50 matches per seat and bot, i.e. 100 per bot; the 95% interval is about ±0.1 at 0.5 and ±0.06 at 0.9.
- **Evaluation only.**
  - The bots never enter training (TRAINING_PLAN principle 3); these curves are monitoring, not the
    pre-registered E3.
  - A point is recorded only after the run has saved past its epoch, so a requeue never records points
    from a discarded trajectory.
- **Cost.** 3 CPU processes at nice 5 next to the training, inside the job's 24 CPUs (training uses ~3–4).

### D92 — Stage 3 launched per the pre-registration (user's approval and choices, 2026-10-04)
- **User:**
  - "Once everything is done - and verified (and meticuolosuly checked) - I hope you launch the
    training runs required";
  - 1B learner steps per run;
  - 4 GPUs at a time;
  - main as the default with grid2 kept ready (D89).
- **What runs.** EXPERIMENTS "Stage-3 pre-registration":
  - A / B / C × seeds 0–2, plus Clo / Chi (seed 0);
  - the frozen main settings at the stage-1 sizes;
  - snapshot interval 477 with 64 kept;
  - shaping annealed over the first 50%;
  - the D90 throughput package;
  - the D91 eval section.
- **How.**
  - ONE job array (`hpc/stage3_submit.sh` → `hpc/stage3_array.sbatch` → `hpc/league.sbatch`),
    `--array=0-10%4`, one H100 per task, 24 CPUs;
  - 72 h wall time, which selects QOS gpu168: the cluster's 4-GPU per-user allowance for jobs over 48 h.
    gpu48 fell to 2 GPUs per user that day and is shared with the user's other project.
- **Verification before launch:** EXPERIMENTS "Stage-3 verification". Every check passes; two bugs were
  found and fixed before any real run (D90).
- **Engine freeze:** the launch commit (EXPERIMENTS, Stage-3 runs table); its code equals the verified
  d388d4f.

### D93 — Stage 4 starts from random init; the stage-3 finals become a periodic Hog-mirror check (user's decision, 2026-10-05)
- **User:**
  - raised the concern: "if we do this - the 'weights' will implicitely carry some non-zero bias
    for/against the cards that it was trained against";
  - proposed the evaluation: "We can use those policies as an eval against the Stage4 polciies (on
    the same deck, of course) we trained periodically";
  - agreed with the order stage 3 → E1 → stage 4 (from scratch, checked against the stage-3
    specialists) → stage 4o.
- **Change.** TRAINING_PLAN §8.4 and T15: stage 4 starts from **random initialisation**, not from the
  best stage-3 policy. A warm-start run remains an **optional comparison arm** at the same budget.
- **Evidence (S3_C_s0 final model, checked 2026-10-05):**
  - **Card bias:** the 8 Hog 2.6 card-embedding rows (16-d) have norms 1.1–1.9 (mean 1.46). The
    other 56 cards' rows never received a gradient: mean norm 4.03, as for an untouched N(0, 1) init
    (√16 = 4). The card-stat projection was fit on those 8 cards only.
  - **Strategy bias:** card play rates in S3_A_s0's last logging window: Ice Spirit 0.95, Skeletons
    0.95, Fireball 0.012, Hog Rider 0.007. The mean episode length was 573 of at most 600 env steps,
    so at least 89% of games went past regulation (360 steps) into overtime. Action entropy fell
    from ~1.42 at init to ~0.42–0.54 by the end of training.
  - **RQ3:** the deck meta-game pilots every deck with the stage-4 policy, so a Hog-practised pilot
    would favour Hog 2.6.
  - **Cost of starting from scratch:** stage 1 reached 0.90 vs the heuristic bot in 7.4–11.5M steps,
    about 1% of the stage-4 budget.
- **Periodic specialist check (TRAINING_PLAN §8.4 item 3, T18).** The stage-4 policy pilots Hog 2.6
  against the frozen stage-3 finals in the Hog mirror (fair score 0.5), both seats, every K epochs and
  on the final model. The stage-3 policies are never stage-4 training opponents. K, the opponent set
  and the match count are fixed in the stage-4 pre-registration.

### D94 — Stage 4o (deck specialists) added to the plan as a proposal (user's idea, 2026-10-05)
- **User:** "we fix 1-5 decks for ourselves - and play AGAINST RANDOM decks ? (Not self-play but
  soemthing) ... This should gie us a best policy FOR a deck against any card ? Less general than
  final proposed Stage 4 polciy - but something still ?"; "DO not implement anything - this is just
  an idea"; then "lets add this to the proprosal".
- **What it is (TRAINING_PLAN §8.4o, T19).**
  - The learner pilots one fixed deck D; one network per deck is the clean version.
  - The opponent seat draws from the stage-4 deck mix (real-deck pool + random decks, held-out decks
    excluded).
  - The opponent decks are piloted by frozen stage-4 snapshots (PFSP). There is no self-play, and no
    scripted bots, which are too weak.
  - In game-theoretic terms, a best response to a population, not a minimax policy.
- **Order.** It comes after stage 4, which supplies the strong random-deck pilots.
- **Open, for its own pre-registration:**
  - which 1–5 decks;
  - random init vs warm start from the stage-4 generalist;
  - the budget, after a pilot run.
- **Nothing implemented.** Plan text only.

### D95 — Card roster: Champions, Evolutions and Heroes deferred to last; card batch 3 proposed (user's decision, 2026-10-05)
- **User:**
  - asked: "I know there are a LOT of other cards clash royale has -> is it posible for you to search
    up and let me know if its possible for us to add more ?";
  - on Champions, Evolutions and Heroes: "That sucks, yeah leave this out -> we'll do that at last if
    time permits - but update th eplan with our discussion".
- **Decided.** Champions, Evolutions and Heroes are deferred to the very end (stage 5, only if time
  permits). Their needs:
  - a new "activate ability" action (a policy-head change);
  - per-match cycle counters;
  - deck-slot rules;
  - more card ids than the 128 the observation has;
  - data the repo lacks: `cards-15.535.json` states "evolutions and hero forms are excluded from
    `cards`".
- **Proposed, awaiting approval: card batch 3.**
  - About 20–25 of the 58 deck-legal cards whose level-1 stats are already in
    `data/source/cards-15.535.json` but not in the engine: tier 1 plus the simpler spells.
  - A new engine work package done like batch 2 (SPEC §16), in parallel with stage 4.
  - Stage 4 starts on the 64 cards and does not wait. A later stage-4 run may use the larger roster.
- **Facts (checked 2026-10-05):**
  - **The real game:** about 123 regular cards, plus 8 Champions, 43 Evolutions, 18 Heroes and 4
    tower troops (deckshop.pro; other sites say 127).
  - **The repo data:** 144 entries. 64 are in the engine; 22 are event or hidden variants; 58 are
    deck-legal and missing.
  - **Capacity:** `CARD_SLOTS = 128`, so 122 cards fit without a layout change.
  - **Effort tiers** (TRAINING_PLAN §8.6), from comparing source fields with `tools/gen_card_db.py`
    and the engine's 7 spell types and 5 buffs:
    - 20 troops/buildings with only standard fields;
    - 19 with extra behaviour fields;
    - 11 spells, mostly needing new mechanics;
    - 8 Champions.
  - **Card coverage:** with 122 cards a random deck contains a given card about half as often, so
    stage 4 on the larger roster needs roughly twice the steps for the same games per card.
- **Constraints.** Engine work: the HPC agent may not change engine code, the spec, tests or the
  golden reference. The engine freeze holds until stage 3 and E1 are done; new cards start a new
  engine lineage.
- **Nothing implemented.** Plan text only (TRAINING_PLAN §8.5, §8.6, T20).

### D96 — E1–E5 start; Clo/Chi get the full E1 probe; stage 4 may later train all three methods (user's decision, 2026-10-07)
- **User:** "Okay - yeah lets go ahead and sure lets add Clo/Cli as well. Also - in the plan - add the
  fact that while we WILL do these - I guess for the full card training - I also want to use all the
  3 methods and NOT only the winner. This is an optional take (subjected to time) - first we'll do the
  trianing only with the winner - and later add others if time permits. Add this to the trainig plan
  and now continue with the work as required".
- **E1–E5 approved and started.** Run per the pre-registration (EXPERIMENTS "Stage-3
  pre-registration"), on the pinned stage-3 checkout, with at most 4 GPUs at a time.
- **Declared addition: Clo/Chi in E1.** S3_Clo_s0 and S3_Chi_s0 get the same probe as the 9
  targets: the calibrated budget, BR seeds 0 and 1, and 1000 matches. They are reported as the
  exploitability-vs-MMD-strength curve on seed 0, never pooled into method C. E3 and E4 also cover
  them. E2 is unchanged.
- **Stage 4 extension (optional, time permitting).** The winner's full-card run comes first. The
  other two methods may follow with identical settings, repeating the A/B/C comparison at full-card
  scale (TRAINING_PLAN §8.4, T21; HANDOFF §11).
- **Also corrected:** HANDOFF §11 still said stage 4 starts `--init-from` the winner's model. It now
  says random init, per D93.

### D97 — E1 best responses train with the D90 throughput paths (2026-10-07)
- **Why.** `best_response.py` built its trainer from league_train's defaults, where the D90 paths are
  off. The measured plain-path BR speed was ~6.2k steps/s on an H100 (smoke job 19346156: 3.01M
  steps in 489 s), about a third of stage-3 training. That would make the 400M calibration run ~19 h
  and E1 ~2–5 days on the 4 GPUs.
- **Change (ae3b6fc).**
  - `best_response.py` gains `--env-threads`, `--fast-opponents` and `--cuda-graphs`. They are passed
    to `league_train.build` exactly as league training does, and are off by default.
  - After training, the learner's CUDA graph is removed (new `fast_league.uninstall_learner_graph`),
    so the evaluation matches run the same plain forward as before.
  - No engine code changed: `git diff 4865c0e ae3b6fc` touches only `scripts/best_response.py`,
    `pufferroyale/fast_league.py` (the new function), `hpc/tests/test_best_response_fast.py` and docs.
- **Checks (job 19348569, pinned `e1_ae3b6fc`).** `pytest hpc/tests`: 23 passed. These include CPU
  `--env-threads` bit-identity of the BR's weights and W/D/L, and a CUDA run with all three flags.
  - **Speed:** 3.01M steps in 166.5 s, ~18.1k steps/s (2.9×).
  - **Training curve:** similar to the plain smoke, with training score vs the target 0.587 vs 0.577
    at 3M steps.
- **What differs from the plain path.** Env stepping is bit-identical. GPU opponent sampling uses a
  different random stream with the same distribution, and the vmapped target forward matches up to
  float rounding (D90). These are statistically equivalent BRs, so every E1 probe uses the same flags.
- **Settings for every probe:** `--init-from-target --train.torch-deterministic= --env.reward-tower 0.3
  --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0 --shaping-anneal-frac 0.5
  --env-threads 16 --fast-opponents --cuda-graphs`, 256 envs (batch 16,384, minibatch 1,024 by
  default), 1000 matches, on the pinned checkout `e1_ae3b6fc`.

### D98 — E1 probes log to W&B (2026-10-07)
- **Why.** The user asked to see the E1 probes in W&B. `best_response.py` had no W&B logging, so E1
  was invisible there; its only trace was a heartbeat line every 5 minutes in the Slurm log.
- **Change.** `best_response.py` accepts league_train.py's logging flags (`--wandb --wandb-project
  --wandb-group --tag --no-model-upload`, through `league_train.make_wandb`). Without `--wandb`, wandb is
  never imported, as before.
  - **Training:** one record per PPO epoch at step = global step: epoch, sps, `losses/*`,
    `results/<target>/score` and `games` (the epoch's finished training matches), `train_score_vs_target`
    and `train_games_vs_target` (the pool's running estimate), and PuffeRL's `environment/*` stats.
  - **Evaluation (1000 matches):** the run summary holds the running `eval/matches`, `eval/wins`,
    `eval/draws`, `eval/losses` and `eval/br_score`, updated every 50 matches.
  - **End:** `final/br_score`, `final/exploitability_proxy`, `final/wins`, `final/draws`, `final/losses`,
    `final/total_timesteps` and `final/train_score_vs_target`.
- **Logging is read-only.** It reads the trainer's losses and drains the env's finished-match list,
  which nothing else in the BR uses. New test `test_br_wandb_offline_same_result`: the same BR with and
  without `--wandb` (WANDB_MODE=offline) has bit-identical weights and identical W/D/L and training
  counts, and the offline run holds the training and evaluation keys.
- **Checks.**
  - CPU job 19376288 (working tree = 43de9fd before the `final/` key rename): `pytest hpc/tests` 11 passed,
    13 CUDA-only skipped; the best_response / wandb / CLI-flag spec and builder tests 24 passed; an online
    CPU smoke synced to W&B (run `E1_smoke_wandbcpu_A_s0`). CPU job 19379568 re-ran `hpc/tests` after the
    rename: 11 passed.
  - GPU job 19374641 (H100, same content as 43de9fd): `pytest hpc/tests` 24 passed (CUDA included); spec /
    builder subset 24 passed; online 3M-step BR smoke with all D90 flags, run `E1_smoke_wandb_A_s0`
    (group `E1_smoke`): 18.1k steps/s, the same as without logging (D97).
  - CPU job 19381556 on the pinned checkout `e1_43de9fd`: extension built (`-O3 -march=x86-64-v3`),
    isolation OK, `pytest hpc/tests` 11 passed, 13 CUDA-only skipped.
- **Probe settings.** Each probe runs as `E1_<name>_s<seed>` in project `pufferroyale`, group `E1`,
  tag = the target's method (A, B, C, Clo, Chi, or scratch), with `--no-model-upload`.
- **Gap.** The three calibration runs (D96; 50M, 150M, 400M) were started before this change and are
  not in W&B. The calibration run at the chosen budget is also the A_s0 seed-0 probe; its numbers are in
  EXPERIMENTS.md.

### D99 — Stage 4: method C first, A as the self-play ablation; 10B steps in chained 20 h jobs (user's decision, 2026-10-09)
- **User** (2026-10-09, after the E1–E4 results): "we decided to move ahead PRIMARILY with C (best overall
  performance) - and A for ablation testing (difference from pure sefl play)"; "lets NOT change the learning
  poitn thing (use the same non decay thing). Also - while chaining - use 20 hours per sbatch script - and
  chain accoridnglt"; "For the specialists -> we are using ONLY the best per method"; 2-GPU jobs dropped;
  "One C and one A on each" GPU type.
- **Method choice.** By the pre-registered rule (E1 first, E2 second; TRAINING_PLAN §8.4): E1 showed no
  difference between methods (mean `br_score` A 0.874, B 0.847, C 0.863; seed spread larger), so E2
  decides: C has the best score against the other methods' finals (0.569; A 0.550, B 0.381), beats B in 8
  of 9 seed pairings and was strongest at the 50% checkpoints. C over A at the end is not established (A
  vs C 0.546). A is kept as the pure self-play baseline; B is dropped (B → C was answered in stage 3). At
  full-card scale, A vs C tests "population + MMD" against pure self-play together. This replaces D96's
  "winner first, the other two only if time permits".
- **Runs.** C and A, seeds 0 and 1, one GPU each: S4_C_s0 and S4_A_s0 on H100/H200, S4_C_s1 and S4_A_s1
  on A100. 10B learner steps each; the 2B and 5B results are the snapshots nearest those step counts
  (with a constant learning rate a checkpoint of the 10B run is equivalent to a run that ends there, so
  separate 2B / 5B runs would add nothing).
- **Unchanged from stage 3:** constant learning rate 3e-4 (no `--anneal-lr`), sizes (256 envs, batch
  16,384, minibatch 1,024), `--snapshot-interval 477` with 64 kept (the pool covers the last ~500M steps;
  same speed as stage 3), shaping 0.3 / 0.2 / 0.01 faded over the first 500M steps (`--shaping-anneal-frac
  0.05` at 10B = stage 3's N = 1,953,125 env steps), the D90 throughput flags, random init (D93).
- **Decks (§8.4):** pool = the 7 presets (hog26, giant, bait, golem, lavaloon, miner_poison, pekka_bridge),
  `random_deck_frac 0.5`, held out = xbow, royal_hogs, `random:50:20261003`, seats draw independently.
- **Chaining.** Each run is a chain of 20 h jobs (`hpc/stage4_submit.sh`, new): link k+1 depends on link
  k (`afterany`) and resumes from the last save (every 477 epochs, so a time-limit kill loses at most
  one snapshot interval); a link that finds the run finished does nothing.
- **Monitoring (new code, cluster-side).**
  - `league_train.py --log-interval N` (N = 10; named so
    that PufferLib's `--wandb` flag is not an argparse abbreviation of it): one W&B call per N epochs, losses and environment/*
    stats averaged, results summed; history.jsonl keeps every epoch; N = 1 is the old path unchanged.
  - `hpc/eval_watch.py --decks NAME=DECKSET --ckpt NAME=PATH`: every 6,104 epochs (≈100M steps) the
    learner plays (a) the heuristic and random bots on the pool decks and on the held-out decks (each deck
    a mirror, 10 matches per seat and deck), and (b) the stage-3 Hog 2.6 specialists S3_A_s1 and S3_C_s0
    (best final per method by E2 Elo) in the Hog mirror, 200 matches each, half per seat (the D93
    specialist check). Logged to the `<run>-eval` W&B run.
  - `hpc/league.sbatch` passes `EVAL_ARGS` to the sidecar.
- **Checks.** CPU job 19487536 (working tree before the commit): `pytest hpc/tests` 16 passed (new
  `hpc/tests/test_stage4.py`: the window's averages and sums; a CPU league run with `--log-interval 3` and
  WANDB_MODE=offline logs epochs 3, 6 and the last partial window 7, whose results add up to history.jsonl;
  the sidecar's deck-set and checkpoint evaluations with per-seat scores; the launcher's dry run), 13
  CUDA-only skipped; `tests/spec/test_hpc_templates.py`, `test_v05_tools.py`, `test_league.py`: 96 passed.
  A first CPU job (19487146) failed the league test: `--wandb-interval` made PufferLib's `--wandb` an argparse
  abbreviation (`expected one argument`); renamed to `--log-interval`.

### D100 — Stage 4: a card-probability floor in the learner's card choice (user's decision, 2026-10-09)
- **User** (2026-10-09, after the pilot's per-card play rates): "There should be something which ensures a
  base frequency of cards playing when they appear"; on the options: "I like the prbablity floor too";
  "I am the mac team too - so I give you the permission"; "Make the change". The user's caveat on the
  stage-3 evidence: "another reason say fireball wasnt played was also becuase we were playing agianst teh
  same deck -> so we might find a way to NOT use fireball and play agianst the same deck".
- **Evidence.** Pilot (`S4pilot_*`, 15–28M steps, play rate = plays / decisions with the card in hand,
  affordable and no lockout): Fireball 0.0001 (C_s1) / 0.0023 (A_s1), Poison 0.0003 / 0.0026, Freeze
  0.0004 / 0.0023, Arrows 0.0020 / 0.0025, The Log 0.0023 / 0.0028, Lightning 0.0035 / 0.0023, Zap 0.53 /
  0.0028; the median card 0.54 / 0.51. Stage 3: Fireball ≈ 0.0000 for the first 250–500M steps in all four
  C/A runs checked although it was in every game, 0.014–0.025 from 500–750M on. A card that is never played
  also never leaves the hand (SPEC §4.1). The entropy bonus barely moves an option of probability p (its
  gradient scales with about p·|ln p|: 0.0009 at p = 1e-4 against 0.32 at p = 0.2).
- **Design** (`pufferroyale/torch.py`, conditional head; the policy code is the Mac team's, changed with the
  user's permission): `Policy.enable_card_floor(eps)` / `set_card_floor(eps)`; for the k legal hand slots of
  a row P'(s) = (1 − k·eps)·P(s) + eps and P'(wait) = (1 − k·eps)·P(wait); the position distribution P(j | s)
  and the mask are unchanged; logits, sampling and PPO's log-probabilities all use P'. eps is a
  non-persistent buffer (state dicts and checkpoints unchanged; a loaded checkpoint plays without a floor),
  read on every call so CUDA-graph replays see each new value; enabled before any graph capture. Off (the
  default), the head runs exactly the statements it ran before.
- **Training** (`scripts/league_train.py --card-floor EPS --card-floor-steps N`): eps = EPS · max(0, 1 −
  step / N) for the epoch starting at learner step `step`, set before the rollout on the learner and on the
  MMD reference (its refreshes are `load_state_dict`, which does not carry the buffer). The "self" opponent
  is the learner object, so it plays with the floor; snapshots, the eval sidecar and all evaluations use
  the unfloored policy. history.jsonl / W&B: `card_floor`; W&B: `cards/below_floor` = the number of cards
  whose play rate is below EPS.
- **Stage-4 settings** (`hpc/stage4_submit.sh`): EPS = 0.005, N = 5B learner steps (half the run). Stage-3
  spell rates once alive (0.014–0.025) are above 0.005, so the floor only binds cards the policy has
  dropped; the 5B and 10B analysis points come from floor-free training (the 2B point trained with eps
  0.003). Example: a card playable for 100 decisions (50 s) is cast at least once with probability
  1 − 0.995^100 ≈ 0.39.
- **Pilot again** before production: the first pilot (no floor) stays as the comparison at equal seeds.

### D101 — Stage 4: Claude Opus 5.5 plays the training policy every 250M steps, recorded to W&B (user's decision, 2026-10-09)
- **User** (2026-10-09): "every 100M steps -is it possible to launch a separate problem where Cluade Opus 5.5
  with one of the unseen coherent decks (not random) plays aginst the training policy - this should be ONE
  separate job per main run - and the logs are added as the main jobs runs once every 100M steps. THe recorded
  game in the simulator should be logged to Wandb"; "It play sonly 1 game (NO SDK)"; choices: the headless
  Claude Code CLI inside the job, and for the policy's deck "Lets do both" (mirror and a pool deck). After the
  usage discussion (the CLI runs on the user's claude.ai Team seat, EMERGE Lab, no API key; the CLI's
  `total_cost_usd` is an estimate at API prices, not a bill; ~400 games would strain the seat's limits): "let
  keep games every 250M - and ONLY for C - any one seed ONLY".
- **Scope.** S4_C_s0 only (H100, the faster seed): one game per 250M learner steps, 40 games over 10B, each
  against the first saved policy at or after the mark (a snapshot, never deleted; the final model for the last
  one), sampled actions, without the D100 floor. Evaluation only: nothing reaches the training.
- **Setup of game k** (i = k − 1; `hpc/llm_game.py game_config`): Claude plays X-Bow or Royal Hogs (the held-out
  presets, alternating); the policy plays the same deck in games with (i // 2) even, else a pool preset
  (cycling through all 7 within 40 games); Claude's seat flips every 4 games. One game per mark ("both" deck
  modes alternate).
- **Mechanism (no SDK).** A CPU job chain per run (`hpc/llm.sbatch`, 2-day links): `hpc/llm_game.py` (in the
  image) plays the match through the repo's text harness (`pufferroyale.llm`: RULES_PROMPT, render_state,
  parse_action; Claude decides every 20 ticks = 1 s and is not asked when it has no legal play), writing each
  decision to a node-local turn file; `hpc/llm_relay.py` (on the host, where the CLI and its login live)
  answers with one stateless call `claude -p --model claude-opus-5-5 --effort low --system-prompt-file
  RULES_PROMPT.txt --tools "" --strict-mcp-config --no-session-persistence --disable-slash-commands
  --setting-sources "" --output-format json`. Checked by hand (2026-10-09, one call): logged in through
  claude.ai, no CLAUDE.md / memory / tools in the context, 12.6 s per call. A failed call (e.g. a usage limit)
  is retried with back-off; after 3 h the game is aborted and retried later (3 attempts).
- **Records.** Per game `<run>/llm/game_<k>/` (meta, every tick's frame, plays, Claude's decisions with replies
  and token usage, result, GIF via `hpc/llm_render.py`); W&B run `<run>-llm` (group S4_C) at step = the policy's
  global step: `llm/score` (Claude's), crowns, decision / error / token counts, the CLI's cost estimate,
  `llm/replay` (the GIF) and `llm/decision_log` (Claude's replies).

### D102 — Stage 4: four more runs with constant dense rewards and a constant card floor (user's decision, 2026-10-10)
- **User** (2026-10-10, after S4_C_s0's specialist scores fell from 0.23 / 0.21 at 500M to 0.09 / 0.06 at
  600M, right after the shaping had faded out): "we will have to launch 4 more runs - everything, literally
  everything same as right now - but the bonus wearing off has to go - I think we are doing it way too soon,
  and we shouldnt do this "curriculum learning" at all - lets keep dense rewards. Anything else we are
  swicthing off like this ?" Answers: card floor "Constant 0.005 forever"; PufferLib's priority beta "Keep
  PufferLib's default"; Claude games on the new C seed-0 run "Yes, same as S4_C_s0".
- **Schedules in the S4 runs (checked in the code):** the shaping fade (0.3 / 0.2 / 0.01 to 0 over 500M), the
  D100 floor fade (0.005 to 0 over 5B), and PuffeRL's prioritized-replay exponent beta = 0.2 + 0.8 · 0.8 ·
  epoch / total_epochs (0.2 to 0.84 over the run; a library default, also in stage 3, kept). Fixed: lr 3e-4,
  ent_coef 0.01, MMD 0.05 (C), the deck mix, the opponent fractions, gamma 0.999.
- **Runs `S4d_*`:** identical to `S4_*` (C and A, seeds 0 and 1, s0 on h100|h200 and s1 on a100, 10B steps, 13
  links of 20 h, evaluations, W&B every 10th epoch, Claude games on S4d_C_s0 every 250M) except
  `--shaping-anneal-frac 0` (the shaping weights stay at 0.3 / 0.2 / 0.01 for the whole run; potential-based, so
  the optimal policy is unchanged) and `--card-floor-steps 0` (the floor stays at 0.005). `hpc/stage4_submit.sh`
  accepts SHAPING_STEPS=0 for this. S4 vs S4d compares "both schedules" against "no schedules" (the two
  changes are not separated).

### D103 — Stage 4: the four league + MMD runs (method C) are stopped (user's decision, 2026-10-20)
- **Why:** all four C runs collapsed (the tile choice became deterministic, then the card set narrowed): S4_C_s0
  at ~1.65B, S4_C_s1 at ~1.9B, S4d_C_s0 at ~2.2-2.5B, S4d_C_s1 sliding at ~2.7-2.9B; none of the four A runs did.
  Evidence in EXPERIMENTS "Stage-4 collapse diagnostics".
- **User** (2026-10-20): "Lets cancel the 4 "C" runs - along with the LLM related and other related runs ONLY
  for "C". All the "A" runs SHOULD remain the way they are".
- **Done:** 43 queued / running jobs cancelled by id (the remaining links of S4_C_s0, S4_C_s1, S4d_C_s0,
  S4d_C_s1 and both Claude chains, S4_C_s0-llm and S4d_C_s0-llm); finished at 13:11. The C-only 2B evaluations
  had already finished. Run directories, snapshots and logs are kept (the ablations branch from them).

### D104 — Stage 4: collapse ablation `S4x_*` and a placement-entropy term (user's decision, 2026-10-20)
- **User:** the three ablations "MMD off; snapshots off; a placement-entropy fix", with the control arm and
  the small code change for the fix ("I like the small code change"), each run isolated from the others.
- **Design:** four branches of S4_C_s0 at snap_94446 (1.547B steps; healthy, ~0.1B before its onset),
  `hpc/branch_run.py`: learner, MMD reference and the 64-snapshot pool copied into each run directory (no
  shared files), step-based schedules continued at the snapshot's step, Adam moments fresh (not saved with
  snapshots), PFSP stats fresh. One 20 h link each (~0.9-1.1B steps, to ~2.4-2.6B). Arms:
  `S4x_ctrl_s0` (unchanged), `S4x_nommd_s0` (mmd_coef 0), `S4x_selfonly_s0` (self_play_frac 1.0, MMD on),
  `S4x_posent_s0` (ent_coef_pos_unweighted 0.01, league + MMD).
- **Code (8401f59):** `pufferroyale/trainer.py` placement_entropy_unweighted + config key
  ent_coef_pos_unweighted (royale.ini, default 0 = off; logged as losses/entropy_pos_unweighted for every
  run, without grad at 0, so other runs train exactly as before); tests in `hpc/tests/test_placement_entropy.py`.
- **W&B layout (8401f59, user's request):** league runs log their matches in `league/` (score vs self, vs all
  snapshots, by snapshot age 0-16M / 16-64M / 64-200M / 200M+, pool p median / min / max, pool counters) and
  `league_snapshots/` (per played snapshot) instead of `environment/league/*` and `results/<spec>/*`;
  best_response.py unchanged. Applied to the A runs' pinned checkouts (s4_d70c336 -> 2be616d, s4d_f7445af ->
  b54b511; league_train.py only, no A job was running) so their next links log it.
- **Pre-existing test failure (not caused by this change):** tests/builder/test_wp_trainer.py::
  test_reward_clip_values also fails on s4d_f7445af (the code of every stage-4 run since launch).
