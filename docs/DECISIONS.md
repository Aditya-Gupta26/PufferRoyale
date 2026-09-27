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
