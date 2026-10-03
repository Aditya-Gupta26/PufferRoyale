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
