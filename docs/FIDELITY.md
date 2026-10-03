# PufferRoyale fidelity notes

This file is the authoritative list of (1) every choice the SPEC marks **[IMPL-DEFINED]** or
leaves open, (2) every place where the engine is simpler than, or differs from, RoyaleSim's
calibration ledger (`data/source/royalesim-calibration.json`, cited as `section.KEY`), and
(3) known divergences from the real game. It tracks **SPEC v0.4.2** (engine state version 4:
Phase E -- the 43 cards of SPEC §16, ids 21-63, the three non-default tower troops and the v0.3
observation -- plus the second audit's engine amendments of SPEC §18). Sections 1-11 describe the
engine as a whole; §12 lists every Phase E choice card by card; §13 the §18 amendments; §14 the
env / observation choices of the SPEC §19 (v0.5-G) work package (the engine is unchanged by it). All
choices are deterministic.

Legend: **[IMPL]** = an implementation choice where the SPEC allows one (the Phase-A choices
were approved by the orchestrator); **[PIN]** = behaviour fixed by a SPEC v0.2 §13
clarification, restated so code and SPEC can be compared; **[DIV]** = a known simplification /
divergence versus RoyaleSim or the live game.

---

## 1. Units, frames, determinism (SPEC §1)

- Positions are `int32` millitiles, durations ms, speeds millitiles/tick. The simulation
  headers contain no floating point (enforced by `tests/builder/test_symmetry_and_source.py`,
  which greps them, `pr_bots.h` included). Integer division is explicit (`pr_floordiv`,
  `pr_ceildiv`); truncation is used only where it is the rule (movement components truncate
  toward zero, SPEC §6.3). Floats appear only in the observation encoder (`pr_obs.h`), the
  env's rewards/logs (`royale.h`) and the renderers (`pr_render.h`).
- State hash: FNV-1a 64 over the raw bytes of `PrState` (a POD of fixed-width integers with
  natural alignment). Every byte (padding included) is zeroed at reset and freed pool slots
  are re-zeroed, so the hash depends only on live content. **[IMPL]** Hashes are comparable
  within one ABI (LP64 clang/gcc on arm64/x86-64 lay the struct out identically).
- RNG streams (PCG32): game `0x5052` (deck shuffles and random decks; inside the state),
  scripted bot `0x424F54`, learner side `0x534944` (both inside the env, never the game's).
- Coordinates entering through the debug API (`Game.spawn`, the entity poke hook) are clamped
  to `+-128000` (4 arena heights) and `Game.play` taps outside the arena are refused, so every
  later frame conversion (`pr_own_x = W - x`) and distance stays far from int32/int64 overflow;
  `Game.restore` enforces the same bounds (§4).
- **Seat symmetry [IMPL]** (SPEC §13.13). Tile centres lie on half-cell boundaries and
  half-open intervals do not rotate onto themselves, so every point query a *moving unit*
  makes is team-oriented: team 0 reads cells as `[a, b)`, team 1 as `(a, b]` (its own frame).
  "Wet" (water for ground units) = water under *either* convention, so the water's boundary
  lines count as water; water ejection is computed in the unit's own frame; unit centres are
  clamped to `[1, W-1] x [1, H-1]`; lane ties go own-left (§6.1). A scenario played by team 0
  and the same own-frame scenario played by team 1 evolve as exact 180-degree rotations
  (tested for all 64 cards over 700 ticks, for 63 two-seat card pairings with the alternating
  play order -- exact including creation order --, for the three tower troops, and for the env
  observations: `tests/builder/test_symmetry_and_source.py`), and by a bot-driven mirror fuzz over
  random 64-card decks, all tower troops and mirrored debug spawns (`tests/c/test_mirror.c`, after
  the second audit's `mirror.c`: 0 divergences; see §13).
- **Simultaneous plays [PIN]** (SPEC §13.13): when both teams play on the same tick, team 0's
  queued plays are applied first (so its units get the lower creation ids, which then win the
  SPEC's lowest-id ties). This is the one permitted seat-dependent rule.
  **Opt-in alternative [IMPL]**: `Game(..., alternate_first=True)` / `Royale(...,
  alternate_first=True)` alternates the first team after every tick on which both teams play
  (state field `first_team`, reset to 0 each match; the flag survives `reset`; a tick counts
  when both teams have an accepted play in it). With it, a
  two-seat scenario and its mirror (roles swapped, the other team first) evolve as exact
  rotations *including creation order* (`tests/c/test_rules.c:
  test_alternating_order_mirror_is_exact`, 500 ticks of interacting units). It is off by
  default because SPEC v0.2 fixes team-0-first.

## 2. Card database (SPEC §2)

- Level 11 = `floor(base * 256 / 100)` for every scaling stat, including spawned units,
  death damage, death bombs, dash damage, variable-damage stages, `damage_special`, projectile,
  area and buff (damage-per-second) damage. Towers and the tower troops use the compounding
  ladder on the *percent* (ledger `combat.TOWER_HITPOINT_LADDER`): pct(1)=100,
  pct(L)=floor(pct(L-1)(100+r)/100), r = 8 (Princess hp and damage, King damage, the tower
  troops) or 7 (King hp) for L <= 9, 10 above; stat = floor(base * pct / 100) -> Princess
  3052 / 109, King 4824 / 109, Cannoneer 2616 / 272, Dagger Duchess 2768 / 91, Royal Chef
  2703 / 109. A tower's projectile carries the tower's ladder damage.
- `tools/gen_card_db.py` asserts at generation time every SPEC §2 example and hand-checked
  level-11 literals for all 64 cards: the hp / damage of every unit a §16 card summons, spawns
  or death-spawns (39 units), death damage and death bombs, charge / dash / variable-damage /
  burrow / collector columns, spawner and death-spawn blocks, every new projectile, area and
  buff number, the tower troops, and that every formation stays within 1600. Ids 0-20 and
  the v0.2 unit / projectile / area / buff indices are unchanged (the v0.1 cards and crown
  towers are registered first).
- Ranged units' damage comes from the projectile row (the data's `damage_source`). A unit with
  a `CustomFirstProjectile` (the Princess) fires that projectile (SPEC §16.6.1: 66 -> 168,
  splash 2000); its 5 `PrincessProjectileDeco` volleys are cosmetic and not modelled.
- `charge_range_raw` 250 is read as centitiles (ledger `charge.CHARGE_RANGE_UNIT`) = 2500.
- Spell damage per hit/wave: Fireball 688, Arrows 122 (per wave), Zap 192, Log 268; crown
  percents 25 / 20 / 25 / 13 from the rows.
- **[DIV]** Columns read but not modelled: `stop_movement_after_ms` / `wait_ms` (the Giant's
  and Ice Golem's stomp pause, ledger `movement.STOMP_*`), `spawn_radius_milli`,
  `projectile_start_radius` of melee units, jump height/speed (jumpers cross water at their
  normal speed), `multiple_projectiles` of Arrows.

## 3. Arena and placement (SPEC §3, v0.2)

- The 2018 tilemap (ledger `arena.ARENA_SOURCE_VINTAGE`); the generator asserts the river and
  bridge geometry and the tower centres of SPEC §3.
- Legality per own-frame tile, exactly SPEC v0.2 §3.1 (one context, `PrLegalCtx`, holds the
  living towers and buildings, so the mask and `play` share every test):
  1. **Troops**: not water, not nodeploy (any half-cell), no positive-area overlap with any
     *living* tower **or building** footprint (either team), and troop territory: own half
     (own `ty >= 17`) or an opened pocket (tile centre not in the river band and not inside
     the closed rect of any living enemy crown tower, King 18x16, Princess 11x21). The river
     band stays closed even on a fallen lane's bridge (as RoyaleSim; unsourced there).
  2. **Buildings**: own half only; the whole footprint inside the arena and the own half; no
     water / nodeploy half-cell under it; no positive-area overlap with a living tower or
     building. **Anchor**: odd F (Cannon, F = 3) at the tile centre; even F (Tesla, F = 2) at
     the tapped tile's own-frame top-left corner `(tx*1000, ty*1000)`, covering own tiles
     `[tx-1, tx] x [ty-1, ty]`. Legality and placement both call `pr_building_anchor`, so the
     placed footprint is exactly the one the legality check saw.
  3. **Fireball, Arrows, Zap, Rocket, Poison, Freeze, Earthquake, Lightning, Giant Snowball**:
     any tile. 4. **Goblin Barrel**: any non-water tile.
  5. **The Log, Barbarian Barrel** (data `can_deploy_on_enemy_side = false`): troop territory
     (own half or an opened pocket), not water; may overlap buildings, towers and nodeploy cells.
  7. **Miner** (SPEC §16.6.3): any non-water tile with no positive-area overlap with a living
     crown-tower footprint of either team (enemy half included; building footprints and
     no-deploy cells allowed).
  The class is generated per card (`PrCardDef.placement`: data kind, `can_deploy_on_enemy_side`,
  the Goblin Barrel's spawn projectile, the Miner's `spawn_pathfind`); every §16 building has
  an odd footprint (F = 3: Tombstone, Inferno Tower, Mortar, X-Bow, Elixir Collector, Bomb
  Tower) and anchors on the tile centre.
  6. A destroyed tower's footprint blocks nothing (troops, buildings, pathing occlusion).
- **Placement point [IMPL, approved]**: `play_tile` and every env action place at the tile
  centre (buildings at their anchor). `Game.play(team, slot, x, y)` treats (x, y) as the
  literal tap point: legality is that of the tile containing the tap *in the acting team's
  own frame* (rotation, then floor division, so a boundary tap is judged seat-symmetrically:
  team 0 at y = 17000 and team 1 at y = 15000 are both on their tile 17); troops (their
  formation) and spells are placed at the tap point itself; buildings at that tile's anchor.
  A troop tapped on the water's boundary line is moved onto dry land by the formation rule.
  (RoyaleSim's shipped `placement.TAP_SNAP = none` also uses the raw tap.)
- Error precedence [IMPL]: GAME_OVER, BAD_SLOT, LOCKOUT, NOT_ENOUGH_ELIXIR, ILLEGAL_POSITION.
- **[DIV]** Not modelled: RoyaleSim's illegal-tap relocation (`placement.ILLEGAL_TAP`), the
  half-open King block for troop taps (`placement.TROOP_TOWER_TAPS`), the Log's
  clamp-to-legal-edge (`spells.ILLEGAL_SPELL_TAP`). Illegal taps are refused.

## 4. Match rules (SPEC §4)

- Elixir exactly as SPEC: +50/+100/+150 units per tick by tick index, start 16800, cap 28000
  with the excess counted as leaked (`state()['leaked']`).
- **Plays [PIN §13.1, §14.1]**: a play queued when `state()['tick'] == t` is checked against
  the *current* state (hand, elixir, tick, board) and the code is returned; accepted plays are
  applied in Upkeep of tick `t` (the next `tick()`). There, **every** play of the tick is
  validated against the state at the start of the Upkeep (after regen, before any play of this
  tick is applied): a team's own queued plays in queue order against its own running elixir and
  hand, while plays of different teams never invalidate each other (a troop landing where the
  enemy places a building in the same tick is resolved by collision; the auditor's pocket case).
  The validated plays are then applied, team 0's first (§1). A play that fails (its slot now
  holds another card after an earlier play of the same team cycled the hand, the team's elixir
  ran out, or a debug setter changed hand / board / tick) is dropped and counted in
  `state()['dropped_plays']` and in the env log key `dropped_plays`. The legal mask is the same
  test on the same state, so every mask-legal env action is applied (`dropped_plays` is 0 in env
  play; checked in `tests/builder/test_audit_c1.py`). The debug API may queue up to 4 plays per
  team per tick.
- Lockout: a play applied in tick `t` needs `t >= deploy_lockout_ticks`.
- King activation: trigger in Resolve (any damage > 0 to the King after reductions) or Reap
  (own Princess destroyed); the 71-tick countdown is decremented in Upkeep, so a trigger while
  processing tick T activates the King from tick T+71.
- Judge exactly as SPEC; **[PIN §13.8]** `end_reason = "DRAW"` for every drawn result (both
  Kings on one tick, or an exact tiebreak tie), `"TIEBREAK"` only for a decisive tiebreak;
  `result = [0, 0]` and `end_reason = None` while running.
- **Debug setters [IMPL, approved]**: `set_tower_hp(team, idx, hp <= 0)` destroys the tower at
  once with all side effects (crowns, King trigger, pocket) and runs the King / crown end checks
  immediately (never the tiebreak); hp above max is clamped; a destroyed tower cannot be
  revived (ValueError). `set_hand` sets hand = order[0:4], queue = order[4:8], discards the
  team's queued plays and makes those 8 distinct cards the team's deck **for later deals too**
  (a `'random'` deck stops being redrawn).
- **Reset**: `reset()` re-deals the *current* decks (the constructor's, or the ones `set_hand`
  installed; `'random'` decks are redrawn) and keeps the configuration (lockout, tiebreak, play
  order); `reset(seed=None)` continues the game RNG stream, `reset(seed=s)` reseeds it first.
  So `reset(seed=s)` equals a fresh `Game(seed=s)` built with the same decks and configuration
  -- not, after a `set_hand`, one built with the constructor's decks.
- **Restore [PIN §14.5, §18.5, §18.8]**: `Game.restore(bytes)` validates a private copy with
  `pr_state_check` before committing it: magic/version, tick in [0, 6000] (a running match below
  6000), every pool count within capacity, entity ids strictly increasing and below `next_id`
  (with id headroom for the rest of the match), every unit / projectile / area / card / buff index
  in range and each entity's kind matching its unit, crown-tower slots holding the configured
  towers and crowns matching the fallen princess towers, 8 distinct deck cards with hand + queue a
  permutation of them, pending plays well-formed, pending unit spawns naming a valid card and a
  non-tower unit, and tight bounds on everything the engine multiplies or accumulates:
  `max_hp` <= the unit's level-11 value (level 12 once Chef-levelled; so always <= the level-12
  value, §18.8b), hp <= max_hp, the lifetime counter <= `lifetime_ms/50 + 1` (0 without a
  lifetime), `load_ms` <= `load_time`, attack progress <= `tick x 10050` and charge run-up <=
  `tick x 1000` (sound per-tick growth bounds), role timers (Collector / Chef / Duchess / dash)
  and spawner timers within their unit's periods, hide timers within the Tesla's, projectile /
  effect damage <= 100,000, pulse-area timers, spent <= `tick x 4 x 10` elixir, plays / dropped
  plays <= 4 / 8 per tick. The engine keeps every bound: a validated state never evolves into an
  invalid or overflowing one (§18.5; `tests/c/test_fuzz.c` re-checks every accepted mutated
  snapshot after each of 25 ticks -- 0 of 93,892 accepted states in a 200,000-flip run became
  invalid, also clean under ASan/UBSan). The statistics counters `leaked` / `spawn_overflow`
  saturate at 10^9 instead (never reached in play). Content the engine never reads (unused pool
  slots, stale pending plays, hit-once bits of empty slots) is not rejected but zeroed after
  validation (`pr_state_canonicalize`), so the hash depends only on live content. Invalid input
  raises `ValueError` and leaves the game unchanged. `tests/c/test_fuzz.c` (also under ASan/UBSan) flips random bytes
  of real snapshots: rejected, or simulated / observed / masked / rendered / bot-played for 25
  ticks without a sanitizer report (60,000-flip run: 7,118 accepted, all clean); a state that
  passes stays valid through real play.
- `DECKS` maps the nine preset names (SPEC §0 and §16.4) to tuples of card ids; `'random'` is
  accepted by the constructor (8 distinct cards drawn uniformly from all 64 at every reset) but
  is not a preset.
- **Tower troops** (SPEC §16.3): `Game(..., tower_troop0=, tower_troop1=)` /
  `Royale(..., tower_troop0=, tower_troop1=)` take `princess` (default), `cannoneer`,
  `dagger_duchess` or `royal_chef` (anything else: `ValueError`, §16.6.27); the choice is
  configuration (state field `tower_troop`, kept by `reset`, carried by snapshots) and replaces
  both Princess towers of that team; `state()['towers'][t][i]['max_hp']` and the observation's
  tower hp fractions use the tower's own max hp; `state()['tower_troops']` names them.

## 5. Tick pipeline (SPEC §5)

- Phases in SPEC order; spell objects advance in the Projectile phase right after the
  projectiles.
- **Deploy [PIN §13.2]** (ledger `movement.DEPLOY_TIMING` = spawn-anchored): a unit created
  during tick `t` shows `deploying` in the observations taken at the end of ticks
  `t .. t + D/50 - 1` and targets/moves/attacks from tick `t + D/50` (its deploy timer does
  not run on the creation tick). Staggered member `i` behaves as if created at
  `t + i*delay/50`. The load timer runs every tick including the creation tick.
- **`spawn(deployed=True)` [PIN §13.10]**: not deploying, `load_timer = max(0, load_time -
  deploy_time)`; a Tesla starts hidden.
- Spawn phase: death spawns, death bombs, Goblin Barrel goblins and the Barbarian Barrel's
  Barbarian queued during tick T are created in the Spawn phase of tick T+1 (and are inactive
  for their own deploy time from there); periodic-spawner waves are queued in the Status phase
  of tick T and created in the Spawn phase of the same tick (it runs right after Status).

### 5.1 Formation [IMPL]
- Offsets are in the acting team's own frame (rotated for team 1), precomputed as integers by
  the generator (no trigonometry at runtime).
- n=1: the point; n=2: (-500, 0) = member 0, (+500, 0) = member 1; n=3: (0,-577),
  (+500,+289), (-500,+289) for every 3-member card (Minions, Skeletons, Goblin Barrel).
  **[DIV]** RoyaleSim scales the ring by SummonRadius (Skeletons stand 807 from the tap).
- n=4 (Goblins, Royal Hogs): a 2x2 square, members at (+-500, +-500); n=5 (Barbarians, Bats):
  a ring of radius 800 from own-frame "up"; n=6 (Minion Horde, Goblin Gang): two rows of three,
  1000 apart, the front row (members 0-2) at y = -500 -- Goblin Gang's three stab goblins in
  front of its three spear goblins; Rascals use the n=3 triangle (the boy at the apex). Cards
  with a `second_summon` place the summon's members first, then the second summon's.
- n=15 (Skeleton Army): one member on the point, 6 on a ring of radius 800 (from own-frame
  "up", clockwise, 60 degrees apart), 8 on a ring of 1500 (offset 22.5 degrees).
  **[DIV]** RoyaleSim's spiral reaches ~2.8 tiles; the SPEC bounds it by 1600 at spawn time
  (§13.14).
- Stagger: member k deploys for `deploy_time + k * summon_deploy_delay_ms` (Archers,
  Minions: 100 ms; for a card whose `summon_deploy_delay_ms` is blank, e.g. the Rascals, its
  `summon_deploy_delay_second_ms`); a staggered member is an ordinary deploying unit
  (targetable, pushed).
  **[DIV]** RoyaleSim's untargetable stagger wait (`formation.STAGGER_WAIT`) is not modelled.
- Ground members outside the arena are clamped into it; **[PIN §14.4]** members that land in
  water go straight across to the bank on the tap's side: the own bank for a tap on the own
  half, the far bank for a tap in a pocket -- never across the river (reachable only through
  `Game.play`'s raw taps near the water line, e.g. a Skeleton Army tapped on the river edge).
  Members that land on a bridge deck are on land and stay there. Goblin Barrel goblins,
  knockback and collisions still use the nearest-dry-point ejection (own bank, far bank,
  bridge decks).

## 6. Units and buildings (SPEC §6)

### 6.1 Targeting
- Formulas exactly as SPEC: sight test `dist <= sight + r_t` (+2000 vs crown towers; the
  attacker radius is not added, **[PIN §13.5]**), attack test `dist <= range + r_a + r_t`,
  preference `dist - r_t` then lowest id -- except that two exactly tied crown towers go by the
  attacker's own-frame x, own-left first **[PIN §18.1]** (tower ids follow the engine frame, so the
  id tie-break favoured engine-left for both seats).
- Troops keep the current target while it is valid and within `sight + r_t (+2000) + 500`
  unless a strictly closer candidate appears while `progress == 0` (so a unit mid-cycle does
  not rescan at all). Buildings/towers acquire and keep only inside attack range. Deploying,
  stunned and hidden (Tesla under ground / rising) entities hold no target; a dormant King
  holds none.
- **Mid-swing reach loss [PIN §14.2]** (ledger `targeting.LOGIC_PRESERVE_TARGET_IF_HIT_STARTED
  = projectile_attackers_only`, with `combat.RETARGET_PROGRESS = keep_when_dead_or_in_reach`):
  when a hit is started (`progress % hit_speed > 50`) and the target is out of attack reach,
  a *direct striker* (no projectile: melee units, Tesla) switches to the nearest valid enemy in
  reach and keeps its progress, so the hit lands on schedule on the new target (the auditor's
  Knight / Hog Rider / Cannon case); with nobody in reach the swing is cancelled
  (`progress = 0`) and the usual acquisition resumes, so a melee hit never lands out of reach.
  A *projectile attacker* (troops, Cannon, crown towers) holds a target that left its reach
  while the distance stays within `reach + 500` -- in the Target phase as well, overriding the
  sight hysteresis and the structures' "keep while in range" -- and fires at it from beyond
  reach; farther away the swing is cancelled. Tested in `tests/c/test_combat.c:
  test_mid_swing_reach_loss`. **[DIV]** the ledger's control run (Knight, Hog Rider leaving, no
  other enemy in sight) shows the Knight completing the hit 473 beyond reach; SPEC §14.2
  cancels it, and so do we.
- Scans are exact but fast: per-team bucket grids (2x2-tile blocks) rebuilt at the start of
  the Target phase, crown towers always offered, a square-root-free bound against the
  incumbent (the current target seeds the search) and an explicit lowest-id tie-break, so the
  result equals the plain full scan's (state hashes unchanged by the rewrite).
- **Minimum range [PIN §16.2]** (Mortar): a candidate closer than `minimum_range + r_a + r_t`
  is skipped by every acquisition scan, and a held target that comes inside it is dropped in
  the Target phase (before the §14.2 mid-swing hold is considered).
- **Non-attackers**: a unit whose data has neither damage nor a projectile (Tombstone -- its
  row only has `HitSpeed` 10000 --, Elixir Collector) never targets or attacks
  (`PrUnitDef.no_attack`); `card_info` still reports the data's hit speed.
- **Lane [PIN §13.6]**: decided in the unit's own frame: own-frame x < 9000 = own-left, else
  own-right; exactly 9000 = own-left (ledger `targeting.CENTRE_LANE_FRAME =
  own_frame_tie_left`). Default tower = the enemy tower in that own-frame lane.
- **[DIV]** Not modelled: projectile-attacker target hold and doomed-target drop
  (`targeting.DOOMED_TARGET_DROP`, so towers can overkill), the post-kill retarget pause
  (`combat.POST_KILL_RETARGET_WAIT`), the 8th-frame acquisition delay of death spawns.

### 6.2 Attack cycle (ledger `combat.ATTACK_CYCLE = progress_credit`)
- Exactly the SPEC/ledger counters (load timer, progress, `progress % hit_speed > 50` started
  test, fresh-cycle credit, hit on crossing a multiple). A slowed unit advances
  `50 * (100 + hit_pct) / 100` per tick (35 under a 30 % slow). The `LoadTime > HitSpeed`
  branch of the ledger is implemented (no card uses it).
- **[IMPL, approved]** On a target change, if the new target is **not** in attack range the
  swing is cancelled (progress = 0); if it is in range the progress carries on (a tower killing
  a swarm keeps its 16-tick cadence; ledger `combat.RETARGET_PROGRESS =
  keep_when_dead_or_in_reach`). A started hit whose target left reach follows §14.2 (§6.1).
- Melee/instant hits are buffered damage; area melee (Valkyrie) centres on the attacker if
  `self_as_aoe_center`, else on the target, and filters by the attacker's air/ground flags.
- **Variable damage [PIN §16.6.9]** (Inferno Tower, Inferno Dragon): a per-entity counter of
  hits landed on the current target; hits 1-5 deal stage 1 (`Damage`), 6-10 stage 2
  (`VariableDamage2`), then stage 3 (`VariableDamage3`) (5 = `VariableDamageTime / HitSpeed` =
  2000 / 400); the counter restarts on any target change (including the §14.2 mid-swing switch
  and the loss of the target) and on a stun / freeze. IT 43 / 158 / 847, ID 35 / 120 / 422.
- Kamikaze (Ice Spirit): the unit deals itself hp + shield when it fires and dies in that
  tick's Reap; its projectile flies on.
- Charge (Prince): the actual distance walked accumulates each walking tick; >= 2500 charges
  it (speed x2, next hit 783 with progress snapped to the next multiple of hit speed). A tick
  without walking resets it, as do a landed hit, a stun and a knockback. **[DIV]** ledger
  `charge.ACCUMULATOR` (permille of min(step, 250)) not modelled: our charge completes on the
  42nd walking tick, RoyaleSim measures the 43rd.

### 6.3 Movement & pathing [IMPL]
- Reverse Dijkstra distance fields over the 36x64 half-tile grid, 8-connected (10 / 14), no
  corner cutting past impassable cells; water impassable for non-jumpers; cells whose centre is
  inside a living tower/building footprint cost 6x to enter (ledger
  `pathfinding.OCCLUDED_CELL_TREATMENT = cost_50` over 8), except the goal's own footprint.
  Goal set: the goal structure's footprint cells, or, for a chased troop, every passable cell
  of its **chase block** (2x2 tiles = 4x4 cells, a rotation-symmetric grid); a unit that is
  inside the block without sight of the target routes to the target's own cell. Quantising
  troop goals lets all chasers of nearby targets share one field and keeps a field valid while
  its target moves inside the block: the auditor's 216-Skeleton river fight dropped from 713
  Dijkstra builds per 300 ticks to 42.
- Waypoint: straight at the goal if the segment is clear (sampled every 125 millitiles; the
  start cell and the goal cell/footprint are exempt); otherwise follow the field's
  steepest-descent chain (ties in the mover's own-frame neighbour order) up to 8 cells and aim
  at the farthest chain cell in line of sight. Recomputed when the unit's cell, its goal key or
  the structure set changes (per-entity cache inside the state, holding 32 bits of the
  structure signature).
- Fields are cached outside the state (24 LRU slots of 4.6 KB; the Dijkstra heap is stack
  scratch), keyed by goal, jumper flag and a 64-bit FNV signature of the living structures; the
  cache is a pure function of the state (a cold-cache replay gives identical hashes -- tested,
  and the auditor's `cachepure.c`, which randomly evicts / resets the cache every tick, stays
  identical). 24 slots are kept: typical play builds 21 fields in 40 recorded matches with 12,
  16 or 24 slots alike, but the 216-Skeleton stress needs ~24 (16 slots: 144 builds, 8.7k
  ticks/s; 24: 42 builds, ~13k ticks/s). `PrPathCache` is 120 KB (was 130 KB), `PrGame` 174 KB
  (was 198 KB).
- A cell is occluded when its centre lies strictly inside the footprint (an open square, its own
  rotation; placed buildings and towers never have an edge on a cell centre, so only off-grid
  debug buildings are affected, §18 NIT).
- Every dry cell is reachable (occlusion is a cost, never a wall); a sweep of every legal
  own-half tile x 37 ground-and-air unit types (the v0.1 14 and 23 §16 ones) x both seats
  (16,428 lone walks, `tests/c/test_path.c`) reaches attack range of the default tower (worst
  561 ticks, the Golemite).
- Steps are capped at the remaining distance to the waypoint/goal (no overshoot). Flying units
  fly straight at the goal. Jumpers (Hog Rider, Prince) path over water at their normal speed;
  `jumping` is set while they stand on water and they skip ground collisions then.
- **[DIV]** Not modelled: RoyaleSim's hop timing (`movement.JUMP_WATER_HOP`), its 16.402 path
  search (costs 8/5/50/7), the contact/avoidance law and the stomp pause.

### 6.4 Collision [IMPL]
- One simultaneous unit-unit pass per tick (displacements from start-of-pass positions):
  ground-ground and air-air pairs split the overlap by inverse mass. Deploying, attacking and
  stunned troops take part. A unit's summed displacement is capped at its own radius per tick.
  Pairs are enumerated by a sweep over tile rows (each pair's contribution is independent and
  integer sums commute, so the result equals the all-pairs loop's).
- Coincident centres: different teams each move toward their own side; same team, the lower id
  moves to its own-left, the higher to its own-right.
- Then ground troops are pushed fully out of every structure's collision circle, sequentially
  over structures in id order; then the arena clamp; then wet ground non-jumpers are ejected
  to the nearest dry point.

### 6.5 Buildings
- §16 buildings (Tombstone, Inferno Tower, Mortar, X-Bow, Elixir Collector, Bomb Tower) use the
  same lifetime drain, footprint and occlusion rules; their mechanics are in §12.
- **Lifetime [PIN §13.4]**: elapsed tick k = 0 is the first non-deploying tick (`t + D/50`);
  per tick the SPEC drain `floor(M(k+1)50/L) - floor(Mk50/L)`; hp reaches 0 and the building
  is reaped at the end of tick `t + D/50 + L/50 - 1` (Cannon 600 ticks after deploy end,
  Tesla 500). **[DIV]** RoyaleSim measures a truncated hundredths rate (Tesla +1 tick,
  Cannon +2).
- Tesla hide (ledger `hide.*`): visible while deploying, hidden from deploy end; rises for
  `up_time_ms` (targetable, cannot attack) when a valid enemy passes its sight test; up
  afterwards; hides again after `hide_time_ms` without a target. A stunned hidden Tesla does
  not start rising. Hidden = untargetable and immune to all damage except the lifetime drain;
  area buffs with `affects_hidden` still land.

### 6.6 Deaths
- Death damage victims are chosen at the moment of death (Reap of tick T) and hit in the
  Resolve of tick T+1 (a Royal-Chef-levelled unit's death damage is levelled too).
- Death spawns and death bombs (§12) fire on any death, the lifetime expiry included
  [PIN §16.6.7].
- **Ice Golem death area [PIN §13.7]**: applied once, at death (ledger
  `spells.ONE_SHOT_AREA_EFFECT_APPLICATION = first_update_only`): the 2000 ms slow lands on
  every enemy (hidden included) within 2000 + r at that moment; no lingering object and
  nothing in `effects()` (Zap likewise).

### 6.7 Status effects [PIN §13.3]
- One slot per buff row storing the tick index from which the buff is gone
  (`buff_until`): a D ms buff applied during tick t is active in every later phase of ticks
  `t .. t + ceil(D/50) - 1` and is removed in the Status phase of tick `t + ceil(D/50)`.
  So Zap (applied in Upkeep of cast tick P, 500 ms) blocks P..P+9 and the unit acts again in
  P+10; an Ice Spirit freeze (1100 ms) applied in the Projectile phase of tick N lets the unit
  act again in N+22; `entities()` reports `stunned` / `slowed` exactly while active.
  Re-application keeps the later end.
- The strongest (most negative) speed and hit-speed multipliers apply; a -100 speed buff is a
  stun/freeze. On application: progress 0, charge reset, target cleared (retarget on resume),
  the variable-damage stage back to 1, a Bandit's dash (stand or move) cancelled.
  Stun does not pause deploy timers, lifetimes or the King countdown (ledger `status.*`); it
  pauses periodic spawners (ledger `spawner.STUN_PAUSES_SPAWNER`), the Dagger Duchess's reload
  and the Royal Chef's cooking, not the Elixir Collector [IMPL].
- New buff rows (data): `Poison` (-15 % speed, 36 dps -> 92, crown 23 %, stacking) and
  `Earthquake` (-50 % speed, 32 dps -> 81, building 350 %, crown 60 %, stacking); the Freeze
  spell reuses the Ice Spirit's `Freeze` row (both -100), Lightning `ZapFreeze`, Snowball
  `IceWizardSlowDown`. `slowed` = an active buff with -100 < speed < 0.

## 7. Projectiles & spells (SPEC §7)

- **Unit/tower shots [IMPL]**: created on the hit tick at the start radius toward the target
  (visible in `projectiles()` right after the hit tick), first step on the next tick (ledger
  `combat.PROJECTILE_LAUNCH = start_radius_next_tick`). Homing toward the target's current
  centre; impact when the remaining distance <= speed. A single-target shot at a dead target
  vanishes; a splash shot flies on to the last known point and splashes there.
- **Non-homing shots [PIN §16.6.2]** (data `homing = false`: Princess, Mortar, Bomb Tower,
  Bomber, Wall Breakers): aimed at the target's centre AT LAUNCH (`has_target = 2`), they never
  re-aim, land there and splash there, so a moving target can dodge them; a single-target aimed
  shot (none in the roster today) would hit its target only if it still stands within its own
  radius of the landing point.
- Splash: every valid enemy within `radius + r_target` of the impact point, filtered by the
  projectile's `aoe_to_air` / `aoe_to_ground`; the projectile's `target_buff` (Ice Spirit
  freeze) lands on every splash victim (ledger `status.TARGET_BUFF_ON_SPLASH`).
- **Spell timing [PIN §13.12]**: a spell play is applied in Upkeep of tick P. Fireball and
  Goblin Barrel are point projectiles from the caster's King centre that already move in the
  Projectile phase of tick P and land in tick `P + ceil(d / v) - 1`; the goblins are committed
  in the Spawn phase of landing tick + 1 and deploy for 1100 ms. Fireball knockback 1000
  radially away from the impact point for troops without `ignore_pushback` (air troops too).
- Arrows **[IMPL]**: an effect standing at the tap point (visible in `effects()`), not a flying
  object; the first wave lands in tick `P + ceil(d / 1100) - 1`, the next ones 4 and 8 ticks
  later.
- Zap: damage buffered and stun applied during Upkeep of tick P (at the tap point for `play`,
  the tile centre for `play_tile` / env actions); damage resolved in that tick's Resolve.
- The Log: starts rolling at the tap point (**[DIV]** RoyaleSim's airborne `LogProjectile`
  phase is not modelled), is tested at its current position every tick, then advances
  min(200, remaining), 10100 in total. Rect-vs-circle test, ground enemies only, each hit once
  **[PIN §14.3]**: the hit-once memory is a bitset over all 256 pool slots (remapped when the
  pool is compacted), so no entity can go unrecorded (before: a 64-id list, after which extra
  entities were hit every tick; the auditor's 65-100-Cannon repro now hits each exactly once);
  knockback 700 **along the roll direction** (SPEC) for troops, `pushback_all` so heavy troops
  are pushed too. **[DIV]** RoyaleSim measures a radial push (`knockback.DIRECTION_ROLLING`).
  **[IMPL]** A log that rolls to or past the arena edge (`y <= 0` rolling up, `y >= 32000` rolling
  down: the closed test is seat-symmetric) stops there.
- Knockback **[IMPL]**: instantaneous, clamped to the arena and to land; resets the attack
  cycle and the charge; keeps the target; cancels a Bandit's dash; never moves a burrowing
  Miner. A knockback of 0 (the Barbarian Barrel) does nothing at all (no attack reset).
  **[DIV]** RoyaleSim's measured ladder (`knockback.DISPLACEMENT_LAW = client16402`) is not
  modelled.
- The §16 spells (Rocket, Giant Snowball, Freeze, Poison, Earthquake, Lightning, Barbarian
  Barrel) are in §12.

## 8. Scripted bots (SPEC §8, `pr_bots.h`)

- All bots are integer-only, use their own PCG32 stream and choose from the engine's exact
  legal mask (a final `mask[a]` check returns the no-op otherwise), so they never issue an
  illegal action. In the env the bot decides once per step, before the step's first tick,
  from the same state the learner sees.
- `noop`: never plays. `random`: with probability `bot_play_prob` (kept as parts per million)
  a uniformly random legal non-noop action, else no-op.
- `heuristic` **[IMPL]** (reasons in its own frame; `bot_play_prob` is not used). In priority
  order:
  1. **Finish a tower**: a damage spell in hand (Fireball, Zap, Arrows) whose crown-tower
     damage (Arrows: all three waves) is at least an enemy crown tower's hp is cast on it.
  2. **Defend** when enemy troops stand at own-frame y >= 13000 and their hp exceeds half the
     hp of our troops in our half:
     a. a damage spell on the best cluster if it pays (value = 3 per enemy troop it would kill
        + 1 per other enemy hit, needed >= 2 + cost; the Log is tapped 1.5 tiles behind the
        cluster and scored over its roll corridor, and never counts air units);
     b. against a buildings-only targeter, a building placed centrally (own tile (8|9, 21),
        nearest legal tile);
     c. otherwise the cheapest troop able to hit the most advanced threat (a more expensive one
        when the threat hp >= 2000; buildings-only troops never defend): ranged units behind the
        threatened lane's princess (own (3|14, 27)), melee units 3 tiles in front of the threat
        (own ty clamped to 17..31).
  3. **Offence** (at most one play per 40 ticks unless elixir is about to cap): support our
     most advanced win condition with a troop 2 tiles behind it; with >= 7 elixir (or capping)
     play a win condition on the weaker enemy lane (ties by the bot RNG): Hog Rider / Prince at
     the bridge (own (3|14 +-1, 17)), Giant at own ty 19, Goblin Barrel on the target tower;
     when capping, the cheapest non-spell card on a safe tile (own (3|14, 26), buildings (9, 22)).
  Tile searches spiral outward to the nearest legal tile. Measured: heuristic beats random
  6/6 (C test, both seats) and in every `scripts/eval.py` game; a passive learner loses to it.
- **Seat symmetry [PIN §18.4, §18.8g]**: every scan over towers, lanes and tiles is in the acting
  team's own frame (rule 1 checks the enemy King, own-left, own-right princess in that order;
  entity scans run in id order, which mirrors); a decision depends only on `(kind, seed)` and the
  own-frame state (checked by the mirror fuzz of `tests/c/test_mirror.c`, which compares every bot
  decision of a game and its seat-swapped rotation).
- **Spell valuation**: a damage spell is valued at the tile centre it will really be tapped on
  (rolling spells 1.5 tiles behind the cluster, scored over their corridor from that tap); Lightning
  counts only the targets its bolts would strike (the 3 highest current-hp enemies in range, crown
  towers included but worth nothing).
- **Every card by kind (SPEC §16.4)**: the rules above read card data, not card ids. Win
  conditions = the Prince, every buildings-only troop (Giant, Hog Rider, Golem, Royal Giant,
  Lava Hound, Balloon, Battle Ram, Royal Hogs, Wall Breakers, Ice Golem), the Miner, the Goblin
  Barrel and siege buildings (range >= 10 tiles: X-Bow, Mortar). Damage spells = every spell
  with damage (rolling spells and Earthquake never count air; a pulsing area's and Arrows'
  damage is summed over its events; Lightning counts one bolt on a tower). Placement: Goblin
  Barrel and Miner next to the target tower; siege buildings at the river of the lane; slow
  (speed <= 45) tanks at own ty 19, or behind the King (ty 29) when their hp >= 5000 (Golem,
  Lava Hound is faster); the rest at the bridge. The Tombstone may defend against ground
  threats (its Skeletons), siege buildings and the Elixir Collector are never a defensive
  decoy, the Miner never defends nor "spends" elixir.

## 9. Environment and observation (SPEC §9)

- **Layout, v0.5 [PIN §16.4, §16.6.16-17, §16.6.22, §19.3]** (constants exported by
  `pufferroyale.royale`, read from the binding): `OBS_SIZE = 17715` = spatial 25x32x18 (14400)
  + entities 64x11 (704) + scalars 306 + mask 2305. `CARD_SLOTS = 128`; card identities are the
  float integer `card_id + 1` (0 = empty) in entity feature 0, the hand (4), the next card (1)
  and the opponent's last four (4); the 128-wide multi-hots (opponent cards seen, deduced hand)
  are indexed by `card_id` (slots 64-127 always 0). Entity rows: card id + 1 [0], x/18000 [1],
  y/32000 [2], hp fraction [3], min(1, hp/2000) [4], flying [5], deploying [6] (also 1 while a
  Miner burrows), stunned-or-frozen [7], slowed [8], is_building [9], target_only_buildings
  [10]; own entities in slots 0-31, enemy in 32-63, each group the 32 lowest ids. Scalars, as
  `SCALAR_INDEX` (name -> (offset, length)): elixir 1, hand 4, hand_cost 4, next_card 1,
  affordable 4, tick 1, overtime 1, elixir_rate 1, lockout 1, own_towers 3, enemy_towers 3,
  king_active 2, crowns 2, opp_seen 128, opp_spent 1, opp_last4 4, opp_deduced_hand 128,
  opp_elixir_ub 1, own_tower_troop 4, enemy_tower_troop 4 (index order princess, cannoneer,
  dagger_duchess, royal_chef; the opponent's tower troop is public), own_deck 8 (§14.1). Spawned, death-spawned and
  released units carry the card that created them (Witch Skeletons -> Witch, Golemites ->
  Golem, Barbarian Barrel's Barbarian -> Barbarian Barrel, ...), in the observation and in
  `entities()['card_id']` [PIN §16.6.6]. Checkpoints trained on v0.2 observations do not load.
- **Spatial channels [IMPL]** (own frame, values in [0, 1]; per side, own 0-9 then enemy
  10-19): ground count, air count, building presence, crown tower, hp sum,
  buildings-only-targeter count, ranged count, splash count, deploying count, debuffed count.
  Counts add 1/4 per unit and are clipped to 1; troops and buildings are counted on the tile
  containing their centre, except that building presence is 1 on every tile the footprint
  overlaps; crown-tower channel = the living tower's hp fraction on its footprint tiles (towers
  enter no other channel); hp sum = hp/4000 per unit, clipped; "ranged" = range > 1900 (ledger
  `targeting.MELEE_RANGE_LIMIT`); "splash" = melee area radius > 0 or a splash projectile;
  "debuffed" = stunned or slowed. Shared: 20 water (tile water flag), 21 own troop-legal
  tiles, 22 own building-legal tiles (placement geometry only, ignoring elixir and lockout
  **[PIN §13.14]**; the building channel uses the own deck's first building, else the Cannon),
  23 / 24 own / enemy spell areas (tiles whose centre lies in a Fireball / Goblin Barrel /
  Rocket / Snowball landing disc while in flight -- radius at least 1000 --, an Arrows,
  Poison, Earthquake or pending-Lightning disc, a death bomb's blast disc while on its fuse, or
  a Log's / Barbarian Barrel's remaining roll corridor).
- **Scalars [IMPL details]**: elixir/28000 (= elixir/10 exactly), costs/10, tick/6000, elixir
  rate/150 (= multiplier/3), tower hp fractions `[King, own-frame left, own-frame right]` for
  both sides (own-frame left is engine left for team 0 and engine right for team 1), crowns/3,
  opponent elixir spent in elixir/200 (SPEC §16.4; unclipped), opponent last-4 most recent first, deduced
  hand nonzero only once all 8 opponent cards were seen (then exactly the 4 cards not in the
  last 4 played), elixir upper bound `clip(16800 + income(t) - spent, 0, 28000)/28000`.
- **Hidden information**: nothing about the opponent's current elixir, hand, queue or
  unrevealed deck enters the observation (tested: changing the opponent's elixir, reordering
  its hand/queue or swapping its unrevealed deck leaves the viewer's observation
  bit-identical; `scripts/e2e_check.py` runs the same probe).
- **Mask**: written by `pr_legal_mask`, the function `play` uses. Self-check path:
  `Royale(mask_check=True)` re-validates all 2305 actions with `pr_check_play` after every step
  and reports disagreements in the log key `mask_mismatch` (0 in every run; a corrupted entry is
  detected, `tests/c/test_env.c`).
- **Rewards**: terminal +1/-1/0 plus the optional potential-based shaping of SPEC §19.1 (reward v2,
  §14.2 below; it replaced the v0.4 tower / crown difference shaping), computed once per step for
  team 0 and negated for team 1, so self-play rewards are exactly antisymmetric. PuffeRL clips rewards
  to [-1, 1] (MMDPuffeRL's `reward_clip`, SPEC §19.2, can lift that).
- **Terminals / reset**: terminals = 1 for every row of the env on the ending step; the env
  re-deals inside the same step (the returned observation is the new match's; random decks and
  `learner_side='random'` are re-drawn); truncations are never set. `reset(seed=s)` reseeds env
  i's game, bot and side streams from `s * num_envs + i`; `reset()` continues them.
- Actions: an action the engine refuses (masked-illegal or out of range) is a no-op counted in
  `illegal_actions`. The learner action and the bot action are queued before the step's first
  tick and applied team 0 first (§1).
- **Log [PIN §13.14, §14.6]** (per episode; `vec_log` averages over `n`): the SPEC keys with
  `episode_return` = team 0's summed reward, `episode_length` in steps, `score` = `perf` =
  team 0's win 1 / draw 0.5 / loss 0, `overtime` = the match went past tick 3600, `tiebreak` =
  1 only if the end-of-overtime comparison ran (a TIEBREAK result, or a DRAW with both Kings
  standing; not a King kill or crown lead decided on tick 5999), `leaked_*` in elixir,
  `plays_*` accepted plays, `illegal_actions` summed over both agents; learner-centric
  `learner_return`, `learner_score`, `learner_win` (the policy's team in 1-agent mode, team 0
  in self-play) and `dropped_plays` (0 in env play); extras `match_ticks`, `mask_mismatch`.
  `royale.ini` sets `[sweep] metric = learner_score` (PufferLib's default `score` is team 0's,
  which in 1-agent mode with a random learner side is not the policy's).
- `Royale` sets `agents_per_batch = num_agents` (PuffeRL's LSTM path reads it from the vecenv;
  a native `PufferEnv` only defines `agent_per_batch`).
- **Policy masking**: `pufferroyale.torch` slices the mask out of the observations of every
  forward call (`forward`, `forward_eval`, and `Recurrent`'s, where PuffeRL passes
  `(segments, horizon, OBS)` and the LSTM wrapper's flattened rows keep that order); nothing is
  stashed between calls and a shape mismatch raises. Checked inside real PuffeRL LSTM epochs
  (`tests/builder/test_audit_c1.py`).
- **Policy (v0.3)**: `pufferroyale.torch.Policy` embeds every card id with one shared
  `nn.Embedding(CARD_SLOTS + 1, card_dim = 16)` (index 0 = empty): the entity id is embedded and
  concatenated with the row's other 10 features before the shared entity MLP (a slot is
  occupied iff its id is non-zero); the 9 scalar ids (hand, next card, opponent last four) are
  embedded and concatenated with the remaining 289 scalars. `league.load_policy` infers
  `card_dim` from the checkpoint.
- **Observation memory**: 17,707 float32 = 70,828 bytes per agent row, so a PuffeRL rollout
  buffer costs `batch_size x 70.8 KB` (4096 rows -> 290 MB; 65,536 -> 4.6 GB). Sections that
  could shrink without losing information (each would need a SPEC change; none applied): the
  static water plane (576) and the own troop-legal plane (576; equal to any affordable troop
  slot's mask block) duplicate information available elsewhere; the mask (2305) could be
  factored into per-slot validity plus the legality planes; the {0,1} and count planes could be
  stored as uint8 / float16 (2-4x) with the policy casting on device.

## 10. Rendering

- `render_mode='ansi'` / `Game.ansi()` / `Royale.ansi()`: a status header (tick, elixir,
  crowns, tower hp, hands) plus the 32x18 board in engine frame (legend in `pr_render.h`:
  upper case = team 0, lower case = team 1; `~` water, `=` bridge rows, `:` no-deploy, `*`
  spell area).
- raylib (`tools/get_raylib.sh` downloads the prebuilt raylib 5.5 for macOS or Linux x86_64
  into `third_party/`; `PR_RAYLIB=1 make build` links its static `libraylib.a`, or the one in
  `$RAYLIB_DIR`, plus the macOS frameworks / Linux GL+X11 libraries): a window with the arena,
  footprints, units (hp bars, flying/stun rings), projectiles and spell areas, plus an
  elixir/hand panel; `render_mode='human'`. Window creation is guarded: without
  an active display (macOS `CGGetActiveDisplayList` count 0, or neither `DISPLAY` nor
  `WAYLAND_DISPLAY` on Linux) the env prints the text board instead of letting GLFW crash
  inside `InitWindow`.

## 11. Other known divergences from the live game

- Evolutions, Champions, Heroes and all cards outside the 64 are out of scope (SPEC §0, §16).
- Arena geometry is the 2018 tilemap; the 2026 bridge width is unverified (ledger
  `arena.ARENA_SOURCE_VINTAGE`).
- Several rules constants the ledger marks below "measured" are used as the SPEC states them
  (e.g. `match.OVERTIME_TIEBREAK` community-sourced, `match.DEPLOY_LOCKOUT_TICKS` third-party
  measured).

## 12. SPEC §16 cards and tower troops (Phase E)

Everything below is data-driven: the generator reads the columns named here into
`PrUnitDef` / `PrCardDef` / `PrAreaDef` / `PrBuffDef` rows and the engine never tests a card id.

### 12.1 Compositions
- `second_summon` (Goblin Gang: 3 `Goblin_Stab` + 3 `SpearGoblin`; Rascals: 1 `RascalBoy` + 2
  `RascalGirl`) is a second unit / count on the card; formations in §5.1. `card_info` reports the
  summon's stats flattened (as in v0.2) plus `second_unit` and `count2`; `count` is the data's
  summon count (3 for Goblin Gang).

### 12.2 Periodic spawners (Witch, Night Witch, Tombstone; SPEC §16.2, §16.6.5, §16.6.24)
- **[PIN]** A spawner's timer runs only on ticks it is neither deploying nor stunned (ledger
  `spawner.STUN_PAUSES_SPAWNER`, timers frozen); the first wave lands at elapsed tick
  `k = start_time_ms/50 - 1` (k = 0 the first non-deploying tick; `pause_time_ms/50 - 1` when
  `start_time` is blank), then waves start every `pause_time_ms` (start to start); units of one
  wave are `interval_ms` apart (all on one tick when blank). Witch: 4 Skeletons at k = 19, then
  every 140 ticks; Night Witch: 2 Bats at 19, every 100; Tombstone: 2 Skeletons at 69 and 79,
  then 139 / 149, ...
- Emitted units are queued in the Status phase and created in the Spawn phase of the same tick
  (which runs right after), born deployed (ledger `spawner.SPAWNED_DEPLOY_TIME = zero`) with
  their load timer at `load_time`; they act (target, move, attack) on their creation tick and
  count toward `MAX_ENTITIES` (drops counted in `spawn_overflow`); they carry the spawner's card.
- **Layout [IMPL]** (ledger `spawner.SPAWN_POINT`): a spawner that sets `SpawnRadius` (Witch
  2000, Night Witch 1500) emits its wave on a ring of that radius around itself, in its OWN
  frame, the first unit at `SpawnAngleShift` clockwise from "toward the enemy" (Night Witch 90:
  its Bats on both flanks; Witch 0: front, right, back, left), the rest 360/n apart; a blank
  radius (Tombstone) emits at the tangent point in front of it (`r_spawner + r_unit` toward the
  enemy). Offsets are generated integers (`PR_SPAWN_OFFSETS`); ground units landing in water go
  to the nearest land point.
- **[DIV]** RoyaleSim measured a Tombstone's first Skeleton on its deploy-end tick
  (`spawner.FIRST_WAVE`) and a pause counted after the last unit of a wave
  (`spawner.PAUSE_ANCHOR`); SPEC §16.6.5 / §16.6.24 pin `pause/50 - 1` and start-to-start, and
  we follow the SPEC. Not modelled: the 8th-frame acquisition delay of spawned units
  (`targeting.SPAWNED_UNIT_ACQUIRE_DELAY`), `SpawnLimit` (no §16 card sets one), the facing
  angle (we use the owner's frame, not the unit's heading).

### 12.3 Death spawns (Golem, Lava Hound, Battle Ram, Night Witch, Tombstone; SPEC §16.2)
- On ANY death (lifetime expiry included, §16.6.7) the `death_spawn` units are queued in Reap and
  committed in the next Spawn phase, carrying the dying unit's card, deploying for
  `DeathSpawnDeployTime` (Battle Ram's Barbarians 1000 ms) or born deployed when it is blank
  (ledger `spawner.DEATH_SPAWN_DEPLOY_TIME_DEFAULT = zero`: Golemites, Lava Pups, Bats,
  Skeletons).
- **Layout [IMPL]**: a ring of `DeathSpawnRadius` around the death point in the owner's own
  frame, the first unit at `SpawnAngleShift` from "toward the enemy" (ledger
  `spawner.DEATH_SPAWN_LAYOUT`, with the owner's frame standing in for the unmeasured facing):
  Golem's 2 Golemites at 1500 in front / behind, Lava Hound's 6 Pups on a hexagon of 2500,
  Battle Ram's 2 Barbarians at 600 behind / in front (shift 180), Night Witch's Bat at 500 on
  her right (shift 90); the Tombstone's 4 Skeletons all on its emission point (ledger
  `spawner.DEATH_SPAWN_AT_EMISSION_POINT`), where the collision pass spreads them.
- **[DIV]** `DeathSpawnPushback` (the Golem's and Lava Hound's children appearing 250 out and
  sliding) is not read, as in RoyaleSim's shipped arm (`spawner.DEATH_SPAWN_PUSHBACK =
  not_read`).

### 12.4 Death bombs (Balloon, Giant Skeleton, Bomb Tower; SPEC §16.2, §16.6.8)
- A `DeathSpawnCharacter` row without hitpoints but with `DeployTime` + `DeathDamage` is a bomb,
  not a unit (RoyaleSim `convert_death_bomb`): the death queues it (Reap of tick T), the Spawn
  phase of T+1 creates an untargetable `PR_FX_BOMB` effect at the death point (visible in
  `effects()` as kind `bomb`), and after `DeployTime` (3000 ms) it detonates, in tick T+61, once:
  the row's `DeathDamage` (Balloon 240, Giant Skeleton 688, Bomb Tower 222; crown towers take
  it in full) to every enemy within `DeathDamageRadius` (3000) + r, ground and air per the row.
  The dead unit's own team is never hit.
- **[DIV]** `DeathPushBack` (Giant Skeleton 1800) is not carried by the data file and not
  applied; a levelled unit's bomb keeps the level-11 damage.

### 12.5 Charge, kamikaze, jump (Battle Ram, Wall Breakers, Fire Spirit, Royal Hogs)
- The Battle Ram charges like the Prince (`charge_range_raw` 300 -> 3000 walked, speed x2,
  `DamageSpecial` 573 on the next hit, no windup) and dies on its hit (ledger
  `combat.KAMIKAZE_DEATH = at_fire`), then breaks into its 2 Barbarians (§12.3).
- Fire Spirit and Wall Breakers are projectile kamikazes: they die when their hit lands in the
  SPEC §6.2 sense (the attack-cycle hit, i.e. when they launch their splash projectile; ledger
  `combat.KAMIKAZE_DEATH = at_fire`, a hypothesis for projectile kamikazes), and the projectile
  flies on and splashes (Wall Breakers: non-homing, aimed at the building's centre, 281 per
  breaker, the blast also hits air per the projectile's `aoe_to_air`). If SPEC §16.2's "dies
  when its hit lands" meant the blast's landing, the death would come one or two ticks later;
  the damage and its target are the same either way.
- Royal Hogs are jumpers (their `jump` block), as the Hog Rider (§6.3).

### 12.6 Minimum range (Mortar) -- §6.1. The Mortar's shell is non-homing (§7).

### 12.7 Bandit dash (SPEC §16.2, §16.6.14, ledger `combat.DASH_ATTACK = client_dash`)
- **Trigger [PIN]**: a walking Bandit whose target is not yet in attack range and whose centre
  distance `d` satisfies `DashMinRange + r_a + r_t <= d <= DashMaxRange + r_a + r_t` (3500 / 6000)
  starts a dash, provided the straight segment to the target is dry (a dash never crosses the
  river [IMPL]). **[DIV]** The ledger measured the trigger at `DashMaxRange + r_t` only; the SPEC
  pins both radii.
- **Stand** `DashCooldown` (800 ms, 16 ticks) holding its target, then **move** `JumpSpeed`
  (500) per tick as two half-steps toward the target's current centre, stopping after the first
  half-step whose edge gap (`d - r_a - r_t`) is within its `Range`; on that arrival tick
  `DashDamage` (389) is buffered on the target and the melee cycle restarts with a full load
  timer, so the first ordinary swing (194) lands 19 ticks later (measured +19 in the ledger).
- **Invulnerable** (no damage taken, the lifetime drain aside) from the first moving tick through
  the arrival tick's Resolve, not during the stand (as the ledger measured: tower arrows during
  the stand hit). **[PIN §18.3]** Immunity is decided when each hit is DEALT (`pr_hit`), not in
  Resolve: a Zap / Freeze / Snowball / Fireball / Lightning / Log hitting a moving Bandit buffers no
  damage even though its stun or knockback cancels the dash in the same call; a hit dealt before
  the dash starts moving (e.g. earlier in the tick it leaves the stand) still lands. A dashing Bandit skips the unit-unit collision pass. A stun, a knockback or
  the loss of the target cancels the dash; a slow does not change the dash speed.
- Not modelled: `DashRadius`, `DashPushBack`, `DashLandingTime` (none matter for the Bandit),
  the dash goal cell quantisation (we aim at the target centre).

### 12.8 Miner (SPEC §16.1, §16.6.3, §16.6.4, §16.6.23)
- Placement: §3. A hand-played Miner is created in the play tick P at its own King's centre with
  `hidden = True` (untargetable, immune to damage, holds no target, not pushed, not hit by any
  area and not a Royal Chef candidate) and walks underground on the straight line to the tap at
  `spawn_pathfind.speed` (650) per tick, its position interpolated exactly so that it stands on
  the tap after `N = ceil(d/650)` ticks; it surfaces in the Status phase of tick P+N
  (`hidden = False`) and deploys for 1000 ms from there (the deploy timer does not run on the
  surfacing tick), i.e. hidden in exactly N observations, deploying in the next 20
  [PIN §16.6.23]. It may pass under the river and under buildings.
- Its hits deal 20 % to crown towers (`crown_tower_damage_percent`, 194 -> 39).
- `Game.spawn` (the test hook) puts a Miner straight on its point without the burrow. In the
  observation a burrowing Miner is an entity (its tunnel is public information) with
  `deploying = 1`.

### 12.9 Elixir Collector (SPEC §16.1, §16.6.15)
- +`ManaCollectAmount` (1 elixir = 2800) at elapsed tick 259 after its deploy (k = 0 the first
  non-deploying tick), then every `ManaGenerateTimeMs` (13000 ms = 260 ticks), in the Status
  phase (after the tick's regen); capped at 10, the excess counted in `leaked`. It keeps
  producing while stunned [IMPL].
- `ManaOnDeath` (+2800, in the Reap of its death tick) only when it is destroyed by damage: its
  hp reached 0 in a Resolve where damage was dealt and the lifetime drain alone would not have
  killed it; nothing on expiry (7 productions in its 93 s). The opponent sees the Collector as an
  entity; its production never enters the opponent's elixir estimate (the upper bound
  `opp_elixir_ub` ignores it, SPEC §16.2).

### 12.10 Spells
- **Rocket, Giant Snowball**: point projectiles from the caster's King like Fireball
  (SPEC §16.6.12): Rocket 350/tick, 1484 in 2000 + r (crown 23 % -> 342), Snowball 800/tick,
  179 in 2500 + r (crown 45) and a 3 s `IceWizardSlowDown` slow on every victim; both knock
  troops without `ignore_pushback` 1800 radially from the impact point.
- **Freeze**: applied once at the tap in the play tick, like Zap: 148 (crown 37) and a 4 s
  freeze to every enemy within 3000 + r, crown towers included; the freeze lands on hidden
  units (`affects_hidden`) but hidden units take no damage [PIN §16.6.21]; no standing object.
- **Poison / Earthquake [PIN §16.6.10-11, IMPL details]**: a `PR_FX_PULSE` object at the tap
  living the area's `LifeDuration` (8 s / 3 s). DAMAGE comes in events, from the area's buff: each
  event deals `dps x hit_frequency / 1000` (Poison 92, Earthquake 81) to every enemy within
  3500 + r (Earthquake: ground only), non-crown buildings take `floor(d x building_pct / 100)`
  (Earthquake 283, Poison 92), crown towers `ceil(d x crown_pct / 100)` (Poison 22, Earthquake 49);
  events at P, P+20, ... (8 for Poison, 3 for Earthquake), the first in the play tick like Zap.
  The SLOW (the buff: -15 % / -50 %) is (re)applied to everyone inside every area `HitSpeed`
  (Poison 250 ms, Earthquake 100 ms) while the area lives, for `BuffTime` (1000 ms) -- Earthquake's
  `cap_buff_time_to_area_effect_time` caps it at the area's remaining life --, so a unit is slowed
  while inside and up to 1 s after leaving a Poison (the last Poison refresh at P+155 -> slowed
  through P+174), never past an Earthquake. Two areas stack (`enable_stacking`): each deals its own
  events. **[DIV]** In the game (RoyaleSim `status.BUFF_PULSE_*`) the damage is the buff's own
  pulse on each unit (its clock starting when the unit is first buffed) and continues while the
  buff lasts after the unit leaves; here the damage lands only on the area's per-second events,
  on whoever is inside then. Earthquake's `hit_tick_from_source` is covered by the same choice.
- **Lightning [PIN §16.6.13, §16.6.25]**: at the cast (tick P) the 3 highest current-hp valid
  enemies within 3500 + r (ground and air, not hidden; crown towers included), ties to the
  lowest id, are fixed; bolt k strikes the k-th of them in tick `P + ceil(460 k / 50)` (P, P+10,
  P+19) for the `LighningSpell` damage (1057, crown 25 % -> 265) and a 500 ms `ZapFreeze`; a
  target that died is skipped. The bolt is instantaneous (the projectile's flight is not
  modelled).
- **Barbarian Barrel [PIN §16.6.20]**: The Log's rolling machinery with the
  `BarbLogProjectileRolling` row: 4500 at 200/tick from the tap (the airborne
  `BarbLogProjectile` phase is not modelled, as for the Log), hitbox 1300 x 600, 232 once per
  ground enemy (crown 100 %), no knockback (`pushback 0`). The roll ends with the hit test at
  its end point (tick P+23); one Barbarian (`spawn_character`, deploying 1000 ms) is committed
  there in the next Spawn phase (P+24), carrying the Barbarian Barrel's card.

### 12.11 Tower troops (SPEC §16.3)
- Stats: §2. They keep the Princess tower's range, footprint (F = 3), pocket rects and crowns.
- **Cannoneer**: the data (2200 ms, 1400 ms load, homing 1000/tick shot, 272 damage).
- **Dagger Duchess [IMPL]**: her data `AttackSequence` has 4 throws with `HitSpeedMultiplier`
  100 / 100 / 70 / 90 of her 500 ms hit speed. Throw k of a sequence lands 500 x mult[k] / 100
  after the previous one (the first of a fresh engagement after its own cycle), so a sequence is
  4 daggers of 91 at 10 / 10 / 7 / 9-tick cycles; after the 4th she reloads for 1000 ms (the
  reload runs with or without a target, paused while stunned), and the next sequence starts on
  the tick after the reload ends: in a long fight the gaps repeat 10, 7, 9, 30 ticks. The data
  has no ammo or reload column (`AttackSequenceMode None`); the 1000 ms reload is chosen so that
  her long-run damage rate (4 x 91 per 2.8 s = 130/s) matches the Princess tower's (109 per
  0.8 s = 136/s) while bursting faster.
- **Royal Chef [PIN §16.6.18, §16.6.26, §18.2, IMPL details]**: each living Chef tower cooks for
  28000 ms from the match start (paused while the tower is stunned); after every entity's Status the
  team's Chef towers are served in OWN-frame left-to-right order (a troop levelled by the first is
  not eligible for the second); when a pancake is ready it levels up
  the allied troop within 7500 (centre to centre) of that tower with the highest elixir cost,
  ties to the lowest id; with nobody eligible the pancake waits and is served the first tick
  someone is (then the next 28 s start). First level-up: the tick-559 Status phase (the 560th
  tick). Cost = the card's elixir for the card's own members (summon and second summon), 0 for
  units a spawner, a death or a spell released; a burrowing Miner and buildings are not
  eligible. Level-up (+1 level): max hp, hp and shield `x ladder[12] / ladder[11]` = 281/256,
  floored, and every damage it deals from then on (hits, projectiles, charge, dash, death
  damage) likewise; a troop is levelled at most once (L12) [IMPL, as in the live game]. Units it
  spawns later and its death bomb stay level 11. **[DIV]** The pancake is served instantly (the
  `ChefTower_pancake_projectile` flight is not modelled).
- The King tower is unchanged; both Princess slots of a team always carry the same troop.

### 12.12 Capacities (SPEC §1)
- `MAX_ENTITIES` stays 256 (the tester's invariants assume it): random 64-card matches peak at
  about 36 live entities (40-match benchmark, mean 12), the SPEC §14 216-Skeleton stress at
  216 (224 with its 8 Cannons), the §16 large battle at 176 (+ Witch waves). Spawner-heavy
  extremes overflow gracefully: the spawn is dropped and counted in `spawn_overflow` (a 40-Witch
  scenario in `tests/c/test_cards_v3.c`); no soak run overflows. `MAX_PENDING_SPAWNS` grew from
  64 to 128 (death spawns + spawner waves + bombs of one tick). The state is 65 KB (version 4;
  53.6 KB in v0.2).

### 12.13 Performance (Apple M4, one core, release build, `make bench` / `make bench-stress`)
- Engine, random 64-card decks, random bots: ~500,000 ticks/s (`make bench`; the tester's
  Python method measures ~470,000). SPEC §14 stress (216 Skeletons across the river):
  ~14,700 ticks/s (12,900 with 8 Cannons); the §16 large battle (`bench_stress` variant 2):
  ~47,000 ticks/s. Env: ~25,000 steps/s (frame_skip 10, 2 agents, observation included).

## 13. Second-audit amendments (SPEC §18, v0.4.1 / v0.4.2)

| item | change | where |
|---|---|---|
| M1 §18.2 Chef order | Chef towers served after the Status loop, own-left first per team | `pr_engine.h` Status |
| L1 §18.3 dash immunity | decided when a hit is dealt (`pr_hit`), not at Resolve | `pr_combat.h`, `pr_entity.h` |
| L2 §18.5 / §18.8 snapshot bounds | tight, evolution-invariant bounds (§4); pending spawns need a card and a non-tower unit; unused slots zeroed; statistics counters saturate; a debug King kill cannot push crowns past 3; `Game._set_entity(hp)` clamps to max_hp | `pr_engine.h`, `pr_rules.h`, `binding.c` |
| L3 §18.1 tower ties | tied crown towers go own-left | `pr_target.h` |
| L4 §18.4 bots | own-frame tower order in rule 1; spell values at the real tap, Lightning's 3 targets | `pr_bots.h` |
| NITs | rolling-spell edge closed test; open-square path occluder | `pr_spell.h`, `pr_path.h` |

Two engine fields are now normalised on transitions, with no change to behaviour: a Tesla's
`hide_ms` is reset to 0 when it hides, and a Bandit's `aux_ms` to 0 when its stand ends. Both are
unused in those states; the reset keeps them within the validated bounds. The state layout is
unchanged (version 4). Regression tests: `tests/c/test_audit2.c`, `tests/c/test_mirror.c`,
`tests/c/test_cards_v3.c` (`test_dash_immunity_vs_cancelling_spells`), `tests/c/test_fuzz.c`
(validity preserved), and `tests/builder/test_audit2_engine.py`. The bot-driven mirror fuzz of the
audit (`mirror.c`) reports 0 divergences: 300 matches forced to each tower troop, and 1000 random
ones.


## 14. Training work package, env side (SPEC §19, v0.5-G)

Everything here lives in `royale.h` (env), `pr_obs.h` (encoder), `binding.c`, `royale.py`, `decks.py`
and `league.py`; `pr_*.h` game logic is untouched (golden hashes unchanged). With every new keyword at
its default `Royale` is bit-identical to v0.4 apart from the `own_deck` field: checked against
fingerprints (observations without `own_deck`, rewards, terminals, logs) recorded from the v0.4 build
(`tests/builder/test_wp_env.py::test_defaults_are_bit_identical_to_v04`).

### 14.1 Own deck (SPEC §19.3)
- **[PIN]** `own_deck` = the viewer's 8 deck cards `card_id + 1`, ascending by card id, at scalar
  offset 298 (after `enemy_tower_troop`); `CARD_ID_SCALARS` gains it. Source: the engine's
  `deck[team]` (the dealt deck, also for `random` and sampled decks), sorted in the encoder, so the
  hand / queue order never shows.

### 14.2 Reward v2 (SPEC §19.1)
- **[PIN]** `F = gamma * Phi_new - Phi_prev`, `Phi_new = m_n * Phi_hat(s')` (0 on the ending step),
  `Phi_prev` stored per match (set to 0 at every deal: construction, `reset`, auto-reset), all in
  double; `r_0` is cast to float once.
- **[IMPL]** `r_1 = 0.0f - r_0`: the exact negation for every non-zero reward, and +0 (never -0) for a
  zero reward, which is what v0.4 stored; this keeps all-zero-weight rewards bit-identical.
- **[IMPL]** `T_k` sums the three `pr_obs_tower_frac` float32 values converted to double; `L_k` is
  `leaked[k] / 2800` in double per team, then differenced; the clips are applied to the differences.
- **[IMPL]** The anneal counter `n` is a per-C-env int64 incremented by every `c_step` (never by a
  reset), exposed as `env_info()["env_steps"]`; `env_info()["shaping_multiplier"]` is `m` of the next
  step. `N` and `n0` are accepted as any finite real >= 0 (integers in practice).
- **[IMPL] Validation** (Python first, the C init repeats it): weights must be finite and >= 0
  (NaN / inf rejected too); caps > 0 (+inf allowed = no clip); `gamma` in (0, 1]; `N`, `n0` finite
  and >= 0. With all four weights 0 the potential is not evaluated (Phi = 0 exactly).

### 14.3 Placement grid (SPEC §19.4)
- **[PIN]** Decoding exactly as the SPEC: block tiles, nearest legal tile to the block centre in
  doubled integer coordinates, ties smaller ty then smaller tx; no legal tile = an illegal no-op
  counted in `illegal_actions`; an out-of-range or negative action likewise.
- **[IMPL] Exactness:** the tile-independent refusals (game over, slot, pending, lockout, elixir) come
  from `pr_check_play` on the block's corner tile; the per-tile test is `pr_card_tile_legal_ctx` on one
  `PrLegalCtx`, the same functions `pr_legal_mask` and `pr_check_play` use, so the candidate set is
  exactly the fine mask (checked exhaustively on real states in `tests/c/test_wp_env.c` and
  `tests/builder/test_wp_env.py`).
- **[IMPL] Fine rows (the §19.4 mechanism):** every agent row of a C env has its own grid
  (`row_grid`, default the env's `placement_grid`); grid 1 takes the v0.4 decode path. A 1-agent env's
  scripted opponent always plays fine actions; `Royale.set_row_grid(env_index, row, grid)` changes a
  row, and `LeagueVecEnv` sets, at every episode start, the learner row to the env grid and the
  opponent row to 1 for `bot:` opponents (else the env grid). `env_info()["row_grids"]` reports them.
- **[IMPL]** `Game.coarse_to_fine(team, a, 1)` returns `a` for a legal fine action and 0 for an illegal
  one (the general rule with 1x1 blocks); the env's grid-1 rows queue the fine action directly, as v0.4.
- **[PIN]** The observation's mask stays the fine 2305-mask; `royale.action_mask(obs, g)` derives the
  coarse mask (`coarse[0] = 1`, block = any legal tile; partial blocks of g = 4 cover the existing tiles).

### 14.4 Deck sampling (SPEC §19.5)
- **[PIN]** Active iff `deck_pool` is non-empty or `random_deck_frac > 0`; then every deal (construction,
  `reset`, auto-reset) draws team 0 then team 1 (mirror: team 1 copies team 0), random decks redrawn
  while held out; draws use the env's own PCG32 stream (`PR_DECK_STREAM`), seeded like the bot / side
  streams and reseeded by `reset(seed)`; the game stream is untouched (a one-deck pool plays bit for bit
  like fixed decks given in ascending order).
- **[IMPL] Draw procedure:** per seat one 32-bit draw `u` picks random (`u < round(frac * 2^32)`) or pool;
  a pool deck takes one more 32-bit draw against cumulative weight thresholds `floor(cum_i / W * 2^32)`
  (last = 2^32), so probabilities are exact to 2^-32 (a weight below 2^-32 of the total would never be
  drawn); a random deck is a partial Fisher-Yates over the 64 cards (8 bounded draws per attempt). The
  draw consumes `u` even when `frac` is 0 or 1. A mirror deal makes no draw for team 1.
- **[IMPL] Construction:** the sampler stream is seeded from the env seed, the construction decks are
  drawn, then every stream is reseeded (as v0.4 does for the game stream), so the first `reset()`
  re-deals the construction decks; `deck_counts()` includes the construction deal.
- **[PIN ruling v0.5-G.1] Installed card order** (the order the engine's shuffle starts from): a pool
  deck named by a preset (`PRESET`, `PRESET:W`, a preset string in a list or a `file:` JSON, a
  `{"deck": PRESET}` dict or a `(PRESET, w)` pair) keeps that preset's card order, so
  `deck_pool="hog26"` deals exactly like `deck0 = deck1 = "hog26"` (same game-stream draws); every
  other deck (card lists, `random:N:SEED`, random draws) is installed ascending.
  `decks.deck_entries(spec)` returns `(deck, weight, installed order)`; `parse_deck_set` stays ascending.
- **[IMPL]** Held-out decks are 64-bit card-set masks (linear scan). Capacities: exactly 256 pool decks
  and 1024 held-out decks (more -> `ValueError`), stored inline in the env (no allocation).
  `deck0` / `deck1` are still validated when the sampler is active, then ignored; `env_info()` reports
  `deck_sampler`, `deck0_ignored`, `deck1_ignored` (all equal) and `deck_mirror`. With the sampler
  inactive, `heldout_decks` has no effect (v0.4 `'random'` decks are not filtered).
- **[IMPL] Deck-set strings:** whitespace around items and around `:` is ignored, empty items (e.g. a
  trailing `;`) are skipped, `file:PATH` keeps inner spaces of the path; `random:N:SEED` needs N >= 1;
  in a `file:` JSON list a string element is a preset name only, while a Python list element may be
  any string item, a list of 8 cards, a `{"deck", "weight"}` dict or a `(deck, weight)` pair (so
  `parse_deck_set` output parses to itself). `deck_key(deck)` = ascending ids joined by `-`.
  `random_decks(n, seed)`: numpy PCG64 from `SeedSequence([sign, |seed|])`, `choice(64, 8, replace=False)`
  sorted, repeats skipped (so a smaller `n` is a prefix).

### 14.5 Per-card play rates (SPEC §19.7.6)
- **[IMPL]** Counted in Python (`royale.CardStats`, vectorised numpy) from each decision's observation
  (`hand`, `affordable`, `lockout`, the mask) and action, not in the C `Log`: `Royale` counts every
  row, `LeagueVecEnv` the learner rows. "Allowed by the action mask" uses the coarse mask under a
  grid > 1; out-of-range actions never count as played. The keys `cards/play_rate/<CARD_KEY>` (the
  source key, e.g. `HogRider`) and `cards/decisions` are added to each emitted log; the counters carry
  over a log interval in which no episode finished (nothing is emitted then) and reset after each
  emission.
