# PufferRoyale — Simulator & Environment Specification (v0.2.1)

Status: **binding contract** between the BUILDER (implements) and the TESTER (verifies).
Owner/adjudicator: the orchestrator. If the spec is ambiguous or wrong, raise it; do not silently
pick. Anything marked **[IMPL-DEFINED]** may be chosen by the builder but MUST be documented in
`docs/FIDELITY.md` and MUST be deterministic.

Game target: Clash Royale 1v1 ladder battle rules as of 2026, card statistics from the
**15.535.29 client** (decoded by RoyaleSim, pinned at RoyaleSim commit `72ed062`), rules from the
RoyaleSim calibration ledger (`data/source/royalesim-calibration.json`). All cards at **tournament
level 11**.

---

## 0. Scope of v0.1

In scope: full match rules (clock, elixir regimes, overtime, sudden death, tiebreak, King
activation, deploy lockout, placement zones incl. pockets, card cycle), crown towers (default
Princess tower troop), and **21 cards**:

| id | name (display) | source key in `cards-15.535.json` | kind |
|---|---|---|---|
| 0 | Knight | Knight | troop |
| 1 | Archers | Archer | troop (x2) |
| 2 | Musketeer | Musketeer | troop |
| 3 | Giant | Giant | troop (buildings only) |
| 4 | Hog Rider | HogRider | troop (buildings only, river jump) |
| 5 | Minions | Minions | troop (air, x3) |
| 6 | Baby Dragon | BabyDragon | troop (air, splash) |
| 7 | Valkyrie | Valkyrie | troop (self-centred splash) |
| 8 | Skeleton Army | SkeletonArmy | troop (x15) |
| 9 | Skeletons | Skeletons | troop (x N per data) |
| 10 | Ice Golem | IceGolemite | troop (buildings only, death damage + death slow area) |
| 11 | Ice Spirit | IceSpirits | troop (kamikaze, splash, freeze) |
| 12 | Prince | Prince | troop (charge, river jump) |
| 13 | Wizard | Wizard | troop (splash projectile) |
| 14 | Cannon | Cannon | building (ground only, lifetime) |
| 15 | Tesla | Tesla | building (hides, lifetime) |
| 16 | Fireball | Fireball | spell (projectile, radius, knockback) |
| 17 | Arrows | Arrows | spell (3 waves) |
| 18 | Zap | Zap | spell (instant area, stun) |
| 19 | The Log | Log | spell (rolling, ground only, knockback) |
| 20 | Goblin Barrel | GoblinBarrel | spell (projectile that spawns 3 Goblins) |

`N_CARDS = 21`. One-hot encodings reserve `CARD_SLOTS = 32`. Card ids above are **stable** and
are part of the public API.

Preset decks (public API names):
- `hog26`: Hog Rider, Musketeer, Cannon, Ice Golem, Ice Spirit, Skeletons, Fireball, The Log
- `giant`: Giant, Prince, Baby Dragon, Wizard, Minions, Knight, Arrows, Zap
- `bait`: Goblin Barrel, Skeleton Army, Tesla, Valkyrie, Archers, Knight, The Log, Fireball
- `random`: 8 distinct cards sampled uniformly from the 21 using the env RNG at each reset.

Out of scope for v0.1 (architecture must not preclude them): Evolutions, Heroes, Champions,
tower troops other than the default, Mirror, card levels other than 11, other cards.

---

## 1. Units, frames, determinism

- Positions: `int32` **millitiles** (1 tile = 1000). Arena `x ∈ [0, 18000)`, `y ∈ [0, 32000)`,
  `y = 0` is the **top** edge.
- **Team 0 is the BOTTOM player** (home half `y ≥ 17000`), **team 1 is the TOP player**.
- Time: 1 **tick** = 50 ms (20 ticks/s). All durations in data (ms) convert to ticks as
  `ticks = ms / 50` (data values are multiples of 50; if not, round **up**).
- Speeds in the data (`speed`) are **millitiles per tick** (Knight 60 → 1.2 tiles/s; Hog Rider
  120 → 2.4 tiles/s). Projectile speeds likewise are millitiles per tick.
- **No floating point in simulation state or logic.** Floats are allowed only in observation
  encoding, logging and rendering. Integer division semantics must be explicit (use helpers;
  C truncates toward zero). Distances use exact integer square root on `int64`.
- **Seat symmetry = 180° rotation**: engine point `(x, y)` for team 1's *own frame* is
  `(18000 - x, 32000 - y)`; tile `(tx, ty)` ↔ `(17 - tx, 31 - ty)`.
- Randomness: one seeded PCG32 stream per env for game logic (deck shuffles, random decks) and a
  **separate** PCG32 stream for scripted bots. Same seed + same actions ⇒ bit-identical state.
- The whole game state is a plain-old-data struct (no pointers inside, zero-initialised with
  `memset` so padding is deterministic). `snapshot = memcpy`; `restore = memcpy`.
- `state_hash` = 64-bit FNV-1a over the bytes of the game-state struct.
- Fixed capacities (compile-time): `MAX_ENTITIES = 256` (troops+buildings+towers),
  `MAX_PROJECTILES = 256`, `MAX_EFFECTS = 64` (spell/area objects). No heap allocation after init;
  no allocation inside `step`. On overflow the spawn is dropped and a log counter increments.

---

## 2. Level-11 stat derivation (the card DB)

`tools/gen_card_db.py` reads `data/source/cards-15.535.json` and emits `pufferroyale/csrc/pr_card_db.h`
(generated, committed, with a header comment giving the source sha256).

- Every card/unit/projectile/area stat that scales (`hitpoints`, `damage`, `shield_hitpoints`,
  `death_damage`, charge `damage_special`, projectile `damage`, area-effect `damage`, spawned unit
  `hitpoints`/`damage`) at level 11 = `floor(base * 256 / 100)` (the Common ladder index 10 =
  256%, which RoyaleSim measured as the unified level-11 multiplier for all base objects).
  Examples: Knight HP 690 → 1766, Knight dmg 79 → 202, Hog HP 663 → 1697, Fireball 269 → 688,
  Zap 75 → 192, Log rolling 105 → 268, Goblin HP 79 → 202, Goblin dmg 49 → 125.
- **Crown towers (level 11)** — compounding tower ladder, percent floored each level, rate `r`
  for levels 2..9 then 10% per level after:
  - Princess tower: HP 1400 → **3052**, damage 50 → **109**, hit speed 800 ms, range 7500,
    load time 0, collision radius 1000, homing projectile speed 600, start radius 300,
    targets ground+air, sight 7500.
  - King tower: HP 2400 → **4824** (rate 7), damage 50 → **109** (rate 8), hit speed 1000 ms,
    range 7000, load time 500, collision radius 1400, homing projectile speed 1000,
    start radius 750, targets ground+air, sight 7000.
- Crown-tower damage reduction: a damage instance with `crown_tower_damage_percent = P < 100`
  deals `ceil(D * P / 100)` to **crown towers only** (King/Princess; not Cannon/Tesla).
  Examples at L11: Fireball 688 → 172 (P=25), Zap 192 → 48 (P=25), Arrows wave 122 → 25 (P=20),
  Log 268 → 35 (P=13).

---

## 3. Arena

Derived by `tools/gen_arena.py` from `data/source/royalesim-arena.json` (the shipped 2018
tilemap: 36×64 **half-tile** cells) into `pufferroyale/csrc/pr_arena_db.h`.

- River (water): `y ∈ [15000, 17000)` except the two **bridges** `x ∈ [2500, 4500)` and
  `x ∈ [13500, 15500)` (2 tiles wide, centred 3500 and 14500).
- No-deploy strips (bitmask bit 16) as in the tilemap: the four back-corner strips at
  `y ∈ [0,1000)` / `[31000,32000)` for `x < 5500` or `x ≥ 12500`, the river-bank corner cells,
  and each King block.
- Crown towers (centres):
  - Team 0 (bottom): King (9000, 29000); Princess left (3500, 25500), right (14500, 25500).
  - Team 1 (top): King (9000, 3000); Princess left (3500, 6500), right (14500, 6500).
- Tower index convention everywhere in the API: `0 = King, 1 = Left Princess, 2 = Right Princess`,
  where Left/Right are in **absolute engine x** (left = x 3500).
- Building/tower footprint (placement & path occlusion): square of side
  `F = ceil((2*radius + 1000) / 1000)` tiles centred on the entity (Princess F=3, King F=4,
  Cannon F=3, Tesla F=2).

### 3.1 Placement legality (per tile, own frame of the acting team)

Actions address **tiles** `(tx, ty)` in the acting team's own frame, `tx ∈ [0,18)`, `ty ∈ [0,32)`;
the placement point is the tile centre `(tx*1000+500, ty*1000+500)` in own frame, converted to
engine frame. A tile is:
- `water` if any of its 4 half-cells is water; `nodeploy` if any half-cell has the no-deploy bit.

Legality by card kind (team T, own frame, so own half is `ty ≥ 17`):
1. **Troops** (incl. Skeleton Army etc.): tile not water, not nodeploy, not overlapping any
   living tower's footprint (own or enemy), not overlapping any living building footprint (own or
   enemy) *(v0.2)*, and **territory**: `ty ≥ 17` (own half) OR the tile is
   in an opened **pocket**. Pocket rule (RoyaleSim `EnemyTowerRects`): tile centre not in the
   river band and **not inside the closed rect of any living enemy crown tower**, where rect sizes
   are King 18×16 tiles and Princess 11×21 tiles centred on the tower. Consequence for team 0:
   after the enemy LEFT princess falls, tiles `tx ∈ [0,8], ty ∈ [11,14]` (minus nodeploy/water)
   open; after the RIGHT falls, `tx ∈ [9,17], ty ∈ [11,14]`.
2. **Buildings** (Cannon, Tesla): own half only (`ty ≥ 17`, no pocket), and the whole footprint
   must be inside own half, not water, not nodeploy, not overlapping any **living** tower footprint
   or any other living building footprint (positive-area overlap). *(v0.2)* Anchor: odd `F` → the
   building centre is the tapped tile's centre; **even `F` (Tesla) → the centre is the tapped
   tile's own-frame top-left corner** `(tx*1000, ty*1000)` (own frame), so the footprint covers
   own-frame tiles `[tx−F/2, tx+F/2−1] × [ty−F/2, ty+F/2−1]` (grid-aligned, as in the real game).
3. **Spells** (Fireball, Arrows, Zap): any of the 576 tiles.
4. **Goblin Barrel**: any tile that is not water.
5. **The Log** *(v0.2 — data `can_deploy_on_enemy_side` is false for the Log)*: troop
   **territory** only (own half `ty ≥ 17` or an opened pocket), not water; it MAY overlap
   buildings, towers and nodeploy cells (it is a spell).
6. A destroyed tower's footprint no longer blocks anything *(v0.2)*.

Additional play conditions: card in hand, elixir ≥ cost, tick ≥ `deploy_lockout_ticks`, game not over.

---

## 4. Match rules

| Rule | Value |
|---|---|
| Regulation | 180 s = ticks `[0, 3600)` |
| Overtime | 120 s = ticks `[3600, 6000)`, **sudden death** |
| Elixir rate 1× | ticks `[0, 2400)`: +50 units/tick |
| Elixir rate 2× | ticks `[2400, 4800)` (last 60 s of regulation + first 60 s of OT): +100/tick |
| Elixir rate 3× | ticks `[4800, 6000)`: +150/tick |
| Elixir unit | **1 elixir = 2800 units** (so 1× = one elixir per 2.8 s exactly) |
| Start elixir | **6 elixir** = 16800 units (ledger `START_MANA`, measured) |
| Max elixir | 10 elixir = 28000 units (regen beyond the cap is lost: "leaked") |
| Deploy lockout | no card may be played while `tick < deploy_lockout_ticks` (default **90**); elixir regenerates during it |
| King activation | King is dormant (does not target or attack) until triggered by (a) the King taking any damage, or (b) the owner's Princess tower being destroyed; it becomes active **71 ticks** (3550 ms) after the first trigger |
| Crowns | each destroyed Princess = 1 crown to the destroyer; destroying the King ends the match immediately and the destroyer's crowns become 3 |

Elixir regen for tick `t` is applied at the **start** of tick `t` processing. The elixir *rate* at
tick t is determined by t (so at t=2400 the rate is already 2×).

End conditions, evaluated at the end of every tick (Judge phase):
1. A King destroyed → that King's owner loses immediately (both Kings on the same tick → draw).
2. At the end of tick 3599 (regulation over): if crowns differ → more crowns wins.
3. During overtime (ticks 3600..5999): at the end of any tick where crowns differ → leader wins.
4. At the end of tick 5999 with crowns equal: **tiebreak** — each side's *weakest surviving crown
   tower* (minimum HP among its alive King/Princess towers, absolute HP) is compared; lower loses;
   exact tie → draw. (Config `tiebreak = absolute|fraction`; default `absolute`.)

`result` per team: +1 win, -1 loss, 0 draw. `end_reason ∈ {KING, REGULATION_CROWNS, OVERTIME_CROWNS, TIEBREAK, DRAW}`.

### 4.1 Hand & cycle

- At reset each team's 8-card deck is shuffled with the game RNG; `hand = deck[0:4]` (slots 0..3),
  `queue = deck[4:8]` (FIFO, `queue[0]` = **next card**).
- Playing hand slot `i` (card c): elixir -= cost(c)·2800; `hand[i] = queue.pop_front()`;
  `queue.push_back(c)`. The hand/queue is always a permutation of the deck.
- Card costs (from data): Knight 3, Archers 3, Musketeer 4, Giant 5, Hog Rider 4, Minions 3,
  Baby Dragon 4, Valkyrie 4, Skeleton Army 3, Skeletons 1, Ice Golem 2, Ice Spirit 1, Prince 5,
  Wizard 5, Cannon 3, Tesla 4, Fireball 4, Arrows 3, Zap 2, The Log 2, Goblin Barrel 3.

---

## 5. Tick pipeline

Each tick executes these phases in order (RoyaleSim's measured order; names are normative):

1. **Upkeep** — advance clock; elixir regen (both teams, rate by tick); apply queued card plays
   for this tick (validated; spawns entities/effects; team 0's play is processed before team 1's
   **[deterministic]**); advance King activation countdowns.
2. **Status** — tick down buffs (stun/freeze/slow), deploy timers, building lifetimes (linear HP
   drain), hide/rise state machines, spawn staggers.
3. **Spawn** — commit entities created by the previous tick's deaths/impacts (death spawns,
   Goblin Barrel goblins).
4. **Target** — every active unit/building/tower acquires or keeps a target (§6.1).
5. **Attack** — attack cycles advance; melee hits are **buffered** as damage events; projectile
   launches are created (appear next tick at start radius) (§6.2).
6. **Path/Move** — ground pathing and movement, flying movement, charge accumulation (§6.3).
7. **Collide** — separation (§6.4).
8. **Projectile** — projectiles and spell objects advance; impacts produce buffered damage,
   buffs, knockback and queued spawns (§7).
9. **Resolve** — apply all buffered damage simultaneously (shields first; crown-tower reduction).
10. **Reap** — remove entities with HP ≤ 0; trigger death damage (buffered for next Resolve),
    death area effects and death spawns (committed next Spawn); tower destruction side effects
    (crowns, King activation trigger, pocket opening).
11. **Judge** — end conditions (§4).

Deploying units: occupy space, are targetable and damageable, but do not target, attack or move.
Hand-played troops deploy for `deploy_time_ms` (1000 ms for most). Multi-unit members may be
staggered by `summon_deploy_delay_ms × member_index` **[IMPL-DEFINED formation, see §5.1]**.

### 5.1 Formation

Multi-unit troops spawn around the placement point with deterministic offsets. Required: n=2 —
two members 1000 apart horizontally, centred on the point; n=3 — triangle on radius 577
(`(0,-577), (+500,+289), (-500,+289)` in own frame, apex toward the enemy); larger n — concentric
rings, all members within 1600 of the point **[IMPL-DEFINED exact layout]**. Ground members that
would land in water or outside the arena are moved to the nearest legal land point
**[IMPL-DEFINED]**. Goblin Barrel goblins: 3-member triangle at the landing point.

---

## 6. Units and buildings

### 6.1 Targeting

- Valid target for attacker A: alive enemy entity, not hidden (Tesla underground), matching A's
  `attacks_air`/`attacks_ground` against the target's layer (flying ⇔ `flying_height > 0`);
  if `target_only_buildings`, only buildings and crown towers.
- `dist` = centre-to-centre integer distance. **Sight test**: `dist ≤ sight_range + r_target`
  (+ 2000 extra if the target is a crown tower). **Attack-range test**:
  `dist ≤ range + r_attacker + r_target`.
- Acquisition: among valid targets passing the sight test choose minimum `dist - r_target`
  (tie → lowest entity id). If none, the unit has no attack target and walks its **default route**:
  toward the enemy crown tower of its lane (lane = `x < 9000` ⇒ left, else right; the lane's
  Princess if alive, else the enemy King). Building-only targeters follow the same rule.
- Retention: a unit keeps its current target while it is valid and within `sight + r_target + 500`
  (keep-hysteresis) **unless** a new acquisition finds a strictly closer target *and* the unit is
  not mid-cycle (attack progress 0). Buildings/towers target the nearest valid enemy within
  attack range and keep it while in range. Stun/freeze clears the target (retarget on resume).
- King tower does not target while dormant.

### 6.2 Attack cycle (ledger `combat.ATTACK_CYCLE = progress_credit`)

Per attacker keep `load_timer` (ms) and `progress` (ms). Every tick `load_timer = max(0, load_timer - 50)`.
If the attacker has a target that passes the attack-range test (or a hit is already started,
`progress % hit_speed > 50`): if `progress == 0` (fresh cycle) set
`progress = load_time - load_timer`, `load_timer = load_time`; then `progress += 50·m` where
`m` is the hit-speed multiplier (1, or reduced by slows, 0 when stunned). A **hit** lands on the
tick `progress` crosses a multiple of `hit_speed`; it sets `load_timer = load_time`. Otherwise
(no target in range and no hit started) `progress = 0`. Consequences the TESTER may rely on:
first hit on a fresh target by a fully-loaded unit lands `(hit_speed - load_time)/50 - 1` ticks
after entering range; subsequent hits every `hit_speed/50` ticks. Units spawn with
`load_timer = load_time` (the timer runs down during deploy).
- Melee hit: buffered damage to the target (and splash if `area_damage_radius > 0`; centre = the
  attacker if `self_as_aoe_center` (Valkyrie) else the target).
- Ranged hit: launch the projectile (§7.1).
- Tesla: instant hit (no projectile).
- Kamikaze (Ice Spirit): the unit dies at the moment it fires; its projectile continues.
- Charge (Prince): after walking an uninterrupted `charge_range` (data `charge_range_raw` in
  centitiles → 2500 millitiles) the unit is charged: movement speed ×2 (percent from data) and the
  next hit deals `damage_special` (306 → 783 at L11) and lands with no windup
  (`progress` snaps to the next multiple of hit_speed). Charge resets on hitting, stun/freeze,
  knockback, or stopping.

### 6.3 Movement & pathing

- A deployed, non-stunned unit that is not attacking moves each tick toward its goal by
  `step = effective_speed` millitiles, **never overshooting** its current waypoint/goal (cap step at
  remaining distance). Direction vector components truncate toward zero.
- Effective speed = `floor(speed * (100 + slow_pct) / 100)`; ×2 when charged; 0 when stunned/frozen.
- Goal: the target's position if it has an attack target not yet in range, otherwise the default
  route (§6.1).
- **Ground pathing**: over the 36×64 half-tile grid. Water is impassable for ground units except
  **jump-enabled** units (Hog Rider, Prince: they may cross water cells anywhere). Tower and
  building footprints are high-cost/occluding for ground units other than the goal. Algorithm is
  **[IMPL-DEFINED]** (A*/Dijkstra/flow field) but must be deterministic and must never leave a unit
  stuck: a lone ground unit placed on any legal tile of its half with no enemies must reach attack
  range of its default tower. Re-plan when the goal cell changes or the building set changes.
- **Flying** units move in a straight line; they ignore water and buildings.
- Units stay inside the arena.

### 6.4 Collision

- Ground–ground and air–air pairs overlap when `dist < r_a + r_b`; resolve by pushing apart along
  the centre line, splitting the overlap by inverse mass (`a` moves `overlap·m_b/(m_a+m_b)`).
  Coincident centres: deterministic direction **[IMPL-DEFINED]**. At least one pass per tick in
  deterministic order.
- Ground units are pushed fully out of building/tower collision circles (buildings never move).
- Ground non-jumpers must not end a tick inside water; flying units are unaffected.

### 6.5 Buildings

- Lifetime decay (ledger `lifetime.HP_DECAY = linear_drain`): a building with `lifetime_ms = L`
  loses HP so that it reaches 0 exactly `L` ms after deploy completes
  (per tick: `floor(max_hp·(k+1)·50/L) - floor(max_hp·k·50/L)` at elapsed tick k).
  Cannon dies at 30 s after deploy end, Tesla at 25 s.
- Tesla hide (ledger `hide.*`): after deploy it is **hidden** (untargetable, immune to all damage);
  when a valid enemy is within its sight it **rises** over `up_time_ms` (targetable while rising,
  can attack once up); it re-hides after `hide_time_ms` with no target.

### 6.6 Deaths

- Death damage (Ice Golem: 84 at L11, radius 2000) hits enemy ground+air within `radius + r_target`.
- Death area effect (Ice Golem `FreezeIceGolemite`): radius 2000, applies slow buff
  (-30% speed, -30% hit speed) for 2000 ms to enemies it touches (incl. hidden).
- When a crown tower is destroyed: crowns/King trigger/pocket (§3.1, §4).

### 6.7 Status effects

- **Stun/Freeze** (speed & hit-speed multipliers -100: Zap's `ZapFreeze` 500 ms, Ice Spirit
  `Freeze` 1100 ms): target cannot move or attack; on application attack progress resets to 0,
  charge resets, target is cleared. Crown towers can be stunned.
- **Slow** (-30%): speed and hit speed scaled (§6.2, §6.3).
- Re-application of the same buff refreshes its remaining duration to the max of old and new.

---

## 7. Projectiles & spells

### 7.1 Unit/tower projectiles

Launched at the attacker's position offset by `projectile_start_radius` toward the target;
appears the tick after the hit tick. Moves `speed` millitiles/tick toward the target's current
position (homing). Impact when the remaining distance ≤ speed. On impact: single-target damage to
its target if still alive; splash projectiles (`radius > 0`) damage all valid enemies within
`radius + r_target` of the impact point (respecting `aoe_to_air`/`aoe_to_ground`), and apply
`target_buff` if any. If the target died before impact: single-target projectiles vanish;
splash projectiles fly on to the target's last position and splash there **[IMPL-DEFINED, documented]**.

### 7.2 Spells

All spell damage uses the crown-tower reduction rule (§2).
- **Fireball**: projectile from the caster's King tower centre to the tap point at 600/tick; on
  arrival 688 damage to all enemy ground+air entities within 2500 + r_target (towers 172);
  knockback 1000 away from the impact centre for troops without `ignore_pushback`.
- **Arrows**: 3 waves, 200 ms apart; the first wave lands when a projectile flying at 1100/tick
  from the caster's King tower centre would reach the tap point; each wave deals 122 to every enemy
  ground+air entity within 3500 + r_target (towers 25). A unit is hit at most once per wave.
- **Zap**: instant at the tap point (applied during the play tick's processing): 192 damage
  (towers 48) and a 500 ms stun to enemy ground+air entities (including crown towers) within
  2500 + r_target.
- **The Log**: a rolling hitbox starting at the tap point, rolling toward the enemy side (own-frame
  "up") at 200/tick for 10100 millitiles. Hitbox: half-width 1950 (x) × half-depth 600 (y) around
  the log centre, tested against target circles. Each **ground** enemy (troops, buildings, towers)
  is hit at most once: 268 damage (towers 35) and knockback 700 in the roll direction for troops
  (respect the data's `pushback_all`/`ignore_pushback` semantics as RoyaleSim does — document).
  Never hits air units.
- **Goblin Barrel**: projectile from the caster's King tower centre to the tap point at 400/tick;
  on landing spawns 3 Goblins (HP 202, dmg 125, speed 120, melee range 500, deploy 1100 ms) in the
  §5.1 triangle. No impact damage.

Knockback: displacement of the troop by the pushback distance along the push direction
(instantaneous or over a few ticks, **[IMPL-DEFINED]**), clamped to land/arena; it resets the
attack cycle and charge, keeps the target.

---

## 8. Scripted bots (C, deterministic given the bot RNG)

- `noop`: never plays.
- `random`: each decision, with probability `bot_play_prob` (default 0.2) picks uniformly among
  **legal** non-noop actions, else noop.
- `heuristic`: a simple rule bot [IMPL-DEFINED rules, documented], e.g. defends units on its side,
  counter-pushes at the bridge, casts spells on clusters/towers. Must only issue legal actions.

---

## 9. PufferLib environment (`pufferroyale.Royale`)

Built as a PufferLib **3.0** native env: `pufferroyale/binding.c` (uses the vendored
`env_binding.h`), `pufferroyale/csrc/royale.h` (Env struct + `c_reset/c_step/c_render/c_close`),
`pufferroyale/royale.py` (`class Royale(pufferlib.PufferEnv)`).

Constructor: `Royale(num_envs=1, num_agents=2, frame_skip=10, deck0='hog26', deck1='hog26',
opponent='heuristic', learner_side='random', deploy_lockout_ticks=90, tiebreak='absolute',
reward_tower=0.0, reward_crown=0.0, bot_play_prob=0.2, log_interval=128, render_mode=None, buf=None, seed=0)`.
- `num_agents` per env: 2 = both teams controlled by the policy (self-play; agent row `2i` is
  team 0, row `2i+1` team 1); 1 = the policy controls one team and the other team is the scripted
  `opponent` bot; the learner's team is `learner_side` (0, 1, or `'random'` per episode).
- `self.num_agents` (total rows) = `num_envs * num_agents`.
- **Action space**: `Discrete(2305)`. `0` = no-op. `a ≥ 1`: `slot = (a-1) // 576`,
  `cell = (a-1) % 576`, `tx = cell % 18`, `ty = cell // 18`, in the acting team's own frame.
- One env step = `frame_skip` engine ticks. The agent's action is applied on the first tick of the
  step (subject to §3.1). An illegal action is treated as no-op and counted (`illegal_actions` log).
- **Observation**: `Box(float32, shape=(OBS_SIZE,))`, own frame, with fixed sections and
  offsets exported as Python constants in `pufferroyale.royale` (`OBS_SIZE`, `SPATIAL_OFFSET`,
  `SPATIAL_SHAPE=(C,32,18)`, `ENTITY_OFFSET`, `ENTITY_SHAPE=(48,F)`, `SCALAR_OFFSET`, `SCALAR_SIZE`,
  `MASK_OFFSET`, `MASK_SIZE=2305`) and mirrored in C. Required contents:
  1. Spatial planes (C channels × 32 × 18, own frame, values in [0,1]): per side (own, enemy):
     ground troop count, air troop count, building presence, crown-tower footprint, HP sum,
     buildings-only-targeter count, ranged count, splash count, deploying count, debuffed count;
     shared: water, own troop-legal tiles, own building-legal tiles, own active spell areas,
     enemy active spell areas. **[IMPL-DEFINED normalisation, documented]**
  2. Entity list: 24 own + 24 enemy entity slots (non-tower entities, sorted by entity id,
     zero-padded), each with card one-hot (32; spawned Goblins map to Goblin Barrel, Skeletons from
     Skeleton Army map to card 8), own-frame x/18000, y/32000, hp fraction, hp/2000 (clipped),
     flying, deploying, stunned/frozen, slowed, is_building, target_only_buildings.
  3. Scalars: own elixir/10 (exact, from units), hand card one-hots (4×32), hand costs/10,
     next card one-hot (32), per-slot affordable flags (4), tick/6000, is_overtime, elixir rate/3,
     lockout active, own tower HP fractions [King, own-frame-left, own-frame-right], enemy tower HP
     fractions [same], King-active flags (own, enemy), crowns own/3 and enemy/3, opponent cards
     seen (32 binary), opponent elixir spent (elixir/100), opponent last-4 played (4×32, most recent
     first), opponent **deduced hand** (32 binary; nonzero only once all 8 opponent cards have been
     seen — then exactly the 4 cards not among the last 4 played), opponent elixir upper-bound
     estimate `min(10, 6 + income(t) - spent)/10`.
  4. Action mask (2305 floats ∈ {0,1}): `mask[0] = 1` always; `mask[a] = 1` iff action `a` would
     be accepted by the engine right now (card affordable, tile legal, not locked out). The mask
     MUST equal the engine's legality function exactly.
  The observation must never contain the opponent's hidden information (current elixir, hand,
  queue/next card, unrevealed deck) except the deductions listed above.
- **Reward** (per agent, per step): terminal `+1/-1/0` on the step the match ends; plus optional
  shaping (default 0): `reward_tower × (Δ enemy tower HP fraction lost − Δ own tower HP fraction
  lost)` summed over the three towers, and `reward_crown × (Δ own crowns − Δ enemy crowns)`.
  **Zero-sum invariant**: in self-play, `r(team0) + r(team1) == 0` on every step.
- **Terminals**: `terminals = 1` for both agents on the step the match ends; the env then resets
  itself within the same `c_step` so returned observations belong to the new match (PufferLib 3.0
  convention). `truncations` always 0. Max match length 6000 ticks ⇒ ≤ 600 steps at frame_skip 10.
- **Log** (`struct Log` of floats, `n` last), aggregated by `vec_log`: at least `episode_return`
  (team 0), `episode_length` (steps), `score` (team-0 win=1/draw=0.5/loss=0), `perf` (= score),
  `win_0`, `win_1`, `draw`, `overtime`, `tiebreak`, `crowns_0`, `crowns_1`, `leaked_0`, `leaked_1`
  (elixir), `plays_0`, `plays_1`, `illegal_actions`, `spawn_overflow`, `n`.
- `render_mode='ansi'` returns a text rendering of the board; a raylib renderer is available when
  compiled with raylib (`PR_RAYLIB=1`).

---

## 10. Scenario / debug API (`pufferroyale.Game`) — for tests and tools

Single-match, tick-level access through the same compiled engine:

```python
g = Game(deck0='hog26', deck1='giant', seed=0, deploy_lockout_ticks=90, tiebreak='absolute')
g.reset(seed=None)                    # re-deal (new shuffle from seed)
g.tick(n=1)                           # advance n engine ticks, no plays
g.play(team, slot, x, y) -> int       # queue a play for the next tick; returns error code (0=OK)
                                      #   (x, y) engine-frame millitiles of the tap point
g.play_tile(team, slot, tx, ty) -> int  # same, own-frame tile
g.spawn(team, card_id, x, y, deployed=True) -> list[int]   # TEST HOOK: bypass elixir/legality
g.entities() -> list[dict]            # keys: id, team, card_id, unit (str), kind ('troop'|'building'|'tower'),
                                      #   x, y, hp, max_hp, shield, radius, flying, deploying, stunned,
                                      #   slowed, hidden, target_id (-1 none), charged
g.projectiles() / g.effects() -> list[dict]
g.state() -> dict                     # tick, elixir (units, per team), hand, queue, crowns, towers
                                      #   [{hp,max_hp,alive,active}]*3 per team, over, result, end_reason, overtime
g.set_elixir(team, units); g.set_tower_hp(team, idx, hp); g.set_hand(team, deck8_order)
g.legal_mask(team) -> np.ndarray[2305] uint8
g.hash() -> int; g.snapshot() -> bytes; g.restore(bytes)
card_info(card_id) -> dict            # level-11 stats as compiled into the engine
CARD_NAMES, DECKS                     # tuple of 21 names (index = id); dict of preset decks
```
Error codes (`pufferroyale.PlayError`): `OK=0, NOT_IN_HAND/BAD_SLOT=1, NOT_ENOUGH_ELIXIR=2,
ILLEGAL_POSITION=3, LOCKOUT=4, GAME_OVER=5`.

---

## 11. Non-functional requirements

- Build: `pip install -e .` (or `python setup.py build_ext --inplace`) builds
  `pufferroyale.binding`; `make test-c` builds and runs C tests; `make asan` runs them under
  AddressSanitizer + UBSan; everything compiles warning-free with `-Wall -Wextra` on Apple clang.
- Performance (Apple M4, one core, release build): engine ≥ **20,000 ticks/s** in a typical match;
  env ≥ **2,000 steps/s** (frame_skip 10, 2 agents, obs included). Measured numbers reported.
- The C sources contain no `float`/`double` in simulation code (obs/log/render excluded;
  enforced by a test that greps the simulation headers).
- Deterministic across repeated runs and across processes.
- Training integration: `scripts/train.py` runs PufferLib 3.0 `PuffeRL` on CPU with the provided
  policy (`pufferroyale.torch.Policy`, joint masked logits over 2305 actions, optional LSTM) for a
  short smoke run without errors or NaNs; `scripts/eval.py` evaluates a checkpoint vs scripted bots.
- Docs: `README.md` (build/run/test), `docs/FIDELITY.md` (every [IMPL-DEFINED] choice and known
  divergence from the real game), `docs/DECISIONS.md`.

---

## 12. Test ownership protocol

- The TESTER owns `tests/spec/` (black-box tests derived from this SPEC and from game knowledge /
  reference data, NOT from reading the builder's implementation). The BUILDER owns
  `tests/builder/` and `tests/c/`. Neither edits the other's tests.
- If a tester test contradicts the SPEC, or the SPEC is wrong, the orchestrator adjudicates and
  amends this document (bump version, changelog at bottom).
- Done = every test in `tests/` passes, ASan/UBSan clean, `scripts/e2e_check.py` prints PASS.

## 13. Clarifications (v0.2, normative — resolve the tester's ambiguity list)

1. **Tick indexing.** `state()['tick']` = number of ticks processed so far. A play queued (via
   `Game.play*`) when `state()['tick'] == t` is validated and applied in Upkeep of tick index `t`
   (the next `tick()` call). Lockout blocks plays applied in ticks `t < deploy_lockout_ticks`.
2. **Deploy.** A unit created during tick `t` (hand-played or spawned with `deployed=False`) has
   `deploying = True` in the observations taken at the end of ticks `t … t + D/50 − 1` (exactly
   `D/50` observations) and may target/move/attack starting in tick `t + D/50`.
   Staggered formation members: member `i` behaves as if created at tick `t + i·delay/50`
   (for deploy purposes).
3. **Buffs (stun/freeze/slow).** A buff of duration `D` ms applied during tick `t` is active in
   every phase that runs after its application during ticks `t … t + D/50 − 1` and is gone from
   tick `t + D/50`. Hence Zap (applied in Upkeep of cast tick `P`, 500 ms) blocks ticks
   `P … P+9` (10 ticks) and the unit may act again in `P+10`; an Ice Spirit freeze applied in the
   Projectile phase of tick `N` (1100 ms) lets the unit act again in tick `N+22`.
4. **Lifetime drain.** Elapsed tick `k = 0` is the first tick in which the building is not
   deploying (tick `t + D/50`); the building's HP reaches 0 (and it is reaped) at the end of tick
   `t + D/50 + L/50 − 1`.
5. **Sight** stays `dist ≤ sight_range + r_target (+2000 vs crown towers)` (confirmed; the
   ledger's both-radii rule applies to the *attack range* test only).
6. **Lane choice** uses the unit's **own frame**: own-frame `x < 9000` ⇒ own-left, else own-right;
   ties (exactly 9000) ⇒ own-left. Default tower = the enemy tower in that own-frame lane.
7. **Ice Golem death area** is applied **once**, at death, for 2000 ms (ledger
   `spells.ONE_SHOT_AREA_EFFECT_APPLICATION`).
8. **Result types.** `state()['result']` = `[r0, r1]` with `r ∈ {+1, −1, 0}` (0 also while
   running); `end_reason` = `None` while running, else one of the strings `"KING"`,
   `"REGULATION_CROWNS"`, `"OVERTIME_CROWNS"`, `"TIEBREAK"` (decisive tiebreak), `"DRAW"` (every
   drawn result: exact tiebreak tie or both Kings on one tick).
9. **`card_info(card_id)`** must contain at least: `id, name, kind, cost` (alias `elixir`), and for
   troops/buildings `hitpoints, damage, hit_speed_ms, load_time_ms, speed, range, sight_range,
   collision_radius, deploy_time_ms, count, lifetime_ms, crown_tower_damage_percent`; for spells
   `damage, radius, crown_tower_damage` where applicable.
10. **`spawn(deployed=True)`** = the state a hand-played unit has in the first tick after its deploy
    completed: not deploying, `load_timer = max(0, load_time − deploy_time)`.
11. **Entity-row layout** (F = 42 floats, in this order): card one-hot [0:32], x/18000 [32],
    y/32000 [33], hp/max_hp [34], min(1, hp/2000) [35], flying [36], deploying [37],
    stunned-or-frozen [38], slowed [39], is_building [40], target_only_buildings [41]. Own entities
    in slots 0–23, enemy in 24–47, each group sorted by entity id, zero-padded.
    **Scalar layout**: in the order listed in §9 item 3, exported as
    `pufferroyale.royale.SCALAR_INDEX: dict[name → (offset, length)]` (offsets relative to
    `SCALAR_OFFSET`).
12. **Spell timing.** A spell play is applied in Upkeep of tick `P`. Fireball / Goblin Barrel
    projectiles already move in the Projectile phase of tick `P` and land in tick
    `P + ceil(d / v) − 1` (`d` = distance from the caster's King centre to the tap point,
    `v` = speed); Goblin Barrel's Goblins are committed in the Spawn phase of the landing tick + 1
    and then deploy for 1100 ms (clarification 2). Arrows' first wave lands in tick
    `P + ceil(d / 1100) − 1` and the next waves 4 and 8 ticks later. Zap applies in tick `P`.
13. **Seat symmetry.** Every [IMPL-DEFINED] rule must commute with the 180° rotation. The only
    permitted seat-dependent rule is the documented ordering of simultaneous same-tick events
    (team 0's queued plays are applied first).
14. **Logs.** `illegal_actions` = sum over both agents per episode; `tiebreak` = 1 iff the match
    reached the end-of-overtime comparison (decisive or drawn); the spatial "own troop-legal" and
    "own building-legal" channels show placement geometry only (they ignore elixir and lockout;
    the action mask handles those). Formation bound (1600) holds at spawn time.
15. **`scripts/train.py` CLI** accepts at least: `--total-timesteps`, `--device`, `--num-envs`,
    `--num-agents`, `--opponent`, `--deck0`, `--deck1`, `--seed`, `--data-dir`.

## 14. Audit amendments (v0.2.1, normative)

1. **Same-tick play validation.** All plays applied in Upkeep of tick `t` are validated against
   the state at the start of that Upkeep (after elixir regen, before any of this tick's plays are
   applied). A team's own queued plays are validated sequentially against its own running elixir/
   hand; plays of different teams never invalidate each other (a troop landing where the enemy
   placed a building in the same tick is resolved by collision). Therefore every mask-legal env
   action is applied. The Log gains `dropped_plays` (must be 0 in env play).
2. **Melee retarget mid-swing** (ledger `targeting.LOGIC_PRESERVE_TARGET_IF_HIT_STARTED =
   projectile_attackers_only`, measured HIGH): a *direct striker* (no projectile) whose current
   target leaves its attack range while a swing is started switches to the nearest valid enemy that
   is within attack range, **keeping its progress** (the hit lands on schedule on the new target);
   if none is in range, the swing is cancelled (`progress = 0`) and normal acquisition applies. A
   *projectile attacker* keeps a target that leaves reach mid-swing while its distance stays
   within `range + r_a + r_t + 500`, until the shot is fired. A swing is never completed on a target
   outside attack range (melee) — supersedes §6.1's "unless … not mid-cycle" clause for melee units.
3. **Hit-once effects** (the Log, Arrows waves): an entity may only be hit if it can be recorded in
   the effect's hit list; the list capacity must be ≥ `MAX_ENTITIES`.
4. **Formation ejection** stays on the placing team's side: members of a hand-played troop that would
   land in water are moved to the nearest land point on the placing team's own half (or the pocket
   side when the tap was in a pocket), never across the river. Bridge land inside the river band
   counts as land on either side (a member may stand on a bridge); what is forbidden is ending on
   the far bank or in water.
5. **Snapshot restore** validates the snapshot (pool counts within capacity, ids strictly increasing,
   all unit/projectile/effect/card indices in range, tick within [0, 6000], hand/queue/deck valid);
   invalid input raises `ValueError` and leaves the game unchanged.
6. **Logs**: add learner-centric keys `learner_score`, `learner_return`, `learner_win`
   (1-agent mode: the policy's team; self-play: team 0), and `dropped_plays`. `tiebreak` = 1 only if
   the end-of-overtime comparison ran (end_reason TIEBREAK, or DRAW decided by that comparison).
   `royale.ini` sweeps must use `learner_score`.

## 15. Training & evaluation tooling (v0.3 — Phase D)

Python-side tooling on top of PufferLib 3.0. Everything must be deterministic given seeds, run on
CPU for tests, and never let opponent transitions enter the learner's PPO buffer.

### 15.1 `pufferroyale.league.OpponentPool`
- Opponent specs (strings): `"self"` (the current learner policy — mirrored self-play),
  `"bot:noop" | "bot:random" | "bot:heuristic"`, `"ckpt:<path>"` (a frozen policy checkpoint).
- `OpponentPool(anchors=("bot:heuristic",), max_snapshots=16, pfsp="hard", pfsp_eps=0.05,
  self_play_frac=0.2, anchor_frac=0.2, seed=0)`:
  - `add_snapshot(path)` appends `ckpt:<path>`; beyond `max_snapshots` the **oldest non-anchor**
    snapshot is evicted.
  - `record(opponent, learner_result)` with `learner_result ∈ {+1, 0, −1}` updates games / wins /
    draws / losses and the learner's score vs that opponent `p = (wins + 0.5·draws) / games`
    (`p = 0.5` for an unseen opponent).
  - `sample()` returns one opponent spec: with probability `self_play_frac` → `"self"`; else with
    probability `anchor_frac` → an anchor chosen uniformly; else a snapshot chosen by PFSP weight
    `w(p)`: `hard` = `(1 − p)^2`, `variance` = `p·(1 − p)`, `uniform` = 1, each floored at
    `pfsp_eps`. If there are no snapshots, the snapshot mass goes to anchors (or `"self"` if no
    anchors). Uses its own seeded RNG.
  - `weights()` returns the current sampling distribution (dict spec → probability, sums to 1).
  - `state_dict()` / `load_state_dict()` (JSON-serialisable) round-trip exactly.

### 15.2 `pufferroyale.league.LeagueVecEnv`
- Wraps `num_envs` Royale matches in self-play buffers (2 agents per match) and exposes a
  PufferLib-vecenv-compatible interface to PuffeRL with **one learner row per match**:
  attributes `num_agents` (= num_envs), `single_observation_space`, `single_action_space`,
  `agents_per_batch` **and** `agent_per_batch`, `driver_env`, `emulated=False`; methods
  `async_reset(seed)`, `send(actions)`, `recv() -> (obs, rewards, terminals, truncations, infos,
  env_ids, masks)`, `close()`. Learner rows only; shapes `(num_envs, …)`.
- Per match and per episode: the learner's seat (team 0/1) is drawn with the wrapper's seeded RNG,
  and an opponent is drawn from the pool. The opponent's action for that match each step comes
  from: `"self"` → the current learner policy (no grad, same device); `"ckpt:…"` → the frozen
  policy (loaded once, cached, batched across matches sharing it; recurrent opponents keep their
  own per-match state, reset at episode start); `"bot:…"` → the C scripted bot for that match/team
  (a binding helper may be added).
- The learner's reward/terminal equal the underlying env's values for the learner's row. On a
  terminal the outcome is `record`ed in the pool and a new seat + opponent are drawn.
- `infos` carry the Royale logs plus league stats: `league/pool_size`,
  `league/p/<opponent>` (learner score vs each opponent), `league/episodes`.
- `set_policy(policy)` provides the current learner policy used for `"self"` opponents.

### 15.3 `pufferroyale.trainer.MMDPuffeRL`
- Subclass of `pufferlib.pufferl.PuffeRL`; `train()` adds `mmd_coef · KL(π_θ ‖ π_ref)` (masked
  categorical distributions over the 2305 actions, averaged over the minibatch) to the PPO loss.
  `π_ref` is a frozen copy of the policy, refreshed every `mmd_ref_interval` epochs (0 = keep the
  initial policy). With `mmd_coef = 0` it must behave exactly like the base `train()`
  (same losses given identical RNG state). Logs `losses/mmd_kl`.

### 15.4 Scripts
- `scripts/league_train.py`: PuffeRL (or MMDPuffeRL when `--mmd-coef > 0`) on a LeagueVecEnv;
  snapshots the learner into the pool every `--snapshot-interval` epochs
  (`<data-dir>/league/<run>/snap_<epoch>.pt`), saves `league_state.json` (pool state + RNG), supports
  `--resume <run dir>`. Flags ≥: `--total-timesteps --device --num-envs --anchors --pfsp
  --self-play-frac --anchor-frac --snapshot-interval --max-snapshots --mmd-coef --mmd-ref-interval
  --seed --data-dir --deck0 --deck1 --resume`.
- `scripts/best_response.py`: exploitability probe — trains a fresh learner against a single frozen
  `--target` checkpoint (pool = that target only) for `--total-timesteps`, then plays `--matches`
  full matches (both seats) and prints/writes JSON with `br_score` (BR's score vs target) and
  `exploitability_proxy = 2·br_score − 1`.
- `scripts/tournament.py`: round-robin between agents given as bot specs and/or checkpoint paths,
  over given decks, `--matches` per pair (half per seat); prints the payoff matrix, Elo, and the
  Nash equilibrium of the empirical meta-game; writes JSON.

### 15.5 `pufferroyale.metagame`
- `solve_zero_sum(A) -> (x, y, value)`: Nash equilibrium of the zero-sum matrix game with row payoff
  matrix `A` (row maximises). Exact to 1e-6 on small games (LP via scipy). Must pass: RPS → uniform,
  value 0; matching pennies → (½,½); strictly dominated rows/columns get 0 mass; 1×1 and
  non-square games.
- `payoff_matrix(results)` builds `P[i][j]` = score of agent i vs j in [0,1] with `P + Pᵀ = 1` off
  the diagonal and 0.5 on it; `metagame_nash(P)` solves the symmetric game `A = P − 0.5`.
- `elo(P, games)` returns ratings (mean 1500) consistent with the score matrix ordering.
- `play_match(agent_a, agent_b, deck_a, deck_b, seed)` plays one full match via `Game`
  (agents: bot specs or loaded policies; greedy or sampled via a flag) → result for agent_a.
- `deck_metagame(agent, decks, matches)`: payoff over deck pairs for a fixed agent, and its Nash
  mixture over decks.

### 15.6 HPC templates (not executed locally)
`hpc/pufferroyale.def` (Apptainer: CUDA 12 base, Python 3.12, torch cu12, PufferLib 3.0 from source
with the documented `NO_OCEAN` fix, this package), `hpc/train.sbatch`, `hpc/league.sbatch`,
`hpc/README.md` (NYU Torch usage: account/partition placeholders, `apptainer exec --nv`, scratch
paths). Must pass `bash -n`; every script/flag they reference must exist.

### 15.7 Clarifications (v0.3-D.1, normative)
1. **Pool mass split is absolute:** P(`"self"`) = `self_play_frac`; P(anchors, uniform among them) =
   `anchor_frac`; P(snapshots, PFSP) = `1 − self_play_frac − anchor_frac`; the constructor requires
   the sum ≤ 1. Fallbacks: no snapshots → the snapshot mass goes to anchors (to `"self"` if there are
   no anchors); no anchors → the anchor mass goes to snapshots (to `"self"` if there are none).
2. `pool.stats() -> {spec: {"games", "wins", "draws", "losses", "p"}}`. An evicted snapshot's stats are
   dropped. `add_snapshot` never touches the filesystem.
3. `state_dict()` contains the snapshot list, stats and RNG state; `load_state_dict` is called on a
   pool built with the same constructor arguments.
4. `LeagueVecEnv(pool, num_envs=1, seed=0, device="cpu", opponent_greedy=False, **royale_kwargs)`;
   `royale_kwargs` (frame_skip, deck0, deck1, log_interval, tower troops, …) are forwarded to
   `Royale(num_agents=2, …)`.
5. Royale logs are forwarded unchanged at Royale's `log_interval`; `league/*` keys are attached to
   every non-empty info. `league/pool_size` = number of distinct opponent specs with non-zero
   sampling probability. Inside the league the `learner_score/learner_return/learner_win` keys are
   rewritten to refer to the learner's seat.
6. `"self"`/`ckpt:` opponent actions are **sampled** from the masked policy distribution with a
   seeded generator owned by the wrapper (greedy argmax when `opponent_greedy=True`).
7. `ckpt:` files are exactly what PuffeRL saves: `torch.save(policy.state_dict())` of
   `pufferroyale.torch.Policy` (default sizes) or `Recurrent(Policy)`; recurrent checkpoints are
   detected by `lstm.*` keys.
8. `MMDPuffeRL(config, vecenv, policy, logger=None)` reads `config["mmd_coef"]` and
   `config["mmd_ref_interval"]`. The reference refresh happens after `train()` increments `epoch`,
   when `interval > 0 and epoch % interval == 0`. "Same losses as base" is measured from `train()`
   onward (the constructor may consume RNG; tests re-seed after construction). `losses/mmd_kl` is
   the mean over the epoch's minibatches.
9. `payoff_matrix(results, n_agents)`: `results` is an iterable of `(i, j, r)` with `r ∈ {+1, 0, −1}`
   = agent i's result. `metagame_nash(P) -> (x, value)`. `elo(P, games) -> ratings`: `games` is a
   count matrix; maximum-likelihood Bradley–Terry ratings on the `400·log10` Elo scale, centred at
   1500.
10. `play_match(agent_a, agent_b, deck_a, deck_b, seed, greedy=True) -> +1/0/−1` for agent_a, who plays
    team 0. `deck_metagame(agent, decks, matches) -> (P, x)` (antisymmetric deck-vs-deck score
    matrix and its Nash mixture over decks).
11. `tournament.py --agents A [A …] --decks D [D …] --matches N --out FILE [--seed --device]`; JSON keys
    `agents`, `payoff`, `elo`, `nash`. `best_response.py --target CKPT --total-timesteps T --matches
    M [--device cpu --seed --out FILE]`; the last stdout line is the JSON.
12. `league_train.py`: `--anchors` is comma-separated; the run directory is
    `<data-dir>/league/<run_id>/` holding `snap_<epoch>.pt` (epoch = PuffeRL epoch after `train()`
    increments) and `league_state.json`; with `--resume` the `--total-timesteps` value is the absolute
    target.
13. HPC templates use only flags that appear in the scripts' `--help`.
14. (Builder-2 choices, accepted.) `tournament.py` plays `--matches` per agent pair **per deck
    pairing** (all decks×decks by default; `--deck-mode mirror` restricts to equal decks), seats
    alternating, greedy by default (`--sample` to sample); `best_response.py` evaluates with sampled
    actions by default (`--greedy`). `elo` falls back to one virtual draw per pair only when the
    maximum-likelihood ratings do not exist, reported as `elo_regularised`. Pool corner cases:
    `weights()` lists only non-zero specs; results against an evicted snapshot are ignored;
    re-adding an existing spec is a no-op; with `pfsp_eps=0` and all PFSP weights zero the snapshot
    mass is uniform.

## 16. Card batch 2 + tower troops (v0.3 — Phase E)

Adds 43 cards (ids 21–63, stable, appended after v0.1's 0–20) and the three non-default tower
troops. Same data source and level-11 rules as §2. Everything not pinned here is [IMPL-DEFINED]
and must be documented in `docs/FIDELITY.md` with the RoyaleSim ledger key / data column it follows.
All new behaviour must keep determinism, seat symmetry (§13.13), integer-only simulation, and the
v0.2.1 rules.

### 16.1 New cards (id — display name — `cards-15.535.json` key — mechanics)
| id | name | key | mechanics |
|---|---|---|---|
| 21 | Barbarians | Barbarians | ×5 melee |
| 22 | Mini P.E.K.K.A | MiniPekka | melee |
| 23 | P.E.K.K.A | Pekka | melee |
| 24 | Mega Minion | MegaMinion | air, ranged |
| 25 | Bats | Bats | ×5 air melee |
| 26 | Spear Goblins | SpearGoblins | ×3 ranged |
| 27 | Goblins | Goblins | ×4 melee |
| 28 | Goblin Gang | GoblinGang | 3 `Goblin_Stab` + 3 `SpearGoblin` (`second_summon`) |
| 29 | Royal Giant | RoyalGiant | ranged buildings-only targeter |
| 30 | Bomber | Bomber | ground splash projectile |
| 31 | Princess | Princess | long-range splash projectile |
| 32 | Dart Goblin | BlowdartGoblin | ranged |
| 33 | Minion Horde | MinionHorde | ×6 air ranged |
| 34 | Fire Spirit | FireSpirits | kamikaze splash |
| 35 | Ice Wizard | IceWizard | splash projectile + slow `target_buff` |
| 36 | Golem | Golem | buildings-only; death damage; death spawn 2 Golemites; deploy 3000 ms |
| 37 | Lava Hound | LavaHound | air buildings-only; death spawn 6 LavaPups |
| 38 | Balloon | Balloon | air buildings-only; death bomb (`BalloonBomb`: timed area hit after its deploy time, as RoyaleSim `convert_death_bomb`) |
| 39 | Giant Skeleton | GiantSkeleton | death bomb (`GiantSkeletonBomb`) |
| 40 | Witch | Witch | splash projectile; periodic spawner (Skeletons) |
| 41 | Tombstone | Tombstone | building spawner (Skeletons); death spawn 4 Skeletons |
| 42 | Inferno Tower | InfernoTower | building; **variable damage** (3 stages from raw `Damage`, `VariableDamage2/3`, `VariableDamageTime1/2`; resets on retarget and on stun/freeze) |
| 43 | Inferno Dragon | InfernoDragon | air; variable damage (same rule) |
| 44 | Bandit | Assassin | dash (`dash.*`: triggers between min and max range, dash speed, dash damage, invulnerable while dashing) |
| 45 | Battle Ram | BattleRam | buildings-only; charge; kamikaze on hit; death spawn 2 Barbarians |
| 46 | Royal Hogs | RoyalHogs | ×4 buildings-only; river jump |
| 47 | Miner | Miner | may be placed on any non-water tile not inside a living enemy crown-tower footprint (incl. the enemy half); travels underground from its King at `spawn_pathfind.speed`, untargetable and inactive while burrowing, then deploys; crown-tower damage 20% |
| 48 | Mortar | Mortar | building; long range; **minimum range** 3500 (cannot target closer); splash projectile |
| 49 | X-Bow | Xbow | building; long range |
| 50 | Elixir Collector | Elixir Collector | building; +`ManaCollectAmount` elixir every `ManaGenerateTimeMs` while alive (not while deploying), +`ManaOnDeath` when it dies (not when it expires); capped at 10, overflow counts as leak |
| 51 | Bomb Tower | BombTower | building; ground splash projectile; death bomb (`BombTowerBomb`) |
| 52 | Rocket | Rocket | projectile spell from King; radius 2000; knockback; crown % |
| 53 | Poison | Poison | pulsing area 8 s (hit speed 250) applying a damage-over-time + slow buff (`buff.damage_per_second`, `hit_frequency_ms`, `speed_multiplier`, own crown %); stacking rule per data (`enable_stacking`) [IMPL-DEFINED details] |
| 54 | Freeze | Freeze | area: damage + freeze buff 4 s (incl. crown towers; affects hidden) |
| 55 | Earthquake | Earthquake | pulsing area 3 s, ground only; DoT buff with `building_damage_percent` 350 vs buildings (not crown towers? follow data `crown_tower_damage_percent`); slow |
| 56 | Lightning | Lightning | hits up to 3 highest-HP valid enemies in radius (one bolt per `hit_speed`), each bolt = projectile damage + 500 ms stun; crown % |
| 57 | Giant Snowball | Snowball | projectile spell: damage, knockback 1800, slow 3 s |
| 58 | Barbarian Barrel | BarbLog | rolling (like The Log, shorter range, same territory rule as the Log) then spawns 1 Barbarian at the end of the roll |
| 59 | Rascals | Rascals | 1 RascalBoy + 2 RascalGirls (`second_summon`) |
| 60 | Elite Barbarians | AngryBarbarians | ×2 melee |
| 61 | Skeleton Dragons | SkeletonDragons | ×2 air splash |
| 62 | Wall Breakers | Wallbreakers | ×2 buildings-only kamikaze splash |
| 63 | Night Witch | DarkWitch | periodic spawner (Bats); death spawn 1 Bat |

Spawned units (Golemite, LavaPups, Skeleton, Bat, Barbarian, SpearGoblin, Goblin_Stab, RascalGirl…)
map in observations to the card that created them. Spawners (§16.2) follow RoyaleSim's measured
`spawner.*` keys where applicable (first wave timing, pause, radius, death-spawn layout); ranges of
deaths/spawns are [IMPL-DEFINED] but deterministic and rotation-symmetric.

### 16.2 Mechanic rules (normative where stated)
- **Periodic spawner**: after deploy completes, waves of `spawner.number` units every
  `pause_time_ms` (with `start_time_ms` for the first wave, `interval_ms` between units in a wave);
  paused while stunned/frozen; spawned units are born deployed (no deploy time) unless data says
  otherwise; count toward entity capacity.
- **Death spawn**: on death, `count` units at `radius` around the death point (ring layout, facing
  rule [IMPL-DEFINED]); committed in the next Spawn phase (§5).
- **Death bomb**: a timed area hit at the death point after the bomb's `deploy_time_ms`, dealing its
  `death_damage` to enemies within `death_damage_radius + r_target` (ground and air per data); not a
  targetable entity.
- **Variable damage**: stage 1 `Damage` for the first `VariableDamageTime1` ms on the same target,
  then stage 2 for `VariableDamageTime2` ms, then stage 3 thereafter; retarget/stun/freeze reset to
  stage 1.
- **Minimum range**: a target closer than `minimum_range + r_a + r_t` is not valid for acquisition
  and is dropped.
- **Dash**: ledger `combat.DASH_ATTACK`; invulnerable (no damage taken) while dashing.
- **Burrow (Miner)**: invisible/untargetable/immune while underground; emerges at the tap point.
- **Elixir Collector**: as in the table; its production is visible to the opponent only through the
  entity (not via hidden elixir).
- **Buildings-only kamikaze (Wall Breakers)**: dies when its hit lands; splash per data.

### 16.3 Tower troops
Config `tower_troop0`, `tower_troop1` ∈ `{"princess", "cannoneer", "dagger_duchess", "royal_chef"}`
(default `"princess"`) on `Game` and `Royale`. It replaces **both** Princess towers of that team:
stats from the data rows `Cannoneer`, `DaggerDuchess`, `ChefTower` scaled with the §2 **tower ladder**
at level 11 (Princess rates). Behaviours:
- **Cannoneer**: slow, high-damage single-target tower (data).
- **Dagger Duchess**: attack sequence/ammo per data (`AttackSequence`): fires the sequence quickly,
  then reloads; [IMPL-DEFINED exact reload rule, documented].
- **Royal Chef**: periodically levels up an allied troop (data buff `ChefTower_increase_level_buff`;
  +1 level = stats × (next ladder % / current)), [IMPL-DEFINED targeting/period, documented].
- The opponent's tower troop is **visible** in observations (public information).

### 16.4 Observation & API changes (v0.3)
- `CARD_SLOTS` becomes **128** (room for the full roster). To keep observations compact, **card
  identities are encoded as integer ids** `card_id + 1` (0 = empty) stored as floats in: entity
  rows (replacing the 32-wide one-hot with 1 float → F = 11), hand slots (4), next card (1),
  opponent last-4 played (4). Multi-hot sets stay multi-hot over 128: opponent cards seen, opponent
  deduced hand. Scalars add own and enemy tower-troop one-hots (4 each). The policy embeds ids with
  `nn.Embedding(CARD_SLOTS + 1, d)`. All offsets remain exported (`SCALAR_INDEX` etc.).
- Entity list grows to **32 own + 32 enemy** slots (ENTITY_SHAPE = (64, 11)); the entity-row layout
  becomes: card id+1 [0], x/18000 [1], y/32000 [2], hp/max_hp [3], min(1, hp/2000) [4], flying [5],
  deploying [6], stunned-or-frozen [7], slowed [8], is_building [9], target_only_buildings [10].
- Opponent elixir spent is normalised by **200** (keeps it within [0,1] for a full 5-minute match).
- `N_CARDS = 64`, `CARD_NAMES` has 64 entries; `DECKS` adds exactly these presets:
  - `golem`: Golem, Night Witch, Baby Dragon, Lightning, Mega Minion, Barbarian Barrel, Mini P.E.K.K.A, Zap
  - `lavaloon`: Lava Hound, Balloon, Minions, Mega Minion, Skeleton Dragons, Arrows, Fireball, Tombstone
  - `xbow`: X-Bow, Tesla, Archers, Knight, Skeletons, Ice Spirit, Fireball, The Log
  - `miner_poison`: Miner, Poison, Goblin Gang, Bats, Inferno Tower, Valkyrie, Spear Goblins, The Log
  - `pekka_bridge`: P.E.K.K.A, Battle Ram, Bandit, Minions, Musketeer, Zap, Poison, Dart Goblin
  - `royal_hogs`: Royal Hogs, Earthquake, Fire Spirit, Barbarian Barrel, Goblin Gang, Mega Minion, Zap, Musketeer
  `random` samples from all 64.
- Bots (§8) must handle every card (the heuristic may treat new cards generically by kind).

### 16.6 Clarifications (v0.3-E.1, normative)
1. **Princess (31)** fires `PrincessProjectile` (the unit's `CustomFirstProjectile`): 66 → **168** at
   L11, splash radius 2000, ground+air; the 5 `PrincessProjectileDeco` volleys are cosmetic (no
   damage). `card_info` damage = 168.
2. **Non-homing projectiles** (e.g. Princess, Mortar, Bomb Tower shells, Wall Breakers) fly to the
   target's position **at launch** and apply their effect there (they can miss a moving target);
   homing ones follow §7.1.
3. **Miner placement**: any non-water tile outside every **living** crown-tower footprint (either
   team); building footprints and no-deploy cells are allowed.
4. **Miner travel**: from the play tick the Miner is an entity with `hidden=True` (untargetable,
   immune, inactive) moving `spawn_pathfind.speed` (650) per tick in a straight line from its own
   King's centre; it surfaces (`hidden=False`) at the tap point after `ceil(d/650)` ticks and then
   deploys for its `deploy_time` (targetable while deploying).
5. **Spawner timing**: first wave `start_time_ms/50` ticks after deploy completes (if `start_time`
   is null, `pause_time_ms/50` after); waves repeat every `pause_time_ms`, measured start-to-start;
   units within a wave are `interval_ms` apart. Paused (timers frozen) while stunned/frozen.
6. `entities()['card_id']` of a spawned/death-spawned unit = the creating card (same mapping as
   observations).
7. **Death spawns and death bombs** fire on **any** death, including lifetime expiry. Only the Elixir
   Collector's `ManaOnDeath` is limited to destruction by damage.
8. **Death-bomb timing**: the bomb is committed in the Spawn phase after the death tick and detonates
   `deploy_time_ms/50` ticks later (so its damage is observable at death-observation + D/50 + 1 ±0).
9. **Variable damage**: the stage clock starts at the first hit on the current target; each stage
   lasts `VariableDamageTime/hit_speed` hits (Inferno: 5 hits per stage); retarget/stun/freeze reset
   to stage 1.
10. **Earthquake**: 3 damage events, one per second (`hit_frequency_ms`), each `DPS·1s` = 81 at L11 to
    troops, `floor(81·350/100) = 283` to non-crown buildings, `ceil(81·60/100) = 49` to crown towers;
    ground only; slow −50% while affected.
11. **Poison**: exactly 8 events of `floor(36·2.56) = 92`, 20 ticks apart (from the application tick
    P); crown towers `ceil(92·23/100) = 22`; −15% slow while inside + 1 s buff; two Poisons stack
    (`enable_stacking`).
12. **Area spells** (Freeze, Poison, Earthquake, Lightning) apply their first effect in tick P, like Zap.
    **Rocket / Snowball** land like Fireball (§13.12).
13. **Lightning**: targets = the 3 highest *current-HP* valid enemies within radius at cast (ties →
    lowest id); bolt k ∈ {0,1,2} strikes in tick `P + ceil(460·k/50)`; each bolt deals the projectile
    damage (crown %) + 500 ms stun; a target that died before its bolt is skipped.
14. **Bandit dash** triggers when `min_range + r_a + r_t ≤ dist ≤ max_range + r_a + r_t` and the target is
    not yet in attack range.
15. **Elixir Collector**: +2800 units at elapsed tick 259 after deploy completes (k = 0 is the first
    non-deploying tick), then every 260 ticks; +2800 in the tick it is destroyed by damage; nothing on
    expiry; overflow above 10 counts toward `leaked_*`.
16. **Tower-troop one-hots**: index order `princess, cannoneer, dagger_duchess, royal_chef`;
    `SCALAR_INDEX` keys `own_tower_troop`, `enemy_tower_troop`, appended after `opp_elixir_ub`.
17. **Multi-hot index** = `card_id` (not +1); slots 64–127 are always 0 in v0.3.
18. **Royal Chef**: every N ms (N from data, [IMPL-DEFINED], documented) levels up the allied troop with
    the highest elixir cost within 7500 of an own Chef tower (ties → lowest id); stats multiplied by
    `ladder[L+1]/ladder[L]`, floored; whether a unit can be levelled more than once, and the cap, are
    [IMPL-DEFINED, documented].
19. Invalid `tower_troop` value → `ValueError`.
20. **Barbarian Barrel**: rolls from the tap point 4500 at 200/tick (hitbox 1300×600, each ground enemy
    hit once); one Barbarian is committed at the end point in the next Spawn phase, deploying 1000 ms.
21. **Freeze vs hidden Tesla**: the freeze buff applies (it `affects_hidden`); the damage does not.
22. **Entity overflow in obs**: more than 32 non-tower entities on a side → the 32 lowest ids are shown.
23. **Miner surfacing** (pins item 4): `hidden=True` in exactly `N = ceil(d/650)` observations (ticks
    `P … P+N−1`); `hidden=False` from tick `P+N`, then deploying for `deploy_time/50` observations.
24. **Spawner first wave** (pins item 5): at elapsed tick `k = start_time_ms/50 − 1` where `k = 0` is the
    first non-deploying tick (same convention as the Collector, item 15).
25. **Lightning order**: bolt k strikes the k-th highest current-HP target chosen at cast.
26. **Royal Chef period**: `N = 28000 ms` (the data carries no timing field; value from the
    community-measured 21–35 s range, [IMPL-DEFINED], documented); first level-up after the first full
    period; "within 7500" is centre-to-centre distance to that Chef tower.
27. Invalid `tower_troop0/1` → `ValueError` on both `Game` and `Royale`.
28. **Royal Chef timers are per tower**: each living Chef tower runs its own 28 s period (so two living
    Chef towers can level two troops per period); units created by spawners/deaths/spells rank as
    elixir cost 0 for the Chef's choice.
29. **Burrowing Miner is public**: the real game shows the Miner's tunnel to both players, so the
    burrowing Miner appears in both observations (encoded with `deploying = 1`), while remaining
    untargetable, immune and inactive ("invisible" in §16.2 means untargetable).
30. **Bandit dash** may additionally require that the dash segment does not cross water
    ([IMPL-DEFINED], documented); the §16.6.14 range window is necessary, not sufficient.

### 16.5 Acceptance
- Every new card has codegen-asserted level-11 stats and at least one behavioural scenario test.
- All previous tests keep passing, except observation-layout tests superseded by §16.4.
- Performance: engine ≥ 20k ticks/s in typical matches with random decks from all 64 cards; the
  §14 stress scenario ≥ 10k ticks/s.

## 17. LLM play interface (v0.4 — Phase F, from `ideas.txt`: "play against LLMs and compare models")

A provider-agnostic harness so a language model can play PufferRoyale through text, plus an
optional Anthropic adapter. **Tests must never touch the network**; everything is exercised with
deterministic mock callables.

### 17.1 `pufferroyale.llm`
- `RULES_PROMPT: str` — a static system prompt (rules summary, coordinate convention — own frame,
  tile `(tx, ty)`, `ty = 31` own back row, `ty ≤ 14` enemy half — card list with costs/roles, and the
  exact response format). Identical across turns and matches (prompt-cacheable: no timestamps/ids).
- `render_state(game, team, *, legal=True) -> str` — deterministic text for one decision in the
  acting team's **own frame**: clock (m:ss, phase, elixir rate), own elixir (1 decimal), hand
  (`slot: Card (cost)` + affordable flag), next card, tower HPs (own/enemy, King active flag), own and
  enemy units (`Card @ (tx,ty) hp/max [flags]`, sorted by id), opponent cards seen and last played,
  opponent elixir upper-bound estimate, and — when `legal=True` — per hand slot a compact list of legal
  tile regions (row-run encoding, e.g. `ty=20: tx 0-5,9-17`). It must **never** contain hidden info
  (opponent elixir, hand, queue, unrevealed deck) beyond the §9 deductions.
- `parse_action(text, game, team) -> (action:int, info:dict)` — accepts, case-insensitive, the last
  line matching `WAIT` or `PLAY <slot|card name> AT <tx>,<ty>` (own frame); returns the Discrete(2305)
  action. Unparseable, unknown card/slot, or illegal (mask) ⇒ `(0, {"error": …})` (a no-op).
- `LLMAgent(model_fn, decision_interval=20, max_history=0)` — `model_fn(system: str, user: str) -> str`.
  `act(game, team) -> int` renders, calls, parses; records a transcript entry (prompt hash, response,
  action, error, latency) in `agent.transcript`.
- `play_llm_match(agent, opponent, deck_agent, deck_opp, seed, agent_team=0, decision_interval=20,
  max_ticks=6000) -> dict` — runs a full match via `Game`; `opponent` is a bot spec, a checkpoint path
  or another `LLMAgent`; the agent acts every `decision_interval` ticks (default 20 = 1 s, a realistic
  LLM cadence); returns `{result, crowns, end_reason, ticks, decisions, illegal, parse_errors}`.
- Mock model functions for tests/baselines: `mock_wait` (always WAIT), `mock_first_legal` (plays the
  first affordable card at its first legal tile, via the text it is given — it must parse the
  rendered state, not read the engine), `mock_random(seed)`.

### 17.2 Anthropic adapter (optional; not executed in tests)
`anthropic_model_fn(model="claude-opus-5-5", effort="low", max_tokens=4096)` returns a `model_fn`
using the official `anthropic` Python SDK (imported lazily; clear error if missing):
`client.messages.create(model=…, max_tokens=…, system=[{"type": "text", "text": system,
"cache_control": {"type": "ephemeral"}}], messages=[{"role": "user", "content": user}],
output_config={"effort": effort})` — no assistant prefill; the static `RULES_PROMPT` is the cached
system prompt; `effort` omitted for `claude-haiku-4-5` (unsupported there); thinking left at the
model default (it cannot be disabled on `claude-opus-5-5` / `claude-fable-5-1`); for
`claude-fable-5-1` / `claude-opus-5` include refusal fallbacks
(`betas=["server-side-fallback-2026-07-01"], fallbacks="default"` via `client.beta.messages.create`);
check `stop_reason` (`refusal` ⇒ treat as WAIT and record it); catch `anthropic.RateLimitError` /
`APIStatusError` / `APIConnectionError` specifically and treat a failed call as WAIT. Known model ids:
`claude-opus-5-5`, `claude-opus-5`, `claude-fable-5-1`, `claude-sonnet-5`, `claude-haiku-4-5`.

### 17.3 `scripts/llm_match.py`
`--model {mock_wait,mock_first_legal,mock_random,anthropic:<model-id>} --opponent SPEC --deck-agent D
--deck-opp D --matches N --seed S --decision-interval K --out FILE [--effort low]`; plays N matches
(alternating seats), prints a JSON summary (win/draw/loss, mean crowns, illegal and parse-error rates,
mean latency) as the last stdout line and writes transcripts (JSONL) next to `--out`. For
`anthropic:` models it refuses to run unless `--allow-network` is given (so nothing calls out by
accident).

### 17.4 Clarifications (v0.4-F.1, normative)
1. Transcript entries carry `prompt_sha256`, `response`, `action`, `error`, `latency_s` (plus `tick`,
   `team`).
2. `opponent` uses the §15.1 spec strings `bot:<name>` / `ckpt:<path>`, or an `LLMAgent`.
3. Hand slots are numbered 0–3, as in the action encoding.
4. The clock shows the remaining time of the current phase in m:ss (e.g. `3:00 left (regulation)`).
5. The `llm_match.py` summary contains `wins`, `draws`, `losses` (counts), `mean_crowns`
   `[agent, opponent]`, `illegal_rate`, `parse_error_rate`, `mean_latency_s`.
6. `result` and `crowns` are from the agent's perspective (`crowns = [agent, opponent]`); a match cut by
   `max_ticks` has `end_reason = None` and result 0.
7. `max_history = k` includes the last k (state, response) pairs as prior turns; 0 = stateless.
8. `mock_random(seed)` picks uniformly among the legal commands listed in the text (and WAIT),
   deterministically given its seed.
9. `anthropic_model_fn` lives in `pufferroyale.llm`.
10. The `play_llm_match(decision_interval=…)` argument sets the cadence; `LLMAgent.decision_interval`
    is its default.

## 18. Second-audit amendments (v0.4.1, normative)
1. **Crown-tower ties**: when two enemy crown towers are exactly tied for acquisition, the one with
   the smaller **own-frame x** (own-left) wins; entity id only breaks ties that remain.
2. **Royal Chef serving order**: each team's Chef towers are served in **own-frame left-to-right**
   order within the tick; a troop levelled by the first tower is not eligible for the second.
3. **Dash invulnerability** is evaluated when damage is dealt: a Bandit whose dash is in its moving
   phase (or on its arrival tick before Resolve) takes no damage from any source, including a spell
   whose stun/knockback cancels the dash in the same tick.
4. **Bots are seat-symmetric**: every scan over towers, lanes or tiles in the scripted bots uses the
   acting team's own frame.
5. **Snapshot validation** also bounds every value the engine multiplies (e.g. lifetime ≤
   `lifetime_ms/50 + 1`, max_hp ≤ its unit's level-cap value) and requires pending unit spawns to
   reference a real non-tower unit and a valid card, so a validated state can never evolve into an
   invalid or overflowing one.
6. **Training precision**: our trainers must not reuse autocast weight caches across optimizer steps
   (use `cache_enabled=False`) — or refuse non-float32 precision; bf16 must either learn correctly or
   be rejected, never silently train on stale weights.
7. **League run robustness**: never snapshot/save a non-finite learner; a failed `--resume` exits
   non-zero promptly (no hanging threads); CLI optimizer hyper-parameters are re-applied on resume;
   enabling MMD at resume sets the reference to the loaded learner; runs are relocatable (pool paths
   stored relative to the run dir).
8. **Clarifications (v0.4.2):**
   (a) the per-entity elapsed-lifetime counter k satisfies `k ≤ lifetime_ms/50 + 1`;
   (b) `max_hp` ≤ its unit's level-12 value (one Chef level-up);
   (c) every pending unit spawn carries a card in `0..63` (spell-released units carry their spell's
       card) and a non-tower unit;
   (d) §10 entity dicts include `dash_state`: 0 none, 1 stand, 2 moving;
   (e) `league_state.json["args"]` and the saved optimizer reflect the CLI values of the latest
       (resumed) invocation;
   (f) `--matches 0` is an error in `best_response.py`, `tournament.py`, `llm_match.py`; the
       `total_timesteps ≥ batch_size` check applies to `train.py`, `league_train.py`,
       `best_response.py`; every JSON output (a script's last stdout line or its `--out` file) is strict
       JSON (no NaN/Infinity);
   (g) `Bot(kind, seed)` decisions depend only on `(kind, seed)` and the acting team's own-frame state.

## Changelog
- v0.4.1 (2026-09-27): §18 second-audit amendments (tower tie by own-frame x, Chef serving order, dash invulnerability at damage time, symmetric bots, stronger snapshot validation, bf16/autocast rule, league robustness).
- v0.4-F.1 (2026-09-27): §17.4 clarifications of the LLM interface (transcript/summary keys, clock, perspective).
- v0.4-F (2026-09-26): §17 LLM play interface (text render/parse, LLMAgent, mocks, optional Anthropic adapter, llm_match script).
- v0.3-E.1 (2026-09-26): §16.6 clarifications (Princess projectile, non-homing shots, Miner, spawners, death effects, Inferno stages, Earthquake/Poison/Lightning numbers, Collector timing, tower-troop scalars, Barbarian Barrel).
- v0.3-D.1 (2026-09-26): §15.7 clarifications of the league/metagame/MMD/script interfaces.
- v0.3-E (2026-09-26, planned): §16 card batch 2 (ids 21–63), tower troops, compact card-id observation encoding (CARD_SLOTS 128).
- v0.3-D (2026-09-26): §15 training & evaluation tooling (league pool + PFSP, LeagueVecEnv, MMD trainer, best-response probe, tournament + meta-game Nash, HPC templates).
- v0.2.1 (2026-09-26): §14 audit amendments (same-tick validation, melee mid-swing retarget per
  ledger, hit-list capacity, formation ejection side, restore validation, learner log keys).
- v0.2 (2026-09-26): troops may not be placed on building footprints; destroyed towers free their
  footprint; even-footprint buildings anchor on the tapped tile's own-frame top-left corner; The Log
  restricted to troop territory (own half + pockets), not water; §13 clarifications 1–15.
- v0.1 (2026-09-26): initial.
