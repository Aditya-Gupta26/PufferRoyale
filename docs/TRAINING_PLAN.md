# PufferRoyale: training plan (plan of record)

_Written 2026-10-02 from the training-strategy discussion between Aditya and Claude. This is the plan
of record for everything from "verified simulator" to "course results". `docs/HPC_HANDOFF.md` tells
the cluster agent how to execute it. `docs/SPEC.md` stays the binding contract for code: every code
change listed here is specified there first (as SPEC §19, to be written when the work package
starts) and tested independently._

**Status (2026-10-03):** the plan is agreed and the work package (§7) is **implemented**: its
contract is `docs/SPEC.md` §19 (v0.5-G … G.3), and it passed the independent spec tests, an
independent audit and the CPU learning check (§7). Flags and config keys marked **[WP]** came with
the work package; their final names are in SPEC §19 and each script's `--help`. Everything not marked [WP] exists on `main`
(commit `b6b63da`) and was checked against the code (see §14).

---

## 0. Summary

**Goal.** Train the strongest Clash Royale 1v1 policy we can in our simulator, and answer the
course's game-theory questions:
- **RQ1.** Does a population (a PFSP league of past snapshots) make self-play less exploitable?
- **RQ2.** Does the MMD regulariser reduce exploitability further?
- **RQ3.** What is the deck meta-game, i.e. the Nash mixture over decks, according to our engine?

**The plan in one table**

| Stage | What | Decks | Opponents in training | Main output |
|---|---|---|---|---|
| 0 | HPC bring-up, verification, throughput | – | – | A verified cluster setup and a standard config per GPU type |
| 1 | **Bot ladder** (calibration only) plus ablations | Hog 2.6 mirror | Scripted bots: do-nothing → random → heuristic | Proof that learning works; fixed hyperparameters; ablation results |
| ~~2~~ | ~~Self-play warm-started from stage 1~~ | – | – | **Removed:** it became run A of stage 3 |
| 3 | **Course comparison** A / B / C, **all from random init**, 3 seeds each | Hog 2.6 mirror | A: current self only. B: self + past snapshots (PFSP). C: B + MMD. **No scripted bots** | Exploitability, head-to-head, transitivity (RQ1, RQ2) |
| 4 | **General policy** with the stage-3 winner | 50% real-deck pool + 50% random decks, held-out decks | As the winning method; no anchors by default | Held-out-deck generalisation; deck meta-game Nash (RQ3) |
| 5 | Stretch | – | – | Tower troops, deck-level PSRO, LLM matches (needs approval), fidelity checks |

**Principles we agreed on**
1. **Zero-sum stays exact.** Every reward term is antisymmetric, and shaping is potential-based.
   Evaluation always uses true match outcomes, never shaped returns.
2. **Run A never sees a scripted bot.** It is pure self-play from random weights. B and C use the
   same seeds and differ from A by exactly one ingredient each.
3. **Evaluation opponents are opponents no stage-3 run trained against.** The scripted bots are
   evaluation-only from stage 3 on. The headline metric is exploitability, measured by
   best-response probes.
4. **Stage 1 is calibration only.** Its checkpoints never start a later stage.
5. **Master one deck first** (the Hog 2.6 mirror, stages 1–3), then scale to many decks (stage 4).

---

## 1. Decision log (this discussion)

| # | Decision | Why |
|---|---|---|
| T1 | Denser reward: tower-HP and crown terms plus a small wasted-elixir term, all **antisymmetric and potential-based**, fading to 0 during training (§3) | Sparse ±1 over ~360 decisions is hard credit assignment. Potential-based antisymmetric shaping keeps the game zero-sum and keeps its equilibria |
| T2 | Card-play bonus exists only as a flag, **off by default** | It rewards spending, not winning (cheap-card spam). Kept for experiments only |
| T3 | Policy head: **card first, then position given the card** (conditional head, §4.1), not independent MultiDiscrete heads | Legality depends on the card (exact masks are impossible with independent heads), and "where" depends on "what". Combining the two into the existing 2,305 joint logits leaves PufferLib and the masks unchanged |
| T4 | **Card-stat features** next to the learned card embedding (§4.2) | Lets the policy understand cards by their attributes, which is what makes random-deck training generalise |
| T5 | **Own full deck in the observation** (§4.3) | Needed for multi-deck play: "what am I piloting" |
| T6 | **Deck sampling**: pool, random-deck fraction, held-out decks (§5) | Stage 4: real-deck pool for archetype depth, random decks for card breadth, held-out decks to test generalisation |
| T7 | `placement_grid` ∈ {1, 2, 4} with **deterministic** block representatives (§4.4) | Your "mega-tile" idea as an ablation (and an optional curriculum). Random placement inside a block was rejected because it changes the game |
| T8 | Stage 1–3 deck: **Hog 2.6 mirror** (Hog Rider, Musketeer, Cannon, Ice Golem, Ice Spirit, Skeletons, Fireball, The Log) for training **and** evaluation | In a mirror the fair score is exactly 0.5, so exploitability is clean. Less noise per game; the methods differ only at higher skill. Comparable with prior work (Hasty-CR trained on Hog 2.6 only) |
| T9 | Stage 3 = nested comparison **A ⊂ B ⊂ C**: A self-play, B + PFSP snapshots, C + MMD | One change per step isolates each ingredient |
| T10 | **No anchors** (scripted bots) in any stage-3 training | Anchors would be a second difference between A and B, and would bias bot-based evaluations |
| T11 | **Run A is completely anchor-free** (your requirement), so stage 3 starts from **random initialisation**, with the same seeds for A, B and C | A warm start from stage 1 would carry bot exposure into A through its weights |
| T12 | Stage 1 = **calibration only** (pipeline check, hyperparameters, ablations); old stage 2 removed | Follows from T11 |
| T13 | Evaluation bots must be **unseen** by the evaluated policies. With T10–T11, the heuristic and random bots qualify for stage 3 | Fair, readable strength numbers |
| T14 | The proposed **lookahead (search) evaluation bot is dropped** (see §12) | Not needed: exploitability is the headline, and the unseen heuristic/random bots cover readable strength |
| T15 | Stage 4 uses the stage-3 winner, a 50% real-deck pool + 50% random decks, and held-out decks; warm-started from the best stage-3 policy | Generalisation plus the deck meta-game (RQ3) |
| T16 | GPU budget: up to **~16 GPUs concurrently** (about 16 GPU-days per day at best) | GPU-hours are not the binding constraint. CPU cores per job and queue wait are |
| T17 | Spend the headroom on **3 seeds** per stage-3 method, parallel stage-1 ablations, and several best-response probes per policy | Statistical strength for the headline results |

Items introduced in this document that we did **not** discuss explicitly are listed in §13 for
your review. Each is small and reversible.

---

## 2. Game-theoretic framing and what we measure

- **The game.** Two-player, zero-sum, partially observable stochastic game. Both players act
  simultaneously every 0.5 s (`frame_skip = 10` ticks at 20 Hz). The opponent's hand, elixir and
  unrevealed cards are hidden, and the observation contains only legal deductions about them.
- **Target.** A Nash equilibrium. Quality is measured by **exploitability**: how much a best
  response gains against the policy. In a symmetric game (the mirror) the game value is exactly 0.5
  in score terms (win 1, draw 0.5, loss 0), so a Nash policy scores ≥ 0.5 against anything.
- **Exploitability proxy.** `scripts/best_response.py` trains a fresh learner against one frozen
  policy and reports `br_score` and `exploitability_proxy = 2·br_score − 1`. This is a **lower
  bound**, only as tight as the best response's training (§9, E1).
- **Why exploitability is the headline.** Beating one fixed opponent measures one matchup; a
  policy can crush a bot and still have a hole a targeted opponent finds at once (KataGo was
  beaten this way in 2022). Exploitability measures the worst case, which is what "close to Nash"
  means.

**Pre-registered hypotheses for stage 3** (fixed before launch, §8.3):
- **H1 (RQ1):** exploitability(B) < exploitability(A) at equal learner-step budgets.
- **H2 (RQ2):** exploitability(C) < exploitability(B) at equal budgets.
- **H0 (also a valid finding):** no measurable difference, i.e. plain self-play suffices in this
  game, as it does in Tron. That outcome would say this game's strategy space is mostly
  transitive (the "spinning tops" picture: Balduzzi et al. 2019, Czarnecki et al. 2020), and the
  within-run transitivity diagnostic (§9, E4) tests that directly.

---

## 3. Reward v2 [WP]

### 3.1 What exists today (verified)
`pufferroyale/csrc/royale.h` (`c_step`, step 3) gives team 0 the per-step reward

```
r0 = reward_tower × (Σ_i enemy tower-HP fraction lost − Σ_i own tower-HP fraction lost)   (i = King, left, right)
   + reward_crown × (Δ own crowns − Δ enemy crowns)
   + result0 on the step the match ends (+1 / −1 / 0)
r1 = −r0
```

Both weights default to 0 (`royale.ini`, `Royale(...)`). The shaping is a plain difference of a
potential, without the factor γ. **PufferLib 3.0 clamps every stored reward to [−1, 1]**
(`PuffeRL.evaluate`, `pufferlib/pufferl.py` line 275 at commit `3b5c604`). So today a large
terminal-step reward is silently clipped, **even at the plan's weights**. With
`reward_tower 0.3, reward_crown 0.2`, a King-kill step (crowns jump to 3) gives about 1.4, which is
stored as 1.0 (found by the independent audit of this document on today's env).

### 3.2 Definition (v2)
For team `k` against enemy `j` in state `s`:

| Quantity | Definition | Range before the match ends |
|---|---|---|
| `T_k(s)` | Σ of `k`'s three crown-tower HP fractions (King, left, right; a destroyed tower = 0) | [0, 3] |
| `C_k(s)` | `k`'s crowns | 0–2 (3 crowns ends the match) |
| `L_k(s)` | `k`'s cumulative leaked elixir in elixir (engine `leaked` / 2800) | 0 to ~189 in a full 5-min match (income is 192.9 elixir over 6,000 ticks) |
| `P_k(s)` | `k`'s cumulative number of card plays (only for the off-by-default play bonus) | 0 to ~200 (bounded by total elixir) |

Potential (antisymmetric by construction: `Φ_j = −Φ_k`):

```
Φ_k(s) =  w_tower · (T_k − T_j)
        + w_crown · (C_k − C_j)
        − w_elixir · clip(L_k − L_j, −L_cap, +L_cap)        L_cap = 20 elixir
        + w_play  · clip(P_k − P_j, −P_cap, +P_cap)        P_cap = 20 plays, w_play = 0 by default
```

Per-step reward for team `k` on the step `s_t → s_{t+1}`:

```
r_k,t = R_k · 1[match ends during this step] + F_k,t
F_k,t = γ · Φ̃_k,t+1 − Φ̃_k,t
Φ̃_k,t+1 = Φ_k(s_{t+1}; w_t)   if the match continues,  and 0 if it ended during this step
Φ̃_k,t   = the value stored as Φ̃_k,t+1 on the previous step  (0 on the first step of a match)
```

- **Φ = 0 at the terminal state.** Required for exact policy invariance in episodic tasks (Ng,
  Harada & Russell 1999; Grześ 2017). It also means the match's last step "refunds" the
  accumulated shaping, so for every match `Σ_t γ^t F_k,t = γ^T·0 − Φ̃_k,0 = 0`. **The discounted
  shaped return equals the discounted true return, match by match.** This is the key invariant the
  tester checks.
- **Stored previous potential.** This is *dynamic* potential-based shaping (Devlin & Kudenko 2012),
  so the telescoping stays exact even while the weights `w_t` anneal between steps.
- **γ in the reward must equal the trainer's γ** (`train.gamma`, 0.999 by default). The scripts pass
  it to the env as `reward_gamma` [WP]; it is not a separate CLI/ini setting, so it cannot disagree.
- **Zero-sum per step:** `r_0,t + r_1,t = 0` exactly, for every step, in self-play.
- **Why it still helps learning.** With GAE (γλ = 0.999 × 0.95 ≈ 0.949) the end-of-match refund,
  ~300 steps after an early action, contributes ~0.949³⁰⁰ ≈ 10⁻⁷ to that action's advantage. The
  shaping acts as an immediate, informative signal, while the theory guarantees the optimal policies
  (and the game's equilibria) are unchanged. Equivalently, potential-based shaping is a value-function
  initialisation (Wiewiora 2003).

**Starting weights** (from the discussion): `w_tower = 0.3`, `w_crown = 0.2`, `w_elixir = 0.01`
(per elixir), `w_play = 0`. Scale check: taking a Princess tower is 0.3·1 + 0.2·1 = 0.5 of potential,
while wasting a full bar (10 elixir) is −0.1. A decisive win (+1) still outweighs all shaping.

### 3.3 Annealing [WP]
- `w_t = w_0 · max(0, 1 − n / N_anneal)`, where `n` = learner steps so far and
  `N_anneal = shaping_anneal_frac × total_timesteps`. `shaping_anneal_frac = 0` means constant
  weights.
- Stage 1 uses constant weights, so the shaping on/off ablation is clean. Stages 3–4 anneal to 0 by
  50% of the budget (if stage 1 shows shaping helps).
- **Mechanism (recommended; SPEC §19 decides):** each C env counts its own steps and derives `n`
  from them; the script passes `N_anneal` converted to per-env steps, plus an offset on `--resume`.
  This works unchanged for in-process vectorisation, the `Multiprocessing` backend and
  `LeagueVecEnv`, with no cross-process calls. The alternative is a setter called by the trainer at
  every epoch; it must then also reach `Multiprocessing` workers, or that backend must refuse to
  start with annealing on.

### 3.4 The [−1, 1] clamp [WP]
- **The problem.** A terminal step's reward is `R_k − Φ̃_k,t + …`, which can exceed 1 in magnitude.
  For example, a player who wins while behind on towers gets `1 + |Φ̃|`. With the weights above,
  `|Φ| < 0.9 + 0.4 + 0.2 = 1.5` before the match ends, so `|r| < 2.5` on a terminal step and
  typically far less on other steps.
- **The fix.** `MMDPuffeRL` gets a `reward_clip` config key: 1.0 by default, which is exactly
  today's behaviour; 0 means no clamp. It works by overriding `evaluate()` with a
  statement-for-statement copy of PuffeRL's, the same technique `trainer.py` already uses for
  `train()`.
- **Settings.** v2-shaped runs use `--train.reward-clip 0`.
- **Zero-sum is safe either way.** Clamping is an odd function, so it never breaks zero-sum, but it
  does break the potential-based property on clipped steps.

### 3.5 Evaluation is independent of shaping
- Every score we report (`learner_score`, `eval.py`, `tournament.py`, `best_response.py`,
  `deck_metagame`) comes from the **match outcome**, never from summed rewards.
- **True today for the league:** it uses `win_0` / `win_1` from the env log.
- **True today for `eval.py` only by accident:** it reads win/draw/loss from the **sign of the
  terminal-step reward** (`scripts/eval.py`, `play()`). That is correct only because it builds
  unshaped envs.
- **The WP must make it true everywhere:**
  - the `Royale` constructor defaults stay unshaped;
  - `eval.py` is converted to read the match outcome;
  - every new tool derives results from `result` / the log, never from reward signs (a shaped
    terminal reward can have either sign).

---

## 4. Policy and observation changes [WP]

### 4.1 Conditional head: card first, then position given the card
Today `pufferroyale/torch.py` uses one `Linear(hidden, 2305)` actor over the joint action
(no-op + 4 hand slots × 576 own-frame tiles; `a = 1 + slot·576 + ty·18 + tx`).

**New head (`policy.head = conditional` [WP]; `flat` keeps today's head for the ablation):**
1. **Card head:** 5 logits (wait + 4 hand slots), each from the trunk and that slot's card
   encoding (§4.2). Slot `s` is masked iff the 2305-mask has no legal tile for it. Wait is always
   legal.
2. **Position head:** for each slot `s`, a 32×18 map of logits, the heat map. It comes from a
   convolutional decoder over the board features conditioned on slot `s`'s card encoding (e.g. FiLM
   or concatenation), and is masked with that slot's exact 576-tile mask segment. Flattening the map
   row-major must give index `ty·18 + tx`, the same own frame and order as the action encoding.
3. **Joint logits:** `logit[0] = log P(wait)`, `logit[1 + s·576 + t] = log P(s) + log P(t | s)`.
   Re-apply the existing `mask_logits` (illegal = `finfo.min`) so illegal entries are exactly as
   today.

**Requirements**
- **Nothing outside the policy changes.** The output is still one `(B, 2305)` tensor, so PuffeRL's
  `sample_logits`, PPO, `masked_kl` (MMD) and `select_actions` (league opponents) work as they are.
- **Entropy:** `H_joint = H_card + Σ_s P(s)·H_pos|s` (an identity).
  - Log `entropy_card` and `entropy_pos` separately.
  - Add `ent_coef_card` / `ent_coef_pos` [WP]. When set, the entropy loss is
    `−(c_card·H_card + c_pos·E_s[H_pos|s])` in place of `−ent_coef·H_joint`.
  - Setting both to `ent_coef` must reproduce today's loss.
  - Reason: H_pos is up to log 576 ≈ 6.4 nats vs log 5 ≈ 1.6 for cards, so one shared coefficient
    over-weights placement exploration. Prior Clash Royale projects (ClashAI, ClashRoyaleEnv)
    reported exactly this imbalance.
- **Both policy forms:** must work for `Policy` and `Recurrent`. PufferLib's `LSTMWrapper` passes
  only the LSTM output to `decode_actions`, so a spatial map cannot ride through it. `Recurrent`
  already overrides `forward`/`forward_eval` and must carry the board features to the decoder (or
  the decoder must work from the hidden vector). Tests cover both.
- **Numerics:** no NaN/inf in logits, log-probs, entropy or KL, including rows where only wait is
  legal.
- **Initial behaviour differs.** The current flat head starts near-uniform over all legal joint
  actions, so P(wait) ≈ 1/(number of legal actions), which is tiny. The conditional head starts
  near-uniform over cards, so P(wait) ≈ 1/(1 + affordable slots). That is a plausible advantage
  (prior work saw wait-heavy optimal play), and it's also why the flat head stays available.

### 4.2 Card-stat features
- **Fixed table.** A fixed `(CARD_SLOTS + 1) × K` table (row `card_id + 1`; row 0 = empty and unused
  ids are zeros) built once from `pufferroyale.binding.card_info(card_id)` for the 64 cards. These
  are the engine's own level-11 values, so they are consistent with the simulation by construction.
- **Features, each normalised to [0, 1]** by the maximum over the 64 cards (log scale for HP and
  damage):
  - elixir; kind (troop / building / spell);
  - count (plus second-summon count);
  - hitpoints, damage per hit, hit speed, DPS;
  - range, sight, speed;
  - flying, attacks air, attacks ground, buildings-only targeter;
  - splash radius (melee `area_damage_radius` or projectile radius);
  - spell damage and radius, crown-tower damage %;
  - building lifetime, death damage, deploy time, jumps the river, charge.
- **Missing keys.** `card_info` returns different key sets for troops and spells (checked for Hog
  Rider and Fireball), so a missing key = 0.
- **Encoding.** `enc(c) = [Embedding(c), Linear(stats[c])]`, used everywhere a card id appears:
  entity rows, hand, next card, opponent's last 4, own deck.
- **Ablation.** `policy.card_stats = false` [WP] turns the stats off. This ablation is meaningful
  only with many decks (stage 4); in a mirror only 8 cards appear.

### 4.3 Own deck in the observation
- **New scalar field `own_deck`:** 8 card ids (`card_id + 1`), in ascending id order, i.e. the deck
  as a **set**. `OBS_SIZE` grows from 17,707 to 17,715, and the field is added to
  `CARD_ID_SCALARS` so the policy embeds it.
- **Not the queue order.** At deal time the order of the queue behind the next card is the hidden
  shuffle; a real player knows only the set. Exposing the order would leak information a human
  doesn't have.
- **Tests.** The no-leak and mirror-symmetry tests must cover the new field.

### 4.4 `placement_grid` ∈ {1, 2, 4} (ablation; default 1)
- **Blocks.** The board's 18 × 32 tiles group into `g×g` blocks:

  | `placement_grid` | Blocks | Actions |
  |---|---|---|
  | 1 | 576 | 2,305 |
  | 2 | 9 × 16 = 144 | 577 |
  | 4 | 5 × 8 = 40 (the last column is 2 tiles wide) | 161 |

- **Representative tile.** A coarse action (slot, block) maps to one fine tile: the legal tile in
  the block nearest the block's geometric centre, with ties broken by lower `ty`, then lower `tx`.
  It's computed from the exact fine mask at decision time.
- **Coarse mask.** (slot, block) is legal iff any tile in the block is legal for that slot. Every
  legal coarse action therefore maps to a legal fine play: no random placement and no new illegal
  plays.
- **Observation.** Unchanged: it keeps the exact fine 2305-mask, and the coarse mask is derived from
  it (block legal iff any tile legal; SPEC §19.4). Checkpoints are tied to their `g` through the
  size of their action head.
- **Policy.** The conditional head pools its 32×18 heat map to the grid.
- **Optional curriculum** (not required in the WP): train at `g = 4`, then initialise a `g = 1`
  policy from it.

---

## 5. Deck sampling [WP]
- **New env keys** (forwarded unchanged by `LeagueVecEnv`):
  - `deck_pool`: a list of decks (preset names or 8-card lists) with optional weights;
  - `random_deck_frac`;
  - `heldout_decks`;
  - `deck_draw` = `independent` | `mirror`.
- **Fixed decks still work.** `deck0` / `deck1` keep today's meaning (fixed decks; `random`), for
  stages 1–3.
- **Per match and seat:**
  - with probability `random_deck_frac`, a uniformly random 8-card deck from all 64 cards,
    **redrawn if it equals a held-out deck** (as a set);
  - otherwise a pool deck by weight;
  - with `mirror`, seat 1 gets seat 0's deck.
- **Separate random stream.** Deck draws use their own seeded stream: deterministic, and
  independent of the game and bot streams.
- **Held-out decks are never trained on.** Construction fails if the pool intersects the held-out
  set.
- **Logging:** per-deck episode counts, for the sampling-statistics check and for the report.
- **Evaluation tools** gain deck-set arguments: `eval.py --decks …`, a held-out-set option for
  `eval.py`/`tournament.py`, and `deck_metagame` over any list (it already takes one).
- **Pools for stage 4** (§8.4):
  - **pool:** `hog26`, `giant`, `bait`, `golem`, `lavaloon`, `miner_poison`, `pekka_bridge`;
  - **held out:** `xbow` (X-Bow siege) and `royal_hogs` (split-lane Royal Hogs + Earthquake), plus 50
    seeded random decks. Their win conditions (X-Bow; Royal Hogs) and Earthquake/Fire Spirit appear
    in no pool deck: the policy meets those cards only in random decks. Bridge spam is not held out,
    since `pekka_bridge` is in the pool;
  - **optional:** real decks filtered from the 1.45M-battle HuggingFace dataset (MIT) to our 64
    cards. This needs a network download, so ask first.

---

## 6. Training operations [WP]
1. **Logging.**
   - wandb logging on `train.py` and `league_train.py`, with offline mode (`WANDB_MODE=offline`,
     then `wandb sync`). PufferLib 3.0 ships `pufferlib.pufferl.WandbLogger`. The key comes from the
     environment, never the repo.
   - **Name clash to resolve in SPEC §19:** PufferLib's config parser already accepts `--wandb`,
     `--wandb-project` and `--wandb-group` through both scripts today, and our scripts ignore them:
     no logging happens. Either wire those flags up or reject them; they must not stay silently
     inert.
   - Logged: losses (including `entropy_card`/`entropy_pos`, `mmd_kl`), SPS, the learner's score
     per opponent, the shaping weights, the illegal-action rate, episode length, and **per-card
     play counts plus "played when in hand" rates**. The last one needs new per-card fields in the
     env `Log`; it is the detector for "never casts Fireball" and reward hacking.
   - `scripts/plot_history.py` renders these from `history.jsonl` to PNG.
2. **`train.py --resume`:** weights, optimizer, counters and RNG states, as `league_train.py`
   already does.
3. **`league_train.py --init-from CKPT`:** start a **new** run from given weights only (new pool,
   new optimizer). Needed to chain the stage-1 ladder (pool settings cannot change on `--resume`)
   and for the stage-4 warm start.
4. **Stage runner `scripts/stages.py`:** runs the bot ladder as chained `league_train.py` runs (one
   anchor per rung, `--init-from` the previous rung), checks each gate on `history.jsonl`, and
   stops at the first failure with a report.
5. **Transitivity diagnostic:** for a run's snapshots ordered by epoch, report the fraction of
   (later, earlier) pairs where the later wins (> 0.5), and the number of cyclic triads (i > j > k
   > i, each by a margin). Either in `tournament.py` JSON or as `scripts/transitivity.py`.
6. **`eval.py` snapshot loading fix.**
   - The bug, **checked on 2026-10-02:** `eval.py --checkpoint <run>/snap_N.pt` fails for recurrent
     league runs. Its `load_policy` looks for `snap_N/config.json`, finds none, and builds a
     feed-forward `Policy`.
   - Today's workarounds: pass the run directory (it uses `model_*.pt` + `config.json`), or use
     `tournament.py`, which infers the architecture from the state dict.
   - The fix: use `pufferroyale.league.load_policy`. Also add Wilson 95% CIs to the JSON.
7. **`best_response.py --init-from-target`:** the best response starts from the target's own
   weights instead of random ones (similar in spirit to AlphaStar's exploiters, which start from trained, supervised weights
   rather than from scratch). It
   starts at ~0.5 and only has to find the holes, which gives a much tighter lower bound per GPU-hour
   (§9, E1).

---

## 7. The work package (Track 1, on the Mac)

**Contents:** §3 (reward v2 + `reward_clip`), §4.1–4.4 (head, card stats, own deck,
`placement_grid`), §5 (deck sampling), §6 (operations). All of these change what the agent learns or
how it's measured, so they land **together, before any real training**. Old checkpoints become
incompatible; none matter.

**Process** (as for the simulator):
1. The orchestrator writes SPEC §19 (requirements and acceptance criteria).
2. A builder agent implements it.
3. An independent tester agent writes black-box tests in `tests/spec/` from SPEC §19 alone.
4. Disputes go through the orchestrator.
5. An independent audit follows.
6. **Done** means a clean rebuild, the full `pytest`, `make test-c`, `make asan` and
   `scripts/e2e_check.py` are all green, plus the CPU learning check below.

**Acceptance (the tester's suite must cover at least):**
- **Reward:**
  - per-step zero-sum in self-play, before and after the clamp;
  - **for every finished match, `Σ_t γ^t F_t = 0`** within float tolerance, with constant and with
    annealing weights;
  - Φ = 0 at terminal;
  - all-zero weights reproduce today's rewards bit for bit;
  - the scripts pass `reward_gamma = train.gamma` to the env;
  - the annealing schedule's values;
  - `reward_clip = 1` reproduces PuffeRL bit for bit, and `0` stores env rewards unchanged;
  - evaluation results independent of shaping.
- **Head:**
  - `P(s, t) = P(s)·P(t | s)` within 1e-6, with probabilities summing to 1;
  - illegal actions have probability exactly 0 and legal ones > 0;
  - a slot with no legal tile gets probability 0;
  - a wait-only row has finite entropy;
  - the entropy identity holds;
  - equal split coefficients reproduce today's loss;
  - the heat-map index ↔ action tile mapping;
  - both `Policy` and `Recurrent` pass;
  - MMD KL stays finite.
- **Card stats:** the table equals `card_info` values for all 64 cards; it is finite and in [0, 1].
- **Own deck:** correct ids and order; the observation leaks nothing new; mirror symmetry still
  holds.
- **`placement_grid`:**
  - action counts 2,305 / 577 / 161;
  - every legal coarse action maps to a legal fine play;
  - the coarse mask is exact;
  - the mapping is deterministic.
- **Decks:**
  - sampling frequencies within CIs over ≥ 10⁵ draws;
  - held-out decks never drawn (including random draws);
  - determinism under seeds;
  - mirror mode.
- **Operations:**
  - resume continues correctly;
  - `--init-from` loads weights only;
  - the stage runner stops on a failed gate;
  - the `eval.py` fix (a recurrent snapshot loads);
  - `--init-from-target` starts at the target's weights;
  - wandb offline works without network.

**CPU learning check (acceptance).** Learn to beat the **random** bot on the Mac, with and without
shaping (`league_train.py --anchors bot:random --self-play-frac 0 --anchor-frac 1.0`):
- **Baseline:** measure the untrained policy's score first.
- **Pass:** a clear rise above the baseline within ~5M learner steps, e.g. ≥ 0.5. If that isn't
  reached, it isn't automatically a failure, but it must be explained before any GPU stage.
- **Why not the do-nothing bot (measured 2026-10-02):** an *untrained* policy with today's flat head
  already scores **1.00 vs `bot:noop`** (200/200 matches, both seats). It scores 0.15 vs
  `bot:random` and 0.00 vs `bot:heuristic` (`scripts/eval.py`, 100 matches per seat), so beating the
  do-nothing bot proves nothing about learning. The new head's untrained baseline must be
  re-measured: it waits more often at initialisation.

This catches "learns nothing" bugs before any GPU time is spent.

**Result (2026-10-03, Mac CPU, conditional head, Hog 2.6 mirror, 32 matches, batch 2048, lr 3e-4):**
- **Untrained baseline:** 1.00 vs `noop`, 0.43 vs `random`, 0.00 vs `heuristic` (50 matches per
  seat).
- **Learner's score vs `bot:random`** (training matches, per 100k-step window):

  | Learner steps | No shaping | Shaping (0.3 / 0.2 / 0.01, `reward_clip 0`) |
  |---|---|---|
  | 0–0.1M | 0.24 | 0.29 |
  | 0.2M | 0.24 | **0.76** |
  | 0.3M | 0.38 | **0.89** |
  | 0.4M | **0.84** | **0.95** |
  | 0.5M | 0.83 | 0.93 |
  | 0.6M | 0.94 | 0.89 |

  Each window covers about 180–280 matches (about 66 in the last, partial one).
- **Independent evaluation** of the saved learners (`eval.py`, 50 matches per seat, sampled):
  - vs `random`: 0.94 (no shaping) and 0.98 (shaping);
  - vs `heuristic`: 0.01 for both (expected this early; G3 is budgeted at 100–300M steps).
- **Verdict:** pass. Both learn; shaping roughly halves the steps to the random-bot gate.

**Not in the WP:** imitation learning (BC) from the heuristic bot. It's a fallback for stage 1
only (§11).

---

## 8. Training stages

Common to all stages:
- tower troop: Tower Princess both sides;
- `frame_skip 10`, `deploy_lockout_ticks 90`, `tiebreak absolute` (the defaults);
- float32 (bf16 only after it's validated on GPU);
- **every run records** its git commit, the engine golden-hash check result, the full command,
  `config.json`, the SLURM job id, GPU type and GPU-hours, in `docs/EXPERIMENTS.md`.
- A "learner step" is one learner decision (`global_step`). In the league that's one per match per
  env step.

### 8.0 Stage 0: bring-up (HPC; current `main`, no WP needed)
- **Work:** container/env, all tests, golden hashes and throughput (HPC_HANDOFF phases 1–3).
- **Early learning smoke (current code):** a run against `bot:random` (untrained baseline ≈ 0.15) to
  see learning on GPU before the WP lands. Not `bot:noop`: an untrained policy already beats it.
- **Gate G0:** everything green, and a standard configuration per GPU type is chosen.

### 8.1 Stage 1: bot ladder (calibration only)
- **Decks:** Hog 2.6 mirror.
- **How it trains:** ordinary single-agent PPO against a fixed bot. The bot is part of the
  environment, and only the learner's experience is trained on. Each rung is a `league_train.py`
  run with one bot anchor:
  `--anchors bot:<kind> --self-play-frac 0 --anchor-frac 1.0 --snapshot-interval I` (checked: pool
  weights are then the anchor only).
  - A snapshot interval > 0 is required, because the league only saves (and so can only resume) at
    snapshot epochs.
  - The rungs are chained with `--init-from` [WP] via `scripts/stages.py` [WP].
- **Gates:** each is measured on the last ≥ 2,000 matches against that rung's bot, and confirmed by
  `eval.py` with 500 matches per seat, sampled actions.

  | Gate | Opponent | Untrained baseline (flat head, measured) | Pass | Budget guide |
  |---|---|---|---|---|
  | G1 | `bot:noop` | 1.00 | score ≥ 0.95 | ≤ 5M learner steps. **Sanity only:** catches a broken head (e.g. one that always waits), proves nothing about learning |
  | G2 | `bot:random` (plays with probability 0.2 per decision) | 0.15 | score ≥ 0.90 | ≤ 50M. **The first real learning gate** |
  | G3 | `bot:heuristic` | 0.00 | clearly > 0.5 and rising | 100–300M |

  The baselines above are the v0.4 flat head (`scripts/eval.py`, 100 matches per seat, sampled,
  2026-10-02). The conditional head (the default now) starts at **1.00 / 0.43 / 0.00** (50 matches
  per seat, 2026-10-03): it waits more at initialisation, so it is already closer to the random bot.

- **Main config (2 seeds):** conditional head [WP], feed-forward, shaping on with constant weights
  (`reward_tower 0.3`, `reward_crown 0.2`, `reward_elixir 0.01` [WP], `reward_clip 0` [WP]), grid 1, and the
  `royale.ini` hyperparameters (lr 3e-4, γ 0.999, λ 0.95, clip 0.2, vf_coef 2.0, ent_coef 0.01,
  bptt 64, 1 update epoch).
- **Ablations** (each a full ladder from scratch, 2 seeds, run in parallel; compare steps-to-G2
  and the heuristic score at 100M / 200M / 300M steps):

  | Id | Change vs main | Question |
  |---|---|---|
  | S1-noshape | shaping off | Does the denser reward speed up learning? |
  | S1-grid2, S1-grid4 | `placement_grid` 2 / 4 [WP] | How much does placement precision matter? (your mega-tile hypothesis) |
  | S1-lstm | `--rnn` | Is memory needed beyond the deduction features? |
  | S1-flat | `policy.head flat` [WP] | Does the conditional head help? |
  | S1-lr | lr 1e-4 and 1e-3 | Learning-rate sensitivity |
  | S1-ent | `ent_coef_card` / `ent_coef_pos` split [WP] (e.g. 0.01 / 0.002) | Is exploration balanced? |

- **Decision rule.** Pick the configuration with the best heuristic score at the fixed budget;
  break ties toward the simpler option (feed-forward, grid 1). Record the choice and freeze it for
  stages 3–4.
- **A small leak to state in the report.** Hyperparameters are tuned partly by score vs the
  heuristic bot, so for stage 3 that bot is unseen by the policies' weights but not perfectly unseen
  by the pipeline. The headline therefore rests on exploitability and head-to-head results.
- **Stage-1 checkpoints are never used to start stage 3.**

### 8.2 Stage 2: removed
It was "self-play warm-started from stage 1". Under T11 it is exactly run A of stage 3, so it no
longer exists separately.

### 8.3 Stage 3: the course comparison (RQ1, RQ2)
**Setup.**
- Hog 2.6 mirror; **random init**; seeds 0, 1, 2. The same seed gives identical initial weights for
  A, B and C (checked, §14).
- Settings frozen from stage 1.
- Shaping annealed to 0 over the first 50% of the budget [WP] (if stage 1 kept shaping).
- **No scripted bots anywhere in training.**

| Run | `league_train.py` pool flags | MMD | Isolates |
|---|---|---|---|
| A | `--anchors none --self-play-frac 1.0 --anchor-frac 0` | – | Baseline: pure self-play |
| B | `--anchors none --self-play-frac 0.35 --anchor-frac 0 --pfsp hard` | – | A → B: value of a population |
| C | B + `--mmd-coef 0.05 --mmd-ref-interval 50` | ✓ | B → C: value of the regulariser |

- **A's pool.** A also writes snapshots at the same interval (for evaluation), but with
  `self_play_frac 1.0` and no anchors its sampling distribution stays `{self: 1.0}` (checked).
- **B's pool.**
  - **Mix:** 35% self, 65% snapshots by PFSP "hard" weight `(1 − p)²` floored at 0.05. Before the
    first snapshot, all of it is self.
  - **Snapshot pool size:** choose `--snapshot-interval` and `--max-snapshots` so the pool spans at
    least the last 50% of training: about 128 snapshots over the run, with the newest 64 kept.
    Eviction is oldest-first. If Phase-3 throughput with 64 opponents is poor, use 32 and an
    interval of E/64, where E = number of epochs.
- **C's MMD.** `loss = PPO + 0.05·KL(π_θ ‖ π_ref)`; the magnet `π_ref` is refreshed to the current
  policy every 50 epochs. Supplementary sensitivity runs (one seed each): `--mmd-coef 0.01` and
  `0.1`.
- **Budget.** Equal **learner steps** for every run. Default pre-registration: 1B steps per run,
  with intermediate snapshots at 25/50/75/100% used for analysis. Reduce to 500M if Phase-3
  throughput makes 1B impractical, and decide **before** launch. Wall-clock and GPU-hours are
  reported too: league runs with many opponents step slower.
- **Pre-registration.** Before launch, write into `docs/EXPERIMENTS.md`:
  - the budget; seeds; snapshot interval and capacity;
  - the stage-1 settings;
  - the BR protocol (budget, init, matches, seeds);
  - the evaluation match counts;
  - H1/H2.

  Nothing is changed after seeing results without logging it as a deviation.
- **Engine freeze.** From stage-3 launch, no gameplay-affecting change. A necessary fix starts a new
  lineage and invalidates the comparison runs started before it.
- **Evaluation:** §9 (E1–E5).

### 8.4 Stage 4: general policy (RQ3)
- **Method.** The stage-3 winner, by E1 (exploitability) first and E2 (head-to-head) second.
- **Starting point.** Warm start from that method's best final policy (lowest exploitability). If
  the transfer clearly hurts, train from scratch with the same method.
- **Decks (§5) [WP].**
  - `deck_pool` = the 7 pool presets;
  - `random_deck_frac 0.5`;
  - `heldout_decks` = `xbow`, `royal_hogs` + 50 seeded random decks;
  - seats draw independently.
- **Opponents.** As the winning method, now with sampled decks. Early snapshots (Hog-only skill)
  are easy sparring partners that newer snapshots soon outnumber.
- **No anchors by default**, which keeps the bots unseen for evaluation. If forgetting of simple
  play shows up, a small anchor share may be added, and the bot numbers are then reported as "seen".
- **Card stats on.** One short ablation without them (S4-nostats) is optional.
- **Budget:** 1–2B learner steps, 2 seeds if affordable.
- **Evaluation.**
  1. **Generalisation:** score vs the heuristic and random bots on pool decks vs held-out decks,
     both seats, ≥ 500 matches per seat and deck set. Close numbers mean the policy learned cards,
     not decks.
  2. **Deck meta-game:**
     - `pufferroyale.metagame.deck_metagame(agent, decks, matches, greedy=False)` over the 9
       presets (≥ 200 matches per pair, half per seat), which gives the deck-vs-deck score matrix
       and its Nash mixture: "the meta according to our engine";
     - compare it with real-game deck usage.
  3. **Optional:**
     - a best response on a held-out-deck mirror;
     - the stage-3 policies played with other decks (they should be much weaker, which motivates
       stage 4).

### 8.5 Stage 5: stretch (as time allows)
- **Tower-troop variety:** Cannoneer, Dagger Duchess, Royal Chef.
- **PSRO over decks:**
  1. Take the current deck-Nash mixture.
  2. Search for a deck that beats it (one-card swaps, scored with the trained policy as pilot).
  3. Add the deck, optionally fine-tune briefly, then re-solve.
  4. Repeat.
- **LLM-vs-bot / LLM-vs-policy matches** (`scripts/llm_match.py`): **only with your approval**,
  since it needs network and API use.
- **Fidelity checks** against recorded real matches (start elixir, lockout, tiebreak, speeds).

---

## 9. Evaluation protocol (shared by stages 3–4)

**General rules**
- **Sampled actions, both seats.** Policies are stochastic, and Nash policies in a hidden-information
  game may need to be mixed, so greedy play would evaluate a different policy. Every evaluation
  therefore samples actions:
  - `tournament.py` (samples by default since SPEC §19.11; `--sample` is a no-op);
  - `eval.py` without `--greedy`;
  - `best_response.py` (samples by default);
  - `deck_metagame(..., greedy=False)`.

  Both seats are used, half the matches each.
- **Confidence intervals.** Report 95% CIs for every score: Wilson intervals for a single
  proportion, and a bootstrap over seeds and matches for method-level numbers. With 3 seeds, report
  per-seed values and effect sizes. Do not claim significance the data can't support.
- **Training settings.** Evaluation uses each checkpoint's own training settings, which the tools
  read from `config.json`. Settings must be identical across A, B and C.

**E1. Exploitability (headline).** For each of the 9 final stage-3 policies:
- **Probe runs.** `best_response.py --target <final> --init-from-target` [WP], with 2 BR seeds.
- **Evaluation matches.** `--matches 1000` (500 per seat), giving a 95% CI half-width of about
  ±0.03 on `br_score`.
- **BR budget calibration (before E1).** Run the BR against A seed 0's final policy at 50M, 150M
  and 400M steps. Choose the smallest budget where `br_score` gains < 0.02 at the next level, then
  apply that budget to all targets.
- **Scratch-start BR.** Report it too, on a subset, as a robustness check.
- **Output:** `br_score` and `exploitability_proxy` per target, aggregated per method.

**E2. Head-to-head round robin.**
- **Agents:** the final policies of all 9 runs, plus the 50% checkpoints.
- **Command:** `tournament.py --agents … --decks hog26 --deck-mode mirror --matches 200 --sample`.
- **Output:** the payoff matrix, Elo, and the Nash mixture over policies. Aggregate by method; if
  A beats B beats C beats A, report that cycle.

**E3. Unseen scripted bots.**
- **Command:** `eval.py --checkpoint <run dir> --bots noop random heuristic --seats 0 1 --episodes 500`
  (per bot and seat), with sampled actions.
- **Snapshots:** pass the run directory, or `tournament.py`, until the WP `eval.py` fix lands.
- **Purpose:** readable strength and a sanity floor.

**E4. Within-run transitivity.**
- **What:** for each run, 10 evenly spaced snapshots in a `tournament.py` round robin
  (`--matches 100 --sample`).
- **Output:** the transitivity report (§6.5) [WP]. Steady improvement means a transitive ladder; cycles
  are the signature that B and C should fix.

**E5. Qualitative.**
- Watch games (`scripts/watch.py --checkpoint <run dir>`, text board on the cluster).
- Note strategies, per-card play rates and degenerate behaviour (e.g. never casting Fireball,
  always waiting).

---

## 10. Compute and schedule

**Unknown until Phase 3:** learner steps per second per job (`S`). The engine runs about 500k
ticks/s per core and the env about 26k steps/s per core (STATUS.md). The league steps all matches
in one process, so its throughput is CPU-bound and falls with more distinct opponents (one GPU
forward per opponent group per step). GPU-hours = steps / S / 3600.

| Stage | Runs | Learner steps (default) | GPU-h at S = 15k (illustrative) |
|---|---|---|---|
| 0 | bring-up + smoke | < 0.2B | < 24 |
| 1 | main + 8 ablation configs (S1-lr counts twice) × 2 seeds = 18 runs | ladder 5M + 50M + 300M = 355M each ≈ 6.4B | ≈ 120 |
| 3 | 9 runs + 2 MMD-sensitivity runs | 1B each = 11B | ≈ 205 |
| 3 (E1) | 18 BR probes (+ calibration) | ~0.25B each ≈ 4.5B | ≈ 85 |
| 4 | 1–2 runs + evaluation | 1.5B each ≈ 3B | ≈ 55 |
| **Total** | | ≈ 25B | **≈ 490 GPU-h ≈ 20 GPU-days** |

- **Wall clock, not GPU-days, is what binds.** With up to 16 concurrent GPUs, the total is spread
  over about 4 weeks of stages. One 1B-step run at 15k SPS is ~18.5 h. All stage runs go through
  `hpc/league.sbatch`, which is requeue/resume-safe (`train.sbatch` is not).
- **CPU-bound jobs.** Several league runs can share one GPU when the env is the bottleneck; Phase 3
  measures that.

**Timeline (today is 2026-10-02; course deadline early December)**

| Dates | Track 1 (Mac) | Track 2 (HPC) |
|---|---|---|
| Oct 2 – 9 | Work package: SPEC §19, build, independent tests, audit, CPU learning check | Phases 1–3: environment, G0, throughput, early learning smoke on current `main` |
| Oct 9 – 10 | Support | Pull the WP; re-run G0; re-measure throughput with the new head |
| Oct 10 – 16 | Support and fixes | **Stage 1** (ladder + ablations in parallel); freeze settings |
| Oct 16 – 17 | – | Pre-registration; engine freeze; **stage 3 launch** |
| Oct 17 – Nov 2 | Support | Stage 3 training (wall ~1–3 days), BR calibration, **E1–E5** |
| Nov 2 – 16 | Support | **Stage 4** training and evaluation, deck meta-game |
| Nov 16 – Dec 1 | Plots, report | Stretch (stage 5), final evaluations |
| Dec 1 – 7 | Buffer | Buffer |

---

## 11. Risks and fallbacks

| Risk | Signal | Response |
|---|---|---|
| Stage 1 stalls at G2 (or G1 fails, which points to a broken head) | score flat vs `random` | Debug before scaling: watch games, check the masks, illegal-action rate and per-card rates; try flat vs conditional head, shaping, lr, batch. **BC from the heuristic bot is a fallback for stage 1 only** |
| Stage 3 self-play from scratch learns too slowly | A's snapshots don't improve (E4), scores vs bots stay low | More budget, keep shaping longer, bigger batch. **Any bot-derived warm start would break T11: ask Aditya first** |
| Conditional head underperforms | S1-flat beats main | Use the flat head (that's why it's kept) |
| Reward hacking | per-card play spikes, chip damage without wins | The shaping is potential-based and anneals to 0; adjust weights only in stage 1 |
| League much slower than A | SPS drops with pool size | Compare at equal **steps** (pre-registered), report wall time; reduce `max_snapshots`; optional HPC_HANDOFF O1 (multi-process league stepping) |
| BR too weak (all ≈ 0.5) | calibration curve still rising | Raise the BR budget; `--init-from-target`; state that the proxy is a lower bound |
| Non-finite training | `nonfinite.json`, exit 1 | Resume from the last good save with a lower lr; report |
| Cross-platform determinism fails | `golden_hashes.py` FAIL | Stop; report the first diverging match and tick (never regenerate the reference) |
| Preemption / wall-time limit | job ends early | `league.sbatch` is `--requeue`-safe and resumes from the last snapshot epoch |
| Schedule slips | stage 3 late | Cut stage 4 to one seed / 1B steps; stage 5 is optional |

---

## 12. Considered and dropped

| Idea | Why it was dropped |
|---|---|
| **Lookahead (search) evaluation bot.** Not MCTS: a flat one-step lookahead that forks the engine (`memcpy` snapshot), simulates each candidate play ~5 s with simple bots for both sides, and picks the best tower-damage outcome. Full MCTS would build a tree over many plies; simultaneous moves and the hidden opponent hand make that awkward | Not needed. The headline is exploitability, and the heuristic and random bots are already unseen by every stage-3 policy. **Possible later idea:** search at play time with the trained policy and value (AlphaZero-style) |
| Random placement inside a 4×4 "mega tile" | Adds uncontrollable noise and changes the game, so its equilibrium is not the real game's. Precision matters (Fireball radius 2.5 tiles, Hog pulls, kiting). Replaced by deterministic `placement_grid` and the conditional head |
| Independent MultiDiscrete card and position heads | Masks can't be exact per card; "where depends on what" can't be expressed; the prior project with independent x/y heads plateaued |
| Card-play bonus on by default | Rewards spending, not winning (cheap-card spam). Kept as an off-by-default flag |
| Elixir-advantage reward | Rewards hoarding |
| Troop-damage / elixir-trade reward | Easy to farm (kills on throwaway units); ClashRoyaleEnv needed several patches |
| Anchors (scripted bots) in stage-3 training | A second difference between A and B, and biased bot evaluations |
| Warm-starting stage 3 from stage 1 | Would expose A to bots through its weights (T11) |
| Random decks for the stage-3 comparison | Deck-luck noise, an unknown fair score, and all three methods probably still learning card basics when the budget ends |

---

## 13. New in this document (not discussed explicitly; flag anything you disagree with)

1. **`reward_clip` override in `MMDPuffeRL`** (§3.4). It's required because PuffeRL clamps every
   step's reward to [−1, 1], and that clamp would break exact potential-based shaping on terminal
   steps.
2. **Φ = 0 at the terminal state**, with the refund on the last step (§3.2). This is what "exactly
   potential-based" requires in an episodic game.
3. **Elixir potential capped at ±20 elixir of net leak**, and the play bonus at ±20 plays (§3.2).
   This bounds |Φ| < 1.5, so |r| < 2.5 on terminal steps and |r| < 3 on any step (typically far
   smaller).
4. **B's mix is 35% self / 65% snapshots** (AlphaStar's main-agent share). The earlier
   "~20% self, 10% anchors, 70% snapshots" was a placeholder, and the anchors are gone.
5. **The snapshot pool spans at least the last half of each run**: ~128 snapshots, 64 kept
   (§8.3).
6. **Best responses start from the target's weights**, with a budget calibration step (§9 E1).
7. **Sampled actions in every evaluation** (§9). Every tool samples by default (SPEC §19.11);
   `--greedy` would play the card-first greedy rule.
8. **Stage 1 runs through `league_train.py`** with one bot anchor per rung. That gives one code
   path with stage 3, plus resume and per-opponent history. It needs `--init-from` (§6.3).
9. **The flat head stays available** as an ablation and fallback (§4.1).
10. **Stage 4 has no anchors by default** (§8.4).
11. **Own deck as an unordered set, not the queue order** (§4.3). The order would leak the initial
    shuffle.
12. **The `eval.py` snapshot-loading fix** (§6.6). It's a real bug, found while writing this plan.
13. **Engine freeze from stage-3 launch** (§8.3). The original plan said W7.
14. **3 seeds per stage-3 method** (with the 16-GPU budget), and 2 BR seeds per target.
15. **The random bot, not the do-nothing bot, is the first real learning gate**, and the CPU
    learning check uses it (§7, §8.1). An untrained policy already beats the do-nothing bot
    (measured).

---

## 14. Facts this plan relies on (checked 2026-10-02 against `main` = `b6b63da`)

| Fact | Where |
|---|---|
| Current shaping formula; terminal ±1/0; zero-sum `r1 = −r0` | `pufferroyale/csrc/royale.h`, `c_step` step 3 |
| PuffeRL clamps rewards to [−1, 1] in `evaluate()` | `.venv/.../pufferlib/pufferl.py:275` (PufferLib 3.0 @ `3b5c604`) |
| `MMDPuffeRL` copies `PuffeRL.train()` and adds `mmd_coef·KL(π‖π_ref)` (masked); the magnet is refreshed every `mmd_ref_interval` epochs | `pufferroyale/trainer.py` |
| Pool mass split (self / anchors / PFSP snapshots) with fallbacks; PFSP `hard = (1−p)²` floored at `pfsp_eps`; oldest snapshot evicted first | `pufferroyale/league.py` (`weights()`, `pfsp_weight()`, `add_snapshot()`) |
| A's pool stays `{self: 1.0}` with snapshots present; B with 2 snapshots → `{self: 0.35, each: 0.325}`; A, B and C with the same seed have identical initial weights | run on the Mac, 2026-10-02 (`OpponentPool`, `LeagueRun`) |
| League results come from the env log (`win_0`/`win_1`), not rewards | `league.py` `LeagueVecEnv.send` |
| Pool-defining options can't change on `--resume`; `league_train.py` defaults `anneal_lr=False`; saves happen only at snapshot epochs | `scripts/league_train.py` (`POOL_KEYS`, `DEFAULTS`, `train()`) |
| `best_response.py` = a fresh learner vs one frozen target (pool = target only), evaluation samples by default | `scripts/best_response.py` |
| `tournament.py` samples by default (`--greedy` = card-first rule; `--sample` is a no-op, SPEC §19.11); `--deck-mode mirror` exists | `scripts/tournament.py` |
| `deck_metagame(agent, decks, matches, seed=0, greedy=False, device="cpu", …)` accepts a checkpoint path | `pufferroyale/metagame.py` |
| `eval.py --checkpoint <recurrent snap_N.pt>` fails (state-dict mismatch); a run directory works | reproduced on the Mac, 2026-10-02 |
| Observation: 17,707 floats = 25×32×18 spatial + 64×11 entities + 298 scalars + 2,305 mask; `CARD_SLOTS` 128; card ids `card_id+1` | `pufferroyale.royale` constants |
| `card_info(id)` exposes level-11 stats (hitpoints, damage, hit speed, range, speed, flying, targets, splash, spell damage/radius, crown-tower %) | `pufferroyale/binding.c` → `card_info` |
| Presets: `hog26` = Hog Rider, Musketeer, Cannon, Ice Golem, Ice Spirit, Skeletons, Fireball, The Log (plus giant, bait, golem, lavaloon, xbow, miner_poison, pekka_bridge, royal_hogs) | `pufferroyale.game.DECKS` |
| Elixir: 1× / 2× / 3× = 50 / 100 / 150 units per tick over ticks [0, 2400) / [2400, 4800) / [4800, 6000); 1 elixir = 2,800 units; start 6, cap 10 | `docs/SPEC.md` §4 |
| Random bot plays with probability `bot_play_prob` = 0.2 per decision | `docs/SPEC.md` §8, `royale.ini` |
| Untrained flat-head policy (random init) scores 1.00 vs `noop`, 0.15 vs `random`, 0.00 vs `heuristic` | `scripts/eval.py --episodes 100 --bots noop random heuristic --seats 0 1`, run on the Mac 2026-10-02 |
| A stage-1 rung (`--anchors bot:noop --self-play-frac 0 --anchor-frac 1.0`) trains and logs per-opponent results; checkpoint files work as opponents via `--anchors ckpt:…` | tiny CPU runs, 2026-10-02 |
| `deck_metagame(<path>, decks, matches, greedy=False)` works; tournament/metagame matches take ~0.5–0.65 s each on the Mac CPU | run on the Mac, 2026-10-02 |
| The Mac's installed `pufferlib/pufferl.py`, `pytorch.py`, `models.py` are byte-identical to upstream PufferLib at `3b5c604` (sha256 `ebe89e96…`, `206b8178…`, `eb8e0edb…`) | sha256 vs upstream GitHub, 2026-10-02 |
