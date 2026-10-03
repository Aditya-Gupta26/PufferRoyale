# PufferRoyale

A deterministic, integer-only Clash Royale 1v1 battle simulator written in C (header-only
C99), exposed to Python as a PufferLib 3.0 native environment and as a tick-level scenario /
debug API. It targets the 2026 ladder battle rules with card statistics from the 15.535.29
client (as decoded by [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim)), all cards at
tournament level 11.

> **Disclaimer.** This is an unofficial, non-commercial research project. It is **not
> affiliated with, endorsed, sponsored, or specifically approved by Supercell**, and
> Supercell is not responsible for it (see Supercell's
> [Fan Content Policy](https://supercell.com/en/fan-content-policy/)). The card numbers are
> Supercell's and are used as measurement input only. Automating the real game (bots,
> scripted input, live "advisors") violates Supercell's Terms of Service and gets accounts
> banned; everything here runs in our own simulator. Do not point it at live servers.

Status (SPEC v0.3-E.1, Phase E): the engine (match rules, **64 cards**, spells, pathing, the four
tower troops), the scripted bots, the PufferLib 3.0 env `pufferroyale.Royale` (observations, masks, rewards,
logs, text and raylib rendering), a masked PyTorch policy and the train / eval / watch /
bench / end-to-end scripts are implemented and verified. Phase D (SPEC §15) adds the
game-theoretic training and evaluation tooling: a league with PFSP opponents, an MMD-regularised
PPO trainer, a best-response exploitability probe, a tournament with a meta-game Nash solver, and
NYU Torch HPC templates. Phase E (SPEC §16) adds 43 cards (ids 21-63: spawners, death spawns and
death bombs, Inferno ramps, the Bandit's dash, the Miner's burrow, the Elixir Collector, Rocket,
Poison, Freeze, Earthquake, Lightning, Giant Snowball, Barbarian Barrel, ...), the Cannoneer, Dagger
Duchess and Royal Chef tower troops, the compact v0.3 observation (integer card ids) and a policy
that embeds them. Phase F (SPEC §17) adds a text interface for playing against language models
(state rendering, action parsing, mock baselines and an optional Anthropic adapter). The binding
contract is
[`docs/SPEC.md`](docs/SPEC.md); every implementation choice the spec leaves open, and every
known divergence from the real game, is listed in [`docs/FIDELITY.md`](docs/FIDELITY.md);
design decisions are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Install and build

Requirements: macOS or Linux, a C99 compiler (Apple clang / clang / gcc), Python >= 3.10.
The project venv is `.venv` (Python 3.12, created with `uv`) and already holds everything below.

```bash
source .venv/bin/activate
make build                   # python setup.py build_ext --inplace -> pufferroyale/binding*.so (-O2)
uv pip install -e .          # (or pip install -e .) the package: numpy + gymnasium
uv pip install -e '.[train]' # + torch and PufferLib 3.0 for train.py / eval.py / the policy
make debug                   # -O0 -g -fsanitize=address,undefined build in build/debug/pufferroyale/
                             #   (never over the in-place release .so); use it with
                             #   PUFFERROYALE_BINDING_DIR=build/debug/pufferroyale and the ASan runtime
                             #   preloaded -- `make asan-py` does both for the Python suites
```

**PufferLib 3.0 from source.** The `[train]` extra names `pufferlib>=3.0,<3.1`, but the env is a
native PufferLib env and needs no Ocean environments, so install PufferLib from its 3.0 branch
with `NO_OCEAN=1` (as the project venv was, commit `3b5c604`). That commit's `setup.py` needs a
one-line fix under `NO_OCEAN=1` (`c_extension_paths` is only defined inside `if not NO_OCEAN:`;
docs/DECISIONS.md D6):

```bash
git clone -b 3.0 https://github.com/PufferAI/PufferLib.git && cd PufferLib && git checkout 3b5c604
python - <<'PY'   # the one-line fix: define c_extension_paths before `if not NO_OCEAN:`
s = open("setup.py").read()
open("setup.py", "w").write(s.replace("c_extensions = []\n", "c_extensions = []\nc_extension_paths = []\n", 1))
PY
NO_OCEAN=1 uv pip install .   # (or NO_OCEAN=1 pip install .) then, back here: uv pip install -e '.[train]'
```

**raylib window (optional).** `tools/get_raylib.sh` downloads the prebuilt raylib 5.5 (macOS, or
Linux x86_64) into `third_party/`; `PR_RAYLIB=1 make build` then links it statically. To use
another raylib 5.5 build, point `RAYLIB_DIR` at a directory with `include/raylib.h` and
`lib/libraylib.a`. Linux also needs the GL / X11 development libraries. Without a display
the window is skipped and the text board is printed.

```bash
tools/get_raylib.sh && PR_RAYLIB=1 make build
RAYLIB_DIR=/opt/raylib-5.5 PR_RAYLIB=1 make build
```

Rebuild the extension after editing anything in `pufferroyale/csrc/` (`scripts/e2e_check.py`
refuses a stale build). `setup.py` uses absolute paths, so it also builds when run from
another directory.

## Test

```bash
make test-c                  # C unit tests (clang -std=c99 -Wall -Wextra -Werror -Wshadow), incl.
                             #   the snapshot fuzz test (tests/c/test_fuzz.c)
make asan                    # the same under AddressSanitizer + UBSan (every binary has a
                             #   TEST_TIMEOUT=300 s wall-clock limit; ASAN_SYMBOLIZE=1 symbolizes reports)
make asan-py                 # builder + spec pytest suites against the ASan/UBSan extension
                             #   (minus subprocess and timing tests, which need the release build)
make CC=gcc-15 BUILD=build/gcc test-c   # also warning-free with gcc
make bench                   # engine ticks/s on recorded random matches
make bench-stress            # worst cases: 216 Skeletons across the river, and the SPEC §16 large
                             #   battle (>= 10k ticks/s)
python -m pytest tests/builder -q    # builder suite
python -m pytest tests/spec -q       # independent tester's black-box suite
make test                    # test-c + build + both pytest suites
make gen / make gen-check    # regenerate the card / arena headers (byte-identical) / fail if stale
python scripts/e2e_check.py  # the whole ladder, PASS/FAIL table, non-zero exit on failure
python scripts/e2e_check.py --quick   # skip ASan, smaller soak and training smoke
```

`tests/c/` and `tests/builder/` are the builder's tests; `tests/spec/` belongs to the
independent tester (black-box tests against the SPEC).

## Train, evaluate, watch

```bash
# self-play, both seats controlled by the learner (num_agents=2), CPU
python scripts/train.py --total-timesteps 100000
# against the scripted heuristic bot, learner side drawn per episode
python scripts/train.py --num-agents 1 --opponent heuristic --total-timesteps 100000
# LSTM policy on the Apple GPU, other decks, custom output directory
python scripts/train.py --rnn --device mps --deck0 bait --deck1 random --data-dir runs
# any config key: --section.key value  (see pufferroyale/config/royale.ini)
python scripts/train.py --train.learning-rate 1e-4 --env.reward-tower 0.1

# a checkpoint (run directory or model .pt) against every bot from both seats: W/D/L
python scripts/eval.py --checkpoint experiments/pufferroyale_<run_id> --episodes 10
python scripts/eval.py --bot-policy heuristic --bots random    # a scripted bot as the "learner"

# watch a match as a text board (bot vs bot, or a checkpoint vs a bot)
python scripts/watch.py --p0 heuristic --p1 random
python scripts/watch.py --checkpoint experiments/pufferroyale_<run_id> --p1 heuristic
python scripts/watch.py --raylib     # a window with a PR_RAYLIB=1 build and a display; else text

python scripts/bench.py              # env steps/s for 1 and 16 native envs
python scripts/bench.py --multiprocessing --workers 4
```

Training writes `experiments/pufferroyale_<run_id>/` (PuffeRL's `model_*.pt`,
`trainer_state.pt`, and `config.json` describing the policy for `eval.py` / `watch.py`) and
prints a JSON summary (steps, SPS, final losses, episode stats); it exits non-zero if a loss
becomes non-finite. Judge progress by `learner_score` (the policy's own win 1 / draw 0.5 /
loss 0; `score` is team 0's), which is also the `[sweep]` metric in `royale.ini`.

## League training, exploitability, tournaments (SPEC §15)

```bash
# league: the learner plays itself (20%), the heuristic bot anchor (20%) and PFSP-weighted
# snapshots of its past weights (60%; their mass goes to the anchor until the first snapshot)
python scripts/league_train.py --total-timesteps 200000 --anchors bot:heuristic --num-envs 16
# two anchors, more self-play, "variance" PFSP, and the MMD regulariser (KL to a reference
# policy refreshed every 20 epochs; --mmd-coef > 0 selects MMDPuffeRL)
python scripts/league_train.py --anchors bot:heuristic,bot:random --self-play-frac 0.4 --anchor-frac 0.2 \
    --pfsp variance --mmd-coef 0.05 --mmd-ref-interval 20 --snapshot-interval 10 --max-snapshots 16
# continue a run; --total-timesteps is the absolute target
python scripts/league_train.py --resume experiments/league/<run_id> --total-timesteps 1000000

# exploitability probe: train a fresh best response against one frozen snapshot, then play
# --matches full matches (half per seat); prints JSON with br_score and 2*br_score - 1
python scripts/best_response.py --target experiments/league/<run_id>/snap_100.pt \
    --total-timesteps 200000 --matches 40 --out br.json

# round-robin tournament of bots and checkpoints over decks: payoff matrix, Elo, meta-game Nash
python scripts/tournament.py --agents bot:noop bot:random bot:heuristic experiments/league/<run_id>/snap_100.pt \
    --decks hog26 giant --matches 4 --out tournament.json
```

A league run lives in `<data-dir>/league/<run_id>/`: `snap_<epoch>.pt` (the learner every
`--snapshot-interval` epochs, added to the pool), `league_state.json` (pool snapshots -- stored
relative to the run dir, so runs can be moved -- stats and RNG, epoch, step, args),
`learner.pt` + `trainer_state.pt` (for `--resume`: the resumed run continues the learner, optimizer,
pool and RNG states and re-deals the matches that were in progress, so it is deterministic but not
bit-identical to an uninterrupted run; `--learning-rate` / `--train.adam-*` given at resume are
applied), `nonfinite.json` if an epoch produced NaN/inf (that epoch is never saved and the run
exits 1), `history.jsonl` (per
epoch: losses and the learner's score against each opponent), and at the end `model_<epoch>.pt`
+ `config.json`, so `eval.py`, `watch.py`, `best_response.py` and `tournament.py` accept the run
directory or any snapshot. One progress line per epoch (`--print-interval`) shows the losses, the
pool size and `opponent=score/games` for the episodes that finished in that epoch.

Evaluation uses the settings a checkpoint was trained with: `tournament.py`, `best_response.py`,
`eval.py` and `metagame.play_match` read frame_skip, deploy lockout, tiebreak and tower troops from the
run's `config.json` (each checkpoint decides at its own frame_skip; bots every 10 ticks) and warn when a
checkpoint plays under other settings; `--frame-skip N` / `--ignore-train-config` override. Every
tool that plays a policy samples its actions from the masked policy by default, seeded so results
are reproducible (SPEC §19.11). `--greedy` (`greedy=True` in Python) plays the card-first greedy
rule instead: the most likely of wait and the four card slots by marginal probability, then that
slot's most likely tile (`pufferroyale.league.greedy_actions`). Checkpoint
arguments accept a `.pt` file, `ckpt:<path>` or a run directory; decks accept presets or 8
comma-separated card names / ids. Every script trains with `MMDPuffeRL` (plain PPO at `--mmd-coef 0`),
whose bf16 autocast never reuses stale weight casts (SPEC §18.6).

The pieces are importable too:

```python
from pufferroyale.league import OpponentPool, LeagueVecEnv   # PFSP pool; learner-rows-only vecenv
from pufferroyale.trainer import MMDPuffeRL                  # PuffeRL + mmd_coef * KL(pi || pi_ref)
from pufferroyale import metagame as mg                      # solve_zero_sum, payoff_matrix, elo, ...

pool = OpponentPool(anchors=("bot:heuristic",), self_play_frac=0.2, anchor_frac=0.2, seed=0)
vecenv = LeagueVecEnv(pool, num_envs=16, seed=0, deck0="hog26", deck1="giant")  # hand to PuffeRL
x, y, v = mg.solve_zero_sum([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])     # RPS: uniform, value 0
print(mg.play_match("bot:heuristic", "bot:random", "hog26", "giant", seed=0))  # +1 / 0 / -1
```

`LeagueVecEnv` wraps `num_envs` self-play matches and gives PuffeRL one learner row per match;
the other seat is played by the opponent drawn from the pool for that episode (policy opponents
batched without grad, bots in C), so opponent transitions never reach the PPO buffer. The
learner's seat is drawn per episode. Everything is deterministic given the seeds. HPC (NYU
Torch, Apptainer + SLURM) templates are in [`hpc/`](hpc/README.md).

### Training operations (v0.5, SPEC §19; plan in [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md))

```bash
# reward v2 (potential-based, zero-sum; reward_gamma = train.gamma is injected), annealed to 0 over
# the first 30% of the run; coarse placement grid (4x4 blocks: 161 actions); deck sampling
python scripts/league_train.py --total-timesteps 2000000 --shaping-anneal-frac 0.3 \
    --env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.02 --train.reward-clip 0 \
    --env.placement-grid 4 --env.deck-pool "hog26;giant:2;random:50:0" --env.random-deck-frac 0.1
# policy / trainer keys: --policy.head flat|conditional, --policy.card-stats 0|1, --policy.pos-channels N,
#   --train.ent-coef-card X --train.ent-coef-pos Y (both or neither)
# warm start (weights only, same architecture) and early stop at a snapshot epoch once the score over
# the last N matches vs EVERY anchor is >= X
python scripts/league_train.py --init-from experiments/league/<run> --anchors bot:heuristic \
    --early-stop-score 0.6 --early-stop-window 2000
# train.py: continue a run (absolute target), or start a new one from any checkpoint's weights
python scripts/train.py --resume experiments/pufferroyale_<run_id> --total-timesteps 2000000
python scripts/train.py --init-from experiments/league/<run>/snap_40.pt --shaping-anneal-frac 0.2
# bot ladder: chained league runs gated on the score vs each anchor; writes <data-dir>/stages_<P>.json
python scripts/stages.py --run-prefix S1 --rungs "bot:noop:5000000:0.95;bot:random:50000000:0.90" \
    --gate-window 2000 --num-envs 256 --device cuda          # other flags go to every rung
# eval: per deck (each as a mirror) + pooled rows, Wilson 95% intervals (ci95) in the JSON
python scripts/eval.py --checkpoint experiments/league/<run> --decks "hog26;giant;random:5:0" --json e.json
# best response that starts from the target's own weights
python scripts/best_response.py --target experiments/league/<run>/snap_40.pt --init-from-target
# within-run transitivity of the snapshots (payoff, Elo, Nash, later_beats_earlier, cyclic_triads)
python scripts/transitivity.py --run experiments/league/<run> --snapshots 10 --matches 100 --out tr.json
# score per opponent and losses vs step from history.jsonl (pip install -e '.[plots]' for matplotlib)
python scripts/plot_history.py experiments/league/<run> --out curves.png
# Weights & Biases (train.py and league_train.py; WANDB_MODE=offline works without network; without
# --wandb, wandb is never imported)
python scripts/league_train.py --wandb --wandb-project pufferroyale --wandb-group s1 --tag v05 ...
```

`config.json` records the env kwargs actually used (incl. `reward_gamma`, the anneal keys and
`placement_grid`); every tool plays a checkpoint through its own placement grid.

## Play against LLMs (SPEC §17)

A provider-agnostic text interface: each decision, the model gets a static system prompt
(`pufferroyale.llm.RULES_PROMPT`: rules, own-frame coordinates, all 64 cards with their level-11
numbers, the reply format) and the current state as text (`render_state`), and answers with a final
line `WAIT` or `PLAY <slot|card> AT <tx>,<ty>`. Unreadable replies, cards not in hand and illegal
tiles count as `WAIT` and are recorded.

```bash
# offline baselines: no network, deterministic
python scripts/llm_match.py --model mock_first_legal --opponent bot:heuristic --matches 10 --out runs/fl.json
python scripts/llm_match.py --model mock_random --opponent bot:random --matches 10 --seed 3 --out runs/rnd.json
python scripts/llm_match.py --model mock_wait --print-prompt          # show one rendered state
# a real model (needs `pip install anthropic` + credentials); refused without --allow-network
python scripts/llm_match.py --model anthropic:claude-opus-5-5 --effort low --opponent bot:heuristic \
    --matches 4 --out runs/opus.json --allow-network
# LLM vs LLM, or vs a trained checkpoint
python scripts/llm_match.py --model anthropic:claude-sonnet-5 --opponent model:anthropic:claude-haiku-4-5 \
    --matches 2 --out runs/sonnet_vs_haiku.json --allow-network
python scripts/llm_match.py --model mock_random --opponent experiments/league/<run_id>/snap_100.pt --matches 4
```

Seats alternate; the model decides every `--decision-interval` ticks (20 = 1 s of game time;
bots and checkpoints act every 10 ticks, the env cadence). The last stdout line is a JSON summary
(W/D/L, score, mean crowns, illegal / parse-error / model-error rates per decision, mean latency);
`--out` also writes per-match results and `<out>.transcripts.jsonl` (every decision: tick, prompt
hash, reply, action, error, latency, and the API's stop reason / token usage / refusal / error).
Known Anthropic model ids: `claude-opus-5-5`, `claude-opus-5`, `claude-fable-5-1`,
`claude-sonnet-5`, `claude-haiku-4-5`. The adapter caches the system prompt, sends no assistant
prefill, leaves thinking at the model default (`--effort` trades depth for latency; not sent to
Haiku), adds server-side refusal fallbacks for Fable 5.1 / Opus 5, and treats refusals and failed
calls as `WAIT`.

```python
import pufferroyale as pr
from pufferroyale import llm

agent = llm.LLMAgent(llm.mock_first_legal)            # any model_fn(system, user) -> str
res = llm.play_llm_match(agent, "bot:heuristic", "hog26", "giant", seed=0)
print(res)                    # result, crowns, end_reason, ticks, decisions, illegal, parse_errors, ...
g = pr.Game(seed=0); g.tick(200)
print(llm.render_state(g, 0))                          # the text a model sees (own frame, public info only)
action, info = llm.parse_action("PLAY 1 AT 9,20", g, 0)
```

The rendered state never contains the opponent's elixir, hand, queue or unrevealed deck: only the
cards it has played, the last four, its hand once all eight cards have been seen, and an elixir
upper bound computed from what it spent (tested by changing those hidden values and comparing the
text). The mocks read only the text (`mock_first_legal` plays the first affordable card at the first
tile of its listed legal region), which shows the text alone is enough to play.

## Quick start (Python)

```python
import numpy as np
import pufferroyale as pr
from pufferroyale import royale as R

# the scenario / debug API
g = pr.Game(deck0="hog26", deck1="giant", seed=0)   # deploy lockout 90 ticks by default
g.tick(90)
print(g.state()["hand"][0])                          # 4 card ids of team 0's hand
err = g.play_tile(0, slot=0, tx=3, ty=20)            # own-frame tile; applied in the next tick
g.tick(1)
for e in g.entities():
    print(e["id"], e["unit"], e["team"], e["x"], e["y"], e["hp"])
mask = g.legal_mask(1)                                # uint8[2305], exactly the engine's legality
a = pr.bot_action(g, 1, "heuristic")                  # a scripted bot's (legal) action
snap = g.snapshot(); g.tick(100); g.restore(snap)     # memcpy snapshot of the POD state
print(g.ansi())                                       # text board
print(pr.card_info("Knight")["hitpoints"])            # 1766 (level 11)

# the PufferLib env: 4 matches, self-play (8 agent rows), own-frame observations
env = pr.Royale(num_envs=4, num_agents=2, seed=0)
obs, _ = env.reset(seed=0)                            # (8, R.OBS_SIZE) float32
legal = obs[:, R.MASK_OFFSET:] > 0.5                  # the action mask (action 0 = no-op)
obs, rew, term, trunc, info = env.step(np.zeros(8, np.int32))
print(R.SCALAR_INDEX["elixir"], R.scalar(obs[0], "elixir"))
env.close()
```

Coordinates are engine-frame millitiles (1 tile = 1000, `y = 0` is the top edge; team 0 is
the bottom player). One tick is 50 ms; speeds are millitiles per tick. Env actions are
`Discrete(2305)`: 0 = no-op, else `slot = (a-1)//576`, own-frame tile `cell = (a-1)%576`.
With `pufferlib.vector.make(pufferroyale.Royale, backend=Serial | Multiprocessing, ...)` the
env also runs under PufferLib's Python vectorization (spawn start method on macOS).

## Cards, decks, tower troops

64 cards at level 11, ids stable (`pr.CARD_NAMES[i]`): 0 Knight, 1 Archers, 2 Musketeer, 3 Giant,
4 Hog Rider, 5 Minions, 6 Baby Dragon, 7 Valkyrie, 8 Skeleton Army, 9 Skeletons, 10 Ice Golem,
11 Ice Spirit, 12 Prince, 13 Wizard, 14 Cannon, 15 Tesla, 16 Fireball, 17 Arrows, 18 Zap, 19 The Log,
20 Goblin Barrel, 21 Barbarians, 22 Mini P.E.K.K.A, 23 P.E.K.K.A, 24 Mega Minion, 25 Bats, 26 Spear
Goblins, 27 Goblins, 28 Goblin Gang, 29 Royal Giant, 30 Bomber, 31 Princess, 32 Dart Goblin,
33 Minion Horde, 34 Fire Spirit, 35 Ice Wizard, 36 Golem, 37 Lava Hound, 38 Balloon, 39 Giant
Skeleton, 40 Witch, 41 Tombstone, 42 Inferno Tower, 43 Inferno Dragon, 44 Bandit, 45 Battle Ram,
46 Royal Hogs, 47 Miner, 48 Mortar, 49 X-Bow, 50 Elixir Collector, 51 Bomb Tower, 52 Rocket,
53 Poison, 54 Freeze, 55 Earthquake, 56 Lightning, 57 Giant Snowball, 58 Barbarian Barrel,
59 Rascals, 60 Elite Barbarians, 61 Skeleton Dragons, 62 Wall Breakers, 63 Night Witch.

Preset decks (`pr.DECKS`; `'random'` draws 8 of the 64 at every reset):

| name | cards |
|---|---|
| `hog26` | Hog Rider, Musketeer, Cannon, Ice Golem, Ice Spirit, Skeletons, Fireball, The Log |
| `giant` | Giant, Prince, Baby Dragon, Wizard, Minions, Knight, Arrows, Zap |
| `bait` | Goblin Barrel, Skeleton Army, Tesla, Valkyrie, Archers, Knight, The Log, Fireball |
| `golem` | Golem, Night Witch, Baby Dragon, Lightning, Mega Minion, Barbarian Barrel, Mini P.E.K.K.A, Zap |
| `lavaloon` | Lava Hound, Balloon, Minions, Mega Minion, Skeleton Dragons, Arrows, Fireball, Tombstone |
| `xbow` | X-Bow, Tesla, Archers, Knight, Skeletons, Ice Spirit, Fireball, The Log |
| `miner_poison` | Miner, Poison, Goblin Gang, Bats, Inferno Tower, Valkyrie, Spear Goblins, The Log |
| `pekka_bridge` | P.E.K.K.A, Battle Ram, Bandit, Minions, Musketeer, Zap, Poison, Dart Goblin |
| `royal_hogs` | Royal Hogs, Earthquake, Fire Spirit, Barbarian Barrel, Goblin Gang, Mega Minion, Zap, Musketeer |

Tower troops replace both Princess towers of a team: `Game(..., tower_troop0="cannoneer",
tower_troop1="royal_chef")`, `Royale(..., tower_troop0=..., tower_troop1=...)` or
`--env.tower-troop0 dagger_duchess` in `train.py`; values `princess` (default), `cannoneer`,
`dagger_duchess`, `royal_chef`. Both tower troops are visible in the observation.

**Observation (v0.5, SPEC §16.4 + §19.3).** `OBS_SIZE = 17715` = 25 spatial planes x 32 x 18 + 64 entity
rows x 11 + 306 scalars + the 2305-long action mask. Card identities are integer ids `card_id + 1`
(0 = empty) in entity feature 0 and in the `hand`, `next_card`, `opp_last4` and `own_deck` (the own
deck as a set, ascending ids) scalars; opponent cards seen / deduced hand are 128-wide multi-hots;
`own_tower_troop` / `enemy_tower_troop` are 4-wide one-hots. Offsets: `R.SCALAR_INDEX`,
`R.ENTITY_FEATURES`, `R.CARD_SLOTS`. The mask stays the fine 2305-mask under any `placement_grid`;
`R.action_mask(obs, grid)` derives the coarse one. Checkpoints from before v0.5 do not load.

## Layout

```
pufferroyale/
  csrc/                header-only C99 engine
    pr_math.h          isqrt64, floor/ceil division, PCG32, FNV-1a
    pr_defs.h          card-database row types
    pr_card_db.h       GENERATED level-11 card / unit / projectile / area / buff tables
    pr_arena_db.h      GENERATED 36x64 half-tile arena bitmask, tile flags, tower centres
    pr_types.h         constants and the plain-old-data game state (PrState)
    pr_arena.h         arena queries (team-oriented, seat-symmetric)
    pr_entity.h        entity pool (sorted by id), formations, status queries
    pr_rules.h         elixir, hand & cycle, placement legality + mask, plays, towers, Judge
    pr_target.h        targeting (acquisition, retention, default route)
    pr_combat.h        attack cycle, damage buffer, projectiles, buffs, knockback, Resolve, Reap
    pr_path.h          flow-field ground pathing + line of sight (pure cache outside the state)
    pr_move.h          Path/Move and Collide phases
    pr_spell.h         spells (projectile, waves, area, rolling, pulsing, Lightning), death bombs
    pr_engine.h        reset, the 11-phase tick pipeline (spawners, Collector, Royal Chef in
                       Status), state hash, debug hooks, snapshot validation
    pr_bots.h          scripted bots: noop, random, heuristic (integer-only, own RNG)
    pr_obs.h           observation encoder (spatial planes, entity rows, scalars, mask)
    pr_render.h        ANSI text board; raylib window under PR_RAYLIB
    royale.h           PufferLib 3.0 Env: c_reset / c_step / c_render / c_close, Log
  binding.c            the pufferroyale.binding extension (env protocol + debug API + bots;
                       Phase D: env_log_peek, a read-only view of one env's episode log)
  env_binding.h        vendored PufferLib 3.0 binding header (unmodified)
  game.py              pufferroyale.Game / Bot / bot_action / card_info / DECKS / PlayError
  royale.py            pufferroyale.Royale (PufferEnv) + observation layout constants
  torch.py             Policy (CNN + card-id embeddings + entity pooling + scalar MLP, masked
                       logits), Recurrent
  league.py            OpponentPool (PFSP) + LeagueVecEnv (learner rows only) + checkpoint loader
  trainer.py           MMDPuffeRL: PuffeRL + mmd_coef * KL(pi || pi_ref)
  metagame.py          zero-sum LP solver, payoff matrix, meta-game Nash, Bradley-Terry Elo, play_match
  llm.py               text interface for LLM play: RULES_PROMPT, render_state, parse_action, LLMAgent,
                       play_llm_match, mock model functions, optional Anthropic adapter
  config/royale.ini    PuffeRL config (merged over pufferlib's default.ini)
scripts/               train.py, eval.py, watch.py, bench.py, e2e_check.py,
                       league_train.py, best_response.py, tournament.py, llm_match.py
hpc/                   Apptainer definition + NYU Torch SLURM templates (not run locally)
tools/gen_card_db.py   data/source/cards-15.535.json -> pr_card_db.h
tools/gen_arena.py     data/source/royalesim-arena.json -> pr_arena_db.h
tools/get_raylib.sh    fetch the prebuilt raylib 5.5 into third_party/ (optional renderer)
tests/c/               C unit tests (engine, rules, combat, specials, §16 cards, path, env, fuzz),
                       soak, benchmarks
tests/builder/         builder pytest suite
tests/spec/            tester-owned pytest suite
data/source/           pinned inputs (see data/source/PROVENANCE.md)
docs/                  SPEC.md (contract), FIDELITY.md, DECISIONS.md, PLAN.md
```

## Design in one paragraph

The whole match is one fixed-size POD struct (`PrState`, ~65 KB): no pointers, no heap
after init, zeroed with `memset` so that padding is deterministic; snapshot and restore are
`memcpy`, the state hash is FNV-1a over its bytes. The simulation uses integers only
(millitiles, milliseconds, exact `isqrt` on int64) and runs RoyaleSim's measured tick phase
order: Upkeep, Status, Spawn, Target, Attack, Path/Move, Collide, Projectile, Resolve,
Reap, Judge. All damage is buffered and applied simultaneously in Resolve. Ground pathing
uses Dijkstra distance fields over the half-tile grid with line-of-sight smoothing; the field
cache lives outside the state but is a pure function of it, so snapshots stay exact. A
scenario and its 180-degree rotation evolve as exact rotations of each other. The env wraps
one match per native env; observations are written in each team's own frame by the same C
code that decides legality, so the action mask is the engine's legality function; bots use
their own RNG streams, never the game's.

## Licences

Code: see the repository licence. `pufferroyale/env_binding.h` is from PufferLib (MIT,
`pufferroyale/PUFFERLIB_LICENSE`); the data inputs come from RoyaleSim (MIT,
`data/source/ROYALESIM_LICENSE`; the card numbers themselves are Supercell's).

## Training on the NYU HPC cluster

See `docs/HPC_HANDOFF.md` (a step-by-step guide for an agent or person with cluster access: container build, verification incl. `scripts/golden_hashes.py`, throughput calibration, learning gates, the course experiments and evaluation) and the templates in `hpc/`.
