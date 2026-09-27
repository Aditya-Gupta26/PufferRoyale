# PufferRoyale: HPC training handoff

**Who this is for:** an agent that has shell access to the NYU HPC cluster (Torch, SLURM) and a
checkout of this repository. Your job is to take PufferRoyale from "verified on a laptop" to
"training on GPUs", and then to run the course experiments. Work through the phases in order,
because each phase has a gate. Report back to the user at every gate.

**Deadline:** the course deliverable is due in early December 2026, so favour reliable progress
over breadth.

---

## 0. What PufferRoyale is (one screen)

- **The simulator.** A Clash Royale 1v1 battle simulator written as header-only C99, integer-only
  and deterministic at 20 ticks/s. It covers 64 cards and 3 tower troops, with the full match
  rules. It is exposed as a PufferLib 3.0 environment `pufferroyale.Royale`:
  - one decision every `frame_skip = 10` ticks (0.5 s);
  - action space `Discrete(2305)`: no-op, or 4 hand slots × 576 own-frame tiles;
  - an own-frame observation of 17,707 float32 values, including the exact legal-action mask;
  - zero-sum reward: +1 / −1 / 0 at match end, with optional antisymmetric shaping.
- **Training tools** (PyTorch, PufferLib 3.0):
  - `scripts/train.py`: PPO via `MMDPuffeRL`, which is PuffeRL with an optional KL
    regulariser; coefficient 0 gives plain PPO.
  - `scripts/league_train.py`: a PFSP league over frozen snapshots, scripted anchors and
    self-play. Resumable.
  - `scripts/best_response.py`: exploitability probe.
  - `scripts/tournament.py`: round robin, Elo, and the meta-game Nash.
  - `scripts/eval.py`: evaluation against the scripted bots.
- **Status.** Everything was verified on an Apple M4 laptop: 1,425 pytest tests, 16,651 C checks,
  clean under ASan/UBSan, and a 16/16 end-to-end ladder. It has **never run on Linux or a GPU**,
  and no training beyond 100–150k-step CPU smoke runs has been done. The built-in heuristic bot
  still beats every trained checkpoint.

### Read these first, in order
1. `docs/STATUS.md`: what exists, how it was verified, known approximations.
2. `README.md`: the command reference and PufferLib install notes.
3. `hpc/README.md` and `hpc/pufferroyale.def`, `hpc/train.sbatch`, `hpc/league.sbatch`. These are
   templates written for this cluster but **never run**.
4. `docs/SPEC.md`: the binding contract. You need §9 (env), §15 (league, MMD, evaluation
   tooling) and §18. Skim the rest.
5. `docs/DECISIONS.md` and `docs/FIDELITY.md`, to know what is deliberate.

---

## 1. Ground rules

- **Never compute on login nodes.** Build, test and train inside SLURM jobs (`srun` for
  interactive work, `sbatch` for runs). Put all outputs under `/scratch/$USER`, never `$HOME`.
- **Game logic is governed by `docs/SPEC.md`.** Do not change engine behaviour (`pufferroyale/csrc/*`)
  to make training easier. If you believe a game rule is wrong, write it up and ask the user.
  Engine changes need a SPEC amendment and tests.
- **Keep the suite green.** `pytest tests -q` and `python scripts/e2e_check.py` must pass on the
  cluster, and every new feature or fix gets tests. `tests/spec/` is an independent black-box
  suite: do not edit it to make it pass; report instead.
- **Git.** Work on a branch (e.g. `hpc-training`) and make small commits with clear messages.
  Never commit secrets (wandb keys), checkpoints, run directories or `.sif` images; `.gitignore`
  already covers `experiments/`, `build/` and `*.so`.
- **Write things down.**
  - Record decisions in `docs/DECISIONS.md` (append, next free ID).
  - Record every run and its result in a new `docs/EXPERIMENTS.md`: date, git commit, command,
    SLURM job id, GPU type, GPU-hours, the outcome, and the path to its outputs.
- **Stop and ask the user** when:
  - you lack an account, partition, or build permission;
  - `scripts/golden_hashes.py` fails;
  - any test fails and you cannot explain why;
  - you want to launch anything larger than a budget the user has approved;
  - you are tempted to change game rules or loosen a test.

---

## 2. Ask the user for these before starting

| Item | Why |
|---|---|
| SLURM account, and GPU partition names and types (L40S / H200) | fill the `#SBATCH` placeholders |
| Apptainer/Singularity availability, and whether `--fakeroot` builds are allowed (and where) | building the container |
| Internet access from login and compute nodes (pip, git, wandb) | container build; online vs offline logging |
| Scratch path, quota, and purge policy | where runs and checkpoints live |
| Wall-time limits and preemption policy | job length and resume cadence |
| GPU-hour budget for the semester | sizing the experiments |
| wandb account and project (optional) | experiment tracking |
| Deck and tower-troop setting for the main experiments (default: `hog26` mirror, Tower Princess) | experiment design |

---

## 3. Phase 1: get the code and the environment onto the cluster

```bash
# on a login node (git only)
cd /scratch/$USER
git clone <repo-url> PufferRoyale && cd PufferRoyale
git checkout -b hpc-training
```

### 3a. Container (preferred)

1. Build the image from `hpc/pufferroyale.def` as described in `hpc/README.md` §2, on a node where
   building is allowed:

   ```bash
   apptainer build --fakeroot pufferroyale.sif hpc/pufferroyale.def
   mkdir -p /scratch/$USER/containers && mv pufferroyale.sif /scratch/$USER/containers/
   apptainer test /scratch/$USER/containers/pufferroyale.sif
   ```

   The image contains:
   - CUDA 12.4 devel, Python 3.12, and torch cu124;
   - PufferLib 3.0 at commit `3b5c604`, built from source with `NO_OCEAN=1` plus a one-line
     `setup.py` fix (upstream bug; see DECISIONS D6);
   - the package, built into `/opt/pufferroyale`.

2. **Use a bind-mounted checkout for development and testing.** The image's `%files` omits
   `tests/`, `docs/`, `hpc/` and the `Makefile`, so bind-mount the checkout and build the extension
   for Linux inside the container, once per code change:

   ```bash
   SIF=/scratch/$USER/containers/pufferroyale.sif
   REPO=/scratch/$USER/PufferRoyale
   run() { apptainer exec --nv --bind /scratch/$USER --pwd $REPO $SIF "$@"; }
   run python setup.py build_ext --inplace   # do NOT set PR_RAYLIB on the cluster
   ```

   Jobs can then use `PROJECT=$REPO` (the sbatch templates honour it) so they run your checkout
   instead of the baked-in copy.

3. If the image build fails, fix `hpc/pufferroyale.def`, commit the fix, and record it in
   DECISIONS.md.

### 3b. Fallback: plain venv without containers

Use this only if Apptainer is unavailable.
1. `module load` CUDA 12.x and Python 3.12 (or install Python 3.12 with `uv`).
2. Create a venv on scratch, then `pip install torch --index-url https://download.pytorch.org/whl/cu124 "numpy<2" scipy gymnasium psutil rich rich_argparse Cython`.
3. Clone PufferLib branch `3.0` at commit `3b5c604`.
4. Apply the fix: add the line `c_extension_paths = []` right after `c_extensions = []` in its
   `setup.py`.
5. Install it: `NO_OCEAN=1 pip install --no-build-isolation .`
6. In the repo, build the extension: `python setup.py build_ext --inplace`.

---

## 4. Phase 2: verify on a GPU node (gate G0)

Open an interactive GPU session as in `hpc/README.md` §5 (e.g. `srun ... --gres=gpu:1
--cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash`), then run the following with the
`run` helper:

```bash
run make test-c                         # C unit/scenario tests, expect "0 failures" everywhere
run make asan                           # C tests under ASan+UBSan, expect clean
run python -m pytest tests -q           # ~5-10 min, expect all passed
run python scripts/golden_hashes.py     # cross-platform determinism, MUST print "golden hashes: PASS"
run python scripts/e2e_check.py         # expect "OVERALL PASS 16/16"
# GPU smoke runs (tiny, finite losses expected)
run python scripts/train.py --device cuda --total-timesteps 300000 --num-envs 64 \
    --batch-size 16384 --minibatch-size 4096 --quiet --summary-json /scratch/$USER/pr_smoke_train.json
run python scripts/league_train.py --device cuda --total-timesteps 200000 --num-envs 64 \
    --anchors bot:heuristic --snapshot-interval 2 --data-dir /scratch/$USER/pufferroyale --run-id smoke
```

What to expect and how to handle failures:

- **Golden hashes.** `tests/golden/golden_hashes.json` holds per-100-tick state hashes of six
  seeded bot matches, generated on macOS arm64. The engine is integer-only with zeroed padding,
  so Linux must reproduce them bit for bit. If it doesn't, **stop and report** the first
  diverging match and tick. Do not regenerate the reference.
- **Performance tests.** Their thresholds (engine ≥ 20k ticks/s, env ≥ 2k steps/s, stress ≥ 10k
  ticks/s) were set on an Apple M4. Server cores should pass. If they don't, report the
  measured numbers; never lower a threshold quietly.
- **`make asan-py` on Linux with clang.** The Makefile asks clang for `libasan.so` (the gcc
  runtime name); clang's runtime is `libclang_rt.asan-x86_64.so`. If that target fails, fix the
  runtime lookup in the Makefile, e.g. try the clang name first. This affects tooling only.
  Subprocess tests are skipped under ASan by design.
- **First Linux build warnings or errors.** The C has been compiled only on macOS, with clang and
  gcc-15. Fix genuine portability issues, such as a missing feature-test macro, without
  changing behaviour. The golden hashes prove you didn't.

**G0 passes when** all the commands above pass. Report the results with the GPU type, CUDA and
driver versions, and the torch version.

---

## 5. Phase 3: throughput calibration

Find the fastest stable configuration for each GPU type before any long run. Measure agent
steps/s (SPS), GPU utilisation, GPU memory and CPU utilisation for:

- `scripts/train.py --device cuda`:
  - `--num-envs` ∈ {64, 128, 256, 512};
  - `--batch-size` ∈ {32768, 65536, 131072};
  - `--minibatch-size` ∈ {8192, 16384};
  - `--backend PufferEnv` vs `--backend Multiprocessing --workers {4, 8}`, which runs several
    env processes;
  - with and without `--rnn`.
- `scripts/league_train.py --device cuda`: `--num-envs` ∈ {64, 128, 256}, with a mixed pool.

Keep these facts in mind:

- **Memory.** One observation row is 17,707 float32 = 70.8 KB, so the rollout buffer costs about
  `batch_size × 70.8 KB` (65,536 rows ≈ 4.6 GB) on top of the model.
- **Env speed.** The C env runs about 26k env steps/s per CPU core. `train.py` can use several
  processes (`Multiprocessing`).
- **League bottleneck.** `LeagueVecEnv` steps all matches in **one** process:
  - bot opponents run in C;
  - checkpoint opponents are batched on the GPU;
  - so env stepping is single-core. If league SPS is CPU-bound, see optional improvement O3.
- **Batch arithmetic.** PuffeRL needs `batch_size / bptt_horizon ≥ number of learner rows`, where
  rows = `num_envs × num_agents` for `train.py` and `num_envs` for the league; it also needs the
  minibatch size to be a multiple of `bptt_horizon` (64). Keep `batch_size` a multiple of
  `rows × 64`. The scripts raise `max_minibatch_size` to the minibatch size themselves.

Write a table in `docs/EXPERIMENTS.md` (GPU type × config → SPS, memory) and pick one standard
configuration per GPU type. Then update the defaults in `hpc/*.sbatch` (`NUM_ENVS`, `BATCH_SIZE`,
`MINIBATCH_SIZE`, `--cpus-per-task`) to match.

---

## 6. Phase 4: operations work (small code changes, with tests)

1. **Experiment tracking.** Nothing logs to wandb or TensorBoard yet; the league writes
   `history.jsonl` and every script prints a JSON summary.
   - Add a `--wandb` flag, with `--wandb-project` and `--wandb-group`, to `train.py` and
     `league_train.py`. PufferLib 3.0 ships `pufferlib.pufferl.WandbLogger`, which `PuffeRL`
     accepts as `logger=`, or you can write a thin logger that also records the league's
     per-opponent scores.
   - Support offline mode (`WANDB_MODE=offline`, then `wandb sync` from a login node) in case
     compute nodes have no internet.
   - Keep the API key in the environment, never in the repo.
   - Also add `scripts/plot_history.py`: the learner's score against each opponent over time,
     and losses, from `history.jsonl`, rendered to PNG.
2. **Resume for long runs.** `train.py` (PuffeRL) cannot resume, while `league_train.py` can.
   - Either run long jobs through `league_train.py`, with a pure self-play configuration for
     baseline A (see §8), or add `--resume` to `train.py` with tests.
   - Set `--snapshot-interval` so the league saves at least every ~30 minutes of wall time
     (saves happen at snapshot epochs).
   - Keep `league.sbatch` requeue-safe (it already resumes automatically).
3. **Seeds as job arrays.** Add `#SBATCH --array` support, or a wrapper, so each configuration
   runs with 2–3 seeds.

---

## 7. Phase 5: learning-sanity gates (short runs, 1 GPU)

Before the expensive experiments, prove the learning signal works. Use `train.py --num-agents 1
--opponent <bot>` (the learner's side is random each episode) and watch `learner_score` in the
logs:

| Gate | Opponent | Pass condition | Budget guide |
|---|---|---|---|
| G1 | `noop` | learner_score ≥ 0.95 | ≤ 20M steps |
| G2 | `random` | learner_score ≥ 0.90 | ≤ 50M steps |
| G3 | `heuristic` | learner_score clearly above 0.5, then climbing | 100–300M steps |

**If G1 or G2 fail, debug before scaling:**
1. Watch the policy play: `scripts/watch.py --checkpoint <run dir> --p1 random`.
2. Check the per-card play rates and illegal-action counts in the logs.
3. Check `learning_rate` (default 3e-4), `ent_coef` (0.01), `gamma` (0.999), `gae_lambda`
   (0.95), `bptt_horizon` (64), batch size, and `--rnn`.
4. Only then try light shaping: `--env.reward-tower 0.1–0.5` or `--env.reward-crown 0.1`. Both
   are antisymmetric, so zero-sum is preserved. Anneal them away later.
5. Keep float32. `precision=bfloat16` was fixed to update weights correctly
   (`MMDPuffeRL`, cache-free autocast), but it has only been verified on CPU. Validate it on the
   GPU against a float32 curve before relying on it.

Every override uses the form `--train.<key>` or `--env.<key>`; all keys are listed in
`pufferroyale/config/royale.ini`.

---

## 8. Phase 6: main experiments (the course research questions)

**Setting.** Start with a fixed deck and a fixed tower troop (default: `hog26` mirror, Tower
Princess, `frame_skip 10`). Generalise to random decks (`--deck0 random --deck1 random`) once
this works.

Give each configuration the **same learner-step budget** (e.g. 500M–1B, sized from Phase 3 and
the GPU-hour budget), run ≥ 2 seeds, and record GPU-hours.

| Id | Question | Command core (`league_train.py`) |
|---|---|---|
| A | naive self-play | `--anchors none --self-play-frac 1.0 --anchor-frac 0 --snapshot-interval 50` (snapshots are used only for evaluation) |
| B | PFSP league | `--anchors bot:heuristic,bot:random --self-play-frac 0.2 --anchor-frac 0.1 --pfsp hard --snapshot-interval 50 --max-snapshots 32` |
| C | league + MMD | B plus `--mmd-coef 0.05 --mmd-ref-interval 50` (also try a coefficient in {0.01, 0.1}) |

Common flags for all three: `--device cuda --num-envs <Phase 3> --batch-size <Phase 3>
--data-dir /scratch/$USER/pufferroyale --run-id <id>_s<seed> --seed <seed>`.

`hpc/league.sbatch` already exposes these as environment variables: `RUN_ID`, `TOTAL`,
`NUM_ENVS`, `ANCHORS`, `PFSP`, `SELF_PLAY_FRAC`, `ANCHOR_FRAC`, `SNAPSHOT_INTERVAL`,
`MAX_SNAPSHOTS`, `MMD_COEF`, `MMD_REF_INTERVAL`, `SEED`, `PROJECT`. Submit like this:

```bash
export PROJECT=/scratch/$USER/PufferRoyale TOTAL=500000000
RUN_ID=A_s0 SEED=0 ANCHORS=none SELF_PLAY_FRAC=1.0 ANCHOR_FRAC=0 sbatch hpc/league.sbatch
RUN_ID=B_s0 SEED=0 ANCHORS=bot:heuristic,bot:random SELF_PLAY_FRAC=0.2 ANCHOR_FRAC=0.1 sbatch hpc/league.sbatch
RUN_ID=C_s0 SEED=0 ANCHORS=bot:heuristic,bot:random SELF_PLAY_FRAC=0.2 ANCHOR_FRAC=0.1 \
    MMD_COEF=0.05 MMD_REF_INTERVAL=50 sbatch hpc/league.sbatch
```

Re-submitting the same `RUN_ID` with a larger `TOTAL` resumes the run.

As an extra reference point for A, you can also run symmetric self-play with
`train.py --num-agents 2`, where both seats learn.

---

## 9. Phase 7: evaluation (what goes in the report)

1. **Against the bots.** `scripts/eval.py --checkpoint <snap> --bots noop random heuristic
   --seats 0 1 --episodes 200 --json out.json`. Report W/D/L with 95% CIs.
2. **Exploitability** (research questions 1 and 2). For each final policy of A, B and C, run
   `scripts/best_response.py --target <snap> --total-timesteps <~10% of its training budget>
   --matches 200 --device cuda --out br.json`, with 2 seeds. Compare `br_score` and
   `exploitability_proxy` across A, B and C. Use the same BR budget for all.
3. **Meta-game.** Run `scripts/tournament.py --agents bot:heuristic bot:random <snapshots of A/B/C
   at 25/50/75/100%> --decks hog26 giant bait --matches 20 --device cuda --out t.json`. This gives
   the payoff matrix, Elo and the Nash mixture.
4. **Deck meta-game** (research question 3). Use `pufferroyale.metagame.deck_metagame(agent,
   decks, matches)` with the best policy over the 9 preset decks. It returns the deck-vs-deck
   score matrix and its Nash mixture.
5. **Qualitative check.** Watch a few games with `scripts/watch.py --checkpoint <snap>` (text
   board on the cluster) and note strategies. The LLM harness (`scripts/llm_match.py`) exists,
   but it needs network access and the user's approval before any real API use.

Evaluation uses the environment settings each checkpoint was trained with (frame_skip, tower
troops, lockout, tiebreak), read from the run's `config.json`. Keep evaluation settings fixed
across A, B and C.

---

## 10. Known issues and pitfalls

- **Use the project scripts, not `puffer train`.** PuffeRL's LSTM path reads `agents_per_batch`
  while `PufferEnv` names it `agent_per_batch`; our scripts and `LeagueVecEnv` handle this.
- **Old checkpoints don't load.** Checkpoints from before the v0.3 observation layout fail with a
  clear `ValueError`.
- **Checkpoint format.** Checkpoints are raw `state_dict`s plus `config.json`, and snapshot paths
  inside a league run are relative, so run directories can be moved.
- **The heuristic bot is simple** but currently stronger than any trained policy. Beating it (G3)
  is the first real milestone.
- **Approximate rules.** Some mechanics are documented approximations (see FIDELITY.md). Do not
  tune the engine to improve training numbers.
- **Data and terms.** Card statistics are Supercell's numbers, decoded by RoyaleSim
  (MIT-licensed); use them for non-commercial research only. Never automate the real game
  client: that breaks Supercell's terms of service.

---

## 11. Optional improvements (ranked by value)

- **O1. Logging and plots** (Phase 4, item 1). This is effectively required.
- **O2. `train.py --resume`** (Phase 4, item 2).
- **O3. Faster league env stepping.** Either run `LeagueVecEnv` over several worker processes,
  or add OpenMP to the native `vec_step`. Keep determinism tests passing.
- **O4. Smaller observations.** Store them as uint8/bf16 (mask and flags are binary; most
  features are in [0, 1]) to cut rollout memory about 4×. This changes the observation contract,
  so it needs a SPEC amendment and test updates, and it is a coordinated change.
- **O5. Validate bf16 on the GPU**, then optionally `torch.compile`.
- **O6. Protein hyperparameter sweeps** (`royale.ini [sweep]` already targets `learner_score`).
  The scripts don't wire PufferLib's sweep entry point yet; manual grids through job arrays are
  fine.
- **O7. Engine and card work.** This is a separate track (see STATUS.md "Suggested next
  steps"). It follows the SPEC process with independent tests. Do not mix it into training
  changes.

---

## 12. Report back to the user at each gate

- **G0:** environment verified, including the golden hashes, the test summary lines and the
  hardware.
- **Phase 3:** the throughput table and the chosen standard configurations.
- **G1–G3:** learning curves (plots), final scores, and the hyperparameters used.
- **Phase 6/7:** per-experiment results, exploitability, the tournament and meta-game outputs,
  GPU-hours spent, and the paths to all JSON outputs and checkpoints.
- **Anywhere:** every deviation from this guide and every issue found, logged in
  `docs/DECISIONS.md` and `docs/EXPERIMENTS.md`.
