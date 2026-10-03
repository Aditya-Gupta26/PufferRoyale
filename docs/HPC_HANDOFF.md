# PufferRoyale: HPC training handoff

**Who this is for:** an agent with shell access to the NYU HPC cluster (Torch, SLURM) and a
checkout of this repository.

**Your job:**
1. Build an **isolated** training environment on the cluster. Use the existing PufferDrive setup
   there as a read-only reference, and never touch it (§2).
2. Verify the simulator on the cluster.
3. Measure throughput.
4. Run the training stages of `docs/TRAINING_PLAN.md`.
5. Evaluate, and report.

Work through the phases in order: each one ends at a gate, and you report to the user at every
gate.

**Plan of record:** `docs/TRAINING_PLAN.md` decides *what* to run and why. This document says *how*.
If they ever disagree, the training plan wins; tell the user about the disagreement.

**Deadline:** the course deliverable is due in early December 2026. Favour reliable progress over
breadth.

**Notation**
- **The training work package is in `main`** (SPEC §19, v0.5-G … G.3): everything this guide uses
  exists, and the flag names match each script's `--help` (checked 2026-10-03).
- **`$PD_REF`** is the PufferDrive folder on the cluster that the user pointed to as the reference
  (they described it as "the EMERGE … /PufferDrive" folder).
- **`$PR_ROOT`** is our own isolated directory (§4b).

---

## 0. What PufferRoyale is (one screen)

- **The simulator.** A Clash Royale 1v1 battle simulator in header-only C99: integer-only, and
  deterministic at 20 ticks/s. It has 64 cards, 4 tower troops (Tower Princess default) and full
  match rules. It is exposed as the PufferLib 3.0 environment `pufferroyale.Royale`:
  - one decision every `frame_skip = 10` ticks (0.5 s);
  - action space `Discrete(2305)`: no-op, or 4 hand slots × 576 own-frame tiles (577 / 161 with
    `placement_grid` 2 / 4);
  - an own-frame observation of 17,715 float32 values, including your own deck and the exact
    legal-action mask;
  - zero-sum reward: +1 / −1 / 0 at match end, with optional potential-based antisymmetric shaping
    (tower HP, crowns, wasted elixir) that can anneal to 0;
  - optional deck sampling (pool, random fraction, held-out decks).
- **Policy:** card first, then position given the card (the "conditional" head, default), with
  card-stat features; the v0.4 "flat" head stays available.
- **Training tools** (PyTorch, PufferLib 3.0):
  - `scripts/league_train.py`: PPO on a league (self-play, frozen snapshots chosen by PFSP,
    optional scripted "anchors"), with the optional MMD regulariser. **Resumable.** All stages run
    through this script.
  - `scripts/stages.py`: the stage-1 bot ladder (chained `league_train.py` rungs with gates).
  - `scripts/train.py`: plain PuffeRL PPO (self-play or vs one bot), resumable.
  - `scripts/best_response.py`: the exploitability probe.
  - `scripts/tournament.py`: round robin, Elo, and the meta-game Nash.
  - `scripts/eval.py`: evaluation against the scripted bots (per deck with `--decks`, Wilson CIs).
  - `scripts/transitivity.py`, `scripts/plot_history.py`: run diagnostics and plots.
- **Status.** Everything was verified on an Apple M4 laptop (clean rebuild, 2026-10-03):
  - 1,455 tester spec tests and 416 builder tests;
  - 184,647 C checks, clean under ASan/UBSan;
  - golden hashes PASS;
  - a 21/21 end-to-end ladder.

  It has **never run on Linux or a GPU**. The only training so far:
  - a CPU learning check: 0.6M steps against the random bot, which it beats 0.94–0.98 (TRAINING_PLAN
    §7);
  - smoke runs.

  The built-in heuristic bot still beats every checkpoint.

### Read these first, in order
1. **`docs/TRAINING_PLAN.md`: the stages, the decisions and their reasons, the evaluation
   protocol.** Read §0, §1, §8 and §9 closely.
2. `docs/STATUS.md`: what exists, how it was verified, known approximations.
3. `README.md`: the command reference and the PufferLib install notes.
4. `hpc/README.md`, `hpc/pufferroyale.def`, `hpc/train.sbatch`, `hpc/league.sbatch`: templates
   written for this cluster but **never run**.
5. `docs/SPEC.md` §9 (env), §15 (league, MMD, evaluation tools), §18 (robustness) and §19 (the
   training work package, incl. §19.9–§19.10). Skim the rest.
6. `docs/DECISIONS.md` and `docs/FIDELITY.md`, to know what is deliberate.

---

## 1. Ground rules

- **Never compute on login nodes.** Build, test and train inside SLURM jobs (`srun` for
  interactive work, `sbatch` for runs).
- **Stay inside `$PR_ROOT`.** All our files live there (§4b): checkout, images, environments,
  caches, outputs. Never use `$HOME` for data.
- **PufferDrive is reference-only.** Follow §2 to the letter: read it, never modify, execute,
  mount or share anything of it.
- **Game logic is governed by `docs/SPEC.md`.**
  - Do not change engine behaviour (`pufferroyale/csrc/*`) to make training easier. If you believe
    a game rule is wrong, write it up and ask the user.
  - Do not implement work-package items yourself: they are being built on the Mac with independent
    tests. Your code changes are cluster-side only: `hpc/*`, new sbatch wrappers, `docs/EXPERIMENTS.md`
    and notes.
- **Keep the suite green.**
  - `pytest tests -q` and `python scripts/e2e_check.py` must pass on the cluster.
  - `tests/spec/` is an independent black-box suite: never edit it to make it pass; report
    instead.
  - Note that `tests/spec/test_hpc_templates.py` and `tests/builder/test_audit2_training.py` check
    `hpc/*.sbatch` (§6, pitfalls).
- **Git.**
  - Work on a branch `hpc-training`, with small commits and clear messages.
  - Merge `origin/main` when the user tells you about new commits.
  - Never commit secrets (wandb keys), checkpoints, run directories or `.sif` / overlay images.
    `.gitignore` covers `experiments/`, `build/` and `*.so`.
- **Write things down.**
  - Decisions go in `docs/DECISIONS.md` (append; next free id).
  - Every run and its result goes in a new `docs/EXPERIMENTS.md`: date, git commit, command, SLURM
    job id, GPU type, GPU-hours, outcome, output path.
  - Cluster facts you discover (partition names, GPU request syntax, paths; no secrets) go in an
    "Cluster setup notes" section of the same file.
- **Stop and ask the user** when:
  - you lack an account, partition or build permission;
  - anything would require touching `$PD_REF` or shared configuration (§2);
  - `scripts/golden_hashes.py` fails;
  - a test fails and you cannot explain why;
  - you want more concurrent GPUs, or a larger budget, than the user approved;
  - **before launching stage 3** (show the pre-registration, Phase 6);
  - before any network/API use for LLM matches;
  - you are tempted to change game rules or loosen a test.

---

## 2. The PufferDrive reference: use it, never touch it

The user has a working PufferDrive setup on the cluster (`$PD_REF`) and pointed to it as the
reference for setting up on this cluster. Whatever it contains is what to learn from, for example:
- a container or overlay pattern, and how the environment is created;
- SLURM headers: account and partition names, GPU request syntax;
- module loads and scratch layout.

**Learn from it. Do not use it, and do not change it.** If our setup needs the same thing, make
our own **new copy** and keep our setup completely isolated.

**The only source is the folder on the HPC.** Everything you learn about the PufferDrive setup must
come from `$PD_REF` itself, read on the cluster. This document deliberately makes no claims about
its contents. Do not use other copies, the public repository, or other descriptions of PufferDrive
as a substitute: read what is actually there and don't assume.

**One risk to check there (read-only):** whether the PufferDrive environment carries its own
`pufferlib` package.
- Ours must be **upstream PufferLib 3.0 at commit `3b5c604`**. Our `MMDPuffeRL` is a
  statement-for-statement copy of upstream's `PuffeRL.train()`, so any other `pufferlib` on the path
  would silently change training.
- The hash check in §4d enforces this in every job, whatever you find.

### Hard rules (R1–R10)
- **R1. Read-only access.** Use only commands that cannot write: `ls`, `cat`, `less`, `head`,
  `grep`, `find` (without `-delete`/`-exec`), `stat`, `diff`.
  - No `git` commands inside `$PD_REF`: even `git status` can rewrite the index.
  - No editors that create swap files there.
- **R2. Never execute anything from `$PD_REF`.**
  - Do not run its scripts or sbatch files, and do not `source` its env files.
  - Do not run Python, `pytest`, `make`, `setup.py`, `pip` or `uv` with it as the working directory
    or on `PYTHONPATH`. Running Python there writes `__pycache__`, and importing from there could
    load a different `pufferlib`.
- **R3. Never use its Python environment.** Do not activate, install into, or copy from its conda
  env, venv or uv environment, and do not use its caches.
- **R4. If it uses overlay images, never mount them, not even read-only.** Singularity/Apptainer
  overlays are locked while mounted, so our mount could stop its jobs from starting.
  - Never copy a possibly live overlay (the copy can be inconsistent).
  - If we use overlays, create a **fresh** one from the cluster's clean template into `$PR_ROOT`.
- **R5. Images: copy, then use the copy.** If it uses `.sif` images: these are read-only files. If we
  need the same base image, `cp` it into `$PR_ROOT/containers/` and use only the copy (per the user's
  instruction). Never build into, `--writable`-open, or otherwise touch images in `$PD_REF`'s
  paths.
- **R6. Nothing shared through the environment.** Our jobs source only our own `$PR_ROOT/env.sh`
  (§4b), which sets our own caches and temp dirs, `PYTHONNOUSERSITE=1` and an empty `PYTHONPATH`.
  Do not edit shared dotfiles (`~/.bashrc`, `~/.bash_profile`, `~/.condarc`, `~/.config/pip`,
  `~/.netrc`, `~/.gitconfig`). If one seems necessary, ask the user.
- **R7. SLURM etiquette.**
  - Every job name starts with `pr-` (the templates use `pr-train` / `pr-league`; override with
    `sbatch --job-name=pr-…`).
  - Cancel only by job id or with `scancel --name=pr-…`. **Never `scancel -u $USER`**: it cancels
    every job of the account, including PufferDrive jobs if they run under the same account (§3).
  - Never `scontrol update` / `hold` / `requeue` a job that isn't ours.
- **R8. Quotas and fairshare are shared.** Our files count against the same scratch quota (space
  and inodes), and our GPU use against the same fairshare.
  - Prune checkpoints and keep run sizes in check.
  - Never delete anything outside `$PR_ROOT`.
  - Respect the concurrent-GPU limit the user sets (§3).
- **R9. Record what you learn**, without secrets, in "Cluster setup notes": base image path, Python
  and CUDA versions, partition names, GPU syntax, module names. Reproduce it in our own scripts
  under `hpc/`.
- **R10. Verify isolation in every job** (the check in §4d). No interpreter or package path may lie
  inside `$PD_REF`, and `pufferlib` must be upstream 3.0 @ `3b5c604`.

### What to look for in `$PD_REF` (read-only)
| Look for | Why |
|---|---|
| `*.def`, `*.sif` paths, `singularity`/`apptainer exec` lines (`--nv`, `--overlay`, `--bind`) | The container pattern that works here, and the base image's path |
| Overlay creation (an ext3 template path, size) and what's installed inside (conda/uv, Python version, torch wheel and CUDA version) | Path B below |
| `#SBATCH` headers: `--account`, `--partition`, `--gres` / `--gpus` syntax for L40S and H200, `--cpus-per-task`, `--mem`, `--time`, `--requeue` | Fill our templates correctly |
| `module load …`, environment variables (TMPDIR, cache dirs, WANDB_*), scratch paths | Cluster conventions |
| Any workaround notes (build flags, driver/CUDA mismatches) | Avoid known problems |

---

## 3. Ask the user for these before starting

| Item | Why |
|---|---|
| **The absolute path of `$PD_REF`**, and whether it belongs to the same Unix account as yours | Read-only reference (§2); shared quota/fairshare if the same account |
| **How many GPUs we may use concurrently** (the user said up to 16 at best) without starving PufferDrive work, and on which partitions (L40S / H200) | Concurrency limit (R8) |
| SLURM account (is it the same allocation PufferDrive uses?) | Fill `#SBATCH`; fairshare |
| Apptainer/Singularity availability; whether `--fakeroot` builds are allowed, and where | Path A vs B (§4c) |
| Internet access from login and compute nodes (pip, git, wandb) | Environment build; online vs offline logging |
| Scratch path, quota (space and inodes), and purge policy | `$PR_ROOT` location; retention |
| Wall-time limits and preemption policy | Job length; resume cadence |
| wandb account/project (optional), and how the API key should be provided (env var from a file under `$PR_ROOT`, never `wandb login`, which writes `~/.netrc`) | Experiment tracking without touching shared dotfiles |

---

## 4. Phase 1: an isolated environment (Stage 0)

### 4a. Read the reference
Read `$PD_REF` per §2 and write down your findings ("Cluster setup notes"). Decide between Path A,
B or C (§4c) with the user if unsure.

### 4b. Our directory layout and environment script
Use one root for everything. The default is `/scratch/$USER/pufferroyale`; adjust it to the
cluster's scratch path.

```
$PR_ROOT/                          everything of ours, nothing else
  PufferRoyale/                    git checkout (branch hpc-training)  = $REPO
  containers/pufferroyale.sif      our image (Path A), or our COPY of a base image (Path B)
  overlays/                        our own overlay(s) (Path B only), fresh from the clean template
  envs/                            our venv (Path C only)
  cache/  tmp/  wandb/             our caches, temp files, wandb files
  logs/                            submit sbatch jobs from here, so %x_%j.out/.err land here, not in the checkout
  runs/                            all outputs (= DATA_DIR): league/, train/, best_response/, eval/
  secrets/                         optional, chmod 700 (e.g. wandb key file); never committed
  env.sh                           the script below
```

Every shell and job starts with `source /scratch/$USER/pufferroyale/env.sh`: a literal path,
because `$PR_ROOT` is only defined by that file. If you choose another root, change that path
everywhere.

```bash
# $PR_ROOT/env.sh -- PufferRoyale jobs only. Source it in our jobs; never from ~/.bashrc.
export PR_ROOT=/scratch/$USER/pufferroyale          # adjust to the cluster's scratch path
export REPO=$PR_ROOT/PufferRoyale PROJECT=$PR_ROOT/PufferRoyale
export SCRATCH_DIR=$PR_ROOT                          # the sbatch templates bind ONLY this directory
export SIF=$PR_ROOT/containers/pufferroyale.sif
export DATA_DIR=$PR_ROOT/runs
export PIP_CACHE_DIR=$PR_ROOT/cache/pip UV_CACHE_DIR=$PR_ROOT/cache/uv XDG_CACHE_HOME=$PR_ROOT/cache/xdg
export APPTAINER_CACHEDIR=$PR_ROOT/cache/apptainer SINGULARITY_CACHEDIR=$PR_ROOT/cache/apptainer
export APPTAINER_TMPDIR=$PR_ROOT/tmp TMPDIR=$PR_ROOT/tmp
export WANDB_DIR=$PR_ROOT/wandb WANDB_CACHE_DIR=$PR_ROOT/cache/wandb WANDB_CONFIG_DIR=$PR_ROOT/wandb/config
export PYTHONNOUSERSITE=1                            # ignore ~/.local site-packages (could hold another pufferlib)
unset PYTHONPATH
export PD_REF=/path/given/by/the/user                # used only by the isolation check (§4d)
```

How the templates use these variables:
- `hpc/train.sbatch` and `hpc/league.sbatch` take `SCRATCH_DIR`, `SIF`, `PROJECT` and `DATA_DIR`
  from the environment.
- They bind `$SCRATCH_DIR:$SCRATCH_DIR` into the container. With `SCRATCH_DIR=$PR_ROOT`, that is our
  directory only, rather than all of `/scratch/$USER`.
- **That is not full isolation.** By default Apptainer also mounts `$HOME`, `/tmp`, the current
  directory and any site-configured bind paths, and passes the host environment through.
  - Add `--no-home` to the `apptainer exec` calls (in the templates and `run()`) to keep `$HOME`
    out. Check that nothing breaks; the template tests accept it, since it comes before the script.
  - `PYTHONNOUSERSITE=1` (also set in the `.def`) already keeps `~/.local` packages out.
- With `PROJECT=$REPO`, jobs run your checkout instead of the copy baked into the image.

### 4c. Choose how to build the environment
**Required stack, whichever path you choose (all versions recorded in EXPERIMENTS.md):**
- Python **3.12.x**. Not 3.13: we need numpy < 2, which has no 3.13 wheels.
- numpy **1.26.x**.
- A CUDA build of PyTorch that matches the node driver (check with `nvidia-smi`).
- scipy, gymnasium, psutil, rich, rich_argparse, Cython, **pytest** (needed for G0), and
  matplotlib + wandb (plots and logging). `hpc/pufferroyale.def` already installs all of them.
- **Upstream PufferLib 3.0 at commit `3b5c604`**, built from source with `NO_OCEAN=1` after the
  one-line fix: add `c_extension_paths = []` right after `c_extensions = []` in its `setup.py`
  (DECISIONS D6).
- Our C extension built **in our checkout**: `python setup.py build_ext --inplace`, without
  `PR_RAYLIB`.
- Mac reference versions for comparison: Python 3.12.11, numpy 1.26.4, torch 2.14.0 (CPU build),
  scipy 1.17.1, gymnasium 1.3.0.

**Path A: our Apptainer definition.** Use this if `--fakeroot` builds are allowed.

```bash
source /scratch/$USER/pufferroyale/env.sh && mkdir -p $PR_ROOT/{containers,cache,tmp,runs,logs}
cd $REPO      # the %files paths are relative to the repository root
apptainer build --fakeroot $SIF hpc/pufferroyale.def
apptainer test $SIF
```

- The image contains:
  - CUDA 12.4 devel and Python 3.12;
  - torch from the cu124 wheel index (the version isn't pinned; record it);
  - PufferLib `3b5c604` with the fix, in `/opt/PufferLib`;
  - the package, in `/opt/pufferroyale`.
- The `.def`'s `TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;9.0"` covers the H200 (9.0). The L40S (8.9) runs
  the 8.6 binaries, since a newer minor version of the same major is compatible. Adding `8.9` is a
  harmless optional improvement.

**Path B: reproduce the `$PD_REF` recipe in isolation.** Use this when the cluster's pattern is
"NYU base image + overlay + conda/uv env".
- **Base image:** if it's one of the cluster's public images, use it directly from the public
  location (read-only), or `cp` it to `$PR_ROOT/containers/`. Never use a copy that lives in
  `$PD_REF`'s directories.
- **Overlay:** create a **new** one from the cluster's clean template (e.g. a gzipped ext3 template)
  in `$PR_ROOT/overlays/`. Mount it read-write only while installing, read-only (`:ro`) in training
  jobs, so many jobs can share it.
- **Inside:** install the required stack above. Write the recipe as a script in
  `hpc/` (e.g. `hpc/setup_overlay.sh`), document it in `hpc/README.md`, and commit it on
  `hpc-training`.

**Path C: a plain venv, no container.** Use this only if containers are unavailable.

```bash
source /scratch/$USER/pufferroyale/env.sh     # = $PR_ROOT/env.sh
module load …                                   # the cluster's Python 3.12 / CUDA 12 modules, or install Python 3.12 with uv
python3.12 -m venv $PR_ROOT/envs/venv && . $PR_ROOT/envs/venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124        # torch alone from the PyTorch index
pip install "numpy<2" scipy gymnasium psutil rich rich_argparse Cython pytest   # the rest from PyPI
git clone -b 3.0 https://github.com/PufferAI/PufferLib.git $PR_ROOT/PufferLib && cd $PR_ROOT/PufferLib && git checkout 3b5c604
sed -i 's/^c_extensions = \[\]$/c_extensions = []\nc_extension_paths = []/' setup.py && grep -q '^c_extension_paths = \[\]' setup.py
NO_OCEAN=1 pip install --no-build-isolation .
cd $REPO && python setup.py build_ext --inplace
```

### 4d. Get the code and check isolation
On a login node (git only):

```bash
source /scratch/$USER/pufferroyale/env.sh     # = $PR_ROOT/env.sh
git clone https://github.com/Aditya-Gupta26/PufferRoyale.git $REPO && cd $REPO && git checkout -b hpc-training
```

Inside a job, define a helper. For Path A, with `--pwd $REPO`:

```bash
run() { apptainer exec --nv --bind $PR_ROOT --pwd $REPO $SIF "$@"; }
run python setup.py build_ext --inplace         # once per code change
```

For Path B, add `--overlay $PR_ROOT/overlays/<ours>.ext3:ro` and activate our env inside. For
Path C, activate the venv instead.

**Isolation check.** Run it in every new environment and at the top of every job script:

```bash
run python - <<'EOF'
import hashlib, os, sys
import numpy, torch, pufferlib, pufferlib.pufferl, pufferroyale
ref = os.path.realpath(os.environ.get("PD_REF", "/nonexistent"))
paths = {"python": sys.executable, "pufferlib": pufferlib.__file__, "pufferroyale": pufferroyale.__file__}
print(paths)
print("numpy", numpy.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
assert not any(os.path.realpath(p).startswith(ref + os.sep) for p in paths.values()), "a PufferDrive path is in use"
assert numpy.__version__.startswith("1."), "numpy must be < 2"
h = hashlib.sha256(open(pufferlib.pufferl.__file__, "rb").read()).hexdigest()
assert h == "ebe89e96a2515e7b0370d7b4833ae64bb46c1a68c20d13fd014a10abc4e33323", f"pufferl.py is not upstream 3b5c604: {h}"
print("isolation OK")
EOF
```

The hash is upstream `pufferlib/pufferl.py` at `3b5c604`. It's identical to the Mac install (both
checked 2026-10-02 against the upstream PufferLib repository). `PD_REF` reaches the check inside
the container because Apptainer passes the whole host environment through, unless `--cleanenv` or
`--containall` is used.

---

## 5. Phase 2: verify on a GPU node (gate G0)

Open an interactive GPU session, e.g.:

```bash
srun --account=… --partition=… --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=02:00:00 \
     --job-name=pr-interactive --pty bash
```

Then run the following with the `run` helper:

```bash
run make test-c                         # C unit/scenario tests, expect "0 failures" everywhere
run make asan                           # C tests under ASan+UBSan, expect clean
run python -m pytest tests -q           # ~5-10 min, expect all passed
run python scripts/golden_hashes.py     # cross-platform determinism, MUST print "golden hashes: PASS"
run python scripts/e2e_check.py         # expect the OVERALL line to say PASS and "21/21 checks passed"
# GPU smoke runs (tiny; expect finite losses)
run python scripts/train.py --device cuda --total-timesteps 300000 --num-envs 64 \
    --batch-size 16384 --minibatch-size 4096 --quiet --summary-json $DATA_DIR/smoke_train.json
run python scripts/league_train.py --device cuda --total-timesteps 200000 --num-envs 64 \
    --anchors bot:heuristic --snapshot-interval 2 --data-dir $DATA_DIR --run-id smoke
```

What to expect and how to handle failures:
- **Golden hashes.** `tests/golden/golden_hashes.json` holds per-100-tick state hashes of six seeded
  bot matches, generated on macOS arm64. The engine is integer-only with zeroed padding, so Linux
  must reproduce them bit for bit. If it doesn't, **stop and report** the first diverging match and
  tick. Do not regenerate the reference.
- **Performance tests.** Their thresholds were set on an Apple M4: engine ≥ 20k ticks/s, env ≥ 2k
  steps/s, stress ≥ 10k ticks/s. Server cores should pass. If they don't, report the measured
  numbers; never lower a threshold quietly.
- **`make asan-py` on Linux with clang.** The Makefile asks for `libasan.so`, the gcc runtime's
  name; clang's runtime is `libclang_rt.asan-x86_64.so`. If that target fails, fix the runtime
  lookup in the Makefile. This affects tooling only.
- **First Linux build warnings or errors.** The C has been compiled only on macOS (clang and
  gcc-15). Fix genuine portability issues without changing behaviour; the golden hashes prove you
  didn't.

**G0 passes when** all of the above pass and the isolation check prints `isolation OK`. Report:
- GPU type, driver, CUDA and torch versions;
- the path chosen (A/B/C);
- the test summary lines.

---

## 6. Phase 3: throughput calibration (still Stage 0; current `main`)

Find the fastest stable configuration per GPU type before any long run. For each, measure:
- learner steps/s (SPS);
- GPU utilisation and memory;
- CPU utilisation (`nproc` inside the job);
- wall time per epoch.

**What to measure**
1. **`league_train.py --device cuda`**, the engine of every stage:
   - `--num-envs` ∈ {64, 128, 256, 512}, with `--batch-size` = `num_envs × 64 × k` for k ∈ {1, 2, 4}
     and `--minibatch-size` ∈ {4096, 8192, 16384};
   - **stage-1 shape:** one bot anchor (`--anchors bot:heuristic --self-play-frac 0
     --anchor-frac 1.0`);
   - **stage-3 A shape:** `--anchors none --self-play-frac 1.0 --anchor-frac 0`;
   - **stage-3 B/C shape with 16, 32 and 64 frozen opponents.** Make that many checkpoint files with
     a short run (`--snapshot-interval 1`, tiny batch), then pass them as anchors:
     `--anchors ckpt:<f1>,ckpt:<f2>,… --anchor-frac 0.65 --self-play-frac 0.35`. That has the same
     number of distinct opponents per step as a B run with that many snapshots;
   - with and without `--rnn`;
   - **1, 2 and 4 league processes sharing one GPU** (the env stepping is single-process and
     CPU-bound, so packing runs per GPU may raise total throughput).
2. **`train.py --device cuda`** for reference: `--backend PufferEnv` vs
   `--backend Multiprocessing --workers {4, 8}`.

**Facts to keep in mind**
- **Memory.** One observation row is 17,707 float32 = 70.8 KB, so the rollout buffer is about
  `batch_size × 70.8 KB` (65,536 rows ≈ 4.6 GB) on top of the model.
- **Env speed.** The C env runs about 26k steps/s per CPU core (STATUS.md).
- **League stepping is one process.** `LeagueVecEnv` steps all matches in **one** process:
  - bot opponents act in C, one call per match;
  - each distinct policy opponent costs one batched GPU forward per step.

  So B/C runs with many snapshots are slower than A at equal steps. That's expected; the stage-3
  comparison is at equal **learner steps** (TRAINING_PLAN §8.3).
- **Batch arithmetic.** `league_train.py` requires:
  - `bptt_horizon` (64) divides both batch and minibatch;
  - minibatch ≤ batch;
  - `num_envs ≤ batch / bptt_horizon`.

  It also sets `max_minibatch_size` to the minibatch itself.

**Then**
- Write a table in `docs/EXPERIMENTS.md` (GPU type × config → SPS, memory, CPU %) and pick one
  standard configuration per GPU type.
- Set the template defaults to match: `NUM_ENVS` and `--cpus-per-task` in both sbatch files,
  `BATCH_SIZE`/`MINIBATCH_SIZE` in `train.sbatch`, and the new `BATCH_SIZE`/`MINIBATCH_SIZE` in
  `league.sbatch` (§6a item 2).

### 6a. Cluster-side template changes (yours to make; keep the template tests green)
1. **`EXTRA_ARGS` passthrough** in `hpc/league.sbatch` and `hpc/train.sbatch`.
   - **Why:** `--env.*`, `--train.*` and other extra flags must not be written literally in the
     sbatch files. `tests/spec/test_hpc_templates.py` checks every literal `--flag` after a
     `scripts/*.py` call against that script's `--help`, and the `--section.key` overrides are not
     listed there.
   - **How:** read it as an array, `read -r -a EXTRA <<< "${EXTRA_ARGS:-}"`, and append
     `${EXTRA[@]+"${EXTRA[@]}"}` to both the fresh and the resume invocation. That form is safe under
     the templates' `set -u` even when empty, including on old bash (checked with bash 3.2).
2. **`league.sbatch` variables and defaults.**
   - Add `BATCH_SIZE` / `MINIBATCH_SIZE` variables, passed as `--batch-size` / `--minibatch-size`
     on the **fresh-run** invocation only. A resumed run keeps its saved values. These flags are in
     `league_train.py --help`, so the template test allows them literally.
   - **Resume branch:** today it re-passes `--num-envs "$NUM_ENVS"` and `--snapshot-interval
     "$SNAPSHOT_INTERVAL"`. These are not pool keys, so the command-line values *override* the saved
     run's. A re-submit that forgets them silently switches to the defaults (256 envs, interval 50),
     changing a pre-registered snapshot cadence or failing the batch-size check. Pass them on resume
     only when the submitter set them explicitly, so the saved values are kept otherwise.
   - **Defaults:** change them to the stage-3 style (`ANCHORS=none`, `ANCHOR_FRAC=0`), or make them
     mandatory. The current defaults (`bot:heuristic,bot:random`, 0.3 self, 0.2 anchors) contradict
     TRAINING_PLAN T10.
3. **Add `--sample` to the end-of-job `tournament.py` call.** Evaluations sample actions
   (TRAINING_PLAN §9); `--sample` is in its `--help`, so the test allows it.
4. **Keep these literal lines** in `league.sbatch`, which `tests/builder/test_audit2_training.py`
   asserts: `--decks "${DECKS[@]}"` and `if [ "$DECK1" != "$DECK0" ]`.
5. **New wrappers.** Add a `hpc/br.sbatch` for best-response probes and a `hpc/eval.sbatch` for
   tournaments and evaluation. They must pass `bash -n` and use only advertised flags. The template
   tests check only the two existing sbatch files (a hard-coded list), so run `bash -n` and the
   `--help` checks on the new ones yourself.
6. **Isolation and naming.** Source the env script and run the isolation check (§4d) at the top of
   every job. Use `pr-` job names. Submit from `$PR_ROOT/logs` (`cd $PR_ROOT/logs && sbatch
   $REPO/hpc/league.sbatch`): the templates don't depend on the submission directory.
7. **Check.** Run `pytest tests/spec/test_hpc_templates.py tests/builder/test_audit2_training.py`
   after every template edit.

### 6b. Early learning smoke (a real learning signal on the GPU before stage 1)

```bash
run python scripts/league_train.py --device cuda --num-envs $NE --batch-size $BS --minibatch-size $MB \
    --anchors bot:random --self-play-frac 0 --anchor-frac 1.0 --snapshot-interval 50 \
    --deck0 hog26 --deck1 hog26 --total-timesteps 50000000 --seed 0 \
    --data-dir $DATA_DIR --run-id smoke_random_s0 \
    --env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0
```

- **Expectation:** the learner's score vs `bot:random` in `history.jsonl` climbs from the untrained
  baseline of about **0.25–0.45** (conditional head) to ≥ 0.9.
  - **Reference (Mac CPU, 2026-10-03, TRAINING_PLAN §7):** with this shaping it passed 0.9 at about
    0.4M learner steps, and without shaping at about 0.6M steps.
  - **Don't use `bot:noop` for this:** an untrained policy already scores 1.00 against it (measured
    on the Mac, 200 matches), so it shows nothing.
- **Report** the curve (score vs steps). If it doesn't climb, that's a finding to debug before
  stage 1; report it.

**Gate check snippet** (the learner's score over the last N matches vs one opponent, from a run's
`history.jsonl`):

```bash
run python - $DATA_DIR/league/smoke_random_s0/history.jsonl bot:random 2000 <<'EOF'
import json, sys
path, spec, need = sys.argv[1], sys.argv[2], int(sys.argv[3])
games = score = 0.0
for line in reversed(open(path).read().splitlines()):
    r = json.loads(line).get("results", {}).get(spec)
    if r:
        games += r["games"]; score += r["score"] * r["games"]
    if games >= need:
        break
print(f"{spec}: score {score / max(games, 1):.3f} over the last {int(games)} matches")
EOF
```

---

## 7. Phase 4: (no longer needed)

The work package this phase used to wait for is already in `main`. If the user later pushes new
commits: merge `origin/main` into `hpc-training`, rebuild the extension, re-run G0, and re-measure
throughput if the policy or the environment changed.

---

## 8. Phase 5: Stage 1, the bot ladder (calibration only; gates G1–G3)

TRAINING_PLAN §8.1 has the reasoning, configs and decision rule. Its checkpoints are **never** used
to start later stages.

**How it trains:** single-agent PPO against a fixed scripted bot, as a `league_train.py` run with
one bot anchor (pool = that bot only). `scripts/stages.py` runs the whole ladder: one rung per bot,
each started from the previous rung's final model (`--init-from`), each stopping early once its gate
is met, and the ladder stopping at the first failed gate.

**Main configuration** (seed `s`):

```bash
run python scripts/stages.py --run-prefix S1main_s$s --data-dir $DATA_DIR --device cuda \
    --num-envs $NE --batch-size $BS --minibatch-size $MB --snapshot-interval 50 --seed $s \
    --deck0 hog26 --deck1 hog26 \
    --env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0
```

- **Default rungs** (`--rungs`, anchor:budget:threshold):
  `bot:noop:5000000:0.95;bot:random:50000000:0.90;bot:heuristic:300000000:0.55`. 0.55 is how
  "clearly > 0.5" is operationalised; report whether the score is still rising.
- **Outputs:**
  - rung runs at `$DATA_DIR/league/<prefix>_<i>_<bot>/`;
  - a summary at `$DATA_DIR/stages_<prefix>.json` (per rung: steps, score, matches, passed);
  - exit status 0 when every gate passed, 1 when a gate failed.
- **Gates** use the learner's last `--gate-window` (default 2000) outcomes vs the anchor, as saved in
  the rung's `league_state.json`. Fewer outcomes fail the gate.
- **Preemption is safe:** a rung whose directory already holds a run is resumed. On `--resume`,
  `history.jsonl` is cut back to the last save.
- **`--snapshot-interval` must be > 0.** The league saves (and so can resume) only at snapshot
  epochs.
- **SLURM:** wrap `stages.py` in a small sbatch (as in §6a item 5), or run each rung through
  `league.sbatch`.
- **Confirmation:** after each gate, check the score with
  `eval.py --checkpoint <rung run dir> --bots <bot> --seats 0 1 --episodes 500` (sampled actions).

| Gate | Opponent | Untrained baseline | Pass | Budget guide |
|---|---|---|---|---|
| G1 | `bot:noop` | 1.00 | score ≥ 0.95 | ≤ 5M learner steps. **Sanity only** (catches a broken head) |
| G2 | `bot:random` | 0.43 | score ≥ 0.90 | ≤ 50M. **The first real learning gate** |
| G3 | `bot:heuristic` | 0.00 | ≥ 0.55 and rising | 100–300M |

The baselines are for the default conditional head (50 matches per seat, 2026-10-03); the v0.4 flat
head starts at 1.00 / 0.15 / 0.00. They were measured on the Mac with `eval.py` and **no
`--checkpoint`** (a random-init policy). League runs write no step-0 checkpoint.

**Ablations.** Run them all in parallel, each a full ladder from scratch with 2 seeds, within the
GPU limit the user approved. TRAINING_PLAN §8.1 has the table:

| Id | Change from main |
|---|---|
| S1-noshape | no reward flags (shaping off) |
| S1-grid2 / S1-grid4 | `--env.placement-grid 2` / `4` |
| S1-lstm | `--rnn` |
| S1-flat | `--policy.head flat` |
| S1-lr | `--learning-rate 1e-4` and `1e-3` |
| S1-ent | `--train.ent-coef-card 0.01 --train.ent-coef-pos 0.002` |

**If G2 stalls (or G1 fails), debug before scaling:**
1. Watch the policy: `scripts/watch.py --checkpoint <run dir> --p1 random --deck1 hog26` (`--deck1`
   defaults to `giant`).
2. Check the illegal-action rate and the per-card play rates (`cards/play_rate/<CARD>` in the logs).
3. Check `learning_rate` (3e-4), `ent_coef` (0.01), `gamma` (0.999), `gae_lambda` (0.95),
   `bptt_horizon` (64), batch size, and `--rnn`.
4. Report to the user. Imitation learning (BC) from the heuristic bot is a **stage-1-only**
   fallback, and needs the user's go-ahead.

**Output (report at the gate):**
- learning curves;
- steps to G2;
- heuristic score at 100M / 200M / 300M steps for every configuration;
- the chosen configuration (decision rule in TRAINING_PLAN §8.1), recorded in `docs/DECISIONS.md`
  and frozen for stages 3–4.

---

## 9. Phase 6: Stage 3, the course comparison (training)

TRAINING_PLAN §8.3 has the reasoning; this is the procedure.

**Before launch (gate G-S3; needs the user's OK):**
- Write the **pre-registration** into `docs/EXPERIMENTS.md`:
  - learner-step budget `T` (default 1B; 500M if throughput forces it);
  - seeds 0, 1, 2;
  - `--snapshot-interval` and `--max-snapshots`;
  - the frozen stage-1 settings;
  - the shaping schedule;
  - the BR protocol (§10 E1);
  - the evaluation match counts;
  - hypotheses H1/H2.
- Record the git commit. That's the **engine freeze**: no gameplay-affecting change during stage 3.
- Show it to the user.

**Snapshot sizing.**
- E = `T / batch_size` epochs.
- `--snapshot-interval` ≈ E / 128, `--max-snapshots 64`, so the pool spans the last ~50% of
  training. Use 32 and E / 64 if Phase-3 SPS with 64 opponents is poor.
- Use the same interval for A (A saves snapshots for evaluation but never samples them).

**Runs.** 9 main runs plus 2 MMD-sensitivity runs. **No scripted bots anywhere** (TRAINING_PLAN
T10–T11). Random init; the same seed gives the same initial weights in A, B and C.

```bash
source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs     # needs the §6a template changes
SIZES="TOTAL=$T NUM_ENVS=$NE BATCH_SIZE=$BS MINIBATCH_SIZE=$MB SNAPSHOT_INTERVAL=$I MAX_SNAPSHOTS=64"
COMMON="<frozen stage-1 flags, e.g. --rnn / --learning-rate> \
  --env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0 \
  --shaping-anneal-frac 0.5"                                   # drop all shaping flags if stage 1 rejected shaping
for s in 0 1 2; do
  env $SIZES RUN_ID=A_s$s SEED=$s ANCHORS=none SELF_PLAY_FRAC=1.0 ANCHOR_FRAC=0 EXTRA_ARGS="$COMMON" \
    sbatch --job-name=pr-A_s$s $REPO/hpc/league.sbatch
  env $SIZES RUN_ID=B_s$s SEED=$s ANCHORS=none SELF_PLAY_FRAC=0.35 ANCHOR_FRAC=0 PFSP=hard EXTRA_ARGS="$COMMON" \
    sbatch --job-name=pr-B_s$s $REPO/hpc/league.sbatch
  env $SIZES RUN_ID=C_s$s SEED=$s ANCHORS=none SELF_PLAY_FRAC=0.35 ANCHOR_FRAC=0 PFSP=hard \
    MMD_COEF=0.05 MMD_REF_INTERVAL=50 EXTRA_ARGS="$COMMON" sbatch --job-name=pr-C_s$s $REPO/hpc/league.sbatch
done
# MMD sensitivity, seed 0 only: as C with MMD_COEF=0.01 (RUN_ID=Clo_s0) and MMD_COEF=0.1 (RUN_ID=Chi_s0)
```

**Keeping runs alive**
- `league.sbatch` resumes automatically when `league_state.json` exists.
- **To continue a run, re-submit it with exactly the same variables:** `RUN_ID`, `TOTAL` (the
  absolute target), `NUM_ENVS`, `SNAPSHOT_INTERVAL` and `EXTRA_ARGS`. On today's template the
  resume branch re-passes `--num-envs` and `--snapshot-interval`, and they override the saved run
  (§6a item 2).
- Keep each run's exact submit line in `docs/EXPERIMENTS.md` so it can be repeated verbatim.
- **The anneal length is fixed when a run is created** (SPEC §19.10.1). Extending `TOTAL` later
  does not restart the shaping; it is recomputed only if `--shaping-anneal-frac` is given again.
- SLURM requeues on preemption (`--requeue`), but a **wall-time kill is not requeued**: re-submit.
- Pool settings and `--rnn` always come from the saved run.
- A non-finite epoch is never saved; the job exits 1 with `nonfinite.json`. Report it, and resume
  with a lower learning rate only with the user's OK (log it as a deviation).

**Monitor:**
- `history.jsonl`: per-epoch losses, the learner's score per opponent, pool size, and snapshots;
  `mmd_kl` for C;
- `[league] …` progress lines;
- `--wandb` (offline: `WANDB_MODE=offline`, then `wandb sync`) and
  `scripts/plot_history.py <run dir> --out curve.png`.
- Check A's pool stays `self` only. For B/C, `pool_size` in `history.jsonl` grows to
  `MAX_SNAPSHOTS + 1`, because it counts `self` too.

---

## 10. Phase 7: Stage 3 evaluation (what goes in the report; gate G-E)

**Rules (TRAINING_PLAN §9)**
- **Sampled actions everywhere:** every tool samples by default (SPEC §19.11): `tournament.py`
  (`--sample` is now a no-op), `eval.py` and `best_response.py` without `--greedy`.
- **Both seats.**
- **95% CIs:** Wilson for proportions; bootstrap over seeds and matches for method-level numbers.
- **Training settings:** each checkpoint's own (read from `config.json`), identical across A, B and
  C.
- **"Final policy"** = `<run dir>/model_<epoch>.pt`, written when the run reaches `T`.

**E1. Exploitability (headline).**
1. **Calibrate the BR budget** first, on `A_s0`'s final policy: 50M, 150M and 400M steps. Pick the
   smallest budget where `br_score` gains < 0.02 at the next level. Record it.
2. **Probe.** For each of the 9 final policies, with BR seeds 0 and 1:

   ```bash
   run python scripts/best_response.py --target $DATA_DIR/league/<run> --device cuda \
       --num-envs $NE --total-timesteps $BR --matches 1000 --seed <0|1> \
       --data-dir $DATA_DIR/br/<run>_s<seed> --out $DATA_DIR/eval/br_<run>_s<seed>.json \
       --init-from-target <same shaping flags as training>
   ```

   - The target may be a run directory; it then resolves to its newest `model_*.pt`. With
     `--init-from-target` the architecture, grid and LSTM come from the target.
   - `--num-envs` defaults to only 8, so raise it on a GPU.
   - Also run a scratch-start BR (without `--init-from-target`) on 3 targets (one per method) as a
     robustness check.
3. **Report** `br_score`, `exploitability_proxy = 2·br_score − 1` and W/D/L per target, aggregated
   per method.

**E2. Head-to-head round robin.** The 9 final policies plus each run's 50% snapshot:

```bash
run python scripts/tournament.py --agents <18 checkpoint paths> --decks hog26 --deck-mode mirror \
    --matches 200 --sample --device cuda --seed 0 --out $DATA_DIR/eval/stage3_roundrobin.json
```

- **Output:** the payoff matrix, Elo and the Nash mixture over policies.
- **Cost:** 153 pairs × 200 = 30,600 matches, played sequentially in one process. On the Mac CPU a
  match took ~0.5 s with small checkpoints, i.e. ~4–5 h here. Measure on the cluster first; reduce
  to 100 matches per pair if needed, and record that.

**E3. Unseen scripted bots.** Per final policy:

```bash
run python scripts/eval.py --checkpoint $DATA_DIR/league/<run> --bots noop random heuristic \
    --seats 0 1 --episodes 500 --device cuda --json $DATA_DIR/eval/bots_<run>.json
```

- Pass the run directory, or any `model_*.pt` / `snap_*.pt` file (recurrent or not). JSON rows carry
  `wins`, `draws`, `losses`, `score` and the Wilson `ci95`.

**E4. Within-run transitivity.** For each run:

```bash
run python scripts/transitivity.py --run $DATA_DIR/league/<run> --snapshots 10 --matches 100 \
    --device cuda --out $DATA_DIR/eval/transitivity_<run>.json
```

It reports the payoff matrix, Elo, the Nash mixture, the fraction of (later, earlier) pairs the later
snapshot wins, and the number of cyclic triads.

**E5. Qualitative.**
- Watch a few games: `scripts/watch.py --checkpoint <run dir> --p1 heuristic --deck1 hog26` (text
  board; `--deck1` defaults to `giant`).
- Note strategies, per-card play rates, and degenerate behaviour.

**Report at G-E:**
- every JSON path;
- per-method exploitability with CIs (the headline);
- the round-robin matrix, Elo and Nash;
- bot scores;
- transitivity plots;
- GPU-hours and wall-clock per run.

---

## 11. Phase 8: Stage 4, the general policy

TRAINING_PLAN §8.4 has the reasoning; this is the procedure.
- **Method:** the stage-3 winner (E1 first, then E2).
- **Start:** `--init-from <the winning method's best final model>` (a new run).
- **Decks** (pass them in `EXTRA_ARGS`; `config.json` records the expanded deck lists):

  ```bash
  --env.deck-pool "hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge" --env.random-deck-frac 0.5 \
  --env.heldout-decks "xbow;royal_hogs;random:50:20261003" --env.deck-draw independent
  ```

  Pick the held-out seed once, record it in the pre-registration, and reuse the same string for
  evaluation.
- **No anchors** by default.
- **Budget:** 1–2B learner steps; 2 seeds if the GPU limit allows.

**Evaluation**
1. **Generalisation.** `eval.py` against `random` and `heuristic`, both seats, comparing pool decks
   with held-out decks. Each deck is played as a mirror; the JSON has per-deck rows plus pooled rows.
   - **Presets:** 500 matches per seat each, e.g.
     `eval.py --checkpoint <run> --bots random heuristic --seats 0 1 --episodes 500 --decks "xbow;royal_hogs"`.
   - **The 50 random held-out decks:** `--episodes 10 --decks "random:50:20261003"`, giving 500 per
     seat pooled.
   - **Pool decks:** evaluated the same way, for the comparison.
2. **Deck meta-game** (RQ3):

   ```python
   from pufferroyale.metagame import deck_metagame
   decks = ["hog26", "giant", "bait", "golem", "lavaloon", "xbow", "miner_poison", "pekka_bridge", "royal_hogs"]
   P, x = deck_metagame("<run dir or model .pt>", decks, matches=200, greedy=False, device="cuda")
   # P: deck-vs-deck score matrix (antisymmetric, 0.5 on the diagonal); x: its Nash mixture over decks
   ```

   That's 36 deck pairs × 200 = 7,200 matches, ~1–1.5 h at the ~0.5–0.65 s per match measured on the
   Mac. Save `P` and `x` as JSON.
3. **Optional:**
   - a BR probe on a held-out-deck mirror;
   - stage-3 policies played with other decks;
   - `--policy.card-stats 0` as a short ablation.

---

## 12. Phase 9: Stage 5, stretch (ask the user first)

Options:
- tower-troop variety (`tower_troop0/1` = `cannoneer`, `dagger_duchess`, `royal_chef`);
- deck-level PSRO (TRAINING_PLAN §8.5);
- LLM matches (`scripts/llm_match.py`: needs network, API keys and the user's explicit approval);
- fidelity checks.

---

## 13. Known issues and pitfalls

- **Use the project scripts, not `puffer train`.** PuffeRL's LSTM path reads `agents_per_batch`,
  while `PufferEnv` names it `agent_per_batch`; our scripts and `LeagueVecEnv` handle this.
- **Shaping needs `--train.reward-clip 0`.** PuffeRL clamps every step's reward to [−1, 1] by
  default (`reward_clip = 1`), and shaped terminal rewards exceed 1 (e.g. about 1.4 on a King kill).
  The scripts print a warning when shaping is on with the clamp; zero-sum is preserved either way,
  but exact potential-based shaping needs the clamp off.
- **wandb:** `--wandb [--wandb-project P --wandb-group G]` uses our own logger, which works with
  recent wandb. `global_step` is the step metric, so re-done epochs after a resume are kept.
- **Concurrent best-response probes need separate `--data-dir`s.** `best_response.py` names its
  run directory by a timestamp (to the second) plus the seed, so two probes started together with
  the same seed and data dir could overwrite each other's `br.pt`. Use one data dir per probe, as
  in §10.
- **Greedy is opt-in and card-first (SPEC §19.11).** `tournament.py`, `watch.py`, `play_match`
  and `deck_metagame` sample by default. `--greedy` / `greedy=True` picks the most likely of wait
  and the four card slots, then that slot's best tile, never the plain argmax (which nearly
  always waits). Reported numbers use sampling.
- **League saves happen only at snapshot epochs.** `--snapshot-interval 0` means no save until the
  run finishes, so no resume after a crash.
- **Pool settings are fixed per run.** `--anchors`, `--pfsp`, the fractions, `--max-snapshots` and
  `--rnn` cannot change on `--resume` (a different architecture or grid exits with a clear
  message). Start a new run with `--init-from` instead.
- **No LR annealing in the league by default.** `league_train.py` defaults to no learning-rate
  annealing (`--anneal-lr` turns it on), while `train.py` reads `anneal_lr = True` from
  `royale.ini`. Use what stage 1 decides, consistently.
- **`best_response.py --num-envs` defaults to 8:** raise it on GPUs.
- **The `league.sbatch` defaults don't match stage 3** (§6a): set `ANCHORS`, `SELF_PLAY_FRAC` and
  `ANCHOR_FRAC` explicitly in every submission.
- **No literal `--section.key` flags in sbatch files.** `test_hpc_templates.py` requires every
  literal flag to appear in the script's `--help`. Pass them through `EXTRA_ARGS`.
- **Old checkpoints don't load** across observation or policy changes: anything written before
  v0.5 (the work package) raises a clear `ValueError`.
- **GPU memory with the conditional head.** It stores about 0.94 MB of activations per training row
  in float32 (the flat head about 0.42 MB): roughly 15 GB at minibatch 16,384, before the rollout
  buffer (70.9 KB per row) and the MMD reference forward. Measure `torch.cuda.max_memory_allocated()`
  in the GPU smoke runs before choosing the minibatch.
- **CPU cost of the conditional head.** On the Mac CPU it trains at about half the flat head's speed
  (490 vs 1,020 steps/s with `train.py` defaults). Measure on the GPU (Phase 3) and plan budgets from
  that.
- **`make asan` takes about 30 s on an idle M4,** but up to about 16 minutes on a loaded machine
  (the work package's C test has 168k checks). Run it in the background.
- **The e2e `perf` stress threshold** (10,000 ticks/s) is borderline on the M4 Mac. It occasionally
  fails there, even on the pre-work-package build. Server cores should pass; if not, report the
  numbers.
- **Checkpoint format.** Checkpoints are raw `state_dict`s plus `config.json`. League snapshot
  paths are relative, so run directories can be moved.
- **The heuristic bot is simple** but currently stronger than any trained policy. Beating it (G3)
  is the first real milestone.
- **Approximate rules.** Some mechanics are documented approximations (FIDELITY.md). Never tune the
  engine to improve training numbers.
- **Data and terms.** The card statistics are Supercell's numbers, decoded by RoyaleSim (MIT); use
  them for non-commercial research only. Never automate the real game client: that breaks
  Supercell's terms of service.

---

## 14. Optional improvements (ranked)

- **O1. Faster league stepping.** Run `LeagueVecEnv` over several worker processes, or add OpenMP to
  the native `vec_step`, and batch bot calls in C. Keep the determinism tests passing. Worth it if
  Phase 3 shows the league is CPU-bound.
- **O2. Parallel tournaments.** Split pairs across jobs and merge the result lists into one
  payoff matrix with `pufferroyale.metagame.payoff_matrix`.
- **O3. Smaller observations** (uint8/bf16 storage; about 4× less rollout memory). This changes
  the observation contract: SPEC amendment and tests first, coordinated with the user.
- **O4. Validate bf16 on GPU** against a float32 curve, then optionally `torch.compile`.
- **O5. Protein sweeps** (`royale.ini [sweep]` targets `learner_score`). Manual grids through job
  arrays are fine.
- **O6. Engine and card work** is a separate track with the SPEC process. Don't mix it into
  training changes.

---

## 15. Report back to the user at each gate

| Gate | Report |
|---|---|
| **G0** | Environment path (A/B/C), isolation check, golden hashes, test summary lines, hardware/driver/CUDA/torch |
| **Phase 3** | Throughput table, chosen standard configs, template changes (§6a), early learning smoke curve (§6b) |
| **G1–G3** | Learning curves, gate results, ablation table, chosen stage-1 configuration |
| **G-S3** | The pre-registration (wait for the user's OK) |
| **G-E** | Exploitability per method with CIs, round robin, bot scores, transitivity, GPU-hours |
| **Stage 4** | Generalisation numbers, deck meta-game matrix and Nash |
| **Anywhere** | Every deviation and issue, logged in `docs/DECISIONS.md` and `docs/EXPERIMENTS.md` |
