# PufferRoyale on the NYU Torch cluster

Cluster-side files for training on NYU Torch (Slurm + Apptainer). The procedure is
`docs/HPC_HANDOFF.md`; the plan is `docs/TRAINING_PLAN.md`; what was learned about the cluster is in
`docs/EXPERIMENTS.md` ("Cluster setup notes"); the decisions are DECISIONS D79+.

## Files

| file | what it does |
|---|---|
| `env.sh` | the environment of every PufferRoyale shell and job; installed as `$PR_ROOT/env.sh` |
| `lib.sh` | `run` (our image, only `$PR_ROOT` bound, `--no-home`), the isolation check, the extension build-on-demand, the GPU heartbeat |
| `isolation_check.py` | HANDOFF §4d: no `$PD_REF` path in use, numpy 1.x, upstream `pufferl.py`, `pufferroyale` from the job's checkout |
| `gpu_heartbeat.py` | keeps the job's GPU at ≥ 75% utilization (user instruction; D79) |
| `peak_mem.py` | runs a script unchanged and reports its `torch.cuda.max_memory_allocated` |
| `pufferroyale.def` | the Apptainer image: CUDA 12.4 devel base, Python 3.12, PyTorch (cu124), PufferLib 3.0 @ `3b5c604` with `NO_OCEAN=1` and the D6 fix, this package |
| `build_image.sbatch` | builds `$SIF` with `--fakeroot` on a CPU node |
| `verify.sbatch` | gate G0 (HANDOFF §5) with per-step logs and a summary |
| `calibrate.sbatch`, `calibrate.py` | Phase 3 throughput grid on one GPU type, incl. runs stacked on one GPU |
| `league.sbatch` | league training (`scripts/league_train.py`, resumable), then a sampled round robin of the newest snapshots vs the bots |
| `train.sbatch` | plain PPO (`scripts/train.py`), reference |
| `stages.sbatch` | one stage-1 bot ladder (`scripts/stages.py`) |
| `pack.sbatch` | several runs stacked on one GPU (one command per line of `PACK_FILE`) |
| `br.sbatch`, `eval.sbatch` | best-response probes; bot evaluation, tournaments, transitivity |
| `league_run.py` | start a league run, or resume it if its run directory already holds one |
| `eval_watch.py` | evaluation sidecar of a league run: evaluates each new snapshot (DECISIONS D91) |
| `branch_run.py` | branch a league run at one of its snapshots into a new, self-contained run (the stage-4a arms) |
| `stage1_report.py` | the numbers the stage-1 decision rule needs (HANDOFF §8) |
| `stage3_submit.sh`, `stage3_array.sbatch` | prepare / submit the stage-3 runs (one array task per run) |
| `stage4_submit.sh` | prepare / submit stage-4 runs (chains of 20 h links; card floor options) |
| `stage4_eval.py`, `stage4_eval.sbatch` | stage-4 / stage-5 evaluations and round robins on CPU nodes |
| `collapse_diag.py`, `collapse_diag.sbatch`, `collapse_gpu_check.py`, `collapse_probe.py` | diagnostics of the method-C collapse (2026-10-15/16; DECISIONS D103-D104) |
| `llm.sbatch`, `llm_game.py`, `llm_relay.py` | Claude-vs-policy games (DECISIONS D101): the game runner in the image and the relay that asks the Claude Code CLI (the owner's own login; no API key) |
| `llm_render.py`, `llm_player.py` | a recorded Claude game as an animated GIF / a self-contained HTML player |
| `tests/` | fast tests of the cluster tools (`python -m pytest hpc/tests`) |

The browser play site's job and tools live in `play/` (`play/play_site.sbatch`, see `docs/HUMAN_PLAY.md`); the
real-game advisor's jobs live in `advisor/hpc/` (see `docs/ADVISOR.md`).

## Account, partition, GPU

Torch's job-submit filter picks the partition, so the scripts name no `--partition`. They request an
account and a GPU type:

```bash
#SBATCH --account=torch_pr_355_tandon_priority   # or torch_pr_355_tandon_advanced
#SBATCH --constraint=a100|h100|h200
#SBATCH --gres=gpu:1                             # one GPU per job, always (user's rule)
```

Override per submission (`sbatch --account=... --constraint=h100 ...`). H200 jobs may ask for at most 16
CPUs per GPU (more fails with "No partition fits this job request").

## Layout (everything under `$PR_ROOT` = /scratch/$USER/pufferroyale)

```
PufferRoyale/        our checkout (branch hpc-training) = $REPO
pinned/<name>/       git worktrees of $REPO that queued jobs run from (PR_CHECKOUT=...), so edits never reach them
containers/          pufferroyale.sif (built by build_image.sbatch)
cache/ tmp/ wandb/   our caches and temp files
logs/                submit from here: %x_%j.out / .err land here
runs/                every output (= $DATA_DIR): league/, train/, g0/, calibration/, br/, eval/
secrets/             wandb_key (chmod 600; never committed)
env.sh               copy of hpc/env.sh
```

`$PD_REF` (the PufferDrive reference) is read-only and never mounted, executed or put on a path; our
containers bind only `$PR_ROOT`, so it is not even visible inside them (HANDOFF §2, D79).

## Build the image (once, or after a change to pufferroyale.def)

```bash
source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
sbatch $REPO/hpc/build_image.sbatch          # --fakeroot build in the job's node-local temp dir
```

Jobs run the checkout, not the image's `/opt/pufferroyale` copy: `run` executes
`apptainer exec --nv --no-home --bind $PR_ROOT:$PR_ROOT --pwd $PROJECT $SIF ...`, and each job builds the C
extension in its checkout if it is missing or stale (under a lock).

## Submit

```bash
source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs
git -C $REPO worktree add --detach $PR_ROOT/pinned/<commit> <commit>       # pin the code a job runs
PR_CHECKOUT=$PR_ROOT/pinned/<commit> sbatch $PR_ROOT/pinned/<commit>/hpc/verify.sbatch           # G0
PR_CHECKOUT=... sbatch --constraint=h100 --job-name=pr-cal-h100 .../hpc/calibrate.sbatch          # Phase 3
PREFIX=S1main_s0 SEED=0 NUM_ENVS=... BATCH_SIZE=... MINIBATCH_SIZE=... EXTRA_ARGS="<shaping flags>" \
    sbatch --job-name=pr-S1main_s0 .../hpc/stages.sbatch                                         # stage 1
RUN_ID=A_s0 SEED=0 ANCHORS=none SELF_PLAY_FRAC=1.0 ANCHOR_FRAC=0 TOTAL=... EXTRA_ARGS="..." \
    sbatch --job-name=pr-A_s0 .../hpc/league.sbatch                                              # stage 3
```

- Job names start with `pr-`; cancel only by job id or `scancel --name=pr-...`, never `scancel -u`.
- Extra flags (`--env.*`, `--train.*`, `--rnn`, `--wandb ...`) always go through `EXTRA_ARGS`: the
  template tests check every literal flag in `train.sbatch` / `league.sbatch` against the scripts' `--help`.
- `league.sbatch` resumes when `<run dir>/league_state.json` exists: re-submit with **exactly** the same
  variables (TOTAL is the absolute target). A wall-time kill is not requeued: re-submit.
- `HEARTBEAT=0` disables the GPU heartbeat (not recommended: Torch reclaims idle GPUs).
- wandb: `--wandb` in `EXTRA_ARGS`; the key, entity and project come from `env.sh`; `WANDB_MODE=offline`
  then `wandb sync` from a login node if a compute node cannot reach wandb.
