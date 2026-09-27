# PufferRoyale on the NYU Torch cluster (templates)

These files are **templates**: they were written against the SPEC (§15.6) and checked with
`bash -n` and against the scripts' `--help`, but they have not been run on the cluster. Replace
every placeholder before submitting and check the current NYU HPC documentation for the exact
account, partition and GPU names.

| file | what it does |
|---|---|
| `pufferroyale.def` | Apptainer image: CUDA 12.4 devel base, Python 3.12, PyTorch (cu124 wheels), PufferLib 3.0 from source with `NO_OCEAN=1` and the one-line `setup.py` fix (docs/DECISIONS.md D6), this package with its C extension |
| `train.sbatch` | plain PPO (`scripts/train.py`): self-play or vs a scripted bot, one GPU |
| `league.sbatch` | league training (`scripts/league_train.py`, resumable), then a tournament (`scripts/tournament.py`) of the newest snapshots vs the bots |

## 1. Placeholders to fill in

In both `.sbatch` files:

```bash
#SBATCH --account=YOUR_ACCOUNT          # your SLURM account (project allocation) on Torch
#SBATCH --partition=YOUR_GPU_PARTITION  # a GPU partition you may use
#SBATCH --gres=gpu:1                    # or a typed request, e.g. gpu:<type>:1, per the cluster docs
```

`sacctmgr show associations user=$USER format=account,partition` (or the NYU HPC portal)
lists the accounts and partitions available to you.

## 2. Build the image

Apptainer images are built once, on a machine where you may build (a build node, or your own
Linux machine with `--fakeroot`), from the **repository root** (the `%files` paths are relative
to it), and stored on scratch:

```bash
cd PufferRoyale
apptainer build --fakeroot pufferroyale.sif hpc/pufferroyale.def
mkdir -p /scratch/$USER/containers && mv pufferroyale.sif /scratch/$USER/containers/
apptainer test /scratch/$USER/containers/pufferroyale.sif      # imports + --help smoke test
```

The image holds the project at `/opt/pufferroyale` (scripts in `/opt/pufferroyale/scripts`).
To run a newer checkout without rebuilding, bind it over and point `PROJECT` at it (the C
extension must then be built for Linux inside the container once:
`apptainer exec --nv --bind /scratch/$USER --pwd /scratch/$USER/PufferRoyale $SIF python setup.py build_ext --inplace`).

## 3. Scratch layout

Everything writes under `/scratch/$USER` (bind-mounted into the container), never under
`$HOME`:

```
/scratch/$USER/containers/pufferroyale.sif        the image (SIF)
/scratch/$USER/pufferroyale/train/                train.sbatch: PuffeRL runs + summary_<job>.json
/scratch/$USER/pufferroyale/league/<run_id>/      league.sbatch: snap_<epoch>.pt, league_state.json,
                                                  learner.pt, history.jsonl, tournament_<job>.json
```

Override with `SCRATCH_DIR=...`, `SIF=...`, `DATA_DIR=...` in the environment of `sbatch`.

## 4. Submit

```bash
# plain PPO self-play (100M learner steps by default)
sbatch hpc/train.sbatch
TOTAL=200000000 NUM_AGENTS=1 OPPONENT=heuristic sbatch hpc/train.sbatch

# league: heuristic + random anchors, 30% self-play, PFSP snapshots every 50 epochs
RUN_ID=league_a sbatch hpc/league.sbatch
# the same with the MMD regulariser (reference refreshed every 50 epochs)
RUN_ID=league_mmd MMD_COEF=0.05 MMD_REF_INTERVAL=50 sbatch hpc/league.sbatch
# continue league_a up to 500M steps (TOTAL is absolute; the saved pool state, optimizer and RNG states are reused)
RUN_ID=league_a TOTAL=500000000 sbatch hpc/league.sbatch
```

Every job runs its Python entry point as

```bash
apptainer exec --nv --bind /scratch/$USER:/scratch/$USER --pwd /opt/pufferroyale \
    /scratch/$USER/containers/pufferroyale.sif python scripts/league_train.py ...
```

(`--nv` exposes the node's NVIDIA driver and GPUs inside the container). Monitor with
`squeue -u $USER` and `tail -f pr-league_<jobid>.out`; the league prints one progress line per
epoch (losses and the learner's score vs each opponent) and `history.jsonl` records the same.

`league.sbatch` is `--requeue`-safe: when the run directory already holds `league_state.json` it
resumes (`--resume`), so a preempted or re-submitted job continues from its last save (made at
every snapshot epoch): the learner, optimizer, pool and RNG states continue, while the matches that
were in progress are re-dealt -- deterministic, but not bit-identical to an uninterrupted run. A run
directory can be moved (snapshot paths are stored relative to it). A non-finite epoch is never saved:
the job exits 1 and `nonfinite.json` in the run directory describes it.

## 5. Interactive session

```bash
srun --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --gres=gpu:1 --cpus-per-task=8 \
     --mem=32G --time=01:00:00 --pty /bin/bash
apptainer exec --nv --bind /scratch/$USER --pwd /opt/pufferroyale /scratch/$USER/containers/pufferroyale.sif \
    python scripts/league_train.py --total-timesteps 200000 --device cuda --num-envs 64 --data-dir /scratch/$USER/pufferroyale
```

## 6. After training

```bash
SIF=/scratch/$USER/containers/pufferroyale.sif
RUN=/scratch/$USER/pufferroyale/league/league_a
# exploitability probe of a snapshot
apptainer exec --nv --bind /scratch/$USER --pwd /opt/pufferroyale $SIF python scripts/best_response.py \
    --target $RUN/snap_500.pt --total-timesteps 20000000 --matches 200 --device cuda --out $RUN/br_500.json
# tournament + meta-game Nash over snapshots and bots
apptainer exec --nv --bind /scratch/$USER --pwd /opt/pufferroyale $SIF python scripts/tournament.py \
    --agents bot:heuristic $RUN/snap_100.pt $RUN/snap_300.pt $RUN/snap_500.pt --decks hog26 giant bait \
    --matches 20 --device cuda --out $RUN/tournament.json
```

Notes: the rollout buffer costs ~76 KB per learner row (`batch_size` rows), so raise
`--num-envs` / `--batch-size` together with the GPU memory; `bot:*` opponents run in C on the
CPU, policy opponents are batched on the GPU per opponent.
