# Documentation

Start with the [main README](../README.md). The files here go deeper; pick by what you want to do.

## Use it

| file | what it covers |
|---|---|
| [`HUMAN_PLAY.md`](HUMAN_PLAY.md) | the browser game against the policy: how a match runs, sign-in (name + PIN), records and replays, the permanent link, network design, security |
| [`../models/README.md`](../models/README.md) | the released model: what it is, how it was trained, how to load it, its results and limits |
| [`ADVISOR.md`](ADVISOR.md) | the advisor that reads a real Clash Royale screen and suggests moves: pipeline, checks, results, how to run it, and the Terms-of-Service warning |

## Build and develop

| file | what it covers |
|---|---|
| [`DEVELOPING.md`](DEVELOPING.md) | install and build, tests, every command-line tool (train, evaluate, watch, league, best response, tournament, LLM matches), the Python API, cards and decks, the code layout |

## The simulator

| file | what it covers |
|---|---|
| [`SPEC.md`](SPEC.md) | the binding specification of the engine and the environment (rules, tick pipeline, observation, actions, rewards, training tools in §15-§19) |
| [`FIDELITY.md`](FIDELITY.md) | every implementation choice the spec leaves open and every known divergence from the real game |

## Training and research

| file | what it covers |
|---|---|
| [`TRAINING_PLAN.md`](TRAINING_PLAN.md) | the plan of record (2026-10-02) from a verified simulator to results |
| [`STAGE4_SELECTION.md`](STAGE4_SELECTION.md) | the model-selection pipeline: rules written before any result, then every dated result (round robin, best-response exploitability, the winner, the card-floor ablation, the verdict) |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | the experiment log: every cluster job with its commit, submit line, GPU type, GPU-hours and outcome |
| [`DECISIONS.md`](DECISIONS.md) | every design and experiment decision (D1, D2, ...) with its options, evidence and what would reverse it |
| [`figures/`](figures/) | the README's plots and `make_figures.py`, which rebuilds them from the run outputs |

## Cluster (NYU Torch, Slurm + Apptainer)

| file | what it covers |
|---|---|
| [`HPC_HANDOFF.md`](HPC_HANDOFF.md) | step-by-step guide for running the training on the cluster |
| [`../hpc/README.md`](../hpc/README.md) | every cluster script and job template |

## History

| file | what it covers |
|---|---|
| [`PLAN.md`](PLAN.md) | the original master plan (2026-09-25) |
| [`STATUS.md`](STATUS.md) | the status at the handoff to cluster training (2026-09-27) |

**Reading the logs.** `EXPERIMENTS.md`, `STAGE4_SELECTION.md` and `DECISIONS.md` are dated lab notebooks:
entries are kept as written, so file paths in older entries are the paths of that day. On 2026-11-09 the
browser play site moved from `scripts/` and `hpc/` to `play/` (`scripts/play_human.py` is now
`play/play_server.py`, `hpc/play_human.sbatch` is `play/play_site.sbatch`). The author's cluster user
name is written as `$USER` (in paths) or `<netid>`.
