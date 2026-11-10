# Experiment log (NYU Torch)

Every job that trains, evaluates or verifies: date, git commit, the exact submit line, SLURM job id,
GPU type, GPU-hours (elapsed × GPUs, from `sacct`), the outcome, and where its outputs are. `$PR_ROOT`
= `/scratch/$USER/pufferroyale`, `$DATA_DIR` = `$PR_ROOT/runs`. Every submission starts with
`source /scratch/$USER/pufferroyale/env.sh && cd $PR_ROOT/logs`. Newest last.

## Cluster setup notes (no secrets)

Learned from the cluster itself and from `$PD_REF` = `/scratch/$USER/EMERGE/PufferDrive` (read
only, R1: `ls`/`cat`/`grep`/`find`/`stat`; nothing executed, no git), 2026-10-03.

| Item | Finding | Source |
|---|---|---|
| Cluster, login | Torch (`torch-login-*.hpc-infra.svc.cluster.local`), RHEL 10.2, Slurm in `/opt/slurm/bin` | cluster |
| Accounts | `torch_pr_355_tandon_priority` (PufferDrive's default) and `torch_pr_355_tandon_advanced`; `users` is rejected by the submit filter | `sacctmgr`, `$PD_REF/scripts/cluster_configs/<netid>_priority*.yaml`, `sbatch --test-only` |
| GPU request | `--gres=gpu:1` + `--constraint=a100\|h100\|h200` (also `--gres=gpu:h200:1` style); no `--partition`: a lua job-submit filter routes to e.g. `a100_tandon`, `h100_tandon`, `h200_public`, `l40s_public` | PufferDrive job scripts, `scontrol show job`, `sbatch --test-only` |
| GPU nodes | A100 `ga*` (4 GPUs, 64 cores), H100 `gh0*` (4 GPUs, 96 cores, AMD EPYC 9454), H200 `gh1*` (8 GPUs, 128 cores), L40S `gl*` (4 GPUs, 128 cores) | `scontrol show node`, job `lscpu` |
| CPUs per GPU | H200 jobs: at most 16 CPUs per GPU (24 → "No partition fits this job request"); A100 / H100 accepted 24 with 1 GPU | `sbatch --test-only`, 2026-10-03 |
| Limits | QOS `gpu48`: 48 h per job, 16 GPUs per user (it briefly read 0 on 2026-09-27); `gpu168`: 7 days, 4 GPUs per user; `cpu_short`: 6 h, 32 CPUs, 120 GB per user. Non-`_public` GPU partitions are `PreemptMode=REQUEUE` | `sacctmgr show qos`, `scontrol show partition` |
| Concurrency | PufferDrive jobs run under the same user and accounts (shared per-user GPU cap, fairshare and quota). User limit for us: ≤ 4 GPUs to start, 1 GPU per job | `squeue`, user |
| Idle GPUs | Torch reclaims low-utilization GPU jobs: PufferDrive runs `scripts/gpu_heartbeat.py` ("so the cluster's idle-GPU reclaimer does not scancel long jobs"; it targets 85%). Public NYU pages found: email < 20%, termination < 5% over 2 h (not confirmed for Torch). Ours: `hpc/gpu_heartbeat.py`, floor 75% (user) | `$PD_REF/scripts/gpu_heartbeat.py` docstring, web search |
| Containers | Apptainer 1.5.3 (`/share/apps/apptainer/bin`, a wrapper script); `--fakeroot` works on compute nodes (no `/etc/subuid` → root-mapped namespace + fakeroot command); site config binds `/etc/localtime`, `/etc/hosts`; the login profile exports `APPTAINER_BINDPATH=/scratch,/projects,/share/apps,/mnt,/state/partition1` (breaks `%post`, see D79) | cluster |
| PufferDrive's pattern | `singularity exec --nv --overlay <overlay-15GB-500K.ext3>:ro <cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif>` + `source .venv/bin/activate`; its setup script builds a miniforge (Python 3.12) + uv venv on `/scratch/$USER`, torch cu121, `TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"`; public base images in `/share/apps/images/`, overlay templates in `/share/apps/overlay-fs-ext3/`. It carries its **own `pufferlib/` fork** in its tree: never on our path (isolation check) | `$PD_REF/script*.s`, `$PD_REF/scripts/setup_container.sh`, `$PD_REF/scripts/submit_sbatch.py` |
| Other PufferDrive conventions | `WANDB_INSECURE_DISABLE_SSL=true` in its jobs (not adopted; noted in case wandb TLS fails on compute nodes); job logs to `<save_dir>/slurm-%j.out`; code snapshotted per job before submission so edits never reach queued jobs | same |
| Shell init | `/share/apps/profile.d/nyu.sh` exports `BASH_ENV=/etc/profile.d/cluster.sh` (re-sourced by every non-interactive bash, incl. the `apptainer` launcher script) and sets `APPTAINER_BINDPATH`, `APPTAINER_CACHEDIR`, `SINGULARITY_CACHEDIR`; our `env.sh` unsets `BASH_ENV` (D80) | `grep` of the site profile, `apptainer --debug` |
| Modules | No gcc, CUDA or Python modules (only `anaconda3/*`, `cmake`, `openmpi`, …); host python 3.12.14 without a compiler → everything in the image | `module avail` |
| Network | Login and compute nodes reach PyPI, GitHub, Docker Hub and wandb | `curl`, image builds on compute nodes |
| Scratch | `/scratch/$USER`: 5 TB / 5 M files quota, flushed (NYU's purge policy applies; not confirmed here); ~2.5 TB used (shared with PufferDrive) | `myquota` |
| CPU nodes | `cs*`: Intel Xeon Platinum 8592+ (one core ≈ 2× slower than an M4 on the dense stress benchmark, see the pre-v0.5 notes) | job `lscpu` |

## Gate G0 (environment verification)

| date | commit | job | GPU | GPU-h | submit line | outcome | outputs |
|---|---|---|---|---|---|---|---|
| 2026-10-03 | a76cfd8 | 19133793 | CPU (cs615) | 0 | `sbatch $REPO/hpc/build_image.sbatch` | image built in 8 min 54 s, `apptainer test` PASS: Python 3.12.15, numpy 1.26.4, torch 2.6.0+cu124 (CUDA 12.4), PufferLib 3.0 @ 3b5c604, wandb 0.30.0, matplotlib 3.11.2, pytest 9.1.1 | `$PR_ROOT/containers/pufferroyale.sif` |
| 2026-10-03 | 67ff257 | 19136106 | CPU (cs655) | 0 | `PR_CHECKOUT=$PR_ROOT/pinned/g0cpu_67ff257 STEPS="build-ext isolation test-c asan pytest golden-hashes e2e asan-py" sbatch <verify.sbatch without the GPU lines>` | cancelled after test-c: build-ext PASS, isolation OK (but `PD_REF visible in container: True`), **test-c FAIL: `undefined reference to 'pow'`** linking `test_wp_env`, asan PASS; both fixed in 50999ea (D80) | `$DATA_DIR/g0/19136106/` |
| 2026-10-03 | 50999ea | 19136476 | CPU (cs655, Xeon Platinum 8592+) | 0 | same, `PR_CHECKOUT=$PR_ROOT/pinned/g0cpu_50999ea` | isolation OK (`PD_REF visible in container: False`); test-c **184,647 checks, 0 failures**; asan **184,431 checks, clean**; **golden hashes PASS**; pytest **1,867 passed, 4 failed**; e2e **19/21** (failing: pytest, and perf stress 6,467 / 6,145 < 10,000 ticks/s; every v0.5 check passed: reward_v2, policy_heads, decks, grid, ops); asan-py (non-gating) 1,797 passed, 7 failed, no sanitizer report (D81). The 4 failures are explained platform deviations (D81); the user accepted G0 with them on 2026-10-04 | `$DATA_DIR/g0/19136476/`, diagnosis `$DATA_DIR/diag/diag_g0.py` |
| 2026-10-04 | 50999ea | 19134429 | **H100 80GB HBM3** (gh006; AMD EPYC 9454), driver 610.43.02 (CUDA UMD 13.3) | 1.15 | `PR_CHECKOUT=$PR_ROOT/pinned/g0_67ff257 sbatch $PR_ROOT/pinned/g0_67ff257/hpc/verify.sbatch` (submitted at 67ff257; the pinned worktree was moved to 50999ea before the job started) | **G0 on the GPU node (gate passed with the D81 deviations):** isolation OK; test-c 184,647 checks, 0 failures; asan clean; **golden hashes PASS**; pytest **1,867 passed, 4 failed** (the same 4 as on the CPU node, D81); e2e **19/21** (pytest; perf stress 8,439 < 10,000 ticks/s); **smoke-train PASS** (train.py, 311,296 steps, finite, peak `max_memory_allocated` 14.6 GiB at batch 16,384 / minibatch 4,096, 1,132 SPS in deterministic cuDNN); **smoke-league PASS** (league_train.py vs bot:heuristic, 200,704 steps, 49 epochs, 16 PFSP snapshots in the pool, finite, peak 3.7 GiB); asan-py (non-gating) exits before pytest prints anything on this GPU node (ran fully on the CPU node: run it on CPU nodes). Torch 2.6.0+cu124, cuDNN 9.1.0 | `$DATA_DIR/g0/19134429/` |

## Phase 3: throughput calibration

The jobs run from `$PR_ROOT/pinned/67ff257`, a worktree first created at 67ff257 and moved (before the jobs that use it started) to 50999ea, c92774b and e80c250 (D80, the heartbeat fix, D82). The A100 and H200 jobs start at e80c250.

| date | commit | job | GPU | GPU-h | submit line | outcome | outputs |
|---|---|---|---|---|---|---|---|
| 2026-10-04 | c92774b | 19134430 | H100 (gh002, AMD EPYC 9454) | 0.55 | `PR_CHECKOUT=$PR_ROOT/pinned/67ff257 sbatch --constraint=h100 --job-name=pr-cal-h100 $PR_ROOT/pinned/67ff257/hpc/calibrate.sbatch` | cancelled after 33 min (D82). Stages s1grid and s1mb ran with **deterministic cuDNN** (royale.ini default): stage-1 shape, num_envs 128/256/512, batch ×1/×2, minibatch 4096/8192/16384 → **1,160–1,190 SPS** whatever the size, 0.98 CPU cores; peak `max_memory_allocated` 17.9 GiB (mb 4096), 27.6–31.4 GiB (mb 8192), 58.3 GiB (mb 16384). Kept as the deterministic reference. New heartbeat: device utilization 80–82% (5-min means), heartbeat busy 29–44% | `$DATA_DIR/calibration/NVIDIA_H100_80GB_HBM3_19134430/` |
| 2026-10-04 | e80c250 | 19149010 | H100 (gh002, AMD EPYC 9454) | 0.87 | same submit line (non-deterministic cuDNN by default from e80c250) | completed; table below | `$DATA_DIR/calibration/NVIDIA_H100_80GB_HBM3_19149010/` |
| 2026-10-04 | e80c250 | 19134431 | A100-SXM4-80GB (ga031, Xeon Platinum 8380) | 0.89 | same with `--constraint=a100 --job-name=pr-cal-a100` | completed; table below | `$DATA_DIR/calibration/NVIDIA_A100-SXM4-80GB_19134431/` |
| 2026-10-03 | e80c250 | 19134443 | H200 (gh120) | 0.85 | same with `--constraint=h200 --cpus-per-task=16 --job-name=pr-cal-h200` (24 CPUs is refused for H200) | COMPLETED 2026-10-04 (queued 12.6 h: submitted 2026-10-03 22:56, started 11:32); table below: ~20–25% slower than the H100 node | `$DATA_DIR/calibration/<GPU>_19134443/` |

### Profiling the slow league (2026-10-04, inside G0 job 19134429's allocation, H100 gh006)

| what | result |
|---|---|
| cProfile, `league_train.py` stage-1 shape, 256 envs, batch 16384, 2 epochs, deterministic cuDNN | 37.7 s total: `torch.conv2d` 15.9 s (924 calls, 17 ms each), `vec_step` 2.9 s, backward 2.1 s, `bot_act_env` 0.2 s; 969–1,081 SPS |
| `forward_eval`, 256 rows, default conditional policy | cuDNN default 1.61 ms; `deterministic=True` 73.1 ms; `benchmark=True` 2.66 ms; cuDNN off 95.4 ms; CPU (16 threads) 202 ms |
| PufferLib config parser, `--train.torch-deterministic X` | X = `False` → True, `0` → True, empty (`--train.torch-deterministic=`) → False |

Scripts: `$DATA_DIR/diag/fwd_bench.py`, profile `$DATA_DIR/diag/prof/league_cuda.prof`.

### H100 80GB HBM3 (job 19149010, gh002, AMD EPYC 9454, 24 CPUs; cuDNN non-deterministic, GPU heartbeat on; e80c250)

| stage | shape | envs | workers | rnn | batch | minibatch | copies | SPS total | SPS per copy | peak alloc GiB (per process) | GPU mem used MiB | CPU cores |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1grid | stage1 | 128 | - | no | 8192 | 8192 | 1 | 5410 | 5410 | 27.58 | 17331 | 0.97 |
| s1grid | stage1 | 128 | - | no | 16384 | 8192 | 1 | 5467 | 5467 | 28.12 | 17437 | 0.97 |
| s1grid | stage1 | 256 | - | no | 16384 | 8192 | 1 | 6144 | 6144 | 28.12 | 17421 | 0.97 |
| s1grid | stage1 | 256 | - | no | 32768 | 8192 | 1 | 6085 | 6085 | 29.2 | 19163 | 0.97 |
| s1grid | stage1 | 512 | - | no | 32768 | 8192 | 1 | 6625 | 6625 | 29.2 | 18173 | 0.97 |
| s1grid | stage1 | 512 | - | no | 65536 | 8192 | 1 | 6496 | 6496 | 31.36 | 21213 | 0.97 |
| s1mb | stage1 | 512 | - | no | 65536 | 4096 | 1 | 6402 | 6402 | 17.88 | 13245 | 0.97 |
| s1mb | stage1 | 512 | - | no | 65536 | 16384 | 1 | 6539 | 6539 | 58.33 | 34063 | 0.97 |
| s1stack | stage1 | 256 | - | no | 32768 | 8192 | 2 | 11549 | 5775/5774 | 29.2 | 36457 | 1.95 |
| s1stack | stage1 | 256 | - | no | 32768 | 8192 | 4 | 21703 | 5349/5587/5358/5409 | 29.2 | 71575 | 3.86 |
| s1stack | stage1 | 512 | - | no | 65536 | 8192 | 2 | 12746 | 6409/6337 | 31.36 | 40103 | 1.93 |
| s1stack | stage1 | 512 | - | no | 65536 | 8192 | 4 | 21576 | 5408/5373/5389/5405 | 26.72 | 79527 | 3.87 |
| A | A | 512 | - | no | 65536 | 8192 | 1 | 3537 | 3537 | 31.36 | 20393 | 6.09 |
| ckpt | ckpt16 | 512 | - | no | 65536 | 8192 | 1 | 1093 | 1093 | 31.44 | 20863 | 16.28 |
| ckpt | ckpt32 | 512 | - | no | 65536 | 8192 | 1 | 1222 | 1222 | 31.52 | 21335 | 15.68 |
| ckpt | ckpt64 | 512 | - | no | 65536 | 8192 | 1 | 1091 | 1091 | 31.67 | 21045 | 15.67 |
| rnn | stage1 | 512 | - | yes | 65536 | 8192 | 1 | 6502 | 6502 | 31.38 | 20343 | 0.97 |
| train | - | 64 | - | no | 65536 | 16384 | 1 | 6004 | 6004 | 58.33 | 33965 | 0.89 |
| train | - | 64 | 4 | no | 65536 | 16384 | 1 | 13580 | 13580 | 58.33 | 36319 | 3.61 |
| train | - | 64 | 8 | no | 65536 | 16384 | 1 | 23083 | 23083 | 58.33 | 36401 | 3.98 |
| train | - | 256 | - | no | 65536 | 16384 | 1 | 7804 | 7804 | 58.33 | 36407 | 0.92 |

Reading: the stage-1 league is CPU-bound in ONE process (~1 core): 5.4–6.6k SPS whatever the size; stacking 4 processes on one GPU gives ~21.6k total (5.4k each), but 4 × (512 envs, batch 65,536, minibatch 8,192) reaches 79.5 of 81.6 GB. Minibatch size changes memory (17.9 / 29–31 / 58 GiB at 4,096 / 8,192 / 16,384), not speed. The LSTM costs nothing. Stage-3 shapes are slower: A (self-play) 3.5k SPS on 6 cores; B/C with 16–64 frozen opponents ~1.1–1.2k SPS on ~16 cores.

### A100-SXM4-80GB (job 19134431, ga031, Intel Xeon Platinum 8380 @ 2.30GHz, 24 CPUs; cuDNN non-deterministic, heartbeat on; e80c250)

| stage | shape | envs | workers | rnn | batch | minibatch | copies | SPS total | SPS per copy | peak alloc GiB (per process) | GPU mem used MiB | CPU cores |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1grid | stage1 | 128 | - | no | 8192 | 8192 | 1 | 3292 | 3292 | 27.53 | 16973 | 0.98 | 
| s1grid | stage1 | 128 | - | no | 16384 | 8192 | 1 | 3341 | 3341 | 28.07 | 17565 | 0.98 | 
| s1grid | stage1 | 256 | - | no | 16384 | 8192 | 1 | 3843 | 3843 | 28.07 | 16231 | 0.98 | 
| s1grid | stage1 | 256 | - | no | 32768 | 8192 | 1 | 3828 | 3828 | 29.15 | 17337 | 0.98 | 
| s1grid | stage1 | 512 | - | no | 32768 | 8192 | 1 | 4082 | 4082 | 29.15 | 17905 | 0.98 | 
| s1grid | stage1 | 512 | - | no | 65536 | 8192 | 1 | 4018 | 4018 | 31.32 | 19087 | 0.98 | 
| s1mb | stage1 | 512 | - | no | 65536 | 4096 | 1 | 4047 | 4047 | 17.83 | 13227 | 0.98 | 
| s1mb | stage1 | 512 | - | no | 65536 | 16384 | 1 | 4053 | 4053 | 58.28 | 34895 | 0.98 | 
| s1stack | stage1 | 256 | - | no | 32768 | 8192 | 2 | 7194 | 3594/3600 | 29.15 | 36331 | 1.96 | 
| s1stack | stage1 | 256 | - | no | 32768 | 8192 | 4 | 3165 | 3165 | 29.15 | - | None | a copy exited early; 3 copies unmeasured
| s1stack | stage1 | 512 | - | no | 65536 | 8192 | 2 | 7682 | 3837/3845 | 31.32 | 40363 | 1.96 | 
| s1stack | stage1 | 512 | - | no | 65536 | 8192 | 4 | 14804 | 3733/3659/3654/3758 | 31.32 | 77615 | 3.92 | 
| A | A | 512 | - | no | 65536 | 8192 | 1 | 2189 | 2189 | 31.32 | 20243 | 3.78 | 
| ckpt | ckpt16 | 512 | - | no | 65536 | 8192 | 1 | 1092 | 1092 | 31.39 | 21397 | 14.4 | 
| ckpt | ckpt32 | 512 | - | no | 65536 | 8192 | 1 | 1109 | 1109 | 31.47 | 21285 | 14.2 | 
| ckpt | ckpt64 | 512 | - | no | 65536 | 8192 | 1 | - |  | 31.63 | 21759 | 15.53 | ok; 1 copies unmeasured
| rnn | stage1 | 512 | - | yes | 65536 | 8192 | 1 | 4056 | 4056 | 31.33 | 21295 | 0.98 | 
| train | - | 64 | - | no | 65536 | 16384 | 1 | 3940 | 3940 | 58.28 | 67777 | 0.96 | 
| train | - | 64 | 4 | no | 65536 | 16384 | 1 | 10050 | 10050 | 58.28 | 37805 | 3.73 | 
| train | - | 64 | 8 | no | 65536 | 16384 | 1 | 13218 | 13218 | 58.28 | 33737 | 5.97 | 
| train | - | 256 | - | no | 65536 | 16384 | 1 | 5148 | 5148 | 58.28 | 33591 | 0.96 |

Reading (A100 node): the 2.3 GHz Xeon steps the env ~40% slower than the H100 node's EPYC (stage-1 shape 3.3–4.1k SPS per process vs 5.4–6.6k); stacking scales the same way (4 copies 14.8k). One 4-copy configuration (256 envs, batch 32,768, minibatch 8,192) ran a copy out of GPU memory (CUDA OOM: four ~29 GiB peaks exceed 80 GB) -- large minibatches and stacking do not mix; stage 1 uses minibatch 1,024 (~4 GB per ladder, D84/D85). ckpt64 was too slow (~1k SPS) to log 3 epochs in the 120 s window.

### H200 (job 19134443, gh120, Intel Xeon Platinum 8592+, 16 CPUs; cuDNN non-deterministic, heartbeat on (old version: 2 of 10 windows at 60% / 69%, D87); e80c250; 0.85 GPU-h)

Submit line: as the H100 / A100 calibrations (`hpc/calibrate.sbatch` from `$PR_ROOT/pinned/67ff257` at e80c250), `--constraint=h200`.

| name | shape | envs | workers | rnn | det. cuDNN | batch | minibatch | copies | SPS total | SPS/copy | peak alloc GiB | GPU mem MiB | GPU util % | CPU cores | RSS GiB | status |
| s1grid | stage1 | 128 | - | no | no | 8192 | 8192 | 1 | 3967.6 | 3968 | 27.58 | 17329.0 | 83.41 | 0.98 | 2.27 | ok |
| s1grid | stage1 | 128 | - | no | no | 16384 | 8192 | 1 | 3993.1 | 3993 | 28.12 | 17883.0 | 82.66 | 0.98 | 2.27 | ok |
| s1grid | stage1 | 256 | - | no | no | 16384 | 8192 | 1 | 4624.8 | 4625 | 28.12 | 18051.0 | 77.48 | 0.98 | 2.36 | ok |
| s1grid | stage1 | 256 | - | no | no | 32768 | 8192 | 1 | 4645.4 | 4645 | 29.2 | 19161.0 | 76.59 | 0.98 | 2.37 | ok |
| s1grid | stage1 | 512 | - | no | no | 32768 | 8192 | 1 | 4905.0 | 4905 | 29.2 | 18493.0 | 77.07 | 0.98 | 2.61 | ok |
| s1grid | stage1 | 512 | - | no | no | 65536 | 8192 | 1 | 4901.5 | 4902 | 31.36 | 21085.0 | 77.19 | 0.98 | 2.61 | ok |
| s1mb | stage1 | 512 | - | no | no | 65536 | 4096 | 1 | 4924.0 | 4924 | 17.88 | 13749.0 | 74.34 | 0.98 | 2.6 | ok |
| s1mb | stage1 | 512 | - | no | no | 65536 | 16384 | 1 | 4909.3 | 4909 | 58.33 | 36437.0 | 78.45 | 0.98 | 2.61 | ok |
| s1stack | stage1 | 256 | - | no | no | 32768 | 8192 | 2 | 8886.3 | 4443/4444 | 29.2 | 37646.0 | 80.31 | 1.96 | 4.73 | ok |
| s1stack | stage1 | 256 | - | no | no | 32768 | 8192 | 4 | 16971.0 | 3915/4324/4366/4365 | 29.2 | 74624.0 | 80.27 | 3.91 | 9.53 | ok |
| s1stack | stage1 | 512 | - | no | no | 65536 | 8192 | 2 | 9585.4 | 4834/4752 | 31.36 | 40744.0 | 80.84 | 1.96 | 5.22 | ok |
| s1stack | stage1 | 512 | - | no | no | 65536 | 8192 | 4 | 17589.5 | 4411/4385/4384/4410 | 31.36 | 80814.0 | 81.38 | 3.91 | 10.44 | ok |
| A | A | 512 | - | no | no | 65536 | 8192 | 1 | 2739.6 | 2740 | 31.36 | 20707.0 | 80.16 | 3.32 | 2.67 | ok |
| ckpt | ckpt16 | 512 | - | no | no | 65536 | 8192 | 1 | 1113.4 | 1113 | 31.44 | 21267.0 | 81.21 | 10.19 | 2.66 | ok |
| ckpt | ckpt32 | 512 | - | no | no | 65536 | 8192 | 1 | 1200.1 | 1200 | 31.52 | 20733.0 | 82.67 | 9.96 | 2.69 | ok |
| ckpt | ckpt64 | 512 | - | no | no | 65536 | 8192 | 1 | - | - | 31.67 | 21503.0 | 76.35 | 10.95 | 2.72 | ok; 1 copies unmeasured |
| rnn | stage1 | 512 | - | yes | no | 65536 | 8192 | 1 | 4886.0 | 4886 | 31.38 | 21467.0 | 81.71 | 0.98 | 2.63 | ok |
| train | - | 64 | - | no | no | 65536 | 16384 | 1 | 4966.6 | 4967 | 58.33 | 50393.0 | 85.62 | 0.97 | 2.19 | ok |
| train | - | 64 | 4 | no | no | 65536 | 16384 | 1 | 14891.2 | 14891 | 58.33 | 36315.0 | 94.7 | 3.81 | 4.06 | ok |
| train | - | 64 | 8 | no | no | 65536 | 16384 | 1 | 21250.1 | 21250 | 58.33 | 36431.0 | 90.4 | 5.07 | 6.27 | ok |
| train | - | 256 | - | no | no | 65536 | 16384 | 1 | 6593.8 | 6594 | 58.33 | 36431.0 | 83.59 | 0.95 | 2.33 | ok |

Reading: the H200 node is ~20–25% slower than the H100 node (AMD EPYC 9454) in every league shape, because the league is CPU-bound and the CPU differs: stage 1 4.0–4.9k SPS per process (H100 5.4–6.6k), 4 stacked 17.0–17.6k (21.6k), A 2.7k (3.5k). B/C (ckptN) stay at ~1.1–1.2k SPS on both nodes, on ~10 cores here vs ~16 on the H100 node, so core count does not track their speed (investigated by the SPS research, 2026-10-04). ckpt64 produced no steady-state reading in 120 s. Use H100 nodes for league runs.

## Phase 3: early learning smoke (HANDOFF §6b)

| date | commit | job | GPU | GPU-h | submit line | outcome | outputs |
|---|---|---|---|---|---|---|---|
| 2026-10-04 | adc8125 | 19150097 | H100 (gh015) | 0 |  `PR_CHECKOUT=$PR_ROOT/pinned/run_adc8125 RUN_ID=smoke_random_s0 SEED=0 ANCHORS=bot:random SELF_PLAY_FRAC=0 ANCHOR_FRAC=1.0 TOTAL=50000000 NUM_ENVS=512 BATCH_SIZE=65536 MINIBATCH_SIZE=8192 SNAPSHOT_INTERVAL=50 TOURNAMENT_MATCHES=0 EXTRA_ARGS="--env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0 --train.torch-deterministic= --wandb --wandb-project pufferroyale --wandb-group smoke6b" sbatch --job-name=pr-smoke_random_s0 $PR_ROOT/pinned/run_adc8125/hpc/league.sbatch` | FAILED after 1 s (exit 2): `pr_ensure_ext`'s `ls` of the not-yet-built extension failed under `set -euo pipefail`; fixed in d48de07 | - |
| 2026-10-04 | d48de07 | 19150297 | H100 (gh011) | 0.20 | same, `PR_CHECKOUT=$PR_ROOT/pinned/run_d48de07` | **PASSED the smoke criterion: score vs bot:random 0.927 (last ~1,060 matches) at 3.87M learner steps**, from the untrained ~0.25; ~5.5k SPS. Per 0.5M-step window: 0.26, 0.25, 0.38, 0.47, 0.57, 0.68, 0.81, 0.93, 0.95. ~8× the Mac's 0.4M steps, as its 8× sparser gradient updates predict (D84). Cancelled at 4.13M steps (epoch 63) once the curve was recorded, to free the GPU for stage 1. Plot: `history.png` | `$DATA_DIR/league/smoke_random_s0/`, wandb `<netid>-new-york-university/pufferroyale` group `smoke6b` |
| 2026-10-04 | d48de07 | 19150430 | H100 (gh010) | 0.06 | `PR_CHECKOUT=$PR_ROOT/pinned/run_d48de07 RUN_ID=smoke_random_default_s0 SEED=0 ANCHORS=bot:random SELF_PLAY_FRAC=0 ANCHOR_FRAC=1.0 TOTAL=20000000 NUM_ENVS=256 SNAPSHOT_INTERVAL=50 TOURNAMENT_MATCHES=0 EXTRA_ARGS="--env.reward-tower 0.3 --env.reward-crown 0.2 --env.reward-elixir 0.01 --train.reward-clip 0 --train.torch-deterministic= --wandb --wandb-project pufferroyale --wandb-group smoke6b" sbatch --job-name=pr-smoke_random_default_s0 --time=06:00:00 $PR_ROOT/pinned/run_d48de07/hpc/league.sbatch` (league default sizes: batch 16,384, minibatch 1,024; D84) | cancelled after 3.5 min: superseded by the stage-1 `S1main` ladders, whose rungs use exactly this configuration | `$DATA_DIR/league/smoke_random_default_s0/` |

## Stage 1: bot ladder (HANDOFF §8, TRAINING_PLAN §8.1)

Configuration (D82, D84): stage-1 main = `royale.ini` hyperparameters (lr 3e-4, γ 0.999, λ 0.95, clip 0.2,
vf_coef 2.0, ent_coef 0.01, bptt 64, 1 update epoch), conditional head, grid 1, feed-forward, shaping 0.3 /
0.2 / 0.01 with `--train.reward-clip 0`, Hog 2.6 mirror, Tower Princess, frame_skip 10; league default sizes
(256 envs, batch 16,384, minibatch 1,024); `--snapshot-interval 200`; `--train.torch-deterministic=`; default
rungs `bot:noop:5M:0.95; bot:random:50M:0.90; bot:heuristic:300M:0.55`, gate window 2,000 matches.
Ablations: noshape, grid2, grid4, lstm, flat, lr 1e-4, lr 1e-3, ent (card 0.01 / pos 0.002); 2 seeds each;
18 ladders stacked up to 7 per GPU (`hpc/pack.sbatch`, D85). Pack files: `$DATA_DIR/packs/stage1_pack{0,A,B}.txt`
(the first-draft packs 1–4, 4 per GPU, were cancelled while pending and kept in `packs/superseded/`);
wandb groups `S1<config>`, tag = the run prefix.

| date | commit | job | GPU | GPU-h | submit line | ladders | outcome |
|---|---|---|---|---|---|---|---|
| 2026-10-04 | d48de07 | 19150509 | H100 (gh011) | 0.99 | `PACK_FILE=$PR_ROOT/runs/packs/stage1_pack0.txt PR_CHECKOUT=$PR_ROOT/pinned/run_d48de07 sbatch --job-name=pr-S1pack0 $PR_ROOT/pinned/run_d48de07/hpc/pack.sbatch` | S1main_s0/s1, S1noshape_s0/s1 | COMPLETED in 59 min; all 4 passed G1–G3. G3 (heuristic, 2,000 matches): main 0.883 / 0.656, noshape 0.706 / 0.558 |
| 2026-10-04 | d48de07 | 19150511 | - | 0 | same with `stage1_pack1.txt`, `--job-name=pr-S1pack1` | S1grid2_s0/s1, S1grid4_s0/s1 | cancelled while pending; repacked into packs A/B (D85) |
| 2026-10-04 | d48de07 | 19150512 | - | 0 | same with `stage1_pack2.txt`, `--job-name=pr-S1pack2` | S1lstm_s0/s1, S1flat_s0/s1 | cancelled while pending; repacked into packs A/B (D85) |
| 2026-10-04 | d48de07 | 19150513 | - | 0 | same with `stage1_pack3.txt`, `--job-name=pr-S1pack3` | S1lr1e-4_s0/s1, S1lr1e-3_s0/s1 | cancelled while pending; repacked into packs A/B (D85) |
| 2026-10-04 | d48de07 | 19150514 | - | 0 | same with `stage1_pack4.txt`, `--job-name=pr-S1pack4` | S1ent_s0/s1 | cancelled while pending; repacked into packs A/B (D85) |
| 2026-10-04 | d48de07 | 19150613 | H100 (gh002) | 0.97 | `PACK_FILE=$PR_ROOT/runs/packs/stage1_packA.txt PR_CHECKOUT=$PR_ROOT/pinned/run_d48de07 sbatch --job-name=pr-S1packA --cpus-per-task=16 --mem=128G $PR_ROOT/pinned/run_d48de07/hpc/pack.sbatch` | S1grid2_s0/s1, S1grid4_s0/s1, S1ent_s0/s1, S1lr1e-4_s0 | COMPLETED in 58 min; all 7 passed G1–G3. G3: grid2 0.748 / 0.737, grid4 0.608 / 0.603, ent 0.666 / 0.584, lr1e-4_s0 0.559 |
| 2026-10-04 | d48de07 | 19150614 | H100 (gh004) | 2.28 | same with `stage1_packB.txt`, `--job-name=pr-S1packB` | S1lr1e-4_s1, S1lr1e-3_s0/s1, S1lstm_s0/s1, S1flat_s0/s1 | COMPLETED in 2 h 17 min; all 7 passed G1–G3. G3: lr1e-4_s1 0.637, lr1e-3 0.645 / 0.723, lstm 0.718 / 0.703, flat 0.588 (after 16.4M heuristic-rung steps) / 0.689 (after 32.8M; the slowest ladder) |

### Gate confirmations (HANDOFF §8: `eval.py`, 500 matches per seat, sampled actions, CPU nodes)

| date | commit | job | node | submit line | result |
|---|---|---|---|---|---|
| 2026-10-04 | 8d0c10f | 19150891 | CPU | `PR_CHECKOUT=$PR_ROOT/pinned/eval_8d0c10f MODE=bots DEVICE=cpu CHECKPOINT=$DATA_DIR/league/S1main_s1_2_heuristic BOTS=heuristic EPISODES=500 OUT=$DATA_DIR/eval/gate_S1main_s1_G3_heuristic.json sbatch --job-name=pr-eval-S1main_s1-G3 $PR_ROOT/logs/eval_cpu_8d0c10f.sbatch` (eval.sbatch without its GPU lines) | **G3 confirmed for S1main_s1**: vs heuristic, seat 0 345-3-152 = **0.693 [0.651, 0.732]**, seat 1 337-0-163 = **0.674 [0.632, 0.714]** (Wilson 95%); 7.6 min |
| 2026-10-04 | e95849b | 19153368 | CPU (16 cores) | `PACK_FILE=$PR_ROOT/runs/packs/gate_confirm_1.txt PR_CHECKOUT=$PR_ROOT/pinned/run_e95849b MAX_PARALLEL=14 PACK_THREADS=1 sbatch --job-name=pr-gateconf1 $PR_ROOT/logs/pack_cpu_e95849b.sbatch` (pack.sbatch without its GPU lines) | 44 evals: G1 (noop, 100/seat), G2 (random, 500/seat), G3 (heuristic, 500/seat) for the 15 ladders that passed all gates; outputs `$DATA_DIR/eval/gate_<prefix>_G<n>_<bot>.json`; **COMPLETED in 29 min, all 44 PASS**: G1 1.000 on both seats everywhere; G2 0.920–1.000 per seat (lowest S1noshape_s1 seat 1, 0.920 [0.893, 0.941]); G3 Wilson lower bound > 0.5 on both seats for all 15 (table below) |
| 2026-10-04 | e95849b | 19154535 | CPU (8 cores) | `PACK_FILE=$PR_ROOT/runs/packs/gate_confirm_2.txt PR_CHECKOUT=$PR_ROOT/pinned/run_e95849b MAX_PARALLEL=7 PACK_THREADS=1 sbatch --job-name=pr-gateconf2 --cpus-per-task=8 --mem=32G $PR_ROOT/logs/pack_cpu_e95849b.sbatch` | 7 evals: S1ent_s1 G1–G3, S1flat_s0/s1 G1–G2. COMPLETED in 10.5 min. S1ent_s1: G1 1.000, G2 0.996 / 0.994, G3 0.628 [0.585, 0.669] / 0.635 [0.592, 0.676], all PASS. S1flat_s0: G1 1.000, G2 0.952 [0.930, 0.968] / 0.938 [0.913, 0.956], PASS. S1flat_s1: G1 1.000; **G2 not confirmed on seat 0: 0.892 [0.862, 0.916]** (seat 1 0.920 [0.893, 0.941]; both seats 0.906; in-training gate 0.924 over 2,000). Borderline (the interval contains 0.90); reported as is, not re-run |
| 2026-10-04 | e95849b | 19156724 | CPU (2 cores) | `PACK_FILE=$PR_ROOT/runs/packs/gate_confirm_3.txt PR_CHECKOUT=$PR_ROOT/pinned/run_e95849b MAX_PARALLEL=2 PACK_THREADS=1 sbatch --job-name=pr-gateconf3 --cpus-per-task=2 --mem=8G $PR_ROOT/logs/pack_cpu_e95849b.sbatch` | 2 evals: S1flat_s0/s1 G3. COMPLETED in 7 min, both PASS (table below). **Totals: 54 gate confirmations (18 ladders × G1–G3), 53 confirmed, 1 borderline not confirmed (S1flat_s1 G2 seat 0)** |

G3 confirmations (vs `bot:heuristic`, 500 matches per seat, sampled; score [Wilson 95%]). These test the rung's final
model, while the in-training gate averages the last 2,000 matches of earlier policies too, so they read higher
(e.g. S1noshape_s1: gate 0.558, eval 0.776 / 0.726).

| ladder | model | seat 0 | seat 1 |
|---|---|---|---|
| S1ent_s0 | model_000600.pt | 0.630 [0.587, 0.671] | 0.686 [0.644, 0.725] |
| S1ent_s1 | model_000800.pt | 0.628 [0.585, 0.669] | 0.635 [0.592, 0.676] |
| S1flat_s0 | model_001000.pt | 0.614 [0.571, 0.656] | 0.563 [0.519, 0.606] |
| S1flat_s1 | model_002000.pt | 0.740 [0.700, 0.777] | 0.698 [0.656, 0.737] |
| S1grid2_s0 | model_000400.pt | 0.732 [0.692, 0.769] | 0.742 [0.702, 0.778] |
| S1grid2_s1 | model_000400.pt | 0.664 [0.621, 0.704] | 0.698 [0.656, 0.737] |
| S1grid4_s0 | model_000400.pt | 0.678 [0.636, 0.717] | 0.656 [0.613, 0.696] |
| S1grid4_s1 | model_000600.pt | 0.576 [0.532, 0.619] | 0.589 [0.545, 0.631] |
| S1lr1e-3_s0 | model_000200.pt | 0.647 [0.604, 0.688] | 0.718 [0.677, 0.756] |
| S1lr1e-3_s1 | model_000400.pt | 0.722 [0.681, 0.759] | 0.723 [0.682, 0.760] |
| S1lr1e-4_s0 | model_000400.pt | 0.596 [0.552, 0.638] | 0.584 [0.540, 0.626] |
| S1lr1e-4_s1 | model_000400.pt | 0.692 [0.650, 0.731] | 0.622 [0.579, 0.663] |
| S1lstm_s0 | model_000400.pt | 0.727 [0.686, 0.764] | 0.746 [0.706, 0.782] |
| S1lstm_s1 | model_000400.pt | 0.720 [0.679, 0.758] | 0.773 [0.734, 0.808] |
| S1main_s0 | model_000400.pt | 0.883 [0.852, 0.908] | 0.896 [0.866, 0.920] |
| S1main_s1 | model_000200.pt | 0.693 [0.651, 0.732] | 0.674 [0.632, 0.714] |
| S1noshape_s0 | model_000400.pt | 0.762 [0.723, 0.797] | 0.728 [0.687, 0.765] |
| S1noshape_s1 | model_000800.pt | 0.776 [0.737, 0.810] | 0.726 [0.685, 0.763] |

### Fixed-budget heuristic phase (D86)

| date | commit | job | GPU | GPU-h | submit line | runs | outcome |
|---|---|---|---|---|---|---|---|
| 2026-10-04 | 7492df2 | 19152998 | H100 (gh012) | 5.60 | `PACK_FILE=$PR_ROOT/runs/packs/stage1_H300_packH.txt PR_CHECKOUT=$PR_ROOT/pinned/run_7492df2 sbatch --job-name=pr-S1H300H --cpus-per-task=16 --mem=128G $PR_ROOT/pinned/run_7492df2/hpc/pack.sbatch` | S1main_s0/s1, S1noshape_s0/s1, S1grid2_s0/s1 `_H300` | stopped at 101.6M (D88): every run past `snap_6200.pt`, then cancelled by job id at 13:28 |
| 2026-10-04 | 7492df2 | 19152999 | H100 (gh009) | 5.46 | same with `stage1_H300_packI.txt`, `--job-name=pr-S1H300I` | S1grid4_s0/s1, S1ent_s0/s1, S1lr1e-4_s0/s1 `_H300` | stopped at 101.6M (D88) |
| 2026-10-04 | 7492df2 | 19153517 | H100 (gh002) | 5.35 | same with `stage1_H300_packJ.txt`, `--job-name=pr-S1H300J` | S1lr1e-3_s0/s1, S1lstm_s0/s1, S1flat_s0/s1 `_H300` | started 16 min late on a busy node (D87); stopped at 101.6M (D88) |
| 2026-10-04 | e95849b | 19166231 | CPU (18 cores) | 0 | `PACK_FILE=$PR_ROOT/runs/packs/h300_100M_eval.txt PR_CHECKOUT=$PR_ROOT/pinned/run_e95849b MAX_PARALLEL=18 PACK_THREADS=1 sbatch --job-name=pr-h100Meval --cpus-per-task=18 --mem=48G $PR_ROOT/logs/pack_cpu_e95849b.sbatch` | 18 evals of `snap_6200.pt` vs bot:heuristic, 500/seat | COMPLETED in 11 min 56 s; results in "Stage-1 results" below |

## Stage-1 results (G1–G3 report, 2026-10-04)

Sources:
- `hpc/stage1_report.py --marks 11M,25M,50M,75M,100M` → `$DATA_DIR/reports/stage1/stage1_report.{md,json}`;
- plots `heuristic_rung.png` and `fixed_budget.png` there;
- `eval.py` JSONs `$DATA_DIR/eval/gate_*` and `h100M_*`.

Totals:
- **Ladders:** all 18 passed G1–G3.
- **Gate confirmations:** 53 of 54 confirmed; S1flat_s1 G2 seat 0 is borderline at 0.892 [0.862, 0.916].
- **Stage-1 GPU-hours:** 20.65 (ladders 4.24, fixed-budget phase 16.41).
- **GPU-hours since 2026-10-03:** 25.50 (G0 / calibration / smoke 4.57, heartbeat tests 0.28).

**Ablation table.** Fixed-budget runs (D86/D88) start from each ladder's random-rung model; scores are
vs `bot:heuristic`.

| config | steps to 0.90 (seed mean; s0 / s1) | train window @11M | @25M | @50M | @100M | eval.py @101.6M (seed mean; s0 / s1; 1,000 matches per seed) |
|---|---|---|---|---|---|---|
| grid2 | 7.4M (8.4 / 6.5) | 0.942 | 0.976 | 0.993 | 0.997 | 0.997 (0.996 / 0.998) |
| main | 11.5M (11.4 / 11.5) | 0.885 | 0.955 | 0.970 | 0.990 | 0.990 (0.995 / 0.986) |
| noshape | 15.2M (16.0 / 14.3) | 0.790 | 0.981 | 0.993 | 0.991 | 0.989 (0.993 / 0.985) |
| grid4 | 16.0M (20.4 / 11.7) | 0.785 | 0.964 | 0.982 | 0.985 | 0.988 (0.997 / 0.979) |
| lstm | 16.2M (14.4 / 17.9) | 0.835 | 0.950 | 0.986 | 0.990 | 0.994 (1.000 / 0.989) |
| lr1e-3 | 16.8M (13.3 / 20.3) | 0.696 | 0.905 | 0.936 | 0.701 | **0.688 (0.925 / 0.451)**: s1 collapsed at ~85M (0.982 @75M → 0.510 @100M), s0 oscillates 0.75–0.95 |
| lr1e-4 | 17.2M (15.4 / 19.1) | 0.682 | 0.967 | 0.992 | 0.995 | 0.994 (0.991 / 0.996) |
| ent | 19.3M (18.1 / 20.6) | 0.525 | 0.953 | 0.986 | 0.998 | 0.990 (0.991 / 0.990) |
| flat | 34.1M (38.7 / 29.4) | 0.348 | 0.687 | 0.956 | 0.973 | 0.980 (0.973 / 0.988) |

Per-seat Wilson intervals at 0.99 over 500 matches are about ±0.01. Every configuration except lr1e-3
and flat is within 0.988–0.997 at 100M: a ceiling tie.

Answers to the plan's ablation questions (TRAINING_PLAN §8.1):
- **noshape: does the denser reward speed learning?** Yes, modestly. 0.90 is reached at 11.5M with
  shaping vs 15.2M without; the final score is the same.
- **grid2 / grid4: how much does placement precision matter?** Against this bot, not at all, and
  2×2 blocks learn fastest (7.4M, every mark ≥ main on both seats). 4×4 blocks are no faster (16.0M)
  and have the largest seed spread. The bot cannot test whether coarse placement costs skill against
  stronger opponents.
- **lstm: is memory needed?** No. Same final score, slower (16.2M).
- **flat: does the conditional head help?** Yes, strongly. 3× slower to 0.90 (34.1M), 13–16M steps
  to pass G2 (6.6M for every other config), and the lowest healthy final score (0.980).
- **lr:** 3e-4 is best. 1e-4 is slower (17.2M); 1e-3 is unstable (collapse).
- **ent split** (card 0.01 / position 0.002): slower (19.3M) with no final gain; keep the single
  `ent_coef 0.01`.


## SPS research (2026-10-04; branch `sps-research`, report `research/REPORT.md` there)

| date | commit | job | node / GPU | GPU-h | what | outcome |
|---|---|---|---|---|---|---|
| 2026-10-04 | sps-research (fork of 7577a7c) | 19163675 `pr-sps-base1` | gh012 H100 (EPYC 9454, 24 CPUs) | 0.50 | baseline time breakdowns: stage1 / A / ckpt16 / ckpt64 at 512 and 256 envs, cuDNN benchmark on/off, `OMP_NUM_THREADS=1` | the C engine (one core) is 74–80% of a step; B/C add ~1.4 ms of host time per opponent group; ckpt64 1,748 SPS on 15.4 cores (mostly OpenMP spin) |
| 2026-10-04 | sps-research | 19164429 `pr-sps-var1` | gh004 H100 | 0.19 | prototypes + GPU correctness checks | best package (threaded engine, GPU obs, GPU sampling, batched checkpoints, CUDA graphs): A 4,696 → 18,888 SPS (4.0×), ckpt16 3,084 → 16,396 (5.3×), ckpt64 1,748 → 14,802 (8.5×); all checks pass |
| 2026-10-04 | sps-research | 19163611, 19164342, 19164445, 19164618, 19164695, 19164815 | CPU (Xeon 8592+) | 0 | threaded-step equivalence, test-c, golden hashes, build flags, OpenMP benchmark | `vec_step_range` bit-identical at 2–16 threads (7.1× engine at 16); test-c 184,647 checks 0 failures; golden PASS; `-O3 -march=x86-64-v3` −14% engine, bit-exact |

**Stage-3 wall clock per run at 1B steps:**
- A: 59 h → 14.7 h;
- B/C: 159 h → 18.8 h.

Every code change awaits the user's approval; none is on `hpc-training`.

The calibration's lower B/C baselines (job 19149010) remain unexplained; cuDNN autotuning was ruled out (benchmark off: 1,726 vs 1,748 SPS).
## Stage-3 verification (D90 throughput package, D91 eval sidecar, stage-3 launcher; 2026-10-04)

Before launch, every check ran on H100 nodes (24 CPUs, GPU heartbeat on) unless marked CPU. Commits:
- f25ca58 / 360406a: first implementation;
- e145395: graph-pool fix;
- d388d4f: review fixes. **The verified code: launch code is identical except docs and tests.**
- b8c92fd / 330f06f: test fixes only.

| job | commit | what | result | GPU-h |
|---|---|---|---|---|
| 19172122 / 19172123 (CPU) | f25ca58 | CPU tests, golden hashes, tests/spec/test_league.py; builds `-O3 -march=x86-64-v3` / `-O2` | golden PASS (both builds); test_league 29/29 (both); vec_step_range bit-identical (grid 1, 2); card stats exact (grid 1, 2, 4). 2 failures were a test bug (220 steps never finished a match); fixed | 0 |
| 19172221 | 360406a (`-O2`) | build, isolation, golden, fast-tests, smoke-league | golden PASS, smoke PASS; fast-tests 13/14: **a shared CUDA-graph pool failed an allocator assert once its graphs were freed** → fixed in e145395 (one pool per cache) | 0.11 |
| 19172220 | 360406a (`-O3 -march`) | full gate (test-c, asan, pytest, golden, e2e, smokes) | test-c 184,647 checks 0 failures; asan 184,431 0 failures; golden PASS; pytest 1,867 passed / 4 failed and e2e 19/21, **the same 4 tests and 2 checks as D81** (perf stress 8,599 ticks/s vs D81's 6,145–8,439); smokes PASS | 0.84 |
| (code review, independent agent) | 360406a | read-only review of the whole change | **BUG: the learner's graph closure was copied into MMD's deep-copied reference → MMD KL = 0 (C ≡ B)**; 12 further risks; all addressed in d388d4f (D90) | – |
| 19173477 | d388d4f | build, test-c, pytest, golden, fast-tests, smokes | test-c 0 failures; golden PASS; pytest 1,867 passed / the same 4 D81 failures; smokes PASS; fast-tests 18/19: the failing RNG-repeat test was vacuous (all opponents waited in the deploy lockout); fixed in tests only | 0.61 |
| 19173478 (array, 2 tasks) | d388d4f | end-to-end stage-3 smoke through hpc/stage3_submit.sh: A and C, 8M steps, snapshot every 5 epochs (≈ 100 pool rebuilds), eval every 25 epochs, wandb | A 18,962 SPS (pool = self only), C 12,619 SPS (pool 9). **C's MMD KL > 0** at every epoch (0.0009 → max 0.107). 19/19 eval points each; wandb runs `smk3_*-eval`; DONE / sidecar stop, tournament OK. GPU memory flat (6.7–7.4 GB) across rebuilds | 0.36 |
| 19175308 (array) | d388d4f | resume test: the same runs re-submitted to 10M | both resumed from epoch 489; history 1–611 with no duplicates; eval history continued with no duplicates, same wandb run; device RNG state saved and loaded; MMD reference restored (KL 0.028) | 0.15 |
| 19175968 (array) | d388d4f | grid2 C smoke, fast path | placement_grid 2, 12,458 SPS, pool 9, MMD KL > 0, 19 eval points | 0.21 |
| 19167543 / 19167544 | 4542acd | plain-path C smoke, grid 1 / grid 2 | both complete; MMD KL > 0; grid-2 snapshots and masked KL fine | 0.25 |
| 19173479 | d388d4f | learning equivalence: stage-1 replicate of S1main_s0's fixed-budget run (same init, seed, settings), fast path | 0.90 vs heuristic at 9.7M (original runs: 11.4M / 11.5M); score @5/11/20M 0.819 / 0.913 / 0.948 vs 0.806 / 0.892 / 0.948; 16,311 vs 5,990 SPS | 0.35 |
| 19175969 / 19175970 | d388d4f | self-play A, plain vs fast, seed 0, 10M steps, eval sidecar | 6,292 vs 19,872 SPS; both end at heuristic 0.25 / random 1.00 / noop 1.00. Mid-run trajectories differ (different opponent random stream), so more seeds were run (below) | 0.60 |
| 19179157 | b8c92fd | fast-tests | 19/20; **GPU vs CPU sampler on real masked logits: same distribution** (TV vs CPU no larger than CPU vs CPU over 200,000 draws per row; 0 illegal draws); the RNG-repeat guard was too strict (a random-init policy mostly waits); fixed | – |
| 19179160 / 19179163, 19179874 / 19179875 | b8c92fd / 330f06f | self-play A, plain vs fast, seeds 1 and 2 (10M steps) | **no systematic difference across 3 seed pairs**: final entropy plain 0.31 / 0.46 / 0.48 vs fast 0.55 / 0.40 / 0.20; final score vs heuristic plain 0.25 / 0.45 / 0.05 vs fast 0.25 / 0.12 / 0.40 (fast seed 2 repeats plain seed 0's early entropy drop); random 0.85–1.00 everywhere; SPS plain 4,201–6,292, fast 18,743–19,872 | 1.46 |
| 19179871 | 330f06f | fast-tests | **20/20 PASS**; golden PASS | 0.02 |

**Verdict:**
- Every check passes on the launch code.
- The default paths are unchanged: the full suite fails only the 4 D81 tests and 2 e2e checks.
- The fast paths are bit-identical where designed to be, and equal in distribution where they change the random stream.
- They learn like the default paths: the stage-1 replicate is within seed spread, and self-play shows no systematic difference over 3 seed pairs. They run 2.7–4.5× faster.

**Seed variance to note for stage 3:** self-play A at 10M steps varies widely between seeds (score vs heuristic 0.05–0.45), so 3 seeds per method give limited power (TRAINING_PLAN §9: report per-seed values and effect sizes).

**GPU-hours of the verification:** 5.26 (the jobs above plus the heartbeat tests 19154132 / 19155336, 0.28), plus the SPS research 0.69.

## Stage-3 pre-registration (gate G-S3; TRAINING_PLAN §8.3, HPC_HANDOFF §9; written before launch, 2026-10-04)

The user approved launching after full verification ("Once everything is done - and verified (and
meticulously checked) - I hope you launch the training runs required"), with 1B learner steps per run and
4 GPUs at a time (their choices, 2026-10-04). Nothing below is changed after seeing results without
logging it as a deviation.

**Hypotheses** (TRAINING_PLAN §2, verbatim):
- **H1 (RQ1):** exploitability(B) < exploitability(A) at equal learner-step budgets.
- **H2 (RQ2):** exploitability(C) < exploitability(B) at equal budgets.
- **H0:** no measurable difference (plain self-play suffices; tested by the transitivity diagnostic E4).

**Runs.** Random init. Hog 2.6 mirror. No scripted bots anywhere in training.
- Seeds 0, 1, 2: the same seed gives the same initial weights in A, B and C.
- Run settings:

  | run | opponents in training | MMD |
  |---|---|---|
  | A | self only (`--anchors none --self-play-frac 1.0 --anchor-frac 0`) | – |
  | B | 35% self, 65% snapshots by PFSP "hard" (`--anchors none --self-play-frac 0.35 --anchor-frac 0 --pfsp hard`) | – |
  | C | as B | `--mmd-coef 0.05 --mmd-ref-interval 50` |
  | Clo / Chi (seed 0 only) | as B | `--mmd-coef 0.01` / `0.1`, ref interval 50 |

- **Budget:** 1,000,000,000 learner steps per run (equal for every run).
- **Sizes** (the frozen stage-1 sizes, D84): 256 envs, batch 16,384, minibatch 1,024. That gives
  E = 61,036 epochs (the last one completes the 1B).
- **Snapshots:** `--snapshot-interval 477` (≈ E/128, ~128 snapshots) with `--max-snapshots 64`, so the
  B/C pool spans the last 64 × 477 = 30,528 epochs = 50.0% of training. A writes the same snapshots for
  evaluation and never samples them.
- **Analysis points:**
  - epochs 15,264 / 30,528 / 45,792 (snap_*, ≈ 25 / 50 / 75%);
  - the final model (100%).
  - Snapshot files are never deleted.
- **Frozen stage-1 settings (D89, "main"):**
  - conditional card-first head, feed-forward, `placement_grid 1`;
  - `royale.ini` hyperparameters: lr 3e-4 constant, γ 0.999, λ 0.95, clip 0.2, vf 2.0, ent 0.01,
    bptt 64, 1 update epoch;
  - `--train.torch-deterministic=` (D82).
- **Shaping:**
  - `reward_tower 0.3`, `reward_crown 0.2`, `reward_elixir 0.01`, `--train.reward-clip 0`;
  - annealed to 0 over the first 50% (`--shaping-anneal-frac 0.5`);
  - evaluation always uses true match outcomes.
- **Throughput (D90):**
  - `--env-threads 16 --fast-opponents --cuda-graphs`, OMP threads 1;
  - C extension built with `-O3 -march=x86-64-v3`;
  - same code for every run. GPU opponent sampling is a different random stream from the CPU path, with
    the same distribution.
- **Monitoring (D91):** the learner vs the heuristic / random / noop bots every 100 epochs, 50 matches
  per seat and bot. wandb runs `<RUN_ID>-eval`, groups `S3_<run>`. Evaluation only; not E3.
- **Resources:**
  - one H100 per run; 24 CPUs, 96 GB;
  - 72 h wall time (QOS gpu168: 4 GPUs per user; gpu48 had fallen to 2);
  - ONE job array `--array=0-10%4` in the order s0 A B C, s1 A B C, s2 A B C, Clo, Chi.
  - Requeue or re-submission resumes every run from its last save (every 477 epochs).

**Evaluation protocol (TRAINING_PLAN §9).** Sampled actions, both seats, 95% Wilson intervals; bootstrap
over seeds and matches for method-level numbers.
- **E1 exploitability (headline).**
  - `best_response.py --target <final> --init-from-target`, 2 BR seeds, `--matches 1000` (500 per seat),
    under the target's own settings (config.json).
  - BR budget calibrated before E1 on A seed 0's final policy at 50M / 150M / 400M BR steps: the
    smallest budget where `br_score` gains < 0.02 at the next level, then used for all 9 targets.
  - Scratch-start BR on a subset as a robustness check.
  - BR runs get `--train.torch-deterministic=`.
- **E2:** round robin of the 9 finals and the 9 50% checkpoints, `tournament.py --decks hog26 --deck-mode
  mirror --matches 200 --sample`. Payoff matrix, Elo, Nash mixture; cycles reported.
- **E3:** `eval.py --bots noop random heuristic --seats 0 1 --episodes 500` on each final.
- **E4:** per run, 10 evenly spaced snapshots in a `tournament.py --matches 100 --sample` round robin.
- **E5:** qualitative (watch.py, per-card play rates).

**Declared additions (2026-10-07, before any E1–E5 result; user-approved, D96).** These only add
measurements; nothing above changes.
- **E1 for the two sensitivity runs:** S3_Clo_s0 and S3_Chi_s0 get the same probe as the 9 targets
  (calibrated budget, BR seeds 0 and 1, `--init-from-target`, 1000 matches). They are reported as the
  exploitability-vs-MMD-strength curve on seed 0 (B_s0 = 0, Clo = 0.01, C_s0 = 0.05, Chi = 0.1), and
  never pooled into method C.
- **E3 and E4 for the two sensitivity runs:** the same bot evaluation and within-run round robin, for
  the same reason. E2 stays exactly as above (the 9 finals + the 9 50% checkpoints).
- **Code:** every E1–E5 job runs on the pinned stage-3 checkout (`PR_CHECKOUT=$PR_ROOT/pinned/stage3_4865c0e`).
  Since 4865c0e only `docs/` has changed on `hpc-training`.

**Engine freeze.**
- The launch commit (below) is the stage-3 engine: no gameplay-affecting change while stage 3 runs.
- A necessary fix starts a new lineage and invalidates the comparison runs started before it.


## Stage-3 runs

| date | commit (engine freeze) | job | GPU | GPU-h | submit line | runs | outcome |
|---|---|---|---|---|---|---|---|
| 2026-10-04 18:42 | 4865c0e (code = verified d388d4f) | 19182223 (array 0–10, %4) | H100, 24 CPUs | 200.2 (sum of task wall times, incl. each task's closing tournament) | `source $PR_ROOT/env.sh; cd $PR_ROOT/logs; PR_CHECKOUT=$PR_ROOT/pinned/stage3_4865c0e TOTAL=1000000000 NUM_ENVS=256 BATCH_SIZE=16384 MINIBATCH_SIZE=1024 SNAPSHOT_INTERVAL=477 SUBMIT=1 bash $PR_CHECKOUT/hpc/stage3_submit.sh` → `sbatch --array=0-10%4 --job-name=pr-S3 --cpus-per-task=24 --constraint=h100 --mem=96G --time=72:00:00 .../stage3_array.sbatch $DATA_DIR/stage3/S3/sub_20261004_184226/manifest.txt` | tasks 0–10 = S3_{A,B,C}_s0, S3_{A,B,C}_s1, S3_{A,B,C}_s2, S3_Clo_s0, S3_Chi_s0; wandb groups S3_<run> (+ `<run_id>-eval`) | **All 11 tasks COMPLETED, exit 0**, 2026-10-05 10:16 → 2026-10-07 03:26; every run reached 1,000,013,824 steps (epoch 61,036) with no non-finite loss and no early stop. Results below |

To resume after a failure: submit the same command again; each run resumes from its last save. The
launcher refuses while a pr-S3 job is queued, and league.sbatch refuses a second trainer on a run.

### Stage-3 training results (monitoring only; the pre-registered evaluation is E1–E5)
Recorded 2026-10-07 from the job logs, `eval_watch.log` and `league_state.json`. None of these numbers
tests H1/H2. The closing tournament and the pool scores only compare a run with itself.

| run | method | seed | MMD | wall h | sidecar vs heuristic, last check (min of last 10) | first sidecar check ≥ 0.90 vs heuristic | closing tournament: top-Elo snapshot (Elo) | Nash support | heuristic wins vs the last 4 snapshots | score vs own pool: all / newest 8 | pool snapshots it scores < 0.5 against |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S3_A_s0 | A | 0 | – | 15.5 | 1.000 (0.980) | 67.2M | snap_60579 (1881) | snap_59148 | 0/80 | – | – |
| S3_A_s1 | A | 1 | – | 15.7 | 1.000 (0.980) | 49.2M | snap_60579 (1872) | snap_60579 | 0/80 | – | – |
| S3_A_s2 | A | 2 | – | 16.3 | 1.000 (0.960) | 54.1M | snap_59625 (1924) | snap_60579 | 0/80 | – | – |
| S3_B_s0 | B | 0 | – | 18.5 | 0.990 (0.980) | 67.2M | snap_60102 (1907) | snap_59148 | 0/80 | 0.685 / 0.510 | 4/64 |
| S3_B_s1 | B | 1 | – | 19.4 | 1.000 (0.990) | 34.4M | snap_60102 (1905) | snap_60102 | 0/80 | 0.703 / 0.604 | 1/64 |
| S3_B_s2 | B | 2 | – | 18.7 | 0.980 (0.980) | 77.0M | snap_60579 (1962) | 59148 0.75, 60102 / 60579 0.125 each | 0/80 | 0.622 / 0.520 | 5/64 |
| S3_C_s0 | C | 0 | 0.05 | 18.7 | 1.000 (0.990) | 54.1M | snap_60579 (1903) | snap_60579 | 0/80 | 0.712 / 0.592 | 0/64 |
| S3_C_s1 | C | 1 | 0.05 | 19.4 | 0.990 (**0.895**) | 32.8M | snap_59625 (1999) | snap_59625 | **3/80** | 0.758 / 0.583 | 2/64 |
| S3_C_s2 | C | 2 | 0.05 | 19.4 | 1.000 (0.990) | 77.0M | snap_60102 (1881) | snap_59625 | 0/80 | 0.716 / 0.542 | 2/64 |
| S3_Clo_s0 | C-like | 0 | 0.01 | 19.2 | 1.000 (0.990) | 37.7M | snap_60579 (1933) | snap_60579 | 0/80 | 0.679 / 0.549 | 4/64 |
| S3_Chi_s0 | C-like | 0 | 0.1 | 19.4 | 1.000 (1.000) | 57.3M | snap_60579 (1884) | snap_60579 | 0/80 | 0.726 / 0.587 | 0/64 |

- **Sidecar** (D91): every 100 epochs, 50 matches per seat against each bot. **Closing tournament:**
  league.sbatch's round robin of the 4 newest snapshots plus the 3 bots, 20 matches per pair,
  sampled actions.
- **Pool scores** are cumulative over the time each snapshot sat in the pool, against the learner as
  it was then, weighted by PFSP. They are not a final-model measurement and are not comparable across
  runs. In this window C-type runs beat more of their own past snapshots than B (0–2 vs 1–5 snapshots
  below 0.5). That is a hint of less cycling, to be tested by E2/E4.
- **Flags for E3** (500 episodes per seat):
  - S3_C_s1: its sidecar dipped to 0.895 vs the heuristic in the last 10 checks, and the heuristic
    took 3 of 80 games from its last snapshots.
  - Several runs dipped against the do-nothing bot at single checks (S3_A_s0 0.905 at its last check;
    S3_A_s2 0.960). That fits a policy that plays for the tiebreak and sometimes fails to take
    damage off a passive opponent.
- **Final models:** `runs/league/<run>/model_061036.pt`. The pre-registered 50% checkpoints are
  `snap_30528.pt`, and the 25 / 75% analysis snapshots are `snap_15264.pt` and `snap_45792.pt`. All
  four files were checked to exist in all 11 run directories on 2026-10-07.

## Stage-3 evaluation (E1–E5, started 2026-10-07; D96, D97)

**Code.** E2–E4 run on the pinned stage-3 checkout `stage3_4865c0e` (CPU jobs). E1 runs on `e1_ae3b6fc`,
which has the same engine and adds the BR throughput flags (D97).

**Jobs submitted 2026-10-07.**

| what | job(s) | where | settings | status |
|---|---|---|---|---|
| BR smoke (plain path) | 19346156 | H100 | A_s0, 3M steps, 20 matches | done: ~6.2k steps/s, br_score 0.45 (20 matches; smoke only) |
| E1 pre-flight | 19348569 | H100 | `pytest hpc/tests` + fast BR smoke, 3M steps | done: 23 passed; ~18.1k steps/s |
| CPU timing | 19345978 | CPU | 2-agent tournament, 20 matches | done: 2.5 s per match |
| **E1 calibration** | 19349431 / 19349432 / 19349435 | H100 ×3 | S3_A_s0, BR seed 0, 50M / 150M / 400M, 1000 matches | 50M: br_score **0.6885** (681-15-304), 1 h 27 min; 150M: **0.9225** (920-5-75), 2 h 49 min; gain 50→150 = 0.234 ≥ 0.02, so 50M is ruled out; 400M: **0.609** (594-30-376), 6 h 04 min; gain 150→400 = **−0.3135** < 0.02 → **budget 150M** (rule applied 2026-10-07 19:40) |
| **E1 probes** | 19393469 (array 0–23, %4; `logs/e1_probe_jobs_20261007.txt`) | any of A100/H100/H200, 16 CPUs | 150M, 1000 matches, pinned `e1_43de9fd` (D98: W&B group `E1`); tasks 0–9 seed 0 (all targets but A_s0), 10–20 seed 1 (all 11), 21–23 scratch-start (`--rnn`, no `--init-from-target`) on A_s0 / B_s0 / C_s0, seed 0. The A_s0 seed-0 probe is calibration run 19349432 | all 24 COMPLETED (first started 2026-10-07 19:44, last ended 2026-10-08 19:00; 2 h 20 min – 4 h 35 min each) |
| E1 calibration re-check | 19393474 | CPU | target + the three calibration BRs, `tournament.py` 200 matches/pair, seed 7 (diagnostic, not part of E1) | COMPLETED (1 h 02 min; `runs/eval/e1_calib_recheck_seed7.json`): vs target 50M **0.690**, 150M **0.905**, 400M **0.657**; 50M vs 150M 0.662, 50M vs 400M 0.800, 150M vs 400M 0.552 (row agent's score) |
| **E2** round robin | 19346352 | CPU (`logs/eval_cpu.sbatch`) | 9 finals + 9 `snap_30528`, 200 matches/pair, `--deck-mode mirror` | COMPLETED 2026-10-08 12:18 (24 h 13 min; 30,600 matches; `runs/eval/stage3_roundrobin.json`) |
| **E3** bots | 19346353 … 19346374 (`logs/e3e4_jobs_20261007.txt`) | CPU | 11 runs, noop/random/heuristic, seats 0 and 1, 500 episodes each | all 11 COMPLETED (21–48 min each) |
| **E4** transitivity | 19346354 … 19346375 | CPU | 11 runs, 10 snapshots, 100 matches/pair | all 11 COMPLETED (2 h 59 min – 5 h 25 min; the C, Clo and Chi jobs ran on ga011/ga013 at ~6–7 min per pair vs ~4.4 on cs nodes) |

The calibration rule (pre-registered): take the smallest budget whose `br_score` gain at the next level
is < 0.02. The calibration run at the chosen budget is the A_s0 seed-0 probe itself; no rerun.

**E3 results** (both seats pooled, 1000 matches per bot; W-D-L, score):

| run | vs noop | vs random | vs heuristic |
|---|---|---|---|
| S3_A_s0 | 962-38-0 (0.981) | 1000-0-0 (1.000) | 999-0-1 (0.999) |
| S3_A_s1 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 994-0-6 (0.994) |
| S3_A_s2 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 996-0-4 (0.996) |
| S3_B_s0 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 997-0-3 (0.997) |
| S3_B_s1 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 999-0-1 (0.999) |
| S3_B_s2 | 998-2-0 (0.999) | 1000-0-0 (1.000) | 999-0-1 (0.999) |
| S3_C_s0 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 997-0-3 (0.997) |
| S3_C_s1 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 981-1-18 (0.982) |
| S3_C_s2 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 1000-0-0 (1.000) |
| S3_Clo_s0 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 997-0-3 (0.997) |
| S3_Chi_s0 | 1000-0-0 (1.000) | 1000-0-0 (1.000) | 1000-0-0 (1.000) |

**E4 results** (10 evenly spaced snapshots, 45 pairs × 100 matches; margin 0.05):

| run | later beats earlier | cyclic triads | Elo first → last | Nash support (snapshot index: weight) |
|---|---|---|---|---|
| S3_A_s0 | 0.933 (42/45) | 0/120 | 491 → 1757 (max 1758) | 9: 1.00 |
| S3_A_s1 | 0.911 (41/45) | 0/120 | 1066 → 1765 (max 1781) | 9: 1.00 |
| S3_A_s2 | 0.867 (39/45) | 0/120 | 479 → 1812 (max 1812) | 7: 0.31, 8: 0.15, 9: 0.54 |
| S3_B_s0 | 1.000 (45/45) | 0/120 | 614 → 1856 (max 1856) | 9: 1.00 |
| S3_B_s1 | 0.978 (44/45) | 0/120 | 527 → 1985 (max 1985) | 9: 1.00 |
| S3_B_s2 | 0.978 (44/45) | 0/120 | 568 → 1915 (max 1915) | 9: 1.00 |
| S3_C_s0 | 1.000 (45/45) | 0/120 | 404 → 1956 (max 1956) | 9: 1.00 |
| S3_C_s1 | 0.933 (42/45) | 9/120 | 580 → 1798 (max 1798) | 1: 0.37, 4: 0.29, 9: 0.34 |
| S3_C_s2 | 1.000 (45/45) | 0/120 | 488 → 2024 (max 2024) | 9: 1.00 |
| S3_Clo_s0 | 0.978 (44/45) | 0/120 | 605 → 1824 (max 1824) | 9: 1.00 |
| S3_Chi_s0 | 1.000 (45/45) | 0/120 | 471 → 2070 (max 2070) | 9: 1.00 |

**E2 results** (18 agents, 153 pairs × 200 matches, sampled actions, mirror hog26; Elo regularised; Nash
of the full 18×18 game). `@50%` is the pre-registered 50% checkpoint `snap_30528`.

| agent | Elo | Nash | mean score | | agent | Elo | Nash | mean score |
|---|---|---|---|---|---|---|---|---|
| A_s1 | 1683.5 | 0.597 | 0.741 | | A_s2 | 1499.6 | 0 | 0.501 |
| C_s0 | 1678.8 | 0 | 0.735 | | B_s0 | 1483.2 | 0 | 0.478 |
| C_s2 | 1664.9 | 0.013 | 0.719 | | B_s1 | 1469.9 | 0 | 0.459 |
| C_s1@50% | 1574.3 | 0 | 0.604 | | A_s0@50% | 1434.1 | 0 | 0.410 |
| C_s0@50% | 1573.2 | 0 | 0.603 | | A_s1@50% | 1409.5 | 0 | 0.376 |
| A_s0 | 1558.3 | 0 | 0.583 | | C_s2@50% | 1396.5 | 0 | 0.359 |
| B_s2 | 1546.4 | 0 | 0.566 | | A_s2@50% | 1392.7 | 0 | 0.354 |
| C_s1 | 1506.4 | 0.389 | 0.511 | | B_s1@50% | 1390.4 | 0 | 0.351 |
| | | | | | B_s2@50% | 1381.6 | 0 | 0.340 |
| | | | | | B_s0@50% | 1356.9 | 0 | 0.309 |

Method vs method (mean of the 9 cross-seed pairs, row method's score; raw, CIs come at gate G-E):

| | finals | 50% checkpoints |
|---|---|---|
| A vs B | 0.553 (pairs 0.40–0.79) | 0.585 (0.41–0.72) |
| A vs C | 0.546 (0.10–0.83) | 0.327 (0.10–0.57) |
| B vs C | 0.315 (0.09–0.68) | 0.384 (0.11–0.67) |

E2 observation (raw): seed spread within a method is large (A finals' mean scores 0.501–0.741; C
0.511–0.735). The Nash support is A_s1 and C_s1: A_s1 beats every other final except C_s1 (0.49), and
C_s1 is the one that holds it to even.

**E1 results** (150M steps, `--init-from-target` unless noted, 1000 sampled matches, half per seat;
`runs/eval/br_*.json`; the A_s0 seed-0 probe is calibration run 19349432). Raw; bootstrap CIs and the
method test come at gate G-E.

| target | seed 0: W-D-L, br_score | seed 1: W-D-L, br_score | mean | proxy (2·mean − 1) |
|---|---|---|---|---|
| S3_A_s0 | 920-5-75, 0.9225 | 839-2-159, 0.8400 | 0.8812 | +0.762 |
| S3_A_s1 | 862-0-138, 0.8620 | 810-0-190, 0.8100 | 0.8360 | +0.672 |
| S3_A_s2 | 917-0-83, 0.9170 | 891-0-109, 0.8910 | 0.9040 | +0.808 |
| S3_B_s0 | 918-0-82, 0.9180 | 787-1-212, 0.7875 | 0.8528 | +0.706 |
| S3_B_s1 | 991-0-9, 0.9910 | 924-0-76, 0.9240 | 0.9575 | +0.915 |
| S3_B_s2 | 788-0-212, 0.7880 | 674-0-326, 0.6740 | 0.7310 | +0.462 |
| S3_C_s0 | 877-0-123, 0.8770 | 864-1-135, 0.8645 | 0.8708 | +0.742 |
| S3_C_s1 | 960-0-40, 0.9600 | 927-2-71, 0.9280 | 0.9440 | +0.888 |
| S3_C_s2 | 724-2-274, 0.7250 | 825-0-175, 0.8250 | 0.7750 | +0.550 |
| S3_Clo_s0 | 993-0-7, 0.9930 | 989-0-11, 0.9890 | 0.9910 | +0.982 |
| S3_Chi_s0 | 962-0-38, 0.9620 | 643-2-355, 0.6440 | 0.8030 | +0.606 |

- A: mean br_score 0.8737 (targets 0.881, 0.836, 0.904)
- B: mean br_score 0.8471 (targets 0.853, 0.958, 0.731)
- C: mean br_score 0.8632 (targets 0.871, 0.944, 0.775)

| scratch-start probe (`--rnn`, seed 0) | W-D-L | br_score |
|---|---|---|
| S3_A_s0 | 0-26-974 | 0.0130 |
| S3_B_s0 | 894-3-103 | 0.8955 |
| S3_C_s0 | 16-0-984 | 0.0160 |

E1 observation (raw):
- Seed spread on one target is up to 0.32 (S3_Chi_s0: 0.962 vs 0.644), larger than the gaps between
  method means (0.847–0.874).
- Training vs test. Score over each probe's last 0.5M training steps vs its final `br_score`: 20 of 24
  agree within the 95% noise band. Four do not: S3_C_s2 seed 0 (0.864 → 0.725), S3_B_s0 seed 1 (0.845 →
  0.788), S3_C_s0 seed 0 (0.832 → 0.877), S3_C_s0 seed 1 (0.895 → 0.865, just outside). (Script:
  `$PR_ROOT/logs/wb_scan3.py` on the local W&B files.)
- The win rate swings during training, reading (a). S3_Chi_s0 seed 1, 5M-step windows: 0.65 (55–60M),
  0.37 (80–85M), 0.75 (125–135M), 0.68 (145–150M); its test score is 0.644.
- Scratch-start probes: a fresh learner does not beat S3_A_s0 or S3_C_s0 in 150M steps (training
  score ≤ 0.04 in every full 5M window up to 150M). It does beat S3_B_s0: training score rises from 0.10 at 35–40M to
  0.85 by 150M; test 0.8955.
- Confound in the scratch-start probes: they were launched with `--rnn` (an LSTM learner; config.json
  `rnn_name: Recurrent`), while the targets and the copy-start probes are feed-forward. The plan only
  says "scratch-start BR on a subset"; `--rnn` was a launch-time choice, not a recorded decision. The
  S3_B_s0 result therefore mixes two changes (random start, memory).

E1 calibration observation (raw). The BR's score is not monotone in its budget: 0.689 (50M), 0.923
(150M), 0.609 (400M), each over 1000 matches (95% CIs about ±0.03). The training score vs the target is
cumulative over all training matches and rises throughout: 0.565, 0.681, 0.711. From the cumulative
counts, the 400M run's training matches after 150M steps scored about 0.73
((0.7105·677,550 − 0.681·254,844) / 422,706; a cross-run estimate, since the two runs anneal shaping
over different step counts). So the final 400M BR scores below its own late training average, while the
150M BR scores above its average. Two readings: (a) the BR's win rate swings during training, and a
final checkpoint can land high or low; (b) the training-time opponent differs from the evaluation one.
The CPU re-check 19393474 (`tournament.py`, new match seeds, 200 matches per pair, 95% CIs about
±0.04–0.07) reproduces the pattern: the BRs score 0.690 (50M), 0.905 (150M) and 0.657 (400M) against the
target, so the non-monotone scores are not an artefact of the BR script's evaluation matches. Against each
other the order reverses: 50M beats 150M (0.662) and 400M (0.800), and 150M edges 400M (0.552). The BR that
exploits the target most (150M) is not the strongest player in general, and the 400M BR is weaker than
the 150M one against every opponent tested. This does not separate (a) from (b): it checks the evaluation
side only. First W&B curves (probes at 45–63M steps, 5M-step windows of the per-epoch training score vs
the target) mostly rise, with dips of up to 0.08 (S3_A_s1 seed 0: 0.71 over 35–40M, 0.63 over 40–50M, 0.79 over
55–60M). Comparing each probe's final `br_score` with its late training score will test (b). E1 reports the
final checkpoint, as pre-registered.

E4 observation (raw; method-level analysis comes with E1 and E2 at gate G-E): S3_C_s1 is the only run
with cyclic triads (9/120). Its snapshot 1 (`snap_7155`, ~117M steps) beats the run's last three
snapshots: 0.80 vs snapshot 7, 0.88 vs 8, 0.80 vs 9 (100 matches each). Snapshot 2 (`snap_13833`) is
close to even with the last two (0.39, 0.44). The Nash mixture is snapshot 1: 0.37, 4: 0.29, 9: 0.34. In
the other ten runs the final snapshot carries Nash weight ≥ 0.54 (1.00 in nine of them). S3_C_s1 also has
the most losses to the heuristic bot in E3 (18 of 1000).

## Stage-4 pre-registration (TRAINING_PLAN §8.4, HPC_HANDOFF §11; D93, D99; written before launch, 2026-10-09)

- **Runs** (D99): method C (primary) and method A (pure self-play ablation), seeds 0 and 1, one GPU each,
  random init (D93):

  | run | GPU | opponents in training | MMD |
  |---|---|---|---|
  | S4_C_s0 | H100/H200 | 35% self, 65% snapshots by PFSP "hard" | 0.05, ref interval 50 |
  | S4_A_s0 | H100/H200 | self only | – |
  | S4_C_s1 | A100 | as S4_C_s0 | as S4_C_s0 |
  | S4_A_s1 | A100 | as S4_A_s0 | – |

- **Budget:** 10,000,000,000 learner steps per run. **Analysis points:** the snapshots nearest 2B and 5B
  steps (`snap_122112` = 2.0007B, `snap_305280` = 5.0017B) and the final model (epoch 610,352). With a
  constant learning rate these are equivalent to separate 2B / 5B runs (D99).
- **Unchanged from stage 3** (EXPERIMENTS "Stage-3 pre-registration"): 256 envs, batch 16,384, minibatch
  1,024; lr 3e-4 constant and the other `royale.ini` hyperparameters; conditional head, feed-forward,
  `placement_grid 1`; `--train.torch-deterministic=`; the D90 throughput flags; `--snapshot-interval 477`
  with `--max-snapshots 64` (snapshot files never deleted).
- **Shaping:** `reward_tower 0.3`, `reward_crown 0.2`, `reward_elixir 0.01`, `--train.reward-clip 0`, faded
  to 0 over the first 500M learner steps (`--shaping-anneal-frac 0.05`, N = 1,953,125 env steps, as in
  stage 3).
- **Decks:** `--env.deck-pool hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge
  --env.random-deck-frac 0.5 --env.heldout-decks xbow;royal_hogs;random:50:20261003 --env.deck-draw
  independent`. The held-out string is fixed here and reused for evaluation.
- **Monitoring** (wandb groups `S4_C`, `S4_A`; D99):
  - training: one W&B call per 10 epochs (`--log-interval 10`); history.jsonl keeps every epoch;
  - every 6,104 epochs (≈100M steps), the `<run>-eval` sidecar:
    - the heuristic and random bots on the pool decks and on the held-out decks (each deck a mirror, 10
      matches per seat and deck);
    - the specialist check (D93): the stage-3 Hog 2.6 specialists S3_A_s1 and S3_C_s0 (best final per
      method by E2 Elo) in the Hog 2.6 mirror, 200 matches each, half per seat. Fair score 0.5.
  - Evaluation only: neither bots nor stage-3 policies are ever training opponents.
- **Evaluation at each analysis point** (TRAINING_PLAN §8.4; sampled actions, both seats, Wilson 95%):
  1. Generalisation: `eval.py` vs random and heuristic, pool decks vs held-out decks, ≥ 500 matches per
     seat and deck set.
  2. Deck meta-game: `deck_metagame` over the 9 presets, 200 matches per pair: deck-vs-deck matrix and Nash.
  3. Specialist check: vs S3_A_s1 and S3_C_s0 in the Hog mirror, 500 matches each.
  4. A vs C: a round robin of the four runs' checkpoints on the 9 presets, each deck a mirror.
  5. Optional: a best response on a held-out-deck mirror.
- **Resources and chaining:** one GPU per run, 24 CPUs, 96 GB; each run is a chain of 20 h jobs
  (`hpc/stage4_submit.sh`, `afterany`), resuming from the last save (every 477 epochs); pinned checkout
  `s4_6e8a976`.
- **Pilot first** (agreed): the same launcher with 50M steps, 45 min links (3 per run), 600-epoch evals:
  speed with mixed decks, all 64 cards in the play-rate logs, resume across a time-limit kill, a finished
  run's extra link doing nothing, the new evals and the 10-epoch logging.
- **Amendment before launch (D100, 2026-10-09): card-probability floor.** Every run adds `--card-floor
  0.005 --card-floor-steps 5000000000` (each legal hand slot keeps at least 0.5% of the learner's card
  choice, faded linearly to 0 by 5B steps; evaluations use the unfloored policy). Reason: the first pilot's
  spells were almost never played (Fireball 0.0001–0.0023 per chance) and stage-3 Fireball stayed unused for
  250–500M steps. The pilot is repeated with the floor; the first pilot (no floor, same seeds) is the
  comparison. Pinned checkout: the D100 commit (replaces `s4_6e8a976`).
- **Amendment before launch (D101, 2026-10-09): Claude games.** S4_C_s0 only: every 250M learner steps one
  recorded game of Claude Opus 5.5 (headless Claude Code CLI, the user's claude.ai seat) with X-Bow or Royal
  Hogs against the run's saved policy (same deck or a pool preset, alternating), 40 games, logged to the W&B run
  `S4_C_s0-llm` with the replay GIF. Exploratory: no pre-registered hypothesis rests on it.

- **Amendment (D102, 2026-10-10, before any analysis point): a second arm `S4d_*`.** The same four runs with
  constant shaping (`--shaping-anneal-frac 0`) and a constant floor (`--card-floor-steps 0`), plus Claude games
  on S4d_C_s0. Each analysis point (2B / 5B / 10B) is evaluated for both arms with the same items 1–4; S4 vs S4d
  per method and seed is a new comparison ("schedules" vs "no schedules"; shaping and floor not separated).

## Stage-4 runs (launched 2026-10-09 20:30, D99–D101)

- **Submission** `runs/stage4/S4/sub_20261009_203058/` (manifest, env files, chains.txt), pinned checkout
  `s4_d70c336`: `TOTAL=10000000000 LINKS=13 PR_CHECKOUT=$PR_ROOT/pinned/s4_d70c336 SUBMIT=1 bash
  hpc/stage4_submit.sh` (defaults: 20 h links, card floor 0.005 over 5B, LLM_RUNS C:0 every 250M).
- **Chains** (13 links of 20 h; afterany):
  - S4_C_s0 (h100|h200): 19508255 … 19508269
  - S4_A_s0 (h100|h200): 19508270 … 19508282
  - S4_C_s1 (a100): 19508283 … 19508297
  - S4_A_s1 (a100): 19508298 … 19508331
  - S4_C_s0-llm (CPU, 7 links of 2 days): 19508332 … 19508340
- **Before launch.**
  - First pilot (`S4pilot_*`, no floor, 50M, 45-min links): speed A100 ~13,900 / H100 ~17,400 steps/s with
    mixed decks; all 64 cards in the play-rate logs; W&B every 10th epoch; the sidecar's six eval keys;
    resume after a time-limit kill (C_s1 link 2 resumed at epoch 1909 from the save at 1908, history cut
    back, same W&B runs); a finished run's extra link did nothing (31 s). C_s1's draws rose to 55–62% at
    31–36M (after the pilot's 25M shaping window) and fell back to 8–23% by 46M; A_s1 stayed under 1%.
  - Floor pilot (`S4pf_*`, same seeds and settings, `--card-floor 0.005 --card-floor-steps 250000000`; both
    finished at 50.0M, 2026-10-09 23:22): every card's play rate stays at about the floor (lowest card 0.0045–0.0056
    C / 0.0045–0.0049 A over 25–50M; without: 0.0000 / 0.0005–0.0012). Resume with the floor: C_s1 link 2 resumed at
    epoch 1909 with floor 0.0044 (= 0.005 · (1 − 31.28M / 250M)), ended at 0.0040. **C_s1's passive phase is gone:**
    draws 0.0% at 31–36M / 36–41M / 41–50M (without the floor 54.4% / 29.0% / 14.0%), cards played per seat and
    game 27.4 / 25.8 / 24.0 (without 10.7 / 17.1 / 21.3). A_s1: no draws either way; 20–24 plays per game with the
    floor vs 26–28 without. Bot evals (sidecar, unfloored policy) at 39M / 49M: C_s1 heuristic held-out 0.394 /
    0.394 (without the floor 0.476 / 0.519, 208 matches each), A_s1 0.611 / 0.570 (without 0.534 / 0.596): the C
    seed is lower with the floor at this early point (one seed; to watch in stage 4, whose four runs all use the
    floor). The `cards/below_floor` alarm reads 1.5–4.4 while the floor is on, since cards sitting at the current
    floor (< 0.005) count; it becomes informative after the fade.
  - Tests: `hpc/tests` 36 passed on an H100 (job 19496820); the full `tests/spec` + `tests/builder` fail the
    same 3 tests on the pre-floor commit 6e8a976 and on ee9575a (jobs 19499024 / 19499025, same CPU model):
    `test_recurrent_train_and_eval_paths[flat-1]`, `test_reward_clip_values`, `test_e2e_check_prints_pass`
    (pre-existing; not touched).
  - Claude game verification (job 19499455, CPU): Claude (X-Bow) beat the first pilot's 50M C_s1 policy 1–0
    in 12.8 min, 161 calls, 4.1 s median latency, 0 errors, CLI estimate $2.63; W&B run S4pilot_C_s1-llm.
- **Analysis tooling (2026-10-10, before any analysis point).** `hpc/stage4_eval.py` (plan / run / merge) +
  `hpc/stage4_eval.sbatch` (CPU array, one process per CPU) implement the pre-registered items 1–4 as ~254
  single-CPU tasks per point: gen (eval.py, 72 matches per seat and deck on the 7 pool decks = 504 per seat, 10 per
  seat and deck on the 52 held-out decks = 520), meta (deck_metagame's exact matches and seeds, 200 per pair of
  the 9 presets), spec (500 per specialist), rr (200 per run pair and preset). About 60K matches per point.
  Test: `hpc/tests/test_stage4_eval.py` (tiny plan, all tasks, merge; meta equals deck_metagame).

## Stage-4 dense arm `S4d_*` (launched 2026-10-10 20:24, D102)

- **Submission** `runs/stage4/S4d/sub_20261010_202438/`, pinned checkout `s4d_f7445af` (training code identical to
  `s4_d70c336`): `PREFIX=S4d SHAPING_STEPS=0 CARD_FLOOR_STEPS=0 TOTAL=10000000000 LINKS=13
  PR_CHECKOUT=$PR_ROOT/pinned/s4d_f7445af SUBMIT=1 bash hpc/stage4_submit.sh`.
- **Chains** (13 links of 20 h; afterany):
  - S4d_C_s0 (h100|h200): 19554989 … 19555001
  - S4d_A_s0 (h100|h200): 19555002 … 19555014
  - S4d_C_s1 (a100): 19555015 … 19555033
  - S4d_A_s1 (a100): 19555034 … 19555051
  - S4d_C_s0-llm (CPU, 7 links of 2 days): 19555052 … 19555058

## Stage-4 collapse diagnostics (2026-10-15 .. 10-20; D103, D104)
- **Finding:** every league + MMD run's tile choice became deterministic and its play collapsed (vs heuristic bot
  0.95+ -> 0.24-0.74); no A run collapsed (to 5.4B). Placement-entropy onset: S4_C_s0 ~1.65B, S4_C_s1 ~1.9B,
  S4d_C_s0 ~2.2B, S4d_C_s1 ~2.7B. P(play) unchanged (0.06-0.09); top-tile probability 0.39 -> 1.00.
- **Ruled out:** a snapshot-path bug (`hpc/collapse_gpu_check.py`: fast GPU path 0.581 vs plain CPU 0.608 for
  the 1.5B learner vs five snapshots; learner graph replay exact); the floor / shaping schedules (S4d has none
  and collapses); gradual decline (ladders: each snapshot beats the one 0.25B earlier 0.60-0.80 until 1.75B).
- **The learner's sub-0.5 training score vs its snapshots is the card floor** (only the learner is floored):
  0.581 -> 0.514 with the floor on; 0.43-0.44 (S4) / 0.40-0.42 (S4d) vs snapshots 0-16M steps old.
- **Gradient probe** (`hpc/collapse_probe.py`, 9 runs): the PG term's push on the unweighted placement entropy
  is 10-50x the entropy bonus's; snapshot games do not push harder than self-play games.
- **Stage-3 contrast (no floor):** card entropy rises with the MMD coefficient (A 0.054-0.067, B 0.056-0.067,
  Clo 0.069-0.075, C 0.071-0.089, Chi 0.081-0.091) while the placement entropy is lowest in C: MMD is the lead
  suspect (its KL, like the entropy bonus, weighs the tile choice by P(play) ~ 0.08).
- **cards/below_floor** jumps with the collapse in S4 C runs (S4_C_s0 2 -> 12 between 1.5B and 2.25B): the
  policy stops choosing cards that are useless at its one tile; they are played only by the floor.
- Data: `runs/eval/collapse_diag/` (place, ladder_*, gpu_check_*.json, probe/), hpc/collapse_diag.py.

## Stage-4 collapse ablation `S4x_*` (submitted 2026-10-20 13:27, D104)
- Submission `runs/stage4/S4x/sub_20261020_132717/` (env files from S4_C_s0.env; manifest; chains.txt),
  pinned checkout `s4x_3cd765c`, `hpc/stage3_array.sbatch`, 24 CPUs, 96 GB, h100|a100, one 20 h link:
  S4x_ctrl_s0 20445563, S4x_nommd_s0 20445574, S4x_selfonly_s0 20445575, S4x_posent_s0 20445576.
- **Collapse criterion (written before any result):** placement scan top-tile probability >= 0.95 (or
  unweighted H(pos | slot) < 0.1) and score vs the heuristic bot < 0.8. The control must collapse for the arms to
  be read (the original reached top-tile 0.68 at 1.8B, 0.98 at 2.1B). **Predictions if MMD is the cause:**
  nommd stays spread past 2.1B with card entropy falling toward ~0.06; selfonly collapses; posent stays spread.
  Single seed per arm; the onset varied 1.65-2.9B across runs, so an ambiguous result needs a second seed.
- **Result (2026-10-26; one 20 h link each, all branched from S4_C_s0 snap_94446 = 1.55B).** Placement scan
  `runs/eval/collapse_diag/place_S4x` (job 21000710, 90 matches per snapshot vs bot:heuristic, 9 preset mirrors):
  - ctrl **collapses**: max p(pos | slot) 0.56 -> 0.95 (2.4B) -> 0.998 (2.7B), H(pos | slot) 1.71 -> 0.005, score
    0.989 -> 0.689 (2.4B) -> 0.317 (2.7B); sidecar heuristic pool / held-out 0.51 / 0.40 at 2.7B.
  - nommd (= method B from 1.55B: 35% self, 65% PFSP snapshots, no MMD) does not: max p 0.25-0.76, H 0.96-3.96,
    score 0.84-0.91 to 2.73B; card entropy (training log) 0.069 -> 0.052, as predicted.
  - selfonly does not (the prediction "collapses" fails within 2.89B): max p 0.45-0.72, H 1.09-2.26 (lowest at
    2.8B), score 0.94-0.98; the training log's weighted entropy_pos falls 0.12 -> 0.05 over 1.5-2.8B.
  - posent (c = 0.01) does not, by flattening placement: H 5.67-5.70 (e^H ~ 290 tiles), max p 0.015-0.021, score
    0.73-0.97; vs its pool 0.33 -> 0.46 (training log). The coefficient is too strong.
  - Reading: neither MMD alone nor snapshots alone collapsed in this window; MMD with snapshots did.

## Best A run, the floor-0.20 run `S4df20_A_s0` and Claude games on S4d_A_s0 (2026-10-20, user's request)
- **A round robin at 2.5B** (`runs/eval/stage4_2.5B_A_rr`, 9 presets mirror, 200 per pair and deck): Elo S4_A_s0
  1442, S4_A_s1 1519, S4d_A_s0 1519, S4d_A_s1 1520; S4d_A_s0 vs S4_A_s1 0.502, vs S4d_A_s1 0.501, vs S4_A_s0 0.604;
  Nash mixture 1.00 on S4d_A_s0. With the 2B evaluations (vs heuristic 0.958 pool / 0.890 held-out, vs S3_A_s1
  0.160, the best of the four) **S4d_A_s0 is taken as the best A run** (a near-tie with S4_A_s1 and S4d_A_s1).
- **S4df20_A_s0:** S4d_A_s0's configuration (method A, seed 0, constant shaping, constant floor) with
  `--card-floor 0.2`, a new run from scratch; pinned checkout `s4x_3cd765c` (ent_coef_pos_unweighted = 0, so
  training equals the S4d_A_s0 code apart from logging); `runs/stage4/S4df20/sub_20261020_134329/`, h100|a100,
  13 links of 20 h: 20446630 ... 20446681. Note: with k playable cards the floor forces a card play with
  probability >= 0.2 k per decision (0.8 at k = 4; 0.02 at the old 0.005).
- **Claude games on S4d_A_s0** (D101 settings: claude-opus-5-5, effort low, one game per 250M steps, W&B group
  S4d_A): a fresh `llm/` state, so the first link backfills 0.25B .. the latest snapshot, then follows training;
  `runs/stage4/S4d_A_llm/sub_*/`, 6 links of 2 days: 20446752 ... 20446757.
