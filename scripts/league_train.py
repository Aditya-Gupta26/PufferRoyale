#!/usr/bin/env python3
"""League training (SPEC §15.4, §15.7.12): PuffeRL -- or MMDPuffeRL when --mmd-coef > 0 -- on a
LeagueVecEnv whose opponents come from an OpponentPool (self-play, anchors, PFSP snapshots).

    # heuristic-bot anchor + self-play + PFSP snapshots of the learner, CPU
    python scripts/league_train.py --total-timesteps 200000 --anchors bot:heuristic
    # two anchors, more self-play, MMD regulariser towards a reference refreshed every 20 epochs
    python scripts/league_train.py --anchors bot:heuristic,bot:random --self-play-frac 0.4 \\
        --mmd-coef 0.05 --mmd-ref-interval 20
    # continue a run up to an absolute total of 1M learner steps
    python scripts/league_train.py --resume experiments/league/<run_id> --total-timesteps 1000000

Run directory <data-dir>/league/<run_id>/:
    snap_<epoch>.pt      learner snapshots (added to the pool), every --snapshot-interval epochs;
                         <epoch> is PuffeRL's epoch counter after train() incremented it
    league_state.json    pool state (snapshots, stats, RNG), wrapper RNG, epoch, global step, args
    learner.pt           latest learner weights; trainer_state.pt: optimizer, epoch, global step
    model_<epoch>.pt     final learner weights + config.json (loadable by eval.py / watch.py)
    history.jsonl        one line per epoch: losses and the learner's results vs each opponent
                         (games and score, plus the per-match outcomes +1/0/-1 in order)

With --resume <run dir> the run continues from its saved state; --total-timesteps is the
ABSOLUTE target (a run that already reached it does nothing). Pool-defining options (anchors,
pfsp, fractions, max snapshots) and --rnn always come from the saved run; optimizer settings
given on the command line (--learning-rate, --train.adam-* ...) are applied to the resumed
optimizer; --train.* / --env.* / --policy.* overrides of the original run are kept. The run
directory is relocatable (pool snapshot paths are stored relative to it). A resumed run is
deterministic given the saved state, but not bit-identical to an uninterrupted run: matches in
progress at the save are re-dealt (from seed + 7919 * epoch); the pool, wrapper and torch RNG
states continue from the save.

History after a preemption (SPEC §19.10.3): --resume first cuts history.jsonl back to the records
with epoch <= the saved epoch (the run re-does the later ones); a new run in a directory holding a
history.jsonl but no league_state.json moves it to history.jsonl.stale-<k>. The early-stop window
(early_stop.recent) is in league_state.json at every save, the final one included. A --resume that
would build another architecture, grid or LSTM than the saved run exits with a clear message.

Robustness (SPEC §18.7): every epoch the losses and the learner's weights are checked; a
non-finite epoch is never snapshotted or saved -- the last good save stays, nonfinite.json
describes the failure, and the process exits 1. Any error after start-up exits promptly with a
non-zero status and saves nothing.

v0.5 (SPEC §19.7):
    --init-from CKPT          start a NEW run from a checkpoint's weights only (file, snapshot or run
                              directory; new pool and optimizer); the architecture -- sizes, head, card
                              stats, placement grid, --rnn -- must match (error otherwise); not with --resume
    --shaping-anneal-frac F   anneal the reward-v2 shaping weights to 0 over the first F of
                              --total-timesteps (env: shaping_anneal_steps N = ceil(F * total / num_envs),
                              exact, shaping_step_offset = global_step // num_envs on --resume); 0 =
                              constant. N is fixed when the run is created (league args) and kept on
                              --resume, even with a larger --total-timesteps; giving --shaping-anneal-frac
                              again recomputes it from the new F and total. The env always gets
                              reward_gamma = train.gamma; shaping weights > 0 with train.reward_clip > 0
                              print a warning (use --train.reward-clip 0)
    --early-stop-score X --early-stop-window N
                              at each snapshot epoch, once every anchor has >= N finished matches and the
                              learner's score over its last N matches vs every anchor is >= X, the run
                              saves and finishes normally ("early_stopped": true in the summary); an
                              early-stopped run is finished (--resume does nothing)
    --wandb --wandb-project P --wandb-group G --tag T --no-model-upload
                              log to Weights & Biases (PufferLib's flags, through
                              pufferroyale.trainer.WandbLogger; WANDB_MODE=offline works without network):
                              one log call per history.jsonl record at step = global step, with PuffeRL's
                              environment/* stats (per-card play rates, ...). Without --wandb, wandb is
                              never imported. These flags are not stored with the run.
    --log-interval N          one wandb call per N epochs instead (D99): losses and environment/* stats
                              averaged over the N epochs, results summed (games; score = total / games),
                              epoch / global_step / sps / pool_size of the last one; history.jsonl keeps
                              every epoch. Default 1 (one call per record, unchanged).
    --card-floor EPS --card-floor-steps N
                              card-probability floor (D100, conditional head): every legal hand slot keeps
                              at least eps of the learner's card distribution (pufferroyale.torch
                              Policy.enable_card_floor), eps = EPS * max(0, 1 - step / N) for the epoch that
                              starts at learner step `step` (N = 0: constant). Set on the learner and on the
                              MMD reference before every epoch; snapshots, checkpoints and evaluations play
                              without it. history.jsonl / wandb get card_floor, wandb also
                              cards/below_floor = the number of cards whose play rate is below EPS.
                              Default 0 (off: the policy runs exactly as before).

Any PuffeRL / env / policy key of pufferroyale/config/royale.ini can also be overridden as
--train.key value, --env.key value, --policy.key value (e.g. --train.learning-rate 1e-4,
--train.reward-clip 0, --policy.head flat, --env.placement-grid 2).
--anneal-lr follows one cosine schedule over the absolute --total-timesteps (continued, not
restarted, on --resume). The last stdout line is a JSON summary (valid JSON: non-finite numbers
are written as strings).
"""
import argparse
import collections
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pufferroyale.league import json_safe  # noqa: E402  (numpy only; torch / pufferlib load lazily)
DEFAULT_CONFIG = os.path.join(ROOT, "pufferroyale", "config", "royale.ini")

# options fixed by the run: on --resume they always come from the saved run (the pool state must
# be loaded into a pool built with the same constructor arguments, SPEC §15.7.3, and the saved
# weights need the same architecture)
POOL_KEYS = ("anchors", "pfsp", "pfsp_eps", "self_play_frac", "anchor_frac", "max_snapshots", "rnn")
DEFAULTS = dict(total_timesteps=1_000_000, device="cpu", num_envs=16, anchors="bot:heuristic", pfsp="hard",
                pfsp_eps=0.05, self_play_frac=0.2, anchor_frac=0.2, snapshot_interval=10, max_snapshots=16,
                mmd_coef=0.0, mmd_ref_interval=0, seed=0, data_dir="experiments", deck0="hog26", deck1="hog26",
                frame_skip=None, bptt_horizon=None, batch_size=None, minibatch_size=None, learning_rate=None,
                rnn=False, opponent_greedy=False, anneal_lr=False, run_id=None, print_interval=1,
                shaping_anneal_frac=0.0, init_from=None, early_stop_score=None, early_stop_window=2000,
                shaping_anneal_steps=None,      # N of SPEC §19.10.1: set when the run is created (not a flag)
                env_threads=0, fast_opponents=False, cuda_graphs=False, eval_export_interval=0,   # D90 / D91
                log_interval=1,                                                                  # D99
                card_floor=0.0, card_floor_steps=0)                                              # D100
# PufferLib logging flags: honoured for the current invocation, never stored with the run (a resumed run
# logs only when asked to again); value = number of arguments the flag takes
LOGGING_FLAGS = {"--wandb": 0, "--no-model-upload": 0, "--wandb-project": 1, "--wandb-group": 1, "--tag": 1,
                 "--neptune": 0, "--neptune-name": 1, "--neptune-project": 1}


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-timesteps", type=int, help="learner agent steps (absolute target on --resume); default 1M")
    ap.add_argument("--device", help="torch device: cpu (default), cuda, mps")
    ap.add_argument("--num-envs", type=int, help="parallel matches = learner rows (default 16)")
    ap.add_argument("--anchors", help="comma-separated anchor specs: bot:noop|bot:random|bot:heuristic|ckpt:<path> "
                                      "('none' for no anchors); default bot:heuristic")
    ap.add_argument("--pfsp", choices=["hard", "variance", "uniform"], help="PFSP weighting of snapshots (default hard)")
    ap.add_argument("--pfsp-eps", type=float, help="PFSP weight floor (default 0.05)")
    ap.add_argument("--self-play-frac", type=float, help="P(opponent = current learner) (default 0.2)")
    ap.add_argument("--anchor-frac", type=float, help="P(opponent = an anchor) (default 0.2)")
    ap.add_argument("--snapshot-interval", type=int, help="snapshot the learner into the pool every N epochs "
                                                          "(0 = never; default 10)")
    ap.add_argument("--max-snapshots", type=int, help="pool capacity for snapshots, oldest evicted (default 16)")
    ap.add_argument("--mmd-coef", type=float, help="MMD KL coefficient; > 0 selects MMDPuffeRL (default 0)")
    ap.add_argument("--mmd-ref-interval", type=int, help="refresh the MMD reference every N epochs (0 = keep the "
                                                         "initial policy; default 0)")
    ap.add_argument("--seed", type=int, help="seed of torch, the pool, the matches and the wrapper (default 0)")
    ap.add_argument("--data-dir", help="runs go to <data-dir>/league/<run_id>/ (default experiments)")
    ap.add_argument("--deck0", help="team 0 deck: preset, 'random', or 8 comma-separated cards (default hog26)")
    ap.add_argument("--deck1", help="team 1 deck (default hog26)")
    ap.add_argument("--resume", metavar="RUN_DIR", help="continue the run saved in RUN_DIR")
    ap.add_argument("--run-id", help="name of the run directory (default: a timestamp)")
    ap.add_argument("--frame-skip", type=int, help="engine ticks per env step (default: royale.ini, 10)")
    ap.add_argument("--bptt-horizon", type=int, help="PPO segment length (default: royale.ini, 64)")
    ap.add_argument("--batch-size", type=int, help="learner steps per epoch (default num_envs * bptt_horizon)")
    ap.add_argument("--minibatch-size", type=int, help="default min(1024, batch size)")
    ap.add_argument("--learning-rate", type=float, help="default: royale.ini (3e-4)")
    ap.add_argument("--anneal-lr", action="store_true", default=None, help="cosine learning-rate annealing")
    ap.add_argument("--rnn", action="store_true", default=None, help="LSTM learner (Recurrent)")
    ap.add_argument("--opponent-greedy", action="store_true", default=None,
                    help="policy opponents play the card-first greedy rule (SPEC §19.11) instead of sampling")
    ap.add_argument("--init-from", metavar="CKPT", help="new run only: start from these weights (a model / snapshot "
                                                         ".pt, 'ckpt:<path>' or a run directory); the architecture "
                                                         "must match")
    ap.add_argument("--shaping-anneal-frac", type=float, help="anneal the reward shaping weights to 0 over this "
                                                              "fraction of --total-timesteps (default 0 = constant)")
    ap.add_argument("--early-stop-score", type=float, help="finish at a snapshot epoch once the learner's score over "
                                                           "its last --early-stop-window matches vs every anchor is "
                                                           ">= this (default: off)")
    ap.add_argument("--early-stop-window", type=int, help="matches per anchor for --early-stop-score (default 2000)")
    ap.add_argument("--env-threads", type=int, help="step the matches from N threads (pufferroyale.fast_league; "
                                                    "env results bit-identical; 0/1 = serial, the default)")
    ap.add_argument("--fast-opponents", action=argparse.BooleanOptionalAction, default=None,
                    help="CUDA only: observations copied to the device once per step, policy opponents sampled "
                         "on the device, frozen checkpoints in one vmapped forward (pufferroyale.fast_league: same "
                         "distribution, a different random stream than the CPU path)")
    ap.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=None,
                    help="CUDA only: replay the no-grad policy forwards from CUDA graphs (pufferroyale.fast_league; "
                         "--no-cuda-graphs turns it off on --resume)")
    ap.add_argument("--eval-export-interval", type=int,
                    help="every N epochs write the learner's weights to <run>/eval_ckpt/ for hpc/eval_watch.py "
                         "(0 = off, the default)")
    ap.add_argument("--print-interval", type=int, help="print a progress line every N epochs (default 1)")
    ap.add_argument("--log-interval", type=int, help="with --wandb: one log call per N epochs, losses / environment "
                                                       "stats averaged and results summed over them (default 1)")
    ap.add_argument("--card-floor", type=float, help="every legal hand slot keeps at least this probability of the "
                                                     "learner's card choice (D100; < 0.25, conditional head; "
                                                     "default 0 = off)")
    ap.add_argument("--card-floor-steps", type=int, help="fade --card-floor linearly to 0 over the first N learner "
                                                         "steps (default 0 = constant)")
    ap.add_argument("--dashboard", action="store_true", help="show PuffeRL's dashboard instead of progress lines")
    ap.add_argument("--summary-json", help="also write the final JSON summary here")
    return ap


# ------------------------------------------------------------------------------------ config
def resolve_args(known, saved=None):
    """Merge CLI flags (explicitly given) over the saved run's args over DEFAULTS."""
    out = dict(DEFAULTS)
    if saved:
        out.update({k: v for k, v in saved.items() if k in DEFAULTS})
    for k in DEFAULTS:
        v = getattr(known, k, None)
        if v is None:
            continue
        if saved and k in POOL_KEYS and v != saved.get(k):
            print(f"[league] --{k.replace('_', '-')} ignored on --resume (the saved run uses {saved.get(k)!r})",
                  file=sys.stderr)
            continue
        out[k] = v
    return out


def parse_anchors(text):
    from pufferroyale.league import parse_spec
    if text is None or str(text).strip().lower() in ("", "none"):
        return ()
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parse_spec(item)
        except ValueError:
            if os.path.exists(item):
                item = f"ckpt:{os.path.abspath(item)}"
            else:
                raise
        out.append(item)
    return tuple(out)


def deck_value(v):
    """A deck argument: a preset / 'random' string, or 8 comma-separated card names or ids."""
    if isinstance(v, str) and "," in v:
        return [int(c) if c.strip().lstrip("-").isdigit() else c.strip() for c in v.split(",")]
    return v


def load_ini(rest):
    """royale.ini merged over PufferLib's default.ini, with --section.key overrides from `rest`."""
    import pufferlib.pufferl as pufferl
    argv0 = sys.argv[0]
    sys.argv = [argv0] + list(rest)
    try:
        return pufferl.load_config_file(DEFAULT_CONFIG)
    finally:
        sys.argv = [argv0]


def build_config(a, ini, run_dir):
    """The PuffeRL config dict for a league run."""
    tr = dict(ini["train"])
    horizon = int(a["bptt_horizon"] or tr["bptt_horizon"])
    batch = int(a["batch_size"] or a["num_envs"] * horizon)
    minibatch = int(a["minibatch_size"] or min(int(tr["minibatch_size"]), batch))
    if batch % horizon or minibatch % horizon or batch < minibatch or batch // horizon < a["num_envs"]:
        raise SystemExit(f"inconsistent sizes: batch {batch}, minibatch {minibatch}, horizon {horizon}, "
                         f"num_envs {a['num_envs']} (need horizon | batch, horizon | minibatch, "
                         f"minibatch <= batch, num_envs <= batch / horizon)")
    if int(a["total_timesteps"]) < batch:
        raise SystemExit(f"--total-timesteps {a['total_timesteps']} is smaller than one PPO batch of {batch} learner "
                         f"steps (num_envs x bptt_horizon, or --batch-size): nothing would be trained")
    tr.update(env="pufferroyale", total_timesteps=int(a["total_timesteps"]), device=a["device"], seed=int(a["seed"]),
              data_dir=run_dir, bptt_horizon=horizon, batch_size=batch, minibatch_size=minibatch,
              max_minibatch_size=minibatch,
              use_rnn=bool(a["rnn"]), anneal_lr=bool(a["anneal_lr"]), checkpoint_interval=10 ** 9,
              compile=False, mmd_coef=float(a["mmd_coef"]), mmd_ref_interval=int(a["mmd_ref_interval"]))
    if a["learning_rate"] is not None:
        tr["learning_rate"] = float(a["learning_rate"])
    return tr


def env_kwargs(a, ini, global_step=0):
    """Royale keywords of a league run: royale.ini [env] (+ --env.* overrides), the decks, and the
    reward-v2 plumbing of SPEC §19.7.1 (reward_gamma = train.gamma; the shaping anneal over
    ceil(F * total / num_envs) env steps, offset by global_step // num_envs on resume)."""
    from pufferroyale.trainer import shaping_env_kwargs
    kw = {k: v for k, v in ini["env"].items() if k not in ("num_envs", "num_agents", "opponent", "learner_side")}
    kw["deck0"], kw["deck1"] = deck_value(a["deck0"]), deck_value(a["deck1"])
    if a["frame_skip"]:
        kw["frame_skip"] = int(a["frame_skip"])
    kw.update(shaping_env_kwargs(ini["train"]["gamma"], a["shaping_anneal_frac"] or 0.0, a["total_timesteps"],
                                 a["num_envs"], global_step, anneal_steps=a.get("shaping_anneal_steps")))
    return kw


def fix_anneal_steps(a, known, saved_args=None):
    """SPEC §19.10.1: the run's anneal length N = ceil(F * total / num_envs) (exact), computed when the
    run is created and stored in its args; a --resume keeps it (also when --total-timesteps grows) unless
    --shaping-anneal-frac is given again, which recomputes it from the new F and the new total. A run
    saved before N was recorded keeps the N its last invocation used (its saved F and total)."""
    from pufferroyale.trainer import shaping_anneal_steps
    if saved_args is None or known.shaping_anneal_frac is not None:
        a["shaping_anneal_steps"] = shaping_anneal_steps(a["shaping_anneal_frac"] or 0.0, a["total_timesteps"],
                                                         a["num_envs"])
    elif saved_args.get("shaping_anneal_steps") is None:
        a["shaping_anneal_steps"] = shaping_anneal_steps(saved_args.get("shaping_anneal_frac") or 0.0,
                                                         saved_args["total_timesteps"], saved_args["num_envs"])
    else:
        a["shaping_anneal_steps"] = int(saved_args["shaping_anneal_steps"])


def truncate_history(path, epoch):
    """SPEC §19.10.3: keep the history.jsonl records with epoch <= `epoch` (the saved epoch a --resume
    continues from; later records belong to epochs the run re-does). A torn last line is dropped too.
    Returns the number of records removed."""
    if not os.path.isfile(path):
        return 0
    keep, dropped = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ok = int(json.loads(line)["epoch"]) <= int(epoch)
            except (ValueError, KeyError, TypeError):
                ok = False
            if ok:
                keep.append(line if line.endswith("\n") else line + "\n")
            else:
                dropped += 1
    if dropped:
        with open(path + ".tmp", "w") as f:
            f.writelines(keep)
        os.replace(path + ".tmp", path)
    return dropped


def rotate_stale_history(path):
    """SPEC §19.10.3: a NEW run in a directory that holds a history.jsonl (but no league_state.json)
    moves it to history.jsonl.stale-<k>, k = 1, 2, ... (the first free). Returns the new path or None."""
    if not os.path.isfile(path):
        return None
    k = 1
    while os.path.exists(f"{path}.stale-{k}"):
        k += 1
    os.replace(path, f"{path}.stale-{k}")
    return f"{path}.stale-{k}"


def strip_logging_flags(tokens):
    """`tokens` without PufferLib's logging flags (and their values)."""
    out, skip = [], 0
    for t in tokens:
        if skip:
            skip -= 1
            continue
        name = str(t).split("=", 1)[0]
        if name in LOGGING_FLAGS:
            skip = 0 if "=" in str(t) else LOGGING_FLAGS[name]
            continue
        out.append(t)
    return out


def make_wandb(ini, config, load_id=None):
    """pufferroyale.trainer.WandbLogger (PufferLib's WandbLogger interface, any wandb version) when
    --wandb was given (SPEC §19.7.5), else None (wandb not imported)."""
    if ini.get("neptune"):
        raise SystemExit("--neptune is not supported by the PufferRoyale scripts: use --wandb")
    if not ini.get("wandb"):
        return None
    from pufferroyale.trainer import WandbLogger
    args = {"wandb_project": ini.get("wandb_project"), "wandb_group": ini.get("wandb_group"), "tag": ini.get("tag"),
            "no_model_upload": bool(ini.get("no_model_upload")), **json_safe(config)}
    return WandbLogger(args, load_id=load_id)


def check_card_floor(a):
    """D100: --card-floor in [0, 0.25), --card-floor-steps >= 0."""
    eps, n = a["card_floor"], a["card_floor_steps"]
    if eps is None or not math.isfinite(float(eps)) or not 0.0 <= float(eps) < 0.25:
        raise SystemExit(f"--card-floor must lie in [0, 0.25), got {eps!r}")
    if n is None or int(n) < 0:
        raise SystemExit(f"--card-floor-steps must be >= 0, got {n!r}")


def card_floor_at(a, global_step):
    """D100: the floor for the epoch that starts at learner step `global_step`: --card-floor, faded
    linearly to 0 over the first --card-floor-steps learner steps (0 = constant)."""
    eps, n = float(a.get("card_floor") or 0.0), int(a.get("card_floor_steps") or 0)
    if eps <= 0.0 or n <= 0:
        return max(eps, 0.0)
    return eps * max(0.0, 1.0 - int(global_step) / n)


PLAY_RATE = "environment/cards/play_rate/"
ENV_LEAGUE = "environment/league/"
#: the league/score_vs_snapshots_age_<label> windows: a snapshot's age = learner steps since it was taken
SNAPSHOT_AGE_BINS = ((0, 16e6, "0-16M"), (16e6, 64e6, "16-64M"), (64e6, 200e6, "64-200M"), (200e6, math.inf, "200M+"))


def snapshot_epoch(spec):
    """The epoch of a pool snapshot spec ("ckpt:<dir>/snap_<epoch>.pt"), else None."""
    base = os.path.basename(spec[5:]) if isinstance(spec, str) and spec.startswith("ckpt:") else ""
    if base.startswith("snap_") and base.endswith(".pt") and base[5:-3].isdigit():
        return int(base[5:-3])
    return None


def results_keys(spec, league_sections=False):
    """(score key, games key) under which wandb_record logs the results against opponent `spec`: with
    league_sections, "self" and pool snapshots in the league sections; everything else under results/."""
    e = snapshot_epoch(spec)
    if league_sections and spec == "self":
        return "league/score_vs_self", "league/games_vs_self"
    if league_sections and e is not None:
        return f"league_snapshots/snap_{e}/score", f"league_snapshots/snap_{e}/games"
    return f"results/{short(spec)}/score", f"results/{short(spec)}/games"


def wandb_record(rec, puffer_logs=None, floor_ref=0.0, league_sections=False):
    """One flat numeric dict for wandb from a history.jsonl record (+ PuffeRL's environment/* stats). With
    floor_ref > 0 (D100) also cards/below_floor: the number of cards whose play rate is below floor_ref.

    league_sections (league_train.py, 2026-10-20; best_response.py keeps the plain layout): the league's
    matches get their own wandb sections instead of environment/league/* and results/<spec>/*:
    league/ = score_vs_self, score_vs_snapshots (all snapshots, games-weighted), score_vs_snapshots_age_<bin>
    (SNAPSHOT_AGE_BINS), the games behind each, pool_p_median / min / max (the running PFSP score p over the
    snapshots played in this record) and the pool counters (pool_size, episodes, snapshots); league_snapshots/
    = per snapshot snap_<epoch>/score, /games and /p, only for snapshots played in this record."""
    out = {"epoch": rec["epoch"], "global_step": rec["global_step"], "sps": rec["sps"], "pool_size": rec["pool_size"]}
    if "card_floor" in rec:
        out["card_floor"] = rec["card_floor"]
    for k, v in rec["losses"].items():
        if v is not None:
            out[f"losses/{k}"] = v
    if not league_sections:
        for spec, r in rec["results"].items():
            out[f"results/{short(spec)}/score"] = r["score"]
            out[f"results/{short(spec)}/games"] = r["games"]
        for k, v in (puffer_logs or {}).items():
            if k.startswith("environment/") and isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[k] = float(v)
        rates = [v for k, v in out.items() if k.startswith(PLAY_RATE)]
        if floor_ref > 0 and rates:
            out["cards/below_floor"] = sum(v < floor_ref for v in rates)
        return out
    steps_per_epoch = rec["global_step"] / max(1, rec["epoch"])
    snap = [0, 0.0]
    ages = {label: [0, 0.0] for _, _, label in SNAPSHOT_AGE_BINS}
    played = set()
    for spec, r in rec["results"].items():
        ks, kg = results_keys(spec, league_sections=True)
        out[ks], out[kg] = r["score"], r["games"]
        e = snapshot_epoch(spec)
        if e is None or not r["games"]:
            continue
        played.add(spec)
        snap[0] += r["games"]
        snap[1] += r["score"] * r["games"]
        age = max(0.0, (rec["epoch"] - e) * steps_per_epoch)
        for lo, hi, label in SNAPSHOT_AGE_BINS:
            if lo <= age < hi:
                ages[label][0] += r["games"]
                ages[label][1] += r["score"] * r["games"]
    if snap[0]:
        out["league/score_vs_snapshots"], out["league/games_vs_snapshots"] = snap[1] / snap[0], snap[0]
    for label, (g, s) in ages.items():
        if g:
            out[f"league/score_vs_snapshots_age_{label}"] = s / g
            out[f"league/games_vs_snapshots_age_{label}"] = g
    pool_p = []
    for k, v in (puffer_logs or {}).items():
        if not (k.startswith("environment/") and isinstance(v, (int, float)) and math.isfinite(float(v))):
            continue
        if k.startswith(ENV_LEAGUE):
            sub = k[len(ENV_LEAGUE):]
            if sub.startswith("p/"):
                spec = sub[2:]
                if spec in played:
                    out[f"league_snapshots/snap_{snapshot_epoch(spec)}/p"] = float(v)
                    pool_p.append(float(v))
            else:
                out[f"league/{sub}"] = float(v)
            continue
        out[k] = float(v)
    if pool_p:
        pool_p.sort()
        n = len(pool_p)
        out["league/pool_p_median"] = (pool_p[(n - 1) // 2] + pool_p[n // 2]) / 2
        out["league/pool_p_min"], out["league/pool_p_max"] = pool_p[0], pool_p[-1]
    rates = [v for k, v in out.items() if k.startswith(PLAY_RATE)]
    if floor_ref > 0 and rates:
        out["cards/below_floor"] = sum(v < floor_ref for v in rates)
    return out


class WandbWindow:
    """--log-interval N (D99): one wandb call per N epochs. Losses and PuffeRL's environment/* stats are
    averaged over the epochs that reported them, results are summed (games; score = total score / games);
    epoch, global_step, sps and pool_size are the last epoch's. The trainer logs through wandb_record
    directly when N = 1, so the default is unchanged (SPEC §19.9.12). card_floor (D100) is the last epoch's;
    cards/below_floor is counted on the averaged play rates. league_sections: see wandb_record."""

    def __init__(self, n, floor_ref=0.0, league_sections=False):
        self.n = max(1, int(n))
        self.floor_ref = float(floor_ref)
        self.league_sections = bool(league_sections)
        self.clear()

    def clear(self):
        self.last = None
        self.losses, self.env, self.results = {}, {}, {}

    def add(self, rec, puffer_logs=None):
        self.last = rec
        for k, v in rec["losses"].items():
            if v is not None:
                s = self.losses.setdefault(k, [0.0, 0])
                s[0] += v
                s[1] += 1
        for spec, r in rec["results"].items():
            s = self.results.setdefault(spec, [0, 0.0])
            s[0] += r["games"]
            s[1] += r["score"] * r["games"]
        for k, v in (puffer_logs or {}).items():
            if k.startswith("environment/") and isinstance(v, (int, float)) and math.isfinite(float(v)):
                s = self.env.setdefault(k, [0.0, 0])
                s[0] += float(v)
                s[1] += 1

    def due(self, epoch):
        return int(epoch) % self.n == 0

    def pop(self):
        """(wandb record, step) for the epochs added since the last pop, or None when there are none."""
        if self.last is None:
            return None
        r = self.last
        rec = {"epoch": r["epoch"], "global_step": r["global_step"], "sps": r["sps"], "pool_size": r["pool_size"],
               "losses": {k: s / c for k, (s, c) in self.losses.items()},
               "results": {spec: {"games": g, "score": sc / g} for spec, (g, sc) in self.results.items() if g}}
        if "card_floor" in r:
            rec["card_floor"] = r["card_floor"]
        logs = {k: s / c for k, (s, c) in self.env.items()}
        self.clear()
        return wandb_record(rec, logs, self.floor_ref, self.league_sections), int(r["global_step"])


# ------------------------------------------------------------------------------------ build
class RunLogger:
    """PuffeRL logger with a fixed run id (PuffeRL's NoLogger uses the wall clock)."""

    def __init__(self, run_id):
        self.run_id = run_id

    def log(self, logs, step):
        pass

    def close(self, model_path, early_stop):
        pass


def build(a, ini, pool, run_dir, run_id, dashboard=False, reset_seed=None, env_overrides=None, global_step=0,
          policy_kwargs=None, rnn_kwargs=None):
    """(LeagueVecEnv, policy, trainer, cfg) for resolved args `a`. The trainer is always MMDPuffeRL
    (with mmd_coef = 0 it is plain PuffeRL PPO, bit for bit, plus the SPEC §18.6 AMP fix).
    policy_kwargs / rnn_kwargs replace royale.ini's [policy] / --rnn + [rnn] (best_response.py
    --init-from-target builds the target's architecture)."""
    import torch
    from pufferroyale.league import LeagueVecEnv
    from pufferroyale.torch import Policy, Recurrent
    from pufferroyale.trainer import MMDPuffeRL

    cfg = build_config(a, ini, run_dir)
    if reset_seed is not None:
        cfg["seed"] = int(reset_seed)        # the vecenv's async_reset seed
    kw = env_kwargs(a, ini, global_step)
    kw.update(env_overrides or {})
    fast = dict(env_threads=int(a.get("env_threads") or 0), cuda_graphs=bool(a.get("cuda_graphs")),
                gpu_obs=bool(a.get("fast_opponents")), gpu_sample=bool(a.get("fast_opponents")),
                vmap=bool(a.get("fast_opponents")))
    if fast["env_threads"] > 1 or fast["cuda_graphs"] or fast["gpu_obs"]:     # opt-in throughput paths (D90)
        from pufferroyale.fast_league import FastLeagueVecEnv
        lv = FastLeagueVecEnv(pool, num_envs=int(a["num_envs"]), seed=int(a["seed"]), device=a["device"],
                              opponent_greedy=bool(a["opponent_greedy"]), **fast, **kw)
    else:
        lv = LeagueVecEnv(pool, num_envs=int(a["num_envs"]), seed=int(a["seed"]), device=a["device"],
                          opponent_greedy=bool(a["opponent_greedy"]), **kw)
    try:
        policy = Policy(lv.driver_env, **(ini["policy"] if policy_kwargs is None else policy_kwargs))
        if policy_kwargs is not None and rnn_kwargs is not None:
            policy = Recurrent(lv.driver_env, policy, **rnn_kwargs)
        elif policy_kwargs is None and a["rnn"]:
            policy = Recurrent(lv.driver_env, policy, **ini["rnn"])
        policy = policy.to(a["device"])
        if float(a.get("card_floor") or 0.0) > 0:
            # D100: before the trainer deep-copies the policy (MMD reference) and before any CUDA graph is captured
            try:
                getattr(policy, "policy", policy).enable_card_floor(card_floor_at(a, global_step))
            except ValueError as e:
                raise SystemExit(f"--card-floor: {e}")
        lv.set_policy(policy)
        cls = MMDPuffeRL
        if not dashboard:                 # progress lines instead of the (screen-clearing) dashboard
            cls = type("MMDPuffeRL", (MMDPuffeRL,), {"print_dashboard": lambda self, *args, **kw: None})
        trainer = cls(cfg, lv, policy, logger=RunLogger(run_id))
        if fast["cuda_graphs"]:
            # AFTER the trainer: MMDPuffeRL deep-copies the policy for its reference, and the copy must run the
            # plain forward with its own weights (install_learner_graph also guards this by object identity)
            from pufferroyale.fast_league import install_learner_graph
            install_learner_graph(policy)
    except BaseException:
        lv.close()
        raise
    trainer.save_checkpoint = lambda: None    # PuffeRL saves from inside train(); all our saves are explicit
    coef = float(a["mmd_coef"])
    trainer.trainer_name = f"MMDPuffeRL(mmd_coef={coef:g})" if coef > 0 else "PuffeRL PPO (MMDPuffeRL, mmd_coef=0)"
    return lv, policy, trainer, cfg


def cpu_state(policy):
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}


def atomic_torch_save(obj, path):
    import torch
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def _map_ckpt_paths(pool_state, fn):
    """pool_state with fn applied to the path of every ckpt: snapshot spec and stats key."""
    def conv(spec):
        return f"ckpt:{fn(spec[5:])}" if isinstance(spec, str) and spec.startswith("ckpt:") else spec
    out = dict(pool_state)
    out["snapshots"] = [conv(s) for s in pool_state["snapshots"]]
    out["stats"] = {conv(k): v for k, v in pool_state["stats"].items()}
    return out


def relative_pool_state(pool_state, run_dir):
    """Snapshot paths inside the run dir stored relative to it (the run dir stays relocatable)."""
    root = os.path.abspath(run_dir)

    def rel(p):
        ap = os.path.abspath(p)
        return os.path.relpath(ap, root) if os.path.commonpath([ap, root]) == root else p
    return _map_ckpt_paths(pool_state, rel)


def absolute_pool_state(pool_state, run_dir):
    return _map_ckpt_paths(pool_state, lambda p: p if os.path.isabs(p) else os.path.join(os.path.abspath(run_dir), p))


def short(spec):
    return f"ckpt:{os.path.basename(spec[5:])}" if spec.startswith("ckpt:") else spec


# ------------------------------------------------------------------------------------ run
class LeagueRun:
    """One league run: builds or resumes, trains, snapshots, saves. On any error after build()
    the constructor / main() call abort(): the monitor thread is stopped, the envs are closed and
    nothing is written."""

    def __init__(self, known, rest):
        import torch
        import numpy as np
        from pufferroyale.league import OpponentPool

        self.known = known
        self.t0 = time.time()
        self.saved = None
        self.trainer = None
        self.lv = None
        self.notes = []
        self.nonfinite, self.bad_params = [], []
        self.wandb = None
        self.early_stopped = False
        if known.resume and known.init_from:
            raise SystemExit("--init-from starts a NEW run (weights only); it cannot be combined with --resume")
        if known.resume:
            self.run_dir = os.path.abspath(known.resume)
            state_path = os.path.join(self.run_dir, "league_state.json")
            if not os.path.isfile(state_path):
                raise SystemExit(f"--resume {known.resume}: no league_state.json in {self.run_dir}")
            with open(state_path) as f:
                self.saved = json.load(f)
            self.run_id = self.saved["run_id"]
            a = resolve_args(known, self.saved["args"])
            if known.data_dir is not None:
                a["data_dir"] = known.data_dir
        else:
            a = resolve_args(known)
            self.run_id = a["run_id"] or time.strftime("%Y%m%d_%H%M%S") + f"_s{a['seed']}"
            self.run_dir = os.path.abspath(os.path.join(a["data_dir"], "league", self.run_id))
            if os.path.exists(os.path.join(self.run_dir, "league_state.json")):
                raise SystemExit(f"{self.run_dir} already holds a run: use --resume or another --run-id")
        a["run_id"] = self.run_id
        a["anchors"] = ",".join(parse_anchors(a["anchors"]))
        fix_anneal_steps(a, known, self.saved["args"] if self.saved else None)
        self.a = a
        self.epoch0 = int(self.saved["epoch"]) if self.saved else 0
        self.step0 = int(self.saved["global_step"]) if self.saved else 0
        # --section.key overrides of the original run are kept; new ones given now win (argparse: last wins).
        # Logging flags (--wandb ...) apply to this invocation only and are not stored
        saved_overrides = list(self.saved.get("ini_overrides", [])) if self.saved else []
        self.ini_overrides = saved_overrides + strip_logging_flags(rest)
        self.ini = load_ini(saved_overrides + list(rest))
        if self.ini.get("neptune"):
            raise SystemExit("--neptune is not supported by the PufferRoyale scripts: use --wandb")
        self._check_early_stop_args(a)
        check_card_floor(a)
        self.floor_on = float(a["card_floor"]) > 0
        self.pool = OpponentPool(anchors=parse_anchors(a["anchors"]), max_snapshots=int(a["max_snapshots"]),
                                 pfsp=a["pfsp"], pfsp_eps=float(a["pfsp_eps"]), self_play_frac=float(a["self_play_frac"]),
                                 anchor_frac=float(a["anchor_frac"]), seed=int(a["seed"]))
        if self.saved:
            self.pool.load_state_dict(absolute_pool_state(self.saved["pool"], self.run_dir))
        # per-anchor results of the last early_stop_window matches (persisted, so a resume keeps them)
        es = (self.saved or {}).get("early_stop") or {}
        win = int(a["early_stop_window"])
        self.recent = {an: collections.deque((int(r) for r in (es.get("recent") or {}).get(an, [])), maxlen=win)
                       for an in self.pool.anchors}
        self.early_stopped = bool((self.saved or {}).get("early_stopped", False))
        self.done_already = self.step0 >= int(a["total_timesteps"]) or self.early_stopped
        # SPEC §19.10.3: history.jsonl matches the saved state -- a resume drops the records written after the
        # last save (the run re-does those epochs); a new run moves a stale history (no league_state.json) aside
        hist = os.path.join(self.run_dir, "history.jsonl")
        if self.saved:
            n = truncate_history(hist, self.epoch0)
            if n:
                self._note(f"history.jsonl: dropped {n} record(s) after the saved epoch {self.epoch0}")
        if self.done_already:
            return
        build_config(a, self.ini, self.run_dir)           # size / total checks before anything starts
        os.makedirs(self.run_dir, exist_ok=True)
        if not self.saved:
            moved = rotate_stale_history(hist)
            if moved:
                self._note(f"a stale history.jsonl (no league_state.json) was moved to {os.path.basename(moved)}")
        # a resumed run continues on fresh match seeds (seed + 7919 * epoch), still deterministic
        seed_now = int(a["seed"]) + 7919 * self.epoch0
        torch.manual_seed(seed_now)
        np.random.seed(seed_now % (2 ** 32))
        self.lv, self.policy, self.trainer, self.cfg = build(a, self.ini, self.pool, self.run_dir, self.run_id,
                                                             dashboard=known.dashboard, reset_seed=seed_now,
                                                             global_step=self.step0)
        try:
            from pufferroyale.trainer import warn_shaping_clip                  # SPEC §19.10.2
            warn_shaping_clip(env_kwargs(a, self.ini, self.step0), self.cfg.get("reward_clip", 1.0), "league")
            self.snapshots = list(self.saved.get("snapshot_files", [])) if self.saved else []
            if self.saved:
                self._restore()
            elif a["init_from"]:
                self._init_from(a["init_from"])
            from pufferroyale.decks import expanded_deck_sets
            from pufferroyale.league import policy_architecture
            pkw, _ = policy_architecture(self.policy)
            env_used = env_kwargs(a, self.ini, self.step0)
            env_used.update(expanded_deck_sets(env_used))                # SPEC §19.10.5
            self.config = {"policy": pkw, "rnn_name": "Recurrent" if a["rnn"] else None, "rnn": self.ini["rnn"],
                           "env": env_used, "league": a}
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(self.config, f, indent=2, default=str)
            self.wandb = make_wandb(self.ini, dict(self.config, train={k: v for k, v in self.cfg.items()
                                                                       if isinstance(v, (int, float, str, bool))}),
                                    load_id=(self.saved or {}).get("wandb_id"))
        except BaseException:
            self.abort()
            raise

    def _check_early_stop_args(self, a):
        if a["early_stop_window"] is None or int(a["early_stop_window"]) < 1:
            raise SystemExit("--early-stop-window must be >= 1")
        if a["early_stop_score"] is None:
            return
        if not math.isfinite(float(a["early_stop_score"])):
            raise SystemExit("--early-stop-score must be a finite number")
        if not parse_anchors(a["anchors"]):
            raise SystemExit("--early-stop-score needs at least one anchor (it compares the learner against every anchor)")
        if int(a["snapshot_interval"]) <= 0:
            raise SystemExit("--early-stop-score is checked at snapshot epochs: it needs --snapshot-interval > 0")

    def _init_from(self, ckpt):
        """--init-from: the learner starts from the checkpoint's weights (new pool, new optimizer); an
        architecture mismatch is an error. With MMD on, the reference is the loaded learner."""
        from pufferroyale.league import load_weights
        try:
            path = load_weights(self.policy, ckpt)
        except (ValueError, FileNotFoundError) as e:
            raise SystemExit(f"--init-from: {e}")
        self.a["init_from"] = os.path.abspath(path)
        if self.trainer.ref_policy is not None:
            self.trainer.refresh_reference()
        self._note(f"learner initialised from {path} (weights only)")

    def early_stop_reached(self) -> bool:
        """SPEC §19.7.3: every anchor has >= N finished matches and the learner's score over its last N
        matches vs every anchor is >= X."""
        x = self.a["early_stop_score"]
        if x is None or not self.recent:
            return False
        n = int(self.a["early_stop_window"])
        for d in self.recent.values():
            if len(d) < n or sum(0.5 * (r + 1) for r in d) / n < float(x):
                return False
        return True

    def _restore(self):
        """Load learner + optimizer + counters + RNG states of the saved run (SPEC §18.7)."""
        import torch
        from pufferroyale.league import architecture_diff
        a, tr = self.a, self.trainer
        dev = a["device"]
        sd = torch.load(os.path.join(self.run_dir, "learner.pt"), map_location=dev, weights_only=True)
        # SPEC §19.10.8: another architecture / grid / rnn than the saved run is a clear exit, not a traceback
        diff = architecture_diff(self.policy, sd)
        try:
            if not diff:
                self.policy.load_state_dict(sd)
        except RuntimeError as e:
            diff = [str(e).strip().splitlines()[-1].strip()]
        if diff:
            raise SystemExit(f"--resume {self.run_dir}: this invocation builds a different policy than the saved run "
                             f"({'; '.join(diff)}). The architecture, placement grid and rnn setting come from the run: "
                             f"drop the overrides that change them (e.g. --env.placement-grid, --policy.*, --rnn.*)")
        ts = torch.load(os.path.join(self.run_dir, "trainer_state.pt"), map_location=dev, weights_only=False)
        # optimizer moments from the save; lr / betas / eps from THIS invocation's config; the cosine
        # schedule (with --anneal-lr) continued at the saved epoch
        tr.load_training_state(ts["optimizer_state_dict"], ts["epoch"], ts["global_step"])
        if ts.get("torch_rng") is not None:
            torch.set_rng_state(ts["torch_rng"].cpu())
        self.lv.episodes = int(self.saved.get("episodes", 0))
        self._drop_eval_exports_after(int(ts["epoch"]))
        rng = (self.saved.get("rng") or {}).get("league")
        if rng:
            self.lv.load_rng_state(rng)
        saved_coef = float(self.saved["args"].get("mmd_coef", 0.0) or 0.0)
        coef = float(a["mmd_coef"])
        if saved_coef != coef:
            self._note(f"mmd_coef changes on resume: {saved_coef:g} -> {coef:g}")
        if tr.ref_policy is not None:
            ref = os.path.join(self.run_dir, "mmd_reference.pt")
            if saved_coef > 0 and os.path.isfile(ref):
                tr.ref_policy.load_state_dict(torch.load(ref, map_location=dev, weights_only=True))
            else:                                          # MMD switched on now: the magnet is the loaded learner
                tr.refresh_reference()
                self._note("MMD reference initialised to the resumed learner"
                           + ("" if saved_coef > 0 else " (MMD was off in the saved run)"))

    def apply_card_floor(self):
        """D100: set the floor of the coming epoch on the learner and on the MMD reference (a refresh is a
        load_state_dict, which does not carry the non-persistent floor buffer). Returns it."""
        eps = card_floor_at(self.a, self.trainer.global_step)
        for p in (self.policy, getattr(self.trainer, "ref_policy", None)):
            if p is not None:
                getattr(p, "policy", p).set_card_floor(eps)
        return eps

    def _note(self, msg):
        self.notes.append(msg)
        print(f"[league] note: {msg}", file=sys.stderr, flush=True)

    # -------------------------------------------------------------- persistence
    def save(self):
        """learner.pt + trainer_state.pt + mmd_reference.pt + league_state.json, each written
        atomically. Only called after a finite epoch. Returns the learner path."""
        import torch
        tr = self.trainer
        learner = os.path.join(self.run_dir, "learner.pt")
        atomic_torch_save(cpu_state(self.policy), learner)
        if getattr(tr, "ref_policy", None) is not None:
            atomic_torch_save(cpu_state(tr.ref_policy), os.path.join(self.run_dir, "mmd_reference.pt"))
        atomic_torch_save({"optimizer_state_dict": tr.optimizer.state_dict(), "global_step": int(tr.global_step),
                           "epoch": int(tr.epoch), "run_id": self.run_id, "torch_rng": torch.get_rng_state()},
                          os.path.join(self.run_dir, "trainer_state.pt"))
        state = {"version": 2, "run_id": self.run_id, "args": self.a, "ini_overrides": self.ini_overrides,
                 "epoch": int(tr.epoch), "global_step": int(tr.global_step),
                 "pool": relative_pool_state(self.pool.state_dict(), self.run_dir),
                 "rng": {"league": self.lv.rng_state()}, "learner": "learner.pt",
                 "snapshot_files": self.snapshots, "episodes": int(self.lv.episodes),
                 "early_stop": {"recent": {an: list(d) for an, d in self.recent.items()}},
                 "early_stopped": bool(self.early_stopped)}
        if self.wandb is not None:
            state["wandb_id"] = self.wandb.run_id
        elif self.saved and self.saved.get("wandb_id"):
            state["wandb_id"] = self.saved["wandb_id"]
        path = os.path.join(self.run_dir, "league_state.json")
        with open(path + ".tmp", "w") as f:
            json.dump(json_safe(state), f, indent=1, allow_nan=False)
        os.replace(path + ".tmp", path)
        return learner

    def snapshot(self):
        path = os.path.join(self.run_dir, f"snap_{self.trainer.epoch}.pt")
        atomic_torch_save(cpu_state(self.policy), path)
        self.pool.add_snapshot(path)
        self.snapshots.append(os.path.basename(path))
        return path

    def _drop_eval_exports_after(self, epoch):
        """On resume: eval exports newer than the resumed epoch came from a trajectory that no longer exists
        (the run restarts at its last save); delete them so hpc/eval_watch.py never evaluates them."""
        d = os.path.join(self.run_dir, "eval_ckpt")
        if not os.path.isdir(d):
            return
        dropped = 0
        for n in os.listdir(d):
            if n.startswith("policy_") and (n.endswith(".json") or n.endswith(".pt")):
                try:
                    e = int(n.split("_", 1)[1].split(".", 1)[0])
                except ValueError:
                    continue
                if e > epoch:
                    os.remove(os.path.join(d, n))
                    dropped += n.endswith(".json")
        if dropped:
            self._note(f"eval exports after the resumed epoch {epoch} removed: {dropped}")

    def export_for_eval(self, keep=24):
        """<run>/eval_ckpt/policy_<epoch>.pt (+ .json with epoch / global_step, written last) for the
        evaluation sidecar hpc/eval_watch.py (D91). config.json is copied next to them so eval.py applies
        the run's env settings (placement grid included). Only the newest `keep` exports are kept; nothing
        here touches the run's state, RNG streams or saves."""
        import shutil
        tr = self.trainer
        d = os.path.join(self.run_dir, "eval_ckpt")
        os.makedirs(d, exist_ok=True)
        cfg = os.path.join(self.run_dir, "config.json")
        if os.path.isfile(cfg) and not os.path.isfile(os.path.join(d, "config.json")):
            shutil.copyfile(cfg, os.path.join(d, "config.json"))
        stem = os.path.join(d, f"policy_{int(tr.epoch):07d}")
        atomic_torch_save(cpu_state(self.policy), stem + ".pt")
        with open(stem + ".json.tmp", "w") as fh:
            json.dump({"epoch": int(tr.epoch), "global_step": int(tr.global_step), "run_id": self.run_id}, fh)
        os.replace(stem + ".json.tmp", stem + ".json")
        old = sorted(n for n in os.listdir(d) if n.startswith("policy_") and n.endswith(".json"))[:-keep]
        for n in old:
            for ext in (".json", ".pt"):
                try:
                    os.remove(os.path.join(d, n[:-5] + ext))
                except FileNotFoundError:
                    pass

    def _write_diagnostic(self, losses):
        diag = {"epoch": int(self.trainer.epoch), "global_step": int(self.trainer.global_step),
                "nonfinite_losses": self.nonfinite, "nonfinite_params": self.bad_params, "losses": losses,
                "last_good_save": {"epoch": self.last_saved_epoch, "learner": "learner.pt"
                                   if self.last_saved_epoch is not None else None},
                "note": "this epoch was NOT snapshotted or saved; lower the learning rate / check the inputs"}
        path = os.path.join(self.run_dir, "nonfinite.json")
        with open(path, "w") as f:
            json.dump(json_safe(diag), f, indent=1, allow_nan=False)
        return path

    # -------------------------------------------------------------- loop
    def train(self):
        import torch
        tr, a = self.trainer, self.a
        hist_path = os.path.join(self.run_dir, "history.jsonl")
        t_start, s_start = time.time(), tr.global_step
        interval = int(a["snapshot_interval"])
        pi = max(1, int(a["print_interval"]))
        floor_ref = float(a["card_floor"]) if self.floor_on else 0.0
        win = WandbWindow(a.get("log_interval") or 1, floor_ref, league_sections=True)
        self.last_saved_epoch = self.epoch0 if self.saved else None
        self.diagnostic = None
        while tr.global_step < int(a["total_timesteps"]):
            if not self.known.dashboard:
                tr.last_log_time = -1e18   # PuffeRL only publishes an epoch's losses when it logs (>= 0.25 s apart)
            eps = self.apply_card_floor() if self.floor_on else None   # D100: fixed for the epoch's rollout + update
            tr.evaluate()
            logs = tr.train()
            losses = {k: float(v) for k, v in tr.losses.items()}
            # SPEC §18.7: check BEFORE any snapshot / save
            self.nonfinite = [(int(tr.epoch), k, str(v)) for k, v in losses.items()
                              if k != "explained_variance" and not math.isfinite(v)]
            self.bad_params = [n for n, p in self.policy.named_parameters() if not torch.isfinite(p).all()]
            window, outcomes = collections.OrderedDict(), collections.OrderedDict()
            for spec, res, seat in self.lv.drain_results():
                g = window.setdefault(spec, [0, 0.0])
                g[0] += 1
                g[1] += 0.5 * (res + 1)
                outcomes.setdefault(spec, []).append(int(res))
                if spec in self.recent:
                    self.recent[spec].append(int(res))
            bad = bool(self.nonfinite or self.bad_params)
            snap = None
            if not bad and interval > 0 and tr.epoch % interval == 0:
                snap = self.snapshot()
                self.early_stopped = self.early_stop_reached()      # SPEC §19.7.3: checked at snapshot epochs
                self.save()
                self.last_saved_epoch = int(tr.epoch)
            ev = int(a.get("eval_export_interval") or 0)
            if not bad and ev > 0 and tr.epoch % ev == 0:
                self.export_for_eval()
            sps = (tr.global_step - s_start) / max(time.time() - t_start, 1e-9)
            rec = {"epoch": int(tr.epoch), "global_step": int(tr.global_step), "sps": round(sps, 1),
                   "losses": {k: (v if math.isfinite(v) else None) for k, v in losses.items()},
                   "results": {s: {"games": g, "score": sc / g} for s, (g, sc) in window.items()},
                   "outcomes": outcomes,
                   "pool_size": self.pool.pool_size(), "snapshot": os.path.basename(snap) if snap else None}
            if eps is not None:
                rec["card_floor"] = eps
            if bad:
                rec["nonfinite"] = True
            if self.early_stopped:
                rec["early_stopped"] = True
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec, allow_nan=False) + "\n")
            if self.wandb is not None and win.n == 1:        # SPEC §19.9.12: one call per record
                self.wandb.log(wandb_record(rec, logs, floor_ref, league_sections=True), int(tr.global_step))
            elif self.wandb is not None:                     # D99: one call per --log-interval epochs
                win.add(rec, logs)
                if win.due(tr.epoch) or bad or self.early_stopped:
                    self.wandb.log(*win.pop())
            if not self.known.dashboard and (tr.epoch % pi == 0 or snap or bad):
                res = " ".join(f"{short(s)}={sc / g:.2f}/{g}" for s, (g, sc) in window.items()) or "-"
                kl = f" kl {losses['mmd_kl']:.2e}" if "mmd_kl" in losses and float(a["mmd_coef"]) > 0 else ""
                print(f"[league] step {tr.global_step:>8d} epoch {tr.epoch:>4d} sps {sps:6.0f} "
                      f"pg {losses.get('policy_loss', float('nan')):+.4f} v {losses.get('value_loss', float('nan')):.4f} "
                      f"ent {losses.get('entropy', float('nan')):.3f}{kl}"
                      + (f" floor {eps:.4f}" if eps is not None else "") + f" pool {self.pool.pool_size()} | {res}"
                      + (f" | snapshot {os.path.basename(snap)}" if snap else "")
                      + (" | NON-FINITE: not saved" if bad else ""), flush=True)
            if bad:
                self.diagnostic = self._write_diagnostic(losses)
                break
            if self.early_stopped:
                print(f"[league] early stop at step {tr.global_step} (epoch {tr.epoch}): score >= "
                      f"{float(a['early_stop_score']):g} over the last {int(a['early_stop_window'])} matches vs every "
                      f"anchor", flush=True)
                break
        if self.wandb is not None:                           # D99: the last, partial window
            last = win.pop()
            if last is not None:
                self.wandb.log(*last)

    def finish(self):
        """After a clean run: final save + model_<epoch>.pt, then release the trainer and envs."""
        import torch
        tr = self.trainer
        try:
            self.save()
            model = os.path.join(self.run_dir, f"model_{tr.epoch:06d}.pt")
            atomic_torch_save(cpu_state(self.policy), model)
            if self.wandb is not None:
                wb, self.wandb = self.wandb, None
                wb.close(model, early_stop=bool(self.early_stopped))
        finally:
            self.abort()
        return model

    def abort(self):
        """Release the trainer's monitor thread and the envs without writing anything."""
        if self.trainer is not None:
            self.trainer.utilization.stop()
        if self.lv is not None:
            self.lv.close()
        if self.wandb is not None:
            wb, self.wandb = self.wandb, None
            wb.abort()                           # finish(exit_code=1); never raises


def summary_of(run, model=None, extra=None):
    pool = run.pool
    out = {"run_dir": run.run_dir, "run_id": run.run_id,
           "global_step": int(run.trainer.global_step) if run.trainer else run.step0,
           "epoch": int(run.trainer.epoch) if run.trainer else run.epoch0,
           "elapsed_s": round(time.time() - run.t0, 2), "model": model,
           "pool": {short(k): round(v, 4) for k, v in pool.weights().items()},
           "stats": {short(k): v for k, v in pool.stats().items()}}
    out["early_stopped"] = bool(run.early_stopped)
    if run.notes:
        out["notes"] = run.notes
    if extra:
        out.update(extra)
    return out


def main():
    known, rest = make_parser().parse_known_args()
    run = LeagueRun(known, rest)                 # aborts (threads stopped, envs closed) on any error
    status = 0
    if run.done_already:
        why = ("early-stopped" if run.early_stopped else
               f"already reached {run.step0} >= --total-timesteps {run.a['total_timesteps']}")
        print(f"[league] {run.run_dir} {why}: nothing to do")
        summary = summary_of(run, extra={"nothing_to_do": True})
    else:
        print(f"[league] run dir {run.run_dir}  trainer {run.trainer.trainer_name}  "
              f"batch {run.cfg['batch_size']}  pool {dict((short(k), round(v, 3)) for k, v in run.pool.weights().items())}",
              flush=True)
        try:
            run.train()
        except BaseException:
            run.abort()                          # no save from a failed epoch; exit non-zero with the traceback
            raise
        if run.nonfinite or run.bad_params:
            run.abort()
            status = 1
            summary = summary_of(run, None, {"nonfinite_losses": run.nonfinite, "nonfinite_params": run.bad_params,
                                             "diagnostic": run.diagnostic, "last_good_epoch": run.last_saved_epoch})
        else:
            model = run.finish()
            summary = summary_of(run, model, {"nonfinite_losses": [], "nonfinite_params": []})
    line = json.dumps(json_safe(summary), default=str, allow_nan=False)
    if known.summary_json:
        with open(known.summary_json, "w") as f:
            f.write(line + "\n")
    print(line)
    if status:
        print(f"NON-FINITE values during training: nothing saved for this epoch (see {run.diagnostic})", file=sys.stderr)
        sys.exit(status)


if __name__ == "__main__":
    main()
