#!/usr/bin/env python3
"""Train a PufferRoyale policy with PufferLib 3.0 PuffeRL (SPEC §11, §19.7).

Loads pufferroyale/config/royale.ini (merged over pufferlib's default.ini); every config key
can be overridden on the command line as --section.key value (e.g. --train.learning-rate 1e-4,
--env.opponent random, --train.reward-clip 0, --policy.head flat, --env.placement-grid 2).
Convenience flags cover the common ones.

    python scripts/train.py --total-timesteps 100000                 # self-play smoke run
    python scripts/train.py --num-agents 1 --opponent heuristic --total-timesteps 100000
    python scripts/train.py --rnn --device mps
    python scripts/train.py --backend Multiprocessing --workers 4 --num-envs 8
    python scripts/train.py --deck0 bait --deck1 random --seed 3 --data-dir /tmp/runs
    python scripts/train.py --resume experiments/pufferroyale_<run_id> --total-timesteps 200000
    python scripts/train.py --init-from experiments/league/<run> --total-timesteps 100000

Checkpoints go to <data-dir>/pufferroyale_<run_id>/ (PuffeRL's model_*.pt and trainer_state.pt
-- optimizer, epoch, global step and the torch RNG state --, plus config.json: the policy, rnn,
env kwargs actually used and train settings). A JSON summary (steps, SPS, final losses, episode
stats) is printed at the end and optionally written with --summary-json.

v0.5 (SPEC §19.7):
    --resume RUN_DIR         continue a run from its latest save (weights, optimizer, epoch, global
                             step, torch RNG); --total-timesteps is the ABSOLUTE target. The run's
                             config.json settings are reused; flags given again now override them
                             (the architecture always comes from the run). Matches in progress are
                             re-dealt (from seed + 7919 * epoch).
    --init-from CKPT         a NEW run starting from a checkpoint's weights (file, snapshot or run
                             directory); its architecture must match this run's (error otherwise)
    --shaping-anneal-frac F  anneal the reward-v2 shaping weights to 0 over the first F of
                             --total-timesteps (shaping_anneal_steps N = ceil(F * total / R) exactly, R =
                             the vecenv's agent rows). N is fixed when the run is created (config.json
                             train.shaping_anneal_steps) and reused on --resume, even with a larger
                             --total-timesteps; giving --shaping-anneal-frac again recomputes it from the
                             new F and total. The env always gets reward_gamma = train.gamma. Shaping
                             weights > 0 with train.reward_clip > 0 print a warning (use
                             --train.reward-clip 0)
    --wandb --wandb-project P --wandb-group G --tag T --no-model-upload
                             Weights & Biases logging (pufferroyale.trainer.WandbLogger, PufferLib's
                             logger interface; WANDB_MODE=offline works without network); without
                             --wandb, wandb is never imported
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT_CONFIG = os.path.join(ROOT, "pufferroyale", "config", "royale.ini")
ENV_NAME = "pufferroyale"
#: convenience flag -> the config entries it sets (section, key)
CONVENIENCE = {"total_timesteps": [("train", "total_timesteps")], "device": [("train", "device")],
               "num_envs": [("env", "num_envs")], "num_agents": [("env", "num_agents")],
               "opponent": [("env", "opponent")], "deck0": [("env", "deck0")], "deck1": [("env", "deck1")],
               "batch_size": [("train", "batch_size")], "minibatch_size": [("train", "minibatch_size"),
                                                                           ("train", "max_minibatch_size")],
               "bptt_horizon": [("train", "bptt_horizon")], "seed": [("train", "seed")],
               "shaping_anneal_frac": [("train", "shaping_anneal_frac")]}
#: env keys of config.json that are derived per invocation / recorded only, never passed back to the env
DERIVED_ENV_KEYS = ("reward_gamma", "shaping_anneal_steps", "shaping_step_offset", "deck_pool_decks",
                    "heldout_decks_decks")


def deck_value(v):
    """A deck argument: a preset / 'random' string, or 8 comma-separated card names or ids."""
    if isinstance(v, str) and "," in v:
        return [int(c) if c.strip().lstrip("-").isdigit() else c.strip() for c in v.split(",")]
    return v


def effective_batch_size(tr, total_agents):
    """PuffeRL's batch size: train.batch_size, or total_agents * bptt_horizon when 'auto'."""
    if tr["batch_size"] == "auto":
        return int(total_agents) * int(tr["bptt_horizon"])
    return int(tr["batch_size"])


def learner_rows(args):
    """R of SPEC §19.7.1: agent rows per vector step (= the vecenv's num_agents)."""
    env, vec = args["env"], args["vec"]
    rows = int(env["num_envs"]) * int(env["num_agents"])
    return rows if vec["backend"] == "PufferEnv" else rows * int(vec["num_envs"])


def explicit_keys(rest):
    """(section, key) of every --section.key flag given on the command line."""
    out = set()
    for t in rest:
        name = str(t).split("=", 1)[0]
        if name.startswith("--") and "." in name:
            sec, key = name[2:].split(".", 1)
            out.add((sec, key.replace("-", "_")))
    return out


def make_conv_parser():
    conv = argparse.ArgumentParser(add_help=False)
    conv.add_argument("--config", default=DEFAULT_CONFIG)
    conv.add_argument("--total-timesteps", type=int, help="agent steps (the ABSOLUTE target with --resume)")
    conv.add_argument("--device")
    conv.add_argument("--backend", choices=["PufferEnv", "Serial", "Multiprocessing"])
    conv.add_argument("--workers", type=int, help="Multiprocessing: worker processes (= vec.num_envs)")
    conv.add_argument("--num-envs", type=int, help="native matches per process (env.num_envs)")
    conv.add_argument("--num-agents", type=int, choices=[1, 2])
    conv.add_argument("--opponent", choices=["noop", "random", "heuristic"])
    conv.add_argument("--deck0", help="team 0 deck: preset name, 'random', or 8 comma-separated card names or ids "
                                      "(e.g. 'Hog Rider,Musketeer,...' or '4,2,14,...')")
    conv.add_argument("--deck1", help="team 1 deck (same forms as --deck0)")
    conv.add_argument("--batch-size", type=int)
    conv.add_argument("--minibatch-size", type=int)
    conv.add_argument("--bptt-horizon", type=int)
    conv.add_argument("--rnn", action="store_true", help="wrap the policy in the LSTM (Recurrent)")
    conv.add_argument("--seed", type=int)
    conv.add_argument("--data-dir")
    conv.add_argument("--resume", metavar="RUN_DIR", help="continue the run saved in RUN_DIR "
                                                          "(<data-dir>/pufferroyale_<run_id>)")
    conv.add_argument("--init-from", metavar="CKPT", help="new run starting from these weights (model / snapshot "
                                                          ".pt, 'ckpt:<path>' or a run directory)")
    conv.add_argument("--shaping-anneal-frac", type=float, help="anneal the reward shaping weights to 0 over this "
                                                                "fraction of --total-timesteps (default 0 = constant)")
    conv.add_argument("--quiet", action="store_true", help="concise log lines instead of the dashboard")
    conv.add_argument("--summary-json")
    conv.add_argument("-h", "--help", action="store_true")
    return conv


def load_args(config_path, rest):
    import pufferlib.pufferl as pufferl
    argv0 = sys.argv[0]
    sys.argv = [argv0] + list(rest)
    try:
        return pufferl.load_config_file(config_path)
    finally:
        sys.argv = [argv0] + list(rest)


def apply_convenience(known, args):
    tr, env, vec = args["train"], args["env"], args["vec"]
    if known.total_timesteps is not None:
        tr["total_timesteps"] = known.total_timesteps
    if known.device:
        tr["device"] = known.device
    if known.backend:
        vec["backend"] = known.backend
    if known.workers:
        vec["num_envs"] = known.workers
        vec["num_workers"] = known.workers
    if known.num_envs:
        env["num_envs"] = known.num_envs
    if known.num_agents:
        env["num_agents"] = known.num_agents
    if known.opponent:
        env["opponent"] = known.opponent
    for key in ("deck0", "deck1"):
        val = getattr(known, key)
        if val:
            env[key] = deck_value(val)
    if known.batch_size:
        tr["batch_size"] = known.batch_size
    if known.minibatch_size:
        tr["minibatch_size"] = known.minibatch_size
        tr["max_minibatch_size"] = max(tr["max_minibatch_size"], known.minibatch_size)
    if known.bptt_horizon:
        tr["bptt_horizon"] = known.bptt_horizon
    if known.seed is not None:
        tr["seed"] = known.seed
        vec["seed"] = known.seed
    if known.data_dir:
        tr["data_dir"] = known.data_dir
    if known.rnn:
        args["rnn_name"] = "Recurrent"
    tr.setdefault("shaping_anneal_frac", 0.0)
    if known.shaping_anneal_frac is not None:
        tr["shaping_anneal_frac"] = float(known.shaping_anneal_frac)


def merge_saved(known, rest, args, saved):
    """--resume: the run's saved config.json settings, overridden by what is given again now (the
    architecture -- policy, rnn_name, rnn -- always from the run)."""
    given = explicit_keys(rest)
    for flag, targets in CONVENIENCE.items():
        if getattr(known, flag, None) is not None:
            given.update(targets)
    for sec in ("env", "train"):
        for k, v in (saved.get(sec) or {}).items():
            if (sec, k) not in given and not (sec == "env" and k in DERIVED_ENV_KEYS):
                args[sec][k] = v
    # SPEC §19.10.8 (ruling v0.5-G.4): the architecture always comes from the run; say so for any
    # architecture flag given again instead of ignoring it silently
    ignored = sorted(f"--{sec}.{k.replace('_', '-')}" for sec, k in given if sec in ("policy", "rnn"))
    if known.rnn and saved.get("rnn_name") is None:
        ignored.append("--rnn")
    if ignored:
        print(f"[train] {' '.join(ignored)} ignored on --resume (the run's architecture is kept)", file=sys.stderr)
    args["policy"] = dict(saved.get("policy") or args["policy"])
    args["rnn"] = dict(saved.get("rnn") or args["rnn"])
    args["rnn_name"] = saved.get("rnn_name")
    if known.seed is not None:
        args["vec"]["seed"] = known.seed
    else:
        args["vec"]["seed"] = args["train"]["seed"]


def parse_args(argv=None):
    conv = make_conv_parser()
    known, rest = conv.parse_known_args(argv)
    if known.help:
        print(__doc__)
        conv.print_help()
        print("\nplus every config key as --section.key (see pufferroyale/config/royale.ini)")
        sys.exit(0)
    if known.resume and known.init_from:
        raise SystemExit("--init-from starts a NEW run (weights only); it cannot be combined with --resume")
    args = load_args(known.config, rest)
    resume = None
    if known.resume:
        run_dir = os.path.abspath(known.resume)
        base = os.path.basename(os.path.normpath(run_dir))
        cfg_path, st_path = os.path.join(run_dir, "config.json"), os.path.join(run_dir, "trainer_state.pt")
        if not (os.path.isfile(cfg_path) and os.path.isfile(st_path)):
            raise SystemExit(f"--resume {known.resume}: no config.json + trainer_state.pt in {run_dir}")
        if not base.startswith(ENV_NAME + "_"):
            raise SystemExit(f"--resume {known.resume}: a train.py run directory is named {ENV_NAME}_<run_id>")
        with open(cfg_path) as f:
            saved = json.load(f)
        merge_saved(known, rest, args, saved)
        # SPEC §19.10.1: the run's anneal length N is reused unless --shaping-anneal-frac is given again
        frac_given = known.shaping_anneal_frac is not None or ("train", "shaping_anneal_frac") in explicit_keys(rest)
        n_saved = (saved.get("train") or {}).get("shaping_anneal_steps",
                                                 (saved.get("env") or {}).get("shaping_anneal_steps"))
        resume = {"run_dir": run_dir, "run_id": base[len(ENV_NAME) + 1:], "state_path": st_path,
                  "anneal_steps": None if (frac_given or n_saved is None) else int(n_saved)}
    apply_convenience(known, args)
    if resume:                                # the architecture and the location are the run's
        args["rnn_name"] = saved.get("rnn_name")
        args["train"]["data_dir"] = os.path.dirname(resume["run_dir"])
    tr = args["train"]
    tr["use_rnn"] = args["rnn_name"] is not None
    tr["env"] = ENV_NAME
    if args.get("neptune"):
        raise SystemExit("--neptune is not supported by the PufferRoyale scripts: use --wandb")
    return known, args, resume


def build(args):
    import pufferlib.vector
    import pufferroyale
    from pufferroyale.torch import Policy, Recurrent

    vec, env_kwargs = args["vec"], dict(args["env"])
    backend = vec["backend"]
    if backend == "PufferEnv":
        vecenv = pufferlib.vector.make(pufferroyale.Royale, env_kwargs=env_kwargs,
                                       backend=pufferlib.vector.PufferEnv)
    else:
        kw = dict(num_workers=vec["num_workers"], batch_size=vec["batch_size"])
        if backend == "Multiprocessing":
            kw["zero_copy"] = vec["zero_copy"]
        vecenv = pufferlib.vector.make(pufferroyale.Royale, env_kwargs=env_kwargs,
                                       backend=getattr(pufferlib.vector, backend),
                                       num_envs=vec["num_envs"], seed=vec["seed"], **kw)
    policy = Policy(vecenv.driver_env, **args["policy"])
    if args["rnn_name"]:
        policy = Recurrent(vecenv.driver_env, policy, **args["rnn"])
    return vecenv, policy.to(args["train"]["device"])


def resume_mismatch(run_dir, diff):
    """The SystemExit message of a --resume whose policy differs from the saved run's (SPEC §19.10.8)."""
    return (f"--resume {run_dir}: this invocation builds a different policy than the saved run ({'; '.join(diff)}). "
            f"The architecture, placement grid and rnn setting come from the run: drop the overrides that change "
            f"them (e.g. --env.placement-grid, --policy.*, --rnn.*)")


class RunLogger:
    """PuffeRL logger with a fixed run id (PuffeRL's NoLogger uses the wall clock), used on --resume."""

    def __init__(self, run_id):
        self.run_id = run_id

    def log(self, logs, step):
        pass

    def close(self, model_path, early_stop):
        pass


def main():
    known, args, resume = parse_args()
    import numpy as np
    import torch
    from pufferroyale.league import json_safe
    from pufferroyale.trainer import shaping_env_kwargs, warn_shaping_clip

    tr = args["train"]
    saved_state = None
    if resume:
        saved_state = torch.load(resume["state_path"], map_location="cpu", weights_only=False)
        if int(saved_state["global_step"]) >= int(tr["total_timesteps"]):
            summary = {"global_step": int(saved_state["global_step"]), "epochs": int(saved_state["update"]),
                       "nothing_to_do": True, "run_dir": resume["run_dir"]}
            print(f"[train] {resume['run_dir']} already reached {saved_state['global_step']} >= --total-timesteps "
                  f"{tr['total_timesteps']}: nothing to do")
            print(json.dumps(summary, allow_nan=False))
            return
    step0 = int(saved_state["global_step"]) if saved_state else 0
    epoch0 = int(saved_state["update"]) if saved_state else 0
    # SPEC §19.7.1 / §19.10.1: reward_gamma = train.gamma; shaping anneal over N = ceil(F * total / R) env
    # steps, fixed when the run is created (recorded under train) and reused on --resume unless F is given again
    rows = learner_rows(args)
    args["env"].update(shaping_env_kwargs(tr["gamma"], tr.get("shaping_anneal_frac") or 0.0, tr["total_timesteps"],
                                          rows, step0, anneal_steps=(resume or {}).get("anneal_steps")))
    tr["shaping_anneal_steps"] = int(args["env"]["shaping_anneal_steps"])
    warn_shaping_clip(args["env"], tr.get("reward_clip", 1.0), "train")         # SPEC §19.10.2
    base_seed = int(tr["seed"])
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    if resume:                              # fresh match seeds for the continuation, still deterministic
        args["vec"]["seed"] = int(args["vec"]["seed"]) + 7919 * epoch0
    vecenv, policy = build(args)
    if vecenv.num_agents != rows:
        vecenv.close()
        raise SystemExit(f"internal: the vecenv has {vecenv.num_agents} agent rows, expected {rows}")
    if resume:                              # SPEC §19.10.8: a clear message, not a load_state_dict traceback
        from pufferroyale.league import architecture_diff, load_state_dict_file
        diff = architecture_diff(policy, load_state_dict_file(os.path.join(resume["run_dir"], saved_state["model_name"])))
        if diff:
            vecenv.close()
            raise SystemExit(resume_mismatch(resume["run_dir"], diff))
    batch = effective_batch_size(tr, vecenv.num_agents)
    if int(tr["total_timesteps"]) < batch:
        vecenv.close()
        raise SystemExit(f"--total-timesteps {tr['total_timesteps']} is smaller than one PPO batch ({batch} agent "
                         f"steps = batch_size, or num_agents x bptt_horizon when 'auto'): nothing would be trained")
    if known.init_from:
        from pufferroyale.league import load_weights
        try:
            tr["init_from"] = os.path.abspath(load_weights(policy, known.init_from))
        except (ValueError, FileNotFoundError) as e:
            vecenv.close()
            raise SystemExit(f"--init-from: {e}")
    # SPEC §19.7.5: PufferLib's --wandb flags through our WandbLogger (PufferLib's interface; its own
    # class needs wandb.util.generate_id, gone in recent wandb); wandb is imported only then
    run_id = resume["run_id"] if resume else None
    if args["wandb"]:
        from pufferroyale.trainer import WandbLogger
        logger = WandbLogger(args, load_id=run_id)
    else:
        logger = RunLogger(run_id) if run_id else None
    trainer_cfg = dict(tr)
    if resume:
        trainer_cfg["seed"] = int(args["vec"]["seed"])          # the vecenv's async_reset seed
    # MMDPuffeRL with mmd_coef = 0 is PuffeRL's PPO bit for bit, with the bf16 autocast-cache fix
    # (SPEC §18.6), a daemon utilization thread, reward_clip and the entropy split (SPEC §19.2)
    from pufferroyale.trainer import MMDPuffeRL
    try:
        trainer = MMDPuffeRL(trainer_cfg, vecenv, policy, logger)
    except BaseException:
        vecenv.close()
        if hasattr(logger, "abort"):
            logger.abort()
        raise
    run_dir = os.path.join(tr["data_dir"], f"{tr['env']}_{trainer.logger.run_id}")
    os.makedirs(run_dir, exist_ok=True)
    if resume:
        dev = tr["device"]
        try:
            policy.load_state_dict(torch.load(os.path.join(resume["run_dir"], saved_state["model_name"]),
                                              map_location=dev, weights_only=True))
        except RuntimeError as e:
            trainer.utilization.stop()
            vecenv.close()
            if hasattr(trainer.logger, "abort"):
                trainer.logger.abort()
            raise SystemExit(resume_mismatch(resume["run_dir"], [str(e).strip().splitlines()[-1].strip()]))
        trainer.load_training_state(saved_state["optimizer_state_dict"], epoch0, step0)
        if saved_state.get("torch_rng") is not None:
            torch.set_rng_state(saved_state["torch_rng"].cpu())
    save_checkpoint = trainer.save_checkpoint

    def guarded_save():
        """PuffeRL saves from inside train(); never write a checkpoint with non-finite weights. The
        trainer state also gets the torch RNG state (for --resume)."""
        if not all(torch.isfinite(p).all() for p in policy.parameters()):
            print("[train] non-finite weights: checkpoint NOT written", file=sys.stderr)
            return None
        path = save_checkpoint()
        st_path = os.path.join(run_dir, "trainer_state.pt")
        if path and os.path.isfile(st_path):
            st = torch.load(st_path, map_location="cpu", weights_only=False)
            if int(st.get("update", -1)) == int(trainer.epoch):
                st["torch_rng"] = torch.get_rng_state()
                torch.save(st, st_path + ".tmp")
                os.replace(st_path + ".tmp", st_path)
        return path

    trainer.save_checkpoint = guarded_save
    if known.quiet:
        trainer.print_dashboard = lambda *a, **k: None
    from pufferroyale.decks import expanded_deck_sets
    with open(os.path.join(run_dir, "config.json"), "w") as f:      # + the expanded deck sets (SPEC §19.10.5)
        json.dump({"policy": args["policy"], "rnn_name": args["rnn_name"], "rnn": args["rnn"],
                   "env": dict(args["env"], **expanded_deck_sets(args["env"])),
                   "train": {k: v for k, v in tr.items() if isinstance(v, (int, float, str, bool))}},
                  f, indent=2, default=str)
    t0 = time.time()
    nonfinite = []
    last_losses, stats_hist, sps_hist = {}, {}, []
    epoch_print = 0
    bad_params = []
    model_path = None
    try:
        while trainer.global_step < tr["total_timesteps"]:
            stats = trainer.evaluate()
            for k, v in stats.items():
                if isinstance(v, list) and v:
                    stats_hist.setdefault(k, []).extend(float(x) for x in v)
            trainer.train()
            if trainer.losses:
                last_losses = dict(trainer.losses)
                for k, v in last_losses.items():
                    if k != "explained_variance" and not math.isfinite(float(v)):
                        nonfinite.append((trainer.global_step, k, str(float(v))))
            sps = (trainer.global_step - step0) / max(time.time() - t0, 1e-9)
            sps_hist.append(sps)
            if known.quiet and trainer.epoch >= epoch_print:
                epoch_print = trainer.epoch + 10
                key = "learner_score"  # SPEC §14.6: the policy's own score (team 0's `score` in self-play)
                score = np.mean(stats_hist[key][-50:]) if stats_hist.get(key) else float("nan")
                print(f"[train] step {trainer.global_step:>9d}  epoch {trainer.epoch:>4d}  sps {sps:7.0f}  "
                      f"pg {last_losses.get('policy_loss', float('nan')):+.4f}  "
                      f"v {last_losses.get('value_loss', float('nan')):.4f}  "
                      f"ent {last_losses.get('entropy', float('nan')):.3f}  learner_score {score:.3f}", flush=True)
            if nonfinite:
                break
        bad_params = [n for n, p in policy.named_parameters() if not torch.isfinite(p).all()]
        if nonfinite or bad_params:                 # keep the last good checkpoint; write nothing new
            vecenv.close()
            model_path = None
            if hasattr(trainer.logger, "abort"):
                trainer.logger.abort()
        else:
            model_path = trainer.close()
            trainer.logger.close(model_path, early_stop=False)
    finally:
        trainer.utilization.stop()
    elapsed = time.time() - t0
    summary = {
        "global_step": int(trainer.global_step),
        "epochs": int(trainer.epoch),
        "elapsed_s": round(elapsed, 2),
        "sps": round((trainer.global_step - step0) / max(elapsed, 1e-9), 1),
        "final_losses": {k: (float(v) if v == v else None) for k, v in last_losses.items()},
        "nonfinite_losses": nonfinite,
        "nonfinite_params": bad_params,
        "episodes": len(stats_hist.get("score", [])) and sum(stats_hist.get("n", [0])),
        "env_stats": {k: float(np.mean(v)) for k, v in stats_hist.items() if v},
        "checkpoint": model_path,
        "run_dir": run_dir,
        "resumed_from_step": step0 if resume else None,
    }
    summary = json_safe(summary)                        # strict JSON: non-finite numbers as strings
    print(json.dumps(summary, indent=2, default=str, allow_nan=False))
    if known.summary_json:
        with open(known.summary_json, "w") as f:
            json.dump(summary, f, indent=2, default=str, allow_nan=False)
    if nonfinite or bad_params:
        print("NON-FINITE values during training", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
