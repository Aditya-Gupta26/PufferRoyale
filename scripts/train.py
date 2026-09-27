#!/usr/bin/env python3
"""Train a PufferRoyale policy with PufferLib 3.0 PuffeRL (SPEC §11).

Loads pufferroyale/config/royale.ini (merged over pufferlib's default.ini); every config key
can be overridden as --section.key value (e.g. --train.learning-rate 1e-4,
--env.opponent random). Convenience flags cover the common ones.

    python scripts/train.py --total-timesteps 100000                 # self-play smoke run
    python scripts/train.py --num-agents 1 --opponent heuristic --total-timesteps 100000
    python scripts/train.py --rnn --device mps
    python scripts/train.py --backend Multiprocessing --workers 4 --num-envs 8
    python scripts/train.py --deck0 bait --deck1 random --seed 3 --data-dir /tmp/runs

Checkpoints go to experiments/pufferroyale_<run_id>/ (PuffeRL's model_*.pt and
trainer_state.pt, plus config.json describing the policy for scripts/eval.py). A JSON
summary (steps, SPS, final losses, episode stats) is printed at the end and optionally
written with --summary-json.
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


def parse_args():
    conv = argparse.ArgumentParser(add_help=False)
    conv.add_argument("--config", default=DEFAULT_CONFIG)
    conv.add_argument("--total-timesteps", type=int)
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
    conv.add_argument("--quiet", action="store_true", help="concise log lines instead of the dashboard")
    conv.add_argument("--summary-json")
    conv.add_argument("-h", "--help", action="store_true")
    known, rest = conv.parse_known_args()
    if known.help:
        print(__doc__)
        conv.print_help()
        print("\nplus every config key as --section.key (see pufferroyale/config/royale.ini)")
        sys.exit(0)
    import pufferlib.pufferl as pufferl

    argv0 = sys.argv[0]
    sys.argv = [argv0] + rest
    try:
        args = pufferl.load_config_file(known.config)
    finally:
        sys.argv = [argv0] + rest
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
    tr["use_rnn"] = args["rnn_name"] is not None
    tr["env"] = "pufferroyale"
    return known, args


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


def main():
    known, args = parse_args()
    import numpy as np
    import torch
    import pufferlib.pufferl as pufferl

    tr = args["train"]
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])
    vecenv, policy = build(args)
    batch = effective_batch_size(tr, vecenv.num_agents)
    if int(tr["total_timesteps"]) < batch:
        vecenv.close()
        raise SystemExit(f"--total-timesteps {tr['total_timesteps']} is smaller than one PPO batch ({batch} agent "
                         f"steps = batch_size, or num_agents x bptt_horizon when 'auto'): nothing would be trained")
    # MMDPuffeRL with mmd_coef = 0 is PuffeRL's PPO bit for bit, with the bf16 autocast-cache fix
    # (SPEC §18.6) and a daemon utilization thread; see pufferroyale/trainer.py
    from pufferroyale.trainer import MMDPuffeRL
    trainer = MMDPuffeRL(tr, vecenv, policy)
    save_checkpoint = trainer.save_checkpoint

    def guarded_save():
        """PuffeRL saves from inside train(); never write a checkpoint with non-finite weights."""
        if all(torch.isfinite(p).all() for p in policy.parameters()):
            return save_checkpoint()
        print("[train] non-finite weights: checkpoint NOT written", file=sys.stderr)
        return None

    trainer.save_checkpoint = guarded_save
    if known.quiet:
        trainer.print_dashboard = lambda *a, **k: None
    run_dir = os.path.join(tr["data_dir"], f"{tr['env']}_{trainer.logger.run_id}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"policy": args["policy"], "rnn_name": args["rnn_name"], "rnn": args["rnn"],
                   "env": args["env"], "train": {k: v for k, v in tr.items() if isinstance(v, (int, float, str, bool))}},
                  f, indent=2, default=str)
    t0 = time.time()
    nonfinite = []
    last_losses, stats_hist, sps_hist = {}, {}, []
    epoch_print = 0
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
            sps = trainer.global_step / max(time.time() - t0, 1e-9)
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
        else:
            model_path = trainer.close()
    finally:
        trainer.utilization.stop()
    elapsed = time.time() - t0
    summary = {
        "global_step": int(trainer.global_step),
        "epochs": int(trainer.epoch),
        "elapsed_s": round(elapsed, 2),
        "sps": round(trainer.global_step / max(elapsed, 1e-9), 1),
        "final_losses": {k: (float(v) if v == v else None) for k, v in last_losses.items()},
        "nonfinite_losses": nonfinite,
        "nonfinite_params": bad_params,
        "episodes": len(stats_hist.get("score", [])) and sum(stats_hist.get("n", [0])),
        "env_stats": {k: float(np.mean(v)) for k, v in stats_hist.items() if v},
        "checkpoint": model_path,
        "run_dir": run_dir,
    }
    from pufferroyale.league import json_safe
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
