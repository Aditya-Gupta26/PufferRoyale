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

With --resume <run dir> the run continues from its saved state; --total-timesteps is the
ABSOLUTE target (a run that already reached it does nothing). Pool-defining options (anchors,
pfsp, fractions, max snapshots) and --rnn always come from the saved run; optimizer settings
given on the command line (--learning-rate, --train.adam-* ...) are applied to the resumed
optimizer; --train.* / --env.* / --policy.* overrides of the original run are kept. The run
directory is relocatable (pool snapshot paths are stored relative to it). A resumed run is
deterministic given the saved state, but not bit-identical to an uninterrupted run: matches in
progress at the save are re-dealt (from seed + 7919 * epoch); the pool, wrapper and torch RNG
states continue from the save.

Robustness (SPEC §18.7): every epoch the losses and the learner's weights are checked; a
non-finite epoch is never snapshotted or saved -- the last good save stays, nonfinite.json
describes the failure, and the process exits 1. Any error after start-up exits promptly with a
non-zero status and saves nothing.

Any PuffeRL / env / policy key of pufferroyale/config/royale.ini can also be overridden as
--train.key value, --env.key value, --policy.key value (e.g. --train.learning-rate 1e-4).
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
                rnn=False, opponent_greedy=False, anneal_lr=False, run_id=None, print_interval=1)


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
                    help="policy opponents take the argmax instead of sampling")
    ap.add_argument("--print-interval", type=int, help="print a progress line every N epochs (default 1)")
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


def env_kwargs(a, ini):
    kw = {k: v for k, v in ini["env"].items() if k not in ("num_envs", "num_agents", "opponent", "learner_side")}
    kw["deck0"], kw["deck1"] = deck_value(a["deck0"]), deck_value(a["deck1"])
    if a["frame_skip"]:
        kw["frame_skip"] = int(a["frame_skip"])
    return kw


# ------------------------------------------------------------------------------------ build
class RunLogger:
    """PuffeRL logger with a fixed run id (PuffeRL's NoLogger uses the wall clock)."""

    def __init__(self, run_id):
        self.run_id = run_id

    def log(self, logs, step):
        pass

    def close(self, model_path, early_stop):
        pass


def build(a, ini, pool, run_dir, run_id, dashboard=False, reset_seed=None, env_overrides=None):
    """(LeagueVecEnv, policy, trainer, cfg) for resolved args `a`. The trainer is always MMDPuffeRL
    (with mmd_coef = 0 it is plain PuffeRL PPO, bit for bit, plus the SPEC §18.6 AMP fix)."""
    import torch
    from pufferroyale.league import LeagueVecEnv
    from pufferroyale.torch import Policy, Recurrent
    from pufferroyale.trainer import MMDPuffeRL

    cfg = build_config(a, ini, run_dir)
    if reset_seed is not None:
        cfg["seed"] = int(reset_seed)        # the vecenv's async_reset seed
    kw = env_kwargs(a, ini)
    kw.update(env_overrides or {})
    lv = LeagueVecEnv(pool, num_envs=int(a["num_envs"]), seed=int(a["seed"]), device=a["device"],
                      opponent_greedy=bool(a["opponent_greedy"]), **kw)
    try:
        policy = Policy(lv.driver_env, **ini["policy"])
        if a["rnn"]:
            policy = Recurrent(lv.driver_env, policy, **ini["rnn"])
        policy = policy.to(a["device"])
        lv.set_policy(policy)
        cls = MMDPuffeRL
        if not dashboard:                 # progress lines instead of the (screen-clearing) dashboard
            cls = type("MMDPuffeRL", (MMDPuffeRL,), {"print_dashboard": lambda self, *args, **kw: None})
        trainer = cls(cfg, lv, policy, logger=RunLogger(run_id))
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
        self.a = a
        self.epoch0 = int(self.saved["epoch"]) if self.saved else 0
        self.step0 = int(self.saved["global_step"]) if self.saved else 0
        # --section.key overrides of the original run are kept; new ones given now win (argparse: last wins)
        self.ini_overrides = list(self.saved.get("ini_overrides", [])) if self.saved else []
        self.ini_overrides += list(rest)
        self.ini = load_ini(self.ini_overrides)
        self.pool = OpponentPool(anchors=parse_anchors(a["anchors"]), max_snapshots=int(a["max_snapshots"]),
                                 pfsp=a["pfsp"], pfsp_eps=float(a["pfsp_eps"]), self_play_frac=float(a["self_play_frac"]),
                                 anchor_frac=float(a["anchor_frac"]), seed=int(a["seed"]))
        if self.saved:
            self.pool.load_state_dict(absolute_pool_state(self.saved["pool"], self.run_dir))
        self.done_already = self.step0 >= int(a["total_timesteps"])
        if self.done_already:
            return
        build_config(a, self.ini, self.run_dir)           # size / total checks before anything starts
        os.makedirs(self.run_dir, exist_ok=True)
        # a resumed run continues on fresh match seeds (seed + 7919 * epoch), still deterministic
        seed_now = int(a["seed"]) + 7919 * self.epoch0
        torch.manual_seed(seed_now)
        np.random.seed(seed_now % (2 ** 32))
        self.lv, self.policy, self.trainer, self.cfg = build(a, self.ini, self.pool, self.run_dir, self.run_id,
                                                             dashboard=known.dashboard, reset_seed=seed_now)
        try:
            self.snapshots = list(self.saved.get("snapshot_files", [])) if self.saved else []
            if self.saved:
                self._restore()
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump({"policy": self.ini["policy"], "rnn_name": "Recurrent" if a["rnn"] else None,
                           "rnn": self.ini["rnn"], "env": env_kwargs(a, self.ini), "league": a}, f, indent=2, default=str)
        except BaseException:
            self.abort()
            raise

    def _restore(self):
        """Load learner + optimizer + counters + RNG states of the saved run (SPEC §18.7)."""
        import torch
        a, tr = self.a, self.trainer
        dev = a["device"]
        self.policy.load_state_dict(torch.load(os.path.join(self.run_dir, "learner.pt"), map_location=dev,
                                               weights_only=True))
        ts = torch.load(os.path.join(self.run_dir, "trainer_state.pt"), map_location=dev, weights_only=False)
        # optimizer moments from the save; lr / betas / eps from THIS invocation's config; the cosine
        # schedule (with --anneal-lr) continued at the saved epoch
        tr.load_training_state(ts["optimizer_state_dict"], ts["epoch"], ts["global_step"])
        if ts.get("torch_rng") is not None:
            torch.set_rng_state(ts["torch_rng"].cpu())
        self.lv.episodes = int(self.saved.get("episodes", 0))
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
                 "snapshot_files": self.snapshots, "episodes": int(self.lv.episodes)}
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
        self.last_saved_epoch = self.epoch0 if self.saved else None
        self.diagnostic = None
        while tr.global_step < int(a["total_timesteps"]):
            if not self.known.dashboard:
                tr.last_log_time = -1e18   # PuffeRL only publishes an epoch's losses when it logs (>= 0.25 s apart)
            tr.evaluate()
            tr.train()
            losses = {k: float(v) for k, v in tr.losses.items()}
            # SPEC §18.7: check BEFORE any snapshot / save
            self.nonfinite = [(int(tr.epoch), k, str(v)) for k, v in losses.items()
                              if k != "explained_variance" and not math.isfinite(v)]
            self.bad_params = [n for n, p in self.policy.named_parameters() if not torch.isfinite(p).all()]
            window = collections.OrderedDict()
            for spec, res, seat in self.lv.drain_results():
                g = window.setdefault(spec, [0, 0.0])
                g[0] += 1
                g[1] += 0.5 * (res + 1)
            bad = bool(self.nonfinite or self.bad_params)
            snap = None
            if not bad and interval > 0 and tr.epoch % interval == 0:
                snap = self.snapshot()
                self.save()
                self.last_saved_epoch = int(tr.epoch)
            sps = (tr.global_step - s_start) / max(time.time() - t_start, 1e-9)
            rec = {"epoch": int(tr.epoch), "global_step": int(tr.global_step), "sps": round(sps, 1),
                   "losses": {k: (v if math.isfinite(v) else None) for k, v in losses.items()},
                   "results": {s: {"games": g, "score": sc / g} for s, (g, sc) in window.items()},
                   "pool_size": self.pool.pool_size(), "snapshot": os.path.basename(snap) if snap else None}
            if bad:
                rec["nonfinite"] = True
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec, allow_nan=False) + "\n")
            if not self.known.dashboard and (tr.epoch % pi == 0 or snap or bad):
                res = " ".join(f"{short(s)}={sc / g:.2f}/{g}" for s, (g, sc) in window.items()) or "-"
                kl = f" kl {losses['mmd_kl']:.2e}" if "mmd_kl" in losses and float(a["mmd_coef"]) > 0 else ""
                print(f"[league] step {tr.global_step:>8d} epoch {tr.epoch:>4d} sps {sps:6.0f} "
                      f"pg {losses.get('policy_loss', float('nan')):+.4f} v {losses.get('value_loss', float('nan')):.4f} "
                      f"ent {losses.get('entropy', float('nan')):.3f}{kl} pool {self.pool.pool_size()} | {res}"
                      + (f" | snapshot {os.path.basename(snap)}" if snap else "")
                      + (" | NON-FINITE: not saved" if bad else ""), flush=True)
            if bad:
                self.diagnostic = self._write_diagnostic(losses)
                break

    def finish(self):
        """After a clean run: final save + model_<epoch>.pt, then release the trainer and envs."""
        import torch
        tr = self.trainer
        try:
            self.save()
            model = os.path.join(self.run_dir, f"model_{tr.epoch:06d}.pt")
            atomic_torch_save(cpu_state(self.policy), model)
        finally:
            self.abort()
        return model

    def abort(self):
        """Release the trainer's monitor thread and the envs without writing anything."""
        if self.trainer is not None:
            self.trainer.utilization.stop()
        if self.lv is not None:
            self.lv.close()


def summary_of(run, model=None, extra=None):
    pool = run.pool
    out = {"run_dir": run.run_dir, "run_id": run.run_id,
           "global_step": int(run.trainer.global_step) if run.trainer else run.step0,
           "epoch": int(run.trainer.epoch) if run.trainer else run.epoch0,
           "elapsed_s": round(time.time() - run.t0, 2), "model": model,
           "pool": {short(k): round(v, 4) for k, v in pool.weights().items()},
           "stats": {short(k): v for k, v in pool.stats().items()}}
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
        print(f"[league] {run.run_dir} already reached {run.step0} >= --total-timesteps {run.a['total_timesteps']}: nothing to do")
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
