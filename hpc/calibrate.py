#!/usr/bin/env python3
"""Throughput calibration on one GPU (docs/HPC_HANDOFF.md §6, Phase 3). Runs INSIDE our image, from
the checkout (hpc/calibrate.sbatch does that), and measures every combination of the list options:

    python hpc/calibrate.py --out DIR --name stage1 --script league --shape stage1 \\
        --num-envs 256 512 --batch-k 1 2 --minibatch-size 8192 --copies 1 2 4 --seconds 120 -- <extra flags>

league shapes (scripts/league_train.py):
    stage1   --anchors bot:heuristic --self-play-frac 0 --anchor-frac 1.0     (a stage-1 rung)
    A        --anchors none --self-play-frac 1.0 --anchor-frac 0             (stage-3 run A)
    ckptN    N frozen checkpoints as anchors, --anchor-frac 0.65 --self-play-frac 0.35 (stage-3 B/C with N
             snapshots: the same number of distinct opponents per step). The N files are made once by a
             short run (--snapshot-interval 1) under <out>/prep/.
train (scripts/train.py, reference): self-play, PufferEnv or --workers W (Multiprocessing).

Batch: --batch-size B, or --batch-k K for num_envs x 64 x K (league: one learner row per match; train:
num_envs x 2 rows). Configurations PuffeRL would reject are skipped. --copies C runs C identical processes
(own run ids and seeds) on the same GPU at once: stacking. Each process runs through hpc/peak_mem.py, so
its torch.cuda.max_memory_allocated is recorded.

Measurement: league runs train until --seconds after every copy logged its first epoch, then are stopped
(SIGTERM). SPS per copy is taken between its first and last logged epoch (history.jsonl: cumulative SPS
per epoch), so start-up and CUDA initialisation do not count; train.py runs a fixed number of epochs
(--train-epochs) and reports its own SPS (from the end of start-up). nvidia-smi gives the device's
utilization (including the GPU heartbeat of the job, which holds it >= 75%) and memory; psutil gives the
CPU use (cores, summed over every process tree) and RSS.

cuDNN: every run gets `--train.torch-deterministic=` (an EMPTY value: PufferLib's bool parser turns any
non-empty string, "False" and "0" included, into True) unless --deterministic is given, because the
deterministic cuDNN algorithms make the conditional head's convolutions ~45x slower on H100 (DECISIONS
D82). Each row records the mode ("cudnn_deterministic").

Output: <out>/results.jsonl (one object per configuration, appended) and a markdown table on stdout.
"""
import argparse
import glob
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
HORIZON = 64
SHAPES = {"stage1": ["--anchors", "bot:heuristic", "--self-play-frac", "0", "--anchor-frac", "1.0"],
          "A": ["--anchors", "none", "--self-play-frac", "1.0", "--anchor-frac", "0"]}


def prepare_ckpts(out, n, device):
    """n frozen checkpoint files for the ckptN shapes (made once by a tiny run with snapshot interval 1)."""
    run = os.path.join(out, "prep", "league", "ckpts")
    snaps = sorted(glob.glob(os.path.join(run, "snap_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    if len(snaps) < n:
        shutil.rmtree(run, ignore_errors=True)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "league_train.py"), "--device", device,
               "--num-envs", "8", "--batch-size", "512", "--minibatch-size", "512", "--anchors", "bot:random",
               "--self-play-frac", "0", "--anchor-frac", "1.0", "--snapshot-interval", "1", "--max-snapshots",
               str(max(n, 64)), "--total-timesteps", str(512 * max(n, 64)), "--data-dir", os.path.join(out, "prep"),
               "--run-id", "ckpts", "--print-interval", "1000"]
        print(f"[calibrate] making {max(n, 64)} checkpoint opponents: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        snaps = sorted(glob.glob(os.path.join(run, "snap_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    return snaps[-n:]


def configs(a):
    out, skipped = [], []
    batches = [("B", b) for b in (a.batch_size or [])] + [("K", k) for k in (a.batch_k or [])]
    grid = itertools.product(a.shape if a.script == "league" else ["-"], a.num_envs, batches, a.minibatch_size,
                             a.rnn, a.workers or [0], a.copies)
    for shape, envs, (bkind, bval), mb, rnn, workers, copies in grid:
        rows = envs if a.script == "league" else envs * 2 * max(1, workers)
        batch = bval if bkind == "B" else envs * HORIZON * bval
        c = dict(name=a.name, script=a.script, shape=shape, num_envs=envs, batch_size=batch, minibatch_size=mb,
                 rnn=rnn == "on", workers=workers, copies=copies, rows=rows, cudnn_deterministic=bool(a.deterministic))
        why = None
        if batch < rows * HORIZON:
            why = f"batch {batch} < rows {rows} x {HORIZON}"
        elif batch % mb or mb % HORIZON or mb > batch:
            why = f"minibatch {mb} must divide batch {batch}, be <= it and a multiple of {HORIZON}"
        elif a.script == "train" and batch % (rows * HORIZON):
            why = f"batch {batch} not a multiple of rows x {HORIZON} = {rows * HORIZON}"
        (skipped if why else out).append(dict(c, skip=why) if why else c)
    return out, skipped


def command(c, a, run_dir, run_id, seed, ckpts):
    wrap = [sys.executable, os.path.join(HERE, "peak_mem.py")]
    if c["script"] == "league":
        pool = SHAPES.get(c["shape"])
        if pool is None:                                   # ckptN
            pool = ["--anchors", ",".join(f"ckpt:{p}" for p in ckpts[c["shape"]]), "--anchor-frac", "0.65",
                    "--self-play-frac", "0.35"]
        cmd = wrap + [os.path.join(ROOT, "scripts", "league_train.py"), "--device", a.device,
                      "--total-timesteps", str(10 ** 12), "--num-envs", str(c["num_envs"]),
                      "--batch-size", str(c["batch_size"]), "--minibatch-size", str(c["minibatch_size"]),
                      "--snapshot-interval", "100000", "--seed", str(seed), "--deck0", a.deck, "--deck1", a.deck,
                      "--data-dir", run_dir, "--run-id", run_id, "--print-interval", "100000"] + pool
    else:
        cmd = wrap + [os.path.join(ROOT, "scripts", "train.py"), "--device", a.device,
                      "--total-timesteps", str(c["batch_size"] * a.train_epochs), "--num-envs", str(c["num_envs"]),
                      "--num-agents", "2", "--batch-size", str(c["batch_size"]), "--minibatch-size",
                      str(c["minibatch_size"]), "--deck0", a.deck, "--deck1", a.deck, "--seed", str(seed),
                      "--data-dir", run_dir, "--quiet", "--summary-json", os.path.join(run_dir, f"{run_id}.json")]
        if c["workers"]:
            cmd += ["--backend", "Multiprocessing", "--workers", str(c["workers"])]
    if c["rnn"]:
        cmd.append("--rnn")
    if not c["cudnn_deterministic"]:
        cmd.append("--train.torch-deterministic=")         # empty value = False (see the module docstring)
    return cmd + list(a.extra)


def league_points(path):
    """[(seconds since the loop start, global_step)] of a league run's logged epochs."""
    pts = []
    if not os.path.isfile(path):
        return pts
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("sps"):
            pts.append((r["global_step"] / r["sps"], r["global_step"]))
    return pts


def steady(pts):
    if len(pts) < 3:
        return None
    (t0, s0), (t1, s1) = pts[1], pts[-1]
    return (s1 - s0) / (t1 - t0) if t1 > t0 else None


class Sampler(threading.Thread):
    def __init__(self, pids, every):
        super().__init__(daemon=True)
        self.pids, self.every = pids, every
        self.util, self.mem, self.cpu, self.rss = [], [], [], []
        self.halt = threading.Event()

    def run(self):
        import psutil
        roots = []
        for p in self.pids:
            try:
                roots.append(psutil.Process(p))
            except psutil.Error:
                pass
        seen = {}
        while not self.halt.wait(self.every):
            cpu, rss = 0.0, 0
            for root in roots:
                try:
                    tree = [root] + root.children(recursive=True)
                except psutil.Error:
                    continue
                for p in tree:
                    try:
                        if p.pid not in seen:
                            seen[p.pid] = p
                            p.cpu_percent(None)
                            continue
                        cpu += seen[p.pid].cpu_percent(None)
                        rss += p.memory_info().rss
                    except psutil.Error:
                        continue
            self.cpu.append(cpu / 100.0)
            self.rss.append(rss)
            try:
                line = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                       "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                      timeout=10).stdout.strip().splitlines()[0]
                u, m = (float(x) for x in line.split(","))
                self.util.append(u)
                self.mem.append(m)
            except Exception:  # noqa: BLE001 -- a missed sample is fine
                pass


def mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def measure(c, a, idx, ckpts):
    tag = f"{a.name}{idx:03d}"
    run_dir = os.path.join(a.out, "runs")
    os.makedirs(run_dir, exist_ok=True)
    procs, logs, hist, memj = [], [], [], []
    t_launch = time.time()
    for k in range(c["copies"]):
        run_id = f"{tag}_c{k}"
        env = dict(os.environ, PR_PEAK_MEM_JSON=os.path.join(a.out, f"{run_id}_mem.json"))
        memj.append(env["PR_PEAK_MEM_JSON"])
        logf = open(os.path.join(a.out, f"{run_id}.log"), "w")
        logs.append(logf)
        procs.append(subprocess.Popen(command(c, a, run_dir, run_id, 1000 + k, ckpts), cwd=ROOT, stdout=logf,
                                      stderr=subprocess.STDOUT, env=env, start_new_session=True))
        hist.append(os.path.join(run_dir, "league", run_id, "history.jsonl") if c["script"] == "league"
                    else os.path.join(run_dir, f"{run_id}.json"))
    sampler = Sampler([p.pid for p in procs], a.sample_s)
    status, t_first = "ok", None
    try:
        while True:
            alive = [p.poll() is None for p in procs]
            now = time.time()
            if c["script"] == "league":
                if t_first is None and all(os.path.isfile(h) and os.path.getsize(h) > 0 for h in hist):
                    t_first = now
                    sampler.start()
                if t_first is not None and now - t_first >= a.seconds:
                    break
                if not all(alive):
                    status = "a copy exited early"
                    break
            else:
                if t_first is None and now - t_launch > 20:
                    t_first = now
                    sampler.start()
                if not any(alive):
                    break
                if now - t_launch > a.train_timeout:
                    status = "timeout"
                    break
            if t_first is None and now - t_launch > a.startup_timeout:
                status = f"no epoch within {a.startup_timeout:.0f}s"
                break
            time.sleep(1.0)
    finally:
        sampler.halt.set()
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for f in logs:
            f.close()
    sps = []
    for h in hist:
        if c["script"] == "league":
            sps.append(steady(league_points(h)))
        else:
            try:
                sps.append(float(json.load(open(h))["sps"]))
            except (OSError, ValueError, KeyError):
                sps.append(None)
    peaks = []
    for m in memj:
        try:
            peaks.append(json.load(open(m))["max_memory_allocated"] / 2 ** 30)
        except (OSError, ValueError, KeyError):
            pass
    ok = [s for s in sps if s]
    res = dict(c, tag=tag, status=status if len(ok) == len(sps) else f"{status}; {len(sps) - len(ok)} copies unmeasured",
               sps_total=round(sum(ok), 1) if ok else None, sps_per_copy=[round(s, 1) if s else None for s in sps],
               peak_alloc_gib=round(max(peaks), 2) if peaks else None,
               gpu_mem_used_max_mib=max(sampler.mem) if sampler.mem else None,
               gpu_util_mean=mean(sampler.util), cpu_cores_mean=mean(sampler.cpu),
               rss_max_gib=round(max(sampler.rss) / 2 ** 30, 2) if sampler.rss else None,
               startup_s=round(t_first - t_launch, 1) if t_first else None)
    bad = [p.returncode for p in procs if p.returncode not in (0, -signal.SIGTERM, 128 + signal.SIGTERM, 143)]
    if bad or not ok:
        tails = []
        for k in range(c["copies"]):
            with open(os.path.join(a.out, f"{tag}_c{k}.log")) as f:
                tails.append(f.read()[-1200:])
        res["log_tail"] = tails
    if not a.keep_runs:
        for k in range(c["copies"]):
            shutil.rmtree(os.path.join(run_dir, "league", f"{tag}_c{k}"), ignore_errors=True)
            for p in glob.glob(os.path.join(run_dir, f"pufferroyale_*")):
                shutil.rmtree(p, ignore_errors=True)
    return res


def table(rows):
    head = ("| name | shape | envs | workers | rnn | det. cuDNN | batch | minibatch | copies | SPS total | SPS/copy "
            "| peak alloc GiB | GPU mem MiB | GPU util % | CPU cores | RSS GiB | status |")
    out = [head, "|" + "---|" * 17]
    for r in rows:
        f = lambda v: "-" if v is None else v  # noqa: E731
        per = "/".join(str(round(s)) if s else "-" for s in r["sps_per_copy"])
        out.append(f"| {r['name']} | {r['shape']} | {r['num_envs']} | {r['workers'] or '-'} | {'yes' if r['rnn'] else 'no'} | "
                   f"{'yes' if r.get('cudnn_deterministic') else 'no'} | "
                   f"{r['batch_size']} | {r['minibatch_size']} | {r['copies']} | {f(r['sps_total'])} | {per} | "
                   f"{f(r['peak_alloc_gib'])} | {f(r['gpu_mem_used_max_mib'])} | {f(r['gpu_util_mean'])} | "
                   f"{f(r['cpu_cores_mean'])} | {f(r['rss_max_gib'])} | {r['status']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="stage name (row label and run-id prefix)")
    ap.add_argument("--script", choices=["league", "train"], default="league")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shape", nargs="+", default=["stage1"], help="league: stage1 | A | ckpt16 | ckpt32 | ckpt64")
    ap.add_argument("--num-envs", type=int, nargs="+", default=[256])
    ap.add_argument("--batch-size", type=int, nargs="+")
    ap.add_argument("--batch-k", type=int, nargs="+", help="batch = num_envs x 64 x K")
    ap.add_argument("--minibatch-size", type=int, nargs="+", default=[8192])
    ap.add_argument("--rnn", nargs="+", choices=["off", "on"], default=["off"])
    ap.add_argument("--workers", type=int, nargs="+", help="train: Multiprocessing workers (0 = PufferEnv)")
    ap.add_argument("--copies", type=int, nargs="+", default=[1], help="processes stacked on the GPU")
    ap.add_argument("--deck", default="hog26")
    ap.add_argument("--seconds", type=float, default=120.0, help="league: measured training wall time")
    ap.add_argument("--train-epochs", type=int, default=8, help="train: epochs per measurement")
    ap.add_argument("--train-timeout", type=float, default=1800.0)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--sample-s", type=float, default=2.0)
    ap.add_argument("--deterministic", action="store_true",
                    help="keep royale.ini's torch_deterministic = True (deterministic cuDNN; ~45x slower convs)")
    ap.add_argument("--keep-runs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="after --: extra flags for every run (e.g. shaping)")
    a = ap.parse_args()
    if not (a.batch_size or a.batch_k):
        a.batch_k = [1]
    cfgs, skipped = configs(a)
    for s in skipped:
        print(f"[calibrate] skip {s}", file=sys.stderr)
    if a.dry_run:
        for c in cfgs:
            print(json.dumps(c))
        print(json.dumps({"configs": len(cfgs), "skipped": len(skipped)}))
        return
    os.makedirs(a.out, exist_ok=True)
    need = sorted({int(c["shape"][4:]) for c in cfgs if c["shape"].startswith("ckpt")})
    ckpts = {f"ckpt{n}": prepare_ckpts(a.out, n, a.device) for n in need}
    done = sum(1 for _ in open(os.path.join(a.out, "results.jsonl"))) if os.path.isfile(
        os.path.join(a.out, "results.jsonl")) else 0
    rows = []
    for i, c in enumerate(cfgs):
        r = measure(c, a, done + i, ckpts)
        rows.append(r)
        with open(os.path.join(a.out, "results.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        print("[calibrate] " + json.dumps({k: v for k, v in r.items() if k != "log_tail"}), flush=True)
    print(table(rows), flush=True)


if __name__ == "__main__":
    main()
