"""GPU heartbeat for every PufferRoyale GPU job (user's instruction 2026-10-03: keep the GPU at a
minimum of 75% utilization at all times; DECISIONS D79).

Torch reclaims GPUs whose jobs stay at low utilization (the reason the user's other project runs a
heartbeat). Our league runs step their environments on the CPU and can sit at a few percent of GPU
utilization, so this process tops the device up to the floor:

  * every --period seconds it runs float32 matmuls ("burst") for a fraction f of the period and stays
    idle for the rest;
  * a sampler thread reads the DEVICE utilization from NVML (training + heartbeat together) every
    --sample-s seconds and keeps its mean over the last --avg-s seconds, u. NVML reports utilization
    over a short recent window, so single readings jump between ~0 and ~100 within a second (seen on
    an H100: 99, 99, ..., 0); only the time average is meaningful;
  * after each period f is adjusted by integral control towards --target + --margin (default
    75 + 5 = 80%):
        f <- clip(f + gain * (setpoint - u) / 100, 0, --max-frac)
    so it adds only what the training does not already use, and backs off to 0 when training alone
    is above the setpoint;
  * the matmuls are small (2048 x 2048, well under a millisecond on A100/H100/H200), so training's own
    kernels wait at most about one matmul when the GPU time-slices between the two processes;
  * they are queued in batches of about --batch-ms of GPU time (measured at start) and double-buffered:
    the next batch is queued before waiting (on a blocking CUDA event, so no core spins) for the
    previous one. When the job's cores are all busy (a 6-run pack starting up) the process may wait a
    while for the CPU; the queued batch keeps the GPU busy meanwhile. The first version synchronized
    after every 4 matmuls (~1.4 ms) and held only 34% (pack I, job 19152999), 64% (pack H, 19152998)
    and ~1% for 15 min (pack J, 19153517) at the start of a pack while it reported itself busy 56-99%.
    Measured with 24 busy loops on the job's 16 cores (jobs 19154132, 19155336, H100): this version at
    nice 0 holds 80.0% (0 of 30 one-second samples below 75% in every 30 s window); the old one at
    nice 0 holds 54-73%; at nice 19 neither even started within 130 s. Hence nice 0 (D87).

Started by hpc/lib.sh (pr_heartbeat_start) at nice 0 (HEARTBEAT_NICE) on the job's own GPU and stopped by the job's
EXIT trap; HEARTBEAT=0 disables it. ~50 MB of GPU memory. Logs a status line every --status-every s
(mean device utilization and the heartbeat's share of the time).
"""
import argparse
import collections
import signal
import sys
import threading
import time

import torch

STOP = False


def _stop(signum, frame):
    global STOP
    STOP = True


def utilization():
    """Device-wide GPU utilization in percent from NVML, or None when unreadable."""
    try:
        return float(torch.cuda.utilization())
    except Exception:  # noqa: BLE001 -- no pynvml / no permission: handled by the caller
        return None


class Sampler(threading.Thread):
    """NVML device utilization every `every` seconds; mean over the last `window` seconds."""

    def __init__(self, every, window):
        super().__init__(daemon=True)
        self.every, self.window = every, window
        self.samples = collections.deque()
        self.lock = threading.Lock()
        self.readable = True

    def run(self):
        while not STOP:
            u = utilization()
            now = time.monotonic()
            with self.lock:
                if u is None:
                    self.readable = False
                else:
                    self.samples.append((now, u))
                while self.samples and self.samples[0][0] < now - self.window:
                    self.samples.popleft()
            time.sleep(self.every)

    def mean(self):
        with self.lock:
            return sum(u for _, u in self.samples) / len(self.samples) if self.samples else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=75.0, help="utilization floor in percent (default 75)")
    ap.add_argument("--margin", type=float, default=5.0, help="aim this many points above the floor (default 5)")
    ap.add_argument("--period", type=float, default=1.0, help="seconds per burst + idle cycle (default 1)")
    ap.add_argument("--max-frac", type=float, default=0.95, help="largest share of each period spent bursting")
    ap.add_argument("--gain", type=float, default=0.1, help="integral gain of the duty-cycle controller")
    ap.add_argument("--sample-s", type=float, default=0.1, help="seconds between NVML readings (default 0.1)")
    ap.add_argument("--avg-s", type=float, default=5.0, help="seconds averaged into the controlled utilization")
    ap.add_argument("--size", type=int, default=2048, help="square matmul size (default 2048)")
    ap.add_argument("--batch-ms", type=float, default=20.0,
                    help="GPU time queued per batch, in ms (default 20; two batches are kept in flight)")
    ap.add_argument("--status-every", type=float, default=300.0, help="seconds between status lines")
    a = ap.parse_args()
    if not 0 < a.target <= 100 or a.period <= 0 or not 0 < a.max_frac <= 1:
        ap.error("--target must be in (0, 100], --period > 0, --max-frac in (0, 1]")
    if not torch.cuda.is_available():
        print("[gpu_heartbeat] no CUDA device: exiting", file=sys.stderr, flush=True)
        return
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    torch.set_num_threads(1)
    dev = torch.device("cuda")
    x = torch.randn(a.size, a.size, device=dev)
    y = torch.randn(a.size, a.size, device=dev)
    z = torch.empty(a.size, a.size, device=dev)
    for _ in range(3):                                   # warm-up (cuBLAS handle, kernel choice)
        torch.mm(x, y, out=z)
    torch.cuda.synchronize()
    t = time.monotonic()
    for _ in range(20):
        torch.mm(x, y, out=z)
    torch.cuda.synchronize()
    t_mm = max(1e-6, (time.monotonic() - t) / 20)
    n_batch = max(1, round(a.batch_ms / 1000.0 / t_mm))
    setpoint = min(100.0, a.target + a.margin)
    sampler = Sampler(a.sample_s, a.avg_s)
    sampler.start()
    frac = min(a.max_frac, setpoint / 100.0)            # start high; the controller backs off
    t_status, busy_total, u_sum, n_u, below = time.monotonic(), 0.0, 0.0, 0, 0
    print(f"[gpu_heartbeat] started on {torch.cuda.get_device_name(0)}: floor {a.target:.0f}%, "
          f"setpoint {setpoint:.0f}%, period {a.period:g}s, matmul {a.size} ({1000 * t_mm:.2f} ms alone), "
          f"{n_batch} per batch", file=sys.stderr, flush=True)
    while not STOP:
        t0 = time.monotonic()
        busy = frac * a.period
        pending = None                                   # the batch queued before the current one
        while time.monotonic() - t0 < busy and not STOP:
            for _ in range(n_batch):
                torch.mm(x, y, out=z)
            ev = torch.cuda.Event(blocking=True)
            ev.record()
            if pending is not None:
                pending.synchronize()
            pending = ev
        if pending is not None:
            pending.synchronize()
        busy_total += time.monotonic() - t0
        rest = a.period - (time.monotonic() - t0)
        if rest > 0:
            time.sleep(rest)
        u = sampler.mean()
        if u is None and not sampler.readable:
            frac = min(a.max_frac, setpoint / 100.0)       # unreadable: a fixed duty cycle at the setpoint
        elif u is not None:
            u_sum, n_u = u_sum + u, n_u + 1
            below += u < a.target
            frac = min(a.max_frac, max(0.0, frac + a.gain * (setpoint - u) / 100.0))
        if time.monotonic() - t_status >= a.status_every:
            el = time.monotonic() - t_status
            print(f"[gpu_heartbeat] last {el:.0f}s: device utilization {(u_sum / n_u) if n_u else float('nan'):.1f}% "
                  f"(below {a.target:.0f}% in {below}/{n_u} samples), heartbeat busy {100 * busy_total / el:.1f}% "
                  f"of the time", file=sys.stderr, flush=True)
            t_status, busy_total, u_sum, n_u, below = time.monotonic(), 0.0, 0.0, 0, 0
    print("[gpu_heartbeat] stopped", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
