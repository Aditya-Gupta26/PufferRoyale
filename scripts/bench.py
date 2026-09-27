#!/usr/bin/env python3
"""Environment throughput (SPEC §11: >= 2,000 env steps/s at frame_skip 10, 2 agents, obs).

Random legal actions (from each row's mask) with probability --play-prob per decision.
Reports env steps/s (one step = frame_skip ticks for every native env) and agent steps/s,
for 1 native env and for N native envs in one process; optionally the PufferLib
Multiprocessing backend.

    python scripts/bench.py                    # 1 and 16 envs, 4 s each
    python scripts/bench.py --envs 1 64 --seconds 10
    python scripts/bench.py --multiprocessing --workers 4
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def bench_native(num_envs, num_agents, frame_skip, seconds, play_prob, seed=0):
    import pufferroyale
    from pufferroyale import royale as R

    env = pufferroyale.Royale(num_envs=num_envs, num_agents=num_agents, frame_skip=frame_skip,
                              opponent="heuristic", seed=seed, log_interval=10 ** 9)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rows = env.num_agents
    acts = np.zeros(rows, dtype=np.int32)
    steps, t_env, episodes = 0, 0.0, 0
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        acts[:] = 0
        pick = rng.random(rows) < play_prob
        for r in np.nonzero(pick)[0]:
            leg = np.flatnonzero(obs[r, R.MASK_OFFSET + 1:R.MASK_OFFSET + R.MASK_SIZE] > 0.5)
            if len(leg):
                acts[r] = int(leg[rng.integers(len(leg))]) + 1
        t0 = time.perf_counter()
        obs, rew, term, trunc, info = env.step(acts)
        t_env += time.perf_counter() - t0
        steps += 1
        episodes += int(term[::num_agents].sum())
    env.close()
    return dict(num_envs=num_envs, env_sps=steps * num_envs / t_env, agent_sps=steps * rows / t_env,
                episodes=episodes)


def bench_multiprocessing(workers, envs_per_worker, num_agents, frame_skip, seconds, seed=0):
    import pufferlib.vector
    import pufferroyale

    vec = pufferlib.vector.make(pufferroyale.Royale, backend=pufferlib.vector.Multiprocessing,
                                num_envs=workers, num_workers=workers, batch_size=workers,
                                env_kwargs=dict(num_envs=envs_per_worker, num_agents=num_agents,
                                                frame_skip=frame_skip, log_interval=10 ** 9))
    vec.reset(seed=seed)
    acts = np.zeros(vec.num_agents, dtype=np.int32)
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        vec.step(acts)
        steps += 1
    dt = time.perf_counter() - t0
    vec.close()
    n_envs = workers * envs_per_worker
    return dict(num_envs=n_envs, env_sps=steps * n_envs / dt, agent_sps=steps * vec.num_agents / dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--num-agents", type=int, default=2)
    ap.add_argument("--frame-skip", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--play-prob", type=float, default=0.1)
    ap.add_argument("--multiprocessing", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs-per-worker", type=int, default=16)
    args = ap.parse_args()
    print(f"frame_skip={args.frame_skip} num_agents={args.num_agents} (obs + mask included)")
    for n in args.envs:
        r = bench_native(n, args.num_agents, args.frame_skip, args.seconds, args.play_prob)
        print(f"native  {n:4d} env(s): {r['env_sps']:9.0f} env steps/s  {r['agent_sps']:9.0f} agent steps/s"
              f"  ({r['episodes']} episodes finished)")
    if args.multiprocessing:
        r = bench_multiprocessing(args.workers, args.envs_per_worker, args.num_agents, args.frame_skip,
                                  args.seconds)
        print(f"multiprocessing {args.workers}x{args.envs_per_worker}: {r['env_sps']:9.0f} env steps/s"
              f"  {r['agent_sps']:9.0f} agent steps/s (no-op actions)")


if __name__ == "__main__":
    main()
