"""pufferroyale.fast_league -- opt-in throughput paths for league training (DECISIONS D90; measured in
research/REPORT.md on branch sps-research). Off by default: scripts/league_train.py builds a
FastLeagueVecEnv only when --env-threads > 1, --fast-opponents or --cuda-graphs is given; LeagueVecEnv
itself is unchanged apart from two hook methods (_vec_step, _update_card_stats) whose base versions run
exactly the statements they replaced.

  env_threads K   every step, binding.vec_step_range steps ~4K slices of the matches from K Python threads
                  (the GIL is released in C; c_step touches only its own env, and nothing in the engine's
                  step path uses global state). Env results are bit-identical to vec_step.
  gpu_obs         the env's observation buffer is page-locked; after every step ONE asynchronous
                  host-to-device copy moves all 2 x num_envs rows, and the learner / opponent rows are gathered
                  on the device. Observations are bit-identical; the learner's reach PuffeRL as a device tensor.
                  The per-card counters read only the columns they need from the host buffer
                  (card_stats_update_rows, the same arithmetic as CardStats.update).
  gpu_sample      policy opponents' forwards are queued without host syncs and ALL their actions are sampled
                  at once on the device: torch.multinomial(softmax(logits)), the sampler of the CPU path, with
                  a device generator seeded from the wrapper's streams (and saved in rng_state()). Same
                  distribution, a different random stream than the CPU generator.
  vmap            frozen non-recurrent checkpoint opponents share ONE vmapped forward
                  (torch.func.stack_module_state + functional_call), each model's rows padded to a power-of-two
                  bucket. Same model and math, different kernels: logits equal up to float rounding.
  cuda_graphs     no-grad forwards replayed from CUDA graphs captured once per input shape: the learner's
                  forward_eval (install_learner_graph; the "self" opponent uses the same graphed method, its
                  rows padded to a multiple of 32) and the vmapped bank. The replayed kernels are the eager ones for that shape; train() (grad
                  enabled) stays eager, and in-place optimizer updates keep the captured weight addresses.
gpu_obs, gpu_sample, vmap and cuda_graphs need a CUDA device; gpu_obs, vmap and cuda_graphs imply
gpu_sample (so "self" rows are padded to a multiple of 32 and few shapes are captured).
The learner's graph wrapper is a bound method that replays ONLY for the policy object it was installed on:
a copy.deepcopy of the policy (MMDPuffeRL's reference "magnet") re-binds it to the copy, which then runs
the eager forward with its own weights (review finding, 2026-10-04: a plain closure made the magnet
replay the learner's graph, so the MMD KL was 0).
"""
from __future__ import annotations

import collections
import copy
import types
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .league import LeagueVecEnv, greedy_actions, is_recurrent_policy, parse_spec

class GraphCache:
    """CUDA-graph replays of a no-grad forward fn(x) -> tuple of tensors, one graph per input shape. The
    graphs of one cache share the cache's OWN memory pool (outputs are cloned right after each replay, so a
    replay can never overwrite live data of another graph); the pool lives as long as the cache's graphs.
    A process-wide pool is NOT safe: once every graph using it has been freed (e.g. a rebuilt vmap bank),
    PyTorch retires the pool and the next capture into it fails an allocator assert (use_count > 0, found
    by hpc/tests on 2026-10-04). Beyond max_graphs shapes, fn runs eagerly."""

    def __init__(self, fn, max_graphs: int = 48):
        import torch
        self.pool = torch.cuda.graph_pool_handle()
        self.fn, self.graphs, self.max_graphs, self.captures = fn, {}, int(max_graphs), 0

    def __call__(self, x):
        import torch
        key = (tuple(x.shape), x.dtype, torch.is_autocast_enabled())
        if key in self.graphs and self.graphs[key] is None:      # capture failed once: eager for this shape
            return self.fn(x)
        g = self.graphs.get(key)
        if g is None:
            if len(self.graphs) >= self.max_graphs:
                return self.fn(x)
            try:
                g = self._capture(x, key)
            except Exception as e:  # noqa: BLE001 -- never end a run over an optimisation
                print(f"[fast_league] CUDA graph capture failed for {key[0]} ({e!r}); eager forward", flush=True)
                self.graphs[key] = None
                return self.fn(x)
        g["x"].copy_(x)
        g["g"].replay()
        return tuple(o.clone() for o in g["out"])

    def _capture(self, x, key):
        import torch
        sx = x.detach().clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):                     # warm-up on a side stream (cuDNN algorithm choice)
                self.fn(sx)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        # thread_local: other threads' CUDA calls (PuffeRL's utilization monitor polls cudaMemGetInfo) do not
        # invalidate the capture; the env-stepping threads make no CUDA calls
        with torch.no_grad(), torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
            out = self.fn(sx)
        entry = {"x": sx, "g": g, "out": tuple(out)}
        self.graphs[key] = entry
        self.captures += 1
        return entry


_LEARNER_GRAPHS = weakref.WeakKeyDictionary()     # policy object -> its GraphCache (never deep-copied)


def learner_graph(policy):
    """The GraphCache installed on exactly this policy object, or None (copies have none)."""
    try:
        return _LEARNER_GRAPHS.get(policy)
    except TypeError:
        return None


def install_learner_graph(policy) -> bool:
    """Replay the learner's no-grad forward_eval from CUDA graphs (PuffeRL.evaluate calls it under
    torch.no_grad()). Grad-enabled or non-CUDA calls, and calls on any OTHER object (a deepcopy of the
    policy, e.g. the MMD reference), run the class's forward_eval. Feed-forward policies only; returns
    whether it was installed."""
    import torch
    if is_recurrent_policy(policy) or learner_graph(policy) is not None:
        return False
    cls_fe = type(policy).forward_eval
    gc = GraphCache(lambda x: cls_fe(policy, x, {}))
    target = weakref.ref(policy)

    def forward_eval(self_, x, state=None):
        if self_ is not target() or torch.is_grad_enabled() or not x.is_cuda:
            return cls_fe(self_, x, state)
        return gc(x)
    policy.forward_eval = types.MethodType(forward_eval, policy)
    _LEARNER_GRAPHS[policy] = gc
    return True


def uninstall_learner_graph(policy) -> bool:
    """Undo install_learner_graph: forward_eval is the class's again (e.g. before a best response's batch-1
    evaluation matches, which then run the same plain forward as without --cuda-graphs). Returns whether a
    graph was installed."""
    if _LEARNER_GRAPHS.pop(policy, None) is None:
        return False
    policy.__dict__.pop("forward_eval", None)
    return True


def card_stats_update_rows(cs, obs, rows, actions) -> None:
    """CardStats.update(obs[rows], actions) without copying the rows: the same arithmetic, reading only the
    hand / affordable / lockout / mask columns of `obs` (the env's full host buffer)."""
    from . import royale as R
    rows = np.asarray(rows, dtype=np.int64)
    hand = obs[rows[:, None], cs._hand].astype(np.int64)
    ok = (obs[rows[:, None], cs._afford] > 0.5) & (obs[rows, cs._lockout][:, None] < 0.5) & (hand > 0)
    cs.available += np.bincount(hand[ok], minlength=len(cs.available))
    a = np.asarray(actions).reshape(-1).astype(np.int64)
    r = np.flatnonzero((a >= 1) & (a < cs.n_actions))
    if r.size:
        slot, block = np.divmod(a[r] - 1, cs.n_blocks)
        if cs.grid == 1:
            legal = obs[rows[r], R.MASK_OFFSET + a[r]] > 0.5
        else:
            cells = R.MASK_OFFSET + 1 + slot[:, None] * R.N_TILES + cs._cells[block]
            legal = (obs[rows[r][:, None], cells] > 0.5).any(-1)
        card = hand[r, slot]
        cs.played += np.bincount(card[legal & (card > 0)], minlength=len(cs.played))
    cs.decisions += rows.shape[0]


def _bucket(m: int) -> int:
    return max(4, 1 << (int(m) - 1).bit_length())


class FastLeagueVecEnv(LeagueVecEnv):
    """LeagueVecEnv with the opt-in throughput paths of this module (see the module docstring)."""

    def __init__(self, pool, num_envs=1, seed=0, device="cpu", opponent_greedy=False, results_maxlen=100_000,
                 env_threads: int = 0, gpu_obs: bool = False, gpu_sample: bool = False, vmap: bool = False,
                 cuda_graphs: bool = False, **royale_kwargs):
        import torch
        from . import royale as R
        n = int(num_envs)
        dev = torch.device(device)
        self.gpu_obs, self.vmap, self.cuda_graphs = bool(gpu_obs), bool(vmap), bool(cuda_graphs)
        self.gpu_sample = bool(gpu_sample or gpu_obs or vmap or cuda_graphs)
        if (self.gpu_sample or self.cuda_graphs) and dev.type != "cuda":
            raise ValueError("gpu_obs / gpu_sample / vmap / cuda_graphs need a CUDA device "
                             f"(got {device!r}); env_threads works on any device")
        self._pinned = None
        if self.gpu_obs:
            if "buf" in royale_kwargs:
                raise ValueError("gpu_obs allocates the env buffers itself: do not pass buf")
            t = torch.zeros((2 * n, R.OBS_SIZE), dtype=torch.float32).pin_memory()
            royale_kwargs["buf"] = {"observations": t.numpy(), "rewards": np.zeros(2 * n, np.float32),
                                    "terminals": np.zeros(2 * n, bool), "truncations": np.zeros(2 * n, bool),
                                    "masks": np.ones(2 * n, bool), "actions": np.zeros(2 * n, np.int32)}
            self._pinned = t
        self._tpool = None                            # close() may run if the base constructor fails
        self._obs_dev = None
        self._h2d_done = None                         # CUDA event of the last observation copy (gpu_obs)
        self._vbank = None
        self._vbank_failed = None
        super().__init__(pool, num_envs=n, seed=seed, device=device, opponent_greedy=opponent_greedy,
                         results_maxlen=results_maxlen, **royale_kwargs)
        try:
            self.royale_kwargs.pop("buf", None)
            self.env_threads = int(env_threads)
            if self.env_threads > 1:
                if not hasattr(self._binding, "vec_step_range"):
                    raise RuntimeError("env_threads > 1 needs binding.vec_step_range: rebuild the C extension")
                self._tpool = ThreadPoolExecutor(self.env_threads, thread_name_prefix="pr_env")
                edges = np.linspace(0, n, min(n, 4 * self.env_threads) + 1).astype(int)
                self._slices = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------ RNG streams (+ the device generator)
    def _reseed(self, reset_seed):
        super()._reseed(reset_seed)
        if getattr(self, "gpu_sample", False):
            torch = self._torch
            self._gen_dev = torch.Generator(device=self.device)
            self._gen_dev.manual_seed((int(self._gen.initial_seed()) ^ 0x2545F4914F6CDD1D) & ((1 << 63) - 1))

    def rng_state(self) -> dict:
        out = super().rng_state()
        if self.gpu_sample:
            out["opponent_generator_device"] = self._gen_dev.get_state().tolist()
        return out

    def load_rng_state(self, state: dict) -> None:
        super().load_rng_state(state)
        if self.gpu_sample and "opponent_generator_device" in state:
            self._gen_dev.set_state(self._torch.tensor(state["opponent_generator_device"], dtype=self._torch.uint8))

    # ------------------------------------------------------------ hooks of LeagueVecEnv.send
    def _vec_step(self) -> None:
        if self._h2d_done is not None:   # the C step overwrites the pinned buffer: the last copy must be done
            self._h2d_done.synchronize()
            self._h2d_done = None
        if self._tpool is None:
            self._binding.vec_step(self.env.c_envs)
            return
        c, f = self.env.c_envs, self._binding.vec_step_range
        list(self._tpool.map(lambda s: f(c, s[0], s[1]), self._slices))   # list(): re-raise any error

    def _update_card_stats(self, learner_rows, actions) -> None:
        if self._pinned is None:
            super()._update_card_stats(learner_rows, actions)
        else:
            card_stats_update_rows(self.card_stats, self.env.observations, learner_rows, actions)

    def _gather_obs(self) -> None:
        if self._pinned is None:
            super()._gather_obs()
            return
        torch = self._torch
        if self._obs_dev is None:
            self._obs_dev = torch.empty(tuple(self._pinned.shape), dtype=torch.float32, device=self.device)
        rows = self._to_dev(self._learner_rows().astype(np.int64))
        self._obs_dev.copy_(self._pinned, non_blocking=True)
        self._h2d_done = torch.cuda.Event()
        self._h2d_done.record()
        self.observations = self._obs_dev.index_select(0, rows)

    # ------------------------------------------------------------ helpers
    def _to_dev(self, arr):
        t = self._torch.from_numpy(np.ascontiguousarray(arr))
        return t.pin_memory().to(self.device, non_blocking=True) if self.device.type == "cuda" else t

    def _opp_x(self, rows, dev):
        if self._obs_dev is not None:
            return self._obs_dev.index_select(0, self._to_dev(np.asarray(rows, dtype=np.int64)))
        return self._torch.as_tensor(self.env.observations[rows]).to(dev)

    # ------------------------------------------------------------ vmap bank of frozen checkpoints
    def _bank(self):
        specs = sorted(s for s, (_, rec) in self._ckpt_cache.items() if not rec)
        key = tuple(specs)
        if self._vbank is not None and self._vbank["key"] == key:
            return self._vbank
        if self._vbank_failed == key or not specs:
            return None
        from torch.func import functional_call, stack_module_state, vmap
        models = [self._ckpt_cache[s][0] for s in specs]
        self._vbank = None
        try:
            params, _ = stack_module_state(models)
            buffers = dict(models[0].named_buffers())
            for m in models[1:]:
                for k, v in m.named_buffers():
                    if v.shape != buffers[k].shape or not bool((v == buffers[k]).all()):
                        raise ValueError(f"buffer {k} differs between checkpoints")
            base_m = copy.deepcopy(models[0]).to("meta")

            def f(p, b, x):
                return functional_call(base_m, (p, b), (x, {}))
            fv = vmap(f, in_dims=(0, None, 0))
            bank = {"key": key, "pos": {s: i for i, s in enumerate(specs)}, "K": len(specs),
                    "fwd": lambda X: fv(params, buffers, X), "params": params}
            if self.cuda_graphs:
                bank["fwd"] = GraphCache(bank["fwd"], max_graphs=8)
            self._vbank = bank
        except Exception as e:  # noqa: BLE001 -- e.g. a heterogeneous pool: per-checkpoint forwards
            print(f"[fast_league] vmap bank unavailable ({e!r}); per-checkpoint forwards", flush=True)
            self._vbank_failed = key
        return self._vbank

    def _vmap_logits(self, vm, opp_rows):
        bank = self._bank()
        if bank is None:
            return None
        if any(spec not in bank["pos"] for spec, _ in vm):
            return None
        K = bank["K"]
        M = _bucket(max(len(ix) for _, ix in vm))
        flat = np.full(K * M, opp_rows[vm[0][1][0]], dtype=np.int64)     # padding: any real row
        sel = []
        for spec, ix in vm:
            k = bank["pos"][spec]
            flat[k * M:k * M + len(ix)] = opp_rows[ix]
            sel.append(np.arange(k * M, k * M + len(ix), dtype=np.int64))
        X = self._opp_x(flat, self.device).view(K, M, -1)
        try:
            logits, _ = bank["fwd"](X)
        except Exception as e:  # noqa: BLE001 -- e.g. OOM or a missing batching rule: per-checkpoint forwards
            print(f"[fast_league] vmap bank forward failed ({e!r}); per-checkpoint forwards", flush=True)
            self._vbank, self._vbank_failed = None, bank["key"]
            return None
        logits = logits.reshape(K * M, -1).index_select(0, self._to_dev(np.concatenate(sel)))
        return [(np.concatenate([ix for _, ix in vm]), logits)]

    # ------------------------------------------------------------ opponents
    def _opponent_actions(self) -> np.ndarray:
        if not self.gpu_sample:
            return super()._opponent_actions()
        torch = self._torch
        n = self.num_envs
        out = np.zeros(n, dtype=np.int32)
        opp_rows = 2 * np.arange(n) + (1 - self.seats)
        groups = collections.OrderedDict()
        for i, spec in enumerate(self.opponents):
            groups.setdefault(spec, []).append(i)
        pend, vm = [], []
        with torch.no_grad():
            for spec, idx in groups.items():
                kind, _ = parse_spec(spec)
                if kind == "bot":
                    for i in idx:
                        out[i] = self._binding.bot_act_env(self._bots[i], self._handles[i], int(1 - self.seats[i]))
                    continue
                policy, recurrent, dev = self._opponent_policy(spec)
                ix = np.asarray(idx)
                if self.vmap and kind == "ckpt" and not recurrent:
                    vm.append((spec, ix))
                    continue
                rows = opp_rows[ix]
                m = len(rows)
                if recurrent:
                    x = self._opp_x(rows, dev)
                    if spec not in self._lstm:
                        hs = int(policy.hidden_size)
                        self._lstm[spec] = (torch.zeros(n, hs, device=dev), torch.zeros(n, hs, device=dev))
                    h, c = self._lstm[spec]
                    ti = torch.as_tensor(ix, device=dev)
                    state = {"lstm_h": h[ti], "lstm_c": c[ti]}
                    logits, _ = policy.forward_eval(x, state)
                    h[ti] = state["lstm_h"]
                    c[ti] = state["lstm_c"]
                elif kind == "self" and self.cuda_graphs and learner_graph(policy) is not None:
                    # the learner's forward_eval replays CUDA graphs (install_learner_graph): bucket the
                    # shape to the next multiple of 32 rows so few graphs are captured
                    if m % 32:
                        rows = np.concatenate([rows, np.full(32 - m % 32, rows[0])])
                    logits, _ = policy.forward_eval(self._opp_x(rows, dev), {})
                else:
                    logits, _ = policy.forward_eval(self._opp_x(rows, dev), {})
                pend.append((ix, logits[:m]))
            if vm:
                got = self._vmap_logits(vm, opp_rows)
                if got is None:                    # fallback: eager per checkpoint
                    for spec, ix in vm:
                        policy, _, dev = self._opponent_policy(spec)
                        logits, _ = policy.forward_eval(self._opp_x(opp_rows[ix], dev), {})
                        pend.append((ix, logits))
                else:
                    pend.extend(got)
            if pend:
                ix_all = np.concatenate([p[0] for p in pend])
                logits = torch.cat([p[1].float() for p in pend]) if len(pend) > 1 else pend[0][1].float()
                if self.opponent_greedy:
                    out[ix_all] = greedy_actions(logits)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    act = torch.multinomial(probs, 1, generator=self._gen_dev).squeeze(-1)
                    out[ix_all] = act.to(torch.int32).cpu().numpy()
        return out

    def close(self) -> None:
        if self._tpool is not None:
            self._tpool.shutdown(wait=True)
            self._tpool = None
        self._vbank = None
        self._obs_dev = None
        self._h2d_done = None
        if getattr(self, "env", None) is not None:
            super().close()
        self._pinned = None


__all__ = ["FastLeagueVecEnv", "GraphCache", "install_learner_graph", "learner_graph", "card_stats_update_rows"]
