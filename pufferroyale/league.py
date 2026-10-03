"""pufferroyale.league -- league training: an opponent pool with PFSP and a learner-only vecenv.

SPEC §15.1 / §15.2 / §15.7. The design follows AlphaStar's league (Vinyals et al. 2019): the
learner trains against a mixture of itself, fixed *anchor* opponents (scripted bots or frozen
checkpoints that are never evicted) and *snapshots* of its own past weights, the latter chosen by
prioritised fictitious self-play (PFSP) so that opponents the learner still loses to are
sampled more often.

    OpponentPool   opponent specs, win/draw/loss bookkeeping, the PFSP sampling distribution
                   (its own seeded RNG; state_dict() is JSON-serialisable and round-trips exactly)
    LeagueVecEnv   `num_envs` self-play Royale matches exposed to PuffeRL as ONE learner row per
                   match; the other seat of every match is played by an opponent drawn from the
                   pool, so opponent transitions never enter the learner's PPO buffer
    greedy_actions the card-first greedy rule (SPEC §19.11): the single greedy action choice of
                   every tool (select_actions(logits, greedy=True, ...))

Opponent specs (strings):
    "self"                              the current learner policy (set_policy), no grad
    "bot:noop" | "bot:random" | "bot:heuristic"     the C scripted bots (SPEC §8)
    "ckpt:<path>"                       a frozen checkpoint: torch.save(policy.state_dict()) of
                                        pufferroyale.torch.Policy or Recurrent(Policy)
                                        (recurrent iff it has lstm.* keys, §15.7.7)

torch / pufferlib are imported lazily, so the pool can be used without them.
"""
from __future__ import annotations

import collections
import os
import types
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

SELF = "self"
BOT_PREFIX = "bot:"
CKPT_PREFIX = "ckpt:"
BOT_KINDS = {"noop": 0, "random": 1, "heuristic": 2}  # PR_BOT_* (pufferroyale.game.BOT_KINDS)
PFSP_MODES = ("hard", "variance", "uniform")


# ======================================================================================
# opponent specs
# ======================================================================================
def parse_spec(spec: str) -> Tuple[str, str]:
    """('self', '') | ('bot', kind) | ('ckpt', path); ValueError for anything else."""
    if not isinstance(spec, str):
        raise ValueError(f"an opponent spec is a string, got {spec!r}")
    if spec == SELF:
        return "self", ""
    if spec.startswith(BOT_PREFIX):
        kind = spec[len(BOT_PREFIX):]
        if kind not in BOT_KINDS:
            raise ValueError(f"unknown bot {spec!r}; use one of {['bot:' + k for k in BOT_KINDS]}")
        return "bot", kind
    if spec.startswith(CKPT_PREFIX) and len(spec) > len(CKPT_PREFIX):
        return "ckpt", spec[len(CKPT_PREFIX):]
    raise ValueError(f"unknown opponent spec {spec!r}: use 'self', 'bot:<noop|random|heuristic>' or 'ckpt:<path>'")


def pfsp_weight(mode: str, p: float, eps: float) -> float:
    """PFSP weight of an opponent the learner scores `p` against (SPEC §15.1), floored at eps.
    hard = (1-p)^2 (focus on opponents the learner loses to), variance = p(1-p) (focus on even
    matchups), uniform = 1."""
    if mode == "hard":
        w = (1.0 - p) ** 2
    elif mode == "variance":
        w = p * (1.0 - p)
    elif mode == "uniform":
        w = 1.0
    else:
        raise ValueError(f"pfsp must be one of {PFSP_MODES}, got {mode!r}")
    return max(w, eps)


# ======================================================================================
# OpponentPool (SPEC §15.1, §15.7.1-3)
# ======================================================================================
class OpponentPool:
    """Opponent pool with an absolute mass split (SPEC §15.7.1):

        P("self")           = self_play_frac
        P(anchors)          = anchor_frac, uniform among the anchors
        P(snapshots)        = 1 - self_play_frac - anchor_frac, by PFSP weight w(p)

    Fallbacks: no snapshots -> their mass goes to the anchors (to "self" if there are none);
    no anchors -> their mass goes to the snapshots (to "self" if there are none).

    Bookkeeping is per spec; `p` is the LEARNER's score against that opponent,
    (wins + 0.5 draws) / games, 0.5 when unseen. Sampling uses the pool's own seeded RNG
    (numpy PCG64), so a pool is deterministic given its seed and the call sequence.
    """

    def __init__(self, anchors: Iterable[str] = ("bot:heuristic",), max_snapshots: int = 16,
                 pfsp: str = "hard", pfsp_eps: float = 0.05, self_play_frac: float = 0.2,
                 anchor_frac: float = 0.2, seed: int = 0):
        anchors = tuple(anchors)
        for a in anchors:
            if parse_spec(a)[0] == "self":
                raise ValueError("'self' is not an anchor: use self_play_frac")
        if len(set(anchors)) != len(anchors):
            raise ValueError(f"duplicate anchors: {anchors}")
        if pfsp not in PFSP_MODES:
            raise ValueError(f"pfsp must be one of {PFSP_MODES}, got {pfsp!r}")
        self_play_frac, anchor_frac, pfsp_eps = float(self_play_frac), float(anchor_frac), float(pfsp_eps)
        if not (0.0 <= self_play_frac <= 1.0 and 0.0 <= anchor_frac <= 1.0):
            raise ValueError("self_play_frac and anchor_frac must lie in [0, 1]")
        if self_play_frac + anchor_frac > 1.0 + 1e-12:
            raise ValueError(f"self_play_frac + anchor_frac = {self_play_frac + anchor_frac} > 1 (SPEC §15.7.1)")
        if pfsp_eps < 0.0:
            raise ValueError("pfsp_eps must be >= 0")
        if int(max_snapshots) < 1:
            raise ValueError("max_snapshots must be >= 1")
        self.anchors = anchors
        self.max_snapshots = int(max_snapshots)
        self.pfsp = pfsp
        self.pfsp_eps = pfsp_eps
        self.self_play_frac = self_play_frac
        self.anchor_frac = anchor_frac
        self.seed = int(seed)
        self.snapshots: List[str] = []            # "ckpt:<path>", oldest first (never anchors)
        self._stats: Dict[str, List[int]] = {}    # spec -> [games, wins, draws, losses]
        self._rng = np.random.Generator(np.random.PCG64(self.seed))

    # ---------------------------------------------------------------- membership
    def add_snapshot(self, path: str) -> str:
        """Append "ckpt:<path>" (never touches the filesystem). Beyond max_snapshots the oldest
        snapshot is evicted and its stats are dropped. A path that is already a snapshot or an
        anchor is left as it is (anchors are never evicted). Returns the spec."""
        spec = f"{CKPT_PREFIX}{path}"
        if spec in self.anchors or spec in self.snapshots:
            return spec
        self.snapshots.append(spec)
        self._stats.setdefault(spec, [0, 0, 0, 0])
        while len(self.snapshots) > self.max_snapshots:
            old = self.snapshots.pop(0)
            self._stats.pop(old, None)
        return spec

    def members(self) -> List[str]:
        """Every spec the pool can currently sample (zero-probability ones included)."""
        return [SELF] + list(self.anchors) + list(self.snapshots)

    def __contains__(self, spec: str) -> bool:
        return spec == SELF or spec in self.anchors or spec in self.snapshots

    # ---------------------------------------------------------------- bookkeeping
    def record(self, opponent: str, learner_result) -> None:
        """Record one finished match against `opponent` from the learner's view: +1 win,
        0 draw, -1 loss. Results against a checkpoint that is no longer in the pool (evicted
        while the match was running) are ignored, so evicted stats stay dropped."""
        r = float(learner_result)
        if r not in (1.0, 0.0, -1.0):
            raise ValueError(f"learner_result must be +1, 0 or -1, got {learner_result!r}")
        kind, _ = parse_spec(opponent)
        if kind == "ckpt" and opponent not in self:
            return
        s = self._stats.setdefault(opponent, [0, 0, 0, 0])
        s[0] += 1
        s[1 if r > 0 else (2 if r == 0 else 3)] += 1

    def score(self, spec: str) -> float:
        """The learner's score p against `spec` (0.5 when unseen)."""
        g, w, d, _ = self._stats.get(spec, (0, 0, 0, 0))
        return (w + 0.5 * d) / g if g else 0.5

    def stats(self) -> Dict[str, Dict[str, float]]:
        """{spec: {"games", "wins", "draws", "losses", "p"}} for every spec with bookkeeping
        (current snapshots and every opponent a result was recorded against)."""
        out = {}
        for spec, (g, w, d, l) in self._stats.items():
            out[spec] = {"games": g, "wins": w, "draws": d, "losses": l, "p": self.score(spec)}
        return out

    # ---------------------------------------------------------------- distribution
    def weights(self) -> Dict[str, float]:
        """The current sampling distribution {spec: probability} over the specs with non-zero
        probability (ordered: "self", anchors, snapshots oldest first); sums to 1."""
        spf, af = self.self_play_frac, self.anchor_frac
        m_self, m_anchor, m_snap = spf, af, max(0.0, 1.0 - spf - af)
        if not self.snapshots:                       # §15.7.1 fallbacks
            if self.anchors:
                m_anchor += m_snap
            else:
                m_self += m_snap
            m_snap = 0.0
        if not self.anchors:
            if self.snapshots:
                m_snap += m_anchor
            else:
                m_self += m_anchor
            m_anchor = 0.0
        w: Dict[str, float] = {}
        if m_self > 0:
            w[SELF] = m_self
        if m_anchor > 0:
            for a in self.anchors:
                w[a] = m_anchor / len(self.anchors)
        if m_snap > 0:
            raw = [pfsp_weight(self.pfsp, self.score(s), self.pfsp_eps) for s in self.snapshots]
            total = sum(raw)
            if total <= 0.0:                         # only possible with pfsp_eps = 0
                raw, total = [1.0] * len(raw), float(len(raw))
            for s, r in zip(self.snapshots, raw):
                if r > 0:
                    w[s] = m_snap * r / total
        return w

    def sample(self) -> str:
        """One opponent spec drawn from weights() with the pool's RNG (one uniform per call)."""
        w = self.weights()
        keys = list(w)
        cum = np.cumsum(np.fromiter(w.values(), dtype=np.float64, count=len(keys)))
        u = self._rng.random() * cum[-1]
        i = int(np.searchsorted(cum, u, side="right"))
        return keys[min(i, len(keys) - 1)]

    def pool_size(self) -> int:
        """Number of distinct specs with non-zero sampling probability (SPEC §15.7.5)."""
        return len(self.weights())

    # ---------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        """JSON-serialisable state: snapshots, stats and the RNG state (SPEC §15.7.3). The
        constructor arguments are included for reference; load_state_dict expects a pool built
        with the same arguments."""
        return {
            "version": 1,
            "config": {"anchors": list(self.anchors), "max_snapshots": self.max_snapshots, "pfsp": self.pfsp,
                       "pfsp_eps": self.pfsp_eps, "self_play_frac": self.self_play_frac,
                       "anchor_frac": self.anchor_frac, "seed": self.seed},
            "snapshots": list(self.snapshots),
            "stats": {k: list(v) for k, v in self._stats.items()},
            "rng": _jsonable_rng_state(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        snaps = [str(s) for s in state["snapshots"]]
        for s in snaps:
            if parse_spec(s)[0] != "ckpt":
                raise ValueError(f"snapshot {s!r} is not a ckpt: spec")
        self.snapshots = snaps
        self._stats = {str(k): [int(x) for x in v] for k, v in state["stats"].items()}
        self._rng = np.random.Generator(np.random.PCG64())
        self._rng.bit_generator.state = state["rng"]

    def __repr__(self) -> str:
        w = ", ".join(f"{k}: {v:.3f}" for k, v in self.weights().items())
        return f"OpponentPool({w})"


def _jsonable_rng_state(state: dict) -> dict:
    """numpy bit-generator state as plain JSON types (ints stay exact: JSON has big ints)."""
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    return conv(dict(state))


# ======================================================================================
# checkpoints (SPEC §15.7.7)
# ======================================================================================
def _strip_prefixes(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for pre in ("_orig_mod.", "module."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def is_recurrent_state_dict(sd: dict) -> bool:
    """SPEC §15.7.7: recurrent checkpoints are detected by lstm.* keys."""
    return any(k.startswith("lstm.") for k in sd)


def policy_kwargs_from_state_dict(sd: dict) -> Tuple[dict, Optional[dict]]:
    """Infer Policy(...) (and Recurrent(...)) keywords from the tensors of a state dict, so every
    checkpoint the tools write loads with its own architecture (SPEC §19.6, §19.9.8): the sizes,
    head ("flat" iff actor.* keys), card_stats (card_stat_proj.* keys), pos_channels (conditional
    head) and placement_grid (the action_grid buffer, else the flat head's action count, else 1).
    Returns (policy_kwargs, rnn_kwargs | None); an unknown layout gives ({}, ...) and the strict load
    explains."""
    rec = is_recurrent_state_dict(sd)
    pre = "policy." if rec else ""
    kw = {}
    try:
        kw["hidden_size"] = int(sd[pre + "value.weight"].shape[1])
        kw["cnn_channels"] = int(sd[pre + "cnn.0.weight"].shape[0])
        kw["entity_hidden"] = int(sd[pre + "entity.0.weight"].shape[0])
        kw["scalar_hidden"] = int(sd[pre + "scalars.0.weight"].shape[0])
        kw["card_dim"] = int(sd[pre + "card_embed.weight"].shape[1])
    except KeyError:
        kw = {}                                       # unknown layout: defaults, strict load will explain
    if kw:
        from .torch import GRID_OF_ACTIONS
        kw["card_stats"] = int(pre + "card_stat_proj.weight" in sd)
        grid = None
        if pre + "actor.weight" in sd:
            kw["head"] = "flat"
            grid = GRID_OF_ACTIONS.get(int(sd[pre + "actor.weight"].shape[0]))
        else:
            kw["head"] = "conditional"
            if pre + "pos_out.weight" in sd:
                kw["pos_channels"] = int(sd[pre + "pos_out.weight"].shape[1])
        if pre + "action_grid" in sd:
            grid = int(sd[pre + "action_grid"])
        kw["placement_grid"] = grid if grid is not None else 1
    rnn = None
    if rec:
        rnn = {"input_size": int(sd["lstm.weight_ih_l0"].shape[1]),
               "hidden_size": int(sd["lstm.weight_hh_l0"].shape[1])}
    return kw, rnn


def _obs_stub(n_actions: Optional[int] = None):
    """The only things pufferroyale.torch.Policy / Recurrent read from `env`: the observation space
    and (for the action head) the action space."""
    import gymnasium
    from . import royale as R
    return types.SimpleNamespace(
        single_observation_space=gymnasium.spaces.Box(low=0.0, high=np.inf, shape=(R.OBS_SIZE,), dtype=np.float32),
        single_action_space=gymnasium.spaces.Discrete(int(n_actions or R.N_ACTIONS)))


def json_safe(obj):
    """obj with every non-finite float replaced by its string ('nan', 'inf', '-inf'), so that
    json.dumps(..., allow_nan=False) always produces valid JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, (np.floating,)):
        return json_safe(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def _snap_epoch(p: str) -> int:
    base = os.path.basename(p)
    try:
        return int(base[5:-3])
    except ValueError:
        return -1


def resolve_checkpoint_path(spec: str) -> str:
    """A checkpoint file from 'ckpt:<path>' or a path. A directory resolves to its newest
    model_*.pt (a finished train.py / league run), else learner.pt (a league run in progress),
    else its newest snap_<epoch>.pt. FileNotFoundError when nothing matches."""
    import glob
    path = spec[len(CKPT_PREFIX):] if isinstance(spec, str) and spec.startswith(CKPT_PREFIX) else spec
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        models = sorted(glob.glob(os.path.join(path, "model_*.pt")))
        if models:
            return models[-1]
        if os.path.isfile(os.path.join(path, "learner.pt")):
            return os.path.join(path, "learner.pt")
        snaps = sorted(glob.glob(os.path.join(path, "snap_*.pt")), key=_snap_epoch)
        if snaps:
            return snaps[-1]
        raise FileNotFoundError(f"{path}: no model_*.pt, learner.pt or snap_*.pt in this directory")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: no such checkpoint")
    return path


#: env settings a policy is trained under that change what its actions mean in a match (SPEC §19.4:
#: placement_grid is one of them)
MATCH_ENV_KEYS = ("frame_skip", "deploy_lockout_ticks", "tiebreak", "tower_troop0", "tower_troop1", "placement_grid")


def checkpoint_env_config(spec: str) -> Optional[dict]:
    """The env settings (MATCH_ENV_KEYS) a checkpoint was trained with, from the config.json that
    train.py / league_train.py write next to it (the checkpoint's directory, or <file stem>/ for
    train.py's copied <run>.pt). None when no config.json is found."""
    import json
    try:
        path = resolve_checkpoint_path(spec)
    except FileNotFoundError:
        return None
    for cand in (os.path.join(os.path.dirname(path), "config.json"), os.path.join(os.path.splitext(path)[0], "config.json")):
        if os.path.isfile(cand):
            try:
                with open(cand) as f:
                    env = json.load(f).get("env") or {}
            except (OSError, ValueError):
                return None
            return {k: env[k] for k in MATCH_ENV_KEYS if k in env}
    return None


_LAYOUT_ERROR = ("this checkpoint does not fit the current observation / policy layout (OBS_SIZE and the card "
                 "encoding changed in v0.3, SPEC §16.4, and again in v0.5: the own_deck observation field, SPEC "
                 "§19.3, the policy heads and card stats, SPEC §19.6; checkpoints from before v0.5 must be retrained)")


def load_state_dict_file(path: str) -> dict:
    """The (prefix-stripped) state dict of a checkpoint file / 'ckpt:<path>' / run directory."""
    import torch
    path = resolve_checkpoint_path(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: not a policy state_dict")
    return _strip_prefixes(sd)


def build_policy(sd: dict):
    """An (untrained) Policy / Recurrent(Policy) with the architecture of state dict `sd`."""
    from .royale import n_actions
    from .torch import Policy, Recurrent
    kw, rnn = policy_kwargs_from_state_dict(sd)
    stub = _obs_stub(n_actions(kw.get("placement_grid", 1)))
    policy = Policy(stub, **kw)
    if rnn is not None:
        policy = Recurrent(stub, policy, **rnn)
    return policy


def load_policy(path: str, device="cpu"):
    """Load a PuffeRL checkpoint (torch.save(policy.state_dict()) of pufferroyale.torch.Policy or
    Recurrent(Policy), either head, any placement grid) as an eval-mode policy without grad.
    Returns (policy, recurrent). `path` may be 'ckpt:<path>' or a run directory (see
    resolve_checkpoint_path). Checkpoints of an older layout raise a clear ValueError."""
    path = resolve_checkpoint_path(path)
    sd = load_state_dict_file(path)
    policy = build_policy(sd)
    try:
        policy.load_state_dict(sd)
    except RuntimeError as e:
        msg = str(e)
        if "size mismatch" in msg or "Missing key" in msg or "Unexpected key" in msg:
            raise ValueError(f"{path}: {_LAYOUT_ERROR}: {msg.splitlines()[-1].strip()}") from e
        raise
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy, is_recurrent_state_dict(sd)


def policy_architecture(policy) -> Tuple[dict, Optional[dict]]:
    """(policy_kwargs, rnn_kwargs | None) of a live Policy / Recurrent, in the form of
    policy_kwargs_from_state_dict."""
    sd = _strip_prefixes(policy.state_dict())
    return policy_kwargs_from_state_dict(sd)


def architecture_diff(policy, sd: dict) -> List[str]:
    """The differences between the architecture of a live policy and that of state dict `sd` (sizes,
    head, card stats, pos channels, grid; recurrence and LSTM sizes), one readable item each; [] when
    they agree."""
    want, got = policy_architecture(policy), policy_kwargs_from_state_dict(_strip_prefixes(sd))
    diff = [f"{k}: checkpoint {got[0].get(k)!r} vs this run {want[0].get(k)!r}"
            for k in sorted(set(want[0]) | set(got[0])) if want[0].get(k) != got[0].get(k)]
    if want[1] != got[1]:
        diff.append(f"recurrent: checkpoint {got[1]!r} vs this run {want[1]!r}")
    return diff


def load_weights(policy, path: str) -> str:
    """Load a checkpoint's weights into `policy` in place (--init-from: weights only). The
    checkpoint's architecture (sizes, head, card stats, grid, recurrent or not) must equal the
    policy's; otherwise ValueError naming the differences. Returns the resolved path."""
    path = resolve_checkpoint_path(path)
    sd = load_state_dict_file(path)
    diff = architecture_diff(policy, sd)
    if diff:
        raise ValueError(f"{path}: architecture mismatch ({'; '.join(diff)})")
    dev = next(iter(policy.parameters())).device
    try:
        policy.load_state_dict({k: v.to(dev) for k, v in sd.items()})
    except RuntimeError as e:
        raise ValueError(f"{path}: {_LAYOUT_ERROR}: {str(e).splitlines()[-1].strip()}") from e
    return path


def is_recurrent_policy(policy) -> bool:
    import pufferlib.models
    return isinstance(policy, pufferlib.models.LSTMWrapper) or hasattr(policy, "lstm")


def policy_device(policy):
    import torch
    for p in policy.parameters():
        return p.device
    return torch.device("cpu")


def _logits_f64(logits) -> Tuple[np.ndarray, float]:
    """(float64 numpy copy of `logits`, illegal floor): entries <= the floor are illegal. The floor
    is finfo(dtype).min of the input's own dtype, but never below float32's (so float32-masked
    logits handed over as float64 still read as masked)."""
    if hasattr(logits, "detach"):                              # a torch tensor (any device / dtype)
        import torch
        fl = float(torch.finfo(logits.dtype).min) if logits.dtype.is_floating_point else -np.inf
        x = logits.detach().cpu().to(torch.float64).numpy()          # (MPS has no float64)
    else:
        x = np.asarray(logits)
        fl = float(np.finfo(x.dtype).min) if np.issubdtype(x.dtype, np.floating) else -np.inf
        x = x.astype(np.float64)
    return x, max(fl, float(np.finfo(np.float32).min))


def greedy_actions(logits) -> np.ndarray:
    """Card-first greedy actions (SPEC §19.11.1) from masked joint logits `(..., A)`,
    A = 1 + 4·B (2305 / 577 / 161; slot-major: action 1 + s·B + j = slot s, tile/block j), torch or
    numpy, any dtype and batch shape. With p = softmax(logits) (float64), P(wait) = p_0 and
    P(s) = sum_j p_{1+s·B+j}: c* = argmax over (wait, slot 0..3) of these marginals, ties -> the
    earliest; wait -> 0, slot c* -> 1 + c*·B + argmax_j logits[1 + c*·B + j], ties -> the smallest j.
    Illegal entries (= finfo.min) are never chosen; a row with no legal entry at all gives 0.
    Returns int32 of shape `logits.shape[:-1]`. (The plain argmax would pick "wait" whenever the
    card probability is spread over many tiles.)"""
    x, floor = _logits_f64(logits)
    A = int(x.shape[-1]) if x.ndim else 0
    if A < 5 or (A - 1) % 4:
        raise ValueError(f"greedy_actions: expected joint logits over 1 + 4*B actions, got {A}")
    B = (A - 1) // 4
    lead = x.shape[:-1]
    x = x.reshape(-1, A)
    n = x.shape[0]
    legal = x > floor                                          # NaN also reads as illegal
    m = np.where(legal, x, -np.inf).max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.where(legal, np.exp(np.where(legal, x - m, 0.0)), 0.0)   # softmax up to its row constant
    cards = np.ascontiguousarray(e[:, 1:]).reshape(n, 4, B).sum(-1)
    c = np.concatenate([e[:, :1], cards], axis=1).argmax(-1)   # first maximum: wait, slot 0, 1, 2, 3
    s = np.clip(c - 1, 0, 3)
    seg = np.where(legal[:, 1:], x[:, 1:], -np.inf).reshape(n, 4, B)[np.arange(n), s]
    j = seg.argmax(-1)                                         # first maximum: the smallest j
    a = np.where(c == 0, 0, 1 + s * B + j)
    return a.astype(np.int32).reshape(lead)


def select_actions(logits, greedy: bool, generator) -> np.ndarray:
    """Actions from masked logits (illegal = finfo.min): the card-first greedy rule
    (greedy_actions, SPEC §19.11) when greedy, else a sample from softmax(logits) drawn with
    `generator` (a CPU torch.Generator)."""
    import torch
    if greedy:
        return greedy_actions(logits)
    probs = torch.softmax(logits.float(), dim=-1).cpu()
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1).numpy().astype(np.int32)


# ======================================================================================
# LeagueVecEnv (SPEC §15.2, §15.7.4-6)
# ======================================================================================
class LeagueVecEnv:
    """`num_envs` self-play Royale matches (num_agents=2) exposed to PuffeRL as one LEARNER row
    per match (a PufferLib-vecenv-compatible object: async_reset / send / recv / close).

    Per match and per episode the learner's seat (team 0/1) is drawn with the wrapper's seeded
    RNG and an opponent spec is drawn from the pool. Every step the opponent's action for its
    seat comes from: "self" -> the policy given to set_policy (no grad, on its own device);
    "ckpt:..." -> the frozen checkpoint (loaded once, cached, batched across the matches that
    share it; recurrent opponents keep a per-match LSTM state, zeroed at episode start);
    "bot:..." -> the C scripted bot for that match/team (a fresh bot per episode, seeded from
    the wrapper's RNG). Policy opponents SAMPLE from the masked distribution with the wrapper's
    torch.Generator (the card-first greedy rule, greedy_actions, with opponent_greedy=True).

    Learner rows only: rewards/terminals are the underlying env's values of the learner's row;
    observations/actions of the opponent never reach the caller, so they cannot enter the PPO
    buffer. On a terminal the result is recorded in the pool and a new seat + opponent are drawn
    (the env has already re-dealt, so the returned observation is the new match's, seen from
    the new seat).

    Logs: Royale's vec_log output at Royale's log_interval, unchanged except learner_score /
    learner_return / learner_win, rewritten to the learner's seat (the self-play env reports
    team 0), plus league/pool_size, league/episodes, league/snapshots and league/p/<opponent>, and
    the learner rows' per-card play rates cards/play_rate/<CARD_KEY> and cards/decisions (SPEC
    §19.7.6).

    Placement grid (SPEC §19.4): with placement_grid g > 1 the learner and policy opponents act
    in the coarse Discrete(1 + 4 B) space, while a "bot:..." opponent's row is switched to fine
    actions (Royale.set_row_grid) for its episode, so its plays are applied exactly as in v0.4.
    """

    def __init__(self, pool: OpponentPool, num_envs: int = 1, seed: int = 0, device="cpu",
                 opponent_greedy: bool = False, results_maxlen: int = 100_000, **royale_kwargs):
        import gymnasium
        import torch
        import pufferlib.spaces
        from . import binding
        from .royale import CardStats, Royale

        if royale_kwargs.pop("num_agents", 2) != 2:
            raise ValueError("LeagueVecEnv wraps self-play matches: num_agents is always 2")
        if int(num_envs) < 1:
            raise ValueError("num_envs must be >= 1")
        self._binding = binding
        self._torch = torch
        self.pool = pool
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.opponent_greedy = bool(opponent_greedy)
        self.royale_kwargs = dict(royale_kwargs)
        self.env = Royale(num_envs=self.num_envs, num_agents=2, seed=self.seed, **royale_kwargs)
        self.grid = self.env.placement_grid
        self.card_stats = CardStats(self.grid)                 # learner rows only (SPEC §19.7.6)
        prob = float(royale_kwargs.get("bot_play_prob", 0.2))
        self.bot_ppm = int(round(min(1.0, max(0.0, prob if np.isfinite(prob) else 0.2)) * 1e6))  # as the C env clamps

        # ---- PufferLib vecenv surface (learner rows only)
        n = self.num_envs
        self.num_agents = n
        self.agents_per_batch = n            # read by PuffeRL's LSTM path
        self.agent_per_batch = n             # PufferLib 3.0 native spelling
        self.single_observation_space = self.env.single_observation_space
        self.single_action_space = self.env.single_action_space
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, n)
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, n)
        self.emulated = False
        self.done = False
        self.driver_env = self.env
        self.agent_ids = np.arange(n)

        self.observations = np.zeros((n, *self.single_observation_space.shape), dtype=np.float32)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.terminals = np.zeros(n, dtype=self.env.terminals.dtype)
        self.truncations = np.zeros(n, dtype=self.env.truncations.dtype)
        self.masks = np.ones(n, dtype=bool)
        self.env_ids = np.arange(n, dtype=np.int32)
        self.infos: list = []

        # ---- per-match league state
        self.seats = np.zeros(n, dtype=np.int64)               # learner's team in match i
        self.opponents: List[str] = [SELF] * n                 # opponent spec of match i
        self._bots: List[object] = [None] * n                  # per-match C bot capsule
        self._ep_return = np.zeros(n, dtype=np.float64)        # learner's return this episode
        self._log_base = np.zeros((n, 3), dtype=np.float64)    # (win_0, win_1, n) since vec_log
        self._policy = None                                    # "self"
        self._ckpt_cache: Dict[str, Tuple[object, bool]] = {}  # spec -> (policy, recurrent)
        self._lstm: Dict[str, Tuple[object, object]] = {}      # spec -> (h, c) per match
        self._handles = list(self.env._c_env_list)
        self._tick = 0
        self.episodes = 0                                      # finished episodes (lifetime)
        # (opponent, learner result, seat) of recent episodes; bounded (drain_results() empties it)
        self.results: collections.deque = collections.deque(maxlen=int(results_maxlen))
        self._acc = np.zeros(4, dtype=np.float64)              # learner (return, score, win, n) since log
        self._reseed(None)

    # ---------------------------------------------------------------- RNG streams
    def _reseed(self, reset_seed: Optional[int]) -> None:
        """Wrapper streams (seats + bot seeds, policy-opponent sampling) as a pure function of
        (constructor seed, reset seed)."""
        def words(x):                        # injective for any int, including negative / huge seeds
            return [1 if x < 0 else 0, abs(int(x))]
        ss = np.random.SeedSequence(words(self.seed) + ([1] + words(reset_seed) if reset_seed is not None else [0])
                                    + [0x1EA6E])
        a, b = ss.spawn(2)
        self._rng = np.random.Generator(np.random.PCG64(a))
        self._gen = self._torch.Generator(device="cpu")
        self._gen.manual_seed(int(b.generate_state(1, np.uint64)[0]) & ((1 << 63) - 1))

    def rng_state(self) -> dict:
        """JSON-serialisable state of the wrapper's own streams (for run records / resume)."""
        return {"seat_rng": _jsonable_rng_state(self._rng.bit_generator.state),
                "opponent_generator": self._gen.get_state().tolist()}

    def load_rng_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state["seat_rng"]
        self._gen.set_state(self._torch.tensor(state["opponent_generator"], dtype=self._torch.uint8))

    # ---------------------------------------------------------------- policies
    def set_policy(self, policy) -> None:
        """The current learner policy, used for "self" opponents (inference only, no grad)."""
        self._policy = policy
        self._lstm.pop(SELF, None)

    def _opponent_policy(self, spec: str):
        if spec == SELF:
            if self._policy is None:
                raise RuntimeError("a 'self' opponent was drawn but set_policy() was never called")
            return self._policy, is_recurrent_policy(self._policy), policy_device(self._policy)
        if spec not in self._ckpt_cache:
            self._ckpt_cache[spec] = load_policy(parse_spec(spec)[1], self.device)
        pol, rec = self._ckpt_cache[spec]
        return pol, rec, self.device

    def _ckpt_grid(self, spec: str) -> int:
        from .torch import policy_grid
        return policy_grid(self._opponent_policy(spec)[0])

    def _prune_cache(self) -> None:
        """Forget frozen checkpoints that left the pool and are not playing any match."""
        live = set(self.opponents)
        for spec in list(self._ckpt_cache):
            if spec not in live and spec not in self.pool:
                del self._ckpt_cache[spec]
                self._lstm.pop(spec, None)

    # ---------------------------------------------------------------- episodes
    def _begin_episode(self, i: int) -> None:
        """New seat + opponent for match i (its env has just dealt a new match)."""
        self.seats[i] = int(self._rng.integers(2))
        spec = self.pool.sample()
        self.opponents[i] = spec
        kind, arg = parse_spec(spec)
        self._bots[i] = None
        if kind == "bot":
            seed = int(self._rng.integers(1 << 62))
            self._bots[i] = self._binding.bot_new(BOT_KINDS[arg], seed, self.bot_ppm)
        # SPEC §19.4: bots keep fine actions; a checkpoint plays through its own grid (D49, D63). Both rows
        # are set every episode (a no-op when everything is grid 1)
        seat = int(self.seats[i])
        opp_grid = 1 if kind == "bot" else (self.grid if kind == "self" else self._ckpt_grid(spec))
        self.env.set_row_grid(i, seat, self.grid)
        self.env.set_row_grid(i, 1 - seat, opp_grid)
        for h, c in self._lstm.values():                       # recurrent opponents start fresh
            h[i].zero_()
            c[i].zero_()
        self._ep_return[i] = 0.0

    def _learner_rows(self) -> np.ndarray:
        return 2 * np.arange(self.num_envs) + self.seats

    def _gather_obs(self) -> None:
        np.take(self.env.observations, self._learner_rows(), axis=0, out=self.observations, mode="clip")

    def _peek_all(self) -> None:
        for i, h in enumerate(self._handles):
            d = self._binding.env_log_peek(h)
            self._log_base[i] = (d["win_0"], d["win_1"], d["n"])

    # ---------------------------------------------------------------- vecenv protocol
    def async_reset(self, seed: Optional[int] = None) -> None:
        """Re-deal every match. With `seed` the matches are reseeded as Royale.reset(seed) does
        and the wrapper's streams restart from (constructor seed, seed); without, all streams
        continue."""
        if seed is not None:
            self._reseed(seed)
        self.env.reset(seed)
        self._tick = 0
        self._peek_all()                    # reset() does not clear the env logs: re-baseline
        self._acc[:] = 0.0
        for i in range(self.num_envs):
            self._begin_episode(i)
        self._gather_obs()
        self.rewards[:] = 0.0
        self.terminals[:] = 0
        self.truncations[:] = 0
        self.infos = []

    def reset(self, seed: Optional[int] = None):
        self.async_reset(seed)
        return self.observations, self.infos

    def _opponent_actions(self) -> np.ndarray:
        torch = self._torch
        n = self.num_envs
        out = np.zeros(n, dtype=np.int32)
        opp_rows = 2 * np.arange(n) + (1 - self.seats)
        groups: Dict[str, List[int]] = collections.OrderedDict()
        for i, spec in enumerate(self.opponents):
            groups.setdefault(spec, []).append(i)
        for spec, idx in groups.items():
            kind, _ = parse_spec(spec)
            if kind == "bot":
                for i in idx:
                    out[i] = self._binding.bot_act_env(self._bots[i], self._handles[i], int(1 - self.seats[i]))
                continue
            policy, recurrent, dev = self._opponent_policy(spec)
            ix = np.asarray(idx)
            with torch.no_grad():
                x = torch.as_tensor(self.env.observations[opp_rows[ix]]).to(dev)
                if recurrent:
                    if spec not in self._lstm:
                        hs = int(policy.hidden_size)
                        self._lstm[spec] = (torch.zeros(n, hs, device=dev), torch.zeros(n, hs, device=dev))
                    h, c = self._lstm[spec]
                    ti = torch.as_tensor(ix, device=dev)
                    state = {"lstm_h": h[ti], "lstm_c": c[ti]}
                    logits, _ = policy.forward_eval(x, state)
                    h[ti] = state["lstm_h"]
                    c[ti] = state["lstm_c"]
                else:
                    logits, _ = policy.forward_eval(x, {})
                out[ix] = select_actions(logits, self.opponent_greedy, self._gen)
        return out

    def send(self, actions) -> None:
        actions = np.asarray(actions).reshape(self.num_envs)
        n = self.num_envs
        learner_rows = self._learner_rows()
        self.card_stats.update(self.observations, actions)   # the learner's decision observations
        opp_actions = self._opponent_actions()
        full = self.env.actions
        full[learner_rows] = actions
        full[2 * np.arange(n) + (1 - self.seats)] = opp_actions
        self._binding.vec_step(self.env.c_envs)
        self._tick += 1

        # learner's view of this step (before any seat change)
        # mode="clip" writes straight into `out` (mode="raise" buffers a copy); the rows are valid
        np.take(self.env.rewards, learner_rows, out=self.rewards, mode="clip")
        np.take(self.env.terminals, learner_rows, out=self.terminals, mode="clip")
        np.take(self.env.truncations, learner_rows, out=self.truncations, mode="clip")
        self._ep_return += self.rewards

        done = np.flatnonzero(self.env.terminals[0::2])
        for i in done:
            i = int(i)
            d = self._binding.env_log_peek(self._handles[i])
            w0, w1, cnt = d["win_0"] - self._log_base[i, 0], d["win_1"] - self._log_base[i, 1], d["n"] - self._log_base[i, 2]
            self._log_base[i] = (d["win_0"], d["win_1"], d["n"])
            if cnt < 0.5:                      # cannot happen: a terminal always logs an episode
                raise RuntimeError(f"match {i} ended without an episode log")
            r0 = 1 if w0 > 0.5 else (-1 if w1 > 0.5 else 0)
            res = r0 if self.seats[i] == 0 else -r0
            spec = self.opponents[i]
            self.pool.record(spec, res)
            self.results.append((spec, res, int(self.seats[i])))
            self.episodes += 1
            self._acc += (self._ep_return[i], 0.5 * (res + 1), float(res == 1), 1.0)
            self._begin_episode(i)
        if len(done):
            self._prune_cache()

        infos = []
        if self._tick % self.env.log_interval == 0:
            log = self._binding.vec_log(self.env.c_envs)
            self._log_base[:] = 0.0            # vec_log cleared every env's accumulators
            if log:
                if self._acc[3] > 0:           # §15.7.5: learner keys refer to the learner's seat
                    k = self._acc[3]
                    log["learner_return"] = float(self._acc[0] / k)
                    log["learner_score"] = float(self._acc[1] / k)
                    log["learner_win"] = float(self._acc[2] / k)
                log.update(self.league_stats())
                infos.append(self.card_stats.emit(log))
            self._acc[:] = 0.0
        self._gather_obs()
        self.infos = infos

    def recv(self):
        return (self.observations, self.rewards, self.terminals, self.truncations, self.infos,
                self.env_ids, self.masks)

    def step(self, actions):
        self.send(actions)
        return self.observations, self.rewards, self.terminals, self.truncations, self.infos

    def drain_results(self) -> List[Tuple[str, int, int]]:
        """Return and forget the (opponent, learner result, seat) of the episodes finished since the
        last call (bounded by results_maxlen)."""
        out = list(self.results)
        self.results.clear()
        return out

    def league_stats(self) -> dict:
        """league/* keys: pool size, finished episodes, snapshots, learner score per opponent."""
        w = self.pool.weights()
        out = {"league/pool_size": float(len(w)), "league/episodes": float(self.episodes),
               "league/snapshots": float(len(self.pool.snapshots))}
        st = self.pool.stats()
        for spec in list(w) + [s for s in st if s not in w]:
            out[f"league/p/{spec}"] = float(st[spec]["p"]) if spec in st else 0.5
        return out

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        self._ckpt_cache.clear()
        self._lstm.clear()
        self._bots = [None] * self.num_envs


__all__ = ["OpponentPool", "LeagueVecEnv", "parse_spec", "pfsp_weight", "load_policy", "select_actions",
           "greedy_actions", "resolve_checkpoint_path", "checkpoint_env_config", "json_safe", "MATCH_ENV_KEYS",
           "is_recurrent_state_dict", "policy_kwargs_from_state_dict", "SELF", "load_weights", "build_policy",
           "load_state_dict_file", "policy_architecture", "architecture_diff"]
