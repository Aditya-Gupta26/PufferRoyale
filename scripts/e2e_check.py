#!/usr/bin/env python3
"""End-to-end verification ladder (SPEC §12: "scripts/e2e_check.py prints PASS").

Runs every check below, prints a PASS/FAIL table and exits non-zero on any failure:

   build       the compiled extension imports and is not older than its C sources
   codegen     generated headers are committed and byte-identical on regeneration
   c-tests     make test-c (warning-free)          asan   make asan (skip with --quick)
   pytest      builder suite (tests/builder)       (the tester suite is run separately)
   determinism identical env rollouts in two separate processes
   soak        N full matches through the env (random legal play vs the random bot and in
               self-play): finite obs, spatial in [0,1], zero-sum rewards, logs, <= 600 steps
   mask        env mask self-check (mask_check=True) + mask == play acceptance via Game
   zero-sum    shaped self-play rewards sum to 0 every step
   no-leak     the viewer's observation does not change when only the opponent's hidden
               deck / hand / elixir changes
   auto-reset  the terminal step returns the NEW match's observation
   reward_v2   SPEC §19.1: shaped bot matches (heuristic vs random, constant and annealed weights,
               gamma 1 / 0.999 / 0.99): r_1 = -r_0 every step and sum_t gamma^t F_t = 0 per match
   policy_heads SPEC §19.6: flat + conditional heads (Policy and Recurrent) x placement grid 1/2/4
               on real observations: masks, illegal logits == finfo.min, P(s) P(j | s) factorisation
   decks       SPEC §19.5: deck-sampler statistics ~ weights (5 sigma), held-out decks never dealt
               (rejection counted), mirror deals, determinism
   grid        SPEC §19.4: every coarse-legal action (g 1/2/4) maps to a fine play the engine
               accepts, on real states; env with g 2/4 and random coarse-legal play: 0 illegal
   perf        engine >= 20,000 ticks/s and env >= 2,000 steps/s (SPEC §11)
   train       scripts/train.py smoke run (no NaN)   eval   scripts/eval.py on its checkpoint
   llm         SPEC §17: scripts/llm_match.py, mock_first_legal (text-only) vs the heuristic bot, 2 matches,
               no network; anthropic: models are refused without --allow-network
   league      SPEC §15: a tiny league run (heuristic anchor + self-play + PFSP snapshots), a tiny
               MMD league run, and a tournament of the 3 bots + the league learner with a
               meta-game Nash solve (equilibrium verified)
   ops         SPEC §19.7: train.py --resume, league_train.py --init-from, two tiny stages.py rungs,
               eval.py --decks on a recurrent (LSTM, placement grid 2) league snapshot

    python scripts/e2e_check.py            # full ladder (a few minutes)
    python scripts/e2e_check.py --quick    # skip ASan, smaller soak / train
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
RESULTS = []


def check(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                detail = fn(*a, **k) or ""
                ok = True
            except Exception as e:  # noqa: BLE001 -- report every failure, keep going
                ok = False
                detail = f"{type(e).__name__}: {e}".strip()
                if not isinstance(e, AssertionError):
                    detail += "\n" + traceback.format_exc(limit=3)
            RESULTS.append((name, ok, time.time() - t0, detail))
            print(f"[{'PASS' if ok else 'FAIL'}] {name:12s} {time.time() - t0:6.1f}s  {detail.splitlines()[0] if detail else ''}",
                  flush=True)
            return ok
        return run
    return deco


def sh(cmd, timeout=900, env=None):
    """Run a command in its own process group; on timeout the WHOLE group (make and the test
    binaries it started) is killed, so nothing is left running, and the check fails."""
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = p.communicate()
        return 124, (out or "") + f"\nTIMEOUT after {timeout}s: {' '.join(cmd)}"
    return p.returncode, out


@check("build")
def check_build():
    import pufferroyale  # noqa: F401
    import pufferroyale.royale  # noqa: F401
    import pufferroyale.torch  # noqa: F401
    so = glob.glob(os.path.join(ROOT, "pufferroyale", "binding*.so"))
    assert so, "no compiled extension: run `make build`"
    srcs = glob.glob(os.path.join(ROOT, "pufferroyale", "csrc", "*.h")) + [os.path.join(ROOT, "pufferroyale", "binding.c")]
    newest = max(os.path.getmtime(p) for p in srcs)
    assert os.path.getmtime(so[0]) >= newest, "extension older than its C sources: run `make build`"
    from pufferroyale import binding
    if not os.environ.get("PUFFERROYALE_BINDING_DIR"):
        assert binding.royale_layout()["DEBUG_BUILD"] == 0, \
            "the in-place extension is a sanitizer build: run `make build` (debug builds go to build/debug)"
    return os.path.basename(so[0])


@check("codegen")
def check_codegen():
    for script in ("gen_card_db.py", "gen_arena.py"):
        code, out = sh([PY, os.path.join("tools", script), "--check"])
        assert code == 0, out[-500:]
    with tempfile.TemporaryDirectory() as d:
        for sub in ("tools", "data"):
            shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub))
        os.makedirs(os.path.join(d, "pufferroyale", "csrc"))
        runs = []
        for _ in range(2):
            for script in ("gen_card_db.py", "gen_arena.py"):
                subprocess.run([PY, os.path.join(d, "tools", script)], cwd=d, check=True, capture_output=True,
                               timeout=300)
            runs.append({f: open(os.path.join(d, "pufferroyale", "csrc", f), "rb").read()
                         for f in ("pr_card_db.h", "pr_arena_db.h")})
        assert runs[0] == runs[1], "regeneration is not byte-identical"
        for f, data in runs[0].items():
            assert data == open(os.path.join(ROOT, "pufferroyale", "csrc", f), "rb").read(), f"{f} is stale"
    return "byte-identical x2, committed copies up to date"


@check("c-tests")
def check_c_tests():
    code, out = sh(["make", "-B", "test-c"])
    assert code == 0, out[-1500:]
    warns = [l for l in out.splitlines() if "warning:" in l]
    assert not warns, "\n".join(warns[:10])
    checks = sum(int(l.split()[0]) for l in out.splitlines() if l.endswith("0 failures"))
    return f"{checks} checks, 0 failures, warning-free"


@check("asan")
def check_asan():
    code, out = sh(["make", "asan"], timeout=1200)
    assert code == 0, out[-1500:]
    for marker in ("ERROR: AddressSanitizer", "runtime error:", "ERROR: LeakSanitizer"):
        assert marker not in out, marker
    return "clean"


@check("pytest")
def check_pytest():
    code, out = sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/builder"])
    assert code == 0, out[-1500:]
    return out.strip().splitlines()[-1]


ROLLOUT = r"""
import sys, hashlib, numpy as np
sys.path.insert(0, sys.argv[1])
import pufferroyale
from pufferroyale import royale as R
env = pufferroyale.Royale(num_envs=2, num_agents=2, deck0='random', deck1='random', seed=5, log_interval=10**9)
obs, _ = env.reset(seed=5)
rng = np.random.default_rng(3)
h = hashlib.sha256()
for t in range(700):
    acts = np.zeros(4, np.int32)
    for r in range(4):
        leg = np.flatnonzero(obs[r, R.MASK_OFFSET:] > 0.5)
        if rng.random() < 0.3:
            acts[r] = int(leg[rng.integers(len(leg))])
    obs, rew, term, trunc, _ = env.step(acts)
    h.update(obs.tobytes()); h.update(rew.tobytes()); h.update(term.tobytes())
print(h.hexdigest(), env.env_info(0)['hash'])
"""


@check("determinism")
def check_determinism():
    outs = set()
    for i in range(2):
        env = dict(os.environ, PYTHONHASHSEED=str(1000 + i))
        code, out = sh([PY, "-c", ROLLOUT, ROOT], env=env)
        assert code == 0, out[-800:]
        outs.add(out.strip().splitlines()[-1])
    assert len(outs) == 1, f"rollouts differ across processes: {outs}"
    return "700-step rollouts identical across 2 processes"


@check("soak")
def check_soak(matches):
    import pufferroyale
    from pufferroyale import royale as R

    done = 0
    logs = []
    for mode in ("selfplay", "vsbot"):
        kw = dict(num_agents=2) if mode == "selfplay" else dict(num_agents=1, opponent="random", learner_side="random")
        env = pufferroyale.Royale(num_envs=4, deck0="random", deck1="random", seed=11, log_interval=1,
                                  mask_check=False, **kw)
        obs, _ = env.reset(seed=11)
        rng = np.random.default_rng(1)
        steps = np.zeros(4, dtype=np.int64)
        target = matches // 2
        n_done = 0
        while n_done < target:
            acts = np.zeros(env.num_agents, np.int32)
            for r in range(env.num_agents):
                if rng.random() < 0.25:
                    leg = np.flatnonzero(obs[r, R.MASK_OFFSET:] > 0.5)
                    acts[r] = int(leg[rng.integers(len(leg))])
            obs, rew, term, trunc, infos = env.step(acts)
            assert np.isfinite(obs).all(), "non-finite observation"
            sp = obs[:, R.SPATIAL_OFFSET:R.SPATIAL_OFFSET + int(np.prod(R.SPATIAL_SHAPE))]
            assert sp.min() >= 0.0 and sp.max() <= 1.0, "spatial planes outside [0, 1]"
            assert not trunc.any()
            if mode == "selfplay":
                assert np.allclose(rew[0::2] + rew[1::2], 0.0), "self-play rewards not zero-sum"
            steps += 1
            per_env = term.reshape(4, -1)[:, 0]
            for i in np.flatnonzero(per_env):
                assert steps[i] <= 600, f"match lasted {steps[i]} steps"
                steps[i] = 0
                n_done += 1
            for lg in infos:
                if lg:
                    logs.append(lg)
                    assert lg["illegal_actions"] == 0, "a masked-legal action was refused"
                    assert abs(lg["win_0"] + lg["win_1"] + lg["draw"] - 1.0) < 1e-5
        env.close()
        done += n_done
    assert logs, "no episode logs"
    return f"{done} matches (self-play + vs random bot), all invariants held"


@check("mask")
def check_mask():
    import pufferroyale
    from pufferroyale import royale as R

    env = pufferroyale.Royale(num_envs=2, num_agents=2, deck0="random", deck1="random", seed=4, mask_check=True,
                              log_interval=1)
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(4)
    mismatches, eps = 0.0, 0
    for t in range(1300):
        acts = np.zeros(4, np.int32)
        for r in range(4):
            if rng.random() < 0.3:
                leg = np.flatnonzero(obs[r, R.MASK_OFFSET:] > 0.5)
                acts[r] = int(leg[rng.integers(len(leg))])
        obs, rew, term, trunc, infos = env.step(acts)
        for lg in infos:
            if lg:
                mismatches += lg["mask_mismatch"] * lg["n"]
                eps += lg["n"]
    env.close()
    assert eps > 0 and mismatches == 0, f"{mismatches} mask/legality mismatches in {eps} episodes"
    g = pufferroyale.Game(deck0="random", deck1="random", seed=9, deploy_lockout_ticks=0)
    checked = 0
    for step in range(12):
        for team in (0, 1):
            snap = g.snapshot()
            m = g.legal_mask(team)
            for a in rng.choice(np.arange(1, R.MASK_SIZE), 200, replace=False):
                ok = g.play_action(team, int(a)) == 0
                assert ok == bool(m[a]), f"mask {m[a]} but play accepted={ok} for action {a}"
                g.restore(snap)
                checked += 1
        g.tick(60)
    return f"self-check 0 mismatches over {int(eps)} episodes; {checked} play/mask spot checks agree"


@check("zero-sum")
def check_zero_sum():
    import pufferroyale
    from pufferroyale import royale as R

    env = pufferroyale.Royale(num_envs=2, num_agents=2, reward_tower=0.5, reward_crown=0.3, seed=2, log_interval=10 ** 9)
    obs, _ = env.reset(seed=2)
    rng = np.random.default_rng(2)
    nonzero = 0
    for t in range(1500):
        acts = np.array([int(rng.choice(np.flatnonzero(obs[r, R.MASK_OFFSET:] > 0.5))) if rng.random() < 0.5 else 0
                         for r in range(4)], np.int32)
        obs, rew, *_ = env.step(acts)
        assert abs(rew[0] + rew[1]) < 1e-6 and abs(rew[2] + rew[3]) < 1e-6, f"step {t}: {rew}"
        nonzero += int(rew[0] != 0)
    env.close()
    assert nonzero > 0
    return f"1500 shaped steps zero-sum ({nonzero} non-zero)"


@check("no-leak")
def check_no_leak():
    import pufferroyale

    for view in (0, 1):
        decks_a = ["hog26", "giant"] if view == 0 else ["giant", "hog26"]
        decks_b = ["hog26", "bait"] if view == 0 else ["bait", "hog26"]
        ea = pufferroyale.Royale(num_agents=2, deck0=decks_a[0], deck1=decks_a[1], seed=7)
        eb = pufferroyale.Royale(num_agents=2, deck0=decks_b[0], deck1=decks_b[1], seed=7)
        oa, _ = ea.reset(seed=7)
        ob, _ = eb.reset(seed=7)
        for t in range(40):
            assert np.array_equal(oa[view], ob[view]), f"view {view} sees the opponent deck at step {t}"
            oa, *_ = ea.step(np.zeros(2, np.int32))
            ob, *_ = eb.step(np.zeros(2, np.int32))
        ea.close()
        eb.close()
    # hidden hand order and elixir of the opponent: set directly on a Game
    g = pufferroyale.Game(deck0="bait", deck1="giant", seed=3)
    g.tick(120)
    base = g.obs(0)
    s = g.state()
    order = s["queue"][1][::-1] + s["hand"][1][::-1]
    g.set_hand(1, order)
    g.set_elixir(1, 1234)
    assert np.array_equal(base, g.obs(0)), "team 0's obs depends on team 1's hidden hand/elixir"
    return "deck swap, hand reorder and elixir change of the opponent are invisible"


@check("auto-reset")
def check_auto_reset():
    import pufferroyale
    from pufferroyale import royale as R

    env = pufferroyale.Royale(num_agents=2, frame_skip=50, seed=0, log_interval=1)
    obs, _ = env.reset(seed=0)
    for t in range(1, 121):
        obs, rew, term, trunc, infos = env.step(np.zeros(2, np.int32))
    assert term.all() and not trunc.any(), "no terminal at tick 6000"
    tick = R.scalar(obs, "tick")
    assert np.all(tick == 0.0), "the returned observation must be the new match (tick 0)"
    assert np.all(R.mask(obs)[:, 1:].sum(1) == 0), "new match: lockout mask"
    lg = [i for i in infos if i][-1]
    assert lg["draw"] == 1.0 and lg["episode_length"] == 120
    obs, rew, term, *_ = env.step(np.zeros(2, np.int32))
    assert not term.any() and np.all(rew == 0)
    env.close()
    return "terminal step returns the new match, log written, next step continues"


def _bot_actions(env, bots):
    """One decision of each scripted bot for its agent row (row r = team r % 2 of match r // 2)."""
    return np.array([b.act_env(env, r % 2, r // 2) for r, b in enumerate(bots)], np.int32)


@check("reward_v2")
def check_reward_v2():
    """SPEC §19.1 on shaped bot matches (heuristic vs random bot, self-play rows): r_1 = -r_0 exactly on
    every step, and per finished match and team sum_t gamma^t r_t = gamma^(T-1) * result (|.| <= 1e-4 in
    float64 of the stored float32 rewards), i.e. sum_t gamma^t F_t = 0 -- constant and annealed weights."""
    import pufferroyale

    shaped, matches = 0, 0
    # (gamma, anneal steps N, offset n0, matches): the third reaches m = 0 inside its first match
    for k, (gamma, anneal, offset, n_matches) in enumerate(((1.0, 0, 0, 2), (0.999, 0, 0, 2), (0.99, 200, 0, 2),
                                                            (0.999, 500, 250, 2))):
        env = pufferroyale.Royale(num_envs=1, num_agents=2, frame_skip=20, deck0="random", deck1="random",
                                  reward_tower=0.3, reward_crown=0.2, reward_elixir=0.05, reward_play=0.02,
                                  reward_elixir_cap=2.0, reward_play_cap=3.0, reward_gamma=gamma,
                                  shaping_anneal_steps=anneal, shaping_step_offset=offset, log_interval=1, seed=k)
        env.reset(seed=k)
        bots = [pufferroyale.Bot("heuristic", seed=10 * k), pufferroyale.Bot("random", seed=10 * k + 1, play_prob=0.4)]
        rews, done, steps = [], 0, 0
        while done < n_matches:
            _, rew, term, _, infos = env.step(_bot_actions(env, bots))
            steps += 1
            assert rew[0] == -rew[1] and rew[0] + rew[1] == 0.0, f"gamma {gamma}: not zero-sum: {rew}"
            shaped += int(rew[0] != 0.0)
            rews.append(float(rew[0]))
            if term[0]:
                lg = [d for d in infos if d]
                assert lg and lg[-1]["n"] == 1, "one episode log per finished match"
                res = 1.0 if lg[-1]["win_0"] else (-1.0 if lg[-1]["win_1"] else 0.0)
                x = np.asarray(rews, np.float32).astype(np.float64)
                disc = gamma ** np.arange(len(x))
                for team, sign in ((0, 1.0), (1, -1.0)):           # team 1's rewards are -r_0
                    tot = float(np.sum(disc * sign * x))
                    want = gamma ** (len(x) - 1) * sign * res
                    assert abs(tot - want) <= 1e-4, f"gamma {gamma} N {anneal}: team {team} {tot} != {want}"
                rews, done = [], done + 1
        info = env.env_info(0)
        m = max(0.0, 1.0 - (offset + steps) / anneal) if anneal else 1.0
        assert info["env_steps"] == steps and abs(info["shaping_multiplier"] - m) < 1e-9, (info, m)
        env.close()
        matches += done
    assert shaped > 100, f"only {shaped} shaped steps"
    return f"{matches} bot matches (gamma 1 / 0.999 / 0.99, constant + annealed): zero-sum, sum gamma^t F_t = 0"


def _bot_observations(grid, num_envs=2, steps=80, seed=0):
    """(env, observations) of real states: heuristic bots on both seats (fine actions) in a self-play
    env of placement grid `grid`, sampled every 10 steps."""
    import pufferroyale

    env = pufferroyale.Royale(num_envs=num_envs, num_agents=2, placement_grid=grid, deck0="random", deck1="random",
                              seed=seed, log_interval=10 ** 9)
    env.reset(seed=seed)
    for i in range(num_envs):
        env.set_row_grid(i, 0, 1)
        env.set_row_grid(i, 1, 1)
    bots = [pufferroyale.Bot("heuristic", seed=seed * 10 + r) for r in range(2 * num_envs)]
    out = []
    for t in range(steps):
        obs, *_ = env.step(_bot_actions(env, bots))
        if t % 10 == 9:
            out.append(obs.copy())
    return env, np.concatenate(out)


@check("policy_heads")
def check_policy_heads():
    """SPEC §19.6 on real observations, both heads x g in {1, 2, 4}: the mask is the env's coarse mask,
    illegal logits exactly finfo.min, legal probabilities > 0, finite logits / values / gradients; the
    conditional head factorises (softmax(joint) = P(s) P(j | s)); Recurrent training forward and
    forward_eval mask the same way."""
    import torch
    import pufferlib.pytorch
    from pufferroyale import royale as R
    from pufferroyale import torch as prt

    neg = torch.finfo(torch.float32).min
    rows = 0
    for g in (1, 2, 4):
        env, obs = _bot_observations(g, seed=g)
        x = torch.as_tensor(obs)
        mask = prt.action_mask(x, g)
        assert torch.equal(mask, torch.as_tensor(R.action_mask(obs, g))), f"g {g}: torch / numpy masks differ"
        assert mask[:, 1:].any(), f"g {g}: no legal play in the sampled states"
        for head in ("flat", "conditional"):
            torch.manual_seed(g)
            pol = prt.Policy(env, head=head)
            assert pol.placement_grid == g and pol.n_actions == R.n_actions(g)
            with torch.no_grad():                              # break the near-uniform init
                for q in pol.parameters():
                    q.add_(torch.randn_like(q) * 0.05)
            logits, value = pol.forward_eval(x)
            assert logits.shape == (len(obs), R.n_actions(g)), logits.shape
            assert (logits[~mask] == neg).all(), f"{head} g {g}: illegal logits are not finfo.min"
            p = torch.softmax(logits, -1)
            assert (p[mask] > 0).all() and (p[~mask] == 0).all(), f"{head} g {g}: probabilities vs mask"
            assert torch.isfinite(logits).all() and torch.isfinite(value).all()
            a, lp, ent = pufferlib.pytorch.sample_logits(logits)
            assert mask[torch.arange(len(obs)), a].all(), f"{head} g {g}: sampled an illegal action"
            (lp.mean() + ent.mean() + value.mean()).backward()
            assert all(q.grad is None or torch.isfinite(q.grad).all() for q in pol.parameters())
            if head == "conditional":
                with torch.no_grad():
                    parts = pol.conditional_parts(x)
                pc, pp, m = parts["card_logp"].exp(), parts["pos_logp"].exp(), parts["mask"][:, 1:]
                N, B = len(obs), pol.n_blocks
                joint = (pc[:, 1:, None] * pp).reshape(N, 4 * B)
                pd = p.detach()
                assert (pd[:, 0] - pc[:, 0]).abs().max() < 1e-5, f"g {g}: P(wait)"
                assert (pd[:, 1:] - torch.where(m, joint, 0)).abs().max() < 1e-5, f"g {g}: P(s) P(j | s)"
                assert torch.equal(pc[:, 1:] > 0, m.reshape(N, 4, B).any(-1)), f"g {g}: slot masking"
            rec = prt.Recurrent(env, prt.Policy(env, head=head))
            seq = x[:12].reshape(3, 4, -1)                     # (segments, horizon, OBS)
            lg2, v2 = rec.forward(seq, {"lstm_h": None, "lstm_c": None})
            lg3, _ = rec.forward_eval(x[:12], {"lstm_h": None, "lstm_c": None})
            m12 = mask[:12]
            assert (lg2[~m12] == neg).all() and (lg3[~m12] == neg).all() and v2.shape == (3, 4)
            assert torch.isfinite(lg2).all() and torch.isfinite(lg3).all()
            rows += len(obs)
        env.close()
    return f"flat + conditional heads, g 1/2/4, {rows} real observation rows: masks, finfo.min, factorisation"


@check("decks")
def check_decks():
    """SPEC §19.5: draws ∝ pool weights and random_deck_frac (5-sigma bounds over many deals), held-out
    decks never dealt (a held-out random draw is redrawn and counted), mirror deals, determinism."""
    import math
    from pufferroyale import decks as D
    from pufferroyale import royale as R

    env = R.Royale(num_envs=4, deck_pool="hog26:1;giant:3", random_deck_frac=0.25, heldout_decks="xbow;random:200:5",
                   seed=1)
    held = {D.deck_key(d) for d, _ in D.parse_deck_set("xbow;random:200:5")}
    deals = 4                                                  # construction deals every match
    for k in range(1200):
        env.reset(seed=1 if k == 0 else None)
        deals += 4
        for i in range(4):
            info = env.env_info(i)
            assert info["deck_sampler"] and info["deck0_ignored"] and info["deck1_ignored"]
            for d in info["decks"]:
                assert D.deck_key(d) not in held, f"held-out deck dealt: {d}"
    c = env.deck_counts()
    env.close()
    seats = 2 * deals
    assert int(c["pool"].sum()) + c["random"] == seats, c
    for got, p, what in ((c["random"], 0.25, "random"), (c["pool"][0], 0.75 / 4, "hog26 (w 1)"),
                         (c["pool"][1], 0.75 * 3 / 4, "giant (w 3)")):
        sd = math.sqrt(p * (1 - p) * seats)
        assert abs(got - p * seats) < 5 * sd, f"{what}: {got} seats, expected {p * seats:.0f} +- {5 * sd:.0f}"
    # a random draw equal to a held-out deck is redrawn: hold out the first deck seed 6 deals
    first = R.Royale(num_envs=1, random_deck_frac=1.0, seed=6)
    d0 = first.env_info(0)["decks"][0]
    first.close()
    env = R.Royale(num_envs=1, random_deck_frac=1.0, heldout_decks=[list(d0)], seed=6)
    d1 = env.env_info(0)["decks"][0]
    rejected = env.deck_counts()["random_rejected"]
    env.close()
    assert D.deck_key(d1) != D.deck_key(d0) and rejected == 1, (d0, d1, rejected)
    # mirror: both seats get the same deck, counted twice; same seed -> same deals
    seqs = []
    for _ in range(2):
        env = R.Royale(num_envs=2, deck_pool="hog26;golem;bait", random_deck_frac=0.5, deck_draw="mirror", seed=4)
        s = []
        for k in range(30):
            env.reset(seed=4 if k == 0 else None)
            for i in range(2):
                d = env.env_info(i)["decks"]
                assert D.deck_key(d[0]) == D.deck_key(d[1]), f"mirror deal differs: {d}"
                s.append(D.deck_key(d[0]))
        mc = env.deck_counts()
        env.close()
        assert mc["random"] % 2 == 0 and all(int(x) % 2 == 0 for x in mc["pool"])
        seqs.append(s)
    assert seqs[0] == seqs[1], "deals are not deterministic"
    return (f"{seats} seats: random {c['random']} (exp {0.25 * seats:.0f}), pool {c['pool'].tolist()} "
            f"(exp {0.75 / 4 * seats:.0f}/{0.75 * 3 / 4 * seats:.0f}); held-out never dealt; mirror + rejection ok")


@check("grid")
def check_grid():
    """SPEC §19.4 on real states: the coarse mask is the block-max of the fine mask, and EVERY coarse-legal
    action (g = 2, 4; g = 1 the identity) maps through Game.coarse_to_fine to a fine action the engine
    accepts (play_action returns OK); a Royale with placement_grid g playing random coarse-legal actions
    has 0 illegal actions."""
    import pufferroyale
    from pufferroyale import royale as R

    accepted = 0
    g0 = pufferroyale.Game(deck0="pekka_bridge", deck1="royal_hogs", seed=11, deploy_lockout_ticks=0)
    bots = [pufferroyale.Bot("heuristic", 1), pufferroyale.Bot("random", 2, play_prob=0.5)]
    for k in range(25):
        for team in (0, 1):
            fine = g0.legal_mask(team).astype(bool)
            obs = g0.obs(team)
            assert np.array_equal(R.action_mask(obs, 1), fine), "obs mask != engine legal mask"
            for g in (1, 2, 4):
                cm = R.action_mask(obs, g)
                rows, cols = R.grid_shape(g)
                for a in np.flatnonzero(cm[1:]) + 1:
                    f = g0.coarse_to_fine(team, int(a), g)
                    assert f > 0 and fine[f], f"g {g}: coarse-legal {a} -> {f} not legal"
                    s, b = divmod(int(a) - 1, rows * cols)
                    tx, ty = (f - 1) % 18, ((f - 1) % 576) // 18
                    assert (f - 1) // 576 == s and ty // g == b // cols and tx // g == b % cols, \
                        f"g {g}: {a} -> tile ({tx}, {ty}) outside its block"
                    snap = g0.snapshot()
                    err = g0.play_action(team, f)
                    g0.restore(snap)
                    assert err == 0, f"g {g}: engine refused fine action {f} for coarse {a}: {err!r}"
                    accepted += 1
                for a in np.flatnonzero(~cm[1:]) + 1:
                    assert g0.coarse_to_fine(team, int(a), g) == 0, f"g {g}: coarse-illegal {a} maps to a play"
        for team in (0, 1):
            a = bots[team].act(g0, team)
            if a:
                g0.play_action(team, a)
        g0.tick(15)
        if g0.state()["over"]:
            break
    assert accepted > 3000, f"only {accepted} coarse-legal actions checked"
    plays = {}
    for g in (2, 4):
        env = pufferroyale.Royale(num_envs=2, placement_grid=g, deck0="miner_poison", deck1="lavaloon", frame_skip=10,
                                  log_interval=1, seed=g)
        obs, _ = env.reset(seed=g)
        rng = np.random.default_rng(g)
        n_play = illegal = 0
        for _ in range(600):
            m = R.action_mask(obs, g)
            act = np.zeros(len(obs), np.int32)
            for r in range(len(obs)):
                legal = np.flatnonzero(m[r, 1:]) + 1
                if legal.size and rng.random() < 0.5:
                    act[r] = legal[rng.integers(legal.size)]
            obs, _, _, _, infos = env.step(act)
            for d in infos:
                if d:
                    n_play += (d["plays_0"] + d["plays_1"]) * d["n"]
                    illegal += d["illegal_actions"] * d["n"]
        env.close()
        assert n_play > 50 and illegal == 0, f"g {g}: {n_play} plays, {illegal} illegal"
        plays[g] = int(n_play)
    return (f"{accepted} coarse-legal actions (g 1/2/4) accepted by the engine; env g2/g4 plays "
            f"{plays[2]}/{plays[4]}, 0 illegal")


@check("perf")
def check_perf():
    import pufferroyale
    from pufferroyale import royale as R

    rng = np.random.default_rng(0)
    best_tick = 0.0
    for rep in range(3):   # SPEC §16.5: random decks from all 64 cards, tower troops varied
        g = pufferroyale.Game(deck0="random", deck1="random", seed=rep,
                              tower_troop0=pufferroyale.TOWER_TROOPS[rep % 4],
                              tower_troop1=pufferroyale.TOWER_TROOPS[(rep + 1) % 4])
        t_eng, n = 0.0, 0
        while not g.state()["over"]:
            for team in (0, 1):
                if rng.random() < 0.5:
                    leg = np.flatnonzero(g.legal_mask(team)[1:]) + 1
                    if len(leg):
                        g.play_action(team, int(leg[rng.integers(len(leg))]))
            t0 = time.perf_counter()
            g.tick(20)
            t_eng += time.perf_counter() - t0
            n += 20
        best_tick = max(best_tick, n / t_eng)
    # SPEC §14 stress: 7 Skeleton Armies per side fighting across the river (>= 10,000 ticks/s)
    best_stress = 0.0
    for rep in range(3):
        g = pufferroyale.Game(deck0="bait", deck1="bait", seed=1, deploy_lockout_ticks=0)
        for k in range(7):
            g.spawn(0, "Skeleton Army", 3000 + (k % 4) * 4000, 20000 + (k // 4) * 3000)
            g.spawn(1, "Skeleton Army", 3000 + (k % 4) * 4000, 12000 - (k // 4) * 3000)
        t0 = time.perf_counter()
        g.tick(300)
        best_stress = max(best_stress, 300 / (time.perf_counter() - t0))
    env = pufferroyale.Royale(num_envs=1, num_agents=2, frame_skip=10, seed=0, log_interval=10 ** 9,
                              deck0="random", deck1="random", tower_troop0="dagger_duchess",
                              tower_troop1="royal_chef")
    obs, _ = env.reset(seed=0)
    acts = np.zeros(2, np.int32)
    best_step = 0.0
    for rep in range(3):
        t_env = 0.0
        for _ in range(1500):
            for r in range(2):
                acts[r] = 0
                if rng.random() < 0.1:
                    leg = np.flatnonzero(obs[r, R.MASK_OFFSET + 1:] > 0.5)
                    if len(leg):
                        acts[r] = int(leg[rng.integers(len(leg))]) + 1
            t0 = time.perf_counter()
            obs, *_ = env.step(acts)
            t_env += time.perf_counter() - t0
        best_step = max(best_step, 1500 / t_env)
    env.close()
    assert best_tick >= 20000, f"engine {best_tick:.0f} ticks/s < 20,000"
    assert best_stress >= 10000, f"stress {best_stress:.0f} ticks/s < 10,000"
    assert best_step >= 2000, f"env {best_step:.0f} steps/s < 2,000"
    return (f"engine {best_tick:,.0f} ticks/s (random 64-card decks), stress {best_stress:,.0f} ticks/s, "
            f"env {best_step:,.0f} steps/s (fs 10, 2 agents, obs)")


@check("llm")
def check_llm():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "llm.json")
        code, text = sh([PY, os.path.join("scripts", "llm_match.py"), "--model", "mock_first_legal", "--opponent",
                         "bot:heuristic", "--matches", "2", "--seed", "0", "--out", out], timeout=600)
        assert code == 0, text[-1500:]
        s = json.loads(text.strip().splitlines()[-1])
        assert s["matches"] == 2 and s["wins"] + s["draws"] + s["losses"] == 2, s
        assert s["decisions"] > 0 and s["plays_per_match"] > 0, "the text-only mock never played"
        assert s["illegal_rate"] == 0 and s["parse_error_rate"] == 0 and s["model_error_rate"] == 0, s
        rows = open(os.path.splitext(out)[0] + ".transcripts.jsonl").read().splitlines()
        assert len(rows) == s["decisions"], "one transcript line per decision"
    code, text = sh([PY, os.path.join("scripts", "llm_match.py"), "--model", "anthropic:claude-opus-5-5", "--matches", "1"],
                    timeout=120)
    assert code == 2 and "--allow-network" in text, "network models must be refused without --allow-network"
    return (f"mock_first_legal vs heuristic: W{s['wins']} D{s['draws']} L{s['losses']}, {s['decisions']} decisions, "
            f"0 illegal / parse errors; anthropic refused without --allow-network")


@check("train")
def check_train(steps, workdir):
    summary = os.path.join(workdir, "train_summary.json")
    code, out = sh([PY, os.path.join("scripts", "train.py"), "--total-timesteps", str(steps), "--quiet",
                    "--data-dir", workdir, "--summary-json", summary], timeout=1500)
    assert code == 0, out[-1500:]
    s = json.load(open(summary))
    assert not s["nonfinite_losses"] and not s["nonfinite_params"], "NaN during training"
    assert s["global_step"] >= steps
    return f"{s['global_step']} agent steps at {s['sps']:.0f} SPS, finite losses ({s['checkpoint']})"


@check("eval")
def check_eval(workdir):
    runs = sorted(glob.glob(os.path.join(workdir, "pufferroyale_*", "model_*.pt")))
    assert runs, "no checkpoint from the train check"
    out_json = os.path.join(workdir, "eval.json")
    code, out = sh([PY, os.path.join("scripts", "eval.py"), "--checkpoint", os.path.dirname(runs[-1]), "--episodes", "1",
                    "--json", out_json], timeout=900)
    assert code == 0, out[-1500:]
    rows = json.load(open(out_json))["results"]
    assert len(rows) == 6
    return " ".join(f"{r['opponent']}/{r['seat']}:{r['wins']}-{r['draws']}-{r['losses']}" for r in rows)


def _league_run(workdir, run_id, extra):
    code, out = sh([PY, os.path.join("scripts", "league_train.py"), "--total-timesteps", "2048", "--num-envs", "4",
                    "--bptt-horizon", "64", "--frame-skip", "50", "--anchors", "bot:heuristic", "--self-play-frac",
                    "0.4", "--anchor-frac", "0.4", "--snapshot-interval", "2", "--seed", "0", "--data-dir", workdir,
                    "--run-id", run_id, *extra], timeout=900)
    assert code == 0, out[-1500:]
    s = json.loads(out.strip().splitlines()[-1])
    assert not s["nonfinite_losses"] and not s["nonfinite_params"], f"{run_id}: non-finite values"
    assert s["global_step"] >= 2048 and s["epoch"] == 8, s
    run = os.path.join(workdir, "league", run_id)
    snaps = sorted(glob.glob(os.path.join(run, "snap_*.pt")))
    assert len(snaps) == 4, f"{run_id}: snapshots {snaps}"
    hist = [json.loads(l) for l in open(os.path.join(run, "history.jsonl"))]
    games = sum(r["games"] for h in hist for r in h["results"].values())
    assert games > 0, f"{run_id}: no finished league episode"
    stats = s["stats"]
    assert "bot:heuristic" in stats, f"{run_id}: the anchor never played: {stats}"
    return s, hist, run


@check("league")
def check_league(workdir):
    import numpy as np
    from pufferroyale import metagame as mg

    s, hist, run = _league_run(workdir, "plain", [])
    sm, hist_m, _ = _league_run(workdir, "mmd", ["--mmd-coef", "0.1", "--mmd-ref-interval", "2"])
    kls = [h["losses"]["mmd_kl"] for h in hist_m]
    assert all(np.isfinite(kls)) and max(kls) > 0, f"MMD KL {kls}"
    # tournament: 3 bots + the plain league learner, meta-game Nash checked as an equilibrium
    out_json = os.path.join(workdir, "tournament.json")
    agents = ["bot:noop", "bot:random", "bot:heuristic", s["model"]]
    code, out = sh([PY, os.path.join("scripts", "tournament.py"), "--agents", *agents, "--decks", "hog26",
                    "--matches", "2", "--seed", "0", "--out", out_json, "--quiet"], timeout=900)
    assert code == 0, out[-1500:]
    t = json.load(open(out_json))
    P, x = np.asarray(t["payoff"]), np.asarray(t["nash"])
    assert np.allclose(P + P.T, 1.0) and abs(x.sum() - 1) < 1e-9 and (x >= -1e-12).all()
    A = P - 0.5
    assert (x @ A >= -1e-6).all() and (A @ x <= 1e-6).all(), "the reported mixture is not a Nash equilibrium"
    x2, v2 = mg.metagame_nash(P)
    assert abs(v2) < 1e-6
    assert P[2, 0] > 0.5 and t["elo"][2] > t["elo"][0], "heuristic must beat noop"
    ls = {k: v["p"] for k, v in s["stats"].items() if v["games"]}
    return (f"league {s['global_step']} steps/{len(hist)} epochs (p {', '.join(f'{k}={v:.2f}' for k, v in ls.items())}); "
            f"MMD KL max {max(kls):.1e}; tournament nash {np.round(x, 2).tolist()}")


def _last_json(text):
    """The last stdout JSON: a one-line summary (league_train.py, stages.py) or train.py's indented block."""
    text = text.strip()
    last = text.splitlines()[-1]
    if last.startswith("{"):
        return json.loads(last)
    return json.loads(text[text.rindex("\n{") + 1:])


@check("ops")
def check_ops(workdir):
    """SPEC §19.7 operations on tiny CPU runs: train.py --resume (absolute target, counters and the run
    continued), league_train.py --init-from (weights only: with lr 0 the final model equals the source),
    two stages.py rungs (both gates pass, the second starts from the first's model), and eval.py on a
    recurrent placement_grid-2 league snapshot (strict JSON rows with Wilson ci95)."""
    import torch

    d = os.path.join(workdir, "ops")
    os.makedirs(d, exist_ok=True)
    tiny = ["--num-envs", "4", "--bptt-horizon", "16", "--frame-skip", "50", "--seed", "0", "--data-dir", d]
    league = [PY, os.path.join("scripts", "league_train.py")]
    # train.py --resume
    common = ["--num-envs", "2", "--batch-size", "64", "--minibatch-size", "32", "--bptt-horizon", "16", "--quiet",
              "--env.frame-skip", "50", "--train.checkpoint-interval", "1"]
    code, out = sh([PY, os.path.join("scripts", "train.py"), "--total-timesteps", "128", "--data-dir", d,
                    "--shaping-anneal-frac", "0.25", "--env.reward-tower", "0.3", *common], timeout=600)
    assert code == 0, out[-1500:]
    rd = _last_json(out)["run_dir"]
    code, out = sh([PY, os.path.join("scripts", "train.py"), "--resume", rd, "--total-timesteps", "256", "--quiet"],
                   timeout=600)
    assert code == 0, out[-1500:]
    s = _last_json(out)
    assert s["global_step"] == 256 and s["epochs"] == 4 and s["run_dir"] == rd and s["resumed_from_step"] == 128, s
    st = torch.load(os.path.join(rd, "trainer_state.pt"), map_location="cpu", weights_only=False)
    cfg = json.load(open(os.path.join(rd, "config.json")))
    assert st["global_step"] == 256 and st.get("torch_rng") is not None
    assert cfg["env"]["shaping_step_offset"] == 128 // 4 and cfg["train"]["shaping_anneal_frac"] == 0.25, cfg["env"]
    # a recurrent placement_grid-2 league run (its snapshot is evaluated below), then --init-from it
    code, out = sh(league + ["--total-timesteps", "256", *tiny, "--run-id", "rec", "--rnn", "--env.placement-grid", "2",
                             "--anchors", "bot:random", "--self-play-frac", "0.5", "--anchor-frac", "0.5",
                             "--snapshot-interval", "2"], timeout=600)
    assert code == 0, out[-1500:]
    base = _last_json(out)
    code, out = sh(league + ["--total-timesteps", "128", *tiny, "--run-id", "warm", "--rnn", "--env.placement-grid", "2",
                             "--init-from", base["run_dir"], "--learning-rate", "0", "--anchors", "bot:noop",
                             "--snapshot-interval", "2"], timeout=600)
    assert code == 0, out[-1500:]
    warm = _last_json(out)
    a = torch.load(warm["model"], map_location="cpu", weights_only=True)
    b = torch.load(base["model"], map_location="cpu", weights_only=True)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "--init-from did not load the weights"
    code, out = sh(league + ["--total-timesteps", "128", *tiny, "--run-id", "bad", "--init-from", base["model"]],
                   timeout=600)
    assert code != 0 and "architecture mismatch" in out, "an --init-from architecture mismatch must be an error"
    # stages.py: two rungs, both gates pass; rung 1 starts from rung 0's final model
    code, out = sh([PY, os.path.join("scripts", "stages.py"), "--run-prefix", "lad", "--rungs",
                    "bot:noop:512:0.0;bot:random:512:0.0", "--gate-window", "1", "--snapshot-interval", "2", *tiny],
                   timeout=900)
    assert code == 0, out[-1500:]
    rep = json.load(open(os.path.join(d, "stages_lad.json")))
    r0, r1 = rep["rungs"]
    assert rep["passed"] and r0["passed"] and r1["passed"] and r1["init_from"] == r0["model"], rep
    # eval.py on the recurrent grid-2 snapshot, two decks (per deck + pooled rows)
    snap = sorted(glob.glob(os.path.join(base["run_dir"], "snap_*.pt")))[0]
    out_json = os.path.join(d, "eval.json")
    code, out = sh([PY, os.path.join("scripts", "eval.py"), "--checkpoint", snap, "--bots", "noop", "--seats", "0",
                    "--episodes", "1", "--decks", "hog26;giant", "--json", out_json], timeout=600)
    assert code == 0, out[-1500:]
    res = json.load(open(out_json))
    assert res["env_settings"]["placement_grid"] == 2, res["env_settings"]
    rows = res["results"]
    assert [r["deck"] for r in rows] == ["hog26", "giant", "pooled"], rows
    for r in rows:
        n = r["wins"] + r["draws"] + r["losses"]
        lo, hi = r["ci95"]
        assert n > 0 and abs(r["score"] - (r["wins"] + 0.5 * r["draws"]) / n) < 1e-9 and 0 <= lo <= r["score"] <= hi <= 1
    return (f"train resume 128->256; league --init-from (lr 0: identical weights, mismatch refused); stages 2/2 "
            f"rungs passed; eval {os.path.basename(snap)} (LSTM, g 2): "
            + " ".join(f"{r['deck']}:{r['wins']}-{r['draws']}-{r['losses']}" for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip ASan, smaller soak and train")
    ap.add_argument("--skip-train", action="store_true", help="skip the train / eval / league / ops checks")
    ap.add_argument("--skip-c", action="store_true", help="skip make test-c / asan")
    args = ap.parse_args()
    t0 = time.time()
    check_build()
    check_codegen()
    if not args.skip_c:
        check_c_tests()
        if not args.quick:
            check_asan()
    check_pytest()
    check_determinism()
    check_soak(8 if args.quick else 24)
    check_mask()
    check_zero_sum()
    check_no_leak()
    check_auto_reset()
    check_reward_v2()
    check_policy_heads()
    check_decks()
    check_grid()
    check_perf()
    check_llm()
    if not args.skip_train:
        with tempfile.TemporaryDirectory() as work:
            if check_train(8192 if args.quick else 32768, work):
                check_eval(work)
            check_league(work)
            check_ops(work)
    print()
    print(f"{'check':14s} {'result':6s} {'time':>7s}  detail")
    print("-" * 78)
    for name, ok, dt, detail in RESULTS:
        print(f"{name:14s} {'PASS' if ok else 'FAIL':6s} {dt:6.1f}s  {detail.splitlines()[0][:110] if detail else ''}")
    failed = [r for r in RESULTS if not r[1]]
    print("-" * 78)
    print(f"{'OVERALL':14s} {'PASS' if not failed else 'FAIL':6s} {time.time() - t0:6.1f}s  "
          f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, ok, dt, detail in failed:
        print(f"\n--- {name} ---\n{detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
