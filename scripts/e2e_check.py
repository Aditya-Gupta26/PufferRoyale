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
   perf        engine >= 20,000 ticks/s and env >= 2,000 steps/s (SPEC §11)
   train       scripts/train.py smoke run (no NaN)   eval   scripts/eval.py on its checkpoint
   llm         SPEC §17: scripts/llm_match.py, mock_first_legal (text-only) vs the heuristic bot, 2 matches,
               no network; anthropic: models are refused without --allow-network
   league      SPEC §15: a tiny league run (heuristic anchor + self-play + PFSP snapshots), a tiny
               MMD league run, and a tournament of the 3 bots + the league learner with a
               meta-game Nash solve (equilibrium verified)

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip ASan, smaller soak and train")
    ap.add_argument("--skip-train", action="store_true", help="skip the train / eval / league checks")
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
    check_perf()
    check_llm()
    if not args.skip_train:
        with tempfile.TemporaryDirectory() as work:
            if check_train(8192 if args.quick else 32768, work):
                check_eval(work)
            check_league(work)
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
