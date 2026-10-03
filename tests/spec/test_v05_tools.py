"""SPEC §19.7 scripts and tools (v0.5-G), §19.4 tool grids, §19.8 golden hashes.

1. reward plumbing: train.py / league_train.py / best_response.py pass reward_gamma = train.gamma and
   shaping_anneal_steps = ceil(F * total_timesteps / R) (R = learner rows per vector step: league / BR
   num_envs; train.py the vecenv's num_agents); on resume shaping_step_offset = global_step // R; F persisted
   with the run (league args); config.json records the env kwargs actually used.
2. train.py --resume RUN_DIR (absolute --total-timesteps), --init-from CKPT (weights only; new run).
3. league_train.py --init-from (new runs only; with --resume an error; architecture mismatch an error),
   --shaping-anneal-frac, --early-stop-score X / --early-stop-window N (summary "early_stopped": true).
4. stages.py --rungs anchor:budget:threshold;... --run-prefix --gate-window; writes
   <data-dir>/stages_<P>.json (run dir, steps, score, matches, passed per rung); stops at the first failed
   gate and exits non-zero.
5. wandb (--wandb, --wandb-project, --wandb-group, --tag); WANDB_MODE=offline works; without --wandb
   nothing is imported.
7. metagame.transitivity(P, margin=0.05) and scripts/transitivity.py.
8. eval.py: any checkpoint like league.load_policy (snapshot, recurrent); --decks; Wilson ci95.
9. best_response.py --init-from-target (learner starts from the target's weights; arch + grid from the target).
10. plot_history.py RUN_DIR [--out PNG] (clear error without matplotlib; `plots` extra).
11. hpc/pufferroyale.def installs pytest, matplotlib, wandb.  12. e2e_check.py gains the new checks.
"""
import glob
import itertools
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import helpers as H
import leaguekit as L
import v05kit as V

ROOT = H.ROOT
TINY = ["--batch-size", "256", "--minibatch-size", "256", "--bptt-horizon", "64"]


def run(args, timeout=600, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=e)


def ok(out):
    assert out.returncode == 0, (out.stdout + out.stderr)[-4000:]
    return out


def flags_of(script):
    out = run([script, "--help"], timeout=180)
    assert out.returncode == 0, f"{script} --help failed:\n{(out.stdout + out.stderr)[-2000:]}"
    return set(re.findall(r"(--[A-Za-z0-9][A-Za-z0-9_-]*)", out.stdout + out.stderr)), out.stdout + out.stderr


def epoch_of(path):
    return int(re.findall(r"(\d+)", os.path.basename(path))[-1])


def models(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=epoch_of)


def load_sd(path):
    import torch
    return torch.load(path, map_location="cpu")


def same_weights(a, b):
    import torch
    sa, sb = load_sd(a), load_sd(b)
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def config_values(run_dir, key, section="env"):
    """§19.9.9: config.json keeps the v0.4 top level (policy, rnn_name, rnn, env, + train / league);
    `env` holds the env kwargs actually used (reward_gamma, shaping_anneal_steps, shaping_step_offset,
    placement_grid)."""
    p = os.path.join(run_dir, "config.json")
    assert os.path.isfile(p), f"no config.json in {run_dir}"
    js = V.load_json(p)
    assert section in js and isinstance(js[section], dict), f"config.json lacks the {section!r} section: {sorted(js)}"
    assert key in js[section], f"config.json [{section}] does not record {key!r}: {sorted(js[section])}"
    return [js[section][key]]


def small_ckpt(path, g=1, head="conditional", hidden=256, **kw):
    import torch
    import pufferroyale
    import pufferroyale.torch as prt
    env = pufferroyale.Royale(num_envs=1, num_agents=2, placement_grid=g)
    torch.manual_seed(0)
    p = prt.Policy(env, head=head, hidden_size=hidden, **kw)
    obs, _ = env.reset(seed=0)
    with torch.no_grad():
        lg, _ = p.forward_eval(torch.as_tensor(obs), dict(lstm_h=None, lstm_c=None))
    assert lg.shape[-1] == V.N_ACT[g], f"setup: a grid-{g} policy has {V.N_ACT[g]} logits, got {lg.shape[-1]}"
    torch.save(p.state_dict(), str(path))
    env.close()
    return str(path)


# ==========================================================================================
# --help flags
# ==========================================================================================
NEW_FLAGS = {
    "scripts/train.py": ["--resume", "--init-from", "--shaping-anneal-frac"],
    "scripts/league_train.py": ["--init-from", "--shaping-anneal-frac", "--early-stop-score", "--early-stop-window"],
    "scripts/best_response.py": ["--init-from-target", "--shaping-anneal-frac"],
    "scripts/eval.py": ["--decks"],
    "scripts/stages.py": ["--rungs", "--run-prefix", "--gate-window"],
    "scripts/transitivity.py": ["--run", "--snapshots", "--matches", "--margin", "--device", "--seed", "--out"],
    "scripts/plot_history.py": ["--out"],
}


@pytest.mark.parametrize("script", sorted(NEW_FLAGS))
def test_new_cli_flags(pr, script):
    assert os.path.isfile(os.path.join(ROOT, script)), f"{script} missing (§19.7)"
    flags, _ = flags_of(script)
    missing = [f for f in NEW_FLAGS[script] if f not in flags]
    assert not missing, f"{script} --help lacks {missing}"


def test_e2e_check_lists_new_checks(pr):
    _, text = flags_of("scripts/e2e_check.py")
    for name in ("reward_v2", "policy_heads", "decks", "grid", "ops"):
        assert re.search(rf"\b{name}\b", text), f"e2e_check.py --help does not list the {name!r} check (§19.7.12)"


def test_plots_extra_and_hpc_def(pr):
    txt = ""
    for f in ("setup.py", "pyproject.toml"):
        p = os.path.join(ROOT, f)
        if os.path.isfile(p):
            txt += open(p).read()
    assert re.search(r"[\"']plots[\"']\s*[:=]\s*\[[^\]]*matplotlib", txt) or \
        re.search(r"^\s*plots\s*=\s*\[[^\]]*matplotlib", txt, re.M), "a `plots` extra with matplotlib (§19.7.10)"
    d = open(os.path.join(ROOT, "hpc", "pufferroyale.def")).read()
    for pkg in ("pytest", "matplotlib", "wandb"):
        assert re.search(rf"\b{pkg}\b", d), f"hpc/pufferroyale.def must install {pkg} (§19.7.11)"


# ==========================================================================================
# metagame.transitivity
# ==========================================================================================
def T(P, **kw):
    import pufferroyale.metagame as mg
    return mg.transitivity(P, **kw)


def ref_transitivity(P, margin):
    P = np.asarray(P, float)
    n = len(P)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    lbe = sum(P[j][i] > 0.5 for i, j in pairs) / len(pairs)
    cyc = 0
    for a, b, c in itertools.combinations(range(n), 3):
        th = 0.5 + margin
        if (P[a][b] > th and P[b][c] > th and P[c][a] > th) or (P[b][a] > th and P[c][b] > th and P[a][c] > th):
            cyc += 1
    return lbe, len(pairs), cyc, math.comb(n, 3)


def antisym(upper):
    n = len(upper)
    P = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            P[i][j] = upper[i][j]
            P[j][i] = 1 - upper[i][j]
    return P


def test_transitivity_fully_transitive(pr):
    n = 5
    P = antisym([[0.2] * n for _ in range(n)])            # P[i][j] = 0.2 for i < j: later beats earlier
    r = T(P)
    assert set(r) >= {"later_beats_earlier", "pairs", "cyclic_triads", "triads"}
    assert r["later_beats_earlier"] == 1.0 and r["cyclic_triads"] == 0
    assert r["pairs"] == 10 and r["triads"] == 10
    r = T(antisym([[0.8] * n for _ in range(n)]).tolist())
    assert r["later_beats_earlier"] == 0.0 and r["cyclic_triads"] == 0


def test_transitivity_rock_paper_scissors_and_margin(pr):
    P = np.array([[0.5, 0.1, 0.9], [0.9, 0.5, 0.1], [0.1, 0.9, 0.5]])   # 1 > 0, 2 > 1, 0 > 2
    r = T(P)
    assert abs(r["later_beats_earlier"] - 2 / 3) < 1e-12 and r["cyclic_triads"] == 1 and r["triads"] == 1
    weak = np.array([[0.5, 0.47, 0.53], [0.53, 0.5, 0.47], [0.47, 0.53, 0.5]])
    assert T(weak)["cyclic_triads"] == 0, "edges at 0.53 are not > 0.5 + 0.05"
    assert T(weak, margin=0.01)["cyclic_triads"] == 1
    tie = np.full((3, 3), 0.5)
    assert T(tie)["later_beats_earlier"] == 0.0 and T(tie)["cyclic_triads"] == 0, "0.5 is not > 0.5"


@pytest.mark.parametrize("seed", range(4))
def test_transitivity_random_tournaments(pr, seed):
    rng = np.random.default_rng(seed)
    n = 7
    P = antisym(rng.choice([0.1, 0.3, 0.52, 0.58, 0.7, 0.95], size=(n, n)))
    for margin in (0.0, 0.05, 0.1):
        lbe, pairs, cyc, tri = ref_transitivity(P, margin)
        r = T(P, margin=margin)
        assert abs(r["later_beats_earlier"] - lbe) < 1e-12 and r["pairs"] == pairs
        assert r["cyclic_triads"] == cyc and r["triads"] == tri, f"margin {margin}: {r} vs {(cyc, tri)}"


# ==========================================================================================
# train.py: plumbing, resume, init-from
# ==========================================================================================
@pytest.fixture(scope="module")
def train_runs(tmp_path_factory):
    data = tmp_path_factory.mktemp("train_v05")
    base = ["scripts/train.py", "--num-envs", "2", "--num-agents", "2", *TINY, "--seed", "1", "--device", "cpu",
            "--data-dir", str(data), "--quiet", "--train.gamma", "0.97", "--shaping-anneal-frac", "0.5",
            "--env.reward-tower", "0.3", "--train.checkpoint-interval", "1"]
    s1 = data / "s1.json"
    a = run([*base, "--total-timesteps", "1024", "--summary-json", str(s1)])
    dirs = sorted(d for d in glob.glob(str(data / "*")) if os.path.isdir(d))
    return dict(data=data, base=base, a=a, dirs=dirs, s1=s1)


def _train_dir(train_runs):
    ok(train_runs["a"])
    assert len(train_runs["dirs"]) == 1, f"expected one run dir: {train_runs['dirs']}"
    return train_runs["dirs"][0]


def test_train_reward_plumbing(pr, train_runs):
    """reward_gamma = train.gamma (0.97); shaping_anneal_steps = ceil(0.5 * 1024 / R), R = 4 rows."""
    d = _train_dir(train_runs)
    assert all(abs(float(v) - 0.97) < 1e-9 for v in config_values(d, "reward_gamma"))
    assert all(int(v) == math.ceil(0.5 * 1024 / 4) for v in config_values(d, "shaping_anneal_steps"))
    assert all(int(v) == 0 for v in config_values(d, "shaping_step_offset"))
    assert all(abs(float(v) - 0.3) < 1e-9 for v in config_values(d, "reward_tower"))


def test_train_resume_continues_and_sets_offset(pr, train_runs):
    d = _train_dir(train_runs)
    before = models(d)
    assert before, "no model_*.pt"
    out = ok(run([*train_runs["base"], "--resume", d, "--total-timesteps", "1024"]))
    assert models(d) == before, "resuming at the reached absolute target must not train"
    out = ok(run([*train_runs["base"], "--resume", d, "--total-timesteps", "2048"]))
    after = models(d)
    assert epoch_of(after[-1]) > epoch_of(before[-1]), f"resume must continue the epoch count: {before} -> {after}"
    assert epoch_of(after[-1]) == 8, f"absolute target 2048 at 256 per epoch -> epoch 8, got {after[-1]}"
    assert all(int(v) == 1024 // 4 for v in config_values(d, "shaping_step_offset")), \
        "on resume shaping_step_offset = global_step // R = 1024 // 4"
    assert all(int(v) == math.ceil(0.5 * 2048 / 4) for v in config_values(d, "shaping_anneal_steps"))
    assert len([x for x in glob.glob(str(train_runs["data"] / "*")) if os.path.isdir(x)]) == 1, "resume = same run"
    assert float(config_values(d, "shaping_anneal_frac", "train")[0]) == 0.5, "§19.9.9: F recorded under train"
    no_f = [a for i, a in enumerate(train_runs["base"])
            if a != "--shaping-anneal-frac" and train_runs["base"][i - 1] != "--shaping-anneal-frac"]
    ok(run([*no_f, "--resume", d, "--total-timesteps", "2560"]))
    assert int(config_values(d, "shaping_anneal_steps")[0]) == 256, \
        "§19.10.1: N is fixed per run (256 from the last F given); a resume without --shaping-anneal-frac reuses it"
    assert int(config_values(d, "shaping_step_offset")[0]) == 2048 // 4
    ok(run([*no_f, "--resume", d, "--total-timesteps", "3072", "--shaping-anneal-frac", "0.25"]))
    assert int(config_values(d, "shaping_anneal_steps")[0]) == math.ceil(0.25 * 3072 / 4), \
        "§19.10.1: N is recomputed from the new F and the new absolute total when F is given again"
    assert int(config_values(d, "shaping_step_offset")[0]) == 2560 // 4
    assert float(config_values(d, "shaping_anneal_frac", "train")[0]) == 0.25


def test_train_init_from_loads_weights_only(pr, train_runs, tmp_path):
    d = _train_dir(train_runs)
    ckpt = models(d)[0]
    data = tmp_path / "init"
    out = ok(run(["scripts/train.py", "--num-envs", "2", "--num-agents", "2", *TINY, "--seed", "2", "--device", "cpu",
                  "--data-dir", str(data), "--quiet", "--total-timesteps", "512", "--init-from", ckpt,
                  "--train.learning-rate", "0", "--env.reward-tower", "0.3"]))
    dirs = [x for x in glob.glob(str(data / "*")) if os.path.isdir(x)]
    assert len(dirs) == 1 and os.path.abspath(dirs[0]) != os.path.abspath(d), "--init-from starts a new run"
    final = models(dirs[0])[-1]
    assert same_weights(final, ckpt), "with learning rate 0 the new run's weights must equal --init-from's"


# ==========================================================================================
# league_train.py
# ==========================================================================================
LEAGUE = ["scripts/league_train.py", "--num-envs", "4", *TINY, "--frame-skip", "50", "--anchors", "bot:noop",
          "--self-play-frac", "0", "--anchor-frac", "1.0", "--snapshot-interval", "1", "--seed", "1",
          "--device", "cpu"]


def league_dir(out, data):
    """§15.7.12: the run directory is <data-dir>/league/<run_id>/; the last stdout line is the summary."""
    ok(out)
    dirs = sorted(x for x in glob.glob(os.path.join(str(data), "league", "*")) if os.path.isdir(x))
    assert len(dirs) == 1, f"expected one run dir under {data}/league: {dirs}"
    res = L.last_json(out.stdout)
    assert isinstance(res, dict), f"no JSON summary on stdout: {out.stdout[-1500:]}"
    return dirs[0], res


@pytest.fixture(scope="module")
def league_run(tmp_path_factory):
    data = tmp_path_factory.mktemp("league_v05")
    out = run([*LEAGUE, "--total-timesteps", "1536", "--data-dir", str(data), "--shaping-anneal-frac", "0.25",
               "--train.gamma", "0.98", "--env.reward-tower", "0.3"])
    return dict(out=out, data=data)


def test_league_reward_plumbing_and_persisted_frac(pr, league_run):
    d, res = league_dir(league_run["out"], league_run["data"])
    assert all(abs(float(v) - 0.98) < 1e-9 for v in config_values(d, "reward_gamma"))
    assert all(int(v) == math.ceil(0.25 * 1536 / 4) for v in config_values(d, "shaping_anneal_steps"))
    st = V.load_json(os.path.join(d, "league_state.json"))
    fr = [v for k in ("shaping_anneal_frac", "shaping-anneal-frac") for v in V.find_values(st, k)]
    assert fr and all(abs(float(v) - 0.25) < 1e-12 for v in fr), "F is persisted with the run (league args)"


def test_league_resume_sets_offset(pr, league_run):
    d, res = league_dir(league_run["out"], league_run["data"])
    out = ok(run(["scripts/league_train.py", "--resume", d, "--total-timesteps", "2048", "--device", "cpu"]))
    gs = 1536                                          # 6 epochs x 256 learner steps (batch 256 = 4 envs x 64)
    assert all(int(v) == gs // 4 for v in config_values(d, "shaping_step_offset")), \
        f"shaping_step_offset must be global_step // num_envs = {gs // 4}"
    assert all(int(v) == math.ceil(0.25 * 1536 / 4) for v in config_values(d, "shaping_anneal_steps")), \
        "§19.10.1: N is fixed per run (computed at creation) and reused on --resume"
    st = V.load_json(os.path.join(d, "league_state.json"))
    assert all(int(v) == 96 for v in V.find_values(st, "shaping_anneal_steps")) and \
        V.find_values(st, "shaping_anneal_steps"), "§19.10.1: N is recorded in the league args"
    ok(run(["scripts/league_train.py", "--resume", d, "--total-timesteps", "2560", "--device", "cpu",
            "--shaping-anneal-frac", "0.5"]))
    assert int(config_values(d, "shaping_anneal_steps")[0]) == math.ceil(0.5 * 2560 / 4), \
        "§19.10.1: recomputed from the new F and the new absolute total when F is given again"
    assert int(config_values(d, "shaping_step_offset")[0]) == 2048 // 4


def test_league_init_from(pr, league_run, tmp_path):
    d, res = league_dir(league_run["out"], league_run["data"])
    ckpt = models(d)[0]
    out = run([*LEAGUE, "--total-timesteps", "512", "--data-dir", str(tmp_path / "a"), "--init-from", ckpt,
               "--learning-rate", "0"])
    d2, res2 = league_dir(out, tmp_path / "a")
    assert os.path.abspath(d2) != os.path.abspath(d)
    assert same_weights(models(d2)[-1], ckpt), "lr 0: the new run's weights must equal --init-from's"
    bad = run([*LEAGUE, "--total-timesteps", "512", "--data-dir", str(tmp_path / "b"), "--init-from", ckpt,
               "--resume", d])
    assert bad.returncode != 0, "--init-from together with --resume must be an error"
    small = small_ckpt(tmp_path / "small.pt", hidden=32)
    bad = run([*LEAGUE, "--total-timesteps", "512", "--data-dir", str(tmp_path / "c"), "--init-from", small])
    assert bad.returncode != 0, "an architecture mismatch with --init-from must be an error"


def test_league_early_stop(pr, tmp_path):
    out = run([*LEAGUE, "--total-timesteps", "20000", "--data-dir", str(tmp_path / "es"), "--early-stop-score",
               "0.0", "--early-stop-window", "1"])
    d, res = league_dir(out, tmp_path / "es")
    assert res.get("early_stopped") is True, f"summary must report early_stopped: true: {res}"
    assert models(d), "an early-stopped run saves normally (model_<epoch>.pt)"
    assert epoch_of(models(d)[-1]) * 256 < 20000, "the run stopped before --total-timesteps"
    out = run([*LEAGUE, "--total-timesteps", "1024", "--data-dir", str(tmp_path / "ne"), "--early-stop-score",
               "1.01", "--early-stop-window", "1"])
    d, res = league_dir(out, tmp_path / "ne")
    assert not res.get("early_stopped", False) and epoch_of(models(d)[-1]) == 4


# ==========================================================================================
# stages.py
# ==========================================================================================
STAGE_FWD = ["--num-envs", "4", *TINY, "--frame-skip", "50", "--snapshot-interval", "1", "--device", "cpu",
             "--seed", "1"]                      # §19.9.10: --gate-window is passed on as --early-stop-window


def rung_rows(js):
    """Per-rung records (§19.7.4: run dir, steps, score, matches, passed) -- dicts carrying both `passed`
    and `steps` (an overall `passed` flag elsewhere in the file is not a rung)."""
    return [d for _, d in V.walk(js) if "passed" in d and "steps" in d]


def stages_json(data, prefix):
    p = os.path.join(str(data), f"stages_{prefix}.json")
    assert os.path.isfile(p), f"stages.py must write {p}"
    return V.load_json(p)


def test_stages_pass_chain(pr, tmp_path):
    out = run(["scripts/stages.py", "--rungs", "bot:noop:768:0.0;bot:random:768:0.0", "--run-prefix", "pp",
               "--gate-window", "1", "--data-dir", str(tmp_path), *STAGE_FWD], timeout=900)
    ok(out)
    js = stages_json(tmp_path, "pp")
    rung_dicts = rung_rows(js)
    passed = [r["passed"] for r in rung_dicts]
    assert passed == [True, True], f"both rungs pass (threshold 0.0): {passed}"
    for r in rung_dicts:
        for k in ("steps", "score", "matches"):
            assert k in r, f"stages json rung lacks {k!r}: {sorted(r)}"
        assert r["matches"] >= 1 and 0.0 <= r["score"] <= 1.0
    dirs = [v for r in rung_dicts for k, v in r.items() if isinstance(v, str) and os.path.isdir(v)]
    assert len(dirs) == 2, f"each rung records its run dir: {rung_dicts}"
    st1 = json.dumps(V.load_json(os.path.join(dirs[1], "league_state.json")))
    assert os.path.basename(os.path.normpath(dirs[0])) in st1, "rung 1 must --init-from rung 0's final model"


def test_stages_stop_at_failed_gate(pr, tmp_path):
    out = run(["scripts/stages.py", "--rungs", "bot:noop:512:1.01;bot:random:512:0.0", "--run-prefix", "ff",
               "--gate-window", "1", "--data-dir", str(tmp_path), *STAGE_FWD], timeout=900)
    assert out.returncode != 0, "a failed gate must exit non-zero"
    passed = [r["passed"] for r in rung_rows(stages_json(tmp_path, "ff"))]
    assert passed == [False], f"stops at the first failed gate (later rungs not run): {passed}"


# ==========================================================================================
# eval.py, transitivity.py, best_response.py, plot_history.py
# ==========================================================================================
@pytest.fixture(scope="module")
def rnn_league(tmp_path_factory):
    data = tmp_path_factory.mktemp("league_rnn")
    out = run([*LEAGUE, "--total-timesteps", "1024", "--data-dir", str(data), "--rnn"])
    return out, data


def check_ci95(js):
    scored = [d for _, d in V.walk(js) if "score" in d and isinstance(d["score"], (int, float))]
    assert scored, "eval JSON has no score entries"
    for d in scored:
        assert "ci95" in d, f"every score gets a Wilson 95% interval (ci95): {sorted(d)}"
        lo, hi = (float(x) for x in d["ci95"])
        s = float(d["score"])
        assert 0.0 <= lo <= s <= hi <= 1.0, (lo, s, hi)
        for k in ("wins", "draws", "losses"):                          # §19.9.6 row layout
            assert k in d, f"eval row lacks {k!r}: {sorted(d)}"
        n = int(d["wins"]) + int(d["draws"]) + int(d["losses"])
        assert n > 0
        assert abs(s - (int(d["wins"]) + 0.5 * int(d["draws"])) / n) < 1e-9, "score = (wins + 0.5 draws) / n"
        wlo, whi = V.wilson(s, n)
        assert abs(lo - wlo) < 1e-4 and abs(hi - whi) < 1e-4, f"ci95 {(lo, hi)} is not Wilson {(wlo, whi)} (n={n})"
    return scored


def test_eval_loads_recurrent_snapshot_and_reports_ci95(pr, rnn_league, tmp_path):
    d, _ = league_dir(*rnn_league)
    snaps = sorted(glob.glob(os.path.join(d, "snap_*.pt")), key=epoch_of)
    assert snaps
    js_path = tmp_path / "eval.json"
    ok(run(["scripts/eval.py", "--checkpoint", snaps[-1], "--bots", "noop", "--episodes", "2", "--seed", "1",
            "--json", str(js_path)]))
    check_ci95(V.load_json(js_path))


def test_eval_decks_per_deck_and_pooled(pr, tmp_path):
    js_path = tmp_path / "eval_decks.json"
    ok(run(["scripts/eval.py", "--bot-policy", "random", "--bots", "noop", "--episodes", "2", "--seed", "1",
            "--decks", "hog26;giant", "--deck0", "xbow", "--json", str(js_path)]))
    js = V.load_json(js_path)
    rows = check_ci95(js)
    decks = {r.get("deck") for r in rows}
    assert decks == {"hog26", "giant", "pooled"}, f"§19.9.6: rows carry deck in hog26/giant/pooled: {decks}"
    keys = {(str(r.get("opponent")), str(r.get("seat"))) for r in rows}
    for opp, seat in keys:
        mine = [r for r in rows if (str(r.get("opponent")), str(r.get("seat"))) == (opp, seat)]
        pooled = [r for r in mine if r["deck"] == "pooled"]
        assert len(pooled) == 1, f"one pooled row per opponent and seat ({opp}, {seat})"
        for k in ("wins", "draws", "losses"):
            assert pooled[0][k] == sum(r[k] for r in mine if r["deck"] != "pooled"), f"pooled {k}"


def test_eval_honours_placement_grid(pr, tmp_path):
    data = tmp_path / "g2"
    ok(run(["scripts/train.py", "--num-envs", "2", "--num-agents", "2", *TINY, "--seed", "1", "--device", "cpu",
            "--data-dir", str(data), "--quiet", "--total-timesteps", "256", "--env.placement-grid", "2"]))
    d = [x for x in glob.glob(str(data / "*")) if os.path.isdir(x)][0]
    assert all(int(v) == 2 for v in config_values(d, "placement_grid")), "placement_grid recorded in config.json"
    js_path = tmp_path / "e.json"
    ok(run(["scripts/eval.py", "--checkpoint", d, "--bots", "noop", "--episodes", "2", "--json", str(js_path)]))
    check_ci95(V.load_json(js_path))


def test_transitivity_script(pr, league_run, tmp_path):
    d, _ = league_dir(league_run["out"], league_run["data"])
    outp = tmp_path / "tr.json"
    out = ok(run(["scripts/transitivity.py", "--run", d, "--snapshots", "3", "--matches", "2", "--margin", "0.05",
                  "--device", "cpu", "--seed", "1", "--out", str(outp)], timeout=900))
    txt = outp.read_text()
    js = json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    for k in ("later_beats_earlier", "pairs", "cyclic_triads", "triads"):
        assert V.find_values(js, k), f"transitivity JSON lacks {k}"
    for k in ("payoff", "elo", "nash"):
        assert V.find_values(js, k), f"transitivity JSON lacks {k}"
    P = np.asarray(V.find_values(js, "payoff")[0], float)
    assert P.shape == (3, 3)
    assert V.find_values(js, "pairs")[0] == 3 and V.find_values(js, "triads")[0] == 1


def test_best_response_init_from_target(pr, tmp_path):
    target = small_ckpt(tmp_path / "target.pt", g=2, head="flat", hidden=64)
    data = tmp_path / "br"
    out = ok(run(["scripts/best_response.py", "--target", target, "--total-timesteps", "256", "--matches", "2",
                  "--num-envs", "4", "--bptt-horizon", "64", "--learning-rate", "0", "--init-from-target",
                  "--data-dir", str(data), "--device", "cpu", "--seed", "1"]))
    res = V.strict_json_line(out.stdout)
    assert 0.0 <= res["br_score"] <= 1.0
    brs = glob.glob(str(data / "**" / "br.pt"), recursive=True)
    assert brs, "BR weights kept under <data-dir>/best_response/<run_id>/br.pt"
    assert same_weights(brs[0], target), "--init-from-target with lr 0: the BR's weights are the target's"
    cfgp = os.path.join(os.path.dirname(brs[0]), "config.json")              # §19.9.9
    assert os.path.isfile(cfgp), "best_response.py --data-dir writes config.json next to br.pt"
    import configparser
    cp = configparser.ConfigParser()
    cp.read(os.path.join(ROOT, "pufferroyale", "config", "royale.ini"))
    js = V.load_json(cfgp)
    assert abs(float(js["env"]["reward_gamma"]) - float(cp["train"]["gamma"])) < 1e-12, \
        "§19.7.1: best_response.py passes reward_gamma = train.gamma"
    assert int(js["env"]["placement_grid"]) == 2, "the grid comes from the target"


def test_plot_history(pr, league_run, tmp_path):
    d, _ = league_dir(league_run["out"], league_run["data"])
    png = tmp_path / "h.png"
    out = run(["scripts/plot_history.py", d, "--out", str(png)])
    try:
        import matplotlib  # noqa: F401
        have = True
    except ImportError:
        have = False
    if have:
        ok(out)
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    else:
        assert out.returncode != 0, "without matplotlib plot_history.py must fail clearly"
        msg = out.stdout + out.stderr
        assert "matplotlib" in msg and "Traceback" not in msg, f"clear error expected, got: {msg[-800:]}"


# ==========================================================================================
# wandb
# ==========================================================================================
def test_no_wandb_import_without_flag(pr, tmp_path):
    fake = tmp_path / "fake"
    (fake / "wandb").mkdir(parents=True)
    marker = tmp_path / "imported"
    (fake / "wandb" / "__init__.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n")
    env = {"PYTHONPATH": str(fake) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    ok(run(["scripts/train.py", "--num-envs", "2", "--num-agents", "2", *TINY, "--total-timesteps", "256", "--quiet",
            "--data-dir", str(tmp_path / "d1")], env=env))
    ok(run([*LEAGUE, "--total-timesteps", "256", "--data-dir", str(tmp_path / "d2")], env=env))
    assert not marker.exists(), "without --wandb nothing may import wandb (§19.7.5)"


def _wandb_available():
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("script", ["train", "league"])
def test_wandb_offline_smoke(pr, tmp_path, script):
    if not _wandb_available():
        pytest.skip("wandb not installed (optional dependency; the offline smoke needs it)")
    wd = tmp_path / "wandb_dir"
    wd.mkdir()
    env = {"WANDB_MODE": "offline", "WANDB_DIR": str(wd), "WANDB_SILENT": "true", "WANDB_API_KEY": ""}
    common = ["--wandb", "--wandb-project", "pr-test", "--wandb-group", "g", "--tag", "t"]
    if script == "train":
        args = ["scripts/train.py", "--num-envs", "2", "--num-agents", "2", *TINY, "--total-timesteps", "512",
                "--quiet", "--data-dir", str(tmp_path / "d"), *common]
    else:
        args = [*LEAGUE, "--total-timesteps", "512", "--data-dir", str(tmp_path / "d"), *common]
    ok(run(args, env=env))
    runs = glob.glob(str(wd / "**" / "offline-run-*"), recursive=True)
    assert runs, "WANDB_MODE=offline must produce an offline run under WANDB_DIR"


# ==========================================================================================
# §19.8 golden hashes, determinism with every new feature on
# ==========================================================================================
def test_golden_hashes_unchanged(pr):
    out = run(["scripts/golden_hashes.py"], timeout=900)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]


def test_env_with_all_new_features_is_deterministic(pr):
    import envkit as E

    def trace():
        env = E.make(num_envs=2, num_agents=2, seed=5, placement_grid=2, deck_pool="hog26:2;golem", random_deck_frac=0.3,
                     heldout_decks="xbow", deck_draw="independent", reward_tower=0.3, reward_crown=0.2,
                     reward_elixir=0.01, reward_play=0.01, reward_gamma=0.99, shaping_anneal_steps=300, frame_skip=20,
                     log_interval=1)
        obs, _ = env.reset(seed=5)
        m = E.R()
        rng = np.random.default_rng(0)
        out = []
        for _ in range(300):
            acts = []
            for row in obs:
                leg = np.nonzero(np.asarray(m.action_mask(row, grid=2)))[0]
                acts.append(int(rng.choice(leg[1:])) if len(leg) > 1 and rng.random() < 0.4 else 0)
            obs, rew, term, _, infos = E.step(env, acts)
            out.append((obs.copy(), rew.copy(), term.copy(), json.dumps(E.logs_in(infos), sort_keys=True, default=str)))
        out.append(json.dumps({k: np.asarray(v).tolist() for k, v in env.deck_counts().items()}))
        env.close()
        return out
    a, b = trace(), trace()
    for x, y in zip(a[:-1], b[:-1]):
        assert np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2]) and x[3] == y[3]
    assert a[-1] == b[-1]


@pytest.mark.parametrize("g", [2, 4])
def test_metagame_plays_grid_checkpoints(pr, tmp_path, g):
    """§19.4: metagame.play_match / deck_metagame play a checkpoint's actions through its own grid
    (the grid is inferred from the checkpoint, §19.6)."""
    import pufferroyale.metagame as mg
    ck = small_ckpt(tmp_path / f"g{g}.pt", g=g, hidden=64)
    for seed in range(2):
        r = mg.play_match(ck, "bot:noop", "hog26", "hog26", seed)
        assert r in (-1, 0, 1)
    r = mg.play_match("bot:random", ck, "giant", "bait", 3)
    assert r in (-1, 0, 1)



# ==========================================================================================
# §19.10 audit amendments (v0.5-G.3)
# ==========================================================================================
def test_anneal_length_exact_rational(pr, tmp_path):
    """§19.10.1: N = ceil(F*T/R) in exact rational arithmetic on the decimal F: F = 0.07, T = 400,
    R = 4 gives exactly 7 (floating point gives 0.07*400/4 = 7.000000000000001 -> 8)."""
    assert math.ceil(0.07 * 400 / 4) == 8, "tester premise: the float computation is off by one"
    out = ok(run([*LEAGUE, "--total-timesteps", "400", "--data-dir", str(tmp_path / "lg"), "--shaping-anneal-frac",
                  "0.07"]))
    d, _ = league_dir(out, tmp_path / "lg")
    assert int(config_values(d, "shaping_anneal_steps")[0]) == 7
    st = V.load_json(os.path.join(d, "league_state.json"))
    assert [int(v) for v in V.find_values(st, "shaping_anneal_steps")][:1] == [7]
    data = tmp_path / "tr"
    ok(run(["scripts/train.py", "--num-envs", "2", "--num-agents", "2", *TINY, "--seed", "1", "--device", "cpu",
            "--data-dir", str(data), "--quiet", "--total-timesteps", "400", "--shaping-anneal-frac", "0.07"]))
    dt = [x for x in glob.glob(str(data / "*")) if os.path.isdir(x)][0]
    assert int(config_values(dt, "shaping_anneal_steps")[0]) == 7


def clip_warnings(err):
    return [l for l in err.splitlines() if re.search(r"reward[-_ ]clip", l, re.I)]


def test_shaping_with_clip_warns_once(pr, train_runs, league_run, tmp_path):
    """§19.10.2: shaping weight > 0 and reward_clip > 0 -> exactly one stderr warning (not an error);
    none with --train.reward-clip 0 or without shaping."""
    assert len(clip_warnings(train_runs["a"].stderr)) == 1, train_runs["a"].stderr[-2000:]
    assert len(clip_warnings(league_run["out"].stderr)) == 1, league_run["out"].stderr[-2000:]
    base = ["scripts/train.py", "--num-envs", "2", "--num-agents", "2", *TINY, "--seed", "1", "--device", "cpu",
            "--quiet", "--total-timesteps", "256"]
    o = ok(run([*base, "--data-dir", str(tmp_path / "a"), "--env.reward-tower", "0.3", "--train.reward-clip", "0"]))
    assert not clip_warnings(o.stderr), o.stderr[-1500:]
    o = ok(run([*base, "--data-dir", str(tmp_path / "b")]))
    assert not clip_warnings(o.stderr), o.stderr[-1500:]
    o = ok(run([*LEAGUE, "--total-timesteps", "256", "--data-dir", str(tmp_path / "c"), "--env.reward-crown", "0.2",
                "--train.reward-clip", "0"]))
    assert not clip_warnings(o.stderr), o.stderr[-1500:]


def _history(d):
    with open(os.path.join(d, "history.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def test_league_resume_truncates_history(pr, tmp_path):
    """§19.10.3: on --resume, history.jsonl is first cut back to the records with epoch <= the saved
    epoch (records written after the last save are dropped)."""
    data = tmp_path / "h"
    out = ok(run([*LEAGUE, "--total-timesteps", "512", "--data-dir", str(data)]))
    d, _ = league_dir(out, data)
    recs = _history(d)
    saved = max(int(e) for e in V.find_values(V.load_json(os.path.join(d, "league_state.json")), "epoch"))
    assert recs and max(r["epoch"] for r in recs) == saved
    with open(os.path.join(d, "history.jsonl"), "a") as f:            # records from after the last save
        for k in (1, 2):
            f.write(json.dumps(dict(recs[-1], epoch=saved + k, tester_fake=True)) + "\n")
    ok(run(["scripts/league_train.py", "--resume", d, "--total-timesteps", "1024", "--device", "cpu"]))
    recs2 = _history(d)
    assert not any(r.get("tester_fake") for r in recs2), "records after the saved epoch must be dropped"
    ep = [r["epoch"] for r in recs2]
    assert ep == sorted(set(ep)) and ep[:len(recs)] == [r["epoch"] for r in recs] and max(ep) == 4, ep


def test_new_run_moves_stale_history_aside(pr, tmp_path):
    """§19.10.3: a new run in a directory holding history.jsonl but no league_state.json moves the old
    file to history.jsonl.stale-<k> (k = 1, 2, ...) before writing."""
    data = tmp_path / "s"
    rd = data / "league" / "fixed_id"
    rd.mkdir(parents=True)
    (rd / "history.jsonl").write_text('{"epoch": 99, "old": 1}\n')
    (rd / "history.jsonl.stale-1").write_text('{"epoch": 98, "older": 1}\n')
    ok(run([*LEAGUE, "--total-timesteps", "256", "--data-dir", str(data), "--run-id", "fixed_id"]))
    assert (rd / "history.jsonl.stale-1").read_text() == '{"epoch": 98, "older": 1}\n', "stale-1 untouched"
    assert (rd / "history.jsonl.stale-2").read_text() == '{"epoch": 99, "old": 1}\n', "old history -> stale-2"
    recs = _history(str(rd))
    assert recs and not any("old" in r for r in recs) and recs[0]["epoch"] == 1


def test_stages_gate_needs_window_outcomes(pr, tmp_path):
    """§19.9.10 / §19.10.4: the gate uses the persisted window; fewer than N outcomes vs the anchor
    fails even with threshold 0."""
    out = run(["scripts/stages.py", "--rungs", "bot:noop:512:0.0", "--run-prefix", "ww", "--gate-window", "1000",
               "--data-dir", str(tmp_path), *STAGE_FWD], timeout=900)
    assert out.returncode != 0, "a gate with fewer than N outcomes fails"
    rows = rung_rows(stages_json(tmp_path, "ww"))
    assert [r["passed"] for r in rows] == [False] and rows[0]["matches"] < 1000


def test_stages_relative_paths_are_the_callers(pr, tmp_path):
    """§19.10.4: rungs run in the caller's working directory: a relative file: deck set forwarded from
    another cwd works; §19.10.5: the rung's config.json env records the expanded decks."""
    cwd = tmp_path / "caller"
    cwd.mkdir()
    (cwd / "decks.json").write_text(json.dumps([{"deck": [19, 2, 16, 4, 11, 9, 14, 10], "weight": 2}]))
    e = dict(os.environ)
    e["PYTHONPATH"] = ROOT + os.pathsep + e.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "stages.py"), "--rungs", "bot:noop:512:0.0",
                          "--run-prefix", "rel", "--gate-window", "1", "--data-dir", str(tmp_path / "data"),
                          *STAGE_FWD, "--env.deck-pool", "file:decks.json"],
                         cwd=str(cwd), capture_output=True, text=True, timeout=900, env=e)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    rows = rung_rows(stages_json(tmp_path / "data", "rel"))
    rd = next(v for r in rows for v in r.values() if isinstance(v, str) and os.path.isdir(v))
    js = V.load_json(os.path.join(rd, "config.json"))
    got = [[list(map(int, c)), float(w)] for c, w in js["env"]["deck_pool_decks"]]
    assert got == [[[2, 4, 9, 10, 11, 14, 16, 19], 2.0]], f"deck_pool_decks {got} (explicit list: installed ascending)"


def test_config_records_expanded_decks(pr, tmp_path):
    """§19.10.5: config.json env records deck_pool_decks / heldout_decks_decks as [cards, weight] with
    the cards in installed order (preset name: the preset's order; everything else ascending)."""
    import pufferroyale as p
    import pufferroyale.decks as dk
    data = tmp_path / "d"
    ok(run(["scripts/train.py", "--num-envs", "2", "--num-agents", "2", *TINY, "--seed", "1", "--device", "cpu",
            "--data-dir", str(data), "--quiet", "--total-timesteps", "256", "--env.deck-pool", "hog26:2;random:1:3",
            "--env.heldout-decks", "xbow", "--env.random-deck-frac", "0.5"]))
    d = [x for x in glob.glob(str(data / "*")) if os.path.isdir(x)][0]
    env = V.load_json(os.path.join(d, "config.json"))["env"]
    pool = [[list(map(int, c)), float(w)] for c, w in env["deck_pool_decks"]]
    assert pool == [[[int(c) for c in p.DECKS["hog26"]], 2.0], [list(dk.random_decks(1, 3)[0]), 1.0]], pool
    held = [list(map(int, c)) for c, w in env["heldout_decks_decks"]]
    assert held == [sorted(int(c) for c in p.DECKS["xbow"])], f"§19.10.5 (ruling v0.5-G.4): held-out cards ascending: {held}"


@pytest.mark.parametrize("w", ["2e9", "inf", "nan", "1e10"])
def test_deck_weight_must_be_finite_and_at_most_1e9(pr, w):
    """§19.10.5: each deck weight must be finite and <= 1e9, else ValueError."""
    import pufferroyale.decks as dk
    with pytest.raises(ValueError):
        dk.parse_deck_set(f"hog26:{w}")
    with pytest.raises(ValueError):
        dk.parse_deck_set([{"deck": "giant", "weight": float(w)}])
    assert float(dk.parse_deck_set("hog26:1e9")[0][1]) == 1e9


@pytest.mark.parametrize("g", [1, 2, 4])
def test_bf16_autocast_conditional_head(pr, g):
    """§19.10.6: under bf16 autocast (CPU) the conditional head has no NaN/inf in its outputs or
    gradients; masked logits stay exactly at finfo.min of the logits' dtype, legal ones finite."""
    import torch
    import pufferlib.pytorch
    import pufferroyale
    import pufferroyale.torch as prt
    import envkit as E
    m = E.R()
    env = pufferroyale.Royale(num_envs=3, num_agents=2, seed=0, placement_grid=g)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    rows = [obs.copy()]
    for _ in range(14):
        obs, *_ = env.step(np.array([E.random_bot_action(rng, r, 0.5) for r in obs], np.int32))
        rows.append(obs.copy())
    batch = np.concatenate(rows)
    torch.manual_seed(0)
    policy = prt.Policy(env, head="conditional")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, value = policy(torch.as_tensor(batch), dict(action=None, lstm_h=None, lstm_c=None))
        a, logp, ent = pufferlib.pytorch.sample_logits(logits)
        loss = -logp.float().mean() - 0.01 * ent.float().mean() + value.float().pow(2).mean()
    cm = torch.as_tensor(np.asarray(m.action_mask(batch, grid=g)))
    assert logits.shape[-1] == V.N_ACT[g]
    assert torch.all(logits[~cm] == torch.finfo(logits.dtype).min) and torch.isfinite(logits[cm]).all()
    assert torch.isfinite(value).all() and torch.isfinite(logp).all() and torch.isfinite(ent).all()
    loss.backward()
    for n, p_ in policy.named_parameters():
        if p_.grad is not None:
            assert torch.isfinite(p_.grad).all(), f"non-finite gradient in {n} under bf16 autocast"
    env.close()


RESUME_CASES = [
    # (script, extra flags, total, expected) -- the league run is at 2560, the train run at 3072 here
    ("league", ["--env.placement-grid", "2"], 2816, "exit"),
    ("league", ["--rnn"], 2816, "ignored"),
    ("league", ["--policy.hidden-size", "64"], 3072, "exit"),
    ("train", ["--env.placement-grid", "4"], 3328, "exit"),
    ("train", ["--rnn"], 3328, "ignored"),
    ("train", ["--policy.hidden-size", "64"], 3584, "ignored"),
]
IGNORED_NAME = {"--rnn": r"rnn", "--policy.hidden-size": r"policy\.hidden[-_]size"}


@pytest.mark.parametrize("script,extra,total,expect", RESUME_CASES,
                         ids=["league-grid", "league-rnn", "league-hidden", "train-grid", "train-rnn", "train-hidden"])
def test_resume_with_different_architecture(pr, league_run, train_runs, script, extra, total, expect):
    """§19.10.8 (ruling v0.5-G.4): on --resume the run's architecture wins: --rnn and --policy.* / --rnn.*
    given again are ignored with one stderr note naming them and the run continues (exit 0) --
    except that league_train.py keeps --policy.* overrides (v0.4), so a changed size there, like any
    effective setting that does not fit the saved weights (a different placement_grid), exits with a
    clear message (SystemExit), not a traceback."""
    if script == "league":
        d, _ = league_dir(league_run["out"], league_run["data"])
        out = run(["scripts/league_train.py", "--resume", d, "--total-timesteps", str(total), "--device", "cpu",
                   *extra])
    else:
        out = run([*train_runs["base"], "--resume", _train_dir(train_runs), "--total-timesteps", str(total), *extra])
    assert "Traceback" not in out.stderr, f"{script} --resume {extra}: traceback:\n{out.stderr[-1500:]}"
    if expect == "exit":
        assert out.returncode != 0, f"{script} --resume with {extra} does not fit the saved weights: must exit"
        assert out.stderr.strip(), "the exit must carry a clear message"
    else:
        assert out.returncode == 0, f"{script} --resume with {extra}: ignored, the run continues\n{out.stderr[-1500:]}"
        notes = [l for l in out.stderr.splitlines() if re.search(r"ignored", l, re.I)]
        assert len(notes) == 1, f"exactly one stderr note about ignored flags expected: {notes}"
        assert re.search(IGNORED_NAME[extra[0]], notes[0], re.I), f"the note must name {extra[0]}: {notes[0]!r}"
