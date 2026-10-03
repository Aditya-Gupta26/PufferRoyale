"""Builder tests, SPEC §19.7 (v0.5): script operations -- train.py --resume / --init-from, league
--init-from / --early-stop-score, stages.py, eval.py (any checkpoint, outcomes, --decks, ci95),
best_response.py --init-from-target, transitivity, plot_history.py, wandb offline, config.json."""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
TINY = ["--num-envs", "4", "--bptt-horizon", "16", "--frame-skip", "50", "--seed", "0"]


def run(script, *args, env=None, check=True, timeout=600):
    p = subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True, timeout=timeout, env=env)
    if check:
        assert p.returncode == 0, f"{script} {args}:\n{(p.stdout + p.stderr)[-3000:]}"
    return p


def last_json(text):
    return json.loads([l for l in text.strip().splitlines() if l.strip()][-1])


def league_run(data_dir, run_id, *extra, total=512):
    p = run("league_train.py", "--total-timesteps", total, *TINY, "--data-dir", data_dir, "--run-id", run_id, *extra)
    return last_json(p.stdout), os.path.join(data_dir, "league", run_id)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("wp_tools"))
    base, base_dir = league_run(d, "base", "--anchors", "bot:noop", "--self-play-frac", "0.5", "--anchor-frac", "0.5",
                                "--snapshot-interval", "2", "--env.reward-tower", "0.3", "--train.reward-clip", "0",
                                "--shaping-anneal-frac", "0.5")
    rec, rec_dir = league_run(d, "rec_g2", "--rnn", "--env.placement-grid", "2", "--anchors", "bot:random",
                              "--self-play-frac", "0.5", "--anchor-frac", "0.5", "--snapshot-interval", "4")
    return {"dir": d, "base": base, "base_dir": base_dir, "rec": rec, "rec_dir": rec_dir}


def sd_of(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def same_weights(a, b):
    a, b = sd_of(a), sd_of(b)
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------------------------ config / help
def test_help_lists_the_new_flags():
    want = {"train.py": ("--resume", "--init-from", "--shaping-anneal-frac", "--wandb"),
            "league_train.py": ("--init-from", "--shaping-anneal-frac", "--early-stop-score", "--early-stop-window",
                                "--wandb", "--wandb-project", "--wandb-group", "--tag"),
            "best_response.py": ("--init-from-target", "--shaping-anneal-frac"),
            "eval.py": ("--decks", "--checkpoint"),
            "stages.py": ("--rungs", "--run-prefix", "--gate-window", "--data-dir"),
            "transitivity.py": ("--run", "--snapshots", "--matches", "--margin", "--out"),
            "plot_history.py": ("--out",)}
    for script, flags in want.items():
        text = run(script, "--help").stdout
        for f in flags:
            assert f in text, f"{script} --help lacks {f}"


def test_league_config_json_records_the_env_kwargs(runs):
    cfg = json.load(open(os.path.join(runs["base_dir"], "config.json")))
    assert {"policy", "rnn_name", "rnn", "env", "league"} <= set(cfg)
    env = cfg["env"]
    assert env["reward_gamma"] == 0.999 and env["placement_grid"] == 1 and env["shaping_step_offset"] == 0
    assert env["shaping_anneal_steps"] == int(np.ceil(0.5 * 512 / 4))
    assert cfg["league"]["shaping_anneal_frac"] == 0.5 and cfg["policy"]["head"] == "conditional"
    rcfg = json.load(open(os.path.join(runs["rec_dir"], "config.json")))
    assert rcfg["env"]["placement_grid"] == 2 and rcfg["rnn_name"] == "Recurrent"
    assert rcfg["policy"]["placement_grid"] == 2
    hist = [json.loads(l) for l in open(os.path.join(runs["base_dir"], "history.jsonl"))]
    for h in hist:
        assert {"entropy_card", "entropy_pos"} <= set(h["losses"])
        for spec, r in h["results"].items():
            assert len(h["outcomes"][spec]) == r["games"]
            assert abs(sum(0.5 * (x + 1) for x in h["outcomes"][spec]) / r["games"] - r["score"]) < 1e-9


# ------------------------------------------------------------------------------------ train.py
def test_train_resume_and_init_from(runs, tmp_path):
    d = str(tmp_path)
    common = ["--num-envs", "2", "--batch-size", "64", "--minibatch-size", "32", "--bptt-horizon", "16", "--quiet",
              "--env.frame-skip", "50", "--train.checkpoint-interval", "1"]
    s1 = last_json_block(run("train.py", "--total-timesteps", 128, "--data-dir", d, "--shaping-anneal-frac", 0.25,
                             *common).stdout)
    rd = s1["run_dir"]
    st = torch.load(os.path.join(rd, "trainer_state.pt"), weights_only=False)
    assert st["global_step"] == 128 and "torch_rng" in st
    cfg = json.load(open(os.path.join(rd, "config.json")))
    assert cfg["train"]["shaping_anneal_frac"] == 0.25 and cfg["env"]["shaping_anneal_steps"] == 8
    # --resume: absolute target, the run's settings reused (shaping frac included), counters continued
    s2 = last_json_block(run("train.py", "--resume", rd, "--total-timesteps", 256, "--quiet").stdout)
    assert s2["global_step"] == 256 and s2["epochs"] == 4 and s2["run_dir"] == rd
    cfg = json.load(open(os.path.join(rd, "config.json")))
    assert cfg["env"]["num_envs"] == 2 and cfg["env"]["frame_skip"] == 50 and cfg["train"]["shaping_anneal_frac"] == 0.25
    # SPEC §19.10.1: N = ceil(0.25 * 128 / 4) = 8 is fixed when the run is created (reused on --resume)
    assert cfg["env"]["shaping_step_offset"] == 128 // 4 and cfg["env"]["shaping_anneal_steps"] == 8
    assert cfg["train"]["shaping_anneal_steps"] == 8
    assert len(glob.glob(os.path.join(rd, "model_*.pt"))) == 4
    p = run("train.py", "--resume", rd, "--total-timesteps", 256, check=False)
    assert p.returncode == 0 and last_json(p.stdout)["nothing_to_do"]
    # --init-from: weights only (lr 0 keeps them), any checkpoint form; mismatch and --resume are errors
    model = runs["base"]["model"]
    s3 = last_json_block(run("train.py", "--total-timesteps", 64, "--data-dir", d, "--init-from", runs["base_dir"],
                             "--train.learning-rate", 0, *common).stdout)
    assert same_weights(glob.glob(os.path.join(s3["run_dir"], "model_*.pt"))[-1], model)
    p = run("train.py", "--total-timesteps", 64, "--data-dir", d, "--init-from", model, "--rnn", *common, check=False)
    assert p.returncode != 0 and "architecture mismatch" in p.stdout + p.stderr
    p = run("train.py", "--resume", rd, "--init-from", model, check=False)
    assert p.returncode != 0 and "--resume" in p.stdout + p.stderr


def last_json_block(text):
    """train.py prints an indented JSON summary at the end."""
    return json.loads(text[text.rindex("\n{") + 1:])


# ------------------------------------------------------------------------------------ league_train.py
def test_league_init_from(runs, tmp_path):
    d = str(tmp_path)
    s, rd = league_run(d, "warm", "--init-from", runs["base_dir"], "--learning-rate", 0, "--anchors", "bot:noop",
                       "--snapshot-interval", 2, total=128)
    assert same_weights(s["model"], runs["base"]["model"])
    assert json.load(open(os.path.join(rd, "config.json")))["league"]["init_from"] == runs["base"]["model"]
    p = run("league_train.py", "--total-timesteps", 128, *TINY, "--data-dir", d, "--run-id", "bad",
            "--init-from", runs["base"]["model"], "--policy.head", "flat", check=False)
    assert p.returncode != 0 and "architecture mismatch" in p.stdout + p.stderr
    p = run("league_train.py", "--resume", rd, "--init-from", runs["base"]["model"], check=False)
    assert p.returncode != 0 and "--resume" in p.stdout + p.stderr


def test_league_early_stop(tmp_path):
    d = str(tmp_path)
    s, rd = league_run(d, "es", "--anchors", "bot:noop", "--self-play-frac", "0", "--anchor-frac", "1",
                       "--snapshot-interval", "2", "--early-stop-score", "0", "--early-stop-window", "1", total=4096)
    assert s["early_stopped"] and s["global_step"] < 4096 and s["model"]
    hist = [json.loads(l) for l in open(os.path.join(rd, "history.jsonl"))]
    assert hist[-1].get("early_stopped") and hist[-1]["snapshot"]
    assert sum(len(h["outcomes"].get("bot:noop", [])) for h in hist) >= 1
    p = run("league_train.py", "--resume", rd, "--total-timesteps", 8192)
    assert last_json(p.stdout)["nothing_to_do"] and last_json(p.stdout)["early_stopped"]
    s2, _ = league_run(d, "no_es", "--anchors", "bot:noop", "--snapshot-interval", "2", "--early-stop-score", "1.01",
                       "--early-stop-window", "1", total=256)
    assert not s2["early_stopped"] and s2["global_step"] == 256
    for bad in (["--anchors", "none", "--early-stop-score", "0.5"],
                ["--anchors", "bot:noop", "--snapshot-interval", "0", "--early-stop-score", "0.5"]):
        p = run("league_train.py", "--total-timesteps", 128, *TINY, "--data-dir", d, "--run-id", "x", *bad, check=False)
        assert p.returncode != 0 and "early-stop" in p.stdout + p.stderr


# ------------------------------------------------------------------------------------ stages.py
def test_stages_two_rungs_and_a_failed_gate(tmp_path):
    d = str(tmp_path)
    p = run("stages.py", "--run-prefix", "lad", "--rungs", "bot:noop:512:0.0;bot:random:512:1.01", "--gate-window", "1",
            "--data-dir", d, "--num-envs", "4", "--bptt-horizon", "16", "--frame-skip", "50", "--snapshot-interval",
            "2", "--seed", "0", check=False)
    assert p.returncode == 1, p.stdout[-2000:] + p.stderr[-2000:]
    rep = json.load(open(os.path.join(d, "stages_lad.json")))
    assert rep == last_json(p.stdout) and not rep["passed"] and rep["failed_rung"] == 1
    r0, r1 = rep["rungs"]
    assert r0["passed"] and r0["matches"] == 1 and r0["early_stopped"] and r0["anchor"] == "bot:noop"
    assert r1["init_from"] == r0["model"] and not r1["passed"] and r1["reason"] == "score below threshold"
    hist = [json.loads(l) for l in open(os.path.join(r1["run_dir"], "history.jsonl"))]
    assert set().union(*[h["results"] for h in hist]) <= {"bot:random"}
    # a gate with fewer finished matches than the window fails as "insufficient matches"
    p = run("stages.py", "--run-prefix", "few", "--rungs", "bot:noop:128:0.0", "--gate-window", "100000",
            "--data-dir", d, "--num-envs", "4", "--bptt-horizon", "16", "--frame-skip", "50", "--snapshot-interval",
            "2", check=False)
    assert p.returncode == 1 and last_json(p.stdout)["rungs"][0]["reason"] == "insufficient matches"
    p = run("stages.py", "--run-prefix", "x", "--anchors", "bot:noop", check=False)
    assert p.returncode != 0


# ------------------------------------------------------------------------------------ eval.py
def test_eval_recurrent_snapshot_with_decks(runs, tmp_path):
    snap = sorted(glob.glob(os.path.join(runs["rec_dir"], "snap_*.pt")))[0]
    out = str(tmp_path / "e.json")
    run("eval.py", "--checkpoint", snap, "--bots", "noop", "--seats", 0, 1, "--episodes", 1, "--decks", "hog26;giant",
        "--json", out)
    res = json.load(open(out))
    assert res["env_settings"]["placement_grid"] == 2 and res["env_settings"]["frame_skip"] == 50
    rows = res["results"]
    assert [(r["deck"], r["seat"]) for r in rows] == [("hog26", 0), ("hog26", 1), ("giant", 0), ("giant", 1),
                                                      ("pooled", 0), ("pooled", 1)]
    for r in rows:
        n = r["wins"] + r["draws"] + r["losses"]
        assert r["score"] == pytest.approx((r["wins"] + 0.5 * r["draws"]) / n)
        lo, hi = r["ci95"]
        assert 0 <= lo <= r["score"] <= hi <= 1
    assert rows[4]["wins"] == rows[0]["wins"] + rows[2]["wins"]


def test_eval_results_come_from_outcomes_not_rewards(runs, tmp_path):
    """With heavy shaping a lost match can end on a positive reward; eval must still count it from
    the outcome. The scripted bot in the learner's seat (no policy) keeps this fast."""
    out = str(tmp_path / "b.json")
    run("eval.py", "--bot-policy", "heuristic", "--bots", "noop", "--seats", 0, "--episodes", 2, "--json", out,
        "--frame-skip", 50)
    r = json.load(open(out))["results"][0]
    assert r["wins"] == 2 and r["ci95"][1] == 1.0


# ------------------------------------------------------------------------------------ best_response.py
def test_best_response_init_from_target(runs, tmp_path):
    d = str(tmp_path)
    p = run("best_response.py", "--target", runs["rec_dir"], "--init-from-target", "--total-timesteps", 64,
            "--num-envs", 4, "--bptt-horizon", 16, "--learning-rate", 0, "--matches", 2, "--data-dir", d,
            "--shaping-anneal-frac", 0.5)
    res = last_json(p.stdout)
    assert res["init_from_target"] and res["env_settings"]["placement_grid"] == 2
    target = sorted(glob.glob(os.path.join(runs["rec_dir"], "model_*.pt")))[-1]
    assert same_weights(res["br_checkpoint"], target), "lr 0: the BR keeps the target's weights"
    cfg = json.load(open(os.path.join(os.path.dirname(res["br_checkpoint"]), "config.json")))
    assert {"policy", "rnn_name", "rnn", "env"} <= set(cfg) and cfg["rnn_name"] == "Recurrent"
    assert cfg["env"]["placement_grid"] == 2 and cfg["env"]["shaping_anneal_steps"] == 8


# ------------------------------------------------------------------------------------ transitivity / plots
def test_transitivity_function():
    from pufferroyale.metagame import transitivity
    P = np.array([[0.5, 0.2, 0.1], [0.8, 0.5, 0.3], [0.9, 0.7, 0.5]])          # later always better
    t = transitivity(P)
    assert t == {"later_beats_earlier": 1.0, "pairs": 3, "cyclic_triads": 0, "triads": 1}
    R = np.array([[0.5, 0.9, 0.1], [0.1, 0.5, 0.9], [0.9, 0.1, 0.5]])          # rock-paper-scissors
    t = transitivity(R)
    assert t["cyclic_triads"] == 1 and t["later_beats_earlier"] == pytest.approx(1 / 3)
    assert transitivity(R, margin=0.45)["cyclic_triads"] == 0
    assert transitivity(np.full((1, 1), 0.5))["later_beats_earlier"] is None


def test_transitivity_script(runs, tmp_path):
    out = str(tmp_path / "t.json")
    p = run("transitivity.py", "--run", runs["base_dir"], "--snapshots", 3, "--matches", 2, "--out", out)
    res = json.load(open(out))
    assert res == last_json(p.stdout) and len(res["snapshots"]) == 3 and res["epochs"] == sorted(res["epochs"])
    assert res["transitivity"]["pairs"] == 3 and res["transitivity"]["triads"] == 1
    P = np.asarray(res["payoff"])
    assert np.allclose(P + P.T, 1.0) and abs(sum(res["nash"]) - 1) < 1e-9 and res["deck"] == "hog26"


def test_plot_history(runs, tmp_path):
    out = str(tmp_path / "h.png")
    run("plot_history.py", runs["base_dir"], "--out", out)
    assert os.path.getsize(out) > 10_000
    block = tmp_path / "block"
    (block / "matplotlib").mkdir(parents=True)
    (block / "matplotlib" / "__init__.py").write_text("raise ImportError('blocked for the test')\n")
    env = dict(os.environ, PYTHONPATH=str(block))
    p = run("plot_history.py", runs["base_dir"], "--out", str(tmp_path / "x.png"), env=env, check=False)
    assert p.returncode != 0 and "needs matplotlib" in p.stderr


# ------------------------------------------------------------------------------------ wandb
def test_wandb_offline_and_not_imported_without_flag(tmp_path):
    pytest.importorskip("wandb")
    d = str(tmp_path)
    env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR=d, WANDB_SILENT="true")
    s, rd = None, None
    p = run("league_train.py", "--total-timesteps", 128, *TINY, "--data-dir", d, "--run-id", "wb", "--anchors",
            "bot:noop", "--snapshot-interval", 2, "--wandb", "--wandb-project", "pr-test", "--no-model-upload", env=env)
    state = json.load(open(os.path.join(d, "league", "wb", "league_state.json")))
    assert state.get("wandb_id") and "--wandb" not in state["ini_overrides"]
    assert glob.glob(os.path.join(d, "wandb", "offline-run-*"))
    driver = ("import runpy, sys; sys.argv = sys.argv[1:]; "
              "runpy.run_path(sys.argv[0], run_name='__main__'); print('WANDB_IMPORTED', 'wandb' in sys.modules)")
    p = subprocess.run([sys.executable, "-c", driver, os.path.join(SCRIPTS, "league_train.py"), "--total-timesteps",
                        "64", *TINY, "--data-dir", d, "--run-id", "nowb", "--anchors", "bot:noop"], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 and "WANDB_IMPORTED False" in p.stdout, p.stderr[-2000:]
    p = run("train.py", "--total-timesteps", 64, "--num-envs", 2, "--batch-size", 64, "--minibatch-size", 32,
            "--bptt-horizon", 16, "--quiet", "--data-dir", d, "--env.frame-skip", 50, "--wandb", "--wandb-project",
            "pr-test", "--no-model-upload", env=env)
    assert len(glob.glob(os.path.join(d, "wandb", "offline-run-*"))) == 2


# ------------------------------------------------------------------------------------ watch.py
def test_watch_plays_a_grid_checkpoint(runs):
    p = run("watch.py", "--checkpoint", runs["rec_dir"], "--p1", "random", "--fps", 0, "--no-clear", "--every", 1000,
            "--max-steps", 80, "--frame-skip", 20)
    assert "policy (team 0)" in p.stdout
