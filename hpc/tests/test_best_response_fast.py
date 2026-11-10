"""Checks of the D90 throughput flags in scripts/best_response.py (E1 probes, 2026-10-07). Cluster-side (not
under tests/spec): run with `python -m pytest -q hpc/tests`.

CPU: uninstall_learner_graph restores the class forward_eval; a best response trained with --env-threads is
bit-identical (weights and evaluation W/D/L) to one trained without. CUDA: a best response with all three
flags trains, removes its learner graph and plays its evaluation matches.
--wandb (D98, CPU, WANDB_MODE=offline): the same weights and W/D/L as without it, and an offline run that
holds the per-epoch training records and the evaluation summary.
"""
import glob
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pufferroyale import royale as R               # noqa: E402

torch = pytest.importorskip("torch")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _target(path, seed=7):
    from pufferroyale.league import _obs_stub
    from pufferroyale.torch import Policy
    torch.manual_seed(seed)
    pol = Policy(_obs_stub(R.n_actions(1)), placement_grid=1)
    torch.save({k: v.detach().cpu() for k, v in pol.state_dict().items()}, path)
    return path


def _br(tmp, name, target, extra, device="cpu", num_envs=8, steps=1024, matches=2, env=None):
    d = os.path.join(tmp, name)
    out = os.path.join(tmp, f"{name}.json")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "best_response.py"), "--target", target,
           "--device", device, "--num-envs", str(num_envs), "--total-timesteps", str(steps),
           "--matches", str(matches), "--seed", "3", "--data-dir", d, "--out", out, "--init-from-target",
           "--train.torch-deterministic="] + extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800,
                       env=None if env is None else {**os.environ, **env})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    res = json.load(open(out))
    return res, torch.load(res["br_checkpoint"], map_location="cpu")


def test_uninstall_learner_graph_restores_class_method():
    from pufferroyale.fast_league import install_learner_graph, learner_graph, uninstall_learner_graph
    from pufferroyale.league import _obs_stub
    from pufferroyale.torch import Policy
    pol = Policy(_obs_stub(R.n_actions(1)), placement_grid=1)
    assert install_learner_graph(pol)
    assert "forward_eval" in pol.__dict__ and learner_graph(pol) is not None
    assert uninstall_learner_graph(pol)
    assert "forward_eval" not in pol.__dict__ and learner_graph(pol) is None
    assert type(pol).forward_eval is pol.forward_eval.__func__
    assert not uninstall_learner_graph(pol)                  # idempotent


def test_br_env_threads_bit_identical(tmp_path):
    target = _target(str(tmp_path / "target.pt"))
    base, w0 = _br(str(tmp_path), "plain", target, [])
    thr, w1 = _br(str(tmp_path), "threads", target, ["--env-threads", "4"])
    assert base["total_timesteps"] == thr["total_timesteps"] > 0
    for k in ("wins", "draws", "losses", "train_games_vs_target", "train_score_vs_target"):
        assert base[k] == thr[k], k
    assert w0.keys() == w1.keys()
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k


@needs_cuda
def test_br_fast_flags_train_and_evaluate(tmp_path):
    target = _target(str(tmp_path / "target.pt"))
    res, w = _br(str(tmp_path), "fast", target, ["--env-threads", "4", "--fast-opponents", "--cuda-graphs"],
                 device="cuda", num_envs=64, steps=64 * 64 * 3, matches=4)
    assert res["matches"] == 4 and res["wins"] + res["draws"] + res["losses"] == 4
    assert res["total_timesteps"] >= 64 * 64 * 3 and res["train_games_vs_target"] >= 0
    assert all(torch.isfinite(v).all() for v in w.values() if v.is_floating_point())


def test_br_wandb_offline_same_result(tmp_path):
    pytest.importorskip("wandb")
    target = _target(str(tmp_path / "target.pt"))
    wd = tmp_path / "wandb_dir"
    wd.mkdir()
    env = {"WANDB_MODE": "offline", "WANDB_DIR": str(wd), "WANDB_SILENT": "true", "WANDB_API_KEY": ""}
    base, w0 = _br(str(tmp_path), "plain", target, [], steps=2048, matches=4)
    logged, w1 = _br(str(tmp_path), "wandb", target, ["--wandb", "--wandb-project", "pr-test", "--wandb-group", "g",
                                                      "--tag", "t", "--no-model-upload"], steps=2048, matches=4, env=env)
    for k in ("total_timesteps", "wins", "draws", "losses", "train_games_vs_target", "train_score_vs_target"):
        assert base[k] == logged[k], k
    assert w0.keys() == w1.keys()
    for k in w0:
        assert torch.equal(w0[k], w1[k]), k
    runs = glob.glob(str(wd / "**" / "offline-run-*" / "*.wandb"), recursive=True)
    assert len(runs) == 1, runs
    blob = open(runs[0], "rb").read()
    for key in (b"train_score_vs_target", b"losses/policy_loss", b"eval/br_score", b"final/exploitability_proxy"):
        assert key in blob, key
