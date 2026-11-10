"""hpc/stage4_eval.py (the stage-4 evaluation at an analysis point, split into tasks): a tiny plan over two fake
runs, every task run, the merge's tables; the meta tasks reproduce metagame.deck_metagame exactly."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hpc", "tests"))

torch = pytest.importorskip("torch")


def test_stage4_eval_plan_run_merge(tmp_path):
    from test_stage4 import _random_policy
    lg = tmp_path / "league"
    for k, r in enumerate(("S4_C_s0", "S4_A_s0")):
        (lg / r).mkdir(parents=True)
        # batch 1e9, interval 1: the 2B point is snap_2
        json.dump({"env": {"frame_skip": 10}, "league": {"batch_size": 1_000_000_000, "snapshot_interval": 1}},
                  open(lg / r / "config.json", "w"))
        _random_policy(lg / r / "snap_2.pt", 10 + k)
    spec = _random_policy(tmp_path / "spec.pt", 20)
    out = tmp_path / "eval"
    cli = [sys.executable, os.path.join(ROOT, "hpc", "stage4_eval.py")]
    r = subprocess.run(cli + ["plan", "--point", "2B", "--out", str(out), "--data-dir", str(tmp_path), "--runs", "S4_C_s0",
                              "S4_A_s0", "--pool", "hog26;giant", "--heldout", "xbow", "--presets", "hog26;giant;xbow",
                              "--specialists", f"SPEC={spec}", "--gen-pool-episodes", "1", "--gen-heldout-episodes", "1",
                              "--meta-matches", "2", "--spec-matches", "2", "--spec-chunk", "1", "--rr-matches", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    tasks = [json.loads(ln) for ln in open(out / "tasks.jsonl")]
    kinds = [t["kind"] for t in tasks]
    assert kinds.count("gen") == 8 and kinds.count("meta") == 6 and kinds.count("spec") == 4 and kinds.count("rr") == 3
    plan = json.load(open(out / "plan.json"))
    assert plan["checkpoints"]["S4_C_s0"]["epoch"] == 2 and plan["checkpoints"]["S4_C_s0"]["global_step"] == 2 * 10 ** 9
    for t in tasks:
        r = subprocess.run(cli + ["run", str(out), "--task", str(t["id"])], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, (t, r.stderr[-3000:])
    mtime = os.path.getmtime(out / "out" / "0000.json")
    subprocess.run(cli + ["run", str(out), "--task", "0"], check=True, cwd=ROOT)       # done tasks are skipped
    assert os.path.getmtime(out / "out" / "0000.json") == mtime
    r = subprocess.run(cli + ["merge", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    S = json.load(open(out / "summary.json"))
    assert S["missing_tasks"] == []
    g = S["generalisation"]["S4_C_s0"]["heuristic"]
    assert g["pool"]["matches"] == 4 and g["heldout"]["matches"] == 2 and "gap_pool_minus_heldout" in g
    assert S["specialists"]["S4_A_s0"]["SPEC"]["matches"] == 2
    rr = S["round_robin"]
    assert rr["agents"] == ["S4_C_s0", "S4_A_s0"] and rr["games"][0][1] == 6 and rr["C_vs_A"]["matches"] == 6
    assert abs(rr["P"][0][1] + rr["P"][1][0] - 1) < 1e-9 and set(rr["per_deck"]) == {"hog26", "giant", "xbow"}
    # the meta tasks are deck_metagame's matches
    from pufferroyale.metagame import deck_metagame
    P, x = deck_metagame(str(lg / "S4_C_s0" / "snap_2.pt"), ["hog26", "giant", "xbow"], 2, seed=0)
    assert S["deck_metagame"]["S4_C_s0"]["P"] == [[round(float(v), 4) for v in row] for row in P]
    assert (out / "summary.md").read_text().startswith("# Stage-4 evaluation at 2B")
