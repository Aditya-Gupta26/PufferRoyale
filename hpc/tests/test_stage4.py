"""Checks of the stage-4 additions (D99, 2026-10-09). Cluster-side (not under tests/spec): run with
`python -m pytest -q hpc/tests`.

- league_train.wandb_record league sections (2026-10-20): league/ and league_snapshots/ instead of
  environment/league/* and results/<spec>/* in league runs; best_response.py's plain layout unchanged.
- league_train.WandbWindow (--log-interval N): averages losses and environment/* stats, sums results, and
  a CPU league run with --log-interval 3 (WANDB_MODE=offline) logs one history row per 3 epochs plus the
  last partial window, whose results add up to history.jsonl's.
- hpc/eval_watch.evaluate: bots on named deck sets (eval.py --decks, pooled rows) and frozen checkpoint
  opponents (tournament.py), with per-seat scores.
- hpc/stage4_submit.sh (dry run): the env files carry the D99 settings and the D100 card floor.
"""
import glob
import json
import math
import os
import struct
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "hpc"))

torch = pytest.importorskip("torch")


def _rec(epoch, losses, results, step=None):
    return {"epoch": epoch, "global_step": step or 100 * epoch, "sps": 10.0 * epoch, "pool_size": 3,
            "losses": losses, "results": results}


def test_wandb_window_aggregates():
    import league_train as LT
    w = LT.WandbWindow(3)
    assert [e for e in range(1, 8) if w.due(e)] == [3, 6]
    w.add(_rec(1, {"policy_loss": 1.0, "entropy": None}, {"self": {"games": 2, "score": 0.5}}),
          {"environment/a": 1.0, "environment/b": float("nan"), "losses/x": 9.0})
    w.add(_rec(2, {"policy_loss": 3.0, "entropy": 2.0}, {"self": {"games": 6, "score": 0.25},
                                                         "ckpt:/x/snap_7.pt": {"games": 1, "score": 1.0}}),
          {"environment/a": 3.0})
    out, step = w.pop()
    assert step == 200
    assert out["epoch"] == 2 and out["global_step"] == 200 and out["sps"] == 20.0 and out["pool_size"] == 3
    assert out["losses/policy_loss"] == 2.0 and out["losses/entropy"] == 2.0          # mean over reporting epochs
    assert out["results/self/games"] == 8 and out["results/self/score"] == pytest.approx((1 + 1.5) / 8)
    assert out["results/ckpt:snap_7.pt/games"] == 1 and out["results/ckpt:snap_7.pt/score"] == 1.0
    assert out["environment/a"] == 2.0 and "environment/b" not in out and "losses/x" not in out
    assert w.pop() is None                                                           # cleared


def test_wandb_window_one_epoch_equals_record():
    import league_train as LT
    rec = _rec(5, {"policy_loss": 0.1, "value_loss": None}, {"self": {"games": 3, "score": 2 / 3}})
    logs = {"environment/cards/play_rate/hog_rider": 0.25, "train/x": 1.0}
    w = LT.WandbWindow(1)
    w.add(rec, logs)
    out, step = w.pop()
    ref = LT.wandb_record(rec, logs)
    assert step == rec["global_step"] and out.keys() == ref.keys()
    for k in ref:
        assert out[k] == pytest.approx(ref[k], rel=1e-12), k


def test_wandb_record_league_sections():
    """league_train's wandb layout (2026-10-20): self / snapshot results and environment/league/* move to the
    league and league_snapshots sections; anchors stay under results/; best_response's plain layout is unchanged."""
    import league_train as LT
    bs, ep = 16384, 20000
    snaps = {e: f"ckpt:/runs/league/X/snap_{e}.pt" for e in (19900, 18000, 12000, 5000, 4000)}
    res = {"self": {"games": 10, "score": 0.5}, "bot:heuristic": {"games": 4, "score": 0.75}}
    played = {19900: (2, 0.5), 18000: (4, 0.25), 12000: (2, 1.0), 5000: (8, 0.125)}   # snap_4000: in the pool, unplayed
    for e, (g, sc) in played.items():
        res[snaps[e]] = {"games": g, "score": sc}
    logs = {"environment/league/pool_size": 6.0, "environment/league/episodes": 99.0, "environment/score": 0.4,
            "environment/league/p/self": 0.5, "environment/league/p/bot:heuristic": 0.7}
    for e, p in zip(snaps, (0.4, 0.3, 0.6, 0.2, 0.5)):
        logs[f"environment/league/p/{snaps[e]}"] = p
    rec = _rec(ep, {"policy_loss": 0.1}, res, step=ep * bs)
    out = LT.wandb_record(rec, logs, league_sections=True)
    assert not [k for k in out if k.startswith("environment/league/")]
    assert out["environment/score"] == 0.4 and out["league/pool_size"] == 6.0 and out["league/episodes"] == 99.0
    assert out["league/score_vs_self"] == 0.5 and out["league/games_vs_self"] == 10
    assert "results/self/score" not in out and out["results/bot:heuristic/score"] == 0.75
    for e, (g, sc) in played.items():
        assert out[f"league_snapshots/snap_{e}/games"] == g and out[f"league_snapshots/snap_{e}/score"] == sc
    assert out["league_snapshots/snap_19900/p"] == 0.4 and out["league_snapshots/snap_5000/p"] == 0.2
    assert not [k for k in out if "snap_4000" in k]                                  # unplayed: no series
    assert out["league/games_vs_snapshots"] == 16
    assert out["league/score_vs_snapshots"] == pytest.approx((1 + 1 + 2 + 1) / 16)
    ages = {"0-16M": (2, 0.5), "16-64M": (4, 0.25), "64-200M": (2, 1.0), "200M+": (8, 0.125)}   # 1.6M, 33M, 131M, 246M
    for label, (g, sc) in ages.items():
        assert out[f"league/games_vs_snapshots_age_{label}"] == g
        assert out[f"league/score_vs_snapshots_age_{label}"] == pytest.approx(sc)
    assert out["league/pool_p_min"] == 0.2 and out["league/pool_p_max"] == 0.6
    assert out["league/pool_p_median"] == pytest.approx((0.3 + 0.4) / 2)
    plain = LT.wandb_record(rec, logs)                                             # best_response.py's layout
    assert plain["results/self/score"] == 0.5 and plain["results/ckpt:snap_5000.pt/games"] == 8
    assert plain["environment/league/p/self"] == 0.5 and not [k for k in plain if k.startswith("league")]
    w = LT.WandbWindow(1, league_sections=True)
    w.add(rec, logs)
    got, step = w.pop()
    assert step == ep * bs and got.keys() == out.keys()
    for k in out:
        assert got[k] == pytest.approx(out[k], rel=1e-12), k


def _wandb_history(path):
    """History rows of an offline .wandb file (LevelDB-log framing: 7-byte file header, 32 KiB blocks)."""
    from wandb.proto import wandb_internal_pb2 as pb2
    buf, pos, parts, rows = open(path, "rb").read(), 7, b"", []
    assert buf[:4] == b":W&B"
    while pos + 7 <= len(buf):
        left = 32768 - pos % 32768
        if left < 7:
            pos += left
            continue
        _, n, typ = struct.unpack("<IHB", buf[pos:pos + 7])
        data, pos = buf[pos + 7:pos + 7 + n], pos + 7 + n
        if typ in (1, 4):
            r = pb2.Record()
            r.ParseFromString(parts + data if typ == 4 else data)
            if r.WhichOneof("record_type") == "history":
                rows.append({(i.key or "/".join(i.nested_key)): json.loads(i.value_json) for i in r.history.item})
        elif typ == 2:
            parts = data
        elif typ == 3:
            parts += data
    return rows


def test_league_log_interval_offline(tmp_path):
    pytest.importorskip("wandb")
    wd = tmp_path / "wandb_dir"
    wd.mkdir()
    env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR=str(wd), WANDB_SILENT="true", WANDB_API_KEY="")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "league_train.py"), "--total-timesteps", str(8 * 64 * 7),
           "--num-envs", "8", "--anchors", "none", "--self-play-frac", "0.5", "--anchor-frac", "0",
           "--snapshot-interval", "2", "--max-snapshots", "4", "--seed", "1", "--data-dir", str(tmp_path),
           "--run-id", "wbint", "--log-interval", "3", "--wandb", "--wandb-project", "pr-test", "--wandb-group", "g",
           "--tag", "t", "--no-model-upload", "--train.torch-deterministic="]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    hist = [json.loads(line) for line in open(tmp_path / "league" / "wbint" / "history.jsonl")]
    assert [h["epoch"] for h in hist] == list(range(1, 8))
    files = glob.glob(str(wd / "**" / "offline-run-*" / "*.wandb"), recursive=True)
    assert len(files) == 1, files
    rows = [x for x in _wandb_history(files[0]) if "epoch" in x]
    assert [x["epoch"] for x in rows] == [3, 6, 7]
    assert [x["global_step"] for x in rows] == [hist[2]["global_step"], hist[5]["global_step"], hist[6]["global_step"]]
    import league_train as LT
    for spec in {s for h in hist for s in h["results"]}:
        key = LT.results_keys(spec, league_sections=True)[1]                # league runs log the league sections
        logged = sum(x.get(key, 0) for x in rows)
        assert logged == sum(h["results"].get(spec, {}).get("games", 0) for h in hist), spec
    pl = [h["losses"]["policy_loss"] for h in hist[:3]]
    assert rows[0]["losses/policy_loss"] == pytest.approx(sum(pl) / 3, rel=1e-9)


def _random_policy(path, seed):
    from pufferroyale import royale as R
    from pufferroyale.league import _obs_stub
    from pufferroyale.torch import Policy
    torch.manual_seed(seed)
    pol = Policy(_obs_stub(R.n_actions(1)), placement_grid=1)
    torch.save({k: v.detach().cpu() for k, v in pol.state_dict().items()}, path)
    return str(path)


def test_eval_watch_decks_and_checkpoints(tmp_path):
    import eval_watch as EW
    pt = _random_policy(tmp_path / "policy_000010.pt", 3)
    other = _random_policy(tmp_path / "other.pt", 4)
    res = EW.evaluate(pt, ["random"], 1, 0, 1, decks=[("pool", "hog26;giant")], ckpts=[("spec", other)],
                      ckpt_matches=4, ckpt_deck="hog26")
    assert set(res) == {"random_pool", "spec"}
    assert res["random_pool"]["matches"] == 2 * 1 * 2                   # 2 decks x 1 episode x 2 seats
    assert res["spec"]["matches"] == 4
    for r in res.values():
        assert 0.0 <= r["lo"] <= r["score"] <= r["hi"] <= 1.0
        assert r["seat0"] is not None and r["seat1"] is not None
        assert r["score"] == pytest.approx((r["seat0"] + r["seat1"]) / 2)  # equal matches per seat
    plain = EW.evaluate(pt, ["noop"], 1, 0, 1)                          # unchanged default: one key per bot
    assert set(plain) == {"noop"} and plain["noop"]["matches"] == 2


def test_stage4_submit_dry_run(tmp_path):
    data = tmp_path / "runs"
    for r in ("S3_A_s1", "S3_C_s0"):
        (data / "league" / r).mkdir(parents=True)
        (data / "league" / r / "model_061036.pt").write_bytes(b"x")
    env = dict(os.environ, DATA_DIR=str(data), TOTAL="10000000000", LINKS="11", PR_CHECKOUT=ROOT, SUBMIT="0")
    r = subprocess.run(["bash", os.path.join(ROOT, "hpc", "stage4_submit.sh")], env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = data / "stage4" / "S4" / "dryrun"
    files = [line.strip() for line in open(out / "manifest.txt")]
    assert [os.path.basename(f) for f in files] == ["S4_C_s0.env", "S4_A_s0.env", "S4_C_s1.env", "S4_A_s1.env"]
    assert "--constraint=h100|h200" in r.stdout and "--constraint=a100" in r.stdout and "--time=20:00:00" in r.stdout
    assert not (out / "chains.txt").exists()
    for f in files:
        kv = dict(line.split("=", 1) for line in open(f).read().splitlines() if line and not line.startswith("#"))
        kv = {k: v.strip("'") for k, v in kv.items()}
        x = kv["EXTRA_ARGS"].split()
        assert x[x.index("--shaping-anneal-frac") + 1] == "0.05"
        assert x[x.index("--log-interval") + 1] == "10"
        assert x[x.index("--card-floor") + 1] == "0.005" and x[x.index("--card-floor-steps") + 1] == "5000000000"
        assert x[x.index("--env.deck-pool") + 1] == "hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge"
        assert x[x.index("--env.heldout-decks") + 1] == "xbow;royal_hogs;random:50:20261003"
        assert x[x.index("--env.random-deck-frac") + 1] == "0.5" and "--anneal-lr" not in x
        assert kv["TOTAL"] == "10000000000" and kv["SNAPSHOT_INTERVAL"] == "477" and kv["MAX_SNAPSHOTS"] == "64"
        assert kv["EVAL_EVERY"] == "6104" and kv["TOURNAMENT_MATCHES"] == "0"
        e = kv["EVAL_ARGS"].split()
        assert f"S3_A_s1={data}/league/S3_A_s1/model_061036.pt" in e and f"S3_C_s0={data}/league/S3_C_s0/model_061036.pt" in e
        assert "pool=hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge" in e
        assert "heldout=xbow;royal_hogs;random:50:20261003" in e
        method = os.path.basename(f).split("_")[1]
        assert kv["MMD_COEF"] == ("0.05" if method == "C" else "0")
        assert kv["SELF_PLAY_FRAC"] == ("0.35" if method == "C" else "1.0")
    # D101: one Claude-game chain, for S4_C_s0 only
    assert sorted(os.path.basename(f) for f in glob.glob(str(out / "*.llm.env"))) == ["S4_C_s0.llm.env"]
    kv = dict(line.split("=", 1) for line in open(out / "S4_C_s0.llm.env").read().splitlines()
              if line and not line.startswith("#"))
    kv = {k: v.strip("'") for k, v in kv.items()}
    assert kv["RUN_ID"] == "S4_C_s0" and kv["LLM_EVERY"] == "250000000" and kv["LLM_GROUP"] == "S4_C"
    assert kv["LLM_MODEL"] == "claude-opus-5-5" and kv["PR_CHECKOUT"] == ROOT
    assert "--job-name=pr-S4_C_s0-llm" in r.stdout and "--time=2-00:00:00" in r.stdout
    from pufferroyale.trainer import shaping_anneal_steps
    assert shaping_anneal_steps(0.05, 10_000_000_000, 256) == 1953125 == math.ceil(500_000_000 / 256)


def test_stage4_submit_dry_run_dense(tmp_path):
    """D102: SHAPING_STEPS=0 and CARD_FLOOR_STEPS=0 give constant shaping and a constant floor."""
    data = tmp_path / "runs"
    for r in ("S3_A_s1", "S3_C_s0"):
        (data / "league" / r).mkdir(parents=True)
        (data / "league" / r / "model_061036.pt").write_bytes(b"x")
    env = dict(os.environ, DATA_DIR=str(data), TOTAL="10000000000", LINKS="13", PR_CHECKOUT=ROOT, SUBMIT="0",
               PREFIX="S4d", SHAPING_STEPS="0", CARD_FLOOR_STEPS="0")
    r = subprocess.run(["bash", os.path.join(ROOT, "hpc", "stage4_submit.sh")], env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = data / "stage4" / "S4d" / "dryrun"
    files = [line.strip() for line in open(out / "manifest.txt")]
    assert [os.path.basename(f) for f in files] == ["S4d_C_s0.env", "S4d_A_s0.env", "S4d_C_s1.env", "S4d_A_s1.env"]
    for f in files:
        kv = dict(line.split("=", 1) for line in open(f).read().splitlines() if line and not line.startswith("#"))
        x = kv["EXTRA_ARGS"].strip("'").split()
        assert x[x.index("--shaping-anneal-frac") + 1] == "0"
        assert x[x.index("--card-floor") + 1] == "0.005" and x[x.index("--card-floor-steps") + 1] == "0"
        assert x[x.index("--wandb-group") + 1] in ("S4d_C", "S4d_A")
    from pufferroyale.trainer import shaping_anneal_steps
    assert shaping_anneal_steps("0", 10_000_000_000, 256) == 0
    import league_train as LT
    assert LT.card_floor_at({"card_floor": 0.005, "card_floor_steps": 0}, 9_999_000_000) == 0.005
    assert (out / "S4d_C_s0.llm.env").exists() and "--job-name=pr-S4d_C_s0-llm" in r.stdout

