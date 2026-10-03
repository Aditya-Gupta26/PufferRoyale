"""Builder tests, SPEC §19.10 (v0.5-G.3 audit amendments): fixed anneal length, the shaping / clamp
warning, league history after a preemption, the stages gate source and working directory, recorded
deck sets and the weight cap, the conditional head under bf16 autocast, the wandb step metric and
clean --resume errors. Every script run is a tiny CPU run."""
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

import pufferlib.pytorch
import pufferroyale
import pufferroyale.torch as prt
from pufferroyale import royale as R
from pufferroyale.decks import MAX_WEIGHT, deck_entries, expanded_deck_sets, parse_deck_set
from pufferroyale.game import DECKS
from pufferroyale.trainer import shaping_anneal_steps, shaping_env_kwargs, warn_shaping_clip

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
TINY = ["--num-envs", "4", "--bptt-horizon", "16", "--frame-skip", "50", "--seed", "0"]
WARN = "warning: reward shaping is on"
NEG = torch.finfo(torch.float32).min


def run(script, *args, cwd=ROOT, check=True, env=None, timeout=600):
    p = subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *map(str, args)], cwd=cwd,
                       capture_output=True, text=True, timeout=timeout, env=env)
    if check:
        assert p.returncode == 0, f"{script} {args}:\n{(p.stdout + p.stderr)[-3000:]}"
    return p


def last_json(text):
    return json.loads([l for l in text.strip().splitlines() if l.strip()][-1])


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load(path):
    with open(path) as f:
        return json.load(f)


# ==========================================================================================
# 6. the conditional head under bf16 autocast
# ==========================================================================================
def bot_states(grid, steps=50, seed=0, num_envs=3):
    """Observations of real states (heuristic bots on both seats, fine actions)."""
    env = pufferroyale.Royale(num_envs=num_envs, num_agents=2, seed=seed, placement_grid=grid, deck0="random",
                              deck1="random")
    env.reset(seed=seed)
    for i in range(num_envs):
        env.set_row_grid(i, 0, 1)
        env.set_row_grid(i, 1, 1)
    bots = [pufferroyale.Bot("heuristic", seed=seed * 10 + r) for r in range(2 * num_envs)]
    out = []
    for t in range(steps):
        acts = np.array([bots[r].act_env(env, r % 2, r // 2) for r in range(2 * num_envs)], np.int32)
        obs, *_ = env.step(acts)
        if t % 10 == 9:
            out.append(obs.copy())
    return env, np.concatenate(out)


@pytest.fixture(scope="module")
def states():
    out = {}
    for g in (1, 2, 4):
        out[g] = bot_states(g)
    yield out
    for env, _ in out.values():
        env.close()


@pytest.mark.parametrize("grid", [1, 2, 4])
@pytest.mark.parametrize("head", ["flat", "conditional"])
def test_bf16_autocast_forward_and_backward_are_finite(states, grid, head):
    env, obs = states[grid]
    x = torch.as_tensor(obs)
    mask = prt.action_mask(x, grid)
    assert (~mask[:, 1:]).reshape(len(obs), 4, -1).all(-1).any(), "the states must contain fully masked slots"
    torch.manual_seed(0)
    pol = prt.Policy(env, head=head)
    with torch.no_grad():                                     # away from the near-uniform init
        for q in pol.parameters():
            q.add_(torch.randn_like(q) * 0.05)
    with torch.no_grad():
        ref, _ = pol.forward_eval(x)                          # float32 reference
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        logits, value = pol.forward_eval(x)
        parts = pol.conditional_parts(x) if head == "conditional" else {}
        _, lp, ent = pufferlib.pytorch.sample_logits(logits)
        loss = lp.mean() + ent.mean() + value.float().mean()
        if parts:                                             # the factors' legal entries enter the graph too
            pos_ok = mask[:, 1:].reshape(len(obs), 4, -1)
            card_ok = torch.cat([mask[:, :1], pos_ok.any(-1)], dim=-1)
            loss = loss + 1e-3 * (torch.where(pos_ok, parts["pos_logp"], 0.0).sum()
                                  + torch.where(card_ok, parts["card_logp"], 0.0).sum())
    for k, v in parts.items():
        if v.dtype.is_floating_point:
            assert v.dtype == torch.float32, f"{k} must be float32 under bf16 autocast, got {v.dtype}"
            assert torch.isfinite(v).all(), f"conditional_parts[{k!r}] has NaN / inf under bf16 (g={grid})"
    assert torch.isfinite(logits).all() and torch.isfinite(value).all() and torch.isfinite(loss)
    assert (logits[~mask] == torch.finfo(logits.dtype).min).all(), "illegal logits stay exactly finfo.min"
    p = torch.softmax(logits.float(), -1)
    assert (p[mask] > 0).all() and torch.allclose(p.sum(-1), torch.ones(len(obs)), atol=1e-3)
    if head == "conditional":
        assert logits.dtype == torch.float32
        assert (logits[mask] - ref[mask]).abs().max() < 0.1, "bf16 logits far from the float32 ones"
    loss.backward()
    for n, q in pol.named_parameters():
        assert q.grad is None or torch.isfinite(q.grad).all(), f"non-finite grad in {n} (g={grid}, {head})"


@pytest.mark.parametrize("grid", [2, 4])
def test_float32_head_has_no_autocast_side_effects(states, grid):
    """Outside autocast the float32 head computes exactly what it did (pooling matmul in float32)."""
    env, obs = states[grid]
    torch.manual_seed(3)
    pol = prt.Policy(env)
    x = torch.as_tensor(obs)
    with torch.no_grad():
        parts = pol.conditional_parts(x)
        h = pol.encode_observations(x)
    assert not torch.is_autocast_enabled("cpu")
    # the pooled position logits equal an independent float32 recomputation
    from pufferroyale.torch import _blocks
    _, pool = _blocks(grid)
    C, H, W = pol.spatial_shape
    with torch.no_grad():
        hand = pol.encode_cards(pol._ids(x[:, R.SCALAR_OFFSET:R.SCALAR_OFFSET + R.SCALAR_SIZE]
                                         .index_select(1, pol.sc_hand_idx)))
        cond = pol.slot_cond(torch.cat([h.unsqueeze(1).expand(len(x), 4, h.shape[-1]), hand], dim=-1))
        board = pol.pos_board(x[:, R.SPATIAL_OFFSET:R.SPATIAL_OFFSET + C * H * W].reshape(len(x), C, H, W))
        gamma, beta = pol.film(cond).reshape(len(x), 4, 2, -1).unbind(2)
        feat = torch.relu(board.unsqueeze(1) * (1.0 + gamma)[..., None, None] + beta[..., None, None])
        pos = pol.pos_out(feat.reshape(len(x) * 4, -1, H, W)).reshape(len(x), 4, H * W) @ torch.as_tensor(pool)
        slot_mask = prt.action_mask(x, grid)[:, 1:].reshape(len(x), 4, -1)
        want = torch.log_softmax(torch.where(slot_mask, pos, NEG), dim=-1)
    assert torch.equal(parts["pos_logp"], want)


# ==========================================================================================
# 1. anneal length: exact and fixed per run
# ==========================================================================================
def test_anneal_steps_exact_rational():
    assert shaping_anneal_steps(0.07, 300_000_000, 3000) == 7000           # float: ceil(0.07 * 3e8 / 3000) = 7001
    assert shaping_anneal_steps(0.07, 3e8, 3000) == 7000
    assert shaping_anneal_steps("0.07", 300_000_000, 3000) == 7000
    assert shaping_anneal_steps(0.3, 1001, 4) == 76 and shaping_anneal_steps(0.0, 10 ** 9, 7) == 0
    assert shaping_anneal_steps(0.1, 30, 3) == 1 and shaping_anneal_steps(1e-5, 10 ** 9, 1) == 10_000
    kw = shaping_env_kwargs(0.99, 0.07, 6e8, 3000, 123_456, anneal_steps=7000)
    assert kw == {"reward_gamma": 0.99, "shaping_anneal_steps": 7000, "shaping_step_offset": 123_456 // 3000}
    for bad in (-0.1, float("nan"), float("inf"), "x"):
        with pytest.raises(ValueError):
            shaping_anneal_steps(bad, 100, 4)
    with pytest.raises(ValueError):
        shaping_env_kwargs(0.99, 0.5, 100, 4, anneal_steps=-1)


# ==========================================================================================
# 2. the shaping / clamp warning
# ==========================================================================================
def test_warn_shaping_clip(capsys):
    assert not warn_shaping_clip({"reward_tower": 0.0, "reward_crown": 0.0}, 1.0)
    assert not warn_shaping_clip({"reward_tower": 0.3}, 0.0)
    assert not warn_shaping_clip({"reward_tower": 0.3}, -1.0)
    assert capsys.readouterr().err == ""
    assert warn_shaping_clip({"reward_tower": 0.3, "reward_elixir": 0.01}, 1.0, "x")
    err = capsys.readouterr().err
    assert err.count(WARN) == 1 and "reward_tower" in err and "reward_elixir" in err and "--train.reward-clip 0" in err


# ==========================================================================================
# 5. deck sets: weight cap and the expanded record
# ==========================================================================================
def test_deck_weight_cap(tmp_path):
    assert MAX_WEIGHT == 1e9
    assert parse_deck_set("hog26:1e9") == [(tuple(sorted(DECKS["hog26"])), 1e9)]
    for spec in ("hog26:1e308", "hog26:1.5e9", "hog26:inf", "hog26:nan", [("hog26", 2e9)],
                 [{"deck": "giant", "weight": float("inf")}], [[list(DECKS["bait"]), 1e300]]):
        with pytest.raises(ValueError):
            parse_deck_set(spec)
    f = tmp_path / "pool.json"
    f.write_text(json.dumps([{"deck": "hog26", "weight": 1e300}]))
    with pytest.raises(ValueError):
        parse_deck_set(f"file:{f}")
    with pytest.raises(ValueError):
        pufferroyale.Royale(num_envs=1, num_agents=2, deck_pool="hog26:1e308;giant")
    env = pufferroyale.Royale(num_envs=1, num_agents=2, deck_pool="hog26:1e9;giant:1e-9")
    env.close()


def test_expanded_deck_sets_installed_order():
    env = {"deck_pool": "hog26:2;random:2:5", "heldout_decks": "giant"}
    ex = expanded_deck_sets(env)
    assert ex["deck_pool_decks"] == [[list(o), w] for _, w, o in deck_entries(env["deck_pool"])]
    assert ex["deck_pool_decks"][0] == [list(DECKS["hog26"]), 2.0], "a preset keeps its own order"
    assert ex["heldout_decks_decks"] == [[sorted(DECKS["giant"]), 1.0]], "held-out decks are installed ascending"
    assert expanded_deck_sets({"deck_pool": "", "heldout_decks": None}) == {"deck_pool_decks": [],
                                                                          "heldout_decks_decks": []}
    json.dumps(ex, allow_nan=False)


# ==========================================================================================
# 7. wandb: global_step is the step metric of every record
# ==========================================================================================
def test_wandb_logger_logs_global_step_as_step_metric(monkeypatch):
    calls = {"define": [], "log": []}

    class FakeRun:
        id = "fake123"
        summary = {}

        def define_metric(self, name, **kw):
            calls["define"].append((name, kw))

        def log(self, data, **kw):
            calls["log"].append((dict(data), kw))

        def finish(self, **kw):
            pass

    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: FakeRun()
    fake.Settings = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "wandb", fake)
    from pufferroyale.trainer import WandbLogger
    lg = WandbLogger({"wandb_project": "p", "wandb_group": None, "tag": None, "no_model_upload": True})
    assert ("global_step", {}) in calls["define"]
    assert ("*", {"step_metric": "global_step"}) in calls["define"]
    lg.log({"losses/x": 1.0}, 512)
    lg.log({"losses/x": 2.0}, 256)                         # a re-done epoch after a resume: not dropped
    assert [d["global_step"] for d, _ in calls["log"]] == [512, 256]
    assert all("step" not in kw for _, kw in calls["log"]), "wandb's own step must not be passed"


# ==========================================================================================
# league_train.py: fixed N, history after a preemption, persisted window, clean errors, decks
# ==========================================================================================
LEAGUE = ["--anchors", "bot:noop", "--self-play-frac", "0", "--anchor-frac", "1.0", "--snapshot-interval", "2"]


@pytest.fixture(scope="module")
def league(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("wp_audit_league"))
    p = run("league_train.py", "--total-timesteps", 512, *TINY, *LEAGUE, "--data-dir", d, "--run-id", "base",
            "--shaping-anneal-frac", 0.25, "--env.reward-tower", 0.3, "--env.deck-pool", "hog26:2;giant",
            "--env.heldout-decks", "bait")
    return {"dir": d, "run": os.path.join(d, "league", "base"), "first": p, "summary": last_json(p.stdout)}


def test_league_first_run_records_n_decks_window_and_warns(league):
    rd, p = league["run"], league["first"]
    assert p.stderr.count(WARN) == 1, "one shaping / clamp warning (shaping on, reward_clip 1)"
    cfg, st = load(os.path.join(rd, "config.json")), load(os.path.join(rd, "league_state.json"))
    assert cfg["env"]["shaping_anneal_steps"] == 32 == st["args"]["shaping_anneal_steps"]   # ceil(0.25 * 512 / 4)
    assert cfg["env"]["deck_pool"] == "hog26:2;giant" and cfg["env"]["heldout_decks"] == "bait"
    assert cfg["env"]["deck_pool_decks"] == [[list(DECKS["hog26"]), 2.0], [list(DECKS["giant"]), 1.0]]
    assert cfg["env"]["heldout_decks_decks"] == [[sorted(DECKS["bait"]), 1.0]]
    # the early-stop window at the final save = the run's last <= N outcomes vs the anchor
    outs = [r for rec in read_jsonl(os.path.join(rd, "history.jsonl")) for r in rec["outcomes"].get("bot:noop", [])]
    assert outs, "the base run must finish matches vs its anchor"
    assert st["early_stop"]["recent"]["bot:noop"] == outs[-2000:]


def test_league_resume_keeps_n_truncates_history_and_reports_clean_errors(league, tmp_path):
    import shutil
    rd = str(tmp_path / "copy")
    shutil.copytree(league["run"], rd)
    st = load(os.path.join(rd, "league_state.json"))
    assert st["epoch"] == 8
    hist = os.path.join(rd, "history.jsonl")
    with open(hist, "a") as f:                             # what a preempted process wrote after its last save
        for e in (9, 10):
            f.write(json.dumps({"epoch": e, "global_step": 64 * e, "rolled_back": True,
                                "outcomes": {"bot:noop": [-1] * 7}, "results": {}, "losses": {}}) + "\n")
        f.write('{"epoch": 11, "glo')                      # torn last line
    # 8. another grid / head / LSTM than the saved run: a clear SystemExit, no traceback
    for bad in (["--env.placement-grid", "2"], ["--policy.head", "flat"], ["--policy.hidden-size", "128"]):
        p = run("league_train.py", "--resume", rd, "--total-timesteps", 768, *bad, check=False)
        assert p.returncode != 0 and "Traceback" not in p.stderr, p.stderr[-2000:]
        assert "different policy than the saved run" in p.stderr, p.stderr[-2000:]
    # 1. + 3. a larger total without --shaping-anneal-frac: N kept; the rolled-back records are dropped
    p = run("league_train.py", "--resume", rd, "--total-timesteps", 768)
    assert p.stderr.count(WARN) == 1
    cfg, st = load(os.path.join(rd, "config.json")), load(os.path.join(rd, "league_state.json"))
    assert cfg["env"]["shaping_anneal_steps"] == 32 == st["args"]["shaping_anneal_steps"]
    assert cfg["env"]["shaping_step_offset"] == 512 // 4
    recs = read_jsonl(hist)
    assert [r["epoch"] for r in recs] == list(range(1, 13)), "history = epochs 1..12, each exactly once"
    assert not any(r.get("rolled_back") for r in recs)
    outs = [r for rec in recs for r in rec["outcomes"].get("bot:noop", [])]
    assert st["early_stop"]["recent"]["bot:noop"] == outs[-2000:]
    # --shaping-anneal-frac given again: N recomputed from the new F and the new absolute total
    run("league_train.py", "--resume", rd, "--total-timesteps", 1024, "--shaping-anneal-frac", 0.5)
    assert load(os.path.join(rd, "config.json"))["env"]["shaping_anneal_steps"] == 128      # ceil(0.5 * 1024 / 4)
    assert load(os.path.join(rd, "league_state.json"))["args"]["shaping_anneal_steps"] == 128


def test_league_new_run_moves_a_stale_history_aside(tmp_path):
    d = str(tmp_path)
    rd = os.path.join(d, "league", "s")
    os.makedirs(rd)
    for name, tag in (("history.jsonl", "old"), ("history.jsonl.stale-1", "older")):
        with open(os.path.join(rd, name), "w") as f:
            f.write(json.dumps({"epoch": 99, "tag": tag}) + "\n")
    p = run("league_train.py", "--total-timesteps", 128, *TINY, *LEAGUE, "--data-dir", d, "--run-id", "s")
    assert WARN not in p.stderr, "no shaping: no warning"
    assert read_jsonl(os.path.join(rd, "history.jsonl.stale-1"))[0]["tag"] == "older"
    assert read_jsonl(os.path.join(rd, "history.jsonl.stale-2"))[0]["tag"] == "old"
    assert [r["epoch"] for r in read_jsonl(os.path.join(rd, "history.jsonl"))] == [1, 2]


# ==========================================================================================
# 4. stages.py: gate from the persisted window; the caller's working directory
# ==========================================================================================
def stages_module():
    sys.path.insert(0, SCRIPTS)
    try:
        import stages
    finally:
        sys.path.remove(SCRIPTS)
    return stages


def test_stages_gate_reads_the_persisted_window_not_the_history(tmp_path):
    stages = stages_module()
    rd = str(tmp_path)
    with open(os.path.join(rd, "league_state.json"), "w") as f:
        json.dump({"early_stop": {"recent": {"bot:noop": [1, 1, 0, 1], "ckpt:/abs/x.pt": [-1]}}}, f)
    with open(os.path.join(rd, "history.jsonl"), "w") as f:      # rolled-back epochs: never counted
        for e in range(1, 4):
            f.write(json.dumps({"epoch": e, "outcomes": {"bot:noop": [-1] * 50}}) + "\n")
    assert stages.gate(rd, "bot:noop", 4) == (pytest.approx(0.875), 4)
    assert stages.gate(rd, "bot:noop", 2) == (pytest.approx(0.75), 2)          # the last N of the window
    assert stages.gate(rd, "bot:noop", 10) == (pytest.approx(0.875), 4)         # fewer than N: main() fails it
    assert stages.gate(rd, "bot:random", 4) == (None, 0)
    assert stages.gate(rd, "ckpt:/abs/x.pt", 4) == (0.0, 1)
    assert stages.gate(str(tmp_path / "missing"), "bot:noop", 4) == (None, 0)


def test_stages_after_a_preemption_and_from_another_cwd(tmp_path):
    d = str(tmp_path / "data")
    # a rung that already reached its budget (league_train --resume: nothing to do), its history holding
    # rolled-back records with losses: the gate is the persisted window only
    run("league_train.py", "--total-timesteps", 128, *TINY, *LEAGUE, "--data-dir", d, "--run-id", "pre_0_noop")
    rd = os.path.join(d, "league", "pre_0_noop")
    st = load(os.path.join(rd, "league_state.json"))
    st["early_stop"]["recent"]["bot:noop"] = [1, 1, 1]
    with open(os.path.join(rd, "league_state.json"), "w") as f:
        json.dump(st, f)
    with open(os.path.join(rd, "history.jsonl"), "a") as f:
        f.write(json.dumps({"epoch": 3, "outcomes": {"bot:noop": [-1] * 40}}) + "\n")
    common = ["--run-prefix", "pre", "--rungs", "bot:noop:128:0.9", "--data-dir", d, *TINY, "--snapshot-interval", 2]
    p = run("stages.py", *common, "--gate-window", 3, check=False)
    rep = last_json(p.stdout)
    assert p.returncode == 0 and rep["passed"], p.stdout[-2000:] + p.stderr[-2000:]
    assert rep["rungs"][0]["score"] == 1.0 and rep["rungs"][0]["matches"] == 3
    assert [r["epoch"] for r in read_jsonl(os.path.join(rd, "history.jsonl"))] == [1, 2], \
        "a --resume (even with nothing to do) cuts the history back to the saved epoch"
    p = run("stages.py", *common, "--gate-window", 5, check=False)
    rep = last_json(p.stdout)
    assert p.returncode == 1 and rep["rungs"][0]["reason"] == "insufficient matches" and rep["rungs"][0]["matches"] == 3
    # launched from another directory with a relative file: deck set and a relative --data-dir
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    (cwd / "pool.json").write_text(json.dumps(["giant", {"deck": "hog26", "weight": 3}]))
    p = run("stages.py", "--run-prefix", "cw", "--rungs", "bot:noop:128:0.0", "--gate-window", 1, "--data-dir", "out",
            *TINY, "--snapshot-interval", 2, "--env.deck-pool", "file:pool.json", cwd=str(cwd), check=False)
    assert p.returncode in (0, 1), p.stdout[-2000:] + p.stderr[-2000:]
    assert os.path.isfile(cwd / "out" / "stages_cw.json")
    cfg = load(cwd / "out" / "league" / "cw_0_noop" / "config.json")
    assert cfg["env"]["deck_pool"] == "file:pool.json"
    assert cfg["env"]["deck_pool_decks"] == [[list(DECKS["giant"]), 1.0], [list(DECKS["hog26"]), 3.0]]


# ==========================================================================================
# train.py and best_response.py
# ==========================================================================================
def last_json_block(text):
    return json.loads(text[text.rindex("\n{") + 1:])


def test_train_fixed_n_decks_warning_and_clean_resume_error(tmp_path):
    d = str(tmp_path)
    common = ["--num-envs", "2", "--batch-size", "64", "--minibatch-size", "32", "--bptt-horizon", "16", "--quiet",
              "--env.frame-skip", "50", "--train.checkpoint-interval", "1"]
    p = run("train.py", "--total-timesteps", 128, "--data-dir", d, "--shaping-anneal-frac", 0.07, *common,
            "--env.reward-tower", 0.3, "--env.deck-pool", "hog26:2;giant", "--env.heldout-decks", "bait")
    assert p.stderr.count(WARN) == 1
    rd = last_json_block(p.stdout)["run_dir"]
    cfg = load(os.path.join(rd, "config.json"))
    assert cfg["train"]["shaping_anneal_steps"] == 3 == cfg["env"]["shaping_anneal_steps"]   # ceil(0.07 * 128 / 4)
    assert cfg["env"]["deck_pool_decks"] == [[list(DECKS["hog26"]), 2.0], [list(DECKS["giant"]), 1.0]]
    assert cfg["env"]["heldout_decks_decks"] == [[sorted(DECKS["bait"]), 1.0]]
    # 8. another grid on --resume: a clear SystemExit, no traceback
    p = run("train.py", "--resume", rd, "--total-timesteps", 256, "--env.placement-grid", 2, check=False)
    assert p.returncode != 0 and "Traceback" not in p.stderr and "different policy than the saved run" in p.stderr, \
        p.stderr[-2000:]
    # 1. a larger total: N kept (the recorded deck lists are not passed back to the env)
    run("train.py", "--resume", rd, "--total-timesteps", 256, "--quiet")
    cfg = load(os.path.join(rd, "config.json"))
    assert cfg["train"]["shaping_anneal_steps"] == 3 == cfg["env"]["shaping_anneal_steps"]
    assert cfg["env"]["shaping_step_offset"] == 128 // 4 and cfg["env"]["deck_pool_decks"][0][1] == 2.0
    # F given again: recomputed from the new F and the new total
    p = run("train.py", "--resume", rd, "--total-timesteps", 320, "--quiet", "--shaping-anneal-frac", 0.5,
            "--train.reward-clip", 0)
    assert WARN not in p.stderr, "reward_clip 0: no warning"
    cfg = load(os.path.join(rd, "config.json"))
    assert cfg["train"]["shaping_anneal_steps"] == 40 == cfg["env"]["shaping_anneal_steps"]  # ceil(0.5 * 320 / 4)


def test_best_response_warns_and_records_n_and_decks(league, tmp_path):
    d = str(tmp_path)
    p = run("best_response.py", "--target", league["run"], "--total-timesteps", 64, "--num-envs", 4, "--bptt-horizon",
            16, "--matches", 1, "--data-dir", d, "--shaping-anneal-frac", 0.07, "--env.reward-crown", 0.2,
            "--env.deck-pool", "giant")
    assert p.stderr.count(WARN) == 1
    res = last_json(p.stdout)
    cfg = load(os.path.join(os.path.dirname(res["br_checkpoint"]), "config.json"))
    assert cfg["env"]["shaping_anneal_steps"] == 2 == cfg["league"]["shaping_anneal_steps"]   # ceil(0.07 * 64 / 4)
    assert cfg["env"]["deck_pool_decks"] == [[list(DECKS["giant"]), 1.0]] and cfg["env"]["heldout_decks_decks"] == []
