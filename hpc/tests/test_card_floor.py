"""Checks of the card-probability floor (DECISIONS D100, 2026-10-09). Cluster-side (not under tests/spec):
run with `python -m pytest -q hpc/tests`.

- pufferroyale.torch.Policy.enable_card_floor / set_card_floor: with eps = 0 the logits are unchanged; with eps > 0
  every legal hand slot gets (1 - k eps) P(s) + eps, wait (1 - k eps) P(wait), the position distribution and the
  mask are unchanged; gradients stay finite; state dicts carry no new key, so a loaded checkpoint plays without
  a floor; a CUDA-graph replay sees a new eps (GPU only).
- scripts/league_train.py --card-floor / --card-floor-steps: the linear schedule, the learner and the MMD
  reference get the same eps every epoch, history.jsonl / wandb get card_floor and cards/below_floor.
"""
import copy
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "hpc", "tests"))

torch = pytest.importorskip("torch")

POOL = "hog26;giant;bait;golem;lavaloon;miner_poison;pekka_bridge"
HELD = "xbow;royal_hogs;random:50:20261003"


def _obs(n_envs=8, steps=(0, 4, 12, 24, 40), seed=3):
    """Both seats' observations of stage-4 deals (mixed decks) at a few steps of waiting (elixir builds up,
    so rows have 0..4 legal hand slots)."""
    from pufferroyale import royale as R
    env = R.Royale(num_envs=n_envs, num_agents=2, seed=seed, deck_pool=POOL, random_deck_frac=0.5, heldout_decks=HELD)
    env.reset(seed=seed)
    out = []
    for t in range(max(steps) + 1):
        if t in steps:
            out.append(env.observations.copy())
        env.step(np.zeros(env.num_agents, dtype=np.int32))
    env.close()
    return torch.from_numpy(np.concatenate(out))


def _policy(seed=0, **kw):
    from pufferroyale import royale as R
    from pufferroyale.league import _obs_stub
    from pufferroyale.torch import Policy
    torch.manual_seed(seed)
    p = Policy(_obs_stub(R.n_actions(1)), placement_grid=1, **kw)
    if p.head == "conditional":
        with torch.no_grad():
            p.slot_logit.bias.fill_(-12.0)          # nearly dead cards: the floor decides their probability
    return p


def test_card_floor_math():
    obs = _obs()
    p0 = _policy()
    p = copy.deepcopy(p0)
    assert set(p.state_dict()) == set(p0.state_dict()) and not p.card_floor_on
    with torch.no_grad():
        ref = p0.conditional_parts(obs)
        p.enable_card_floor(0.0)
        assert torch.equal(p.conditional_parts(obs)["logits"], ref["logits"])        # eps = 0: unchanged
        eps = 0.02
        p.set_card_floor(eps)
        out = p.conditional_parts(obs)
    B = p.n_blocks
    probs = torch.softmax(out["logits"], -1)
    card = torch.cat([probs[:, :1], probs[:, 1:].reshape(-1, 4, B).sum(-1)], -1)        # P'(wait), P'(slot s)
    ok = ref["mask"][:, 1:].reshape(-1, 4, B).any(-1)
    k = ok.sum(-1, keepdim=True).float()
    assert k.max() == 4 and len(set(k.flatten().tolist())) >= 2, "the sample should hold rows with 4 and fewer legal slots"
    P0 = ref["card_logp"].exp()
    assert P0[:, 1:][ok].max() < 1e-4                                                  # the premise: dead cards
    assert torch.allclose(card.sum(-1), torch.ones(len(card)), atol=1e-5)
    assert torch.allclose(card[:, 1:][ok], ((1 - k * eps) * P0[:, 1:] + eps)[ok], rtol=1e-5, atol=1e-7)
    assert torch.allclose(card[:, :1], (1 - k * eps) * P0[:, :1], rtol=1e-5, atol=1e-7)
    assert (card[:, 1:][ok] >= eps * (1 - 1e-5)).all() and (card[:, 1:][~ok] == 0).all()
    assert torch.allclose(out["card_logp"].exp(), card, atol=1e-6)                     # the returned parts agree
    assert torch.equal(out["pos_logp"], ref["pos_logp"]) and torch.equal(out["mask"], ref["mask"])
    pos = probs[:, 1:].reshape(-1, 4, B) / card[:, 1:, None].clamp_min(1e-30)          # P'(j | s) = P(j | s)
    assert torch.allclose(pos[ok], ref["pos_logp"].exp()[ok], atol=1e-5)
    assert (out["logits"][~out["mask"]] == torch.finfo(torch.float32).min).all()


def test_card_floor_gradients_finite():
    obs = _obs()
    p = _policy()
    p.enable_card_floor(0.02)
    logits, value = p(obs)
    a = torch.distributions.Categorical(logits=logits).sample()
    loss = -torch.log_softmax(logits, -1).gather(1, a[:, None]).sum() + value.sum()
    loss.backward()
    for n, prm in p.named_parameters():
        assert prm.grad is None or torch.isfinite(prm.grad).all(), n
    assert p.slot_logit.bias.grad is not None


def test_card_floor_errors_and_checkpoints(tmp_path):
    from pufferroyale.league import load_policy
    p = _policy()
    with pytest.raises(RuntimeError):
        p.set_card_floor(0.01)
    for bad in (-0.01, 0.25, float("nan"), True, "0.1"):
        with pytest.raises(ValueError):
            p.enable_card_floor(bad)
    assert not p.card_floor_on                                                         # a failed enable leaves it off
    with pytest.raises(ValueError):
        _policy(head="flat").enable_card_floor(0.01)
    p.enable_card_floor(0.03)
    torch.save(p.state_dict(), tmp_path / "floored.pt")
    q, _ = load_policy(str(tmp_path / "floored.pt"))
    assert not q.card_floor_on and float(q.card_floor) == 0.0                          # checkpoints play unfloored


def test_card_floor_schedule():
    import league_train as LT
    a = dict(card_floor=0.005, card_floor_steps=1000)
    assert LT.card_floor_at(a, 0) == 0.005 and LT.card_floor_at(a, 500) == pytest.approx(0.0025)
    assert LT.card_floor_at(a, 1000) == 0.0 and LT.card_floor_at(a, 5000) == 0.0
    assert LT.card_floor_at(dict(card_floor=0.005, card_floor_steps=0), 10 ** 10) == 0.005
    assert LT.card_floor_at(dict(card_floor=0.0, card_floor_steps=10), 0) == 0.0
    for bad in (dict(card_floor=0.25, card_floor_steps=0), dict(card_floor=-1e-3, card_floor_steps=0),
                dict(card_floor=0.01, card_floor_steps=-1)):
        with pytest.raises(SystemExit):
            LT.check_card_floor(bad)


def test_league_learner_and_reference_floors(tmp_path):
    import league_train as LT
    known, rest = LT.make_parser().parse_known_args(
        ["--total-timesteps", "2048", "--num-envs", "8", "--anchors", "none", "--self-play-frac", "0.5",
         "--anchor-frac", "0", "--snapshot-interval", "2", "--mmd-coef", "0.05", "--mmd-ref-interval", "1",
         "--data-dir", str(tmp_path), "--run-id", "fl", "--card-floor", "0.02", "--card-floor-steps", "1024"])
    run = LT.LeagueRun(known, rest)
    try:
        pol, ref = run.policy, run.trainer.ref_policy
        assert pol.card_floor_on and ref.card_floor_on and float(pol.card_floor) == pytest.approx(0.02)
        run.trainer.global_step = 512
        run.trainer.refresh_reference()                                                # load_state_dict: no buffer
        assert run.apply_card_floor() == pytest.approx(0.01)
        assert float(pol.card_floor) == pytest.approx(0.01) and float(ref.card_floor) == pytest.approx(0.01)
    finally:
        run.abort()


def test_league_card_floor_offline(tmp_path):
    pytest.importorskip("wandb")
    from test_stage4 import _wandb_history
    wd = tmp_path / "wandb_dir"
    wd.mkdir()
    env = dict(os.environ, WANDB_MODE="offline", WANDB_DIR=str(wd), WANDB_SILENT="true", WANDB_API_KEY="")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "league_train.py"), "--total-timesteps", str(8 * 64 * 6),
           "--num-envs", "8", "--anchors", "none", "--self-play-frac", "0.5", "--anchor-frac", "0",
           "--snapshot-interval", "2", "--max-snapshots", "4", "--seed", "1", "--data-dir", str(tmp_path),
           "--run-id", "flw", "--mmd-coef", "0.05", "--mmd-ref-interval", "2", "--card-floor", "0.02",
           "--card-floor-steps", "2048", "--env.deck-pool", POOL, "--env.random-deck-frac", "0.5",
           "--env.heldout-decks", HELD, "--wandb", "--wandb-project", "pr-test", "--wandb-group", "g", "--tag", "t",
           "--no-model-upload", "--train.torch-deterministic="]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    import league_train as LT
    hist = [json.loads(line) for line in open(tmp_path / "league" / "flw" / "history.jsonl")]
    a = dict(card_floor=0.02, card_floor_steps=2048)
    assert [h["card_floor"] for h in hist] == pytest.approx([LT.card_floor_at(a, 512 * e) for e in range(6)])
    assert all(math_ok(v) for h in hist for v in h["losses"].values() if v is not None)
    rows = [x for x in _wandb_history(glob.glob(str(wd / "**" / "offline-run-*" / "*.wandb"), recursive=True)[0])
            if "epoch" in x]
    assert [x["card_floor"] for x in rows] == pytest.approx([h["card_floor"] for h in hist])
    below = [x for x in rows if "cards/below_floor" in x]
    assert below, "no epoch logged play rates"
    for x in below:
        rates = [v for k, v in x.items() if k.startswith(LT.PLAY_RATE)]
        assert x["cards/below_floor"] == sum(v < 0.02 for v in rates)


def math_ok(v):
    return v == v and abs(v) != float("inf")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_card_floor_reaches_cuda_graph_replays():
    from pufferroyale.fast_league import install_learner_graph
    obs = _obs().cuda()
    p = _policy().cuda()
    p.enable_card_floor(0.02)                                                          # before the capture
    assert install_learner_graph(p)
    with torch.no_grad():
        g1 = p.forward_eval(obs)[0].clone()                                            # captured
        g1b = p.forward_eval(obs)[0].clone()                                           # replayed
        e1 = type(p).forward_eval(p, obs)[0]
        p.set_card_floor(0.005)
        g2 = p.forward_eval(obs)[0].clone()                                            # replayed, new eps
        e2 = type(p).forward_eval(p, obs)[0]
    assert torch.allclose(g1b, e1, atol=1e-4) and torch.allclose(g2, e2, atol=1e-4)
    assert not torch.allclose(g1, g2, atol=1e-4)
