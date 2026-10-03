"""Builder tests for the second-audit fixes to the training tooling (SPEC v0.4.1 §18.6-18.7, audit
items M2-M6, L5-L12 and the nits). CPU only; the league runs are tiny."""
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest
import torch

import pufferroyale
import pufferroyale.torch  # noqa: F401  (pufferroyale.torch.Policy / Recurrent)
from pufferroyale import royale as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
TINY = ["--num-envs", "2", "--bptt-horizon", "16", "--frame-skip", "50", "--anchors", "bot:random"]


def ppo_cfg(data_dir, batch=64, horizon=16, minibatch=16, total=64 * 50, **extra):
    cfg = dict(env="pufferroyale", seed=1, torch_deterministic=True, cpu_offload=False, device="cpu",
               optimizer="adam", precision="float32", total_timesteps=total, learning_rate=3e-4, anneal_lr=False,
               min_lr_ratio=0.0, gamma=0.995, gae_lambda=0.9, update_epochs=1, clip_coef=0.2, vf_coef=2.0,
               vf_clip_coef=0.2, max_grad_norm=1.5, ent_coef=0.001, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
               data_dir=str(data_dir), checkpoint_interval=10 ** 9, batch_size=batch, minibatch_size=minibatch,
               max_minibatch_size=minibatch, bptt_horizon=horizon, compile=False, compile_mode="default",
               compile_fullgraph=False, vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.8, prio_beta0=0.2,
               use_rnn=False, name="t", project="t", amp=True)
    cfg.update(extra)
    return cfg


def quiet(cls):
    return type(cls.__name__, (cls,), {"print_dashboard": lambda self, *a, **k: None})


def epochs(tr, n):
    for _ in range(n):
        tr.last_log_time = -1e18
        tr.evaluate()
        tr.train()


def run_script(args, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def strict_json(line):
    def bad(c):
        raise ValueError(f"non-standard JSON constant {c}")
    return json.loads(line, parse_constant=bad)


def lt():
    sys.path.insert(0, SCRIPTS)
    try:
        import league_train
    finally:
        sys.path.pop(0)
    return league_train


def non_daemon_threads():
    return [t for t in threading.enumerate() if t is not threading.main_thread() and not t.daemon and t.is_alive()]


# ------------------------------------------------------------------------------------ M2 bf16 / autocast
@pytest.mark.parametrize("mmd_coef", [0.0, 1.0])
def test_bf16_training_updates_the_live_forward(tmp_path, mmd_coef):
    """SPEC §18.6: under bf16 autocast (CPU) the forward after a few updates must reflect the new
    weights: KL(live || initial) > 0, and no autocast state leaks out of train()."""
    from pufferroyale.trainer import MMDPuffeRL, masked_kl, obs_mask
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0)
    torch.manual_seed(0)
    pol = pufferroyale.torch.Policy(env)
    init = copy.deepcopy(pol)
    tr = quiet(MMDPuffeRL)(ppo_cfg(tmp_path, learning_rate=3e-3, precision="bfloat16", mmd_coef=mmd_coef,
                                   mmd_ref_interval=0), env, pol)
    try:
        assert isinstance(tr.amp_context, torch.autocast) and tr.amp_context._cache_enabled is False
        epochs(tr, 3)
        assert not torch.is_autocast_enabled("cpu"), "train() must not leave autocast entered"
        obs = tr.observations.reshape(-1, R.OBS_SIZE)[:64]
        m = obs_mask(obs)
        with torch.no_grad(), tr.amp_context:
            live, _ = pol(obs)
            ini, _ = init(obs)
            kl = masked_kl(live, ini, m).item()
            if mmd_coef:
                ref, _ = tr.ref_policy(obs)
                assert masked_kl(live, ref, m).item() > 1e-5
        assert kl > 1e-5, f"the live bf16 forward still equals the initial policy (KL {kl})"
        assert all(math.isfinite(float(v)) for k, v in tr.losses.items() if k != "explained_variance")
    finally:
        tr.utilization.stop()
        env.close()


def test_float32_has_no_autocast_and_monitor_is_daemon(tmp_path):
    import contextlib
    from pufferroyale.trainer import MMDPuffeRL
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0)
    tr = quiet(MMDPuffeRL)(ppo_cfg(tmp_path), env, pufferroyale.torch.Policy(env))
    try:
        assert isinstance(tr.amp_context, contextlib.nullcontext)
        assert tr.utilization.daemon, "PuffeRL's monitor must not keep the process alive (SPEC §18.7)"
    finally:
        tr.utilization.stop()
        env.close()


def test_trainer_rejects_total_smaller_than_batch(tmp_path):
    from pufferroyale.trainer import MMDPuffeRL
    env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=10, seed=0)
    before = set(non_daemon_threads())
    with pytest.raises(ValueError, match="batch_size"):
        quiet(MMDPuffeRL)(ppo_cfg(tmp_path, total=32), env, pufferroyale.torch.Policy(env))
    env.close()
    assert set(non_daemon_threads()) <= before


def test_all_scripts_use_the_fixed_trainer():
    for script in ("train.py", "league_train.py", "best_response.py"):
        src = open(os.path.join(SCRIPTS, script)).read()
        assert "pufferl.PuffeRL(" not in src and "PuffeRL if" not in src, f"{script} still builds the base class"
    assert "MMDPuffeRL(trainer_cfg, vecenv, policy, logger)" in open(os.path.join(SCRIPTS, "train.py")).read()


# ------------------------------------------------------------------------------------ M3 non-finite learners
def test_nan_epoch_is_never_saved(tmp_path):
    out = run_script(["scripts/league_train.py", "--total-timesteps", "256", *TINY, "--snapshot-interval", "1",
                      "--learning-rate", "1e12", "--data-dir", str(tmp_path), "--run-id", "nan", "--seed", "0"])
    assert out.returncode == 1, (out.stdout + out.stderr)[-2000:]
    summary = strict_json(out.stdout.strip().splitlines()[-1])       # valid JSON despite NaN losses
    run = tmp_path / "league" / "nan"
    diag = json.load(open(run / "nonfinite.json"))
    bad_epoch = diag["epoch"]
    assert summary["model"] is None and summary["nonfinite_losses"]
    assert not (run / f"snap_{bad_epoch}.pt").exists(), "the non-finite epoch was snapshotted"
    assert not list(run.glob("model_*.pt")), "no final model after a non-finite epoch"
    for f in list(run.glob("snap_*.pt")) + ([run / "learner.pt"] if (run / "learner.pt").exists() else []):
        sd = torch.load(f, weights_only=True)
        assert all(torch.isfinite(v).all() for v in sd.values()), f"{f.name} holds non-finite weights"
    if (run / "league_state.json").exists():
        st = json.load(open(run / "league_state.json"))
        assert st["epoch"] < bad_epoch and st["epoch"] == diag["last_good_save"]["epoch"]


# ------------------------------------------------------------------------------------ resume (M4, M5, M6, L8, L9)
@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("audit2")
    out = run_script(["scripts/league_train.py", "--total-timesteps", "128", *TINY, "--snapshot-interval", "1",
                      "--data-dir", str(d), "--run-id", "r1", "--seed", "3", "--train.adam-eps", "1e-6"])
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    return d / "league" / "r1"


def copy_run(base_run, tmp_path, name="run"):
    dst = tmp_path / name
    shutil.copytree(base_run, dst)
    return dst


def test_failed_resume_exits_promptly(base_run, tmp_path):
    bad = copy_run(base_run, tmp_path)
    os.remove(bad / "learner.pt")
    out = run_script(["scripts/league_train.py", "--resume", str(bad), "--total-timesteps", "256"], timeout=90)
    assert out.returncode != 0 and "learner.pt" in out.stderr
    LT = lt()
    known, rest = LT.make_parser().parse_known_args(["--resume", str(bad), "--total-timesteps", "256"])
    with pytest.raises(FileNotFoundError):
        LT.LeagueRun(known, rest)
    assert not non_daemon_threads(), "a failed resume left a non-daemon thread running"


def test_resume_applies_cli_optimizer_settings_and_restores_state(base_run, tmp_path):
    run_dir = copy_run(base_run, tmp_path)
    st = json.load(open(run_dir / "league_state.json"))
    LT = lt()
    known, rest = LT.make_parser().parse_known_args(["--resume", str(run_dir), "--total-timesteps", "256",
                                                     "--learning-rate", "1e-5", "--train.adam-beta1", "0.8"])
    run = LT.LeagueRun(known, rest)
    try:
        g = run.trainer.optimizer.param_groups[0]
        assert g["lr"] == 1e-5 and g["initial_lr"] == 1e-5 and g["betas"][0] == 0.8
        assert g["eps"] == 1e-6, "--train.* overrides of the original run are kept on resume"
        assert run.trainer.epoch == st["epoch"] and run.trainer.global_step == st["global_step"]
        assert run.lv.episodes == st["episodes"]
        assert run.lv.rng_state() == st["rng"]["league"], "the wrapper's RNG state is restored"
    finally:
        run.abort()


def test_enabling_mmd_at_resume_uses_the_loaded_learner(base_run, tmp_path):
    run_dir = copy_run(base_run, tmp_path)
    learner = torch.load(run_dir / "learner.pt", weights_only=True)
    LT = lt()
    known, rest = LT.make_parser().parse_known_args(["--resume", str(run_dir), "--total-timesteps", "256",
                                                     "--mmd-coef", "0.1"])
    run = LT.LeagueRun(known, rest)
    try:
        ref = run.trainer.ref_policy.state_dict()
        assert max((ref[k].cpu() - learner[k]).abs().max().item() for k in learner) == 0.0
        assert any("mmd_coef changes" in n for n in run.notes)
    finally:
        run.abort()


def test_run_dir_is_relocatable(base_run, tmp_path):
    moved = copy_run(base_run, tmp_path, "somewhere_else")
    st = json.load(open(moved / "league_state.json"))
    assert st["pool"]["snapshots"] and all(not os.path.isabs(s[5:]) for s in st["pool"]["snapshots"])
    out = run_script(["scripts/league_train.py", "--resume", str(moved), "--total-timesteps", "192"])
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    s = strict_json(out.stdout.strip().splitlines()[-1])
    assert s["global_step"] == 192 and s["run_dir"] == str(moved)


def test_exception_during_training_saves_nothing(tmp_path, monkeypatch):
    LT = lt()
    boom = RuntimeError("simulated failure")

    def train(self):
        self.trainer.evaluate()                              # rollout steps counted, never trained on
        raise boom

    monkeypatch.setattr(LT.LeagueRun, "train", train)
    monkeypatch.setattr(sys, "argv", ["league_train.py", "--total-timesteps", "128", *TINY, "--data-dir",
                                      str(tmp_path), "--run-id", "x"])
    with pytest.raises(RuntimeError, match="simulated"):
        LT.main()
    run = tmp_path / "league" / "x"
    assert not (run / "learner.pt").exists() and not (run / "league_state.json").exists()
    assert not non_daemon_threads()


def test_anneal_lr_continues_across_resume(tmp_path):
    """The cosine schedule after load_training_state equals the uninterrupted one."""
    from pufferroyale.trainer import MMDPuffeRL

    def make():
        env = pufferroyale.Royale(num_envs=2, num_agents=2, frame_skip=50, seed=0)
        torch.manual_seed(0)
        return quiet(MMDPuffeRL)(ppo_cfg(tmp_path, batch=64, minibatch=64, total=64 * 8, anneal_lr=True,
                                         learning_rate=1e-3), env, pufferroyale.torch.Policy(env))

    a = make()
    lrs, saved = [], None
    for e in range(8):
        epochs(a, 1)
        lrs.append(a.optimizer.param_groups[0]["lr"])
        if e == 3:
            saved = (copy.deepcopy(a.optimizer.state_dict()), a.epoch, a.global_step)
    a.utilization.stop()
    a.vecenv.close()
    b = make()
    b.load_training_state(*saved)
    assert abs(b.optimizer.param_groups[0]["lr"] - lrs[3]) < 1e-12
    resumed = []
    for _ in range(4):
        epochs(b, 1)
        resumed.append(b.optimizer.param_groups[0]["lr"])
    b.utilization.stop()
    b.vecenv.close()
    assert np.allclose(resumed, lrs[4:], rtol=1e-9, atol=1e-15), (resumed, lrs[4:])


# ------------------------------------------------------------------------------------ L5 / L6 / L12 CLI checks
@pytest.mark.parametrize("args", [
    ["scripts/league_train.py", "--total-timesteps", "10", *TINY],
    ["scripts/train.py", "--total-timesteps", "10", "--num-envs", "1"],
])
def test_total_timesteps_below_one_batch_is_a_clear_error(tmp_path, args):
    out = run_script(args + ["--data-dir", str(tmp_path)], timeout=120)
    assert out.returncode != 0 and "smaller than one PPO batch" in out.stderr, (out.stdout + out.stderr)[-1500:]
    assert not (tmp_path / "league").exists() or not any((tmp_path / "league").iterdir())


def test_best_response_accepts_run_dirs_ckpt_and_comma_decks(base_run, tmp_path):
    deck = "Hog Rider,Musketeer,Cannon,Ice Golem,Ice Spirit,Skeletons,Fireball,The Log"
    out_file = tmp_path / "br.json"
    out = run_script(["scripts/best_response.py", "--target", f"ckpt:{base_run}", "--total-timesteps", "32",
                      "--num-envs", "2", "--bptt-horizon", "16", "--matches", "2", "--deck0", deck, "--out", str(out_file)])
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    res = strict_json(out.stdout.strip().splitlines()[-1])
    assert res["target"].endswith("model_000004.pt") and res["env_settings"]["frame_skip"] == 50
    for bad, msg in ((["--matches", "0"], "--matches must be >= 1"), (["--total-timesteps", "5"], "PPO batch")):
        out = run_script(["scripts/best_response.py", "--target", str(base_run), *bad], timeout=120)
        assert out.returncode != 0 and msg in out.stderr, out.stderr[-800:]


def test_deck_values_accept_names_and_ids():
    LT = lt()
    assert LT.deck_value("hog26") == "hog26"
    assert LT.deck_value("4, 2,14,10,11,9,16,19") == [4, 2, 14, 10, 11, 9, 16, 19]
    assert LT.deck_value("Knight,Archers") == ["Knight", "Archers"]
    sys.path.insert(0, SCRIPTS)
    try:
        import train
    finally:
        sys.path.pop(0)
    assert train.deck_value("0,1,2,3,4,5,6,7") == list(range(8))
    env = pufferroyale.Royale(num_envs=1, num_agents=2, deck0=train.deck_value("0,1,2,3,4,5,6,7"))
    env.close()


# ------------------------------------------------------------------------------------ L7 training env settings
@pytest.fixture
def fs50_ckpt(tmp_path):
    run = tmp_path / "run_fs50"
    run.mkdir()
    env = pufferroyale.Royale(num_envs=1, num_agents=2)
    torch.manual_seed(0)
    torch.save(pufferroyale.torch.Policy(env).state_dict(), run / "model_000001.pt")
    env.close()
    json.dump({"env": {"frame_skip": 50, "deploy_lockout_ticks": 60, "tiebreak": "fraction",
                       "tower_troop0": "cannoneer", "tower_troop1": "cannoneer"}}, open(run / "config.json", "w"))
    return run


def test_play_match_uses_the_checkpoint_training_settings(fs50_ckpt):
    from pufferroyale import metagame as mg
    ck = str(fs50_ckpt / "model_000001.pt")
    with warnings.catch_warnings():
        warnings.simplefilter("error", mg.MatchSettingsWarning)
        _, info = mg.play_match(ck, "bot:noop", "hog26", "hog26", seed=1, return_info=True)
    assert info["settings"]["frame_skip"] == [50, 10]
    assert info["settings"]["game"] == {"deploy_lockout_ticks": 60, "tiebreak": "fraction",
                                        "tower_troop0": "cannoneer", "tower_troop1": "cannoneer"}
    with pytest.warns(mg.MatchSettingsWarning, match="frame_skip=50"):
        _, info = mg.play_match(ck, "bot:noop", "hog26", "hog26", seed=1, return_info=True, frame_skip=20)
    assert info["settings"]["frame_skip"] == [20, 20]
    _, info = mg.play_match(f"ckpt:{fs50_ckpt}", "bot:noop", "hog26", "hog26", seed=1, return_info=True,
                            use_train_config=False)
    assert info["settings"]["frame_skip"] == [10, 10] and info["settings"]["game"]["tiebreak"] == "absolute"


def test_tournament_and_eval_report_training_settings(fs50_ckpt, tmp_path):
    out_file = tmp_path / "t.json"
    out = run_script(["scripts/tournament.py", "--agents", "bot:noop", str(fs50_ckpt), "--decks", "hog26", "hog26",
                      "--matches", "2", "--out", str(out_file)])
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    t = json.load(open(out_file))
    assert t["decks"] == ["hog26"], "repeated decks are played once"
    assert t["match_settings"]["frame_skip"] == [10, 50] and t["match_settings"]["game"]["tiebreak"] == "fraction"
    ev = run_script(["scripts/eval.py", "--checkpoint", str(fs50_ckpt), "--episodes", "1", "--bots", "noop",
                     "--seats", "0", "--json", str(tmp_path / "e.json")])
    assert ev.returncode == 0, (ev.stdout + ev.stderr)[-2000:]
    assert json.load(open(tmp_path / "e.json"))["env_settings"]["frame_skip"] == 50


def test_llm_checkpoint_opponent_uses_its_training_cadence(fs50_ckpt):
    from pufferroyale import llm
    from pufferroyale.llm import _make_opponent
    _, k = _make_opponent(str(fs50_ckpt), 0, 1, None, True, "cpu")
    assert k == 50
    _, k = _make_opponent("bot:heuristic", 0, 1, None, True, "cpu")
    assert k == 10
    assert llm.play_llm_match(llm.LLMAgent(llm.mock_wait), str(fs50_ckpt), "hog26", "hog26", seed=0,
                              max_ticks=200)["ticks"] == 200


# ------------------------------------------------------------------------------------ L9 / L10 / L11 / nits
def test_results_are_bounded_and_drained():
    import pufferroyale.league as L
    pool = L.OpponentPool(anchors=("bot:noop",), self_play_frac=0.0, anchor_frac=1.0)
    lv = L.LeagueVecEnv(pool, num_envs=2, seed=0, frame_skip=100, results_maxlen=3)
    lv.async_reset(0)
    for _ in range(62 * 3):
        lv.send(np.zeros(2, np.int32))
    assert lv.episodes >= 5 and len(lv.results) == 3
    got = lv.drain_results()
    assert len(got) == 3 and len(lv.results) == 0
    lv.close()


def test_recurrent_keeps_the_policy_init():
    env = pufferroyale.Royale(num_envs=1, num_agents=2)
    torch.manual_seed(0)
    for head, finals in (("flat", ("actor",)), ("conditional", ("wait_logit", "slot_logit", "pos_out"))):
        p = pufferroyale.torch.Policy(env, head=head)
        before = {k: v.clone() for k, v in p.state_dict().items()}
        r = pufferroyale.torch.Recurrent(env, p)
        after = r.policy.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before), "LSTMWrapper re-initialised the Policy"
        for name in finals:
            assert getattr(r.policy, name).weight.std() < 0.05, f"{name} keeps its std-0.01 init"
    env.close()


def test_bot_play_prob_is_clamped_and_seeds_do_not_collide():
    import pufferroyale.league as L
    pool = L.OpponentPool(anchors=("bot:random",), self_play_frac=0.0, anchor_frac=1.0)
    for prob, ppm in ((5.0, 1_000_000), (-1.0, 0), (0.25, 250_000)):
        lv = L.LeagueVecEnv(pool, num_envs=1, seed=0, bot_play_prob=prob)
        assert lv.bot_ppm == ppm
        lv.close()

    def seats(seed, reset):
        lv = L.LeagueVecEnv(L.OpponentPool(anchors=("bot:noop",), self_play_frac=0.0, anchor_frac=1.0), num_envs=1,
                            seed=seed)
        lv.async_reset(reset)
        out = [int(lv._rng.integers(1 << 30)) for _ in range(4)]
        lv.close()
        return out

    assert seats(0, 5) != seats(2 ** 32, 5), "seeds differing by 2**32 must not collide"
    assert seats(-1, 5) != seats(1, 5) and seats(3, None) != seats(3, 0)


def test_learner_rows_still_match_after_unbuffered_take():
    import pufferroyale.league as L
    pool = L.OpponentPool(anchors=("bot:random",), self_play_frac=0.0, anchor_frac=1.0, seed=1)
    lv = L.LeagueVecEnv(pool, num_envs=3, seed=2, frame_skip=100)
    lv.async_reset(2)
    for _ in range(80):
        rows = 2 * np.arange(3) + lv.seats
        lv.send(np.zeros(3, np.int32))
        assert np.array_equal(lv.rewards, lv.env.rewards[rows])
        assert np.array_equal(lv.observations, lv.env.observations[2 * np.arange(3) + lv.seats])
    lv.close()


def test_docs_and_templates_nits():
    heads = re.findall(r"^### (D\d+) ", open(os.path.join(ROOT, "docs", "DECISIONS.md")).read(), re.M)
    dupes = sorted({h for h in heads if heads.count(h) > 1})
    assert not dupes, f"duplicate decision ids {dupes}"
    sb = open(os.path.join(ROOT, "hpc", "league.sbatch")).read()
    assert '--decks "${DECKS[@]}"' in sb and 'if [ "$DECK1" != "$DECK0" ]' in sb
    assert "card ids/names" not in open(os.path.join(SCRIPTS, "train.py")).read()
