"""Builder tests, SPEC §19.11 (v0.5-G.5): the card-first greedy rule (league.greedy_actions) and
sampling as the default of every tool that plays a policy (watch.py, tournament.py, metagame,
LLM-match policy opponents); --greedy / greedy=True routes through the one rule everywhere."""
import importlib
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import pufferroyale
from pufferroyale import royale as R

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
F32_MIN = float(np.finfo(np.float32).min)
WIDTHS = (2305, 577, 161)


def L():
    import pufferroyale.league as mod
    return mod


def run(script, *args, check=True, timeout=600):
    p = subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True, timeout=timeout)
    if check:
        assert p.returncode == 0, f"{script} {args}:\n{(p.stdout + p.stderr)[-3000:]}"
    return p


def last_json(text):
    return json.loads([ln for ln in text.strip().splitlines() if ln.strip()][-1])


def reference(row):
    """The §19.11.1 rule written out directly: normalised softmax in float64, marginals, then the
    best tile of the chosen slot."""
    row = np.asarray(row, dtype=np.float64)
    A = row.size
    B = (A - 1) // 4
    legal = row > F32_MIN
    p = np.zeros(A)
    p[legal] = np.exp(row[legal] - row[legal].max())
    p /= p.sum()
    marg = [p[0]] + [p[1 + s * B:1 + (s + 1) * B].sum() for s in range(4)]
    c = int(np.argmax(marg))
    if c == 0:
        return 0
    seg = np.where(legal[1 + (c - 1) * B:1 + c * B], row[1 + (c - 1) * B:1 + c * B], -np.inf)
    return 1 + (c - 1) * B + int(np.argmax(seg))


def masked(A, rng, p_legal=0.3, scale=2.0):
    x = rng.normal(0, scale, A).astype(np.float32)
    keep = rng.random(A) < p_legal
    keep[0] = True
    x[~keep] = F32_MIN
    return x


def make_flat_ckpt(path, noop_bias=3.0, seed=0):
    """A small flat-head checkpoint whose wait logit is raised by `noop_bias`: the plain argmax is
    always wait, while the card-first rule (and sampling) play cards once a slot has many legal tiles."""
    from pufferroyale.torch import Policy
    env = pufferroyale.Royale(num_envs=1, num_agents=2)
    torch.manual_seed(seed)
    pol = Policy(env, head="flat", hidden_size=64, cnn_channels=16, entity_hidden=32, scalar_hidden=32)
    with torch.no_grad():
        pol.actor.bias[0] += noop_bias
    env.close()
    torch.save(pol.state_dict(), str(path))
    return str(path)


@pytest.fixture(scope="module")
def flat_ckpt(tmp_path_factory):
    return make_flat_ckpt(tmp_path_factory.mktemp("greedy") / "noopish.pt")


@pytest.fixture
def spy(monkeypatch):
    """Counts calls of league.greedy_actions (every greedy path must go through it)."""
    mod = L()
    real = mod.greedy_actions
    calls = []

    def wrapped(logits):
        out = real(logits)
        calls.append(np.asarray(out).copy())
        return out
    monkeypatch.setattr(mod, "greedy_actions", wrapped)
    return calls


# ------------------------------------------------------------------------------------ the rule
@pytest.mark.parametrize("A", WIDTHS)
def test_spread_probability_picks_the_card_not_wait(A):
    """Wait is the single most likely action, but slot 1 holds more mass over its tiles."""
    B = (A - 1) // 4
    x = np.full(A, F32_MIN, np.float32)
    x[0] = 1.0
    x[1 + B:1 + 2 * B] = 0.0                    # every tile of slot 1 legal, each less likely than wait
    x[1 + B + 7] = 0.5                          # its best tile
    assert int(x.argmax()) == 0                 # the plain argmax plays nothing
    assert int(L().greedy_actions(x)) == 1 + B + 7 == reference(x)


@pytest.mark.parametrize("A", WIDTHS)
def test_wait_when_its_marginal_is_largest(A):
    B = (A - 1) // 4
    x = np.zeros(A, np.float32)
    x[0] = np.log(4 * B) + 1.0                  # P(wait) > every P(s), although every tile is legal
    assert int(L().greedy_actions(x)) == 0 == reference(x)
    x[0] = np.log(B) - 1.0                      # now each slot outweighs wait: slot 0 (earliest), j = 0
    assert int(L().greedy_actions(x)) == 1


@pytest.mark.parametrize("A", WIDTHS)
def test_ties(A):
    B = (A - 1) // 4
    g = L().greedy_actions
    x = np.full(A, F32_MIN, np.float32)
    x[0] = 0.0
    x[1 + 2 * B + 3] = 0.0                      # P(wait) == P(slot 2) exactly -> wait
    assert int(g(x)) == 0
    x[0] = F32_MIN                              # (wait masked: a degenerate row) -> slot 2
    assert int(g(x)) == 1 + 2 * B + 3
    y = np.full(A, F32_MIN, np.float32)
    y[0] = -5.0
    for s in (1, 3):                            # identical slots 1 and 3 -> slot 1
        y[1 + s * B + 4] = y[1 + s * B + 9] = 2.0   # tile tie inside the slot -> the smaller j
    assert int(g(y)) == 1 + B + 4
    z = np.zeros(A, np.float32)                 # all equal: slot 0 has B times wait's mass; j = 0
    assert int(g(z)) == 1


@pytest.mark.parametrize("A", WIDTHS)
def test_wait_only_rows_and_fully_masked_slots(A):
    B = (A - 1) // 4
    x = np.full((3, A), F32_MIN, np.float32)
    x[:, 0] = [-50.0, 0.0, 30.0]                # only wait legal: always 0, whatever its logit
    assert L().greedy_actions(x).tolist() == [0, 0, 0]
    x[1, 1 + 3 * B + 5] = -60.0                 # one legal tile far below wait: still wait
    x[2, 1 + 3 * B:1 + 4 * B] = 28.0            # slot 3 fully legal, each tile below wait: slot 3 wins
    assert L().greedy_actions(x).tolist() == [0, 0, 1 + 3 * B]
    assert L().greedy_actions(np.full(A, F32_MIN, np.float32)) == 0      # nothing legal at all -> 0


@pytest.mark.parametrize("A", WIDTHS)
def test_matches_the_reference_and_never_illegal(A):
    rng = np.random.default_rng(A)
    for p_legal, scale in ((0.02, 1.0), (0.3, 2.0), (0.9, 0.1), (0.001, 5.0)):
        x = np.stack([masked(A, rng, p_legal, scale) for _ in range(64)])
        a = L().greedy_actions(x)
        assert a.dtype == np.int32 and a.shape == (64,)
        assert all(x[i, a[i]] > F32_MIN for i in range(64)), "an illegal action was chosen"
        assert a.tolist() == [reference(r) for r in x]


@pytest.mark.parametrize("A", WIDTHS)
def test_batch_shapes_dtypes_and_inputs(A):
    rng = np.random.default_rng(1)
    x = np.stack([masked(A, rng) for _ in range(6)])
    g = L().greedy_actions
    want = g(x)
    assert g(x.reshape(2, 3, A)).shape == (2, 3) and g(x.reshape(2, 3, A)).ravel().tolist() == want.tolist()
    one = g(x[2])
    assert one.shape == () and int(one) == int(want[2])
    assert g(np.zeros((0, A), np.float32)).shape == (0,)
    assert g(torch.as_tensor(x)).tolist() == want.tolist()                       # torch input
    assert g(x.astype(np.float64)).tolist() == want.tolist()                     # float32 mask in float64
    t64 = torch.as_tensor(x, dtype=torch.float64).masked_fill(torch.as_tensor(x) <= F32_MIN,
                                                              torch.finfo(torch.float64).min)
    assert g(t64).tolist() == want.tolist()
    xb = torch.as_tensor(x).bfloat16().masked_fill(torch.as_tensor(x) <= F32_MIN, torch.finfo(torch.bfloat16).min)
    ab = g(xb)
    assert all(x[i, ab[i]] > F32_MIN for i in range(6))                          # bf16: still legal


def test_bad_widths_raise():
    for A in (4, 2304, 10):
        with pytest.raises(ValueError):
            L().greedy_actions(np.zeros(A))


def test_select_actions_greedy_uses_the_rule_and_sampling_is_unchanged():
    rng = np.random.default_rng(3)
    x = torch.as_tensor(np.stack([masked(2305, rng) for _ in range(8)]))
    assert np.array_equal(L().select_actions(x, True, None), L().greedy_actions(x))
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    a = L().select_actions(x, False, g1)
    b = torch.multinomial(torch.softmax(x.float(), -1), 1, generator=g2).squeeze(-1).numpy()
    assert a.dtype == np.int32 and np.array_equal(a, b)


@pytest.mark.parametrize("grid", [1, 2, 4])
def test_real_policy_logits_are_legal(grid):
    from pufferroyale.torch import Policy
    env = pufferroyale.Royale(num_envs=4, num_agents=2, placement_grid=grid, deploy_lockout_ticks=0, seed=grid)
    obs, _ = env.reset(seed=grid)
    torch.manual_seed(0)
    pol = Policy(env, hidden_size=64).eval()
    with torch.no_grad():
        for _ in range(20):
            logits, _ = pol.forward_eval(torch.as_tensor(obs), {})
            a = L().greedy_actions(logits)
            m = R.action_mask(obs, grid)
            assert all(m[i, a[i]] for i in range(len(a)))
            assert a.tolist() == [reference(r) for r in logits.double().numpy()]
            obs, *_ = env.step(a)
    env.close()


def test_noop_biased_policy_plain_argmax_waits_but_card_first_plays(flat_ckpt):
    pol, _ = L().load_policy(flat_ckpt)
    g = pufferroyale.Game(deck0="hog26", deck1="giant", seed=0)
    g.tick(200)
    with torch.no_grad():
        logits, _ = pol.forward_eval(torch.as_tensor(np.stack([g.obs(0), g.obs(1)])), {})
    assert logits.argmax(-1).tolist() == [0, 0]
    a = L().greedy_actions(logits)
    assert all(a > 0) and all(g.legal_mask(t)[a[t]] for t in (0, 1))


# ------------------------------------------------------------------------------------ defaults
def test_library_defaults_are_sampling():
    from pufferroyale import llm, metagame
    for fn in (metagame.play_match, metagame.deck_metagame, llm.play_llm_match):
        assert inspect.signature(fn).parameters["greedy"].default is False, fn.__name__
    assert inspect.signature(L().LeagueVecEnv.__init__).parameters["opponent_greedy"].default is False


def test_scripts_help_advertise_greedy_flags():
    for script, flag in (("watch.py", "--greedy"), ("tournament.py", "--greedy"), ("tournament.py", "--sample"),
                         ("eval.py", "--greedy"), ("best_response.py", "--greedy"),
                         ("league_train.py", "--opponent-greedy"), ("llm_match.py", "--opponent-greedy")):
        text = run(script, "--help").stdout
        assert flag in text, f"{script} --help lacks {flag}"
        if flag != "--sample":
            assert "card-first" in text, f"{script} --help does not name the card-first rule"


# ------------------------------------------------------------------------------------ routing
def test_metagame_play_match_routes_greedy(spy, flat_ckpt):
    from pufferroyale import metagame as mg
    r1 = mg.play_match(flat_ckpt, "bot:noop", "hog26", "hog26", seed=3)
    assert spy == [], "play_match samples by default"
    assert r1 == mg.play_match(flat_ckpt, "bot:noop", "hog26", "hog26", seed=3)       # seeded
    mg.play_match(flat_ckpt, "bot:noop", "hog26", "hog26", seed=3, greedy=True)
    assert spy and any(int(a[0]) > 0 for a in spy), "greedy=True goes through greedy_actions and plays cards"
    n = len(spy)
    P, x = mg.deck_metagame(flat_ckpt, ["hog26", "giant"], matches=1, seed=0)
    assert len(spy) == n
    mg.deck_metagame(flat_ckpt, ["hog26", "giant"], matches=1, seed=0, greedy=True)
    assert len(spy) > n


def test_league_opponent_greedy_routes_and_plays(spy, flat_ckpt):
    lv = L().LeagueVecEnv(L().OpponentPool(anchors=(f"ckpt:{flat_ckpt}",), self_play_frac=0.0, anchor_frac=1.0,
                                           seed=0),
                          num_envs=2, seed=0, opponent_greedy=True, frame_skip=50)
    lv.async_reset(0)
    for _ in range(40):
        lv.recv()
        lv.send(np.zeros(2, np.int32))                  # the learner never plays
    lv.close()
    assert spy, "opponent_greedy=True must use greedy_actions"
    assert sum(int((a > 0).sum()) for a in spy) > 0, "the card-first opponent plays cards (the argmax would not)"


def test_llm_policy_opponent_samples_by_default(spy, flat_ckpt):
    from pufferroyale import llm
    r = llm.play_llm_match(llm.LLMAgent(llm.mock_wait), flat_ckpt, "hog26", "hog26", seed=0, max_ticks=600)
    assert spy == [] and r["ticks"] == 600
    llm.play_llm_match(llm.LLMAgent(llm.mock_wait), flat_ckpt, "hog26", "hog26", seed=0, max_ticks=600, greedy=True)
    assert spy and any(int(a[0]) > 0 for a in spy)


def test_eval_play_routes_greedy_and_is_seeded(spy, flat_ckpt):
    sys.path.insert(0, SCRIPTS)
    ev = importlib.import_module("eval")
    pol, _, rec = ev.load_policy(flat_ckpt, "cpu")
    kw = dict(deck0="hog26", deck1="hog26", device="cpu", seed=4, num_envs=2,
              env_settings={"frame_skip": 50, "placement_grid": 1})
    s1 = ev.play(pol, rec, "noop", 0, 2, greedy=False, **kw)
    assert spy == [] and s1 == ev.play(pol, rec, "noop", 0, 2, greedy=False, **kw)
    ev.play(pol, rec, "noop", 0, 2, greedy=True, **kw)
    assert spy and sum(int((a > 0).sum()) for a in spy) > 0


# ------------------------------------------------------------------------------------ scripts
def _watch(ckpt, *extra):
    p = run("watch.py", "--checkpoint", ckpt, "--p1", "random", "--fps", 0, "--no-clear", "--every", 1000,
            "--max-steps", 120, *extra)
    return p.stdout.strip().splitlines()[-1]


def test_watch_samples_by_default_greedy_plays_cards_and_seeds_reproduce(flat_ckpt):
    import re
    s0 = _watch(flat_ckpt, "--seed", 1)
    assert "sampled" in s0 and int(re.search(r"(\d+) plays", s0).group(1)) > 0
    assert _watch(flat_ckpt, "--seed", 1) == s0, "a given --seed reproduces the match"
    gr = _watch(flat_ckpt, "--seed", 1, "--greedy")
    assert "greedy" in gr and int(re.search(r"(\d+) plays", gr).group(1)) > 0, \
        "card-first greedy plays cards with a wait-biased policy (the plain argmax played none)"


def test_tournament_samples_by_default_and_greedy_flag(flat_ckpt, tmp_path):
    base = ["--agents", "bot:noop", flat_ckpt, "--decks", "hog26", "--matches", 2, "--quiet", "--frame-skip", 50]
    outs = {}
    for name, extra in (("default", []), ("again", []), ("sample", ["--sample"]), ("greedy", ["--greedy"]),
                        ("both", ["--greedy", "--sample"])):
        f = tmp_path / f"{name}.json"
        p = run("tournament.py", *base, *extra, "--out", f)
        outs[name] = json.load(open(f))
        assert ("card-first greedy" if outs[name]["greedy"] else "sampled") in p.stdout
    assert outs["default"]["greedy"] is False and outs["sample"]["greedy"] is False
    assert outs["greedy"]["greedy"] is True and outs["both"]["greedy"] is True
    assert outs["default"]["results"] == outs["again"]["results"] == outs["sample"]["results"]


def test_llm_match_opponent_greedy_flag(flat_ckpt, tmp_path):
    common = ["--model", "mock_wait", "--opponent", flat_ckpt, "--matches", 1, "--max-ticks", 400]
    assert last_json(run("llm_match.py", *common).stdout)["opponent_greedy"] is False
    assert last_json(run("llm_match.py", *common, "--opponent-greedy").stdout)["opponent_greedy"] is True


def test_best_response_greedy_flag_recorded(flat_ckpt):
    common = ["--target", flat_ckpt, "--total-timesteps", 32, "--num-envs", 2, "--bptt-horizon", 16, "--matches", 2,
              "--frame-skip", 50]
    assert last_json(run("best_response.py", *common).stdout)["greedy"] is False
    assert last_json(run("best_response.py", *common, "--greedy").stdout)["greedy"] is True
