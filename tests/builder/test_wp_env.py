"""Builder tests for the SPEC §19 env items (v0.5 work package): reward v2 (§19.1), own deck in the
observation (§19.3), the placement grid (§19.4), deck sampling and pufferroyale.decks (§19.5), and
the per-card play-rate logs (§19.7.6) in Royale and LeagueVecEnv."""
import hashlib
import json
import math

import numpy as np
import pytest

import pufferroyale
from pufferroyale import Game, Bot, binding
from pufferroyale import decks as D
from pufferroyale import royale as R

N_TILES = 576


# ------------------------------------------------------------------------------------ helpers
def strip_own_deck(obs):
    """The observation without the §19.3 field = the v0.4 layout."""
    off, n = R.SCALAR_INDEX["own_deck"]
    a = R.SCALAR_OFFSET + off
    return np.concatenate([obs[:, :a], obs[:, a + n:]], axis=1)


def random_legal(obs, rng, p, grid=1):
    m = R.action_mask(obs, grid)
    act = np.zeros(len(obs), np.int32)
    for r in range(len(obs)):
        legal = np.flatnonzero(m[r, 1:]) + 1
        if legal.size and rng.random() < p:
            act[r] = legal[rng.integers(legal.size)]
    return act


def fingerprint(cfg, steps=1300, p=0.3):
    """sha256 of the obs (own_deck removed), rewards, terminals and logs of a seeded run with random
    legal actions (the same procedure produced the v0.4 references below from the v0.4 build)."""
    env = R.Royale(log_interval=50, **cfg)
    obs, _ = env.reset(seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    h = hashlib.sha256()
    h.update(strip_own_deck(obs).tobytes())
    for _ in range(steps):
        m = obs[:, R.MASK_OFFSET:R.MASK_OFFSET + R.MASK_SIZE] > 0.5
        act = np.zeros(len(obs), np.int32)
        for r in range(len(obs)):
            legal = np.flatnonzero(m[r, 1:]) + 1
            if legal.size and rng.random() < p:
                act[r] = legal[rng.integers(legal.size)]
        obs, rew, term, trunc, info = env.step(act)
        for x in (strip_own_deck(obs), rew, term, trunc):
            h.update(x.tobytes())
        for d in info:
            h.update(repr(sorted((k, v) for k, v in d.items() if not k.startswith("cards/"))).encode())
    env.close()
    return h.hexdigest()[:24]


# generated with the v0.4 build (git HEAD b6b63da) by the same fingerprint()
V04 = [
    (dict(num_envs=2, deck0="hog26", deck1="giant", seed=3), "80a1df21b82eef036438b3f5"),
    (dict(num_envs=2, num_agents=1, opponent="heuristic", deck0="random", deck1="random", seed=5),
     "c9dc8adb7bf3b71ca99e4510"),
    (dict(num_envs=1, deck0="golem", deck1="xbow", frame_skip=5, tower_troop0="cannoneer",
          tower_troop1="royal_chef", seed=7), "7ce04272cbd396dbde0e7aff"),
    (dict(num_envs=2, deck0="bait", deck1="royal_hogs", reward_tower=0.0, reward_crown=0.0,
          alternate_first=True, seed=9), "6302528b85b327468d499de8"),
]


def sorted_ids(deck):
    return tuple(sorted(pufferroyale.DECKS[deck] if isinstance(deck, str) else deck))


# ------------------------------------------------------------------------------------ §19.8 defaults
@pytest.mark.parametrize("i", range(len(V04)))
def test_defaults_are_bit_identical_to_v04(i):
    cfg, ref = V04[i]
    assert fingerprint(cfg) == ref


def test_explicit_defaults_and_inactive_features_are_bit_identical():
    """Zero weights with any gamma / anneal, grid 1, a held-out set without an active sampler."""
    cfg, ref = V04[0]
    extra = dict(placement_grid=1, reward_gamma=0.9, shaping_anneal_steps=100, shaping_step_offset=7,
                 reward_elixir_cap=3.0, heldout_decks="xbow;random:3:1", deck_pool="", random_deck_frac=0.0,
                 deck_draw="mirror")
    assert fingerprint(dict(cfg, **extra)) == ref


def test_layout_v05():
    assert R.SCALAR_SIZE == 306 and R.OBS_SIZE == 17715
    assert R.SCALAR_INDEX["own_deck"] == (298, 8)
    assert list(R.SCALAR_INDEX)[-1] == "own_deck"
    assert R.CARD_ID_SCALARS == ("hand", "next_card", "opp_last4", "own_deck")
    assert R.MASK_OFFSET == 14400 + 704 + 306 and R.MASK_SIZE == 2305
    lay = binding.royale_layout()
    assert lay["SCALAR_FIELDS"]["own_deck"] == (298, 8) and lay["OBS_SIZE"] == 17715


# ------------------------------------------------------------------------------------ §19.3 own deck
def test_own_deck_in_game_and_env_obs():
    g = Game(deck0="golem", deck1="xbow", seed=3)
    for team, deck in ((0, "golem"), (1, "xbow")):
        v = R.scalar(g.obs(team), "own_deck")
        assert tuple(int(x) - 1 for x in v) == sorted_ids(deck)
    # the queue order (hidden shuffle) and the opponent's deck never enter
    before = R.scalar(g.obs(0), "own_deck").copy()
    g.set_hand(0, list(reversed(pufferroyale.DECKS["golem"])))
    g.set_hand(1, list(pufferroyale.DECKS["bait"]))
    assert np.array_equal(R.scalar(g.obs(0), "own_deck"), before)
    # env rows, random decks re-drawn per deal; a mirror match gives both seats the same field
    env = R.Royale(num_envs=1, deck0="random", deck1="random", frame_skip=100, seed=2)
    obs, _ = env.reset(seed=2)
    seen = set()
    for _ in range(3):
        decks = env.env_info(0)["decks"]
        for row in (0, 1):
            assert tuple(int(x) - 1 for x in R.scalar(obs[row], "own_deck")) == tuple(sorted(decks[row]))
        seen.add(tuple(sorted(decks[0])))
        for _ in range(60):
            obs, *_ = env.step(np.zeros(2, np.int32))
    assert len(seen) == 3
    env.close()
    env = R.Royale(num_envs=1, deck0="pekka_bridge", deck1="pekka_bridge", seed=1)
    obs, _ = env.reset(seed=1)
    assert np.array_equal(R.scalar(obs[0], "own_deck"), R.scalar(obs[1], "own_deck"))
    env.close()


# ------------------------------------------------------------------------------------ §19.1 reward v2
def run_shaped(gamma, anneal, offset, matches, seed, num_envs=2):
    """Shaped self-play with random legal actions: per-step zero-sum and per-match telescoping."""
    env = R.Royale(num_envs=num_envs, frame_skip=20, reward_tower=0.3, reward_crown=0.2, reward_elixir=0.05,
                   reward_play=0.02, reward_elixir_cap=2.0, reward_play_cap=3.0, reward_gamma=gamma,
                   shaping_anneal_steps=anneal, shaping_step_offset=offset, log_interval=10 ** 9, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rews = [[] for _ in range(num_envs)]
    done, shaped = 0, 0
    while done < matches:
        obs, rew, term, _, infos = env.step(random_legal(obs, rng, 0.4))
        r = rew.reshape(num_envs, 2)
        assert np.all(r[:, 0] == -r[:, 1]) and np.all(r[:, 0] + r[:, 1] == 0.0)
        assert not np.any(np.signbit(r[r == 0.0]))           # +0, as in v0.4
        shaped += int(np.count_nonzero(r[:, 0]))
        peeks = {i: binding.env_log_peek(env._c_env_list[i]) for i in range(num_envs) if term[2 * i]}
        if peeks:
            binding.vec_log(env.c_envs)                  # clear the env logs: one finished match per read
        for i in range(num_envs):
            rews[i].append(float(r[i, 0]))
            if i in peeks:  # sum_t gamma^t r_t (float64 of the stored float32) = gamma^(T-1) result
                res = 1.0 if peeks[i]["win_0"] else (-1.0 if peeks[i]["win_1"] else 0.0)
                x = np.asarray(rews[i], np.float32).astype(np.float64)
                tot = float(np.sum(gamma ** np.arange(len(x)) * x))
                assert abs(tot - gamma ** (len(x) - 1) * res) <= 1e-4, (tot, res, len(x))
                rews[i] = []
                done += 1
    env.close()
    return shaped


def test_reward_v2_zero_sum_and_telescoping_on_real_matches():
    assert run_shaped(1.0, 0, 0, 3, seed=1) > 0
    assert run_shaped(0.999, 0, 0, 3, seed=2) > 0
    assert run_shaped(0.99, 500, 0, 5, seed=3) > 0       # the weights reach 0 inside the matches
    assert run_shaped(0.999, 800, 400, 3, seed=4) > 0    # resumed with an offset


def test_reward_v2_terminal_matches_engine_result_and_episode_return():
    env = R.Royale(num_envs=1, frame_skip=20, reward_tower=0.5, reward_crown=0.3, reward_gamma=0.999,
                   log_interval=10 ** 9, seed=5)
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    ep = []
    for _ in range(2000):
        obs, rew, term, _, _ = env.step(random_legal(obs, rng, 0.5))
        ep.append(float(rew[0]))
        if term[0]:
            break
    log = binding.env_log_peek(env._c_env_list[0])
    res = 1.0 if log["win_0"] else (-1.0 if log["win_1"] else 0.0)
    x = np.asarray(ep, np.float64)
    assert abs(np.sum(0.999 ** np.arange(len(x)) * x) - 0.999 ** (len(x) - 1) * res) <= 1e-4
    assert abs(log["episode_return"] - np.float32(np.sum(np.asarray(ep, np.float32)))) < 1e-4
    env.close()


def test_anneal_counter_and_offset_in_env_info():
    env = R.Royale(num_envs=1, reward_tower=1.0, shaping_anneal_steps=100, shaping_step_offset=20, seed=0)
    env.reset(seed=0)
    for _ in range(30):
        env.step(np.zeros(2, np.int32))
    env.reset(seed=1)                                       # never resets the counter
    info = env.env_info(0)
    assert info["env_steps"] == 30 and info["shaping_multiplier"] == pytest.approx(0.5)
    env.close()


@pytest.mark.parametrize("kw", [
    dict(reward_tower=-0.1), dict(reward_crown=-1.0), dict(reward_elixir=-0.01), dict(reward_play=-1.0),
    dict(reward_tower=float("nan")), dict(reward_elixir_cap=0.0), dict(reward_play_cap=-2.0),
    dict(reward_gamma=0.0), dict(reward_gamma=1.01), dict(reward_gamma=-0.5), dict(shaping_anneal_steps=-1),
    dict(shaping_step_offset=-5), dict(placement_grid=3), dict(placement_grid=0), dict(placement_grid=True),
    dict(deck_draw="both"), dict(random_deck_frac=1.5), dict(random_deck_frac=-0.1),
    dict(random_deck_frac=0.5), dict(deck_pool="hog26;xbow", heldout_decks="xbow"),
    dict(deck_pool="hog26;hog26:2"), dict(deck_pool="hog26:0"), dict(deck_pool="random"),
    dict(deck_pool="nope"), dict(deck_pool="random:3"), dict(heldout_decks="random:2:1;random:2:1"),
])
def test_construction_errors(kw):
    with pytest.raises(ValueError):
        R.Royale(num_envs=1, **kw)


# ------------------------------------------------------------------------------------ §19.4 grid
def test_grid_shapes_and_action_spaces():
    assert [R.grid_shape(g) for g in (1, 2, 4)] == [(32, 18), (16, 9), (8, 5)]
    assert [R.n_actions(g) for g in (1, 2, 4)] == [2305, 577, 161]
    for g in (1, 2, 4):
        env = R.Royale(num_envs=1, placement_grid=g)
        assert env.single_action_space.n == R.n_actions(g) and env.placement_grid == g
        assert env.env_info(0)["placement_grid"] == g and env.env_info(0)["row_grids"] == (g, g)
        env.close()
    with pytest.raises(ValueError):
        R.n_actions(3)


def brute_coarse(fine_mask, g):
    rows, cols = R.grid_shape(g)
    out = np.zeros(1 + 4 * rows * cols, bool)
    out[0] = True
    for s in range(4):
        for by in range(rows):
            for bx in range(cols):
                tiles = [(tx, ty) for ty in range(g * by, min(32, g * by + g)) for tx in range(g * bx, min(18, g * bx + g))]
                out[1 + s * rows * cols + by * cols + bx] = any(fine_mask[1 + s * N_TILES + ty * 18 + tx] for tx, ty in tiles)
    return out


def brute_fine(fine_mask, a, g):
    rows, cols = R.grid_shape(g)
    s, b = divmod(a - 1, rows * cols)
    by, bx = divmod(b, cols)
    x0, x1, y0, y1 = g * bx, min(18, g * bx + g), g * by, min(32, g * by + g)
    cands = [((2 * tx + 1 - x0 - x1) ** 2 + (2 * ty + 1 - y0 - y1) ** 2, ty, tx)
             for ty in range(y0, y1) for tx in range(x0, x1) if fine_mask[1 + s * N_TILES + ty * 18 + tx]]
    if not cands:
        return 0
    _, ty, tx = min(cands)
    return 1 + s * N_TILES + ty * 18 + tx


def test_coarse_mask_and_decoding_exhaustive_on_real_states():
    g0 = Game(deck0="pekka_bridge", deck1="royal_hogs", seed=11, deploy_lockout_ticks=0)
    bots = [Bot("heuristic", 1), Bot("random", 2, play_prob=0.5)]
    checked = 0
    for k in range(40):
        for team in (0, 1):
            fine = g0.legal_mask(team)
            obs = g0.obs(team)
            for g in (1, 2, 4):
                cm = R.action_mask(obs, g)
                assert cm.dtype == bool and cm.shape == (R.n_actions(g),)
                assert np.array_equal(cm, brute_coarse(fine, g))
                for a in range(R.n_actions(g)):
                    f = g0.coarse_to_fine(team, a, g)
                    assert f == (brute_fine(fine, a, g) if a else 0)
                    if cm[a] and a:
                        assert fine[f] == 1                  # every coarse-legal action maps to a legal play
                        checked += 1
                assert g0.coarse_to_fine(team, R.n_actions(g), g) == 0
        for team in (0, 1):
            a = bots[team].act(g0, team)
            if a:
                g0.play_action(team, a)
        g0.tick(15)
        if g0.state()["over"]:
            break
    assert checked > 3000
    # batched / leading shapes
    o = np.stack([g0.obs(0), g0.obs(1)])[None]
    assert R.action_mask(o, 4).shape == (1, 2, 161)


@pytest.mark.parametrize("g", [2, 4])
def test_env_plays_every_coarse_legal_action(g):
    env = R.Royale(num_envs=2, placement_grid=g, deck0="miner_poison", deck1="lavaloon", frame_skip=10,
                   log_interval=1, seed=g)
    obs, _ = env.reset(seed=g)
    rng = np.random.default_rng(g)
    plays = illegal = 0
    for _ in range(700):
        obs, rew, term, _, infos = env.step(random_legal(obs, rng, 0.5, grid=g))
        for d in infos:
            plays += d["plays_0"] * d["n"] + d["plays_1"] * d["n"]
            illegal += d["illegal_actions"] * d["n"]
    env.close()
    assert plays > 50 and illegal == 0


def test_illegal_coarse_actions_counted():
    env = R.Royale(num_envs=1, placement_grid=4, frame_skip=300, log_interval=1, seed=0)
    env.reset(seed=0)
    total = 0.0
    for _ in range(25):
        _, _, term, _, infos = env.step(np.array([R.n_actions(4), 10 ** 6], np.int32))  # out of range
        for d in infos:
            total += d["illegal_actions"]
    assert total == 40.0                                   # 2 per step over one 20-step match
    env.close()


def hashes(env):
    return [env.env_info(i)["hash"] for i in range(env.num_envs_native)]


def test_single_agent_bot_stays_fine_under_coarse_grids():
    """The scripted opponent of a 1-agent env plays exactly as with grid 1 (learner idle)."""
    trails = []
    for g in (1, 2, 4):
        env = R.Royale(num_envs=2, num_agents=1, opponent="heuristic", placement_grid=g, seed=4, log_interval=1)
        env.reset(seed=4)
        t = []
        for _ in range(650):
            _, rew, _, _, infos = env.step(np.zeros(2, np.int32))
            t.append((tuple(hashes(env)), rew.tobytes()))
            assert all(d["illegal_actions"] == 0 for d in infos)
        trails.append(t)
        env.close()
    assert trails[0] == trails[1] == trails[2]


def test_league_bot_opponent_plays_fine_actions_under_coarse_grids():
    import pufferroyale.league as L
    trails = []
    for g in (1, 4):
        pool = L.OpponentPool(anchors=("bot:heuristic", "bot:random"), self_play_frac=0.0, anchor_frac=1.0, seed=0)
        lv = L.LeagueVecEnv(pool, num_envs=2, seed=3, placement_grid=g, bot_play_prob=0.5, log_interval=1)
        lv.async_reset(3)
        assert lv.single_action_space.n == R.n_actions(g)
        t, illegal, plays = [], 0.0, 0.0
        for _ in range(700):
            for i in range(2):
                rg = lv.env.env_info(i)["row_grids"]
                assert rg[int(lv.seats[i])] == g and rg[1 - int(lv.seats[i])] == 1
            lv.send(np.zeros(2, np.int32))
            _, rew, term, _, infos, _, _ = lv.recv()
            t.append((tuple(hashes(lv.env)), rew.tobytes(), tuple(lv.seats), tuple(lv.opponents)))
            for d in infos:
                illegal += d["illegal_actions"]
                plays += d["plays_0"] + d["plays_1"]
        lv.close()
        assert illegal == 0 and plays > 0
        trails.append(t)
    assert trails[0] == trails[1], "a bot: opponent must play exactly as in v0.4 under placement_grid > 1"


def test_league_learner_coarse_actions_vs_bot():
    import pufferroyale.league as L
    pool = L.OpponentPool(anchors=("bot:random",), self_play_frac=0.0, anchor_frac=1.0, seed=1)
    lv = L.LeagueVecEnv(pool, num_envs=2, seed=1, placement_grid=2, bot_play_prob=0.5, log_interval=1)
    lv.async_reset(1)
    rng = np.random.default_rng(1)
    obs = lv.observations
    illegal = 0.0
    for _ in range(700):
        lv.send(random_legal(obs, rng, 0.5, grid=2))
        obs, _, _, _, infos, _, _ = lv.recv()
        illegal += sum(d["illegal_actions"] for d in infos)
    lv.close()
    assert illegal == 0


# ------------------------------------------------------------------------------------ §19.5 decks.py
def test_parse_deck_set_forms(tmp_path):
    hog, giant = sorted_ids("hog26"), sorted_ids("giant")
    assert D.parse_deck_set("") == [] and D.parse_deck_set(None) == [] and D.parse_deck_set([]) == []
    assert D.parse_deck_set(" hog26 ; giant : 2.5 ;") == [(hog, 1.0), (giant, 2.5)]
    rnd = D.random_decks(3, 7)
    assert D.parse_deck_set("random:3:7") == [(d, 1.0) for d in rnd]
    knight8 = ["Knight", "Archers", "Musketeer", "Giant", "Hog Rider", "Minions", "Baby Dragon", "Valkyrie"]
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(["bait", knight8, {"deck": "golem", "weight": 3}, {"deck": list(range(8, 16))}]))
    got = D.parse_deck_set(f"file:{path}; xbow:0.5")
    assert got == [(sorted_ids("bait"), 1.0), (tuple(range(8)), 1.0), (sorted_ids("golem"), 3.0),
                   (tuple(range(8, 16)), 1.0), (sorted_ids("xbow"), 0.5)]
    lst = D.parse_deck_set(["hog26", "giant:2", list(range(16, 24)), {"deck": "bait", "weight": 4}, "random:2:1"])
    assert lst[:4] == [(hog, 1.0), (giant, 2.0), (tuple(range(16, 24)), 1.0), (sorted_ids("bait"), 4.0)]
    assert D.parse_deck_set(lst) == lst                       # its own output parses to itself
    for bad in ("random", "hog26:x", "hog26:-1", "hog26:inf", "random:2", "random:a:1", "random:0:1",
                "file:", f"file:{tmp_path / 'missing.json'}", "hog26:1:2", "nope", 5, [list(range(7))],
                [[0, 0, 1, 2, 3, 4, 5, 6]], [{"deck": "hog26", "w": 1}], "hog26;giant;hog26:3",
                [list(range(8)), list(reversed(range(8)))]):
        with pytest.raises(ValueError):
            D.parse_deck_set(bad)


def test_random_decks_and_deck_key():
    a = D.random_decks(200, 3)
    assert a == D.random_decks(200, 3) and a[:50] == D.random_decks(50, 3) and a != D.random_decks(200, 4)
    assert len(set(a)) == 200 and all(len(d) == 8 and list(d) == sorted(set(d)) for d in a)
    counts = np.bincount(np.concatenate(a), minlength=64)
    assert counts.sum() == 1600 and counts.min() > 5 and counts.max() < 50   # all 64 cards, roughly uniform
    assert D.random_decks(0, 1) == []
    assert D.deck_key("hog26") == D.deck_key(list(reversed(pufferroyale.DECKS["hog26"])))
    assert D.deck_key("hog26") == "-".join(map(str, sorted_ids("hog26")))
    assert D.deck_key("hog26") != D.deck_key("giant")
    with pytest.raises(ValueError):
        D.random_decks(-1, 0)


# ------------------------------------------------------------------------------------ §19.5 sampler
def test_sampler_statistics_and_counts():
    env = R.Royale(num_envs=4, deck_pool="hog26:1;giant:3", random_deck_frac=0.25, heldout_decks="xbow;random:20:5",
                   seed=1)
    held = {sorted_ids("xbow")} | set(D.random_decks(20, 5))
    deals = 4                                                # construction
    for k in range(1500):
        env.reset(seed=1 if k == 0 else None)                # every reset re-deals every match
        deals += 4
        for i in range(4):
            info = env.env_info(i)
            assert info["deck_sampler"] and info["deck0_ignored"] and info["deck1_ignored"]
            assert not {tuple(sorted(d)) for d in info["decks"]} & held
    env.step(np.zeros(8, np.int32))
    c = env.deck_counts()
    assert c["pool"].dtype == np.int64 and c["pool"].shape == (2,)
    seats = 2 * deals
    assert int(c["pool"].sum()) + c["random"] == seats and c["random_rejected"] == 0
    for got, p in ((c["random"], 0.25), (c["pool"][0], 0.75 / 4), (c["pool"][1], 0.75 * 3 / 4)):
        assert abs(got - p * seats) < 5 * math.sqrt(p * (1 - p) * seats), (got, p * seats)
    env.close()


def test_sampler_mirror_heldout_rejection_and_determinism():
    env = R.Royale(num_envs=2, deck_pool="hog26;giant;bait", random_deck_frac=0.5, deck_draw="mirror", seed=4)
    env.reset(seed=4)
    for _ in range(20):
        for i in range(2):
            d = env.env_info(i)["decks"]
            assert sorted(d[0]) == sorted(d[1])
        env.reset()
    c = env.deck_counts()
    assert c["random"] % 2 == 0 and all(x % 2 == 0 for x in c["pool"])
    env.close()
    # a random draw equal to a held-out deck is redrawn: hold out the first deck seed 6 deals
    first = R.Royale(num_envs=1, random_deck_frac=1.0, seed=6)
    d0 = tuple(sorted(first.env_info(0)["decks"][0]))
    first.close()
    env = R.Royale(num_envs=1, random_deck_frac=1.0, heldout_decks=[list(d0)], seed=6)
    assert tuple(sorted(env.env_info(0)["decks"][0])) != d0
    assert env.deck_counts()["random_rejected"] == 1
    env.close()
    # determinism: same seed -> same deals; reset(seed) reproduces them
    seqs = []
    for _ in range(2):
        env = R.Royale(num_envs=2, deck_pool="hog26;golem", random_deck_frac=0.5, seed=9)
        s = []
        for k in range(6):
            env.reset(seed=9 if k == 3 else None)
            s.append([env.env_info(i)["decks"] for i in range(2)])
        seqs.append(s)
        env.close()
    assert seqs[0] == seqs[1] and seqs[0][3] == seqs[0][0]


def test_sampler_leaves_the_game_stream_alone():
    """The sampler has its own stream (ruling v0.5-G.1 on the installed order): a pool of the preset
    'hog26' deals exactly like deck0 = deck1 = 'hog26'; a pool deck given as a card list is installed
    ascending, so it deals like the sorted list given as deck0 / deck1."""
    pre = list(pufferroyale.DECKS["hog26"])
    asc = sorted(pre)
    assert pre != asc
    fixed_pre = fingerprint(dict(num_envs=2, deck0="hog26", deck1="hog26", seed=3), steps=700)
    fixed_asc = fingerprint(dict(num_envs=2, deck0=asc, deck1=asc, seed=3), steps=700)
    assert fixed_pre != fixed_asc                             # the order matters to the shuffle
    for spec in ("hog26", "hog26:2", ["hog26"], [{"deck": "hog26"}]):
        assert fingerprint(dict(num_envs=2, deck_pool=spec, deck0="giant", deck1="random", seed=3), steps=700) == fixed_pre
    assert fingerprint(dict(num_envs=2, deck_pool=[pre], seed=3), steps=700) == fixed_asc
    # 1-agent mode with a random bot and a random learner side: the bot / side streams are untouched too
    kw = dict(num_envs=2, num_agents=1, opponent="random", bot_play_prob=0.5, seed=4)
    assert fingerprint(dict(kw, deck_pool="xbow"), steps=400) == fingerprint(dict(kw, deck0="xbow", deck1="xbow"), steps=400)


def test_deck_entries_installed_order(tmp_path):
    """Ruling v0.5-G.1: a deck named by a preset keeps the preset's card order; every other deck
    (card lists, random:N:SEED, file card lists) is ascending; parse_deck_set stays ascending."""
    path = tmp_path / "d.json"
    path.write_text(json.dumps(["bait", {"deck": "golem", "weight": 2}, list(range(15, 7, -1))]))
    ents = D.deck_entries(f"hog26;giant:3;random:1:2;file:{path}")
    orders = [o for _, _, o in ents]
    assert orders[0] == pufferroyale.DECKS["hog26"] and orders[1] == pufferroyale.DECKS["giant"]
    assert orders[2] == D.random_decks(1, 2)[0]
    assert orders[3] == pufferroyale.DECKS["bait"] and orders[4] == pufferroyale.DECKS["golem"]
    assert orders[5] == tuple(range(8, 16))
    assert [(d, w) for d, w, _ in ents] == D.parse_deck_set(f"hog26;giant:3;random:1:2;file:{path}")
    assert all(tuple(sorted(o)) == d for d, _, o in ents)
    lst = D.deck_entries([list(pufferroyale.DECKS["xbow"]), ("royal_hogs", 2.0)])
    assert lst[0][2] == tuple(sorted(pufferroyale.DECKS["xbow"])) and lst[1][2] == pufferroyale.DECKS["royal_hogs"]
    env = R.Royale(num_envs=1, deck_pool="pekka_bridge", deck_draw="mirror", seed=1)
    env.reset(seed=1)
    assert env.env_info(0)["decks"] == [list(pufferroyale.DECKS["pekka_bridge"])] * 2
    assert env.deck_pool == [(sorted_ids("pekka_bridge"), 1.0)]
    env.close()


def test_sampler_inactive_reports():
    env = R.Royale(num_envs=1, seed=0)
    info = env.env_info(0)
    assert not info["deck_sampler"] and not info["deck0_ignored"]
    c = env.deck_counts()
    assert c["pool"].shape == (0,) and c["random"] == 0 and c["random_rejected"] == 0
    env.close()


# ------------------------------------------------------------------------------------ §19.7.6 card stats
def expected_counts(obs, act, grid):
    avail, played = {}, {}
    hand = R.SCALAR_INDEX["hand"][0]
    aff = R.SCALAR_INDEX["affordable"][0]
    lock = R.SCALAR_INDEX["lockout"][0]
    cm = R.action_mask(obs, grid)
    B = (R.n_actions(grid) - 1) // 4
    for r in range(len(obs)):
        sc = obs[r, R.SCALAR_OFFSET:]
        for s in range(4):
            c = int(sc[hand + s]) - 1
            if c >= 0 and sc[aff + s] == 1.0 and sc[lock] == 0.0:
                avail[c] = avail.get(c, 0) + 1
        a = int(act[r])
        if 1 <= a < R.n_actions(grid) and cm[r, a]:
            c = int(sc[hand + (a - 1) // B]) - 1
            played[c] = played.get(c, 0) + 1
    return avail, played


@pytest.mark.parametrize("grid", [1, 2])
def test_royale_card_stats(grid):
    env = R.Royale(num_envs=2, placement_grid=grid, deck0="hog26", deck1="random", frame_skip=60, log_interval=50,
                   seed=grid)
    obs, _ = env.reset(seed=grid)
    rng = np.random.default_rng(grid)
    avail, played, dec, emitted = {}, {}, 0, 0
    for t in range(400):
        act = random_legal(obs, rng, 0.3, grid)
        if t % 7 == 0:
            act[0] = rng.integers(R.n_actions(grid))           # often masked-out: never counted as played
        a, p = expected_counts(obs, act, grid)
        for k, v in a.items():
            avail[k] = avail.get(k, 0) + v
        for k, v in p.items():
            played[k] = played.get(k, 0) + v
        dec += len(obs)
        obs, _, _, _, infos = env.step(act)
        for d in infos:
            emitted += 1
            assert d["cards/decisions"] == dec
            rates = {k[len("cards/play_rate/"):]: v for k, v in d.items() if k.startswith("cards/play_rate/")}
            assert rates == pytest.approx({pufferroyale.CARD_KEYS[c]: played.get(c, 0) / n for c, n in avail.items()})
            avail, played, dec = {}, {}, 0
    assert emitted >= 3
    env.close()


def test_league_card_stats_count_learner_rows_only():
    import pufferroyale.league as L
    pool = L.OpponentPool(anchors=("bot:heuristic",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.LeagueVecEnv(pool, num_envs=2, seed=2, deck0="hog26", deck1="xbow", frame_skip=60, log_interval=50)
    lv.async_reset(2)
    rng = np.random.default_rng(2)
    obs = lv.observations.copy()
    avail, played, dec, emitted = {}, {}, 0, 0
    for _ in range(300):
        act = random_legal(obs, rng, 0.4)
        a, p = expected_counts(obs, act, 1)
        for k, v in a.items():
            avail[k] = avail.get(k, 0) + v
        for k, v in p.items():
            played[k] = played.get(k, 0) + v
        dec += len(obs)
        lv.send(act)
        o, _, _, _, infos, _, _ = lv.recv()
        obs = o.copy()
        for d in infos:
            emitted += 1
            assert d["cards/decisions"] == dec
            rates = {k[len("cards/play_rate/"):]: v for k, v in d.items() if k.startswith("cards/play_rate/")}
            assert rates == pytest.approx({pufferroyale.CARD_KEYS[c]: played.get(c, 0) / n for c, n in avail.items()})
            avail, played, dec = {}, {}, 0
    lv.close()
    assert emitted >= 3
