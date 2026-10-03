"""SPEC §19.5 deck sampling (v0.5-G).

- Env kwargs deck_pool (deck-set, default empty), random_deck_frac ([0, 1], default 0), heldout_decks
  (deck-set), deck_draw ("independent" | "mirror").
- Deck-set: string of `;`-separated items (whitespace ignored): PRESET or PRESET:WEIGHT (a DECKS name other
  than random; weight > 0, default 1), random:N:SEED (random_decks(N, SEED), weight 1 each), file:PATH
  (JSON list of preset names, 8-card lists of names/ids, or {"deck": preset|list, "weight": w}); or a
  Python list of the same element forms. Two equal decks (as sets) -> ValueError.
- pufferroyale.decks: parse_deck_set(spec) -> [(deck, weight)] (deck = ascending tuple of 8 ids);
  random_decks(n, seed) (n distinct uniform 8-card decks of all N_CARDS; pure function); deck_key(deck).
- Sampler active iff deck_pool non-empty or random_deck_frac > 0 (deck0/deck1 then ignored); inactive ->
  bit-identical to v0.4. Every deal (construction, reset, auto-reset): team 0 then team 1 (mirror: team 1
  takes team 0's deck): w.p. random_deck_frac a uniformly random deck, redrawn while it equals a held-out
  deck, else a pool deck w.p. proportional to weight; a dedicated PCG32 stream (reseeded by reset(seed));
  the game RNG is not used by the sampler.
- Construction errors (ValueError); capacity >= 256 pool / 1024 held-out decks.
- deck_counts() -> {"pool": int64 (len(pool),), "random": int, "random_rejected": int}, cumulative since
  construction: +1 per seat per pool deck (mirror deals count both seats), +1 per seat with a random deck,
  +1 per rejected random draw.
"""
import json
import math

import numpy as np
import pytest

import envkit as E
import helpers as H
import leaguekit as L
import v05kit as V


def D():
    import pufferroyale.decks as mod
    return mod


def preset(name):
    import pufferroyale as p
    return tuple(sorted(int(c) for c in p.DECKS[name]))


def names(deck):
    import pufferroyale as p
    return [p.CARD_NAMES[c] for c in deck]


def counts(env):
    c = env.deck_counts()
    assert set(c) >= {"pool", "random", "random_rejected"}, f"deck_counts keys {sorted(c)}"
    pool = np.asarray(c["pool"])
    assert pool.dtype == np.int64, f"deck_counts()['pool'] dtype {pool.dtype} (SPEC: int64)"
    assert isinstance(c["random"], (int, np.integer)) and isinstance(c["random_rejected"], (int, np.integer))
    return pool.copy(), int(c["random"]), int(c["random_rejected"])


def chi2_ok(obs, exp, alpha=1e-6):
    from scipy.stats import chi2
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    stat = float(np.sum((obs - exp) ** 2 / exp))
    return stat, float(chi2.ppf(1 - alpha, len(obs) - 1))


# ==========================================================================================
# pufferroyale.decks
# ==========================================================================================
def test_parse_preset_items_and_weights(pr):
    out = D().parse_deck_set(" hog26 : 2 ;giant;  bait:0.5 ")
    assert [tuple(d) for d, _ in out] == [preset("hog26"), preset("giant"), preset("bait")], "order and decks"
    assert [float(w) for _, w in out] == [2.0, 1.0, 0.5]
    for d, _ in out:
        assert isinstance(d, tuple) and len(d) == 8 and list(d) == sorted(d) and all(isinstance(c, int) for c in d)
    assert list(D().parse_deck_set("")) == [] and list(D().parse_deck_set([])) == []


def test_parse_random_and_file_items(pr, tmp_path):
    rd = D().random_decks(3, 7)
    out = D().parse_deck_set("random:3:7")
    assert [tuple(d) for d, _ in out] == [tuple(d) for d in rd] and all(float(w) == 1.0 for _, w in out)
    xb = preset("xbow")
    spec = ["golem", names(preset("lavaloon")), list(preset("royal_hogs"))[::-1],
            {"deck": "xbow", "weight": 3}, {"deck": names(preset("pekka_bridge"))[::-1], "weight": 0.25},
            {"deck": list(preset("miner_poison")), "weight": 2}]
    f = tmp_path / "decks.json"
    f.write_text(json.dumps(spec))
    out = D().parse_deck_set(f"bait:4; file:{f}; random:2:11")
    want = [(preset("bait"), 4.0), (preset("golem"), 1.0), (preset("lavaloon"), 1.0), (preset("royal_hogs"), 1.0),
            (xb, 3.0), (preset("pekka_bridge"), 0.25), (preset("miner_poison"), 2.0)] + \
           [(tuple(d), 1.0) for d in D().random_decks(2, 11)]
    assert [(tuple(d), float(w)) for d, w in out] == want
    assert [(tuple(d), float(w)) for d, w in D().parse_deck_set(spec)] == want[1:7], "Python list form"


BAD_SPECS = ["random", "random:3", "random:x:1", "nosuchdeck", "hog26:0", "hog26:-1", "hog26:abc",
             "hog26;hog26", "hog26;giant:2;hog26:3",
             [["Knight"] * 8], [list(range(7))], [list(range(9))], [["Knight", "Archers", "Musketeer", "Giant",
                                                                    "Hog Rider", "Minions", "Baby Dragon", "Nope"]],
             [{"deck": "hog26", "weight": 0}], [{"deck": "hog26", "weight": -2}], ["hog26", list(preset("hog26"))[::-1]],
             [list(range(60, 68))]]


def test_python_list_may_hold_string_items(pr):
    """§19.9.1: a Python deck-set list may also hold PRESET:W and random:N:SEED strings."""
    out = D().parse_deck_set(["hog26:2", "random:2:5", "giant"])
    want = [(preset("hog26"), 2.0)] + [(tuple(d), 1.0) for d in D().random_decks(2, 5)] + [(preset("giant"), 1.0)]
    assert [(tuple(d), float(w)) for d, w in out] == want


@pytest.mark.parametrize("spec", BAD_SPECS, ids=[repr(s)[:40] for s in BAD_SPECS])
def test_parse_rejects_bad_specs(pr, spec):
    """§19.5: bad syntax, a `random` preset item, weight <= 0, not 8 distinct real cards, duplicates
    (equal as sets) -> ValueError."""
    with pytest.raises(ValueError):
        D().parse_deck_set(spec)


def test_parse_file_duplicate_and_bad_json(pr, tmp_path):
    f = tmp_path / "dup.json"
    f.write_text(json.dumps([names(preset("hog26"))[::-1]]))
    with pytest.raises(ValueError):
        D().parse_deck_set(f"hog26;file:{f}")
    g = tmp_path / "bad.json"
    g.write_text(json.dumps({"deck": "hog26"}))                 # not a list
    with pytest.raises(ValueError):
        D().parse_deck_set(f"file:{g}")


def test_random_decks_pure_distinct_uniform(pr):
    import random as pyrandom
    np.random.seed(1)
    pyrandom.seed(1)
    a = D().random_decks(3000, 5)
    np.random.seed(999)
    pyrandom.seed(999)
    b = D().random_decks(3000, 5)
    assert [tuple(x) for x in a] == [tuple(x) for x in b], "random_decks must be a pure function of (n, seed)"
    assert len(a) == 3000
    for d in a:
        assert len(d) == 8 and len(set(d)) == 8 and list(d) == sorted(d) and all(0 <= c < 64 for c in d)
    assert len({tuple(d) for d in a}) == 3000, "the n decks are distinct"
    assert [tuple(x) for x in D().random_decks(10, 6)] != [tuple(x) for x in a[:10]], "seed changes the decks"
    freq = np.bincount(np.concatenate([np.asarray(d) for d in a]), minlength=64)
    stat, crit = chi2_ok(freq, np.full(64, 3000 * 8 / 64))
    assert stat < crit, f"card frequencies not uniform over all 64 cards: chi2 {stat:.1f} > {crit:.1f}"


def test_deck_key_is_a_set_key(pr):
    k = D().deck_key
    d = preset("hog26")
    assert k(d) == k(tuple(reversed(d))) == k(list(d[3:]) + list(d[:3]))
    assert k(d) != k(preset("giant"))
    hash(k(d))


# ==========================================================================================
# Env: inactive sampler, construction errors, counts
# ==========================================================================================
def _trace(env, steps, seed, p=0.4):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    o, r = [obs.copy()], []
    for _ in range(steps):
        obs, rew, *_ = E.step(env, [E.random_bot_action(rng, x, p) for x in obs])
        o.append(obs.copy())
        r.append(rew.copy())
    return np.array(o), np.array(r)


def test_inactive_sampler_is_v04(pr):
    """Inactive (empty pool, frac 0) -> bit-identical to v0.4 even with held-out decks / mirror set."""
    base = dict(num_envs=2, num_agents=2, deck0="giant", deck1="bait", frame_skip=50, seed=3)
    ref = _trace(E.make(**base), 300, 3)
    for kw in (dict(deck_pool="", random_deck_frac=0.0, heldout_decks="", deck_draw="independent"),
               dict(deck_pool=[], heldout_decks="xbow;random:5:1", deck_draw="mirror")):
        env = E.make(**base, **kw)
        got = _trace(env, 300, 3)
        assert all(np.array_equal(a, b) for a, b in zip(ref, got)), f"inactive sampler {kw} changed the env"
        env.close()


def test_active_sampler_ignores_deck0_deck1(pr):
    env = E.make(num_envs=2, num_agents=2, deck0="xbow", deck1="giant", deck_pool="golem", frame_skip=100, seed=1)
    o, _ = _trace(env, 120, 1)
    for frame in o:
        assert all(V.own_deck(r) == preset("golem") for r in frame)
    env.close()


def test_single_deck_pool_leaves_the_game_streams_untouched(pr):
    """The sampler uses its own stream, never the game RNG; ruling v0.5-G.1: a pool item given as a
    preset name is installed in that preset's card order, so a one-preset pool deals exactly like
    deck0 = deck1 = that preset (whole trajectory identical, incl. a random bot and the learner side);
    every other deck is installed ascending, so a one-list pool (given in any order) deals exactly like
    deck0 = deck1 = the ascending list."""
    shuffled = [19, 2, 16, 4, 11, 9, 14, 10]                       # hog26's ids, not ascending
    for kw in (dict(num_agents=2), dict(num_agents=1, opponent="random", learner_side="random", bot_play_prob=0.5)):
        for fixed, pool in (("hog26", "hog26"), (sorted(shuffled), [shuffled]),
                            (list(D().random_decks(1, 4)[0]), "random:1:4")):
            a = E.make(num_envs=2, deck0=fixed, deck1=fixed, frame_skip=50, seed=4, **kw)
            b = E.make(num_envs=2, deck0="xbow", deck1="giant", deck_pool=pool, frame_skip=50, seed=4, **kw)
            ta, tb = _trace(a, 250, 4), _trace(b, 250, 4)
            assert all(np.array_equal(x, y) for x, y in zip(ta, tb)), \
                f"{kw}: pool {pool!r} does not deal like deck0 = deck1 = {fixed!r}"
            a.close()
            b.close()


BAD_ENV = [
    dict(deck_pool="hog26", heldout_decks="giant;hog26"),
    dict(deck_pool="hog26", random_deck_frac=-0.1),
    dict(deck_pool="hog26", random_deck_frac=1.5),
    dict(random_deck_frac=0.5),                                   # frac < 1 with an empty pool
    dict(deck_pool="hog26", deck_draw="sometimes"),
    dict(deck_pool="hog26:"),
    dict(deck_pool="nosuch"),
    dict(deck_pool="hog26", heldout_decks="random:4"),
    dict(deck_pool="hog26;hog26"),
]


@pytest.mark.parametrize("kw", BAD_ENV, ids=[repr(k)[:50] for k in BAD_ENV])
def test_construction_errors(pr, kw):
    with pytest.raises(ValueError):
        E.make(**kw)


def test_pool_equal_to_heldout_as_sets_is_rejected(pr, tmp_path):
    with pytest.raises(ValueError):
        E.make(deck_pool="hog26", heldout_decks=[names(preset("hog26"))[::-1]])
    with pytest.raises(ValueError):
        E.make(deck_pool=["bait", list(preset("giant"))], heldout_decks="giant")


def test_bad_deck_draw_rejected_even_when_inactive(pr):
    """§19.5 lists 'bad deck_draw' among the construction errors unconditionally."""
    with pytest.raises(ValueError):
        E.make(deck_draw="sometimes")


def test_frac_one_with_empty_pool_and_capacity(pr):
    env = E.make(num_envs=1, random_deck_frac=1.0)
    pool, rnd, rej = counts(env)
    assert len(pool) == 0 and rnd == 2, f"construction deals both seats a random deck: random={rnd}"
    env.close()
    big_pool = [list(d) for d in D().random_decks(256, 1)]
    held = [list(d) for d in D().random_decks(1024, 2)]
    assert not ({D().deck_key(d) for d in big_pool} & {D().deck_key(d) for d in held})
    env = E.make(num_envs=1, deck_pool=big_pool, heldout_decks=held, random_deck_frac=0.3)
    pool, rnd, rej = counts(env)
    assert len(pool) == 256 and int(pool.sum()) + rnd == 2
    env.close()
    env = E.make(num_envs=1, deck_pool="random:256:1", heldout_decks="random:1024:2")
    assert len(counts(env)[0]) == 256
    env.close()


# ==========================================================================================
# Sampling statistics (>= 1e5 seat draws)
# ==========================================================================================
POOL = "hog26:1; giant:2; bait:3; golem:4"
WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0])
POOL_DECKS = ["hog26", "giant", "bait", "golem"]


def _deal_many(mode, frac, n_envs=64, resets=800, heldout="xbow;random:20:99"):
    env = E.make(num_envs=n_envs, num_agents=2, deck_pool=POOL, random_deck_frac=frac, heldout_decks=heldout,
                 deck_draw=mode, seed=12345)
    pool0, rnd0, _ = counts(env)
    assert int(pool0.sum()) + rnd0 == 2 * n_envs, "construction is one deal of both seats of every env"
    held = {D().deck_key(d) for d, _ in D().parse_deck_set(heldout)}
    index = {preset(n): i for i, n in enumerate(POOL_DECKS)}
    tally = np.zeros(len(POOL_DECKS), np.int64)
    tally_rnd, pairs_equal, rnd_decks = 0, 0, []
    obs, _ = env.reset(seed=0)
    base = counts(env)
    for k in range(1, resets + 1):
        obs, _ = env.reset(seed=7919 * k + 1)
        for i in range(n_envs):
            d0, d1 = V.own_deck(obs[2 * i]), V.own_deck(obs[2 * i + 1])
            pairs_equal += int(d0 == d1)
            for d in (d0, d1):
                assert D().deck_key(d) not in held, f"held-out deck {d} was dealt"
                if d in index:
                    tally[index[d]] += 1
                else:
                    tally_rnd += 1
                    rnd_decks.append(d)
    pool1, rnd1, rej1 = counts(env)
    env.close()
    return dict(tally=tally, tally_rnd=tally_rnd, dpool=pool1 - base[0], drnd=rnd1 - base[1], rej=rej1,
                seats=2 * n_envs * resets, pairs_equal=pairs_equal, rnd_decks=rnd_decks, n_envs=n_envs,
                resets=resets)


@pytest.fixture(scope="module")
def independent_stats():
    return _deal_many("independent", 0.2)


def test_counts_match_observed_deals(pr, independent_stats):
    s = independent_stats
    assert np.array_equal(s["dpool"], s["tally"]), f"deck_counts pool {s['dpool']} != observed {s['tally']}"
    assert s["drnd"] == s["tally_rnd"], f"deck_counts random {s['drnd']} != observed {s['tally_rnd']}"
    assert int(s["tally"].sum()) + s["tally_rnd"] == s["seats"] >= 100_000


def test_pool_and_random_frequencies(pr, independent_stats):
    """P(seat gets pool deck i) = (1 - f) w_i / sum(w); P(random) = f (chi-square over >= 1e5 seats)."""
    s = independent_stats
    n, f = s["seats"], 0.2
    exp = np.concatenate([n * (1 - f) * WEIGHTS / WEIGHTS.sum(), [n * f]])
    stat, crit = chi2_ok(np.concatenate([s["tally"], [s["tally_rnd"]]]), exp)
    assert stat < crit, f"pool/random frequencies off: chi2 {stat:.1f} > {crit:.1f}; {s['tally']} {s['tally_rnd']}"


def test_random_decks_are_uniform_and_independent_seats(pr, independent_stats):
    s = independent_stats
    rd = s["rnd_decks"]
    assert len(rd) > 15000
    for d in rd[:2000]:
        assert len(set(d)) == 8 and list(d) == sorted(d)
    freq = np.bincount(np.concatenate([np.asarray(d) for d in rd]), minlength=64)[:64]
    stat, crit = chi2_ok(freq, np.full(64, len(rd) * 8 / 64))
    assert stat < crit, f"random decks are not uniform over the 64 cards: chi2 {stat:.1f} > {crit:.1f}"
    assert len({tuple(d) for d in rd}) > 0.99 * len(rd), "random decks repeat far too often"
    # independent draws: both seats equal with probability sum p_i^2 (pool) only
    p = np.concatenate([(1 - 0.2) * WEIGHTS / WEIGHTS.sum()])
    expect = s["seats"] / 2 * float(np.sum(p ** 2))
    sd = math.sqrt(expect)
    assert abs(s["pairs_equal"] - expect) < 6 * sd, f"seat decks not independent: {s['pairs_equal']} equal pairs vs {expect:.0f}"


def test_mirror_mode(pr):
    s = _deal_many("mirror", 0.3, n_envs=32, resets=400)
    assert s["pairs_equal"] == s["seats"] // 2, "mirror: team 1 takes team 0's deck at every deal"
    assert np.array_equal(s["dpool"], s["tally"]) and s["drnd"] == s["tally_rnd"]
    assert np.all(s["dpool"] % 2 == 0) and s["drnd"] % 2 == 0, "a mirror deal counts both seats"
    deals = s["seats"] // 2
    exp = np.concatenate([deals * 0.7 * WEIGHTS / WEIGHTS.sum(), [deals * 0.3]])
    stat, crit = chi2_ok(np.concatenate([s["tally"] // 2, [s["tally_rnd"] // 2]]), exp)
    assert stat < crit, f"mirror deal frequencies off: chi2 {stat:.1f} > {crit:.1f}"


def test_heldout_never_dealt_as_random_draw(pr):
    """frac = 1 (pool empty) with 1024 held-out decks (the capacity): no dealt deck is held out; every
    seat counts as random."""
    held = "xbow;random:1023:3"
    env = E.make(num_envs=32, num_agents=2, random_deck_frac=1.0, heldout_decks=held, seed=8)
    hk = {D().deck_key(d) for d, _ in D().parse_deck_set(held)}
    before = counts(env)
    for k in range(60):
        obs, _ = env.reset(seed=31 * k + 5)
        for r in obs:
            assert D().deck_key(V.own_deck(r)) not in hk
    after = counts(env)
    assert after[1] - before[1] == 60 * 64 and len(after[0]) == 0
    env.close()


def test_rejection_redraws_and_is_counted(pr):
    """Make the first random draw after reset(seed) a held-out deck (observed in an identical env without
    held-out decks): the draw is rejected (+1 random_rejected), redrawn, and the seat gets another deck."""
    a = E.make(num_envs=1, num_agents=2, random_deck_frac=1.0, seed=21)
    oa, _ = a.reset(seed=77)
    first = V.own_deck(oa[0])
    a.close()
    b = E.make(num_envs=1, num_agents=2, random_deck_frac=1.0, heldout_decks=[list(first)], seed=21)
    before = counts(b)
    ob, _ = b.reset(seed=77)
    after = counts(b)
    assert after[2] - before[2] == 1, f"random_rejected went {before[2]} -> {after[2]} (one rejected draw)"
    assert after[1] - before[1] == 2, "two seats received random decks"
    assert V.own_deck(ob[0]) != first and D().deck_key(V.own_deck(ob[0])) != D().deck_key(first)
    b.close()


# ==========================================================================================
# Determinism and stream independence
# ==========================================================================================
def _deal_sequence(env, seeds):
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        out.append(tuple(V.own_deck(r) for r in obs))
    return out


def test_reset_seed_reproducible(pr):
    kw = dict(num_envs=4, num_agents=2, deck_pool=POOL, random_deck_frac=0.5, seed=3)
    a, b = E.make(**kw), E.make(**kw)
    seeds = [5, 6, 5, 100, 6]
    sa, sb = _deal_sequence(a, seeds), _deal_sequence(b, seeds)
    assert sa == sb, "same seeds -> same deals"
    assert sa[0] == sa[2] and sa[1] == sa[4], "reset(seed) reseeds the deck stream"
    assert sa[0] != sa[1], "different reset seeds -> different deals"
    a.close()
    b.close()


def test_auto_reset_deals_do_not_depend_on_play(pr):
    """The k-th auto-reset deal is the same whether the match was played or not (dedicated stream: the
    game, bot and side streams are consumed differently)."""
    def deals(p):
        env = E.make(num_envs=1, num_agents=1, opponent="random", learner_side="random", bot_play_prob=p,
                     random_deck_frac=0.6, deck_pool="hog26;giant", frame_skip=100, seed=2)
        rng = np.random.default_rng(1)
        obs, _ = env.reset(seed=2)
        out = [V.own_deck(obs[0])]
        for _ in range(400):
            a = E.random_bot_action(rng, obs[0], p)
            obs, rew, term, *_ = E.step(env, [a])
            if term[0]:
                out.append(V.own_deck(obs[0]))
                if len(out) == 5:
                    break
        env.close()
        return out
    da, db = deals(0.0), deals(0.7)
    assert len(da) == len(db) == 5
    # the learner's seat may differ per episode only through the side stream, which is the same in both
    assert da == db, f"deal sequence depends on play: {da} vs {db}"


def test_league_forwards_deck_kwargs(pr):
    pool = L.make_pool(anchors=("bot:noop",), self_play_frac=0.0, anchor_frac=1.0, seed=0)
    lv = L.make_league(pool, num_envs=4, seed=0, deck_pool="golem;xbow", deck_draw="mirror", deck0="hog26")
    lv.async_reset(0)
    obs = lv.recv()[0]
    allowed = {preset("golem"), preset("xbow")}
    assert all(V.own_deck(r) in allowed for r in obs)
    lv.close()


def test_royale_ini_env_keys(pr):
    """§19.5: royale.ini [env] gains deck_pool, random_deck_frac, heldout_decks, deck_draw,
    placement_grid and the reward-v2 weight/cap keys with these defaults."""
    import configparser
    import os
    cp = configparser.ConfigParser()
    cp.read(os.path.join(H.ROOT, "pufferroyale", "config", "royale.ini"))
    env = cp["env"]
    want = dict(deck_pool="", random_deck_frac=0.0, heldout_decks="", deck_draw="independent", placement_grid=1,
                reward_elixir=0.0, reward_play=0.0, reward_elixir_cap=20.0, reward_play_cap=20.0)
    for k, v in want.items():
        assert k in env, f"royale.ini [env] lacks {k}"
        got = env[k].strip()
        if isinstance(v, str):
            assert got == v, f"[env] {k} = {got!r}, want {v!r}"
        else:
            assert float(got) == float(v), f"[env] {k} = {got!r}, want {v}"


# ==========================================================================================
# env_info keys (ruling v0.5-G.1)
# ==========================================================================================
ENV_INFO_KEYS = ("deck_sampler", "deck0_ignored", "deck1_ignored", "deck_mirror", "placement_grid", "row_grids",
                 "env_steps", "shaping_multiplier")


@pytest.mark.parametrize("kw,active,mirror", [
    (dict(), False, False),
    (dict(deck_pool="hog26;giant"), True, False),
    (dict(deck_pool="hog26;giant", deck_draw="mirror"), True, True),
    (dict(random_deck_frac=1.0), True, False),
])
def test_env_info_sampler_flags(pr, kw, active, mirror):
    """§19.5: when the sampler is active deck0/deck1 are ignored, reported by env_info."""
    env = E.make(num_envs=1, num_agents=2, **kw)
    info = env.env_info()
    for k in ENV_INFO_KEYS:
        assert k in info, f"env_info lacks {k!r} (ruling v0.5-G.1): {sorted(info)}"
    assert info["deck_sampler"] is active or bool(info["deck_sampler"]) == active
    assert bool(info["deck0_ignored"]) == active and bool(info["deck1_ignored"]) == active
    if active:
        assert bool(info["deck_mirror"]) == mirror
    env.close()


@pytest.mark.parametrize("g", [1, 2, 4])
def test_env_info_grid(pr, g):
    """§19.9.2: row_grids has one entry per team; a scripted-bot team (single-agent env) is always 1."""
    env = E.make(num_envs=1, num_agents=2, placement_grid=g)
    info = env.env_info(0)
    assert int(info["placement_grid"]) == g
    assert [int(x) for x in info["row_grids"]] == [g, g], f"self-play: both teams on grid {g}: {info['row_grids']}"
    env.close()
    for side in (0, 1):
        env = E.make(num_envs=1, num_agents=1, opponent="random", learner_side=side, placement_grid=g)
        env.reset(seed=0)
        info = env.env_info(0)
        want = [g, 1] if side == 0 else [1, g]
        assert [int(x) for x in info["row_grids"]] == want, f"learner side {side}: row_grids {info['row_grids']}"
        env.close()


def test_env_info_env_steps_and_shaping_multiplier(pr):
    """§19.9.2: env_steps = c_steps since that C env was created (never reset by reset());
    shaping_multiplier = the multiplier the next step will use, m_{env_steps}."""
    N, n0 = 100, 10
    env = E.make(num_envs=2, num_agents=2, reward_tower=0.3, shaping_anneal_steps=N, shaping_step_offset=n0)
    for i in (0, 1):
        assert int(env.env_info(i)["env_steps"]) == 0
        assert abs(float(env.env_info(i)["shaping_multiplier"]) - V.anneal(0, N, n0)) < 1e-12
    env.reset(seed=1)
    for k in range(1, 8):
        E.step(env, [0, 0, 0, 0])
        for i in (0, 1):
            info = env.env_info(i)
            assert int(info["env_steps"]) == k
            assert abs(float(info["shaping_multiplier"]) - V.anneal(k, N, n0)) < 1e-12, \
                f"after {k} steps the next multiplier is m_{k} = {V.anneal(k, N, n0)}, got {info['shaping_multiplier']}"
    env.reset(seed=2)
    assert int(env.env_info(0)["env_steps"]) == 7, "reset() does not reset the step count"
    env.close()
    env = E.make(num_envs=1, num_agents=2)
    env.reset(seed=0)
    E.step(env, [0, 0])
    assert float(env.env_info(0)["shaping_multiplier"]) == 1.0
    env.close()
    env = E.make(num_envs=1, num_agents=2, reward_tower=0.3, shaping_anneal_steps=5, shaping_step_offset=5)
    assert float(env.env_info(0)["shaping_multiplier"]) == 0.0
    env.close()
