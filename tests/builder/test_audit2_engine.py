"""Builder tests: the second audit's engine items through the Python API (SPEC §18, v0.4.1 / v0.4.2).

18.1 crown-tower ties go own-left; 18.2 Royal Chef serving order; 18.3 dash immunity evaluated when
damage is dealt; 18.4 seat-symmetric bots; 18.5 / 18.8 snapshot validation (multiplied values,
pending spawns) with unused slots normalised on restore."""
import numpy as np
import pytest

from pufferroyale import CARD_NAMES, Bot, Game, PlayError

C = {n: i for i, n in enumerate(CARD_NAMES)}


def rot(x, y):
    return 18000 - x, 32000 - y


def own(team, x, y):
    return (x, y) if team == 0 else rot(x, y)


@pytest.mark.parametrize("team", [0, 1])
def test_crown_tower_tie_goes_own_left(team):
    g = Game("hog26", "hog26", seed=1, deploy_lockout_ticks=0)
    k = g.spawn(team, "Knight", *own(team, 9000, 12000))[0]
    g.tick()
    target = g.entity(k)["target_id"]
    own_left = 1 if team == 0 else 2
    assert target == g.state()["towers"][1 - team][own_left]["id"]


@pytest.mark.parametrize("team", [0, 1])
@pytest.mark.parametrize("musk_x", [2000, 16000])
def test_royal_chef_serves_own_left_first(team, musk_x):
    g = Game("hog26", "giant", seed=1, deploy_lockout_ticks=0, **{f"tower_troop{team}": "royal_chef"})
    g.tick(555)
    golem = g.spawn(team, "Golem", *own(team, 9000, 22000))[0]
    musk = g.spawn(team, "Musketeer", *own(team, musk_x, 22000))[0]
    g.tick(6)                                                  # past the first pancake (tick 560)
    assert g.entity(golem)["level"] == 12                      # own-left tower, within reach of both
    assert g.entity(musk)["level"] == (12 if musk_x == 16000 else 11)


@pytest.mark.parametrize("spell", ["Zap", "Freeze", "The Log", "Arrows"])
def test_dash_immunity_holds_against_any_spell(spell):
    """A spell cast on the Bandit while it dashes deals 0 damage, even if its stun / knockback
    cancels the dash in the same tick."""
    g = Game("hog26", "hog26", seed=1, deploy_lockout_ticks=0)
    g.spawn(1, "Golem", 9000, 9000, deployed=False)
    b = g.spawn(0, "Bandit", 9000, 14000)[0]
    for _ in range(60):
        g.tick()
        if g.entity(b)["dash_state"] == 2:
            break
    e = g.entity(b)
    assert e["dash_state"] == 2, "setup: the Bandit must be dashing"
    g.set_hand(1, [C[spell]] + [c for c in range(64) if c != C[spell]][:7])
    g.set_elixir(1, 28000)
    assert g.play(1, 0, e["x"], e["y"]) == PlayError.OK
    g.tick()
    assert g.entity(b)["hp"] == e["hp"], f"{spell} damaged the dashing Bandit"


def test_heuristic_finish_tower_is_seat_symmetric():
    acts = []
    for team in (0, 1):
        g = Game("hog26", "hog26", seed=3, deploy_lockout_ticks=0)
        g.set_tower_hp(1 - team, 1, 100)
        g.set_tower_hp(1 - team, 2, 100)
        g.set_hand(team, [C["Fireball"]] + [c for c in range(64) if c != C["Fireball"]][:7])
        g.set_elixir(team, 28000)
        acts.append(Bot("heuristic", seed=7).act(g, team))
    assert acts[0] == acts[1] != 0
    assert ((acts[0] - 1) % 576) % 18 < 9, "the own-left enemy princess"


def test_restore_rejects_unbounded_values_and_normalises_unused_slots():
    g = Game("giant", "giant", seed=3, deploy_lockout_ticks=0)
    g.spawn(0, "Cannon", 9000, 21000)
    g.tick(20)
    base = g.snapshot()
    h = g.hash()
    words = np.frombuffer(base[:len(base) // 4 * 4], dtype="<i4").copy()
    idx = [i for i in range(len(words)) if words[i] == 824]      # the Cannon's max_hp (level 11)
    assert idx, "setup: the Cannon's max_hp word"
    rejected = 0
    for i in idx:
        bad = words.copy()
        bad[i] = 10 ** 9
        try:
            g.restore(bad.tobytes() + base[len(base) // 4 * 4:])
        except ValueError:
            rejected += 1
            assert g.hash() == h, "a rejected restore left the game changed"
    assert rejected >= 1
    # garbage in an unused entity slot (the last one) is not live content: accepted, then zeroed
    blob = bytearray(base)
    blob[-1] ^= 0x5A
    g.restore(bytes(blob))
    assert g.snapshot() == base and g.hash() == h
