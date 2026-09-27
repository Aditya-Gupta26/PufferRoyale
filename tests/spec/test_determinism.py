"""SPEC §1 determinism: seeded shuffles, bit-identical replays (in-process and across processes),
snapshot/restore, FNV-1a state hash, 180-degree seat symmetry; §11 no float/double (and no libc
RNG) in simulation C code."""
import json
import os
import re
import subprocess
import sys

import pytest

import gamekit as K
import helpers as H

HERE = os.path.dirname(os.path.abspath(__file__))


def script_plays(deck):
    """A fixed, legal play script: (tick, team, card, own tile). One play per tick at most."""
    d = H.DECKS[deck]
    plays = []
    t = 100
    tiles = [(9, 21), (4, 22), (13, 22), (8, 19), (3, 20), (14, 20), (9, 24), (5, 18)]
    btiles = [(1, 28), (15, 28), (4, 29), (12, 29)]      # clear of every troop tile above
    bi = 0
    for i in range(16):
        team = i % 2
        card = d[(i * 3) % 8]
        if H.CARD_KIND[card] == "building":
            tile = btiles[bi % len(btiles)]
            bi += 1
        else:
            tile = tiles[i % len(tiles)]
        plays.append((t, team, card, tile))
        t += 37
    return plays


def run_script(deck, seed, n_ticks=900):
    g = K.new_game(deck, deck, seed=seed)
    plays = {p[0]: p for p in script_plays(deck)}
    hashes = []
    for _ in range(n_ticks):
        tk = K.tick_of(g)
        if tk in plays:
            _, team, card, tile = plays[tk]
            code = K.cast_tile(g, team, card, *tile)
            assert code == H.OK, f"script play at tick {tk} refused with {code}"
        g.tick(1)
        hashes.append(int(g.hash()))
        if K.over(g):
            break
    return hashes


@pytest.mark.parametrize("deck", ["hog26", "giant", "bait"])
def test_replay_is_bit_identical(pr, deck):
    a = run_script(deck, 7)
    b = run_script(deck, 7)
    assert a == b
    assert len(set(a)) > len(a) // 2, "the hash must change as the state evolves"


def test_replay_identical_across_processes(pr):
    local = run_script("hog26", 3, 600)
    code = (
        "import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "import test_determinism as T\n"
        "print(json.dumps(T.run_script('hog26', 3, 600)))\n" % (HERE, H.ROOT)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         cwd=H.ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    remote = json.loads(out.stdout.strip().splitlines()[-1])
    assert remote == local, "same seed + same script must give identical hashes in another process"


def test_different_seeds_diverge(pr):
    hs = {int(K.new_game("hog26", "hog26", seed=s).hash()) for s in range(8)}
    assert len(hs) >= 6, "different seeds must deal different decks (different state hashes)"


def test_seed_zero_default_reproducible(pr):
    assert int(K.new_game().hash()) == int(K.new_game().hash())


def test_snapshot_restore_reproduces_future(pr):
    g = K.new_game("bait", "bait", seed=5)
    plays = {p[0]: p for p in script_plays("bait")}
    for _ in range(400):
        tk = K.tick_of(g)
        if tk in plays:
            _, team, card, tile = plays[tk]
            assert K.cast_tile(g, team, card, *tile) == H.OK
        g.tick(1)
    snap = g.snapshot()
    assert isinstance(snap, (bytes, bytearray))
    h0 = int(g.hash())
    fut = []
    for _ in range(300):
        g.tick(1)
        fut.append(int(g.hash()))
    g.restore(snap)
    assert int(g.hash()) == h0
    again = []
    for _ in range(300):
        g.tick(1)
        again.append(int(g.hash()))
    assert again == fut
    other = K.new_game("hog26", "giant", seed=99)
    other.restore(snap)
    assert int(other.hash()) == h0, "restore into another Game instance"
    for i in range(50):
        other.tick(1)
        assert int(other.hash()) == fut[i]


def test_snapshot_restore_mid_flight(pr):
    """Snapshot while a Fireball and a Goblin Barrel are in flight and a Log is rolling."""
    g = K.new_game("bait", "bait", seed=2, deploy_lockout_ticks=0)
    assert K.cast(g, 0, H.FIREBALL, 3500, 6500) == H.OK
    g.tick(1)
    assert K.cast(g, 0, H.GOBLIN_BARREL, 14500, 6500) == H.OK
    g.tick(1)
    assert K.cast(g, 1, H.LOG, 9000, 12000) == H.OK
    g.tick(3)
    assert len(g.projectiles()) + len(g.effects()) >= 2
    snap = g.snapshot()
    a = []
    for _ in range(120):
        g.tick(1)
        a.append(int(g.hash()))
    g.restore(snap)
    b = []
    for _ in range(120):
        g.tick(1)
        b.append(int(g.hash()))
    assert a == b


def fnv1a64(data):
    h = 0xcbf29ce484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def test_hash_is_fnv1a_of_state_bytes(pr):
    """SPEC §1: state_hash = FNV-1a-64 over the state struct; snapshot = memcpy of that struct."""
    g = K.new_game("giant", "hog26", seed=4)
    g.tick(137)
    snap = bytes(g.snapshot())
    assert fnv1a64(snap) == int(g.hash()) & 0xFFFFFFFFFFFFFFFF


# ------------------------------------------------------------------------------------------
# Seat symmetry (180-degree rotation)
# ------------------------------------------------------------------------------------------
MIRROR_IDX = [0, 2, 1]   # absolute left <-> right under the rotation


def mirrored_equal(ga, gb):
    """State of ga rotated by 180 degrees with teams swapped must equal the state of gb."""
    ea = sorted((1 - int(e["team"]), int(e["card_id"]), H.ARENA_W - int(e["x"]), H.ARENA_H - int(e["y"]),
                 int(e["hp"]), bool(e["deploying"]), bool(e["stunned"])) for e in K.ents(ga))
    eb = sorted((int(e["team"]), int(e["card_id"]), int(e["x"]), int(e["y"]), int(e["hp"]),
                 bool(e["deploying"]), bool(e["stunned"])) for e in K.ents(gb))
    if ea != eb:
        return False, (ea, eb)
    for t in (0, 1):
        for i in range(3):
            if K.tower_hp(ga, t, i) != K.tower_hp(gb, 1 - t, MIRROR_IDX[i]):
                return False, ("tower", t, i)
    if K.elixir(ga, 0) != K.elixir(gb, 1) or K.elixir(ga, 1) != K.elixir(gb, 0):
        return False, "elixir"
    return True, None


# name -> ((deck0, deck1) of game A, script of (tick, team, card, own tile) for game A).
# Game B uses the swapped decks and plays each action for the OTHER team at the same own-frame
# tile, i.e. the 180-degree-rotated counterpart. Plays never coincide in a tick, so entity
# creation order (and ids) correspond between A and B.
MIRROR_SCENARIOS = {
    "knight": (("giant", "giant"), [(100, 0, H.KNIGHT, (4, 22))]),
    "hog_giant": (("hog26", "giant"), [(100, 0, H.HOG, (13, 20)), (140, 1, H.GIANT, (4, 22))]),
    "fireball_log": (("hog26", "giant"), [(100, 1, H.KNIGHT, (9, 20)), (160, 0, H.FIREBALL, (8, 11)),
                                          (200, 0, H.LOG, (8, 17))]),
    "arrows_zap": (("giant", "giant"), [(100, 1, H.MINIONS, (9, 20)), (150, 0, H.ARROWS, (9, 12)),
                                        (190, 0, H.ZAP, (9, 12))]),
    "barrel": (("bait", "bait"), [(100, 0, H.GOBLIN_BARREL, (3, 6)), (180, 1, H.SKARMY, (14, 22))]),
    "buildings": (("bait", "hog26"), [(100, 0, H.TESLA, (9, 20)), (130, 1, H.HOG, (4, 20)),
                                      (170, 0, H.VALKYRIE, (5, 23)), (210, 1, H.CANNON, (12, 21))]),
    "lane_tie": (("giant", "giant"), [(100, 0, H.MINIONS, (9, 20)), (130, 1, H.KNIGHT, (8, 22))]),
}


@pytest.mark.parametrize("name", sorted(MIRROR_SCENARIOS))
def test_mirror_symmetry(pr, name):
    (d0, d1), script = MIRROR_SCENARIOS[name]
    ga = K.new_game(d0, d1, seed=0)
    gb = K.new_game(d1, d0, seed=0)
    plays = {p[0]: p for p in script}
    for _ in range(420):
        tk = K.tick_of(ga)
        if tk in plays:
            _, team, card, tile = plays[tk]
            ca = K.cast_tile(ga, team, card, *tile)
            cb = K.cast_tile(gb, 1 - team, card, *tile)
            assert ca == cb == H.OK, (ca, cb)
        ga.tick(1)
        gb.tick(1)
        ok, why = mirrored_equal(ga, gb)
        assert ok, f"{name}: seat symmetry broken at tick {K.tick_of(ga)}: {str(why)[:1500]}"
        if K.over(ga):
            assert K.over(gb) and K.result(ga, 0) == K.result(gb, 1)
            break


# ------------------------------------------------------------------------------------------
# Static checks on the C sources (SPEC §11, §1)
# ------------------------------------------------------------------------------------------
CSRC = os.path.join(H.ROOT, "pufferroyale", "csrc")
EXCLUDE = re.compile(r"(obs|log|render|raylib|binding|env)", re.I)


def strip_comments(src):
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    src = re.sub(r'"(\\.|[^"\\])*"', '""', src)
    return src


def sim_sources():
    assert os.path.isdir(CSRC), "pufferroyale/csrc/ must exist"
    files = []
    for root, _, names in os.walk(CSRC):
        for n in names:
            if n.endswith((".h", ".c")) and not EXCLUDE.search(n) and n != "royale.h":
                files.append(os.path.join(root, n))
    return sorted(files)


def test_generated_db_headers_exist(pr):
    for n in ("pr_card_db.h", "pr_arena_db.h"):
        assert os.path.isfile(os.path.join(CSRC, n)), f"{n} must be generated and committed (SPEC §2/§3)"


def test_no_float_or_double_in_simulation_code(pr):
    files = sim_sources()
    assert len(files) >= 3, f"expected several simulation headers in csrc/, found {files}"
    bad = []
    for f in files:
        src = strip_comments(open(f, encoding="utf-8", errors="replace").read())
        for m in re.finditer(r"\b(float|double|long\s+double)\b|\b(sqrtf?|powf?|sinf?|cosf?|atan2f?)\s*\(", src):
            line = src[:m.start()].count("\n") + 1
            bad.append(f"{os.path.relpath(f, H.ROOT)}:{line}: {m.group(0)}")
    assert not bad, "floating point in simulation code:\n" + "\n".join(bad[:40])


def test_no_libc_rng_or_clock_in_simulation_code(pr):
    """SPEC §1: randomness only from seeded PCG32 streams; nothing time-dependent."""
    bad = []
    for f in sim_sources():
        src = strip_comments(open(f, encoding="utf-8", errors="replace").read())
        for m in re.finditer(r"\b(rand|srand|random|drand48|time|clock|gettimeofday|qsort)\s*\(", src):
            line = src[:m.start()].count("\n") + 1
            bad.append(f"{os.path.relpath(f, H.ROOT)}:{line}: {m.group(0)}")
    assert not bad, "non-deterministic libc calls in simulation code:\n" + "\n".join(bad[:40])
