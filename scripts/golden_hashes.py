"""Cross-platform determinism check (SPEC §1): replay seeded scripted-bot matches and compare
the engine's state hash every 100 ticks against a committed reference.

    python scripts/golden_hashes.py            # compare with tests/golden/golden_hashes.json
    python scripts/golden_hashes.py --write    # (re)generate the reference -- only on purpose

The reference was generated on macOS arm64 (Apple clang). The same hashes on another platform
(e.g. Linux x86_64 on the HPC cluster) show the engine is bit-for-bit deterministic across
platforms. A mismatch is a real portability finding: report the first diverging match and tick,
do not simply regenerate the reference.
"""
import argparse
import json
import pathlib
import platform
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pufferroyale import Bot, Game  # noqa: E402

REF = ROOT / "tests" / "golden" / "golden_hashes.json"
EVERY = 100

# (deck0, deck1, seed, tower_troop0, tower_troop1, bot0, bot1): covers the v0.1 and v0.3 cards,
# every tower troop and both bot kinds that act.
SCENARIOS = [
    ("hog26", "giant", 0, "princess", "princess", "heuristic", "heuristic"),
    ("golem", "lavaloon", 1, "cannoneer", "royal_chef", "heuristic", "heuristic"),
    ("xbow", "miner_poison", 2, "dagger_duchess", "princess", "heuristic", "random"),
    ("pekka_bridge", "royal_hogs", 3, "royal_chef", "cannoneer", "random", "heuristic"),
    ("bait", "random", 4, "princess", "dagger_duchess", "random", "random"),
    ("random", "random", 5, "cannoneer", "cannoneer", "heuristic", "heuristic"),
]


def trail(deck0, deck1, seed, tt0, tt1, b0, b1):
    g = Game(deck0=deck0, deck1=deck1, seed=seed, tower_troop0=tt0, tower_troop1=tt1)
    bots = [Bot(b0, seed=1000 + seed), Bot(b1, seed=2000 + seed)]
    hashes = []
    while not g.state()["over"]:
        for team in (0, 1):
            a = bots[team].act(g, team)
            if a:
                slot, cell = (a - 1) // 576, (a - 1) % 576
                g.play_tile(team, slot, cell % 18, cell // 18)
        for _ in range(10):
            g.tick(1)
            t = g.state()["tick"]
            if t % EVERY == 0:
                hashes.append(f"{g.hash():016x}")
            if g.state()["over"]:
                break
    s = g.state()
    return {"hashes": hashes, "final_tick": s["tick"], "final_hash": f"{g.hash():016x}",
            "result": s["result"], "end_reason": s["end_reason"]}


def run_all():
    return [dict(zip(("deck0", "deck1", "seed", "tower_troop0", "tower_troop1", "bot0", "bot1"), sc),
                 **trail(*sc)) for sc in SCENARIOS]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="regenerate the reference file")
    args = ap.parse_args()
    got = run_all()
    if args.write:
        REF.parent.mkdir(parents=True, exist_ok=True)
        REF.write_text(json.dumps({"platform": f"{platform.system()} {platform.machine()}",
                                   "every": EVERY, "matches": got}, indent=1) + "\n")
        print(f"wrote {REF} ({len(got)} matches)")
        return 0
    ref = json.loads(REF.read_text())
    ok = True
    for r, g in zip(ref["matches"], got):
        name = f"{r['deck0']} vs {r['deck1']} seed {r['seed']}"
        if r["hashes"] == g["hashes"] and r["final_hash"] == g["final_hash"]:
            print(f"OK    {name}: {len(g['hashes'])} checkpoints, final tick {g['final_tick']}")
            continue
        ok = False
        first = next((i for i, (a, b) in enumerate(zip(r["hashes"], g["hashes"])) if a != b),
                     min(len(r["hashes"]), len(g["hashes"])))
        print(f"DIFF  {name}: first divergence at tick {(first + 1) * EVERY} "
              f"(reference from {ref['platform']}, this is {platform.system()} {platform.machine()})")
    print("golden hashes:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
