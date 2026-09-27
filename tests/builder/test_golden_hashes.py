"""Cross-platform determinism: the engine must reproduce the committed golden hash trails
(tests/golden/golden_hashes.json, generated on macOS arm64) bit for bit on every platform.
See scripts/golden_hashes.py; a failure here on a new platform is a portability bug to report,
not a reason to regenerate the reference."""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location("golden_hashes", ROOT / "scripts" / "golden_hashes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_golden_hash_trails_match_reference():
    gh = _load_script()
    ref = json.loads(gh.REF.read_text())
    got = gh.run_all()
    assert len(ref["matches"]) == len(got) == len(gh.SCENARIOS)
    for r, g in zip(ref["matches"], got):
        name = f"{r['deck0']} vs {r['deck1']} seed {r['seed']}"
        assert g["hashes"] == r["hashes"], f"hash trail diverged: {name} (reference: {ref['platform']})"
        assert g["final_hash"] == r["final_hash"] and g["final_tick"] == r["final_tick"], name
        assert g["result"] == r["result"] and g["end_reason"] == r["end_reason"], name
