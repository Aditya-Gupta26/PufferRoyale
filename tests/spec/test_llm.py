"""SPEC §17 (v0.4 Phase F): the LLM play interface `pufferroyale.llm` and `scripts/llm_match.py`.

HARD RULE: no network. Every test runs with the `anthropic` package import-blocked and socket
connections refused (autouse fixture); subprocesses get the same guard through a sitecustomize.
The Anthropic adapter is only checked for its behaviour when the SDK is absent.

Formats the SPEC leaves open are handled leniently and listed as ambiguities in the report:
hand-slot numbering (0- or 1-based, detected from the rendered `slot: Card (cost)` lines and then
used consistently), the clock (elapsed or remaining m:ss), and the opponent spec strings
(`bot:<name>` / `ckpt:<path>` as in §15.1)."""
import hashlib
import importlib
import json
import math
import os
import re
import socket
import subprocess
import sys

import numpy as np
import pytest

import cardsv3 as C
import gamekit as K
import helpers as H

ROOT = H.ROOT
SCRIPT = os.path.join("scripts", "llm_match.py")
NAMES = C.ALL_NAMES


# ------------------------------------------------------------------------------------------
# No-network / no-anthropic guard
# ------------------------------------------------------------------------------------------
class _BlockAnthropic:
    attempts = []

    def find_spec(self, name, path=None, target=None):
        if name == "anthropic" or name.startswith("anthropic."):
            _BlockAnthropic.attempts.append(name)
            raise ImportError("tests/spec/test_llm.py: importing 'anthropic' is forbidden in tests")
        return None


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    blocker = _BlockAnthropic()
    sys.meta_path.insert(0, blocker)

    def refuse(*a, **k):
        raise OSError("network access is forbidden in tests")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    _BlockAnthropic.attempts.clear()
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
    assert not _BlockAnthropic.attempts, f"something tried to import anthropic: {_BlockAnthropic.attempts}"
    assert "anthropic" not in sys.modules


SITECUSTOMIZE = '''
import os, socket, sys
_MARK = os.environ["PR_TEST_MARK"]
def _note(what):
    with open(_MARK, "a") as f:
        f.write(what + "\\n")
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "anthropic" or name.startswith("anthropic."):
            _note("import " + name)
            raise ImportError("No module named 'anthropic' (blocked by the PufferRoyale test harness)")
        return None
sys.meta_path.insert(0, _Block())
def _refuse(*a, **k):
    _note("network")
    raise OSError("network access blocked by the PufferRoyale test harness")
socket.socket.connect = _refuse
socket.socket.connect_ex = _refuse
socket.create_connection = _refuse
'''


def run_guarded(tmp_path, args=None, code=None, timeout=900, extra_env=None):
    """Run python (a script or -c code) with anthropic import-blocked and sockets refused.
    Returns (CompletedProcess, marker text: 'import anthropic' / 'network' lines)."""
    site = tmp_path / "guard_site"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    mark = tmp_path / "guard_mark.txt"
    if mark.exists():
        mark.unlink()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(site), ROOT, env.get("PYTHONPATH", "")])
    env["PR_TEST_MARK"] = str(mark)
    env.update(extra_env or {})
    cmd = [sys.executable] + (["-c", code] if code is not None else list(args))
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return out, (mark.read_text() if mark.exists() else "")


def L():
    import pufferroyale.llm as mod
    return mod


# ------------------------------------------------------------------------------------------
# Scene helpers
# ------------------------------------------------------------------------------------------
def game(d0="giant", d1="hog26", seed=1, **kw):
    kw.setdefault("deploy_lockout_ticks", 0)
    return K.new_game(d0, d1, seed=seed, **kw)


def ready(g, team, order=None, elixir=H.MAX_ELIXIR):
    if order is not None:
        g.set_hand(team, list(order))
    g.set_elixir(team, elixir)


def mask_block(g, team, slot):
    m = np.asarray(g.legal_mask(team))
    base = 1 + slot * H.N_CELLS
    return {(int(i) % 18, int(i) // 18) for i in np.nonzero(m[base:base + H.N_CELLS])[0]}


def slot_base(text, hand):
    """0 or 1: numbering of the rendered `slot: Card (cost)` hand lines (then used consistently)."""
    bases = set()
    for s, c in enumerate(hand):
        m = re.search(r"(?m)^\W*(?:slot\s*)?(\d)\s*:\s*" + re.escape(NAMES[c]) + r"\s*\(\s*" + str(C.cost(c)) + r"\s*\)", text)
        assert m, f"hand line '<slot>: {NAMES[c]} ({C.cost(c)})' not found in:\n{text}"
        bases.add(int(m.group(1)) - s)
    assert len(bases) == 1 and bases <= {0, 1}, f"inconsistent hand slot numbering {bases}"
    return bases.pop()


RE_TY = re.compile(r"^\W*ty\s*=\s*(\d+)(?:\s*-\s*(\d+))?\s*:\s*tx\s+([\d\s,\-]+?)\s*$", re.I)


def regions(text):
    """Blocks of consecutive row-run lines (`ty=20: tx 0-5,9-17`; `ty=a-b` also accepted)."""
    blocks, cur = [], None
    for line in text.splitlines():
        m = RE_TY.match(line)
        if not m:
            cur = None
            continue
        if cur is None:
            cur = set()
            blocks.append(cur)
        ty0 = int(m.group(1))
        ty1 = int(m.group(2)) if m.group(2) else ty0
        for part in m.group(3).split(","):
            part = part.strip()
            if not part:
                continue
            a, _, b = part.partition("-")
            for ty in range(ty0, ty1 + 1):
                for tx in range(int(a), int(b or a) + 1):
                    cur.add((tx, ty))
    return blocks


def clock_tokens(text):
    return [(int(a), int(b)) for a, b in re.findall(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])", text)]


class Recorder:
    """A model_fn that records (system, user) and answers with a fixed text or a function."""

    def __init__(self, reply="WAIT"):
        self.calls = []
        self.reply = reply

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.reply(system, user) if callable(self.reply) else self.reply


def phash(entry):
    """The transcript's prompt hash (SPEC: 'prompt hash'; key name not pinned: prompt_hash / prompt_sha*)."""
    keys = [k for k in entry if re.match(r"prompt_(hash|sha)", k)]
    return entry[keys[0]] if keys and entry[keys[0]] else None


END_REASONS = K.END_REASONS
RESULT_KEYS = {"result", "crowns", "end_reason", "ticks", "decisions", "illegal", "parse_errors"}


def check_result(res, max_ticks=6000):
    assert isinstance(res, dict) and RESULT_KEYS <= set(res), f"result keys {sorted(res)}"
    assert int(res["result"]) in (-1, 0, 1)
    assert len(res["crowns"]) == 2 and all(0 <= int(c) <= 3 for c in res["crowns"])
    assert 0 < int(res["ticks"]) <= max_ticks
    if res["end_reason"] is None:
        assert int(res["ticks"]) == max_ticks < 6000 and int(res["result"]) == 0, \
            "end_reason may be None only for a match cut by max_ticks"
    else:
        assert res["end_reason"] in END_REASONS
    for k in ("decisions", "illegal", "parse_errors"):
        assert int(res[k]) >= 0 and float(res[k]) == int(res[k])
    assert int(res["illegal"]) + int(res["parse_errors"]) <= int(res["decisions"])


# ==========================================================================================
# RULES_PROMPT
# ==========================================================================================
def test_rules_prompt_is_static(pr, tmp_path):
    rp = L().RULES_PROMPT
    assert isinstance(rp, str) and len(rp) > 300
    assert importlib.reload(L()).RULES_PROMPT == rp, "identical after a module reload"
    digest = hashlib.sha256(rp.encode()).hexdigest()
    code = "import hashlib, pufferroyale.llm as L; print(hashlib.sha256(L.RULES_PROMPT.encode()).hexdigest())"
    for hs in ("0", "12345"):
        out, mark = run_guarded(tmp_path, code=code, timeout=300, extra_env={"PYTHONHASHSEED": hs})
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().splitlines()[-1] == digest, "RULES_PROMPT differs across processes"
        assert mark == ""


def test_rules_prompt_has_no_timestamps_or_ids(pr):
    rp = L().RULES_PROMPT
    for pat, what in ((r"\b(19|20)\d\d-\d\d-\d\d\b", "a date"), (r"\b\d{1,2}:\d{2}:\d{2}\b", "a time of day"),
                      (r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "a uuid"),
                      (r"\b[0-9a-f]{16,}\b", "a hex id")):
        assert not re.search(pat, rp, re.I), f"RULES_PROMPT contains {what}: {re.search(pat, rp, re.I).group(0)}"


def test_rules_prompt_states_frame_and_response_format(pr):
    rp = L().RULES_PROMPT
    assert re.search(r"own[\s-]+frame", rp, re.I), "own-frame convention"
    assert re.search(r"\(\s*tx\s*,\s*ty\s*\)", rp), "tile (tx, ty)"
    t31 = r"ty\s*(?:=\s*)?31\b"
    assert re.search(r"(?i)(back\s+row[^\n]{0,40}" + t31 + "|" + t31 + r"[^\n]{0,40}back\s+row)", rp), \
        "ty = 31 is the own back row"
    t14 = r"ty\s*(?:(?:<=|≤|=)\s*14\b|0\s*[-–]\s*14\b|\s14\b)"
    assert re.search(r"(?i)(enemy\s+half[^\n]{0,40}" + t14 + "|" + t14 + r"[^\n]{0,40}enemy\s+half)", rp), \
        "ty <= 14 is the enemy half"
    assert re.search(r"\bWAIT\b", rp), "response format: WAIT"
    assert re.search(r"PLAY\s+<[^>\n]+>\s+AT\s+<\s*tx\s*>\s*,\s*<\s*ty\s*>", rp), \
        "response format: PLAY <slot|card name> AT <tx>,<ty>"


def test_rules_prompt_lists_every_card_with_its_cost(pr):
    rp = L().RULES_PROMPT
    lines = rp.splitlines()
    missing = []
    for cid, name in enumerate(NAMES):
        pat = re.compile(r"(?<![\d.])" + str(C.cost(cid)) + r"(?![\d.])")
        if not any(name in ln and pat.search(ln.replace(name, " ")) for ln in lines):
            missing.append((name, C.cost(cid)))
    assert not missing, f"cards without a line naming them with their cost: {missing}"


# ==========================================================================================
# render_state
# ==========================================================================================
def test_render_state_is_deterministic(pr):
    a, b = game(seed=3), game(seed=3)
    for g in (a, b):
        K.spawn(g, 0, H.KNIGHT, 9500, 21500)
        K.spawn(g, 1, H.GIANT, 5500, 12500)
    ta = L().render_state(a, 0)
    assert isinstance(ta, str) and ta == L().render_state(a, 0) == L().render_state(b, 0)
    a.tick(37)
    b.tick(37)
    assert L().render_state(a, 1) == L().render_state(b, 1)
    assert L().render_state(a, 0) != ta, "the rendering reflects the state"


def mirrored_pair():
    """Game A and its 180-degree mirror B with the seats swapped: A.team0 <-> B.team1."""
    ga, gb = game("hog26", "giant", seed=3), game("giant", "hog26", seed=3)
    oh = [H.HOG, H.CANNON, H.ICE_SPIRIT, H.FIREBALL, H.MUSKETEER, H.ICE_GOLEM, H.SKELETONS, H.LOG]
    og = [H.KNIGHT, H.ARROWS, H.GIANT, H.MINIONS, H.PRINCE, H.BABY_DRAGON, H.WIZARD, H.ZAP]
    ready(ga, 0, oh, 19600)
    ready(gb, 1, oh, 19600)
    ready(ga, 1, og, 22400)
    ready(gb, 0, og, 22400)
    ga.set_tower_hp(0, 1, 2000)        # A team 0 own-left Princess (engine x 3500)
    gb.set_tower_hp(1, 2, 2000)        # B team 1 own-left Princess (engine x 14500)
    ga.set_tower_hp(1, 0, 4000)
    gb.set_tower_hp(0, 0, 4000)
    for (team, card, x, y, dep) in ((0, H.KNIGHT, 9500, 21500, True), (1, H.GIANT, 5500, 12500, True),
                                    (0, H.MUSKETEER, 3500, 19500, False)):
        K.spawn(ga, team, card, x, y, deployed=dep)
        K.spawn(gb, 1 - team, card, H.ARENA_W - x, H.ARENA_H - y, deployed=dep)
    return ga, gb


def test_render_state_team1_is_the_rotation_of_team0(pr):
    ga, gb = mirrored_pair()
    for _ in range(2):
        assert L().render_state(ga, 0) == L().render_state(gb, 1), "own frame: seat-mirrored scenes read the same"
        assert L().render_state(ga, 1) == L().render_state(gb, 0)
        ga.tick(30)
        gb.tick(30)


def test_render_state_clock_phase_and_rate(pr):
    g = game(d0="hog26", d1="hog26")
    t0 = L().render_state(g, 0, legal=False)                     # no 'tx 1,..' runs to confuse the regexes
    assert set(clock_tokens(t0)) & {(0, 0), (3, 0)}, f"clock at tick 0 (elapsed 0:00 or remaining 3:00): {t0[:300]}"
    assert re.search(r"(?i)(\b1\s*[x×]|[x×]\s*1\b|single)", t0), "elixir rate 1x at tick 0"
    assert not re.search(r"(?i)overtime", t0)
    g.tick(1200)
    t1 = L().render_state(g, 0, legal=False)
    assert set(clock_tokens(t1)) & {(1, 0), (2, 0)}, "clock after 60 s"
    g.tick(1300)                                                   # tick 2500: double elixir
    t2 = L().render_state(g, 0, legal=False)
    assert re.search(r"(?i)(\b2\s*[x×]|[x×]\s*2\b|double)", t2), "elixir rate 2x from tick 2400"
    g.tick(1200)                                                   # tick 3700: overtime (no crowns)
    assert not K.over(g)
    assert re.search(r"(?i)overtime", L().render_state(g, 0, legal=False)), "phase: overtime"


def test_render_state_own_elixir_one_decimal_and_no_opponent_elixir(pr):
    g = game(d0="hog26", d1="giant")
    g.set_elixir(0, 18200)                                        # 6.5
    g.set_elixir(1, 26040)                                        # 9.3 (hidden)
    t = L().render_state(g, 0)
    assert re.search(r"(?<![\d.])6\.5(?![\d])", t), "own elixir with one decimal"
    assert not re.search(r"(?<![\d.])9\.3(?![\d])", t), "the opponent's current elixir is hidden"


def test_render_state_hand_costs_affordability_and_next_card(pr):
    g = game(d0="hog26", d1="giant")
    order = [H.HOG, H.CANNON, H.ICE_SPIRIT, H.FIREBALL, H.MUSKETEER, H.ICE_GOLEM, H.SKELETONS, H.LOG]
    hand = order[:4]
    texts = {}
    for el in (28000, 0, 8400):
        ready(g, 0, order, el)
        texts[el] = L().render_state(g, 0)
    base = slot_base(texts[28000], hand)

    def line(t, s):
        pat = re.compile(r"(?m)^.*?\b" + str(base + s) + r"\s*:\s*" + re.escape(NAMES[hand[s]]) + r"\s*\(.*$")
        m = pat.search(t)
        assert m, f"no hand line for slot {s}"
        return m.group(0)
    for s, c in enumerate(hand):
        full, empty, three = line(texts[28000], s), line(texts[0], s), line(texts[8400], s)
        assert full != empty, f"slot {s}: the affordable flag must differ between 10 and 0 elixir"
        if C.cost(c) <= 3:
            assert three == full, f"slot {s} ({NAMES[c]}) is affordable at 3.0 elixir: {three!r} vs {full!r}"
        else:
            assert three != full, f"slot {s} ({NAMES[c]}) is NOT affordable at 3.0 elixir: {three!r}"
    nxt = [ln for ln in texts[28000].splitlines() if re.search(r"(?i)\bnext\b", ln)]
    assert any(NAMES[order[4]] in ln for ln in nxt), f"next card {NAMES[order[4]]} not on a 'next' line: {nxt}"


def test_render_state_tower_hps(pr):
    g = game(d0="hog26", d1="giant")
    vals = {(0, 0): 4000, (0, 1): 1234, (0, 2): 2345, (1, 0): 4100, (1, 1): 3001, (1, 2): 1111}
    for (t, i), v in vals.items():
        g.set_tower_hp(t, i, v)
    for team in (0, 1):
        txt = L().render_state(g, team)
        for v in vals.values():
            assert re.search(r"(?<!\d)" + str(v) + r"(?!\d)", txt), f"tower hp {v} missing from team {team}'s view"


def test_render_state_units_in_own_frame_sorted_by_id(pr):
    g = game(d0="hog26", d1="giant")
    ghp = H.expected_unit_stats(H.GIANT)["hp"]
    K.spawn(g, 0, H.GIANT, 9500, 24500, deployed=True)
    K.spawn(g, 0, H.KNIGHT, 9500, 21500, deployed=False)
    K.spawn(g, 1, H.MUSKETEER, 12500, 12500, deployed=True)

    def find(t, name, tx, ty, hp):
        m = re.search(re.escape(name) + r"\s*@\s*\(\s*" + str(tx) + r"\s*,\s*" + str(ty) + r"\s*\)\s*"
                      + str(hp) + r"\s*/\s*" + str(hp) + r".*", t)
        assert m, f"'{name} @ ({tx},{ty}) {hp}/{hp}' not found in:\n{t}"
        return m
    t0 = L().render_state(g, 0)
    gi, kn = find(t0, "Giant", 9, 24, ghp), find(t0, "Knight", 9, 21, 1766)
    find(t0, "Musketeer", 12, 12, 721)
    assert gi.start() < kn.start(), "own units sorted by entity id"
    assert re.search(r"(?i)deploy", kn.group(0)) and not re.search(r"(?i)deploy", gi.group(0)), \
        "flags: the deploying Knight is marked, the deployed Giant is not"
    t1 = L().render_state(g, 1)
    find(t1, "Giant", 8, 7, ghp)
    find(t1, "Knight", 8, 10, 1766)
    find(t1, "Musketeer", 5, 19, 721)


def test_render_state_opponent_cards_seen(pr):
    g = game(d0="hog26", d1="giant")
    assert "Zap" not in L().render_state(g, 0)
    assert K.cast(g, 1, H.ZAP, 9000, 20000) == H.OK
    g.tick(2)
    assert "Zap" in L().render_state(g, 0), "a card the opponent played is listed (seen / last played)"


@pytest.mark.parametrize("decks", [("hog26", "giant"), ("miner_poison", "golem"), ("xbow", "royal_hogs"),
                                   ("bait", "pekka_bridge")])
@pytest.mark.parametrize("team", [0, 1])
def test_render_state_legal_regions_match_the_mask(pr, decks, team):
    """With every slot playable the text holds 4 row-run blocks, in slot order, each exactly the
    slot's legal_mask block (own frame)."""
    g = game(*decks, seed=2)
    ready(g, team)
    blocks = regions(L().render_state(g, team))
    want = [mask_block(g, team, s) for s in range(4)]
    assert all(want), "setup: every slot playable"
    assert len(blocks) == 4, f"expected 4 legal-region blocks, found {len(blocks)}"
    for s in range(4):
        assert blocks[s] == want[s], \
            f"slot {s}: text-only {sorted(blocks[s] - want[s])[:8]}, mask-only {sorted(want[s] - blocks[s])[:8]}"
    assert regions(L().render_state(g, team, legal=False)) == [], "legal=False omits the regions"


@pytest.mark.parametrize("team", [0, 1])
def test_render_state_legal_regions_with_pocket_buildings_and_elixir(pr, team):
    g = game("miner_poison", "royal_hogs", seed=4)
    g.set_tower_hp(1 - team, 1, 1)
    deck = K.deck_of(g, team)
    assert K.cast(g, team, C.POISON if C.POISON in deck else H.ZAP, *H.TOWER_POS[1 - team][1]) == H.OK
    K.run_until(g, lambda gg: not K.tower_alive(gg, 1 - team, 1), 60, "enemy Princess destroyed")
    x, y = H.own_to_engine_point(team, 6500, 24500)
    K.spawn(g, team, C.INFERNO_TOWER, x, y)
    for el in (H.MAX_ELIXIR, 8400, 0):
        ready(g, team, elixir=el)
        want = [(s, mask_block(g, team, s)) for s in range(4)]
        want = [w for w in want if w[1]]
        blocks = regions(L().render_state(g, team))
        assert len(blocks) == len(want), f"elixir {el}: {len(blocks)} blocks for {len(want)} playable slots"
        for b, (s, w) in zip(blocks, want):
            assert b == w, f"elixir {el}, slot {s}: text != mask"


def test_render_state_no_regions_during_lockout(pr):
    g = K.new_game("hog26", "giant", seed=1)                      # default lockout 90
    assert regions(L().render_state(g, 0)) == []


def test_render_state_never_leaks_hidden_information(pr):
    """Opponent hand order, queue, current elixir and even its unrevealed deck do not change a
    single character of the acting team's text (no opponent play yet, so no deductions)."""
    own = [H.HOG, H.CANNON, H.ICE_SPIRIT, H.FIREBALL, H.MUSKETEER, H.ICE_GOLEM, H.SKELETONS, H.LOG]
    g1, g2 = game("hog26", "giant", seed=5), game("hog26", "bait", seed=5)
    for g in (g1, g2):
        ready(g, 0, own, 19600)
    ready(g1, 1, [H.ZAP, H.GIANT, H.KNIGHT, H.ARROWS, H.PRINCE, H.MINIONS, H.WIZARD, H.BABY_DRAGON], 5600)
    ready(g2, 1, list(H.DECKS["bait"])[::-1], 27000)
    for _ in range(3):
        t1 = L().render_state(g1, 0)
        assert t1 == L().render_state(g2, 0), "the opponent's hidden deck/hand/elixir leaks into the text"
        g1.set_hand(1, K.deck_of(g1, 1)[::-1])
        g1.set_elixir(1, (K.elixir(g1, 1) + 9000) % 28000)
        assert L().render_state(g1, 0) == t1
        for g in (g1, g2):
            g.tick(40)


# ==========================================================================================
# parse_action
# ==========================================================================================
ORDER_G = [H.KNIGHT, H.ARROWS, H.GIANT, H.MINIONS, H.PRINCE, H.BABY_DRAGON, H.WIZARD, H.ZAP]


def parse_scene(team=0):
    g = game("giant", "giant", seed=1)
    ready(g, team, ORDER_G)
    return g, slot_base(L().render_state(g, team), ORDER_G[:4])


def test_parse_wait(pr):
    g, _ = parse_scene()
    for txt in ("WAIT", "wait", "  Wait  ", "I will hold.\nWAIT"):
        a, info = L().parse_action(txt, g, 0)
        assert a == 0 and not info.get("error"), (txt, info)


@pytest.mark.parametrize("team", [0, 1])
def test_parse_play_by_slot_and_by_card_name(pr, team):
    g, base = parse_scene(team)
    kn = H.action_id(0, 9, 21)
    for txt in ("PLAY Knight AT 9,21", "play knight at 9,21", "Play KNIGHT At 9,21", f"PLAY {base} AT 9,21"):
        a, info = L().parse_action(txt, g, team)
        assert a == kn and not info.get("error"), (txt, a, info)
    a, _ = L().parse_action(f"PLAY {base + 2} AT 4,20", g, team)
    assert a == H.action_id(2, 4, 20), "slot 2 (Giant)"
    a, _ = L().parse_action("PLAY Arrows AT 9,5", g, team)
    assert a == H.action_id(1, 9, 5), "spells may target the enemy half (own-frame tile)"


def test_parse_last_matching_line_wins(pr):
    g, base = parse_scene()
    cases = [("PLAY Knight AT 9,21\nWAIT", 0),
             ("WAIT\nThinking about it...\nPLAY Giant AT 9,21", H.action_id(2, 9, 21)),
             ("PLAY Knight AT 9,21\nPLAY Arrows AT 9,5", H.action_id(1, 9, 5)),
             ("PLAY Knight AT 9,21\nThat is my move.", H.action_id(0, 9, 21))]
    for txt, want in cases:
        a, info = L().parse_action(txt, g, 0)
        assert a == want and not info.get("error"), (txt, a, info)


@pytest.mark.parametrize("txt", ["PLAY 7 AT 9,21", "PLAY -1 AT 9,21", "PLAY Dragon Lord AT 9,21",
                                 "PLAY Wizard AT 9,21", "PLAY Knight AT 18,21", "PLAY Knight AT 9,32",
                                 "PLAY Knight AT 9,15", "PLAY Knight AT 9,5", "PLAY Knight", "hello there",
                                 "", "PLAY Knight AT nine,21"])
def test_parse_invalid_maps_to_noop_with_error(pr, txt):
    g, base = parse_scene()
    a, info = L().parse_action(txt, g, 0)
    assert a == 0 and isinstance(info, dict) and info.get("error"), f"{txt!r} -> {a}, {info}"


def test_parse_unaffordable_is_illegal(pr):
    g, base = parse_scene()
    g.set_elixir(0, 2 * H.ELIXIR_UNIT)
    a, info = L().parse_action("PLAY Giant AT 9,21", g, 0)
    assert a == 0 and info.get("error")


def test_parse_during_lockout_is_illegal(pr):
    g = K.new_game("giant", "giant", seed=1)
    a, info = L().parse_action("PLAY Knight AT 9,21" if H.KNIGHT in K.hand(g, 0) else "PLAY 1 AT 9,21", g, 0)
    assert a == 0 and info.get("error")


@pytest.mark.parametrize("decks", [("giant", "hog26"), ("pekka_bridge", "golem"), ("xbow", "miner_poison")])
@pytest.mark.parametrize("team", [0, 1])
def test_parse_round_trips_every_legal_action(pr, decks, team):
    g = game(*decks, seed=6)
    ready(g, team)
    hand = K.hand(g, team)
    base = slot_base(L().render_state(g, team), hand)
    legal = np.nonzero(np.asarray(g.legal_mask(team))[1:])[0] + 1
    assert len(legal) > 100
    for i, a in enumerate(legal):
        s, tx, ty = H.decode_action(int(a))
        for txt in (f"PLAY {base + s} AT {tx},{ty}", f"PLAY {NAMES[hand[s]]} AT {tx},{ty}"):
            got, info = L().parse_action(txt, g, team)
            assert got == a and not info.get("error"), f"{txt!r} -> {got} ({info}), want {a}"
        if i % 37 == 0:
            got, _ = L().parse_action(f"play {NAMES[hand[s]].lower()} at {tx},{ty}", g, team)
            assert got == a


# ==========================================================================================
# LLMAgent and play_llm_match
# ==========================================================================================
def test_agent_act_and_transcript_entry(pr):
    g, base = parse_scene()
    rec = Recorder("I'll defend.\nPLAY Knight AT 9,21")
    agent = L().LLMAgent(rec)
    a = agent.act(g, 0)
    assert a == H.action_id(0, 9, 21)
    assert len(rec.calls) == 1 and rec.calls[0][0] == L().RULES_PROMPT, "system prompt = RULES_PROMPT"
    assert L().render_state(g, 0) in rec.calls[0][1], "the user prompt carries the rendered state"
    assert len(agent.transcript) == 1
    e = agent.transcript[0]
    assert {"response", "action", "error"} <= set(e), f"transcript keys {sorted(e)}"
    assert phash(e), f"no prompt-hash key (prompt_hash / prompt_sha...) in {sorted(e)}"
    lat = [k for k in e if k.startswith("latency")]
    assert lat and float(e[lat[0]]) >= 0.0
    assert e["response"] == rec.reply and e["action"] == a and not e["error"]
    agent2 = L().LLMAgent(Recorder("no idea"))
    assert agent2.act(g, 0) == 0 and agent2.transcript[0]["error"]


def test_agent_without_history_sends_only_the_current_state(pr):
    """max_history=0 (default): the user prompt of a later call carries nothing from earlier turns."""
    g = game(seed=2)
    rec = Recorder("remember-marker-7Q\nWAIT")
    agent = L().LLMAgent(rec, max_history=0)
    agent.act(g, 0)
    g.tick(20)
    agent.act(g, 0)
    assert "remember-marker-7Q" not in rec.calls[1][1]
    agent2 = L().LLMAgent(Recorder("WAIT"), max_history=2)       # accepted
    assert agent2.act(g, 0) == 0


def test_agent_prompt_hash_is_deterministic(pr):
    g = game(seed=2)
    a1, a2 = L().LLMAgent(Recorder()), L().LLMAgent(Recorder())
    a1.act(g, 0)
    a2.act(g, 0)
    assert phash(a1.transcript[0]) == phash(a2.transcript[0])
    g.tick(20)
    a1.act(g, 0)
    assert phash(a1.transcript[1]) != phash(a1.transcript[0])


@pytest.mark.parametrize("k", [20, 40])
def test_decision_interval_is_respected(pr, k):
    rec = Recorder("WAIT")
    agent = L().LLMAgent(rec, decision_interval=k)
    res = L().play_llm_match(agent, "bot:noop", "hog26", "giant", seed=1, decision_interval=k, max_ticks=800)
    check_result(res, 800)
    assert res["ticks"] == 800
    n = len(rec.calls)
    assert n == res["decisions"] == len(agent.transcript) and n in (800 // k, math.ceil(800 / k)), n
    secs = [clock_tokens(u)[0] for _, u in rec.calls]
    secs = [m * 60 + s for m, s in secs]
    assert all(abs(b - a) == k // 20 for a, b in zip(secs, secs[1:])), f"clock steps {secs[:6]}"
    assert all(sysp == L().RULES_PROMPT for sysp, _ in rec.calls)


def test_rules_prompt_identical_across_matches_and_seeds(pr):
    rec = Recorder("WAIT")
    for seed, team in ((1, 0), (2, 1)):
        L().play_llm_match(L().LLMAgent(rec), "bot:random", "giant", "bait", seed=seed, agent_team=team,
                           max_ticks=400)
    assert len({s for s, _ in rec.calls}) == 1 and rec.calls[0][0] == L().RULES_PROMPT


def test_mock_wait_never_plays(pr):
    agent = L().LLMAgent(L().mock_wait)
    res = L().play_llm_match(agent, "bot:noop", "hog26", "giant", seed=2)
    check_result(res)
    assert res["result"] == 0 and res["end_reason"] == "DRAW" and list(res["crowns"]) == [0, 0]
    assert res["ticks"] == 6000 and res["illegal"] == 0 and res["parse_errors"] == 0
    assert len(agent.transcript) == res["decisions"] > 0 and all(e["action"] == 0 for e in agent.transcript)
    g = game()
    assert L().parse_action(L().mock_wait(L().RULES_PROMPT, L().render_state(g, 0)), g, 0)[0] == 0


def test_mock_first_legal_plays_first_affordable_card_at_first_listed_tile(pr):
    g = game("giant", "hog26", seed=3)
    order = [H.GIANT, H.KNIGHT, H.ARROWS, H.MINIONS, H.PRINCE, H.BABY_DRAGON, H.WIZARD, H.ZAP]
    ready(g, 0, order, 3 * H.ELIXIR_UNIT)                     # Giant (5) unaffordable -> Knight, slot 1
    text = L().render_state(g, 0)
    a, info = L().parse_action(L().mock_first_legal(L().RULES_PROMPT, text), g, 0)
    first_line = next(ln for ln in text.splitlines() if RE_TY.match(ln))   # first listed tile of the
    m = RE_TY.match(first_line)                                            # first playable slot's block
    first_tile = (int(m.group(3).split(",")[0].split("-")[0]), int(m.group(1)))
    assert not info.get("error") and a == H.action_id(1, *first_tile), (a, info, first_tile)
    ready(g, 0, order, 0)
    a, info = L().parse_action(L().mock_first_legal(L().RULES_PROMPT, L().render_state(g, 0)), g, 0)
    assert a == 0 and not info.get("error"), "nothing affordable -> WAIT"


def test_mock_first_legal_reads_the_rendered_text(pr):
    """Doctor the first slot's region lines down to one tile: the mock must pick that tile, so it
    parses the text it is given rather than reading the engine."""
    g = game("giant", "hog26", seed=3)
    ready(g, 0, ORDER_G)
    lines = L().render_state(g, 0).splitlines()
    idx = [i for i, ln in enumerate(lines) if RE_TY.match(ln)]
    first_block = [idx[0]]
    while first_block[-1] + 1 in idx:
        first_block.append(first_block[-1] + 1)
    new = re.sub(r"(ty\s*=\s*)\d+(\s*-\s*\d+)?", r"\g<1>20", lines[idx[0]], count=1)
    new = re.sub(r"(tx\s+).*$", r"\g<1>7-7", new)
    doctored = "\n".join(lines[:first_block[0]] + [new] + lines[first_block[-1] + 1:])
    assert regions(doctored)[0] == {(7, 20)}
    resp = L().mock_first_legal(L().RULES_PROMPT, doctored)
    a, info = L().parse_action(resp, g, 0)
    assert a == H.action_id(0, 7, 20), f"mock_first_legal answered {resp!r}"


def test_mock_first_legal_only_plays_legal_actions_and_really_plays(pr):
    agent = L().LLMAgent(L().mock_first_legal)
    res = L().play_llm_match(agent, "bot:noop", "hog26", "giant", seed=4)
    check_result(res)
    assert res["illegal"] == 0 and res["parse_errors"] == 0
    assert all(not e["error"] for e in agent.transcript)
    assert sum(e["action"] != 0 for e in agent.transcript) >= 5, "it must actually play cards"
    assert res["result"] in (0, 1), "a noop opponent can never beat it"


@pytest.mark.parametrize("agent_team", [0, 1])
def test_result_is_from_the_agents_perspective(pr, agent_team):
    agent = L().LLMAgent(L().mock_first_legal)
    res = L().play_llm_match(agent, "bot:noop", "giant", "hog26", seed=5, agent_team=agent_team)
    check_result(res)
    assert res["result"] in (0, 1) and int(res["crowns"][1]) == 0, \
        f"agent seat {agent_team}: result/crowns must be [agent, opponent]: {res}"


def test_play_llm_match_is_deterministic(pr):
    runs = []
    for _ in range(2):
        agent = L().LLMAgent(L().mock_random(7))
        res = L().play_llm_match(agent, "bot:heuristic", "royal_hogs", "xbow", seed=9, max_ticks=2400)
        check_result(res, 2400)
        det = {k: v for k, v in res.items() if "latency" not in k}           # wall-clock fields aside
        runs.append((det, [(e["response"], e["action"], phash(e)) for e in agent.transcript]))
    assert runs[0] == runs[1]
    assert runs[0][0]["parse_errors"] == 0, "mock_random answers in the response format"


def test_opponent_can_be_another_llm_agent(pr):
    me, opp = L().LLMAgent(L().mock_wait), L().LLMAgent(L().mock_first_legal)
    res = L().play_llm_match(me, opp, "hog26", "giant", seed=3)
    check_result(res)
    assert opp.transcript and any(e["action"] != 0 for e in opp.transcript)
    assert res["result"] in (-1, 0) and int(res["crowns"][0]) == 0, "a passive agent cannot win or score"


def test_opponent_can_be_a_checkpoint(pr, tmp_path):
    import leaguekit
    ckpt = leaguekit.make_ckpt(tmp_path / "opp.pt")
    agent = L().LLMAgent(L().mock_first_legal)
    res = L().play_llm_match(agent, f"ckpt:{ckpt}", "hog26", "giant", seed=1, max_ticks=1200)
    check_result(res, 1200)
    assert res["decisions"] == len(agent.transcript) > 0


def test_illegal_and_parse_error_counts(pr):
    """Decisions at ticks 0..80 all fall inside the default 90-tick lockout."""
    res = L().play_llm_match(L().LLMAgent(Recorder("PLAY 1 AT 9,21")), "bot:noop", "giant", "hog26", seed=1,
                             max_ticks=100)
    assert res["decisions"] == 5 and res["illegal"] == 5 and res["parse_errors"] == 0, res
    res = L().play_llm_match(L().LLMAgent(Recorder("I am not sure.")), "bot:noop", "giant", "hog26", seed=1,
                             max_ticks=100)
    assert res["decisions"] == 5 and res["parse_errors"] == 5 and res["illegal"] == 0, res


# ==========================================================================================
# No network: module import, adapter without the SDK
# ==========================================================================================
def test_guard_blocks_anthropic_in_subprocesses(pr, tmp_path):
    out, mark = run_guarded(tmp_path, code="import anthropic", timeout=120)
    assert out.returncode != 0 and "import anthropic" in mark, "the subprocess guard must be active"


def test_llm_module_import_does_not_touch_anthropic_or_network(pr, tmp_path):
    code = ("import sys, pufferroyale.llm as L\n"
            "assert 'anthropic' not in sys.modules\n"
            "assert callable(L.render_state) and callable(L.parse_action) and callable(L.play_llm_match)\n"
            "print('ok')")
    out, mark = run_guarded(tmp_path, code=code, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]
    assert mark == "", f"importing pufferroyale.llm touched: {mark}"


def test_anthropic_adapter_without_sdk_raises_a_clear_error(pr, tmp_path):
    code = ("import sys, pufferroyale.llm as L\n"
            "fn = getattr(L, 'anthropic_model_fn', None)\n"
            "if fn is None:\n"
            "    import importlib\n"
            "    fn = importlib.import_module('pufferroyale.llm_anthropic').anthropic_model_fn\n"
            "try:\n"
            "    f = fn('claude-opus-5-5')\n"
            "    f(L.RULES_PROMPT, 'state')\n"
            "except Exception as e:\n"
            "    print('ERR', type(e).__name__, e)\n"
            "    sys.exit(3)\n"
            "print('NOERR')")
    out, mark = run_guarded(tmp_path, code=code, timeout=300)
    assert out.returncode == 3 and "ERR" in out.stdout, (out.stdout[-500:], out.stderr[-1500:])
    assert re.search(r"(?i)anthropic", out.stdout.split("ERR", 1)[1]), f"unclear error: {out.stdout!r}"
    assert "network" not in mark, "the adapter attempted a network connection"


# ==========================================================================================
# scripts/llm_match.py
# ==========================================================================================
FLAGS = {"--model", "--opponent", "--deck-agent", "--deck-opp", "--matches", "--seed", "--decision-interval",
         "--out", "--effort", "--allow-network"}


def test_llm_match_help_flags(pr, tmp_path):
    out, mark = run_guarded(tmp_path, [SCRIPT, "--help"], timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    flags = set(re.findall(r"(--[A-Za-z0-9][A-Za-z0-9_-]*)", out.stdout + out.stderr))
    assert FLAGS <= flags, f"missing flags {sorted(FLAGS - flags)}"
    assert mark == ""


def _summary_value(summary, *pats):
    keys = [k for k in summary if any(re.search(p, k, re.I) for p in pats)]
    assert keys, f"summary has no key matching {pats}: {sorted(summary)}"
    return keys, [summary[k] for k in keys]


def test_llm_match_mock_run_summary_and_transcripts(pr, tmp_path):
    outdir = tmp_path / "res"
    outdir.mkdir()
    args = [SCRIPT, "--model", "mock_first_legal", "--opponent", "bot:noop", "--deck-agent", "hog26",
            "--deck-opp", "giant", "--matches", "2", "--seed", "3", "--decision-interval", "40",
            "--out", str(outdir / "summary.json")]
    out, mark = run_guarded(tmp_path, args)
    assert out.returncode == 0, (out.stdout[-1500:], out.stderr[-2500:])
    assert mark == "", f"a mock run touched: {mark}"
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    summary = json.loads(lines[-1])
    assert isinstance(summary, dict)
    wdl = []
    for pat in (r"^wins?\b|^win_|^wins?$", r"^draws?\b|^draw_|^draws?$", r"^loss(es)?\b|^loss_|^losses$"):
        keys, vals = _summary_value(summary, pat)
        wdl.append(float(vals[0]))
    assert abs(sum(wdl) - 2) < 1e-9 or abs(sum(wdl) - 1) < 1e-9, f"win/draw/loss {wdl} (counts for 2 or rates)"
    assert wdl[2] == 0, "mock_first_legal never loses to a noop bot"
    _summary_value(summary, r"crown")
    _, ill = _summary_value(summary, r"illegal")
    _, par = _summary_value(summary, r"parse")
    assert all(float(v) == 0 for v in ill + par), "mock_first_legal: no illegal or parse-error decisions"
    _, lat = _summary_value(summary, r"latency")
    assert all(float(v) >= 0 for v in lat)
    jl = sorted(outdir.glob("*.jsonl"))
    assert jl, f"no JSONL transcripts next to --out: {sorted(p.name for p in outdir.iterdir())}"
    n, seats = 0, set()
    for p in jl:
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                assert isinstance(rec, dict) and "action" in rec and "response" in rec, sorted(rec)
                seat_key = next((k for k in ("agent_team", "seat") if k in rec), None)
                if seat_key is not None:
                    seats.add(int(rec[seat_key]))
                n += 1
    assert n >= 20, f"only {n} transcript records for 2 matches"
    if seats:
        assert seats == {0, 1}, f"2 matches must alternate the agent's seat, saw seats {seats}"


def test_llm_match_refuses_anthropic_without_allow_network(pr, tmp_path):
    outdir = tmp_path / "res"
    outdir.mkdir()
    args = [SCRIPT, "--model", "anthropic:claude-opus-5-5", "--opponent", "bot:noop", "--deck-agent", "hog26",
            "--deck-opp", "giant", "--matches", "1", "--seed", "0", "--decision-interval", "20",
            "--out", str(outdir / "summary.json")]
    out, mark = run_guarded(tmp_path, args, timeout=300)
    assert out.returncode != 0, "anthropic: models must be refused without --allow-network"
    assert "allow-network" in (out.stdout + out.stderr), "the refusal should name --allow-network"
    assert mark == "", f"refusal must come before any anthropic import or network use: {mark}"
    assert not list(outdir.glob("*.jsonl")), "nothing is played"
