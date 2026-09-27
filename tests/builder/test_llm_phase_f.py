"""Builder tests for Phase F (SPEC §17): the LLM text interface. Everything runs offline with
deterministic mock model functions or a fake `anthropic` module; nothing touches the network."""
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

import pufferroyale as pr
from pufferroyale import llm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bot_game(ticks, seed=3, deck0="hog26", deck1="giant", step=10):
    """A game advanced `ticks` ticks with two heuristic bots (a realistic mid-match state)."""
    g = pr.Game(deck0=deck0, deck1=deck1, seed=seed)
    bots = (pr.Bot("heuristic", seed * 2), pr.Bot("heuristic", seed * 2 + 1))
    while g.state()["tick"] < ticks and not g.state()["over"]:
        for team in (0, 1):
            a = bots[team].act(g, team)
            if a:
                g.play_action(team, a)
        g.tick(step)
    return g


def mask_tiles(g, team, slot):
    m = g.legal_mask(team)[1 + slot * 576:1 + (slot + 1) * 576]
    return {(int(c) % 18, int(c) // 18) for c in np.flatnonzero(m)}


# ------------------------------------------------------------------------------------ RULES_PROMPT
def test_rules_prompt_is_static_and_complete():
    assert llm.RULES_PROMPT == llm._build_rules_prompt(), "must be identical on every build (cacheable)"
    t = llm.RULES_PROMPT
    for c, name in enumerate(pr.CARD_NAMES):
        assert f"- {name} ({pr.CARD_COSTS[c]}): " in t, f"card list lacks {name}"
    for needle in ("ty 31", "ty 0-14", "WAIT", "PLAY <slot> AT <tx>,<ty>", "own frame", "deploy lockout"):
        assert needle.lower() in t.lower(), needle
    assert "lasts 30 s" in t and "lasts 3 s" not in t, "number formatting"
    import re
    assert not re.search(r"20\d\d-\d\d-\d\d|\d\d:\d\d:\d\d", t), "no dates / clock times (cacheable)"


# ------------------------------------------------------------------------------------ render_state
def test_render_sections_and_determinism():
    g = bot_game(1400)
    a, b = llm.render_state(g, 0), llm.render_state(g, 0)
    assert a == b
    for sec in ("TIME ", "CROWNS you", "YOUR ELIXIR", "YOUR HAND", "next:", "YOUR TOWERS:", "ENEMY TOWERS:",
                "YOUR UNITS", "ENEMY UNITS:", "OPPONENT CARDS SEEN", "OPPONENT LAST PLAYED", "OPPONENT ELIXIR: at most",
                "LEGAL TILES"):
        assert sec in a, sec
    st = g.state()
    assert f"YOUR ELIXIR {int(st['elixir'][0] * 10 // 2800) / 10:.1f} / 10" in a
    assert "TIME 1:50 left (regulation)" in a
    no_legal = llm.render_state(g, 0, legal=False)
    assert "LEGAL TILES" not in no_legal and no_legal.splitlines()[0] == a.splitlines()[0]


def test_render_lockout_and_unaffordable_lines():
    g = pr.Game(deck0="hog26", deck1="giant", seed=0)
    t = llm.render_state(g, 1)
    assert "DEPLOY LOCKOUT" in t and "TIME 3:00 left (regulation)" in t
    assert t.count("none (deploy lockout)") == 4 and "- deploy lockout" in t
    g2 = pr.Game(deck0="hog26", deck1="giant", seed=0, deploy_lockout_ticks=0)
    g2.set_elixir(0, 2800)                                  # 1 elixir
    t2 = llm.render_state(g2, 0)
    hand = g2.state()["hand"][0]
    for s, c in enumerate(hand):
        if pr.CARD_COSTS[c] > 1:
            assert f"slot {s} {pr.CARD_NAMES[c]}: none (costs {pr.CARD_COSTS[c]}, you have 1.0)" in t2
            assert f"{s}: {pr.CARD_NAMES[c]} ({pr.CARD_COSTS[c]}) - need {pr.CARD_COSTS[c] - 1:.1f} more elixir" in t2


@pytest.mark.parametrize("ticks,team", [(100, 0), (1400, 1), (2600, 0), (3300, 1)])
def test_legal_text_is_exactly_the_mask(ticks, team):
    """The LEGAL TILES text is complete: parsing it gives exactly the engine's legal tiles."""
    g = bot_game(ticks, seed=7)
    parsed = llm.parse_rendered_legal(llm.render_state(g, team))
    for s in range(4):
        assert set(parsed.get(s, [])) == mask_tiles(g, team, s), f"slot {s}"
    hand = llm.parse_rendered_hand(llm.render_state(g, team))
    assert [h["slot"] for h in hand] == [0, 1, 2, 3]
    assert [h["playable"] for h in hand] == [bool(mask_tiles(g, team, s)) for s in range(4)]


def test_render_never_leaks_opponent_hidden_info():
    """Changing the opponent's hand order, queue, unrevealed deck and elixir via the debug API leaves
    the viewer's text unchanged (only the §9 deductions -- cards seen, last played, spent-based
    elixir bound -- appear, and none of them depends on those)."""
    g = bot_game(1500, seed=11)
    for viewer in (0, 1):
        opp = 1 - viewer
        base = llm.render_state(g, viewer)
        snap = g.snapshot()
        st = g.state()
        order = st["queue"][opp][::-1] + st["hand"][opp][::-1]
        g.set_hand(opp, order)                               # new hand + queue order, same deck
        g.set_elixir(opp, 0)
        assert llm.render_state(g, viewer) == base, "opponent hand/queue/elixir leaked"
        g.set_elixir(opp, 28000)
        assert llm.render_state(g, viewer) == base
        unseen = [c for c in range(pr.N_CARDS) if c not in st["deck"][opp]][:8]
        g.set_hand(opp, unseen)                              # a different (unrevealed) deck
        assert llm.render_state(g, viewer) == base, "unrevealed deck leaked"
        g.restore(snap)
        g.set_elixir(viewer, 0)                              # sanity: the test is sensitive
        assert llm.render_state(g, viewer) != base
        g.restore(snap)


def test_deduced_hand_shown_only_after_eight_cards_and_stable_under_hand_changes():
    g = pr.Game(deck0="hog26", deck1="bait", seed=4, deploy_lockout_ticks=0)
    seen_all = False
    for i in range(12):                     # team 1 plays slot i % 4 every second: it reveals its deck
        g.set_elixir(1, 28000)
        s = i % 4
        tiles = np.flatnonzero(g.legal_mask(1)[1 + s * 576:1 + (s + 1) * 576])
        g.play_action(1, 1 + s * 576 + int(tiles[len(tiles) // 2]))
        g.tick(20)
        text = llm.render_state(g, 0)
        seen = text.split("OPPONENT CARDS SEEN (")[1].split(" of 8")[0]
        assert ("OPPONENT HAND (deduced" in text) == (seen == "8")
        if seen == "8":
            seen_all = True
            break
    assert seen_all, "setup: the random bot should reveal its whole deck"
    last = g.state()["last_played"][1]
    deck = g.state()["deck"][1]
    want = sorted(pr.CARD_NAMES[c] for c in deck if c not in last)
    line = [l for l in text.splitlines() if l.startswith("OPPONENT HAND (deduced")][0]
    assert sorted(line.split("): ")[1].split(", ")) == want
    st = g.state()
    g.set_hand(1, st["hand"][1][::-1] + st["queue"][1])
    g.set_elixir(1, 1000)
    assert llm.render_state(g, 0) == text


def test_render_is_seat_symmetric():
    """A scenario and its 180-degree rotation give the same text to the two seats."""
    X, Y = list(pr.DECKS["hog26"]), list(pr.DECKS["giant"])
    a = pr.Game(deck0=X, deck1=Y, seed=1, deploy_lockout_ticks=0)
    b = pr.Game(deck0=Y, deck1=X, seed=1, deploy_lockout_ticks=0)
    a.set_hand(0, X), a.set_hand(1, Y)
    b.set_hand(1, X), b.set_hand(0, Y)
    a.spawn(0, "Knight", 4500, 22500), a.spawn(1, "Giant", 14500, 9500), a.spawn(1, "Minions", 9000, 12000)
    b.spawn(1, "Knight", 13500, 9500), b.spawn(0, "Giant", 3500, 22500), b.spawn(0, "Minions", 9000, 20000)
    for _ in range(4):
        assert llm.render_state(a, 0) == llm.render_state(b, 1)
        assert llm.render_state(a, 1) == llm.render_state(b, 0)
        a.tick(25)
        b.tick(25)


def test_units_merge_identical_lines_and_show_flags():
    g = pr.Game(deck0="bait", deck1="giant", seed=2, deploy_lockout_ticks=0)
    g.spawn(0, "Skeleton Army", 9000, 22000, deployed=False)
    g.spawn(1, "Minions", 9000, 8000)
    t = llm.render_state(g, 0)
    own = t.split("YOUR UNITS")[1].split("ENEMY UNITS")[0]
    assert "Skeleton Army" in own and "[deploying]" in own and "x Skeleton Army" in own
    enemy = t.split("ENEMY UNITS:")[1].split("OPPONENT")[0]
    assert "Minions" in enemy and "[air]" in enemy


# ------------------------------------------------------------------------------------ parse_action
def lockout_free_game():
    g = pr.Game(deck0="hog26", deck1="giant", seed=3)
    g.tick(120)
    return g


@pytest.mark.parametrize("reply,ok", [
    ("WAIT", True), ("wait.", True), ("Let me think.\nPLAY 0 AT 9,20", True), ("**PLAY 0 AT (9, 20)**", True),
    ("Action: play 0 at 9 20", True), ("PLAY 0 AT 9,20\nActually no.\nWAIT", True),
])
def test_parse_action_accepts_formats(reply, ok):
    g = lockout_free_game()
    a, info = llm.parse_action(reply, g, 0)
    assert "error" not in info, info
    want = 0 if reply.strip().lower().endswith(("wait", "wait.")) else 1 + 0 * 576 + 20 * 18 + 9
    assert a == want


def test_parse_action_by_card_name_and_errors():
    g = lockout_free_game()
    hand = g.state()["hand"][0]
    name = pr.CARD_NAMES[hand[2]]
    a, info = llm.parse_action(f"PLAY {name.upper()} AT 9,20", g, 0)
    assert a == 1 + 2 * 576 + 20 * 18 + 9 and info["slot"] == 2
    absent = next(c for c in range(pr.N_CARDS) if c not in g.state()["deck"][0])
    cases = {
        "I have no idea": "parse", "PLAY 7 AT 9,20": "parse", "PLAY Dragonzilla AT 9,20": "parse",
        f"PLAY {pr.CARD_NAMES[absent]} AT 9,20": "parse", "PLAY 0 AT 9,5": "illegal", "PLAY 0 AT 18,20": "illegal",
        "PLAY 0 AT 9,15": "illegal",
    }
    for reply, kind in cases.items():
        a, info = llm.parse_action(reply, g, 0)
        assert a == 0 and info.get("error") and info["error_kind"] == kind, (reply, info)
    early = pr.Game(deck0="hog26", deck1="giant", seed=3)
    a, info = llm.parse_action("PLAY 0 AT 9,20", early, 0)
    assert a == 0 and "lockout" in info["error"]


# ------------------------------------------------------------------------------------ mocks
def test_mock_first_legal_reads_only_text():
    synthetic = "\n".join([
        "YOUR HAND (slot: card (elixir)):", "  0: Giant (5) - need 1.0 more elixir", "  1: Zap (2) - playable",
        "  2: Knight (3) - playable", "  3: Arrows (3) - playable", "  next: Prince (5)",
        "YOUR TOWERS: ...", 'LEGAL TILES per hand slot (own frame; "ty=<rows>: tx <columns>"):',
        "slot 0 Giant: none (costs 5, you have 4.0)", "slot 1 Zap:", "  ty=3-5: tx 7,9-10", "slot 2 Knight:",
        "  ty=17: tx 1-16", "slot 3 Arrows:", "  ty=0-31: tx 0-17", "Reply ..."])
    assert llm.mock_first_legal("", synthetic) == "PLAY 1 AT 7,3"
    assert llm.mock_first_legal("", "nothing here") == "WAIT"
    for ticks in (0, 300, 1500):
        g = bot_game(ticks, seed=5)
        reply = llm.mock_first_legal(llm.RULES_PROMPT, llm.render_state(g, 1))
        a, info = llm.parse_action(reply, g, 1)
        assert "error" not in info
        legal = g.legal_mask(1)
        if legal[1:].any():
            first_slot = min(s for s in range(4) if mask_tiles(g, 1, s))
            tile = min(mask_tiles(g, 1, first_slot), key=lambda p: (p[1], p[0]))
            assert a == 1 + first_slot * 576 + tile[1] * 18 + tile[0]
        else:
            assert a == 0


def test_mock_random_is_seeded_and_legal():
    g = bot_game(900, seed=6)
    text = llm.render_state(g, 0)
    f1, f2 = llm.mock_random(3), llm.mock_random(3)
    r1 = [f1("", text) for _ in range(30)]
    assert r1 == [f2("", text) for _ in range(30)]
    assert any(r.startswith("PLAY") for r in r1)
    for r in r1:
        assert "error" not in llm.parse_action(r, g, 0)[1]


def test_mock_random_is_uniform_over_wait_and_listed_commands():
    """SPEC §17.4.8: uniformly one of WAIT and the legal commands listed in the text."""
    text = "\n".join(["YOUR HAND (slot: card (elixir)):", "  0: Zap (2) - playable", "  1: Giant (5) - need 3.0 more elixir",
                      "  2: Knight (3) - no legal tile", "  3: Arrows (3) - playable", "YOUR TOWERS: ...",
                      'LEGAL TILES per hand slot (own frame; "ty=<rows>: tx <columns>"):', "slot 0 Zap:", "  ty=3: tx 7-8",
                      "slot 1 Giant: none (costs 5, you have 2.0)", "slot 2 Knight: none (no free tile)", "slot 3 Arrows:",
                      "  ty=9: tx 4", "Reply ..."])
    f = llm.mock_random(0)
    draws = [f("", text) for _ in range(4000)]
    want = {"WAIT", "PLAY 0 AT 7,3", "PLAY 0 AT 8,3", "PLAY 3 AT 4,9"}
    assert set(draws) == want
    for c in want:
        assert abs(draws.count(c) / 4000 - 0.25) < 0.04, (c, draws.count(c))


# ------------------------------------------------------------------------------------ agent + matches
def test_agent_transcript_history_and_model_errors():
    g = lockout_free_game()
    seen = []

    def fn(system, user):
        seen.append((system, user))
        return "PLAY 9 AT 0,0" if len(seen) == 1 else "PLAY 0 AT 9,20"

    ag = llm.LLMAgent(fn, max_history=2)
    assert ag.act(g, 0) == 0
    assert seen[0][0] == llm.RULES_PROMPT
    e = ag.transcript[0]
    assert e["error_kind"] == "parse" and e["decoded"] == "WAIT" and len(e["prompt_sha256"]) == 16
    assert e["response"] == "PLAY 9 AT 0,0" and e["latency_s"] >= 0 and e["tick"] == 120
    assert ag.act(g, 0) == 1 + 20 * 18 + 9
    # SPEC §17.4.7: the last k (state, response) pairs as prior turns
    first_state = llm.render_state(g, 0)
    assert seen[1][1].startswith("PREVIOUS TURNS (your last 1, oldest first):")
    assert first_state in seen[1][1] and "PLAY 9 AT 0,0" in seen[1][1] and "treated as WAIT" in seen[1][1]
    assert seen[1][1].endswith(first_state), "the current state comes last"
    ag.act(g, 0)
    ag.act(g, 0)
    assert seen[3][1].count("=== turn -") == 4, "max_history=2 keeps two (state, reply) pairs"

    def boom(system, user):
        raise RuntimeError("model down")

    bad = llm.LLMAgent(boom)
    assert bad.act(g, 0) == 0 and bad.transcript[0]["error_kind"] == "model"


def test_play_llm_match_counts_and_determinism():
    r = llm.play_llm_match(llm.LLMAgent(llm.mock_wait), "bot:noop", "hog26", "giant", seed=1)
    assert r["result"] == 0 and r["end_reason"] == "DRAW" and r["ticks"] == 6000
    assert r["decisions"] == 300 and r["plays"] == 0 and r["waits"] == 300 and r["crowns"] == [0, 0]
    runs = []
    for _ in range(2):
        ag = llm.LLMAgent(llm.mock_first_legal)
        runs.append((llm.play_llm_match(ag, "bot:heuristic", "hog26", "giant", seed=9, agent_team=1),
                     [(e["tick"], e["action"]) for e in ag.transcript]))
    for r_ in runs:
        r_[0].pop("mean_latency_s")                          # wall-clock, not part of the game
    assert runs[0] == runs[1]
    res = runs[0][0]
    assert res["agent_team"] == 1 and res["illegal"] == 0 and res["parse_errors"] == 0 and res["plays"] > 10
    slow = llm.play_llm_match(llm.LLMAgent(llm.mock_wait, decision_interval=40), "bot:noop", "hog26", "hog26", seed=1,
                              max_ticks=400)
    assert slow["decisions"] == 10, "SPEC §17.4.10: LLMAgent.decision_interval is the default cadence"
    cut = llm.play_llm_match(llm.LLMAgent(llm.mock_wait), "bot:noop", "hog26", "hog26", seed=1, max_ticks=400,
                             decision_interval=40)
    assert cut["ticks"] == 400 and not cut["finished"] and cut["end_reason"] is None and cut["decisions"] == 10
    skip = llm.play_llm_match(llm.LLMAgent(llm.mock_wait), "bot:noop", "hog26", "hog26", seed=1, max_ticks=400,
                              skip_idle=True)
    assert skip["skipped"] == 5 and skip["decisions"] == 15, "the 5 decisions inside the lockout are skipped"


def test_first_legal_beats_noop_and_llm_vs_llm(tmp_path):
    r = llm.play_llm_match(llm.LLMAgent(llm.mock_first_legal), "bot:noop", "hog26", "hog26", seed=2)
    assert r["result"] == 1
    opp = llm.LLMAgent(llm.mock_random(1), decision_interval=40)
    ag = llm.LLMAgent(llm.mock_first_legal)
    r2 = llm.play_llm_match(ag, opp, "hog26", "bait", seed=3, max_ticks=2000)
    assert r2["decisions"] == 100 and len(opp.transcript) == 50
    assert all(e["team"] == 1 for e in opp.transcript) and all(e["team"] == 0 for e in ag.transcript)


def test_checkpoint_opponent(tmp_path):
    import torch
    from pufferroyale.torch import Policy
    env = pr.Royale(num_envs=1, num_agents=2)
    torch.manual_seed(0)
    path = tmp_path / "p.pt"
    torch.save(Policy(env).state_dict(), path)
    env.close()
    r = llm.play_llm_match(llm.LLMAgent(llm.mock_wait), str(path), "hog26", "hog26", seed=0, max_ticks=600)
    assert r["ticks"] == 600 and r["decisions"] == 30


# ------------------------------------------------------------------------------------ Anthropic adapter (offline)
def fake_anthropic():
    mod = types.ModuleType("anthropic")

    class APIError(Exception):
        def __init__(self, message="err"):
            super().__init__(message)
            self.message = message

    class APIStatusError(APIError):
        def __init__(self, message="status", status_code=500):
            super().__init__(message)
            self.status_code = status_code

    class RateLimitError(APIStatusError):
        def __init__(self, message="slow down"):
            super().__init__(message, 429)

    class APIConnectionError(APIError):
        pass

    def no_client(*a, **k):
        raise AssertionError("tests must inject a fake client; never build a real one")

    mod.APIError, mod.APIStatusError, mod.RateLimitError, mod.APIConnectionError = (
        APIError, APIStatusError, RateLimitError, APIConnectionError)
    mod.Anthropic = no_client
    return mod


class FakeClient:
    def __init__(self, outcomes):
        self.calls, self.outcomes = [], list(outcomes)
        self.messages = types.SimpleNamespace(create=lambda **kw: self._call("messages", kw))
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **kw: self._call("beta", kw)))

    def _call(self, path, kw):
        self.calls.append((path, kw))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def reply(text, stop="end_turn"):
    return types.SimpleNamespace(
        stop_reason=stop, model="m", stop_details=types.SimpleNamespace(category="cyber", explanation="x"),
        content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=text)],
        usage=types.SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3000,
                                    cache_creation_input_tokens=0))


def test_anthropic_adapter_missing_sdk_is_a_clear_error():
    if importlib.util.find_spec("anthropic") is not None:
        pytest.skip("the anthropic SDK is installed here")
    with pytest.raises(ImportError, match="pip install anthropic"):
        llm.anthropic_model_fn("claude-opus-5-5")


def test_anthropic_adapter_requests_offline(monkeypatch):
    mod = fake_anthropic()
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    c = FakeClient([reply("thinking...\nPLAY 1 AT 9,20")])
    fn = llm.anthropic_model_fn("claude-opus-5-5", effort="low", client=c)
    assert fn("SYS", "STATE") == "thinking...\nPLAY 1 AT 9,20"
    path, kw = c.calls[0]
    assert path == "messages" and kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 4096
    assert kw["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert kw["messages"] == [{"role": "user", "content": "STATE"}], "single user turn, no prefill"
    assert kw["output_config"] == {"effort": "low"} and "thinking" not in kw
    assert fn.last_call["usage"]["cache_read_input_tokens"] == 3000
    c = FakeClient([reply("WAIT")])
    llm.anthropic_model_fn("claude-haiku-4-5", client=c)("S", "U")
    assert "output_config" not in c.calls[0][1], "effort is not supported on claude-haiku-4-5"
    for model in ("claude-fable-5-1", "claude-opus-5"):
        c = FakeClient([reply("WAIT")])
        llm.anthropic_model_fn(model, client=c)("S", "U")
        path, kw = c.calls[0]
        assert path == "beta" and kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    c = FakeClient([reply("", stop="refusal"), mod.RateLimitError(), mod.APIStatusError("bad", 400),
                    mod.APIConnectionError("offline")])
    fn = llm.anthropic_model_fn("claude-sonnet-5", client=c)
    assert fn("S", "U") == "WAIT" and fn.last_call["refusal"]["category"] == "cyber"
    assert fn("S", "U") == "WAIT" and fn.last_call["error"].startswith("rate_limit")
    assert fn("S", "U") == "WAIT" and "400" in fn.last_call["error"]
    assert fn("S", "U") == "WAIT" and fn.last_call["error"].startswith("connection")
    g = lockout_free_game()
    c = FakeClient([reply("", stop="refusal")])
    ag = llm.LLMAgent(llm.anthropic_model_fn("claude-opus-5-5", client=c))
    assert ag.act(g, 0) == 0 and ag.transcript[0]["meta"]["refusal"]


def test_make_model_fn_refuses_network_without_permission():
    with pytest.raises(PermissionError):
        llm.make_model_fn("anthropic:claude-opus-5-5")
    assert llm.make_model_fn("mock_wait") is llm.mock_wait
    with pytest.raises(ValueError):
        llm.make_model_fn("gpt")


# ------------------------------------------------------------------------------------ script
def test_llm_match_script(tmp_path):
    out = tmp_path / "r.json"
    p = subprocess.run([sys.executable, "scripts/llm_match.py", "--model", "mock_first_legal", "--opponent", "bot:random",
                        "--matches", "2", "--seed", "4", "--out", str(out)], cwd=ROOT, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    s = json.loads(p.stdout.strip().splitlines()[-1])
    for k in ("wins", "draws", "losses", "mean_crowns", "illegal_rate", "parse_error_rate", "mean_latency_s"):
        assert k in s
    assert s["matches"] == 2 and s["wins"] + s["draws"] + s["losses"] == 2 and s["illegal_rate"] == 0
    rows = [json.loads(l) for l in open(tmp_path / "r.transcripts.jsonl")]
    assert len(rows) == s["decisions"] and {r["agent_team"] for r in rows} == {0, 1}
    assert json.load(open(out))["summary"] == s
    p = subprocess.run([sys.executable, "scripts/llm_match.py", "--model", "anthropic:claude-opus-5-5", "--matches", "1"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2 and "--allow-network" in p.stderr
    if importlib.util.find_spec("anthropic") is None:
        p = subprocess.run([sys.executable, "scripts/llm_match.py", "--model", "anthropic:claude-opus-5-5", "--matches",
                            "1", "--allow-network"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert p.returncode == 2 and "pip install anthropic" in p.stderr
