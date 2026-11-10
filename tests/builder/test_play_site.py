"""Builder tests for play/play_server.py (the browser human-play mode)."""
import glob
import json
import os
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import numpy as np
import pytest

import pufferroyale as pr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "play"))
import play_server as ph  # noqa: E402


@pytest.fixture(scope="module")
def policy_player():
    """A random-init policy of the default architecture (the same code path as a checkpoint)."""
    import torch
    from watch import PolicyPlayer
    torch.set_num_threads(2)            # as the server does; torch's default (all cores) is ~300x slower on a busy node
    torch.manual_seed(0)                # the random init comes from torch's global generator: fixed weights, so a run
    return PolicyPlayer(None, greedy=False, seed=0)   # never draws a policy that almost never plays (seen 2026-11-09)


def make_args(tmp_path, **kw):
    a = dict(checkpoint="test.pt", label="test policy", greedy=False, data_dir=str(tmp_path / "data"), max_players=5,
             max_replays=5, claude_relay_dir=None, claude_daily_calls=200, comment_every=200, claude_timeout=120.0,
             threads=2, voice_dir=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


def make_server(tmp_path, policy_player, **kw):
    return ph.Server(make_args(tmp_path, **kw), player=policy_player)


def params(seat=0, deck="hog26", opp="giant", speed=1.0):
    label, ids = ph.deck_spec(deck, pr)
    olabel, oids = ph.deck_spec(opp, pr, allow_random=True)
    return {"seat": seat, "deck_label": label, "deck_ids": ids, "opp_label": olabel, "opp_ids": oids, "speed": speed}


def live(server, seat=0, owner=False, **kw):
    s = ph.Session("p" * 16, "Tester", owner, "match")
    return ph.LiveMatch(server, s, params(seat, **kw))


def run_to(m, tick):
    while m.game.tick_count < tick and not m.done:
        m.step()


def troop_slot(game, team):
    for s, c in enumerate(game.state()["hand"][team]):
        if pr.card_info(int(c)).get("kind") == "troop":
            return s, int(c)
    pytest.skip("no troop in hand")


# ---------------------------------------------------------------------------------------------- pure helpers
@pytest.mark.parametrize("team", [0, 1])
def test_to_frame_matches_the_engine(team):
    g = pr.Game(deck0="hog26", deck1="hog26", seed=0, deploy_lockout_ticks=0)
    g.set_elixir(team, 28000)
    slot, _ = troop_slot(g, team)
    before = {e["id"] for e in g.entities()}
    assert g.play_tile(team, slot, 3, 20) == pr.PlayError.OK
    g.tick(1)
    new = [e for e in g.entities() if e["id"] not in before]
    want = ph.to_frame(team, 3500, 20500)
    assert new and abs(np.mean([e["x"] for e in new]) - want[0]) <= 600 and abs(np.mean([e["y"] for e in new]) - want[1]) <= 600
    assert ph.to_frame(team, *ph.to_frame(team, 1234, 5678)) == (1234, 5678)


def test_terrain_rows():
    g = pr.Game(seed=0)
    t0, t1 = ph.terrain_rows(g, 0), ph.terrain_rows(g, 1)
    assert len(t0) == 32 and all(len(r) == 18 for r in t0)
    assert t0[15][3] == "=" and t0[15][14] == "=" and t0[15][0] == "~"
    assert t1 == [r[::-1] for r in t0[::-1]]
    assert set("".join(t0)) == set(".:~=T") and "".join(t0).count("T") == 2 * (16 + 9 + 9)   # King 4x4, Princess 3x3
    assert all(t0[ty][tx] == "T" for ty in range(27, 31) for tx in range(7, 11))           # the bottom King


def test_deck_spec_and_names():
    assert ph.deck_spec("bait", pr) == ("bait", list(pr.DECKS["bait"]))
    assert ph.deck_spec(list(range(8)), pr) == ("custom", list(range(8)))
    for bad in ("xbow", "random", [1] * 8, list(range(7)), list(range(60, 68)), 5):
        with pytest.raises((ValueError, TypeError)):
            ph.deck_spec(bad, pr)
    assert ph.deck_spec("random", pr, allow_random=True) == ("random", "random")
    assert set(ph.POOL_DECKS) <= set(pr.DECKS) and len(ph.POOL_DECKS) == 7
    assert ph.clean_name("  <b>Ann</b>   O'Neil!!  ") == "bAnnb O'Neil"
    assert ph.clean_name("x" * 40) == "x" * 24 and ph.clean_name("<>") == ""


def test_parse_claude_reply():
    assert ph.parse_claude_reply('thinking...\n{"commentary": "Nice push.", "p_win": 0.62}') == ("Nice push.", 0.62)
    assert ph.parse_claude_reply('{"commentary": "a"} then {"commentary": "b", "p_win": 1.7}') == ("b", 1.0)
    assert ph.parse_claude_reply('{"commentary": "c", "p_win": "nan"}') == ("c", None)
    assert ph.parse_claude_reply("WAIT") is None and ph.parse_claude_reply("") is None
    # the user's rule: at most 10 words, no numbers
    assert ph.clean_commentary("Your tower at 256 HP crumbles like a sandcastle, truly tragic stuff here folks") == \
        "Your tower at HP crumbles like a sandcastle!"                       # cut at the comma, not mid-phrase
    assert ph.clean_commentary("Gladiators frozen at the gates, the robot overlord sharpens its claws for dessert!") == \
        "Gladiators frozen at the gates!"
    assert ph.clean_commentary("Robot overlords everywhere just quietly sharpening their metal claws on the bridge") == \
        "Robot overlords everywhere just quietly sharpening their metal claws!"  # no break: 10 words, no dangling "on"
    assert ph.clean_commentary("Short and sweet!") == "Short and sweet!"
    assert ph.clean_commentary("Level 11 Hog Rider? Bold!") == "Level Hog Rider? Bold!"
    assert len(ph.parse_claude_reply('{"commentary": "' + "word " * 30 + '", "p_win": 0.4}')[0].split()) == 10


# ---------------------------------------------------------------------------------------------- the match
@pytest.mark.parametrize("seat", [0, 1])
def test_beat_and_play(tmp_path, policy_player, seat):
    srv = make_server(tmp_path, policy_player)
    m = live(srv, seat=seat)
    card = m.game.state()["hand"][m.ht][0]
    ok, why = m.queue_play(card, 9, 25)
    assert not ok and "lockout" in why
    run_to(m, 100)
    m.game.set_elixir(m.ht, 28000)
    slot, card = troop_slot(m.game, m.ht)
    assert m.queue_play(card, 9, 25)[0]
    run_to(m, 101)
    ev = [e for e in m.events if e["who"] == "human"]
    assert len(ev) == 1 and ev[0]["result"] == "OK" and ev[0]["tick"] == 100 and ev[0]["xy_human_frame"] == [9500, 25500]
    assert [p for p in m.plays if p[1] == m.ht] == [[100, m.ht, slot, 9, 25, card]]
    view = json.loads(m.frame)
    assert view["seat"] == seat and view["ranked"] is True and "model_hand" not in view and "model_elixir" not in view
    assert m.model_seen and view["model_seen"] == [] and view["model_last"] == []       # like a real match
    assert "deck_labels" not in view and "model_deck" not in view
    assert "giant" not in json.dumps({k: v for k, v in view.items() if k != "events"})
    mine = [e for e in view["ents"] if e[1] == 1 and e[4] == "troop"]
    assert mine and all(e[6] > 16000 for e in mine)          # the player's troops are drawn on the bottom half
    run_to(m, 400)                                           # (a random-init policy may wait on its first beats)
    model = [e for e in m.events if e["who"] == "model"]
    assert model and all(e["result"] == "OK" for e in model)
    tw = {(t["side"], t["idx"]): (t["x"], t["y"]) for t in view["towers"]}
    assert tw[("me", 0)] == (9000, 29000) and tw[("foe", 0)] == (9000, 3000)         # Kings: bottom = the player
    assert {tw[("me", 1)], tw[("me", 2)]} == {(3500, 25500), (14500, 25500)}
    assert {tw[("foe", 1)], tw[("foe", 2)]} == {(3500, 6500), (14500, 6500)}


def test_replay_reproduces_a_match_exactly(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player)
    m = live(srv, seat=1, deck="golem", opp="random")
    rng = np.random.default_rng(0)
    while not m.done:                           # a "human" that plays a random legal card now and then
        if m.game.tick_count % 30 == 5 and rng.random() < 0.5:
            st = m.game.state()
            mask = m.game.legal_mask(m.ht)
            legal = np.flatnonzero(mask[1:]) if st["tick"] >= st["lockout_ticks"] else []
            if len(legal):
                slot, cell = divmod(int(rng.choice(legal)), pr.N_TILES)
                m.queue_play(int(st["hand"][m.ht][slot]), cell % 18, cell // 18)
        m.step()
    end = json.loads(m.frame)
    assert end["model_deck"] == m.deck_ids["model"] and end["deck_labels"]["model"].startswith("random: ")
    assert end["model_seen"]                                   # revealed once the match is over
    rec = m.record()
    json.dumps(rec)
    assert rec["summary"]["result"] in (-1, 0, 1) and rec["summary"]["ranked"]
    assert rec["replay"]["plays"] and len(rec["replay"]["hashes"]) >= 5
    assert any(p[1] == m.ht for p in rec["replay"]["plays"]) and any(p[1] == m.mt for p in rec["replay"]["plays"])
    r = ph.Replay(srv, rec)
    while not r.done:
        r.step()
    assert r.verified and r.mismatch is None
    assert json.loads(r.frame)["crowns"] == rec["summary"]["crowns"]
    r.seek(1234)
    assert r.game.tick_count == 1234 and not r.done
    r.seek(10 ** 9)
    assert r.done and r.verified


def test_advisor(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player)
    m = live(srv)
    run_to(m, 120)
    m.control({"advisor": True}, owner=False)
    a = json.loads(m.frame)["advice"]
    hand = list(m.game.state()["hand"][m.ht])
    mask = m.game.legal_mask(m.ht)
    assert 0 <= a["wait"] <= 1 and a["cards"]
    for c in a["cards"]:
        assert c["card"] in hand and 0 < c["p"] <= 1 and 0 < c["p_tile"] <= 1
        assert mask[1 + hand.index(c["card"]) * pr.N_TILES + c["ty"] * 18 + c["tx"]]
    run_to(m, 140)
    assert json.loads(m.frame)["advice"] is not None
    m.control({"advisor": False}, owner=False)
    assert m.assist_max == 1 and not m.ranked()
    with pytest.raises(ValueError):
        m.control({"speed": 3.0}, owner=False)
    m.control({"speed": 0.5}, owner=False)
    assert m.min_speed == 0.5


def test_autopilot(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player)
    m = live(srv, seat=1)
    with pytest.raises(ValueError):
        m.control({"assist": 3}, owner=False)
    m.control({"assist": 2}, owner=False)
    assert json.loads(m.frame)["assist"] == 2 and not m.ranked()
    run_to(m, 400)
    auto = [e for e in m.events if e["who"] == "human" and e.get("auto")]
    assert auto and all(e["result"] == "OK" for e in auto)                     # the policy plays the player's side
    # the player's own play always comes first: on that beat the autopilot does not play
    # (no elixir top-up here: the replay below must reproduce the match from its recorded plays alone)
    card = None
    while card is None and not m.done:
        m.step()
        if m.game.tick_count % ph.FRAME_SKIP == 0:
            st = m.game.state()
            ok = [int(c) for c in st["hand"][m.ht] if pr.card_info(int(c)).get("kind") == "troop"
                  and st["elixir"][m.ht] >= pr.CARD_COSTS[int(c)] * ph.ELIXIR_UNIT]
            if ok and m.queue_play(ok[0], 9, 25)[0]:
                card = ok[0]
    assert card is not None
    beat = m.game.tick_count
    m.step()
    mine = [e for e in m.events if e["who"] == "human" and e["tick"] == beat]
    assert len(mine) == 1 and not mine[0].get("auto") and mine[0]["card"] == card and mine[0]["result"] == "OK"
    m.control({"assist": 0}, owner=False)
    n = len([e for e in m.events if e.get("auto")])
    run_to(m, beat + 200)
    assert len([e for e in m.events if e.get("auto")]) == n                    # off again: no more autopilot plays
    while not m.done:
        m.step()
    rec = m.record()
    assert rec["summary"]["autopilot_used"] and rec["summary"]["assist_max"] == 2 and not rec["summary"]["ranked"]
    r = ph.Replay(srv, rec)                                                    # its plays replay exactly
    while not r.done:
        r.step()
    assert r.verified


def test_claude_toggles_and_relay(tmp_path, policy_player):
    relay = tmp_path / "relay"
    srv = make_server(tmp_path, policy_player, claude_relay_dir=str(relay), claude_daily_calls=2, comment_every=100)
    assert (relay / "RULES_PROMPT.txt").read_text().startswith("You are the live commentator")
    m = live(srv, owner=True)
    with pytest.raises(ValueError):
        m.control({"commentary": True}, owner=False)          # owner only
    m.control({"commentary": True, "winprob": True}, owner=True)
    run_to(m, 100)
    d = relay / ("g" + m.id)
    assert sorted(os.listdir(d)) == ["turn_0001.txt"]
    text = (d / "turn_0001.txt").read_text()
    assert "YOUR HAND" in text and "Reply with" not in text
    assert "OPPONENT ELIXIR" not in text and "OPPONENT CARDS SEEN" not in text and "OPPONENT LAST PLAYED" not in text
    (d / "reply_0001.json").write_text(json.dumps({"text": 'x\n{"commentary": "Even start.", "p_win": 0.5}', "meta": {}}))
    run_to(m, 102)
    v = json.loads(m.frame)
    assert v["comments"][-1]["text"] == "Even start." and v["winprob"][-1][1] == 0.5
    run_to(m, 260)                                             # the 2nd call; then the daily cap (2) is reached
    assert (d / "turn_0002.txt").exists()
    (d / "reply_0002.json").write_text(json.dumps({"text": "WAIT", "meta": {"fatal": True}}))
    run_to(m, 420)
    v = json.loads(m.frame)
    assert v["claude"] == {"commentary": False, "winprob": False} and "used up" in v["claude_note"]
    assert len(m.comments) == 1 and not (d / "turn_0003.txt").exists()


def test_neural_voice_wiring(tmp_path, policy_player):
    """The server side of the voice: a fake worker (heartbeat + a written clip) stands in for play/tts_worker.py."""
    relay, voice = tmp_path / "relay", tmp_path / "voice"
    srv = make_server(tmp_path, policy_player, claude_relay_dir=str(relay), comment_every=100, voice_dir=str(voice))
    assert not srv.voice_on() and srv.meta(True)["neural_voice"] is False          # no worker: browser voice
    (voice / "HEARTBEAT").write_text("x")
    assert srv.voice_on()
    m = live(srv, owner=True)
    with pytest.raises(ValueError):
        m.control({"voice": "not_a_voice"}, owner=True)
    m.control({"commentary": True}, owner=True)
    run_to(m, 61)
    d = relay / ("g" + m.id)
    (d / "reply_0001.json").write_text(json.dumps({"text": '{"commentary": "What a disaster, folks!", "p_win": 0.4}'}))
    run_to(m, 63)
    c = json.loads(m.frame)["comments"][-1]
    cid = f"{m.id}_0001"
    assert c["audio"] == cid and c["ready"] is False
    assert json.loads((voice / "req" / (cid + ".txt")).read_text()) == {"text": "What a disaster, folks!", "voice": "am_fenrir"}
    (voice / (cid + ".wav")).write_bytes(b"RIFFfake")                               # the worker wrote the clip
    run_to(m, 64)
    assert json.loads(m.frame)["comments"][-1]["ready"] is True
    assert srv.voice_clip(cid) == b"RIFFfake" and srv.voice_clip("../etc/passwd") is None
    m.control({"voice": "bm_george"}, owner=True)                                   # the owner picks another voice
    assert json.loads(m.frame)["voice"] == "bm_george" and srv.meta(True)["voices"][0] == "am_fenrir"
    srv.request_voice("x" * 12, "Next line!", m.voice)
    assert json.loads((voice / "req" / ("x" * 12 + ".txt")).read_text())["voice"] == "bm_george"
    os.utime(voice / "HEARTBEAT", (time.time() - 60, time.time() - 60))           # the worker died: no new requests
    assert not srv.voice_on()


# ---------------------------------------------------------------------------------------------- records
def test_name_and_pin_are_the_login(tmp_path):
    st = ph.Store(str(tmp_path))
    assert st.hello(None) == (None, {}, False, None) and st.players == {}      # nobody is made without a name
    with pytest.raises(ValueError, match="choose a PIN"):
        st.hello(None, "Aditya")                                               # a new name needs a PIN
    with pytest.raises(ValueError, match="choose a PIN"):
        st.hello(None, "Aditya", "12a4")
    pid, prof, created, new_pin = st.hello(None, "Aditya", "1234")
    assert created and new_pin is None and "pin" not in prof                   # the page never gets the PIN record
    assert "1234" not in json.dumps(st.players) and len(st.players[pid]["pin"]["hash"]) == 64   # only a salted hash
    st.update(pid, deck=list(range(8)), speed=0.5)
    for other in ("aditya", "  ADITYA ", "Aditya"):                            # another device / browser / address
        again, prof, created2, np2 = st.hello(None, other, "1234")
        assert again == pid and not created2 and np2 is None and prof["deck"] == list(range(8)) and prof["speed"] == 0.5
    assert st.hello("some-other-browser-id", "aditya", "1234")[0] == pid
    assert st.hello(pid)[0] == pid                                             # the browser's own stored id: no PIN
    with pytest.raises(ValueError, match="enter it"):
        st.hello(None, "Aditya")
    bob, _, c, _ = st.hello(pid, "Bob", "0000")                                # a different name is a different player
    assert c and bob != pid and len(st.players) == 2
    assert ph.Store(str(tmp_path)).hello(None, "bob", "0000")[0] == bob        # also after a restart
    st.set_pin(pid, "9876")                                                    # "Change my PIN"
    with pytest.raises(ValueError, match="wrong PIN"):
        st.hello(None, "Aditya", "1234")
    assert st.hello(None, "Aditya", "9876")[0] == pid
    with pytest.raises(ValueError):
        st.set_pin(pid, "12345")


def test_wrong_pins_lock_the_name(tmp_path, monkeypatch):
    st = ph.Store(str(tmp_path))
    pid = st.hello(None, "Ann", "1111")[0]
    for i in range(ph.PIN_TRIES - 1):
        with pytest.raises(ValueError, match="left"):
            st.hello(None, "Ann", "2222")
    with pytest.raises(ValueError, match="locked"):
        st.hello(None, "Ann", "2222")
    with pytest.raises(ValueError, match="try again"):
        st.hello(None, "Ann", "1111")                                          # even the right PIN waits out the lock
    assert st.hello(pid)[0] == pid                                             # the player's own browser still works
    later = time.time() + ph.PIN_LOCK_S + 1
    monkeypatch.setattr(ph.time, "time", lambda: later)
    assert st.hello(None, "Ann", "1111")[0] == pid


def test_players_from_before_pins_get_one_once(tmp_path):
    (tmp_path / "players.json").write_text(json.dumps({"aaaaaaaaaaaa": {"name": "Cleo"}, "bbbbbbbbbbbb": {"name": "Dev"}}))
    st = ph.Store(str(tmp_path))
    pid, prof, created, pin = st.hello(None, "cleo", "")                       # by name: the site gives a PIN
    assert pid == "aaaaaaaaaaaa" and not created and ph.PIN_RE.match(pin) and "pin" not in prof
    assert st.hello(None, "Cleo", pin)[3] is None                              # shown once; from now on it is needed
    with pytest.raises(ValueError):
        st.hello(None, "Cleo")
    pid2, _, _, pin2 = st.hello("bbbbbbbbbbbb")                                # by the browser's stored id
    assert pid2 == "bbbbbbbbbbbb" and ph.PIN_RE.match(pin2)
    assert ph.Store(str(tmp_path)).hello(None, "Dev", pin2)[0] == pid2       # persisted (as a hash)


def test_admin_file_drops_a_player_and_resets_a_pin(tmp_path):
    st = ph.Store(str(tmp_path))
    a = st.hello(None, "Old Name", "1111")[0]
    b = st.hello(None, "Bob", "2222")[0]
    for i, pid in enumerate((a, b)):
        st.save_match({"summary": {"id": f"20261106_12000{i}_abcdef", "player_id": pid, "name": "x", "result": 1,
                                   "crowns": [1, 0], "ranked": True, "decks": {}, "abandoned": False}})
    (tmp_path / "admin.json").write_text(json.dumps({"drop": ["old name"], "reset_pin": ["BOB"]}))
    st2 = ph.Store(str(tmp_path))                                              # the restarted server
    assert set(st2.players) == {b} and [r["name"] for r in st2.leaderboard()["rows"]] == ["Bob"]
    assert os.listdir(tmp_path / "removed") == ["20261106_120000_abcdef.json"]  # kept aside, not deleted
    assert sorted(os.listdir(tmp_path / "matches")) == ["20261106_120001_abcdef.json"]
    assert not (tmp_path / "admin.json").exists() and len(glob.glob(str(tmp_path / "admin.done.*.json"))) == 1
    log = [json.loads(l) for l in open(tmp_path / "merge_log.jsonl")]
    assert any(e.get("admin_drop") == a for e in log) and any(e.get("admin_reset_pin") == b for e in log)
    pid, _, _, pin = st2.hello(None, "Bob")                                    # reset: a new PIN, shown once
    assert pid == b and ph.PIN_RE.match(pin)
    assert ph.Store(str(tmp_path)).hello(None, "old name", "1111")[2]      # the name is free again (a new player)


def test_two_servers_keep_each_others_pins(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "players.json").write_text(json.dumps({"aaaaaaaaaaaa": {"name": "Cleo"}}))
    a, b = ph.Store(str(root)), ph.Store(str(root))           # the old and the new site job, both running
    pin = a.hello(None, "Cleo")[3]                           # a gives Cleo a PIN
    b.update("aaaaaaaaaaaa", speed=0.5)                      # b saves its own copy, made before that PIN
    assert json.load(open(root / "players.json"))["aaaaaaaaaaaa"]["pin"]       # not wiped
    assert b.hello(None, "Cleo", pin)[3] is None             # b uses a's PIN: no second PIN, no free login
    with pytest.raises(ValueError, match="wrong PIN"):
        b.hello(None, "Cleo", f"{(int(pin) + 1) % 10000:04d}")


def test_old_duplicate_names_are_merged(tmp_path):
    os.makedirs(tmp_path / "matches")
    players = {"aaaaaaaaaaaa": {"name": "Aditya", "last_seen": "2026-11-06T15:00:00"},
               "bbbbbbbbbbbb": {"name": "aditya", "last_seen": "2026-11-06T16:00:00", "deck": "golem"},
               "cccccccccccc": {"name": ""}, "dddddddddddd": {"name": "Zoe"}}
    (tmp_path / "players.json").write_text(json.dumps(players))
    for i, pid in enumerate(["aaaaaaaaaaaa", "aaaaaaaaaaaa", "bbbbbbbbbbbb"]):
        (tmp_path / "matches" / f"20261106_15000{i}_abcdef.json").write_text(json.dumps({"summary": {
            "id": f"20261106_15000{i}_abcdef", "player_id": pid, "name": "Aditya", "result": 1, "crowns": [1, 0],
            "ranked": True, "decks": {}, "abandoned": False}}))
    st = ph.Store(str(tmp_path))
    assert set(st.players) == {"aaaaaaaaaaaa", "dddddddddddd"}                  # kept: the one with more matches
    assert st.leaderboard()["rows"][0]["n"] == 3 and st.hello(None, "ADITYA", "1234")[0] == "aaaaaaaaaaaa"
    assert all(json.load(open(f))["summary"]["player_id"] == "aaaaaaaaaaaa" for f in (tmp_path / "matches").iterdir())
    log = [json.loads(l) for l in open(tmp_path / "merge_log.jsonl")]
    assert {"merged": "bbbbbbbbbbbb", "into": "aaaaaaaaaaaa"}.items() <= log[0].items()
    assert any(l.get("dropped_unnamed") == "cccccccccccc" for l in log)
    assert len(ph.Store(str(tmp_path)).players) == 2                           # idempotent


def test_store_leaderboard_hides_player_ids(tmp_path):
    st = ph.Store(str(tmp_path))
    pid, prof, created, _ = st.hello(None, "Ann", "1234")
    assert ph.ID_RE.match(pid) and prof["name"] == "Ann" and created
    assert st.hello(pid)[0] == pid and st.hello("not-a-known-id-123") == (None, {}, False, None)
    st.update(pid, deck=list(range(8)))
    assert ph.Store(str(tmp_path)).players[pid]["deck"] == list(range(8))        # persisted
    for i, (res, ranked) in enumerate([(1, True), (-1, True), (0, True), (1, False)]):
        st.save_match({"summary": {"id": f"20261106_1200{i:02d}_abcdef", "player_id": pid, "name": "Ann", "result": res,
                                   "crowns": [1, 0], "ranked": ranked, "decks": {}, "abandoned": False}})
    lb = st.leaderboard()
    assert lb["rows"] == [{"n": 3, "w": 1, "d": 1, "l": 1, "best_crowns": 1, "name": "Ann", "score": 0.5,
                           "pub": ph.public_id(pid)}]
    assert pid not in json.dumps(lb) and pid not in json.dumps(st.history(pid))
    assert len(st.history(pid)) == 4 and lb["totals"] == {"n": 3, "w": 1, "d": 1, "l": 1}
    assert len(ph.Store(str(tmp_path)).index) == 4                                  # rebuilt from the files
    with pytest.raises(ValueError):
        st.load_match("../../etc/passwd")
    assert st.claude_take(1) and not st.claude_take(1)


def test_spectators(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player)
    pid = srv.store.hello(None, "Ann", "1234")[0]
    s = srv.join(pid, "Ann", False, params())
    m = s.board
    t0 = time.time()
    while m.tick_now < 120 and time.time() - t0 < 20:
        time.sleep(0.05)
    live = srv.live_list()
    assert [(x["match_id"], x["name"], x["spectators"]) for x in live] == [(m.id, "Ann", 0)]
    w = srv.spectate(False, m.id)
    assert w.board is m and m.spectators == 1 and srv.live_count() == 1          # not a seat
    v = json.loads(m.frame)
    assert v["player_name"] == "Ann" and v["model_seen"] == [] and "deck_labels" not in v
    with pytest.raises(ValueError):
        srv.spectate(False, "20990101_000000_nosuchid")
    # a spectator never keeps the player's match alive
    srv.session(w.sid)
    m.last_client = time.time() - ph.IDLE_ABANDON_S - 5
    srv.sweep(time.time())
    t0 = time.time()
    while not os.listdir(os.path.join(srv.args.data_dir, "matches")) and time.time() - t0 < 10:
        time.sleep(0.05)
    assert m.done and m.abandoned and s.status == "ended"
    srv.leave(w)
    assert w.status == "ended" and m.spectators == 0
    with pytest.raises(ValueError):
        srv.spectate(False, m.id)                                                 # not live any more
    srv.shutdown()


# ---------------------------------------------------------------------------------------------- HTTP
def test_http_seats_line_and_auth(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player, max_players=1)
    token, owner = "t" * 22, "o" * 22
    httpd = ph.ThreadingHTTPServer(("127.0.0.1", 0), ph.make_handler(srv, token, owner, "<html>ok</html>"))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None, tok=token, own=None):
        url = f"{base}{path}{'&' if '?' in path else '?'}token={tok}" + (f"&owner={own}" if own else "")
        req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    try:
        for path in ("/", "/api/meta"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + path, timeout=5)
            assert e.value.code == 403
        assert call("/api/meta")["owner"] is False and call("/api/meta", own=owner)["owner"] is True
        assert call("/api/meta", own="x" * 22)["owner"] is False
        a = call("/api/hello", {"name": "Ann", "pin": "1234"})["player_id"]
        b = call("/api/hello", {"name": "Bob", "pin": 2222})["player_id"]                 # a number works too
        h = call("/api/hello", {"name": " ann ", "pin": "1234"})                        # Ann again, from anywhere
        assert h["player_id"] == a and h["created"] is False and h["profile"]["name"] == "Ann"
        assert "pin" not in h["profile"] and "new_pin" not in h
        for bad in ({"name": "Ann", "pin": "9999"}, {"name": "Ann"}, {"name": "Newcomer"}):
            with pytest.raises(urllib.error.HTTPError) as e:                            # wrong / missing PIN
                call("/api/hello", bad)
            assert e.value.code == 400
        assert call("/api/pin", {"player_id": a, "pin": "4321"})["ok"] is True
        assert call("/api/hello", {"name": "Ann", "pin": "4321"})["player_id"] == a
        assert call("/api/hello", {})["player_id"] is None                              # a visitor without a name
        with pytest.raises(urllib.error.HTTPError) as e:
            call("/api/hello", {"name": "<>"})
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:
            call("/api/join", {"player_id": a, "deck": [1] * 8})
        assert e.value.code == 400
        ja = call("/api/join", {"player_id": a, "deck": "hog26", "opp_deck": "giant"})
        jb = call("/api/join", {"player_id": b, "name": "Bob", "deck": list(range(8)), "opp_deck": "random"})
        assert ja["status"] == "playing" and jb["status"] == "queued" and jb["position"] == 1
        st = call(f"/api/state?sid={jb['sid']}")
        assert st["kind"] == "queued" and st["position"] == 1 and st["playing"] == 1
        f = call(f"/api/state?sid={ja['sid']}")
        assert f["kind"] == "match" and len(f["hand"]) == 4
        t1 = time.time()
        g = call(f"/api/state?sid={ja['sid']}&after={f['v']}")              # long poll: the next frame, not the same one
        assert g["v"] > f["v"] and time.time() - t1 < 1.0
        with pytest.raises(urllib.error.HTTPError) as e:
            call("/api/control", {"sid": ja["sid"], "commentary": True})                # not the owner
        assert e.value.code == 400
        with urllib.request.urlopen(f"{base}/api/stream?sid={ja['sid']}&token={token}", timeout=10) as r:
            line = r.readline()
            assert line.startswith(b"data: ") and json.loads(line[6:])["kind"] == "match"
        live = call("/api/leaderboard")["live"]
        assert len(live) == 1 and live[0]["name"] == "Ann" and "player_id" not in live[0]
        sp = call("/api/spectate", {"match_id": live[0]["match_id"]})["sid"]
        fs = call(f"/api/state?sid={sp}")
        fs = call(f"/api/state?sid={sp}&after={fs['v']}")                       # the next frame counts the watcher
        assert fs["kind"] == "match" and fs["player_name"] == "Ann" and fs["spectators"] == 1
        for path, body in (("/api/play", {"card": 0, "tx": 9, "ty": 25}), ("/api/control", {"paused": True}),
                           ("/api/cancel", {})):
            with pytest.raises(urllib.error.HTTPError) as e:
                call(path, dict(body, sid=sp))
            assert e.value.code == 400
        call("/api/leave", {"sid": sp})
        fa = call(f"/api/state?sid={ja['sid']}")
        assert call(f"/api/state?sid={ja['sid']}&after={fa['v']}")["spectators"] == 0
        call("/api/leave", {"sid": ja["sid"]})
        t0 = time.time()
        while srv.sessions[jb["sid"]].status != "playing" and time.time() - t0 < 10:
            time.sleep(0.1)
        assert srv.sessions[jb["sid"]].status == "playing"                              # the line moved on
        mdir = os.path.join(srv.args.data_dir, "matches")
        while not os.listdir(mdir) and time.time() - t0 < 10:
            time.sleep(0.1)
        recs = [json.load(open(os.path.join(mdir, f))) for f in os.listdir(mdir)]
        assert len(recs) == 1 and recs[0]["summary"]["abandoned"] and not recs[0]["summary"]["ranked"]
        lb = call("/api/leaderboard")
        assert lb["rows"] == [] and lb["recent"][0]["name"] == "Ann" and a not in json.dumps(lb)
        w = call("/api/watch", {"player_id": a, "match_id": recs[0]["summary"]["id"]})
        f = call(f"/api/state?sid={w['sid']}")
        assert f["kind"] == "replay" and f["replay"]["summary"]["name"] == "Ann"
        call("/api/leave", {"sid": jb["sid"]})
    finally:
        srv.shutdown()
        httpd.shutdown()


# ---------------------------------------------------------------------------------------------- real time (2026-11-09)
def test_a_play_goes_in_at_the_next_tick_not_the_next_beat(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player)
    m = live(srv, seat=1)
    run_to(m, 103)                                            # between beats (the policy acts at 100, 110, ...)
    m.game.set_elixir(m.ht, 28000)
    slot, card = troop_slot(m.game, m.ht)
    ok, why = m.queue_play(card, 9, 25, pid="p1")
    assert ok and why == "queued for the next tick"
    m.step()
    ev = [e for e in m.events if e["who"] == "human"]
    assert len(ev) == 1 and ev[0]["tick"] == 103 and ev[0]["result"] == "OK" and ev[0]["pid"] == "p1"
    assert [p for p in m.plays if p[1] == m.ht] == [[103, m.ht, slot, 9, 25, card]]


def test_the_queue_counts_elixir_already_on_its_way(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player)
    m = live(srv, seat=0)
    run_to(m, 100)
    deck = [int(c) for c in m.game.state()["deck"][m.ht]]
    troops = [c for c in deck if pr.card_info(c).get("kind") == "troop"]
    assert len(troops) >= 2                                     # Hog 2.6: Hog Rider, Musketeer, Ice Golem, ...
    m.game.set_hand(m.ht, troops[:2] + [c for c in deck if c not in troops[:2]])   # both troops in hand
    a, b = troops[:2]
    m.game.set_elixir(m.ht, pr.CARD_COSTS[a] * ph.ELIXIR_UNIT)      # exactly enough for the first one
    assert m.queue_play(a, 9, 25)[0]
    assert m.queue_play(a, 8, 25) == (False, "that card is already on its way")
    assert m.queue_play(b, 8, 25) == (False, "not enough elixir")
    m.step()
    assert [e["card"] for e in m.events if e["who"] == "human" and e["result"] == "OK"] == [a]


def _ws_open(port, path):
    """A minimal WebSocket client for the tests (raw socket; client frames masked as RFC 6455 requires)."""
    import base64
    import socket
    key = base64.b64encode(os.urandom(16)).decode()
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall((f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    f = s.makefile("rb")
    status = f.readline()
    headers = {}
    while True:
        line = f.readline()
        if line in (b"\r\n", b""):
            break
        k, _, v = line.decode().partition(":")
        headers[k.strip().lower()] = v.strip()
    return s, f, status, headers, key


def _ws_send(s, obj, op=1, masked=True):
    import struct
    data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    mask = os.urandom(4)
    n = len(data)
    head = struct.pack("!BB", 0x80 | op, (0x80 if masked else 0) | n) if n < 126 else \
        struct.pack("!BBH", 0x80 | op, (0x80 if masked else 0) | 126, n)
    body = bytes(c ^ mask[i % 4] for i, c in enumerate(data)) if masked else data
    s.sendall(head + (mask if masked else b"") + body)


def _ws_recv(f):
    import struct
    b0, b1 = f.read(2)
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack("!H", f.read(2))[0]
    elif n == 127:
        n = struct.unpack("!Q", f.read(8))[0]
    return b0 & 0x0F, f.read(n)


def test_websocket_frames_frame_helpers():
    import io
    assert ph.ws_accept("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="     # RFC 6455 section 1.3 example
    for n in (0, 5, 125, 126, 1000, 70000):
        fr = ph.ws_frame(b"x" * n)
        assert fr[0] == 0x81 and fr.endswith(b"x" * n)
    raw = bytearray()
    for i, (op, fin) in enumerate(((1, False), (0, True))):         # a fragmented, masked text message
        part, mask = (b"hello " if i == 0 else b"world"), b"\x01\x02\x03\x04"
        raw += bytes([(0x80 if fin else 0) | op, 0x80 | len(part)]) + mask + bytes(c ^ mask[j % 4] for j, c in enumerate(part))
    assert ph.ws_read(io.BytesIO(bytes(raw))) == (1, b"hello world")
    with pytest.raises(ValueError):
        ph.ws_read(io.BytesIO(bytes([0x81, 0x03]) + b"abc"))         # unmasked client frame
    with pytest.raises(ValueError):
        ph.ws_read(io.BytesIO(bytes([0x81, 0xFF]) + (10 ** 9).to_bytes(8, "big") + b"\0" * 4), limit=100)
    with pytest.raises(EOFError):
        ph.ws_read(io.BytesIO(b"\x81"))


def test_websocket_live_channel(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player, max_players=2)
    token = "t" * 22
    httpd = ph.ThreadingHTTPServer(("127.0.0.1", 0), ph.make_handler(srv, token, None, "<html>ok</html>"))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None):
        url = f"{base}{path}{'&' if '?' in path else '?'}token={token}"
        req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Host": "abc.trycloudflare.com"},
                                     method="GET" if body is None else "POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r, r.read()

    try:
        r, _ = call("/")
        assert "wss://abc.trycloudflare.com" in r.headers["Content-Security-Policy"]
        pid = json.loads(call("/api/hello", {"name": "Ann", "pin": "1234"})[1])["player_id"]
        sid = json.loads(call("/api/join", {"player_id": pid, "deck": "hog26", "opp_deck": "giant"})[1])["sid"]
        s, f, status, headers, key = _ws_open(port, f"/api/ws?sid={sid}&token={'x' * 22}")      # wrong token
        assert b"403" in status
        s.close()
        s, f, status, headers, key = _ws_open(port, f"/api/ws?sid={sid}&token={token}")
        assert b"101" in status and headers["sec-websocket-accept"] == ph.ws_accept(key)
        ticks, t0 = [], time.time()
        while len(ticks) < 15:                                   # pushed frames: one per engine tick
            op, data = _ws_recv(f)
            assert op == 1
            fr = json.loads(data)
            assert fr["kind"] == "match"
            ticks.append(fr["tick"])
        assert ticks == sorted(ticks) and ticks[-1] - ticks[0] >= 10 and time.time() - t0 < 5
        m = srv.sessions[sid].board
        while m.game.tick_count < 100:
            time.sleep(0.05)
        with m.lock:
            m.game.set_elixir(m.ht, 28000)
            slot, card = troop_slot(m.game, m.ht)
        _ws_send(s, {"t": "ping", "c": 123.5, "rtt": 42.0})
        _ws_send(s, {"t": "play", "id": "a1", "card": card, "tx": 9, "ty": 25})
        _ws_send(s, b"ping!", op=9)
        got = {}
        while not {"pong", "ack", "pongframe"} <= set(got):
            op, data = _ws_recv(f)
            if op == 10:
                got["pongframe"] = data
            elif op == 1:
                msg = json.loads(data)
                if msg.get("t") in ("pong", "ack"):
                    got[msg["t"]] = msg
        assert got["pong"] == {"t": "pong", "c": 123.5} and got["pongframe"] == b"ping!"
        assert got["ack"] == {"t": "ack", "id": "a1", "ok": True, "reason": "queued for the next tick"}
        deadline = time.time() + 5
        while not [e for e in m.events if e["who"] == "human"] and time.time() < deadline:
            time.sleep(0.02)
        ev = [e for e in m.events if e["who"] == "human"]
        assert ev and ev[0]["pid"] == "a1" and ev[0]["result"] == "OK" and ev[0]["waited_ms"] < 200
        _ws_send(s, {"t": "nonsense", "id": "z"})
        while True:
            op, data = _ws_recv(f)
            if op == 1 and json.loads(data).get("t") == "error":
                assert json.loads(data) == {"t": "error", "id": "z", "error": "unknown message"}
                break
        assert "ws" in m.transports and m.client_rtt == [42.0]
        sp = json.loads(call("/api/spectate", {"match_id": m.id})[1])["sid"]       # a watcher may not play
        s2, f2, status, _, _ = _ws_open(port, f"/api/ws?sid={sp}&token={token}")
        assert b"101" in status
        _ws_send(s2, {"t": "play", "id": "b1", "card": card, "tx": 9, "ty": 24})
        while True:
            op, data = _ws_recv(f2)
            if op == 1 and json.loads(data).get("t") == "error":
                assert "spectators cannot play" in json.loads(data)["error"]
                break
        _ws_send(s2, {"t": "ping"}, masked=False)                # a protocol error closes the connection
        deadline = time.time() + 5
        while srv.streams > 1 and time.time() < deadline:
            time.sleep(0.05)
        assert srv.streams == 1
        s2.close()
        _ws_send(s, struct_close := (1000).to_bytes(2, "big"), op=8)
        deadline = time.time() + 5
        while srv.streams > 0 and time.time() < deadline:
            time.sleep(0.05)
        assert srv.streams == 0
        s.close()
        rec = m.record()
        assert rec["client"]["transports"] == ["ws"] and rec["client"]["rtt_ms"]["median"] == 42.0
    finally:
        srv.shutdown()
        httpd.shutdown()


# ---------------------------------------------------------------------------------------------- permanent link (2026-11-09)
def test_two_servers_on_one_data_dir_keep_each_others_players(tmp_path):
    root = str(tmp_path / "data")
    a, b = ph.Store(root), ph.Store(root)                    # the old and the new site job, both running
    pa = a.hello(None, "Ann", "1111")[0]
    pb = b.hello(None, "Bob", "2222")[0]                     # b loaded the file before Ann existed
    disk = json.load(open(os.path.join(root, "players.json")))
    assert set(disk) == {pa, pb}
    assert b.hello(None, "ann", "1111")[0] == pa             # b took Ann in: the same login, not a new player
    c = ph.Store(root)
    assert {v["name"] for v in c.players.values()} == {"Ann", "Bob"}


def test_alive_needs_no_token_and_allows_other_sites(tmp_path, policy_player):
    srv = make_server(tmp_path, policy_player)
    httpd = ph.ThreadingHTTPServer(("127.0.0.1", 0), ph.make_handler(srv, "t" * 22, None, "<html>ok</html>"))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/alive", timeout=10) as r:
            assert json.loads(r.read()) == {"ok": True} and r.headers["Access-Control-Allow-Origin"] == "*"
        with pytest.raises(urllib.error.HTTPError) as e:                 # everything else still needs the token
            urllib.request.urlopen(base + "/api/meta", timeout=10)
        assert e.value.code == 403
    finally:
        srv.shutdown()
        httpd.shutdown()


def test_play_link_against_a_fake_github(tmp_path, monkeypatch):
    import base64
    import importlib.util
    spec = importlib.util.spec_from_file_location("play_link", os.path.join(ROOT, "play", "play_link.py"))
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    key = tmp_path / "github_pages_token"
    fake = "github_pat_" + "A" * 22 + "_" + "B" * 59                      # the shape of a fine-grained key
    key.write_text("\x1b[?2004h" + fake + "\x1b]11;rgb:0000/0000/0000\x1b\\\n")   # with terminal codes around it
    monkeypatch.setattr(pl, "TOKEN_FILE", str(key))
    site = tmp_path / "play_site_token"
    site.write_text("s" * 24 + "\n")
    monkeypatch.setattr(pl, "SITE_TOKEN_FILE", str(site))
    files, seen_tokens = {}, set()

    def fake_api(method, repo, path, token, body=None):
        seen_tokens.add(token)
        assert repo == "Aditya-Gupta26/pr-play"
        if method == "GET":
            if path not in files:
                return 404, {"message": "Not Found"}
            return 200, {"content": base64.b64encode(files[path][0]).decode(), "sha": files[path][1]}
        if path in files and body.get("sha") != files[path][1]:
            return 409, {"message": "sha mismatch"}
        files[path] = (base64.b64decode(body["content"]), f"sha{len(files)}{time.time()}")
        return 200, {}
    monkeypatch.setattr(pl, "api", fake_api)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["play_link.py", *argv])
        pl.main()

    run("set", "https://abc-def.trycloudflare.com", "--job", "111")
    t = json.loads(files["target.json"][0])
    assert t["status"] == "up" and t["url"] == "https://abc-def.trycloudflare.com" and t["job"] == "111"
    assert t["token"] == "s" * 24                                           # open link: the page adds the token
    run("set", "https://abc-def.trycloudflare.com", "--job", "111", "--private")
    assert "token" not in json.loads(files["target.json"][0])
    run("set", "https://new-one.trycloudflare.com/", "--job", "222")      # the successor took over
    run("down", "--job", "111")                                            # the old job ending: left alone
    assert json.loads(files["target.json"][0])["url"] == "https://new-one.trycloudflare.com"
    assert json.loads(files["target.json"][0])["status"] == "up"
    run("down", "--job", "222")
    assert json.loads(files["target.json"][0])["status"] == "down"
    with pytest.raises(SystemExit):
        run("set", "https://evil.example.com", "--job", "333")             # only quick-tunnel addresses
    run("publish-page")
    page = files["index.html"][0].decode()
    assert 'const REPO = "Aditya-Gupta26/pr-play";' in page and "__REPO__" not in page
    assert fake not in json.dumps({k: v[0].decode() for k, v in files.items()})
    assert seen_tokens == {fake}                                            # the codes around it were dropped
    key.write_text(fake + "\n" + fake.replace("A", "C"))
    with pytest.raises(SystemExit):
        pl.read_token()                                                     # two keys: refuse, do not guess
    assert pl.pages_url("Aditya-Gupta26/pr-play") == "https://aditya-gupta26.github.io/pr-play/"
