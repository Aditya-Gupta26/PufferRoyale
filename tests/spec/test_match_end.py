"""SPEC §4 end conditions (evaluated at the END of each tick), crowns, results, end_reason."""
import numpy as np
import pytest

import gamekit as K
import helpers as H


def zap_tower(g, caster, team, idx, hp=1):
    g.set_tower_hp(team, idx, hp)
    x, y = H.TOWER_POS[team][idx]
    assert K.cast(g, caster, H.ZAP, x, y) == H.OK


def check_result(g, r0):
    assert K.over(g) is True
    assert K.result(g, 0) == r0 and K.result(g, 1) == -r0, \
        f"result team0={K.result(g, 0)} team1={K.result(g, 1)}, want {r0}/{-r0}"


@pytest.mark.parametrize("winner", [0, 1])
def test_king_kill_instant_three_crowns(pr, winner):
    g = K.new_game("giant", "giant")
    g.tick(200)
    loser = 1 - winner
    zap_tower(g, winner, loser, 0)
    assert not K.over(g)
    g.tick(1)
    assert K.tower_alive(g, loser, 0) is False
    check_result(g, 1 if winner == 0 else -1)
    c = K.crowns(g)
    assert c[winner] == 3 and c[loser] == 0
    assert K.end_reason(g) == "KING"
    assert K.tick_of(g) == 201, "the match ends at the end of the tick the King died"


def test_king_kill_beats_crown_lead(pr):
    g = K.new_game("giant", "giant")
    g.tick(100)
    zap_tower(g, 1, 0, 1)           # team 1 takes a princess: 0-1
    g.tick(1)
    assert K.crowns(g) == (0, 1) and not K.over(g)
    zap_tower(g, 1, 0, 2)           # 0-2
    g.tick(1)
    assert K.crowns(g) == (0, 2) and not K.over(g)
    zap_tower(g, 0, 1, 0)           # team 0 kills the King
    g.tick(1)
    check_result(g, +1)
    assert K.crowns(g) == (3, 2)
    assert K.end_reason(g) == "KING"


def test_both_kings_same_tick_is_draw(pr):
    g = K.new_game("giant", "giant")
    g.tick(150)
    zap_tower(g, 0, 1, 0)
    zap_tower(g, 1, 0, 0)
    g.tick(1)
    assert not K.tower_alive(g, 0, 0) and not K.tower_alive(g, 1, 0)
    check_result(g, 0)
    assert K.end_reason(g) == "DRAW", "both Kings on one tick is a draw (SPEC §13.8)"


def test_running_state_result_and_reason(pr):
    g = K.new_game()
    g.tick(10)
    s = K.st(g)
    assert list(s["result"]) == [0, 0], "result is [0, 0] while running (SPEC §13.8)"
    assert s["end_reason"] is None


def test_princess_crown_goes_to_destroyer(pr):
    g = K.new_game("giant", "giant")
    g.tick(100)
    zap_tower(g, 0, 1, 2)
    g.tick(1)
    assert K.tower_alive(g, 1, 2) is False
    assert K.crowns(g) == (1, 0)
    assert not K.over(g), "a princess in regulation does not end the match"


@pytest.mark.parametrize("leader", [0, 1])
def test_regulation_crown_lead_ends_at_3600(pr, leader):
    g = K.new_game("giant", "giant")
    g.tick(500)
    zap_tower(g, leader, 1 - leader, 1)
    g.tick(1)
    g.tick(3599 - K.tick_of(g))
    assert K.tick_of(g) == 3599 and not K.over(g)
    g.tick(1)                        # processes tick 3599: regulation over at its end
    check_result(g, 1 if leader == 0 else -1)
    assert K.end_reason(g) == "REGULATION_CROWNS"
    assert K.tick_of(g) == 3600


@pytest.mark.parametrize("leader", [0, 1])
def test_overtime_sudden_death(pr, leader):
    g = K.new_game("giant", "giant")
    g.tick(3600)
    assert not K.over(g), "0-0 at the end of regulation goes to overtime"
    g.tick(300)
    assert not K.over(g)
    zap_tower(g, leader, 1 - leader, 2)
    g.tick(1)
    check_result(g, 1 if leader == 0 else -1)
    assert K.end_reason(g) == "OVERTIME_CROWNS"
    assert K.tick_of(g) == 3901


def test_tiebreak_absolute_weakest_tower(pr):
    g = K.new_game("giant", "giant")
    g.tick(100)
    g.set_tower_hp(0, 1, 1600)        # team 0 weakest: 1600 absolute (52% of a princess)
    g.set_tower_hp(1, 0, 2000)        # team 1 weakest: 2000 absolute (41% of the king)
    g.tick(5999 - K.tick_of(g))
    assert not K.over(g)
    g.tick(1)
    check_result(g, -1)
    assert K.end_reason(g) == "TIEBREAK"
    assert K.tick_of(g) == 6000


def test_tiebreak_fraction_config(pr):
    g = K.new_game("giant", "giant", tiebreak="fraction")
    g.tick(100)
    g.set_tower_hp(0, 1, 1600)
    g.set_tower_hp(1, 0, 2000)
    g.tick(6000 - K.tick_of(g))
    check_result(g, +1)
    assert K.end_reason(g) == "TIEBREAK"


def test_tiebreak_ignores_destroyed_towers(pr):
    g = K.new_game("giant", "giant")
    g.tick(100)
    zap_tower(g, 0, 1, 1)            # 1-0
    g.tick(1)
    zap_tower(g, 1, 0, 2)            # 1-1
    g.tick(1)
    assert K.crowns(g) == (1, 1)
    g.set_tower_hp(0, 0, 3000)       # team 0 alive: King 3000, Left 3052 -> weakest 3000
    g.set_tower_hp(1, 2, 3052)       # (team 1's right princess is alive and full)
    g.set_tower_hp(1, 0, 2999)       # team 1 alive: King 2999, Right 3052 -> weakest 2999
    g.tick(6000 - K.tick_of(g))
    check_result(g, +1)
    assert K.end_reason(g) == "TIEBREAK"


def test_untouched_match_is_exact_tie_draw(pr):
    g = K.new_game("hog26", "giant")
    g.tick(5999)
    assert not K.over(g)
    g.tick(1)
    check_result(g, 0)
    assert K.end_reason(g) == "DRAW", "an exact tiebreak tie is reported as DRAW (SPEC §13.8)"
    assert K.crowns(g) == (0, 0)


def test_game_over_refuses_plays(pr):
    g = K.new_game("giant", "giant")
    g.tick(100)
    zap_tower(g, 0, 1, 0)
    g.tick(1)
    assert K.over(g)
    K.put_first(g, 1, H.KNIGHT)
    g.set_elixir(1, H.MAX_ELIXIR)
    assert g.play_tile(1, 0, 9, 21) == H.GAME_OVER
    m = np.asarray(g.legal_mask(1))
    assert m[0] == 1 and m[1:].sum() == 0, "no card is legal once the game is over"


def test_finished_match_stays_finished(pr):
    g = K.new_game("giant", "giant")
    g.tick(100)
    zap_tower(g, 0, 1, 0)
    g.tick(1)
    snap = (K.result(g, 0), K.end_reason(g), K.crowns(g))
    g.tick(50)
    assert K.over(g) and (K.result(g, 0), K.end_reason(g), K.crowns(g)) == snap
