"""Tracker: a sequence of FrameReadings -> BoardStates.

What one frame cannot tell and the tracker supplies:
  * unit identity over time (tracks, matched frame to frame by position) -> the entity order, deploy flags;
  * which CARD each unit came from (the observation's entity card id): a group of units appearing together
    is matched against the cards' play compositions (measured on the engine, engine_db.play_compositions),
    after first checking whether it is the output of a periodic spawner (Witch -> Skeletons) or a death
    spawn (Golem -> Golemites) of a unit of the same side nearby;
  * the opponent's play history (cards seen, last four, elixir spent) from the groups judged to be plays;
  * the match clock (tick = 20 x seconds since the battle started) and own elixir between bar steps.

Robustness to a real detector (docs/ADVISOR.md 6a: about 25% of units missed per frame, 5% wrong type, 21% wrong
side from the side classifier alone, phantoms on the crown towers):
  * detections inside a STANDING crown tower's footprint (shrunk by `tower_margin`) are dropped -- no ground unit
    can stand there; Balloons fly over towers and are kept;
  * detections are matched to tracks by position (Hungarian), with a penalty for a different type / side, so one
    wrong-type or wrong-side frame does not start a new unit;
  * a track counts only after `min_hits` detections; its type and side are then fixed by majority vote (side:
    a unit the viewer's deck cannot produce is an enemy; otherwise the votes plus the half it appeared in);
  * the opponent has 8 cards: after 8 distinct cards are seen, plays are explained by those 8 only;
  * an enemy (own) play must appear where that side can deploy (its half, or a pocket opened by a destroyed
    tower), except Miner, Goblin Barrel and Barbarian Barrel; otherwise the group is not counted as a play.

Known blind spot (docs/ADVISOR.md §5): spells leave no units, so the opponent's spell plays are missed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import engine_db as D
from .board import BoardState, Tower, Unit
from .reading import FrameReading

# units the detector cannot tell apart -> one association key; the card decides the engine unit
KEY_ALIASES = {"Goblin": "Goblin_Stab"}
CARD_UNIT_OVERRIDES = {("Goblin_Stab", 20): "Goblin"}   # Goblin Barrel's goblins are the unit "Goblin"
DEPLOY_ANYWHERE = {20, 47, 58}                          # Goblin Barrel, Miner, Barbarian Barrel
OVER_TOWERS = {"Balloon"}                               # kept even inside a tower footprint
TYPE_PENALTY, SIDE_PENALTY = 1500.0, 500.0              # association cost (millitiles) of a type / side mismatch


def unit_key(unit: str) -> str:
    return KEY_ALIASES.get(unit, unit)


def deck_units(deck) -> set:
    """Association keys of every unit the deck can put on the board (plays, spawner and death-spawn output)."""
    out = set()
    for c in deck:
        for u in D.play_compositions()[c]:
            out.add(unit_key(u))
            ud = D.unit(u)
            for p in (ud.spawn_unit, ud.dspawn_unit):
                if p:
                    out.add(unit_key(p))
    return out


@dataclass
class Track:
    id: int
    x: float
    y: float
    born: float                 # game seconds of the first detection
    last_seen: float
    hits: int = 1
    votes: Counter = field(default_factory=Counter)   # association key -> detections
    own_votes: int = 0
    enemy_votes: int = 0
    confirmed: bool = False
    key: str = ""               # fixed at confirmation (majority vote)
    own: bool = False           # fixed at confirmation
    card: int = -1
    play: Optional[bool] = None  # None = not attributed yet; True = from a play; False = spawner / death / other
    hp_frac: Optional[float] = None
    label: str = ""

    @property
    def known(self) -> bool:
        return not self.key.startswith("label:")

    @property
    def unit(self) -> str:
        return CARD_UNIT_OVERRIDES.get((self.key, self.card), self.key)

    def vote_key(self) -> str:
        return self.votes.most_common(1)[0][0]


@dataclass
class PlayEvent:
    t: float
    own: bool
    card: int
    x: float
    y: float
    play: bool                  # False = spawner / death-spawn output, or not a plausible play
    units: dict
    joined: Counter = field(default_factory=Counter)   # units assigned to this event so far, by key


class Tracker:
    def __init__(self, deck, tower_troops=(0, 0), keep_s: float = 0.6, event_window_s: float = 0.35,
                 settle_s: float = 1.0, event_radius: int = 3500, spawn_radius: int = 3000, min_hits: int = 3,
                 tower_margin: int = 300, reid_s: float = 5.0, late_s: float = 2.0, lockout_ticks: int = 90):
        self.deck = [int(c) for c in deck]
        if len(self.deck) != 8 or len(set(self.deck)) != 8:
            raise ValueError("deck must be 8 distinct card ids")
        self.tower_troops = tuple(tower_troops)
        self.keep_s, self.window, self.settle_s, self.radius = keep_s, event_window_s, settle_s, event_radius
        self.spawn_radius, self.min_hits, self.lockout_ticks = spawn_radius, min_hits, lockout_ticks
        self.tower_margin, self.reid_s, self.late_s = tower_margin, reid_s, late_s
        self.bar_ref_s = 10.0
        self._bar_ref = [0.0] * 6     # per tower (own K, L, R, enemy K, L, R): bar reading at full HP
        self.comp = D.play_compositions()
        self.own_keys = deck_units(self.deck)
        self.tracks: list = []
        self.dead: list = []          # (track, t_removed) of confirmed tracks
        self.events: list = []        # PlayEvent log
        self.next_id = 0
        self.opp_seen: set = set()
        self.opp_spent = 0
        self.opp_last4 = [-1, -1, -1, -1]
        self.hand, self.next_card = [-1, -1, -1, -1], -1
        self.own_tower_frac = [1.0, 1.0, 1.0]
        self.enemy_tower_frac = [1.0, 1.0, 1.0]
        self.elixir_units: Optional[int] = None
        self._elixir_int: Optional[int] = None
        self._phase = 0.0             # elixir units regenerated since the bar last stepped
        self._own_played: list = []   # (t, card) from hand changes
        self.t: Optional[float] = None
        tu = D.tower_units()
        self._tower_fp = []           # own-frame footprints [(x0, y0, x1, y1)] in tower order own K, L, R, enemy K, L, R
        for side, xy, tt in ((0, D.OWN_TOWER_XY, self.tower_troops[0]), (1, D.ENEMY_TOWER_XY, self.tower_troops[1])):
            for i, (cx, cy) in enumerate(xy):
                h = D.unit(tu["king"] if i == 0 else tu[tt]).footprint_tiles * 500
                self._tower_fp.append((cx - h, cy - h, cx + h, cy + h))

    # ------------------------------------------------------------------ clock / elixir / hand / towers
    def tick(self, t: float) -> int:
        return int(min(max(round(t * 1000 / D.TICK_MS), 0), D.TICKS_MAX))

    def _update_elixir(self, t: float, e: Optional[float]) -> None:
        if e is None:
            return
        n = int(min(max(round(e), 0), 10))
        dt_ticks = 0.0 if self.t is None else (t - self.t) * 1000 / D.TICK_MS
        self._phase += D.elixir_rate(self.tick(t)) * dt_ticks
        if self._elixir_int is None or n > self._elixir_int:
            # the bar stepped up somewhere since the last frame: on average half a frame ago
            self._phase = 0.5 * D.elixir_rate(self.tick(t)) * dt_ticks
        self._elixir_int = n
        self._phase = min(max(self._phase, 0.0), D.ELIXIR_UNIT - 1)
        self.elixir_units = D.ELIXIR_MAX if n >= 10 else n * D.ELIXIR_UNIT + int(self._phase)

    def _update_hand(self, t: float, hand: list, next_card: int) -> None:
        new = [int(c) if c in self.deck else -1 for c in hand]
        if all(c >= 0 for c in self.hand) and all(c >= 0 for c in new):
            for c in self.hand:
                if c not in new:              # a card left the hand: we played it
                    self._own_played.append((t, c))
        for s, c in enumerate(new):
            if c >= 0:
                self.hand[s] = c
        if next_card in self.deck:
            self.next_card = int(next_card)

    def _update_towers(self, t: float, own: list, enemy: list) -> None:
        """Tower HP fractions. A screen bar does not read exactly 1.0 when full (its end cap, the HP digits drawn
        on it), so each bar is scaled by its largest reading in the first `bar_ref_s` seconds, when every tower is
        still at full HP. 0.0 = destroyed."""
        for side, (dst, src) in enumerate(((self.own_tower_frac, own), (self.enemy_tower_frac, enemy))):
            for i, f in enumerate(src):
                if f is None:
                    continue
                f = float(min(max(f, 0.0), 1.0))
                k = 3 * side + i
                if t < self.bar_ref_s and f > 0:
                    self._bar_ref[k] = max(self._bar_ref[k], f)
                if f > 0 and self._bar_ref[k] > 0:
                    f = min(1.0, f / self._bar_ref[k])
                dst[i] = f

    def _in_standing_tower(self, key: str, x: float, y: float) -> bool:
        if key in OVER_TOWERS:
            return False
        m = self.tower_margin
        for alive, (x0, y0, x1, y1) in zip(
                [f > 0 for f in self.own_tower_frac] + [f > 0 for f in self.enemy_tower_frac], self._tower_fp):
            if alive and x0 + m <= x <= x1 - m and y0 + m <= y <= y1 - m:
                return True
        return False

    # ------------------------------------------------------------------ tracks
    def _associate(self, r: FrameReading) -> None:
        dt = 0.5 if self.t is None else max(r.t - self.t, 1e-3)
        dets = []
        for d in r.detections:
            key = unit_key(d.unit) if d.unit else "label:" + d.label
            if self._in_standing_tower(key, d.x, d.y):
                continue
            own = d.own and key in self.own_keys      # a unit our deck cannot produce is the enemy's
            dets.append((key, own, d))
        trs = self.tracks
        matched_t, matched_d = set(), set()
        if trs and dets:
            big = 1e9
            cost = np.full((len(trs), len(dets)), big)
            for i, tr in enumerate(trs):
                tkey = tr.key if tr.confirmed else tr.vote_key()
                town = tr.own if tr.confirmed else tr.own_votes >= tr.enemy_votes
                for j, (key, own, d) in enumerate(dets):
                    dist = float(np.hypot(tr.x - d.x, tr.y - d.y))
                    sp = max(D.unit(tkey).speed if not tkey.startswith("label:") else 120,
                             D.unit(key).speed if not key.startswith("label:") else 120)
                    if dist > max(1500.0, sp * dt * 1000 / D.TICK_MS * 1.5 + 1000.0):
                        continue
                    cost[i, j] = dist + TYPE_PENALTY * (key != tkey) + SIDE_PENALTY * (own != town)
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] >= big:
                    continue
                tr, (key, own, d) = trs[i], dets[j]
                tr.x, tr.y, tr.last_seen, tr.hits, tr.hp_frac, tr.label = d.x, d.y, r.t, tr.hits + 1, d.hp_frac, d.label
                if not tr.confirmed:
                    tr.votes[key] += 1
                    tr.own_votes += own
                    tr.enemy_votes += not own
                matched_t.add(i)
                matched_d.add(j)
        for j, (key, own, d) in enumerate(dets):
            if j in matched_d:
                continue
            self.tracks.append(Track(id=self.next_id, x=d.x, y=d.y, born=r.t, last_seen=r.t, votes=Counter({key: 1}),
                                     own_votes=int(own), enemy_votes=int(not own), hp_frac=d.hp_frac, label=d.label))
            self.next_id += 1
        # confirm: fix type and side by majority vote (side ties -> the half the unit appeared in)
        for tr in self.tracks:
            if not tr.confirmed and tr.hits >= self.min_hits:
                tr.key = tr.vote_key()
                if tr.key not in self.own_keys:
                    tr.own = False
                elif tr.own_votes != tr.enemy_votes:
                    tr.own = tr.own_votes > tr.enemy_votes
                else:
                    tr.own = tr.y >= 16000
                tr.confirmed = True
                self._reidentify(tr)
        keep = []
        for tr in self.tracks:
            if r.t - tr.last_seen > self.keep_s:
                if tr.confirmed:
                    self.dead.append((tr, tr.last_seen))
            else:
                keep.append(tr)
        self.tracks = keep
        self.dead = [(tr, td) for tr, td in self.dead if r.t - td <= max(2.0, self.reid_s)]

    def _reidentify(self, tr: Track) -> None:
        """A newly confirmed track that is a recently lost track seen again (the detector missed it for a few
        frames) takes back the lost track's identity: id (entity order), birth time, card and play status, so
        it is not counted as a new play."""
        best, best_cost = None, None
        for old, t_lost in self.dead:
            if old.key != tr.key or old.own != tr.own:
                continue
            gap = tr.born - t_lost
            if gap < 0 or gap > self.reid_s:
                continue
            reach = D.unit(tr.key).speed * gap * 1000 / D.TICK_MS * 1.5 + 1500.0 if tr.known else 1500.0 + 2000.0 * gap
            dist = float(np.hypot(old.x - tr.x, old.y - tr.y))
            if dist <= reach and (best_cost is None or dist < best_cost):
                best, best_cost = old, dist
        if best is None:
            return
        self.dead = [(o, t) for (o, t) in self.dead if o is not best]
        tr.id, tr.born, tr.card, tr.play = best.id, best.born, best.card, best.play

    # ------------------------------------------------------------------ attribution
    def _parent(self, own: bool, key: str, cx: float, cy: float, t: float, death: bool):
        """A same-side unit nearby whose spawner (death=False) / death spawn (death=True) produces `key`."""
        best, best_d = None, float(self.spawn_radius)
        if death:
            cands = [tr for tr, td in self.dead if t - td <= 1.5] + [tr for tr in self.tracks if tr.last_seen < t]
        else:
            cands = [tr for tr in self.tracks if tr.born < t]
        for tr in cands:
            if not tr.confirmed or tr.own != own or not tr.known or tr.card < 0:
                continue
            u = D.unit(tr.unit)
            prod = u.dspawn_unit if death else u.spawn_unit
            if not prod or unit_key(prod) != key:
                continue
            dist = float(np.hypot(tr.x - cx, tr.y - cy))
            if dist <= best_d:
                best, best_d = tr, dist
        return best

    def _side_override(self, own: bool, comp: Counter, cx: float, cy: float, t: float):
        """Side evidence stronger than the per-frame side votes, or None:
        (1) spawner / death-spawn output belongs to its spawner's owner (a parent of either side nearby);
        (2) units of a card the viewer just played (it left the hand), appearing where the viewer can deploy
            it, are the viewer's."""
        if len(comp) == 1:
            key = next(iter(comp))
            for side in (own, not own):
                if self._parent(side, key, cx, cy, t, death=False) or self._parent(side, key, cx, cy, t, death=True):
                    return side
        if not own:
            for tp, c in self._own_played:
                if abs(t - tp) > 2.0:
                    continue
                cc = Counter({unit_key(u): n for u, n in self.comp[c].items()})
                if cc and set(comp) <= set(cc) and self._deployable(True, c, cx, cy):
                    return True
        return None

    def _late_join(self, own: bool, comp: Counter, cx: float, cy: float, born: float):
        """A recent event of the same side nearby whose card's composition still has room for these units."""
        best, best_d = None, float(self.radius)
        for ev in reversed(self.events):
            if born - ev.t > self.late_s:
                break
            if ev.own != own or ev.card < 0 or born < ev.t:
                continue
            if ev.play:       # a play: room up to its card's composition
                cc = Counter({unit_key(u): n for u, n in self.comp[ev.card].items()})
                if any(ev.joined[k] + n > cc.get(k, 0) for k, n in comp.items()):
                    continue
            elif not set(comp) <= set(ev.joined):   # spawner / death output: more units of the same kinds
                continue
            d = float(np.hypot(ev.x - cx, ev.y - cy))
            if d <= best_d:
                best, best_d = ev, d
        return best

    def _deployable(self, own: bool, card: int, x: float, y: float, normal_only: bool = False) -> bool:
        """Could `own`'s side have played `card` with its units appearing at own-frame (x, y)?
        normal_only: ignore the deploy-anywhere exception (is (x, y) in the side's normal deploy zone?)."""
        if card in DEPLOY_ANYWHERE and not normal_only:
            return True
        if own:
            if y >= 16500:
                return True
            fr, cx = self.enemy_tower_frac, 9000           # a destroyed enemy princess tower opens a pocket
            return y >= 5500 and ((x < cx and fr[1] <= 0) or (x >= cx and fr[2] <= 0))
        if y <= 15500:
            return True
        fr, cx = self.own_tower_frac, 9000
        return y <= 26500 and ((x < cx and fr[1] <= 0) or (x >= cx and fr[2] <= 0))

    def _best_card(self, own: bool, comp: Counter, t: float, cx: float, cy: float) -> int:
        if own:
            cands = self.deck
        else:
            cands = sorted(self.opp_seen) if len(self.opp_seen) >= 8 else range(D.N_CARDS)
        recent_own = {c for (tp, c) in self._own_played if t - tp <= 2.0}
        best, best_key = -1, None
        for c in cands:
            cc = Counter({unit_key(u): n for u, n in self.comp[c].items()})
            if not cc or not set(comp) <= set(cc):
                continue
            extra = sum(max(0, comp[u] - cc[u]) for u in comp)
            missing = sum(max(0, cc[u] - comp[u]) for u in cc)
            pref = (c in recent_own) if own else (c in self.opp_seen)
            # a deploy-anywhere card (Goblin Barrel, Miner, Barbarian Barrel) explains units in the side's normal
            # deploy zone only as a last resort: there, 3 Goblins are far more often Goblins with one missed
            anywhere_penalty = 2 if (c in DEPLOY_ANYWHERE and self._deployable(own, -1, cx, cy, normal_only=True)) else 0
            k = (not self._deployable(own, c, cx, cy), 2 * extra + missing + anywhere_penalty, not pref, c)
            if best_key is None or k < best_key:
                best, best_key = c, k
        return best

    def _attribute(self, t: float) -> None:
        pending = [tr for tr in self.tracks if tr.confirmed and tr.play is None and tr.known]
        pending.sort(key=lambda tr: (tr.born, tr.id))
        done = set()
        for seed in pending:
            if seed.id in done:
                continue
            group = [tr for tr in pending if tr.id not in done and tr.own == seed.own
                     and abs(tr.born - seed.born) <= self.window
                     and np.hypot(tr.x - seed.x, tr.y - seed.y) <= self.radius]
            comp = Counter(tr.key for tr in group)
            cx = float(np.mean([tr.x for tr in group]))
            cy = float(np.mean([tr.y for tr in group]))
            side = self._side_override(seed.own, comp, cx, cy, seed.born)
            if side is not None and side != seed.own:   # side votes overruled by spawner ownership / our own play
                for tr in group:                         # (seed is in the group)
                    tr.own = side
            card, play = -1, True
            late = self._late_join(seed.own, comp, cx, cy, seed.born)
            if late is not None:              # units of a recent play first detected late: not a new play
                for tr in group:
                    tr.card, tr.play = late.card, late.play
                    done.add(tr.id)
                    late.joined[tr.key] += 1
                continue
            if len(comp) == 1:
                key = next(iter(comp))
                parent = self._parent(seed.own, key, cx, cy, seed.born, death=False) or \
                    self._parent(seed.own, key, cx, cy, seed.born, death=True)
                if parent is not None:
                    card, play = parent.card, False
            if card < 0:
                card = self._best_card(seed.own, comp, seed.born, cx, cy)
                exact = card >= 0 and Counter({unit_key(u): n for u, n in self.comp[card].items()}) == comp
                if not exact and t - seed.born < self.settle_s:
                    continue                  # the group may still be appearing / confirming: decide later
                if card < 0 or not self._deployable(seed.own, card, cx, cy):
                    play = False              # not explainable as a play here: keep the units, no history
            for tr in group:
                tr.card, tr.play = card, play
                done.add(tr.id)
            self.events.append(PlayEvent(t=seed.born, own=seed.own, card=card, x=cx, y=cy, play=play,
                                         units=dict(comp), joined=Counter(comp)))
            if play and not seed.own and card >= 0:
                self.opp_seen.add(card)
                self.opp_last4 = [card] + self.opp_last4[:3]
                self.opp_spent += D.CARD_COSTS[card] * D.ELIXIR_UNIT

    # ------------------------------------------------------------------ main entry
    def update(self, r: FrameReading) -> BoardState:
        self._update_elixir(r.t, r.elixir)
        self._update_hand(r.t, r.hand, r.next_card)
        self._update_towers(r.t, r.own_towers, r.enemy_towers)
        self._associate(r)
        self._attribute(r.t)
        self.t = r.t
        return self.board()

    def board(self) -> BoardState:
        t = self.t or 0.0
        tick = self.tick(t)
        tu = D.tower_units()

        def towers(fracs, tt):
            return [Tower(round(f * D.unit(tu["king"] if i == 0 else tu[tt]).hp),
                          D.unit(tu["king"] if i == 0 else tu[tt]).hp) for i, f in enumerate(fracs)]

        own_t, en_t = towers(self.own_tower_frac, self.tower_troops[0]), towers(self.enemy_tower_frac, self.tower_troops[1])
        crowns_own = 3 if not en_t[0].alive else sum(not x.alive for x in en_t[1:])
        crowns_en = 3 if not own_t[0].alive else sum(not x.alive for x in own_t[1:])
        king_own = own_t[0].hp < own_t[0].max_hp or any(not x.alive for x in own_t[1:])
        king_en = en_t[0].hp < en_t[0].max_hp or any(not x.alive for x in en_t[1:])
        units = []
        for tr in sorted(self.tracks, key=lambda tr: tr.id):
            if not tr.confirmed or not tr.known:
                continue
            u = D.unit(tr.unit)
            deploy_ms = D.card_info(tr.card).get("deploy_time_ms", u.deploy_ms) if (tr.play and tr.card >= 0) else 0
            hp = u.hp if tr.hp_frac is None else max(1, round(tr.hp_frac * u.hp))
            units.append(Unit(unit=tr.unit, card=tr.card, own=tr.own, x=int(tr.x), y=int(tr.y), hp=hp, max_hp=u.hp,
                              inactive=(t - tr.born) * 1000 < deploy_ms))
        return BoardState(
            tick=tick, elixir=self.elixir_units if self.elixir_units is not None else 0, hand=list(self.hand),
            next_card=self.next_card, deck=list(self.deck), own_towers=own_t, enemy_towers=en_t,
            king_active=(king_own, king_en), crowns=(crowns_own, crowns_en), tower_troops=self.tower_troops,
            units=units, spells=[], opp_seen=set(self.opp_seen), opp_spent=self.opp_spent,
            opp_last4=list(self.opp_last4), lockout_ticks=self.lockout_ticks,
            over=not own_t[0].alive or not en_t[0].alive)

    def unknown_units(self) -> list:
        """Confirmed tracks of units outside our 64 cards (for the overlay)."""
        return [tr for tr in self.tracks if tr.confirmed and not tr.known]
