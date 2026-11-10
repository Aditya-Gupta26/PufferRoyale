"""Draw the advisor's view: the game frame with the suggested tile(s) and a side panel of numbers.

Everything is drawn on a COPY shown in the advisor's own window (or written to a video); the game is never
touched.
"""
from __future__ import annotations

import numpy as np

from . import engine_db as D

COLOURS = [(60, 220, 60), (255, 200, 40), (90, 160, 255)]     # BGR: 1st, 2nd, 3rd suggestion
PANEL_W = 360


def _txt(cv2, img, s, xy, scale=0.55, colour=(235, 235, 235), thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def render(frame_rgb: np.ndarray, layout, suggestion=None, board=None, tracker=None, status: str = "",
           show_units: bool = True, heatmap: bool = True) -> np.ndarray:
    """BGR image: the frame with overlays, plus a text panel on the right."""
    import cv2
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    over = img.copy()
    if suggestion is not None and suggestion.cards:
        top = suggestion.cards[0]
        if heatmap:   # P(tile | top card), faint
            rows, cols = top.tile_probs.shape
            m = top.tile_probs / max(top.tile_probs.max(), 1e-9)
            g = 32 // rows          # placement grid: rows = ceil(32 / g) gives 32, 16, 8 for g = 1, 2, 4
            for r in range(rows):
                for c in range(cols):
                    if m[r, c] < 0.05:
                        continue
                    (x0, y0), (x1, y1) = layout.tile_to_px(c * g, r * g), layout.tile_to_px((c + 1) * g, (r + 1) * g)
                    cv2.rectangle(over, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), -1)
            img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
        for k, cs in enumerate(suggestion.cards):
            tx0, ty0, tx1, ty1 = cs.block
            (x0, y0), (x1, y1) = layout.tile_to_px(tx0, ty0), layout.tile_to_px(tx1, ty1)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), COLOURS[k % 3], 3 if k == 0 else 2)
            _txt(cv2, img, f"{k + 1}", (int(x0) + 2, int(y0) - 4), 0.6, COLOURS[k % 3], 2)
    if show_units and tracker is not None:
        for tr in tracker.tracks:
            if not getattr(tr, "confirmed", True):
                continue
            px, py = layout.milli_to_px(tr.x, tr.y)
            col = (255, 160, 60) if tr.own else (60, 60, 255)
            cv2.circle(img, (int(px), int(py)), 4, col, -1)
            name = D.CARD_NAMES[tr.card] if 0 <= tr.card < D.N_CARDS else (tr.label if not tr.known else "?")
            _txt(cv2, img, name[:10], (int(px) + 5, int(py) - 3), 0.38, col)
    panel = np.zeros((h, PANEL_W, 3), np.uint8)
    y = 26
    _txt(cv2, panel, "PufferRoyale advisor (suggestions only)", (8, y), 0.5, (200, 200, 200))
    y += 26
    if status:
        _txt(cv2, panel, status, (8, y), 0.5, (120, 200, 255))
        y += 26
    if suggestion is not None:
        _txt(cv2, panel, f"wait: {100 * suggestion.p_wait:4.0f}%", (8, y), 0.65)
        y += 30
        for k, cs in enumerate(suggestion.cards):
            _txt(cv2, panel, f"{k + 1}. {cs.name}  {100 * cs.p_card:3.0f}%", (8, y), 0.65, COLOURS[k % 3], 2)
            y += 24
            _txt(cv2, panel, f"   tile ({cs.tile[0]},{cs.tile[1]})  {100 * cs.p_tile:3.0f}% of its tiles", (8, y), 0.45)
            y += 26
        _txt(cv2, panel, f"value (own view): {suggestion.value:+.2f}", (8, y), 0.5)
        y += 30
    if board is not None:
        t = board.tick / 20.0
        _txt(cv2, panel, f"clock {int(t // 60)}:{int(t % 60):02d}  elixir {board.elixir / D.ELIXIR_UNIT:4.1f}", (8, y), 0.5)
        y += 24
        hand = ", ".join(D.CARD_NAMES[c] if c >= 0 else "?" for c in board.hand)
        _txt(cv2, panel, f"hand: {hand}"[:52], (8, y), 0.45)
        y += 24
        seen = ", ".join(D.CARD_NAMES[c] for c in sorted(board.opp_seen))
        _txt(cv2, panel, f"opp seen ({len(board.opp_seen)}): {seen}"[:52], (8, y), 0.45)
        y += 22
        last4 = ", ".join(D.CARD_NAMES[c] for c in board.opp_last4 if c >= 0)
        _txt(cv2, panel, f"opp last: {last4}"[:52], (8, y), 0.45)
        y += 22
        ub = min(max(D.ELIXIR_START + D.income(board.tick) - board.opp_spent, 0), D.ELIXIR_MAX) / D.ELIXIR_UNIT
        _txt(cv2, panel, f"opp elixir <= {ub:4.1f} (spells not seen)", (8, y), 0.45)
        y += 22
    if tracker is not None and tracker.unknown_units():
        names = sorted({tr.label for tr in tracker.unknown_units()})
        _txt(cv2, panel, ("not in our 64: " + ", ".join(names))[:52], (8, y), 0.45, (80, 80, 255))
    return np.hstack([img, panel])
