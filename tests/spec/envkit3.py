"""SPEC §16.4 / §16.6 (v0.3) observation helpers -- thin aliases of envkit (which now implements the
v0.3 layout for the whole suite)."""
import envkit as E
from envkit import (CARD_SLOTS, N_OWN, R, entity_rows, hand_of, id_vec, ids_to_cards, multihot,  # noqa: F401
                    next_of, scalar, tt_onehot)

V3_FIELDS = E.SCALAR_FIELDS
V3_TOTAL = E.SCALAR_TOTAL                                  # 298
TT_LEN = 4
F_ID, F_X, F_Y, F_HPF, F_HP2K, F_FLY, F_DEPLOY, F_STUN, F_SLOW, F_BUILDING, F_TOB = range(11)


def v3_layout():
    return E.scalar_layout()
