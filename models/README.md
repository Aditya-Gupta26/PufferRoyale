# Model card: `S4_A_s1_10B`

The agent behind the [play site](https://aditya-gupta26.github.io/pr-play/): a PPO policy trained by pure
self-play on the PufferRoyale simulator for 10 billion steps.

| | |
|---|---|
| Files | `S4_A_s1_10B.pt` (the weights, a PyTorch state dict) and `S4_A_s1_10B/config.json` (the run's full configuration; the loader reads the environment settings from it) |
| sha256 of `S4_A_s1_10B.pt` | `46c62cddee6a1a9266fba2c4a5999bd547c4859ef7ace0b25198138c6557f9a4` |
| Origin | run `S4_A_s1`, file `model_610352.pt` (epoch 610,352 = 10,000,007,168 learner steps), finished 2026-11-01 |
| Size | 1,287,556 parameters, float32, 5.2 MB; feed-forward (no LSTM) |
| Licence | MIT, like the rest of the repository |

The only edit to `config.json` is its `data_dir`, generalised to `/scratch/$USER/pufferroyale/runs`. The
weights are byte-identical to the original file.

## Using it

```bash
python play/play_server.py                                                    # play it in a browser (its default opponent)
python scripts/watch.py --checkpoint models/S4_A_s1_10B.pt --p1 heuristic    # watch it play the scripted bot
python scripts/eval.py --checkpoint models/S4_A_s1_10B.pt --episodes 10       # score it against the scripted bots
```

From Python, `scripts/watch.py`'s `PolicyPlayer` wraps it:

```python
import sys; sys.path.insert(0, "scripts")
import pufferroyale as pr
from watch import PolicyPlayer

agent = PolicyPlayer("models/S4_A_s1_10B.pt", greedy=False, seed=0)   # sampled actions, as in every evaluation
g = pr.Game(deck0="hog26", deck1="giant", seed=0)
a = agent.act(g.obs(0))           # an action in Discrete(2305): 0 = wait, else 1 + slot * 576 + tile
```

It expects the environment it was trained with:

- frame skip 10 (one decision per 0.5 s), deploy lockout 90 ticks, absolute-HP tiebreak;
- Princess towers on both sides, the fine placement grid.

`league.load_policy` and the play server read these settings from `S4_A_s1_10B/config.json`.

## Training

| Setting | Value |
|---|---|
| Algorithm | PPO (PufferLib 3.0 `PuffeRL`), 256 learner rows, batch 16,384, minibatch 1,024, 1 update epoch, BPTT 64 |
| Optimiser | Adam, learning rate 3e-4 constant, γ = 0.999, GAE λ = 0.95, clip 0.2, value coefficient 2.0, max grad norm 1.5, entropy 0.01 |
| Opponents | itself only (method A: self-play fraction 1.0, no anchors, no MMD) |
| Reward | win/loss/draw ±1/0, plus potential-based shaping (tower 0.3, crown 0.2, elixir 0.01) annealed to 0 over the first 0.5B steps; reward clipping off |
| Card floor | 0.005, faded to 0 over the first 5B steps (training only) |
| Decks | per seat: a uniformly random 8-card deck with probability 0.5, else one of 7 presets (hog26, giant, bait, golem, lavaloon, miner_poison, pekka_bridge); 52 decks held out (xbow, royal_hogs, 50 random) |
| Volume | 10,000,007,168 steps; 23,862,912 self-play matches |
| Hardware | one NVIDIA A100 80GB, 188.7 h in 13 links of 20 h (about 14,800 steps/s) |
| Seed | 1 |

The exact command line is in the [main README](../README.md#build-test-and-reproduce).

## Evaluation

| Opponent | Score [Wilson 95%] | Matches | Checkpoint |
|---|---|---|---|
| Heuristic bot, held-out decks | 0.939 [0.923, 0.952] | 1,040 | 9.9B |
| Heuristic bot, pool decks | 0.950 [0.900, 0.976] | 140 | 9.9B |
| Heuristic bot, pool / held-out (pre-registered analysis point) | 0.912 [0.893, 0.928] / 0.906 [0.886, 0.922] | 1,008 / 1,040 | 5.0B |
| Random bot, pool / held-out | 1.000 / 0.998 | 140 / 1,040 | 9.9B |
| Hog-mirror specialists S3_A_s1 / S3_C_s0 (fair = 0.5) | 0.210 / 0.075 | 200 each | 9.9B |
| Copy-start attackers, 150M steps (br_score; < 0.5 = no exploit) | 0.4845, 0.4565 | 1,000 each | 7.81B |
| From-scratch attackers, 1B steps | 0.0745, 0.166 | 1,000 each | 7.81B |
| Humans, ranked (human's score) | 0.15 [0.05, 0.36] (3 W, 17 L) | 20 | 10.0B (this file) |

Score = (wins + draws/2) / matches, both seats, sampled actions, no card floor. No large-sample bot evaluation
was run on this exact checkpoint; the 9.9B point (100M steps earlier) is the closest one. The selection pipeline
measured exploitability on the 7.81B snapshot; no attacker has been trained against this file yet.

## Intended use

- **Research:** a baseline opponent for the simulator, a starting point for further training, and a subject for
  exploitability studies.
- **Playing for fun** on the play site or a local server.
- **Not for the real game.** Using it to play or advise in Clash Royale matches breaks Supercell's Terms of
  Service (see the advisor warning in the [main README](../README.md#the-advisor-for-the-real-game)).

## Limitations

- **Simulator only.** It learned in a simulator that is not calibrated against real matches. 64 cards are
  modelled, all at level 11, and it trained with Princess towers only.
- **Weak against specialists.** It loses clearly to agents trained on a single deck when they play that deck.
- **Exploitability is a lower bound**, from 4 attackers on an earlier checkpoint.
- **Seeds matter.** The same configuration with seed 0 ranked last of 9 in the selection round robin.
- **May be replaced.** The stage-5 card-floor verdict is pending (see
  [`docs/STAGE4_SELECTION.md`](../docs/STAGE4_SELECTION.md)); if it names a different model, it will be added
  here with its own card.
