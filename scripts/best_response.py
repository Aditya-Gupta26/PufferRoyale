#!/usr/bin/env python3
"""Exploitability probe (SPEC §15.4, §15.7.11): train a FRESH learner against one frozen target
checkpoint, then measure how well it exploits the target.

    python scripts/best_response.py --target experiments/league/<run>/snap_40.pt \\
        --total-timesteps 200000 --matches 40 --out br.json

1. Training: PuffeRL on a LeagueVecEnv whose pool is the target only (self-play 0, anchor
   fraction 1, no snapshots); the learner's seat is drawn per episode, the target's actions are
   sampled from its masked policy (as in the league).
2. Evaluation: --matches full matches through pufferroyale.Game, the best response (BR) in seat 0
   for the even-numbered matches and in seat 1 for the others (team 0 always plays --deck0);
   actions are sampled for both sides (as in training), or the card-first greedy rule with
   --greedy (SPEC §19.11: the most likely of wait / the 4 card slots, then that slot's best tile).

v0.5 (SPEC §19.7.9): the BR trains and is evaluated under the target's env settings, including its
placement_grid (from its run's config.json; the grid always matches the target's weights). With
--init-from-target the learner starts from the target's own weights (architecture, recurrence and
grid all come from the target; --rnn is then ignored); otherwise it is a fresh policy of the
royale.ini / --policy.* architecture. --shaping-anneal-frac F anneals the reward-v2 shaping over
N = ceil(F * total / num_envs) env steps (exact; league args shaping_anneal_steps); the env always gets
reward_gamma = train.gamma. Shaping weights > 0 with train.reward_clip > 0 print a warning (use
--train.reward-clip 0). With --data-dir a config.json (policy, rnn_name, rnn, env -- with the expanded
deck_pool_decks / heldout_decks_decks --, league) is written next to br.pt.

Output: br_score = the BR's score vs the target (win 1 / draw 0.5 / loss 0) and
exploitability_proxy = 2 * br_score - 1 (0 = the BR cannot beat the target, 1 = it always wins;
a lower bound on the target's exploitability, only as tight as the BR training). The JSON is the
last stdout line and is also written to --out. BR weights are kept only with --data-dir
(<data-dir>/best_response/<run_id>/br.pt).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="frozen checkpoint to exploit: a model .pt, 'ckpt:<path>', "
                                                    "or a run directory (its newest model_*.pt / learner.pt / snap)")
    ap.add_argument("--total-timesteps", type=int, default=100_000, help="BR training steps (default 100k)")
    ap.add_argument("--matches", type=int, default=20, help="evaluation matches, half per seat (default 20)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the result JSON here too")
    ap.add_argument("--num-envs", type=int, default=8, help="parallel training matches (default 8)")
    ap.add_argument("--deck0", default="hog26", help="team 0 deck: preset, 'random', or 8 comma-separated card "
                                                     "names / ids (default hog26)")
    ap.add_argument("--deck1", default="hog26", help="team 1 deck (default hog26)")
    ap.add_argument("--frame-skip", type=int, help="env frame_skip for training and evaluation (default: the "
                                                   "target's training frame_skip, else 10)")
    ap.add_argument("--ignore-train-config", action="store_true",
                    help="do not apply the target's training env settings (frame_skip, lockout, tiebreak, "
                         "tower troops) from its run's config.json")
    ap.add_argument("--greedy", action="store_true",
                    help="evaluate with card-first greedy actions (SPEC §19.11; default: sampled)")
    ap.add_argument("--bptt-horizon", type=int, help="PPO segment length (default: royale.ini, 64)")
    ap.add_argument("--learning-rate", type=float, help="default: royale.ini (3e-4)")
    ap.add_argument("--rnn", action="store_true", help="LSTM best response (ignored with --init-from-target)")
    ap.add_argument("--init-from-target", action="store_true",
                    help="the BR starts from the target's weights (its architecture and grid)")
    ap.add_argument("--shaping-anneal-frac", type=float, default=0.0,
                    help="anneal the reward shaping weights to 0 over this fraction of --total-timesteps "
                         "(default 0 = constant)")
    ap.add_argument("--data-dir", help="keep the BR weights under <data-dir>/best_response/<run_id>/")
    ap.add_argument("--verbose", action="store_true", help="print a progress line per epoch")
    return ap


def main():
    known, rest = make_parser().parse_known_args()
    import numpy as np
    import torch
    import league_train as LT
    from pufferroyale import metagame as mg
    from pufferroyale.league import OpponentPool

    from pufferroyale.league import checkpoint_env_config, resolve_checkpoint_path
    from pufferroyale.game import _deck_arg

    if known.matches < 1:
        raise SystemExit("--matches must be >= 1 (the probe needs at least one evaluation match)")
    try:
        target = os.path.abspath(resolve_checkpoint_path(known.target))
        decks = (LT.deck_value(known.deck0), LT.deck_value(known.deck1))
        for d in decks:
            _deck_arg(d)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"best_response.py: {e}")
    # the BR trains and is evaluated under the env settings the target was trained with (audit L7); the
    # placement grid is the target's own (SPEC §19.4: it is fixed by the size of its action head)
    from pufferroyale.league import load_state_dict_file, policy_kwargs_from_state_dict
    target_sd = load_state_dict_file(target)
    target_kw, target_rnn = policy_kwargs_from_state_dict(target_sd)
    env_over = {} if known.ignore_train_config else dict(checkpoint_env_config(target) or {})
    grid = int(target_kw.get("placement_grid", 1))
    if "placement_grid" in env_over and int(env_over["placement_grid"]) != grid:
        print(f"[br] warning: config.json says placement_grid {env_over['placement_grid']} but the target's weights "
              f"act on grid {grid}: using {grid}", file=sys.stderr)
    env_over["placement_grid"] = grid
    if known.frame_skip:
        env_over["frame_skip"] = int(known.frame_skip)
    if env_over:
        print(f"[br] env settings for training and evaluation: {env_over}", flush=True)
    t0 = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_s{known.seed}"
    keep = known.data_dir is not None
    run_dir = (os.path.abspath(os.path.join(known.data_dir, "best_response", run_id)) if keep
               else tempfile.mkdtemp(prefix="pufferroyale_br_"))
    os.makedirs(run_dir, exist_ok=True)

    a = dict(LT.DEFAULTS)
    a.update(total_timesteps=known.total_timesteps, device=known.device, num_envs=known.num_envs, seed=known.seed,
             deck0=known.deck0, deck1=known.deck1, bptt_horizon=known.bptt_horizon, learning_rate=known.learning_rate,
             frame_skip=env_over.get("frame_skip"), shaping_anneal_frac=known.shaping_anneal_frac,
             rnn=(target_rnn is not None) if known.init_from_target else known.rnn, snapshot_interval=0, mmd_coef=0.0,
             anchors=f"ckpt:{target}", self_play_frac=0.0, anchor_frac=1.0)
    ini = LT.load_ini(rest)
    try:
        LT.fix_anneal_steps(a, known)                     # SPEC §19.10.1: exact N, recorded in config.json
    except ValueError as e:
        raise SystemExit(f"best_response.py: {e}")
    pool = OpponentPool(anchors=(f"ckpt:{target}",), max_snapshots=1, self_play_frac=0.0, anchor_frac=1.0,
                        seed=known.seed)
    torch.manual_seed(known.seed)
    np.random.seed(known.seed)
    try:
        LT.build_config(a, ini, run_dir)                  # size / total checks before anything starts
        arch = dict(policy_kwargs=target_kw, rnn_kwargs=target_rnn) if known.init_from_target else {}
        lv, policy, trainer, cfg = LT.build(a, ini, pool, run_dir, run_id,
                                            env_overrides={k: v for k, v in env_over.items() if k != "frame_skip"},
                                            **arch)
        try:
            from pufferroyale.trainer import warn_shaping_clip                  # SPEC §19.10.2
            env_used = LT.env_kwargs(a, ini)
            env_used.update({k: v for k, v in env_over.items() if k != "frame_skip"})
            warn_shaping_clip(env_used, cfg.get("reward_clip", 1.0), "br")
            if known.init_from_target:                    # SPEC §19.7.9: start from the target's weights
                policy.load_state_dict({k: v.to(next(policy.parameters()).device) for k, v in target_sd.items()})
            if keep:
                from pufferroyale.decks import expanded_deck_sets
                from pufferroyale.league import policy_architecture
                env_used.update(expanded_deck_sets(env_used))            # SPEC §19.10.5
                with open(os.path.join(run_dir, "config.json"), "w") as f:
                    json.dump({"policy": policy_architecture(policy)[0], "rnn_name": "Recurrent" if a["rnn"] else None,
                               "rnn": target_rnn if (known.init_from_target and target_rnn) else ini["rnn"],
                               "env": env_used, "league": a}, f, indent=2, default=str)
            while trainer.global_step < known.total_timesteps:
                trainer.last_log_time = -1e18
                trainer.evaluate()
                trainer.train()
                if known.verbose:
                    st = pool.stats().get(f"ckpt:{target}", {})
                    print(f"[br] step {trainer.global_step:>8d} epoch {trainer.epoch:>4d} "
                          f"train score vs target {st.get('p', 0.5):.3f} ({st.get('games', 0)} games)", flush=True)
        finally:
            trainer.utilization.stop()
            lv.close()
        train_stats = pool.stats().get(f"ckpt:{target}", {"games": 0, "p": 0.5})
        policy.eval()
        if keep:
            torch.save(LT.cpu_state(policy), os.path.join(run_dir, "br.pt"))

        # ---- evaluation: full matches, half per seat, under the training settings
        match_kw = dict(match_config={k: v for k, v in env_over.items() if k in mg.DEFAULT_MATCH},
                        use_train_config=False)
        w = d = l = 0
        for k in range(known.matches):
            s = mg.match_seed(known.seed, 0xB5, k)
            if k % 2 == 0:
                r = mg.play_match(policy, target, decks[0], decks[1], s, greedy=known.greedy, device=known.device,
                                  **match_kw)
            else:
                r = -mg.play_match(target, policy, decks[0], decks[1], s, greedy=known.greedy, device=known.device,
                                  **match_kw)
            w, d, l = w + (r > 0), d + (r == 0), l + (r < 0)
        n = max(1, known.matches)
        br_score = (w + 0.5 * d) / n
        result = {
            "target": target,
            "br_score": br_score,
            "exploitability_proxy": 2.0 * br_score - 1.0,
            "matches": int(known.matches), "wins": int(w), "draws": int(d), "losses": int(l),
            "total_timesteps": int(trainer.global_step),
            "train_games_vs_target": int(train_stats["games"]), "train_score_vs_target": float(train_stats["p"]),
            "greedy": bool(known.greedy), "deck0": known.deck0, "deck1": known.deck1, "seed": int(known.seed),
            "init_from_target": bool(known.init_from_target), "shaping_anneal_frac": float(known.shaping_anneal_frac),
            "env_settings": env_over,
            "br_checkpoint": os.path.join(run_dir, "br.pt") if keep else None,
            "elapsed_s": round(time.time() - t0, 2),
        }
    finally:
        if not keep:
            shutil.rmtree(run_dir, ignore_errors=True)
    from pufferroyale.league import json_safe
    line = json.dumps(json_safe(result), allow_nan=False)                 # strict JSON (SPEC §18.8f)
    if known.out:
        with open(known.out, "w") as f:
            f.write(line + "\n")
    print(f"[br] best response vs {os.path.basename(target)}: W {w} D {d} L {l} over {known.matches} matches "
          f"-> br_score {br_score:.3f}, exploitability proxy {2 * br_score - 1:+.3f}", flush=True)
    print(line)


if __name__ == "__main__":
    main()
