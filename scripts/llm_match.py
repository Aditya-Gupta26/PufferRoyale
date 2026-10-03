#!/usr/bin/env python3
"""Play PufferRoyale matches with a text model (SPEC §17.3): a mock baseline or an Anthropic model.

    # offline baselines (no network)
    python scripts/llm_match.py --model mock_first_legal --opponent bot:heuristic --matches 10 --out runs/fl.json
    python scripts/llm_match.py --model mock_random --opponent bot:random --matches 10 --seed 3 --out runs/rnd.json
    # a real model: needs `pip install anthropic`, credentials, and an explicit --allow-network
    python scripts/llm_match.py --model anthropic:claude-opus-5-5 --effort low --opponent bot:heuristic \\
        --matches 4 --out runs/opus.json --allow-network

The model plays --deck-agent, the opponent --deck-opp; seats alternate (even-numbered matches:
the model is team 0). The model decides every --decision-interval ticks (20 = 1 s of game time);
bot / checkpoint opponents act every 10 ticks (the env cadence). Opponent: a bot spec
(bot:noop | bot:random | bot:heuristic), a checkpoint path, or another model spec prefixed with
"model:" (e.g. model:mock_random). A checkpoint opponent samples its actions from its masked policy
(seeded by the match seed; SPEC §19.11); --opponent-greedy makes it play the card-first greedy rule.

Output: one progress line per match, then a JSON summary as the LAST stdout line (win/draw/loss,
score, mean crowns, illegal / parse-error / model-error rates per decision, mean latency). With
--out, the summary plus per-match results are written there and every decision (tick, prompt
hash, reply, action, error, latency, API metadata) goes to <out stem>.transcripts.jsonl.
anthropic:<model> is refused unless --allow-network is given, so nothing calls out by accident.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MOCK_MODELS = ("mock_wait", "mock_first_legal", "mock_random")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="mock_wait | mock_first_legal | mock_random | anthropic:<model-id> "
                         "(claude-opus-5-5, claude-opus-5, claude-fable-5-1, claude-sonnet-5, claude-haiku-4-5)")
    ap.add_argument("--opponent", default="bot:heuristic",
                    help="bot:noop | bot:random | bot:heuristic | <checkpoint path> | model:<model spec> "
                         "(default bot:heuristic)")
    ap.add_argument("--opponent-greedy", action="store_true",
                    help="a checkpoint opponent plays the card-first greedy rule (SPEC §19.11) instead of "
                         "sampling")
    ap.add_argument("--deck-agent", default="hog26", help="the model's deck (preset or 'random'; default hog26)")
    ap.add_argument("--deck-opp", default="hog26", help="the opponent's deck (default hog26)")
    ap.add_argument("--matches", type=int, default=10, help="number of matches, seats alternating (default 10)")
    ap.add_argument("--seed", type=int, default=0, help="base seed; each match gets its own derived seed")
    ap.add_argument("--decision-interval", type=int, default=20, help="ticks between model decisions (default 20)")
    ap.add_argument("--out", help="write the summary + per-match results here; transcripts next to it")
    ap.add_argument("--effort", default="low", help="anthropic: output_config effort (default low; ignored "
                                                    "for claude-haiku-4-5)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="anthropic: max_tokens per reply (default 4096)")
    ap.add_argument("--allow-network", action="store_true",
                    help="required for anthropic: models (they call the Anthropic API)")
    ap.add_argument("--max-history", type=int, default=0, help="include the model's last N decisions in each prompt")
    ap.add_argument("--max-ticks", type=int, default=6000, help="cut matches after this many ticks (default 6000)")
    ap.add_argument("--skip-idle", action="store_true",
                    help="do not call the model when it has no legal play (deploy lockout, nothing affordable)")
    ap.add_argument("--save-prompts", action="store_true", help="store the full state text in the transcripts")
    ap.add_argument("--print-prompt", action="store_true", help="print the first rendered state text and exit")
    return ap


def summarize(model, opponent, args, results, transcripts_path):
    n = len(results)
    w = sum(r["result"] > 0 for r in results)
    d = sum(r["result"] == 0 for r in results)
    lo = n - w - d
    dec = sum(r["decisions"] for r in results)
    lat = [r["mean_latency_s"] for r in results if r["decisions"]]

    def rate(key):
        return (sum(r[key] for r in results) / dec) if dec else 0.0

    return {
        "model": model, "opponent": opponent, "deck_agent": args.deck_agent, "deck_opp": args.deck_opp,
        "matches": n, "wins": w, "draws": d, "losses": lo, "score": (w + 0.5 * d) / n if n else 0.0,
        "mean_crowns": [sum(r["crowns"][0] for r in results) / max(n, 1), sum(r["crowns"][1] for r in results) / max(n, 1)],
        "decisions": dec, "plays_per_match": sum(r["plays"] for r in results) / max(n, 1),
        "illegal_rate": rate("illegal"), "parse_error_rate": rate("parse_errors"), "model_error_rate": rate("model_errors"),
        "mean_latency_s": (sum(r["mean_latency_s"] * r["decisions"] for r in results) / dec) if dec else 0.0,
        "mean_ticks": sum(r["ticks"] for r in results) / max(n, 1),
        "seed": args.seed, "decision_interval": args.decision_interval, "transcripts": transcripts_path,
        "opponent_greedy": bool(args.opponent_greedy),
    }


def main():
    args = make_parser().parse_args()
    if args.matches < 1 and not args.print_prompt:
        print("llm_match.py: --matches must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.decision_interval < 1:
        print("llm_match.py: --decision-interval must be >= 1", file=sys.stderr)
        sys.exit(2)
    from pufferroyale import llm
    from pufferroyale.league import json_safe
    from pufferroyale.metagame import match_seed

    specs = [args.model] + ([args.opponent[6:]] if args.opponent.startswith("model:") else [])
    for spec in specs:
        if spec.startswith("anthropic:") and not args.allow_network:
            print(f"refusing to run {spec}: it would call the Anthropic API over the network. "
                  f"Re-run with --allow-network if that is intended.", file=sys.stderr)
            sys.exit(2)
        if spec not in MOCK_MODELS and not spec.startswith("anthropic:"):
            print(f"unknown model {spec!r}: use {', '.join(MOCK_MODELS)} or anthropic:<model-id>", file=sys.stderr)
            sys.exit(2)
        if spec.startswith("anthropic:") and spec.split(":", 1)[1] not in llm.KNOWN_ANTHROPIC_MODELS:
            print(f"warning: {spec} is not one of the known ids {llm.KNOWN_ANTHROPIC_MODELS}", file=sys.stderr)

    def model_fn(spec, seed):
        return llm.make_model_fn(spec, seed=seed, effort=args.effort, max_tokens=args.max_tokens,
                                 allow_network=args.allow_network)

    try:
        agent = llm.LLMAgent(model_fn(args.model, args.seed), max_history=args.max_history, name=args.model,
                             keep_prompts=args.save_prompts)
        opponent = args.opponent
        if opponent.startswith("model:"):
            opponent = llm.LLMAgent(model_fn(opponent[6:], args.seed + 1), decision_interval=args.decision_interval,
                                    max_history=args.max_history, name=opponent[6:])
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    if isinstance(opponent, str) and not opponent.startswith("bot:"):
        from pufferroyale.metagame import resolve_agent
        try:                                           # load the checkpoint once, up front
            resolve_agent(opponent)
        except (ValueError, FileNotFoundError, OSError) as e:
            print(f"cannot use opponent {opponent!r}: {e}", file=sys.stderr)
            sys.exit(2)

    if args.print_prompt:
        from pufferroyale import Game
        g = Game(deck0=args.deck_agent, deck1=args.deck_opp, seed=match_seed(args.seed, 0))
        g.tick(100)
        print(llm.render_state(g, 0))
        return

    transcripts_path = None
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        transcripts_path = os.path.splitext(args.out)[0] + ".transcripts.jsonl"
        open(transcripts_path, "w").close()
    results, t0 = [], time.time()
    for k in range(args.matches):
        team = k % 2
        seed = match_seed(args.seed, k)
        n0 = len(agent.transcript)
        r = llm.play_llm_match(agent, opponent, args.deck_agent, args.deck_opp, seed, agent_team=team,
                               decision_interval=args.decision_interval, max_ticks=args.max_ticks,
                               skip_idle=args.skip_idle, greedy=args.opponent_greedy)
        r.update(match=k, seed=seed)
        results.append(r)
        if transcripts_path:
            with open(transcripts_path, "a") as f:
                for e in agent.transcript[n0:]:
                    f.write(json.dumps(json_safe({"match": k, "seed": seed, "agent_team": team, **e}),
                                       allow_nan=False) + "\n")
        outcome = {1: "WIN", 0: "DRAW", -1: "LOSS"}[r["result"]]
        print(f"[llm] match {k + 1}/{args.matches} seat {team}: {outcome:4s} crowns {r['crowns'][0]}-{r['crowns'][1]} "
              f"({r['end_reason']}, {r['ticks']} ticks)  decisions {r['decisions']} plays {r['plays']} "
              f"illegal {r['illegal']} parse errors {r['parse_errors']} model errors {r['model_errors']}  "
              f"latency {r['mean_latency_s'] * 1000:.1f} ms", flush=True)
    summary = summarize(args.model, args.opponent, args, results, transcripts_path)
    summary["elapsed_s"] = round(time.time() - t0, 2)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(json_safe({"summary": summary, "matches": results}), f, indent=1, allow_nan=False)
    print(json.dumps(json_safe(summary), allow_nan=False))


if __name__ == "__main__":
    main()
