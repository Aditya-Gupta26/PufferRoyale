#!/usr/bin/env python3
"""Claude-vs-policy games for one league run (DECISIONS D101; the user's request of 2026-10-09: Claude Opus 5.5
with an unseen coherent deck plays the training policy, one game per LLM_EVERY learner steps, the recorded game in
W&B). Runs in the image next to hpc/llm_relay.py (on the host), started by hpc/llm.sbatch.

Game k (k = 1 .. total // every) is played against the first saved policy at or after k * every learner steps:
the run's snapshot snap_<epoch>.pt (never deleted, written with the save, so never from a discarded trajectory)
or, at the end, its final model_<epoch>.pt; the policy samples its actions (unfloored, as in every evaluation).
Setup of game k (i = k - 1): Claude plays CLAUDE_DECKS[i % 2] (xbow, royal_hogs); the policy plays the same deck
when (i // 2) % 2 == 0 (mirror), else POOL[(i // 4) % 7]; Claude is team (i // 4) % 2. Claude decides every
--decision-interval ticks (20 = 1 s) through the repo's text harness (pufferroyale.llm: RULES_PROMPT,
render_state, parse_action) and skips decisions with no legal play; the policy acts every frame_skip ticks.

Relay protocol (node-local --relay-dir): RULES_PROMPT.txt, then per decision g<k>/turn_<n>.txt (a header with the
opponent's plays since Claude's last turn and Claude's recent decisions, then the rendered state), answered by
g<k>/reply_<n>.json = {"text": <the model's reply>, "meta": {...}} (meta.fatal: the model stayed unavailable, the
game is aborted and retried later, at most --attempts times).

Per game, <run>/llm/game_<k>/: meta.json, frames.jsonl.gz (every tick), plays.jsonl, decisions.jsonl,
turn_*.txt, result.json, game_<k>.gif (hpc/llm_render.py). <run>/llm/state.json records every finished game;
W&B run "<run_id>-llm" (job_type llm, the run's group, resumed by id) gets at step = the policy's global step:
llm/score (Claude's: win 1, draw 0.5), llm/claude_crowns, llm/policy_crowns, decision / error / token counts,
llm/replay (the GIF) and llm/decision_log (a table of Claude's replies).

    python hpc/llm_game.py RUN_DIR --relay-dir DIR [--every-steps 250000000] [--wandb --wandb-group G]
"""
import argparse
import gzip
import json
import os
import re
import shutil
import signal
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLAUDE_DECKS = ("xbow", "royal_hogs")
POOL = ("hog26", "giant", "bait", "golem", "lavaloon", "miner_poison", "pekka_bridge")
N_TILES = 576


# BaseException, not Exception: pufferroyale.llm.LLMAgent.act turns any Exception of the model function into a
# WAIT, and these two must reach main()
class Stop(BaseException):
    """SIGTERM: leave at once (finished games are already recorded)."""


class GameAborted(BaseException):
    """The relay gave up on a decision (the model stayed unavailable) or never answered."""


# ------------------------------------------------------------------------------------ schedule
def game_config(k, claude_decks=CLAUDE_DECKS, pool=POOL):
    i = int(k) - 1
    cd = claude_decks[i % len(claude_decks)]
    mirror = (i // 2) % 2 == 0
    return {"k": int(k), "claude_deck": cd, "policy_deck": cd if mirror else pool[(i // 4) % len(pool)],
            "mirror": mirror, "claude_team": (i // 4) % 2}


def saved_policies(run_dir):
    """[(epoch, path, kind)] of the run's snapshots and final model, by epoch."""
    out = []
    for n in os.listdir(run_dir):
        m = re.fullmatch(r"(snap|model)_(\d+)\.pt", n)
        if m:
            out.append((int(m.group(2)), os.path.join(run_dir, n), m.group(1)))
    return sorted(out)


def pick_policy(run_dir, batch, target_step):
    """The first saved policy at or after `target_step` learner steps, or None (not there yet)."""
    for epoch, path, kind in saved_policies(run_dir):
        if epoch * batch >= target_step:
            return {"epoch": epoch, "path": path, "kind": kind, "global_step": epoch * batch}
    return None


def run_info(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    lg = cfg["league"]
    batch = int(lg.get("batch_size") or int(lg["num_envs"]) * int(lg.get("bptt_horizon") or 64))
    return {"run_id": lg["run_id"], "total": int(lg["total_timesteps"]), "batch": batch}


def run_finished(run_dir, total):
    try:
        st = json.load(open(os.path.join(run_dir, "league_state.json")))
    except (OSError, ValueError):
        return False
    return int(st.get("global_step", 0)) >= total and any(k == "model" for _, _, k in saved_policies(run_dir))


def drop_live(live):
    """Remove a game's live relay directory: renamed first (the relay only looks at g*/), so it never sees a turn
    whose reply is already deleted and never answers a finished game."""
    if os.path.isdir(live):
        gone = os.path.join(os.path.dirname(live), f".gone_{os.path.basename(live)}_{time.time_ns()}")
        os.rename(live, gone)
        shutil.rmtree(gone, ignore_errors=True)


# ------------------------------------------------------------------------------------ one game
def decode(action):
    if action <= 0:
        return None
    slot, cell = divmod(int(action) - 1, N_TILES)
    return slot, cell % 18, cell // 18


def frame(g):
    from pufferroyale.game import CARD_NAMES
    st = g.state()
    ents, towers = [], []
    for e in g.entities():
        if e["kind"] == "tower":
            towers.append([e["team"], e["unit"], e["x"], e["y"], e["hp"], e["max_hp"]])
        else:
            ents.append([e["id"], e["team"], e["unit"], e["card_id"], e["x"], e["y"], e["hp"], e["max_hp"],
                         int(bool(e.get("flying"))), int(bool(e.get("deploying"))), int(bool(e.get("stunned"))),
                         e.get("radius", 0)])
    proj = [[p["team"], p["name"], p["x"], p["y"]] for p in g.projectiles()]
    eff = [[f.get("team", -1), f.get("name", ""), f.get("x", 0), f.get("y", 0), f.get("radius", 0)] for f in g.effects()]
    return {"t": int(st["tick"]), "el": [round(v / 2800, 2) for v in st["elixir"]], "cr": st["crowns"],
            "hand": [[CARD_NAMES[c] for c in h] for h in st["hand"]],
            "q": [CARD_NAMES[q[0]] if q else None for q in st["queue"]],
            "tw": towers, "e": ents, "p": proj, "f": eff, "ot": bool(st["overtime"])}


def _atomic_write(path, text):
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


class FileRelay:
    """model_fn(system, user) for pufferroyale.llm.LLMAgent: writes turn_<n>.txt (header + state) to the live
    directory, waits for reply_<n>.json from hpc/llm_relay.py; .last_call = the reply's meta."""

    def __init__(self, live, archive, header_fn, timeout):
        self.live, self.archive, self.header_fn, self.timeout = live, archive, header_fn, float(timeout)
        self.n, self.last_call = 0, None

    def __call__(self, system, user):
        n = self.n
        text = self.header_fn(n) + "\n" + user
        _atomic_write(os.path.join(self.archive, f"turn_{n:04d}.txt"), text)
        _atomic_write(os.path.join(self.live, f"turn_{n:04d}.txt"), text)
        rp = os.path.join(self.live, f"reply_{n:04d}.json")
        t0 = time.time()
        while not os.path.exists(rp):
            if time.time() - t0 > self.timeout:
                raise GameAborted(f"no reply to decision {n} within {self.timeout:.0f} s (is hpc/llm_relay.py running?)")
            time.sleep(0.2)
        rep = json.load(open(rp))
        self.last_call = rep.get("meta") or {}
        if self.last_call.get("fatal"):
            raise GameAborted(f"decision {n}: {self.last_call.get('error')}")
        self.n += 1
        return rep.get("text") or ""


def play_game(cfg, policy, live, archive, seed, decision_interval=20, max_ticks=6000, reply_timeout=14400):
    """One recorded match; returns the result dict (also written to result.json)."""
    from pufferroyale import llm
    from pufferroyale.game import CARD_NAMES, Game
    os.makedirs(live, exist_ok=True)
    os.makedirs(archive, exist_ok=True)
    me, opp_team = int(cfg["claude_team"]), 1 - int(cfg["claude_team"])
    decks = (cfg["claude_deck"], cfg["policy_deck"]) if me == 0 else (cfg["policy_deck"], cfg["claude_deck"])
    g = Game(deck0=decks[0], deck1=decks[1], seed=int(seed))
    opp, ko = llm._make_opponent(policy["path"], int(seed), opp_team, None, False, "cpu")
    k = int(decision_interval)
    opp_since, recent = [], []

    def header(n):
        st = g.state()
        opp = "; ".join(f"t={p['t']} {p['card']} at ({p['mx']},{p['my']})" for p in opp_since) or "none"
        mine = "; ".join(f"#{r[0]} t={r[1]} {r[2]}" for r in recent[-3:]) or "none"
        return "\n".join([f"=== DECISION {n} | tick {st['tick']} | you are team {me} (your own frame) ===",
                          f"OPPONENT PLAYS SINCE YOUR LAST TURN: {opp}   [tiles in YOUR frame]",
                          f"YOUR LAST DECISIONS: {mine}"])

    relay = FileRelay(live, archive, header, reply_timeout)
    agent = llm.LLMAgent(relay, decision_interval=k, name="claude-code-headless")
    meta = dict(cfg, seed=int(seed), policy=policy, decision_interval_ticks=k, opponent_interval_ticks=ko,
                started=time.strftime("%Y-%m-%dT%H:%M:%S"), claude_deck_name=cfg["claude_deck"],
                policy_deck_name=cfg["policy_deck"],
                claude_deck_cards=[CARD_NAMES[c] for c in g.state()["deck"][me]],
                policy_deck_cards=[CARD_NAMES[c] for c in g.state()["deck"][opp_team]])
    json.dump(meta, open(os.path.join(archive, "meta.json"), "w"), indent=1)

    def log_play(team, action):
        slot, tx, ty = decode(action)
        hand = g.state()["hand"][team]
        gx, gy = (tx, ty) if team == 0 else (17 - tx, 31 - ty)          # own frame -> global (team 0's)
        mx, my = (gx, gy) if me == 0 else (17 - gx, 31 - gy)            # global -> Claude's frame
        return {"t": int(g.state()["tick"]), "team": team, "who": "claude" if team == me else "policy",
                "card": CARD_NAMES[hand[slot]] if slot < len(hand) else "?", "slot": slot, "tx": tx, "ty": ty,
                "gx": gx, "gy": gy, "mx": mx, "my": my, "err": 0}

    plays_f = open(os.path.join(archive, "plays.jsonl"), "w")
    dec_f = open(os.path.join(archive, "decisions.jsonl"), "w")
    frames = gzip.open(os.path.join(archive, "frames.jsonl.gz"), "wt")
    skipped = 0
    try:
        frames.write(json.dumps(frame(g)) + "\n")
        max_ticks = min(int(max_ticks), llm.MAX_TICKS)
        while True:
            st = g.state()
            t = int(st["tick"])
            if st["over"] or t >= max_ticks:
                break
            if t % k == 0:
                if not g.legal_mask(me)[1:].any():
                    skipped += 1
                else:
                    n = relay.n
                    x = agent.act(g, me)                      # GameAborted propagates
                    entry = agent.transcript[-1]
                    rec = log_play(me, x) if x else None
                    err = g.play_action(me, x) if x else 0
                    if rec:
                        rec["err"] = int(err)
                        plays_f.write(json.dumps(rec) + "\n")
                    dec_f.write(json.dumps({"n": n, "t": t, "opp_since": opp_since, "reply": entry["response"],
                                            "action": int(x), "decoded": entry["decoded"], "error": entry["error"],
                                            "error_kind": entry["error_kind"], "play_err": int(err),
                                            "meta": entry["meta"]}) + "\n")
                    dec_f.flush()
                    recent.append((n, t, entry["decoded"] if x else "WAIT"))
                    opp_since = []
            if t % ko == 0:
                b = opp.act(g, opp_team)
                if b:
                    rec = log_play(opp_team, b)
                    err = g.play_action(opp_team, b)
                    rec["err"] = int(err)
                    plays_f.write(json.dumps(rec) + "\n")
                    if not err:
                        opp_since.append(rec)
            g.tick(1)
            frames.write(json.dumps(frame(g)) + "\n")
    finally:
        frames.close()
        plays_f.close()
        dec_f.close()
    st = g.state()
    decs = [json.loads(ln) for ln in open(os.path.join(archive, "decisions.jsonl"))]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    cost, latency = 0.0, []
    for d in decs:
        m = d.get("meta") or {}
        for key in usage:
            usage[key] += int(((m.get("usage") or {}).get(key)) or 0)
        cost += float(m.get("cost_usd_estimate") or 0.0)
        if m.get("latency_s") is not None:
            latency.append(float(m["latency_s"]))
    r = int(st["result"][me])
    res = {"k": cfg["k"], "result": r, "score": {1: 1.0, 0: 0.5, -1: 0.0}[r],
           "claude_crowns": int(st["crowns"][me]), "policy_crowns": int(st["crowns"][opp_team]),
           "end_reason": st["end_reason"], "ticks": int(st["tick"]), "finished": bool(st["over"]),
           "decisions": len(decs), "skipped_idle": skipped, "plays": sum(d["action"] != 0 for d in decs),
           "illegal": sum(d["error_kind"] == "illegal" for d in decs),
           "parse_errors": sum(d["error_kind"] == "parse" for d in decs),
           "model_errors": sum(bool((d.get("meta") or {}).get("error")) for d in decs),
           "mean_latency_s": round(sum(latency) / len(latency), 3) if latency else None,
           "usage": usage, "cost_usd_estimate": round(cost, 4), "ended": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(res, open(os.path.join(archive, "result.json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------------------------ W&B
def wandb_start(args, info, state):
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "pufferroyale"), group=args.wandb_group,
                     name=f"{info['run_id']}-llm", job_type="llm", id=state.get("wandb_id"), resume="allow",
                     config={"run_id": info["run_id"], "every_steps": args.every_steps, "model": args.model,
                             "effort": args.effort, "decision_interval": args.decision_interval,
                             "claude_decks": list(CLAUDE_DECKS), "pool": list(POOL)})
    run.define_metric("global_step")
    run.define_metric("llm/*", step_metric="global_step")
    state["wandb_id"] = run.id                       # wandb 0.30 has no wandb.util.generate_id: let init pick it
    return run


def wandb_game(run, cfg, policy, res, archive, gif):
    import wandb
    table = wandb.Table(columns=["n", "tick", "opponent plays since", "Claude's reply", "action", "error"])
    for ln in open(os.path.join(archive, "decisions.jsonl")):
        d = json.loads(ln)
        opp = "; ".join(f"{p['card']} ({p['mx']},{p['my']})" for p in d["opp_since"])
        table.add_data(d["n"], d["t"], opp, d["reply"], d["decoded"] if d["action"] else "WAIT", d["error"] or "")
    cap = (f"game {cfg['k']}: Claude {cfg['claude_deck']} (team {cfg['claude_team']}) vs policy "
           f"{cfg['policy_deck']} @ {os.path.basename(policy['path'])}: Claude {res['claude_crowns']}-"
           f"{res['policy_crowns']} ({res['end_reason']})")
    run.log({"global_step": policy["global_step"], "llm/score": res["score"], "llm/result": res["result"],
             "llm/claude_crowns": res["claude_crowns"], "llm/policy_crowns": res["policy_crowns"],
             "llm/ticks": res["ticks"], "llm/decisions": res["decisions"], "llm/skipped_idle": res["skipped_idle"],
             "llm/plays": res["plays"], "llm/illegal": res["illegal"], "llm/parse_errors": res["parse_errors"],
             "llm/model_errors": res["model_errors"], "llm/mean_latency_s": res["mean_latency_s"] or 0.0,
             "llm/output_tokens": res["usage"]["output_tokens"], "llm/cost_usd_estimate": res["cost_usd_estimate"],
             "llm/mirror": int(cfg["mirror"]), "llm/claude_team": cfg["claude_team"], "llm/epoch": policy["epoch"],
             "llm/replay": wandb.Video(gif, caption=cap), "llm/decision_log": table})


# ------------------------------------------------------------------------------------ main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--relay-dir", required=True, help="node-local directory shared with hpc/llm_relay.py")
    ap.add_argument("--every-steps", type=int, default=250_000_000, help="one game per this many learner steps")
    ap.add_argument("--decision-interval", type=int, default=20, help="ticks between Claude's decisions (20 = 1 s)")
    ap.add_argument("--max-ticks", type=int, default=6000, help="cut games here (tests)")
    ap.add_argument("--max-games", type=int, default=0, help="stop after this many games (0 = all; tests)")
    ap.add_argument("--poll", type=float, default=120.0, help="seconds between looks for the next saved policy")
    ap.add_argument("--reply-timeout", type=float, default=14400.0, help="abort a game after this long without a reply")
    ap.add_argument("--attempts", type=int, default=3, help="tries per game (an aborted game is retried later)")
    ap.add_argument("--retry-after", type=float, default=1800.0, help="seconds before an aborted game is retried")
    ap.add_argument("--model", default="claude-opus-5-5", help="recorded in W&B (hpc/llm_relay.py calls the model)")
    ap.add_argument("--effort", default="low", help="recorded in W&B")
    ap.add_argument("--render-every", type=int, default=4, help="ticks per GIF frame (4 = 5 fps, real time)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default=None)
    a = ap.parse_args()

    def on_term(*_):
        raise Stop()
    signal.signal(signal.SIGTERM, on_term)
    sys.path.insert(0, os.path.join(ROOT, "hpc"))
    import llm_render
    from pufferroyale import llm

    run_dir = os.path.abspath(a.run_dir)
    out_dir = os.path.join(run_dir, "llm")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(a.relay_dir, exist_ok=True)
    _atomic_write(os.path.join(a.relay_dir, "RULES_PROMPT.txt"), llm.RULES_PROMPT)
    state_path = os.path.join(out_dir, "state.json")
    state = json.load(open(state_path)) if os.path.isfile(state_path) else {"games": {}}

    def save_state():
        _atomic_write(state_path, json.dumps(state, indent=1))

    while not os.path.isfile(os.path.join(run_dir, "config.json")):     # the training has not started yet
        time.sleep(a.poll)
    info = run_info(run_dir)
    ks = list(range(1, info["total"] // a.every_steps + 1))
    print(f"[llm_game] {run_dir}: {len(ks)} games (one per {a.every_steps} learner steps), "
          f"{sum(1 for k in ks if state['games'].get(str(k), {}).get('status') in ('done', 'skipped'))} already done",
          flush=True)
    wb = None
    played = 0
    try:
        while True:
            todo = []
            for k in ks:
                g = state["games"].get(str(k), {})
                if g.get("status") in ("done", "skipped"):
                    continue
                if g.get("status") == "aborted" and g.get("attempts", 0) >= a.attempts:
                    continue
                todo.append(k)
            if not todo or (a.max_games and played >= a.max_games):
                break
            k = todo[0]
            g = state["games"].get(str(k), {})
            if g.get("status") == "aborted" and time.time() < g.get("retry_at", 0):
                time.sleep(a.poll)
                continue
            target = k * a.every_steps
            policy = pick_policy(run_dir, info["batch"], target)
            if policy is None:
                if run_finished(run_dir, info["total"]):
                    state["games"][str(k)] = {"status": "skipped", "why": "no saved policy at or after the target"}
                    save_state()
                    continue
                time.sleep(a.poll)
                continue
            cfg = game_config(k)
            seed = zlib.crc32(f"{info['run_id']}:{k}".encode()) & 0x7FFFFFFF
            archive = os.path.join(out_dir, f"game_{k:03d}")
            live = os.path.join(a.relay_dir, f"g{k:03d}")
            shutil.rmtree(archive, ignore_errors=True)     # a game cut by a previous job end restarts from scratch
            drop_live(live)
            print(f"[llm_game] game {k}: target {target}, {os.path.basename(policy['path'])} "
                  f"(step {policy['global_step']}), {cfg}", flush=True)
            t0 = time.time()
            try:
                res = play_game(cfg, policy, live, archive, seed, a.decision_interval, a.max_ticks, a.reply_timeout)
            except GameAborted as e:
                n = g.get("attempts", 0) + 1
                state["games"][str(k)] = {"status": "aborted", "attempts": n, "why": str(e),
                                          "retry_at": time.time() + a.retry_after}
                save_state()
                print(f"[llm_game] game {k} aborted (attempt {n}): {e}", flush=True)
                continue
            gif = os.path.join(archive, f"game_{k:03d}.gif")
            llm_render.render(archive, gif, every=a.render_every,
                              title=f"game {k} | policy @ step {policy['global_step']:,}")
            res.update(status="done", policy=policy, config=cfg, seconds=round(time.time() - t0, 1),
                       attempts=g.get("attempts", 0) + 1)
            if a.wandb:
                try:
                    if wb is None:
                        wb = wandb_start(a, info, state)
                    wandb_game(wb, cfg, policy, res, archive, gif)
                    res["wandb"] = True
                except Exception as e:  # noqa: BLE001 -- W&B must never cost a played game
                    res["wandb"] = f"failed: {type(e).__name__}: {e}"
            state["games"][str(k)] = res
            save_state()
            drop_live(live)
            played += 1
            print(f"[llm_game] game {k} done in {res['seconds']:.0f} s: Claude {res['claude_crowns']}-"
                  f"{res['policy_crowns']} ({res['end_reason']}), {res['decisions']} decisions, "
                  f"est. ${res['cost_usd_estimate']:.2f}", flush=True)
    except Stop:
        print("[llm_game] SIGTERM: stopping (finished games are recorded)", flush=True)
    finally:
        if wb is not None:
            wb.finish()
    if all(state["games"].get(str(k), {}).get("status") in ("done", "skipped")
           or state["games"].get(str(k), {}).get("attempts", 0) >= a.attempts for k in ks):
        _atomic_write(os.path.join(out_dir, "DONE"), time.strftime("%Y-%m-%dT%H:%M:%S\n"))


if __name__ == "__main__":
    main()
