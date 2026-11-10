#!/usr/bin/env python3
"""Stage-4 collapse diagnostics: gradient probe (2026-10-16; see hpc/collapse_diag.py). Which part of the PPO
update pushes the placement entropy down, and is it stronger on games against frozen snapshots than on
self-play games?

The run's own trainer is built exactly as league_train builds it (its saved args and ini overrides; on the CPU,
without the CUDA-only fast paths), the learner is loaded from a snapshot (with the run's card floor at that
step), and the pool holds the 64 snapshots before it, as the run's pool did (stats unseen, so PFSP is uniform).
--mode picks the opponents: self (self_play_frac 1, the self-play runs' data), snap (frozen snapshots only) or
mixed (the run's own split). For each collected batch (trainer.evaluate(): num_envs x bptt_horizon steps) it
computes, on the whole batch, the gradients of the loss terms of MMDPuffeRL.train() at ratio 1:

  pg    -mean(c * A_norm * log pi(a))   (A = the trainer's compute_puff_advantage; c = the expected
                                          prioritised-replay weight of the segment, prio_alpha / prio_beta0)
  ent   -ent_coef * mean(H_joint)
  mmd   mmd_coef * KL(pi || ref), ref = the snapshot --ref-lag snapshots earlier (a proxy for the magnet,
        which lags at most mmd_ref_interval epochs)
  value vf_coef * 0.5 * mean((V - R)^2)

and their first-order effect on the UNWEIGHTED placement entropy H_u = mean over states and legal slots of
H(p(. | slot)) (computed on the batch's own states): dH_T = -grad(H_u) . grad(L_T) (the change of H_u per unit
step of plain gradient descent on term T), with the cosine between -grad(L_T) and grad(H_u).

    python hpc/collapse_probe.py --run S4_C_s0 --at 1.55 --mode snap --epochs 8 --out OUT.json
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "hpc")):
    sys.path.insert(0, p)


def flat_grad(params, loss, retain):
    import torch
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def placement_entropy(policy, obs):
    """H_u: mean over rows and legal slots of H(p(. | slot)) (differentiable), and the number of terms."""
    import torch
    parts = policy.conditional_parts(obs)
    pos_logp, mask = parts["pos_logp"], parts["mask"]
    N = obs.shape[0]
    B = pos_logp.shape[-1]
    legal = mask[:, 1:].reshape(N, 4, B)
    slot_ok = legal.any(-1)
    p = pos_logp.exp()
    h = -torch.where(legal, p * pos_logp, torch.zeros_like(p)).sum(-1)          # (N, 4)
    n = slot_ok.sum()
    return torch.where(slot_ok, h, torch.zeros_like(h)).sum() / n.clamp(min=1), int(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--at", type=float, required=True, help="learner snapshot, billions of steps")
    ap.add_argument("--mode", choices=["self", "snap", "mixed"], required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--ref-lag", type=int, default=1, help="MMD reference = this many snapshots earlier")
    ap.add_argument("--coef-from", default=None, help="take mmd_coef / ent_coef from this run instead (A runs)")
    ap.add_argument("--env-threads", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--h-rows", type=int, default=4096, help="batch rows used for grad(H_u)")
    ap.add_argument("--num-envs", type=int, default=0, help="override (smoke tests): batch = num_envs x 64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", os.path.join(ROOT, "experiments")))
    ap.add_argument("--out", required=True)
    o = ap.parse_args()

    import torch
    import league_train as LT
    from collapse_diag import snapshot
    from pufferroyale.league import OpponentPool, load_weights
    from pufferroyale.trainer import compute_puff_advantage, masked_kl, obs_mask
    from pufferroyale.torch import grid_of_actions

    run_dir = os.path.join(o.data_dir, "league", o.run)
    st = json.load(open(os.path.join(run_dir, "league_state.json")))
    a = dict(st["args"])
    if o.coef_from:
        a2 = json.load(open(os.path.join(o.data_dir, "league", o.coef_from, "league_state.json")))["args"]
        a["mmd_coef"], a["mmd_ref_interval"] = a2["mmd_coef"], a2["mmd_ref_interval"]
    a.update(device="cpu", fast_opponents=False, cuda_graphs=False, env_threads=o.env_threads)
    if o.num_envs:
        a.update(num_envs=o.num_envs, batch_size=o.num_envs * 64, minibatch_size=min(1024, o.num_envs * 64))
    ini = LT.load_ini(list(st.get("ini_overrides", [])))
    name, lpath, e_l, gs = snapshot(o.data_dir, o.run, o.at)
    interval = int(a["snapshot_interval"])
    pool_eps = [e_l - interval * k for k in range(64, 0, -1) if e_l - interval * k > 0]
    sp = {"self": 1.0, "snap": 0.0, "mixed": float(a["self_play_frac"])}[o.mode]
    pool = OpponentPool(anchors=(), max_snapshots=64, pfsp=a["pfsp"], pfsp_eps=float(a["pfsp_eps"]),
                        self_play_frac=sp, anchor_frac=0.0, seed=o.seed)
    for e in pool_eps:
        pool.add_snapshot(os.path.join(run_dir, f"snap_{e}.pt"))
    tmp = tempfile.mkdtemp(prefix="probe_", dir=os.environ.get("TMPDIR"))
    torch.manual_seed(o.seed)
    lv, policy, trainer, cfg = LT.build(a, ini, pool, tmp, "probe", reset_seed=o.seed, global_step=gs)
    load_weights(policy, lpath)
    ref_e = e_l - interval * o.ref_lag
    coef = float(a["mmd_coef"])
    ref = trainer.ref_policy
    if ref is None:                                   # an A run: build a reference just for the probe
        ref = trainer._frozen_copy()
    load_weights(ref, os.path.join(run_dir, f"snap_{ref_e}.pt"))
    if float(a.get("card_floor") or 0) > 0:
        eps = LT.card_floor_at(a, gs)
        for p in (policy, ref):
            getattr(p, "policy", p).set_card_floor(eps)
    else:
        eps = 0.0
    tc = trainer.config
    ent_coef, vf_coef = float(tc["ent_coef"]), float(tc["vf_coef"])
    mmd_coef = coef if coef > 0 else 0.05
    alpha, b0 = float(tc["prio_alpha"]), float(tc["prio_beta0"])
    total_epochs = int(a["total_timesteps"]) // int(cfg["batch_size"])
    beta = b0 + (1 - b0) * alpha * e_l / total_epochs
    params = [p for p in policy.parameters() if p.requires_grad]
    out = {"run": o.run, "at": name, "learner": lpath, "global_step": gs, "mode": o.mode, "self_play_frac": sp,
           "ref": f"snap_{ref_e}.pt", "card_floor": eps, "ent_coef": ent_coef, "vf_coef": vf_coef,
           "mmd_coef": mmd_coef, "mmd_coef_in_run": coef, "prio": [alpha, b0, beta], "epochs": []}
    print(f"[probe] {name} mode {o.mode} floor {eps:.4f} pool {len(pool.snapshots)} snapshots, ref snap_{ref_e}",
          flush=True)
    t_all = time.time()
    try:
        for ep in range(o.epochs):
            t0 = time.time()
            policy.eval()
            trainer.evaluate()
            res = lv.drain_results()
            obs = trainer.observations.reshape(-1, trainer.observations.shape[-1]).float()
            acts = trainer.actions.reshape(-1).long()
            S, T = trainer.values.shape
            with torch.no_grad():
                adv = torch.zeros_like(trainer.values)
                adv = compute_puff_advantage(trainer.values, trainer.rewards, trainer.terminals,
                                             torch.ones_like(trainer.values), adv, tc["gamma"], tc["gae_lambda"],
                                             tc["vtrace_rho_clip"], tc["vtrace_c_clip"])
                ret = (adv + trainer.values).reshape(-1)
                seg = adv.abs().sum(1)
                w = torch.nan_to_num(seg ** alpha, 0, 0, 0)
                probs = (w + 1e-6) / (w.sum() + 1e-6)
                c = (S * probs) * (S * probs) ** (-beta)            # expected sampling frequency x importance weight
                c = (c / c.mean()).repeat_interleave(T)
                advf = adv.reshape(-1)
                adv_n = c * (advf - advf.mean()) / (advf.std() + 1e-8)
            policy.train()
            N = obs.shape[0]
            G = {k: torch.zeros(sum(p.numel() for p in params)) for k in ("pg", "ent", "mmd", "value")}
            vals = {k: 0.0 for k in ("pg", "ent", "mmd", "value", "H_joint", "kl")}
            for i in range(0, N, o.chunk):
                x, a_ = obs[i:i + o.chunk], acts[i:i + o.chunk]
                logits, v = policy(x, dict(action=a_, lstm_h=None, lstm_c=None))
                logp_all = torch.log_softmax(logits.float(), dim=-1)
                logp = logp_all.gather(1, a_[:, None]).squeeze(1)
                pr = logp_all.exp()
                H = -torch.where(pr > 0, pr * logp_all, torch.zeros_like(pr)).sum(-1)
                with torch.no_grad():
                    rl, _ = ref(x, dict(action=a_, lstm_h=None, lstm_c=None))
                kl = masked_kl(logits, rl, obs_mask(x, grid_of_actions(logits.shape[-1])))
                f = x.shape[0] / N
                terms = {"pg": -(adv_n[i:i + o.chunk] * logp).sum() / N,
                         "ent": -ent_coef * H.sum() / N,
                         "mmd": mmd_coef * kl * f,
                         "value": vf_coef * 0.5 * ((v.reshape(-1) - ret[i:i + o.chunk]) ** 2).sum() / N}
                for j, (k, L) in enumerate(terms.items()):
                    G[k] += flat_grad(params, L, retain=j < len(terms) - 1)
                    vals[k] += float(L)
                vals["H_joint"] += float(H.sum()) / N
                vals["kl"] += float(kl) * f
            hx = obs[torch.randperm(N, generator=torch.Generator().manual_seed(ep))[:o.h_rows]]
            Hu, nterms = placement_entropy(policy, hx)
            gH = flat_grad(params, Hu, retain=False)
            row = {"epoch": ep, "H_u": float(Hu), "H_u_terms": nterms, "values": vals,
                   "episodes": len(res), "score": (sum((r + 1) / 2 for _, r, _ in res) / len(res)) if res else None,
                   "adv_abs_mean": float(advf.abs().mean()), "seconds": round(time.time() - t0, 1), "terms": {}}
            gsum = sum(G.values())
            for k, g in list(G.items()) + [("total", gsum)]:
                d = float(-(gH * g).sum())
                cos = d / (float(gH.norm()) * float(g.norm()) + 1e-30)
                row["terms"][k] = {"dH": d, "cos": cos, "norm": float(g.norm())}
            out["epochs"].append(row)
            print(f"[probe] epoch {ep}: H_u {row['H_u']:.3f} score {row['score']} "
                  + " ".join(f"{k} dH {v['dH']:+.2e} cos {v['cos']:+.3f}" for k, v in row["terms"].items()),
                  flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(o.out)), exist_ok=True)
            json.dump(out, open(o.out, "w"), indent=1)
    finally:
        lv.close()
    ks = list(out["epochs"][0]["terms"])
    out["mean"] = {k: {m: float(np.mean([r["terms"][k][m] for r in out["epochs"]])) for m in ("dH", "cos", "norm")}
                   for k in ks}
    out["sem"] = {k: float(np.std([r["terms"][k]["dH"] for r in out["epochs"]], ddof=1) / np.sqrt(len(out["epochs"])))
                  if len(out["epochs"]) > 1 else None for k in ks}
    out["seconds"] = round(time.time() - t_all, 1)
    json.dump(out, open(o.out, "w"), indent=1)
    print(json.dumps({k: [round(v["dH"], 8), round(v["cos"], 4)] for k, v in out["mean"].items()}))


if __name__ == "__main__":
    main()
