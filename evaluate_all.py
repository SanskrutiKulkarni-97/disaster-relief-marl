"""Evaluate baselines and trained MAPPO actors on identical held-out seeds,
and measure the actor/critic gradient imbalance of the v1 update rule."""
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

from disaster_env import DisasterReliefEnv
from train_mappo import mlp, MAPPO as MAPPOv1, rollout as rollout_v1

N_EVAL = 300
SEED0 = 900_000


def run_policy(policy, n=N_EVAL):
    env = DisasterReliefEnv(seed=SEED0)
    rows = []
    for i in range(n):
        obs, state, masks = env.reset(seed=SEED0 + i)
        done = False
        tot = energy = served = 0.0
        disp = steps = 0
        while not done:
            a = policy(env, obs, masks)
            obs, state, masks, r, done, m = env.step(a)
            tot += r
            energy += m["energy_proxy"]
            served += m["satisfied_demand"]
            disp += m["dispatches"]
            steps += 1
        rows.append([tot, m["unmet_critical_demand"], energy, served, disp,
                     m["task_completion_rate"], steps])
    rows = np.array(rows)
    names = ["reward", "crit_unmet", "energy", "served", "dispatches",
             "completion", "length"]
    return {k: float(rows[:, j].mean()) for j, k in enumerate(names)}, rows[:, 0], rows[:, 1]


# ---------------- baselines ----------------
_rng = np.random.default_rng(0)


def noop(env, obs, m):
    return np.full(env.n_warehouses, env.noop_action)


def random_pol(env, obs, m):
    return np.array([_rng.choice(np.flatnonzero(m[w])) for w in range(env.n_warehouses)])


def greedy(env, obs, m):
    acts, unmet = [], env.unmet().copy()
    for w in range(env.n_warehouses):
        best, bs = None, -1e18
        for a in np.flatnonzero(m[w]):
            d = env.decode(a)
            if d is None:
                continue
            c, z, q = d
            if unmet[z, c] <= 0:
                continue
            wgt = {1: 4, 2: 2, 3: 1}[int(env.priorities[z])]
            sc = wgt * min(q, unmet[z, c]) * 100 - q * (1 + 0.25 * (z + 1))
            if sc > bs:
                bs, best = sc, a
        if best is None:
            acts.append(env.noop_action)
        else:
            c, z, q = env.decode(best)
            unmet[z, c] -= q
            acts.append(best)
    return np.array(acts)


def actor_policy(actor, greedy_act):
    gen = torch.Generator().manual_seed(0)

    @torch.no_grad()
    def pol(env, obs, m):
        logits = actor(torch.as_tensor(obs, dtype=torch.float32))
        logits = logits.masked_fill(~torch.as_tensor(m), -1e9)
        if greedy_act:
            return logits.argmax(-1).numpy()
        p = torch.softmax(logits, -1)
        return torch.multinomial(p, 1, generator=gen).squeeze(-1).numpy()
    return pol


results = {}
for name, pol in [("noop", noop), ("random", random_pol), ("greedy_heuristic", greedy)]:
    results[name], _, _ = run_policy(pol)
    print(name, results[name])

env = DisasterReliefEnv()
for version in ["v1", "v2"]:
    for mode in ["stochastic", "argmax"]:
        per_seed = []
        for path in sorted(glob.glob(f"runs_{version}_s*/mappo_model.pt")):
            actor = mlp(env.local_obs_dim, env.n_actions)
            actor.load_state_dict(torch.load(path, map_location="cpu")["actor"])
            actor.eval()
            r, _, _ = run_policy(actor_policy(actor, mode == "argmax"))
            r["run"] = path
            per_seed.append(r)
        keys = [k for k in per_seed[0] if k != "run"]
        agg = {k: float(np.mean([p[k] for p in per_seed])) for k in keys}
        agg.update({k + "_std": float(np.std([p[k] for p in per_seed])) for k in keys})
        agg["n_seeds"] = len(per_seed)
        agg["per_seed"] = per_seed
        results[f"mappo_{version}_{mode}"] = agg
        print(version, mode, {k: round(agg[k], 1) for k in ["reward", "reward_std", "crit_unmet", "crit_unmet_std"]})

# ---------------- gradient diagnostic for the v1 update ----------------
torch.manual_seed(0)
genv = DisasterReliefEnv(seed=42)
ag = MAPPOv1(genv.local_obs_dim, genv.global_state_dim, genv.n_actions)
actor_pre, critic_pre, actor_post, closs = [], [], [], []
for ep in range(1, 51):
    b, _, _ = rollout_v1(genv, ag, 42 + ep)
    obs = torch.tensor(b["obs"]); T, n = obs.shape[:2]
    f_obs = obs.reshape(T * n, -1)
    f_m = torch.tensor(b["masks"]).reshape(T * n, -1)
    f_a = torch.tensor(b["actions"], dtype=torch.long).reshape(-1)
    f_old = torch.tensor(b["logp"], dtype=torch.float32).reshape(-1)
    adv = torch.tensor(b["advantages"])[:, None].expand(T, n).reshape(-1)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    d = ag.dist(f_obs, f_m)
    ratio = torch.exp(d.log_prob(f_a) - f_old)
    al = -torch.minimum(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
    v = ag.critic(torch.tensor(b["states"])).squeeze(-1)
    cl = ((v - torch.tensor(b["returns"])) ** 2).mean()
    ag.opt.zero_grad()
    (al + 0.5 * cl - 0.01 * d.entropy().mean()).backward()
    ga = torch.sqrt(sum((p.grad ** 2).sum() for p in ag.actor.parameters())).item()
    gc = torch.sqrt(sum((p.grad ** 2).sum() for p in ag.critic.parameters())).item()
    tot = (ga ** 2 + gc ** 2) ** 0.5
    actor_pre.append(ga); critic_pre.append(gc)
    actor_post.append(ga * min(1.0, 0.5 / tot)); closs.append(cl.item())
    ag.update(b)  # continue training with the original rule
results["grad_diag_v1"] = {
    "actor_grad_pre": float(np.median(actor_pre)),
    "critic_grad_pre": float(np.median(critic_pre)),
    "ratio": float(np.median(np.array(critic_pre) / np.array(actor_pre))),
    "actor_grad_post_clip": float(np.median(actor_post)),
    "shrink_factor": float(np.median(np.array(actor_pre) / np.array(actor_post))),
    "critic_loss": float(np.median(closs)),
}
print("grad diag", results["grad_diag_v1"])

# demand / capacity numbers
cap_env = DisasterReliefEnv()
init = []
for i in range(N_EVAL):
    cap_env.reset(seed=SEED0 + i)
    init.append(cap_env.demand.sum())
results["capacity"] = {"initial_demand_mean": float(np.mean(init)),
                       "max_deliverable": float(cap_env.vehicles.sum() * max(cap_env.quantities))}
print(results["capacity"])


# ---------------- comparison algorithms (outputs of compare_algos.py) ----------------
def convergence(prefix, threshold=-50.0, window=100):
    """Episodes until the 100-episode moving-average training reward first
    reaches the threshold, one entry per seed (None = never reached)."""
    import csv
    eps = []
    for path in sorted(glob.glob(f"{prefix}_s*/training_metrics.csv")):
        with open(path) as f:
            v = np.array([float(r["reward"]) for r in csv.DictReader(f)])
        if len(v) < window:
            eps.append(None)
            continue
        ma = np.convolve(v, np.ones(window) / window, mode="valid")
        idx = np.flatnonzero(ma >= threshold)
        eps.append(int(idx[0] + window) if len(idx) else None)
    return eps


comparison = {}
for algo in ["qlearning", "sarsa", "dqn", "ppo", "mppo"]:
    evals = [json.load(open(p)) for p in sorted(glob.glob(f"cmp_{algo}_s*/eval.json"))]
    if not evals:
        print(f"no compare_algos results for {algo}, skipping")
        continue
    for mode in ["greedy", "stochastic"]:
        if mode not in evals[0]:
            continue
        keys = list(evals[0][mode])
        d = {k: float(np.mean([e[mode][k] for e in evals])) for k in keys}
        d.update({k + "_std": float(np.std([e[mode][k] for e in evals])) for k in keys})
        comparison[f"{algo}_{mode}"] = d
    if "n_states" in evals[0]:
        comparison[f"{algo}_states"] = float(np.mean([e["n_states"] for e in evals]))
        comparison[f"{algo}_unseen"] = float(np.mean([e["eval_unseen_frac"] for e in evals]))
    comparison[f"{algo}_conv"] = convergence(f"cmp_{algo}")
comparison["mappo_v2_conv"] = convergence("runs_v2")
comparison["mappo_v1_conv"] = convergence("runs_v1")
results["comparison"] = comparison
for k, v in comparison.items():
    if isinstance(v, dict):
        print(f"{k:18s} reward {v['reward']:8.1f} +/- {v['reward_std']:5.1f} | critical unmet {v['crit_unmet']:6.1f}")

with open("final_results.json", "w") as f:
    json.dump(results, f, indent=1)
print("wrote final_results.json")
