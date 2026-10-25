"""Baseline RL algorithms on the unchanged DisasterReliefEnv.

All methods use the same environment, reward, 4000-episode budget and
held-out evaluation as MAPPO. Every warehouse is controlled by one shared
learner acting on its own local view and receiving the team reward
(independent learners with parameter sharing), so only the learning
algorithm differs between rows of the comparison.

  qlearning  tabular Q-learning, epsilon-greedy, no action masking
  sarsa      tabular SARSA, epsilon-greedy, no action masking
  dqn        Deep Q-Network (replay + target net), no action masking
  ppo        independent PPO, local critic, no action masking
  mppo       independent PPO, local critic, WITH action masking
(MAPPO = masked actor + centralized critic is train_mappo_v2.py.)

Without masking an infeasible action is executed by the environment as a
no-op (no vehicle is used), exactly as in disaster_env.py.

Usage: python compare_algos.py --algo dqn --seed 42 --episodes 4000
"""
import argparse
import csv
import json
import os
import pickle
import random
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from disaster_env import DisasterReliefEnv
from train_mappo_v2 import RunningNorm, mlp

N_EVAL, EVAL_SEED0 = 300, 900_000


# --------------------------------------------------------------- utilities
def eps_at(ep, n_eps, start=1.0, end=0.05, frac=0.6):
    return max(end, start - (start - end) * ep / (frac * n_eps))


def tab_state(env, w):
    """Discrete state for tabular methods: own vehicles left,
    zone priorities and own road status."""
    return (int(env.vehicles[w]), tuple(int(p) for p in env.priorities),
            tuple(int(r) for r in env.roads[w]))


def evaluate(policy):
    env = DisasterReliefEnv(seed=EVAL_SEED0)
    rows = []
    for i in range(N_EVAL):
        obs, state, masks = env.reset(seed=EVAL_SEED0 + i)
        done = False
        tot = energy = served = 0.0
        steps = 0
        while not done:
            a = policy(env, obs, masks)
            obs, state, masks, r, done, m = env.step(a)
            tot += r; energy += m["energy_proxy"]; served += m["satisfied_demand"]; steps += 1
        rows.append([tot, m["unmet_critical_demand"], energy, served,
                     m["task_completion_rate"], steps])
    rows = np.array(rows)
    keys = ["reward", "crit_unmet", "energy", "served", "completion", "length"]
    return {k: float(rows[:, j].mean()) for j, k in enumerate(keys)}


# --------------------------------------------------------------- tabular
def train_tabular(args, env, log, sarsa):
    rng = np.random.default_rng(args.seed)
    Q = defaultdict(lambda: np.zeros(env.n_actions))
    alpha, gamma = 0.1, 0.99

    def choose(s, eps):
        if rng.random() < eps:
            return int(rng.integers(env.n_actions))
        q = Q[s]
        return int(rng.choice(np.flatnonzero(q == q.max())))

    for ep in range(1, args.episodes + 1):
        env.reset(seed=args.seed * 100_000 + ep)
        eps = eps_at(ep, args.episodes)
        states = [tab_state(env, w) for w in range(env.n_warehouses)]
        acts = [choose(s, eps) for s in states]
        done, ep_r, ep_e = False, 0.0, 0.0
        while not done:
            _, _, _, r, done, m = env.step(np.array(acts))
            ep_r += r; ep_e += m["energy_proxy"]
            nstates = [tab_state(env, w) for w in range(env.n_warehouses)]
            nacts = [choose(s, eps) for s in nstates]
            for w in range(env.n_warehouses):
                s, a, s2 = states[w], acts[w], nstates[w]
                if done:
                    target = r
                elif sarsa:
                    target = r + gamma * Q[s2][nacts[w]]
                else:
                    target = r + gamma * Q[s2].max()
                Q[s][a] += alpha * (target - Q[s][a])
            states, acts = nstates, nacts
        log.append([ep, ep_r, m["unmet_critical_demand"], ep_e, m["task_completion_rate"]])

    Qd = dict(Q)
    eval_rng = np.random.default_rng(0)
    coverage = {"seen": 0, "unseen": 0}

    def policy(env, obs, masks):
        # States never visited in training fall back to a uniform random
        # action (the same untrained behaviour as epsilon = 1).
        out = []
        for w in range(env.n_warehouses):
            q = Qd.get(tab_state(env, w))
            if q is None:
                coverage["unseen"] += 1
                out.append(int(eval_rng.integers(env.n_actions)))
            else:
                coverage["seen"] += 1
                out.append(int(np.argmax(q)))
        return np.array(out)
    return {"greedy": policy}, {"q_table": Qd, "n_states": len(Qd), "coverage": coverage}


# --------------------------------------------------------------- DQN
def train_dqn(args, env, log):
    rng = np.random.default_rng(args.seed)
    q = mlp(env.local_obs_dim, env.n_actions)
    tgt = mlp(env.local_obs_dim, env.n_actions)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=5e-4)
    buf = deque(maxlen=50_000)
    gamma, batch, warmup, tgt_every, r_scale = 0.99, 64, 500, 500, 0.1
    step = 0
    for ep in range(1, args.episodes + 1):
        obs, _, _ = env.reset(seed=args.seed * 100_000 + ep)
        eps = eps_at(ep, args.episodes)
        done, ep_r, ep_e = False, 0.0, 0.0
        while not done:
            with torch.no_grad():
                greedy = q(torch.as_tensor(obs)).argmax(-1).numpy()
            acts = np.where(rng.random(env.n_warehouses) < eps,
                            rng.integers(env.n_actions, size=env.n_warehouses), greedy)
            nobs, _, _, r, done, m = env.step(acts)
            ep_r += r; ep_e += m["energy_proxy"]
            for w in range(env.n_warehouses):
                buf.append((obs[w], acts[w], r * r_scale, nobs[w], float(done)))
            obs = nobs
            step += 1
            if len(buf) >= warmup:
                idx = rng.integers(len(buf), size=batch)
                o, a, rr, o2, d = map(np.array, zip(*[buf[i] for i in idx]))
                o = torch.as_tensor(o); o2 = torch.as_tensor(o2)
                a = torch.as_tensor(a, dtype=torch.long)
                rr = torch.as_tensor(rr, dtype=torch.float32)
                d = torch.as_tensor(d, dtype=torch.float32)
                with torch.no_grad():
                    y = rr + gamma * (1 - d) * tgt(o2).max(-1).values
                qa = q(o).gather(1, a[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(qa, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
            if step % tgt_every == 0:
                tgt.load_state_dict(q.state_dict())
        log.append([ep, ep_r, m["unmet_critical_demand"], ep_e, m["task_completion_rate"]])

    @torch.no_grad()
    def policy(env, obs, masks):
        return q(torch.as_tensor(obs)).argmax(-1).numpy()
    return {"greedy": policy}, {"state_dict": q.state_dict()}


# --------------------------------------------------------------- independent PPO
class IPPO:
    """PPO with a local (per-agent) critic, using the corrected update
    (separate optimizers, value normalization, multi-episode batches)."""

    def __init__(self, env, use_mask):
        self.use_mask = use_mask
        self.actor = mlp(env.local_obs_dim, env.n_actions)
        self.critic = mlp(env.local_obs_dim, 1)
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.vn = RunningNorm()
        self.gamma, self.lam, self.clip, self.ent = 0.99, 0.95, 0.2, 0.01

    def mask(self, m):
        m = torch.as_tensor(m, dtype=torch.bool)
        return m if self.use_mask else torch.ones_like(m)

    def dist(self, o, m):
        return Categorical(logits=self.actor(o).masked_fill(~m, -1e9))

    @torch.no_grad()
    def act(self, obs, masks, greedy=False):
        o = torch.as_tensor(obs)
        d = self.dist(o, self.mask(masks))
        a = d.probs.argmax(-1) if greedy else d.sample()
        v = self.vn.denorm(self.critic(o).squeeze(-1).numpy())
        return a.numpy(), d.log_prob(a).numpy(), v

    def rollout(self, env, seed):
        obs, _, masks = env.reset(seed=seed)
        B = {k: [] for k in ["obs", "masks", "act", "logp", "val", "rew"]}
        done, ep_r, ep_e = False, 0.0, 0.0
        while not done:
            a, lp, v = self.act(obs, masks)
            B["obs"].append(obs); B["masks"].append(self.mask(masks).numpy())
            B["act"].append(a); B["logp"].append(lp); B["val"].append(v)
            obs, _, masks, r, done, m = env.step(a)
            B["rew"].append(r); ep_r += r; ep_e += m["energy_proxy"]
        rew = np.array(B["rew"], dtype=np.float32)
        val = np.vstack(B["val"] + [np.zeros(env.n_warehouses)]).astype(np.float32)
        adv = np.zeros((len(rew), env.n_warehouses), dtype=np.float32)
        gae = np.zeros(env.n_warehouses, dtype=np.float32)
        for t in reversed(range(len(rew))):
            delta = rew[t] + self.gamma * val[t + 1] - val[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        out = {k: np.array(B[k]) for k in ["obs", "masks", "act", "logp"]}
        out["adv"], out["ret"] = adv, adv + val[:-1]
        return out, ep_r, m, ep_e

    def update(self, batches, epochs=8, bs=256):
        cat = lambda k: np.concatenate([b[k] for b in batches]).reshape(
            -1, *batches[0][k].shape[2:])
        o = torch.as_tensor(cat("obs")); m = torch.as_tensor(cat("masks"))
        a = torch.as_tensor(cat("act"), dtype=torch.long)
        old = torch.as_tensor(cat("logp"))
        adv = torch.as_tensor(cat("adv")); ret = cat("ret")
        self.vn.update(ret)
        ret_n = torch.as_tensor((ret - self.vn.mean) / self.vn.std, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        n = len(a)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                i = perm[s:s + bs]
                d = self.dist(o[i], m[i])
                ratio = torch.exp(d.log_prob(a[i]) - old[i])
                loss = -torch.minimum(ratio * adv[i], torch.clamp(
                    ratio, 1 - self.clip, 1 + self.clip) * adv[i]).mean() - self.ent * d.entropy().mean()
                self.aopt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5); self.aopt.step()
                closs = ((self.critic(o[i]).squeeze(-1) - ret_n[i]) ** 2).mean()
                self.copt.zero_grad(); closs.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5); self.copt.step()


def train_ippo(args, env, log, use_mask):
    ag = IPPO(env, use_mask)
    ep = 0
    while ep < args.episodes:
        batches = []
        for _ in range(min(8, args.episodes - ep)):
            ep += 1
            b, r, m, e = ag.rollout(env, args.seed * 100_000 + ep)
            batches.append(b)
            log.append([ep, r, m["unmet_critical_demand"], e, m["task_completion_rate"]])
        ag.update(batches)
    return ({"stochastic": lambda env, o, m: ag.act(o, m)[0],
             "greedy": lambda env, o, m: ag.act(o, m, greedy=True)[0]},
            {"actor": ag.actor.state_dict(), "critic": ag.critic.state_dict()})


# --------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["qlearning", "sarsa", "dqn", "ppo", "mppo"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or f"cmp_{args.algo}_s{args.seed}"
    os.makedirs(out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = DisasterReliefEnv(seed=args.seed)
    log = []

    if args.algo in ("qlearning", "sarsa"):
        policies, model = train_tabular(args, env, log, sarsa=args.algo == "sarsa")
        with open(os.path.join(out, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    elif args.algo == "dqn":
        policies, model = train_dqn(args, env, log)
        torch.save(model, os.path.join(out, "model.pt"))
    else:
        policies, model = train_ippo(args, env, log, use_mask=args.algo == "mppo")
        torch.save(model, os.path.join(out, "model.pt"))

    with open(os.path.join(out, "training_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "unmet_critical_demand",
                    "energy_proxy_episode", "task_completion_rate"])
        w.writerows(log)

    torch.manual_seed(0)
    res = {mode: evaluate(pol) for mode, pol in policies.items()}
    if "n_states" in model:
        res["n_states"] = model["n_states"]
        c = model["coverage"]
        res["eval_unseen_frac"] = c["unseen"] / max(1, c["seen"] + c["unseen"])
    json.dump(res, open(os.path.join(out, "eval.json"), "w"), indent=1)
    last = np.array(log)[-500:, 1].mean()
    print(f"{args.algo} seed {args.seed}: last-500 train reward {last:.1f} | eval {json.dumps({k: round(v['reward'], 1) for k, v in res.items() if isinstance(v, dict)})}", flush=True)


if __name__ == "__main__":
    main()
