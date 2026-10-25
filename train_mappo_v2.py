"""MAPPO trainer, v2.

The environment and its reward are unchanged (controlled comparison).
Changes are confined to the learning algorithm:

  1. Separate actor / critic optimizers with separate gradient clipping.
     In v1 one clip_grad_norm_(0.5) covered both nets, and the critic's
     MSE (raw returns of roughly -100) produces gradients orders of
     magnitude larger than the actor's, so clipping shrank the actor's
     update to almost nothing.
  2. Value normalization (running mean/std of returns) so the critic
     regresses on a unit-scale target.
  3. Several episodes per PPO update (--episodes-per-update, default 8)
     instead of one, so advantage normalization and the clipped
     objective see a real batch instead of ~4 time steps.
  4. Critic trained on every collected state each minibatch pass,
     not on random draws with replacement.

Logged metrics stay in raw reward units, comparable with v1.
"""
import os
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disaster_env import DisasterReliefEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(inp, out):
    return nn.Sequential(
        nn.Linear(inp, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, out),
    )


class RunningNorm:
    """Running mean/std for value-target normalization (PopArt-lite)."""

    def __init__(self, eps=1e-4):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        b_mean, b_var, b_n = x.mean(), x.var(), x.size
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var, self.count = m2 / tot, tot

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


class MAPPO:
    """Shared decentralized actor with a centralized critic (CTDE)."""

    def __init__(self, local_dim, global_dim, n_actions,
                 actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.01):
        self.actor = mlp(local_dim, n_actions).to(DEVICE)
        self.critic = mlp(global_dim, 1).to(DEVICE)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.lam, self.clip, self.ent_coef = gamma, lam, clip, ent_coef
        self.value_norm = RunningNorm()

    def dist(self, obs, masks):
        logits = self.actor(obs).masked_fill(~masks, -1e9)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs, state, masks, greedy=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        masks_t = torch.as_tensor(masks, dtype=torch.bool, device=DEVICE)
        d = self.dist(obs_t, masks_t)
        actions = d.probs.argmax(-1) if greedy else d.sample()
        log_probs = d.log_prob(actions)
        v_norm = self.critic(
            torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        ).item()
        value = self.value_norm.denorm(v_norm)  # raw-scale value for GAE
        return actions.cpu().numpy(), log_probs.cpu().numpy(), float(value)

    def update(self, batch, epochs=8, batch_size=256):
        t = lambda k, dt=torch.float32: torch.as_tensor(batch[k], dtype=dt, device=DEVICE)
        obs, masks = t("obs"), t("masks", torch.bool)
        actions, old_lp = t("actions", torch.long), t("logp")
        states, returns, adv = t("states"), t("returns"), t("advantages")

        self.value_norm.update(batch["returns"])
        returns_n = (returns - self.value_norm.mean) / self.value_norm.std

        T, n_agents = obs.shape[:2]
        f_obs = obs.reshape(T * n_agents, -1)
        f_masks = masks.reshape(T * n_agents, -1)
        f_act = actions.reshape(-1)
        f_old = old_lp.reshape(-1)
        f_adv = adv[:, None].expand(T, n_agents).reshape(-1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std(unbiased=False) + 1e-8)

        stats = {"actor_grad": [], "critic_grad": [], "critic_loss": []}
        for _ in range(epochs):
            # Actor
            perm = torch.randperm(len(f_act), device=DEVICE)
            for s in range(0, len(f_act), batch_size):
                idx = perm[s:s + batch_size]
                d = self.dist(f_obs[idx], f_masks[idx])
                ratio = torch.exp(d.log_prob(f_act[idx]) - f_old[idx])
                surr = torch.minimum(
                    ratio * f_adv[idx],
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * f_adv[idx],
                )
                loss = -surr.mean() - self.ent_coef * d.entropy().mean()
                self.actor_opt.zero_grad()
                loss.backward()
                stats["actor_grad"].append(
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5).item())
                self.actor_opt.step()
            # Critic (separate optimizer + clipping, unit-scale targets)
            perm = torch.randperm(T, device=DEVICE)
            for s in range(0, T, batch_size):
                idx = perm[s:s + batch_size]
                v = self.critic(states[idx]).squeeze(-1)
                closs = ((v - returns_n[idx]) ** 2).mean()
                self.critic_opt.zero_grad()
                closs.backward()
                stats["critic_grad"].append(
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5).item())
                self.critic_opt.step()
                stats["critic_loss"].append(closs.item())
        return {k: float(np.mean(v)) for k, v in stats.items()}


def rollout(env, agent, seed):
    obs, state, masks = env.reset(seed=seed)
    keys = ["obs", "masks", "actions", "logp", "states", "rewards", "values"]
    buf = {k: [] for k in keys}
    ep_reward = ep_energy = 0.0
    ep_disp = 0
    done = False
    while not done:
        actions, lp, value = agent.act(obs, state, masks)
        buf["obs"].append(obs.copy()); buf["masks"].append(masks.copy())
        buf["actions"].append(actions.copy()); buf["logp"].append(lp.copy())
        buf["states"].append(state.copy()); buf["values"].append(value)
        obs, state, masks, reward, done, metrics = env.step(actions)
        buf["rewards"].append(reward)
        ep_reward += reward
        ep_energy += metrics["energy_proxy"]
        ep_disp += metrics["dispatches"]
        last = metrics

    rewards = np.asarray(buf["rewards"], dtype=np.float32)
    values = np.asarray(buf["values"] + [0.0], dtype=np.float32)
    adv = np.zeros_like(rewards)
    gae = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + agent.gamma * values[i + 1] - values[i]
        gae = delta + agent.gamma * agent.lam * gae
        adv[i] = gae

    out = {k: np.asarray(buf[k]) for k in ["obs", "masks", "actions", "logp", "states"]}
    out["advantages"] = adv
    out["returns"] = adv + values[:-1]
    ep_metrics = {
        "unmet_critical_demand": last["unmet_critical_demand"],
        "energy_proxy": ep_energy,
        "dispatches": ep_disp,
        "task_completion_rate": last["task_completion_rate"],
        "length": len(rewards),
    }
    return out, ep_reward, ep_metrics


def concat(batches):
    return {k: np.concatenate([b[k] for b in batches], axis=0) for k in batches[0]}


def evaluate(agent, n=100, seed0=10_000):
    env = DisasterReliefEnv(seed=seed0)
    R, U = [], []
    for i in range(n):
        obs, state, masks = env.reset(seed=seed0 + i)
        done, tot = False, 0.0
        while not done:
            a, _, _ = agent.act(obs, state, masks, greedy=True)
            obs, state, masks, r, done, m = env.step(a)
            tot += r
        R.append(tot); U.append(m["unmet_critical_demand"])
    return float(np.mean(R)), float(np.mean(U))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--episodes-per-update", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results_v2")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = DisasterReliefEnv(seed=args.seed)
    agent = MAPPO(env.local_obs_dim, env.global_state_dim, env.n_actions)

    log = {k: [] for k in ["reward", "unmet", "energy", "completion", "dispatches"]}
    episode = 0
    while episode < args.episodes:
        batches = []
        for _ in range(min(args.episodes_per_update, args.episodes - episode)):
            episode += 1
            b, r, m = rollout(env, agent, args.seed + episode)
            batches.append(b)
            log["reward"].append(r); log["unmet"].append(m["unmet_critical_demand"])
            log["energy"].append(m["energy_proxy"]); log["completion"].append(m["task_completion_rate"])
            log["dispatches"].append(m["dispatches"])
            if episode % 25 == 0:
                print(
                    f"Episode {episode}/{args.episodes} | "
                    f"mean reward(25)={np.mean(log['reward'][-25:]):.2f} | "
                    f"critical unmet={np.mean(log['unmet'][-25:]):.1f} | "
                    f"completion={np.mean(log['completion'][-25:]):.3f} | "
                    f"energy/episode={np.mean(log['energy'][-25:]):.1f}",
                    flush=True,
                )
        agent.update(concat(batches))

    eval_r, eval_u = evaluate(agent)
    print(f"Greedy eval on 100 held-out seeds: reward={eval_r:.2f} | critical unmet={eval_u:.1f}")

    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()},
               os.path.join(args.out, "mappo_model.pt"))

    with open(os.path.join(args.out, "training_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "unmet_critical_demand",
                    "energy_proxy_episode", "task_completion_rate", "dispatches_episode"])
        w.writerows(zip(range(1, args.episodes + 1), log["reward"], log["unmet"],
                        log["energy"], log["completion"], log["dispatches"]))

    window = min(25, args.episodes)
    smooth = lambda v: np.convolve(v, np.ones(window) / window, mode="valid")
    for data, title, ylabel, fn in [
        (log["reward"], "MAPPO training reward", "Episode reward", "mappo_reward.png"),
        (log["unmet"], "Critical unmet demand", "Unmet critical units", "mappo_critical_unmet.png"),
        (log["energy"], "Episode energy proxy", "Energy proxy per episode", "mappo_energy.png"),
        (log["completion"], "Task completion rate", "Completion rate", "mappo_completion.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(1, len(data) + 1), data, alpha=0.3, label="Episode")
        ax.plot(range(window, len(data) + 1), smooth(data), label=f"{window}-episode moving average")
        ax.set_title(title); ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.grid(True); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(args.out, fn), dpi=160)
        plt.close(fig)
    print(f"Saved model, CSV and graphs in {args.out}/")


if __name__ == "__main__":
    main()
