import os
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from disaster_env import DisasterReliefEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(inp, out):
    return nn.Sequential(
        nn.Linear(inp, 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, out),
    )


class MAPPO:
    """Shared decentralized actor with a centralized critic (CTDE)."""

    def __init__(
        self,
        local_dim,
        global_dim,
        n_actions,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip=0.2,
    ):
        self.actor = mlp(local_dim, n_actions).to(DEVICE)
        self.critic = mlp(global_dim, 1).to(DEVICE)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        self.gamma = gamma
        self.lam = lam
        self.clip = clip

    def dist(self, obs, masks):
        logits = self.actor(obs).masked_fill(~masks.bool(), -1e9)
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs, state, masks):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        masks_t = torch.tensor(masks, dtype=torch.bool, device=DEVICE)
        distribution = self.dist(obs_t, masks_t)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        value = self.critic(
            torch.tensor(state, dtype=torch.float32, device=DEVICE)
        ).squeeze()
        return actions.cpu().numpy(), log_probs.cpu().numpy(), float(value.item())

    def update(self, batch, epochs=8, batch_size=128):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=DEVICE)
        masks = torch.tensor(batch["masks"], dtype=torch.bool, device=DEVICE)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=DEVICE)
        old_log_probs = torch.tensor(batch["logp"], dtype=torch.float32, device=DEVICE)
        states = torch.tensor(batch["states"], dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=DEVICE)
        advantages = torch.tensor(
            batch["advantages"], dtype=torch.float32, device=DEVICE
        )

        time_steps, n_agents = obs.shape[:2]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_masks = masks.reshape(-1, masks.shape[-1])
        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)

        # A team-level advantage is shared by agents at each time step.
        flat_advantages = advantages[:, None].expand(time_steps, n_agents).reshape(-1)
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / (flat_advantages.std(unbiased=False) + 1e-8)

        for _ in range(epochs):
            permutation = torch.randperm(len(flat_actions), device=DEVICE)
            for start in range(0, len(flat_actions), batch_size):
                indices = permutation[start:start + batch_size]
                distribution = self.dist(flat_obs[indices], flat_masks[indices])
                new_log_probs = distribution.log_prob(flat_actions[indices])

                ratio = torch.exp(new_log_probs - flat_old_log_probs[indices])
                surrogate1 = ratio * flat_advantages[indices]
                surrogate2 = (
                    torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip)
                    * flat_advantages[indices]
                )
                actor_loss = -torch.minimum(surrogate1, surrogate2).mean()
                entropy = distribution.entropy().mean()

                # Critic regression on sampled global states and team returns.
                time_indices = torch.randint(
                    0, time_steps, (min(batch_size, time_steps),), device=DEVICE
                )
                values = self.critic(states[time_indices]).squeeze(-1)
                critic_loss = ((values - returns[time_indices]) ** 2).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    0.5,
                )
                self.opt.step()


def rollout(env, agent, seed):
    obs, state, masks = env.reset(seed=seed)
    batch = {
        key: []
        for key in [
            "obs", "masks", "actions", "logp", "states", "rewards", "values"
        ]
    }
    episode_reward = 0.0
    episode_energy = 0.0
    episode_dispatches = 0
    last_metrics = {}
    done = False

    while not done:
        actions, log_probs, value = agent.act(obs, state, masks)
        batch["obs"].append(obs.copy())
        batch["masks"].append(masks.copy())
        batch["actions"].append(actions.copy())
        batch["logp"].append(log_probs.copy())
        batch["states"].append(state.copy())
        batch["values"].append(value)

        obs, state, masks, reward, done, metrics = env.step(actions)
        batch["rewards"].append(reward)
        episode_reward += reward

        # FIX: aggregate these metrics across every step in the episode.
        episode_energy += metrics["energy_proxy"]
        episode_dispatches += metrics["dispatches"]
        last_metrics = metrics

    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    values = np.asarray(batch["values"] + [0.0], dtype=np.float32)
    advantages = np.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + agent.gamma * values[t + 1] - values[t]
        gae = delta + agent.gamma * agent.lam * gae
        advantages[t] = gae

    result = {
        key: np.asarray(batch[key])
        for key in ["obs", "masks", "actions", "logp", "states"]
    }
    result["returns"] = advantages + values[:-1]
    result["advantages"] = advantages

    episode_metrics = {
        "unmet_critical_demand": last_metrics["unmet_critical_demand"],
        "energy_proxy": episode_energy,
        "dispatches": episode_dispatches,
        "task_completion_rate": last_metrics["task_completion_rate"],
    }
    return result, episode_reward, episode_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results_fixed")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = DisasterReliefEnv(seed=args.seed)
    agent = MAPPO(env.local_obs_dim, env.global_state_dim, env.n_actions)

    rewards = []
    unmet = []
    energy = []
    completion = []
    dispatches = []

    for episode in range(1, args.episodes + 1):
        batch, episode_reward, metrics = rollout(
            env, agent, args.seed + episode
        )
        agent.update(batch)

        rewards.append(episode_reward)
        unmet.append(metrics["unmet_critical_demand"])
        energy.append(metrics["energy_proxy"])
        completion.append(metrics["task_completion_rate"])
        dispatches.append(metrics["dispatches"])

        if episode % 25 == 0:
            print(
                f"Episode {episode}/{args.episodes} | "
                f"mean reward(25)={np.mean(rewards[-25:]):.2f} | "
                f"critical unmet={np.mean(unmet[-25:]):.1f} | "
                f"completion={np.mean(completion[-25:]):.3f} | "
                f"energy/episode={np.mean(energy[-25:]):.1f}"
            )

    torch.save(
        {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()},
        os.path.join(args.out, "mappo_model.pt"),
    )

    csv_path = os.path.join(args.out, "training_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "episode", "reward", "unmet_critical_demand",
            "energy_proxy_episode", "task_completion_rate", "dispatches_episode"
        ])
        writer.writerows(zip(
            range(1, args.episodes + 1),
            rewards, unmet, energy, completion, dispatches
        ))

    window = min(25, args.episodes)

    def smooth(values):
        return np.convolve(values, np.ones(window) / window, mode="valid")

    for data, title, ylabel, filename in [
        (rewards, "MAPPO training reward", "Episode reward", "mappo_reward.png"),
        (unmet, "Critical unmet demand", "Unmet critical units", "mappo_critical_unmet.png"),
        (energy, "Episode energy proxy", "Energy proxy per episode", "mappo_energy.png"),
        (completion, "Task completion rate", "Completion rate", "mappo_completion.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(1, len(data) + 1), data, alpha=0.3, label="Episode")
        ax.plot(
            range(window, len(data) + 1),
            smooth(data),
            label=f"{window}-episode moving average",
        )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, filename), dpi=160)
        plt.close(fig)

    print(f"Saved model, CSV and graphs in {args.out}/")


if __name__ == "__main__":
    main()
