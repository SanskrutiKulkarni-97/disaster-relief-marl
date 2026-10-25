import numpy as np


class DisasterReliefEnv:
    """Cooperative multi-warehouse relief simulator with stochastic demand and road failures."""

    def __init__(
        self,
        n_warehouses=3,
        n_zones=5,
        max_steps=20,
        seed=42,
        road_failure_prob=0.03,
        stochastic_demand=True,
    ):
        self.n_warehouses = n_warehouses
        self.n_zones = n_zones
        self.max_steps = max_steps
        self.road_failure_prob = road_failure_prob
        self.stochastic_demand = stochastic_demand
        self.rng = np.random.default_rng(seed)

        self.n_commodities = 3
        self.quantities = (30, 60)
        self.noop_action = self.n_commodities * n_zones * len(self.quantities)
        self.n_actions = self.noop_action + 1

        self.local_obs_dim = 3 + 1 + n_zones * 3 + n_zones + n_zones + 1
        self.global_state_dim = (
            n_warehouses * 4
            + n_zones * 10
            + n_warehouses * n_zones
            + 1
        )
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.inventory = np.tile([500.0, 400.0, 200.0], (self.n_warehouses, 1))
        self.vehicles = np.full(self.n_warehouses, 4.0, dtype=np.float32)
        self.priorities = self.rng.choice(
            [1, 2, 3], self.n_zones, p=[0.25, 0.35, 0.40]
        )

        base = np.array([140.0, 105.0, 45.0])
        self.demand = np.maximum(
            1.0, base * self.rng.uniform(0.8, 1.2, (self.n_zones, 3))
        )
        self.received = np.zeros_like(self.demand)
        self.roads = np.ones((self.n_warehouses, self.n_zones), dtype=np.float32)
        self.roads[
            self.rng.integers(self.n_warehouses),
            self.rng.integers(self.n_zones),
        ] = 0.0

        return self.local_observations(), self.global_state(), self.action_masks()

    def unmet(self):
        return np.maximum(self.demand - self.received, 0.0)

    def local_observations(self):
        unmet_ratio = self.unmet() / np.maximum(self.demand, 1.0)
        return np.stack([
            np.concatenate([
                self.inventory[w] / [500.0, 400.0, 200.0],
                [self.vehicles[w] / 4.0],
                unmet_ratio.reshape(-1),
                self.priorities / 3.0,
                self.roads[w],
                [self.t / self.max_steps],
            ])
            for w in range(self.n_warehouses)
        ]).astype(np.float32)

    def global_state(self):
        return np.concatenate([
            (self.inventory / [500.0, 400.0, 200.0]).reshape(-1),
            self.vehicles / 4.0,
            (self.demand / [180.0, 140.0, 60.0]).reshape(-1),
            (self.received / [180.0, 140.0, 60.0]).reshape(-1),
            (self.unmet() / [180.0, 140.0, 60.0]).reshape(-1),
            self.priorities / 3.0,
            self.roads.reshape(-1),
            [self.t / self.max_steps],
        ]).astype(np.float32)

    def decode(self, action):
        if int(action) == self.noop_action:
            return None
        quantity_index = int(action) % len(self.quantities)
        x = int(action) // len(self.quantities)
        commodity = x // self.n_zones
        zone = x % self.n_zones
        return commodity, zone, self.quantities[quantity_index]

    def action_masks(self):
        masks = np.zeros((self.n_warehouses, self.n_actions), dtype=bool)
        masks[:, self.noop_action] = True
        remaining_demand = self.unmet()

        for w in range(self.n_warehouses):
            if self.vehicles[w] <= 0:
                continue
            for c in range(self.n_commodities):
                if self.inventory[w, c] <= 0:
                    continue
                for z in range(self.n_zones):
                    # Prevent dispatches to already satisfied demand.
                    if remaining_demand[z, c] <= 0:
                        continue
                    if self.roads[w, z] <= 0:
                        continue
                    for qi in range(len(self.quantities)):
                        action_index = (
                            (c * self.n_zones + z) * len(self.quantities) + qi
                        )
                        masks[w, action_index] = True
        return masks

    def step(self, actions):
        self.t += 1
        served_by_zone = np.zeros(self.n_zones)
        energy = 0.0
        dispatches = 0

        for w, action in enumerate(actions):
            decoded = self.decode(action)
            if decoded is None or self.vehicles[w] <= 0:
                continue

            c, z, requested = decoded
            if self.roads[w, z] <= 0 or self.inventory[w, c] <= 0:
                continue

            # Re-check demand sequentially because another warehouse may
            # have served this same zone/commodity earlier in this step.
            unmet_before = max(0.0, self.demand[z, c] - self.received[z, c])
            if unmet_before <= 0:
                continue

            qty = min(float(requested), float(self.inventory[w, c]))
            delivered = min(qty, unmet_before)

            self.inventory[w, c] -= qty
            # Only actually satisfied demand is counted as received.
            # Any surplus shipment is treated as wasted.
            self.received[z, c] += delivered
            self.vehicles[w] -= 1
            served_by_zone[z] += delivered
            energy += qty * (1.0 + 0.25 * (z + 1))
            dispatches += 1

        if self.stochastic_demand:
            self.demand += self.rng.poisson([2.0, 1.5, 0.7], size=self.demand.shape)

        if self.road_failure_prob > 0:
            self.roads[
                self.rng.random(self.roads.shape) < self.road_failure_prob
            ] = 0.0

        weights = np.where(
            self.priorities == 1,
            4.0,
            np.where(self.priorities == 2, 2.0, 1.0),
        )
        weighted_service = float(np.sum(weights * served_by_zone))
        critical_unmet = float(self.unmet()[self.priorities == 1].sum())

        # Intentionally unchanged for a controlled comparison.
        reward = 0.05 * weighted_service - 0.015 * energy - 0.08 * critical_unmet

        done = self.t >= self.max_steps or self.vehicles.sum() <= 0
        metrics = {
            "satisfied_demand": float(served_by_zone.sum()),
            "unmet_critical_demand": critical_unmet,
            "energy_proxy": energy,
            "dispatches": dispatches,
            "task_completion_rate": float(np.mean(self.received >= self.demand)),
        }
        return (
            self.local_observations(),
            self.global_state(),
            self.action_masks(),
            float(reward),
            done,
            metrics,
        )
