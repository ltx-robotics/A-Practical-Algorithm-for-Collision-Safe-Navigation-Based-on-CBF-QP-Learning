
from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, max_rows: int):
        self.capacity = int(capacity)
        self.max_rows = int(max_rows)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_nominal = np.zeros((capacity, act_dim), dtype=np.float32)
        self.action_executed = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.cons_a = np.zeros((capacity, max_rows, 2), dtype=np.float32)
        self.cons_b = np.zeros((capacity, max_rows), dtype=np.float32)
        self.cons_mask = np.zeros((capacity, max_rows), dtype=np.float32)
        self.next_cons_a = np.zeros((capacity, max_rows, 2), dtype=np.float32)
        self.next_cons_b = np.zeros((capacity, max_rows), dtype=np.float32)
        self.next_cons_mask = np.zeros((capacity, max_rows), dtype=np.float32)
        self._index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action_nominal, action_executed, reward, next_obs, done,
            constraints, next_constraints) -> None:
        i = self._index
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.action_nominal[i] = action_nominal
        self.action_executed[i] = action_executed
        self.reward[i] = reward
        self.done[i] = float(done)
        self.cons_a[i], self.cons_b[i], self.cons_mask[i] = constraints
        self.next_cons_a[i], self.next_cons_b[i], self.next_cons_mask[i] = next_constraints
        self._index = (self._index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device,
               rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        index = rng.integers(0, self._size, size=batch_size)

        def tensor(array):
            return torch.as_tensor(array[index], device=device)

        return {
            "obs": tensor(self.obs),
            "next_obs": tensor(self.next_obs),
            "action_nominal": tensor(self.action_nominal),
            "action_executed": tensor(self.action_executed),
            "reward": tensor(self.reward),
            "done": tensor(self.done),
            "cons_a": tensor(self.cons_a),
            "cons_b": tensor(self.cons_b),
            "cons_mask": tensor(self.cons_mask),
            "next_cons_a": tensor(self.next_cons_a),
            "next_cons_b": tensor(self.next_cons_b),
            "next_cons_mask": tensor(self.next_cons_mask),
        }
