

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def mlp(sizes: Sequence[int], activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int],
                 log_std_min: float = -20.0, log_std_max: float = 2.0):
        super().__init__()
        self.body = mlp([obs_dim] + list(hidden), output_activation=nn.ReLU)
        self.mean = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
            action = torch.tanh(pre_tanh)
            log_prob = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
            return action, log_prob
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)
        # Gaussian log-density minus the tanh change-of-variables term.
        log_prob = (-0.5 * noise.pow(2) - log_std - LOG_SQRT_2PI).sum(dim=-1)
        log_prob = log_prob - (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(dim=-1)
        return action, log_prob


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int]):
        super().__init__()
        sizes = [obs_dim + act_dim] + list(hidden) + [1]
        self.q1 = mlp(sizes)
        self.q2 = mlp(sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat([obs, action], dim=-1)
        return self.q1(joint).squeeze(-1), self.q2(joint).squeeze(-1)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
