"""
actor.py

Author: Caleb Scott

---

Actor for the atari game environment.

"""

# IMPORTS
import gymnasium as gym
import numpy as np

import torch
from torch import nn

# CONSTANTS

# FUNCTIONS
def greedy_epsilon(q_net: nn.Module, env: gym.Env, state: torch.Tensor, epsilon: float, device):
    """
    Performs epsilon-greedy selection of action (exploration/exploitation tradeoff).
    """
    if epsilon > np.random.uniform():
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = state.float().div_(255.).to(device)
            q_values = q_net(state)
            predicted_actions = q_values.argmax(dim=1).cpu().numpy()
            return predicted_actions

# CLASSES
class EpsilonScheduler:
    """
    This decays epsilon at a linear rate by the number of steps past a
    defined threshold.

    Credit: https://www.kaggle.com/code/shreydan/deep-q-learning-playing-atari-breakout#Double-DQN-(DDQN)
    """

    def __init__(
        self,
        eps_min: float,
        eps_max: float,
        steps: int,
        exploration_ratio: float = 0.1
    ):
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.steps = steps
        self.exploration_ratio = exploration_ratio

        self.exploration_steps = int(steps * self.exploration_ratio)

    def __call__(self, step):
        if step < self.exploration_steps:
            return self.eps_max

        decay_steps = self.steps - self.exploration_steps
        decay_rate = (self.eps_max - self.eps_min) / decay_steps

        epsilon = self.eps_max - (step - self.exploration_steps) * decay_rate

        return max(self.eps_min, epsilon)

class DQNActor:

    def __init__(self):
        pass
