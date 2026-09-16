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
import torch.nn.functional as F

from modules.networks import DQN

# CONSTANTS

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

class EpsilonDQNActor:

    def __init__(
        self,
        stacked_frames: int,
        action_space,
        epsilon_sched: EpsilonScheduler,
        epsilon_sched_init_step: int,
        lr: float,
        discount_rate: float,
        model_save: str,
        device
    ):
        self.q_net = DQN(stacked_frames, action_space.n)
        self.action_space = action_space
        self.epsilon_sched = epsilon_sched
        self.epsilon = self.epsilon_sched(epsilon_sched_init_step)
        self.discount_rate = discount_rate
        self.device = device
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.model_save = model_save

    def update_epsilon(self, i: int):
        self.epsilon = self.epsilon_sched(i)

    def action_greedy_epsilon(self, state):
        if self.epsilon > np.random.uniform():
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state = state.float().to(self.device)
                q_values = self.q_net(state)
                # Since we're doing one action prediction, we index at 0
                #   This won't be the case if we're doing multi-env selection.
                predicted_actions = q_values.argmax(dim=1).cpu().numpy()[0]
                return predicted_actions

    def action_q_net(self, state):
        return self.q_net(state)

    def train_step(self, sample: tuple):
        """
        Given sampled observation, action, next observation, reward, done,
        calculate the loss and iterate a single training step.
        """
        o, a, no, r, d = sample

        # Make a prediction for the action, given the observation
        pred_act = self.q_net(o)
        pred_next_act_idxs = self.q_net(no).argmax(dim=1)

        # Use actual actions taken to get the "value" of those actions
        pred_act_q = pred_act.gather(1, a.int().unsqueeze(1)).flatten()
        pred_next_act_q = self.q_net(no).gather(1, pred_next_act_idxs.unsqueeze(1)).flatten()

        # Discounted rewards, using q-values of actions
        target_q = r + self.discount_rate * (1-d) * pred_next_act_q
        loss = F.smooth_l1_loss(pred_act_q, target_q)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        return loss.item()

    def save_state(self):
        torch.save(self.q_net.state_dict(), self.model_save)
