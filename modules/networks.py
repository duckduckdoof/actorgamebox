"""
networks.py

Author: Caleb Scott

---

Neural Network declarations, to be used by actors.

"""

# IMPORTS
import torch

from torch import nn

# CONSTANTS

# FUNCTIONS

# CLASSES
class DQN(nn.Module):
    """
    Vanilla NN for predicting action, given current observation state.
    DQN credit:
        https://www.kaggle.com/code/shreydan/deep-q-learning-playing-atari-breakout#Double-DQN-(DDQN)
        |-> in this tutorial, we convert to grayscale, and pass 4 frames into a CNN.
        |-> thus, one kernel per channel, summed together for each channel after one convolution.
    """

    def __init__(self, stacked_gray_frames: int = 4, action_space: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(stacked_gray_frames, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            # This step wasn't included in the original article.
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        return self.model(x)

class PPONetwork(nn.Module):
    """
    NN for doing PPO, actor-critic style.

    Actors select the best action, given the current observation.

    Critics estimate the value of the action taken at the current observation.
    """

    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(512, 4)
        )
        self.critic = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, x, action=None):
        x = self.base(x)
        probs = torch.distributions.Categorical(logits=self.actor(x))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
