"""
networks.py

Author: Caleb Scott

---

Neural Network declarations, to be used by actors.

"""

# IMPORTS
from torch import nn

# CONSTANTS

# FUNCTIONS

# CLASSES
class DQN(nn.Module):
    """
    Vanilla NN for predicting action, given current observation state.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 4)
        )

    def forward(self, x):
        return self.model(x)
