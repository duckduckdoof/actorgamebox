"""
eye.py

Author: Caleb Scott

---

NN Design of the eye saccade module.

"""

# IMPORTS
import torch
from torch import nn
from torch.distributions import Normal


# CLASSES
class BoundedGaussianPolicy(nn.Module):

    def __init__(self, x_dim, out_dim, log_std_min, log_std_max, *, compile_, device=None):
        super().__init__()

        # Predictor networks
        self.mean_pred = compile_(nn.Linear(x_dim, out_dim, device=device))
        self.log_std_pred = compile_(nn.Linear(x_dim, out_dim, device=device))

        # Numerical stability for std dev.
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        mean = self.mean_pred(x)
        log_std = self.log_std_pred(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        norm = Normal(mean, std)

        pred_x = norm.rsample()

        pred_pos = torch.tanh(pred_x)

        # Log probability with Jacobian correction for tanh squashing
        log_prob = norm.log_prob(pred_x)
        log_prob -= torch.log(1.0 - pred_pos.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return pred_pos, log_prob, mean
