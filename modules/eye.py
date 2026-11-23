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

from modules import nets


# CLASSES
class BoundedGaussianPolicy(nn.Module):

    def __init__(self, x_dim, out_dim, log_std_min, log_std_max, 
                 dims, norm, act, init, out_init, *, compile_, device=None):
        super().__init__()

        # Predictor networks
        mean_modules, mean_backbone_dim = nets.mlp(x_dim, dims, norm, act, init, out_bias=True, out_norm=True, device=device)
        mean_head = nn.Linear(mean_backbone_dim, out_dim, bias=True, device=device)
        mean_modules.append(nets.init_(mean_head, out_init))

        std_modules, std_backbone_dim = nets.mlp(x_dim, dims, norm, act, init, out_bias=True, out_norm=True, device=device)
        std_head = nn.Linear(std_backbone_dim, out_dim, bias=True, device=device)
        std_modules.append(nets.init_(std_head, out_init))

        self.mean_pred = compile_(nn.Sequential(*mean_modules))
        self.log_std_pred = compile_(nn.Sequential(std_modules))

        # Numerical stability for std dev.
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_stats(self, inp):
        mean = self.mean_pred(inp)
        log_std = self.log_std_pred(inp)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        norm = Normal(mean, std)
        return norm

    @torch.no_grad()
    def predict(self, stats):
        pred_x = stats.rsample()
        pred_pos = torch.tanh(pred_x)

        # Log probability with Jacobian correction for tanh squashing
        log_prob = stats.log_prob(pred_x)
        log_prob -= torch.log(1.0 - pred_pos.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return pred_pos, log_prob

    @torch.no_grad()
    def forward(self, x):
        stats = self.get_stats(x)
        sample_pos, entropy = self.predict(stats)
        return sample_pos, entropy
