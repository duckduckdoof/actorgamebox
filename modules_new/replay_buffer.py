"""
replay_buffer.py

Author: Caleb Scott

---

ReplayBuffer to store transitions during gameplay.

"""

# IMPORTS
import numpy as np
import torch

from modules_new import utils


# CLASSES
class ReplayBuffer:
    """ For storing transitions """

    def __init__(self, obs_space, act_space, capacity, start_o, device):
        self.obs_space = obs_space
        self.act_space = act_space

        self.obs = utils.space_zeros(obs_space, capacity, device=device)
        self.acts = utils.space_zeros(act_space, capacity, device=device)
        self.next_rew = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_terms = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_truncs = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_obs = utils.space_zeros(obs_space, capacity, device=device)

        self.obs[0] = torch.as_tensor(np.array(start_o)).to(device=device, non_blocking=True)
        self.len = 0

        self.episode_reward = 0.0
        self.episode_rewards = []

    def to(self, device):
        if device == self.obs.device:
            return self

        self.obs = self.obs.to(device, non_blocking=True)
        self.acts = self.acts.to(device, non_blocking=True)
        self.next_rew = self.next_rew.to(device, non_blocking=True)
        self.next_terms = self.next_terms.to(device, non_blocking=True)
        self.next_truncs = self.next_truncs.to(device, non_blocking=True)
        self.next_obs = self.next_obs.to(device, non_blocking=True)

        return self

    @property
    def device(self):
        return self.obs.device

    @property
    def capacity(self):
        return self.obs.shape[0]

    @property
    def cont_o(self):
        return self.obs[self.len].unsqueeze(0)

    def __len__(self):
        return self.len

    def get(self, idx, *keys):
        """ Return transitions at selcted indices """
        if len(keys) == 0:
            keys = ('o', 'a', 'next_r', 'next_term', 'next_trunc', 'next_o')

        tensors = {
            'o': self.obs,
            'a': self.acts,
            'next_r': self.next_rew,
            'next_term': self.next_terms,
            'next_trunc': self.next_truncs,
            'next_o': self.next_obs
        }
        tensors = tuple(tensors[key] for key in keys)
        device = self.device

        if isinstance(idx, np.ndarray):
            idx = torch.as_tensor(idx).to(device=device, non_blocking=True)

        if torch.is_tensor(idx):
            if device is not None and idx.device != device:
                idx = idx.to(device=device, non_blocking=True)

            assert torch.all(idx >= 0)
            assert torch.all(idx < self.len)

            flat_idx = idx.flatten()
            get_item = lambda t: t[flat_idx].reshape(idx.shape + t.shape[1:])
        else:
            assert torch.all(idx < self.len)
            get_item = lambda t: t[idx]

        if isinstance(tensors, (tuple, list)):
            return tuple(get_item(t) for t in tensors)
        elif isinstance(tensors, dict):
            return {key: get_item(t) for key, t in tensors.items()}
        elif torch.is_tensor(tensors):
            return get_item(tensors)
        else:
            raise ValueError()

    def step(self, a, next_o, next_r, next_term, next_trunc, cont_o):
        """ Add transition tuple to buffer """
        if self.len >= self.capacity:
            raise ValueError('Buffer is full')

        i = self.len
        next_o = torch.as_tensor(np.array(next_o), dtype=self.obs.dtype)\
                .to(device=self.obs.device, non_blocking=True)
        self.next_obs[i] = next_o

        self.acts[i] = a
        self.next_rew[i] = next_r
        self.next_terms[i] = next_term
        self.next_truncs[i] = next_trunc

        self.episode_reward += next_r

        # If we have reached the end of the episode
        if next_term or next_trunc:
            next_o = torch.as_tensor(np.array(cont_o), dtype=self.obs.dtype)\
                    .to(device=self.obs.device, non_blocking=True)
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0

        self.len += 1
        if self.len < self.capacity:
            self.obs[self.len] = next_o

        next_r = self.next_rew[i]
        next_term = self.next_terms[i]
        next_trunc = self.next_truncs[i]
        return next_r.unsqueeze(0), next_term.unsqueeze(0), next_trunc.unsqueeze(0), next_o.unsqueeze(0)

    def sample_idx(self, count, rng):
        idx = rng.choice(self.len, count, replace=False, shuffle=False)
        idx = torch.as_tensor(idx).to(self.device, non_blocking=True)
        return idx

    def get_stats(self):
        """ Buffer statistics, from episode rewards """
        episode_rewards = np.array(self.episode_rewards)
        return {
            'buffer_size': self.len,
            'buffer_episodes': len(episode_rewards),
            'buffer_max_episode_reward': np.max(episode_rewards),
            'buffer_total_reward': np.sum(episode_rewards)
        }
