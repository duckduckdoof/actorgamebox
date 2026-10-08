"""
memory.py

Author: Caleb Scott

---

File for holding different memory constructions, such
as simple frame buffers.

Credit goes to the following for inspiration/code:
    1. https://www.kaggle.com/code/auxeno/ppo-on-atari-rl
    2. https://www.kaggle.com/code/shreydan/deep-q-learning-playing-atari-breakout#DQN-(Deep-Q-Network)

"""

# IMPORTS
from collections import deque

import numpy as np
import torch

from modules_new import utils


# CLASSES
class FrameBuffer:
    """
    Class for holding a finite queue of frames captured from the environment.
    """

    def __init__(self, frame_limit: int):
        self.frames = deque(maxlen=frame_limit)

    def add(self, frame: np.ndarray):
        self.frames.append(frame)

    def get(self):
        stk = np.stack(self.frames)
        # Remember to add batch dimension.
        return torch.from_numpy(stk).unsqueeze(0)
        # We may want to do this, although I'd prefer that
        # things like datatype are handled elsewhere.
        # return torch.from_numpy(stk).float()

class MultiFrameBuffer:
    """
    Capturing frames from environments running in parallel.
    """

    def __init__(self, envs: int, frame_limit: int):
        self.envs = envs
        self.frames = {fid: deque(maxlen=frame_limit) for fid in range(envs)}

    def add(self, frames: np.ndarray):
        for eid in range(self.envs):
            self.frames[eid].append(frames[eid])

    def get(self):
        stk = np.stack([np.stack(f) for _, f in self.frames.items()])
        return torch.from_numpy(stk)
        # Maybe we need this instead, but I want datatype handled elsewhere.
        # return torch.from_numpy(stk).float()

class CircularReplayBuffer:
    """
    Instead of storing frames in sequence like the FrameBuffer,
    we will store tuples of (s, a, s', r, d):
        s => state at (t-1)
        a => chosen action by actor at (t-1)
        s' => state at (t)
        r => reward given at (t)
        d => 'done', indicates that we have stopped to end the episode.

    Items stored are circular; if we add a record past the max size,
    then this wraps and overwrites the first element.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        obs_type: np.dtype,
        replay_limit: int,
        batch_size: int,
        device: str
    ):
        # Init buffer attributes
        self.obs_shape = obs_shape
        self.replay_limit = replay_limit
        self.batch_size = batch_size
        self.device = device
        self.size = 0
        self.ptr = 0

        self.rng = np.random.default_rng()

        # Init memory for buffer
        # NOTE: we may want to specify the datatype as a parameter
        state_shp = (self.replay_limit, *obs_shape)
        self.states = np.zeros(state_shp, dtype=obs_type)
        self.actions = np.zeros((self.replay_limit,), dtype=np.int64)
        self.next_states = np.zeros(state_shp, dtype=obs_type)
        self.rewards = np.zeros((self.replay_limit,), dtype=np.float32)
        self.terminals = np.zeros((self.replay_limit,), dtype=np.bool_)

    def add(self, s, a, ns, r, d):
        """
        Add state, action, next state, reward, terminal flag.
        """
        self.states[self.ptr] = s
        self.next_states[self.ptr] = ns
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.terminals[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.replay_limit
        self.size = min(self.size + 1, self.replay_limit)

    def sample(self, to_gpu=True) -> tuple:
        """
        Pulls a sampled batch from the buffer.
        """
        idxs = self.rng.integers(0, self.size, size=self.batch_size)

        # Get sampled data
        s = self.states[idxs]
        a = self.actions[idxs]
        ns = self.next_states[idxs]
        r = self.rewards[idxs]
        t = self.terminals[idxs]

        # Send sample to GPU
        if to_gpu:
            ts = torch.from_numpy(s).float().to(self.device)
            ta = torch.from_numpy(a).float().to(self.device)
            tns = torch.from_numpy(ns).float().to(self.device)
            tr = torch.from_numpy(r).float().to(self.device)
            tt = torch.from_numpy(t).float().to(self.device)

            return ts, ta, tns, tr, tt

        # Return data for viewing (remember these aren't from the gpu!)
        return s, a, ns, r, t

    def ready(self):
        return self.size >= self.batch_size

class ReplayBuffer:
    """ 
    Stores experience of the actor in a buffer. 
    """

    def __init__(
            self,
            obs_space,
            act_space,
            capacity,
            start_o,
            device
        ):
        self.obs_space = obs_space
        self.act_space = act_space

        self.obs = utils.space_zeros(obs_space, capacity, device=device)
        self.acts = utils.space_zeros(act_space, capacity, device=device)
        self.next_rew = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_terms = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_truncs = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_obs = utils.space_zeros(obs_space, capacity, device=device)

        # Async put the first observation
        self.obs[0] = torch.as_tensor(np.array(start_o)).to(device=device, non_blocking=True)
        self.len = 0

        # Track rewards in episodes.
        self.episode_reward = 0.0
        self.episode_rewards = []
        self.episode_idxs = []
        self.episode_idx = self.len
        self.episode_idx_start = self.len

    def to(self, device):
        if device == self.obs.device:
            return self

        # Async put data onto (new) device
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
        """ We've initialized memory for each var to be the capacity. """
        return self.obs.shape[0]

    @property
    def last_o(self):
        """ Most recent observation. """
        # Adds batch dimension at the beginning for convenience.
        return self.obs[self.len].unsqueeze(0)

    def __len__(self):
        return self.len

    def get(self, idx, *keys):
        """ 
        Returns selected variables:
            - obs
            - act
            - next_rew
            - next_term
            - next_trunc
            - next_obs

        ...specified by 'keys', indexed by 'idx'
        """
        if not keys:
            keys = ('obs', 'act', 'next_rew', 'next_term', 'next_trunc', 'next_obs')

        tensors = {
            'obs': self.obs,
            'act': self.acts,
            'next_rew': self.next_rew,
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

        # This will always be a tuple
        return tuple(get_item(t) for t in tensors)

    def step(self, a, next_o, next_r, next_term, next_trunc, cont_o):
        """ 
        Add transition tuple to buffer.
        This is a little more complicated, since we load our first observation
        into the buffer before any additional actions, rewards, and next obs.

        NOTE: cont_o is an observation made upon reset of the environment. Otherwise,
        the trainer just copies the next observation (but it's ignored in this case).
        """
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

            start, end = self.episode_idx_start, self.episode_idx
            self.episode_idxs.append((start, end+1))
            self.episode_idx_start = end
            self.episode_idx = self.episode_idx_start

        # Update current episode
        self.episode_idx += 1

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

