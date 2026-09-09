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
import numpy as np
from numpy.core.multiarray import dtype
import torch

from collections import deque

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
        return torch.from_numpy(stk)
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
        obs_shape: tuple,
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
            torch.from_numpy(s).float().to(self.device)
            torch.from_numpy(a).float().to(self.device)
            torch.from_numpy(ns).float().to(self.device)
            torch.from_numpy(r).float().to(self.device)
            torch.from_numpy(t).float().to(self.device)

        # Return data for viewing
        return s, a, ns, r, t
