"""
environment.py

Author: Caleb Scott

---

Create and run atari-based environment.

"""

# IMPORTS
# This is needed to specify which video driver to use (linux).
import os
os.environ['SDL_VIDEODRIVER'] = "x11"

import ale_py
import gymnasium as gym

from datetime import datetime, timezone
from configs.defaults import DEFAULT_DATETIME

# Register ALE before using the Atari ROMs
gym.register_envs(ale_py)

# CLASSES
class Environment:
    """
    Wrapper class for gym Atari environment + configs.
    """

    def __init__(
        self,
        display_mode: str,
        game_name: str,
        mode: str,
        record_dir: str,
        verbose: bool,
        seed: int,
        episode_rec_freq: int,
        wrappers: list,
        wrappers_kwargs: list
    ):
        # Init environment based on display mode (human/rgb_array)
        mode = "human" if display_mode == "human" else "rgb_array"
        self.env = gym.make(game_name, render_mode=mode)
        self.env.action_space.seed(seed)

        # Recording wrapper
        today = datetime.now(tz=timezone.utc).strftime(DEFAULT_DATETIME)
        if display_mode == "record":
            self.env = gym.wrappers.RecordVideo(
                self.env,
                episode_trigger=lambda x: x % episode_rec_freq == 0,
                video_folder=record_dir,
                name_prefix=f"{today}-rec"
            )

        # Add other wrappers to this env
        for wrapper, kwargs in zip(wrappers, wrappers_kwargs):
            self.env = wrapper(self.env, **kwargs)

    def reset(self):
        return self.env.reset()

    def obs_space(self):
        return self.env.observation_space

    def act_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()
