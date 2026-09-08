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

gym.register_envs(ale_py)

# CLASSES
class Environment:

    def __init__(
        self,
        display_mode: str,
        game_name: str,
        mode: str,
        record_dir: str,
        verbose: bool,
        seed: int
    ):
        # Init environment based on display mode (human/rgb_array)
        mode = "human" if display_mode == "human" else "rgb_array"
        self.env = gym.make(game_name, render_mode=mode)

        # Recording wrapper
        if display_mode == "record":
            self.env = gym.wrappers.RecordVideo(
                self.env,
                episode_trigger=lambda num: num % 2 == 0,
                video_folder=record_dir,
                name_prefix="recording-"
            )

        # Reset environment before running
        self.reset()

    def reset(self):
        self.env.reset()
