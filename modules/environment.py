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

from datetime import datetime, timezone
from functools import partial

import ale_py
import gymnasium as gym
import numpy as np

from configs.defaults import DEFAULT_DATETIME

# Register ALE before using the Atari ROMs
gym.register_envs(ale_py)

def simple_atari_env(
        game_name: str, 
        max_frames: int = 108000, 
        full_action_space=False, 
        episodic_life: bool = True,
        make=True
    ):
    """ Much simpler atari environment. """
    env_id = game_name
    env = gym.make(env_id, full_action_space=False)
    wrappers = [
        partial(
            gym.wrappers.TimeLimit,
            max_episode_steps=max_frames
        ),
        EpisodicLifeWrapper if episodic_life else None
    ]
    wrappers = [w for w in wrappers if w]
    kwargs = {'full_action_space': full_action_space}

    if make:
        env = gym.make(env_id, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    else:
        return env_id, wrappers, kwargs

def eye_env(
        # Gym game args
        game_name: str,
        max_frames: int = 108000, 
        full_action_space=False, 
        episodic_life: bool = True,

        # Eye env args
        patch_size: int | tuple[int, int] = 30,
        patch_pos: tuple[int, int] | np.ndarray = (0,0),
        mov_scale: float = 1.0,
        max_sacchades: int = 10,
        sacchade_penalty: float = -0.1,

        # Make the game or set it up for later
        make=True
    ):
    """
    Atari environment, including eye sacchade wrapper for additional actions.
    """
    env_id = game_name
    env = gym.make(env_id, full_action_space=False)
    wrappers = [
        partial(
            gym.wrappers.TimeLimit,
            max_episode_steps=max_frames
        ),
        EpisodicLifeWrapper if episodic_life else None,
        partial(
            EyePatchWrapper,
            patch_size=patch_size,
            patch_pos=patch_pos,
            mov_scale=mov_scale,
            max_sacchades=max_sacchades,
            sacchade_penalty=sacchade_penalty
        )
    ]
    wrappers = [w for w in wrappers if w]
    kwargs = {'full_action_space': full_action_space}

    if make:
        env = gym.make(env_id, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    else:
        return env_id, wrappers, kwargs

def atari_env(
        game_name: str,
        episodic_life: bool = False,
        full_action_space: bool = False,
        make: bool = True,
        grayscale: bool = False,
        frame_skip: int | None = 1,
        frame_stack: int | None = None,
        resolution: int | tuple[int, int] | None = None,
        noop_max: int = 30,
        max_frames: int = 108000,
    ):
    if max_frames > 108000:
        raise NotImplementedError("NoFrameskip-v4 does not support max frames > 108000")

    atari_args = {
        'noop_max': noop_max,
        'frame_skip': frame_skip,
        'terminal_on_life_loss': False,
        'grayscale_obs': grayscale,
        'grayscale_newaxis': True
    }
    if resolution:
        atari_args['screen_size'] = resolution

    env_id = game_name
    wrappers = [
        partial(
            gym.wrappers.TimeLimit,
            max_episode_steps=max_frames
        ),
        partial(
            gym.wrappers.AtariPreprocessing, 
            **atari_args
        ),
        partial(
            gym.wrappers.FrameStackObservation,
            stack_size=frame_stack
        ) if frame_stack else None
    ]
    wrappers = [w for w in wrappers if w]

    if episodic_life:
        wrappers.append(EpisodicLifeWrapper)

    kwargs = {'full_action_space': full_action_space}

    if make:
        env = gym.make(env_id, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    else:
        return env_id, wrappers, kwargs


# CLASSES
class EyePatchWrapper(gym.Wrapper):
    """ 
    Custom environment wrapper for 'eye' sacchade movements given
    frames fed from the overarching game environment.

    For this project, it is expected that there is a framestack dimension:
    (F, C, H, W)

    We give (H, W, C), so a dimension is added, but this must be reflected
    in the config YAML file.
    """

    def __init__(
            self, 
            env: gym.Env,
            patch_size: int | tuple[int, int] = 30,
            patch_pos: tuple[int, int] | np.ndarray = (0,0),
            mov_scale: float = 1.0,
            max_sacchades: int = 10,
            sacchade_penalty: float = -0.1
        ):
        super().__init__(env)
        self.game_env = env

        # Perform sanity checks here.
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, tuple):
            self.patch_size = patch_size
        else:
            raise TypeError("patch_size must be either int or tuple[int, int].")

        # The last dim of frame is color channels (RGB)
        self.obs_shape = (1, *self.patch_size, 3)
        self.observation_space = gym.spaces.Box(0, 255, self.obs_shape, dtype=np.uint8)

        if isinstance(patch_pos, tuple):
            patch_pos = np.array(patch_pos)
        if np.issubdtype(patch_pos.dtype, np.integer):
            patch_pos = patch_pos.astype(int)

        # Sanity check initial position
        self.patch_pos = patch_pos

        # Eye movement map from discrete space.
        self.eye_action_to_displacement = {
            0: np.array([0., 0.]),    # NOOP
            1: np.array([1., 0.]),    # DOWN
            2: np.array([-1., 0.]),   # UP
            3: np.array([0., 1.]),    # RIGHT
            4: np.array([0., -1.]),   # LEFT
            5: np.array([1., 1.]),    # DOWN_RIGHT
            6: np.array([-1., 1.]),   # UP_RIGHT
            7: np.array([-1., -1.]),  # UP_LEFT
            8: np.array([1., -1.]),   # DOWN_LEFT
        }

        # Organize action spaces for eye + game
        self.game_action_space = env.action_space
        self.eye_action_space = gym.spaces.Discrete(len(self.eye_action_to_displacement))
        total_actions = self.game_action_space.n + self.eye_action_space.n
        self.action_space = gym.spaces.Discrete(total_actions)

        # Sacchade movement scale
        self.mov_scale = mov_scale

        # Reset iterations
        self.eye_itrs, self.game_itrs = 0, 0

        # Sacchade tracking + penalty
        self.max_sacchades = max_sacchades
        self.sacchade_penalty = sacchade_penalty

    def _get_bounds(self):
        return self.frame.shape[0] - self.patch_size[0], self.frame.shape[1] - self.patch_size[1]

    def _in_bounds(self, pos: np.ndarray):
        br = np.array(self._get_bounds())
        tl = np.array([0,0])
        return np.all((pos >= tl) & (pos < br))

    def _rand_patch_pos(self):
        """ Returns a random position within the frame (accounting for patch size) """
        h, w = self._get_bounds()
        return self.np_random.integers((0, 0), (h, w))

    def _get_patch(self):
        """ Given the location of the top-left position of the patch, return the patch. """
        y, x = self.patch_pos
        h, w = self.patch_size[0], self.patch_size[1]
        return self.frame[y:y+h, x:x+w, :]

    def _update_patch_pos(self, disp: np.ndarray):
        """ Given vector displacement of patch position, update location """
        updated_pos = self.patch_pos + self.mov_scale * disp
        if self._in_bounds(updated_pos):
            self.patch_pos = updated_pos.astype(int)

    def _get_info(self):
        return {
            'patch_pos': self.patch_pos,
            'frame': self.frame,
            'bounds': self._get_bounds(),
            'eye_itrs': self.eye_itrs,
            'game_itrs': self.game_itrs
        }

    def reset(self, seed: int | None = None):
        """ Starts a new episode. """
        super().reset(seed=seed)

        self.eye_itrs, self.game_itrs = 0, 0

        # Reset the game environment + eye postition on frame.
        self.frame, _ = self.game_env.reset()

        if not self._in_bounds(self.patch_pos):
            raise ValueError(f"Initial position {self.patch_pos} must be within bounds: {self._get_bounds()}")

        p = self._get_patch()
        p = np.expand_dims(p, axis=0)
        info = self._get_info()
        return p, info

    def _eye_act(self, action):
        """ Eye sacchade movement. """
        self.eye_itrs += 1
        if self.eye_itrs >= self.max_sacchades:
            print("We've looked around too much and spent too much time. Penalize.")

        di = self.eye_action_to_displacement[action]
        self._update_patch_pos(di)
        obs = self._get_patch()
        penalty = self.sacchade_penalty
        return obs, penalty, False, False

    def _game_act(self, action):
        """ Game actions. """
        self.eye_itrs = 0
        self.game_itrs += 1
        self.frame, rew, term, trunc, _ = self.game_env.step(action)
        obs = self._get_patch()
        return obs, rew, term, trunc

    def step(self, action):
        # Filter which environment we are affecting (eye vs. game)
        eas = self.eye_action_space.n
        act = self._eye_act if action < eas else self._game_act
        action = action if action < eas else action - eas
        
        obs, rew, term, trunc = act(action)
        obs = np.expand_dims(obs, axis=0)
        info = self._get_info()

        return obs, rew, term, trunc, info

class Environment:
    """
    Wrapper class for gym Atari environment + configs.
    """

    def __init__(
        self,
        display_mode: str,
        game_name: str,
        record_dir: str,
        verbose: bool,
        seed: int,
        episode_rec_freq: int,
        wrappers: list,
        wrappers_kwargs: list
    ):
        # Init environment based on display mode (human/rgb_array)
        d_mode = "human" if display_mode == "human" else "rgb_array"
        self.env = gym.make(game_name, render_mode=d_mode)
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

    def step(self, act):
        return self.env.step(act)

    def obs_space(self):
        return self.env.observation_space

    def act_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()

class EpisodicLifeWrapper(gym.Wrapper):
    # different from AtariPreprocessing: real reset only when game over

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.game_over = True

    def _ale_lives(self):
        return self.env.unwrapped.ale.lives()

    def reset(self, **kwargs):
        if self.game_over:
            o, info = self.env.reset(**kwargs)
        else:
            # noop after lost life
            o, _, _, _, info = self.env.step(0)
        self.lives = self._ale_lives()
        return o, info

    def step(self, action):
        next_o, next_r, next_term, next_trunc, info = self.env.step(action)
        self.game_over = next_term or next_trunc or self.game_over
        lives = self._ale_lives()
        if lives < self.lives and lives > 0:
            next_term = True
        self.lives = lives
        return next_o, next_r, next_term, next_trunc, info
