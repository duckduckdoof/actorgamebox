"""
defaults.py

Author: Caleb Scott

---

Default config settings for a run.

Modify this if you don't want to bother with loquacious argparse args.

"""

# IMPORTS
from configs.globals import ROMS

import gymnasium.wrappers as wrp

# CONSTANTS

# Rom selection
DEFAULT_ROM = ROMS[14]

# Seed
DEFAULT_SEED = 278691

# Log formatting
DEFAULT_LOGGER_NAME = "actorgb"
DEFAULT_LOG_FMT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_STDOUT_FMT = "%(message)s"
DEFAULT_DATETIME = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DATETIME_PREFIX = "%Y-%m-%d_%H:%M"

# Wrappers for environment
DEFAULT_WRAPPER_STACK = [
    wrp.AtariPreprocessing
]
# Corresponding kwargs for wrapper stack
DEFAULT_WRAPPER_KWARGS = [
    {'grayscale_obs': True, 'screen_size': 84, 'scale_obs': True, 'frame_skip': 1},
]

# Env/misc params
DEFAULT_FRAME_LIMIT = 4
DEFAULT_ADAM_LR = 2e-4

# Training an actor
DEFAULT_NUM_STEPS = 1_000_000
DEFAULT_START_TRAINING_STEP = 500
DEFAULT_BUFFER_SIZE = 500_000
DEFAULT_BATCH_SIZE = 64
DEFAULT_UPDATE_FREQ = 1000
DEFAULT_EPS_MAX = 1.0
DEFAULT_EPS_MIN = 0.01
DEFAULT_EXPLORATION_FRAC = 0.2
DEFAULT_DISCOUNT_RATE = 0.98
