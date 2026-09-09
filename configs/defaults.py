"""
defaults.py

Author: Caleb Scott

---

Default config settings for a run.

Modify this if you don't want to bother with loquacious argparse args.

"""

# IMPORTS
from configs.globals import ROMS

# CONSTANTS

# Rom selection
DEFAULT_ROM = ROMS[15]

# Seed
DEFAULT_SEED = 278691

# Log formatting
DEFAULT_LOGGER_NAME = "actorgb"
DEFAULT_LOG_FMT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_DATETIME = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DATETIME_PREFIX = "%Y-%m-%d_%H:%M"
