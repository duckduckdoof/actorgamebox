"""
config.py

Author: Caleb Scott

---

Configurations for atari game environment.

"""

# CONSTANTS

# ROM Info
ROMS = [
    "ALE/Adventure-v5",
    "ALE/AirRaid-v5",
    "ALE/Breakout-v5",
    "ALE/Boxing-v5",
    "ALE/CrazyClimber-v5",
    "ALE/Frogger-v5",
    "ALE/MontezumaRevenge-v5",
    "ALE/Pong-v5",
    "ALE/Seaquest-v5",
    "ALE/Surround-v5"
]

DEFAULT_ROM = ROMS[7]

# Gameplay recording
RECORDINGS_DIR = "./recordings/"

# Config file
DEFAULT_CONFIG_FILE = "./configs/default.ini"

# Seed
DEFAULT_SEED = 278691
