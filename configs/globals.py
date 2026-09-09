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
    "AdventureNoFrameskip-v4",
    "ALE/AirRaid-v5",
    "AirRaidNoFrameskip-v4",
    "ALE/Breakout-v5",
    "BreakoutNoFrameskip-v4",
    "ALE/Boxing-v5",
    "BoxingNoFrameskip-v4",
    "ALE/CrazyClimber-v5",
    "CrazyClimberNoFrameskip-v4",
    "ALE/Frogger-v5",
    "FroggerNoFrameskip-v4",
    "ALE/MontezumaRevenge-v5",
    "MontezumaRevengeNoFrameskip-v4",
    "ALE/Pong-v5",
    "PongNoFrameskip-v4",
    "ALE/Seaquest-v5",
    "SeaquestNoFrameskip-v4",
    "ALE/Surround-v5",
    "SurroundNoFrameskip-v4"
]

# Gameplay recording
RECORDINGS_DIR = "./recordings/"

# Logging
LOGGING_DIR = "./logs/"

# Tmp catch-all file
TMP_DIR = "./tmp/"
