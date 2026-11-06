"""
config.py

Author: Caleb Scott

---

Configurations for atari game environment.

Globals include things like paths, or any information to be
made widely available to modules.

"""

from dataclasses import dataclass

# CONSTANTS

# ROM Info
@dataclass
class ROM:
    Adventure_ALE: str                  = "ALE/Adventure-v5"
    Adventure_NoFrameskip: str          = "AdventureNoFrameskip-v4"
    AirRaid_ALE: str                    = "ALE/AirRaid-v5"
    AirRaid_NoFrameskip: str            = "AirRaidNoFrameskip-v4"
    Breakout_ALE: str                   = "ALE/Breakout-v5"
    Breakout_NoFrameskip: str           = "BreakoutNoFrameskip-v4"
    Boxing_ALE: str                     = "ALE/Boxing-v5"
    Boxing_NoFrameskip: str             = "BoxingNoFrameskip-v4"
    CrazyClimber_ALE: str               = "ALE/CrazyClimber-v5"
    CrazyClimber_NoFrameskip: str       = "CrazyClimberNoFrameskip-v4"
    Frogger_ALE: str                    = "ALE/Frogger-v5"
    Frogger_NoFrameskip: str            = "FroggerNoFrameskip-v4"
    MontezumaRevenge_ALE: str           = "ALE/MontezumaRevenge-v5"
    MontezumaRevenge_NoFrameskip: str   = "MontezumaRevengeNoFrameskip-v4"
    Pong_ALE: str                       = "ALE/Pong-v5"
    Pong_NoFrameskip: str               = "PongNoFrameskip-v4"
    Seaquest_ALE: str                   = "ALE/Seaquest-v5"
    Seaquest_NoFrameskip: str           = "SeaquestNoFrameskip-v4"
    Surround_ALE: str                   = "ALE/Surround-v5"
    Surround_NoFrameskip: str           = "SurroundNoFrameskip-v4"

# Gameplay recording
RECORDINGS_DIR = "./recordings/"

# Logging
LOGGING_DIR = "./logs/"

# Tmp catch-all file
TMP_DIR = "./tmp/"

# Models
MODELS_DIR = "./models/"
