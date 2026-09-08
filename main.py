"""
main.py

Author: Caleb Scott

---

Main file for kicking off atari game + actor.

"""

# IMPORTS
import argparse

from config import ROMS, DEFAULT_ROM, RECORDINGS_DIR

# CONSTANTS

# FUNCTIONS

# CLASSES

# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="argparse for Atari games testbed."
    )

    # Args
    parser.add_argument(
        "-g", "--game_name",
        choices=ROMS,
        default=DEFAULT_ROM,
        help=f"The atari game to choose (default: {DEFAULT_ROM})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="More verbose output."
    )
    parser.add_argument(
        "-r", "--record-at",
        default=RECORDINGS_DIR,
        help="Store replays at this location."
    )
    parser.add_argument(
        "-n", "--no-gameplay",
        action="store_true",
        help="Don't show gameplay."
    )

    args = parser.parse_args()
