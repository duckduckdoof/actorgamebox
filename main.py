"""
main.py

Author: Caleb Scott

---

Main file for kicking off atari game + actor.

"""

# IMPORTS
import argparse
from pprint import pprint

import configs.config as cfg

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
        choices=cfg.ROMS,
        default=cfg.DEFAULT_ROM,
        help=f"The atari game to choose (default: {cfg.DEFAULT_ROM})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="More verbose output."
    )
    parser.add_argument(
        "-r", "--recording",
        default=cfg.RECORDINGS_DIR,
        help="Store replays at this location."
    )
    parser.add_argument(
        "-n", "--no-gameplay",
        action="store_true",
        help="Don't show gameplay."
    )
    parser.add_argument(
        "-c", "--config-file",
        default=cfg.DEFAULT_CONFIG_FILE,
        help="Name of config file."
    )
    parser.add_argument(
        "-u", "--use-config",
        action="store_true",
        help="Use config file instead of parse args."
    )

    args = parser.parse_args()
    args_dict = vars(args)

    print("Loading atari gamebox settings...")

    # Load from config file, if indicated
    if args.use_config:
        pass

    if args.verbose:
        print("Arguments:")
        pprint(args_dict)

    # Pass the args.
