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
from environment import Environment

# FUNCTIONS
def parse():
    """
    Get all arguments necessary before running.
    """
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
        "-d", "--display-mode",
        choices=["human", "record"],
        default="human",
        help="Store replays at this location."
    )
    parser.add_argument(
        "-r", "--record-dir",
        default=cfg.RECORDINGS_DIR,
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
    parser.add_argument(
        "-m", "--mode",
        choices=["train", "test"],
        default="train",
        help="Train/Test the actor for the game."
    )
    parser.add_argument(
        "--seed",
        default=cfg.DEFAULT_SEED,
        help="Default seed for controlling randomness."
    )

    args = parser.parse_args()
    args_dict = vars(args)

    print("Loading atari gamebox settings...")

    # Load from config file, if indicated
    if args.use_config:
        pass
    else:
        del args_dict['use_config']
        del args_dict['config_file']

    if args.verbose:
        print("Arguments:")
        pprint(args_dict)

    # Pass the args.
    return args_dict

def run(kwargs: dict):
    """
    Run the environment.
    """
    # Create the environment
    e = Environment(**kwargs)

    for _ in range(10):
        obs, info = e.env.reset()
        done = False

        while not done:
            act = e.env.action_space.sample()
            obs, rew, term, trunc, info = e.env.step(act)
            done = term or trunc

    e.env.close()

# MAIN
if __name__ == "__main__":
    args = parse()
    run(args)
