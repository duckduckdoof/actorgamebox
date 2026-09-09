"""
main.py

Author: Caleb Scott

---

Main file for kicking off atari game + actor.

"""

# IMPORTS
import argparse
from logging import Logger
from typing import Any

import configs.globals as globals
import configs.defaults as defaults

from modules.environment import Environment
from modules.logging import configure_logger

# FUNCTIONS
def init_logging(verbose: bool) -> Logger:
    """
    Pull logging config defaults and return logger.
    """
    lgr_defaults = {
        'logger_name': defaults.DEFAULT_LOGGER_NAME,
        'logger_dir': globals.LOGGING_DIR,
        'log_format_str': defaults.DEFAULT_LOG_FMT,
        'log_datetime_prefix': defaults.DEFAULT_LOG_DATETIME_PREFIX,
        'datetime_format': defaults.DEFAULT_DATETIME,
        'verbose': verbose
    }
    return configure_logger(**lgr_defaults)

def parse() -> dict[str, Any]:
    """
    Get all arguments necessary before running.
    """
    parser = argparse.ArgumentParser(
        description="argparse for Atari games testbed."
    )

    # Args
    parser.add_argument(
        "-g", "--game_name",
        choices=globals.ROMS,
        default=defaults.DEFAULT_ROM,
        help=f"The atari game to choose (default: {defaults.DEFAULT_ROM})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Output to stdout."
    )
    parser.add_argument(
        "-d", "--display-mode",
        choices=["human", "record"],
        default="human",
        help="Either show in real-time or record progress."
    )
    parser.add_argument(
        "-r", "--record-dir",
        default=globals.RECORDINGS_DIR,
        help="Don't show gameplay."
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["train", "test"],
        default="train",
        help="Train/Test the actor for the game."
    )
    parser.add_argument(
        "--seed",
        default=defaults.DEFAULT_SEED,
        help="Default seed for controlling randomness."
    )

    args = parser.parse_args()
    args_dict = vars(args)

    print("Loading atari gamebox settings...")

    # Pass the args.
    return args_dict

def run(kwargs: dict, lgr: Logger):
    """
    Run the environment.
    """
    # Create the environment
    lgr.info("Initializing environment...")
    lgr.info(kwargs)
    e = Environment(**kwargs)

    lgr.info("running 10 episodes...")
    for i in range(10):
        obs, info = e.reset()
        done = False

        lgr.info(f"Running episdode: {i}...")
        while not done:
            act = e.env.action_space.sample()
            obs, rew, term, trunc, info = e.env.step(act)
            done = term or trunc
            lgr.info(f"  {act} | {rew}: Done: {done}")

    lgr.info("Finished! Cleaning up...")
    e.env.close()

# MAIN
if __name__ == "__main__":
    args = parse()
    lgr = init_logging(args['verbose'])
    run(args, lgr)
