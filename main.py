"""
main.py

Author: Caleb Scott

---

Main file for kicking off atari game + actor.

"""

# IMPORTS
import argparse, torch
import torch.nn.functional as F
from logging import Logger
from pprint import pformat
from typing import Any

import configs.globals as globals
import configs.defaults as defaults

from modules.actor import EpsilonScheduler, greedy_epsilon
from modules.environment import Environment
from modules.logging import configure_logger
from modules.memory import CircularReplayBuffer, FrameBuffer
from modules.networks import DQN, desc_network

# FUNCTIONS
def get_cuda_options(lgr: Logger, debug: bool = True):
    if torch.cuda.is_available():
        options = []
        if debug:
            lgr.debug(f"Number of CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            if debug:
                lgr.debug(f"Device ({i}): {torch.cuda.get_device_name(i)}")
                lgr.debug(f"    Properties: {torch.cuda.get_device_properties(i)}")
            options.append(torch.cuda.get_device_name(i))
        lgr.debug("")
        return options
    else:
        if debug:
            lgr.debug("CUDA is not available...")
            lgr.debug("")
        return []

def init_logging(verbose: bool) -> Logger:
    """
    Pull logging config defaults and return logger.
    """
    lgr_defaults = {
        'logger_name': defaults.DEFAULT_LOGGER_NAME,
        'logger_dir': globals.LOGGING_DIR,
        'log_format_str': defaults.DEFAULT_LOG_FMT,
        'log_stdout_str': defaults.DEFAULT_STDOUT_FMT,
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
        "-f", "--episode-rec-freq",
        default=defaults.DEFAULT_EP_REC_FREQ,
        help="How many episodes to wait for recording."
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
    lgr.debug("Initializing environment...")
    lgr.debug(pformat(kwargs, indent=4))
    env = Environment(**kwargs)
    lgr.debug(f"Environment observation space: {env.obs_space()}")
    lgr.debug(f"Environment action space: {env.act_space()}")
    lgr.debug("")

    # Initialization of memory, device, etc. (we will need to refactor this later)
    cudas = get_cuda_options(lgr)
    device = "cuda" if len(cudas) > 0 else "cpu"

    fb = FrameBuffer(defaults.DEFAULT_FRAME_LIMIT)
    obs_shape = (4, *env.obs_space().shape)
    rb = CircularReplayBuffer(
        obs_shape=obs_shape,
        obs_type=env.obs_space().dtype,
        replay_limit=defaults.DEFAULT_BUFFER_SIZE,
        batch_size=defaults.DEFAULT_BATCH_SIZE,
        device=device
    )

    eps_sched = EpsilonScheduler(
        eps_max=defaults.DEFAULT_EPS_MAX,
        eps_min=defaults.DEFAULT_EPS_MIN,
        steps=defaults.DEFAULT_NUM_STEPS,
        exploration_ratio=defaults.DEFAULT_EXPLORATION_FRAC
    )

    episode_rewards = []
    episode_losses = []

    q_net = DQN(stacked_gray_frames=4, action_space=env.act_space().n).to(device)
    desc_network(q_net, lgr)
    optim = torch.optim.Adam(q_net.parameters(), lr=defaults.DEFAULT_ADAM_LR)

    done = False
    episode_loss = 0
    episode_reward = 0.0
    episode_step = 0
    num_episodes = 0

    lgr.debug("Begin simulation...")
    for i in range(defaults.DEFAULT_NUM_STEPS):
        if done or i == 0:
            lgr.debug("Resetting episode...")
            obs, _ = env.reset()

            # Fill frame buffer on new episode...
            for _ in range(4):
                fb.add(obs)

            # Add up losses from the prev episode before reset.
            if i > 0:
                episode_losses.append(episode_loss / episode_step)
                episode_rewards.append(episode_reward)
                episode_loss, episode_reward, episode_step = 0,0,0

            # Reset
            num_episodes += 1
            done = False

        # Determine epsilon, given the step in the training.
        epsilon = eps_sched(i)
        lgr.debug(f"GS({i:8}) E({num_episodes:4}) ES({episode_step:4}) | Epsilon: {epsilon}")

        # Get frames from frame buffer
        frames = fb.get()
        # lgr.debug(f"{episode_step}) Current frames: {frames.shape}")

        # Determine selected action, using epsilon-greedy strat with Q-network
        act = greedy_epsilon(q_net, env.env, frames, epsilon, device)
        lgr.debug(f"GS({i:8}) E({num_episodes:4}) ES({episode_step:4}) | Selected action: {act}")

        # Step the environment, given the selected action.
        next_obs, rew, term, trunc, info = env.env.step(act)
        # lgr.debug(f"{episode_step}) Env: {rew} | {term} | {trunc}")
        done = term or trunc
        fb.add(next_obs)

        # Get next stack of 4 frames (with single updated frame from env step)
        next_frames = fb.get()

        # Add to replay buffer
        rb.add(frames, act, next_frames, rew, done)

        # We have progressed one step within the current episode.
        episode_step += 1

        # Don't train unless the replay buffer has enough data & we're past the threshold.
        if rb.ready() and i >= defaults.DEFAULT_START_TRAINING_STEP:
            # Get batch of obs, act, next_obs, rew, done tuple
            o, a, no, r, d = rb.sample()
            # lgr.debug(f"Sampled from replay buffer: {o.shape}, {a.shape}, {no.shape}, {r.shape}, {d.shape}")
            # lgr.debug(f"Actions: {a[0:5]}")

            # Make a prediction for the action, given the observation
            pred_act = q_net(o)
            pred_next_act_idxs = q_net(no).argmax(dim=1)

            # Use actual actions taken to get the "value" of those actions
            pred_act_q = pred_act.gather(1, a.int().unsqueeze(1)).flatten()
            pred_next_act_q = q_net(no).gather(1, pred_next_act_idxs.unsqueeze(1)).flatten()

            # Discounted rewards, using q-values of actions
            target_q = r + defaults.DEFAULT_DISCOUNT_RATE * (1-d) * pred_next_act_q
            loss = F.smooth_l1_loss(pred_act_q, target_q)
            loss.backward()
            optim.step()
            optim.zero_grad()

            episode_loss += loss.item()
            lgr.debug(f"GS({i:8}) E({num_episodes:4}) ES({episode_step:4}) | Loss: {episode_loss:5f} | Loss/Steps: {episode_loss/episode_step:5f}")

    lgr.debug("Finished! Cleaning up...")
    env.close()

    lgr.debug("Saving model...")
    torch.save(q_net.state_dict(), f"{globals.MODELS_DIR}model_save.pt")

# MAIN
if __name__ == "__main__":
    # Get args
    args = parse()

    # Get any wrappers for env
    args['wrappers'] = defaults.DEFAULT_WRAPPER_STACK
    args['wrappers_kwargs'] = defaults.DEFAULT_WRAPPER_KWARGS

    # Start logger
    lgr = init_logging(args['verbose'])

    # Run the environment!
    run(args, lgr)
