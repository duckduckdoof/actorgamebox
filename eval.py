"""
eval.py

Author: Caleb Scott

---

Evaluation of trained models.
For the purpose of seeing the frame + focus.

"""

# IMPORTS
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import torch
from ruamel import yaml

from configs.globals import ROM
from modules import environment as e
from modules import utils
from modules.actor_critic import ActorCriticPolicy
from modules.agent import Agent
from modules.wm import WorldModel


# MAIN
def main(game=ROM.Pong_NoFrameskip):
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help='Device used for training')
    parser.add_argument("--game", type=str, default=game, help="Choice of Atari game")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--compile", default=False, action='store_true', help="Whether to use torch.compile")
    parser.add_argument("--dir", type=str, default="models/pong/base/", help="Base dir for models.")
    parser.add_argument("--num", type=int, default=1, help="Default number of episodes.")
    args = parser.parse_args()

    # Load from config file
    with open(args.config, 'r') as f:
        config = yaml.YAML(typ='safe', pure=True).load(f)

    # Update config with command line args
    config = {
        **config, 
        'config': args.config, 
        'game': args.game,
        'seed': args.seed,
        'compile': args.compile,
        'dir': args.dir,
        'num': args.num
    }
    config = SimpleNamespace(**config)
    # print(f"Config: {config}")

    # Device, autocast, compile
    device = torch.device(args.device)
    compile_ = lambda mod: torch.compile(mod, dynamic=False, disable=not config.compile)

    # Env, policy, agent, wm
    seed = (config.seed + 42) * 27
    utils.seed_all(seed)
    
    # Create eye-sacchade game wrapper env
    print("Creating env...")
    env_selection = e.eye_env
    env = env_selection(config.game, **config.env)

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}\n")

    y_dim = config.wm['y_dim']
    a_dim = env.action_space.n

    print("Making Agent/WM...")
    policy = ActorCriticPolicy(
        y_dim,
        a_dim,
        config.policy['actor'],
        config.policy['critic'],
        compile_=compile_,
        device=device
    )
    agent = Agent(policy, env.action_space, config.action_stack)
    wm = WorldModel(
        env.observation_space, 
        agent.stack_act_space, 
        **config.wm, 
        compile_=compile_, 
        device=device
    )

    print("Importing WM weights...")
    wm_path = Path(config.dir + "wm.pt")
    wm_checkpoint = torch.load(wm_path, map_location=device, weights_only=True) 
    clean_wm_checkpoint = {
        key.replace("_orig_mod.", ""): value
        for key, value in wm_checkpoint.items()
    }
    wm.load_state_dict(clean_wm_checkpoint, strict=True)

    print("Importing Agent weights...")
    ag_path = Path(config.dir + "agent.pt")
    ag_checkpoint = torch.load(ag_path, map_location=device, weights_only=True) 
    clean_ag_checkpoint = {
        key.replace("_orig_mod.", ""): value
        for key, value in ag_checkpoint.items()
    }
    agent.load_state_dict(clean_ag_checkpoint, strict=True)

    print("Putting into eval mode and running...")

    wm.eval()
    agent.eval()

    stats = []
    frame_data = []

    with torch.no_grad():
        for i in range(config.num):
            # Observation
            o, _ = env.reset()
            done = False
            rew = 0.0
            data = []

            encode = wm.encode
            policy = agent.policy

            while not done:
                o = torch.tensor(o, dtype=torch.uint8, device=device).unsqueeze(0)
                y = encode(o)
                a = policy(y)

                # Observation, reward, termination, truncation, info
                o, reward, term, trunc, info = env.step(a.item())
                data.append(info)
                done = term or trunc
                rew += reward

            print(f"Episode {i} final reward: {rew}")
            stats.append({
                "itr": i,
                "total reward": rew
            })
            frame_data.append(data)

if __name__ == "__main__":
    main()