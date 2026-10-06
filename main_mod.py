"""
main_sgf.py

Author: Caleb Scott

---

Kick-off file for SGF-like training/eval of agent.
"""

# IMPORTS
import pprint
from argparse import ArgumentParser
from pathlib import Path

import torch
from ruamel import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from configs import globals as g
from modules import environment as e
from modules_new import utils
from modules_new.actor_critic import ActorCriticPolicy
from modules_new.agent import Agent
from modules_new.trainer import Trainer
from modules_new.wm import WorldModel


# MAIN
def main(game=g.ROMS[15]):
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help='Device used for training')
    parser.add_argument("--game", type=str, required=True, default=game, help="Choice of Atari game")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--mode", type=str, nargs='?', help="W&B mode")
    parser.add_argument("--project", type=str, nargs='?', help="W&B project")
    parser.add_argument("--notes", type=str, nargs='?', help="W&B notes")
    parser.add_argument("--wm_eval", type=str, default='none', help="World model evaluation: 'none', 'decoder'")
    parser.add_argument("--agent_eval", type=str, default='all', help="Agent model evaluation: 'none', 'all', 'final'")
    parser.add_argument("--amp", default=False, action='store_true', help="Use automatic mixed precision training")
    parser.add_argument("--compile", default=False, action='store_true', help="Whether to use torch.compile")
    parser.add_argument("--save", default=False, action='store_true', help="Save the model after training.")
    parser.add_argument("--more_gpu", default=False, action='store_true', help="Enable distributed parallel gpu use.")
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
        'wm_eval': args.wm_eval,
        'agent_eval': args.agent_eval,
        'amp': args.amp,
        'compile': args.compile,
        'save': args.save,
        'more_gpu': args.more_gpu
    }

    # W&B setup
    wandb.init(project=args.project, mode=args.mode, notes=args.notes, config=config)
    config = wandb.config

    print("Config:")
    pprint.pp(config.as_dict())
    print()

    # Device, autocast, compile
    if config.more_gpu:
        l_rank = utils.setup_distributed()
        device = torch.device(f"cuda:{l_rank}")
        compile_ = lambda mod: torch.compile(DDP(mod, device_ids=[l_rank]), dynamic=True, disable=not config.compile)
        
        # Make sure to reduce size of each buffer per parallel processing group.
        config.trainer.env_steps = config.trainer.env_steps // len(device)
    else:
        device = torch.device(args.device)
        compile_ = lambda mod: torch.compile(mod, dynamic=True, disable=not config.compile)
    autocast = lambda: torch.autocast(device_type=device.type, enabled=config.amp)

    # Env, policy, agent, wm
    seed = (config.seed + 42) * 27
    rng = utils.seed_all(seed, local_rank=l_rank if l_rank else 0)
    
    # Create both game env and sacchade env
    genv = e.simple_atari_env(config.game, **config.game_env)
    env = e.EyePatchEnv(screen_env=genv, **config.eye_env)

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}\n")

    y_dim = config.wm['y_dim']
    a_dim = env.action_space.n
    g_policy = ActorCriticPolicy(
        y_dim,
        a_dim,
        config.policy['actor'],
        config.policy['critic'],
        compile_=compile_,
        device=device
    )
    g_agent = Agent(g_policy, env.action_space, config.action_stack)
    wm = WorldModel(env.observation_space, g_agent.stack_act_space, **config.wm, compile_=compile_, device=device)

    # Trainer
    trainer = Trainer(
        env, 
        config.game,
        wm,
        g_agent,
        seed,
        **config.trainer,
        wm_eval=config.wm_eval,
        agent_eval=config.agent_eval,
        buffer_device=device,
        rng=rng,
        autocast=autocast,
        compile_=compile_
    )

    print(f"Starting... (seed: {seed})\n")
    print(f"World Model # params: {utils.num_params(wm)}")
    print(f"Game Agent  # params: {utils.num_params(g_agent)}")

    # Train agent and WM
    while not trainer.is_finished():
        metrics = trainer.train()

        if len(metrics) > 0:
            wandb.log(metrics, step=trainer.it)

        if trainer.it == 0:
            print("Training...\n")

    # Save models if we indicated so
    if config.save:
        torch.save(wm.state_dict(), Path(wandb.run.dir) / 'wm.pt')
        torch.save(g_agent.state_dict(), Path(wandb.run.dir) / 'g_agent.pt')
        wandb.save('wm.pt')
        wandb.save('g_agent.pt')
        if config.wm_eval == "decoder":
            torch.save(trainer.wm_trainer.decoder.state_dict(), Path(wandb.run.dir) / 'decoder.pt')
            wandb.save('decoder.pt')

    # Clean up
    trainer.close()
    wandb.finish()
    utils.cleanup_devices()

if __name__ == "__main__":
    main()