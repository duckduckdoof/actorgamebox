"""
main_sgf.py

Author: Caleb Scott

---

Kick-off file for SGF-like training/eval of agent.
"""

# IMPORTS
import os
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import randomname
import torch
import torch.distributed as dist
from ruamel import yaml

import wandb
from modules_new import envs, utils
from modules_new.actor_critic import ActorCriticPolicy
from modules_new.agent import Agent
from modules_new.trainer import Trainer
from modules_new.wm import WorldModel


# MAIN
def main():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help='Device used for training')
    parser.add_argument("--game", type=str, required=True, help="Choice of Atari game")
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

    # Device, autocast, compile
    if config['more_gpu']:
        print("Multi GPU job selected...")
        l_rank = utils.setup_distributed()
        print(f"Rank: {l_rank}")
        world_size = dist.get_world_size()
        print(f"world size: {world_size}")
        device = torch.device(f"cuda:{l_rank}")
        print(f"Device: {device}")
        compile_ = partial(utils.compile_ddp, disable=not config['compile'], l_rank=l_rank)
        
        # Make sure to reduce size of each buffer per parallel processing group.
        config['trainer']['env_steps'] = config['trainer']['env_steps'] // world_size
    else:
        device = torch.device(args.device)
        compile_ = lambda mod: torch.compile(mod, dynamic=True, disable=not config['compile'])
    autocast = lambda: torch.autocast(device_type=device.type, enabled=config['amp'])

    # W&B setup
    if config['more_gpu']:
        exp_name = randomname.get_name()
        run_id = os.environ.get("TORCH_RUN_ID", "unique_job_id")
        wandb.init(
            project=args.project, 
            mode=args.mode, 
            notes=args.notes, 
            config=config,
            group=f"{exp_name}-{run_id}",
            name=f"gpu-{os.environ.get('RANK', '0')}"
        )
    else:
        wandb.init(
            project=args.project, 
            mode=args.mode, 
            notes=args.notes, 
            config=config
        )
    config = wandb.config

    # Env, policy, agent, wm
    seed = (config.seed + 42) * 27
    rng = utils.seed_all(seed, local_rank=l_rank if l_rank else 0)

    env = envs.atari_env(config.game, make=True, **config.env)

    y_dim = config.wm['y_dim']
    a_dim = env.action_space.n
    policy = ActorCriticPolicy(
        y_dim,
        a_dim,
        config.policy['actor'],
        config.policy['critic'],
        compile_=compile_,
        device=device
    )
    agent = Agent(policy, env.action_space, config.action_stack)
    wm = WorldModel(env.observation_space, agent.stack_act_space, **config.wm, compile_=compile_, device=device)

    # Trainer
    trainer = Trainer(
        env, 
        config.game,
        wm,
        agent,
        seed,
        **config.trainer,
        wm_eval=config.wm_eval,
        agent_eval=config.agent_eval,
        buffer_device=device,
        rng=rng,
        autocast=autocast,
        compile_=compile_
    )

    print(f"Starting... (seed: {seed})")
    print(f"World Model # params: {utils.num_params(wm)}")
    print(f"Agent       # params: {utils.num_params(agent)}")

    # Train agent and WM
    while not trainer.is_finished():
        metrics = trainer.train()

        if len(metrics) > 0:
            wandb.log(metrics, step=trainer.it)

        if trainer.it == 0:
            print("Training...")

    # Save models if we indicated so
    if config.save:
        torch.save(wm.state_dict(), Path(wandb.run.dir) / 'wm.pt')
        torch.save(agent.state_dict(), Path(wandb.run.dir) / 'agent.pt')
        wandb.save('wm.pt')
        wandb.save('agent.pt')
        if config.wm_eval == "decoder":
            torch.save(trainer.wm_trainer.decoder.state_dict(), Path(wandb.run.dir) / 'decoder.pt')
            wandb.save('decoder.pt')

    # Clean up
    trainer.close()
    wandb.finish()
    if config.more_gpu:
        utils.cleanup_devices()

if __name__ == "__main__":
    main()