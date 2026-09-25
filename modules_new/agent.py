"""
agent.py

Author: Caleb Scott

---

Agent wrapper for actor.

"""

# IMPORTS
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from modules_new import envs, utils


# CLASSES
@dataclass
class AgentState:
    action_stack: Optional[Any]

    def map(self, fn):
        """ Apply a function to the state, element-wise """
        action_stack = utils.map_structure(fn, self.action_stack)
        return AgentState(action_stack)

class Agent(nn.Module):
    """ Policy wrapper for agent state """

    def __init__(self, policy, single_action_space, action_stack):
        super().__init__()
        self.policy = policy
        self.single_act_space = single_action_space
        self.stack_act_space = gym.vector.utils.batch_space(single_action_space, action_stack)
        self.act_stack = action_stack

    @torch.no_grad()
    def start(self):
        stack = None
        state = AgentState(stack)
        return state

    @torch.no_grad()
    def _advance_state(self, state, cont_mask, a):
        stack = state.action_stack
        # When the stack is None (on init), make a stack of 0s mimicking the datatype of the action chosen.
        # In the case of atari, '0' as a choice means NOOP.
        if stack is None:
            stack = torch.stack(tuple(torch.zeros_like(a) for _ in range(self.act_stack)), 1)

        # Otherwise, we'll need to shape the mask (init: 1s) to match the size of the stack
        else:
            stack = cont_mask.apply(stack)

        # Add the newly selected action to the stack of copied actions
        # This applies a shift to the action stack, with the most recent action taken at the end.
        stack = torch.cat([stack[:, 1:], a.unsqueeze(1)], 1)
        stacked_a = stack
        state = AgentState(stack)
        return state, stacked_a

    def act_random(self, state, cont_mask, rng):
        batch_size = cont_mask.shape[0]
        a = torch.as_tensor(
            rng.choice(self.single_act_space.n, batch_size),
            dtype=torch.long,
            device=cont_mask.device
        )
        next_state, stacked_a = self._advance_state(state, cont_mask, a)
        return a, stacked_a, next_state

    def act(self, state, cont_mask, x, temperature=1):
        a = self.policy(x, temperature=temperature)
        next_state, stacked_a = self._advance_state(state, cont_mask, a)
        return a, stacked_a, next_state

class AgentTrainer:

    def __init__(self, game, agent, world_model, replay_buffer, batch_size, horizon, policy_trainer,
                eval_env, eval_num_parallel, eval_temperature, eval_epsilon, eval_episodes,
                final_eval_episodes, eval_mode, *, total_its, rng, autocast, compile_=None):
        if eval_mode not in ('none', 'final', 'all'):
            raise ValueError("Agent eval mode must be in (none, final, all)")

        self.agent = agent
        self.world_model = world_model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.horizon = horizon
        self.eval_temp = eval_temperature
        self.eval_eps = eval_epsilon
        self.eval_epi = eval_episodes
        self.final_eval_epi = final_eval_episodes
        self.eval_mode = eval_mode
        self.rng = rng
        self.autocast = autocast

        if eval_mode in ('all', 'final'):
            eval_env_id, eval_env_wrappers, eval_env_kwargs = envs.atari_env(game, make=False, **eval_env)
            self.eval_collector = utils.EpisodeCollector(eval_env_id, eval_env_wrappers, eval_env_kwargs, eval_num_parallel)

        self.policy_trainer = agent.policy.create_trainer(
            policy_trainer,
            total_its=total_its,
            rng=rng,
            autocast=autocast,
            compile_=compile_
        )

    def train(self, it, start_y=None):
        """ Single-iteration training step """

        agent = self.agent
        wm = self.world_model
        buffer = self.replay_buffer
        batch_size = self.batch_size

        wm.eval()
        agent.eval()
        with self.autocast(), torch.no_grad():
            if start_y is None:
                idx = buffer.sample_idx(batch_size, self.rng)
                o = buffer.get(idx, 'next_o')[0]
                start_y = wm.encode(o)
            elif start_y.shape[0] >= batch_size:
                start_y = start_y[:batch_size]
            else:
                idx = buffer.sample_idx(batch_size - start_y.shape[0], self.rng)
                remaining_o = buffer.get(idx, 'next_o')[0]
                remaining_y = wm.encode(remaining_o)
                start_y = torch.cat([start_y, remaining_y], 0)

            ys, as_, _, next_ys, next_rs, next_terms = wm.imagine(agent, self.horizon, start_y)

            ys = torch.stack(ys, 0)
            as_ = torch.stack(as_, 0)
            next_terms = torch.stack(next_terms, 0)
            next_rs = torch.stack(next_rs, 0)

        metrics = self.policy_trainer.train(it, ys, next_ys[-1], as_, next_rs, next_terms)
        return metrics

    @torch.no_grad()
    def evaluate(self, is_final, seed):
        """ Actor evaluation """

        if self.eval_mode not in ('all', 'final') or (self.eval_mode == 'final' and not is_final):
            return {}

        agent = self.agent
        wm = self.world_model
        device = utils.device(agent)
        agent.eval()
        wm.eval()

        rng = utils.rng(seed)

        num_eps = self.final_eval_epi if is_final else self.eval_epi

        @torch.no_grad()
        def policy(os, state, just_done):
            with self.autocast():
                if state is not None:
                    mask = torch.as_tensor(~just_done, device=device)
                    state = state.map(lambda x: x[mask])
                else:
                    state = agent.start()

                n = os.shape[0]
                o = torch.as_tensor(os, device=device)
                y = wm.encode(o)
                cont_mask = utils.get_mask(torch.ones(n, dtype=torch.bool, device=device))
                if rng.random() < self.eval_eps:
                    a, _, state = agent.act_randomly(state, cont_mask, rng)
                else:
                    a, _, state = agent.act(state, cont_mask, y, temperature=self.eval_temp)
                a_cpu = a.cpu().numpy()
                return a_cpu, state

        stats = self.eval_collector.collect_episode_stats(seed, num_eps, policy)

        def aggregate(xs, key):
            return {
                key: np.mean(xs),
                f"{key}_std": np.std(xs),
                f"{key}_min": np.min(xs),
                f"{key}_max": np.max(xs)
            }

        metrics = {
            **aggregate(stats['episode_reward'], 'episode_reward'),
            **aggregate(stats['episode_length'], 'episode_length')
        }

        return metrics

    def close(self):
        self.eval_collector.close()
