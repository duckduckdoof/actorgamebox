"""
trainer.py

Author: Caleb Scott

---

Wrapper Trainer class for both WM + Actor

"""

# IMPORTS
import time

import replay_buffer as rb
import torch
import utils
from agent import AgentTrainer
from wm import WorldModelTrainer


# CLASSES
class Trainer:
    """ Trainer class for both WM and Actor """

    def __init__(self, env, game, wm, agent, seed, env_steps, init_steps, env_eps, env_temp,
                wm_every, agent_every, log_every, eval_every, wm_trainer, agent_trainer,
                wm_eval, agent_eval, buffer_device, *, rng, autocast, compile_):
        self.env = env
        self.wm = wm
        self.agent = agent
        self.seed = seed
        self.env_steps = env_steps
        self.init_steps = init_steps
        self.env_eps = env_eps
        self.env_temp = env_temp
        self.wm_every = wm_every
        self.agent_every = agent_every
        self.log_every = log_every
        self.eval_every = eval_every
        self.rng = rng
        self.autocast = autocast

        # Init replay buffer
        start_o, _ = env.reset(seed=seed)
        replay_buffer = rb.ReplayBuffer(
            env.observation_space,
            agent.stack_act_space,
            env_steps,
            start_o,
            device=buffer_device
        )
        self.replay_buffer = replay_buffer

        # WM + Agent Trainers
        self.wm_trainer = WorldModelTrainer(
            wm, replay_buffer, **wm_trainer, init_steps=init_steps, eval_mode=wm_eval,
            total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
        )
        self.agent_trainer = AgentTrainer(
            game, agent, wm, replay_buffer, **agent_trainer,
            eval_mode=agent_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
        )

        self.it = -1
        self.wm_it = 0
        self.agent_it = 0
        self.wm_agg = utils.Aggregator(op='mean')
        self.agent_agg = utils.Aggregator(op='mean')
        self.train_time = 0.0
        self.wm_time = 0.0
        self.agent_time = 0.0
        self.last_log = -1
        self.last_eval = -1

        # Init the agent state + replay buffer of tuples
        agent.eval()
        with autocast(), torch.no_grad():
            self.agent_state = agent.start()
            self.cont_mask = utils.get_mask(torch.ones(1, dtype=torch.bool, device=replay_buffer.device))
            for _ in range(init_steps):
                a, stacked_a, self.agent_state = agent.act_random(self.agent_state, self.cont_mask, self.rng)
                self._env_step(a, stacked_a)

    def close(self):
        self.agent_trainer.close()

    def is_finished(self):
        return self.it >= (self.env_steps - self.init_steps - 1)

    def _env_step(self, a, stacked_a):
        env = self.env
        next_o, next_r, next_term, next_trunc, _ = env.step(a.cpu().item())
        cont_o = next_o
        if next_term or next_trunc:
            cont_o, _ = env.reset()
        next_r, next_term, next_trunc, next_o = self.replay_buffer.step(
            stacked_a, next_o, next_r, next_term, next_trunc, cont_o
        )
        self.cont_mask = utils.get_mask(~(next_term | next_trunc))

    def train(self):
        """ Train both the actor + wm in one iteration """

        wm, agent, replay_buffer = self.wm, self.agent, self.replay_buffer

        it = self.it + 1

        agent.eval()
        wm.eval()
        with self.autocast(), torch.no_grad():
            o = self.replay_buffer.cont_o
            y = wm.encode(o)
            if self.rng.random() < self.env_eps:
                a, stacked_a, self.agent_state = agent.act_random(self.agent_state, self.cont_mask, self.rng)
            else:
                a, stacked_a, self.agent_state = agent.act(self.agent_state, self.cont_mask, y, temperature=self.env_temp)

        # Take one env step
        self._env_step(a, stacked_a)

        # Train the WM by one step
        start_time = time.time()
        start_y = None
        while self.wm_it <= it:
            wm_metrics, start_y = self.wm_trainer.train(it)
            self.wm_agg.append(wm_metrics)
            self.wm_it += self.wm_every
        wm_end_time = time.time()

        # Train the agent
        while self.agent_it <= it:
            agent_metrics = self.agent_trainer.train(it, start_y)
            self.agent_agg.append(agent_metrics)
            self.agent_it += self.agent_every
        end_time = time.time()

        self.train_time += (end_time - start_time)
        self.wm_time += (wm_end_time - start_time)
        self.agent_time += (end_time - wm_end_time)

        metrics = {}
        is_first = it == 0
        is_final = (it == self.env_steps - self.init_steps - 1)

        if is_first or is_final or (it - self.last_log >= self.log_every):
            stats = {
                **replay_buffer.get_stats(),
                'train_time': self.train_time,
                'wm_time': self.wm_time,
                'agent_time': self.agent_time
            }
            metrics.update({f"stats/{k}": v for k, v in stats.items()})
            metrics.update({f"wm/{k}": v for k, v in self.wm_agg.aggregate().items()})
            metrics.update({f"agent/{k}": v for k, v in self.agent_agg.aggregate().items()})
            self.last_log = it

        if is_first or is_final or (it - self.last_eval >= self.eval_every):
            wm.eval()
            agent.eval()
            wm_eval_metrics = self.wm_trainer.evaluate(agent, self.seed)
            agent_eval_metrics = self.agent_trainer.evaluate(is_final, self.seed)
            metrics.update({f"eval/{k}": v for k, v in wm_eval_metrics.items()})
            metrics.update({f"eval/{k}": v for k, v in agent_eval_metrics.items()})
            self.last_eval = it

        self.it = it
        return metrics
