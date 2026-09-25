"""
wm.py

Author: Caleb Scott

---

World Model description.

"""

# IMPORTS
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torchvision.utils import make_grid

from modules_new import nets, utils


# CLASSES
class WorldModel(nn.Module):
    """ World model for the actor, of the environment """

    def __init__(self, obs_space, stack_act_space, y_dim, z_dim, encoder, projector, predictor,
                trans_predictor, rew_predictor, term_predictor, contrastive, sim_coef, var_coef, cov_coef,
                rew_coef, term_coef, *, compile_, device=None):
        super().__init__()

        if not (isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 4):
            raise ValueError('Observation Space is not supported')

        if not isinstance(stack_act_space, gym.spaces.MultiDiscrete):
            raise TypeError('Action space is not supported')

        self.obs_space = obs_space
        self.stack_act_space = stack_act_space
        flat_stack_act_space = gym.spaces.flatten_space(stack_act_space)

        num_frames, h, w, c = obs_space.shape
        o_dim = num_frames * c
        o_res = h
        a_dim = flat_stack_act_space.shape[0]

        self.o_dim = o_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.encoder = compile_(nn.Sequential(*nets.cnn(o_dim, o_res, y_dim, **encoder, memory_format=torch.channels_last, device=device)))
        self.projector = compile_(nets.VectorMLP(y_dim, z_dim, **projector, device=device))
        self.predictor = compile_(nets.VectorMLP(z_dim + a_dim, y_dim, **predictor, device=device))
        self.trans_predictor = compile_(nets.VectorMLP(y_dim + a_dim, y_dim, **trans_predictor, device=device))
        self.rew_predictor = compile_(nets.ScalarMLP(2 * y_dim + a_dim, **rew_predictor, device=device))
        self.term_predictor = compile_(nets.ScalarMLP(2 * y_dim + a_dim, **term_predictor, device=device))

        self.contrastive = contrastive
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.rew_coef = rew_coef
        self.term_coef = term_coef

    def representation_modules(self):
        return (self.encoder, self.projector, self.predictor, self.rew_predictor, self.term_predictor)

    def transition_modaules(self):
        return (self.trans_predictor,)

    @torch.no_grad()
    def preprocess(self, o, dtype=None):
        if dtype is None:
            dtype = torch.float

        # (B, F, H, W, C) -> (B, F * C, H, W) F=num frames
        o = o.detach().permute(0, 1, 4, 2, 3).flatten(1,2)

        if o.dtype != dtype:
            was_uint8 = (o.dtype == torch.uint8)
            o = o.to(dtype=dtype, memory_format=torch.channels_last)
            if was_uint8:
                o = o / 255
        else:
            o = o.to(memory_format=torch.channels_last)
        return o

    @torch.no_grad()
    def flatten_actions(self, stacked_a, dtype=None):
        if dtype is None:
            dtype = torch.float
        flat_a = utils.space_flatten(self.stack_act_space, stacked_a, dtype=dtype)
        return flat_a

    @torch.no_grad()
    def encode(self, o, dtype=None):
        o = self.preprocess(o, dtype=dtype)
        y = self.encoder(o)
        return y

    @torch.no_grad()
    def imagine(self, agent, horizon, start_y, start_cont_mask=None):
        assert not self.training

        y = start_y
        if start_cont_mask is not None:
            y = start_cont_mask.apply(y)
        agent_state = agent.start()
        cont_mask = start_cont_mask
        history = []

        for t in range(horizon):
            a, stacked_a, next_agent_state = agent.act(agent_state, cont_mask, y)
            flat_a = self.flatten_actions(stacked_a, dtype=y.dtype)
            inp = torch.cat([y, flat_a], -1)

            # Skip connection residuals
            next_y = y + self.transition_predictor(inp)

            inp = torch.cat([y, flat_a, next_y], -1)
            next_r = self.rew_predictor(inp)
            next_term = self.term_predictor(inp)

            history.append([y, a, stacked_a, next_y, next_r, next_term])

            next_cont_mask = utils.get_mask(1 - next_term.float())
            y = next_cont_mask.apply(next_y)
            agent_state = next_agent_state
            cont_mask = next_cont_mask

        return tuple(map(tuple, zip(*history)))

    # By this time, we have preprocessed observations from B, F, C, H, W to
    # B, F * C, H, W
    def representation_loss(self, ot, next_ot, flat_a, next_r, next_term):
        """ Loss in the predicted representation of the WM. """

        assert self.training

        yt = self.encoder(ot)
        z = self.projector(yt)

        next_yt = self.encoder(next_ot)
        next_z = self.projector(next_yt)

        # var/covar regularization terms
        var_loss1, cov_loss1, std1 = utils.var_covar_loss(z, contrastive=self.contrastive)
        var_loss2, cov_loss2, std2 = utils.var_covar_loss(next_z, contrastive=self.contrastive)
        var_loss = (var_loss1 + var_loss2) / 2
        cov_loss = (cov_loss1 + cov_loss2) / 2

        # Similarity between predicted z and actual z
        inp = torch.cat([z, flat_a], -1)
        pred_z = self.predictor(inp)
        sim_loss = self.predictor.loss(pred_z, next_z)

        inp = torch.cat([yt, flat_a, next_yt], -1)
        reward_stats = self.rew_predictor.get_stats(inp, full_precision=True)
        reward_loss = self.rew_predictor.loss(reward_stats, next_r)
        term_stats = self.term_predictor.get_states(inp, full_precision=True)
        term_loss = self.term_predictor.loss(term_stats, next_r)

        repr_loss = self.sim_coef * sim_loss + self.var_coef * var_loss + self.cov_coef * cov_loss + \
            self.rew_coef * reward_loss + self.term_coef * term_loss

        pred_r = self.rew_predictor.predict(reward_stats)
        pred_term = self.term_predictor.predict(term_stats)
        if pred_term.dtype != torch.bool:
            pred_term = pred_term > 0.5

        metrics = {
            'z_sim_loss': sim_loss,
            'z_var_loss': var_loss,
            'z_cov_loss': cov_loss,
            'reward_loss': reward_loss,
            'terminal_loss': term_loss,
            'representation_loss': repr_loss,
            'z_std': (pred_r - next_r).abs().mean(),
            'terminal_acc': (pred_term == next_term).float().mean()
        }
        return repr_loss, metrics, yt, next_yt

    def transition_loss(self, y, flat_a, next_y):
        """ Transition loss for model """
        inp = torch.cat([y, flat_a], -1)

        # Another skip connection residual
        pred_y = y + self.trans_predictor(inp)
        trans_loss = self.trans_predictor.loss(pred_y, next_y)

        metrics = {
            'transition_loss': trans_loss,
            'y_mse': F.mse_loss(y, next_y),
            'y_norm': (torch.linalg.vector_norm(y, dim=-1).mean() +
                torch.linalg.vector_norm(next_y, dim=-1).mean()) / 2,
            'transition_mae': F.l1_loss(pred_y, next_y)
        }
        return trans_loss, metrics

class WorldModelTrainer:

    def __init__(self, wm, replay_buffer, batch_size, augmentation, repr_optim, trans_optim,
                debug, init_steps, eval_mode, total_its, rng, *, autocast, compile_):

        if eval_mode not in ('none', 'decoder'):
            raise ValueError("World Model eval_mode must be: (none, decoder)")

        self.wm = wm
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.augmentation = compile_(nets.augmentation(augmentation))

        self.repr_optim = nets.Optimizer(nn.ModuleList(wm.representation_modules()), **repr_optim, total_its=total_its, autocast=autocast)
        self.trans_optim = nets.Optimizer(nn.ModuleList(wm.representation_modules()), **trans_optim, total_its=total_its, autocast=autocast)

        self.init_steps = init_steps
        self.eval_mode = eval_mode
        self.total_its = total_its
        self.dream_horizon = debug['dream_horizon']
        self.rng = rng
        self.autocast = autocast

        if eval_mode == 'decoder':
            device = next(wm.parameters()).device
            self.decoder = compile_(nn.Sequential(*nets.transpose_cnn(
                wm.y_dim, wm.o_dim, **debug['decoder'], memory_format=torch.channels_last, device=device
            )))
            self.decoder_optim = nets.Optimizer(self.decoder, **debug['optimizer'], total_its=total_its, autocast=autocast)
        elif eval_mode != 'none':
            raise ValueError("Invalid eval mode")

        self._optimize = compile_(self._optimize)
        self._optimize_decoder = compile_(self._optimize_decoder)

    # Remember that we have B, F, C, H, W for our observation.
    def _optimize(self, o, stacked_a, next_r, next_term, next_o, it):
        wm = self.wm

        autocast = self.autocast()
        with autocast:
            with torch.no_grad():
                dtype = torch.half if autocast._enabled else torch.float

                o = wm.preprocess(o, dtype=dtype)
                next_o = wm.preprocess(next_o, dtype=dtype)

                ot = self.augmentation(o)
                next_ot = self.augmentation(next_o)

                # Actions are two-D in the stack (ex: [[0, 0, 3, 4]])
                flat_a = wm.flatten_actions(stacked_a, dtype=dtype)

            repr_loss, repr_loss_metrics, yt, next_yt = wm.representation_loss(ot, next_ot, flat_a, next_r, next_term)

        # Representation loss
        with self.autocast():
            if isinstance(self.augmentation, nn.Identity):
                y = yt.detach()
                next_y = next_yt.detach()
            else:
                wm.encoder.eval()
                with torch.no_grad():
                    y = wm.encoder(o)
                    next_y = wm.encoder(next_o)

            trans_loss, trans_loss_metrics = wm.transition_loss(y, flat_a, next_y)

        # Optimize WM
        self.repr_optim.step(repr_loss, self.batch_size, it)
        self.trans_optim.step(trans_loss, self.batch_size, it)
        metrics = {**repr_loss_metrics, **trans_loss_metrics}
        return metrics, y

    def _optimize_decoder(self, o, y, it):
        autocast = self.autocast()
        with autocast:
            dtype = torch.half if autocast._enabled else torch.float
            o = self.wm.preprocess(o, dtype=dtype)
            recon = self.decoder(y)
            loss = F.mse_loss(recon, o, reduction='none').sum([-3, -2, -1]).mean()
            metrics = {'decoder_loss': loss}
        self.decoder_optim.step(loss, o.shape[0], it)
        return metrics

    def train(self, it):
        """ Train for a single iteration of the world model. """
        idx = self.replay_buffer.sample_idx(self.batch_size, self.rng)
        o, stacked_a, next_r, next_term, next_o = self.replay_buffer.get(idx, 'o', 'a', 'next_r', 'next_term', 'next_o')

        self.wm.train()
        metrics, y = self._optimize(o, stacked_a, next_r, next_term, next_o, it)

        if self.eval_mode == 'decoder':
            self.decoder.train()
            decoder_metrics = self._optimize_decoder(o, y, it)
            metrics.update(decoder_metrics)

        return metrics, y

    @torch.no_grad()
    def evaluate(self, agent, seed):
        """ Evaluation for the WM """
        if self.eval_mode != 'decoder':
            return {}

        wm, decoder, replay_buffer = self.wm, self.decoder, self.replay_buffer
        metrics = {}

        eval_rng = np.random.Generator(np.random.PCG64(seed))

        wm.eval()
        decoder.eval()
        with self.autocast():
            # Select some fixed, some random observations from buffer, and visualize using decoder
            num_obs = 3
            fixed_idx = torch.linspace(0, self.init_steps, num_obs).long()
            fixed_o, fixed_term, fixed_trunc = replay_buffer.get(fixed_idx, 'next_o', 'next_term', 'next_trunc')
            random_idx = eval_rng.choice(len(replay_buffer), num_obs, replace=False)
            random_o, random_term, random_trunc = replay_buffer.get(random_idx, 'next_o', 'next_term', 'next_trunc')
            o = torch.cat([fixed_o, random_o], 0)
            term = torch.cat([fixed_term, random_term], 0)
            trunc = torch.cat([fixed_trunc, random_trunc], 0)
            cont_mask = utils.get_mask(~(term | trunc))

            y = wm.encode(o)
            ohat = decoder(y)

            num_frames = wm.obs_space.shape[0]
            o = wm.preprocess(o, dtype=ohat.dtype)
            o, ohat = [x.unflatten(1, (num_frames, -1)) for x in (o, ohat)]

            # Visualize reconstructions
            img = torch.cat([utils.viz_obs(o), utils.viz_obs(ohat.clamp(0., 1.0))], 0)
            img = make_grid(img, nrow=num_obs * 2)
            img = (img.permute(1, 2, 0) * 255.).byte().cpu().numpy()
            metrics['recons'] = wandb.Image(img)

            ys = wm.imagine(agent, self.dream_horizon, y, cont_mask)[0]
            ys = torch.stack(ys, 0)
            ohat = utils.apply_seq(decoder, ys)
            ohat = ohat.unflatten(2, (num_frames, -1))

            # Make video
            vid = utils.apply_seq(utils.viz_obs, ohat.clamp(0., 1.))
            vid = [make_grid(vid[t], nrow=num_obs) for t in range(vid.shape[0])]
            vid.append(torch.zeros_like(vid[-1]))
            vid = torch.stack(vid, 0)
            vid = (vid * 255.).byte().cpu().numpy()
            metrics['dream'] = wandb.Video(vid, fps=10)

        return metrics
