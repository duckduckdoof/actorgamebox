"""
utils.py

Author: Caleb Scott

---

Utility functions for global use.

From https://github.com/jrobine/sgf/blob/main/src/utils.py

"""

# IMPORTS
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import singledispatch

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from configs.constants import LUMA_VALS

# CONSTANTS
_numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

# FUNCTIONS
# See above. Helpful for conversions between numpy and torch.
def np_to_torch_dtype(np_dtype):
    """
    Numpy dtypes are classes. So, we get some weird behavior when comparing between
    numpy and torch dtypes that are supposed to be the same.

    So, we want to make sure np.dtype("float64") and np.dtype("float64").type are equivalent.
    """
    try:
        return _numpy_to_torch_dtype_dict[np_dtype]
    except KeyError:
        return _numpy_to_torch_dtype_dict[np_dtype.type]

# Seeding.

def rng(seed):
    """ Generator, given seed """
    return np.random.Generator(np.random.PCG64(seed))

def seed_all(seed):
    """ Seed every rng we have """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return rng(seed)

# Helpful mathematical functions.

def hinge(x, target):
    """ Hinge loss """
    return F.relu(target - x)

def off_diag(x, keepdim=False):
    """ Get off-diagonal elements of square matrix """
    n, m = x.shape
    assert n == m
    out = x.flatten()[:-1].view(n-1, n+1)[:, 1:]
    if keepdim:
        return out.reshape(n, n-1)
    else:
        return out.flatten()

def var_covar_loss(z, contrastive=False, eps=1e-4):
    """ VICReg's variance/cov loss """
    n = z.shape[0]
    d = z.shape[-1]

    if contrastive:
        z = z.T

    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = hinge(std, 1)
    var_loss = var_loss.mean()

    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n-1)
    cov_loss = off_diag(cov).pow_(2).sum(-1) / d
    cov_loss = cov_loss.sum()

    return var_loss, cov_loss, std

def lamdba_return(next_rs, next_vs, next_gammas, lmbda):
    """ Compute Lambda return using generalized advantage estimation """
    T = next_rs.shape[0]
    rets = [None for _ in range(T+1)]
    rets[T] = next_vs[-1]
    interms = next_rs + (1-lmbda) * next_gammas * next_vs
    for t in reversed(range(T)):
        rets[t] = interms[t] + lmbda * next_gammas[t] * rets[t+1]
    rets.pop()
    rets = torch.stack(rets, 0)
    return rets

def symlog(x):
    """ Symmetric log transform (DreamerV3) """
    return x.sign() * (x.abs() + 1).log()

def symexp(x):
    """ Symmetric exponential transform (DreamerV3) """
    return x.sign() * (x.abs().exp() - 1)

@torch._dynamo.disable()
def sample_categorical(probs):
    """ Sample from a categorical distribution """
    shape = probs.shape
    *batch_shape, out_dim = shape
    probs = probs.reshape(-1, out_dim)
    samples = torch.multinomial(probs, 1, replacement=True)
    samples = samples.reshape(batch_shape)
    return samples

@torch._dynamo.disable()
def sample_bernoulli(probs):
    """ Sample from Bernoulli distribution """
    return torch.bernoulli(probs)

def bins(low, high, num, device=None, decimals=4):
    """ Creates num bins with fixed decimal precision """
    bins = torch.linspace(low, high, num, device=device)
    bins = torch.round(bins, decimals=decimals)

def two_hot(tensor, bins):
    """ Create a two-hot encoding of values in tensor, given bins """
    num_bins = len(bins)
    below_mask = bins <= tensor
    below = below_mask.long().sum(-1, keepdim=True) - 1
    above = num_bins - (~below_mask).long().sum(-1, keepdim=True)
    equal = below == above
    dist_to_below = torch.where(equal, 1, torch.abs(bins[below] - tensor))
    dist_to_above = torch.where(equal, 1, torch.abs(bins[above] - tensor))
    total = dist_to_above + dist_to_below
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    return torch.nn.functional.one_hot(below.squeeze(-1), num_bins) * weight_below + \
        torch.nn.functional.one_hot(above.squeeze(-1), num_bins) * weight_above

# Utilities for visualization from data.

def grayscale(x, dim, keepdim=False, lumas=LUMA_VALS):
    """ Convert to grayscale (LUMA) """
    coefs = x.new_tensor(lumas)
    dim %= x.ndim
    coefs = coefs.reshape(*((1,)) * dim + (3,) + (1,) * (x.ndim - dim - 1))
    return (x * coefs).sum(dim, keepdim=keepdim)

def viz_obs(o):
    """ Visualize observation space for wandb """
    # time(?), number of frames, number of channels, height, width of images
    _, nf, c, _, _ = o.shape
    if nf == 1:
        return o

    if c != 1:
        o = grayscale(o, 2, keepdim=False)
    if nf >= 4:
        if nf > 4:
            o = o[:, -4:]
        f012 = o[:, [0,1,2]]
        f3 = o[:, [3]]
        o = (f012 + f3) / 2
    elif nf == 3:
        f01 = o[:, [0,1]]
        f2 = o[:, [2]]
        o = torch.cat(((f01 + f2) / 2, f2), dim=1)
    elif nf == 2:
        f0 = o[: [0]]
        f1 = o[: [1]]
        o = torch.cat(((f0 + f1) / 2, f1, f1), dim=1)
    else:
        assert False
    return o

# Helpful for getting information from a NN module

def num_params(mod):
    """ Total number of (differentiable) parameters in a NN module """
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)

def device(mod):
    """ Get the device of a NN module """
    return next(mod.parameters()).device

# The following are used for yielding flattened/zeroed torch tensors,
# derived from gym spaces.

@singledispatch
def space_flatten(space, x, dtype=None):
    raise NotImplementedError(f"Unknown space: '{space}'")

@space_flatten.register(gym.spaces.Box)
@space_flatten.register(gym.spaces.MultiBinary)
def _flatten_box_multibinary(space, x, dtype=None):
    if dtype is None:
        dtype = np_to_torch_dtype(space.dtype)
    return x.type(dtype).flatten(-len(space.shape))

@space_flatten.register(gym.spaces.Discrete)
def _flatten_discrete(space, x, dtype=None):
    if dtype is None:
        dtype = np_to_torch_dtype(space.dtype)
    return F.one_hot(x.long(), space.n).type(dtype)

@space_flatten.register(gym.spaces.MultiDiscrete)
def _flatten_multidiscrete(space, x, dtype=None):
    num_cats = int(space.nvec[0])
    if dtype is None:
        dtype = np_to_torch_dtype(space.dtype)
    return F.one_hot(x.long(), num_cats).type(dtype).flatten(-2, -1)

@singledispatch
def space_zeros(space, size=tuple(), device=None):
    raise NotImplementedError(f"Unknown space: '{space}'")

@space_zeros.register(gym.spaces.Box)
@space_zeros.register(gym.spaces.MultiBinary)
def _zeros_box_multibinary(space, size=tuple(), device=None):
    if isinstance(size, int):
        size = (size,)
    return torch.zeros(size + tuple(space.shape), dtype=np_to_torch_dtype(space.dtype), device=device)

@space_zeros.register(gym.spaces.Discrete)
def _zeros_discrete(space, size=tuple(), device=None):
    if isinstance(size, int):
        size = (size,)
    return torch.zeros(size, dtype=torch.int32, device=device)

@space_zeros.register(gym.spaces.MultiDiscrete)
def _zeros_discrete(space, size=tuple(), device=None):
    if isinstance(size, int):
        size = (size,)
    return torch.zeros(size + tuple(space.shape), dtype=torch.int32, device=device)

# Returning a Mask (see class below)
@torch.no_grad()
def get_mask(values):
    """ Get a mask from a tensor """
    values = values.detach()
    if values.dtype == torch.bool:
        complete = torch.all(values).item()
        values = values.float()
        if complete:
            count = torch.as_tensor(values.ndim, dtype=torch.float, device=values.device)
        else:
            count = values.sum()
    else:
        if values.dtype != torch.float:
            values = values.float()
        count = values.sum()
        complete = (count.long() == values.ndim).item()
    return Mask(values, count, complete)

# Function utility methods
def map_structure(fn, value, skip_none=True):
    """ Map a function over a (possibly nested) structure """
    if isinstance(value, (tuple, list)):
        return tuple(map_structure(fn, x) for x in value)
    elif isinstance(value, dict):
        return {k: map_structure(fn, v) for k, v, in value.items()}
    elif value is None and skip_none:
        return None
    else:
        return fn(value)

def flatten_structure(value, skip_none=False):
    """ Flatten a (possibly nested) structure into a tuple """
    if skip_none and value is None:
        return tuple()
    if isinstance(value, (tuple, list)):
        result = []
        for v in value:
            result.extend(flatten_structure(v, skip_none))
        return tuple(result)
    else:
        return (value,)

def flatten_seq(value):
    """ Flatten time/batch dimensions of all tensors """
    return map_structure(lambda x: x.flatten(0, 1), value)

def unflatten_seq(value, shape):
    """ Un-flatten the time/batch dimensions of all tensors """
    return map_structure(lambda x: x.unflatten(0, shape), value)

def apply_seq(fn, *args, unflatten=True, **kwargs):
    """ Flatten time/batch dimensions, call function, and then unflatten """
    all_values = flatten_structure(args, skip_none=True)
    shape = all_values[0].shape[:2]
    for i in range(1, len(all_values)):
        if all_values[i].shape[:2] != shape:
            raise ValueError("Batch shapes of all inputs must match.")

    args = flatten_seq(args)
    result = fn(*args, **kwargs)
    if unflatten:
        result = unflatten_seq(result, shape)
    return result

def unsqueeze_as(tensor, target_tensor):
    """ Unsqueeze tensor to match the shape of another. """
    n = tensor.ndim
    if n > target_tensor.ndim or tensor.shape != target_tensor.shape[:n]:
        raise ValueError(tensor.shape, target_tensor.shape)
    return tensor.reshape(*tensor.shape, *((1,) * (target_tensor.ndim - n)))

def target_network(src_net):
    """ Create a copy of a target network """
    trg_net = deepcopy(src_net)
    for src_param, trg_param in zip(src_net.parameters(), trg_net.parameters()):
        trg_param.data.copy_(src_param.data)
    trg_net.requires_grad_(False).eval()
    return trg_net

@torch.no_grad()
def ema_update(src_net, trg_net, decay):
    """ Update a target network using exponential moving average. """
    if decay == 1:
        return
    elif decay == 0:
        for src_param, trg_param in zip(src_net.parameters(), trg_net.parameters()):
            trg_param.data.copy_(src_param.data)
    else:
        one_m_decay = 1.0 - decay
        for src_param, trg_param in zip(src_net.parameters(), trg_net.parameters()):
            tmp = trg_param - src_param
            tmp.mul_(one_m_decay)
            trg_param.sub_(tmp)


# CLASSES
@dataclass
class Mask:
    """ Masking utility class """
    values: torch.Tensor
    count: torch.Tensor
    complete: bool

    @property
    def shape(self):
        return self.values.shape

    @property
    def device(self):
        return self.values.device

    def type(self, dtype):
        return Mask(self.values.type(dtype), self.count.type(dtype), self.complete)

    def __getitem__(self, idx):
        values = self.values[idx]
        if self.complete:
            count = torch.as_tensor(values.ndim, dtype=torch.float, device=values.device)
            complete = True
            return Mask(values, count, complete)
        count = values.sum()
        complete = (count.long() == values.ndim).item()
        return Mask(values, count, complete)

    def reshape(self, shape):
        return Mask(self.values.reshape(shape), self.count, self.complete)

    def squeeze(self, dim):
        return Mask(self.values.squeeze(dim), self.count, self.complete)

    def unsqueeze(self, dim):
        return Mask(self.values.unsqueeze(dim), self.count, self.complete)

    def flatten(self, start_dim=0, end_dim=-1):
        return Mask(self.values.flatten(start_dim, end_dim), self.count, self.complete)

    def unflatten(self, dim, sizes):
        return Mask(self.values.unflatten(dim, sizes), self.count, self.complete)

    def select(self, value):
        if self.complete:
            return value
        select = (self.values == 1)
        return map_structure(lambda x: x[select], value)

    def apply(self, value):
        if self.complete:
            return value
        if value.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            values = (self.values == 1)
            return map_structure(lambda x: unsqueeze_as(values, x) * x, value)
        return map_structure(lambda x: unsqueeze_as(self.values, x) * x, value)

    def sum(self, value):
        if self.complete:
            return map_structure(lambda x: x.sum(), value)
        return map_structure(lambda x: (unsqueeze_as(self.values, x) * x).sum(), value)

    def mean(self, value):
        if self.complete:
            return map_structure(lambda x: x.mean(), value)
        return map_structure(lambda x: (unsqueeze_as(self.values, x) * x).sum() / self.count, value)

    def clone(self):
        return Mask(self.values, self.count, self.complete)

    def cumulative_sequence(self, shifted=False, batch_first=False):
        if self.complete:
            return self.clone()

        values = self.values
        if shifted:
            if batch_first:
                values = torch.cat([torch.ones_like(values[:, :1]), values[:, :-1]], 1)
            else:
                values = torch.cat([torch.ones_like(values[:1]), values[:-1]], 0)

        values = torch.cumprod(values, dim=(1 if batch_first else 0))
        count = values.sum()
        complete = (count.long() == values.ndim).item()
        return Mask(values, count, complete)

    def cat(self, other, dim):
        values = torch.cat([self.values, other.values], dim)
        count = self.count + other.count
        complete = self.complete and other.complete
        return Mask(values, count, complete)

class Aggregator:
    """ Metric Aggregator class """

    def __init__(self, op, non_numeric='raise', same_keys=True):
        if op not in ('mean', 'max', 'min'):
            raise ValueError(op)
        if non_numeric not in ('raise', 'first', 'last'):
            raise ValueError(non_numeric)

        self.op = op
        self.non_numeric = non_numeric
        self.same_keys = same_keys
        self.history = None

    def append(self, metrics):
        def prepare(k, v):
            if not isinstance(k, str):
                raise TypeError(k)
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    v = v.reshape(1)
                v = v.detach().clone()
            return v

        metrics = {k: prepare(k, v) for k, v in metrics.items()}
        if self.history is None:
            self.history = {k: [v] for k, v in metrics.items()}
        elif self.same_keys:
            if metrics.keys() != self.history.keys():
                raise ValueError()

            for k, v in metrics.items():
                self.history[k].append(v)
        else:
            for k, v in metrics.items():
                self.history.setdefault(k, []).append(v)

    def aggregate(self):
        result = dict()
        if self.history is None:
            return result

        def raise_non_numeric(k, vs):
            raise ValueError(f"Non-numeric value for key {k}.")

        op = {'mean': np.mean, 'max': np.max, 'min': np.min}[self.op]
        non_numeric = {
            'raise': raise_non_numeric,
            'first': lambda k, vs: vs[0],
            'last': lambda k, vs: vs[-1],
            }[self.non_numeric]

        for k, values in self.history.items():
            values = tuple(v.item() if isinstance(v, torch.Tensor) else v for v in values)
            is_non_numeric = False
            for v in values:
                if not isinstance(v, (int, float, torch.Tensor)):
                    is_non_numeric = True
                    break
            if is_non_numeric:
                result[k] = non_numeric(k, values)
            else:
                result[k] = op(np.array(values))

        self.history = None
        return result

class EpisodeCollector:
    """ Collection of episodes in a parallel setup. """

    def __init__(self, env_id, wrappers, kwargs, num_parallel):
        self.vector_env = gym.make_vec(
            env_id,
            num_envs=num_parallel,
            vectorization_mode='async',
            wrappers=wrappers,
            **kwargs
        )

    def close(self):
        self.vector_env.close()

    def collect(self, seed, num_episodes, policy_fn, reset_fn, step_fn, aggregate_fn):
        vector_env = self.vector_env
        num_parallel = vector_env.num_envs
        if (num_episodes % num_parallel) != 0:
            warnings.warn("Number of episodes is not divisible by number of parallel environments.")

        aggs = []
        num_done = 0
        a = gym.vector.utils.create_empty_array(vector_env.action_space, n=1)[0]

        while num_done < num_episodes:
            cont = np.ones(num_parallel, dtype=bool)
            just_done = np.zeros(num_parallel)
            overflow = num_done + num_parallel - num_episodes
            if overflow > 0:
                cont[-overflow:] = False
                just_done = just_done[:-overflow]
            o, _ = vector_env.reset(seed=seed + num_done)
            agg = reset_fn(cont, o)
            state = None
            while np.any(cont):
                cont_a, state = policy_fn(o[cont], state, just_done)

                # We assume '0' is noop for actions
                a.fill(0)
                a[cont] = cont_a
                next_o, next_r, next_term, next_trunc, info = vector_env.step(a)
                next_done = next_term | next_trunc
                agg = step_fn(agg, cont, o, a, next_o, next_r, next_term, next_trunc)
                just_done = next_done[cont]
                cont = cont & ~next_done
                o = next_o
            num_done += num_parallel
            aggs.append(agg)

        return aggregate_fn(aggs)

    def collect_episode_stats(self, seed, num_episodes, policy_fn):
        def reset(cont, o):
            n = cont.shape[0]
            return {
                'episode_reward': np.zeros(n, dtype=np.float64),
                'episode_length': np.zeros(n, dtype=np.int64)
            }

        def step(agg, cont, o, a, next_o, next_r, next_term, next_trunc):
            agg['episode_reward'][cont] += next_r[cont]
            agg['episode_length'][cont] += 1
            return agg

        def aggregate(aggs):
            return {
                'episode_reward': np.concatenate([agg['episode_reward'] for agg in aggs]),
                'episode_length': np.concatenate([agg['episode_length'] for agg in aggs]),
            }

        return self.collect(seed, num_episodes, policy_fn, reset, step, aggregate)
