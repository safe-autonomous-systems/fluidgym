import logging
import os
from datetime import datetime as dt

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import fluidgym
from fluidgym.integration.gymnasium import GymFluidEnv
import time
import numpy as np
import torch.nn as nn
from copy import deepcopy
import re
import torch.nn.functional as F
import torch
import pandas as pd
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("fluidgym.training")


OmegaConf.register_new_resolver("eval", lambda x: eval(x))

local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
fluidgym.config.update("local_data_path", local_data_path)


__REDUCE__ = lambda b: "mean" if b else "none"


def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (
        (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x)
        .squeeze(0)
        .shape
    )


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class NormalizeImg(nn.Module):
    """Normalizes pixel observations to [0,1) range."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0)


class Flatten(nn.Module):
    """Flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def enc(env, cfg):
    """Returns a TOLD encoder."""
    if cfg.modality == "pixels":
        C = int(3 * cfg.frame_stack)
        layers = [
            NormalizeImg(),
            nn.Conv2d(C, cfg.num_channels, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
            nn.ReLU(),
        ]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
        layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
    else:
        layers = [
            nn.Linear(env.observation_space.shape[0], cfg.enc_dim),
            nn.ELU(),
            nn.Linear(cfg.enc_dim, cfg.latent_dim),
        ]
    return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]),
        act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]),
        act_fn,
        nn.Linear(mlp_dim[1], out_dim),
    )


def q(env, cfg, act_fn=nn.ELU()):
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(
        nn.Linear(cfg.latent_dim + np.prod(env.action_space.shape), cfg.mlp_dim),
        nn.LayerNorm(cfg.mlp_dim),
        nn.Tanh(),
        nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
        nn.ELU(),
        nn.Linear(cfg.mlp_dim, 1),
    )


class RandomShiftsAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, cfg):
        super().__init__()
        self.pad = int(cfg.img_size / 21) if cfg.modality == "pixels" else None

    def forward(self, x):
        if not self.pad:
            return x
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, env, cfg, init_obs):
        episode_length = env.unwrapped.episode_length
        action_dim = np.prod(env.action_space.shape)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        dtype = torch.float32 if cfg.modality == "state" else torch.uint8
        self.obs = torch.empty(
            (episode_length + 1, *init_obs.shape), dtype=dtype, device=self.device
        )
        self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
        self.action = torch.empty(
            (episode_length, action_dim), dtype=torch.float32, device=self.device
        )
        self.reward = torch.empty(
            (episode_length,), dtype=torch.float32, device=self.device
        )
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done):
        self.obs[self._idx + 1] = torch.tensor(
            obs, dtype=self.obs.dtype, device=self.obs.device
        )
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1


class ReplayBuffer:
    """
    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default."""

    def __init__(self, env, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Calculate initial target capacity
        raw_capacity = int(min(cfg.train_steps, cfg.max_buffer_size))
        self.episode_length = env.unwrapped.episode_length

        # --- FIX ---
        # Snap capacity to a perfect multiple of episode_length
        self.capacity = (raw_capacity // self.episode_length) * self.episode_length
        # ---------------

        action_dim = np.prod(env.action_space.shape)

        dtype = torch.float32
        obs_shape = env.observation_space.shape
        self._obs = torch.empty(
            (self.capacity + 1, *obs_shape), dtype=dtype, device=self.device
        )
        self._last_obs = torch.empty(
            (self.capacity // self.episode_length, *obs_shape),
            dtype=dtype,
            device=self.device,
        )
        self._action = torch.empty(
            (self.capacity, action_dim), dtype=torch.float32, device=self.device
        )
        self._reward = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._priorities = torch.ones(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        self._obs[self.idx : self.idx + self.episode_length] = episode.obs[:-1]
        self._last_obs[self.idx // self.episode_length] = episode.obs[-1]
        self._action[self.idx : self.idx + self.episode_length] = episode.action
        self._reward[self.idx : self.idx + self.episode_length] = episode.reward
        if self._full:
            max_priority = self._priorities.max().to(self.device).item()
        else:
            max_priority = (
                1.0
                if self.idx == 0
                else self._priorities[: self.idx].max().to(self.device).item()
            )
        mask = (
            torch.arange(self.episode_length) >= self.episode_length - self.cfg.horizon
        )
        new_priorities = torch.full(
            (self.episode_length,), max_priority, device=self.device
        )
        new_priorities[mask] = 0
        self._priorities[self.idx : self.idx + self.episode_length] = new_priorities
        self.idx = (self.idx + self.episode_length) % self.capacity
        self._full = self._full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        if self.cfg.modality == "state":
            return arr[idxs]
        obs = torch.empty(
            (self.cfg.batch_size, 3 * self.cfg.frame_stack, *arr.shape[-2:]),
            dtype=arr.dtype,
            device=torch.device("cuda"),
        )
        obs[:, -3:] = arr[idxs].cuda()
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg.frame_stack):
            mask[_idxs % self.episode_length == 0] = False
            _idxs[mask] -= 1
            obs[:, -(i + 1) * 3 : -i * 3] = arr[_idxs].cuda()
        return obs.float()

    def sample(self):
        probs = (
            self._priorities if self._full else self._priorities[: self.idx]
        ) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(
                total,
                self.cfg.batch_size,
                p=probs.cpu().numpy(),
                replace=not self._full,
            )
        ).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()

        obs = self._get_obs(self._obs, idxs)
        next_obs_shape = (
            self._last_obs.shape[1:]
            if self.cfg.modality == "state"
            else (3 * self.cfg.frame_stack, *self._last_obs.shape[-2:])
        )
        next_obs = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *next_obs_shape),
            dtype=obs.dtype,
            device=obs.device,
        )
        action = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *self._action.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )
        reward = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size),
            dtype=torch.float32,
            device=self.device,
        )
        for t in range(self.cfg.horizon + 1):
            _idxs = idxs + t
            next_obs[t] = self._get_obs(self._obs, _idxs + 1)
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]

        mask = (_idxs + 1) % self.episode_length == 0
        next_obs[-1, mask] = (
            self._last_obs[_idxs[mask] // self.episode_length].cuda().float()
        )
        if not action.is_cuda:
            action, reward, idxs, weights = (
                action.cuda(),
                reward.cuda(),
                idxs.cuda(),
                weights.cuda(),
            )

        return obs, next_obs, action, reward.unsqueeze(2), idxs, weights

    def save(self, fp):
        """Save replay buffer to filepath."""
        torch.save(
            {
                "obs": self._obs.cpu(),
                "last_obs": self._last_obs.cpu(),
                "action": self._action.cpu(),
                "reward": self._reward.cpu(),
                "priorities": self._priorities.cpu(),
                "idx": self.idx,
                "full": self._full,
            },
            fp,
        )

    def load(self, fp):
        """Load replay buffer from filepath."""
        d = torch.load(fp)
        self._obs = d["obs"].to(self.device)
        self._last_obs = d["last_obs"].to(self.device)
        self._action = d["action"].to(self.device)
        self._reward = d["reward"].to(self.device)
        self._priorities = d["priorities"].to(self.device)
        self.idx = d["idx"]
        self._full = d["full"]

def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)


class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, env, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = enc(env, cfg)
        self._dynamics = mlp(
            cfg.latent_dim + np.prod(env.action_space.shape), cfg.mlp_dim, cfg.latent_dim
        )
        self._reward = mlp(cfg.latent_dim + np.prod(env.action_space.shape), cfg.mlp_dim, 1)
        self._pi = mlp(cfg.latent_dim, cfg.mlp_dim, + np.prod(env.action_space.shape))
        self._Q1, self._Q2 = q(env, cfg), q(env, cfg)
        self.apply(orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            set_requires_grad(m, enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)


class TDMPC:
    """Implementation of TD-MPC learning + inference."""

    def __init__(self, env, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.std = linear_schedule(cfg.std_schedule, 0)
        self.model = TOLD(env, cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.aug = RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
        }

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d["model"])
        self.model_target.load_state_dict(d["model_target"])

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G

    @torch.no_grad()
    def plan(self, env, obs, eval_mode=False, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        # Seed steps
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(
                np.prod(env.action_space.shape), dtype=torch.float32, device=self.device
            ).uniform_(-1, 1)

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(
            min(self.cfg.horizon, linear_schedule(self.cfg.horizon_schedule, step))
        )
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(
                horizon, num_pi_trajs, np.prod(env.action_space.shape), device=self.device
            )
            z = self.model.h(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z, _ = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.model.h(obs).repeat(self.cfg.num_samples + num_pi_trajs, 1)
        mean = torch.zeros(horizon, np.prod(env.action_space.shape), device=self.device)
        std = 2 * torch.ones(horizon, np.prod(env.action_space.shape), device=self.device)
        if not t0 and hasattr(self, "_prev_mean"):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(
                    horizon,
                    self.cfg.num_samples,
                    np.prod(env.action_space.shape),
                    device=std.device,
                ),
                -1,
                1,
            )
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites, dim=0
            ).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9
            )
            _std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
                / (score.sum(0) + 1e-9)
            )
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(np.prod(env.action_space.shape), device=std.device)
        return a

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a = self.model.pi(z, self.cfg.min_std)
            Q = torch.min(*self.model.Q(z, a))
            pi_loss += -Q.mean() * (self.cfg.rho**t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.h(next_obs)
        td_target = reward + self.cfg.discount * torch.min(
            *self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std))
        )
        return td_target

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.optim.zero_grad(set_to_none=True)
        self.std = linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # Representation
        z = self.model.h(self.aug(obs))
        zs = [z.detach()]

        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        for t in range(self.cfg.horizon):
            # Predictions
            Q1, Q2 = self.model.Q(z, action[t])
            z, reward_pred = self.model.next(z, action[t])
            with torch.no_grad():
                next_obs = self.aug(next_obses[t])
                next_z = self.model_target.h(next_obs)
                td_target = self._td_target(next_obs, reward[t])
            zs.append(z.detach())

            # Losses
            rho = self.cfg.rho**t
            consistency_loss += rho * torch.mean(mse(z, next_z), dim=1, keepdim=True)
            reward_loss += rho * mse(reward_pred, reward[t])
            value_loss += rho * (mse(Q1, td_target) + mse(Q2, td_target))
            priority_loss += rho * (l1(Q1, td_target) + l1(Q2, td_target))

        # Optimize model
        total_loss = (
            self.cfg.consistency_coef * consistency_loss.clamp(max=1e4)
            + self.cfg.reward_coef * reward_loss.clamp(max=1e4)
            + self.cfg.value_coef * value_loss.clamp(max=1e4)
        )
        weighted_loss = (total_loss.squeeze(1) * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False
        )
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update policy + target network
        pi_loss = self.update_pi(zs)
        if step % self.cfg.update_freq == 0:
            ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "weighted_loss": float(weighted_loss.mean().item()),
            "grad_norm": float(grad_norm),
        }


def evaluate(env, agent, num_episodes, step, env_step) -> pd.DataFrame:
    """Evaluate a trained agent and optionally save a video."""
    all_dfs = []
    for episode_idx in range(num_episodes):
        # Do not randomize initial state for first episode for comparability
        obs, _ = env.reset(randomize=not episode_idx == 0)
        done = False

        if episode_idx == 0:
            env.render(save=True, filename="test_eval_episode_0_initial")

        t = 0
        ep_metrics = defaultdict(list)

        while not done:
            action = agent.plan(env, obs, eval_mode=True, step=step, t0=t == 0)

            # Reshape to env.action_space if necessary
            if action.shape != env.action_space.shape:
                action_env = action.reshape(env.action_space.shape)
            else:
                action_env = action

            obs, reward, term, trunc, info = env.step(action_env.cpu().numpy())

            if episode_idx == 0:
                env.render()

            done = term or trunc
            ep_metrics["reward"].append(reward)
            for action_idx in range(min(action.shape[0], 12)):
                ep_metrics[f"action_{action_idx}"].append(action[action_idx].item())
            for metric in env.unwrapped.metrics:
                ep_metrics[metric].append(info[metric])
            t += 1

        if episode_idx == 0:
            env.render(save=True, filename="test_eval_episode_0_final")
            env.save_gif("test_eval_episode_0")

        episode_df = pd.DataFrame(ep_metrics)
        episode_df["episode"] = episode_idx
        episode_df["step"] = np.arange(len(episode_df))
        all_dfs.append(episode_df)

    return pd.concat(all_dfs, ignore_index=True)


def train(env: GymFluidEnv, cfg,) -> TDMPC:
    """Training script for TD-MPC. Requires a CUDA-enabled device."""
    agent, buffer = TDMPC(env, cfg), ReplayBuffer(env, cfg)

    training_logs = []

    if cfg.continue_training:
        logger.info("Continuing training from existing model and replay buffer.")
        agent.load("tdmpc_agent.pth")
        buffer.load("replay_buffer_latest.pth")
        logs = pd.read_csv("training_log.csv")
        trained_steps = logs["train/step"].max()
        training_logs += logs.to_dict("records")
        remaining_steps = int(cfg.train_steps) - trained_steps

        # DEBUG LOGGING
        logger.info(f"Trained steps: {trained_steps}")
        logger.info(f"Remaining training steps: {remaining_steps}")
        logger.info(f"Last training log entry: {training_logs[-1]}")
        logger.info(f"Replay buffer idx: {buffer.idx}, full: {buffer._full}")
    else:
        trained_steps = 0
    
    # Run training
    episode_length = env.unwrapped.episode_length
    episode_idx = trained_steps // episode_length
    start_time = time.time()
    for step in range(trained_steps, int(cfg.train_steps) + episode_length, episode_length):
        # Collect trajectory
        obs, _ = env.reset()
        episode = Episode(env, cfg, obs)
        while not episode.done:
            action = agent.plan(env, obs, step=step, t0=episode.first)

            # Reshape to env.action_space if necessary
            if action.shape != env.action_space.shape:
                action_env = action.reshape(env.action_space.shape)
            else:
                action_env = action

            obs, reward, term, trunc, _ = env.step(action_env.cpu().numpy())
            done = term or trunc
            episode += (obs, action, reward, done)
        assert len(episode) == episode_length
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i))

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            "train/episode": episode_idx,
            "train/step": step,
            "train/env_step": env_step,
            "train/total_time": time.time() - start_time,
            "train/episode_reward": episode.cumulative_reward,
            "train/mean_reward": episode.cumulative_reward / len(episode),
        }
        # Evaluate agent periodically
        if env_step > 0 and env_step % cfg.eval_freq == 0:
            eval_metrics = evaluate(env, agent, cfg.n_eval_episodes, step, env_step)
            common_metrics["eval/mean_reward"] = eval_metrics["reward"].mean()

        train_metrics.update(common_metrics)

        if cfg.wandb.enable:
            wandb.log(train_metrics, step=env_step)

        training_logs.append(train_metrics)

        if env_step > 0 and env_step % cfg.ckpt_freq == 0:
            pd.DataFrame(training_logs).to_csv("training_log.csv", index=False)
            agent.save(f"tdmpc_agent.pth")
            buffer.save(f"replay_buffer_latest.pth")

    print("Training completed successfully")

    pd.DataFrame(training_logs).to_csv("training_log.csv", index=False)

    return agent


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_tdmpc")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise

def run_experiment(cfg: DictConfig):
    logger.info("Training script started.")

    logger.info("Initializing environments...")
    fluid_env = fluidgym.make(cfg.env_id, **cfg.env_kwargs)
    fluid_eval_env = fluidgym.make(cfg.env_id, **cfg.eval_env_kwargs)

    if cfg.rl_mode == "marl":
        raise NotImplementedError("MARL mode is not implemented yet.")
    elif cfg.rl_mode == "sarl":
        env = GymFluidEnv(fluid_env)
        eval_env = GymFluidEnv(fluid_eval_env)
    else:
        raise ValueError(f"Unknown rl_mode: {cfg.rl_mode}")
    env.seed(cfg.seed)
    eval_env.seed(cfg.seed + 42)
    eval_env.val()
    logger.info("Done.")

    now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"TD-MPC_{cfg.env_id}_{cfg.seed}_{now}"
    run_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    assert isinstance(run_config, dict), "Run config must be a dictionary."

    if cfg.wandb.enable:
        logger.info("Initializing Weights & Biases...")
        wandb.init(
            id=run_id,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=run_config,  # type: ignore
            group=env.id,
        )
        logger.info("Done.")

    if cfg.run_training:
        logger.info("Starting training...")
        agent = train(env, cfg)
        agent.save("tdmpc_agent.pth")
        logger.info("Training finished.")
    else:
        agent = TDMPC(env, cfg)
        agent.load("tdmpc_agent.pth")
        logger.info("Loaded existing model from tdmpc_agent.pth.")

    logger.info("Starting evaluation...")
    eval_results = evaluate(eval_env, agent, cfg.n_test_episodes, step=0, env_step=0)

    output_dir = Path("./test")
    output_dir.mkdir(exist_ok=True)
    eval_results.to_csv(output_dir / "test_eval_sequences.csv", index=False)
    logger.info(
        f"Evaluation completed. Average reward: {eval_results['reward'].mean():.2f}"
    )
    eval_results.to_csv("test_eval_sequences.csv", index=False)
    if cfg.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    main()
