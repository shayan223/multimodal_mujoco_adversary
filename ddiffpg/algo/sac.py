from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ddiffpg.algo.ac_base import ActorCriticBase
from ddiffpg.replay.nstep_replay import NStepReplay
from ddiffpg.utils.torch_util import soft_update
from ddiffpg.utils.common import handle_timeout, DensityTracker
from ddiffpg.utils.intrinsic import IntrinsicM


@dataclass
class AgentSAC(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor) if not self.cfg.algo.no_tgt_actor else self.actor

        self.obs = None
        if self.cfg.algo.alpha is None:
            self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
            self.alpha_optim = torch.optim.AdamW([self.log_alpha], lr=self.cfg.algo.alpha_lr)

        self.target_entropy = -self.action_dim

        self.n_step_buffer = NStepReplay(self.obs_dim,
                                         self.action_dim,
                                         self.cfg.num_envs,
                                         self.cfg.algo.nstep,
                                         device=self.device)
        
        if 'antmaze' in self.cfg.env.name:
            self.pos_history = DensityTracker(self.cfg.env.env_kwargs, resolution=self.cfg.env.resolution)
        self.intrinsic = IntrinsicM(self.obs_dim, 
                                    type=self.cfg.intrinsic.type, 
                                    env_name=self.cfg.env.name, 
                                    normalize=self.cfg.intrinsic.normalize,
                                    pos_enc=self.cfg.intrinsic.pos_enc,
                                    L=self.cfg.intrinsic.L,
                                    device=self.device)
        
    def get_alpha(self, detach=True, scalar=False):
        if self.cfg.algo.alpha is None:
            alpha = self.log_alpha.exp()
            if detach:
                alpha = alpha.detach()
            if scalar:
                alpha = alpha.item()
        else:
            alpha = self.cfg.algo.alpha
        return alpha

    def get_actions(self, obs, sample=True):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        actions = self.actor.get_actions(obs, sample=sample)
        return actions

    @torch.no_grad()
    def explore_env(self, env, timesteps: int, random: bool) -> list:
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_obs = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_actions = torch.empty((self.cfg.num_envs, timesteps) + (self.action_dim,), device=self.device)
        traj_rewards = torch.empty((self.cfg.num_envs, timesteps), device=self.device)
        traj_next_obs = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,), device=self.device)
        traj_dones = torch.empty((self.cfg.num_envs, timesteps), device=self.device)

        obs = self.obs
        for i in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(obs)
            if random:
                action = torch.rand((self.cfg.num_envs, self.action_dim),
                                    device=self.cfg.device) * 2.0 - 1.0
            else:
                action = self.get_actions(obs, sample=True)

            next_obs, reward, done, info = env.step(action)
            self.update_tracker(reward, done)

            # to draw exploration density
            if 'antmaze' in self.cfg.env.name:
                self.pos_history.update_mat(obs[:, :2].cpu())

            if self.cfg.algo.handle_timeout:
                done = handle_timeout(done, info)

            traj_obs[:, i] = obs
            traj_actions[:, i] = action
            traj_dones[:, i] = done
            traj_rewards[:, i] = reward
            traj_next_obs[:, i] = next_obs
            obs = next_obs
        self.obs = obs

        traj_rewards = self.cfg.algo.reward_scale * traj_rewards.reshape(self.cfg.num_envs, timesteps, 1)
        traj_dones = traj_dones.reshape(self.cfg.num_envs, timesteps, 1)
        data = self.n_step_buffer.add_to_buffer(traj_obs, traj_actions, traj_rewards, traj_next_obs, traj_dones)

        return data, timesteps * self.cfg.num_envs

    def update_net(self, memory):
        critic_loss_list = list()
        actor_loss_list = list()
        for i in range(self.cfg.algo.update_times):
            obs, action, _, reward, next_obs, done = memory.sample_batch(self.cfg.algo.batch_size)
            # intrinsic reward
            reward_intrinsic = self.intrinsic.compute_reward(obs, next_obs)
            reward = reward + reward_intrinsic

            if self.cfg.algo.obs_norm:
                obs = self.obs_rms.normalize(obs)
                next_obs = self.obs_rms.normalize(next_obs)
            critic_loss, critic_grad_norm = self.update_critic(obs, action, reward, next_obs, done)
            critic_loss_list.append(critic_loss)

            # update RND
            if self.cfg.intrinsic.type == 'rnd':
                dynamic_loss, dynamic_grad_norm = self.intrinsic.update(obs)
            elif self.cfg.intrinsic.type == 'noveld':
                dynamic_loss, dynamic_grad_norm = self.intrinsic.update(torch.cat([obs, next_obs]))
            else:
                raise NotImplementedError

            actor_loss, actor_grad_norm = self.update_actor(obs)
            actor_loss_list.append(actor_loss)

            soft_update(self.critic_target, self.critic, self.cfg.algo.tau)
            if not self.cfg.algo.no_tgt_actor:
                soft_update(self.actor_target, self.actor, self.cfg.algo.tau)

        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "train/return": self.return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            'train/alpha': self.get_alpha(scalar=True),
        }
        if 'antmaze' in self.cfg.env.name:
            log_info["train/state_coverage"] = self.pos_history.get_density()
        return log_info

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_actions, _, log_prob = self.actor.get_actions_logprob(next_obs)
            target_Q = self.critic_target.get_q_min(next_obs, next_actions) - self.get_alpha() * log_prob
            target_Q = reward + (1 - done) * (self.cfg.algo.gamma ** self.cfg.algo.nstep) * target_Q
        current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        grad_norm = self.optimizer_update(self.critic_optimizer, critic_loss)
        return critic_loss.item(), grad_norm

    def update_actor(self, obs, skip_weight_update=False):
        self.critic.requires_grad_(False)
        actions, _, log_prob = self.actor.get_actions_logprob(obs)
        Q = self.critic.get_q_min(obs, actions)
        actor_loss = (self.get_alpha() * log_prob - Q).mean()
        grad_norm = self.optimizer_update(self.actor_optimizer, actor_loss,skip_weight_update=skip_weight_update)
        self.critic.requires_grad_(True)

        if self.cfg.algo.alpha is None:
            alpha_loss = (self.get_alpha(detach=False) * (-log_prob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, alpha_loss)
        return actor_loss.item(), grad_norm

    def get_attack_input_gradient(self, obs):
        """Return d(actor objective)/d(obs) without mutating learner state."""
        if not obs.requires_grad:
            obs = obs.requires_grad_(True)

        actions, _, log_prob = self.actor.get_actions_logprob(obs)
        q_value = self.critic.get_q_min(obs, actions)
        actor_loss = (self.get_alpha() * log_prob - q_value).mean()
        (input_grad,) = torch.autograd.grad(
            actor_loss,
            obs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        return input_grad.detach()
