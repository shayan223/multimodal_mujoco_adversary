from copy import deepcopy
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ddiffpg.algo.ac_base import ActorCriticBase
from ddiffpg.replay.nstep_replay import NStepReplay
from ddiffpg.utils.noise import add_mixed_normal_noise
from ddiffpg.utils.noise import add_normal_noise
from ddiffpg.utils.schedule_util import ExponentialSchedule
from ddiffpg.utils.schedule_util import LinearSchedule
from ddiffpg.utils.torch_util import soft_update
from ddiffpg.utils.common import handle_timeout, DensityTracker
from ddiffpg.replay.diffusion_replay import DiffusionGoalBuffer
from ddiffpg.utils.distl_util import projection
from ddiffpg.utils.torch_util import add_embedding
from ddiffpg.utils.intrinsic import IntrinsicM


@dataclass
class AgentDDiffPG(ActorCriticBase):
    def __post_init__(self):
        super().__post_init__()
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor) if not self.cfg.algo.no_tgt_actor else self.actor

        if self.cfg.algo.noise.decay == 'linear':
            self.noise_scheduler = LinearSchedule(start_val=self.cfg.algo.noise.std_max,
                                                  end_val=self.cfg.algo.noise.std_min,
                                                  total_iters=self.cfg.algo.noise.lin_decay_iters
                                                  )
        elif self.cfg.algo.noise.decay == 'exp':
            self.noise_scheduler = ExponentialSchedule(start_val=self.cfg.algo.noise.std_max,
                                                       gamma=self.cfg.algo.exp_decay_rate,
                                                       end_val=self.cfg.algo.noise.std_min)
        else:
            self.noise_scheduler = None

        self.n_step_buffer = NStepReplay(self.obs_dim,
                                         self.action_dim,
                                         self.cfg.num_envs,
                                         self.cfg.algo.nstep,
                                         device=self.device)

        self.diffusion_buffer = DiffusionGoalBuffer(
                cfg=self.cfg,
                capacity=self.cfg.algo.memory_size,
                obs_dim=self.obs_dim[0], 
                action_dim=self.action_dim,
                num_envs=self.cfg.num_envs,
                max_episode_len=self.max_episode_len, 
                device=self.device
        )

        if 'antmaze' in self.cfg.env.name:
            self.pos_history = DensityTracker(self.cfg.env.env_kwargs, resolution=self.cfg.env.resolution)
        self.intrinsic = IntrinsicM(self.obs_dim, 
                                    type=self.cfg.intrinsic.type, 
                                    env_name=self.cfg.env.name, 
                                    normalize=self.cfg.intrinsic.normalize,
                                    pos_enc=self.cfg.intrinsic.pos_enc,
                                    L=self.cfg.intrinsic.L,
                                    device=self.device)
        self.reward_mean = deque(maxlen=int(1e4))
        self.explore_n = self.cfg.algo.batch_size
        self.explore_embedding = None
        self.mode_embedding = []
        self.num_mode = 1
        self.exp_scheduler = None

    def get_actions(self, obs, sample=True):
        if self.cfg.algo.obs_norm:
            obs = self.obs_rms.normalize(obs)
        
        actions = self.actor(obs)

        if sample:
            if self.cfg.algo.noise.type == 'fixed':
                actions = add_normal_noise(actions,
                                           std=self.get_noise_std(),
                                           out_bounds=[-1., 1.])
            elif self.cfg.algo.noise.type == 'mixed':
                actions = add_mixed_normal_noise(actions,
                                                 std_min=self.cfg.algo.noise.std_min,
                                                 std_max=self.cfg.algo.noise.std_max,
                                                 out_bounds=[-1., 1.])
            else:
                raise NotImplementedError
        return actions

    def get_tgt_policy_actions(self, obs, sample=True):
        actions = self.actor_target(obs)
        if sample:
            actions = add_normal_noise(actions,
                                    std=self.cfg.algo.noise.tgt_pol_std,
                                    noise_bounds=[-self.cfg.algo.noise.tgt_pol_noise_bound,
                                                    self.cfg.algo.noise.tgt_pol_noise_bound],
                                    out_bounds=[-1., 1.])
        return actions
    
    def get_exp_p(self, steps):
        if self.cfg.algo.exp.type == 'fixed':
            p = min(self.cfg.algo.exp.fix_ratio, 1-self.explore_n/self.cfg.algo.batch_size)
        elif self.cfg.algo.exp.type == 'linear':
            if len(self.mode_embedding) != 0 and self.exp_scheduler is None:
                iters = (self.cfg.algo.exp.stop_ratio * self.cfg.max_step - steps) // self.cfg.num_envs
                self.exp_scheduler = LinearSchedule(start_val=0.0,
                                                    end_val=1.0,
                                                    total_iters=iters
                                                    )
                p = self.exp_scheduler.val()
            elif self.exp_scheduler is not None:
                self.exp_scheduler.step()
                p = self.exp_scheduler.val()
            else:
                assert len(self.mode_embedding) == 0
                p = 0.0
        elif self.cfg.algo.exp.type == 'prop':
            p = 1-self.explore_n/self.cfg.algo.batch_size
        else:
            raise NotImplementedError
        
        if steps is not None:
            if steps >= self.cfg.algo.exp.stop_ratio * self.cfg.max_step:
                p = 1.0
        return p
    
    def explore_env(self, env, timesteps: int, random: bool = False, total_steps: int = None) -> list:
        obs_dim = (self.obs_dim,) if isinstance(self.obs_dim, int) else self.obs_dim
        traj_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,)).to(self.device)
        traj_actions = torch.empty((self.cfg.num_envs, timesteps) + (self.action_dim,)).to(self.device)
        traj_rewards = torch.empty((self.cfg.num_envs, timesteps)).to(self.device)
        traj_next_states = torch.empty((self.cfg.num_envs, timesteps) + (*obs_dim,)).to(self.device)
        traj_dones = torch.empty((self.cfg.num_envs, timesteps)).to(self.device)
        
        obs = self.obs
        self.p = self.get_exp_p(total_steps)
        for i in range(timesteps):
            if self.cfg.algo.obs_norm:
                self.obs_rms.update(obs)
            if random:
                action = torch.rand((self.cfg.num_envs, self.action_dim),
                                    device=self.cfg.device) * 2.0 - 1.0
            else:
                embedded_obs = add_embedding(obs, self.explore_embedding, p=self.p, modes=self.mode_embedding if self.cfg.algo.exp.mode_embedding else [])
                action = self.get_actions(embedded_obs, sample=True)

            next_obs, reward, done, info = env.step(action)
            traj_info = self.update_tracker(reward, done)

            # to draw exploration density
            if 'antmaze' in self.cfg.env.name:
                self.pos_history.update_mat(obs[:, :2].cpu())

            if self.cfg.algo.handle_timeout:
                done = handle_timeout(done, info)
            
            # add data to diffusion buffer
            if 'success' in info.keys():
                traj_info['success'] = info['success']
            self.diffusion_buffer.add_to_buffer(obs, action, reward * self.cfg.algo.reward_scale, next_obs, done, traj_info)

            traj_states[:, i] = obs
            traj_actions[:, i] = action
            traj_dones[:, i] = done
            traj_rewards[:, i] = reward
            traj_next_states[:, i] = next_obs
            obs = next_obs
        self.obs = obs

        return timesteps * self.cfg.num_envs

    def update_net(self):
        critic_loss_list = list()
        critic_grad_list = list()
        actor_loss_list = list()
        actor_grad_list = list()
        dynamic_loss_list = list()
        dynamic_grad_list = list()
        for i in range(self.cfg.algo.update_times):
            data_list = self.diffusion_buffer.sample_batch(self.cfg.algo.batch_size)

            # intrinsic reward
            obs, next_obs, reward = [], [], []
            self.num_mode = len(data_list)
            for i in range(len(data_list)):
                obs.append(data_list[i]["batch"][0])
                next_obs.append(data_list[i]["batch"][4])
                reward.append(data_list[i]["batch"][3])
            obs = torch.cat(obs)
            next_obs = torch.cat(next_obs)
            reward = torch.cat(reward) 
            reward_intrinsic = self.intrinsic.compute_reward(obs, next_obs)
            rewards = reward + reward_intrinsic
            
            prev = 0
            return_list, state_list, action_list = [], [], []
            self.mode_embedding = []
            for i in range(len(data_list)):
                cur_batch = data_list[i]["batch"][0].shape[0]
                critic_target = data_list[i]["Q"]["target_Q"]

                # add embedding for states
                state = data_list[i]["batch"][0]
                next_state = data_list[i]["batch"][4]
                # for exploratory mode when i == 0
                if i == 0:
                    reward = reward_intrinsic[prev:prev+cur_batch]
                    self.explore_n = state.shape[0]
                    embedding = data_list[i]["embedding"]
                    embedded_state = add_embedding(state, embedding, p=0)
                    embedded_next_state = add_embedding(next_state, embedding, p=0)
                else:
                    reward = rewards[prev:prev+cur_batch]
                    if self.cfg.algo.use_embedding:
                        embedding = data_list[i]["embedding"]
                        self.mode_embedding.append(embedding)
                    else:
                        embedding = torch.zeros(self.cfg.algo.embedding_dim, device=state.device)
                    embedded_state = add_embedding(state, embedding)
                    embedded_next_state = add_embedding(next_state, embedding)

                # update Q
                critic, critic_loss, critic_grad_norm = self.update_critic(
                        critic=data_list[i]["Q"]["Q"], 
                        critic_target=data_list[i]["Q"]["target_Q"], 
                        critic_optimizer=data_list[i]["Q"]["optimizer"], 
                        obs=data_list[i]["batch"][0],
                        action=data_list[i]["batch"][1], 
                        reward=reward, 
                        next_obs=data_list[i]["batch"][4],
                        embedded_next_obs=embedded_next_state,
                        done=data_list[i]["batch"][5])
                soft_update(critic_target, critic, self.cfg.algo.tau)
                critic_loss_list.append(critic_loss)
                critic_grad_list.append(critic_grad_norm)
                
                # get target action
                mean_action, new_action = self.update_target_action(
                        obs=data_list[i]["batch"][0],
                        action=data_list[i]["batch"][2],
                        critic=critic
                        )
                
                state_list.append(embedded_state)
                action_list.append(new_action)
                Q = {"Q": critic, "target_Q": critic_target, "optimizer": data_list[i]["Q"]["optimizer"]}
                return_list.append({
                    "Q": Q,
                    "indices": data_list[i]["indices"],
                    "new_action": new_action
                    })
                prev += cur_batch

            # update diffusion policy
            self.diffusion_buffer.update_target_action(return_list)
            state = torch.cat(state_list)
            target_action = torch.cat(action_list)
            actor_loss, actor_grad_norm = self.update_actor(state, target_action)
            actor_loss_list.append(actor_loss)
            actor_grad_list.append(actor_grad_norm)

            # update RND
            if self.cfg.intrinsic.type == 'rnd':
                dynamic_loss, dynamic_grad_norm = self.intrinsic.update(obs)
            elif self.cfg.intrinsic.type == 'noveld':
                dynamic_loss, dynamic_grad_norm = self.intrinsic.update(torch.cat([obs, next_obs]))
            else:
                raise NotImplementedError
            dynamic_loss_list.append(dynamic_loss)
            dynamic_grad_list.append(dynamic_grad_norm)

        log_info = {
            "train/critic_loss": np.mean(critic_loss_list),
            "train/actor_loss": np.mean(actor_loss_list),
            "train/dynamic_loss": np.mean(dynamic_loss_list),
            "train/return": self.return_tracker.mean(),
            "train/episode_length": self.step_tracker.mean(),
            "train/actor_grad": np.mean(actor_grad_list),
            "train/critic_grad": np.mean(critic_grad_list),
            "train/dynamic_grad": np.mean(dynamic_grad_list),
            "train/mean_action": mean_action,
            "train/mean_intrinsic": reward_intrinsic.mean().item(),
            "train/p": self.p
        }
        if 'antmaze' in self.cfg.env.name:
            log_info["train/state_coverage"] = self.pos_history.get_density()
        return log_info
    
    def update_critic(self, critic, critic_target, critic_optimizer, obs, action, reward, next_obs, embedded_next_obs, done):
        next_actions = self.get_tgt_policy_actions(embedded_next_obs)

        with torch.no_grad():   
            target_Q1, target_Q2 = critic_target.get_q1_q2(next_obs, next_actions)
            target_Q1_projected = projection(next_dist=target_Q1,
                                                     reward=reward,
                                                     done=done,
                                                     gamma=self.cfg.algo.gamma ** self.cfg.algo.nstep,
                                                     v_min=critic.v_min,
                                                     v_max=critic.v_max,
                                                     num_atoms=self.cfg.algo.num_atoms,
                                                     support=critic.z_atoms,
                                                     device=self.device)
            target_Q2_projected = projection(next_dist=target_Q2,
                                                     reward=reward,
                                                     done=done,
                                                     gamma=self.cfg.algo.gamma ** self.cfg.algo.nstep,
                                                     v_min=critic.v_min,
                                                     v_max=critic.v_max,
                                                     num_atoms=self.cfg.algo.num_atoms,
                                                     support=critic.z_atoms,
                                                     device=self.device)
            target_Q = torch.min(target_Q1_projected, target_Q2_projected)

        current_Q1, current_Q2 = critic.get_q1_q2(obs, action)
        critic_loss = F.binary_cross_entropy(current_Q1, target_Q) + F.binary_cross_entropy(current_Q2, target_Q)
        grad_norm = self.optimizer_update(critic_optimizer, critic_loss)

        return critic, critic_loss.item(), grad_norm.item()

    def update_actor(self, obs, target_action):
        actor_loss = self.actor.get_loss(obs, target_action)
        grad_norm = self.optimizer_update(self.actor_optimizer, actor_loss)
        return actor_loss.item(), grad_norm.item()

    def update_target_action(self, obs, action, critic):
        critic.requires_grad_(False)
        lim = 1 - 1e-5
        action.clamp_(-lim, lim)
        action_optimizer = torch.optim.Adam([action], lr=self.cfg.diffusion.action_lr, eps=1e-5)
        for i in range(self.cfg.diffusion.update_times):
            action.requires_grad_(True)
            Q = critic.get_q_min(obs, action)
            loss = -Q.mean()
            self.optimizer_update(action_optimizer, loss)
            action.requires_grad_(False)
            action.clamp_(-lim, lim)
        target_action = action.detach()
        update = deepcopy(target_action)
        critic.requires_grad_(True)
        return torch.abs(action).mean().item(), update

