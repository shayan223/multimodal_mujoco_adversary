import torch
from copy import deepcopy


def create_buffer(capacity, obs_dim, action_dim, device='cuda'):
    if isinstance(capacity, int):
        capacity = (capacity,)
    buf_obs_size = (*capacity, obs_dim) if isinstance(obs_dim, int) else (*capacity, *obs_dim)
    buf_obs = torch.empty(buf_obs_size,
                          dtype=torch.float32, device=device)
    buf_action = torch.empty((*capacity, int(action_dim)),
                             dtype=torch.float32, device=device)
    buf_reward = torch.empty((*capacity, 1),
                             dtype=torch.float32, device=device)
    buf_next_obs = torch.empty(buf_obs_size,
                               dtype=torch.float32, device=device)
    buf_done = torch.empty((*capacity, 1),
                           dtype=torch.bool, device=device)
    return buf_obs, buf_action, buf_next_obs, buf_reward, buf_done


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device='cpu'):
        self.obs_dim = obs_dim
        if isinstance(obs_dim, int):
            self.obs_dim = (self.obs_dim,)
        self.action_dim = action_dim
        self.device = device
        self.next_p = 0  # next pointer
        self.if_full = False
        self.cur_capacity = 0  # current capacity
        self.capacity = int(capacity)
        self.total_samples = 0
        self.sample_idx = None

        ret = create_buffer(capacity=self.capacity, obs_dim=obs_dim, action_dim=action_dim, device=device)
        self.buf_obs, self.buf_action, self.buf_next_obs, self.buf_reward, self.buf_done = ret
        self.buf_target_action = torch.empty_like(self.buf_action)

    @torch.no_grad()
    def add_to_buffer(self, trajectory):
        obs, actions, rewards, next_obs, dones = trajectory
        obs = obs.reshape(-1, *self.obs_dim)
        actions = actions.reshape(-1, self.action_dim)
        rewards = rewards.reshape(-1, 1)
        next_obs = next_obs.reshape(-1, *self.obs_dim)
        dones = dones.reshape(-1, 1).bool()
        p = self.next_p + rewards.shape[0]
        self.total_samples += rewards.shape[0]

        if p > self.capacity:
            self.if_full = True

            self.buf_obs[self.next_p:self.capacity] = obs[:self.capacity - self.next_p]
            self.buf_action[self.next_p:self.capacity] = actions[:self.capacity - self.next_p]
            self.buf_target_action[self.next_p:self.capacity] = actions[:self.capacity - self.next_p]
            self.buf_reward[self.next_p:self.capacity] = rewards[:self.capacity - self.next_p]
            self.buf_next_obs[self.next_p:self.capacity] = next_obs[:self.capacity - self.next_p]
            self.buf_done[self.next_p:self.capacity] = dones[:self.capacity - self.next_p]

            p = p - self.capacity
            self.buf_obs[0:p] = obs[-p:]
            self.buf_action[0:p] = actions[-p:]
            self.buf_target_action[0:p] = actions[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_next_obs[0:p] = next_obs[-p:]
            self.buf_done[0:p] = dones[-p:]
        else:
            self.buf_obs[self.next_p:p] = obs
            self.buf_action[self.next_p:p] = actions
            self.buf_target_action[self.next_p:p] = actions
            self.buf_reward[self.next_p:p] = rewards
            self.buf_next_obs[self.next_p:p] = next_obs
            self.buf_done[self.next_p:p] = dones

        self.next_p = p  # update pointer
        self.cur_capacity = self.capacity if self.if_full else self.next_p

    @torch.no_grad()
    def sample_batch(self, batch_size, device='cuda'):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=device)
        self.sample_idx = indices

        return (
            self.buf_obs[indices].to(device),
            self.buf_action[indices].to(device),
            self.buf_target_action[indices].to(device),
            self.buf_reward[indices].to(device),
            self.buf_next_obs[indices].to(device),
            self.buf_done[indices].to(device).float()
        )

    @torch.no_grad()
    def update_target_action(self, new_action):
        self.buf_target_action[self.sample_idx] = new_action


class DiffusionReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device='cpu'):
        self.obs_dim = obs_dim
        if isinstance(obs_dim, int):
            self.obs_dim = (self.obs_dim,)
        self.action_dim = action_dim
        self.device = device
        self.cur_capacity = 0  # current capacity
        self.capacity = int(capacity)
        self.last_sample = None

        self.buf_obs = None
        self.buf_action = None 
        self.buf_next_obs = None
        self.buf_reward = None 
        self.buf_done = None
        self.buf_id = None
        self.buf_target_action = None

    @torch.no_grad()
    def add_to_buffer(self, trajectory, traj_id):
        obs, actions, target_actions, rewards, next_obs, dones = trajectory
        # reformat data
        obs = obs.reshape(-1, *self.obs_dim)
        actions = actions.reshape(-1, self.action_dim)
        target_actions = target_actions.reshape(1, -1, self.action_dim)
        rewards = rewards.reshape(-1, 1)
        next_obs = next_obs.reshape(-1, *self.obs_dim)
        dones = dones.reshape(-1, 1).bool()
        traj_id = torch.ones_like(rewards) * traj_id
        
        if self.buf_obs is None:
            self.buf_obs = obs
            self.buf_action = actions
            self.buf_next_obs = next_obs
            self.buf_reward = rewards
            self.buf_done = dones
            self.buf_id = traj_id
            self.buf_target_action = target_actions
        else:
            target_actions = target_actions.repeat(self.buf_target_action.shape[0], 1, 1)
            self.buf_obs = torch.cat([self.buf_obs, obs])
            self.buf_action = torch.cat([self.buf_action, actions])
            self.buf_next_obs = torch.cat([self.buf_next_obs, next_obs])
            self.buf_reward = torch.cat([self.buf_reward, rewards])
            self.buf_done = torch.cat([self.buf_done, dones])
            self.buf_id = torch.cat([self.buf_id, traj_id])
            self.buf_target_action = torch.cat([self.buf_target_action, target_actions], dim=1)

        self.cur_capacity = self.buf_obs.shape[0]

    @torch.no_grad()
    def sample_batch(self, batch_size, cluster_idx, target_idx, device='cuda'):
        available_idx = torch.where(torch.isin(self.buf_id, torch.tensor(cluster_idx, device=device)))[0]
        indices = torch.randint(available_idx.shape[0], size=(batch_size,), device=device)
        indices = available_idx[indices]

        return (
            self.buf_obs[indices].to(device),
            self.buf_action[indices].to(device),
            self.buf_target_action[target_idx, indices].to(device),
            self.buf_reward[indices].to(device),
            self.buf_next_obs[indices].to(device),
            self.buf_done[indices].to(device).float()
        ), indices
    
    def remove(self, target_idx, device='cuda'):
        remove_idx = torch.where(torch.isin(self.buf_id, torch.tensor(target_idx, device=device)))[0]
        keep_idx = torch.ones(self.buf_obs.shape[0], dtype=bool)
        keep_idx[remove_idx] = False
        prev = len(set(torch.unique(self.buf_id).tolist()))

        self.buf_obs = self.buf_obs[keep_idx]
        self.buf_action = self.buf_action[keep_idx]
        self.buf_next_obs = self.buf_next_obs[keep_idx]
        self.buf_reward = self.buf_reward[keep_idx]
        self.buf_done = self.buf_done[keep_idx]
        self.buf_id = self.buf_id[keep_idx]
        self.buf_target_action = self.buf_target_action[:, keep_idx]
        after = len(set(torch.unique(self.buf_id).tolist()))

        assert prev == after + len(target_idx)

    def get_buffer_size(self, cluster_idx):
        if self.buf_id is None:
            return 0
        return torch.where(torch.isin(self.buf_id, torch.tensor(cluster_idx).to(self.device)))[0].shape[0]
    
    def update_target_action_dim(self, indices):
        if len(indices) == 0:
            return
        new_target_action = [deepcopy(self.buf_target_action[0])]
        assert max(indices) < self.buf_target_action.shape[0]
        for idx in indices:
            if idx == -1:
                new_target_action.append(deepcopy(self.buf_action))
            else:
                new_target_action.append(deepcopy(self.buf_target_action[idx]))
        self.buf_target_action = torch.stack(new_target_action)

    @torch.no_grad()
    def update_target_action(self, new_action, indices, i):
        self.buf_target_action[i, indices] = new_action
