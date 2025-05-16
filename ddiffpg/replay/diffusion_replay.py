import torch
import numpy as np
import random
from copy import deepcopy
from ddiffpg.replay.simple_replay import create_buffer, DiffusionReplayBuffer
from ddiffpg.utils.Q_scheduler import Q_scheduler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from dtaidistance import dtw_ndim
from collections import deque


class DiffusionGoalBuffer:
    def __init__(self, cfg, capacity: int, obs_dim: int, action_dim: int, num_envs: int, max_episode_len=1000, device='cpu'):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.env_num = num_envs
        self.max_episode_len = max_episode_len
        self.capacity = capacity
        
        if 'antmaze' in self.cfg.env.name:
            self.traj_dim = 2
        else:
            self.traj_dim = 3
        
        self.unsuccess = []
        self.unsuccess_id = []
        self.success_id = []
        self.success = []
        self.lengths = deque(maxlen=self.capacity)
        self.count = 0
        self.map = {}
        self.plot = None
        self.clusters, self.plot = [], []

        self.Q_scheduler = Q_scheduler(cfg, obs_dim, action_dim)
        explore_Q, Qs, indices, embeddings = self.Q_scheduler.update_cluster(self.clusters)
        self.Qs = [explore_Q] + Qs
        self.embeddings = embeddings
        # temp trajectory storage
        ret = create_buffer(capacity=(self.max_episode_len, self.env_num), obs_dim=obs_dim, action_dim=action_dim, device=device)
        self.traj_state, self.traj_action, self.traj_next_state, self.traj_reward, self.traj_done = ret
        self.traj_target_action = torch.empty_like(self.traj_action)
        self.replay_buffer = DiffusionReplayBuffer(capacity=capacity,
                              obs_dim=obs_dim,
                              action_dim=action_dim,
                              device=device)

    @torch.no_grad()
    def add_to_buffer(self, obs, action, reward, next_obs, done, info):
        assert self.env_num == info['step'].shape[0]
        
        self.temp_state, self.temp_action, self.temp_reward, self.temp_next_state, self.temp_done = [], [], [], [], []
        # copy the obs, action, ..., into temp storage
        for env_id in range(self.env_num):
            self.traj_state[int(info['step'][env_id])-1][env_id] = obs[env_id].clone()
            self.traj_action[int(info['step'][env_id])-1][env_id] = action[env_id].clone()
            self.traj_target_action[int(info['step'][env_id])-1][env_id] = deepcopy(action[env_id])
            self.traj_reward[int(info['step'][env_id])-1][env_id] = reward[env_id].clone()
            self.traj_next_state[int(info['step'][env_id])-1][env_id] = next_obs[env_id].clone()
            self.traj_done[int(info['step'][env_id])-1][env_id] = done[env_id].clone()
            
            self.temp_state.append(self.traj_state[:int(info['step'][env_id]), env_id].clone())
            self.temp_action.append(self.traj_action[:int(info['step'][env_id]), env_id].clone())
            self.temp_reward.append(self.traj_reward[:int(info['step'][env_id]), env_id].clone())
            self.temp_next_state.append(self.traj_next_state[:int(info['step'][env_id]), env_id].clone())
            self.temp_done.append(self.traj_done[:int(info['step'][env_id]), env_id].clone())

        self.temp_state = torch.cat(self.temp_state).reshape(-1, self.obs_dim)
        self.temp_action = torch.cat(self.temp_action).reshape(-1, self.action_dim)
        self.temp_reward = torch.cat(self.temp_reward).reshape(-1, 1)
        self.temp_next_state = torch.cat(self.temp_next_state).reshape(-1, self.obs_dim)
        self.temp_done = torch.cat(self.temp_done).reshape(-1, 1).bool()

        # if there is trajectory done
        if 'success' in info.keys():
            assert info['success'].shape[0] == self.env_num

            for i in range(info['indices'].shape[0]):
                k = info['indices'][i].item()
                success = info['success'][k].item()
                # store full traj into buffer
                s = self.traj_state[:int(info['step'][k].item()), k].clone()
                a = self.traj_action[:int(info['step'][k].item()), k].clone()
                t_a = self.traj_target_action[:int(info['step'][k].item()), k].clone()
                r = self.traj_reward[:int(info['step'][k].item()), k].clone()
                n_s = self.traj_next_state[:int(info['step'][k].item()), k].clone()
                d = self.traj_done[:int(info['step'][k].item()), k].clone()
                
                if success != 0:
                    cur_traj = trajectory(self.count, s, success!=0, self.traj_dim)
                    self.success.append(cur_traj)
                    self.lengths.append(s.shape[0])
                else:
                    cur_traj = trajectory(self.count, s, success!=0, self.traj_dim)
                    self.unsuccess.append(cur_traj)
                    self.unsuccess_id.append(self.count)
                    assert len(self.unsuccess_id) == len(self.unsuccess)

                self.replay_buffer.add_to_buffer((s, a, t_a, r, n_s, d), self.count)
                self.count += 1
        
        # empty temp traj storage
        if info['indices'].shape[0] != 0:
            self.traj_state[:, info['indices']] = torch.zeros((self.max_episode_len, info['indices'].shape[0], self.obs_dim)).to(self.device)
            self.traj_action[:, info['indices']] = torch.zeros((self.max_episode_len, info['indices'].shape[0], self.action_dim)).to(self.device)
            self.traj_target_action[:, info['indices']] = torch.zeros((self.max_episode_len, info['indices'].shape[0], self.action_dim)).to(self.device)
            self.traj_reward[:, info['indices']] = torch.zeros((self.max_episode_len, info['indices'].shape[0], 1)).to(self.device)
            self.traj_next_state[:, info['indices']] = torch.zeros((self.max_episode_len, info['indices'].shape[0], self.obs_dim)).to(self.device)
            self.traj_done[:, info['indices']] = torch.zeros((self.max_episode_len, info['indices'].shape[0], 1), dtype=torch.bool).to(self.device)
    
    def update_cluster(self):
        self.clusters, self.success_id, self.plot = self.cluster()

        # update the Q functions
        if len(self.clusters) > 10:
            print("Num clusters:", len(self.clusters), "More than 10 clusters, consider as one")
            self.clusters = [self.success_id] 
            explore_Q, Qs, indices, embeddings = self.Q_scheduler.update_cluster(self.clusters)
        else:
            explore_Q, Qs, indices, embeddings = self.Q_scheduler.update_cluster(self.clusters)
            print("Num clusters:", len(self.clusters), "Indices:", indices)

        # balance the mode trajectories
        if len(self.clusters) != 0:
            maximum_length = self.capacity // (2 * len(self.clusters))
        for i in range(len(self.clusters)):
            if len(self.clusters[i]) > maximum_length:
                remove_idx = random.sample(self.clusters[i], len(self.clusters[i])-maximum_length)
                a = []
                for j in range(len(self.success)):
                    if self.success[j].id in remove_idx:
                        a.append(self.success[j])
                for k in a:
                    self.clusters[i].remove(k.id)
                    self.success_id.remove(k.id)
                    self.success.remove(k)
                self.replay_buffer.remove(remove_idx)
            assert len(self.success_id) == len(self.success)

        tt = [[] for _ in range(len(self.clusters))]
        for i in range(len(self.success)):
            for j in range(len(self.clusters)):
                if self.success[i].id in self.clusters[j]:
                    tt[j].append(self.success[i].length)
        
        # add unsuccess traj into the cluster
        self.unsuccess_clusters, self.unsuccess_plot = self.unsuccess_cluster()
        assert len(self.unsuccess_clusters) == len(self.clusters)
        if len(self.unsuccess_clusters) == 0:
            self.unsuccess_clusters = [deepcopy(self.unsuccess_id)]
            maximum_length = self.capacity
        for i in range(len(self.unsuccess_clusters)):
            if len(self.unsuccess_clusters[i]) > maximum_length:
                remove_idx = random.sample(self.unsuccess_clusters[i], len(self.unsuccess_clusters[i])-maximum_length)
                a = []
                for j in range(len(self.unsuccess)):
                    if self.unsuccess[j].id in remove_idx:
                        a.append(self.unsuccess[j])
                for k in a:
                    self.unsuccess_clusters[i].remove(k.id)
                    self.unsuccess_id.remove(k.id)
                    self.unsuccess.remove(k)
                self.replay_buffer.remove(remove_idx)
            assert len(self.unsuccess_id) == len(self.unsuccess)

        # the indices means the cluster rearrangement, also for the target actions
        # there is a one-to-one mapping between Q-cluster-target_actions
        self.replay_buffer.update_target_action_dim(indices)
        self.Qs = [explore_Q] + Qs
        self.embeddings = embeddings

    @torch.no_grad()
    def cluster(self):
        num_success = len(self.success)
        # only if there are at least two success trajectories then we can do cluster
        if num_success > 1:
            distance_mat = np.zeros((num_success, num_success))
            success_id, plot_traj = [], []
            for i in range(len(self.success)):
                success_id.append(self.success[i].id)
                plot_traj.append(self.success[i].get_2d())
                for j in range(len(self.success)):
                    # hashmap key
                    map_id_1 = str(self.success[i].id) + "+" + str(self.success[j].id)
                    map_id_2 = str(self.success[j].id) + "+" + str(self.success[i].id)
                    if distance_mat[i][j] != 0:
                        dis = distance_mat[i][j]
                    elif i == j:
                        dis = 0
                    elif map_id_1 not in self.map.keys():
                        # use 2d pos to do clustering
                        avg_len = sum(self.lengths)/len(self.lengths)
                        s1 = self.success[i].get_2d(target_len=avg_len if self.cfg.algo.use_downsampling else None) - 12
                        s2 = self.success[j].get_2d(target_len=avg_len if self.cfg.algo.use_downsampling else None) - 12
                        # compute DWT distance
                        dis = dtw_ndim.distance(s1, s2, use_c=True)
                        self.map[map_id_1] = dis
                        self.map[map_id_2] = dis
                    else:
                        assert self.map[map_id_1] == self.map[map_id_2]
                        dis = self.map[map_id_1]
                
                    distance_mat[i][j] = dis
                    distance_mat[j][i] = dis
            
            # convert distance matrix to condensed format
            distance_mat = squareform(distance_mat)
            # methods = ['average', 'centroid', 'single']
            Z = linkage(distance_mat, method='average')
            if self.cfg.algo.cluster_threshold is None:
                threshold = 0.7*max(Z[:,2])
            else:
                threshold = self.cfg.algo.cluster_threshold
            output = fcluster(Z, t=threshold, criterion='distance')
            num_clusters = len(set(output))
            clusters = []
            plot_clusters = []
            output = torch.tensor(output)
            count = 0
            for i in range(num_clusters):
                plot_idx = torch.where(output==i+1)[0]
                plot_clusters.append(plot_idx.tolist())
                # convert idx (e.g., [1, 1, 1, 2, 2, 2]) to actual idx in traj_buffer
                converted_idx = torch.tensor(success_id)[plot_idx].tolist()
                clusters.append(converted_idx)
                count += len(converted_idx)

            assert len(success_id) == count
        else:
            plot_traj, plot_clusters = [], []
            success_id, clusters = [], []
            Z = None
            for i in range(len(self.success)):
                success_id.append(self.success[i].id)
                clusters.append([self.success[i].id])
        
        return clusters, success_id, [plot_traj, plot_clusters, Z]

    @torch.no_grad()
    def sample_batch(self, batch_size, device=None):
        if device is None:
            device = self.device
       
        # define traj idx for each group
        groups = [self.success_id + list(self.unsuccess_id)]
        for i in range(len(self.clusters)):
            groups.append(self.clusters[i] + self.unsuccess_clusters[i])
        
        # determine batch_size for each group
        if batch_size % len(groups) != 0:
            batch_sizes = [batch_size // len(groups) for i in range(len(groups))]
            batch_sizes[0] = batch_sizes[0] + batch_size % len(groups)
        else:
            batch_sizes = [batch_size // len(groups) for i in range(len(groups))]
        assert sum(batch_sizes) == batch_size
        assert len(self.Qs) == len(groups)
        assert len(self.Qs) == len(self.embeddings)
        if self.replay_buffer.buf_target_action is not None:
            assert len(self.Qs) == self.replay_buffer.buf_target_action.shape[0]
        data_list = []
        for i in range(len(groups)):
            data, indices = self.add_temp_data(batch_sizes[i], groups[i], i, if_add_temp=i==0, device=device)
            cur = {
                    "Q": self.Qs[i],
                    "batch": data,
                    "indices": indices,
                    "embedding": self.embeddings[i],
                    }
            data_list.append(cur)
        return data_list

    def add_temp_data(self, batch_size, cluster_idx, target_idx, if_add_temp=True, device=None):
        temp_size = self.temp_state.shape[0]
        buffer_size = self.replay_buffer.get_buffer_size(cluster_idx)
        if if_add_temp:
            b_temp = int((temp_size / (temp_size + buffer_size)) * batch_size)
        else:
            b_temp = 0
        b_sample = batch_size - b_temp
        states = []
        actions = []
        target_actions = []
        rewards = []
        next_states = []
        dones = []
        if b_sample != 0:
            data, sample_indices = self.replay_buffer.sample_batch(b_sample, cluster_idx, target_idx)
            states.append(data[0])
            actions.append(data[1])
            target_actions.append(data[2])
            rewards.append(data[3])
            next_states.append(data[4])
            dones.append(data[5])
        else:
            sample_indices = None
        if b_temp != 0:
            indices = torch.randint(self.temp_state.shape[0], size=(b_temp,), device=self.device)
            states.append(self.temp_state[indices])
            actions.append(self.temp_action[indices])
            target_actions.append(self.temp_action[indices])
            rewards.append(self.temp_reward[indices])
            next_states.append(self.temp_next_state[indices])
            dones.append(self.temp_done[indices])
        
        return (
            torch.cat(states).to(device),
            torch.cat(actions).to(device),
            torch.cat(target_actions).to(device),
            torch.cat(rewards).to(device),
            torch.cat(next_states).to(device),
            torch.cat(dones).to(device).float()
        ), sample_indices

    @torch.no_grad()
    def update_target_action(self, data_list):
        Qs = []
        for i in range(len(data_list)):
            data = data_list[i]
            if data["indices"] is not None:
                self.replay_buffer.update_target_action(data["new_action"][:data["indices"].shape[0]], data["indices"], i)
            Qs.append(data["Q"])
        self.Q_scheduler.update_Qs(Qs[0], Qs[1:])

    def unsuccess_cluster(self):
        unsuccess_clusters = [[] for i in range(len(self.clusters))]
        plot_traj = []
        plot_clusters = [[] for i in range(len(self.clusters))]
        for i in range(len(self.unsuccess)):
            min_dis = None
            belong_group = None
            for j in range(len(self.clusters)):
                if len(self.clusters[j]) < 3:
                    sample_len = len(self.clusters[j])
                else:
                    sample_len = 3
                sample_idx = random.sample(self.clusters[j], sample_len)
                total_dis = 0
                for k in sample_idx:
                    cur_success = self.success[self.success_id.index(k)]
                    assert cur_success.id == k
                    # hashmap key
                    map_id_1 = str(cur_success.id) + "+" + str(self.unsuccess[i].id)
                    map_id_2 = str(self.unsuccess[i].id) + "+" + str(cur_success.id)
                    if map_id_1 not in self.map.keys():
                        # use 2d pos to do clustering
                        s1 = cur_success.get_2d() - 12
                        s2 = self.unsuccess[i].get_2d() - 12
                        # compute DWT distance
                        dis = dtw_ndim.distance(s1, s2, use_c=True)
                        self.map[map_id_1] = dis
                        self.map[map_id_2] = dis
                    else:
                        assert self.map[map_id_1] == self.map[map_id_2]
                        dis = self.map[map_id_1]
                    total_dis += dis
                
                total_dis = total_dis / sample_len
                if min_dis is None:
                    min_dis = total_dis
                    belong_group = j
                elif total_dis < min_dis:
                    min_dis = total_dis
                    belong_group = j
            
            if belong_group is not None:
                unsuccess_clusters[belong_group].append(self.unsuccess[i].id)
                plot_clusters[belong_group].append(i)
                plot_traj.append(self.unsuccess[i].get_2d())
            else:
                assert len(self.clusters) == 0
        return unsuccess_clusters, [plot_traj, plot_clusters]


class trajectory:
    def __init__(self, traj_id: int, state, success=bool, traj_dim=2):
        self.id = traj_id
        self.state = state[:, :traj_dim]
        self.length = state.shape[0]
        self.success = success

    def get_2d(self, target_len=None):
        # return numpy for fast distance computation
        if target_len is None:
            return self.state.cpu().numpy().astype(np.double)
        else:
            return self.downsample(target_len).cpu().numpy().astype(np.double)
        
    def downsample(self, target_len):
        indices = torch.linspace(0, self.state.shape[0]-1, steps=int(target_len)).long()
        return self.state[indices]
    
