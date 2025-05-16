import torch
import numpy as np


class PybulletEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = np.zeros(self.env.observation_space['observation'].shape[1])
        self.action_space = np.zeros(self.env.action_space.shape[1])
        self.max_episode_length = 100
        self.device = torch.device("cuda:0")

    def reset(self):
        ob, info = self.env.reset()
        ob = self.cast(ob['observation'])
        return self.cast(ob)

    def step(self, actions):
        actions = actions.cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
        next_obs = self.cast(next_obs['observation'])
        dones = np.logical_or(terminated, truncated)
        timeout = torch.tensor(truncated).bool().to(self.device)
        success = torch.tensor(terminated).to(self.device)
        info_ret = {'time_outs': timeout, 'success': success}

        return self.cast(next_obs), self.cast(rewards), self.cast(dones).long(), info_ret

    def cast(self, x):
        x = torch.tensor(x).to(self.device)
        return x