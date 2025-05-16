from dataclasses import dataclass
from typing import Any
import itertools
from gym import spaces
import torch
import numpy as np


class D4RLEnvWrapper:
    def __init__(self, env, episode_len):
        self.env = env
        self.observation_space = np.zeros(self.env.observation_space.shape[1])
        self.action_space = np.zeros(self.env.action_space.shape[1])
        self.max_episode_length = episode_len
        self.device = torch.device("cuda:0")

    def reset(self):
        ob = self.env.reset()
        return self.cast(ob)

    def step(self, actions):
        actions = actions.cpu().numpy()
        next_obs, rewards, dones, infos = self.env.step(actions)
        timeout = torch.zeros(dones.shape).bool().to(self.device)
        success = torch.zeros(dones.shape).to(self.device)
        for i in range(len(infos)):
            if len(infos[i]) == 0:
                pass
            else:
                if "TimeLimit.truncated" in infos[i].keys():
                    timeout[i] = infos[i]["TimeLimit.truncated"]
                if "success" in infos[i].keys():
                    success[i] = infos[i]["success"]
        info_ret = {"time_outs": timeout, "success": success}

        return (
            self.cast(next_obs),
            self.cast(rewards),
            self.cast(dones).long(),
            info_ret,
        )

    def cast(self, x):
        x = torch.Tensor(x).to(self.device)
        return x


@dataclass
class D4RLRPGEnvWrapper:
    env: Any

    def __post_init__(self):
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.env.observation_space.shape[1],),
            dtype=np.float32,
        )
        self.observation_space = [self.observation_space for _ in range(5)]
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.env.action_space.shape[1],), dtype=np.float32
        )
        self.action_space = [self.action_space for _ in range(5)]
        self._max_episode_steps = 500
        self._max_episode_steps = [self._max_episode_steps for _ in range(5)]
        self.device = torch.device("cuda:0")

    def reset(self):
        ob = self.env.reset()
        return ob

    def step(self, actions):
        next_obs, rewards, dones, infos = self.env.step(actions)
        return next_obs, rewards, dones, infos

    def cast(self, x):
        x = torch.Tensor(x).to(self.device)
        return x


@dataclass
class D4RLConsistencyEnvWrapper:
    env: Any

    def __post_init__(self):
        self.single_observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.env.observation_space.shape[1],),
            dtype=np.float32,
        )
        self.observation_space = [self.single_observation_space for _ in range(5)]
        self.single_action_space = spaces.Box(
            -1.0, 1.0, shape=(self.env.action_space.shape[1],), dtype=np.float32
        )
        self.action_space = [self.single_action_space for _ in range(5)]
        self._max_episode_steps = 500
        self._max_episode_steps = [self._max_episode_steps for _ in range(5)]
        self.device = torch.device("cuda:0")

    def reset(self):
        ob = self.env.reset()
        return ob

    def step(self, actions):
        next_obs, rewards, dones, infos = self.env.step(actions)
        return next_obs, rewards, dones, infos

    def cast(self, x):
        x = torch.Tensor(x).to(self.device)
        return x
