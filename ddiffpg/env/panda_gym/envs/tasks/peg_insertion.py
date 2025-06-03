from typing import Any, Dict
import os
import numpy as np
import random
import pybullet as p
from ddiffpg.env.panda_gym.envs.core import Task
from panda_gym.utils import distance
import panda_gym
from copy import deepcopy

import os.path

MODULE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class PegInsertion(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.1,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        # hole params
        self.hole_file = MODULE_PATH + "/assets/objects/Hole/Hole.urdf"
        z_hole = 0.03
        self.z_hole_offset = -0.02
        # add hole here!
        self.init_hole_poses = [np.array([0.05, 0.15, z_hole]),
                                np.array([0.05, -0.15, z_hole])]
        self.r_pos_hole = 0.1
        self.random_hole = False
        # for internal usage
        self.target_hole_poses = []
        # multiple holes possible
        self.body_holes = []
        ang = -np.pi * 0.5
        self.hole_ori = p.getQuaternionFromEuler([ang, 0, 0])

        with self.sim.no_rendering():
            self._create_scene()
        self.once = True  # TODO: delete me


    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.3, width=0.7, height=0.4, x_offset=-0.3)
        self.create_hole()

    def create_hole(self):
        # loading hole here
        # positining in reset
        # make it a list for multiple ones

        for i, init_target_pos in enumerate(self.init_hole_poses):

            hole_name = f"hole_{i}"

            self.sim.loadURDF(body_name=hole_name, fileName=self.hole_file, basePosition=init_target_pos,
                              baseOrientation=(self.hole_ori[0], self.hole_ori[1], self.hole_ori[2], self.hole_ori[3]),
                              globalScaling=1, useFixedBase=True)
            self.body_holes.append(hole_name)
            self.target_hole_poses.append(init_target_pos)


    def _reset_holes(self, randomize=False):
        # update target
        targets = []
        if randomize:
            for i in range(len(self.init_hole_poses)):
                noise = np.random.uniform(low=-self.r_pos_hole, high=self.r_pos_hole, size=3)
                noise[2] = 0  # no height randomization
                targets.append(self.init_hole_poses[i] + noise)
        else:
            targets = deepcopy(self.init_hole_poses)
        # reset holes
        for body, target in zip(self.body_holes, targets):
            self.sim.set_base_pose(body=body,  position=target, orientation=self.hole_ori)
        return targets

    def get_hole_poses(self):
        return self.target_hole_poses

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        return self.get_ee_position()

    def get_goal(self):
        goal = np.array([0.05, 0.15, 0.01, 0.05, -0.15, 0.01])
        return goal

    def reset(self) -> None:
        self.target_hole_poses = self._reset_holes(randomize=False)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d_1 = distance(achieved_goal, desired_goal[:3])
        d_2 = distance(achieved_goal, desired_goal[-3:])
        d_close = min(d_1, d_2)
        return np.array(d_close < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        special_reward = False

        if not special_reward:
            d_1 = distance(achieved_goal, desired_goal[:3])
            d_2 = distance(achieved_goal, desired_goal[-3:])
            d_close = min(d_1, d_2)
            if self.reward_type == "sparse":
                if d_close > self.distance_threshold:
                    return np.array(0, dtype=np.float32)
                else:
                    return np.array(10, dtype=np.float32)
            else:
                return -d_close.astype(np.float32)
        else:
            # special reward from minitouch 
            if distance(achieved_goal[0:2], desired_goal[0:2]) < self.distance_threshold:
                if achieved_goal[2] <= self.distance_threshold:
                    finish_reward = 1
                else:
                    finish_reward = 0
                insertion_reward = 1*(0.2 - achieved_goal[2])
                return finish_reward + insertion_reward
            else:
                align_penalty = 5*(self.distance_threshold - distance(achieved_goal[0:2], desired_goal[0:2]))
                return align_penalty
