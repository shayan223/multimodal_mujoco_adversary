from typing import Any, Dict

import math
import numpy as np
from ddiffpg.env.panda_gym.envs.core import Task
from panda_gym.utils import distance
import os
MODULE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Cabinet(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",  # TODO: specify options (if there are any)
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.get_ee_position = get_ee_position
        # cabinet
        self.cabinet_file_path = MODULE_PATH + "/assets/objects/cabinet/cabinet_0004.urdf"

        self.cabinet_joint = 1
        self.cabinet_body = "cabinet"

        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.0, width=1.2, height=0.4, x_offset=-0.3)
        self._create_cabinet()

    def _create_cabinet(self):
        cabinet_ori = [0, 0, math.pi]  # math.pi/2
        self.sim.loadURDF(body_name=self.cabinet_body,
                          fileName=self.cabinet_file_path, basePosition=[-0.22, 0.1, 0.53],  # -0.1, 0.0, 0.43
                          globalScaling=1, baseOrientation=self.sim.get_quat_euler(cabinet_ori),
                          useFixedBase=True)

        self._reset_cabinet()

    def _reset_cabinet(self):
        init_cabinet_joint_state = 0.0
        self.sim.set_joint_angle(body=self.cabinet_body, joint=self.cabinet_joint, angle=init_cabinet_joint_state)

    def _get_cabinet_joint_pos(self):
        j_pos = self.sim.get_joint_angle(self.cabinet_body, self.cabinet_joint)
        return j_pos

    def get_obs(self) -> np.ndarray:
        cabinet_joint_pos = self._get_cabinet_joint_pos()
        return np.array([cabinet_joint_pos])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        cabinet_joint_pos = self._get_cabinet_joint_pos()
        return np.array([cabinet_joint_pos])

    def get_goal(self):
        # fixed goal (open cabinet_joint)
        return np.array([0.4])

    def reset(self) -> None:
        self._reset_cabinet()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return achieved_goal >= desired_goal

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            rew = 10 if self.is_success(achieved_goal, desired_goal) else 0
            return np.array([rew])
        else:
            # opening distance measured in joint-space of cabinet
            return -d.astype(np.float32)
