from typing import Any, Dict
import os
import numpy as np
import pybullet as p
from ddiffpg.env.panda_gym.envs.core import Task
MODULE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class DrawerMulti(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.get_ee_position = get_ee_position  # that's a method

        # drawer
        self.drawer_setting = 3
        self.drawer_joint = 1
        self.drawer_file_path = MODULE_PATH + "/assets/objects/cabinet/drawer_1.urdf"

        if self.drawer_setting == 1:
            self.drawer_names = ["drawer_1", "drawer_2", "drawer_3"]
            self.drawer_poses = [[0.3, -0.5, 0.18], [0.3, 0.0, 0.18], [0.3, 0.5, 0.18]]
            self.drawer_j_poses = [0.17, 0.17, 0.17]
            self.drawer_scale = 0.75
        elif self.drawer_setting == 2:
            self.drawer_names = ["drawer_1", "drawer_2", "drawer_3"]
            z_offset = 0.25
            x_drawer = -0.35
            self.drawer_poses = [[x_drawer, 0.0, 0.1+z_offset],
                                 [x_drawer, 0.0, 0.3+z_offset],
                                 [x_drawer, 0.0, 0.5+z_offset]]
            self.drawer_j_poses = [0.17, 0.17, 0.17]
            self.drawer_scale = 0.55
        elif self.drawer_setting == 3:
            self.drawer_names = ["drawer_1", "drawer_2", "drawer_3", "drawer_4"]
            z_offset = 0.28  # 0.25
            x_drawer = -0.35
            y_offset = 0.185
            self.drawer_poses = [[x_drawer, -y_offset, 0.2+z_offset],
                                 [x_drawer, -y_offset, 0.4+z_offset],
                                 [x_drawer, y_offset, 0.2+z_offset],
                                 [x_drawer, y_offset, 0.4+z_offset]]
            self.drawer_j_poses = [0.17, 0.17, 0.17, 0.17]
            self.drawer_scale = 0.55
        else:
            print("using default drawer setup. change drawer setting for other setup")
            self.drawer_names = ["drawer"]
            self.drawer_poses = [[0.3, 0.0, 0.18]]
            self.drawer_j_poses = [0.15]

        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=2.0, width=1.2, height=0.4, x_offset=-0.5)
        self._create_drawers()

    def _create_drawers(self):
        for name, d_pos in zip(self.drawer_names, self.drawer_poses):
            self.sim.loadURDF(body_name=name,
                              fileName=self.drawer_file_path, basePosition=d_pos,
                              globalScaling=self.drawer_scale, baseOrientation=[0, 180, 0, 1],
                              useFixedBase=True)
        self._reset_drawers()

    def _reset_drawers(self):
        for name, j_pos in zip(self.drawer_names, self.drawer_j_poses):
            self.sim.set_joint_angle(body=name, joint=0, angle=j_pos)

    def _get_drawer_joint_poses(self):
        j_poses = []
        for name in self.drawer_names:
            j_poses.append(self.sim.get_joint_angle(name, 0))
        return j_poses

    def get_obs(self) -> np.ndarray:
        return np.array(self._get_drawer_joint_poses())

    def get_achieved_goal(self) -> np.ndarray:
        drawer_joint_poses = self._get_drawer_joint_poses()
        min_drawer_j = np.min(np.array(drawer_joint_poses))
        return np.array([min_drawer_j])

    def get_goal(self):
        # fixed goal (close drawer_joint)
        return np.array([0.03])

    def reset(self) -> None:
        self._reset_drawers()

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        return achieved_goal <= desired_goal

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:

        if self.reward_type == "sparse":
            if achieved_goal <= desired_goal:
                return np.array(10, dtype=np.float32)
            else:
                return np.array(0, dtype=np.float32)
        else:
            return -achieved_goal

    def debug_sphere(self, pos, radius=0.05, color=None):
        if color is None:
            color = np.array([0.1, 0.9, 0.1, 0.3])
        self.sim.create_sphere(
            body_name="sphere",
            radius=radius,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=color,
        )