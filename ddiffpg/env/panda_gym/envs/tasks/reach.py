from typing import Any, Dict

import numpy as np
import pybullet as p
from ddiffpg.env.panda_gym.envs.core import Task
from panda_gym.utils import distance


class Reach(Task):
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
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=self.distance_threshold,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self._create_obstacle()

    def _create_obstacle(self):
        self.ob_poses = [
            [-0.22, 0.0, 0.67],
            [-0.22, 0.0, 0.67],
        ]
        self.ob_extents = [
            [0.02, 0.005, 0.2],
            [0.02, 0.2, 0.005],
        ]
        for pos, ext in zip(self.ob_poses, self.ob_extents):
            self.create_cube(position=pos,
                        halfExtents=ext,
                        mass=0,
                        collision=True)

    @staticmethod
    def create_cube(position, halfExtents, mass=0, collision=True):
        visual_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                        halfExtents=halfExtents,
                                        rgbaColor=[1, 0, 1, 1],
                                        specularColor=[0.8, .0, 0])

        collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                            halfExtents=halfExtents)

        if collision:
            p.createMultiBody(baseMass=mass,
                              baseInertialFramePosition=[0, 0, 0],
                              baseCollisionShapeIndex=collision_id,
                              baseVisualShapeIndex=visual_id,
                              basePosition=position,
                              useMaximalCoordinates=True)
        else:
            p.createMultiBody(baseMass=mass,
                              baseInertialFramePosition=[0, 0, 0],
                              # baseCollisionShapeIndex=collision_id,
                              baseVisualShapeIndex=visual_id,
                              basePosition=position,
                              useMaximalCoordinates=True)

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([-0.05, 0.0, 0.65])
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            if d > self.distance_threshold:
                return np.array(0, dtype=np.float32)
            else:
                return np.array(10, dtype=np.float32)
        else:
            return -d.astype(np.float32)
