from typing import Optional

import numpy as np
from gymnasium import spaces
import os
import panda_gym
from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet
import pybullet as p

MODULE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        random_init_pos: bool = False,
        has_peg: bool = False,
        cabinet: bool = False,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        self.has_peg = has_peg
        # print(MODULE_PATH)
        if self.has_peg:
            file_name_peg = MODULE_PATH + "/assets/franka_panda/panda_peg.urdf"
            super().__init__(
                sim,
                body_name="panda",
                file_name=file_name_peg,  # "franka_panda/panda.urdf",
                base_position=base_position,
                action_space=action_space,
                joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
                joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
            )
            self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
            # self.neutral_joint_values = np.array([0.00, -1.5, 0.00, -3, 0.00, 2.26, 0.79, 0.00, 0.0])
            self.ee_link = 14
        elif cabinet:
            super().__init__(
                sim,
                body_name="panda",
                file_name="franka_panda/panda.urdf",
                base_position=base_position,
                action_space=action_space,
                joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
                joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0])*100,
            )
            # self.neutral_joint_values = np.array([-0.2, 0.41, 0.0, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

            # little knick
            # self.neutral_joint_values = np.array([-0.2, -0.3, 0.0, -2.3, 0.00, 2.0, 0.79, 0.00, 0.00])

            self.neutral_joint_values = np.array([-0.2, -0.5, 0.0, -2.6, 0.00, 2.2, 0.79, 0.00, 0.00])
            
            # side init
            # self.neutral_joint_values = np.array([-0.9, 0.5, 0.5, -1.5, 0.9, 1.6, 0.79, 0.00, 0.00])

            # self.neutral_joint_values = np.array([0.00, -1.5, 0.00, -3, 0.00, 2.26, 0.79, 0.00, 0.0])
            self.ee_link = 11
        else:
            super().__init__(
                sim,
                body_name="panda",
                file_name="franka_panda/panda.urdf",
                base_position=base_position,
                action_space=action_space,
                joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
                joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
            )
            # self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
            self.neutral_joint_values = np.array([0.00, -1.5, 0.00, -3, 0.00, 2.26, 0.79, 0.00, 0.0])  # from reaching
            # self.neutral_joint_values = np.array([0.00, -0.5, 0.00, -2.2, 0.00, 1.8, 0.79, 0.025, 0.025])
            self.ee_link = 11
        self.fingers_indices = np.array([9, 10])

        self.random_init_pos = random_init_pos
        # last two joint are not controlled
        # TODO: tune starting pos randomization
        self.init_random_range = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0])

        # joint limits from urdf
        # lower, upper = self.sim.get_joint_limits(self.body_name, self.joint_indices)
        self.joint_limit_lower = np.array([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0.0, 0.0])
        self.joint_limit_upper = np.array([2.9671, 1.8326, 2.9671, 0., 2.9671, 3.8223, 2.9671, 0.04, 0.04])

        # self.sim.get_info("panda")

        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

        # add peg to panda
        from pybullet_utils import urdfEditor as ed
        # ang = -np.pi * 0.5
        # # peg in EE of robot (fixed)
        # peg_ori = p.getQuaternionFromEuler([ang, 0, 0])
        # module_path = os.path.dirname(os.path.abspath(panda_gym.__file__))
        # self.object_file_path = module_path + "/assets/objects/Peg/Peg.urdf"
        # self.get_ee_position()
        # self.sim.loadURDF("peg", fileName=self.object_file_path, basePosition=(
        #         self.get_ee_position()[0], self.get_ee_position()[1], self.get_ee_position()[2] + 0.03),
        #                                     baseOrientation=(peg_ori[0], peg_ori[1], peg_ori[2], peg_ori[3]),
        #                                     globalScaling=1)

        # p.changeDynamics(self.objectUid, -1, 1, lateralFriction=50, rollingFriction=50, spinningFriction=50, )
        # p.changeDynamics(self.objectUid, 0, 1, lateralFriction=0., rollingFriction=0., spinningFriction=0., )

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0.03  # 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        if self.random_init_pos:
            self.set_joint_random()
        else:
            self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def set_joint_random(self) -> None:
        """Set the robot to a random joint pos near the neutral pose."""
        # add noise to the neutral position
        noise = np.random.uniform(low=-self.init_random_range, high=self.init_random_range)
        joint_values = self.neutral_joint_values + noise
        # clip joint limits
        joint_values = np.clip(joint_values, self.joint_limit_lower, self.joint_limit_upper)

        self.set_joint_angles(joint_values)


    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def get_joint_limits(self) -> (np.ndarray, np.ndarray):
        return self.sim.get_joint_limits(self.body_name, self.joint_indices)