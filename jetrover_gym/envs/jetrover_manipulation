from gymnasium import spaces
import numpy as np

from kinematics import transform
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import get_ik, get_position_ik, set_joint_range, get_joint_range
from servo_controller import getServoPulse, runActionGroup

servo_ids = (1, 2, 3, 4, 5, 10)
gripper_range = (100, 600)

class JetRoverManipulation():
    """Hiwonder JetRover manipulation environment backed by ROS 2 and IK."""

    def __init__(self, max_episode_steps=100, rgb=True, debug=False):
        """Initialize the environment.

        Args:
            max_episode_steps: max step after which the environment will be terminated.
            rgb: whether to include RGB image in observation.
            debug: if true, verbose output will be printed.
        """
        self._max_episode_steps = max_episode_steps
        self.rgb = rgb
        self._img_rgb_height = 640
        self._img_rgb_width = 480
        self._fk = ForwardKinematics(debug=debug)

        self.observation_space = spaces.Dict()
        # EE pos (xyz), rot (rpy), and gripper_state (open/close)
        self.observation_space['state'] = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=float
        )
        if self._rgb:
            self.observation_space['rgb'] = spaces.Box(
                low=0,
                high=255,
                shape=(self._img_rgb_width, self._img_rgb_height, 3),
                dtype=np.uint8
            )

        # EE delta_pos (xyz), delta_rot (rpy), and gripper_state (open/close)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=float
        ) 

    def _get_observation(self):
        """Get observation of the state

        Returns:
            obs: EE position (xyz), rotation (rpy), and gripper_state (open/close)
        """
        obs_state = []

        pulses = [getServoPulse(id) for id in servo_ids]
        angles = transform.pulse2angle(pulses[:-1])
        pos, quat = fk.get_fk(angles)
        rpy = transform.qua2rpy(quat)
        gripper_pulse = pulses[-1]
        gripper_state = (gripper_pulse - gripper_range[0]) / (gripper_range[1] - gripper_range[0])

        obs_state.extend(pos)
        obs_state.extend(rpy)
        obs_state.append(state)

        obs = spaces.Dict()
        obs['state'] = np.array(obs_state)
        if self._rgb:
            obs['rgb'] = ...
        return obs

    def _control(self, action: np.ndarray) -> bool:
        """Apply a delta action by solving IK and sending commands.

        Args:
            action: EE delta_position (xyz), delta_rotation (rpy), and gripper_state (open/close)

        Returns:
            result: True if success to apply the action
        """
        # 1. Get xyz and rpy using Forward kinematics 
        # 2. Calculate target xyz and rpy
        # 3. Get values using inverse kinematics
        # 4. Change the servos using ik and gripper_state 
        result = False
        return result
    
    def reset(self):
        """Reset the robot to initial position."""
        runActionGroup('init')
        obs = self._get_observation()
        info = dict()
        return obs, info

    def step(self, action: np.ndarray):
        """Advance the environment by one step.

        Args:
            action: EE delta_position (xyz), delta_rotation (rpy), and gripper_state (open/close)
        """
        result = self._control(action)

        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            'action_result': result
        }
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = JetRoverManipulation(rgb=False, debug=True)

    obs, info = env.reset()
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(f"action: {action}, obs: {obs}, reward: {reward}")

    # Flow:
    # 1. Implement Gym interface (get_observation, reset, control)
    # 2. Implement Teleoperation using keyboard and store dataset
    # 3. Replay buffer to load dataset and pre-process state/action/reward
    # 4. Modify IQL 