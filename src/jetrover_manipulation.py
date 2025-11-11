from time import sleep

from gymnasium import Env, spaces
import numpy as np

from jetrover_gym.src.manipulation_node import ManipulationNode
from manipulation.utils.kinematics_utils import pulse2angle, angle2pulse
from manipulation.kinematics import forward_kinematics, inverse_kinematics
from jetrover_gym.src.utils import T_to_xyz_rpy, xyz_rpy_to_T

ACTION_GROUPSAVE_DIR = '/home/ubuntu/software/arm_pc/ActionGroups'
GRIPPER_PULSE_RANGE = (100, 550)
INITIAL_GRIPPER_PULSE = 500
CONTROL_DURATION = 2.0 
SLEEP_DURATION = 1.0
EE_POS_LOWER = np.array([0.15, -0.001, 0.02], dtype=np.float32)
EE_POS_UPPER = np.array([0.23, 0.001, 0.27], dtype=np.float32)

pos_scale = 0.02
rpy_scale = 0.05

class JetRoverManipulation(Env):
    """Hiwonder JetRover manipulation environment backed by ROS 2 and IK."""

    def __init__(self,
                 name='jetrover_manipulation',
                 max_episode_steps=100,
                 rgb=True,
                 debug=False,
                 ik_error_threshold=0.01):
        """Initialize the environment.

        Args:
            name: name of the node to be used.
            max_episode_steps: max step after which the environment will be terminated.
            rgb: whether to include RGB image in observation.
            debug: if true, verbose output will be printed.
        """
        self._max_episode_steps = max_episode_steps
        self._rgb = rgb
        self._img_rgb_height = 640
        self._img_rgb_width = 480
        self._ik_error_threshold = ik_error_threshold
        self._debug = debug

        self._node = ManipulationNode(name)
        self._gripper_pulse = INITIAL_GRIPPER_PULSE

        self.observation_space = spaces.Dict()
        # EE pos (xyz), rot (rpy), and gripper_state (open/close)
        self.observation_space['state'] = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=float)
        if self._rgb:
            self.observation_space['rgb'] = spaces.Box(
                low=0,
                high=255,
                shape=(self._img_rgb_width, self._img_rgb_height, 3),
                dtype=np.uint8
            )

        # EE delta_pos (xyz), delta_rot (rpy), and gripper_state (open/close)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.float32),
            shape=(7,),
            dtype=np.float32
        ) 


    def _get_observation(self):
        """Get observation of the state

        Returns:
            obs: EE position (xyz), rotation (rpy), and gripper_state (open/close)
        """
        obs_state = []

        joint_pulses = self._node.get_joint_positions_pulse()
        joint_angles = pulse2angle(joint_pulses)
        transform_matrix = forward_kinematics(joint_angles, 'tcp')
        pos, rpy = T_to_xyz_rpy(transform_matrix)

        gmin, gmax = GRIPPER_PULSE_RANGE
        gripper_state = (self._gripper_pulse - gmin) / (gmax - gmin)

        obs_state.extend(pos)
        obs_state.extend(rpy)
        obs_state.append(gripper_state)

        obs = {'state': np.asarray(obs_state)}
        if self._rgb:
            obs['rgb'] = ...
        return obs

    def _bounded_ee_pos(self, pos: np.ndarray) -> np.ndarray:
        """Limit the position of the end-effector to be within the workspace.

        Args:
            pos: end-effector position

        Returns:
            pos: end-effector position
        """
        return np.clip(pos, EE_POS_LOWER, EE_POS_UPPER)

    def _control(self, action: np.ndarray) -> bool:
        """Apply a delta action by solving IK and sending commands.

        Args:
            action: EE delta_position (xyz), delta_rotation (rpy), and gripper_state (open/close)

        Returns:
            result: True if success to apply the action
        """
        # Get xyz and rpy using forward kinematics 
        joint_pulses = self._node.get_joint_positions_pulse()
        joint_angles = pulse2angle(joint_pulses)
        joint_angles = joint_angles.astype(np.float32)
        transform_matrix = forward_kinematics(joint_angles, 'tcp')
        pos, rpy = T_to_xyz_rpy(transform_matrix)

        # Calculate target xyz and rpy
        delta_pos = action[:3] * pos_scale
        delta_rpy = action[3:6] * rpy_scale
        gripper_action = action[6]

        # Get values using inverse kinematics
        target_pos = pos + delta_pos
        target_pos = self._bounded_ee_pos(target_pos)
        target_rpy = rpy + delta_rpy
        
        target_transform_matrix = xyz_rpy_to_T(target_pos, target_rpy)
        res = inverse_kinematics(joint_angles, target_transform_matrix)
        ik_error = res["pos_error"]

        if ik_error > self._ik_error_threshold:
            print(f"IK: no solution!: error = {res['pos_error']}")
            return False

        sol = res["sol"].astype(np.float32)
        target_joint_pulses = angle2pulse(sol)
        target_joint_pulses = target_joint_pulses.astype(np.int64)

        # Change servos
        # Joints
        self._node.set_joint_positions_pulse(target_joint_pulses, CONTROL_DURATION)
        gmin, gmax = GRIPPER_PULSE_RANGE

        # Gripper
        target_gripper_pulse = int(gmin + (1 - gripper_action) * (gmax - gmin))
        self._node._set_position_pulse([(10, target_gripper_pulse)], CONTROL_DURATION)

        sleep(CONTROL_DURATION + SLEEP_DURATION)

        if self._debug:
            pulses_debug = angle2pulse(joint_angles)
            p2a2p = (pulses_debug - joint_pulses).sum()

            res_debug = inverse_kinematics(joint_angles, transform_matrix)
            ik2fk = (res_debug["sol"].astype(np.float32) - joint_angles).sum()

            T_debug = xyz_rpy_to_T(pos, rpy)
            T2xyzrpy2T = (T_debug - transform_matrix).sum()

            print("==================")
            print(f"[Debug] current pulses: {joint_pulses}")
            print(f"[Debug] current xyz, rpy: {pos}, {rpy}")
            print(f"[Debug] action: {action}")
            print(f"[Debug] target xyz, rpy: {target_pos}, {target_rpy}")
            print(f"[Debug] ik_error = {ik_error}")
            print(f"[Debug] target joint pulses: {target_joint_pulses}")

            print(f"[Debug] pulse2angle - angle2pulse: {p2a2p}")
            print(f"[Debug] ik - fk: {ik2fk}")
            print(f"[Debug] T_to_xyz_rpy - xyz_rpy_to_T: {T2xyzrpy2T}")

        return True
    
    def reset(self):
        """Reset the robot to initial position."""
        self._node.execute_action_group('init')
        print("RESET")
        sleep(CONTROL_DURATION)
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
    print(f"obs: {obs}")
    for _ in range(5):
        # action = env.action_space.sample()
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        obs, reward, terminated, truncated, info = env.step(action)

    # v Implement Gym interface (get_observation, reset, control)
    # 2. Implement Teleoperation using keyboard and store dataset
    # 3. Replay buffer to load dataset and pre-process state/action/reward
    # 4. Modify IQL 
