from time import sleep

from gymnasium import Env, spaces
import numpy as np

from jetrover_gym.src.manipulation_node import ManipulationNode
from manipulation.utils.kinematics_utils import pulse2angle, angle2pulse
from manipulation.kinematics import forward_kinematics, inverse_kinematics
from jetrover_gym.src.utils import T_to_xyz_rpy, xyz_rpy_to_T

ACTION_GROUPSAVE_DIR = '/home/ubuntu/software/arm_pc/ActionGroups'
INITIAL_GRIPPER_PULSE = 500
GRIPPER_PULSE_RANGE = (120, 550)
EE_POS_LOWER = np.array([0.15, -0.001, 0.02], dtype=np.float32)
EE_POS_UPPER = np.array([0.24, 0.001, 0.29], dtype=np.float32)
CONTROL_DURATION = 2.0 
SLEEP_DURATION = 1.0

class JetRoverManipulationEnv(Env):
    """Hiwonder JetRover manipulation environment backed by ROS 2 and IK."""

    def __init__(self,
                 name='jetrover_manipulation',
                 max_episode_steps=100,
                 use_rgb=False,
                 debug=False,
                 ik_error_threshold=0.01,
                 delta_pulse_max=50,
                 pos_scale=0.04,
                 rpy_scale=0.1
                 ):
        """Initialize the environment.

        Args:
            name: name of the node to be used.
            max_episode_steps: max step after which the environment will be terminated.
            use_rgb: whether to include RGB image in observation.
            debug: if true, verbose output will be printed.
            ik_error_threshold: threshold to determine if IK solver succeeded.
            delta_pulse_max: max delta pulse to be applied in each step.
            pos_scale: scaling factor for delta position.
            rpy_scale: scaling factor for delta rotation.
        """
        self._max_episode_steps = max_episode_steps
        self._use_rgb = use_rgb
        self._img_rgb_height = 640
        self._img_rgb_width = 480
        self._debug = debug
        self._ik_error_threshold = ik_error_threshold
        self._delta_pulse_max = delta_pulse_max
        self._pos_scale = pos_scale
        self._rpy_scale = rpy_scale

        self._node = ManipulationNode(name)
        self._gripper_pulse = INITIAL_GRIPPER_PULSE

        self.observation_space = spaces.Dict()
        # EE pos (xyz), rot (rpy), and gripper_state (open/close)
        self.observation_space['state'] = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=float)
        self.global_step = 0

        if self._use_rgb:
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
        if self._use_rgb:
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
        delta_pos = action[:3] * self._pos_scale
        delta_rpy = action[3:6] * self._rpy_scale
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

        if np.any(target_joint_pulses < 0):
            print("Negative pulses!")
            print(f"target pulses: {target_joint_pulses}")
            return False

        delta_joint_pulses = np.abs(target_joint_pulses - joint_pulses)
        if np.any(delta_joint_pulses >= self._delta_pulse_max):
            print("Too different pulses!")
            print(f"current pulses: {joint_pulses}")
            print(f"target pulses: {target_joint_pulses}")
            return False

        # Change servos
        # Joints
        self._node.set_joint_positions_pulse(target_joint_pulses, CONTROL_DURATION)
        gmin, gmax = GRIPPER_PULSE_RANGE

        # Gripper
        target_gripper_pulse = int(gmin + (1 - gripper_action) * (gmax - gmin))
        target_gripper_pulse = gmax if target_gripper_pulse > (gmax + gmin) / 2 else gmin
        self._node._set_position_pulse([(10, target_gripper_pulse)], CONTROL_DURATION)

        sleep(CONTROL_DURATION + SLEEP_DURATION)

        if self._debug:
            pulses_debug = angle2pulse(joint_angles)
            p2a2p = (pulses_debug - joint_pulses)

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
            print(f"[Debug] target pulses: {target_joint_pulses}")

        return True
    
    def reset(self):
        """Reset the robot to initial position."""
        self._node.execute_action_group('init')
        sleep(CONTROL_DURATION)
        self.global_step = 0
        obs = self._get_observation()
        info = dict()
        return obs, info

    def step(self, action: np.ndarray):
        """Advance the environment by one step.

        Args:
            action: EE delta_position (xyz), delta_rotation (rpy), and gripper_state (open/close)
        """
        result = self._control(action)
        self.global_step += 1

        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            'action_result': result
        }
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = JetRoverManipulationEnv(debug=True)

    obs, info = env.reset()
    print(f"obs: {obs}")
    for _ in range(5):
        # action = env.action_space.sample()
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
        obs, reward, terminated, truncated, info = env.step(action)

    # v Implement Gym interface (get_observation, reset, control)
    # v Implement Teleoperation using keyboard and store dataset
    # 3. Replay buffer to load dataset and pre-process state/action/reward
    # 4. Modify IQL 
