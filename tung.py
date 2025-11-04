class MobileRobot():
    def __init__(self, max_episode_steps=20):
        # Declare used variables

        self.observation_space = ... # Mixed
        self.action_space = ... # Box 
        pass

    def get_observation():
        obs = dict()
        # obs['state'] = EE coordinate (xyz), EE linear velocity (v_x, v_y, v_z), rotation (wxvz/euler angle), gripper state (open/close)
        # obs['rgb'] = image

        # in ROS: get image, get state
        # Get directly infos from ROS 2

        return obs
    
    def control(self, action):
        # Do something with ROS to pass the action
        pass

    def reset(self):
        # Reset robot to initial position
        # - Define initial location within environment
        # - Define initial position of arm (reset to neutral position)

        obs = self.get_observation()
        pass

    def step(self, action):
        # 1. Define action space: in Sawyer, displacement: delta_x, delta_y, delta_z, delta_yaw, gipper (open/close)
        # self.control(action)
        
        
        obs = self.get_observation()

        reward = 0.0
        done = False
        infos = {}
        return obs, reward, done, infos



if __name__ == "__main__":
    env = MobileRobot()
    obs = env.reset()
    # print(obs)

    # action = random_action
    # obs, reward, done, infos = env.step(action)

    # Flow:
    # 1. Implement Gym interface
    # 2. Implement Teleoperation using keyboard and store dataset
    # 3. Replay buffer to load dataset and pre-process state/action/reward
    # 4. Modify IQL 