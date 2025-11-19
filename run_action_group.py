import time
import rclpy
from rclpy.node import Node
from servo_controller_msgs.msg import ServosPosition, ServoPosition
from manipulation.utils.action_group_controller import ActionGroupController

ACTION_GROUP_SAVE_DIR = '/home/ubuntu/software/arm_pc/ActionGroups'

# Specify the action group name to execute
ACTION_GROUP_NAME = 'init'   

class ActionGroupExecution(Node):
    def __init__(self, name):
        super().__init__(name)
        self.joints_pub = self.create_publisher(ServosPosition, 'servo_controller', 1)
        self.controller = ActionGroupController(
            self.joints_pub, ACTION_GROUP_SAVE_DIR)

    def execute_action_group(self, name):
        self.controller.run_action(name)
        
def main():
    rclpy.init()
    node = ActionGroupExecution("action_group_execution")
	
    time.sleep(2.)
    node.execute_action_group(ACTION_GROUP_NAME)

    
if __name__ == '__main__':
    main()
