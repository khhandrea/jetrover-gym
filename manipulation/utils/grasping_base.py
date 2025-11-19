import rclpy
from rclpy.node import Node

from ros_robot_controller_msgs.srv import GetBusServoState
from ros_robot_controller_msgs.msg import ServoPosition, ServosPosition
from ros_robot_controller_msgs.msg import GetBusServoCmd

from servo_controller_msgs.msg import ServosPosition as ServosPositionMsg
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import numpy as np

from ..marker_detector import MarkerDetector
from .kinematics_utils import pulse2angle, angle2pulse
from .action_group_controller import ActionGroupController


class GraspingNodeBase(Node):
    def __init__(self, name):
        # Initialize ROS2 node
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        # Servo control
        self.servo_client = self.create_client(
            GetBusServoState, '/ros_robot_controller/bus_servo/get_state')
        self.servo_client.wait_for_service()
        self.servo_position_pub = self.create_publisher(
            ServosPosition, 'ros_robot_controller/bus_servo/set_position', 1)

        # Joint control
        self.joints_pub = self.create_publisher(ServosPositionMsg, 'servo_controller', 1)
        self.controller = ActionGroupController(
            self.joints_pub, '/home/ubuntu/software/arm_pc/ActionGroups')

        # For Marker detection 
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()
        self.image = None
        
        self.marker_detector = MarkerDetector()
        self.marker_ids = {"red" : 1, "blue" : 6, "green" : 7}
    
    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def get_joint_positions_pulse(self):
        try:
            req = GetBusServoState.Request()
            for i in range(1, 5+1):
                cmd = GetBusServoCmd()
                cmd.id = i
                cmd.get_id = 1
                cmd.get_position = 1
                req.cmd.append(cmd)
            
            self.future = self.servo_client.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            response = self.future.result()

            values = [-1, -1, -1, -1, -1]
            for state in response.state:
                values[state.present_id[0] - 1] = state.position[0]
                # values.append(state.position[0])
            q = np.array(values)
            self.last_q = pulse2angle(q)
        except:
            q = angle2pulse(self.last_q)
        return q

    def set_joint_positions_pulse(self, pulse, duration):
        pulse = [(1+i, p) for i, p in enumerate(pulse)]
        duration = 0.02 if duration < 0.02 else 30 if duration > 30 else duration

        self._set_position_pulse(pulse, duration)
        
    def _set_position_pulse(self, pulse_formatted, duration):
        msg = ServosPosition()
        msg.duration = float(duration)

        for i in pulse_formatted:
            position = int(i[1])
            position = 0 if position < 0 else 1000 if position > 1000 else position
            servo_msg = ServoPosition()
            servo_msg.id = i[0]
            servo_msg.position = position
            msg.position.append(servo_msg)
        self.servo_position_pub.publish(msg)
