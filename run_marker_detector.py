import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import time
from sensor_msgs.msg import Image

from manipulation.marker_detector import MarkerDetector

class ImageCallbackNode(Node):
    def __init__(self, name):
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        self.image = None

        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def get_image(self, timeout=1):
        return self.image

def main():
    rclpy.init()
    marker_detector = MarkerDetector()

    # Initialize ROS2 and create the node
    node = ImageCallbackNode("aruco")
    rclpy.spin_once(node)
    img = node.get_image(timeout=1)
    while True:
        rclpy.spin_once(node)
        img = node.get_image(timeout=1)
        if img is not None:
            detected_markers = marker_detector.detect_markers_with_pose(img)

            if detected_markers:
                for marker_id, data in detected_markers.items():
                    img = marker_detector.draw_marker(img, data)
            cv2.imshow("image", img)
            node.image = None
        else:
            print("No image received yet.")
        time.sleep(0.01)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("exit.")
            break
            


if __name__ == "__main__":
    main()