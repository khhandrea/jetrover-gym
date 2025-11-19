import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import time

from manipulation.grasping import GraspingNode

def main():
    node = GraspingNode("grasping")

    ## B: 6
    ## G: 7
    ## R: 1

    grasp_targets = [6, 7, 1]  # Marker ID to be grasped
    action_names = ["blue_3", "green_3", "red_3"]

    try:
        # call the method to grasp the object with the specified ID
        for i, target_marker_id in enumerate(grasp_targets):
            print(f"Marker ID {target_marker_id} grasping start")
            success = node.grasp(target_marker_id)
            if success:
                place_success = node.place(action_names[i])
                print(f"Place task {place_success}")

            else:
                print(f"Marker ID {target_marker_id} grasping failed.")
            print("Waiting before trying to grasp the next object...")
            time.sleep(3.0)

    finally:
        print("Shutting down the program.")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()