#!/usr/bin/env python3
"""
ROS2 Image Saver (from queue): press ENTER to save the latest frame.
- Subscribes to a sensor_msgs/Image topic
- Maintains an internal queue of recent frames
- Shows a preview window and saves when ENTER is pressed
- Press Q or ESC to quit

Usage:
  python save_on_enter_ros.py \
    --topic /camera/image_raw \
    --out ./captures \
    --prefix img \
    --ext png \
    --queue-size 5 \
    --width 0 --height 0

Notes:
- Requires: rclpy, cv_bridge, sensor_msgs, OpenCV (cv2)
- If your camera publishes a different encoding, adjust desired_encoding.
"""

import argparse
import glob
import os
from pathlib import Path
import sys
import time
import queue

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageQueueSubscriber(Node):
    def __init__(self, 
                 topic: str = "/depth_cam/rgb/image_raw", 
                 queue_size: int = 5, 
                 desired_encoding: str = "bgr8"):
        
        super().__init__("image_queue_subscriber")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, queue_size)
        )

        self.bridge = CvBridge()
        self.desired_encoding = desired_encoding
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self.latest = None  # type: ignore

        self.sub = self.create_subscription(Image, topic, self.img_callback, qos)
        self.get_logger().info(f"Subscribed to: {topic}")

    def img_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.desired_encoding)
        except CvBridgeError as e:
            self.get_logger().warning(f"CvBridge error: {e}")
            return

        self.latest = img
        # put into queue; if full, drop the oldest to keep things moving
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(img)
        except queue.Full:
            pass

# def parse_args():
#     p = argparse.ArgumentParser(description="ROS2 image saver (press ENTER)")
#     p.add_argument("--topic", type=str, default="/depth_cam/rgb/image_raw", help="Image topic")
#     p.add_argument("--out", type=str, default="calibration/captures", help="Output directory")
#     p.add_argument("--prefix", type=str, default="img", help="Filename prefix")
#     p.add_argument("--ext", type=str, choices=["png", "jpg", "jpeg"], default="png", help="File extension")
#     p.add_argument("--queue-size", type=int, default=5, help="Max frames to buffer in queue")
#     p.add_argument("--width", type=int, default=0, help="Optional preview width (0 = original)")
#     p.add_argument("--height", type=int, default=0, help="Optional preview height (0 = original)")
#     p.add_argument("--show-fps", action="store_true", help="Overlay FPS text on preview")
#     p.add_argument("--save-npy", action="store_true", help="Also save a .npy copy alongside the image")
#     p.add_argument("--encoding", type=str, default="bgr8", help="desired encoding for CvBridge (e.g., bgr8)")
#     return p.parse_args()

# def next_index(out_dir: Path, prefix: str, ext: str) -> int:
#     pattern = str(out_dir / f"{prefix}_*.{ext}")
#     existing = glob.glob(pattern)
#     max_idx = 0
#     for path in existing:
#         name = os.path.basename(path)
#         stem, _ = os.path.splitext(name)
#         parts = stem.split("_")
#         if parts and parts[-1].isdigit():
#             max_idx = max(max_idx, int(parts[-1]))
#     return max_idx + 1

# def main():
#     args = parse_args()

#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     rclpy.init()
#     node = ImageQueueSubscriber(args.topic, args.queue_size, args.encoding)

#     window = "ROS Image (ENTER to save, Q/ESC to quit)"
#     cv2.namedWindow(window, cv2.WINDOW_NORMAL)

#     idx = next_index(out_dir, args.prefix, args.ext)

#     prev_time = time.time()
#     fps = 0.0
#     current = None

#     try:
#         while rclpy.ok():
#             # Process ROS callbacks quickly
#             rclpy.spin_once(node, timeout_sec=0.01)

#             # Pull the freshest available frame from the queue
#             try:
#                 while True:  # drain to the most recent
#                     current = node.queue.get(timeout=0.05)
#                     if node.queue.empty():
#                         break
#             except queue.Empty:
#                 pass

#             if current is None:
#                 # no frame yet
#                 cv2.waitKey(1)
#                 continue

#             frame = current

#             # Optional preview resize
#             if args.width > 0 and args.height > 0:
#                 frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

#             # FPS overlay
#             if args.show_fps:
#                 now = time.time()
#                 dt = max(1e-6, now - prev_time)
#                 fps = 0.9 * fps + 0.1 * (1.0 / dt)
#                 prev_time = now
#                 # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

#             # Next filename overlay
#             filename = f"{args.prefix}_{idx:04d}.{args.ext}"
#             # cv2.putText(frame, f"ENTER=save -> {filename}", (10, frame.shape[0]-20),
#             #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#             cv2.imshow(window, frame)
#             key = cv2.waitKey(1) & 0xFF

#             # ENTER to save (both 13 and 10 for cross-platform)
#             if key in (13, 10):
#                 save_path = out_dir / filename
#                 ok = cv2.imwrite(str(save_path), current)
#                 if ok:
#                     print(f"[OK] Saved: {save_path}")
#                     if args.save_npy:
#                         npy_path = out_dir / f"{args.prefix}_{idx:04d}.npy"
#                         np.save(str(npy_path), current)
#                         print(f"[OK] Saved NPY: {npy_path}")
#                     idx += 1
#                 else:
#                     print(f"[ERROR] Failed to save: {save_path}")

#             elif key in (27, ord('q'), ord('Q')):
#                 break

#     finally:
#         cv2.destroyAllWindows()
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
