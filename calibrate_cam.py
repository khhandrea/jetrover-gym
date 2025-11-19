from calibration_utils.capture_img import ImageQueueSubscriber
import cv2
import numpy as np
import rclpy
import queue

SHOW_CORNERS = True # for debugging
OUTPUT_FILE = 'resources/camera_calibration.npz' # output file of calibration results

def main():
    rclpy.init()
    node = ImageQueueSubscriber()

    window = "ROS Image (ENTER to save, Q to skip, ESC to finish)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_list = []
    TARGET_NUM = 10

    # Collect images for calibration
    while rclpy.ok():
        # Process ROS callbacks quickly
        rclpy.spin_once(node, timeout_sec=0.01)

        # Pull the freshest available frame from the queue
        frame = None
        try:
            while True:  # drain to the most recent
                frame = node.queue.get(timeout=0.05)
                if node.queue.empty():
                    break
        except queue.Empty:
            pass

        if frame is None: # no frame yet
            cv2.waitKey(1)
            continue
        
        cv2.imshow(window, frame)
        print("Press ENTER to save this image for calibration, Q to skip, or ESC to finish.")
        key = cv2.waitKey(0) & 0xFF # get key input

        if key == ord('q'):
            print("Skip current image")
            continue

        if len(frame_list) == TARGET_NUM or key == 27: # ESC to finish
            print("Finished image collection.")
            break

        # ENTER to save (both 13 and 10 for cross-platform)
        if key == 13:
            frame_list.append(frame)
            print(f"Saved image {len(frame_list)}/{TARGET_NUM}")
            continue


    # Callibrate camera
    CHESSBOARD_SIZE = (8, 6)
    SQUARE_SIZE_MM = 24  


    # Prepare object points
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM  # 실제 크기(mm)를 곱해줍니다.

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Detect corners in each image
    for i, img in enumerate(frame_list):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray image for better corner detection
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None) # find corners

        if ret:
            print(f"Image {i+1}: Detected corners successfully.")
            objpoints.append(objp)

            # Define the criteria for corner refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # (optionally) Draw and display the corners
            if SHOW_CORNERS:
                cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
                h, w = img.shape[:2]
                scale = 800 / w if w > 800 else 1.0
                img_display = cv2.resize(img, (0,0), fx=scale, fy=scale)
                cv2.imshow('Chessboard Corners', img_display)
                cv2.waitKey(500)  # 0.5 seconds
        else:
            print(f"Image {i+1}: Failed to detect corners.")
            ## Visualize failure image
            cv2.imshow('Chessboard Corners - Failure', gray)
            cv2.waitKey(500)  # 0.5 seconds

    if SHOW_CORNERS:
        cv2.destroyAllWindows()

    if not objpoints:
        print("Error: No corners were detected in any image. Calibration failed.")
        exit()
    print(f"\nDetected corners in {len(objpoints)}/{len(frame_list)} images.")

    # using objpoints and imgpoints, compute the camera matrix, distortion coefficients, rotation and translation vectors.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        print("Error: Camera calibration failed.")
        exit()

    # Save and print the results

    print("\n--- Calibration Results ---")
    print("Camera Intrinsic Matrix:\n", mtx)
    print("\nDistortion Coefficients:\n", dist)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Mean Re-projection Error: {mean_error / len(objpoints)}, which is good if < 1.0")

    # Save to npz file
    np.savez(OUTPUT_FILE, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"\nCalibration results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()