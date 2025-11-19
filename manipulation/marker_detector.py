import os
import cv2
import numpy as np

class MarkerDetectionResult:
    """
    dataclass for holding the detection result of a single ArUco marker.
    
    Attributes:
        corners (np.ndarray): The 4 corner coordinates of the marker (array of shape (4, 2)).
        rvec (np.ndarray): The rotation vector of the marker (array of size 3).
        tvec (np.ndarray): The translation vector of the marker (array of size 3).
    """
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray

    def __init__(self, rvec, tvec, corners=None):
        self.corners = corners
        self.rvec = rvec
        self.tvec = tvec

    def check_validity(self):
        if self.corners.shape != (4, 2):
            raise ValueError(f"Invalid corners shape: {self.corners.shape}, expected (4, 2)")
        if self.rvec.shape != (3,):
            raise ValueError(f"Invalid rvec shape: {self.rvec.shape}, expected (3,)")
        if self.tvec.shape != (3,):
            raise ValueError(f"Invalid tvec shape: {self.tvec.shape}, expected (3,)")

# aruco
class MarkerDetector():
    def __init__(self, calib_file_path='resources/camera_calibration.npz'):
        self.marker_size =  0.02
        
        ## Load camera calibration parameters
        if not os.path.exists(calib_file_path):
            raise FileNotFoundError(f"Calibration file {calib_file_path} not found. Please run camera calibration first.")
            exit(-1)
        else:
            with np.load(calib_file_path) as X:
                self.camera_intrinsic, self.dist_coeffs = [X[i] for i in ('mtx','dist')]
        
        print("Camera intrinsic:\n", self.camera_intrinsic)
        print("Distortion coeffs: \n", self.dist_coeffs.ravel())

    def draw_marker(self, image, result: MarkerDetectionResult):
        '''
        Draw the detected marker and its pose axes on the image.
        Args:
            image: cv2 image.
            result: MarkerDetectionResult object containing corners, rvec, tvec.
        Returns:
            image with drawn marker and axes.
        '''
        result.check_validity()
        rvec, tvec, corner = result.rvec, result.tvec, result.corners
        
        # Draw the marker's bounding box
        (topLeft, topRight, bottomRight, bottomLeft) = corner
        topRightPoint = (int(topRight[0]), int(topRight[1]))
        topLeftPoint = (int(topLeft[0]), int(topLeft[1]))
        bottomRightPoint = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeftPoint = (int(bottomLeft[0]), int(bottomLeft[1]))
        
        cv2.circle(image, topLeftPoint, 4, (255,0,0), -1)
        cv2.circle(image, topRightPoint, 4, (255,0,0), -1)
        cv2.circle(image, bottomRightPoint, 4, (255,0,0), -1)
        cv2.circle(image, bottomLeftPoint, 4, (255,0,0), -1)

        cv2.drawFrameAxes(
            image, self.camera_intrinsic, self.dist_coeffs,
            rvec, tvec, self.marker_size/2)
        return image
    
    def detect_markers_with_pose(self, rgb_image):
        """
        Detect ArUco markers in the given RGB image and estimate their poses.
        Args:
            rgb_image: cv2 image.
        Returns:
            dict[int, MarkerDetectionResult]: A dictionary mapping marker IDs to their detection results.
        """
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # ArUco Dectection
        corners, ids, _ = detector.detectMarkers(gray)

        detected_data: dict[int, MarkerDetectionResult] = {}
        # If Markers are detected
        if ids is not None:
            # get rotation and translation vectors for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_intrinsic, self.dist_coeffs)

            # Store information for all detected markers in the data
            for i, marker_id in enumerate(ids.flatten()):
                detected_data[marker_id] = MarkerDetectionResult(
                    corners=corners[i][0],
                    rvec=rvecs[i][0],
                    tvec=tvecs[i][0]
                )
                
        return detected_data