import time
import cv2
import rclpy
import numpy as np

from .kinematics import *
from .utils.kinematics_utils import *
from .utils.grasping_base import GraspingNodeBase
from .marker_detector import MarkerDetectionResult

### UNUSED (Only for solution) 
from enum import Enum, auto
class RobotState(Enum):
    INITIALIZING = auto()         # 초기화 및 홈 위치로 이동
    IDLE = auto()                 # 1차 마커 탐색
    MOVING_TO_APPROACH = auto()   # 1차 목표(접근 지점)로 이동 중
    SCANNING = auto()             # 접근 지점에서 2차 정밀 스캔 중
    MOVING_TO_GRASP = auto()      # 2차 목표(최종 파지 지점)로 이동 중
    GRASPING = auto()             # 그리퍼 닫는 중
    RETURNING_HOME = auto()       # 홈으로 복귀 중
    POST_GRASP = auto()
    FAILED = auto()               # 실패 상태
    FAILED_SCAN = auto()          # 스캔 실패 상태

###

class GraspingNode(GraspingNodeBase):
    def __init__(self, name):
        super().__init__(name)
        self.running = True
        
        self.last_q = None
        self.grasping = False
        self.grasp_targets = []

        #####
        ## TODO : Define custom class variables, if needed.
        self.home_pose = [125, 791, 36, 123, 500]

        self.approach_z_offset = 0.10
        self.final_grasp_z_offset = -0.02
        self.scan_timeout = 5.0
        self.move_time_approach = 2.0
        self.move_time_grasp = 3.0
        self.move_time_home = 2.0
        self.grasp_duration = 1.5
        self.cam_stabilize_time = 1.0
        self.idle_wait_time = 4.0

        #####


    def gripper_close(self, duration=1.5):
        '''
        Close the gripper.
        '''
        self._set_position_pulse([(10, 550)], duration)

    def gripper_open(self, duration=1.5):
        '''
        Open the gripper.
        '''
        self._set_position_pulse([(10, 100)], duration)

    def get_joint_positions(self):
        '''
        Returns: current joint positions in "radians" as a numpy array
        '''
        q = self.get_joint_positions_pulse() # Base function method
        return pulse2angle(q)
    
    def set_joint_positions(self, q, duration):
        '''
        q: "radians", list or numpy array of joint angles
        duration: time to move in seconds
        '''
        pulse = angle2pulse(q)
        self.set_joint_positions_pulse(pulse, duration) # Base function method

    def get_detected_markers(self):
        rclpy.spin_once(self, timeout_sec=0.01)
        if self.image is not None:
            detected_markers = self.marker_detector.detect_markers_with_pose(self.image)
            self.image = None
        else:
            detected_markers = {}
        return detected_markers

    #########################################################################################################################
    ### Functions for solution file

    @staticmethod
    def joint_space_distance(q_cur, q_tgt, weights=None, wrap=True):
        """Return weighted L2 distance in joint space.
        q_cur, q_tgt: iterable of joint angles (rad)
        weights: optional per-joint weights; same length as q
        wrap: if True, wrap angle diffs to [-pi, pi]
        """
        q_cur = np.asarray(q_cur, dtype=float)
        q_tgt = np.asarray(q_tgt, dtype=float)
        diff = q_tgt - q_cur
        if wrap:
            # wrap angle error into [-pi, pi]
            diff = (diff + np.pi) % (2*np.pi) - np.pi
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            diff = w * diff
        return float(np.linalg.norm(diff, ord=2))

    @staticmethod
    def shrink_xy(tvec, shrink=0.03):
        x, y, z = tvec.flatten()  # tvec이 (3,1) 형태일 수 있음
        r = np.sqrt(x**2 + y**2)
        if r <= shrink:
            # 너무 가까우면 그냥 원점으로 수축하지 않고 그대로 둠
            return np.array([x, y, z])
        r_new = r - shrink
        theta = np.arctan2(y, x)
        x_new = r_new * np.cos(theta)
        y_new = r_new * np.sin(theta)
        return np.array([x_new, y_new, z])

    def get_top_down_grasp_pose_simple(self, T_block, z_offset=0.005, local_xy_offset=None):
        """
        블록의 Pose를 기준으로 '상대적인 Top-down' Grasp Pose 후보 4개를 생성합니다.
        그리퍼의 접근 방향(Z축)은 블록의 Z축과 정반대이며,
        이 접근 축을 기준으로 90도씩 회전한 4개의 Pose를 반환합니다.

        Args:
            T_block (np.ndarray): 마커로부터 계산된 4x4 변환 행렬 (블록의 Pose).
            z_offset (float): 블록 표면으로부터 그리퍼를 띄울 거리 (블록의 Z축 방향).
            local_xy_offset (list or np.ndarray): 블록의 로컬 좌표계 기준 XY 오프셋 [x, y].
                                                마커가 블록 중심이 아닐 때 보정용으로 사용합니다.

        Returns:
            list[np.ndarray]: 계산된 4개의 4x4 Grasp Pose 행렬 리스트.
        """
        # --- 1. 기본 Grasp 위치(Position) 계산 ---
        # 먼저 블록의 원점 위치를 가져옵니다.
        base_position = T_block[:3, 3].copy()
        
        # 마커가 블록 중심이 아닌 경우, 블록 로컬 좌표계 기준 오프셋을 적용
        if local_xy_offset is not None:
            offset_in_block_frame = np.array([local_xy_offset[0], local_xy_offset[1], 0.0])
            # 블록의 회전을 적용하여 월드 좌표계 오프셋으로 변환
            offset_in_world_frame = T_block[:3, :3] @ offset_in_block_frame
            base_position += offset_in_world_frame

        # 블록의 Z축 방향으로 z_offset만큼 떨어진 위치를 최종 파지 접근 위치로 설정
        # block_z_axis = T_block[:3, 2]
        # grasp_position = base_position + block_z_axis * z_offset
        grasp_position = base_position + np.array([0, 0, z_offset])

        # --- 2. 기본 Grasp 자세(Orientation) 계산 ---
        # 그리퍼의 Z축 (접근 방향)은 블록의 Z축과 정반대 방향입니다.
        z_axis_grasp = -T_block[:3, 2]
        z_axis_grasp /= np.linalg.norm(z_axis_grasp) # 정규화

        z_axis_grasp = np.array([0, 0, -1])  

        # 그리퍼의 X축은 블록의 X축을 기준으로 설정합니다.
        # 단, z_axis_grasp와 수직이어야 하므로 Gram-Schmidt 방법을 이용해 직교 벡터를 구합니다.
        x_axis_block = T_block[:3, 0]
        x_axis_grasp = x_axis_block - np.dot(x_axis_block, z_axis_grasp) * z_axis_grasp
        x_axis_grasp /= np.linalg.norm(x_axis_grasp) # 정규화
        
        # 그리퍼의 Y축은 오른손 법칙에 따라 Z축과 X축의 외적으로 계산합니다.
        y_axis_grasp = np.cross(z_axis_grasp, x_axis_grasp)

        # 기본 Grasp 회전 행렬을 조립합니다.
        R_grasp_base = np.stack([x_axis_grasp, y_axis_grasp, z_axis_grasp], axis=1)

        # --- 3. 90도씩 회전시킨 4개의 Pose 생성 ---
        grasp_poses = []
        for angle_deg in [0, 90, 180, 270]:
            angle_rad = np.deg2rad(angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # 그리퍼의 로컬 Z축을 기준으로 회전시키는 변환 행렬
            R_z_local = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])
            
            # 기본 자세에 로컬 회전을 적용하여 새로운 자세를 계산
            R_new = R_grasp_base @ R_z_local
            
            # 최종 4x4 Pose 행렬 생성
            T_new = np.eye(4)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = grasp_position
            grasp_poses.append(T_new)
            
        return grasp_poses

    def find_best_ik_solution(self, q_current, target_poses, ik_pos_error_threshold=0.02, joint_weights=None, target='tcp'):
        """
        주어진 목표 Pose 후보 리스트에 대해 IK를 계산하고,
        현재 관절 각도에서 가장 이동 거리가 짧은 유효한 해를 찾습니다.
        """
        if joint_weights is None:
            joint_weights = np.array([1, 1, 1, 1, 1], dtype=float)
            
        candidates = []
        for i, T_target in enumerate(target_poses):
            result = inverse_kinematics(q_current, T_target)

            if result is None or result["pos_error"] > ik_pos_error_threshold:
                continue

            q_sol = np.asarray(result["sol"], dtype=float)
            print(f"q_sol {i}:", q_sol)
            jdist = self.joint_space_distance(q_current, q_sol, weights=joint_weights, wrap=True)
            # pulses = [(j + 1, int(v)) for j, v in enumerate(angle2pulse(q_sol))]
            pulses = angle2pulse(q_sol)
            
            candidates.append({
                "idx": i,
                "joint_dist": jdist,
                "pulses": pulses,
                "T_target": T_target,
                "ik_result": result,
            })
            print(f"  - 유효한 IK 후보 [{i}] 발견 (관절 거리: {jdist:.4f})")
            print(f"Target Pos : {T_target[:3, 3]}")

        if not candidates:
            return None
        else:
            best = min(candidates, key=lambda c: c["joint_dist"])
            print(f"  -> 최적 IK 후보 선택: [{best['idx']}] (관절 거리: {best['joint_dist']:.4f})")
            return best

    def grav_comp_heuristic(self, pulses):
        pulses[1] += 10 if pulses[1] < 500 else -10
        pulses[2] += 10 if pulses[2] < 500 else -10
        pulses[3] += 5 if pulses[3] < 500 else -5

        return pulses

    def _grasp_object_by_id(self, target_marker_id: int, view_angle: int = 0) -> bool:
        """
        특정 마커 ID를 가진 객체를 집는 전체 과정을 수행합니다.
        성공 시 True, 실패 시 False를 반환합니다.
        """
        robot_state = RobotState.INITIALIZING
        action_start_time = 0
        scan_time = 1.0
        scan_step = 0
        max_scan_steps = 15

        target_marker_status = None

        home_pose = self.home_pose
        # home_pose[0] = (1, angle2pulse_single(radians(view_angle), 0) - 500 + self.home_pose[0][1])

        print(f"Current home pose: {home_pose}")
    

        print(f"\n>>> GRASPING PROCESS STARTED FOR MARKER ID: {target_marker_id} <<<")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            # q_current = self.get_joint_positions()

            if robot_state == RobotState.INITIALIZING:
                print("[상태: INITIALIZING] 로봇을 홈 위치로 이동하고 그리퍼를 엽니다.")
                self.set_joint_positions_pulse(home_pose, self.move_time_home)
                time.sleep(self.move_time_home)
                self.gripper_open()
                time.sleep(1.0)
                print(f"카메라 안정화를 위해 {self.cam_stabilize_time}초 대기...")
                time.sleep(self.cam_stabilize_time)
                print("초기화 완료. 1차 마커 탐색을 시작합니다.")
                action_start_time = time.time()
                robot_state = RobotState.IDLE

            elif robot_state == RobotState.IDLE:
                # Detect Marker
                # detected_markers = self.marker_detector.detect_markers_with_pose(self.image)
                # self.image = None

                detected_markers = self.get_detected_markers()

                if target_marker_id in detected_markers:
                    print(f"\n[상태: IDLE] 1차 마커(ID: {target_marker_id}) 검출! 중간 접근 지점을 계산합니다.")
                    marker_res = detected_markers[target_marker_id]
                    # marker_res = MarkerDetectionResult(
                    #     rvec=marker_res['rvec'],
                    #     tvec=marker_res['tvec']
                    # ) 

                    T_block_rough = self.get_block_pose(marker_res)

                    tvec = T_block_rough[:3, 3]  # numpy array (3,) or (3,1)
                    tvec_shrunk = self.shrink_xy(tvec, shrink=0.05)
                    T_block_rough[:3, 3] = tvec_shrunk

                    T_approach_list = self.get_top_down_grasp_pose_simple(T_block_rough, z_offset=self.approach_z_offset)
                    q_current = self.get_joint_positions()
                    best_ik = self.find_best_ik_solution(q_current, T_approach_list)


                    if best_ik:
                        print(f"중간 접근 위치로 이동합니다...")

                        ## TODO: Gravity compensation
                        best_ik['pulses'] = self.grav_comp_heuristic(best_ik['pulses'])

                        self.set_joint_positions_pulse(best_ik["pulses"], self.move_time_approach)
                        action_start_time = time.time()
                        robot_state = RobotState.MOVING_TO_APPROACH
                    else:
                        print("오류: 중간 접근 지점에 대한 IK 해를 찾지 못했습니다.")
                        robot_state = RobotState.FAILED
                else:
                    print(f"[상태: IDLE] 1차 마커(ID: {target_marker_id})를 찾는 중...")
                    print(f"  - 현재 검출된 마커들: {list(detected_markers.keys())}")
                    if time.time() - action_start_time > self.idle_wait_time:
                        print(f"{self.idle_wait_time}초 동안 마커를 찾지 못했습니다. 파지 작업을 중단합니다.")
                        robot_state = RobotState.FAILED

                time.sleep(0.1)

            elif robot_state == RobotState.MOVING_TO_APPROACH:
                if time.time() - action_start_time >= self.move_time_approach:
                    print(f"\n[상태: MOVING_TO_APPROACH] 중간 접근 지점 도착. 정밀 스캔을 시작합니다.")
                    action_start_time = time.time()
                    robot_state = RobotState.SCANNING

            elif robot_state == RobotState.SCANNING:
                if (time.time() - action_start_time) < self.cam_stabilize_time:
                    print(f"[상태: SCANNING] 카메라 안정화 중... {self.cam_stabilize_time - (time.time() - action_start_time):.1f}초 남음", end='\r')
                    time.sleep(0.1)
                    target_marker_status = None
                    continue

                elif scan_step < max_scan_steps and time.time() - action_start_time < self.scan_timeout:
                    print(f"[상태: SCANNING] 마커 프레임 수집 중... {scan_step} / {max_scan_steps}", end='\r')

                    # detected_markers = self.marker_detector.detect_markers_with_pose(self.image)
                    # self.image = None

                    detected_markers = self.get_detected_markers()

                    if target_marker_id in detected_markers:
                        meas = detected_markers[target_marker_id]  # {'rvec': ..., 'tvec': ...}

                        # 첫 샘플: 그대로 초기화(나누지 말 것!)
                        if target_marker_status is None:
                            target_marker_status = {}
                            
                            # for k, v in meas.items():
                            #     target_marker_status[k] = np.array(v, dtype=float)
                            target_marker_status['rvec'] = meas.rvec
                            target_marker_status['tvec'] = meas.tvec


                            scan_step = 1  # 첫 샘플 반영
                        else:
                            beta = 1.0 / (scan_step + 1)  # 누적 평균; EMA 원하면 beta = eta(예:0.2)
                            # for k, v in meas.items():
                            #     v = np.array(v, dtype=float)
                            #     if k not in target_marker_status:
                            #         target_marker_status[k] = v.copy()
                            #     else:
                            

                            target_marker_status['rvec'] = target_marker_status['rvec'] + beta * (meas.rvec - target_marker_status['rvec'])
                            target_marker_status['tvec'] = target_marker_status['tvec'] + beta * (meas.tvec - target_marker_status['tvec'])
                                    # target_marker_status[k] = target_marker_status[k] + beta * (v - target_marker_status[k])
                            scan_step += 1

                            print(f"updated rvec: {target_marker_status['rvec']}, tvec: {target_marker_status['tvec']}")
                    else: 
                        print(f"Marker ID {target_marker_id} not detected in this frame.")

                elif target_marker_id in detected_markers and scan_step >= max_scan_steps:
                    print(f"\n[상태: SCANNING] 2차 정밀 마커(ID: {target_marker_id}) 검출! 최종 파지 위치를 계산합니다.")
                    print(f"Final averaged rvec: {target_marker_status['rvec']}, tvec: {target_marker_status['tvec']}")
                    marker_res = target_marker_status  
                    marker_res = MarkerDetectionResult(
                        corners=None,
                        rvec=marker_res['rvec'],
                        tvec=marker_res['tvec']
                    ) 

                    T_block_final = self.get_block_pose(marker_res)

                    T_grasp_list = self.get_top_down_grasp_pose_simple(T_block_final, z_offset=self.final_grasp_z_offset)
                    q_current = self.get_joint_positions()
                    best_ik = self.find_best_ik_solution(q_current, T_grasp_list)

                    if best_ik:
                        print(f"최종 파지 위치로 이동합니다...")
                        
                        ## TODO: Gravity compensation
                        best_ik['pulses'] = self.grav_comp_heuristic(best_ik['pulses'])
                        
                        self.set_joint_positions_pulse(best_ik["pulses"], self.move_time_grasp)
                        action_start_time = time.time()
                        robot_state = RobotState.MOVING_TO_GRASP
                    else:
                        print("오류: 최종 파지 위치에 대한 IK 해를 찾지 못했습니다.")
                        robot_state = RobotState.FAILED
                
                elif time.time() - action_start_time > self.scan_timeout:
                    print("\n[상태: SCANNING] 스캔 시간 초과. 마커를 찾지 못했습니다.")
                    robot_state = RobotState.FAILED_SCAN

            elif robot_state == RobotState.MOVING_TO_GRASP:
                if time.time() - action_start_time >= self.move_time_grasp:
                    print("\n[상태: MOVING_TO_GRASP] 파지 위치 도착. 그리퍼를 닫습니다.")
                    self.gripper_close(self.grasp_duration)
                    action_start_time = time.time()
                    robot_state = RobotState.GRASPING

            elif robot_state == RobotState.GRASPING:
                if time.time() - action_start_time >= self.grasp_duration:
                    print("\n[상태: GRASPING] 파지 완료. 홈으로 복귀합니다.")
                    action_start_time = time.time()
                    robot_state = RobotState.POST_GRASP

            elif robot_state == RobotState.POST_GRASP:
                if time.time() - action_start_time >= self.move_time_grasp:
                    print("\n[상태: POST_GRASP] 그리퍼를 들어올립니다.")
                    q_current = self.get_joint_positions()

                    T_curr = forward_kinematics(q_current, 'tcp')
                    T_curr = np.array(T_curr, dtype=float)
                    T_lifted = T_curr.copy()
                    T_lifted[2, 3] += 0.10  # Z축으로 10cm 들어올림
                    result = inverse_kinematics(q_current, T_lifted)
                    
                    if result and result["pos_error"] <= 0.02:
                        q_lifted = angle2pulse(result["sol"])
                        self.set_joint_positions_pulse(q_lifted, self.move_time_grasp)
                        action_start_time = time.time()
                        robot_state = RobotState.RETURNING_HOME
                    else:
                        print("오류: 들어올리기 위치에 대한 IK 해를 찾지 못했습니다.")
                        robot_state = RobotState.RETURNING_HOME

                    
            elif robot_state == RobotState.RETURNING_HOME:
                self.set_joint_positions_pulse(home_pose, self.move_time_home)
                time.sleep(self.move_time_home)
                
                print("\n>>> GRASPING PROCESS SUCCEEDED <<<")
                return 1 # 성공적으로 종료
            
            elif robot_state == RobotState.FAILED:
                print("파지 작업 실패. 홈으로 복귀합니다.")
                self.set_joint_positions_pulse(home_pose, self.move_time_home)
                time.sleep(self.move_time_home)
                print("\n>>> GRASPING PROCESS FAILED <<<")
                return 0 # 실패로 종료

            elif robot_state == RobotState.FAILED_SCAN:
                print("물체를 찾지 못했습니다. 홈으로 복귀합니다.")
                self.set_joint_positions_pulse(home_pose, self.move_time_home)
                time.sleep(self.move_time_home)
                print("\n>>> GRASPING PROCESS FAILED <<<")
                return -1 # 실패로 종료

            time.sleep(0.1)
    #########################################################################################################################

    ## TO-BE-IMPLEMENTED
    def get_block_pose(self, 
        detection_result : MarkerDetectionResult, 
    ):
        """
            Calculate the block's pose in the world coordinate frame 
            given the marker's rotation and translation vectors.
            Returns a 4x4 transformation matrix representing the block's pose.
        """
        ######
        ## TODO : Implement this function to compute the block's pose, given the marker's rvec and tvec.
        rvec, tvec = detection_result.rvec, detection_result.tvec

        R, _ = cv2.Rodrigues(rvec)
        T_marker = np.eye(4)
        T_marker[:3, :3] = R
        T_marker[:3, 3] = tvec.flatten()

        T_block = T_marker.copy()
        q = self.get_joint_positions()
        base_to_cam = forward_kinematics(q, 'cam')
        base_to_cam = np.array(base_to_cam, dtype=float)
        T_block = base_to_cam @ T_block

        ######

        return T_block

    def grasp(self, target_marker_id: int | str) -> bool:
        """
            Execute grasping action for the object with the specified marker ID.
            If target_marker_id is a string, it should map to a predefined marker ID.
            Returns True if successful, False otherwise.
        """

        ######
        ## TODO : Implement this function to perform the complete grasping sequence.

        if isinstance(target_marker_id, str):
            target_marker_id = self.marker_ids[target_marker_id]
        view_angles = [0, 90, 180]
        for angle in view_angles:
            result = self._grasp_object_by_id(target_marker_id, view_angle=angle)
            if result == 1:
                return True
            elif result == -1:
                ## Grasp failed
                return False
            else:
                ## Scan failed, try next angle
                continue

    def place(self, action_name) -> bool:
        """
            Execute placing action given an action name.
            You may utilize predefined action groups if available.
        """

        ######
        ## TODO : Implement this function to perform the placing action.
        ## You may utilize predefined action groups if available.

        try:
            self.controller.run_action(action_name)
            return True
        except Exception as e:
            print(f"Error executing action '{action_name}': {e}")
            return False