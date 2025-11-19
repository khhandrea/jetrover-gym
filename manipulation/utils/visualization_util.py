from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import yourdfpy
import viser
from viser.extras import ViserUrdf
from scipy.spatial.transform import Rotation
from functools import partial
import time
# from .kinematics_utils import *

URDF_PATH = Path("resources/jetrover_description/urdf/jetrover.urdf")


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config

def visualize_frame(
    name: str,
    server: viser.ViserServer,
    T: np.ndarray,
    axis_length: float = 0.1,
    axis_radius: float = 0.01,
) -> None:
    """Visualize a coordinate frame with three arrows.

    Args:
        name: Name of the frame to visualize.
        server: Viser server.
        T: 4x4 transformation matrix of the frame.
    """
    position = T[:3, 3]
    rotation = T[:3, :3].copy()
    quat = Rotation.from_matrix(rotation).as_quat()
    wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
    server.scene.add_frame(
        name, 
        position=position,
        wxyz=wxyz,
        axes_length=axis_length,
        axes_radius=axis_radius
    )

def draw_robot_frames(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
    q: np.ndarray,
    fk_func: Callable = None
):
    """
    Draw robot frames using custom forward kinematics function.

    Args:
        server: Viser server.
        viser_urdf: ViserUrdf instance.
        q: joint configuration.
    """

    frames_to_visualize = ['j1', 'j2', 'j3', 'j4', 'j5', 'cam', 'tcp']
    frames = []

    for link_name in frames_to_visualize:
        try:
            T_world_link = fk_func(q, target=link_name)
            T_world_link = np.array(T_world_link)
            frames.append(T_world_link)
            visualize_frame(
                name=f"/custom_fk/{link_name}",
                server=server,
                T=T_world_link
            )
        except Exception as e:
            print(f"Could not compute FK for link '{link_name}': {e}")
            pass
    return frames

def draw_robot_frames_yourdf(
    server: viser.ViserServer,
    robot_urdf: yourdfpy.URDF,
    q: np.ndarray,
):
    """
    Draw robot frames using the 'yourdf' library for ground-truth forward kinematics.

    Args:
        server: Viser server instance.
        robot_urdf: The loaded 'yourdf' URDF object.
        q: The joint configuration array.
    """
    joint_values = dict(zip(robot_urdf.actuated_joint_names, q))
    robot_urdf.update_cfg(q)
    frames_to_visualize = ['link1', 'link2', 'link3', 'link4', 'link5', 'depth_cam_frame', 'end_effector_link']
    frames = []
    for link_name in frames_to_visualize:
        try:
            T_world_link = robot_urdf.get_transform(
                frame_to=link_name,
            )

            frames.append(T_world_link)

            visualize_frame(
                name=f"/ground_truth/{link_name}", # 이름 충돌을 피하기 위해 prefix 추가
                server=server,
                T=T_world_link
            )
        except Exception as e:
            print(f"Could not compute FK for link '{link_name}' with yourdfpy: {e}")
            pass
    return frames

def run_urdf_viewer():
    server = viser.ViserServer()
    
    viser_urdf = ViserUrdf(
        server, urdf_or_path=URDF_PATH,
    )

    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    while True:
        time.sleep(0.01) 

def calculate_fk_accuracy(T_predicted_list, T_ground_truth_list):
    """
    예측된 FK 결과와 실제(Ground-truth) 결과를 비교하여 정확도 메트릭을 계산합니다.

    Args:
        T_predicted_list (list[np.ndarray]): 예측된 4x4 변환 행렬 리스트
        T_ground_truth_list (list[np.ndarray]): 실제 4x4 변환 행렬 리스트

    Returns:
        dict: 위치 및 회전 오차에 대한 통계 정보를 담은 딕셔너리
    """
    trans_errors = []
    rot_errors_deg = []

    for T_pred, T_gt in zip(T_predicted_list, T_ground_truth_list):
        p_pred, p_gt = T_pred[:3, 3], T_gt[:3, 3]
        trans_errors.append(np.linalg.norm(p_pred - p_gt))

        R_pred, R_gt = T_pred[:3, :3], T_gt[:3, :3]
        R_err = R_pred @ R_gt.T
        
        trace_val = np.trace(R_err)
        angle_rad = np.arccos(np.clip((trace_val - 1) / 2.0, -1.0, 1.0))
        rot_errors_deg.append(np.rad2deg(angle_rad))
        
    return trans_errors, rot_errors_deg


def run_urdf_viewer_with_fk(print_error: bool = True, fk_func: Callable = None):
    server = viser.ViserServer()
    
    viser_urdf = ViserUrdf(
        server, urdf_or_path=URDF_PATH,
    )

    robot_urdf = yourdfpy.URDF.load(URDF_PATH,
                                    filename_handler=partial(
                                        yourdfpy.filename_handler_magic,
                                        dir=URDF_PATH.parent,
                                    ),
        )

    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )
    
    def _update_robot_and_frames():
        q = np.array([slider.value for slider in slider_handles])
        fk = draw_robot_frames(server, viser_urdf, q, fk_func=fk_func)
        fk_gt = draw_robot_frames_yourdf(server, robot_urdf, q)

        if print_error:
            trans_errors, rot_errors_deg = calculate_fk_accuracy(fk, fk_gt) # Except tcp frame
            header = ['j1', 'j2', 'j3', 'j4', 'j5', 'cam', 'tcp']

            # print errors with a nice table
            print(f"{'Link':<6} {'Trans Error (m)':<15} {'Rot Error (deg)':<15}")
            for i, link_name in enumerate(header):
                print(f"{link_name:<6} {trans_errors[i]:<15.4f} {rot_errors_deg[i]:<15.4f}")

    for slider in slider_handles:
        slider.on_update(lambda _: _update_robot_and_frames())
    
    print("Setting initial robot configuration and drawing frames...")
    q = np.array([slider.value for slider in slider_handles])
    viser_urdf.update_cfg(q)

    _update_robot_and_frames()

    while True:
        time.sleep(0.01) 

def run_urdf_viewer_with_ik(print_error: bool = True, ik_func: Callable = None):
    server = viser.ViserServer()
    
    viser_urdf = ViserUrdf(
        server, urdf_or_path=URDF_PATH,
    )

    robot_urdf = yourdfpy.URDF.load(URDF_PATH,
                                    filename_handler=partial(
                                        yourdfpy.filename_handler_magic,
                                        dir=URDF_PATH.parent,
                                    ),
        )

    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )
    
    def _update_robot_and_frames():
        q = np.array([slider.value for slider in slider_handles])
        fk_gt = draw_robot_frames_yourdf(server, robot_urdf, q)

        # Solve ik 
        T_tcp = fk_gt[-1]
        q_ik = ik_func(q[:5], T_tcp)
        if q_ik is not None:
            print("IK Solution Found:", q_ik)
            if print_error:
                # Compute error
                q_error = q[:5] - q_ik["sol"]
                print(f"IK mean joint error: {np.mean(np.abs(q_error))} rad")
                

    for slider in slider_handles:
        slider.on_update(lambda _: _update_robot_and_frames())
    
    print("Setting initial robot configuration and drawing frames...")
    q = np.array([slider.value for slider in slider_handles])
    viser_urdf.update_cfg(q)

    _update_robot_and_frames()

    while True:
        time.sleep(0.01) 

if __name__ == "__main__":
    run_urdf_viewer()
    # run_urdf_viewer_with_fk()