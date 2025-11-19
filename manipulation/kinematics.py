import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import least_squares
import time
from .utils.kinematics_utils import get_urdf, check_se3
from jaxlie import SO3
from dataclasses import dataclass


##### Codees for solution
def T_y_axis(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    T = jnp.array([
        [c, 0., s, 0.],
        [0., 1., 0., 0.],
        [-s, .0, c, 0.],
        [0., 0., 0., 1.]
    ])
    return T

def T_z_axis(theta):
    c, s = jnp.cos(-theta), jnp.sin(-theta)
    T = jnp.array([
        [c, -s, 0., 0.],
        [s, c, 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    return T

def log_SO3(R):
    return SO3.from_matrix(R).log()

def residual_fn_(q, T_goal, target='tcp'):
    T_curr = forward_kinematics(q, target=target)
    pos_error = T_goal[:3, -1] - T_curr[:3, -1]
    rot_error = log_SO3(T_curr[:3, :3].T @ T_goal[:3, :3])
    error = jnp.hstack([pos_error, rot_error])
    return error # vector6d

jac_ = jax.jit(jax.jacfwd(residual_fn_)) 

#####

def forward_kinematics(
    q: np.ndarray, 
    target: str = 'j1'
) -> np.ndarray:
    """
    Compute the forward kinematics for the robotic arm.
    Args:
        q (np.ndarray): Joint angles (1D array of size 5) in radian.
        target (str): Target frame to compute the pose for. Options are:
                      'j1', 'j2', 'j3', 'j4', 'j5', 'cam', 'tcp'.
                      Default is 'j1'.
    Returns:
        jnp.ndarray: 4x4 transformation matrix of the target frame.
    """

    ######
    ## TODO : Implement Forward Kinematics using the robot's URDF parameters.
    ## You can refer to the URDF parameters defined in get_urdf() function in

    urdf = get_urdf()
    pose = urdf["T_base"] @ urdf["T_joint_1"] @ T_z_axis(q[0])
    if target == "j1": return pose
    pose = pose @ urdf["T_joint_2"] @ T_y_axis(q[1])
    if target == "j2": return pose
    pose = pose @ urdf["T_joint_3"] @ T_y_axis(q[2])
    if target == "j3": return pose
    pose = pose @ urdf["T_joint_4"] @ T_y_axis(q[3])
    if target == "j4": return pose
    
    if target == "cam":
        pose = pose @ urdf["T_cam_offset"]
        return pose
    else:
        pose = pose @ urdf["T_joint_5"] @ T_z_axis(q[4])
    if target == "j5": return pose
    elif target == "tcp":
        pose = pose @ urdf["T_TCP_offset"]


    #####

    return pose

def inverse_kinematics(
    q: np.ndarray, 
    T_target: np.ndarray
) -> dict | None:
    """
    Compute the inverse kinematics for the robotic arm.
    Args:
        q (np.ndarray): Initial joint angles (1D array of size 5) in radian.
        T_target (np.ndarray): 4x4 transformation matrix of the target end-effector pose (tcp).
    """

    #####
    ## TODO : Implement Inverse Kinematics using numerical optimization.
    ## You may use scipy.optimize.least_squares and forward_kinematics function for this purpose.
    ## When the optimization fails, return None.

    residual_fn = partial(residual_fn_, T_goal=T_target)
    jac = partial(jac_, T_goal=T_target)

    try:
        res = least_squares(residual_fn, q, jac=jac)
    except Exception as e:
        print(e)
        print("singular")
        return None
    if res.success == False:
        return None

    #####

    result = {}
    result["sol"] = np.array(res.x)
    result["pos_error"] = np.linalg.norm(residual_fn(res.x))

    return result
