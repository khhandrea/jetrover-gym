import jax.numpy as jnp
from math import degrees, radians, atan2, asin, sqrt
import numpy as np

def get_urdf():
    ### --- kinematics ---
    T_base = jnp.array([[1.        , 0.        , 0.          , 0.],
                        [0.        , 1.        , 0.          , 0.],
                        [0.        , 0.        , 1.          , 0.11609108],
                        [0.        , 0.        , 0.          , 1.]])

    T_joint1 = jnp.array([[1.        , 0.        , 0.        , 0.02513281],
                        [0.        , 1.        , 0.        , 0.],
                        [0.        , 0.        , 1.        , 0.07740269],
                        [0.        , 0.        , 0.        , 1.]])

    T_joint2 = jnp.array([[1.        , 0.        , 0.        , 0.],
                        [0.        , 1.        , 0.        , 0.],
                        [0.        , 0.        , 1.        , 0.0338648],
                        [0.        , 0.        , 0.        , 1.]])

    T_joint3 = jnp.array([[1.        , 0.        , 0.        , 0.],
                        [0.        , 1.        , 0.        , 0.],
                        [0.        , 0.        , 1.        , 0.12941645],
                        [0.        , 0.        , 0.        , 1.]])

    T_joint4 = jnp.array([[1.        , 0.        , 0.        , 0.],
                        [0.        , 1.        , 0.        , 0.],
                        [0.        , 0.        , 1.        , 0.12944463],
                        [0.        , 0.        , 0.        , 1.]])

    T_joint5 = jnp.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.05448333],
                        [0., 0., 0., 1.]])
    ##end-effector frame
    T_TCP_offset = jnp.array([[1.        , 0.        , 0.        , 0.],
                            [0.        , 1.        , 0.        , 0.],
                            [0.        , 0.        , 1.        , 0.113],  # 0.113 (Measured with closed gripper)
                            [0.        , 0.        , 0.        , 1.]])

    T_cam_offset = jnp.array([[0.        , 1.        , 0.        , -0.0507],
                            [-1.       , 0.        , 0.        , 0.],
                            [0.        , 0.        , 1.        , 0.065],
                            [0.        , 0.        , 0.        , 1.]])

    j1_axis = np.array([0, 0, -1])
    j2_axis = np.array([0, 1, 0])
    j3_axis = np.array([0, 1, 0])
    j4_axis = np.array([0, 1, 0])
    j5_axis = np.array([0, 0, -1])
    
    urdf_dict = {
        "T_base": T_base,
        "T_joint_1": T_joint1,
        "T_joint_2": T_joint2,
        "T_joint_3": T_joint3,
        "T_joint_4": T_joint4,
        "T_joint_5": T_joint5,
        "Axis_joint_1": j1_axis,
        "Axis_joint_2": j2_axis,
        "Axis_joint_3": j3_axis,
        "Axis_joint_4": j4_axis,
        "Axis_joint_5": j5_axis,
        "T_TCP_offset": T_TCP_offset,
        "T_cam_offset": T_cam_offset
    }


    return urdf_dict
                            
joint1_map = [0, 1000, 500, 120, -120, 0]
joint2_map = [0, 1000, 500, 120, -120, 0]
joint3_map = [0, 1000, 500, 120, -120, 0]
joint4_map = [0, 1000, 500, 120, -120, 0]
joint5_map = [0, 1000, 500, 120, -120, 0]


def angle_transform(angle, param, inverse=False):
    if inverse:
        new_angle = ((angle - param[5]) / (param[4] - param[3])) * (param[1] - param[0]) + param[2]
    else:
        new_angle = ((angle - param[2]) / (param[1] - param[0])) * (param[4] - param[3]) + param[5]

    return new_angle

def pulse2angle(pulse):
    theta1 = angle_transform(pulse[0], joint1_map)
    theta2 = angle_transform(pulse[1], joint2_map)
    theta3 = angle_transform(pulse[2], joint3_map)
    theta4 = angle_transform(pulse[3], joint4_map)
    theta5 = angle_transform(pulse[4], joint5_map)
    
    #print(theta1, theta2, theta3, theta4, theta5)
    q = radians(theta1), radians(theta2), radians(theta3), radians(theta4), radians(theta5)
    return np.array(q)

def angle2pulse(angle):
    pulse = []
    theta1 = int(angle_transform(degrees(angle[0]), joint1_map, True))
    theta2 = int(angle_transform(degrees(angle[1]), joint2_map, True))
    theta3 = int(angle_transform(degrees(angle[2]), joint3_map, True))
    theta4 = int(angle_transform(degrees(angle[3]), joint4_map, True))
    theta5 = int(angle_transform(degrees(angle[4]), joint5_map, True))

    pulse = np.array([theta1, theta2, theta3, theta4, theta5])

    return pulse

def angle2pulse_single(angle, joint_idx):
    if joint_idx == 0:
        pulse = int(angle_transform(degrees(angle), joint1_map, True))
    elif joint_idx == 1:
        pulse = int(angle_transform(degrees(angle), joint2_map, True))
    elif joint_idx == 2:
        pulse = int(angle_transform(degrees(angle), joint3_map, True))
    elif joint_idx == 3:
        pulse = int(angle_transform(degrees(angle), joint4_map, True))
    elif joint_idx == 4:
        pulse = int(angle_transform(degrees(angle), joint5_map, True))
    else:
        raise ValueError("Invalid joint index")

    return pulse


def check_se3(T):
    assert T.shape == (4,4), "T must be 4x4 matrix"
    assert jnp.allclose(T[3,:], jnp.array([0.,0.,0.,1.])), "Last row must be [0,0,0,1]"
    assert jnp.allclose(T[:3,:3] @ T[:3,:3].T, jnp.eye(3)), "Rotation part must be orthogonal"
    assert jnp.isclose(jnp.linalg.det(T[:3,:3]), 1.0), "Rotation part must have determinant 1"
